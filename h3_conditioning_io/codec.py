"""Lossless, pickle-free persistence for MiniMax H3 CONDITIONING values.

Supports the nested tensor/dict/list/tuple/scalar structure used by ComfyUI's
MiniMax H3 CONDITIONING. Does NOT save an initial AV latent or Motion Context.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT_KEY = "h3_conditioning_io"
FORMAT_VERSION = 1


def validate_conditioning(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise TypeError("Expected a nonempty ComfyUI CONDITIONING list")
    for idx, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError(f"Conditioning item {idx} must be a [tensor, metadata] pair")
        if not isinstance(item[0], torch.Tensor) or not isinstance(item[1], dict):
            raise TypeError(f"Conditioning item {idx} must contain tensor and metadata dict")


def _pack(value: Any, table: dict[str, torch.Tensor], where: str) -> dict:
    if isinstance(value, torch.Tensor):
        name = f"tensor_{len(table):07d}"
        # Safetensors requires ordinary contiguous CPU tensors.
        table[name] = value.detach().to(device="cpu").contiguous()
        return {"type": "tensor", "name": name}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "scalar", "value": value}
    if isinstance(value, dict):
        pairs = []
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Unsupported non-string dictionary key at {where}: {key!r}")
            pairs.append([key, _pack(child, table, f"{where}.{key}")])
        return {"type": "dict", "items": pairs}
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [_pack(child, table, f"{where}[{i}]") for i, child in enumerate(value)],
        }
    raise TypeError(f"Unsupported conditioning data at {where}: {type(value).__qualname__}")


def _unpack(node: Any, table: dict[str, torch.Tensor]) -> Any:
    if not isinstance(node, dict):
        raise ValueError("Malformed H3 conditioning structure")
    kind = node.get("type")
    if kind == "tensor":
        key = node.get("name")
        if not isinstance(key, str) or key not in table:
            raise ValueError(f"Missing conditioning tensor: {key!r}")
        return table[key]
    if kind == "scalar":
        value = node.get("value")
        if value is None or isinstance(value, (str, int, bool, float)):
            return value
        raise ValueError("Invalid scalar in H3 conditioning file")
    if kind in ("list", "tuple"):
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError("Invalid sequence in H3 conditioning file")
        values = [_unpack(item, table) for item in items]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        pairs = node.get("items")
        if not isinstance(pairs, list):
            raise ValueError("Invalid dictionary in H3 conditioning file")
        result = {}
        for entry in pairs:
            if (not isinstance(entry, list) or len(entry) != 2
                    or not isinstance(entry[0], str) or entry[0] in result):
                raise ValueError("Invalid or duplicate H3 conditioning dictionary entry")
            result[entry[0]] = _unpack(entry[1], table)
        return result
    raise ValueError(f"Unknown H3 conditioning structure type: {kind!r}")


def save_conditioning(conditioning: Any, path: Path, *, overwrite: bool = True) -> int:
    """Atomically save CONDITIONING to a persistent .safetensors file."""
    validate_conditioning(conditioning)
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("H3 conditioning files must end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")

    tensors: dict[str, torch.Tensor] = {}
    structure = _pack(conditioning, tensors, "conditioning")
    metadata = {
        FORMAT_KEY: json.dumps(
            {"version": FORMAT_VERSION, "structure": structure},
            ensure_ascii=True, allow_nan=False, separators=(",", ":")
        )
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        # Check the file can be read as safetensors before publishing it.
        with safe_open(str(temporary), framework="pt", device="cpu") as handle:
            if FORMAT_KEY not in (handle.metadata() or {}):
                raise RuntimeError("H3 conditioning save verification failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


def load_conditioning(path: Path) -> Any:
    """Load CONDITIONING with exact nested structure, dtypes and values."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw_meta = (handle.metadata() or {}).get(FORMAT_KEY)
        if raw_meta is None:
            raise ValueError("Not an H3 Conditioning IO file (missing format metadata)")
        descriptor = json.loads(raw_meta)
        if not isinstance(descriptor, dict) or descriptor.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported H3 Conditioning IO file version")
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    result = _unpack(descriptor["structure"], tensors)
    validate_conditioning(result)
    return result
