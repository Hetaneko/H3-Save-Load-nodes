"""Persistence for native MiniMax H3 video/audio ComfyUI LATENT structures.

The .h3latent extension denotes a safetensors file, not a pickle archive.
The new format remains readable by the older basic H3 AV loader because
latent_0, latent_1, tensor_count, and format_version retain their key names.
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

from .codec import _pack, _unpack

FORMAT_KEY = "h3_persistent_io_latent"
FORMAT_VERSION = 2
LEGACY_VERSION = 1


def _is_nested(value: Any) -> bool:
    # H3 employs ComfyUI NestedTensor for AV streams.
    # Avoid importing ComfyUI here, so format codec can be unit-tested separately.
    try:
        import comfy.nested_tensor
        if isinstance(value, comfy.nested_tensor.NestedTensor):
            return True
    except ImportError:
        pass
    return bool(getattr(value, "is_nested", False))


def _members(value: Any) -> tuple[list[torch.Tensor], bool]:
    nested = _is_nested(value)
    if nested:
        members = list(value.unbind())
    elif isinstance(value, torch.Tensor):
        members = [value]
    else:
        raise TypeError("LATENT['samples'] must be a torch.Tensor or ComfyUI NestedTensor")
    if not members or any(not isinstance(t, torch.Tensor) for t in members):
        raise ValueError("H3 LATENT must contain at least one tensor stream")
    # Validate each stream before writing.
    for i, t in enumerate(members):
        if t.layout != torch.strided:
            raise TypeError(f"LATENT stream {i} has unsupported tensor layout: {t.layout}")
    return members, nested


def validate_latent(latent: Any) -> None:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("Expected ComfyUI LATENT dictionary containing 'samples'")
    _members(latent["samples"])


def save_latent(latent: dict[str, Any], path: Path, *, overwrite: bool = True) -> int:
    """Save exact stream dtypes and optional LATENT fields, without any .float() casts."""
    validate_latent(latent)
    path = Path(path)
    if path.suffix.lower() != ".h3latent":
        raise ValueError("H3 latent files must end in .h3latent")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"Refusing to overwrite existing latent: {path}")

    members, nested = _members(latent["samples"])
    tensors: dict[str, torch.Tensor] = {
        "format_version": torch.tensor([FORMAT_VERSION], dtype=torch.int32),
        "tensor_count": torch.tensor([len(members)], dtype=torch.int32),
    }
    for i, t in enumerate(members):
        tensors[f"latent_{i}"] = t.detach().to(device="cpu").contiguous()

    # Retain other ComfyUI LATENT data (for example noise_mask/batch_index) too.
    extra = {k: v for k, v in latent.items() if k != "samples"}
    extra_structure = _pack(extra, tensors, "latent_extra")
    metadata = {
        FORMAT_KEY: json.dumps({
            "version": FORMAT_VERSION,
            "samples_container": "nested" if nested else "tensor",
            "extras": extra_structure,
        }, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        "format": "minimax_h3_av_latent",
        "format_version": str(FORMAT_VERSION),
        "tensor_count": str(len(members)),
    }
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(tensors, str(tmp), metadata=metadata)
        with safe_open(str(tmp), framework="pt", device="cpu") as handle:
            if handle.metadata() is None or FORMAT_KEY not in handle.metadata():
                raise RuntimeError("H3 latent save verification failed")
        # Atomic publishing. Do not overwrite if told not to.
        if not overwrite and path.exists():
            raise FileExistsError(f"Refusing to overwrite existing latent: {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path.stat().st_size


def load_latent(path: Path) -> dict[str, Any]:
    """Load persistent format or version-1 H3 AV files written by older custom node."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    if "tensor_count" not in tensors:
        raise ValueError("Not a MiniMax H3 AV latent: missing 'tensor_count'")
    tensor_count_tensor = tensors["tensor_count"]
    if tensor_count_tensor.numel() != 1:
        raise ValueError("Malformed H3 latent tensor_count")
    count = int(tensor_count_tensor.item())
    if count < 1 or count > 32:
        raise ValueError(f"Unreasonable H3 latent stream count: {count}")
    members = []
    for i in range(count):
        key = f"latent_{i}"
        if key not in tensors:
            raise ValueError(f"Missing H3 AV latent tensor {key}")
        members.append(tensors[key])  # Never cast to float32.

    spec_raw = metadata.get(FORMAT_KEY)
    if spec_raw is None:
        # Backward compatibility with provided v1 SaveMiniMaxH3AVLatent.
        # A one-member legacy file was an ordinary tensor.
        nested = count > 1
        extras: dict[str, Any] = {}
    else:
        try:
            spec = json.loads(spec_raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid H3 latent metadata") from exc
        if not isinstance(spec, dict) or spec.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported H3 persistent latent version")
        if spec.get("samples_container") not in ("nested", "tensor"):
            raise ValueError("Invalid H3 latent samples container")
        nested = spec["samples_container"] == "nested"
        if not nested and count != 1:
            raise ValueError("Non-nested H3 LATENT must have one sample stream")
        extras = _unpack(spec.get("extras"), tensors)
        if not isinstance(extras, dict) or "samples" in extras:
            raise ValueError("Invalid H3 latent auxiliary data")

    if nested:
        import comfy.nested_tensor
        samples = comfy.nested_tensor.NestedTensor(members)
    else:
        samples = members[0]
    result = {**extras, "samples": samples}
    validate_latent(result)
    return result
