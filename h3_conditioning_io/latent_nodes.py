"""ComfyUI nodes for persistent MiniMax H3 AV LATENT saving/loading."""
from __future__ import annotations

from pathlib import Path
import folder_paths

from .latent_codec import load_latent, save_latent


def _latent_path(filename: str, source: str = "h3_latents") -> tuple[Path, str]:
    # A stable path is essential for cross-workflow, per-shot deferred upscaling.
    if source not in {"h3_latents", "output", "input"}:
        raise ValueError(f"Unsupported H3 latent folder: {source}")
    if source == "h3_latents":
        root = Path(folder_paths.get_output_directory()) / "h3_latents"
    elif source == "output":
        root = Path(folder_paths.get_output_directory())
    else:
        root = Path(folder_paths.get_input_directory())
    root = root.resolve()
    raw = str(filename).strip().replace("\\", "/")
    if not raw:
        raise ValueError("H3 latent filename must not be empty")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or ":" in raw:
        raise ValueError("H3 latent filename must be a relative path inside the selected folder")
    if not relative.suffix:
        relative = relative.with_suffix(".h3latent")
    if relative.suffix.lower() != ".h3latent":
        raise ValueError("H3 latent filename must end in .h3latent")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("H3 latent filename escapes the allowed folder")
    return resolved, relative.as_posix()


class H3LatentIOSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "filename": ("STRING", {"default": "shot_0001/base.h3latent"}),
            "overwrite": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "saved_relative_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "H3/Persistent IO"
    DESCRIPTION = "Save H3 NestedTensor video/audio LATENT without dtype conversion, for later upscaling."

    def run(self, samples, filename, overwrite):
        path, relative = _latent_path(filename)
        count = save_latent(samples, path, overwrite=overwrite)
        print(f"[H3 Persistent IO] Saved latent {relative} ({count:,} bytes)")
        return (samples, relative)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class H3LatentIOLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "filename": ("STRING", {"default": "shot_0001/base.h3latent"}),
            "source": (["h3_latents", "output", "input"], {"default": "h3_latents"}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "run"
    CATEGORY = "H3/Persistent IO"
    DESCRIPTION = "Load saved MiniMax H3 video/audio LATENT from disk. Supports legacy .h3latent files."

    def run(self, filename, source="h3_latents"):
        path, relative = _latent_path(filename, source)
        if not path.is_file():
            raise FileNotFoundError(f"H3 latent not found under {source}: {relative}")
        latent = load_latent(path)
        print(f"[H3 Persistent IO] Loaded latent from {source}/{relative}")
        return (latent,)

    @classmethod
    def IS_CHANGED(cls, filename, source="h3_latents"):
        path, _ = _latent_path(filename, source)
        if not path.is_file():
            return ("missing", str(path))
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)


NODE_CLASS_MAPPINGS = {
    "H3PersistentIOSaveLatent": H3LatentIOSave,
    "H3PersistentIOLoadLatent": H3LatentIOLoad,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PersistentIOSaveLatent": "H3 Save AV Latent (Safetensors)",
    "H3PersistentIOLoadLatent": "H3 Load AV Latent (Safetensors)",
}
