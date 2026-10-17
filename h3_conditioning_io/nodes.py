"""ComfyUI save/load nodes for persistent MiniMax H3 CONDITIONING."""
from __future__ import annotations

from pathlib import Path

import folder_paths

from .codec import load_conditioning, save_conditioning


def _file_path(relative_name: str) -> tuple[Path, str]:
    # Deliberately isolate generated conditioning files under ComfyUI/output.
    root = (Path(folder_paths.get_output_directory()) / "h3_conditioning").resolve()
    given = Path(str(relative_name).strip().replace("\\", "/"))
    if not str(relative_name).strip() or given.is_absolute() or ".." in given.parts:
        raise ValueError("Provide a relative path under output/h3_conditioning")
    if given.suffix == "":
        given = given.with_suffix(".safetensors")
    if given.suffix != ".safetensors":
        raise ValueError("Filename must end in .safetensors")
    resolved = (root / given).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path must remain inside output/h3_conditioning")
    return resolved, str(given).replace("\\", "/")


class H3ConditioningSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "filename": ("STRING", {"default": "shot_0001/positive.safetensors"}),
            "overwrite": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "saved_relative_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "H3/Conditioning IO"

    def run(self, conditioning, filename, overwrite):
        path, relative = _file_path(filename)
        byte_count = save_conditioning(conditioning, path, overwrite=overwrite)
        print(f"[H3 Conditioning IO] Saved {relative} ({byte_count:,} bytes)")
        return (conditioning, relative)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Persistent save is a deliberate action on each queued run.
        return float("nan")


class H3ConditioningLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "filename": ("STRING", {"default": "shot_0001/positive.safetensors"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "H3/Conditioning IO"

    def run(self, filename):
        path, relative = _file_path(filename)
        conditioning = load_conditioning(path)
        print(f"[H3 Conditioning IO] Loaded {relative}")
        return (conditioning,)

    @classmethod
    def IS_CHANGED(cls, filename):
        path, _ = _file_path(filename)
        if not path.is_file():
            return ("missing", str(path))
        s = path.stat()
        return (str(path), s.st_mtime_ns, s.st_size)


NODE_CLASS_MAPPINGS = {
    "H3ConditioningIOSave": H3ConditioningSave,
    "H3ConditioningIOLoad": H3ConditioningLoad,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ConditioningIOSave": "H3 Save Conditioning (Safetensors)",
    "H3ConditioningIOLoad": "H3 Load Conditioning (Safetensors)",
}
