"""Smoke tests using a small ComfyUI mock, no H3 weights or GPU needed.

Run from package directory: python test_persistent_io.py
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib
import json
import sys
import types

import torch
from safetensors.torch import save_file

# Lightweight ComfyUI interface used by the codec and nodes.
comfy = types.ModuleType("comfy")
comfy.__path__ = []
nested = types.ModuleType("comfy.nested_tensor")

class NestedTensor:
    def __init__(self, members):
        self.tensors = list(members)

    def unbind(self):
        return tuple(self.tensors)

nested.NestedTensor = NestedTensor
comfy.nested_tensor = nested
sys.modules["comfy"] = comfy
sys.modules["comfy.nested_tensor"] = nested

folder_paths = types.ModuleType("folder_paths")
sys.modules["folder_paths"] = folder_paths

from h3_conditioning_io.latent_codec import save_latent, load_latent
from h3_conditioning_io.latent_nodes import H3LatentIOSave, H3LatentIOLoad, _latent_path
from h3_conditioning_io.nodes import H3ConditioningSave, H3ConditioningLoad

with TemporaryDirectory() as root:
    root = Path(root)
    output = root / "output"
    input_ = root / "input"
    output.mkdir()
    input_.mkdir()
    folder_paths.get_output_directory = lambda: str(output)
    folder_paths.get_input_directory = lambda: str(input_)

    video = torch.arange(2*3*2*3, dtype=torch.float32).reshape(2,3,2,3).bfloat16()
    audio = torch.arange(24, dtype=torch.float16).reshape(1,2,3,4)
    latent = {
        "samples": NestedTensor([video, audio]),
        "noise_mask": torch.tensor([1,0,1],dtype=torch.uint8),
        "batch_index": [2,3],
        "metadata": {"producer": "test", "enabled": False},
    }
    path = root / "av.h3latent"
    size = save_latent(latent,path)
    assert size > 0
    loaded = load_latent(path)
    assert isinstance(loaded["samples"], NestedTensor)
    assert len(loaded["samples"].unbind()) == 2
    for expected,got in zip([video,audio],loaded["samples"].unbind()):
        assert expected.dtype == got.dtype
        assert torch.equal(expected,got)
    assert loaded["batch_index"] == [2,3]
    assert torch.equal(loaded["noise_mask"],latent["noise_mask"])
    assert loaded["metadata"] == latent["metadata"]

    # Normal float tensor samples are not turned into NestedTensors.
    scalar_latent = {"samples": torch.ones(1,4,2,2,dtype=torch.float16)}
    save_latent(scalar_latent,root/"plain.h3latent")
    got = load_latent(root/"plain.h3latent")
    assert isinstance(got["samples"],torch.Tensor)
    assert got["samples"].dtype == torch.float16
    # Even a single-member NestedTensor preserves its container type.
    save_latent({"samples":NestedTensor([video])},root/"one_nested.h3latent")
    assert isinstance(load_latent(root/"one_nested.h3latent")["samples"],NestedTensor)

    # Files from the user's original .h3latent node still load correctly.
    legacy = {
        "tensor_count":torch.tensor([2],dtype=torch.int32),
        "format_version":torch.tensor([1],dtype=torch.int32),
        "latent_0": video,
        "latent_1": audio,
    }
    save_file(legacy,str(root/"old.h3latent"))
    got = load_latent(root/"old.h3latent")
    assert isinstance(got["samples"],NestedTensor)
    assert got["samples"].unbind()[0].dtype == torch.bfloat16

    # UI save/load round trip and predictable path.
    saved = H3LatentIOSave().run(latent,"movie/shot_01/base.h3latent",True)
    assert saved[1] == "movie/shot_01/base.h3latent"
    assert (output/"h3_latents/movie/shot_01/base.h3latent").exists()
    restored = H3LatentIOLoad().run("movie/shot_01/base.h3latent")[0]
    assert isinstance(restored["samples"], NestedTensor)
    assert H3LatentIOLoad.IS_CHANGED("movie/shot_01/base.h3latent")
    # Load files saved by old node anywhere below output without relocation.
    legacy_dst = output / "latents/old_style.h3latent"
    legacy_dst.parent.mkdir(parents=True)
    legacy_dst.write_bytes((root/"old.h3latent").read_bytes())
    assert H3LatentIOLoad().run("latents/old_style.h3latent", "output")[0]

    try:
        _latent_path("../secrets.h3latent")
        assert False, "Path traversal should fail"
    except ValueError:
        pass
    try:
        save_latent(latent,path,overwrite=False)
        assert False, "overwrite=False should fail"
    except FileExistsError:
        pass

    # Also verify original conditioning nodes still work unchanged.
    conditioning = [[torch.ones(1,3,dtype=torch.bfloat16),{"minimax_refs":[{"kind":"image","latent":torch.ones(1,2,dtype=torch.float16)}]}]]
    saved = H3ConditioningSave().run(conditioning,"movie/shot_01/positive.safetensors",True)
    assert saved[1]=="movie/shot_01/positive.safetensors"
    restored = H3ConditioningLoad().run(saved[1])[0]
    assert restored[0][0].dtype == torch.bfloat16
    assert restored[0][1]["minimax_refs"][0]["latent"].dtype == torch.float16

print("PASS: AV streams, FP16/BF16, nested/single, auxiliary metadata, legacy files, nodes, paths, conditioning")
