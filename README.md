# H3 Persistent IO — ComfyUI custom nodes

Permanent **conditioning + AV latent** persistence for a two-pass MiniMax H3 workflow. No pickle, no cache eviction, no session-scoped signatures, no re-encoding of decoded videos.

This package **extends** the previously supplied `h3_conditioning_io` custom-node folder. It retains the original two conditioning nodes (same IDs) and adds two H3 video/audio latent nodes.

## Install

Extract `h3_conditioning_io` into `ComfyUI/custom_nodes/` (merge/replace the older `h3_conditioning_io` folder if already installed). Restart ComfyUI. PyTorch and `safetensors` are provided by a normal ComfyUI install.

## Nodes

| Node | Inputs | Outputs | Persistent storage |
| --- | --- | --- | --- |
| **H3 Save Conditioning (Safetensors)** | `conditioning`, `filename`, `overwrite` | `CONDITIONING`, saved relative filename | `output/h3_conditioning/` |
| **H3 Load Conditioning (Safetensors)** | `filename` | `CONDITIONING` | `output/h3_conditioning/` |
| **H3 Save AV Latent (Safetensors)** | `samples` (`LATENT`), `filename`, `overwrite` | original `LATENT`, saved relative filename | `output/h3_latents/` |
| **H3 Load AV Latent (Safetensors)** | `filename`, `source` | restored `LATENT` | source-dependent; see below |

All filenames can include *relative subdirectories*, making them suitable for film/shot identifiers. For example, give the latent saver `film_A/shot_003/base.h3latent`; the file will be at `ComfyUI/output/h3_latents/film_A/shot_003/base.h3latent`. To load it, set **the same filename** and keep `source=h3_latents`.

The conditioning saver likewise accepts `film_A/shot_003/positive.safetensors`, stored under `output/h3_conditioning/`. Use a second conditioning save/load for negative conditioning where applicable.

**Load AV Latent `source` options:**

- `h3_latents` (default): `ComfyUI/output/h3_latents/` — files written by our new node.
- `output`: `ComfyUI/output/` — supports `.h3latent` files written by the original `SaveMiniMaxH3AVLatent` node (for example `latents/MiniMaxH3_00001_.h3latent`) without moving them.
- `input`: `ComfyUI/input/` — supports files manually placed there or written by other workflows.

All file arguments are validated as relative paths confined to their respective roots. Save operations use a temporary file and atomic rename. Set `overwrite=false` to prevent unintended replacement of an existing shot's latent.

## Two-pass MiniMax H3 usage

**Base pass** (one save for each shot):

1. Save **original pre-Motion-Context** positive/negative conditioning with `H3 Save Conditioning (Safetensors)`.
2. Build any Motion Context conditioning *dynamically* using the previous shot's saved latent; don't save those augmented conditionings.
3. Generate H3 output and apply any intended **base** refinement/blending.
4. Save the finalized **base-resolution** H3 AV `LATENT` using `H3 Save AV Latent (Safetensors)`. Preserve both the video and audio streams.

**Upscale pass** (later, possibly after reboot):

1. Load the shot's saved original conditioning and base AV latent by their stable filenames.
2. Perform your H3-aware conditioning geometry adjustment and upscale/refine.
3. For chained shots, use the prior shot's compatible finalized **high-resolution tail** to rebuild Motion Context dynamically, protecting overlap appropriately. Your upscaling workflow must implement this mechanism; these file nodes only persist/reload the tensors.
4. Save only high-resolution tail latents needed for subsequent shots, and export the finished video.

### Important details

- MiniMax H3 may use `comfy.nested_tensor.NestedTensor` carrying separate video/audio streams. The AV saver serializes all streams into `.h3latent` (safetensors data) and the loader reconstructs the NestedTensor on load.
- **Dtypes are preserved.** No `.float()` conversion; BF16 and FP16 tensors remain BF16/FP16. The loader keeps tensors on CPU until downstream ComfyUI nodes use them.
- The new format also preserves non-`samples` LATENT values composed of tensors, dicts, lists, tuples, or scalar values. Unsupported custom Python objects are rejected rather than silently discarded.
- Files saved by the supplied older v1 `SaveMiniMaxH3AVLatent` node (with `latent_0`, `tensor_count` keys) can be read by the new loader. They did not preserve additional LATENT keys, so those cannot be reconstructed from old files.
- New format includes the legacy tensor keys for compatibility with basic H3 AV loaders, but old loaders may discard extra fields and convert stream dtypes. Prefer this package's loader.
- A persistent conditioning file may contain embedded image/video reference tensors and can be large. Files are *not* model-independent; rerun a real H3 compatibility test after changing models or ComfyUI's H3 conditioning format.
- Neither save node itself rescales, crops, alters spatial reference keyframes, trims AV context, or applies upscaling. Those operations belong to your separate workflows.
- `.h3latent` is a safetensors file extension; it is **not** a native video container.

## Test

The optional `test_persistent_io.py` runs round-trip smoke tests with CPU torch and a small ComfyUI mock:

```bash
python test_persistent_io.py
```

The test suite covers nested video/audio streams, FP16/BF16, optional fields, legacy file reading, save/load node paths, and original conditioning node compatibility. These checks are **not** a substitute for running a real H3 workflow with your ComfyUI build and nodes.
