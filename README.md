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
