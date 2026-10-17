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
