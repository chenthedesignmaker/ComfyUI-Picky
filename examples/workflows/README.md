# Example workflows

These are ComfyUI API-format graphs, exported from the editor, for each
quality tier. Load one in ComfyUI to see exactly what the editor queues.

| File | Tier | Steps |
| --- | --- | --- |
| `h3_single_pass_quick_8_t2v.api.json` | preview: one pass with the turbo LoRA | 8 |
| `h3_two_pass_turbo_6x2_t2v.api.json` | turbo 2-pass: base pass, latent upscale, refine | 6 + 2 |
| `h3_single_pass_fine_25_t2v.api.json` | final: one pass without the LoRA | 25 |
| `h3_two_pass_fine_15x5_t2v.api.json` | final 2-pass | 15 + 5 |

`../h3_t2v.json` is the plain UI-format text-to-video graph that render
profiles start from.
