# Video Prompter: an open-weights film studio for MiniMax H3

Video Prompter is a ComfyUI custom node pack and a browser editor for directing
short films with **MiniMax H3**, using only open weights on your own GPU. It
works like a timeline editor. You fill a bin with shots, write each shot's
prompt as a storyboard, render a few seeds, compare the takes side by side,
keep the best one, trim it and cut it into the timeline.

Built for [#OpenSourceHiggsfield](https://comfy.org/open-source-higgsfield),
as an open alternative to Higgsfield's *Cinema Studio*.

**What you get:**

- **Storyboard prompts, not prompt soup.** Each shot is a structured document
  with style, overview, timed beats, camera, dialogue, soundscape and music.
  It is composed into MiniMax's official H3 prompt format for you.
- **A prompt linter that knows H3.** It catches the ways H3 misreads a
  well-formed prompt before you spend GPU time on them: overfull beats, cuts
  with no new information, wasted quality tokens, missing soundscape and so on.
- **A cast.** Characters and locations are defined once, with reference
  pictures. You write `@Mara` in any prompt, and at render time it becomes the
  right `<Subject N>` with that member's current references. Renaming a cast
  member rewrites every mention.
- **Text, image and reference to video.** Supports first and last frames,
  ref2v with up to six references, and a Krea 2 keyframe studio for generating
  stills.
- **Extends that keep the shot.** A continuation runs on the shot's own graph
  plus Motion Context, so the framing, cast and checkpoint carry over.
- **Takes and lineage.** Sweep seeds, compare candidates, keep a full history
  of every prompt version, and fold extension chains back into one clip.
- **Quality tiers.** Use a fast low-resolution preview for checking adherence,
  turbo HD for drafts, and a 2-pass latent upscale for final renders. SeedVR2
  can enhance a take afterwards.
- **An AI writer that can stay on your machine.** Describe a scene in a
  sentence, or ask for a change ("make the second beat handheld"). The writer
  edits only the fields in scope and then repairs its draft against the
  linter. It runs on any local OpenAI-compatible server (Ollama, LM Studio,
  llama.cpp, vLLM). Gemini is an optional alternative.
- **Agent-driven.** A [Claude Code](https://claude.com/claude-code) skill
  (`.claude/skills/video-prompt`) can author, revise and render through the
  editor's HTTP API, and its edits appear live in the open editor.

## Requirements

- An NVIDIA GPU with **24 GB of VRAM** (tested on an RTX 4090). 720p at about
  10 seconds fits in 24 GB. See
  [`docs/MINIMAX_H3_VRAM_AND_CONTEXT_GUIDE.md`](docs/MINIMAX_H3_VRAM_AND_CONTEXT_GUIDE.md)
  for the full table.
- A recent [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with built-in
  MiniMax H3 support. The portable Windows build with Python 3.12 is tested.
- `ffmpeg` on your `PATH`, for export and clip assembly.
- About 60 GB of disk space for the H3 models.

## Install

### 1. Install the node pack

```bash
cd ComfyUI/custom_nodes
git clone <this repo> video_prompter
```

The face-restoration features need a few extra Python packages, installed
into ComfyUI's Python. The rest of the editor needs only the standard library.

```bash
python -m pip install -r video_prompter/requirements-faces.txt
```

### 2. Install the companion node packs

| Pack | Needed for |
| --- | --- |
| [ComfyUI-MiniMaxH3-Contex-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop) | Extends (Motion Context) |
| [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | 2-pass tiers (latent upscale) |
| [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) | Enhance (optional) |
| [ComfyUI-FeiHou-Easy-H3](https://github.com/FX-FeiHou/ComfyUI-FeiHou-Easy-H3) and [ComfyUI-FeiHou-Toolbox](https://github.com/FX-FeiHou/ComfyUI-FeiHou-Toolbox) | Face refine (optional) |

Plain text-, image- and reference-to-video renders need only ComfyUI's
built-in H3 nodes.

### 3. Download the models

These are all open weights. Put each file in the ComfyUI `models/` subfolder
shown.

| File | Folder | Source |
| --- | --- | --- |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `diffusion_models` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) (ref2v) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `minimax_h3_video_vae_fp16.safetensors` | `vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `loras` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) (preview and turbo tiers) |
| `minimax_h3_latent_upscaler_3d_fp16.safetensors` | `latent_upscale_models` | [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) (2-pass tiers) |
| `krea2_turbo_fp8_scaled.safetensors` | `diffusion_models` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors) (keyframe studio) |
| `qwen3vl_4b_fp8_scaled.safetensors` | `text_encoders` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) (keyframe studio) |

The SeedVR2 pack downloads its own models on first use.

### 4. Optional: a local model for the AI writer

Install [Ollama](https://ollama.com) and pull a model that handles structured
JSON well. A Qwen 3 or Gemma 3 model of 8B or more works, for example
`ollama pull qwen3:8b`. Then create
`ComfyUI/user/video_prompter/settings.json`, or edit it if it exists:

```json
{
  "llm_url": "http://127.0.0.1:11434/v1",
  "llm_model": "qwen3:8b"
}
```

Any OpenAI-compatible server works. LM Studio's default is
`http://127.0.0.1:1234/v1`, and llama.cpp's `llama-server` defaults to
`http://127.0.0.1:8080/v1`. If you leave out `llm_model`, the first model the
server lists is used. You can set the same values with the
`VIDEO_PROMPTER_LLM_URL` and `VIDEO_PROMPTER_LLM_MODEL` environment variables.

If you prefer Gemini, set `gemini_key` instead. When both are set, the local
server is used. Without either, the editor works normally and the writer
buttons are disabled.

> Run `GET /video_prompter/llm/status` to see which file path the settings are
> read from, and which provider is in use.

### 5. Start

Restart ComfyUI, then open **http://127.0.0.1:8188/video_prompter/**.

## A first film in five minutes

1. **New project**, then **Add source**.
2. Write a sentence in the writer bar, for example *"night market, a cook
   flips a wok of noodles into the air, slow push in, 8 seconds"*. Or fill in
   the Scene tab yourself.
3. Fix anything the **Findings** panel flags in red.
4. Render on the **preview** tier with three or four seeds. Double-click the
   source to compare the takes, then choose one.
5. Re-render the chosen seed on **turbo_hd** or **final**. Trim it on the
   timeline, add the next shot, then **Export**.

## Documentation

- [`docs/REFERENCE.md`](docs/REFERENCE.md): the full manual, covering the
  prompt model, projects, takes, profiles, export and enhancement.
- [`docs/EDITOR_USAGE.md`](docs/EDITOR_USAGE.md): editor navigation.
- [`docs/MINIMAX_H3_VRAM_AND_CONTEXT_GUIDE.md`](docs/MINIMAX_H3_VRAM_AND_CONTEXT_GUIDE.md):
  VRAM limits and long-form renders.
- [`.claude/skills/video-prompt/SKILL.md`](.claude/skills/video-prompt/SKILL.md):
  the agent skill, which is also a compact guide to writing H3 prompts.
- [`examples/workflows/`](examples/workflows): the ComfyUI graphs the tiers
  produce.

## Tests

```bash
python -m pytest
```

The tests need no GPU and no ComfyUI. The editor's JavaScript tests run from
the same command and need Node 18 or newer on your `PATH`.

## License

GPL-3.0, the same license as ComfyUI. See [LICENSE](LICENSE). The bundled
fonts are under the SIL Open Font License (`js/standalone/screening-room/fonts/`).
Model weights are covered by their own licenses.
