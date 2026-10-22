# Long-form chains and master upscaling

Read this for 1080p work, multi-scene chains rendered in one unattended
ComfyUI graph, or a SeedVR2 master upscale. Ordinary continuation of one take
is `tools/render_sources.py --extend TAKE_ID`.

## Multi-shot chains (long-form and 1088p)

When authoring and generating multi-scene stories or extending clips for long-form video with true motion, subject, and audio continuity:

### The 1088p memory limit
- On a 24 GB GPU (RTX 4090), generating at **1920×1088** (2.09 MP) has a strict single-shot limit of **124 frames (5.1s)** due to DiT 3D activation memory (~23.1 GB peak).
- **Never attempt a single 243-frame (10.1s) pass at 1088p** — it will hit CUDA OOM.
- To produce seamless 10s, 30s, or multi-minute sequences at 1088p, always decompose into 124-frame scenes chained via latent context.

### Method A: one unattended chain graph
For rendering complete 2+ scene continuous narratives in a single unattended execution with automatic clip trimming and assembly:

1. **Author Structured Scene Sequence**:
   - Scene 1 (Opening Shot): Sets the character, environment, dialogue `<d>[Language] ...</d>`, and soundscape.
   - Subsequent Scenes (Context Continuations): Begin with `integrated_multimodal_description: [Shot 1] ... continues directly from the incoming H3 Motion Context... Preserve character identity, clothing, lighting, camera position...`.
2. **Build Sequence Plan Graph**:
   - Use `MiniMaxH3ChainPlan` node with `run_name`, `generation_fingerprint`, resolution (e.g. `1920x1088` or `1280x720` @ 24fps), `context_length: 22`, `audio_context_length: 22`, and `plan_json` specifying the array of `shots` with `id`, `prompt`, `length: 124`, `steps` (8 for Turbo draft, 25 for Base master), and per-shot `seed`.
   - Engine configuration: match the render profile the rest of the project
     uses (`quality-tiers.md` lists the three). All three sample with
     `res_multistep`; `euler` was tested on the two-pass tiers and ruled out,
     and none of them uses `lcm`. Drafts run 8 steps and masters 25 — with the
     turbo LoRA at strength 1.0 for drafts only if you chain on
     `H3 T2V w Turbo`.
   - Node chain: `MiniMaxH3ChainPlan` → `MiniMaxH3ChainLoopStart` → `MiniMaxH3ChainCurrent` → `MiniMaxH3ChainContext` → `MiniMaxH3ImageToVideo` → `SamplerCustomAdvanced` → `MiniMaxH3LoopTrim` (removes 22 pinned frames + aligns audio) → `MiniMaxH3ChainSegmentSave` → `MiniMaxH3ChainLoopEnd` → `MiniMaxH3ChainAssemble` (stitches all scenes into master MP4 + master WAV with 0.00ms audio drift).
   - Set `MiniMaxH3ChainReview` input `enabled: false` for unattended programmatic rendering.
3. **Queue and Monitor**: Submit to ComfyUI `POST /prompt` and poll `GET /history/{prompt_id}` until the assembled master output is written to `output/h3_chains/<run_name>/final/`.

### Method B: extend one take at a time (`/render/extend`)
For continuing a specific existing take on demand:

1. **Prepare Extension**: `POST /video_prompter/render/extend` with:
   ```json
   {
     "project": "My Film",
     "profile": "H3 Hybrid local t2v",
     "prompt": { "...the segment's document..." },
     "source": "video/take_00001.mp4 [output]",
     "prompt_text": "The camera pushes further down the alley as rain falls harder",
     "width": 1920,
     "height": 1088,
     "frames": 124,
     "seconds": 5.166,
     "quality": "preview",
     "context": 22,
     "latent": "h3_context/take_00001.safetensors"
   }
   ```
   - `profile` and `prompt` are what make a continuation the shot's **own**
     render with Motion Context wired in, rather than a plainer pipeline beside
     it. The profile carries the weights, LoRAs and sampler; the prompt carries
     the media, and the mode (t2v / i2v / ref2v) is chosen off it exactly as
     `/render/prepare` chooses it. Send both, or a continuation of a
     `<Subject 1>` shot renders text-only on stock weights and the model invents
     a new face for the second half of the chain. A shot with keyframes or
     references and no `profile` is refused with a **400**.
   - `context`: number of tail frames to pin (default `22` for optimal VAE grid alignment).
   - `latent`: optional path to previous run's saved AV latent checkpoint (bypasses lossy decode/re-encode cycles for pristine audio).
   - Returns `{workflow, width, height, frames, seconds, seed, prompt, mode,
     latent_path}`. `latent_path` is the continuation's **own** saved AV latent
     — pass it as `latent` when extending *this* take next, so a chain stays on
     the clean-audio path link after link.
2. **Queue on ComfyUI**: Mint `prompt_id` UUID and `POST /prompt` with `workflow` and `extra_data.video_prompter`.
3. **Poll, Record & Pick**: Poll `GET /history/{prompt_id}`, record the output to gallery via `/gallery/record`, and pick the take into the timeline.


## SeedVR2 master upscale (manual ComfyUI graph — validated 720p → 1080p)

For master delivery, upscale the finished H3 take with **SeedVR2** (temporal
diffusion video super-resolution). This is a **manual ComfyUI graph**, not a tool
endpoint — run it in ComfyUI directly; nothing in the video-prompter API drives
it.

- **Validated path:** 1344×736 (720p) → **1972×1080** (~1.43×), 24 fps, audio
  preserved. Higher factors toward 2K/4K are possible but were not validated
  here — scale up from this known-good pass rather than assuming 2×/4K works.
- **Source:** the master H3 MP4 take (CRF 16).
- **Model:** a `SeedVR2` video upscaler checkpoint.
- **VRAM (24 GB / RTX 4090):** spatial tiling (~512×512 tiles, 64px overlap) plus
  temporal windowing (16–24 frame chunks) to keep peak VRAM in budget.
- **Audio passthrough:** mux the original 48 kHz track from the source take so
  the upscale never re-encodes the audio:
  ```bash
  ffmpeg -y -i upscaled_video_only.mp4 -i source_take.mp4 \
    -c:v copy -c:a copy -map 0:v:0 -map 1:a:0? final_upscaled.mp4
  ```
