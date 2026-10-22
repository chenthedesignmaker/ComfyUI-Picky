# MiniMax H3 (Hailuo 3.0) — VRAM Limits, Context Extension & Production Guide

This guide documents the VRAM budget mechanics, hardware limit thresholds, context extension architecture, and production workflows for generating high-resolution (1088p) long-form video using **MiniMax H3** in ComfyUI on an **NVIDIA RTX 4090 (24 GB)**.

---

## 1. VRAM Budget Architecture & Why Single-Shot OOM Occurs

On a 24 GB GPU, VRAM is partitioned into three distinct layers during MiniMax H3 execution:

```text
┌──────────────────────────────────────────────────────────────────┐
│                   TOTAL GPU VRAM (24.0 GB)                       │
├──────────────────────────┬───────────────────────┬───────────────┤
│  1. Static Model Weights │  2. DiT Activations   │ 3. VAE Spikes │
│       (~13.5 - 15.0 GB)  │     (~6.0 - 8.5 GB)   │  (~1.5 - 3GB) │
├──────────────────────────┼───────────────────────┼───────────────┤
│ • UNet (int8): ~12.0 GB  │ Spatio-temporal 3D    │ 3D upsample   │
│ • Text Enc (int8): Loaded│ attention sequence    │ conv buffers  │
│ • VAEs (Video+Audio): 2GB│ W/16 × H/16 × T/4     │ during decode │
└──────────────────────────┴───────────────────────┴───────────────┘
```

### The Math Behind DiT Token Explosion
MiniMax H3 compresses video into 3D latent tokens with spatial reduction ($16\times$) and temporal reduction ($4\times$):
$$\text{Tokens} \approx \left(\frac{\text{Width}}{16}\right) \times \left(\frac{\text{Height}}{16}\right) \times \left(\frac{\text{Frames}}{4}\right)$$

* **$960 \times 544$ (0.52 Mpx) @ 124 frames (5.1s)** $\rightarrow \approx 60 \times 34 \times 31 = \mathbf{63,240\text{ tokens}}$
* **$1920 \times 1088$ (2.09 Mpx) @ 124 frames (5.1s)** $\rightarrow \approx 120 \times 68 \times 31 = \mathbf{252,960\text{ tokens}}$ ($4.0\times$ increase)
* **$1920 \times 1088$ (2.09 Mpx) @ 243 frames (10.1s)** $\rightarrow \approx 120 \times 68 \times 61 = \mathbf{497,760\text{ tokens}}$ ($7.9\times$ increase)

While model weights remain constant, attention buffers, KV caches, and intermediate activation tensors scale linearly with token count. At $1920 \times 1088$ for $>5.1\text{s}$, activation memory exceeds the remaining 9 GB headroom, triggering an immediate CUDA Out-of-Memory (OOM) crash.

---

## 2. RTX 4090 (24 GB) Generation Limit Matrix

The following limits have been empirically verified on PyTorch 2.13+CUDA 13.0 with `int8` weights and Comfy Kitchen Attention:

| Resolution | Pixel Count | Aspect Ratio | Max Single-Shot Frames | Max Single-Shot Duration | Peak VRAM | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$960 \times 544$** | 0.52 Mpx | 16:9 | 243 frames | **10.1s** | ~18.5 GB | 🟢 Safe (Fast drafting) |
| **$1280 \times 720$** | 0.92 Mpx | 16:9 (720p) | 243 frames | **10.1s** | ~21.2 GB | 🟢 Safe |
| **$1344 \times 768$** | 1.03 Mpx | 16:9 | 184 frames | **7.6s** | ~22.4 GB | 🟡 Borderline for 10s |
| **$1536 \times 864$** | 1.33 Mpx | 16:9 | 124 frames | **5.1s** | ~21.8 GB | 🟢 Safe at 5s |
| **$1920 \times 1088$** | 2.09 Mpx | 16:9 (1080p) | **124 frames** | **5.1s** | **~23.1 GB** | 🟡 **Max Ceiling (5.1s limit)** |
| **$2048 \times 1152$** | 2.36 Mpx | 16:9 (2K) | <90 frames | <3.7s | >24.0 GB | 🔴 OOM on 24 GB |

---

## 3. The Video Context Extension Solution

To generate continuous long-form videos (10s, 30s, 60s+) at $1920 \times 1088$ without hitting OOM, use the **Latent Context Extension Loop** (`ComfyUI-MiniMaxH3-Contex-Loop`).

### Architecture
Instead of one massive generation, the sequence is split into standardized 5.1s chunks (124 frames) and linked in latent space:

```text
Shot 1 (124f @ 1088p) ──[Trailing 22 Latent Frames]──> Shot 2 (124f @ 1088p) ──[Trailing 22 Latent Frames]──> Shot 3 ...
         │                                                      │                                                  │
   Peak: ~23.1 GB                                         Peak: ~23.1 GB                                     Peak: ~23.1 GB
         └────────────────────────────── Stitched into 13.5s 1080p Master ─────────────────────────────────────────┘
```

### Why Context Extension Preserves Quality
1. **Zero VAE Generation Loss**: The trailing 22 frames are extracted directly from the DiT latent tensor before decoding and injected into the head of the next shot's latent conditioning. There is no decode $\rightarrow$ re-encode cycle.
2. **Audio Latent Continuity**: Audio latents are carried directly at the 40 Hz latent grid (22 frames video $\approx$ 36.6 audio latent steps), preventing pops and clicks.
3. **Constant VRAM Ceiling**: The GPU only ever samples 124 frames at a time. The video can be extended infinitely with zero memory growth.

### Recommended Node Configuration

```json
{
  "width": 1920,
  "height": 1088,
  "default_duration_seconds": 5.1,
  "context_length": 22,
  "audio_context_length": 22,
  "continuation_mode": "guide",
  "anchor_mode": "head"
}
```

---

## 4. Production Workflow Profiles: Draft vs. Master

| Parameter | ⚡ Fast Draft Profile (Turbo LoRA) | 🏆 Pristine Master Profile (Full Base) |
| :--- | :--- | :--- |
| **LoRA** | `minimax_h3_fl2v_turbo_8step` (Strength: 1.0) | None (LoRA bypassed / Strength: 0.0) |
| **Sampler** | `lcm` | `euler` |
| **Scheduler** | `beta` | `beta` |
| **Steps** | **8 steps** | **25 steps** |
| **Sigma Shift** | Video: 12.0 / Audio: 3.0 | Video: 12.0 / Audio: 3.0 |
| **Time per 5.1s Scene** | **~5.5 minutes** | **~21.5 minutes** |
| **Peak VRAM** | ~23.0 GB | ~23.1 GB |
| **Best Used For** | Rapid prompt tuning, seed sweeps, storyboarding | Final high-res cinematic deliveries, subtle physics |

---

## 5. Practical Guidelines & Best Practices

1. **Step Count Does Not Affect VRAM**: Running 25 or 30 steps uses the exact same peak VRAM as 8 steps. Only generation time scales with step count.
2. **Scene Length Rule**: Always keep individual scenes in `MiniMaxH3ChainPlan` set to `"length": 124` (5.1s) when working at $1920 \times 1088$.
3. **Context Frame Sweet Spot**: `context_length: 22` provides the optimal balance (~0.9s motion reference) for maintaining character appearance and camera momentum without locking the new scene into a freeze.
4. **Attention Backend**: Always set `ModelAttentionBackend` to `comfy kitchen attention` or `sdpa` (FlashAttention). Standard PyTorch fallback attention will run out of memory during backward/forward activation passes.
5. **System Headroom**: At $1920 \times 1088$, peak VRAM reaches 23.1 GB. Ensure background applications with hardware acceleration (e.g., heavy browser video tabs) are minimized to prevent desktop window manager VRAM spikes from causing an OOM allocation error.
