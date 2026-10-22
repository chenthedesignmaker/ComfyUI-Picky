# Agent Reference: Krea 2 Turbo Keyframe Generation for MiniMax H3

This guide provides the exact specifications, model checkpoints, ComfyUI node graph schema, and Python invocation methods for generating consistent character keyframes with **Krea 2 Turbo** and feeding them into **MiniMax H3 Image-to-Video (I2V)**.

---

## 1. Quick Python Invocation (Recommended)

Any agent working in this repo can generate a keyframe in 3 lines using the built-in [`video_prompter.t2i`](file:///c:/PROJECTS/video%20prompter/video_prompter/t2i.py) module:

```python
import json
import urllib.request
from video_prompter.t2i import prepare_t2i

# 1. Build the workflow payload
prep = prepare_t2i(
    prompt="A Japanese woman in her late 20s with short piecey dark shag hair, wearing cream pajamas, sleeping in bed",
    seed=2048,
    steps=8,
    style="pastel-dream",  # Optional: "golden-hour", "noir", "neon-noir", "kodak-2383", etc.
    width=1344,            # Native MiniMax H3 I2V resolution (or 1920x1088)
    height=768,
    supersample=1.0,
    image_prefix="image/shot_01_keyframe"
)

# 2. Dispatch to ComfyUI
req = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=json.dumps({"prompt": prep["workflow"], "client_id": "agent_worker"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
res = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
prompt_id = res["prompt_id"]
print(f"Dispatched job: {prompt_id}")
```

---

## 2. Required Models & Checkpoints

The local ComfyUI instance (`127.0.0.1:8188`) uses the following exact safetensors files located in `ComfyUI/models/`:

| Component | Node Class | Target File | Notes |
| :--- | :--- | :--- | :--- |
| **UNet / Model** | `UNETLoader` | `krea2_turbo_fp8_scaled.safetensors` | Single-stream MMDiT (Wan 2.1 latent format) |
| **Text Encoder** | `CLIPLoader` | `qwen3vl_4b_fp8_scaled.safetensors` | **CRITICAL:** `type` must be set to `"krea2"` |
| **VAE** | `VAELoader` | `Wan2_1_VAE_bf16.safetensors` | Wan 2.1 16-channel VAE |

---

## 3. Raw ComfyUI Node Graph Schema

If constructing the JSON graph manually without importing `video_prompter.t2i`:

```json
{
  "unet": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "krea2_turbo_fp8_scaled.safetensors",
      "weight_dtype": "default"
    }
  },
  "clip": {
    "class_type": "CLIPLoader",
    "inputs": {
      "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
      "type": "krea2",
      "device": "default"
    }
  },
  "vae": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "Wan2_1_VAE_bf16.safetensors"
    }
  },
  "positive": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "<YOUR_STRUCTURED_PROMPT>",
      "clip": ["clip", 0]
    }
  },
  "neg_zero": {
    "class_type": "ConditioningZeroOut",
    "inputs": {
      "conditioning": ["positive", 0]
    }
  },
  "latent": {
    "class_type": "EmptySD3LatentImage",
    "inputs": {
      "width": 1344,
      "height": 768,
      "batch_size": 1
    }
  },
  "sampler": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["unet", 0],
      "positive": ["positive", 0],
      "negative": ["neg_zero", 0],
      "latent_image": ["latent", 0],
      "seed": 1024,
      "steps": 8,
      "cfg": 1.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1.0
    }
  },
  "decode": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["sampler", 0],
      "vae": ["vae", 0]
    }
  },
  "save": {
    "class_type": "SaveImage",
    "inputs": {
      "filename_prefix": "image/keyframe_output",
      "images": ["decode", 0]
    }
  }
}
```

---

## 4. Critical Architecture Rules & Pitfalls

1. **`CLIPLoader` Type**: Must be `"krea2"`. Setting it to `"flux"` or `"sd3"` will fail or output corrupt latents.
2. **Latent Node**: Use `EmptySD3LatentImage` (or `EmptySD3LatentImage` compatible 16-channel node). Krea 2 uses Wan 2.1's 16-channel latent space.
3. **CFG & Sampling**:
   * Krea 2 Turbo is a distilled model: **CFG must be `1.0`** (guidance-free).
   * Steps: **8 steps** is the quality sweet spot (more steps do not add detail).
   * Sampler: `euler`, Scheduler: `simple`.
   * Negative conditioning: Pass positive conditioning through `ConditioningZeroOut` into KSampler's negative input.
4. **Resolution**:
   * For direct MiniMax H3 I2V starting images: use **1344 × 768** (or any multiple of 16/32).
   * For crisp master keyframes: use **1920 × 1088** with supersampling if needed.

---

## 5. Character Consistency Prompting Recipe (VLM Anchoring)

Because Krea 2 uses **Qwen3-VL**, it interprets semantic character hierarchies with extreme precision. Do **not** use vague references like *"the same person"*.

### The 3-Part Prompt Formula:
$$\text{Full Prompt} = \text{[Master Character Anchor]} + \text{[Scene Action \& Environment]} + \text{[Cinematic Style Preset]}$$

* **Part 1 - Master Character Anchor (Keep 100% Identical across all shots):**
  > *"A Japanese woman in her late 20s with short piecey dark shag hair and soft approachable facial features, wearing loose cream-colored cotton pajamas with subtle whimsical cartoon animal prints..."*
* **Part 2 - Scene Delta (Change per shot):**
  * *Shot 1:* *"...sleeping peacefully on her side in bed, head nestled in a white cotton pillow under a linen duvet, morning dawn light."*
  * *Shot 2:* *"...stepping into a kitchen, pouring coffee from an electric kettle into a white ceramic mug, warm window sunbeams."*
* **Part 3 - Cinematic Style Preset:**
  * e.g., `"golden-hour"`, `"pastel-dream"`, `"kodak-2383"`, `"noir"`.

---

## 6. MiniMax H3 Image-to-Video (I2V) Handoff

Once the image is generated and saved under `ComfyUI/output/image/<filename>.png`:

1. Copy or adopt the image into ComfyUI's input directory (e.g. `video_prompter/<filename>.png`).
2. In your Video Prompter project segment definition:
   ```json
   {
     "first_frame": "video_prompter/test_krea2_sleeping_in_bed_00001_.png",
     "last_frame": "",
     "overview": "The woman slowly awakens as soft sunlight touches the duvet",
     "beats": [...]
   }
   ```
3. When dispatched, [`video_prompter/imagery.py`](file:///c:/PROJECTS/video%20prompter/video_prompter/imagery.py#L109-L147) will automatically augment the H3 profile with `LoadImage` and wire it directly to `MiniMaxH3ImageToVideo.first_frame`.
