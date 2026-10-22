"""Text-to-image via Krea 2 Turbo, for generating keyframes.

The video prompter workflow is: describe a scene → generate a still → feed it
into H3 image-to-video. This module builds the text-to-image workflow, using a
Krea 2 Turbo model for fast (~4 step) generation on a local GPU.

Krea 2 architecture (from ComfyUI's ``comfy/ldm/krea2``):
  - Single-stream MMDiT with Qwen3-VL-4B text encoder (12-layer stack)
  - CLIPLoader type ``krea2`` — *not* ``flux`` or ``flux2``
  - Wan 2.1 latent format (16-channel, same EmptySD3LatentImage as SD3)
  - Wan 2.1 VAE for decode

The turbo variant is distilled to 2–4 steps at cfg 1.0 (guidance-free).
"""

from __future__ import annotations

from typing import Any

# Model defaults — local paths as they appear in ComfyUI's model directories.
DEFAULT_MODELS: dict[str, str] = {
    "unet": "krea2_turbo_fp8_scaled.safetensors",
    "clip": "qwen3vl_4b_fp8_scaled.safetensors",
    "vae": "Wan2_1_VAE_bf16.safetensors",
}

# Delivery canvas — the size of the saved image. Larger than H3's i2v canvas
# (1344×768) for a crisper standalone keyframe; when adopted as an i2v first
# frame it is resized to the shot's canvas, a harmless downscale.
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1088

# Turbo-distilled: few steps, no classifier-free guidance. 8 is the quality
# sweet spot for this distilled model — more plateaus, cfg stays at 1.
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
SAMPLER_NAME = "euler"
SCHEDULER = "simple"

# Supersampling: render at delivery × factor, then lanczos-downscale to the
# delivery size. >1 packs more true detail into backgrounds and anti-aliases
# edges. 1.0 is off. The render canvas is snapped to a multiple of 16 (Wan
# latent requirement). 1.5 (render 2880×1632 → 1920×1088) tested clean on the
# distilled turbo model with no composition duplication; note that changing the
# factor changes the latent size and therefore re-rolls the composition, so a
# consistent factor is used across seed-hunting and the final render.
DEFAULT_SUPERSAMPLE = 1.5
DOWNSCALE_METHOD = "lanczos"


def _round16(n: int) -> int:
    """Snap a dimension to the nearest positive multiple of 16."""
    return max(16, int(round(n / 16.0)) * 16)

# SaveImage puts files under this prefix in ComfyUI's output directory.
OUTPUT_PREFIX = "image/VideoPrompter_t2i"

# Cinematic style bundles — camera / film-stock / lighting / grade / mood
# language appended to the user's prompt to steer the *look* of a keyframe
# without touching what they described. Our own bundles (not a copy of any node
# pack). `styled_prompt` joins the suffix onto the prompt; the original text is
# always preserved and comes first. Keys are stable, lowercase, kebab-case.
STYLE_PRESETS: dict[str, str] = {
    "kodak-2383": (
        "shot on 35mm film, Kodak 2383 print stock, warm teal-and-orange grade, "
        "rich filmic contrast, subtle halation, fine grain, cinematic"
    ),
    "noir": (
        "high-contrast black and white, hard chiaroscuro lighting, deep shadows, "
        "venetian-blind slats of light, 1940s film noir, moody and dramatic"
    ),
    "neon-noir": (
        "neon-noir, wet reflective streets, saturated magenta and cyan practicals, "
        "volumetric haze, anamorphic bokeh, low-key cinematic lighting"
    ),
    "golden-hour": (
        "golden hour, warm low-angle sunlight, long soft shadows, gentle lens flare, "
        "hazy backlight, natural filmic color, cinematic"
    ),
    "vintage-16mm": (
        "shot on vintage 16mm film, visible grain, slightly faded muted colors, "
        "soft focus falloff, gate weave, nostalgic documentary feel"
    ),
    "anamorphic": (
        "anamorphic widescreen, 2.39:1 framing, horizontal blue lens flares, "
        "oval bokeh, shallow depth of field, cinematic scope"
    ),
    "bleach-bypass": (
        "bleach bypass process, desaturated silver-retention look, crushed blacks, "
        "high contrast, gritty metallic highlights, cold cinematic tone"
    ),
    "pastel-dream": (
        "soft pastel palette, diffused dreamy lighting, gentle bloom, low contrast, "
        "airy and ethereal, delicate film grain"
    ),
    "documentary": (
        "naturalistic documentary photography, available light, handheld candid framing, "
        "realistic color, unstyled and authentic"
    ),
    "moody-thriller": (
        "moody thriller lighting, teal shadows and amber highlights, single hard key light, "
        "deep negative fill, tense atmosphere, cinematic grade"
    ),
}


def style_names() -> list[str]:
    """The available cinematic style preset keys, sorted."""
    return sorted(STYLE_PRESETS)


def styled_prompt(prompt: str, style: str | None) -> str:
    """Append a cinematic style bundle to the prompt, keeping the original first.

    ``style`` is a key in ``STYLE_PRESETS``; ``None`` or ``""`` leaves the prompt
    untouched. An unknown key raises ``ValueError`` so callers can surface it.
    """
    if not style:
        return prompt
    suffix = STYLE_PRESETS.get(style)
    if suffix is None:
        raise ValueError(
            f"Unknown style {style!r}. Available: {', '.join(style_names())}."
        )
    prompt = prompt.strip()
    return f"{prompt}, {suffix}" if prompt else suffix


def _node(class_type: str, inp: dict[str, Any], title: str | None = None) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inp, "_meta": {"title": title or class_type}}


def _normalize_loras(loras: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Validate and normalize the LoRA stack, dropping disabled entries.

    Each entry needs a ``name``; ``strength_model``/``strength_clip`` default to
    1.0, and ``enabled`` defaults to True. Raises ``ValueError`` on a bad shape.
    """
    if not loras:
        return []
    out: list[dict[str, Any]] = []
    for i, lora in enumerate(loras):
        if not isinstance(lora, dict):
            raise ValueError(f"LoRA #{i} must be an object, got {type(lora).__name__}.")
        if not lora.get("enabled", True):
            continue
        name = lora.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"LoRA #{i} needs a `name` string.")
        out.append({
            "name": name.strip(),
            "strength_model": float(lora.get("strength_model", 1.0)),
            "strength_clip": float(lora.get("strength_clip", 1.0)),
        })
    return out


def build_t2i_workflow(
    prompt: str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int = 0,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    style: str | None = None,
    loras: list[dict[str, Any]] | None = None,
    supersample: float = DEFAULT_SUPERSAMPLE,
    models: dict[str, str] | None = None,
    image_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """A ComfyUI API-format workflow for text-to-image with Krea 2 Turbo.

    ``width``/``height`` are the *delivery* size of the saved image (default
    1920×1088). ``style`` names a cinematic bundle from ``STYLE_PRESETS``
    appended to the prompt for a film look. ``loras`` is a stack of
    ``{name, strength_model, strength_clip, enabled}`` entries patched onto the
    model and CLIP before sampling.

    ``supersample`` > 1 renders at ``width×height × factor`` (snapped to /16)
    and lanczos-downscales to the delivery size for sharper background detail;
    push it too high and the distilled model duplicates composition.

    Returns the node graph dict ready for ``/prompt``.
    """
    m = {**DEFAULT_MODELS, **(models or {})}
    prompt = styled_prompt(prompt, style)
    stack = _normalize_loras(loras)

    ss = max(1.0, float(supersample))
    render_width = _round16(width * ss) if ss > 1.0 else width
    render_height = _round16(height * ss) if ss > 1.0 else height

    g: dict[str, Any] = {
        "unet": _node("UNETLoader",
                       {"unet_name": m["unet"], "weight_dtype": "default"},
                       "Krea 2 Turbo"),
        "clip": _node("CLIPLoader",
                       {"clip_name": m["clip"], "type": "krea2", "device": "default"},
                       "Qwen3-VL-4B (Krea 2)"),
        "vae":  _node("VAELoader",
                       {"vae_name": m["vae"]},
                       "Wan 2.1 VAE"),
    }

    # LoRA stack — each LoraLoader patches both the model and the CLIP, chaining
    # off the previous one. Downstream nodes read the final model/clip refs.
    model_ref: list[Any] = ["unet", 0]
    clip_ref: list[Any] = ["clip", 0]
    for i, lora in enumerate(stack):
        key = f"lora_{i}"
        g[key] = _node("LoraLoader", {
            "lora_name": lora["name"],
            "strength_model": lora["strength_model"],
            "strength_clip": lora["strength_clip"],
            "model": model_ref,
            "clip": clip_ref,
        }, f"LoRA: {lora['name']}")
        model_ref = [key, 0]
        clip_ref = [key, 1]

    # Text encoding + zeroed-out negative (turbo runs at cfg 1.0, but the
    # KSampler still needs both slots wired).
    g["positive"] = _node("CLIPTextEncode",
                          {"text": prompt, "clip": clip_ref},
                          "Prompt")
    g["neg_zero"] = _node("ConditioningZeroOut",
                          {"conditioning": ["positive", 0]},
                          "Negative (zero)")

    # Latent canvas — Krea 2 uses Wan 2.1's 16-channel latent space, same node
    # as SD3 / FLUX.1 / Wan. Sized to the render (supersampled) resolution.
    g["latent"] = _node("EmptySD3LatentImage",
                        {"width": render_width, "height": render_height, "batch_size": 1},
                        "Canvas")

    # Sampler
    g["sampler"] = _node("KSampler", {
        "model": model_ref,
        "positive": ["positive", 0],
        "negative": ["neg_zero", 0],
        "latent_image": ["latent", 0],
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": SAMPLER_NAME,
        "scheduler": SCHEDULER,
        "denoise": 1.0,
    }, "KSampler")

    # Decode, then (if supersampling) lanczos-downscale to the delivery size.
    g["decode"] = _node("VAEDecode",
                        {"samples": ["sampler", 0], "vae": ["vae", 0]},
                        "Decode")
    images_ref: list[Any] = ["decode", 0]
    if render_width != width or render_height != height:
        g["downscale"] = _node("ImageScale", {
            "image": ["decode", 0],
            "upscale_method": DOWNSCALE_METHOD,
            "width": width,
            "height": height,
            "crop": "disabled",
        }, f"Downscale → {width}×{height}")
        images_ref = ["downscale", 0]

    g["save"]   = _node("SaveImage",
                        {"filename_prefix": image_prefix, "images": images_ref},
                        "Save Image")

    return g


def prepare_t2i(
    prompt: str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int | None = None,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    style: str | None = None,
    loras: list[dict[str, Any]] | None = None,
    supersample: float = DEFAULT_SUPERSAMPLE,
    models: dict[str, str] | None = None,
    image_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Build a t2i workflow and return everything the editor needs to submit it.

    Mirrors ``render.PreparedRender.to_dict()`` in shape so the editor can
    treat the payload the same way: POST the ``workflow`` to ``/prompt``,
    record the result.
    """
    from .profiles import random_seed

    seed = random_seed() if seed is None else int(seed)
    ss = max(1.0, float(supersample))
    render_width = _round16(width * ss) if ss > 1.0 else width
    render_height = _round16(height * ss) if ss > 1.0 else height
    workflow = build_t2i_workflow(
        prompt,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg=cfg,
        style=style,
        loras=loras,
        supersample=ss,
        models=models,
        image_prefix=image_prefix,
    )

    return {
        "workflow": workflow,
        "prompt": prompt,
        "width": width,
        "height": height,
        "render_width": render_width,
        "render_height": render_height,
        "supersample": ss,
        "seed": seed,
        "steps": steps,
        "style": style,
        "loras": _normalize_loras(loras),
        "mode": "t2i",
    }
