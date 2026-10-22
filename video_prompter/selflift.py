"""SelfLift progressive-resolution rendering for MiniMax H3.

Runs early denoising steps at low resolution (where H3 obeys complex camera moves,
character blocking, and timing), lifts the clean latent via the 3D latent upscaler,
and finishes at full target resolution (720p or 1080p). Also applies Temporal
State Transport (TST) for cross-frame consistency.
"""

from __future__ import annotations

from typing import Any

SELFLIFT_SAMPLER_NODE = "SelfLiftH3Sampler"
SELFLIFT_TST_NODE = "SelfLiftH3TST"
DEFAULT_UPSCALER_MODEL = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
AUDIO_DECODE_NODE = "VAEDecodeAudio"
VIDEO_DECODE_NODE = "VAEDecode"
KSAMPLER_SELECT_NODE = "KSamplerSelect"
GUIDER_NODE = "BasicGuider"


class SelfLiftError(ValueError):
    """This workflow cannot be transformed into a SelfLift render."""


def apply(
    workflow: dict[str, Any],
    *,
    pass1_steps: int,
    width: int,
    height: int,
    lowres_scale: float = 0.5,
    tst_tau: float = 0.2,
    upscaler_model: str = DEFAULT_UPSCALER_MODEL,
    rho: float = 0.0,
    w_min: float = 0.5,
    w_max: float = 1.0,
) -> dict[str, str]:
    """Rewrite `workflow` in place into a SelfLift progressive render.

    - Sizes the latent node feeding the sampler to `width` x `height`.
    - Patches the model with `SelfLiftH3TST` (tau = tst_tau).
    - Ensures standard Euler sampler is selected.
    - Replaces the sampling execution with `SelfLiftH3Sampler`.
    - Rewires downstream decodes and latent outputs.
    """
    audio_decode = _find(workflow, AUDIO_DECODE_NODE)
    if audio_decode is None:
        raise SelfLiftError(
            f"SelfLift needs an audio-video workflow, but found no {AUDIO_DECODE_NODE}."
        )

    sampler_id = _wire_source(workflow[audio_decode], "samples")
    if sampler_id is None or sampler_id not in workflow:
        raise SelfLiftError(
            f"The {AUDIO_DECODE_NODE} node is not wired to a sampler."
        )

    sampler = workflow[sampler_id]
    guider_id = _wire_source(sampler, "guider")
    if guider_id is None or guider_id not in workflow:
        raise SelfLiftError("Sampler is not wired to a guider node.")
    guider = workflow[guider_id]

    model_wire = guider.get("inputs", {}).get("model")
    conditioning_wire = guider.get("inputs", {}).get("conditioning")
    if not _is_wire(model_wire) or not _is_wire(conditioning_wire):
        raise SelfLiftError("Guider is missing model or conditioning connections.")

    sigmas_wire = sampler.get("inputs", {}).get("sigmas")
    latent_wire = sampler.get("inputs", {}).get("latent_image")
    if not _is_wire(sigmas_wire) or not _is_wire(latent_wire):
        raise SelfLiftError("Sampler is missing sigmas or latent_image connections.")

    # Find VAE for the video latent
    video_decode = _find(workflow, VIDEO_DECODE_NODE)
    vae_wire = None
    if video_decode and _is_wire(workflow[video_decode].get("inputs", {}).get("vae")):
        vae_wire = workflow[video_decode]["inputs"]["vae"]
    else:
        # Fallback: look for VAELoader or vvae
        for nid, node in workflow.items():
            if isinstance(node, dict) and "vvae" in nid.lower():
                vae_wire = [nid, 0]
                break

    if not vae_wire:
        raise SelfLiftError("Could not locate video VAE in workflow for SelfLift.")

    # Find or force KSamplerSelect to Euler
    ksel_id = _find(workflow, KSAMPLER_SELECT_NODE)
    if ksel_id:
        workflow[ksel_id]["inputs"]["sampler_name"] = "euler"
        sampler_wire = [ksel_id, 0]
    else:
        ksel_id = _free_id(workflow, "vp_ksel_euler")
        workflow[ksel_id] = {
            "class_type": KSAMPLER_SELECT_NODE,
            "inputs": {"sampler_name": "euler"},
            "_meta": {"title": "KSamplerSelect (Euler for SelfLift)"},
        }
        sampler_wire = [ksel_id, 0]

    # Size the node that makes the sampler's latent, and only that node. It is
    # the canvas: the H3 video nodes resize their keyframes and references to
    # it. Written as numbers even when the profile wired them from a
    # megapixel node, which still holds pass 1's small budget — following the
    # wire would render the "finished" take at the preview's size.
    canvas_id = str(latent_wire[0])
    canvas_inputs = (workflow.get(canvas_id) or {}).get("inputs")
    if not isinstance(canvas_inputs, dict) or not {"width", "height"} <= canvas_inputs.keys():
        raise SelfLiftError(
            "SelfLift needs the sampler's latent to come from a node with width "
            "and height inputs, so it can set the finished canvas."
        )
    canvas_inputs["width"] = int(width)
    canvas_inputs["height"] = int(height)

    # Extract seed
    seed = 0
    noise_wire = sampler.get("inputs", {}).get("noise")
    if _is_wire(noise_wire) and noise_wire[0] in workflow:
        noise_node = workflow[noise_wire[0]]
        seed = noise_node.get("inputs", {}).get("noise_seed", 0)

    ids = {
        "tst": _free_id(workflow, "vp_selflift_tst"),
        "sampler": _free_id(workflow, "vp_selflift_sampler"),
    }

    # Inject TST node
    workflow[ids["tst"]] = {
        "class_type": SELFLIFT_TST_NODE,
        "inputs": {
            "model": list(model_wire),
            "tau": float(tst_tau),
            "log_diagnostics": "on",
        },
        "_meta": {"title": f"H3 Temporal State Transport (tau={tst_tau})"},
    }

    # Inject SelfLift sampler
    workflow[ids["sampler"]] = {
        "class_type": SELFLIFT_SAMPLER_NODE,
        "inputs": {
            "model": [ids["tst"], 0],
            "positive": list(conditioning_wire),
            "negative": list(conditioning_wire),
            "vae": list(vae_wire),
            "latent_image": list(latent_wire),
            "sampler": sampler_wire,
            "sigmas": list(sigmas_wire),
            "seed": int(seed),
            "cfg": 1.0,
            "transition_step": int(pass1_steps),
            "lowres_scale": float(lowres_scale),
            "rho": float(rho),
            "w_min": float(w_min),
            "w_max": float(w_max),
            "upscaler_model": str(upscaler_model),
        },
        "_meta": {"title": f"SelfLift Progressive Sampler ({int(width)}x{int(height)})"},
    }

    # Repoint video decode, audio decode, and any saved latent wires
    _repoint(workflow, sampler_id, ids["sampler"], skip=set(ids.values()))

    return ids


def _find(workflow: dict[str, Any], class_type: str) -> str | None:
    return next(
        (
            nid
            for nid, node in workflow.items()
            if isinstance(node, dict) and node.get("class_type") == class_type
        ),
        None,
    )


def _wire_source(node: dict[str, Any], input_name: str) -> str | None:
    wire = (node.get("inputs") or {}).get(input_name)
    return str(wire[0]) if _is_wire(wire) else None


def _is_wire(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _repoint(
    workflow: dict[str, Any], old_id: str, new_id: str, skip: set[str]
) -> None:
    for node_id, node in workflow.items():
        if node_id in skip or not isinstance(node, dict):
            continue
        for name, value in (node.get("inputs") or {}).items():
            if _is_wire(value) and str(value[0]) == old_id:
                node["inputs"][name] = [new_id, value[1]]


def _free_id(workflow: dict[str, Any], base: str) -> str:
    if base not in workflow:
        return base
    suffix = 2
    while f"{base}_{suffix}" in workflow:
        suffix += 1
    return f"{base}_{suffix}"
