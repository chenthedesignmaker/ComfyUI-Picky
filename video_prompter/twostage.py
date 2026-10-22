"""Two-stage rendering: prove the frame small, then finish it big.

A single-pass H3 render has to pick one canvas and live with it. Small obeys the
prompt but looks soft; 720p looks right but drops instructions (see `quality`).
The two-stage pass refuses the choice. It samples most of the schedule at the
small, obedient canvas, upscales the *clean* latent — the sampler's
`denoised_output`, not its noisy working state — and spends the last few steps at
720p. The composition, the timing and the fine text are decided where the model
listens; only surface detail is added where the pixels are.

The mechanism is a split schedule. `SplitSigmas` cuts one sigma run in two, so
both passes ride the same denoise curve rather than being two unrelated renders:
pass 2 resumes at the sigma pass 1 stopped at, which works out to roughly a
quarter of the noise. Injecting more (a fresh half-noise pass, say) is what
destroys LED digits and small text — the model re-invents them instead of
sharpening them.

Between the passes the AV latent has to come apart. H3's latent carries video and
audio together, and only the video half can be upscaled, so the graph separates
them, upscales the video, and concatenates the untouched audio back on.

This is written as a **transform over a finished workflow**, not as a captured
profile, for the reason `imagery` builds its ref2v graph in code: node ids differ
per mode and per user capture, so anything that hardcodes them works for exactly
one graph. Everything here is found by class and by wire, so one implementation
covers t2v, i2v, ref2v and an extend alike.
"""

from __future__ import annotations

from typing import Any

# The upscaler ships as a custom node (Comfyui_Minimax_h3_latent_Upscaler) with
# its weights under `models/latent_upscale_models`.
UPSCALER_NODE = "MinimaxH3LatentUpscaler3D"
DEFAULT_UPSCALER_MODEL = "minimax_h3_latent_upscaler_3d_fp16.safetensors"

# The AV latent splitter/joiner are core ComfyUI (`comfy_extras/nodes_lt.py`),
# not a custom node, so nothing extra has to be installed for this path.
SEPARATE_AV_NODE = "LTXVSeparateAVLatent"
CONCAT_AV_NODE = "LTXVConcatAVLatent"

SAMPLER_NODE = "SamplerCustomAdvanced"
AUDIO_DECODE_NODE = "VAEDecodeAudio"

# The upscaler aligns its target in pixel space; 32 is the node's own default and
# what stops a light band along the bottom edge.
ALIGN = 32

# Slot 1 of SamplerCustomAdvanced is `denoised_output` — the model's estimate of
# the finished frame at the step it stopped on, with the working noise already
# subtracted. Upscaling slot 0 instead would magnify that noise, and pass 2 would
# spend its few steps cleaning up after the upscaler rather than adding detail.
DENOISED_SLOT = 1

DEFAULT_SINGULARITY_UNET = "MiniMax-H3\\Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
DEFAULT_TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
DEFAULT_DENOISE2 = 0.35
DEFAULT_PASS2_STEPS = 4


class TwoStageError(ValueError):
    """This workflow cannot be split into two passes.

    Raised rather than quietly rendering single-pass: the gallery reads a take's
    tier back off the canvas and step count it recorded, so a render that claimed
    to be two-stage and was not would misreport itself for good.
    """


def apply(
    workflow: dict[str, Any],
    *,
    pass1_steps: int,
    width: int,
    height: int,
    upscaler_model: str = DEFAULT_UPSCALER_MODEL,
    align: int = ALIGN,
) -> dict[str, str]:
    """Rewrite `workflow` in place into a two-pass render, returning its new ids.

    `pass1_steps` is where the schedule is cut, counted in steps of the run the
    workflow already asks for — so a 20-step schedule cut at 15 gives 15 + 5.
    `width`/`height` are the finished size in pixels; pass 1 keeps whatever canvas
    the workflow was already set up for.

    Everything downstream of the first sampler is re-pointed at the second, so a
    graph that saved its AV latent for a later Extend goes on saving one — the
    finished, upscaled latent rather than the small first-pass one.
    """
    audio_decode = _find(workflow, AUDIO_DECODE_NODE)
    if audio_decode is None:
        raise TwoStageError(
            f"Two-stage needs a combined audio+video latent to split, and this "
            f"workflow has no {AUDIO_DECODE_NODE}. Render it single-pass instead."
        )

    sampler_id = _wire_source(workflow[audio_decode], "samples")
    if sampler_id is None or sampler_id not in workflow:
        raise TwoStageError(
            f"The {AUDIO_DECODE_NODE} node is not wired to a sampler, so there is "
            f"no first pass to split."
        )

    sampler = workflow[sampler_id]
    if sampler.get("class_type") != SAMPLER_NODE:
        raise TwoStageError(
            f"Two-stage reads the first pass's clean latent off {SAMPLER_NODE}'s "
            f"`denoised_output`, but this workflow samples with "
            f"{sampler.get('class_type')!r}. Re-capture the profile with "
            f"{SAMPLER_NODE}."
        )

    sigmas = (sampler.get("inputs") or {}).get("sigmas")
    if not _is_wire(sigmas):
        raise TwoStageError(
            "The sampler's sigmas are not wired to a scheduler, so there is no "
            "schedule to cut in two."
        )

    if _find(workflow, UPSCALER_NODE) is not None:
        raise TwoStageError("This workflow is already a two-stage render.")

    steps = int(pass1_steps)
    if steps < 1:
        raise TwoStageError(f"Pass 1 needs at least one step, got {steps}.")

    ids = {
        "split": _free_id(workflow, "vp_split"),
        "separate": _free_id(workflow, "vp_sep_av"),
        "upscale": _free_id(workflow, "vp_upscale"),
        "concat": _free_id(workflow, "vp_concat_av"),
        "sampler2": _free_id(workflow, "vp_samp2"),
    }

    workflow[ids["split"]] = {
        "class_type": "SplitSigmas",
        "inputs": {"sigmas": list(sigmas), "step": steps},
        "_meta": {"title": f"Split sigmas (pass 1 = {steps} steps)"},
    }
    workflow[ids["separate"]] = {
        "class_type": SEPARATE_AV_NODE,
        "inputs": {"av_latent": [sampler_id, DENOISED_SLOT]},
        "_meta": {"title": "Separate video from audio"},
    }
    workflow[ids["upscale"]] = {
        "class_type": UPSCALER_NODE,
        "inputs": {
            "latent": [ids["separate"], 0],
            "model_name": upscaler_model,
            # "target dimensions" is the only mode that takes an exact canvas.
            # Its `scale`/`megapixels` inputs are ignored there but must still be
            # present, because ComfyUI validates every declared input.
            "mode": "target dimensions",
            "width": int(width),
            "height": int(height),
            "scale": 2.0,
            "megapixels": 1.0,
            "align": int(align),
            "enable_temporal_chunking": True,
            "force_unload": True,
            "device": "cuda",
            "precision": "fp16",
        },
        "_meta": {"title": f"Upscale latent to {int(width)}x{int(height)}"},
    }
    workflow[ids["concat"]] = {
        "class_type": CONCAT_AV_NODE,
        "inputs": {
            "video_latent": [ids["upscale"], 0],
            "audio_latent": [ids["separate"], 1],
        },
        "_meta": {"title": "Rejoin video and audio"},
    }
    # Pass 2 is the same sampler with the same noise and the same guider: only
    # where it starts on the schedule, and what it starts from, differ.
    workflow[ids["sampler2"]] = {
        "class_type": SAMPLER_NODE,
        "inputs": {
            **{
                name: value
                for name, value in (sampler.get("inputs") or {}).items()
                if name not in ("sigmas", "latent_image")
            },
            "sigmas": [ids["split"], 1],
            "latent_image": [ids["concat"], 0],
        },
        "_meta": {"title": "Sampler (pass 2, refine)"},
    }

    # Pass 1 now runs only the head of the schedule.
    sampler["inputs"]["sigmas"] = [ids["split"], 0]

    # Anything that read the first pass's finished latent should read the second
    # one's: both decodes, and the Extend latent save if the graph has one.
    _repoint(workflow, sampler_id, ids["sampler2"], skip=set(ids.values()))

    return ids


def apply_singularity_2pass(
    workflow: dict[str, Any],
    *,
    pass1_steps: int = 20,
    width: int,
    height: int,
    pass2_steps: int = DEFAULT_PASS2_STEPS,
    denoise2: float = DEFAULT_DENOISE2,
    singularity_unet: str = DEFAULT_SINGULARITY_UNET,
    turbo_lora: str = DEFAULT_TURBO_LORA,
    upscaler_model: str = DEFAULT_UPSCALER_MODEL,
    align: int = ALIGN,
) -> dict[str, str]:
    """Rewrite `workflow` in place into a Singularity 2-pass render, returning its new ids.

    Pass 1: `pass1_steps` (default 20) steps on the pure base model with Turbo LoRA
    bypassed (strength 0.0), at pass 1's canvas.
    Between passes: clean latent (`denoised_output`) upscaled to `width` x `height`
    via `MinimaxH3LatentUpscaler3D`.
    Pass 2: `pass2_steps` (default 4) steps with the 8-step turbo LoRA at `denoise2`
    (default 0.35).
    UNET: Swapped to `singularity_unet` (Singularity Ref2va pruned int8).
    """
    audio_decode = _find(workflow, AUDIO_DECODE_NODE)
    if audio_decode is None:
        raise TwoStageError(
            f"Two-stage needs a combined audio+video latent to split, and this "
            f"workflow has no {AUDIO_DECODE_NODE}. Render it single-pass instead."
        )

    sampler_id = _wire_source(workflow[audio_decode], "samples")
    if sampler_id is None or sampler_id not in workflow:
        raise TwoStageError(
            f"The {AUDIO_DECODE_NODE} node is not wired to a sampler, so there is "
            f"no first pass to split."
        )

    sampler = workflow[sampler_id]
    if sampler.get("class_type") != SAMPLER_NODE:
        raise TwoStageError(
            f"Two-stage reads the first pass's clean latent off {SAMPLER_NODE}'s "
            f"`denoised_output`, but this workflow samples with "
            f"{sampler.get('class_type')!r}. Re-capture the profile with "
            f"{SAMPLER_NODE}."
        )

    sigmas = (sampler.get("inputs") or {}).get("sigmas")
    if not _is_wire(sigmas):
        raise TwoStageError(
            "The sampler's sigmas are not wired to a scheduler, so there is no "
            "schedule to cut in two."
        )

    if _find(workflow, UPSCALER_NODE) is not None:
        raise TwoStageError("This workflow is already a two-stage render.")

    steps = int(pass1_steps)
    if steps < 1:
        raise TwoStageError(f"Pass 1 needs at least one step, got {steps}.")

    p2_steps = int(pass2_steps)
    if p2_steps < 1:
        raise TwoStageError(f"Pass 2 needs at least one step, got {p2_steps}.")

    # Identify or set UNET
    unet_id = _find(workflow, "UNETLoader")
    if unet_id is not None:
        if singularity_unet:
            workflow[unet_id].setdefault("inputs", {})["unet_name"] = singularity_unet

    # Find guider and shift
    guider_wire = (sampler.get("inputs") or {}).get("guider")
    guider_id = str(guider_wire[0]) if _is_wire(guider_wire) else None
    guider = workflow.get(guider_id) if guider_id else None

    shift_wire = (guider.get("inputs") or {}).get("model") if guider else None
    shift_id = str(shift_wire[0]) if _is_wire(shift_wire) else None
    candidate_shift = workflow.get(shift_id) if shift_id else None
    shift = (
        candidate_shift
        if candidate_shift and candidate_shift.get("class_type") == "MiniMaxH3SigmaShift"
        else None
    )

    # Trace upstream to find any pass-1 LoRA and base model wire
    model_upstream = (
        (shift.get("inputs") or {}).get("model")
        if shift
        else (candidate_shift.get("inputs") or {}).get("model") if candidate_shift else None
    )

    existing_lora_node = None
    if candidate_shift and candidate_shift.get("class_type") == "LoraLoaderModelOnly":
        existing_lora_node = candidate_shift
    elif _is_wire(model_upstream) and model_upstream[0] in workflow:
        lora_node = workflow[model_upstream[0]]
        if lora_node.get("class_type") == "LoraLoaderModelOnly":
            existing_lora_node = lora_node

    if existing_lora_node:
        existing_lora_node.setdefault("inputs", {})["strength_model"] = 0.0
        lora_model = (existing_lora_node.get("inputs") or {}).get("model")
        base_model_wire = list(lora_model) if _is_wire(lora_model) else None
        existing_lora_name = (existing_lora_node.get("inputs") or {}).get("lora_name")
    else:
        base_model_wire = None
        existing_lora_name = None

    if base_model_wire is None:
        if shift:
            s_model = (shift.get("inputs") or {}).get("model")
            base_model_wire = list(s_model) if _is_wire(s_model) else None
        elif guider:
            g_model = (guider.get("inputs") or {}).get("model")
            base_model_wire = list(g_model) if _is_wire(g_model) else None

    if base_model_wire is None:
        base_model_wire = [unet_id, 0] if unet_id is not None else ["unet", 0]

    target_lora = turbo_lora
    if turbo_lora == DEFAULT_TURBO_LORA and existing_lora_name:
        target_lora = existing_lora_name

    # Configure Pass 1 scheduler
    sched_id = str(sigmas[0])
    if sched_id in workflow:
        workflow[sched_id].setdefault("inputs", {})["steps"] = steps
        workflow[sched_id]["inputs"]["denoise"] = 1.0

    ids = {
        "separate": _free_id(workflow, "vp_sep_av"),
        "upscale": _free_id(workflow, "vp_upscale"),
        "concat": _free_id(workflow, "vp_concat_av"),
        "lora2": _free_id(workflow, "vp_lora2"),
        "shift2": _free_id(workflow, "vp_shift2"),
        "guider2": _free_id(workflow, "vp_guider2"),
        "sched2": _free_id(workflow, "vp_sched2"),
        "sampler2": _free_id(workflow, "vp_samp2"),
    }

    # Upscaler chain
    workflow[ids["separate"]] = {
        "class_type": SEPARATE_AV_NODE,
        "inputs": {"av_latent": [sampler_id, DENOISED_SLOT]},
        "_meta": {"title": "Separate video from audio"},
    }
    workflow[ids["upscale"]] = {
        "class_type": UPSCALER_NODE,
        "inputs": {
            "latent": [ids["separate"], 0],
            "model_name": upscaler_model,
            "mode": "target dimensions",
            "width": int(width),
            "height": int(height),
            "scale": 2.0,
            "megapixels": 1.0,
            "align": int(align),
            "enable_temporal_chunking": True,
            "force_unload": True,
            "device": "cuda",
            "precision": "fp16",
        },
        "_meta": {"title": f"Upscale latent to {int(width)}x{int(height)}"},
    }
    workflow[ids["concat"]] = {
        "class_type": CONCAT_AV_NODE,
        "inputs": {
            "video_latent": [ids["upscale"], 0],
            "audio_latent": [ids["separate"], 1],
        },
        "_meta": {"title": "Rejoin video and audio"},
    }

    # Pass 2 model chain: Turbo LoRA -> Shift -> Guider
    workflow[ids["lora2"]] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "lora_name": target_lora,
            "strength_model": 1.0,
            "model": list(base_model_wire),
        },
        "_meta": {"title": "Turbo LoRA (pass 2)"},
    }

    shift2_inputs = dict((shift.get("inputs") or {}) if shift else {})
    shift2_inputs["model"] = [ids["lora2"], 0]
    shift2_inputs["shift_video"] = float(shift2_inputs.get("shift_video", 12.0))
    shift2_inputs["shift_audio"] = float(shift2_inputs.get("shift_audio", 3.0))
    shift2_inputs.pop("shift", None)
    workflow[ids["shift2"]] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": shift2_inputs,
        "_meta": {"title": "Shift (pass 2)"},
    }

    guider_inputs = dict((guider.get("inputs") or {}) if guider else {})
    guider_inputs["model"] = [ids["shift2"], 0]
    if "conditioning" not in guider_inputs and guider:
        guider_inputs["conditioning"] = list((guider.get("inputs") or {}).get("conditioning", []))
    workflow[ids["guider2"]] = {
        "class_type": guider.get("class_type", "BasicGuider") if guider else "BasicGuider",
        "inputs": guider_inputs,
        "_meta": {"title": "Guider (pass 2)"},
    }

    # Pass 2 scheduler: p2_steps @ denoise2
    sched_node = workflow.get(sched_id)
    sched_type = sched_node.get("inputs", {}).get("scheduler", "simple") if sched_node else "simple"
    workflow[ids["sched2"]] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "scheduler": sched_type,
            "steps": p2_steps,
            "denoise": float(denoise2),
            "model": [ids["shift2"], 0],
        },
        "_meta": {"title": f"{p2_steps} steps @ {denoise2}"},
    }

    # Pass 2 sampler
    workflow[ids["sampler2"]] = {
        "class_type": SAMPLER_NODE,
        "inputs": {
            **{
                name: value
                for name, value in (sampler.get("inputs") or {}).items()
                if name not in ("sigmas", "latent_image", "guider")
            },
            "guider": [ids["guider2"], 0],
            "sigmas": [ids["sched2"], 0],
            "latent_image": [ids["concat"], 0],
        },
        "_meta": {"title": "Sampler (pass 2, Singularity)"},
    }

    # Repoint downstream decodes and latent saves to sampler2
    _repoint(workflow, sampler_id, ids["sampler2"], skip=set(ids.values()))

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
    """Send every `[old_id, 0]` reference to `[new_id, 0]` instead.

    Only slot 0 moves. Slot 1 is the clean latent the upscaler feeds on, and that
    still has to come from pass 1.
    """
    for node_id, node in workflow.items():
        if node_id in skip or not isinstance(node, dict):
            continue
        for name, value in (node.get("inputs") or {}).items():
            if _is_wire(value) and str(value[0]) == old_id and _slot(value[1]) == 0:
                node["inputs"][name] = [new_id, 0]


def _slot(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _free_id(workflow: dict[str, Any], base: str) -> str:
    if base not in workflow:
        return base
    suffix = 2
    while f"{base}_{suffix}" in workflow:
        suffix += 1
    return f"{base}_{suffix}"
