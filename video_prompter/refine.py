"""Build a ComfyUI workflow that refines an existing take.

The workflow loads a take's rendered video and audio via LoadVideo,
optionally runs a guided resample pass for texture sharpening and/or upscaling,
and optionally runs a face refinement pass. The result is a new take linked
to the original.

Unlike `prepare_render`, this does not start from a profile template: the
workflow is built from scratch because the inputs and node graph are
structurally different from a generation pipeline.

This module is deliberately kept free of aiohttp and of ComfyUI internals:
it produces a plain ``dict[str, Any]`` workflow that the route layer submits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Model weight defaults ───────────────────────────────────────────────────
DEFAULT_REMIX_MODEL = "MiniMax-H3/minimax_h3_fl2va_ref2va_true_hybrid_b25_49.safetensors"
DEFAULT_SECOND_MODEL = "MiniMax-H3/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
DEFAULT_TEXT_ENCODER = "MiniMax-H3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_TAEH3 = "taeh3.safetensors"
DEFAULT_FACE_UPSCALER = "RealESRGAN_x4plus.safetensors"

# Guided resample defaults
GUIDE_PROMPT = (
    "Enhance this video with sharp, crisp details while preserving "
    "a natural photorealistic appearance."
)
GUIDE_STEPS = 4
GUIDE_SAMPLER = "res_multistep"
GUIDE_SCHEDULER = "simple"
GUIDE_SIGMA_SHIFT_VIDEO = 12
GUIDE_SIGMA_SHIFT_AUDIO = 3

# Face refine defaults
FACE_DETECTOR = "face_yolov8m.pt"
FACE_DENOISE = 0.18
SMALL_FACE_DENOISE = 0.25
FACE_STEPS = 8
FACE_CANVAS = "768"
FACE_CONFIDENCE = 0.35
FACE_CROP_FACTOR = 2.5
FACE_SAMPLER = "euler"
FACE_SCHEDULER = "simple"
FACE_PROMPT = (
    "Refine the face in this video crop. Preserve identity, expression, "
    "gaze, mouth movement, lighting and motion. Natural facial detail; "
    "no new motion or camera changes."
)
FACE_REF_PROMPT = (
    "<Picture 1> is the character reference image. "
    "Refine the face in this video crop to match <Picture 1>. "
    "Faithfully preserve identity, facial features, expression, "
    "gaze, mouth movement, lighting and motion from the source video. "
    "Natural facial detail matching <Picture 1>; no new motion or camera changes."
)


def _nid(counter: list[int]) -> str:
    """Mint a sequential node id."""
    counter[0] += 1
    return str(counter[0])


@dataclass
class RefineSpec:
    """What to do when refining a take."""

    guided_resample: bool = True
    face_refine: bool = True
    enhancement_mode: str = "h3"
    enhancement_resolution: str = "original"

    # Guided resample parameters
    guide_steps: int = GUIDE_STEPS
    guide_megapixels: float | None = None  # None = match source resolution

    # ReDetail parameters
    redetail_scale: float = 1.5
    guide_strength: float = 1.0

    # Face refine parameters
    face_steps: int = FACE_STEPS
    face_denoise: float = SMALL_FACE_DENOISE
    face_canvas: str = FACE_CANVAS
    face_target: str = "small_faces"
    reference_face: str | None = None
    face_prompt: str | None = None
    adaptive_face_scale: bool = True
    min_magnification: float = 1.20
    max_magnification: float = 1.60

    @classmethod
    def from_request(cls, data: dict[str, Any]) -> "RefineSpec":
        return cls(
            enhancement_mode=str(data.get("enhancement_mode", "h3")),
            enhancement_resolution=str(data.get("enhancement_resolution", "original")),
            redetail_scale=float(data.get("redetail_scale", data.get("scale", 1.5))),
            guide_strength=float(data.get("guide_strength", 1.0)),
            guided_resample=bool(data.get("guided_resample", True)),
            face_refine=bool(data.get("face_refine", True)),
            guide_steps=int(data.get("guided_resample_steps", GUIDE_STEPS)),
            guide_megapixels=(
                float(data["guided_resample_megapixels"])
                if data.get("guided_resample_megapixels") is not None
                else None
            ),
            face_steps=int(data.get("face_refine_steps", FACE_STEPS)),
            face_denoise=float(next(
                (data[key] for key in ("face_refine_denoise", "face_denoise")
                 if data.get(key) is not None),
                SMALL_FACE_DENOISE if data.get("face_refine_target", "small_faces") == "small_faces" and not data.get("reference_face") else FACE_DENOISE,
            )),
            face_canvas=str(data.get("face_refine_canvas", FACE_CANVAS)),
            face_target=str(data.get("face_refine_target") or ("reference_identity" if data.get("reference_face") else "small_faces")),
            reference_face=data.get("reference_face") or None,
            face_prompt=data.get("face_prompt") or data.get("face_refine_prompt") or None,
            adaptive_face_scale=bool(data.get("adaptive_face_scale", True)),
            min_magnification=float(data.get("min_magnification", 1.20)),
            max_magnification=float(data.get("max_magnification", 1.60)),
        )


def _target_canvas(
    source_w: int, source_h: int, target_megapixels: float | None
) -> tuple[int, int]:
    """Compute target canvas, preserving aspect ratio and aligning to 32."""
    if target_megapixels is None or target_megapixels <= 0:
        return _align32(source_w), _align32(source_h)

    area = target_megapixels * 1_000_000
    aspect = source_w / max(source_h, 1)
    h = math.sqrt(area / max(aspect, 1e-6))
    w = h * aspect
    return _align32(int(round(w))), _align32(int(round(h)))


def _align32(v: int) -> int:
    return max(32, round(v / 32) * 32)


def build_refine_workflow(
    *,
    source_video: str,
    source_width: int,
    source_height: int,
    seconds: float = 5.0,
    prompt_text: str = "",
    spec: RefineSpec,
    seed: int = 42,
    model_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Construct a self-contained ComfyUI API workflow for take refinement.

    Loads the source video and audio from *source_video* (the ComfyUI reference,
    e.g. ``"subfolder/take.mp4 [output]"``), runs guided resample and/or
    face refine, and saves the result as a new video.
    """
    valid_modes = (
        "h3", "seedvr2", "seedvr2_faces", "redetail", "redetail_refine", "redetail_pixel"
    )
    if spec.enhancement_mode not in valid_modes:
        raise ValueError("Unknown enhancement mode.")
    if spec.enhancement_mode.startswith("redetail"):
        from .redetail import build_workflow
        return build_workflow(
            source_video=source_video,
            source_width=source_width,
            source_height=source_height,
            spec=spec,
            seed=seed,
            model_overrides=model_overrides,
        )
    if spec.enhancement_mode != "h3":
        from .seedvr2 import build_workflow
        return build_workflow(source_video, source_width, source_height, spec, seed, model_overrides)
    if not spec.guided_resample and not spec.face_refine:
        raise ValueError("At least one refinement pass must be enabled.")
    if spec.face_target not in ("small_faces", "largest_face", "reference_identity"):
        raise ValueError("Unknown face refinement target.")
    if spec.face_refine and spec.face_target == "small_faces" and spec.reference_face:
        raise ValueError("A reference face selects one person; use reference_identity instead of small_faces.")
    if spec.face_refine and spec.face_target == "reference_identity" and not spec.reference_face:
        raise ValueError("Reference identity refinement requires a reference face.")

    models = {**(model_overrides or {})}
    remix_model = models.get("remix_model", DEFAULT_REMIX_MODEL)
    second_model = models.get("second_model", DEFAULT_SECOND_MODEL)
    text_encoder = models.get("text_encoder", DEFAULT_TEXT_ENCODER)
    video_vae = models.get("video_vae", DEFAULT_VIDEO_VAE)
    audio_vae = models.get("audio_vae", DEFAULT_AUDIO_VAE)

    c: list[int] = [0]
    wf: dict[str, Any] = {}

    # ── 1. Model Loader (FeiHouEasyH3RHRemixLoader) ──────────────────────
    loader_id = _nid(c)
    wf[loader_id] = {
        "class_type": "FeiHouEasyH3RHRemixLoader",
        "inputs": {
            "remix_model": remix_model,
            "text_encoder": text_encoder,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "second_sampling_model": second_model,
            "first_pass_lora_stack": {},
            "second_pass_lora_stack": {},
        },
        "_meta": {"title": "Refine — Model Loader"},
    }

    # ── 2. Context Builder (FeiHouEasyH3RH) ──────────────────────────────
    # Prepares model hooks and MINIMAX_H3_CONTEXT for downstream nodes
    h3_id = _nid(c)
    wf[h3_id] = {
        "class_type": "FeiHouEasyH3RH",
        "inputs": {
            "h3_bundle": [loader_id, 0],
            "mode": "image",
            "prompt": prompt_text,
            "resolution": "540P",
            "aspect_ratio": "16:9",
            "width": _align32(source_width),
            "height": _align32(source_height),
            "audio_duration_auto": "off",
            "seconds": max(0.2, float(seconds)),
            "advanced": False,
            "fps": 24.0,
            "keyframe_role": "first",
            "ref_image_size": "480",
            "reference_mention_mode": "index",
            "prompt_optimizer_enabled": False,
            "prompt_optimizer_provider": "",
            "prompt_optimizer_scene_guide": "none",
            "force_offload": True,
            "low_vram_streamed_attention": False,
            "reference_text_only": False,
            "second_sampling_output_connected": True,
        },
        "_meta": {"title": "Refine — Context Builder"},
    }

    # ── 3. Context Unpacker (FeiHouEasyH3RHOutput) ────────────────────────
    # Unpacks h3_context: [2] video_vae, [4] fps, [6] clip, [8] duration_control
    out_id = _nid(c)
    wf[out_id] = {
        "class_type": "FeiHouEasyH3RHOutput",
        "inputs": {
            "h3_context": [h3_id, 2],
        },
        "_meta": {"title": "Refine — Context Output"},
    }

    # ── 4. Source Video & Audio Loader ───────────────────────────────────
    load_vid_id = _nid(c)
    wf[load_vid_id] = {
        "class_type": "LoadVideo",
        "inputs": {
            "file": source_video,
        },
        "_meta": {"title": "Refine — Load Source Video"},
    }

    split_comp_id = _nid(c)
    wf[split_comp_id] = {
        "class_type": "GetVideoComponents",
        "inputs": {
            "video": [load_vid_id, 0],
        },
        "_meta": {"title": "Refine — Split Frames & Audio"},
    }

    images_source = (split_comp_id, 0)
    audio_source = (split_comp_id, 1)
    fps_source = (out_id, 4)

    # ── 5. Guided Resample Pass (if enabled) ──────────────────────────────
    if spec.guided_resample:
        # MiniMax H3's diffusion DiT runs at native canvas (aligned to 32)
        # to respect the model's trained resolution (~1.0 MP) and fit in VRAM.
        guide_w = _align32(source_width)
        guide_h = _align32(source_height)

        # 5a. Resize frames to native aligned canvas
        resize_id = _nid(c)
        wf[resize_id] = {
            "class_type": "ImageResizeKJv2",
            "inputs": {
                "image": list(images_source),
                "width": guide_w,
                "height": guide_h,
                "upscale_method": "bicubic",
                "keep_proportion": "crop",
                "pad_color": "0, 0, 0",
                "crop_position": "center",
                "divisible_by": 32,
                "device": "cpu",
            },
            "_meta": {"title": "Refine — Resize for Guide"},
        }

        # 5b. ImpactImageInfo gives exact width, height, count for empty latent
        img_info_id = _nid(c)
        wf[img_info_id] = {
            "class_type": "ImpactImageInfo",
            "inputs": {
                "value": [resize_id, 0],
            },
            "_meta": {"title": "Refine — Target Image Info"},
        }

        # 5c. Empty AV Latent at target resolution
        empty_latent_id = _nid(c)
        wf[empty_latent_id] = {
            "class_type": "EmptyMiniMaxH3LatentAV",
            "inputs": {
                "width": [img_info_id, 2],
                "height": [img_info_id, 1],
                "length": [img_info_id, 0],
            },
            "_meta": {"title": "Refine — Empty Target Latent"},
        }

        # 5d. CLIP prompt for guide enhancement
        clip_encode_id = _nid(c)
        wf[clip_encode_id] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": [out_id, 6],
                "text": GUIDE_PROMPT,
            },
            "_meta": {"title": "Refine — Guide Prompt"},
        }

        # 5e. Add guide conditioning
        add_guide_id = _nid(c)
        wf[add_guide_id] = {
            "class_type": "MiniMaxH3AddGuide",
            "inputs": {
                "positive": [clip_encode_id, 0],
                "latent": [empty_latent_id, 0],
                "vae": [out_id, 2],
                "image": [resize_id, 0],
                "frame_idx": 0,
            },
            "_meta": {"title": "Refine — Add Guide Conditioning"},
        }

        # 5f. Model conditioning chain for second sampling model (slot 1)
        sigma_shift_id = _nid(c)
        wf[sigma_shift_id] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": [h3_id, 1],
                "shift_video": GUIDE_SIGMA_SHIFT_VIDEO,
                "shift_audio": GUIDE_SIGMA_SHIFT_AUDIO,
            },
            "_meta": {"title": "Refine — Sigma Shift"},
        }

        attn_id = _nid(c)
        wf[attn_id] = {
            "class_type": "ModelAttentionBackend",
            "inputs": {
                "model": [sigma_shift_id, 0],
                "attention": "comfy kitchen attention",
            },
            "_meta": {"title": "Refine — Attention Backend"},
        }

        bsa_id = _nid(c)
        wf[bsa_id] = {
            "class_type": "BlockSparseAttention",
            "inputs": {
                "model": [attn_id, 0],
                # ComfyUI renamed this option from "Sol-Attn (adaptive tau)".
                "selection": "sol-attn",
                "selection.tau": 1.3,
                "start_percent": 0.2,
                "end_percent": 1.0,
                "dense_blocks": "",
                "min_tokens": 12288,
                "extra_tokens": 256,
                "sink_conditioning": "exact_kv_and_rows",
                "verbose": False,
            },
            "_meta": {"title": "Refine — Block Sparse Attention"},
        }

        # 5g. Guider and scheduler
        cfg_guider_id = _nid(c)
        wf[cfg_guider_id] = {
            "class_type": "CFGGuider",
            "inputs": {
                "model": [bsa_id, 0],
                "positive": [add_guide_id, 0],
                "negative": [add_guide_id, 0],
                "cfg": 1.0,
            },
            "_meta": {"title": "Refine — CFG Guider"},
        }

        guide_scheduler_id = _nid(c)
        wf[guide_scheduler_id] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": [bsa_id, 0],
                "scheduler": GUIDE_SCHEDULER,
                "steps": spec.guide_steps,
                "denoise": 1.0,
            },
            "_meta": {"title": "Refine — Guide Scheduler"},
        }

        guide_sampler_id = _nid(c)
        wf[guide_sampler_id] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": GUIDE_SAMPLER},
            "_meta": {"title": "Refine — Guide Sampler"},
        }

        guide_noise_id = _nid(c)
        wf[guide_noise_id] = {
            "class_type": "RandomSeedNoise",
            "inputs": {"seed": seed},
            "_meta": {"title": "Refine — Guide Noise"},
        }

        # 5h. Sampler Custom Advanced
        guide_sample_id = _nid(c)
        wf[guide_sample_id] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [guide_noise_id, 0],
                "guider": [cfg_guider_id, 0],
                "sampler": [guide_sampler_id, 0],
                "sigmas": [guide_scheduler_id, 0],
                "latent_image": [empty_latent_id, 0],
            },
            "_meta": {"title": "Refine — Guided Resample"},
        }

        # 5i. Decode guided output
        guide_decode_id = _nid(c)
        wf[guide_decode_id] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [guide_sample_id, 0],
                "vae": [out_id, 2],
            },
            "_meta": {"title": "Refine — Decode Guided Video"},
        }

        # 5j. Crop duration and align with audio
        crop_id = _nid(c)
        wf[crop_id] = {
            "class_type": "FeiHouEasyH3RHDurationCrop",
            "inputs": {
                "images": [guide_decode_id, 0],
                "duration_control": [out_id, 8],
                "audio": list(audio_source),
                "fps": [out_id, 4],
            },
            "_meta": {"title": "Refine — Align Duration"},
        }

        images_source = (crop_id, 0)
        fps_source = (crop_id, 1)
        audio_source = (crop_id, 2)

    # ── 6. Face Refine Pass (if enabled) ─────────────────────────────────
    if spec.face_refine:
        face_prompt = spec.face_prompt
        if not face_prompt:
            face_prompt = FACE_REF_PROMPT if spec.reference_face else FACE_PROMPT

        face_inputs: dict[str, Any] = {
            "images": list(images_source),
            "model": [h3_id, 1],
            "h3_context": [h3_id, 2],
            "audio": list(audio_source),
            "enabled": True,
            "detector": FACE_DETECTOR,
            "target": spec.face_target,
            "denoise": spec.face_denoise,
            "steps": spec.face_steps,
            "seed": seed,
            "advanced": False,
            "canvas_size": spec.face_canvas,
            "confidence": FACE_CONFIDENCE,
            "crop_factor": FACE_CROP_FACTOR,
            "chunk_frames": "不分段",
            "sampler_name": FACE_SAMPLER,
            "scheduler": FACE_SCHEDULER,
            "prompt": face_prompt,
            "force_offload": True,
            "audio_lock": False,
            "clean_second_model": True,
            "adaptive_face_scale": spec.adaptive_face_scale,
            "min_magnification": spec.min_magnification,
            "max_magnification": spec.max_magnification,
        }

        if spec.reference_face:
            ref_img_id = _nid(c)
            wf[ref_img_id] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": spec.reference_face,
                },
                "_meta": {"title": "Refine — Reference Face Image"},
            }
            face_inputs["reference_face"] = [ref_img_id, 0]

        face_refine_id = _nid(c)
        wf[face_refine_id] = {
            "class_type": "FeiHouEasyH3RHFaceRefine",
            "inputs": face_inputs,
            "_meta": {"title": "Refine — Face Refine"},
        }
        if spec.face_target == "small_faces":
            # Each tracked person supplies their own crop reference. Never apply
            # a single cast image to everybody in the scene.
            delivery_w, delivery_h = _target_canvas(source_width, source_height, spec.guide_megapixels)
            upscale_faces = delivery_w * delivery_h > _align32(source_width) * _align32(source_height)
            wf[face_refine_id] = {
                "class_type": "VideoPrompterSmallFaceRefine",
                "inputs": {**{k: face_inputs[k] for k in (
                    "images", "model", "h3_context", "audio", "detector", "denoise",
                    "steps", "seed", "canvas_size", "prompt")},
                    "confidence": 0.25, "min_face_size": 8, "max_face_size": 192,
                    "tiled_detection": True,
                    "output_width": delivery_w if upscale_faces else 0,
                    "output_height": delivery_h if upscale_faces else 0},
                "_meta": {"title": "Refine — Small Faces"},
            }
            # Restore tiny source crops before H3; paste the refined detail
            # directly into the delivery canvas instead of shrinking it first.
            face_upscaler = models.get("face_upscale_model", DEFAULT_FACE_UPSCALER)
            if upscale_faces and face_upscaler:
                upscale_model_id = _nid(c)
                wf[upscale_model_id] = {
                    "class_type": "UpscaleModelLoader",
                    "inputs": {"model_name": face_upscaler},
                    "_meta": {"title": "Refine — Small Face Pre-Upscale Model"},
                }
                wf[face_refine_id]["inputs"]["upscale_model"] = [upscale_model_id, 0]

        images_source = (face_refine_id, 0)
        audio_source = (face_refine_id, 1)
        fps_source = (face_refine_id, 2)

    # ── 7. Final Upscale (if target canvas area exceeds native) ───────────
    target_w, target_h = _target_canvas(
        source_width, source_height, spec.guide_megapixels,
    )

    native_w, native_h = _align32(source_width), _align32(source_height)
    if (target_w * target_h) > (native_w * native_h) and not (spec.face_refine and spec.face_target == "small_faces"):
        upscale_id = _nid(c)
        wf[upscale_id] = {
            "class_type": "ImageResizeKJv2",
            "inputs": {
                "image": list(images_source),
                "width": target_w,
                "height": target_h,
                "upscale_method": "lanczos",
                "keep_proportion": "crop",
                "pad_color": "0, 0, 0",
                "crop_position": "center",
                "divisible_by": 32,
                "device": "cpu",
            },
            "_meta": {"title": "Refine — Final Upscale to Deliverable Canvas"},
        }
        images_source = (upscale_id, 0)

    # ── 8. Save Refined Video (VHS_VideoCombine) ─────────────────────────
    save_id = _nid(c)
    wf[save_id] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": list(images_source),
            "audio": list(audio_source),
            "frame_rate": list(fps_source),
            "save_output": True,
            "filename_prefix": "Minimax_H3_Refined",
            "loop_count": 0,
            "pix_fmt": "yuv420p",
            "save_metadata": True,
            "crf": 19,
            "trim_to_audio": False,
            "format": "video/h264-mp4",
            "no_preview": False,
            "pingpong": False,
        },
        "_meta": {"title": "Refine — Save Video"},
    }

    return wf


@dataclass
class PreparedRefine:
    """Everything the client needs to submit a refinement job."""

    workflow: dict[str, Any]
    source_take_id: str
    width: int
    height: int
    seed: int
    refinement: list[str]
    reference_face: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "source_take_id": self.source_take_id,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "refinement": self.refinement,
            "reference_face": self.reference_face,
        }


def prepare_refine(
    *,
    source_video: str,
    source_take_id: str,
    source_width: int,
    source_height: int,
    seconds: float = 5.0,
    prompt_text: str = "",
    seed: int = 42,
    spec: RefineSpec,
    model_overrides: dict[str, str] | None = None,
) -> PreparedRefine:
    """Prepare a refinement of an existing take."""
    target_w, target_h = _target_canvas(
        source_width, source_height, spec.guide_megapixels,
    )
    if spec.enhancement_mode.startswith("redetail"):
        from .redetail import output_canvas
        mode = "pixel" if spec.enhancement_mode == "redetail_pixel" else "refine"
        target_w, target_h = output_canvas(source_width, source_height, spec.redetail_scale, mode=mode)
    elif spec.enhancement_mode != "h3":
        from .seedvr2 import output_canvas
        target_w, target_h = output_canvas(source_width, source_height, spec.enhancement_resolution)

    workflow = build_refine_workflow(
        source_video=source_video,
        source_width=source_width,
        source_height=source_height,
        seconds=seconds,
        prompt_text=prompt_text,
        spec=spec,
        seed=seed,
        model_overrides=model_overrides,
    )

    refinement = []
    if spec.enhancement_mode.startswith("redetail"):
        refinement.append("redetail")
        if spec.enhancement_mode == "redetail_pixel":
            refinement.append("pixel")
        else:
            refinement.append("refine")
    elif spec.enhancement_mode != "h3":
        refinement.append("seedvr2")
        if spec.enhancement_mode == "seedvr2_faces":
            refinement.append("face_blend")
    elif spec.guided_resample:
        refinement.append("guided_resample")
    if spec.enhancement_mode == "h3" and spec.face_refine:
        refinement.append("face_refine")
    native_w, native_h = (
        (source_width, source_height)
        if spec.enhancement_mode != "h3"
        else (_align32(source_width), _align32(source_height))
    )
    if (target_w * target_h) > (native_w * native_h):
        refinement.append("upscale")

    return PreparedRefine(
        workflow=workflow,
        source_take_id=source_take_id,
        width=target_w,
        height=target_h,
        seed=seed,
        refinement=refinement,
        reference_face=spec.reference_face,
    )
