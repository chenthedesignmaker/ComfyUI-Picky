"""Image-to-video (i2v) and reference-to-video (ref2v) for MiniMax H3.

The base render path drives `MiniMaxH3ImageToVideo` from a prompt alone
(text-to-video). H3 has two more task modes that need actual media wired into
the graph, both on weights already loaded for t2v:

  i2v    the same `MiniMaxH3ImageToVideo` node, plus a first (and optional last)
         keyframe image. Because it is the *same* node the stored t2v profile
         already uses, i2v is built by **augmenting** that profile — adding
         `LoadImage` nodes and wiring them in — so the user's sampler, steps and
         models carry over untouched.
  ref2v  a different node, `MiniMaxH3ReferenceToVideo`, with an `audio_vae` and
         omni-reference slots (images / videos / audio). It cannot reuse the t2v
         node, so its graph is built fresh here (the `extend.py` pattern),
         reusing only the model *names* discovered from the stored profile.

The render **mode is chosen from the document**: keyframes -> i2v, references ->
ref2v, neither -> t2v. Both at once is a lint error (the two live on different
nodes). Callers go through `api.prepare_render`, which picks the profile by mode
and then hands off to the unchanged `render.prepare_render`, so composing,
linting, seeding and the extend latent-save all still apply.
"""

from __future__ import annotations

import copy
from typing import Any

from . import h3_spec
from .camera_path import camera_media, motion_path
from .document import VideoPrompt
from .extend import DEFAULT_MODELS
from .profiles import ProfileError, RenderProfile, StaleProfileError, capture

Mode = str  # "t2v" | "i2v" | "ref2v"


def render_mode(prompt: VideoPrompt) -> Mode:
    """Which H3 task mode a shot renders as, from what media it carries.

    Keyframes and references sit on different nodes, so a shot is one or the
    other; `mode.conflict` (lint) fires when both are set. Keyframes win here so
    a conflicted shot still resolves to a single mode rather than raising.
    """
    prompt = camera_media(prompt)
    if prompt.first_frame.strip() or prompt.last_frame.strip():
        return "i2v"
    if prompt.references.items:
        return "ref2v"
    return "t2v"


def models_from_profile(profile: RenderProfile | None) -> dict[str, str]:
    """Model names from a stored profile's loader nodes, defaults for the rest.

    ref2v builds a fresh graph but should still run on whatever weights the
    user's t2v profile names. The two VAEs are told apart by wiring: the one
    feeding `VAEDecodeAudio` is the audio VAE, the other the video VAE.
    """
    models = dict(DEFAULT_MODELS)
    if profile is None:
        return models

    api = profile.api_format or {}

    def _first(class_type: str, input_name: str) -> str | None:
        for node in api.values():
            if isinstance(node, dict) and node.get("class_type") == class_type:
                value = (node.get("inputs") or {}).get(input_name)
                if isinstance(value, str) and value:
                    return value
        return None

    unet = _first("UNETLoader", "unet_name")
    clip = _first("CLIPLoader", "clip_name")
    if unet:
        models["unet"] = unet
    if clip:
        models["clip"] = clip

    # Which VAELoader is the audio one: follow VAEDecodeAudio's `vae` wire.
    audio_loader_id = None
    for node in api.values():
        if isinstance(node, dict) and node.get("class_type") == "VAEDecodeAudio":
            wire = (node.get("inputs") or {}).get("vae")
            if isinstance(wire, (list, tuple)) and wire:
                audio_loader_id = str(wire[0])
            break
    for node_id, node in api.items():
        if not (isinstance(node, dict) and node.get("class_type") == "VAELoader"):
            continue
        name = (node.get("inputs") or {}).get("vae_name")
        if not isinstance(name, str) or not name:
            continue
        if str(node_id) == audio_loader_id:
            models["audio_vae"] = name
        else:
            models["video_vae"] = name
    return models


def _free_id(graph: dict[str, Any], base: str) -> str:
    if base not in graph:
        return base
    n = 2
    while f"{base}_{n}" in graph:
        n += 1
    return f"{base}_{n}"


def augment_i2v(
    api_format: dict[str, Any], first_frame: str, last_frame: str = ""
) -> dict[str, Any]:
    """Clone a t2v workflow and wire keyframe(s) into its H3 image node.

    Adds a `LoadImage` per keyframe and points the node's `first_frame` /
    `last_frame` inputs at them. Everything else — sampler, steps, VAEs, save —
    is the stored profile's own, so an i2v render is a t2v render with an anchor.
    """
    graph = copy.deepcopy(api_format)

    node_id = next(
        (nid for nid, node in graph.items()
         if isinstance(node, dict)
         and node.get("class_type") == h3_spec.NODE_IMAGE_TO_VIDEO),
        None,
    )
    if node_id is None:
        raise StaleProfileError(
            f"Profile has no {h3_spec.NODE_IMAGE_TO_VIDEO} node to attach a "
            f"keyframe to; re-capture the t2v profile."
        )
    inputs = graph[node_id].setdefault("inputs", {})

    def _add_load(image: str, title: str) -> str:
        lid = _free_id(graph, "vp_" + title)
        graph[lid] = {
            "class_type": "LoadImage",
            "inputs": {"image": image},
            "_meta": {"title": f"Keyframe ({title})"},
        }
        return lid

    if first_frame.strip():
        inputs["first_frame"] = [_add_load(first_frame.strip(), "first"), 0]
    if last_frame.strip():
        inputs["last_frame"] = [_add_load(last_frame.strip(), "last"), 0]
    return graph


def _node(class_type: str, inp: dict[str, Any], title: str | None = None) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inp, "_meta": {"title": title or class_type}}


def build_ref2v_workflow(
    prompt: VideoPrompt,
    models: dict[str, str] | None = None,
    *,
    video_prefix: str = "video/VideoPrompter_H3ref",
    reference_size: str = "match",
) -> dict[str, Any]:
    """A fresh ref2v graph: `MiniMaxH3ReferenceToVideo` plus one loader per ref.

    Same overall shape as the t2v workflow (UNET + CLIP + video/audio VAE ->
    reference node -> sampler -> video/audio decode -> CreateVideo -> SaveVideo)
    so `render.prepare_render` discovers prompt/length/width/height/seed the
    usual way and `render.attach_latent_save` can still hook the audio decode.

    References are wired into the node's 0-based autogrow slots by kind. The
    executor nests each slot back into the node's collected dict argument, so the
    input key is the dotted path `<group>.<prefix>N` (e.g. `ref_images.ref_image_0`),
    not the bare slot name. A reference video's soundtrack (`video_audio`) pairs
    with the video of the same index, matching `ref_video_audio_N` <-> `ref_video_N`.
    """
    m = {**DEFAULT_MODELS, **(models or {})}
    camera = motion_path(prompt)
    if camera and "ref2" not in m["unet"].lower():
        raise ProfileError("Motion Frame requires a Ref2VA-capable model. Select a Ref2VA or H3 Hybrid render profile.")
    prompt = camera_media(prompt)
    refs = prompt.references

    g: dict[str, Any] = {
        "unet": _node("UNETLoader", {"unet_name": m["unet"], "weight_dtype": "default"}),
        "clip": _node("CLIPLoader", {"clip_name": m["clip"], "type": "minimax", "device": "default"}),
        "vvae": _node("VAELoader", {"vae_name": m["video_vae"]}),
        "avae": _node("VAELoader", {"vae_name": m["audio_vae"]}),
    }

    ref_inputs: dict[str, Any] = {
        "prompt": "",
        "width": h3_spec.DEFAULT_WIDTH,
        "height": h3_spec.DEFAULT_HEIGHT,
        "length": 124,
        "ref_image_size": reference_size,
        "clip": ["clip", 0],
        "vae": ["vvae", 0],
        "audio_vae": ["avae", 0],
    }

    # images -> ref_images.ref_image_N
    for i, _ref in enumerate(refs.of("image")):
        lid = f"refimg_{i}"
        g[lid] = _node("LoadImage", {"image": _ref.path}, f"Reference image {i + 1}")
        ref_inputs[f"ref_images.ref_image_{i}"] = [lid, 0]

    # videos -> frames from GetVideoComponents into ref_videos.ref_video_N
    for i, _ref in enumerate(refs.of("video")):
        vid = f"refvid_{i}"
        comp = f"refvidc_{i}"
        g[vid] = _node("LoadVideo", {"file": _ref.path}, f"Reference video {i + 1}")
        g[comp] = _node("GetVideoComponents", {"video": [vid, 0]})
        ref_inputs[f"ref_videos.ref_video_{i}"] = [comp, 0]
        if camera and _ref.path == camera["reference_video"]:
            prep = f"vp_camera_frames_{i}"
            g[prep] = _node("VideoPrompterCameraFrames", {"images": [comp, 0], "source_fps": [comp, 2]},
                            "Motion Frame: resample to 24 fps")
            ref_inputs[f"ref_videos.ref_video_{i}"] = [prep, 0]

    # reference-video soundtracks -> ref_video_audios.ref_video_audio_N (index-paired)
    for i, _ref in enumerate(refs.of("video_audio")):
        lid = f"refvidaud_{i}"
        g[lid] = _node("LoadAudio", {"audio": _ref.path}, f"Reference video soundtrack {i + 1}")
        ref_inputs[f"ref_video_audios.ref_video_audio_{i}"] = [lid, 0]

    # standalone audio -> ref_audios.ref_audio_N
    for i, _ref in enumerate(refs.of("audio")):
        lid = f"refaud_{i}"
        g[lid] = _node("LoadAudio", {"audio": _ref.path}, f"Reference audio {i + 1}")
        ref_inputs[f"ref_audios.ref_audio_{i}"] = [lid, 0]

    g["ref"] = _node(h3_spec.NODE_REFERENCE_TO_VIDEO, ref_inputs, "MiniMax H3 Reference to Video")
    g["turbo"] = _node(
        "LoraLoaderModelOnly",
        {
            "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            "strength_model": 1.0,
            "model": ["unet", 0],
        },
        "turbo",
    )
    g["shift"] = _node(
        h3_spec.NODE_SIGMA_SHIFT,
        {"shift_video": 12.0, "shift_audio": 3.0, "model": ["turbo", 0]},
        "MiniMax H3 Sigma Shift",
    )
    g["noise"] = _node("RandomNoise", {"noise_seed": 0})
    g["guider"] = _node("BasicGuider", {"model": ["shift", 0], "conditioning": ["ref", 0]})
    g["ksel"] = _node("KSamplerSelect", {"sampler_name": "res_multistep"})
    g["sched"] = _node("BasicScheduler",
                       {"scheduler": "simple", "steps": 8, "denoise": 1.0, "model": ["shift", 0]})
    g["samp"] = _node("SamplerCustomAdvanced",
                      {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksel", 0],
                       "sigmas": ["sched", 0], "latent_image": ["ref", 1]})
    g["vdec"] = _node("VAEDecode", {"samples": ["samp", 0], "vae": ["vvae", 0]})
    g["adec"] = _node("VAEDecodeAudio", {"samples": ["samp", 0], "vae": ["avae", 0]})
    g["cvid"] = _node("CreateVideo", {"fps": 24.0, "images": ["vdec", 0], "audio": ["adec", 0]})
    g["save"] = _node("SaveVideo",
                      {"filename_prefix": video_prefix, "format": "auto", "codec": "auto",
                       "video": ["cvid", 0]})
    return g


def build_i2v_profile(
    prompt: VideoPrompt, base: RenderProfile, name: str = "H3 i2v"
) -> RenderProfile:
    """An i2v profile: the stored t2v profile with keyframe loaders wired in."""
    prompt = camera_media(prompt)
    graph = augment_i2v(base.api_format, prompt.first_frame, prompt.last_frame)
    return capture(name, graph)


def build_ref2v_profile(
    prompt: VideoPrompt,
    models: dict[str, str] | None = None,
    name: str = "H3 ref2v",
    reference_size: str = "match",
) -> RenderProfile:
    """A ref2v profile built fresh from the prompt's references.

    prompt/length/width/height/seed/steps are discovered on the reference node
    the usual way; the reference files are baked into the graph at build time
    (their count is dynamic, so there is no fixed binding for them).
    """
    profile = capture(
        name, build_ref2v_workflow(prompt, models, reference_size=reference_size)
    )
    # Nothing to pin by hand: the reference loaders carry their filenames as
    # literals, and every scalar the renderer drives is a discovered binding.
    assert profile.controls("prompt"), "ref2v graph lost its prompt binding"
    return profile


def profile_for(
    prompt: VideoPrompt,
    base: RenderProfile | None,
    reference_size: str = "match",
) -> tuple[Mode, RenderProfile]:
    """The (mode, profile) a shot should render through.

    `base` is the user's stored t2v profile: reused verbatim for t2v, augmented
    for i2v, and mined for model names for ref2v. i2v/ref2v both require it to be
    present (i2v to augment, ref2v for the loaders' models) — a caller with no
    profile at all cannot render any mode.

    `reference_size` is the shot's quality tier's answer to how large identity
    references may arrive (`quality.reference_size`). ref2v only: it is baked
    into the graph, not injected, so it has to be known before the profile is
    built rather than set on the way out.
    """
    prompt = camera_media(prompt)
    mode = render_mode(prompt)
    if mode == "t2v":
        if base is None:
            raise ProfileError("No render profile to render through.")
        return mode, base
    if mode == "i2v":
        if base is None:
            raise ProfileError("i2v needs a t2v profile to attach the keyframe to.")
        return mode, build_i2v_profile(prompt, base)
    return mode, build_ref2v_profile(
        prompt, models_from_profile(base), reference_size=reference_size
    )
