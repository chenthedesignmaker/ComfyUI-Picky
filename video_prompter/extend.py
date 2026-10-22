"""Extend an existing clip with H3 Motion Context.

Motion Context lets a *new* clip continue an existing one — its last frames and
sound are pinned as never-denoised conditioning, so motion and audio carry
across the join instead of the clip restarting from a single frame. See the
ComfyUI-MiniMaxH3-Contex-Loop custom node for the mechanism (the older
ComfyUI-H3-Motion-Context package of the same class names sits disabled
alongside it, and its grid is shorter).

**A continuation is an ordinary render with Motion Context wired in.** That is
the whole design, and it is why `augment_extend` is a *transform over a finished
workflow* rather than a graph of its own: whatever the shot would have rendered
as — t2v, i2v, or ref2v with the project's cast wired in — is what it continues
as, on the same weights, the same LoRAs and the same sampler. A fixed graph of
our own could only ever be one mode on one set of models, which is how
continuations quietly lost their cast references and their checkpoint.

Two audio paths, chosen per call:

  video path   the source clip's frames AND sound come from the video itself
               (LoadVideo -> GetVideoComponents), the sound re-encoded through
               the audio VAE. Works on ANY clip — a take, a re-shoot, imported
               footage — with no prior render state.
  latent path  the source clip's saved AV latent carries the sound instead,
               skipping a decode/re-encode round trip that dulls it a little at
               every link of a chain. Available only when a latent was saved
               (every AV render saves one, so chains stay on this path).

Both pin the video frames from the decoded source; only the audio source
differs.
"""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from typing import Any

from . import h3_spec
from .profiles import (
    ROLE_CONTEXT_LATENT,
    ROLE_CONTEXT_LENGTH,
    ROLE_CONTEXT_VIDEO,
    Binding,
    ProfileError,
    RenderProfile,
    capture,
)

# Local H3 weights, matching the "H3 local t2v" profile. Only ever used for the
# built-in fallback graph below — a caller with a profile of their own renders
# on that profile's weights, which is the point.
DEFAULT_MODELS = {
    "unet": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}

# Run lengths the video VAE actually distinguishes, mirroring VIDEO_RUN_GRID in
# the node. The node snaps an off-grid request DOWN and only logs it, so a value
# that is not on this list silently becomes a shorter one — we offer the grid
# instead of a free number so nothing is carried over behind the caller's back.
CONTEXT_GRID = (1, 5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, 209, 226, 243)

# Author-tested continuation settings (see the node's QUICK START). 22 frames of
# video context sits on the VAE grid; head anchoring puts the pinned run at the
# front to be trimmed; timeline audio places the pinned sound on the new clip's
# own timeline so the model continues it rather than imitating it. The node's
# README suggests 39 when the SAME shot should continue seamlessly, against 22
# for a new shot that should merely remember the one before it.
CONTEXT_LENGTH = 22
AUDIO_CONTEXT_LENGTH = 22
FPS = 24.0

# The classes the transform below finds its way around by. Nothing is addressed
# by node id: ids differ per mode and per user capture, so anything hardcoding
# one would work for exactly one graph (the same reasoning as `twostage`).
MOTION_CONTEXT_NODE = "MiniMaxH3MotionContext"
MOTION_CONTEXT_TRIM_NODE = "MiniMaxH3MotionContextTrim"
LOAD_LATENT_NODE = "MiniMaxH3MotionContextLoadLatent"
CONDITIONING_NODES = (h3_spec.NODE_IMAGE_TO_VIDEO, h3_spec.NODE_REFERENCE_TO_VIDEO)
VIDEO_DECODE_NODE = "VAEDecode"
AUDIO_DECODE_NODE = "VAEDecodeAudio"
CREATE_VIDEO_NODE = "CreateVideo"
GUIDER_NODE = "BasicGuider"


class ExtendError(ValueError):
    """This workflow cannot be continued.

    Raised rather than rendering something that merely looks like a
    continuation: an extend that silently lost its Motion Context is a clip
    that restarts, and the only way to notice is to watch the whole thing.
    """


def snap_context(frames: int) -> int:
    """The grid value at or below `frames` — what the node would really pin.

    Mirrors the node's own snap so callers can show the honest number before
    rendering rather than discovering it in a ComfyUI log afterwards.
    """
    return next((g for g in reversed(CONTEXT_GRID) if g <= int(frames)), CONTEXT_GRID[0])


# --- the transform -------------------------------------------------------

def _node(ct: str, inp: dict[str, Any], title: str | None = None) -> dict[str, Any]:
    return {"class_type": ct, "inputs": inp, "_meta": {"title": title or ct}}


def _is_wire(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _find(graph: dict[str, Any], *class_types: str) -> str | None:
    return next(
        (nid for nid, node in graph.items()
         if isinstance(node, dict) and node.get("class_type") in class_types),
        None,
    )


def _wire_source(graph: dict[str, Any], node_id: str, input_name: str) -> list[Any] | None:
    """The `[node, slot]` a node's input is fed from, or None if it is a literal."""
    wire = (graph.get(node_id, {}).get("inputs") or {}).get(input_name)
    return list(wire) if _is_wire(wire) else None


def _free_id(graph: dict[str, Any], base: str) -> str:
    if base not in graph:
        return base
    n = 2
    while f"{base}_{n}" in graph:
        n += 1
    return f"{base}_{n}"


def augment_extend(
    api_format: dict[str, Any], *, latent_backed: bool = False
) -> tuple[dict[str, Any], dict[str, str]]:
    """Wire Motion Context into a finished H3 render graph.

    Returns the new graph and the ids of the nodes added, so a profile can pin
    the extend roles onto them (`build_extend_profile`) and the two-pass rebind
    below can find them again.

    Found by class and by wire, never by id, so one implementation covers a
    stored t2v capture, an i2v augmentation and a freshly built ref2v graph
    alike. What it does, in order: the source clip is loaded and split into
    frames and sound; a Motion Context node takes the H3 node's conditioning and
    empty latent plus that pinned run, and everything that read the conditioning
    now reads Motion Context's instead; the pinned head that comes back at the
    front of the result is trimmed off before the video is assembled.
    """
    graph = copy.deepcopy(api_format)

    cond_id = _find(graph, *CONDITIONING_NODES)
    if cond_id is None:
        raise ExtendError(
            "A continuation needs an H3 conditioning node to attach Motion "
            f"Context to, and this workflow has no {' or '.join(CONDITIONING_NODES)}. "
            "Re-capture the render profile."
        )
    video_decode = _find(graph, VIDEO_DECODE_NODE)
    audio_decode = _find(graph, AUDIO_DECODE_NODE)
    create_video = _find(graph, CREATE_VIDEO_NODE)
    if video_decode is None or audio_decode is None or create_video is None:
        raise ExtendError(
            "A continuation carries picture and sound across the join, so it "
            f"needs {VIDEO_DECODE_NODE}, {AUDIO_DECODE_NODE} and "
            f"{CREATE_VIDEO_NODE} in the graph. This workflow is missing at "
            "least one of them."
        )

    # The two VAEs are told apart by wiring, not by name: whichever one feeds
    # the audio decode is the audio VAE. Motion Context encodes the pinned
    # picture with the video one.
    video_vae = _wire_source(graph, video_decode, "vae")
    audio_vae = _wire_source(graph, audio_decode, "vae")
    if video_vae is None:
        raise ExtendError(
            f"{VIDEO_DECODE_NODE} has no VAE wired, so there is nothing to "
            "encode the pinned frames with."
        )

    ids = {
        "source": _free_id(graph, "vp_extend_src"),
        "components": _free_id(graph, "vp_extend_split"),
        "motion_context": _free_id(graph, "vp_extend_mc"),
        "trim": _free_id(graph, "vp_extend_trim"),
    }

    graph[ids["source"]] = _node("LoadVideo", {"file": ""}, "Source clip (Extend)")
    graph[ids["components"]] = _node(
        "GetVideoComponents", {"video": [ids["source"], 0]}, "Source frames and sound"
    )

    mc_inputs: dict[str, Any] = {
        "conditioning": [cond_id, 0],
        "vae": video_vae,
        # The NEW clip's empty AV latent, slot 1 of the H3 conditioning node.
        # Motion Context reads the canvas off it, so the pinned run is encoded
        # at exactly the size this pass samples at.
        "latent": [cond_id, 1],
        "context_frames": [ids["components"], 0],
        "context_length": CONTEXT_LENGTH,
        "encode_mode": "video",
        "anchor_mode": "head",
        "crop": "disabled",
        "audio_context_length": AUDIO_CONTEXT_LENGTH,
        "audio_mode": "timeline",
    }
    if latent_backed:
        ids["latent"] = _free_id(graph, "vp_extend_latent")
        graph[ids["latent"]] = _node(
            LOAD_LATENT_NODE, {"latent_path": "", "clip_index": 0}, "Source latent"
        )
        mc_inputs["context_latent"] = [ids["latent"], 0]
    else:
        # sound comes straight from the decoded source, re-encoded by the audio VAE
        mc_inputs["context_audio"] = [ids["components"], 1]
        if audio_vae is not None:
            mc_inputs["audio_vae"] = audio_vae

    graph[ids["motion_context"]] = _node(
        MOTION_CONTEXT_NODE, mc_inputs, "H3 Motion Context"
    )

    # Everything that read the H3 node's conditioning reads Motion Context's
    # now — the guider, and anything else a captured graph put in between.
    _repoint_conditioning(
        graph, cond_id, ids["motion_context"], skip={ids["motion_context"]}
    )

    # In head mode the pinned run is regenerated at the front of the result, so
    # it is trimmed off before the clip is assembled.
    graph[ids["trim"]] = _node(
        MOTION_CONTEXT_TRIM_NODE,
        {
            "images": [video_decode, 0],
            "trim_frames": [ids["motion_context"], 1],
            "audio": [audio_decode, 0],
            "fps": FPS,
            "match_tail": True,
        },
        "Trim the carried head",
    )
    create_inputs = graph[create_video].setdefault("inputs", {})
    create_inputs["images"] = [ids["trim"], 0]
    create_inputs["audio"] = [ids["trim"], 1]

    return graph, ids


def _repoint_conditioning(
    graph: dict[str, Any], old_id: str, new_id: str, skip: set[str]
) -> None:
    """Send every conditioning wire from `old_id` slot 0 to `new_id` slot 0.

    Slot 1 is the empty AV latent, which still has to come from the H3 node
    itself — Motion Context returns conditioning and a trim count, not a latent.
    """
    for node_id, node in graph.items():
        if node_id in skip or not isinstance(node, dict):
            continue
        for name, value in (node.get("inputs") or {}).items():
            if _is_wire(value) and str(value[0]) == old_id and str(value[1]) == "0":
                node["inputs"][name] = [new_id, 0]


def rebind_two_stage(
    workflow: dict[str, Any], ids: dict[str, str], two_stage: dict[str, str]
) -> str:
    """Give a two-pass continuation its own Motion Context for pass 2.

    Not an optimisation — without this a two-pass extend cannot run at all. H3
    packs continuation guides into the *target's* spatial grid (see
    `PackedLayout` in `comfy/ldm/minimax/model.py`: `n = vt * frame_rows`, where
    `frame_rows` comes from the latent being sampled). Pass 2 samples an
    upscaled latent, so a guide encoded at pass 1's canvas no longer fills the
    rows the layout reserves for it.

    So pass 2 gets a second Motion Context node reading its canvas off the
    upscaled latent, and a guider of its own to carry it. Same source clip, same
    pinned length, same everything else — only the resolution the pinned run is
    encoded at differs, which is exactly what changed between the passes.

    Returns the new Motion Context node's id.
    """
    mc_id = ids["motion_context"]
    sampler2 = two_stage["sampler2"]
    concat = two_stage["concat"]

    guider = _wire_source(workflow, sampler2, "guider")
    if guider is None or str(guider[0]) not in workflow:
        raise ExtendError(
            "Pass 2 has no guider to re-point, so its Motion Context could not "
            "be attached."
        )

    mc2 = _free_id(workflow, "vp_extend_mc_pass2")
    workflow[mc2] = copy.deepcopy(workflow[mc_id])
    # The only difference: pass 2's canvas, read off the rejoined upscaled latent.
    workflow[mc2]["inputs"]["latent"] = [concat, 0]
    workflow[mc2]["_meta"] = {"title": "H3 Motion Context (pass 2)"}

    guider2 = _free_id(workflow, "vp_extend_guider_pass2")
    workflow[guider2] = copy.deepcopy(workflow[str(guider[0])])
    workflow[guider2]["inputs"]["conditioning"] = [mc2, 0]
    workflow[guider2]["_meta"] = {"title": "Guider (pass 2, refine)"}
    workflow[sampler2]["inputs"]["guider"] = [guider2, 0]

    return mc2


# --- profiles ------------------------------------------------------------

def build_base_workflow(
    models: dict[str, str] | None = None,
    *,
    video_prefix: str = "video/H3extend",
) -> dict[str, Any]:
    """A plain local-H3 t2v graph, for a caller with no profile of their own.

    The CLI and the tests have no stored profile to continue on; the editor
    always does, and should never reach this. Deliberately minimal — it is a
    fallback, not a recommendation.
    """
    m = {**DEFAULT_MODELS, **(models or {})}
    g: dict[str, Any] = {
        "unet": _node("UNETLoader", {"unet_name": m["unet"], "weight_dtype": "default"}),
        "clip": _node("CLIPLoader",
                      {"clip_name": m["clip"], "type": "minimax", "device": "default"}),
        "vvae": _node("VAELoader", {"vae_name": m["video_vae"]}),
        "avae": _node("VAELoader", {"vae_name": m["audio_vae"]}),
        "ksel": _node("KSamplerSelect", {"sampler_name": "res_multistep"}),
        "i2v": _node(
            h3_spec.NODE_IMAGE_TO_VIDEO,
            {"prompt": "", "width": 608, "height": 352, "length": 124,
             "clip": ["clip", 0], "vae": ["vvae", 0]},
        ),
        "noise": _node("RandomNoise", {"noise_seed": 0}),
    }
    g["guider"] = _node(GUIDER_NODE, {"model": ["unet", 0], "conditioning": ["i2v", 0]})
    g["sched"] = _node(
        "BasicScheduler",
        {"scheduler": "simple", "steps": 8, "denoise": 1.0, "model": ["unet", 0]},
    )
    g["samp"] = _node(
        "SamplerCustomAdvanced",
        {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksel", 0],
         "sigmas": ["sched", 0], "latent_image": ["i2v", 1]},
    )
    g["vdec"] = _node(VIDEO_DECODE_NODE, {"samples": ["samp", 0], "vae": ["vvae", 0]})
    g["adec"] = _node(AUDIO_DECODE_NODE, {"samples": ["samp", 0], "vae": ["avae", 0]})
    g["cvid"] = _node(CREATE_VIDEO_NODE,
                      {"fps": FPS, "images": ["vdec", 0], "audio": ["adec", 0]})
    g["save"] = _node(
        "SaveVideo",
        {"filename_prefix": video_prefix, "format": "auto", "codec": "auto",
         "video": ["cvid", 0]},
    )
    return g


def build_extend_workflow(
    *,
    latent_backed: bool = False,
    models: dict[str, str] | None = None,
    video_prefix: str = "video/H3extend",
) -> dict[str, Any]:
    """The fallback base graph with Motion Context wired in, ids discarded."""
    return augment_extend(
        build_base_workflow(models, video_prefix=video_prefix),
        latent_backed=latent_backed,
    )[0]


def build_extend_profile(
    name: str = "H3 extend",
    *,
    base: RenderProfile | None = None,
    latent_backed: bool = False,
    models: dict[str, str] | None = None,
    video_prefix: str = "video/H3extend",
) -> RenderProfile:
    """A render profile for a continuation of `base`.

    `base` is the profile the shot would have rendered through — the user's
    stored t2v capture, or the i2v/ref2v graph `imagery` builds from it. Its
    weights, LoRAs, sampler and references all carry over; only Motion Context
    is added. Without one, the built-in fallback graph stands in.

    Prompt, length, width, height, seed and steps are discovered the usual way.
    The extend roles are pinned by hand: nothing about a string input says "this
    is the clip to continue", and `context_length` rides on the Motion Context
    node the profile system does not know. It drives both the video window and
    the audio window, so the sound carried over always spans the pinned picture.
    """
    graph = (
        copy.deepcopy(base.api_format) if base is not None
        else build_base_workflow(models, video_prefix=video_prefix)
    )
    graph, ids = augment_extend(graph, latent_backed=latent_backed)

    try:
        profile = capture(name, graph)
    except ProfileError as error:
        raise ExtendError(f"The continuation graph cannot be driven: {error}") from error

    profile.bindings.setdefault(ROLE_CONTEXT_VIDEO, []).append(
        Binding(ids["source"], "file")
    )
    profile.bindings.setdefault(ROLE_CONTEXT_LENGTH, []).extend([
        Binding(ids["motion_context"], "context_length"),
        Binding(ids["motion_context"], "audio_context_length"),
    ])
    if latent_backed:
        profile.bindings.setdefault(ROLE_CONTEXT_LATENT, []).append(
            Binding(ids["latent"], "latent_path")
        )
    return profile


def motion_context_ids(profile: RenderProfile) -> dict[str, str]:
    """Where the extend nodes ended up in a built profile.

    Read back off the pinned roles rather than threaded through every caller:
    the bindings already say which node carries the pinned length and which the
    source clip, and those are the two anyone downstream needs.
    """
    targets = profile.targets(ROLE_CONTEXT_LENGTH)
    if not targets:
        raise ExtendError(
            "This profile has no Motion Context binding, so it is not a "
            "continuation profile."
        )
    ids = {"motion_context": targets[0].node_id}
    source = profile.targets(ROLE_CONTEXT_VIDEO)
    if source:
        ids["source"] = source[0].node_id
    return ids


# --- headless submission -------------------------------------------------

def submit(workflow: dict[str, Any], host: str = "http://127.0.0.1:8188",
           timeout_s: float = 2400.0) -> dict[str, Any]:
    """POST a workflow to ComfyUI and wait for it, returning the output files.

    Kept here so the CLI can render an extend on the local GPU the same way the
    editor's client would, without a browser.
    """
    body = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(host + "/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        out = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError("ComfyUI rejected the workflow: " + e.read().decode()) from e

    pid = out["prompt_id"]
    if out.get("node_errors"):
        raise RuntimeError("node errors: " + json.dumps(out["node_errors"]))

    start = time.time()
    while True:
        time.sleep(3)
        hist = json.loads(urllib.request.urlopen(host + "/history/" + pid, timeout=30).read())
        if pid in hist:
            rec = hist[pid]
            files: list[str] = []
            output_ref = None
            for nd in rec.get("outputs", {}).values():
                for group in ("images", "gifs", "videos"):
                    for x in nd.get(group, []):
                        name = x.get("filename")
                        files.append(name)
                        if output_ref is None and name:
                            # a ComfyUI-visible reference for the next LoadVideo:
                            # "subfolder/name.mp4 [type]"
                            sub = x.get("subfolder") or ""
                            typ = x.get("type") or "output"
                            output_ref = (f"{sub}/{name}" if sub else name) + f" [{typ}]"
            status = rec.get("status", {})
            errs = [m for m in status.get("messages", [])
                    if m and m[0] in ("execution_error", "execution_interrupted")]
            return {"prompt_id": pid, "completed": status.get("completed"),
                    "status": status.get("status_str"), "files": sorted(files),
                    "output_ref": output_ref,
                    "errors": errs, "elapsed_s": round(time.time() - start, 1)}
        if time.time() - start > timeout_s:
            return {"prompt_id": pid, "completed": False, "status": "timeout",
                    "files": [], "output_ref": None}


def chain(
    source: str,
    prompts: list[str],
    *,
    width: int = 608,
    height: int = 352,
    frames: int = 124,
    steps: int = 8,
    seed: int | None = None,
    context_length: int = CONTEXT_LENGTH,
    base: RenderProfile | None = None,
    host: str = "http://127.0.0.1:8188",
    models: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Extend `source` once per prompt, each step continuing the previous one.

    Step 1 continues the caller's `source` off its own frames and sound (the
    video path — the source may be any clip, with no saved latent). Every step
    saves its own AV latent, so steps 2+ ride the latent path automatically,
    keeping the sound clean link after link — the whole reason the latent path
    exists. Returns one submit result per step, each carrying its `step`,
    `prompt` and `output_ref`; stops at the first failure.
    """
    from .render import prepare_extend

    seconds = frames / FPS
    results: list[dict[str, Any]] = []
    src_ref = source
    prev_latent: str | None = None

    for i, prompt in enumerate(prompts, start=1):
        latent_backed = prev_latent is not None
        profile = build_extend_profile(
            base=base, latent_backed=latent_backed, models=models,
        )
        prepared = prepare_extend(
            profile, prompt, src_ref,
            width=width, height=height, frames=frames, seconds=seconds,
            seed=None if seed is None else seed + i, steps=steps,
            context_length=context_length,
            source_latent=prev_latent if latent_backed else None,
        )
        result = submit(prepared.workflow, host=host)
        result["step"] = i
        result["prompt"] = prompt
        results.append(result)
        if not result.get("completed") or not result.get("output_ref"):
            break
        # thread this step's outputs into the next: its video for the frames,
        # its own saved latent for the sound
        src_ref = result["output_ref"]
        prev_latent = prepared.latent_path

    return results


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from .paths import storage_root
    from .profiles import ProfileStore
    from .render import prepare_extend

    p = argparse.ArgumentParser(
        prog="video_prompter.extend",
        description="Continue an existing clip with H3 Motion Context. Give one "
                    "prompt to extend once, or several to chain clip->clip->clip.",
    )
    p.add_argument("source", help="clip to continue: a ComfyUI-visible video "
                                  "(e.g. 'video/take.mp4 [output]' or an input filename)")
    p.add_argument("prompt", nargs="+",
                   help="what happens next; pass several to chain them into a sequence")
    p.add_argument("--profile", default=None, metavar="NAME",
                   help="stored render profile to continue on — its weights, "
                        "LoRAs and sampler. Without one a plain local-H3 graph "
                        "stands in, which is rarely what a project wants.")
    p.add_argument("--length", type=int, default=124, help="frames of each new clip")
    p.add_argument("--width", type=int, default=608)
    p.add_argument("--height", type=int, default=352)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--context", type=int, default=CONTEXT_LENGTH, metavar="FRAMES",
                   help="frames of the source to pin (default %(default)s; video "
                        "mode uses 1/5/22/39). Lower for a looser tie and a bigger "
                        "action change, higher for tighter continuity.")
    p.add_argument("--latent", metavar="PATH", default=None,
                   help="saved AV latent of the source (single-extend latent path)")
    p.add_argument("--seconds", type=float, default=None,
                   help="duration to record; defaults to length/24")
    p.add_argument("--submit", action="store_true", help="render on local ComfyUI")
    p.add_argument("--host", default="http://127.0.0.1:8188")
    args = p.parse_args(argv)

    base = None
    if args.profile:
        store = ProfileStore(storage_root() / "profiles")
        if not store.exists(args.profile):
            print(f"error: no render profile named {args.profile!r}. Have: "
                  + ", ".join(store.list()), file=sys.stderr)
            return 2
        base = store.load(args.profile)

    # More than one prompt is a chain. It only makes sense as a real render,
    # since each step's source is the previous step's output.
    if len(args.prompt) > 1:
        if not args.submit:
            print("error: a chain (multiple prompts) needs --submit; each step's "
                  "source is the previous step's rendered output.", file=sys.stderr)
            return 2
        results = chain(
            args.source, args.prompt,
            width=args.width, height=args.height, frames=args.length, steps=args.steps,
            seed=args.seed, context_length=args.context, base=base, host=args.host,
        )
        print(json.dumps(results, indent=2))
        return 0 if results and all(r.get("completed") for r in results) else 1

    profile = build_extend_profile(base=base, latent_backed=bool(args.latent))
    seconds = args.seconds if args.seconds is not None else args.length / FPS
    prepared = prepare_extend(
        profile, args.prompt[0], args.source,
        width=args.width, height=args.height, frames=args.length, seconds=seconds,
        seed=args.seed, steps=args.steps, source_latent=args.latent,
        context_length=args.context,
    )

    if not args.submit:
        json.dump(prepared.to_dict(), sys.stdout, indent=2)
        print()
        return 0

    result = submit(prepared.workflow, host=args.host)
    print(json.dumps(result, indent=2))
    return 0 if result.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
