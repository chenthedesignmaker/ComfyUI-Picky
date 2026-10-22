"""Turning a prompt plus a profile into something ComfyUI will accept.

Kept free of aiohttp and of ComfyUI itself: this is the part worth testing, and
the server layer above it is a thin adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .cast import Cast, appearing, mention_label, resolve
from .compose import compose
from .document import VideoPrompt
# extend.py defers its own import of this module, so this direction is safe.
from .extend import CONTEXT_LENGTH as DEFAULT_CONTEXT_LENGTH
from .extend import motion_context_ids
from .extend import rebind_two_stage as extend_two_stage
from .lint import Finding, Severity, lint
from .profiles import MAX_SEED, RenderProfile, effective_canvas, inject, random_seed
from .provenance import spans
from .selflift import DEFAULT_UPSCALER_MODEL as SELFLIFT_DEFAULT_UPSCALER
from .selflift import apply as apply_selflift
from .twostage import (
    ALIGN as TWOSTAGE_ALIGN,
    DEFAULT_SINGULARITY_UNET,
    DEFAULT_TURBO_LORA,
    DEFAULT_UPSCALER_MODEL as TWOSTAGE_DEFAULT_UPSCALER,
    apply as apply_two_stage,
    apply_singularity_2pass,
)

# More than this in one click is a mis-click, not an intent.
MAX_BATCH = 8

# The custom node that serialises H3's nested video/audio latent, and the audio
# decode whose input latent is exactly what we want to save. Injecting the save
# into every AV render is what makes an ordinary take extendable on the latent
# (cleaner-audio) path later, not just extends-of-extends.
SAVE_LATENT_NODE = "MiniMaxH3MotionContextSaveLatent"
AUDIO_DECODE_NODE = "VAEDecodeAudio"


def attach_latent_save(workflow: dict[str, Any], seed: int) -> str | None:
    """Append a Save Latent node to an H3 AV workflow, and return where it writes.

    Wired to the same latent the audio decode reads, so it saves exactly the AV
    latent the take rendered from. Returns None for a workflow with nothing to
    save — no audio decode (a still or a video-only graph) — leaving it
    untouched, so this is safe to call on any workflow.

    Allocate the path before submission so it can be recorded on the take.
    Each preparation owns a separate file, including reshoots at the same seed:
    older takes must retain the context that actually produced them.
    """
    audio = next(
        (nid for nid, node in workflow.items()
         if isinstance(node, dict) and node.get("class_type") == AUDIO_DECODE_NODE),
        None,
    )
    if audio is None:
        return None
    samples = (workflow[audio].get("inputs") or {}).get("samples")
    if not (isinstance(samples, (list, tuple)) and len(samples) == 2):
        return None
    if any(isinstance(n, dict) and n.get("class_type") == SAVE_LATENT_NODE
           for n in workflow.values()):
        return None  # the workflow already saves one; do not double up

    prefix = f"h3_context/take{int(seed)}_{uuid4().hex}"
    node_id = _free_node_id(workflow, "vp_latent")
    workflow[node_id] = {
        "class_type": SAVE_LATENT_NODE,
        "inputs": {
            "latent": [samples[0], samples[1]],
            "filename_prefix": prefix,
            "clip_index": 1,
        },
        "_meta": {"title": "Save AV latent (for Extend)"},
    }
    return f"{prefix}_00001.safetensors"


def split_into_two_passes(
    workflow: dict[str, Any],
    canvas: VideoPrompt,
    refine: tuple[int, float] | None,
    *,
    is_selflift: bool = False,
    selflift_settings: dict[str, Any] | None = None,
    is_singularity_2pass: bool = False,
    singularity_settings: dict[str, Any] | None = None,
    steps: int | None = None,
) -> tuple[int, int] | None:
    """Rewrite an injected workflow into two passes, returning its finished size.

    `refine` is `quality.refine(...)` — `None` for every single-pass tier, which
    makes this a no-op and lets the callers below stay branch-free.

    When `is_selflift` is True, rewrites via `apply_selflift` progressive sampling
    and TST patch.
    When `is_singularity_2pass` is True, rewrites via `apply_singularity_2pass`.
    Otherwise rewrites via traditional `apply_two_stage`.

    The finished size is the shot's own aspect ratio at the refine budget, worked
    out the same way `effective_canvas` sizes a single pass, so a two-stage take
    lands on the same canvas its one-pass equivalent would.

    Runs before `attach_latent_save`, so the Extend latent that gets hooked is
    the finished one rather than the small first pass.
    """
    if refine is None:
        return None
    pass1_steps, megapixels = refine
    # SelfLift and Singularity 2-pass write their finishes onto the latent / upscaler,
    # so they may exceed the 768-edge nominal; the classic two-stage upscaler keeps
    # the old ceiling.
    width, height = effective_canvas(
        canvas, megapixels, upscale=is_selflift or is_singularity_2pass
    )
    if is_selflift:
        cfg = selflift_settings or {}
        apply_selflift(
            workflow,
            pass1_steps=pass1_steps,
            width=width,
            height=height,
            lowres_scale=float(cfg.get("lowres_scale", 0.5)),
            tst_tau=float(cfg.get("tst_tau", 0.2)),
            upscaler_model=str(cfg.get("upscaler_model", SELFLIFT_DEFAULT_UPSCALER)),
            rho=float(cfg.get("rho", 0.0)),
            w_min=float(cfg.get("w_min", 0.5)),
            w_max=float(cfg.get("w_max", 1.0)),
        )
    elif is_singularity_2pass:
        cfg = singularity_settings or {}
        p2_steps = int(
            cfg.get("pass2_steps", max(steps - pass1_steps, 1) if steps else 4)
        )
        apply_singularity_2pass(
            workflow,
            pass1_steps=pass1_steps,
            width=width,
            height=height,
            pass2_steps=p2_steps,
            denoise2=float(cfg.get("denoise2", 0.35)),
            singularity_unet=str(cfg.get("singularity_unet", DEFAULT_SINGULARITY_UNET)),
            turbo_lora=str(cfg.get("turbo_lora", DEFAULT_TURBO_LORA)),
            upscaler_model=str(cfg.get("upscaler_model", TWOSTAGE_DEFAULT_UPSCALER)),
            align=int(cfg.get("align", TWOSTAGE_ALIGN)),
        )
    else:
        apply_two_stage(workflow, pass1_steps=pass1_steps, width=width, height=height)
    return width, height


def _free_node_id(workflow: dict[str, Any], base: str) -> str:
    if base not in workflow:
        return base
    suffix = 2
    while f"{base}_{suffix}" in workflow:
        suffix += 1
    return f"{base}_{suffix}"


class RenderBlocked(Exception):
    """The prompt has findings that would waste a generation.

    Carries the findings so the caller can show them rather than just failing.
    """

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings
        summary = "; ".join(finding.message for finding in findings[:3])
        super().__init__(f"Prompt has {len(findings)} blocking finding(s): {summary}")


@dataclass
class PreparedRender:
    """Everything the client needs to submit and to explain what it submitted."""

    workflow: dict[str, Any]
    prompt_text: str
    frames: int
    seconds: float
    width: int
    height: int
    seed: int | None
    findings: list[Finding] = field(default_factory=list)
    # Where this render's AV latent is saved (None if the workflow has no audio
    # to save). Recorded on the take so a later Extend can pin from it.
    latent_path: str | None = None
    # Upstream take id ("prompt_id:filename") when this is an extend continuation.
    parent_take: str | None = None
    steps: int | None = None
    # Where a two-pass render cut its schedule; None when it rendered in one
    # pass. Recorded because `steps` and the finished canvas alone cannot tell
    # a two-pass tier from a single-pass one that spends the same budget —
    # `turbo_2pass` and `turbo_hd` are 8 steps at 1.0 MP either way, and the
    # gallery reads a take's tier back off exactly those numbers.
    pass1_steps: int | None = None
    # Frames of the parent clip pinned as motion context, on extends only.
    # Recorded on the take so a re-roll of this continuation carries over the
    # same amount rather than falling back to the default.
    context_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "prompt": self.prompt_text,
            "frames": self.frames,
            "seconds": self.seconds,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "steps": self.steps,
            "pass1_steps": self.pass1_steps,
            "latent_path": self.latent_path,
            "parent_take": self.parent_take,
            "context_length": self.context_length,
            "findings": [finding_dict(finding) for finding in self.findings],
        }


def preview(
    prompt: VideoPrompt,
    megapixels: float | None = None,
    cast: Cast | None = None,
) -> dict[str, Any]:
    """Composed text plus findings, for the editor's live pane.

    Never raises on a bad prompt — the whole point is to show what is wrong
    while it is still being written.

    `megapixels` reports the canvas at the shot's quality budget rather than the
    document's, so the size the editor shows is the size that will render.

    The cast is resolved once, here, and everything below works on the result —
    so the reference tags the editor is handed are the ones the render will use,
    members included.
    """
    findings = lint(prompt, cast)
    on_screen = {member.id for member in appearing(prompt, cast)}
    prompt = resolve(prompt, cast)
    width, height = effective_canvas(prompt, megapixels)
    return {
        "prompt": compose(prompt),
        "frames": prompt.frames,
        "seconds": prompt.rendered_seconds,
        "width": width,
        "height": height,
        "findings": [finding_dict(finding) for finding in findings],
        # Where each authored field landed in that text, so the editor can tint
        # it. Empty when compose and the marker pass disagree; see provenance.
        "spans": spans(prompt),
        # Tags come back from here rather than being worked out in the editor:
        # <Audio j> numbering runs across video soundtracks and standalone
        # audio together, which is not something to reimplement in two places.
        # `member`/`slot` ride along so the editor can tell which of *its* rows
        # each entry is: resolution reorders the list and inserts the cast's
        # files, so position alone stopped identifying a row once links existed.
        "references": [
            {
                "tag": prompt.references.tag(reference),
                "kind": reference.kind,
                "role": reference.role,
                "path": reference.path,
                "member": reference.member,
                "slot": reference.slot,
            }
            for reference in prompt.references.items
        ],
        # The `<Subject N>` layer over those files: what the document actually
        # defines. `sources` are the reference paths each subject speaks for,
        # which is how the editor knows a picture row has no line of its own.
        "subjects": [
            {
                "tag": prompt.subjects.tag(subject),
                "role": subject.role,
                "retention": subject.retention,
                "noun": subject.noun,
                "label": subject.label,
                "member": subject.member,
                "sources": list(subject.sources),
            }
            for subject in prompt.subjects.items
        ],
        # What each cast member's `@mention` composes to right now. Sent for
        # every member, not only the mentioned ones, because the editor needs
        # the answer *before* the mention is typed — it is what the `@` menu
        # shows beside each name.
        #
        # `present` is what keeps that honest for the rest of the cast. A label
        # is a position in *this* segment's subject list, so a member who is not
        # in the segment yet has no number to show; mentioning them puts them in
        # it and mints one. The editor says "joins as a new subject" for those
        # rather than quoting the fallback as though it were the answer.
        "mentions": [
            {
                "id": member.id,
                "name": member.name,
                "kind": member.kind,
                "label": mention_label(prompt, member),
                "present": member.id in on_screen,
            }
            for member in (cast or [])
        ],
    }


def prepare_render(
    prompt: VideoPrompt,
    profile: RenderProfile,
    seed: int | None = None,
    strict: bool = True,
    megapixels: float | None = None,
    steps: int | None = None,
    turbo_strength: float | None = None,
    cast: Cast | None = None,
    refine: tuple[int, float] | None = None,
    *,
    is_selflift: bool = False,
    selflift_settings: dict[str, Any] | None = None,
    is_singularity_2pass: bool = False,
    singularity_settings: dict[str, Any] | None = None,
) -> PreparedRender:
    """Compose, lint and inject.

    With `strict`, error-level findings raise `RenderBlocked` rather than
    burning a generation on a prompt we already know is wrong. Warnings and
    info never block; they ride along on the result.

    The seed is decided here rather than left to `inject`, so the number that
    went into the workflow is the number the take records. A render whose seed
    was chosen and then forgotten cannot be reproduced.

    `megapixels`/`steps` carry the shot's quality preset. The reported
    width/height reflect the megapixel budget, not the document's canvas, so a
    take records the resolution it truly rendered at.

    `refine` turns the render into a two-pass one (see `twostage`). `megapixels`
    is then the *first* pass's canvas and the refine budget is the finished one,
    which is what gets reported — a take should record the size of the file it
    produced, not the size of the pass that decided its composition.
    """
    findings = lint(prompt, cast)
    blocking = [finding for finding in findings if finding.severity is Severity.ERROR]
    if strict and blocking:
        raise RenderBlocked(blocking)

    seed = random_seed() if seed is None else int(seed)
    # Resolved before injection, not just before composing: a member's identity
    # files have to reach the reference node's slots, not only its prose.
    prompt = resolve(prompt, cast)
    prompt_text = compose(prompt)
    workflow = inject(
        profile,
        prompt,
        prompt_text,
        seed=seed,
        megapixels=megapixels,
        steps=steps,
        turbo_strength=turbo_strength,
    )
    width, height = effective_canvas(prompt, megapixels)
    finished = split_into_two_passes(
        workflow,
        prompt,
        refine,
        is_selflift=is_selflift,
        selflift_settings=selflift_settings,
        is_singularity_2pass=is_singularity_2pass,
        singularity_settings=singularity_settings,
        steps=steps,
    )
    if finished is not None:
        width, height = finished
    latent_path = attach_latent_save(workflow, seed)

    return PreparedRender(
        workflow=workflow,
        prompt_text=prompt_text,
        frames=prompt.frames,
        seconds=prompt.rendered_seconds,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        pass1_steps=refine[0] if refine else None,
        findings=findings,
        latent_path=latent_path,
    )


def sweep_base(seed: int | None, count: int) -> int:
    """Where a `count`-long sweep starts, with room left for the rest of it.

    Sweeps run consecutive seeds so the whole batch is reproducible from its
    first number, which means the base has to sit far enough below the ceiling
    for the last one to still be a distinct seed. Past MAX_SEED they collapse
    onto one float64 on the round trip through the browser, and the sweep
    quietly becomes N renders of the same clip.

    A pinned base gets the same treatment as a rolled one rather than being
    trusted: the editor lets the author type the seed a sweep starts from.
    """
    ceiling = MAX_SEED - count + 1
    return max(min(random_seed() if seed is None else int(seed), ceiling), 0)


def prepare_batch(
    prompt: VideoPrompt,
    profile: RenderProfile,
    count: int,
    seed: int | None = None,
    strict: bool = True,
    megapixels: float | None = None,
    steps: int | None = None,
    turbo_strength: float | None = None,
    cast: Cast | None = None,
    refine: tuple[int, float] | None = None,
    *,
    is_selflift: bool = False,
    selflift_settings: dict[str, Any] | None = None,
    is_singularity_2pass: bool = False,
    singularity_settings: dict[str, Any] | None = None,
) -> list[PreparedRender]:
    """`count` renders of one prompt, one per seed.

    A shot is only worth comparing against itself, and comparing needs
    candidates — so the cheap way to get them is the same prompt at different
    seeds. Linting happens once: the prompt does not change between them, and
    failing on the first is the whole point of failing early.
    """
    if count < 1:
        raise ValueError(f"A batch needs at least one render, got {count}.")

    base = sweep_base(seed, count)
    # Consecutive seeds, so a batch is reproducible from the first one alone.
    return [
        prepare_render(
            prompt,
            profile,
            seed=base + offset,
            strict=strict,
            megapixels=megapixels,
            steps=steps,
            turbo_strength=turbo_strength,
            cast=cast,
            refine=refine,
            is_selflift=is_selflift,
            selflift_settings=selflift_settings,
            is_singularity_2pass=is_singularity_2pass,
            singularity_settings=singularity_settings,
        )
        for offset in range(count)
    ]


def prepare_reshoot(
    profile: RenderProfile,
    prompt_text: str,
    width: int,
    height: int,
    frames: int,
    seconds: float,
    seed: int | None = None,
    megapixels: float | None = None,
    steps: int | None = None,
    turbo_strength: float | None = None,
    refine: tuple[int, float] | None = None,
    *,
    is_selflift: bool = False,
    selflift_settings: dict[str, Any] | None = None,
    is_singularity_2pass: bool = False,
    singularity_settings: dict[str, Any] | None = None,
) -> PreparedRender:
    """Re-render a take from its own frozen values, not a structured prompt.

    A take stores the composed text, its canvas, seed and length. Re-shooting
    injects that text verbatim — no re-compose, no lint (it already rendered) —
    and drives length and duration from the take so the clip matches. `Reroll`
    passes no seed for a fresh one; a fine re-shoot passes the take's seed to
    reproduce the same composition at a higher budget.
    """
    canvas = VideoPrompt.blank()
    canvas.width, canvas.height = int(width), int(height)
    out_w, out_h = effective_canvas(canvas, megapixels)

    seed = random_seed() if seed is None else int(seed)
    workflow = inject(
        profile,
        canvas,
        prompt_text,
        seed=seed,
        megapixels=megapixels,
        steps=steps,
        turbo_strength=turbo_strength,
        length=int(frames),
        duration_s=float(seconds),
    )
    finished = split_into_two_passes(
        workflow,
        canvas,
        refine,
        is_selflift=is_selflift,
        selflift_settings=selflift_settings,
        is_singularity_2pass=is_singularity_2pass,
        singularity_settings=singularity_settings,
        steps=steps,
    )
    if finished is not None:
        out_w, out_h = finished
    latent_path = attach_latent_save(workflow, seed)

    return PreparedRender(
        workflow=workflow,
        prompt_text=prompt_text,
        frames=int(frames),
        seconds=float(seconds),
        width=out_w,
        height=out_h,
        seed=seed,
        steps=steps,
        pass1_steps=refine[0] if refine else None,
        findings=[],
        latent_path=latent_path,
    )


def prepare_extend(
    profile: RenderProfile,
    prompt_text: str,
    source_video: str,
    *,
    width: int,
    height: int,
    frames: int,
    seconds: float,
    seed: int | None = None,
    steps: int | None = None,
    turbo_strength: float | None = None,
    megapixels: float | None = None,
    refine: tuple[int, float] | None = None,
    source_latent: str | None = None,
    context_length: int | None = None,
    parent_take: str | None = None,
) -> PreparedRender:
    """Prepare a clip that continues `source_video` via H3 Motion Context.

    Unlike `prepare_render`, the prompt is a freeform continuation, not a
    structured document, so it is injected verbatim and not linted — the source
    clip already rendered, and there is no storyboard to check.

    `source_latent` selects the higher-fidelity audio path: pass it only with a
    profile built `latent_backed`, whose `context_latent` binding it fills. The
    caller decides which profile matches the latent it has (see
    `extend.build_extend_profile`), so nothing is guessed here.

    The canvas works differently here than for a fresh generation. `width` and
    `height` are the clip being continued, and a continuation *finishes* on
    them: the pinned run comes from that clip, and a chain whose resolution
    wandered from link to link would be an edit nobody could cut. So a
    single-pass tier ignores `megapixels` entirely and renders at the parent's
    size. A two-pass tier (`refine`) spends its first pass at `megapixels` — the
    small canvas where H3 obeys — and finishes at the parent's size, which is
    the two-stage trade made without changing what comes out.
    """
    canvas = VideoPrompt.blank()
    canvas.width, canvas.height = int(width), int(height)
    out_w, out_h = effective_canvas(canvas, None)
    # Two passes: the first is sized by the tier, the second by the parent.
    pass1_megapixels = megapixels if refine is not None else None

    seed = random_seed() if seed is None else int(seed)
    workflow = inject(
        profile,
        canvas,
        prompt_text,
        seed=seed,
        megapixels=pass1_megapixels,
        steps=steps,
        turbo_strength=turbo_strength,
        length=int(frames),
        duration_s=float(seconds),
        context_video=source_video,
        context_latent=source_latent,
        context_length=context_length,
    )
    if refine is not None:
        two_stage = apply_two_stage(
            workflow, pass1_steps=refine[0], width=out_w, height=out_h
        )
        # Pass 2 samples a bigger latent, and H3 packs continuation guides into
        # the target's own spatial grid — so it needs its own Motion Context.
        extend_two_stage(workflow, motion_context_ids(profile), two_stage)

    return PreparedRender(
        workflow=workflow,
        prompt_text=prompt_text,
        frames=int(frames),
        seconds=float(seconds),
        width=out_w,
        height=out_h,
        seed=seed,
        steps=steps,
        pass1_steps=refine[0] if refine else None,
        # A continuation saves an AV latent of its own, exactly as a fresh
        # render does, so extending *it* rides the clean-audio path. Recording
        # the parent's path here instead is how a chain used to keep feeding
        # itself its grandparent's sound.
        latent_path=attach_latent_save(workflow, seed),
        context_length=int(context_length) if context_length is not None
        else DEFAULT_CONTEXT_LENGTH,
        parent_take=parent_take,
        findings=[],
    )


def prepare_extend_batch(
    profile: RenderProfile,
    prompt_text: str,
    source_video: str,
    count: int,
    *,
    width: int,
    height: int,
    frames: int,
    seconds: float,
    seed: int | None = None,
    steps: int | None = None,
    turbo_strength: float | None = None,
    megapixels: float | None = None,
    refine: tuple[int, float] | None = None,
    source_latent: str | None = None,
    context_length: int | None = None,
    parent_take: str | None = None,
) -> list[PreparedRender]:
    """`count` continuations of one clip, one per seed.

    A continuation needs candidates for the same reason a fresh generation
    does: the pinned context fixes how the clip opens, but everything after it
    is the model's guess, and the cheap way to see a few guesses is the same
    continuation at different seeds. Everything but the seed is shared — same
    parent, same pinned run, same words — so the takes differ only in what the
    model invented.
    """
    if count < 1:
        raise ValueError(f"A batch needs at least one render, got {count}.")

    base = sweep_base(seed, count)
    return [
        prepare_extend(
            profile,
            prompt_text,
            source_video,
            width=width,
            height=height,
            frames=frames,
            seconds=seconds,
            seed=base + offset,
            steps=steps,
            turbo_strength=turbo_strength,
            megapixels=megapixels,
            refine=refine,
            source_latent=source_latent,
            context_length=context_length,
            parent_take=parent_take,
        )
        for offset in range(count)
    ]


def finding_dict(finding: Finding) -> dict[str, Any]:
    return {
        "rule": finding.rule,
        "severity": str(finding.severity),
        "message": finding.message,
        "where": finding.where,
    }
