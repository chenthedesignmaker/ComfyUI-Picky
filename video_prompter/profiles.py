"""Render profiles: a captured ComfyUI workflow plus where to inject into it.

The editor never serializes a live canvas. It captures a workflow once, records
which `[node_id, input_name]` pairs carry the prompt, the frame count, the
canvas and the seeds, and from then on renders by writing values into a copy of
that template.

The point of the indirection is that the editor knows nothing about H3's node
graph. Swap the sampler, add a LoRA, switch from `MiniMaxH3ImageToVideo` to
`MiniMaxH3ReferenceToVideo` — re-capture and the editor is unchanged.

Workflows are in ComfyUI's API format: `{node_id: {"class_type": ..., "inputs":
{...}}}`, where an input is either a literal or a `[node_id, slot]` wire.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import h3_spec
from .document import VideoPrompt

ROLE_PROMPT = "prompt"
# Frame count, written straight onto an H3 node's `length` input.
ROLE_LENGTH = "length"
# Duration in seconds, for workflows that compute their own frame count from
# it. Cannot be discovered — a float is just a float — so it must be pinned by
# hand with `RenderProfile.bind`.
ROLE_DURATION_S = "duration_s"
ROLE_WIDTH = "width"
ROLE_HEIGHT = "height"
ROLE_SEED = "seed"
# Render-quality knobs, filled from the shot's quality preset rather than the
# document. `megapixels` drives a canvas/resize node; `steps` a sampler.
ROLE_MEGAPIXELS = "megapixels"
ROLE_STEPS = "steps"
# The turbo LoRA's strength. The draft tiers keep it at 1.0; `final` bypasses it
# (0.0) so the full base model renders. Discovered by title (a LoRA loader whose
# node title mentions "turbo"), since strength inputs alone can't say which LoRA
# is the turbo one — so name that node "turbo" in the captured graph.
ROLE_TURBO_STRENGTH = "turbo_strength"
# Extend/continuation roles. `context_video` is the clip being continued (its
# frames and sound are pinned as the new clip's head); `context_latent` is that
# clip's saved AV latent, an optional higher-fidelity path for the audio carry.
# Neither is discovered — the extend profile pins them by hand — so they only
# ever carry a value when a caller passes one.
ROLE_CONTEXT_VIDEO = "context_video"
ROLE_CONTEXT_LATENT = "context_latent"
# How many frames of the source to pin. Fewer = a looser tie to the source and
# more room for a big action change; more = tighter continuity. Not discovered.
ROLE_CONTEXT_LENGTH = "context_length"

ROLES = (
    ROLE_PROMPT,
    ROLE_DURATION_S,
    ROLE_LENGTH,
    ROLE_WIDTH,
    ROLE_HEIGHT,
    ROLE_SEED,
    ROLE_MEGAPIXELS,
    ROLE_STEPS,
    ROLE_TURBO_STRENGTH,
    ROLE_CONTEXT_VIDEO,
    ROLE_CONTEXT_LATENT,
    ROLE_CONTEXT_LENGTH,
)

# Only the prompt is non-negotiable; a workflow may hardcode its own canvas.
REQUIRED_ROLES = (ROLE_PROMPT,)

# Nodes that carry a text prompt for the local H3 path. There is no negative
# prompt input on either.
PROMPT_NODES = (h3_spec.NODE_IMAGE_TO_VIDEO, h3_spec.NODE_REFERENCE_TO_VIDEO)

# Nodes that size the generation.
GEOMETRY_NODES = (
    h3_spec.NODE_EMPTY_LATENT,
    h3_spec.NODE_IMAGE_TO_VIDEO,
    h3_spec.NODE_REFERENCE_TO_VIDEO,
)

SEED_INPUTS = ("seed", "noise_seed")

# Quality-knob inputs, discovered by name on any node the way seeds are. A
# workflow may carry a `megapixels` input on a canvas sizer (ResolutionSelector)
# and again on an image resizer (ImageScaleToTotalPixels); binding both is
# intended, so a preview shrinks the canvas and the reference together.
MEGAPIXEL_INPUTS = ("megapixels",)
STEPS_INPUTS = ("steps",)

# The turbo LoRA is found by node title, not input name: any LoRA loader carries
# `strength_model`/`strength_clip`, so the graph marks the turbo one by titling
# its node with this hint. Both strength inputs get bound, so a bypass zeroes the
# model and CLIP strengths together.
TURBO_TITLE_HINT = "turbo"
TURBO_STRENGTH_INPUTS = ("strength_model", "strength_clip")

# ComfyUI's seed widgets go far higher, but a seed makes the whole round trip
# through the browser — into the workflow, back out in the queue metadata — and
# JavaScript numbers are float64. Past 2**53 consecutive seeds collapse onto the
# same value, which would quietly turn a sweep into N renders of one seed.
MAX_SEED = 2**53 - 1


def random_seed() -> int:
    return random.randint(0, MAX_SEED)


class ProfileError(ValueError):
    pass


class StaleProfileError(ProfileError):
    """A binding points at a node or input the workflow no longer has.

    Means the workflow changed under the profile; the fix is to re-capture, not
    to guess.
    """


@dataclass(frozen=True)
class Binding:
    node_id: str
    input_name: str

    def to_list(self) -> list[str]:
        return [self.node_id, self.input_name]

    @classmethod
    def from_any(cls, value: Any) -> "Binding":
        if isinstance(value, Binding):
            return value
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return cls(str(value[0]), str(value[1]))
        raise ProfileError(f"Not a binding: {value!r}")


@dataclass
class RenderProfile:
    """A workflow template and the bindings that make it renderable."""

    name: str
    api_format: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, list[Binding]] = field(default_factory=dict)
    # The original litegraph workflow, kept verbatim so it can be put back on
    # the canvas and re-captured after an edit.
    canvas_workflow: dict[str, Any] | None = None

    def targets(self, role: str) -> list[Binding]:
        return list(self.bindings.get(role) or [])

    def controls(self, role: str) -> bool:
        return bool(self.bindings.get(role))

    def bind(self, role: str, node_id: str, input_name: str) -> "RenderProfile":
        """Pin a binding discovery could not infer, and return self.

        Needed whenever a workflow routes a value through nodes of its own —
        a duration in seconds feeding a math expression, say — where the type
        alone does not say what the input is for.
        """
        if role not in ROLES:
            raise ProfileError(f"Unknown role {role!r}; expected one of {', '.join(ROLES)}.")
        binding = Binding(str(node_id), str(input_name))
        targets = self.bindings.setdefault(role, [])
        if binding not in targets:
            targets.append(binding)
        return self

    def describe(self) -> str:
        """What this profile can and cannot drive, for the editor and the CLI."""
        held = [role for role in ROLES if self.controls(role)]
        loose = [role for role in ROLES if not self.controls(role)]
        lines = [f"{self.name}: controls {', '.join(held) or 'nothing'}"]
        if loose:
            lines.append(f"  the workflow owns: {', '.join(loose)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "api_format": self.api_format,
            "bindings": {
                role: [binding.to_list() for binding in targets]
                for role, targets in self.bindings.items()
                if targets
            },
            "canvas_workflow": self.canvas_workflow,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderProfile":
        return cls(
            name=data.get("name", ""),
            api_format=data.get("api_format") or {},
            bindings={
                role: [Binding.from_any(entry) for entry in entries or []]
                for role, entries in (data.get("bindings") or {}).items()
            },
            canvas_workflow=data.get("canvas_workflow"),
        )


def discover_bindings(api_format: dict[str, Any]) -> dict[str, list[Binding]]:
    """Work out where to inject by reading the workflow.

    Deliberately narrow: only literal inputs on nodes we recognise. Anything
    already wired to another node is left alone, because a wire means the
    author meant it to come from somewhere else.
    """
    found: dict[str, list[Binding]] = {role: [] for role in ROLES}

    for node_id, node in _nodes(api_format):
        class_type = node.get("class_type")
        inputs = node.get("inputs") or {}

        if class_type in PROMPT_NODES and _is_literal(inputs.get(ROLE_PROMPT)):
            found[ROLE_PROMPT].append(Binding(node_id, ROLE_PROMPT))

        if class_type in GEOMETRY_NODES:
            for role in (ROLE_LENGTH, ROLE_WIDTH, ROLE_HEIGHT):
                if _is_literal(inputs.get(role)):
                    found[role].append(Binding(node_id, role))

        for name in SEED_INPUTS:
            if _is_literal(inputs.get(name)):
                found[ROLE_SEED].append(Binding(node_id, name))

        for name in MEGAPIXEL_INPUTS:
            if _is_literal(inputs.get(name)):
                found[ROLE_MEGAPIXELS].append(Binding(node_id, name))

        for name in STEPS_INPUTS:
            if _is_literal(inputs.get(name)):
                found[ROLE_STEPS].append(Binding(node_id, name))

        title = str((node.get("_meta") or {}).get("title") or "")
        if TURBO_TITLE_HINT in title.casefold():
            for name in TURBO_STRENGTH_INPUTS:
                if _is_literal(inputs.get(name)):
                    found[ROLE_TURBO_STRENGTH].append(Binding(node_id, name))

    return {role: targets for role, targets in found.items() if targets}


def capture(
    name: str,
    api_format: dict[str, Any],
    canvas_workflow: dict[str, Any] | None = None,
    bindings: dict[str, list[Binding]] | None = None,
) -> RenderProfile:
    """Build a profile from a workflow, discovering bindings unless given."""
    profile = RenderProfile(
        name=name,
        api_format=api_format,
        bindings=bindings if bindings is not None else discover_bindings(api_format),
        canvas_workflow=canvas_workflow,
    )
    validate(profile)
    return profile


# Roles added after the first profiles were captured. Those profiles were frozen
# without them, so loading re-discovers just these and fills any that are
# missing — otherwise a fine render is a bigger canvas at the same step count
# ("bigger, not better"), and nobody should have to re-capture to fix that.
BACKFILL_ROLES = (ROLE_MEGAPIXELS, ROLE_STEPS, ROLE_TURBO_STRENGTH)


def backfill_quality_roles(profile: RenderProfile) -> RenderProfile:
    """Fill in newly-added roles a stored profile predates, in place.

    Only touches roles it has none of, so a binding someone deliberately edited
    away is left alone. The discovery pass is the same narrow one a fresh
    capture runs, so the result matches re-capturing.
    """
    discovered = discover_bindings(profile.api_format)
    for role in BACKFILL_ROLES:
        if not profile.bindings.get(role) and discovered.get(role):
            profile.bindings[role] = discovered[role]
    return profile


def validate(profile: RenderProfile) -> None:
    """Check the profile can still drive its workflow.

    Raises `StaleProfileError` when a binding has gone missing, which is the
    signal to re-capture rather than to patch around it.
    """
    if not profile.api_format:
        raise ProfileError(f"Profile {profile.name!r} has no workflow.")

    for role in REQUIRED_ROLES:
        if not profile.targets(role):
            raise ProfileError(
                f"Profile {profile.name!r} has no {role} binding. Nothing to render into."
            )

    for role, targets in profile.bindings.items():
        if role not in ROLES:
            raise ProfileError(f"Profile {profile.name!r} has an unknown role {role!r}.")
        for binding in targets:
            node = profile.api_format.get(binding.node_id)
            if node is None:
                raise StaleProfileError(
                    f"{role} binding points at node {binding.node_id}, which the "
                    f"workflow no longer has. Re-capture the profile."
                )
            if binding.input_name not in (node.get("inputs") or {}):
                raise StaleProfileError(
                    f"{role} binding points at input {binding.input_name!r} on node "
                    f"{binding.node_id}, which no longer exists. Re-capture the profile."
                )


def effective_canvas(
    prompt: VideoPrompt, megapixels: float | None, *, upscale: bool = False
) -> tuple[int, int]:
    """The width/height a render will really use for a megapixel budget.

    Keeps the shot's aspect ratio and re-sizes it to the budget; with no budget
    the document's own canvas stands. Used both to inject the numbers and to
    report the true resolution back, so a take records what it rendered at.
    `upscale` lets a budget above the 768-edge nominal grow the canvas (see
    `h3_spec.adapt_canvas`).
    """
    if megapixels is None:
        return prompt.width, prompt.height
    return h3_spec.adapt_canvas(
        prompt.width, prompt.height, max_pixels=megapixels * 1e6, upscale=upscale
    )


def inject(
    profile: RenderProfile,
    prompt: VideoPrompt,
    prompt_text: str,
    seed: int | None = None,
    megapixels: float | None = None,
    steps: int | None = None,
    turbo_strength: float | None = None,
    length: int | None = None,
    duration_s: float | None = None,
    context_video: str | None = None,
    context_latent: str | None = None,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Return a submittable workflow with this prompt's values written in.

    `prompt_text` is passed separately rather than composed here so the caller
    controls exactly what gets sent — the editor shows a preview, and what it
    shows is what renders.

    `megapixels`, `steps` and `turbo_strength` come from the shot's quality
    preset. A `megapixels` budget both feeds any `megapixels` node and re-sizes
    the width/height bindings, so whichever mechanism a profile uses to size the
    render is driven by the one number. `turbo_strength` drives the turbo LoRA
    (0.0 bypasses it for the keeper). Any left `None` leaves the workflow's own
    value.

    `length` and `duration_s` override the frame count and duration primitive.
    A re-shoot passes the take's own frozen numbers here: an aligned frame count
    does not always map back to a whole-second duration, so it cannot be rebuilt
    from a `Storyboard` without drift.
    """
    validate(profile)

    workflow = json.loads(json.dumps(profile.api_format))
    width, height = effective_canvas(prompt, megapixels)
    values: dict[str, Any] = {
        ROLE_PROMPT: prompt_text,
        # Seconds as authored; the workflow does its own alignment. Frames as
        # aligned; the node would snap anyway, and this way the number in the
        # graph matches the number the linter reported.
        ROLE_DURATION_S: float(prompt.storyboard.duration) if duration_s is None else float(duration_s),
        ROLE_LENGTH: prompt.frames if length is None else int(length),
        ROLE_WIDTH: width,
        ROLE_HEIGHT: height,
    }
    if megapixels is not None:
        values[ROLE_MEGAPIXELS] = float(megapixels)
    if steps is not None:
        values[ROLE_STEPS] = int(steps)
    # A profile without a turbo LoRA simply has no binding, so this is a no-op
    # there; where the LoRA exists, the keeper's 0.0 bypasses it.
    if turbo_strength is not None:
        values[ROLE_TURBO_STRENGTH] = float(turbo_strength)
    # Extend-only: the source clip to continue, and optionally its saved latent.
    # A profile without these bindings simply ignores them.
    if context_video is not None:
        values[ROLE_CONTEXT_VIDEO] = str(context_video)
    if context_latent is not None:
        values[ROLE_CONTEXT_LATENT] = str(context_latent)
    if context_length is not None:
        values[ROLE_CONTEXT_LENGTH] = int(context_length)

    for role, value in values.items():
        for binding in profile.targets(role):
            workflow[binding.node_id]["inputs"][binding.input_name] = value

    # A seed decided here would be lost — the caller records what it passed in,
    # so a render nobody chose a seed for could never be reproduced. Callers
    # that care mint one first; this fallback only covers direct use.
    chosen = random_seed() if seed is None else seed
    for binding in profile.targets(ROLE_SEED):
        workflow[binding.node_id]["inputs"][binding.input_name] = chosen

    return workflow


def _nodes(api_format: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for node_id, node in api_format.items():
        if isinstance(node, dict):
            yield str(node_id), node


def _is_literal(value: Any) -> bool:
    """True for a hardcoded input; False for a `[node_id, slot]` wire or absence."""
    return value is not None and not isinstance(value, (list, tuple, dict))


class ProfileStore:
    """Profiles as one JSON file each. A shared library, not per-project."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, name: str) -> Path:
        from .presets import sanitize_name

        return self.root / f"{sanitize_name(name)}.json"

    def list(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted((path.stem for path in self.root.glob("*.json")), key=str.casefold)

    def exists(self, name: str) -> bool:
        return self.path_for(name).is_file()

    def save(self, profile: RenderProfile) -> Path:
        validate(profile)
        path = self.path_for(profile.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(profile.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def load(self, name: str) -> RenderProfile:
        path = self.path_for(name)
        if not path.is_file():
            raise ProfileError(f"No render profile named {name!r}.")
        with path.open("r", encoding="utf-8") as handle:
            profile = RenderProfile.from_dict(json.load(handle))
        # Upgrade profiles captured before the quality roles existed, so a fine
        # render actually drives steps and megapixels without a re-capture.
        return backfill_quality_roles(profile)

    def delete(self, name: str) -> bool:
        path = self.path_for(name)
        if not path.is_file():
            return False
        path.unlink()
        return True
