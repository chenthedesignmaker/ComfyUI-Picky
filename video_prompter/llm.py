"""Ask a language model for a prompt document, and take back only what it may write.

The tool already publishes everything a model needs to author for it: `contract`
says what the fields mean, `lint` says what "good" is, and `transform` does the
arithmetic. What was missing was a caller. This is that caller.

Two ideas carry the design.

**The schema is the guard rail.** The model is asked for structured JSON matching
`response_schema()`, which covers the *authored* fields and nothing else. The
media attachments, the canvas and the unused `negatives` block are simply not in
the shape it can return, so no rewrite can clobber a keyframe or resize a shot,
however the instruction was phrased. Wanting a field the model must not touch is
a reason to leave it out of the schema, never a reason to ask it nicely.

**Scope is enforced at the merge, not requested in the prose.** A targeted tweak
sends the whole document — "make this faster" may need the beat spans *and* the
camera phrasing, so a model shown one text box cannot do what was asked — and the
answer is then merged only into the fields that were in scope. Asking for
restraint and checking for it are different things; this does the second.

The provider is an interface with one implementation because a second one is a
small file, and guessing now at the shape of it would cost more than it saves.
Nothing here imports `Api`, so it can be tested against a fake provider without
a network.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from . import contract, paths, render
from .document import VideoPrompt
from .lint import Severity, lint
from .serialize import prompt_from_dict

# Where the key and the model name come from. Either way the key is read in the
# server process and never leaves it: the editor is served over plain HTTP on a
# machine that is not the only thing on its network, so a browser that could see
# the key would be handing it to everything else that can reach the port.
#
# Two sources, and the second is the one that matters day to day. An environment
# variable is only read when a process starts, and ComfyUI's own Restart button
# re-executes in place — it reloads every custom node but inherits the
# environment it already had, so a key added to the launcher after startup stays
# invisible until someone finds the console window. The settings file is read at
# each use, so editing it and pressing Restart is enough.
#
# The environment still wins where both are set: it is the more deliberate of
# the two, and it matches every other override in `paths`.
KEY_VARS = ("VIDEO_PROMPTER_GEMINI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
MODEL_VAR = "VIDEO_PROMPTER_GEMINI_MODEL"
SETTINGS_KEY = "gemini_key"
SETTINGS_MODEL = "gemini_model"
# Writing a whole prompt from a sentence is a different job from nudging one
# field, and worth a different model. Unset, it is the same one: a second
# default nobody chose would be a surprise on the bill and in the latency.
SETTINGS_MODEL_WHOLE = "gemini_model_whole"
# An alias rather than a pinned version, so the flash tier keeps working when
# Google retires a point release. `GET /llm/status` lists what the key can
# actually see, which is the honest way to find out what to put here.
DEFAULT_MODEL = "gemini-flash-latest"

GEMINI_ROOT = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT_S = 60

# Statuses that mean "not now" rather than "not ever": rate limiting, and the
# capacity 503 the flash tiers hand out at busy times. Worth sleeping through,
# because the alternative is an editor that shows a red error for something
# that fixes itself in two seconds.
RETRY_STATUSES = (429, 500, 502, 503, 504)
RETRIES = 3
BACKOFF_S = 2.0

# How many times a draft that still trips the linter is handed back with the
# findings. Two covers the mistakes models actually make here — a beat that
# overruns the budget, a missing soundscape — and past that it is not repairing,
# it is rerolling on someone else's clock.
REPAIR_ATTEMPTS = 2

# A wall clock over the whole draft, repairs included. Someone is watching a
# spinner: three repair turns each costing three transport retries at a 60s
# timeout is nine minutes, which is not a slow answer, it is a hung editor.
# Passing this stops the repairs and hands back what there is, findings and all.
DRAFT_BUDGET_S = 100.0


class LlmError(Exception):
    """Something went wrong reaching or reading the model."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


class Provider(Protocol):
    """Whatever can turn an instruction into a prompt document."""

    name: str
    model: str

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return the JSON object the model produced. Raises `LlmError`."""


# -- what the model is allowed to write --------------------------------------

# The authored fields, and only those. Everything absent here is carried over
# from the prompt as it stands: `references`, `first_frame`/`last_frame` name
# files the model has never seen; `width`/`height` come from the aspect ratio and
# the quality budget; `negatives` is deliberately unused (see the contract);
# `storyboard_header`/`label_beats` are formatting the editor owns.
AUTHORED_FIELDS = (
    "duration", "beats", "cues", "style", "overview", "camera",
    "audio", "music", "screen_text",
)

_STR = {"type": "string"}


def response_schema(scope: "Scope | None" = None) -> dict[str, Any]:
    """The JSON shape the model must answer in, narrowed to what may land.

    Only the scoped fields are asked for. This began as "always return the
    whole document, we will keep the part in scope", which was wrong twice
    over. Answering about the music meant regenerating the entire storyboard,
    so a call cost time in proportion to the *document* rather than to the
    request. And a field regenerated from scratch comes back blander than the
    one somebody tuned by hand — which is why asking to change one line of a
    detailed prompt read as the tool quietly simplifying all of it.

    The model still *sees* the whole document; it just does not *write* it.
    Seeing it is what lets a tweak sit properly in its context. Writing it was
    what made a tweak destructive.

    Everything asked for is required: a partial answer merges as "the user
    asked me to delete the music", and an empty string says that far more
    clearly than an absent key.
    """
    aim = scope or Scope.everything()
    line = {
        "type": "object",
        "properties": {
            "speakers": {"type": "array", "items": _STR},
            "text": _STR,
            "language": _STR,
            "verb": _STR,
            "delivery": _STR,
            "describe": _STR,
            "voiceover": {"type": "boolean"},
            "possessive": _STR,
            # A plain string, not an enum: the API refuses an empty string as
            # an enum value, and "no seam here" is the answer for almost every
            # line. `serialize._continuity` already drops anything that is not
            # "into" or "from", and the linter catches a seam tagged on one
            # side only, so tolerance costs nothing and an enum costs a 400.
            "continuity": _STR,
            "cutoff": {"type": "boolean"},
            # Same reasoning as `continuity`: "" is the answer for most lines
            # and an enum cannot carry it. `serialize._placement` drops
            # anything that is not "open".
            "placement": _STR,
        },
        "required": ["speakers", "text"],
    }
    beat = {
        "type": "object",
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
            "angle": _STR,
            "action": _STR,
            "camera": _STR,
            "transition": _STR,
            "lines": {"type": "array", "items": line},
        },
        "required": ["start", "end", "angle", "action", "camera", "transition", "lines"],
    }
    cue = {
        "type": "object",
        "properties": {
            "start": {"type": "number"},
            # There is no nullable number here, and an instant hit is the common
            # case, so it is said out loud rather than by omitting `end`.
            "end": {"type": "number"},
            "instant": {"type": "boolean"},
            "description": _STR,
        },
        "required": ["start", "description"],
    }
    every = {
        "duration": {"type": "integer"},
        "beats": {"type": "array", "items": beat},
        "cues": {"type": "array", "items": cue},
        "style": _STR,
        "overview": _STR,
        "camera": _STR,
        "audio": _STR,
        "music": _STR,
        "screen_text": {"type": "array", "items": _STR},
    }
    wanted = {name: every[name] for name in aim.fields if name in every}
    return {"type": "object", "properties": wanted, "required": list(wanted)}


# -- scope --------------------------------------------------------------------


@dataclass(frozen=True)
class Scope:
    """Which part of the document a rewrite is allowed to land in.

    `fields` are top-level names; `beat` is an index when only one shot is in
    scope. `Scope.everything()` is the whole-document case.
    """

    fields: tuple[str, ...]
    beat: int | None = None

    @classmethod
    def everything(cls) -> "Scope":
        return cls(fields=AUTHORED_FIELDS)

    @classmethod
    def parse(cls, raw: Any) -> "Scope":
        """Read the wire form: "all", a field name, "beat:N", or a list of names."""
        if raw in (None, "", "all"):
            return cls.everything()
        if isinstance(raw, list):
            names = tuple(str(x) for x in raw if str(x) in AUTHORED_FIELDS)
            if not names:
                raise LlmError(f"No writable field in scope {raw!r}.", 400)
            return cls(fields=names)

        text = str(raw)
        if text.startswith("beat:"):
            try:
                index = int(text.split(":", 1)[1])
            except ValueError:
                raise LlmError(f"Unreadable beat scope {text!r}.", 400) from None
            if index < 0:
                raise LlmError(f"Beat index {index} is not a beat.", 400)
            return cls(fields=("beats",), beat=index)
        if text in AUTHORED_FIELDS:
            return cls(fields=(text,))
        raise LlmError(f"{text!r} is not a field a rewrite can land in.", 400)

    def describe(self) -> str:
        """How the scope is put to the model, so its answer is aimed the right way."""
        if self.fields == AUTHORED_FIELDS and self.beat is None:
            return "the whole prompt"
        if self.beat is not None:
            return f"shot {self.beat + 1} only (beats[{self.beat}])"
        return " and ".join(self.fields)


def merge(current: dict[str, Any], drafted: dict[str, Any], scope: Scope) -> dict[str, Any]:
    """Fold a model's answer into the prompt, taking only what was in scope.

    Everything outside it comes through untouched — which is what makes a
    targeted tweak safe to apply without asking first. A single-beat scope
    replaces that one beat and leaves its neighbours, including their timings,
    exactly as they were; a rewrite that needs the spans moved is a `beats`
    scope, not a `beat:N` one.
    """
    merged = dict(current)
    for name in scope.fields:
        if name not in drafted:
            continue
        if name == "beats" and scope.beat is not None:
            beats = list(merged.get("beats") or [])
            replacement = drafted.get("beats") or []
            if not replacement:
                continue
            if scope.beat >= len(beats):
                raise LlmError(f"This prompt has no shot {scope.beat + 1}.", 400)
            # The model was shown the whole document, so when it hands back the
            # whole storyboard the beat in scope is the one at that index. A
            # shorter answer is one beat aimed at the scope; take its first.
            picked = (
                replacement[scope.beat]
                if scope.beat < len(replacement)
                else replacement[0]
            )
            beats[scope.beat] = picked
            merged["beats"] = beats
        else:
            merged[name] = drafted[name]
    return _read_cues(merged)


def _read_cues(prompt: dict[str, Any]) -> dict[str, Any]:
    """Turn the schema's `instant` flag back into the `end: null` the tool stores."""
    cues = prompt.get("cues")
    if not isinstance(cues, list):
        return prompt
    prompt["cues"] = [
        {
            "start": cue.get("start", 0),
            "end": None if cue.get("instant") or cue.get("end") is None else cue["end"],
            "description": cue.get("description", ""),
        }
        if isinstance(cue, dict)
        else cue
        for cue in cues
    ]
    return prompt


# -- the briefing -------------------------------------------------------------


def system_prompt() -> str:
    """What the model is told about the job, built from the published contract.

    Generated rather than written out, so the field rules can never drift from
    the ones the linter actually enforces — a second hand-maintained copy of the
    schema is a bug with a delay on it.
    """
    spec = contract.schema()
    return "\n".join([
        "You author prompt documents for MiniMax-H3, a video generation model, "
        "inside a non-linear editor.",
        "",
        "You never write finished prompt text. You fill in a STRUCTURED document; "
        "the tool composes it into H3's official format itself. Answer with JSON "
        "matching the schema you were given and nothing else.",
        "",
        "## The fields",
        json.dumps(spec["prompt_fields"], indent=1, ensure_ascii=False),
        "",
        "## The model's hard limits",
        json.dumps(spec["limits"], indent=1, ensure_ascii=False),
        "",
        "## Camera vocabulary",
        json.dumps(spec["camera_vocabulary"], indent=1, ensure_ascii=False),
        "",
        "## Dialogue vocabulary",
        json.dumps(spec["dialogue_vocabulary"], indent=1, ensure_ascii=False),
        "",
        "## What the linter will check",
        json.dumps(spec["lint_rules"], indent=1, ensure_ascii=False),
        "",
        "## House rules",
        "- Beat spans must tile the duration exactly: no gaps, no overlaps, the "
        "first starting at 0 and the last ending at the duration.",
        "- Camera motion is woven into each beat's `camera` as motion type + "
        "amplitude ('with small/large amplitude') + speed ('at slow/fast "
        "speed'), phrased to follow 'The camera ...'. Never a trailing label.",
        "- `action` is a phrase that can follow the framing clause ('steam "
        "rising from the bread'), not a predicate ('frames the baker').",
        "- `audio` is diegetic sound only — what the scene itself makes — "
        "written as 1-4 sentences, never a comma list, and never empty: H3 "
        "makes picture and sound in the same pass.",
        "- `music` is the non-diegetic score only the audience hears: "
        "instrumentation, tempo, rhythm, dynamics. Not mood words. Empty is "
        "fine and composes to N/A. Music a character can hear is diegetic and "
        "belongs in the shot instead.",
        "- Leave `screen_text` empty unless on-screen words were actually asked "
        "for; naming text tends to burn it into the frame.",
        "- `overview` is lower-case and unpunctuated ('a rain-soaked alley at "
        "dusk'), not a sentence.",
    ])


def user_prompt(current: dict[str, Any], instruction: str, scope: Scope) -> str:
    """The turn itself: what exists, what was asked, and what may change."""
    blank = not any(
        current.get(name) for name in ("beats", "style", "audio", "music", "overview")
    )
    parts: list[str] = []
    if blank:
        parts.append("There is no prompt yet. Write one from this description:")
    else:
        parts += [
            "Here is the prompt as it stands. All of it is context; only the "
            "fields named at the end are yours to answer with.",
            json.dumps(
                {name: current.get(name) for name in AUTHORED_FIELDS},
                indent=1,
                ensure_ascii=False,
            ),
            "",
            "The person who wrote it asked for this change:",
        ]
    parts += [instruction.strip()]

    if not blank:
        # Without this, a rewrite reads as "write me one of these", and back
        # comes a competent generic version of a prompt somebody spent an hour
        # making specific. These fields are finished work, not a first draft.
        parts += [
            "",
            "This prompt was written deliberately. It is not a draft to "
            "improve and not a text to tidy. Make the change asked for and "
            "leave everything else as it is — the level of detail, the "
            "length, the specific language. Keep named characters and places, "
            "the particular props, materials and light, the established look "
            "and its vocabulary. Do not summarise, generalise, or swap a "
            "concrete phrase for a smoother one. Where the change touches "
            "only part of a field, the rest of that field comes back word for "
            "word.",
        ]

    if scope.beat is not None:
        parts += [
            "",
            f"Answer with `beats` as a list holding only shot {scope.beat + 1}, "
            "rewritten. Its neighbours are not yours to touch.",
        ]
    else:
        named = ", ".join(f"`{name}`" for name in scope.fields)
        parts += ["", f"Answer with {named} and nothing else."]

    if not blank and "duration" in scope.fields:
        # The schema asks for `duration`, so the model has to answer with one,
        # and left alone it re-picks a length to suit the scene it just wrote.
        # That silently overrides the number someone typed into the editor, so
        # it is spelled out: the budget is a decision already taken.
        parts += [
            "",
            f"Keep the duration at {current.get('duration')}s unless the "
            "instruction above asks for a different length. It is a setting "
            "someone chose, not a suggestion — fit the shots to it.",
        ]
    return "\n".join(parts)


def repair_prompt(findings: list[dict[str, Any]]) -> str:
    """The follow-up turn when a draft still trips the linter."""
    lines = [f"- {f.get('rule', '?')}: {f.get('message', '')}" for f in findings]
    return "\n".join([
        "That draft does not pass the tool's linter yet:",
        *lines,
        "",
        "Fix exactly those. Answer with the same fields as before and change "
        "nothing else about them.",
    ])


# -- the loop -----------------------------------------------------------------


@dataclass
class Draft:
    """What came back: the prompt, what moved in it, and whether it is clean."""

    prompt: dict[str, Any]
    findings: list[dict[str, Any]]
    attempts: int
    changed: list[str]
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "findings": self.findings,
            "attempts": self.attempts,
            "changed": self.changed,
            "model": self.model,
        }


# Lint rules about things outside `AUTHORED_FIELDS`: the canvas, the media
# attachments, and the keyframe-versus-references mode they decide between.
# Handing one of these back would ask the model to fix something it has no field
# to fix it with, and it would spend both repair turns finding that out.
UNWRITABLE_RULES = ("canvas.", "references.", "ref.", "mode.")

# Which authored field a lint rule is really about, by rule prefix. Used to keep
# the repair loop inside the scope it was given: with a `beat:2` scope the merge
# keeps only that one beat, so a model trying to close a gap by moving its
# neighbours has its fix thrown away — and is then asked again, and again, for
# something it was never allowed to change. `prompt.` is absent on purpose: an
# over-long prompt can be shortened from any field, so it is always in scope.
RULE_FIELDS = {
    "duration.": "duration",
    "storyboard.": "beats",
    "beat.": "beats",
    "dialogue.": "beats",
    "audio.": "audio",
    "style.": "style",
}


def reachable(rule: str, scope: "Scope") -> bool:
    """Whether a fix for `rule` could survive this scope's merge."""
    field = next((f for prefix, f in RULE_FIELDS.items() if rule.startswith(prefix)), None)
    if field is None:
        return True  # a whole-document rule; any field can answer it
    return field in scope.fields


def findings_for(prompt: VideoPrompt) -> list[dict[str, Any]]:
    """Everything the linter has to say, minus the notes-to-self."""
    return [
        render.finding_dict(finding)
        for finding in lint(prompt)
        if finding.severity in (Severity.ERROR, Severity.WARNING)
    ]


def actionable(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    scope: "Scope | None" = None,
) -> list[dict[str, Any]]:
    """The findings worth spending a repair turn on.

    Three filters, and each one exists because without it the loop asks for
    something it will then throw away:

    - **Writable.** A finding about the canvas or the media attachments is not
      the model's to fix; it has no field for it.
    - **Not inherited.** A prompt that arrived with a keyframe *and* references
      was broken before anyone touched it. Handing that back with "change
      nothing else" burns the turns the real mistakes needed.
    - **In scope.** The merge keeps only the scoped fields, so a fix outside
      them is discarded and the finding comes straight back. That is not a
      repair loop, it is a spin.

    Warnings count, not only errors. A storyboard with a seven-second hole in it
    passes as a warning and would still waste a render, which is exactly the
    mistake worth one more turn.
    """
    aim = scope or Scope.everything()
    was = {(f.get("rule"), f.get("message")) for f in before}
    return [
        finding for finding in after
        if (finding.get("rule"), finding.get("message")) not in was
        and not str(finding.get("rule", "")).startswith(UNWRITABLE_RULES)
        and reachable(str(finding.get("rule", "")), aim)
    ]


def draft(
    provider: Provider,
    current: dict[str, Any],
    instruction: str,
    scope: Scope | None = None,
) -> Draft:
    """Ask for a rewrite, repair it against the linter, and hand it back.

    The repair loop lives here rather than at the caller because a draft with an
    error finding is not a draft: `/render/prepare` refuses it with a 422, so
    passing one on would only move the failure somewhere less useful. The
    findings that come back are everything the linter still says, repaired or
    not — see `actionable` for which of them were worth a turn.
    """
    if not instruction.strip():
        raise LlmError("Say what you want written or changed.", 400)

    aim = scope or Scope.everything()
    schema = response_schema(aim)
    system = system_prompt()
    turn = user_prompt(current, instruction, aim)

    # What was already wrong before anyone touched it, so the repair loop can
    # tell the model's mistakes from the ones it inherited.
    inherited = findings_for(prompt_from_dict(current))

    merged = dict(current)
    findings: list[dict[str, Any]] = []
    attempt = 0
    started = time.monotonic()
    for attempt in range(1, REPAIR_ATTEMPTS + 2):
        answered = provider.complete(system, turn, schema)
        merged = merge(current, answered, aim)
        findings = findings_for(prompt_from_dict(merged))
        todo = actionable(inherited, findings, aim)
        if not todo:
            break
        if time.monotonic() - started > DRAFT_BUDGET_S:
            break
        turn = repair_prompt(todo)

    return Draft(
        prompt=merged,
        findings=findings,
        attempts=attempt,
        changed=changed_fields(current, merged),
        model=provider.model,
    )


def changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Which authored fields actually moved. What the editor shows and logs."""
    return [name for name in AUTHORED_FIELDS if before.get(name) != after.get(name)]


# -- writing an image prompt --------------------------------------------------
#
# A separate path from `draft`, not a scope of it, and the reason is the shape of
# the thing being written rather than the model doing the writing.
#
# Everything protective above is built out of the document's field names. The
# response schema narrows to the fields in scope, so a rewrite of the music has
# nowhere to put a new keyframe. `merge` keeps only those fields, so restraint is
# checked rather than requested. The repair loop asks `lint` whether the answer
# is renderable and spends a turn when it is not. An image prompt is one string:
# no fields, so no narrowing; no "only this part", so no merge; and no image
# linter, so nothing for a repair turn to check against.
#
# What is left is the one guard rail that does survive — the model answers in a
# schema holding a prompt and nothing else, so it cannot reply with commentary
# the editor would paste into the box — plus an undo. `image_prompts.py` is that
# undo, and it is what makes replacing the whole box safe to apply straight away
# rather than as a proposal to accept.

IMAGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"prompt": _STR},
    "required": ["prompt"],
}


@dataclass
class ImageDraft:
    """What came back for an image prompt: the words, and what they replaced."""

    prompt: str
    before: str
    changed: bool
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "before": self.before,
            "changed": self.changed,
            "model": self.model,
        }


def image_system_prompt() -> str:
    """What the model is told about writing for the keyframe generator.

    Hand-written rather than generated from `contract`, because the contract
    describes the H3 prompt document and this is not one. The rules below are
    about the image model actually behind the button — a Krea 2 Turbo distilled
    to a handful of steps at cfg 1.0, read by a Qwen3-VL text encoder — so they
    are kept beside the workflow that states those facts (`t2i.py`) rather than
    invented from general image-prompt folklore.
    """
    return "\n".join([
        "You write prompts for a text-to-image model, inside a video editor "
        "where the picture is used as the opening frame of a shot.",
        "",
        "Answer with JSON matching the schema you were given: the prompt text "
        "and nothing else. No preamble, no explanation, no options to choose "
        "between.",
        "",
        "## The model you are writing for",
        "- Krea 2 Turbo, read by a Qwen3-VL text encoder. It reads ordinary "
        "descriptive English, so write flowing sentences, not a comma-separated "
        "pile of tags.",
        "- It runs guidance-free (cfg 1.0) with the negative conditioning zeroed "
        "out. Negative phrasing does nothing at all — 'no people', 'without "
        "text', 'avoid blur' are read as the very things they forbid. Say what "
        "IS in the picture; if something must be absent, describe the empty "
        "version of the scene instead.",
        "",
        "## House rules",
        "- It is one still frame. Nothing moves, nothing happens over time. Turn "
        "any motion the author describes into the instant it is caught at — "
        "'running' becomes 'mid-stride, one heel lifted'.",
        "- Lead with the subject and what it is doing, then where it is, then "
        "the light, then the framing and lens. Composition is worth more words "
        "than adjectives are.",
        "- No quality filler. 'masterpiece', '8k', 'highly detailed', 'award "
        "winning', trending-on-anything: these buy nothing on this model and "
        "crowd out the description that would.",
        "- A separate Look control appends the film stock, grade and lens "
        "language after your text. Do not write a grade unless the author asked "
        "for one by name — you would be arguing with it.",
        "- A paragraph or two. Long enough to fix the composition, short enough "
        "that every clause is doing work.",
    ])


def image_user_prompt(current: str, instruction: str, style: str = "") -> str:
    """The turn itself: what is in the box, and what to do about it."""
    said = instruction.strip()
    have = (current or "").strip()
    lines: list[str] = []

    if have:
        lines += [
            "The author's current image prompt:",
            "",
            have,
            "",
            "Rewrite it as they ask. Keep everything they did not ask you to "
            "change — the subject, the setting and the framing they already "
            "chose are theirs, and a rewrite that quietly restyles all of it "
            "reads as the tool losing their work.",
        ]
    else:
        lines += [
            "The box is empty. Write the image prompt from their description "
            "below, inventing the specifics they left open — the light, the "
            "framing, the details that make one particular picture rather than "
            "a category of picture.",
        ]

    if style:
        lines += [
            "",
            f"The Look control is set to {style!r}, and its own language is "
            "appended after your text. Leave the grade to it.",
        ]

    lines += ["", "What they asked for:", "", said]
    return "\n".join(lines)


def draft_image(
    provider: Provider,
    current: str,
    instruction: str,
    style: str = "",
) -> ImageDraft:
    """Write or rewrite one image prompt. One turn — there is nothing to repair.

    `draft` loops because `lint` can tell it the answer is unrenderable. No such
    check exists for an image prompt, so a second turn would be asking the same
    question again and hoping. The undo covers what the loop cannot.
    """
    if not instruction.strip():
        raise LlmError("Say what you want written or changed.", 400)

    answered = provider.complete(
        image_system_prompt(),
        image_user_prompt(current, instruction, style),
        IMAGE_RESPONSE_SCHEMA,
    )
    written = str(answered.get("prompt") or "").strip()
    if not written:
        raise LlmError(f"{provider.model} answered with an empty prompt.", 502)

    before = (current or "").strip()
    return ImageDraft(
        prompt=written,
        before=before,
        changed=written != before,
        model=provider.model,
    )


# -- Gemini -------------------------------------------------------------------


def read_settings() -> dict[str, Any]:
    """The settings file as a dict, or empty. Never raises.

    A missing file is the normal case, and a malformed one must not stop
    ComfyUI loading this node — losing the editor entirely over a stray comma
    in an optional file would be a far worse trade than losing the model.
    """
    try:
        text = paths.settings_path().read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class Configured:
    """What was found, and where. `source` is empty when there is no key.

    `source` exists for one reason: when the model is not answering, the first
    question is always which key it is using, and guessing at that from two
    places is exactly the half hour this field saves.
    """

    key: str = ""
    model: str = DEFAULT_MODEL
    # The model for a whole-document write. Falls back to `model`.
    whole_model: str = DEFAULT_MODEL
    source: str = ""


def configured() -> Configured:
    """The key and model in force, from the environment or the settings file."""
    settings = read_settings()
    model = (
        os.environ.get(MODEL_VAR, "").strip()
        or str(settings.get(SETTINGS_MODEL) or "").strip()
        or DEFAULT_MODEL
    )
    whole = str(settings.get(SETTINGS_MODEL_WHOLE) or "").strip() or model

    for name in KEY_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return Configured(key=value, model=model, whole_model=whole,
                              source="environment")

    from_file = str(settings.get(SETTINGS_KEY) or "").strip()
    if from_file:
        return Configured(key=from_file, model=model, whole_model=whole,
                          source="settings file")
    return Configured(model=model, whole_model=whole)


def configured_key() -> str:
    return configured().key


@dataclass
class Gemini:
    """Google's Generative Language REST API, over the standard library.

    No SDK: this runs inside ComfyUI's Python, where adding a dependency to make
    one POST is a cost the user pays at every install and upgrade.
    """

    api_key: str
    model: str = DEFAULT_MODEL
    name: str = "gemini"
    source: str = ""

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        return self._read(self._post(f"models/{self.model}:generateContent", body))

    def models(self) -> list[str]:
        """What this key can actually see. Beats guessing at a model name."""
        listed = self._get("models")
        return sorted(
            entry.get("name", "").removeprefix("models/")
            for entry in listed.get("models") or []
            if "generateContent" in (entry.get("supportedGenerationMethods") or [])
        )

    # -- transport ----------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        return self._send(urllib.request.Request(self._url(path), method="GET"))

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._send(urllib.request.Request(
            self._url(path),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        ))

    def _url(self, path: str) -> str:
        return f"{GEMINI_ROOT}/{path}"

    def _send(self, request: urllib.request.Request) -> dict[str, Any]:
        """One call, retried while the answer is a temporary one.

        The retry is here rather than around `draft` because it is a transport
        concern: a 503 means the request never reached a model, so repeating it
        costs nothing and changes nothing. A repair turn is the opposite — it
        sends different text — and lives up in the loop where it belongs.
        """
        # The key rides in a header rather than the query string: a URL ends up
        # in logs and proxies, and this one would take the secret with it.
        request.add_header("x-goog-api-key", self.api_key)

        for attempt in range(1, RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:400]
                if error.code in RETRY_STATUSES and attempt < RETRIES:
                    self._wait(attempt)
                    continue
                raise LlmError(self._explain(error.code, detail), 502) from error
            except urllib.error.URLError as error:
                if attempt < RETRIES:
                    self._wait(attempt)
                    continue
                raise LlmError(f"Could not reach Gemini: {error.reason}", 504) from error
            except json.JSONDecodeError as error:
                raise LlmError("Gemini's reply was not JSON.", 502) from error
        raise LlmError("Gemini did not answer.", 502)  # unreachable; keeps mypy honest

    def _wait(self, attempt: int) -> None:
        time.sleep(BACKOFF_S * attempt)

    def _explain(self, code: int, detail: str) -> str:
        """Say what the caller can do about it, not just what happened."""
        if code == 429:
            return ("Gemini is rate-limiting this key. Wait a moment, or check "
                    f"the quota on your Google AI Studio project. ({detail})")
        if code in (500, 502, 503, 504):
            return (f"Gemini is busy and stayed busy through {RETRIES} tries. "
                    f"This is on their side and usually passes. ({detail})")
        if code in (401, 403):
            return f"Gemini refused the key. Check `gemini_key`. ({detail})"
        if code == 404:
            return (f"Gemini has no model called {self.model!r}. Ask "
                    f"/llm/status?probe=1 what this key can see. ({detail})")
        return f"Gemini said {code}: {detail}"

    def _read(self, answer: dict[str, Any]) -> dict[str, Any]:
        """Pull the document out of a candidate, or say why there isn't one."""
        candidates = answer.get("candidates") or []
        if not candidates:
            blocked = (answer.get("promptFeedback") or {}).get("blockReason")
            raise LlmError(
                f"Gemini returned nothing ({blocked})."
                if blocked
                else "Gemini returned no candidates.",
                502,
            )
        reason = candidates[0].get("finishReason")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text.strip():
            raise LlmError(f"Gemini returned an empty answer ({reason}).", 502)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            if reason == "MAX_TOKENS":
                raise LlmError(
                    "Gemini ran out of room mid-document. Try a shorter "
                    "description, or a narrower scope.",
                    502,
                ) from error
            raise LlmError("Gemini's answer was not the JSON document.", 502) from error
        if not isinstance(parsed, dict):
            raise LlmError("Gemini answered with something that is not a document.", 502)
        return parsed


# -- a local, open-weights model ----------------------------------------------

# Any server that speaks the OpenAI chat-completions dialect: Ollama, LM Studio,
# llama.cpp's `llama-server`, vLLM. One shape covers all of them, and none of
# them needs a key or leaves the machine — which is the whole point of having
# it next to Gemini rather than instead of nothing.
LOCAL_URL_VAR = "VIDEO_PROMPTER_LLM_URL"
LOCAL_MODEL_VAR = "VIDEO_PROMPTER_LLM_MODEL"
SETTINGS_LOCAL_URL = "llm_url"
SETTINGS_LOCAL_MODEL = "llm_model"
# A local model writing a whole storyboard on a GPU that may also be holding
# H3 is slow in a way a hosted flash model is not.
LOCAL_TIMEOUT_S = 300


@dataclass(frozen=True)
class LocalConfigured:
    url: str = ""
    model: str = ""
    source: str = ""


def local_configured() -> LocalConfigured:
    """The local server in force, from the environment or the settings file."""
    settings = read_settings()
    env_url = os.environ.get(LOCAL_URL_VAR, "").strip()
    url = env_url or str(settings.get(SETTINGS_LOCAL_URL) or "").strip()
    if not url:
        return LocalConfigured()
    model = (
        os.environ.get(LOCAL_MODEL_VAR, "").strip()
        or str(settings.get(SETTINGS_LOCAL_MODEL) or "").strip()
    )
    return LocalConfigured(
        url=url.rstrip("/"), model=model,
        source="environment" if env_url else "settings file",
    )


def _strip_wrapping(text: str) -> str:
    """What is left of an answer once the habits of chat models are removed.

    Reasoning models open with a `<think>` block, and plenty of instruction
    models fence JSON in Markdown even when a schema was asked for. Neither is
    the document, and a server that does not enforce the schema lets both
    through.
    """
    text = text.strip()
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


@dataclass
class OpenAICompatible:
    """A local chat-completions server, over the standard library.

    `model` may be empty: llama.cpp and LM Studio answer with whatever is
    loaded, and Ollama is asked what it has. The first listed model is used,
    which is the one someone just pulled in the common case of exactly one.
    """

    base_url: str
    model: str = ""
    name: str = "local"
    source: str = ""

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        model = self.model or self._first_model()
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "video_prompt", "schema": schema},
            },
            "temperature": 0.7,
        }
        answer = self._send("chat/completions", body)
        choices = answer.get("choices") or []
        if not choices:
            raise LlmError("The local model returned no choices.", 502)
        reason = choices[0].get("finish_reason")
        text = _strip_wrapping(str((choices[0].get("message") or {}).get("content") or ""))
        if not text:
            raise LlmError(f"The local model returned an empty answer ({reason}).", 502)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            if reason == "length":
                raise LlmError(
                    "The local model ran out of room mid-document. Raise its "
                    "context length, or ask for a narrower scope.", 502,
                ) from error
            raise LlmError("The local model's answer was not the JSON document.", 502) from error
        if not isinstance(parsed, dict):
            raise LlmError("The local model answered with something that is not a document.", 502)
        return parsed

    def models(self) -> list[str]:
        """What the server offers, in its own order, minus embedding models.

        Not sorted: Ollama lists the most recently pulled first, which is the
        best guess at what someone meant. An embedding model cannot write a
        document at all, and Ollama installs often carry one for RAG.
        """
        listed = self._send("models", None)
        return [
            str(entry["id"]) for entry in listed.get("data") or []
            if entry.get("id") and "embed" not in str(entry["id"]).lower()
        ]

    def _first_model(self) -> str:
        found = self.models()
        if not found:
            raise LlmError(
                f"The local server at {self.base_url} lists no models. Load or "
                f"pull one, or name it in `{SETTINGS_LOCAL_MODEL}`.", 503,
            )
        self.model = found[0]
        return self.model

    def _send(self, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/{path}",
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=LOCAL_TIMEOUT_S) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:400]
            if error.code == 404:
                raise LlmError(
                    f"The local server has no model {self.model!r} (or no "
                    f"/{path} route). Ask /llm/status?probe=1 what it has. ({detail})",
                    502,
                ) from error
            raise LlmError(f"The local model said {error.code}: {detail}", 502) from error
        except urllib.error.URLError as error:
            raise LlmError(
                f"Could not reach the local model at {self.base_url}: {error.reason}. "
                "Is the server running?", 504,
            ) from error
        except json.JSONDecodeError as error:
            raise LlmError("The local server's reply was not JSON.", 502) from error


def provider_from_settings(scope: "Scope | None" = None) -> "Gemini | OpenAICompatible | None":
    """The configured provider, or None when there is nothing to use.

    A local server wins over a Gemini key when both are set: naming one is a
    deliberate choice to keep prompts on the machine, and quietly sending them
    to Google anyway because a key happened to be lying around would break it.

    A whole-document scope gets the `gemini_model_whole` model when one is set,
    because writing a scene from a sentence rewards a stronger model in a way
    that nudging one field does not — and the nudge is the one being waited on.
    """
    local = local_configured()
    if local.url:
        return OpenAICompatible(base_url=local.url, model=local.model, source=local.source)

    found = configured()
    if not found.key:
        return None
    whole = scope is None or (scope.fields == AUTHORED_FIELDS and scope.beat is None)
    return Gemini(
        api_key=found.key,
        model=found.whole_model if whole else found.model,
        source=found.source,
    )
