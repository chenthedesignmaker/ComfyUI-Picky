"""Which authored field each stretch of the composed prompt came from.

The editor tints the composed text so a writer can see at a glance that this
sentence is the style contract and that one is Shot 2's action. Teaching
`compose` to emit offsets would mean threading positions through every f-string
in it, so this takes a different route: compose the prompt twice, once normally
and once from a shadow copy whose every field value carries an invisible marker
at its front. Deleting the markers from the second pass has to give the first
pass back exactly; when it does, the marker positions are where those fields
landed. When it does not — because `compose` grew a case that reads a field's
leading words — we hand back nothing and the editor shows plain text. The
colouring is a convenience, never a reason to hold up the prompt.

The extent of a span is found by walking forward from its marker for as long as
the composed text still matches the field's own text. That works because
`compose` only ever trims a field from the right (`rstrip(".")`) or re-cases its
first letter; it never rewrites the middle.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from .compose import compose
from .document import VideoPrompt

# Private-use characters: not whitespace, not sentence-ending punctuation, and
# unchanged by `.upper()`, so inserting one cannot change what compose does.
_OPEN = chr(0xE000)
_CLOSE = chr(0xE001)
_MARKER = re.compile(f"{_OPEN}(\\d+){_CLOSE}")

# `_camera_sentence` is the one place compose reads a field's leading words: a
# motion already starting "the camera" is used verbatim, anything else gets the
# subject prepended. A marker in front of those words would send it down the
# other branch, so it goes after them instead and the span reclaims them later.
_CAMERA_LEADS = ("the camera", "camera")


@dataclass
class _Mark:
    field: str  # addresses the editor field, e.g. `beat.1.action`
    kind: str  # colour group
    label: str  # what to call it in the tooltip
    text: str  # the field's own (stripped) text
    lead: int  # how many of those characters sit before the marker


def spans(prompt: VideoPrompt) -> list[dict[str, Any]]:
    """Where each authored field ended up in `compose(prompt)`.

    Returns `{field, kind, label, start, end}` runs, in document order and
    never overlapping. Empty when the two passes disagree.
    """
    marks: list[_Mark] = []
    from .camera_path import materialize_paths
    shadow = copy.deepcopy(materialize_paths(prompt))
    _mark_all(shadow, marks)

    plain = compose(prompt)
    clean, hits = _unmark(compose(shadow))
    if clean.lower() != plain.lower():
        return []

    found = []
    for index, position in hits:
        span = _span(plain, position, marks[index])
        if span:
            found.append(span)

    found.sort(key=lambda item: (item["start"], item["end"]))
    return _without_overlaps(found)


def _span(plain: str, position: int, mark: _Mark) -> dict[str, Any] | None:
    start = position - mark.lead
    if start < 0 or plain[start:position].lower() != mark.text[: mark.lead].lower():
        return None
    end = position + _matching_prefix(plain[position:], mark.text[mark.lead :])
    if end <= start:
        return None
    return {
        "field": mark.field,
        "kind": mark.kind,
        "label": mark.label,
        "start": start,
        "end": end,
    }


def _matching_prefix(composed: str, field_text: str) -> int:
    """How much of `field_text` survives verbatim at the head of `composed`."""
    length = 0
    for left, right in zip(composed, field_text):
        if left.lower() != right.lower():
            break
        length += 1
    return length


def _unmark(marked: str) -> tuple[str, list[tuple[int, int]]]:
    """Strip the markers, reporting where each one sat in the clean text."""
    pieces: list[str] = []
    hits: list[tuple[int, int]] = []
    cursor = 0
    length = 0
    for match in _MARKER.finditer(marked):
        pieces.append(marked[cursor : match.start()])
        length += match.start() - cursor
        hits.append((int(match.group(1)), length))
        cursor = match.end()
    pieces.append(marked[cursor:])
    return "".join(pieces), hits


def _without_overlaps(found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first of any two spans that collide, so the editor can walk the
    list once and slice the plain text between them."""
    kept: list[dict[str, Any]] = []
    for span in found:
        if kept and span["start"] < kept[-1]["end"]:
            continue
        kept.append(span)
    return kept


# -- laying the markers ---------------------------------------------------


def _mark_all(shadow: VideoPrompt, marks: list[_Mark]) -> None:
    def take(field: str, kind: str, label: str, value: str, lead: int = 0) -> str:
        text = value.strip()
        if not text:
            return value  # an empty field composes to nothing, or to a default
        marks.append(_Mark(field, kind, label, text, lead))
        return f"{text[:lead]}{_OPEN}{len(marks) - 1}{_CLOSE}{text[lead:]}"

    shadow.style = take("style", "style", "Style contract", shadow.style)
    shadow.overview = take("overview", "overview", "Scene overview", shadow.overview)
    shadow.camera = take(
        "camera",
        "camera",
        "Camera (shot 1 fallback)",
        shadow.camera,
        _camera_lead(shadow.camera),
    )
    shadow.audio = take("audio", "sound", "Soundscape", shadow.audio)
    shadow.music = take("music", "music", "Non-diegetic music", shadow.music)
    shadow.screen_text = [
        take(f"screen_text.{index}", "screen", f"On-screen text {index + 1}", text)
        for index, text in enumerate(shadow.screen_text)
    ]
    shadow.ref_summary = take(
        "ref_summary", "reference", "Reference summary", shadow.ref_summary
    )
    shadow.ref_task = [
        take(f"ref_task.{index}", "reference", f"Task type {index + 1}", text)
        for index, text in enumerate(shadow.ref_task)
    ]

    for index, item in enumerate(shadow.references.items):
        tag = shadow.references.tag(item)
        item.role = take(f"ref.{index}.role", "reference", f"{tag} role", item.role)
        item.retention = take(
            f"ref.{index}.retention", "reference", f"{tag} retention", item.retention
        )

    # Beats are authored in list order but composed in time order, so the id
    # points at the row the user edits and the label names the shot they read.
    order = {
        id(beat): number
        for number, beat in enumerate(shadow.storyboard.ordered_beats(), start=1)
    }
    for index, beat in enumerate(shadow.storyboard.beats):
        shot = f"Shot {order.get(id(beat), index + 1)}"
        beat.angle = take(f"beat.{index}.angle", "angle", f"{shot} · Angle", beat.angle)
        beat.action = take(f"beat.{index}.action", "action", f"{shot} · Action", beat.action)
        beat.camera = take(
            f"beat.{index}.camera",
            "camera",
            f"{shot} · Camera",
            beat.camera,
            _camera_lead(beat.camera),
        )
        beat.transition = take(
            f"beat.{index}.transition", "transition", f"{shot} · Transition", beat.transition
        )
        for position, line in enumerate(beat.lines):
            where = f"{shot} · Line {position + 1}"
            line.describe = take(
                f"line.{index}.{position}.describe",
                "dialogue",
                f"{where} · Describe",
                line.describe,
            )
            line.verb = take(
                f"line.{index}.{position}.verb", "dialogue", f"{where} · Verb", line.verb
            )
            line.delivery = take(
                f"line.{index}.{position}.delivery",
                "dialogue",
                f"{where} · Delivery",
                line.delivery,
            )
            line.language = take(
                f"line.{index}.{position}.language",
                "dialogue",
                f"{where} · Language",
                line.language,
            )
            line.text = take(
                f"line.{index}.{position}.text", "dialogue", f"{where} · Words", line.text
            )

    for index, cue in enumerate(shadow.storyboard.cues):
        cue.description = take(
            f"cue.{index}.description", "sound", f"Cue {index + 1} · Sound", cue.description
        )


def _camera_lead(motion: str) -> int:
    lowered = motion.strip().lower()
    for prefix in _CAMERA_LEADS:
        if lowered.startswith(prefix):
            return len(prefix)
    return 0
