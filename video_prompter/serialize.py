"""Structure <-> plain dicts, for anything that has to hit disk or the wire.

Deliberately lossless and tolerant: unknown keys are ignored and missing ones
take their defaults, so a project written by an older build still loads.
"""

from __future__ import annotations

from typing import Any
from copy import deepcopy

from .document import (
    REFERENCE_KINDS,
    Reference,
    ReferenceSet,
    Subject,
    SubjectSet,
    VideoPrompt,
)
from .storyboard import AudioCue, Beat, Line, Storyboard


def prompt_to_dict(prompt: VideoPrompt) -> dict[str, Any]:
    return {
        "duration": prompt.storyboard.duration,
        "beats": [
            {
                "start": beat.start,
                "end": beat.end,
                "angle": beat.angle,
                "action": beat.action,
                "camera": beat.camera,
                **({"camera_path": deepcopy(beat.camera_path)} if beat.camera_path is not None else {}),
                "transition": beat.transition,
                "cut": beat.cut,
                "lines": [
                    {
                        "speakers": list(line.speakers),
                        "text": line.text,
                        "language": line.language,
                        "verb": line.verb,
                        "delivery": line.delivery,
                        "describe": line.describe,
                        "voiceover": line.voiceover,
                        "possessive": line.possessive,
                        "continuity": line.continuity,
                        "cutoff": line.cutoff,
                        "placement": line.placement,
                    }
                    for line in beat.lines
                ],
            }
            for beat in prompt.storyboard.beats
        ],
        "cues": [
            {"start": cue.start, "end": cue.end, "description": cue.description}
            for cue in prompt.storyboard.cues
        ],
        "style": prompt.style,
        "overview": prompt.overview,
        "camera": prompt.camera,
        "audio": prompt.audio,
        "music": prompt.music,
        "screen_text": list(prompt.screen_text),
        "negatives": list(prompt.negatives),
        "references": [
            {
                "kind": item.kind,
                "path": item.path,
                "role": item.role,
                "retention": item.retention,
                "member": item.member,
                "slot": item.slot,
            }
            for item in prompt.references.items
        ],
        "subjects": [
            {
                "role": subject.role,
                "retention": subject.retention,
                "sources": list(subject.sources),
                "member": subject.member,
                "noun": subject.noun,
                "label": subject.label,
            }
            for subject in prompt.subjects.items
        ],
        "ref_task": list(prompt.ref_task),
        "ref_summary": prompt.ref_summary,
        "first_frame": prompt.first_frame,
        "last_frame": prompt.last_frame,
        "width": prompt.width,
        "height": prompt.height,
        "cast": list(prompt.cast),
        "undescribed": list(prompt.undescribed),
        "guard_identities": prompt.guard_identities,
        "storyboard_header": prompt.storyboard_header,
        "label_beats": prompt.label_beats,
    }


def prompt_from_dict(data: dict[str, Any]) -> VideoPrompt:
    blank = VideoPrompt.blank()

    # `shots`/`label_shots` are what this tool wrote before beats and edit
    # shots became different things; read them so old prompts still open.
    beats = data.get("beats")
    if beats is None:
        beats = data.get("shots")
    label_beats = data.get("label_beats", data.get("label_shots", blank.label_beats))

    storyboard = Storyboard(
        duration=data.get("duration", blank.storyboard.duration),
        beats=[_beat(entry) for entry in beats or []],
        cues=[_cue(entry) for entry in data.get("cues") or []],
    )

    return VideoPrompt(
        storyboard=storyboard,
        style=data.get("style", ""),
        overview=data.get("overview", ""),
        camera=data.get("camera", ""),
        audio=data.get("audio", ""),
        music=data.get("music", ""),
        screen_text=list(data.get("screen_text") or []),
        cast=[str(member) for member in data.get("cast") or []],
        undescribed=[str(member) for member in data.get("undescribed") or []],
        negatives=list(data.get("negatives") or []),
        references=ReferenceSet(
            [ref for ref in map(_reference, data.get("references") or []) if ref]
        ),
        subjects=SubjectSet(
            [subject for subject in map(_subject, data.get("subjects") or []) if subject]
        ),
        ref_task=[str(task) for task in data.get("ref_task") or []],
        ref_summary=data.get("ref_summary", ""),
        first_frame=str(data.get("first_frame") or ""),
        last_frame=str(data.get("last_frame") or ""),
        width=int(data.get("width", blank.width)),
        height=int(data.get("height", blank.height)),
        guard_identities=bool(
            data.get("guard_identities", blank.guard_identities)
        ),
        storyboard_header=data.get("storyboard_header", blank.storyboard_header),
        label_beats=bool(label_beats),
    )


def _beat(entry: dict[str, Any]) -> Beat:
    return Beat(
        start=float(entry.get("start", 0)),
        end=float(entry.get("end", 0)),
        angle=entry.get("angle", ""),
        action=entry.get("action", ""),
        camera=entry.get("camera", ""),
        camera_path=deepcopy(entry.get("camera_path")),
        transition=entry.get("transition", ""),
        # Absent means a cut: every prompt written before no-cut beats
        # existed had one shot per beat.
        cut=bool(entry.get("cut", True)),
        lines=[_line(item) for item in entry.get("lines") or []],
    )


def _line(entry: dict[str, Any]) -> Line:
    speakers = entry.get("speakers")
    if speakers is None and entry.get("speaker"):  # tolerate a single-key shorthand
        speakers = [entry["speaker"]]
    return Line(
        speakers=[str(key) for key in speakers or []],
        text=entry.get("text", ""),
        language=entry.get("language", "") or "English",
        verb=entry.get("verb", "") or "says",
        delivery=entry.get("delivery", ""),
        describe=entry.get("describe", ""),
        voiceover=bool(entry.get("voiceover", False)),
        possessive=entry.get("possessive", "") or "",
        continuity=_continuity(entry.get("continuity")),
        cutoff=bool(entry.get("cutoff", False)),
        placement=_placement(entry.get("placement")),
    )


def _continuity(value: object) -> str:
    """Only the two seam markers the format defines; anything else is dropped."""
    text = str(value or "").strip().lower()
    return text if text in ("into", "from") else ""


def _placement(value: object) -> str:
    """Only the one slot the format has room for; anything else is the default."""
    return "open" if str(value or "").strip().lower() == "open" else ""


def _cue(entry: dict[str, Any]) -> AudioCue:
    end = entry.get("end")
    return AudioCue(
        start=float(entry.get("start", 0)),
        end=None if end is None else float(end),
        description=entry.get("description", ""),
    )


def _reference(entry: dict[str, Any]) -> Reference | None:
    """Drop references whose kind we do not recognise rather than fail the load."""
    kind = entry.get("kind")
    if kind not in REFERENCE_KINDS:
        return None
    return Reference(
        kind=kind,
        path=entry.get("path", ""),
        role=entry.get("role", ""),
        retention=entry.get("retention", ""),
        member=str(entry.get("member") or ""),
        slot=_slot(entry.get("slot")),
    )


def _subject(entry: Any) -> Subject | None:
    """A `<Subject N>` row. Anything that is not a mapping is skipped.

    Unlike a reference there is no kind to validate, so a row survives as long
    as it says something: a role, or the sources it is cited from.
    """
    if not isinstance(entry, dict):
        return None
    subject = Subject(
        role=str(entry.get("role") or ""),
        retention=str(entry.get("retention") or ""),
        sources=[str(path) for path in entry.get("sources") or [] if str(path)],
        member=str(entry.get("member") or ""),
        noun=str(entry.get("noun") or ""),
        label=str(entry.get("label") or ""),
    )
    if not (subject.role.strip() or subject.sources or subject.member):
        return None
    return subject


def _slot(value: Any) -> int:
    """Which of a member's identity files a linked row points at, 0 otherwise."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
