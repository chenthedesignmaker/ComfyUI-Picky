"""What was generated: a lineage, its segments, and the path chosen through it.

A **Source** is one bin item — everything that came out of generating a thing and
continuing it. It owns:

- **Segments**, one per generation link. A segment is *intent*: the prompt and
  the quality to render at, at depth 0, 1, 2 … Nothing about the edit.
- A **path**: the take id chosen at each depth. One ordered list, one place, so
  "which take follows which" cannot disagree with itself the way it did when the
  answer lived in a per-shot `pick`, a `branch_picks` map and `take.parent_take`
  all at once.

Everything else about a lineage — the branches, whether a path holds up, how long
it runs — is derived from `take.parent_take` by :mod:`lineage`. Nothing here
guesses.

A plain un-extended shot is a Source with one segment. There is deliberately no
"is this a group or a single?" distinction: that special case is what the old
model kept tripping over.

How the edit *uses* a source is not here. That is a :class:`~.sequence.Clip`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable

from . import h3_spec, video_loras
from .document import VideoPrompt
from .serialize import prompt_from_dict, prompt_to_dict


def new_source_id() -> str:
    return f"src_{uuid.uuid4().hex[:8]}"


def new_segment_id() -> str:
    return f"seg_{uuid.uuid4().hex[:8]}"


def new_revision_id() -> str:
    return f"rev_{uuid.uuid4().hex[:8]}"


def _stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# How many superseded prompts a segment keeps. Deep enough to walk back
# through an afternoon of tinkering, shallow enough that `project.json` stays
# a file you can open and read. The oldest goes first.
REVISION_LIMIT = 50


@dataclass
class Revision:
    """A prompt this segment used to hold, kept so a rewrite can be undone.

    Takes already freeze the text they rendered, but only for prompts that
    reached the GPU: edit, dislike, edit again without rendering and the middle
    version is gone. Project backups cover it from the other side and are too
    blunt to use — restoring one to recover a paragraph rolls back the whole
    bin. This is the missing middle: one segment's own past, restorable alone.

    `note` says where the prompt came from rather than what it said — "Gemini:
    made it faster", "restored rev_1a2b" — because the what is right there.
    """

    prompt: VideoPrompt
    at: str = field(default_factory=_stamp)
    note: str = ""
    id: str = field(default_factory=new_revision_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": self.at,
            "note": self.note,
            "prompt": prompt_to_dict(self.prompt),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Revision":
        return cls(
            id=str(data.get("id") or new_revision_id()),
            at=str(data.get("at") or _stamp()),
            note=data.get("note", ""),
            prompt=prompt_from_dict(data.get("prompt") or {}),
        )


@dataclass
class Segment:
    """One generation link: what to render at this depth, and how hard.

    Depth is the segment's position in its source's list, not a stored number —
    a stored one could disagree with the list it sits in.
    """

    id: str = field(default_factory=new_segment_id)
    prompt: VideoPrompt = field(default_factory=VideoPrompt.blank)
    # A quality preset name (see `quality.py`). Reproducing one specific take is
    # a take-level concern, not a segment-level one.
    quality: str = "preview"
    notes: str = ""
    # Superseded prompts, oldest first; `revisions[-1]` is what `prompt` replaced.
    # Only `revise` and `restore` write this, so no caller can change the prompt
    # and forget to file the old one.
    revisions: list[Revision] = field(default_factory=list)

    def revise(self, prompt: VideoPrompt, note: str = "") -> Revision | None:
        """Adopt `prompt`, filing the outgoing one in the history.

        Returns the revision filed, or `None` when the prompt is unchanged — a
        save that only touched the quality must not push a duplicate, or a
        chatty editor would bury the edits worth finding.
        """
        if prompt_to_dict(prompt) == prompt_to_dict(self.prompt):
            return None
        filed = Revision(prompt=deepcopy(self.prompt), note=note)
        self.revisions.append(filed)
        del self.revisions[:-REVISION_LIMIT]
        self.prompt = prompt
        return filed

    def restore(self, revision_id: str) -> Revision | None:
        """Put a past prompt back, filing the current one on the way.

        The trip is round on purpose: restoring is itself an edit, so undoing an
        undo is just another restore rather than a special case.
        """
        found = next((r for r in self.revisions if r.id == revision_id), None)
        if found is None:
            return None
        self.revise(deepcopy(found.prompt), note=f"restored {found.id}")
        return found

    def to_dict(self, *, history: bool = True) -> dict[str, Any]:
        """The segment as stored. `history=False` drops the revisions.

        The disk keeps them; the wire usually should not. Every save answers
        with the whole project, so shipping each segment's past on every
        keystroke would grow the reply without anyone having asked to read it.
        `revision_count` is left behind so the editor can still show that there
        is something to walk back to, and fetch it when asked.
        """
        shape = {
            "id": self.id,
            "prompt": prompt_to_dict(self.prompt),
            "quality": self.quality,
            "notes": self.notes,
            "revision_count": len(self.revisions),
        }
        if history:
            shape["revisions"] = [revision.to_dict() for revision in self.revisions]
        return shape

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            id=str(data.get("id") or new_segment_id()),
            prompt=prompt_from_dict(data.get("prompt") or {}),
            quality=data.get("quality") or "preview",
            notes=data.get("notes", ""),
            revisions=[Revision.from_dict(e) for e in data.get("revisions") or []],
        )


@dataclass
class Source:
    """One lineage in the bin: its segments, and the path chosen through it."""

    id: str = field(default_factory=new_source_id)
    name: str = "Untitled"
    segments: list[Segment] = field(default_factory=list)
    # Take id per depth. The single answer to "which take follows which".
    path: list[str] = field(default_factory=list)
    # Take id -> preferred child take id. Advisory only: it rebuilds the tail of
    # a path after a branch switch. A stale entry costs a mediocre default, never
    # a broken path, which is why it may never be read as the truth.
    remembered: dict[str, str] = field(default_factory=dict)
    # Carried through the v1 migration so nothing is lost. Unused: it held a
    # *guess* about which shot continued which, and lineage now comes from the
    # renderer's own record.
    legacy_continuation_of: str | None = None
    # Applies to the whole source/shot, including its continuation segments.
    video_loras: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # A source with no segments cannot be rendered or drawn. One blank
        # segment is the honest minimum, and matches "a plain shot is a source".
        if not self.segments:
            self.segments = [Segment()]

    @property
    def depth(self) -> int:
        """How many generations this source plans."""
        return len(self.segments)

    def segment_at(self, depth: int) -> Segment | None:
        return self.segments[depth] if 0 <= depth < len(self.segments) else None

    def add_segment(self, segment: Segment | None = None) -> Segment:
        """Plan one more generation on the end of the chain."""
        made = segment or Segment()
        self.segments.append(made)
        return made

    def remember(self, parent_take_id: str, child_take_id: str) -> None:
        if parent_take_id and child_take_id:
            self.remembered[parent_take_id] = child_take_id

    def set_path(self, path: list[str]) -> list[str]:
        """Replace the chosen path, recording each step as a preference.

        Remembering here rather than at the click site means every route into a
        path — the UI, an agent, a migration — leaves the same trail.
        """
        self.path = list(path)
        for position in range(1, len(self.path)):
            self.remember(self.path[position - 1], self.path[position])
        return self.path

    def to_dict(self, *, history: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "segments": [s.to_dict(history=history) for s in self.segments],
            "path": list(self.path),
            "remembered": dict(self.remembered),
            "legacy_continuation_of": self.legacy_continuation_of,
            "video_loras": deepcopy(self.video_loras),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        return cls(
            id=str(data.get("id") or new_source_id()),
            name=data.get("name", "Untitled"),
            segments=[Segment.from_dict(entry) for entry in data.get("segments") or []],
            path=[str(x) for x in data.get("path") or []],
            remembered=dict(data.get("remembered") or {}),
            legacy_continuation_of=data.get("legacy_continuation_of") or None,
            video_loras=video_loras.normalize_shot(data.get("video_loras")),
        )

    def copy(self, *, name: str | None = None) -> "Source":
        """A duplicate with fresh ids and no path — same intent, no results.

        The revisions stay behind too. A fresh bin item that opens showing an
        afternoon of edits it never had would be a lie about its own past.
        """
        return replace(
            self,
            id=new_source_id(),
            name=name or self.name,
            segments=[
                Segment(prompt=deepcopy(segment.prompt), quality=segment.quality,
                        notes=segment.notes)
                for segment in self.segments
            ],
            path=[],
            remembered={},
            video_loras=deepcopy(self.video_loras),
        )


def shot_label(indices: Iterable[int]) -> str:
    """"shot 2", "shots 2-3", "shots 1, 3" — 1-based, the way a storyboard counts.

    The editor and the format both call the first beat Shot 1, so a bin item
    named after the beats it holds has to count the same way, whatever
    0-based indices got it there.
    """
    kept = sorted({int(i) + 1 for i in indices})
    if not kept:
        return "no shots"
    if len(kept) == 1:
        return f"shot {kept[0]}"
    if kept == list(range(kept[0], kept[-1] + 1)):
        return f"shots {kept[0]}-{kept[-1]}"
    return "shots " + ", ".join(str(number) for number in kept)


def split_off(
    source: Source, segment: Segment, indices: Iterable[int], *, name: str = ""
) -> Source:
    """Some of one segment's shots as a bin item of their own. Prompt only.

    A scene written as one generation often wants to be two — the cut inside it
    lands badly, or two of its shots each deserve a whole generation's worth of
    frames. This is the first half of doing that: the picked shots' words, and
    the scene around them, in a source that has rendered nothing.

    A **copy**, not a move. The shots stay where they were, because whether the
    original should lose them is a decision worth making after seeing what the
    split generates, not before.

    One segment, never the chain. A continuation's prompt is written to follow
    the generation before it, so lifting a shot out of depth 2 and asking for
    it on its own is a different intent that needs a different first sentence —
    the honest thing to hand back is a single generation the author can extend
    themselves.
    """
    kept = sorted({int(i) for i in indices})
    return Source(
        name=name or f"{source.name} ({shot_label(kept)})",
        video_loras=deepcopy(source.video_loras),
        segments=[
            Segment(
                prompt=segment.prompt.for_beats(kept),
                quality=segment.quality,
                notes=segment.notes,
            )
        ],
    )


def blank_source(name: str, duration: int = h3_spec.TRAINED_MIN_DURATION_S) -> Source:
    """A placeholder source: named, one segment, timed, nothing written yet.

    A new source starts at the trained floor, not the allowed one: shorter
    budgets are a deliberate choice, never the default nobody picked.
    """
    return Source(name=name, segments=[Segment(prompt=VideoPrompt.blank(duration))])
