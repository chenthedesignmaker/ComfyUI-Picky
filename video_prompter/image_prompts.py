"""Past versions of the image prompt, so a rewrite can be taken back.

The still library already keeps the words that *generated* something: every
`Still` records its `prompt_text`, and the editor offers them back as "Reuse".
That covers a prompt which reached the GPU. It does not cover the ones in
between — write, dislike, rewrite, and the middle version is gone — and it is
about to matter far more than it did, because a model can now replace the whole
box in one press.

That is the difference from the video side. A segment prompt is a document with
named fields, so `llm.merge` can drop everything outside the scope of the
request and a rewrite of the music is physically unable to touch the storyboard.
An image prompt is one string. There is no "only this part" to enforce, so a
rewrite replaces all of it, and the thing that makes that safe to apply straight
away — rather than as a proposal to accept — is being able to undo it. Hence
this: the missing middle, exactly as `sources.Revision` is for a segment.

One trail per project rather than one per destination. The editor's draft boxes
are per destination, but recovery is not: the paragraph you want back is the one
you had, and hunting for it in the right slot's history first is a puzzle the
tool would be setting rather than solving. Each entry records the destination it
was written for so the list can still say where it came from.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .projects import ProjectStore, utc_now_iso, write_json_atomic

IMAGE_PROMPTS_JSON = "image_prompts.json"

# How many past versions a project keeps. Same reasoning as `REVISION_LIMIT` in
# `sources`: deep enough to walk back through an afternoon, shallow enough that
# the file stays one you can open and read. The oldest goes first.
REVISION_LIMIT = 60


def new_revision_id() -> str:
    return f"img_{uuid.uuid4().hex[:8]}"


@dataclass
class ImageRevision:
    """One image prompt this project used to hold.

    `note` says where the *replacement* came from rather than what this text
    said — "Gemini: colder light", "generated" — because the what is right
    there in `text`. Same convention as `sources.Revision`, so the two history
    lists read the same way.
    """

    text: str
    at: str = field(default_factory=utc_now_iso)
    note: str = ""
    # Where the prompt was aimed when it was written: the same string the
    # editor's destination control uses ("first_frame", "member:cast_1a2b3c4d").
    # Free text on purpose — this is a label for a human reading the list, and
    # a destination that has since been deleted must not make the entry invalid.
    target: str = ""
    id: str = field(default_factory=new_revision_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": self.at,
            "note": self.note,
            "target": self.target,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageRevision":
        return cls(
            id=str(data.get("id") or new_revision_id()),
            at=str(data.get("at") or utc_now_iso()),
            note=str(data.get("note") or ""),
            target=str(data.get("target") or ""),
            text=str(data.get("text") or ""),
        )


class ImagePromptStore:
    """Image prompt history per project, in a file beside the project itself."""

    def __init__(self, projects: ProjectStore) -> None:
        self.projects = projects
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def path_for(self, project: str) -> Path:
        return self.projects.path_for(project) / IMAGE_PROMPTS_JSON

    def list(self, project: str) -> list[ImageRevision]:
        """Past versions, newest first — the order the editor lists them in."""
        return list(reversed(self._read(project)))

    def file(
        self,
        project: str,
        text: str,
        *,
        note: str = "",
        target: str = "",
    ) -> ImageRevision | None:
        """Keep `text` as a past version.

        Returns the revision filed, or `None` when there is nothing worth
        keeping: an empty box, or the same words as the newest entry already
        holds. Without that second check a chatty caller would file the same
        paragraph on every generate of a seed sweep and bury the edits worth
        finding — the same reason `Segment.revise` returns `None` unchanged.
        """
        kept = (text or "").strip()
        if not kept:
            return None
        with self._lock_for(project):
            existing = self._read(project)
            if existing and existing[-1].text.strip() == kept:
                return None
            filed = ImageRevision(text=text, note=note, target=target)
            existing.append(filed)
            del existing[:-REVISION_LIMIT]
            self._write(project, existing)
            return filed

    def find(self, project: str, revision_id: str) -> ImageRevision | None:
        return next(
            (rev for rev in self._read(project) if rev.id == revision_id), None
        )

    def clear(self, project: str) -> int:
        """Drop the whole trail. Returns how many versions went."""
        with self._lock_for(project):
            existing = self._read(project)
            if existing:
                self._write(project, [])
            return len(existing)

    def _lock_for(self, project: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(project)
            if lock is None:
                lock = threading.Lock()
                self._locks[project] = lock
            return lock

    def _read(self, project: str) -> list[ImageRevision]:
        """Oldest first, as stored."""
        path = self.path_for(project)
        if not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError:
            return []
        except json.JSONDecodeError:
            # As with takes and stills: move a damaged side-car aside so the
            # next write does not overwrite it, then return empty.
            stamp = utc_now_iso().replace(":", "-")
            aside = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
            try:
                path.replace(aside)
            except OSError:
                pass
            return []
        if not isinstance(data, dict):
            return []
        return [ImageRevision.from_dict(e) for e in data.get("revisions") or []]

    def _write(self, project: str, revisions: list[ImageRevision]) -> None:
        path = self.path_for(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            path, {"revisions": [rev.to_dict() for rev in revisions]}
        )
