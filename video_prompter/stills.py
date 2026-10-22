"""Generated images, kept per project so a keyframe can be found again.

A still is not a take. A take is a delivered clip — it is cut into the timeline,
rated, extended, exported. A still is an *input*: the image a segment starts
from, or the file that fixes what a cast member looks like. Mixing the two would
put a PNG in the take grid and let it be picked into an edit, so they live in
separate files with separate rules, exactly as `takes.json` lives beside
`project.json` for its own reasons.

What is worth keeping is the recipe, not the pixels. ComfyUI already owns the
file; what the editor cannot reconstruct later is which prompt, seed and style
made it — which is precisely what you need when a composition was nearly right
and you want the next one to be closer. So a still records those and points at
the file.

Every still is adopted into ComfyUI's input directory as it is recorded, because
an image that cannot be named by `LoadImage` is not usable as a keyframe, and an
image the author has to adopt by hand is one they will forget to adopt.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .gallery import OUTPUT_KEYS, VIDEO_EXTENSIONS, queue_metadata
from .projects import ProjectStore, utc_now_iso, write_json_atomic

STILLS_JSON = "stills.json"

# What `Api.prepare_t2i` stamps into queue metadata, and what the editor sends
# as a job's mode. Either marks a run as image generation rather than a render.
T2I_MODES = ("t2i", "keyframe (t2i)")


@dataclass
class Still:
    """One generated image, and the recipe that made it."""

    prompt_id: str
    filename: str
    subfolder: str = ""
    # Where the adopted copy lives, relative to ComfyUI's input dir — the value
    # that goes on `first_frame`, a reference `path`, or a cast member's
    # identity file. Empty only when adoption failed (no input dir to write to).
    path: str = ""
    # The source this was generated for, when it was generated from a segment.
    # Free-standing images (generated for a cast member, or from the library)
    # leave it empty; the library shows everything either way.
    source: str = ""
    # The cast member this was generated for, when it was.
    member: str = ""
    prompt_text: str = ""
    style: str = ""
    seed: int | None = None
    steps: int | None = None
    width: int = 0
    height: int = 0
    created: str = field(default_factory=utc_now_iso)
    favorite: bool = False
    discarded: bool = False

    @property
    def id(self) -> str:
        """Same construction as a take's id: the pair recording dedupes on."""
        return f"{self.prompt_id}:{self.filename}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "filename": self.filename,
            "subfolder": self.subfolder,
            "path": self.path,
            "source": self.source,
            "member": self.member,
            "prompt_text": self.prompt_text,
            "style": self.style,
            "seed": self.seed,
            "steps": self.steps,
            "width": self.width,
            "height": self.height,
            "created": self.created,
            "favorite": self.favorite,
            "discarded": self.discarded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Still":
        seed = data.get("seed")
        steps = data.get("steps")
        return cls(
            prompt_id=str(data.get("prompt_id", "")),
            filename=str(data.get("filename", "")),
            subfolder=str(data.get("subfolder") or ""),
            path=str(data.get("path") or ""),
            source=str(data.get("source") or ""),
            member=str(data.get("member") or ""),
            prompt_text=str(data.get("prompt_text") or ""),
            style=str(data.get("style") or ""),
            seed=int(seed) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else None,
            steps=int(steps) if isinstance(steps, (int, float)) and not isinstance(steps, bool) else None,
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            created=str(data.get("created") or utc_now_iso()),
            favorite=bool(data.get("favorite")),
            discarded=bool(data.get("discarded")),
        )


def is_t2i_entry(entry: dict[str, Any]) -> bool:
    """True when a history entry is one of ours *and* was image generation.

    The queue is shared with every other tool on this ComfyUI, and a video
    render saving a preview frame must not be read as a keyframe. The mode
    stamped at submission is the only reliable marker, so nothing without it
    counts.
    """
    mode = queue_metadata(entry).get("mode")
    return str(mode or "").strip().lower() in T2I_MODES


def stills_from_history(prompt_id: str, entry: dict[str, Any]) -> list[Still]:
    """Read a finished image generation out of a ComfyUI history entry.

    Yields nothing for a video render, for someone else's run, or for an entry
    shaped in a way we do not recognise — this runs over whatever the server
    happens to be holding.
    """
    if not is_t2i_entry(entry):
        return []
    meta = queue_metadata(entry)

    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return []

    found: list[Still] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in OUTPUT_KEYS:
            for item in node_output.get(key) or []:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                if item.get("type") == "input":
                    continue
                filename = str(item["filename"])
                # A t2i graph saves one image, but a workflow someone extended
                # could also write a video; that is a take's business, not ours.
                if filename.lower().endswith(VIDEO_EXTENSIONS):
                    continue
                found.append(
                    Still(
                        prompt_id=prompt_id,
                        filename=filename,
                        subfolder=str(item.get("subfolder") or ""),
                        source=str(meta.get("shot") or meta.get("source") or ""),
                        member=str(meta.get("member") or ""),
                        prompt_text=str(meta.get("prompt") or ""),
                        style=str(meta.get("style") or ""),
                        seed=meta.get("seed"),
                        steps=meta.get("steps"),
                        width=int(meta.get("width") or 0),
                        height=int(meta.get("height") or 0),
                    )
                )
    return found


class StillStore:
    """Stills per project, in a file beside the project itself."""

    def __init__(self, projects: ProjectStore) -> None:
        self.projects = projects
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def path_for(self, project: str) -> Path:
        return self.projects.path_for(project) / STILLS_JSON

    def list(self, project: str, source: str | None = None) -> list[Still]:
        """Stills for a project, newest first.

        `source` narrows to the ones generated from that source. The editor
        shows those first and the rest of the library under them, so the filter
        is a convenience for callers that only want the near ones.
        """
        stills = self._read(project)
        if source:
            stills = [still for still in stills if still.source == source]
        return sorted(stills, key=lambda still: still.created, reverse=True)

    def find(self, project: str, still_id: str) -> Still | None:
        return next((still for still in self._read(project) if still.id == still_id), None)

    def record(self, project: str, stills: Iterable[Still]) -> list[Still]:
        """Add stills, ignoring any already recorded. Idempotent like `Gallery.record`."""
        incoming = list(stills)
        if not incoming:
            return []
        with self._lock_for(project):
            existing = self._read(project)
            known = {(still.prompt_id, still.filename) for still in existing}
            added = []
            for still in incoming:
                key = (still.prompt_id, still.filename)
                if key in known:
                    continue
                existing.append(still)
                known.add(key)
                added.append(still)
            if added:
                self._write(project, existing)
            return added

    def annotate(
        self,
        project: str,
        still_id: str,
        *,
        favorite: bool | None = None,
        discarded: bool | None = None,
        source: str | None = None,
        member: str | None = None,
    ) -> Still | None:
        """Set a still's flags, or re-file it against a source or member."""
        with self._lock_for(project):
            stills = self._read(project)
            found = next((still for still in stills if still.id == still_id), None)
            if found is None:
                return None
            if favorite is not None:
                found.favorite = bool(favorite)
            if discarded is not None:
                found.discarded = bool(discarded)
            if source is not None:
                found.source = str(source)
            if member is not None:
                found.member = str(member)
            self._write(project, stills)
            return found

    def forget(self, project: str, still_id: str) -> int:
        """Drop a still from the record. The file on disk is left alone.

        Deliberately not a delete: the adopted copy may already be a segment's
        keyframe, and removing a library entry must never break a prompt that
        points at the file.
        """
        with self._lock_for(project):
            existing = self._read(project)
            kept = [still for still in existing if still.id != still_id]
            if len(kept) != len(existing):
                self._write(project, kept)
            return len(existing) - len(kept)

    def _lock_for(self, project: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(project)
            if lock is None:
                lock = threading.Lock()
                self._locks[project] = lock
            return lock

    def _read(self, project: str) -> list[Still]:
        path = self.path_for(project)
        if not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError:
            return []
        except json.JSONDecodeError:
            # As with takes: move a damaged side-car aside so the next write
            # does not overwrite it, then return empty so the project opens.
            stamp = utc_now_iso().replace(":", "-")
            aside = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
            try:
                path.replace(aside)
            except OSError:
                pass
            return []
        if not isinstance(data, dict):
            return []
        return [Still.from_dict(entry) for entry in data.get("stills") or []]

    def _write(self, project: str, stills: list[Still]) -> None:
        path = self.path_for(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, {"stills": [still.to_dict() for still in stills]})
