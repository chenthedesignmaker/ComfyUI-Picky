"""Bringing a v1 project up to sources and clips.

A v1 project is a list of shots, each one doing three unrelated jobs at once:
what to generate, which take won, and how the edit uses it. v2 splits those into
a :class:`~.sources.Source`, its `path`, and a :class:`~.sequence.Clip`.

The migration is deliberately dull. **Every shot becomes exactly one source with
exactly one segment, plus exactly one clip.** Chains that were spread across
sibling shots stay as separate sources, each with its own clip trimmed as it
already was — so the timeline after the migration is the timeline before it,
frame for frame.

It would be possible to go further and merge a run of shots into one multi-segment
source, because `take.parent_take` says which takes continue which. It is
deliberately not done here. Merging changes the edit, and the old heuristics that
would have to guess *which* shots to merge are the ones this whole rewrite exists
to delete. Merging is an explicit action the author takes, on a source they are
looking at, with the lineage on screen.

Nothing is thrown away: `continuation_of` was a guess, but it is carried across as
`legacy_continuation_of` rather than dropped.
"""

from __future__ import annotations

from typing import Any

from .sources import new_source_id

SCHEMA_V1 = 1
SCHEMA_V2 = 2


def is_v2(data: dict[str, Any]) -> bool:
    """True when a project dict is already sources-and-clips.

    Tested on shape rather than on the version number alone: a file hand-edited
    from a newer build should be read as what it contains, not as what its
    header claims.
    """
    if int(data.get("version") or SCHEMA_V1) >= SCHEMA_V2:
        return True
    return isinstance(data.get("sources"), list) and isinstance(data.get("timeline"), dict)


def to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Return a v2 project dict. Already-v2 input is returned unchanged."""
    if is_v2(data):
        return data

    migrated = {key: value for key, value in data.items() if key not in ("sequence", "clips")}
    sources: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []

    for shot in _shots_of(data):
        source, clip = split_shot(shot)
        sources.append(source)
        clips.append(clip)

    migrated["version"] = SCHEMA_V2
    migrated["sources"] = sources
    migrated["timeline"] = {"clips": clips}
    return migrated


def _shots_of(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The shots of a v1 project, including the pre-sequence `clips` bag."""
    sequence = data.get("sequence")
    if isinstance(sequence, dict):
        entries = sequence.get("shots")
    else:
        entries = data.get("clips")
    return [entry for entry in (entries or []) if isinstance(entry, dict)]


def split_shot(shot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One shot, cut along the joint between generating and editing."""
    # The shot's id becomes the source's id rather than a fresh `src_` one.
    # Takes record the shot they were rendered for, so changing it here would
    # orphan every take in the project to save a naming convention.
    #
    # Projects old enough to predate shot ids have none to keep; those get a
    # minted one. Not a positional id — that reads as stable and is not.
    source_id = str(shot.get("id") or new_source_id())
    pick = shot.get("pick") or ""

    source = {
        "id": source_id,
        "name": shot.get("name", "Untitled"),
        "segments": [
            {
                "id": f"seg_{source_id}_0",
                "prompt": shot.get("prompt") or {},
                "quality": shot.get("quality") or "preview",
                "notes": shot.get("notes", ""),
            }
        ],
        # A v1 shot stood for one take, which is a one-step path.
        "path": [str(pick)] if pick else [],
        "remembered": dict(shot.get("branch_picks") or {}),
        "legacy_continuation_of": shot.get("continuation_of") or None,
    }

    clip = {
        "id": f"clip_{source_id}",
        "source_id": source_id,
        # Following the source is the default; nothing in v1 pinned a branch.
        "path_override": None,
        "source_in": float(shot.get("source_in") or 0.0),
        "source_out": shot.get("source_out"),
        "speed": shot.get("speed", 1.0),
        "enabled": bool(shot.get("enabled", True)),
        "label": "",
    }
    return source, clip
