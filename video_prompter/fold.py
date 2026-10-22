"""Folding a chain that was made before sources existed back into one source.

Every extend used to be authored as its own shot, so a ten-generation chain
landed on disk as ten shots — and the v1 migration deliberately kept it that
way, one source and one clip each, rather than guessing which shots belonged
together. This module is where that guess stops being a guess.

The renderer wrote the answer down at render time: a continuation records the
take it continues in `take.parent_take`. When one source's path ends on exactly
the take another source's path begins with, those two are the same lineage split
across two bin rows. That is a fact from the take record, not an inference from
names, ordering or position, which is why folding can run by itself where the
old `continuation_of` heuristics could not.

Two things keep it honest:

- **The link must be unique in both directions.** A take continued by two
  different sources is a fork the author used twice, and folding it would pick
  one arm for them. Ambiguous runs are left alone and reported.
- **The edit does not move.** Each absorbed clip is rebased onto the surviving
  source by its offset in lineage time, so the timeline plays the same frames in
  the same order. Only where clips already sat next to each other, untrimmed and
  in order, do they collapse into the one clip that run always was.

Folding is idempotent: a folded project has no sibling runs left to find, so a
second pass over it does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import edit
from .edit import Bin
from .projects import Project
from .sequence import Clip, Timeline
from .sources import Source

# Lineage seconds are rounded to milliseconds everywhere, so "these clips meet"
# is judged at that resolution rather than by exact equality.
JOIN_TOLERANCE = 0.002


@dataclass(frozen=True)
class Run:
    """A chain of sibling sources, head first, and the path it plays."""

    source_ids: list[str]
    path: list[str]


@dataclass(frozen=True)
class Skip:
    """A run that links up but is not safe to fold, and why not."""

    source_ids: list[str]
    reason: str


@dataclass
class Plan:
    runs: list[Run] = field(default_factory=list)
    skipped: list[Skip] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.runs)


@dataclass
class Folded:
    """What one fold did, in the terms a person would describe it in."""

    kept: str
    name: str
    absorbed: list[str] = field(default_factory=list)
    absorbed_names: list[str] = field(default_factory=list)
    generations: int = 0
    clips_before: int = 0
    clips_after: int = 0


@dataclass
class Result:
    folds: list[Folded] = field(default_factory=list)
    skipped: list[Skip] = field(default_factory=list)
    # Absorbed source id -> the source id that now owns its takes.
    reassign: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.folds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": [
                {
                    "kept": one.kept,
                    "name": one.name,
                    "absorbed": list(one.absorbed),
                    "absorbed_names": list(one.absorbed_names),
                    "generations": one.generations,
                    "clips_before": one.clips_before,
                    "clips_after": one.clips_after,
                }
                for one in self.folds
            ],
            "skipped": [
                {"sources": list(one.source_ids), "reason": one.reason}
                for one in self.skipped
            ],
        }


# -- finding the runs ------------------------------------------------------


def plan(project: Project, bin: Bin) -> Plan:
    """Which sources are one chain split up, and which cannot be folded."""
    if len(project.sources) < 2:
        return Plan()

    paths = {source.id: bin.path_of(source) for source in project.sources}
    by_id = {source.id: source for source in project.sources}
    follows = _links(project, bin, paths)

    result = Plan()
    taken = set(follows.values())
    for source in project.sources:
        if source.id in taken:
            continue
        chain = _chain_from(source.id, follows)
        if len(chain) < 2:
            continue
        reason = _why_not(chain, by_id, paths, bin)
        if reason:
            result.skipped.append(Skip(source_ids=chain, reason=reason))
        else:
            result.runs.append(
                Run(source_ids=chain, path=[t for sid in chain for t in paths[sid]])
            )
    return result


def _links(project: Project, bin: Bin, paths: dict[str, list[str]]) -> dict[str, str]:
    """Source id -> the source that continues it, where that is unambiguous.

    Both directions have to be unique. One source continued by two others is a
    fork used twice over, and one source continuing two others means two bin
    rows end on the same take — either way there is a choice to be made, and
    making it here is exactly the guessing this rewrite deleted.
    """
    ends_at: dict[str, list[str]] = {}
    for source in project.sources:
        path = paths[source.id]
        if path:
            ends_at.setdefault(path[-1], []).append(source.id)

    successors: dict[str, list[str]] = {}
    predecessors: dict[str, list[str]] = {}
    for source in project.sources:
        path = paths[source.id]
        node = bin.index.get(path[0]) if path else None
        parent = node.take.parent_take if node else ""
        if not parent:
            continue
        for owner in ends_at.get(parent, []):
            if owner == source.id:
                continue
            successors.setdefault(owner, []).append(source.id)
            predecessors.setdefault(source.id, []).append(owner)

    return {
        parent: children[0]
        for parent, children in successors.items()
        if len(children) == 1 and len(predecessors.get(children[0], [])) == 1
    }


def _chain_from(head: str, follows: dict[str, str]) -> list[str]:
    """Every source downstream of `head`, in order. Cycle-safe."""
    chain = [head]
    seen = {head}
    nxt = follows.get(head)
    while nxt and nxt not in seen:
        chain.append(nxt)
        seen.add(nxt)
        nxt = follows.get(nxt)
    return chain


def _why_not(
    chain: list[str], by_id: dict[str, Source], paths: dict[str, list[str]], bin: Bin
) -> str:
    """Empty when this run is safe to fold; a plain sentence when it is not."""
    # A source part-way down the run that plans more than it has rendered is
    # ambiguous: its unrendered segment may well *be* the generation the next
    # source rendered, and nothing on disk says whether it is. Folding would have
    # to either double-count those planned seconds or throw the segment away.
    for source_id in chain[:-1]:
        source = by_id[source_id]
        if len(paths[source_id]) != len(source.segments):
            return (
                f"{source.name or source_id} plans {len(source.segments)} generation(s) "
                f"but has rendered {len(paths[source_id])}, and it is not the last of "
                "the run."
            )

    combined = [take_id for source_id in chain for take_id in paths[source_id]]
    problems = bin.problems_of(combined)
    return problems[0] if problems else ""


# -- doing it --------------------------------------------------------------


def fold(project: Project, bin: Bin, prepared: Plan | None = None) -> Result:
    """Fold every unambiguous run in `project`, in place.

    The project is mutated and the caller saves it. Takes are not touched here:
    the returned `reassign` map says which source id now owns them, because the
    take record is a different store with its own lock.
    """
    prepared = plan(project, bin) if prepared is None else prepared
    result = Result(skipped=list(prepared.skipped))
    for run in prepared.runs:
        result.folds.append(_fold_one(project, bin, run))
        for absorbed in run.source_ids[1:]:
            result.reassign[absorbed] = run.source_ids[0]
    return result


def _fold_one(project: Project, bin: Bin, run: Run) -> Folded:
    by_id = {source.id: source for source in project.sources}
    members = [by_id[source_id] for source_id in run.source_ids]
    paths = [bin.path_of(member) for member in members]
    lengths = [
        edit.source_seconds(member, bin.takes_of(path))
        for member, path in zip(members, paths)
    ]
    keep = members[0]
    before = sum(len(project.timeline.using(member.id)) for member in members)

    # Rebase first, while each member's own length and path are still its own.
    rebased: set[str] = set()
    offset = 0.0
    above: list[str] = []
    for member, path, length in zip(members, paths, lengths):
        for clip in project.timeline.using(member.id):
            _rebase(clip, keep.id, offset, length, above)
            rebased.add(clip.id)
        offset = round(offset + length, 3)
        above = above + list(path)

    for member in members[1:]:
        # The absorbed source's name is the only thing folding really loses, so
        # it goes onto the generation it named. Nothing reads `notes`; this is
        # the same kind of keeping as `legacy_continuation_of`.
        if member.segments and not member.segments[0].notes and member.name:
            member.segments[0].notes = member.name
        keep.segments.extend(member.segments)
        keep.remembered.update(member.remembered)

    # Stored, not left to the default: the default follows the best-rated child,
    # which is not necessarily the take each of these rows was picked onto.
    keep.set_path(run.path)

    absorbed = set(run.source_ids[1:])
    # Filtered rather than removed: `remove_source` ripples into the timeline,
    # and these clips have just been rebased onto the source that survives.
    project.sources = [s for s in project.sources if s.id not in absorbed]

    _join(project.timeline, rebased)

    return Folded(
        kept=keep.id,
        name=keep.name,
        absorbed=[member.id for member in members[1:]],
        absorbed_names=[member.name for member in members[1:]],
        generations=len(keep.segments),
        clips_before=before,
        clips_after=len(project.timeline.using(keep.id)),
    )


def _rebase(clip: Clip, keep_id: str, offset: float, length: float, above: list[str]) -> None:
    """Point a clip at the surviving source, `offset` seconds further in.

    The out point is pinned to a number even when it was open. `None` means "to
    the end of the lineage", and after a fold that end is the end of the whole
    chain — leaving it open is how one clip would silently swallow the ones
    after it.
    """
    out = clip.out_point(length)
    start = min(max(0.0, clip.source_in), out)

    clip.source_id = keep_id
    clip.source_in = round(start + offset, 3)
    clip.source_out = round(out + offset, 3)
    if clip.path_override is not None:
        # An override is a whole path from depth 0, so it gains the generations
        # now sitting above it.
        clip.path_override = list(above) + list(clip.path_override)


def _join(timeline: Timeline, rebased: set[str]) -> int:
    """Collapse rebased clips that already ran back to back into one.

    Only clips this fold moved, and only where the join is invisible: same
    source, same speed, same pin, both shown or both hidden, no labels of their
    own, and the first ending exactly where the second begins. A clip that was
    trimmed does not meet its neighbour, so a trim can never be folded away.
    """
    joined = 0
    index = 0
    while index < len(timeline.clips) - 1:
        left, right = timeline.clips[index], timeline.clips[index + 1]
        if _joinable(left, right, rebased):
            left.source_out = right.source_out
            timeline.clips.pop(index + 1)
            joined += 1
            continue
        index += 1
    return joined


def _joinable(left: Clip, right: Clip, rebased: set[str]) -> bool:
    if left.id not in rebased or right.id not in rebased:
        return False
    if left.source_id != right.source_id:
        return False
    if left.enabled != right.enabled or left.speed != right.speed:
        return False
    if left.label or right.label:
        return False
    if left.path_override != right.path_override:
        return False
    if left.source_out is None or right.source_out is None:
        return False
    return abs(left.source_out - right.source_in) <= JOIN_TOLERANCE
