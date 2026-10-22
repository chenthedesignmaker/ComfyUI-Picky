"""Where the bin, the edit and the take record meet.

A :class:`~.sources.Source` knows what it planned and which path it chose. A
:class:`~.sequence.Clip` knows which slice of a source the edit uses. Neither
knows how long anything actually is — that is in the takes, and only
:mod:`lineage` can read it. This module is the one place those three are joined,
so the timeline, the linter and the export all get the same numbers.

**Every length here is lineage time**: a source's takes laid end to end. A clip's
`source_in`/`source_out` are seconds into that clock, which is what lets a clip
start inside generation two and run across the join into generation three.

A source that is only half rendered still has to hold its place on the ruler, so
a length is *measured where takes exist and planned beyond them*: the path's real
durations, plus the prompt budget of every segment past the end of the path. An
unrendered source is therefore all plan (what a v1 shot did), a finished one is
all measurement, and planning a third generation widens its clip immediately
rather than after the render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from . import lineage
from .gallery import Take
from .lineage import Node, Piece
from .sequence import AudioClip, Clip, Lifted, Mute
from .sources import Source


@dataclass
class Bin:
    """A project's takes read as lineages, indexed for the questions asked here.

    Built from *every* take in the project rather than per source, because a
    chain does not respect source boundaries: extends made before the Source
    model landed are spread across sibling shots, and rebuilding a forest per
    source would sever exactly those links. A source claims the roots whose
    trees contain any of its own takes.
    """

    roots: list[Node] = field(default_factory=list)
    index: dict[str, Node] = field(default_factory=dict)
    _roots_by_source: dict[str, list[Node]] = field(default_factory=dict)

    def roots_for(self, source_id: str) -> list[Node]:
        """The chains that touch a source, whole — never clipped to it."""
        return list(self._roots_by_source.get(source_id, []))

    def path_of(self, source: Source) -> list[str]:
        """The path a source plays: its own choice, or the best default.

        Falling back to a default rather than to nothing means a source that has
        rendered but never been chosen for still shows something, which is what
        the old "newest take wins" display did without ever writing it down.

        **The default stops at the depth the source plans.** A chain does not
        respect source boundaries — one made before sources existed runs on into
        a sibling — so a default that followed it to the end would give this
        source a length that belongs partly to the next one, and the edit would
        play those seconds twice. The plan sets the depth; the path only decides
        which take fills each level of it. An explicit choice may still run
        deeper, because somebody said so.
        """
        if source.path:
            return list(source.path)
        default = lineage.default_path(
            self.roots_for(source.id), remembered=source.remembered
        )
        return default[:max(1, len(source.segments))]

    def takes_of(self, path: Iterable[str]) -> list[Take]:
        return lineage.path_takes(list(path), self.index)

    def problems_of(self, path: Iterable[str]) -> list[str]:
        """What is wrong with a source's path, as plain sentences.

        A source's path is not required to start at a root. Every chain that was
        made before sources existed is spread across sibling shots, so after the
        migration the second shot's path legitimately begins one link down — and
        it plays exactly as it always did. What must hold is the link rule: each
        take's recorded parent is the take before it.
        """
        return lineage.validate(list(path), self.index, must_start_at_root=False)

    def branches_of(self, source: Source, path: list[str] | None = None) -> list[list[Node]]:
        """What is on offer at each depth, given the path chosen above it.

        One entry per depth: every root at depth 0, then the children of
        whatever the path selected at the depth before. A take whose parent is
        not on the path is never in this list, which is what makes the old
        parent/child mismatch impossible to display rather than merely unlikely.
        """
        chosen = self.path_of(source) if path is None else list(path)
        roots = self.roots_for(source.id)
        offered: list[list[Node]] = []
        for depth in range(max(len(chosen), len(source.segments))):
            nodes = lineage.children_at(roots, chosen, depth)
            if not nodes and depth >= len(chosen):
                break
            offered.append(nodes)
        return offered


def read(sources: Iterable[Source], takes: Iterable[Take], *,
         include_discarded: bool = False) -> Bin:
    """Index a project's takes for every length and branch question below."""
    roots = lineage.build_forest(takes, include_discarded=include_discarded)
    index = lineage.index_forest(roots)

    by_source: dict[str, list[Node]] = {}
    for source in sources:
        by_source[source.id] = [
            root for root in roots
            if any(node.take.shot == source.id for node in root.walk())
        ]
    return Bin(roots=roots, index=index, _roots_by_source=by_source)


def planned_seconds(source: Source, from_depth: int = 0) -> float:
    """What a source's unrendered segments budget, in lineage time."""
    return round(
        sum(
            segment.prompt.rendered_seconds
            for segment in source.segments[max(0, from_depth):]
        ),
        3,
    )


def source_seconds(source: Source, path_takes: list[Take]) -> float:
    """A source's full length: measured where it rendered, planned beyond."""
    return round(
        lineage.lineage_seconds(path_takes) + planned_seconds(source, len(path_takes)),
        3,
    )


def lengths(sources: Iterable[Source], clips: Iterable[Clip], bin: Bin) -> dict[str, float]:
    """How much source each clip has to cut from, by clip id.

    Keyed by clip rather than by source because two clips on the same source can
    pin different branches, and a pinned branch can be a different length.
    """
    by_id = {source.id: source for source in sources}
    result: dict[str, float] = {}
    for clip in clips:
        source = by_id.get(clip.source_id)
        if source is None:
            # A clip whose source is gone has nothing to cut from. It stays on
            # the timeline occupying zero rather than vanishing, so the mistake
            # is visible instead of silently shortening the edit.
            result[clip.id] = 0.0
            continue
        path = clip.path(bin.path_of(source))
        result[clip.id] = source_seconds(source, bin.takes_of(path))
    return result


def resolve_clip(clip: Clip, source: Source, bin: Bin) -> list[Piece]:
    """The files covering one clip's trim, in order, starting at zero.

    The planned tail contributes nothing — there is no file to play — so a clip
    trimmed into a segment that has not rendered yet simply comes up short here
    rather than pretending.
    """
    path = clip.path(bin.path_of(source))
    takes = bin.takes_of(path)
    available = source_seconds(source, takes)
    return lineage.resolve(takes, clip.source_in, clip.out_point(available))


def resolve_audio(
    clip: Clip, source: Source, bin: Bin, holder: Source | None = None
) -> list[Piece]:
    """The files carrying one clip's *sound*, in order, starting at zero.

    A muted clip resolves to nothing, and a clip playing its own generation
    resolves to exactly what the picture does — same call, same joins, so the
    common case cannot drift apart from what is on screen.

    A borrowed one reads the same *width* of the lineage it borrows from,
    shifted by `offset`. Width rather than range because the picture is what
    holds the clip's place on the ruler: sound that ran longer would either
    overrun the cut or force the edit to re-time itself around an audio choice,
    and both are surprises. A borrowed lineage that is too short simply runs out
    and the tail is silent, which is visible in the monitor rather than hidden.
    """
    sound = clip.audio
    if sound.muted:
        return []
    if not sound.borrowed:
        return resolve_clip(clip, source, bin)

    from_source = holder or source
    path = sound.path_override or bin.path_of(from_source)
    takes = bin.takes_of(path)
    available = source_seconds(from_source, takes)

    own = source_seconds(source, bin.takes_of(clip.path(bin.path_of(source))))
    width = max(0.0, clip.out_point(own) - max(0.0, clip.source_in))
    if width <= 0:
        return []

    start = max(0.0, clip.source_in + sound.offset)
    return lineage.resolve(takes, start, min(start + width, available))


def mute_windows(
    clip: Clip,
    mutes: Iterable[Mute],
    start: float,
    lineage_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """One clip's mutes as ranges on the **timeline**, merged and in order.

    Mutes are stored in lineage seconds so they ride the footage, but they have
    to be applied in timeline seconds, and that conversion is the point of this
    function. A clip that borrows its sound reads a *different* lineage on a
    different clock, so a mute expressed in the picture's lineage cannot be
    matched against an audio piece's file positions at all. Both meet on the
    ruler, and only there.

    `start` is where the clip sits on the timeline. Windows are clamped to the
    clip's trim, so a mute hanging off either edge contributes only its visible
    part while keeping its full extent on the record.

    Overlapping windows are merged. A caller subtracting these from a run of
    sound would otherwise have to cope with a piece being cut twice in the same
    place, and every one of them would have to cope with it identically.
    """
    speed = clip.speed if clip.speed > 0 else 1.0
    low = max(0.0, clip.source_in)
    high = clip.out_point(lineage_seconds)
    if high <= low:
        return []

    spans: list[tuple[float, float]] = []
    for mute in mutes:
        if mute.clip_id != clip.id:
            continue
        narrowed = mute.clamped(low, high)
        if narrowed is None:
            continue
        spans.append((
            round(start + (narrowed.start - low) / speed, 3),
            round(start + (narrowed.end - low) / speed, 3),
        ))

    merged: list[tuple[float, float]] = []
    for span in sorted(spans):
        if merged and span[0] <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged.append(span)
    return merged


def resolve_lifted(item: AudioClip, source: Source, bin: Bin) -> list[Piece]:
    """The files carrying one lane item, in order, starting at zero.

    Only a lifted item resolves here — an imported one is a file on disk that no
    bin knows about, and the caller that owns that folder resolves it instead.
    Returning nothing rather than raising keeps the rule the rest of sound
    follows: a piece that cannot be found is a silence, and a silence is
    something you can hear and go fix.
    """
    origin = item.origin
    if not isinstance(origin, Lifted):
        return []

    path = origin.path_override or bin.path_of(source)
    takes = bin.takes_of(path)
    available = source_seconds(source, takes)

    start = max(0.0, origin.source_in)
    end = min(origin.source_out, available)
    if end - start <= 0:
        return []
    return lineage.resolve(takes, start, end)


def path_to(bin: Bin, take_id: str) -> list[str]:
    """The path from a root down to one take.

    Walks up through each take's recorded parent, so the answer is the
    renderer's, not a guess. A take whose parent is missing from the forest
    starts a path of its own.
    """
    path: list[str] = []
    seen: set[str] = set()
    current = take_id
    while current and current in bin.index and current not in seen:
        seen.add(current)
        path.append(current)
        current = bin.index[current].take.parent_take or ""
    path.reverse()
    return path


def extend_path(bin: Bin, path: list[str], *, remembered: dict[str, str] | None = None,
                depth: int = 0) -> list[str]:
    """Carry a path on down the chain, preferring what was remembered.

    Stops at `depth` when one is given, so choosing a take at depth 1 of a
    three-segment source fills in depth 2 instead of leaving the clip short.
    Remembered choices stay advisory: a stale one costs a mediocre default.
    """
    remembered = remembered or {}
    filled = list(path)
    while filled and (depth <= 0 or len(filled) < depth):
        node = bin.index.get(filled[-1])
        if node is None or not node.children:
            break
        wanted = remembered.get(node.id)
        chosen = next((c for c in node.children if c.id == wanted), None)
        filled.append((chosen or node.children[0]).id)
    return filled
