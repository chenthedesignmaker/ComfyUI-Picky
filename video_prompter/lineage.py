"""Take lineage: the forest of continuations, and how to play a path through it.

A Motion Context extend renders a *new* take that continues an existing one, and
records which one in `take.parent_take`. That single field is the only honest
record of what follows what — it is written by the renderer, at render time, from
the graph that actually ran. Everything about chains should be derived from it
and nothing should be guessed alongside it.

This module does the deriving. It builds the forest, walks a path through it, and
turns a range of *lineage time* into the real files that cover it.

**Lineage time** is the clock a chain runs on: its takes' true durations, laid
end to end. Nothing overlaps. The Motion Context workflow pins 22 frames of the
parent at the head of each continuation and then trims them off again
(`MiniMaxH3MotionContextTrim`), so a 124-frame generation delivers a 102-frame
file that begins exactly where its parent stopped. Concatenation is the whole
join — there is no cross-fade and no overlap to subtract.

That trim is also why durations must be *measured*. A take's recorded `seconds`
is the length that was asked for at queue time; a continuation delivers 0.917s
less than it was asked for. See `gallery` for the repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .gallery import Take


@dataclass
class Node:
    """One take in the forest, with the takes that continue it."""

    take: Take
    depth: int
    children: list["Node"] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.take.id

    def walk(self) -> Iterable["Node"]:
        """This node and everything under it, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class Piece:
    """One contiguous run of a single take's file inside a resolved range.

    `source_from`/`source_to` are seconds into that take's own file. `start` is
    where the piece lands in the resolved range, which begins at zero — the
    caller adds its own timeline offset.
    """

    take_id: str
    url: str
    depth: int
    source_from: float
    source_to: float
    start: float

    @property
    def seconds(self) -> float:
        return round(self.source_to - self.source_from, 3)


def _sort_key(take: Take) -> tuple:
    """Best first: rating wins, then the newer take."""
    return (-int(take.rating or 0), take.created or "")


def build_forest(takes: Iterable[Take], *, include_discarded: bool = False) -> list[Node]:
    """Group takes into continuation trees. Returns the roots.

    A take is a root when it has no parent, or when its parent is not among the
    takes given — an orphan is a root of its own rather than a take that
    silently vanishes from the tree.
    """
    pool = [t for t in takes if include_discarded or not t.discarded]
    by_id = {t.id: t for t in pool}
    nodes = {t.id: Node(take=t, depth=0) for t in pool}

    roots: list[Node] = []
    for take in pool:
        parent_id = take.parent_take
        if parent_id and parent_id in by_id and parent_id != take.id:
            nodes[parent_id].children.append(nodes[take.id])
        else:
            roots.append(nodes[take.id])

    # Depth is assigned by descent, so a take reached through two links is
    # numbered by the route actually taken rather than by its own record.
    def number(node: Node, depth: int, seen: set[str]) -> None:
        node.depth = depth
        node.children.sort(key=lambda child: _sort_key(child.take))
        for child in list(node.children):
            if child.id in seen:
                # A cycle can only come from corrupt metadata. Cut it rather
                # than recurse forever, and leave the rest of the tree intact.
                node.children.remove(child)
                continue
            number(child, depth + 1, seen | {child.id})

    roots.sort(key=lambda node: _sort_key(node.take))
    for root in roots:
        number(root, 0, {root.id})
    return roots


def index_forest(roots: Iterable[Node]) -> dict[str, Node]:
    """Every node in the forest, by take id."""
    return {node.id: node for root in roots for node in root.walk()}


def children_at(roots: Iterable[Node], path: list[str], depth: int) -> list[Node]:
    """The takes offered at `depth`, given the path chosen above it.

    At depth 0 that is every root. Deeper, it is the children of whatever the
    path selected at `depth - 1` — which is what keeps a child of a parent you
    are not standing on off the screen entirely.
    """
    roots = list(roots)
    if depth <= 0:
        return roots
    if len(path) < depth:
        return []
    index = index_forest(roots)
    parent = index.get(path[depth - 1])
    return list(parent.children) if parent else []


def validate(
    path: list[str], index: dict[str, Node], *, must_start_at_root: bool = True
) -> list[str]:
    """Everything wrong with a path, as plain sentences. Empty means good.

    This is the check that replaces the guessing: a path is sound when each
    take's recorded parent is the take before it. Nothing is inferred, and a
    path that does not hold up says so instead of being quietly repaired.

    `must_start_at_root` asks the extra question "and does this path begin a
    lineage?", which is right for a whole lineage and wrong for a source's own
    path: a chain spread over sibling shots leaves each source owning one link
    part-way down, and those play perfectly well. See `edit.Bin.problems_of`.
    """
    problems: list[str] = []
    for position, take_id in enumerate(path):
        node = index.get(take_id)
        if node is None:
            problems.append(f"Step {position + 1} points at a take that is gone ({take_id}).")
            continue
        expected = path[position - 1] if position else ""
        actual = node.take.parent_take or ""
        if position == 0:
            if must_start_at_root and actual and actual in index:
                problems.append(
                    f"Step 1 ({node.take.filename}) continues another take, so it cannot start a lineage."
                )
        elif actual != expected:
            problems.append(
                f"Step {position + 1} ({node.take.filename}) does not continue step {position}."
            )
    return problems


def default_path(
    roots: Iterable[Node],
    *,
    remembered: dict[str, str] | None = None,
    root_id: str | None = None,
) -> list[str]:
    """The path to show when nobody has chosen one.

    Follows `remembered` where it still fits and falls back to the best-rated
    child otherwise. Remembered choices are advisory by design: a stale one
    costs a mediocre default, never a broken path.
    """
    roots = list(roots)
    if not roots:
        return []
    remembered = remembered or {}

    node = next((r for r in roots if r.id == root_id), None) if root_id else None
    if node is None:
        node = roots[0]

    path = [node.id]
    while node.children:
        wanted = remembered.get(node.id)
        chosen = next((c for c in node.children if c.id == wanted), None)
        node = chosen or node.children[0]
        path.append(node.id)
    return path


def path_takes(path: list[str], index: dict[str, Node]) -> list[Take]:
    """The takes named by a path, skipping any that are missing."""
    return [index[take_id].take for take_id in path if take_id in index]


def lineage_seconds(takes: Iterable[Take]) -> float:
    """How long a path runs, end to end."""
    return round(sum(max(0.0, float(t.seconds or 0.0)) for t in takes), 3)


def segment_bounds(takes: Iterable[Take]) -> list[tuple[float, float]]:
    """Each take's start and end in lineage time, in path order."""
    bounds: list[tuple[float, float]] = []
    cursor = 0.0
    for take in takes:
        length = max(0.0, float(take.seconds or 0.0))
        bounds.append((round(cursor, 3), round(cursor + length, 3)))
        cursor += length
    return bounds


def resolve(
    takes: list[Take],
    start: float = 0.0,
    end: float | None = None,
) -> list[Piece]:
    """Turn a range of lineage time into the files that cover it.

    This is the one function the monitor, the lineage preview, a single-segment
    preview and the export all go through, so a join behaves identically
    everywhere. A range that falls in a gap — past the end, or inside a take
    with no measured length — simply contributes nothing.
    """
    bounds = segment_bounds(takes)
    total = bounds[-1][1] if bounds else 0.0
    lo = max(0.0, float(start))
    hi = total if end is None else min(float(end), total)
    if hi <= lo:
        return []

    pieces: list[Piece] = []
    for depth, (take, (seg_start, seg_end)) in enumerate(zip(takes, bounds)):
        if seg_end <= lo or seg_start >= hi:
            continue
        source_from = round(max(lo, seg_start) - seg_start, 3)
        source_to = round(min(hi, seg_end) - seg_start, 3)
        if source_to <= source_from:
            continue
        pieces.append(
            Piece(
                take_id=take.id,
                url=take.view_url(),
                depth=depth,
                source_from=source_from,
                source_to=source_to,
                start=round(max(lo, seg_start) - lo, 3),
            )
        )
    return pieces


def tree_to_dict(roots: Iterable[Node], path: list[str] | None = None) -> list[dict]:
    """The forest as JSON for the editor, marking which nodes are on `path`."""
    on_path = set(path or [])

    def one(node: Node) -> dict:
        return {
            "take_id": node.id,
            "depth": node.depth,
            "on_path": node.id in on_path,
            "take": node.take.to_dict(),
            "children": [one(child) for child in node.children],
        }

    return [one(root) for root in roots]
