from __future__ import annotations

import pytest

from video_prompter import lineage
from video_prompter.gallery import Take


def take(name: str, *, parent: str = "", seconds: float = 4.25, rating: int = 0,
         created: str = "", discarded: bool = False) -> Take:
    return Take(
        prompt_id="p",
        filename=f"{name}.mp4",
        seconds=seconds,
        parent_take=f"p:{parent}.mp4" if parent else "",
        rating=rating,
        created=created or f"2026-01-01T00:00:{int(name[-1]) if name[-1].isdigit() else 0:02d}Z",
        discarded=discarded,
    )


def chain(*names: str, seconds: float = 4.25) -> list[Take]:
    """A straight lineage: each take continues the one before it."""
    made = []
    for index, name in enumerate(names):
        parent = names[index - 1] if index else ""
        # The root is a full generation; continuations are trimmed shorter.
        made.append(take(name, parent=parent, seconds=5.167 if index == 0 else seconds))
    return made


class TestForest:
    def test_a_straight_chain_is_one_root_with_depth(self):
        roots = lineage.build_forest(chain("a", "b", "c"))
        assert len(roots) == 1
        assert [node.depth for node in roots[0].walk()] == [0, 1, 2]

    def test_unrelated_takes_are_separate_roots(self):
        roots = lineage.build_forest([take("a"), take("b")])
        assert len(roots) == 2

    def test_a_take_whose_parent_is_absent_becomes_its_own_root(self):
        # An orphan must stay visible rather than vanish from the tree.
        orphan = take("b", parent="gone")
        roots = lineage.build_forest([orphan])
        assert [node.id for node in roots] == [orphan.id]

    def test_siblings_branch_from_one_parent(self):
        takes = [take("a"), take("b", parent="a"), take("c", parent="a")]
        roots = lineage.build_forest(takes)
        assert len(roots) == 1
        assert len(roots[0].children) == 2

    def test_discarded_takes_are_left_out_by_default(self):
        takes = [take("a"), take("b", parent="a", discarded=True)]
        assert not lineage.build_forest(takes)[0].children
        assert lineage.build_forest(takes, include_discarded=True)[0].children

    def test_better_rated_children_sort_first(self):
        takes = [take("a"), take("b", parent="a", rating=1), take("c", parent="a", rating=5)]
        roots = lineage.build_forest(takes)
        assert roots[0].children[0].take.filename == "c.mp4"

    def test_a_cycle_in_corrupt_metadata_does_not_hang(self):
        a = take("a", parent="b")
        b = take("b", parent="a")
        roots = lineage.build_forest([a, b])
        # Whichever becomes the root, the walk must terminate and cover no
        # take twice.
        seen = [node.id for root in roots for node in root.walk()]
        assert len(seen) == len(set(seen))


class TestValidate:
    def test_a_real_chain_has_no_problems(self):
        takes = chain("a", "b", "c")
        index = lineage.index_forest(lineage.build_forest(takes))
        assert lineage.validate([t.id for t in takes], index) == []

    def test_a_step_that_does_not_continue_the_one_before_is_reported(self):
        takes = chain("a", "b", "c")
        index = lineage.index_forest(lineage.build_forest(takes))
        # Skipping the middle link is exactly the mismatch the old heuristics
        # used to paper over.
        problems = lineage.validate([takes[0].id, takes[2].id], index)
        assert len(problems) == 1
        assert "does not continue" in problems[0]

    def test_a_missing_take_is_reported_rather_than_substituted(self):
        takes = chain("a", "b")
        index = lineage.index_forest(lineage.build_forest(takes))
        problems = lineage.validate([takes[0].id, "p:gone.mp4"], index)
        assert "gone" in problems[0]

    def test_a_continuation_cannot_start_a_lineage(self):
        takes = chain("a", "b")
        index = lineage.index_forest(lineage.build_forest(takes))
        problems = lineage.validate([takes[1].id], index)
        assert "cannot start a lineage" in problems[0]

    def test_a_path_may_be_allowed_to_start_part_way_down(self):
        # What a source owns after the migration: chains made before sources
        # existed are spread over sibling shots, so the second shot's path
        # legitimately begins one link down and plays exactly as it always did.
        takes = chain("a", "b")
        index = lineage.index_forest(lineage.build_forest(takes))
        assert lineage.validate([takes[1].id], index, must_start_at_root=False) == []

    def test_the_link_rule_still_holds_for_a_path_starting_part_way_down(self):
        takes = chain("a", "b", "c")
        index = lineage.index_forest(lineage.build_forest(takes))
        problems = lineage.validate(
            [takes[0].id, takes[2].id], index, must_start_at_root=False
        )
        assert "does not continue" in problems[0]


class TestDefaultPath:
    def test_it_runs_to_the_end_of_the_chain(self):
        takes = chain("a", "b", "c")
        roots = lineage.build_forest(takes)
        assert lineage.default_path(roots) == [t.id for t in takes]

    def test_remembered_choices_are_followed(self):
        takes = [take("a"), take("b", parent="a", rating=5), take("c", parent="a")]
        roots = lineage.build_forest(takes)
        chosen = lineage.default_path(roots, remembered={takes[0].id: takes[2].id})
        assert chosen[1] == takes[2].id

    def test_a_stale_memory_falls_back_to_the_best_child(self):
        # Advisory by design: a remembered id that no longer exists costs a
        # mediocre default, never a broken path.
        takes = [take("a"), take("b", parent="a", rating=5)]
        roots = lineage.build_forest(takes)
        chosen = lineage.default_path(roots, remembered={takes[0].id: "p:gone.mp4"})
        assert chosen == [takes[0].id, takes[1].id]


class TestResolve:
    def test_lineage_time_is_the_measured_lengths_laid_end_to_end(self):
        takes = chain("a", "b", "c")
        assert lineage.lineage_seconds(takes) == pytest.approx(5.167 + 4.25 + 4.25)

    def test_the_whole_lineage_resolves_to_every_take_in_order(self):
        takes = chain("a", "b", "c")
        pieces = lineage.resolve(takes)
        assert [p.depth for p in pieces] == [0, 1, 2]
        assert pieces[0].start == 0.0
        assert pieces[1].start == pytest.approx(5.167)

    def test_one_segment_resolves_to_one_piece(self):
        takes = chain("a", "b", "c")
        start, end = lineage.segment_bounds(takes)[1]
        pieces = lineage.resolve(takes, start, end)
        assert len(pieces) == 1
        assert pieces[0].source_from == 0.0
        assert pieces[0].source_to == pytest.approx(4.25)

    def test_a_range_across_a_join_splits_into_two_pieces(self):
        takes = chain("a", "b", "c")
        pieces = lineage.resolve(takes, 4.167, 6.167)
        assert [p.depth for p in pieces] == [0, 1]
        # The first piece runs out the tail of the root...
        assert pieces[0].source_from == pytest.approx(4.167)
        assert pieces[0].source_to == pytest.approx(5.167)
        # ...and the second picks up at the head of the continuation, with no
        # overlap: the workflow already trimmed the pinned context frames.
        assert pieces[1].source_from == 0.0
        assert pieces[1].source_to == pytest.approx(1.0)
        assert pieces[1].start == pytest.approx(1.0)

    def test_a_range_past_the_end_is_clamped(self):
        takes = chain("a", "b")
        pieces = lineage.resolve(takes, 0.0, 999.0)
        assert sum(p.seconds for p in pieces) == pytest.approx(lineage.lineage_seconds(takes))

    def test_an_empty_range_resolves_to_nothing(self):
        takes = chain("a", "b")
        assert lineage.resolve(takes, 3.0, 3.0) == []
        assert lineage.resolve([], 0.0, 5.0) == []


class TestChildrenAt:
    def test_only_the_selected_parents_children_are_offered(self):
        # This is the invariant the columns UI rests on: a child of a parent you
        # are not standing on is never shown at all.
        takes = [
            take("a"), take("b", parent="a"), take("c", parent="a"),
            take("d", parent="b"), take("e", parent="c"),
        ]
        roots = lineage.build_forest(takes)
        path = [takes[0].id, takes[1].id]
        offered = [node.take.filename for node in lineage.children_at(roots, path, 2)]
        assert offered == ["d.mp4"]

    def test_depth_zero_offers_every_root(self):
        roots = lineage.build_forest([take("a"), take("b")])
        assert len(lineage.children_at(roots, [], 0)) == 2
