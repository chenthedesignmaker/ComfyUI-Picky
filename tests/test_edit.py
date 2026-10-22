"""Where the bin, the edit and the take record meet.

The numbers here are the ones the timeline, the linter and the export all read,
so they are worth pinning down once rather than three times.
"""

from __future__ import annotations

from video_prompter import Clip, Segment, Source, VideoPrompt, edit
from video_prompter.gallery import Take


def take(name: str, *, source: str = "src_a", parent: str = "", seconds: float = 4.25,
         rating: int = 0) -> Take:
    return Take(
        prompt_id="p",
        filename=f"{name}.mp4",
        shot=source,
        seconds=seconds,
        parent_take=f"p:{parent}.mp4" if parent else "",
        rating=rating,
        created=f"2026-01-01T00:00:{ord(name[-1]) % 60:02d}Z",
    )


def a_source(segments: int = 1, duration: int = 8, **kwargs) -> Source:
    return Source(
        id="src_a",
        segments=[Segment(prompt=VideoPrompt.blank(duration)) for _ in range(segments)],
        **kwargs,
    )


class TestWhichChainsBelongToASource:
    def test_a_source_claims_the_chains_its_takes_are_in(self):
        takes = [take("a"), take("b", source="src_b")]
        bin = edit.read([a_source(), Source(id="src_b")], takes)
        assert len(bin.roots_for("src_a")) == 1
        assert len(bin.roots_for("src_b")) == 1

    def test_a_chain_is_claimed_whole_even_when_it_crosses_sources(self):
        # Extends made before sources existed are spread over sibling shots.
        # Cutting the chain at that boundary is what made them hard to see.
        takes = [take("a"), take("b", source="src_b", parent="a")]
        bin = edit.read([a_source(), Source(id="src_b")], takes)
        assert len(bin.roots_for("src_a")) == 1
        assert len(list(bin.roots_for("src_a")[0].walk())) == 2

    def test_a_source_with_no_takes_claims_nothing(self):
        bin = edit.read([a_source()], [])
        assert bin.roots_for("src_a") == []


class TestWhichPathASourcePlays:
    def test_a_stored_path_is_used_as_written(self):
        takes = [take("a"), take("b", parent="a")]
        source = a_source(path=[takes[0].id])
        assert edit.read([source], takes).path_of(source) == [takes[0].id]

    def test_an_unchosen_source_falls_back_to_the_best_default(self):
        # Better-rated children sort first, so the default follows them down —
        # as far as the source plans to go, which is two generations here.
        takes = [take("a"), take("b", parent="a", rating=1), take("c", parent="a", rating=5)]
        source = a_source(segments=2)
        assert edit.read([source], takes).path_of(source) == [takes[0].id, takes[2].id]

    def test_the_default_stops_at_the_depth_the_source_plans(self):
        # A chain made before sources existed runs on into a sibling source. A
        # default that followed it to the end would hand this source seconds
        # that belong to the next one, and the edit would play them twice.
        takes = [take("a"), take("b", source="src_b", parent="a")]
        source = a_source(segments=1)
        assert edit.read([source], takes).path_of(source) == [takes[0].id]

    def test_an_explicit_choice_may_still_run_deeper_than_the_plan(self):
        takes = [take("a"), take("b", parent="a")]
        source = a_source(segments=1, path=[t.id for t in takes])
        assert len(edit.read([source], takes).path_of(source)) == 2

    def test_the_default_prefers_what_was_remembered(self):
        takes = [take("a"), take("b", parent="a", rating=1), take("c", parent="a", rating=5)]
        source = a_source(segments=2, remembered={takes[0].id: takes[1].id})
        assert edit.read([source], takes).path_of(source) == [takes[0].id, takes[1].id]


class TestLength:
    def test_an_unrendered_source_holds_its_segments_budget(self):
        # A whole sequence can be laid out before a frame is rendered, which is
        # what a v1 shot's fallback to its prompt length bought.
        source = a_source(duration=8)
        assert edit.source_seconds(source, []) == 8.0

    def test_a_rendered_source_is_measured_not_planned(self):
        takes = [take("a", seconds=5.167)]
        source = a_source(duration=8, path=[takes[0].id])
        bin = edit.read([source], takes)
        assert edit.source_seconds(source, bin.takes_of(source.path)) == 5.167

    def test_a_chain_is_the_sum_of_its_takes(self):
        takes = [take("a", seconds=5.167), take("b", parent="a", seconds=4.25)]
        source = a_source(segments=2, path=[t.id for t in takes])
        bin = edit.read([source], takes)
        assert edit.source_seconds(source, bin.takes_of(source.path)) == 9.417

    def test_a_half_rendered_source_measures_what_exists_and_plans_the_rest(self):
        takes = [take("a", seconds=5.167)]
        source = a_source(segments=2, duration=8, path=[takes[0].id])
        bin = edit.read([source], takes)
        # Generation one is real, generation two is still a budget.
        assert edit.source_seconds(source, bin.takes_of(source.path)) == 13.167

    def test_a_lineage_longer_than_the_plan_is_still_all_measured(self):
        takes = [take("a", seconds=5.167), take("b", parent="a", seconds=4.25)]
        source = a_source(segments=1, path=[t.id for t in takes])
        bin = edit.read([source], takes)
        assert edit.source_seconds(source, bin.takes_of(source.path)) == 9.417

    def test_a_clip_whose_source_is_gone_has_nothing_to_cut_from(self):
        clip = Clip(source_id="src_ghost")
        assert edit.lengths([a_source()], [clip], edit.read([], []))[clip.id] == 0.0

    def test_two_clips_on_one_source_each_get_their_own_length(self):
        takes = [take("a", seconds=5.167), take("b", parent="a", seconds=4.25)]
        source = a_source(segments=2, path=[t.id for t in takes])
        following = Clip(source_id="src_a")
        pinned = Clip(source_id="src_a", path_override=[takes[0].id])
        bin = edit.read([source], takes)
        lengths = edit.lengths([source], [following, pinned], bin)
        assert lengths[following.id] == 9.417
        # Pinned to the root alone: one measured generation plus one planned.
        assert lengths[pinned.id] == 13.167


class TestResolvingAClip:
    def test_a_trim_inside_one_generation_is_one_piece(self):
        takes = [take("a", seconds=5.167), take("b", parent="a", seconds=4.25)]
        source = a_source(segments=2, path=[t.id for t in takes])
        clip = Clip(source_id="src_a", source_in=1.0, source_out=3.0)
        pieces = edit.resolve_clip(clip, source, edit.read([source], takes))
        assert len(pieces) == 1
        assert (pieces[0].source_from, pieces[0].source_to) == (1.0, 3.0)

    def test_a_trim_across_the_join_is_two_pieces_back_to_back(self):
        takes = [take("a", seconds=5.167), take("b", parent="a", seconds=4.25)]
        source = a_source(segments=2, path=[t.id for t in takes])
        clip = Clip(source_id="src_a", source_in=4.0, source_out=7.0)
        pieces = edit.resolve_clip(clip, source, edit.read([source], takes))
        assert [p.take_id for p in pieces] == [takes[0].id, takes[1].id]
        assert round(sum(p.seconds for p in pieces), 3) == 3.0

    def test_the_planned_tail_contributes_no_files(self):
        # Trimming into a generation that has not rendered comes up short here
        # rather than pretending there is something to play.
        takes = [take("a", seconds=5.167)]
        source = a_source(segments=2, duration=8, path=[takes[0].id])
        clip = Clip(source_id="src_a")
        pieces = edit.resolve_clip(clip, source, edit.read([source], takes))
        assert round(sum(p.seconds for p in pieces), 3) == 5.167

    def test_an_unrendered_clip_resolves_to_nothing(self):
        source = a_source()
        clip = Clip(source_id="src_a")
        assert edit.resolve_clip(clip, source, edit.read([source], [])) == []


class TestWhatCountsAsABrokenPath:
    def test_a_sound_path_has_no_problems(self):
        takes = [take("a"), take("b", parent="a")]
        source = a_source(segments=2, path=[t.id for t in takes])
        assert edit.read([source], takes).problems_of(source.path) == []

    def test_a_link_that_does_not_hold_up_is_reported(self):
        takes = [take("a"), take("b"), take("c", parent="b")]
        source = a_source(segments=2, path=[takes[0].id, takes[2].id])
        assert edit.read([source], takes).problems_of(source.path)

    def test_a_source_may_own_one_link_part_way_down_a_chain(self):
        # Every chain made before sources existed is spread over sibling shots,
        # so after the migration the second shot's path starts mid-chain. Those
        # play exactly as they always did and are not broken.
        takes = [take("a"), take("b", source="src_b", parent="a")]
        source = Source(id="src_b", path=[takes[1].id])
        assert edit.read([source], takes).problems_of(source.path) == []


class TestPathsFromATake:
    def test_a_path_is_walked_up_through_recorded_parents(self):
        takes = [take("a"), take("b", parent="a"), take("c", parent="b")]
        bin = edit.read([a_source()], takes)
        assert edit.path_to(bin, takes[2].id) == [t.id for t in takes]

    def test_a_take_with_no_parent_is_a_path_of_one(self):
        takes = [take("a")]
        assert edit.path_to(edit.read([a_source()], takes), takes[0].id) == [takes[0].id]

    def test_an_unknown_take_has_no_path(self):
        assert edit.path_to(edit.read([a_source()], []), "p:ghost.mp4") == []

    def test_extending_carries_a_path_on_down_the_chain(self):
        takes = [take("a"), take("b", parent="a"), take("c", parent="b")]
        bin = edit.read([a_source()], takes)
        assert edit.extend_path(bin, [takes[0].id]) == [t.id for t in takes]

    def test_extending_stops_at_the_planned_depth(self):
        takes = [take("a"), take("b", parent="a"), take("c", parent="b")]
        bin = edit.read([a_source()], takes)
        filled = edit.extend_path(bin, [takes[0].id], depth=2)
        assert filled == [takes[0].id, takes[1].id]

    def test_extending_prefers_what_was_remembered(self):
        takes = [take("a"), take("b", parent="a", rating=1), take("c", parent="a", rating=5)]
        bin = edit.read([a_source()], takes)
        filled = edit.extend_path(bin, [takes[0].id], remembered={takes[0].id: takes[1].id})
        assert filled == [takes[0].id, takes[1].id]


class TestBranchesOnOffer:
    def test_every_root_is_offered_at_depth_zero(self):
        takes = [take("a"), take("b")]
        source = a_source()
        offered = edit.read([source], takes).branches_of(source)
        assert len(offered[0]) == 2

    def test_only_the_chosen_parents_children_are_offered_below(self):
        # A child of a parent you are not standing on is never on screen, so a
        # parent/child mismatch becomes impossible to display.
        takes = [
            take("a"), take("b"),
            take("c", parent="a"), take("d", parent="b"),
        ]
        source = a_source(segments=2, path=[takes[0].id, takes[2].id])
        offered = edit.read([source], takes).branches_of(source)
        assert [node.id for node in offered[1]] == [takes[2].id]
