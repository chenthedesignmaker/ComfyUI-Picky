from __future__ import annotations

import pytest

from video_prompter import AudioCue, Beat, Line, Storyboard
from video_prompter.storyboard import format_seconds, format_span, parse_span


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "0s"), (4.0, "4s"), (1.5, "1.5s"), (12.25, "12.25s")],
    )
    def test_whole_seconds_lose_the_decimal(self, value, expected):
        assert format_seconds(value) == expected

    def test_span_matches_official_bracket_style(self):
        assert format_span(0, 1.5) == "[0s-1.5s]"


class TestSpanParsing:
    @pytest.mark.parametrize(
        "text",
        ["[0s-2s]", "[0-2s]", "[0s - 2s]", "[0 to 2 seconds]", "[0s to 2s]"],
    )
    def test_accepts_the_forms_official_prompts_use(self, text):
        assert parse_span(text) == (0.0, 2.0)

    def test_returns_none_when_there_is_no_timecode(self):
        assert parse_span("Camera: locked off, no push in.") is None


class TestBeat:
    def test_renders_with_index_and_angle(self):
        beat = Beat(0, 1.5, "high side angle", "he sprints at the roof edge")
        assert beat.render(1) == "[0s-1.5s] Shot 1: high side angle: he sprints at the roof edge"

    def test_renders_without_the_beat_label(self):
        beat = Beat(0, 2, "wide", "the city wakes")
        assert beat.render(1, label_beats=False) == "[0s-2s] wide: the city wakes"

    def test_round_trips_through_parse(self):
        beat = Beat(2.5, 4, "low-angle", "he lands, rolls and rises")
        assert Beat.parse(beat.render(3)) == beat

    def test_parses_a_line_with_no_angle(self):
        shot = Beat.parse("[0s-2s] the camera holds on the door")
        assert shot == Beat(0, 2, "", "the camera holds on the door")

    def test_touching_beats_do_not_overlap(self):
        assert not Beat(0, 2).overlaps(Beat(2, 4))

    def test_straddling_beats_overlap(self):
        assert Beat(0, 1.5).overlaps(Beat(1, 2.5))


class TestCoverage:
    def test_contiguous_beats_leave_no_gap(self):
        storyboard = Storyboard(10, [Beat(0, 4), Beat(4, 10)])
        assert storyboard.gaps == []
        assert storyboard.is_contiguous()

    def test_gap_is_reported_between_and_after_beats(self):
        storyboard = Storyboard(10, [Beat(0, 2), Beat(5, 7)])
        assert storyboard.gaps == [(2.0, 5.0), (7.0, 10.0)]

    def test_overlap_counts_wall_time_once(self):
        storyboard = Storyboard(5, [Beat(0, 1.5), Beat(1, 2.5)])
        assert storyboard.covered == 2.5

    def test_overlapping_pairs_are_reported_by_rendered_index(self):
        storyboard = Storyboard(5, [Beat(0, 1.5), Beat(1, 2.5), Beat(2.5, 4)])
        assert storyboard.overlaps == [(1, 2)]


class TestRefit:
    def test_stretches_beats_proportionally_onto_a_new_budget(self):
        storyboard = Storyboard(5, [Beat(0, 1), Beat(1, 5)]).refit(10)
        assert storyboard.duration == 10
        assert [(s.start, s.end) for s in storyboard.beats] == [(0.0, 2.0), (2.0, 10.0)]

    def test_moves_audio_cues_with_the_picture(self):
        storyboard = Storyboard(5, [Beat(0, 5)], [AudioCue(4, description="hit")]).refit(10)
        assert storyboard.cues[0].start == 8.0

    def test_keeps_sustained_cue_ends(self):
        storyboard = Storyboard(5, [Beat(0, 5)], [AudioCue(1, 2, "bed")]).refit(10)
        assert (storyboard.cues[0].start, storyboard.cues[0].end) == (2.0, 4.0)

    def test_an_empty_storyboard_just_takes_the_new_budget(self):
        assert Storyboard(5).refit(12).duration == 12


class TestMakeContiguous:
    def test_closes_overlaps_and_fills_the_budget(self):
        fixed = Storyboard(5, [Beat(0, 1.5), Beat(1, 2.5), Beat(2.5, 4), Beat(4, 5)]).make_contiguous()
        assert fixed.is_contiguous()
        assert fixed.beats[0].start == 0.0
        assert fixed.beats[-1].end == 5.0

    def test_preserves_beat_order_and_content(self):
        storyboard = Storyboard(6, [Beat(0, 1.5, "a", "first"), Beat(1, 3, "b", "second")])
        fixed = storyboard.make_contiguous()
        assert [s.action for s in fixed.beats] == ["first", "second"]

    def test_relative_weight_survives(self):
        # A shot twice as long as its neighbour stays twice as long.
        fixed = Storyboard(9, [Beat(0, 1), Beat(1, 3)]).make_contiguous()
        first, second = fixed.beats
        assert first.duration == pytest.approx(3.0)
        assert second.duration == pytest.approx(6.0)

    def test_empty_storyboard_is_left_alone(self):
        assert Storyboard(5).make_contiguous().beats == []


class TestSnapping:
    def test_boundaries_land_on_the_half_second(self):
        storyboard = Storyboard(15, [Beat(0, 1.5), Beat(1, 2.5), Beat(2.5, 4), Beat(4, 5)])
        cuts = [s.start for s in storyboard.make_contiguous().beats]
        assert all(cut % 0.5 == 0 for cut in cuts), cuts

    def test_snapping_still_fills_the_budget_exactly(self):
        fixed = Storyboard(15, [Beat(0, 1.5), Beat(1, 2.5), Beat(2.5, 4), Beat(4, 5)]).make_contiguous()
        assert fixed.beats[-1].end == 15.0
        assert fixed.is_contiguous()

    def test_snapping_can_be_turned_off(self):
        # Weights 1.5 and 2.0 of 3.5 total, so the exact cut is 15 * 1.5/3.5.
        fixed = Storyboard(15, [Beat(0, 1.5), Beat(1, 3)]).make_contiguous(snap=None)
        assert fixed.beats[0].end == pytest.approx(6.429, abs=0.001)
        assert fixed.beats[0].end % 0.5 != 0

    def test_no_beat_is_starved_when_the_grid_is_coarse(self):
        # Eight beats on a 4s budget: a 0.5s grid exactly fits, none collapse.
        beats = [Beat(i * 0.1, (i + 1) * 0.1) for i in range(8)]
        fixed = Storyboard(4, beats).make_contiguous()
        assert all(shot.duration > 0 for shot in fixed.beats)
        assert fixed.is_contiguous()

    def test_an_impossible_grid_falls_back_to_exact_cuts(self):
        # Twelve beats cannot each hold 0.5s inside 4s, so proportions win.
        beats = [Beat(i, i + 1) for i in range(12)]
        fixed = Storyboard(4, beats).make_contiguous()
        assert all(shot.duration > 0 for shot in fixed.beats)
        assert fixed.is_contiguous()


class TestExtract:
    """Lifting some shots out as a ruler of their own, for a bin-item split."""

    def _three(self) -> Storyboard:
        return Storyboard(
            12,
            [
                Beat(0, 4, "wide", "the door opens"),
                Beat(4, 8, "medium", "she crosses the room"),
                Beat(8, 12, "close", "the clock reads midnight"),
            ],
        )

    def test_a_tail_is_rebased_to_zero(self):
        board = self._three().extract([1, 2])
        assert [(b.start, b.end) for b in board.beats] == [(0.0, 4.0), (4.0, 8.0)]
        assert board.duration == 8

    def test_each_shot_keeps_its_own_length(self):
        board = Storyboard(12, [Beat(0, 2), Beat(2, 9), Beat(9, 12)]).extract([1, 2])
        assert [b.duration for b in board.beats] == [7.0, 3.0]
        assert board.duration == 10

    def test_the_words_travel_with_the_shot(self):
        board = self._three().extract([2])
        assert board.beats[0].action == "the clock reads midnight"
        assert board.beats[0].angle == "close"

    def test_a_scattered_pick_closes_the_gap_it_left(self):
        board = self._three().extract([0, 2])
        assert [(b.start, b.end) for b in board.beats] == [(0.0, 4.0), (4.0, 8.0)]
        assert [b.action for b in board.beats] == ["the door opens", "the clock reads midnight"]

    def test_order_of_the_pick_does_not_matter(self):
        assert self._three().extract([2, 1]).beats == self._three().extract([1, 2]).beats

    def test_indices_outside_the_storyboard_are_ignored(self):
        assert len(self._three().extract([1, 9, -3]).beats) == 1

    def test_picking_nothing_leaves_nothing(self):
        assert self._three().extract([]).beats == []

    def test_lines_are_copied_not_shared(self):
        board = self._three()
        board.beats[1].lines = [Line(text="I know.")]
        lifted = board.extract([1])
        lifted.beats[0].lines[0].text = "Changed."
        assert board.beats[1].lines[0].text == "I know."

    def test_the_budget_rounds_to_a_whole_second_and_still_tiles(self):
        # H3 counts in whole seconds, so 2.4 + 3.1 has to land on 6, not 5.5.
        board = Storyboard(12, [Beat(0, 2.4), Beat(2.4, 5.5), Beat(5.5, 12)]).extract([0, 1])
        assert board.duration == 6
        assert board.beats[-1].end == 6.0
        assert board.is_contiguous()

    def test_a_cue_travels_with_the_shot_it_starts_in(self):
        board = self._three()
        board.cues = [AudioCue(9.0, None, "the chime")]
        assert board.extract([1, 2]).cues == [AudioCue(5.0, None, "the chime")]

    def test_a_cue_in_a_shot_nobody_kept_is_dropped(self):
        board = self._three()
        board.cues = [AudioCue(1.0, None, "the latch")]
        assert board.extract([1, 2]).cues == []

    def test_a_bed_is_clipped_to_what_is_left(self):
        board = self._three()
        board.cues = [AudioCue(5.0, 11.0, "the hum carries")]
        kept = board.extract([1]).cues[0]
        assert (kept.start, kept.end) == (1.0, 4.0)

    def test_a_shorter_pick_than_the_trained_floor_is_kept_honest(self):
        # Warned about by the linter, not rounded up to five here: a two-second
        # split is a decision, and inflating it would be a different one.
        board = Storyboard(12, [Beat(0, 2), Beat(2, 12)]).extract([0])
        assert board.duration == 2


class TestShotGrouping:
    """Beats group into the `[Shot N]` blocks they compose into."""

    def _board(self, *cuts: bool) -> Storyboard:
        beats = [
            Beat(index, index + 1, action=f"moment {index}", cut=cut)
            for index, cut in enumerate(cuts)
        ]
        return Storyboard(duration=len(cuts), beats=beats)

    def test_every_beat_cutting_is_one_shot_each(self):
        board = self._board(True, True, True)
        assert [len(shot) for shot in board.shots()] == [1, 1, 1]
        assert board.shot_numbers() == [1, 2, 3]

    def test_a_no_cut_beat_joins_the_shot_before_it(self):
        board = self._board(True, False, False)
        assert [len(shot) for shot in board.shots()] == [3]
        assert board.shot_numbers() == [1, 1, 1]

    def test_the_first_beat_opens_a_shot_whatever_it_says(self):
        # Nothing precedes it, so there is nothing for it to continue.
        board = self._board(False, False)
        assert [len(shot) for shot in board.shots()] == [2]
        assert board.shot_numbers() == [1, 1]

    def test_grouping_follows_time_not_list_order(self):
        board = Storyboard(
            duration=3,
            beats=[
                Beat(2, 3, action="third", cut=False),
                Beat(0, 1, action="first"),
                Beat(1, 2, action="second"),
            ],
        )
        assert [[b.action for b in shot] for shot in board.shots()] == [
            ["first"],
            ["second", "third"],
        ]

    def test_extract_promotes_the_first_kept_beat_to_a_cut(self):
        # It is Shot 1 of a new generation now; what it continued is gone.
        board = self._board(True, False, False)
        taken = board.extract([1, 2])
        assert [beat.cut for beat in taken.beats] == [True, False]
