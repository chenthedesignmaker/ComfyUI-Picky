"""The timeline: clips, their two clocks, and the order that is the edit."""

from __future__ import annotations

import pytest

from video_prompter import Clip, Timeline
from video_prompter.sequence import MAX_SPEED, new_clip_id


def a_clip(source: str = "src_a", **kwargs) -> Clip:
    return Clip(source_id=source, **kwargs)


class TestIds:
    def test_ids_are_unique(self):
        assert new_clip_id() != new_clip_id()

    def test_a_clip_gets_one_without_being_asked(self):
        assert a_clip().id.startswith("clip_")

    def test_an_id_survives_a_round_trip(self):
        clip = a_clip()
        assert Clip.from_dict(clip.to_dict()).id == clip.id

    def test_a_clip_written_without_an_id_is_given_one(self):
        assert Clip.from_dict({"source_id": "src_a"}).id.startswith("clip_")


class TestLineageAndTimelineTime:
    """`lineage_seconds` is passed in: only the take record knows the real length."""

    def test_an_unrendered_clip_has_nothing_to_cut_from(self):
        # Unlike a shot, a clip cannot fall back to a prompt's budget — how long
        # a lineage runs depends on how many generations actually exist.
        assert a_clip().timeline_seconds() == 0.0

    def test_the_lineage_length_is_the_source_length(self):
        assert a_clip().timeline_seconds(15.083) == 15.083

    def test_trimming_shortens_the_timeline(self):
        clip = a_clip(source_in=1.0, source_out=3.0)
        assert clip.timeline_seconds(8.0) == 2.0

    def test_an_open_out_point_runs_to_the_end_of_the_lineage(self):
        assert a_clip(source_in=2.0).timeline_seconds(8.0) == 6.0

    def test_speed_divides_the_timeline_length_not_the_source(self):
        clip = a_clip(speed=2.0)
        assert clip.trimmed_seconds(8.0) == 8.0
        assert clip.timeline_seconds(8.0) == 4.0

    def test_a_half_speed_clip_runs_longer(self):
        assert a_clip(speed=0.5).timeline_seconds(8.0) == 16.0

    def test_an_out_point_past_the_lineage_is_clamped(self):
        assert a_clip(source_out=99.0).timeline_seconds(8.0) == 8.0

    def test_an_inverted_trim_is_zero_not_negative(self):
        assert a_clip(source_in=6.0, source_out=2.0).timeline_seconds(8.0) == 0.0

    @pytest.mark.parametrize("value", [0, -1, "fast", None])
    def test_a_nonsense_speed_falls_back_to_one(self, value):
        assert Clip.from_dict({"speed": value}).speed == 1.0

    def test_an_absurd_speed_is_clamped(self):
        assert Clip.from_dict({"speed": 1000}).speed == MAX_SPEED


class TestWhichPathAClipPlays:
    def test_a_clip_follows_its_source_by_default(self):
        assert a_clip().path(["t1", "t2"]) == ["t1", "t2"]

    def test_a_pinned_clip_keeps_its_own_branch(self):
        clip = a_clip(path_override=["t1", "t9"])
        assert clip.path(["t1", "t2"]) == ["t1", "t9"]

    def test_pinning_survives_a_round_trip(self):
        clip = a_clip(path_override=["t1", "t9"])
        assert Clip.from_dict(clip.to_dict()).path_override == ["t1", "t9"]

    def test_following_is_not_the_same_as_pinning_an_empty_list(self):
        # `None` means follow; an empty list would be a pin onto nothing.
        assert Clip.from_dict({"path_override": None}).path_override is None


class TestTimelineOrder:
    def _three(self) -> Timeline:
        return Timeline([a_clip("src_a"), a_clip("src_b"), a_clip("src_c")])

    def _eight(self, timeline: Timeline) -> dict[str, float]:
        return {clip.id: 8.0 for clip in timeline}

    def test_starts_are_a_running_total(self):
        timeline = self._three()
        starts = timeline.starts_at(self._eight(timeline))
        assert [starts[clip.id] for clip in timeline] == [0.0, 8.0, 16.0]

    def test_duration_is_the_sum(self):
        timeline = Timeline([a_clip("src_a"), a_clip("src_b")])
        assert timeline.duration(self._eight(timeline)) == 16.0

    def test_a_disabled_clip_occupies_nothing(self):
        timeline = self._three()
        timeline.clips[1].enabled = False
        lengths = self._eight(timeline)
        assert timeline.duration(lengths) == 16.0
        # The third clip follows the first directly.
        assert timeline.starts_at(lengths)[timeline.clips[2].id] == 8.0

    def test_a_longer_lineage_moves_everything_after_it(self):
        timeline = Timeline([a_clip("src_a"), a_clip("src_b")])
        first, second = timeline.clips
        starts = timeline.starts_at({first.id: 15.083, second.id: 8.0})
        assert starts[second.id] == 15.083

    def test_moving_a_clip_changes_the_edit(self):
        timeline = self._three()
        timeline.move(timeline.clips[2].id, 0)
        assert [clip.source_id for clip in timeline] == ["src_c", "src_a", "src_b"]

    def test_removing_ripples(self):
        timeline = self._three()
        lengths = self._eight(timeline)
        timeline.remove(timeline.clips[0].id)
        assert timeline.starts_at(lengths)[timeline.clips[0].id] == 0.0
        assert timeline.duration(lengths) == 16.0

    def test_removing_something_absent_says_so(self):
        assert Timeline([a_clip()]).remove("clip_nope") is False

    def test_inserting_at_a_position(self):
        timeline = Timeline([a_clip("src_a"), a_clip("src_c")])
        timeline.add(a_clip("src_b"), at=1)
        assert [clip.source_id for clip in timeline] == ["src_a", "src_b", "src_c"]

    def test_inserting_past_the_end_appends(self):
        timeline = Timeline([a_clip("src_a")])
        timeline.add(a_clip("src_b"), at=99)
        assert [clip.source_id for clip in timeline] == ["src_a", "src_b"]

    def test_index_of_a_missing_clip_raises(self):
        with pytest.raises(KeyError):
            Timeline([a_clip()]).index_of("clip_nope")

    def test_using_finds_every_clip_on_one_source(self):
        timeline = Timeline([a_clip("src_a"), a_clip("src_b"), a_clip("src_a")])
        assert len(timeline.using("src_a")) == 2


class TestSplitAndDuplicate:
    def test_splitting_makes_two_windows_on_one_lineage(self):
        timeline = Timeline([a_clip("src_a")])
        head, tail = timeline.split(timeline.clips[0].id, 3.0, 8.0)
        assert head.source_id == tail.source_id == "src_a"
        assert (head.source_in, head.source_out) == (0.0, 3.0)
        assert (tail.source_in, tail.source_out) == (3.0, 8.0)

    def test_a_split_keeps_the_total_length(self):
        timeline = Timeline([a_clip("src_a")])
        clip_id = timeline.clips[0].id
        timeline.split(clip_id, 3.0, 8.0)
        lengths = {clip.id: 8.0 for clip in timeline}
        assert timeline.duration(lengths) == 8.0

    def test_the_cut_lands_in_source_time_not_timeline_time(self):
        # At 2x, three seconds of timeline is six seconds of source.
        timeline = Timeline([a_clip("src_a", speed=2.0)])
        head, _ = timeline.split(timeline.clips[0].id, 3.0, 8.0)
        assert head.source_out == 6.0

    def test_splitting_outside_the_clip_is_refused(self):
        timeline = Timeline([a_clip("src_a")])
        with pytest.raises(ValueError):
            timeline.split(timeline.clips[0].id, 99.0, 8.0)

    def test_splitting_at_zero_is_refused(self):
        timeline = Timeline([a_clip("src_a")])
        with pytest.raises(ValueError):
            timeline.split(timeline.clips[0].id, 0.0, 8.0)

    def test_duplicating_makes_another_use_of_the_same_lineage(self):
        timeline = Timeline([a_clip("src_a", source_in=1.0)])
        copy = timeline.duplicate(timeline.clips[0].id)
        assert copy.source_id == "src_a"
        assert copy.source_in == 1.0
        assert copy.id != timeline.clips[0].id

    def test_a_duplicated_pin_is_its_own_list(self):
        timeline = Timeline([a_clip("src_a", path_override=["t1"])])
        copy = timeline.duplicate(timeline.clips[0].id)
        copy.path_override.append("t2")
        assert timeline.clips[0].path_override == ["t1"]


class TestRoundTrip:
    def test_a_full_clip_survives(self):
        clip = a_clip(
            source_in=1.5, source_out=6.0, speed=0.5, enabled=False,
            label="the leap", path_override=["t1", "t2"],
        )
        assert Clip.from_dict(clip.to_dict()) == clip

    def test_a_timeline_survives(self):
        timeline = Timeline([a_clip("src_a"), a_clip("src_b")])
        assert Timeline.from_dict(timeline.to_dict()) == timeline
