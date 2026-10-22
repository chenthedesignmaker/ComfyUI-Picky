"""The bin: a source's segments, and the one path that says which take won."""

from __future__ import annotations

from video_prompter import Beat, Segment, Source, VideoPrompt, blank_source
from video_prompter.sources import (
    REVISION_LIMIT,
    new_segment_id,
    new_source_id,
    shot_label,
    split_off,
)


def a_segment(duration: int = 8) -> Segment:
    return Segment(prompt=VideoPrompt.blank(duration))


class TestIds:
    def test_ids_are_unique(self):
        assert new_source_id() != new_source_id()
        assert new_segment_id() != new_segment_id()

    def test_a_source_and_a_segment_get_one_without_being_asked(self):
        source = Source()
        assert source.id.startswith("src_")
        assert source.segments[0].id.startswith("seg_")

    def test_a_source_written_without_an_id_is_given_one(self):
        assert Source.from_dict({"name": "Legacy"}).id.startswith("src_")


class TestSegments:
    def test_a_source_is_never_segmentless(self):
        # Something with no segments can be neither rendered nor drawn, so one
        # blank segment is the honest minimum.
        assert len(Source().segments) == 1

    def test_depth_is_the_number_of_planned_generations(self):
        source = Source(segments=[a_segment(), a_segment()])
        assert source.depth == 2

    def test_adding_a_segment_plans_one_more_generation(self):
        source = Source(segments=[a_segment()])
        added = source.add_segment()
        assert source.segments[-1] is added
        assert source.depth == 2

    def test_segment_at_is_bounded_both_ways(self):
        source = Source(segments=[a_segment()])
        assert source.segment_at(0) is source.segments[0]
        assert source.segment_at(1) is None
        assert source.segment_at(-1) is None

    def test_a_legacy_segment_defaults_the_quality(self):
        assert Segment.from_dict({}).quality == "preview"


class TestPath:
    def test_setting_a_path_records_each_step_as_a_preference(self):
        # Remembering here rather than at the click site means every route into
        # a path leaves the same trail.
        source = Source()
        source.set_path(["t1", "t2", "t3"])
        assert source.remembered == {"t1": "t2", "t2": "t3"}

    def test_a_one_step_path_remembers_nothing(self):
        source = Source()
        source.set_path(["t1"])
        assert source.remembered == {}

    def test_setting_a_path_replaces_the_old_one(self):
        source = Source()
        source.set_path(["t1", "t2"])
        source.set_path(["t1"])
        assert source.path == ["t1"]
        # The old preference survives, which is what makes it advisory: it is a
        # default to come back to, never a claim about the current path.
        assert source.remembered == {"t1": "t2"}

    def test_remembering_ignores_a_blank_end(self):
        source = Source()
        source.remember("t1", "")
        assert source.remembered == {}


class TestRoundTrip:
    def test_a_full_source_survives(self):
        source = Source(
            name="The leap",
            segments=[a_segment(), a_segment(12)],
            path=["t1", "t2"],
            remembered={"t1": "t2"},
            legacy_continuation_of="sh_parent",
        )
        assert Source.from_dict(source.to_dict()) == source

    def test_blank_sources_start_at_the_model_floor(self):
        # Shorter budgets are allowed, but only when someone asks: below 5s the
        # frame count drops under H3's trained range.
        assert blank_source("New").segments[0].prompt.storyboard.duration == 5


class TestCopy:
    def test_a_copy_keeps_the_intent_and_drops_the_results(self):
        source = Source(segments=[a_segment()], path=["t1"], remembered={"t1": "t2"})
        copy = source.copy()
        assert copy.id != source.id
        assert copy.path == [] and copy.remembered == {}
        assert copy.segments[0].prompt == source.segments[0].prompt

    def test_a_copied_prompt_is_its_own(self):
        source = Source(segments=[a_segment()])
        copy = source.copy()
        copy.segments[0].prompt.style = "changed"
        assert source.segments[0].prompt.style != "changed"


class TestRevisions:
    def test_a_changed_prompt_files_the_one_it_replaced(self):
        segment = Segment(prompt=VideoPrompt.blank(8))
        filed = segment.revise(VideoPrompt.blank(12), note="Gemini: longer")

        assert filed is not None
        assert filed.note == "Gemini: longer"
        assert filed.prompt.duration == 8
        assert segment.prompt.duration == 12

    def test_an_unchanged_prompt_files_nothing(self):
        # A save that touched only the quality must not bury the edits worth
        # finding under a stack of identical entries.
        segment = Segment(prompt=VideoPrompt.blank(8))
        assert segment.revise(VideoPrompt.blank(8)) is None
        assert segment.revisions == []

    def test_the_filed_prompt_is_a_copy(self):
        # Filing the live object would let the next edit rewrite the past.
        original = VideoPrompt.blank(8)
        segment = Segment(prompt=original)
        filed = segment.revise(VideoPrompt.blank(12))

        original.style = "changed after the fact"
        assert filed.prompt.style == ""

    def test_the_oldest_is_dropped_once_the_limit_is_reached(self):
        segment = Segment(prompt=VideoPrompt.blank(5))
        for duration in range(6, 6 + REVISION_LIMIT + 5):
            segment.revise(VideoPrompt.blank(duration))

        assert len(segment.revisions) == REVISION_LIMIT
        # The five oldest went; the newest history entry is the prompt just
        # superseded.
        assert segment.revisions[-1].prompt.duration == 5 + REVISION_LIMIT + 4

    def test_restoring_puts_a_past_prompt_back(self):
        segment = Segment(prompt=VideoPrompt.blank(8))
        segment.revise(VideoPrompt.blank(12))
        first = segment.revisions[0]

        assert segment.restore(first.id) is first
        assert segment.prompt.duration == 8

    def test_restoring_files_the_current_prompt_so_the_trip_is_round(self):
        segment = Segment(prompt=VideoPrompt.blank(8))
        segment.revise(VideoPrompt.blank(12))
        segment.restore(segment.revisions[0].id)

        # Undoing the undo is another restore, not a special case.
        latest = segment.revisions[-1]
        assert latest.prompt.duration == 12
        assert latest.note.startswith("restored ")
        segment.restore(latest.id)
        assert segment.prompt.duration == 12

    def test_restoring_something_that_is_not_there_is_refused(self):
        assert Segment().restore("rev_nope") is None

    def test_history_survives_a_round_trip(self):
        segment = Segment(prompt=VideoPrompt.blank(8))
        segment.revise(VideoPrompt.blank(12), note="why")
        read = Segment.from_dict(segment.to_dict())

        assert [r.id for r in read.revisions] == [r.id for r in segment.revisions]
        assert read.revisions[0].note == "why"
        assert read.revisions[0].prompt.duration == 8

    def test_a_segment_written_before_history_existed_simply_has_none(self):
        assert Segment.from_dict({"quality": "final"}).revisions == []

    def test_the_wire_shape_counts_the_history_without_carrying_it(self):
        # Every save answers with the whole project; shipping each segment's
        # past on every keystroke would grow the reply nobody asked to read.
        segment = Segment(prompt=VideoPrompt.blank(8))
        segment.revise(VideoPrompt.blank(12))
        wire = segment.to_dict(history=False)

        assert "revisions" not in wire
        assert wire["revision_count"] == 1
        assert segment.to_dict()["revisions"]

    def test_a_source_passes_the_flag_down_to_its_segments(self):
        source = Source(segments=[a_segment()])
        source.segments[0].revise(VideoPrompt.blank(12))

        assert "revisions" not in source.to_dict(history=False)["segments"][0]
        assert source.to_dict()["segments"][0]["revisions"]

    def test_a_duplicate_starts_with_no_past_of_its_own(self):
        # A fresh bin item that opened showing an afternoon of edits it never
        # had would be lying about itself.
        source = Source(segments=[a_segment()])
        source.segments[0].revise(VideoPrompt.blank(12))

        assert source.copy().segments[0].revisions == []


class TestSplittingShotsOff:
    """Some of a segment's shots lifted out as a bin item of their own."""

    def a_scene(self) -> VideoPrompt:
        prompt = VideoPrompt.blank(12)
        prompt.storyboard.beats = [
            Beat(0, 4, "wide", "the door opens"),
            Beat(4, 8, "medium", "she crosses the room"),
            Beat(8, 12, "close", "the clock reads midnight"),
        ]
        prompt.style = "Live-action, anamorphic, film grain."
        prompt.audio = "Rain on the window."
        prompt.cast = ["mem_1"]
        prompt.first_frame = "open.png"
        prompt.last_frame = "close.png"
        return prompt

    def test_only_the_picked_shots_come_across(self):
        made = self.a_scene().for_beats([1, 2])
        assert [b.action for b in made.storyboard.beats] == [
            "she crosses the room", "the clock reads midnight",
        ]

    def test_the_scene_around_them_comes_across_whole(self):
        # Two shots lifted out still happen in the same place, to the same
        # people, in the same light.
        made = self.a_scene().for_beats([1, 2])
        assert made.style == "Live-action, anamorphic, film grain."
        assert made.audio == "Rain on the window."
        assert made.cast == ["mem_1"]

    def test_the_original_is_left_alone(self):
        prompt = self.a_scene()
        made = prompt.for_beats([1])
        made.storyboard.beats[0].action = "Changed."
        made.style = "Changed."
        assert len(prompt.storyboard.beats) == 3
        assert prompt.storyboard.beats[1].action == "she crosses the room"
        assert prompt.style == "Live-action, anamorphic, film grain."

    def test_a_keyframe_survives_only_the_frame_it_was_pointed_at(self):
        tail = self.a_scene().for_beats([1, 2])
        assert tail.first_frame == ""
        assert tail.last_frame == "close.png"

        head = self.a_scene().for_beats([0])
        assert head.first_frame == "open.png"
        assert head.last_frame == ""

    def test_a_split_carries_the_prompt_and_no_takes(self):
        source = Source(name="Hallway", segments=[Segment(prompt=self.a_scene())])
        source.set_path(["r1:a.mp4"])

        made = split_off(source, source.segments[0], [1, 2])
        assert made.id != source.id
        assert made.path == []
        assert len(made.segments) == 1
        assert made.segments[0].prompt.duration == 8

    def test_a_split_renders_the_way_the_segment_it_came_from_does(self):
        segment = Segment(prompt=self.a_scene(), quality="final")
        source = Source(name="Hallway", segments=[segment])
        assert split_off(source, segment, [1]).segments[0].quality == "final"

    def test_a_split_is_named_for_the_shots_it_holds(self):
        source = Source(name="Hallway", segments=[Segment(prompt=self.a_scene())])
        assert split_off(source, source.segments[0], [1, 2]).name == "Hallway (shots 2-3)"
        assert split_off(source, source.segments[0], [1]).name == "Hallway (shot 2)"
        assert split_off(source, source.segments[0], [0, 2]).name == "Hallway (shots 1, 3)"

    def test_a_given_name_wins(self):
        source = Source(name="Hallway", segments=[Segment(prompt=self.a_scene())])
        assert split_off(source, source.segments[0], [1], name="Clock").name == "Clock"

    def test_the_new_source_has_no_past_of_its_own(self):
        segment = Segment(prompt=self.a_scene())
        segment.revise(self.a_scene())
        source = Source(name="Hallway", segments=[segment])
        assert split_off(source, segment, [1]).segments[0].revisions == []


class TestShotLabel:
    def test_counts_the_way_a_storyboard_does(self):
        assert shot_label([0]) == "shot 1"
        assert shot_label([1, 2]) == "shots 2-3"
        assert shot_label([2, 0]) == "shots 1, 3"
        assert shot_label([]) == "no shots"
