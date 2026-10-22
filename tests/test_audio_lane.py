"""Mutes anchored to footage, and sound that sits where its picture does not.

Two ideas are on trial here. A :class:`Mute` is stored in lineage seconds, so the
tests that matter most are the ones where the *edit* moves and the mute has to
stay over the same frames. An :class:`AudioClip` carries an absolute start, so
the tests that matter most are the ones where it lands nowhere near the clip it
came from and both are still heard.
"""

from __future__ import annotations

import pytest

from video_prompter import (
    AudioClip,
    Clip,
    Imported,
    Lifted,
    Mute,
    Project,
    Segment,
    Sound,
    Source,
    Timeline,
    VideoPrompt,
)
from video_prompter import edit, export
from video_prompter.gallery import Take


def a_source(name: str = "One", duration: int = 8, segments: int = 1, **kwargs) -> Source:
    return Source(
        name=name,
        segments=[Segment(prompt=VideoPrompt.blank(duration)) for _ in range(segments)],
        **kwargs,
    )


def a_take(source: Source, seconds: float = 8.0, filename: str = "a.mp4",
           parent: Take | None = None, **kwargs) -> Take:
    return Take(
        prompt_id=kwargs.pop("prompt_id", "run-1"),
        filename=filename,
        shot=source.id,
        seconds=seconds,
        parent_take=parent.id if parent else "",
        **kwargs,
    )


def a_project(*sources: Source, clips: list[Clip] | None = None,
              mutes: list[Mute] | None = None,
              audio: list[AudioClip] | None = None) -> Project:
    made = clips if clips is not None else [Clip(source_id=s.id) for s in sources]
    return Project(
        name="Cut",
        sources=list(sources),
        timeline=Timeline(made, mutes=list(mutes or []), audio=list(audio or [])),
    )


@pytest.fixture
def output_dir(tmp_path):
    directory = tmp_path / "output"
    directory.mkdir()
    return directory


def graph_of(command: list[str]) -> str:
    return command[command.index("-filter_complex") + 1]


def windows_for(project: Project, takes: list[Take], clip: Clip):
    """One clip's mutes as timeline ranges, placed the way the export places them."""
    bin = edit.read(project.sources, takes)
    lengths = edit.lengths(project.sources, project.clips, bin)
    starts = project.timeline.starts_at(lengths)
    return edit.mute_windows(
        clip, project.timeline.mutes, starts[clip.id], lengths.get(clip.id)
    )


class TestAMuteIsAnchoredToTheFootage:
    """The four behaviours that fall out of storing a mute in lineage seconds."""

    def test_it_lands_where_the_footage_is(self):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id)
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=2.0, end=3.0)])
        assert windows_for(project, [take], clip) == [(2.0, 3.0)]

    def test_trimming_the_head_slides_it_because_the_picture_slid(self):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id, source_in=1.5)
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=2.0, end=3.0)])
        # The mute still covers lineage 2s–3s; the clip now starts at lineage
        # 1.5s, so on the ruler that is half a second in.
        assert windows_for(project, [take], clip) == [(0.5, 1.5)]

    def test_a_clip_pushed_later_carries_its_mute_with_it(self):
        first, second = a_source("First"), a_source("Second")
        takes = [a_take(first, filename="a.mp4"), a_take(second, filename="b.mp4")]
        first.set_path([takes[0].id])
        second.set_path([takes[1].id])
        clip = Clip(source_id=second.id)
        project = a_project(first, second, clips=[Clip(source_id=first.id), clip],
                            mutes=[Mute(clip_id=clip.id, start=1.0, end=2.0)])
        # The first clip is eight seconds, so the second starts there.
        assert windows_for(project, takes, clip) == [(9.0, 10.0)]

    def test_speed_rescales_it_without_anyone_converting_anything(self):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id, speed=2.0)
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=2.0, end=4.0)])
        assert windows_for(project, [take], clip) == [(1.0, 2.0)]

    def test_trimming_past_a_mute_hides_it_and_trimming_back_returns_it(self):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id, source_out=2.0)
        mute = Mute(clip_id=clip.id, start=3.0, end=4.0)
        project = a_project(source, clips=[clip], mutes=[mute])
        assert windows_for(project, [take], clip) == []

        # The record was never touched, so re-extending brings it back whole.
        clip.source_out = None
        assert windows_for(project, [take], clip) == [(3.0, 4.0)]
        assert (mute.start, mute.end) == (3.0, 4.0)

    def test_a_mute_hanging_off_the_edge_contributes_only_its_visible_part(self):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id, source_in=1.0, source_out=5.0)
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=0.0, end=2.0)])
        assert windows_for(project, [take], clip) == [(0.0, 1.0)]


class TestMuteBookkeeping:
    def test_a_band_dragged_right_to_left_is_the_same_band(self):
        assert Mute(start=4.0, end=1.0).seconds == 3.0
        assert Mute(start=4.0, end=1.0).start == 1.0

    def test_overlapping_mutes_are_merged_into_one_window(self):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id)
        project = a_project(source, clips=[clip], mutes=[
            Mute(clip_id=clip.id, start=1.0, end=3.0),
            Mute(clip_id=clip.id, start=2.5, end=4.0),
            Mute(clip_id=clip.id, start=6.0, end=7.0),
        ])
        assert windows_for(project, [take], clip) == [(1.0, 4.0), (6.0, 7.0)]

    def test_another_clips_mutes_are_not_this_clips_business(self):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        mine, theirs = Clip(source_id=source.id), Clip(source_id=source.id)
        project = a_project(source, clips=[mine, theirs],
                            mutes=[Mute(clip_id=theirs.id, start=1.0, end=2.0)])
        assert windows_for(project, [take], mine) == []

    def test_deleting_a_clip_takes_its_mutes_and_leaves_the_lane_alone(self):
        clip = Clip(source_id="src_1")
        item = AudioClip(origin=Lifted(source_id="src_1", source_out=2.0))
        timeline = Timeline([clip], mutes=[Mute(clip_id=clip.id, end=1.0)], audio=[item])
        assert timeline.remove(clip.id)
        assert timeline.mutes == []
        # The lane item is sound the edit is using elsewhere; the picture it was
        # lifted from going away is not a reason to throw it out.
        assert timeline.audio == [item]

    def test_forgetting_a_source_takes_everything_that_reads_it(self):
        source, other = a_source("Gone"), a_source("Stays")
        clip = Clip(source_id=source.id)
        project = a_project(
            source, other,
            clips=[clip, Clip(source_id=other.id)],
            mutes=[Mute(clip_id=clip.id, end=1.0)],
            audio=[
                AudioClip(origin=Lifted(source_id=source.id, source_out=2.0)),
                AudioClip(origin=Lifted(source_id=other.id, source_out=2.0)),
            ],
        )
        assert project.remove_source(source.id)
        assert [c.source_id for c in project.clips] == [other.id]
        assert project.timeline.mutes == []
        assert [i.origin.source_id for i in project.timeline.audio] == [other.id]

    def test_splitting_a_clip_divides_a_mute_that_straddles_the_cut(self):
        clip = Clip(source_id="src_1")
        timeline = Timeline([clip], mutes=[Mute(clip_id=clip.id, start=3.0, end=5.0)])
        head, tail = timeline.split(clip.id, 4.0, 8.0)
        by_clip = {m.clip_id: (m.start, m.end) for m in timeline.mutes}
        assert by_clip == {head.id: (3.0, 4.0), tail.id: (4.0, 5.0)}

    def test_a_mute_reaching_past_the_trim_survives_a_split_whole(self):
        clip = Clip(source_id="src_1", source_out=6.0)
        timeline = Timeline([clip], mutes=[Mute(clip_id=clip.id, start=5.0, end=9.0)])
        _, tail = timeline.split(clip.id, 3.0, 8.0)
        assert [(m.start, m.end) for m in timeline.mutes] == [(5.0, 9.0)]
        assert timeline.mutes[0].clip_id == tail.id

    def test_duplicating_a_clip_duplicates_its_silences(self):
        clip = Clip(source_id="src_1")
        timeline = Timeline([clip], mutes=[Mute(clip_id=clip.id, start=1.0, end=2.0)])
        copy = timeline.duplicate(clip.id)
        assert len(timeline.mutes) == 2
        assert timeline.mutes_for(copy.id)[0].start == 1.0
        # Separate records, so editing one does not edit the other.
        timeline.mutes_for(copy.id)[0].end = 3.0
        assert timeline.mutes_for(clip.id)[0].end == 2.0


class TestMutesInTheExport:
    def test_a_mute_in_the_middle_cuts_the_sound_in_two(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id)
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=3.0, end=5.0)])

        sounds = export.resolve_sound(project, [take], output_dir)
        assert [(s.start, s.length) for s in sounds] == [(0.0, 3.0), (5.0, 3.0)]
        # The second half reads from further into the same file, so the hole in
        # the sound sits under the same picture the hole was drawn over.
        assert [(s.source_in, s.source_out) for s in sounds] == [(0.0, 3.0), (5.0, 8.0)]

    def test_only_the_edges_the_mute_made_get_a_fade(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id)
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=3.0, end=5.0)])

        head, tail = export.resolve_sound(project, [take], output_dir)
        assert (head.fade_in, head.fade_out) == (0.0, 0.04)
        assert (tail.fade_in, tail.fade_out) == (0.04, 0.0)

    def test_a_mute_over_the_whole_clip_leaves_nothing(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id)
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=0.0, end=8.0)])
        assert export.resolve_sound(project, [take], output_dir) == []

    def test_a_mute_silences_borrowed_sound_too(self, output_dir):
        """The case that only works because the windows are in timeline seconds.

        The borrowed lineage is a different clock entirely, so a mute expressed
        in the picture's lineage could not be matched against it any other way.
        """
        source = a_source()
        seen = a_take(source, filename="seen.mp4")
        heard = a_take(source, filename="heard.mp4", prompt_id="run-2")
        source.set_path([seen.id])
        clip = Clip(source_id=source.id, audio=Sound(path_override=[heard.id]))
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=2.0, end=8.0)])

        sounds = export.resolve_sound(project, [seen, heard], output_dir)
        assert [s.path.name for s in sounds] == ["heard.mp4"]
        assert (sounds[0].start, sounds[0].length) == (0.0, 2.0)

    def test_the_graph_ramps_a_made_edge(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id)
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=3.0, end=5.0)])

        clips = export.resolve(project, [take], output_dir)
        sounds = export.resolve_sound(project, [take], output_dir)
        graph = graph_of(export.ffmpeg_command(clips, output_dir / "o.mp4", sounds))
        assert "afade=t=out:st=2.96:d=0.04" in graph
        assert "afade=t=in:st=0:d=0.04" in graph

    def test_a_sliver_fades_across_itself_rather_than_past_its_end(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id)
        project = a_project(source, clips=[clip],
                            mutes=[Mute(clip_id=clip.id, start=0.02, end=8.0)])
        head = export.resolve_sound(project, [take], output_dir)[0]
        assert head.fade_out == 0.01


class TestTheAudioLane:
    def test_a_lifted_item_plays_where_it_was_put_not_where_it_came_from(
        self, output_dir
    ):
        first, second = a_source("First"), a_source("Second")
        takes = [a_take(first, filename="a.mp4"), a_take(second, filename="b.mp4")]
        first.set_path([takes[0].id])
        second.set_path([takes[1].id])
        # Sound from the second shot, pulled two seconds early — under the first.
        lifted = AudioClip(
            origin=Lifted(source_id=second.id, source_in=0.0, source_out=2.0),
            start=6.0,
        )
        project = a_project(first, second, audio=[lifted])

        lane = [s for s in export.resolve_sound(project, takes, output_dir) if s.lane]
        assert [(s.path.name, s.start, s.length) for s in lane] == [("b.mp4", 6.0, 2.0)]

    def test_the_lifted_sound_and_the_picture_under_it_are_both_heard(
        self, output_dir
    ):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, audio=[
            AudioClip(origin=Lifted(source_id=source.id, source_out=2.0), start=1.0),
        ])

        clips = export.resolve(project, [take], output_dir)
        sounds = export.resolve_sound(project, [take], output_dir)
        graph = graph_of(export.ffmpeg_command(clips, output_dir / "o.mp4", sounds))
        # Two audio inputs placed independently and mixed, not concatenated.
        assert "amix=inputs=2" in graph
        assert "adelay=1000:all=1" in graph

    def test_a_lane_item_carries_its_own_ramps(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, audio=[
            AudioClip(origin=Lifted(source_id=source.id, source_out=2.0),
                      fade_in=0.5, fade_out=0.25),
        ])
        item = export.resolve_lane(project, [take], output_dir)[0]
        assert (item.fade_in, item.fade_out) == (0.5, 0.25)

    def test_a_mute_does_not_reach_the_sound_moved_on_top_of_it(self, output_dir):
        """A J-cut is a hole and a fill, and the hole must not swallow the fill."""
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id)
        project = a_project(
            source,
            clips=[clip],
            mutes=[Mute(clip_id=clip.id, start=0.0, end=4.0)],
            audio=[AudioClip(origin=Lifted(source_id=source.id, source_in=6.0,
                                           source_out=8.0), start=1.0)],
        )
        sounds = export.resolve_sound(project, [take], output_dir)
        assert [(s.lane, s.start, s.length) for s in sounds] == [
            (False, 4.0, 4.0),
            (True, 1.0, 2.0),
        ]

    def test_sound_dragged_past_the_last_frame_is_trimmed_to_it(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, audio=[
            AudioClip(origin=Lifted(source_id=source.id, source_out=4.0), start=6.0),
        ])
        lane = export.resolve_lane(project, [take], output_dir)
        # The edit is eight seconds; the item wanted to run to ten.
        assert [(s.start, s.length) for s in lane] == [(6.0, 2.0)]

    def test_with_no_picture_there_is_nowhere_for_it_to_go(self, output_dir):
        """The export is as long as the picture, so no picture is no room.

        Unreachable through the export route, which refuses an edit with no
        enabled clips before it ever asks about sound — pinned here so the rule
        is written down rather than discovered.
        """
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, clips=[], audio=[
            AudioClip(origin=Lifted(source_id=source.id, source_out=2.0)),
        ])
        assert export.resolve_lane(project, [take], output_dir) == []

    def test_a_disabled_item_is_not_in_the_mix(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, audio=[
            AudioClip(origin=Lifted(source_id=source.id, source_out=2.0),
                      enabled=False),
        ])
        assert export.resolve_lane(project, [take], output_dir) == []

    def test_an_item_reading_a_lineage_that_is_gone_is_a_silence_not_a_crash(
        self, output_dir
    ):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, audio=[
            AudioClip(origin=Lifted(source_id="src_nothing", source_out=2.0)),
        ])
        assert export.resolve_lane(project, [take], output_dir) == []

    def test_items_are_read_in_ruler_order_however_they_were_added(self):
        late = AudioClip(origin=Lifted(source_out=1.0), start=9.0)
        early = AudioClip(origin=Lifted(source_out=1.0), start=1.0)
        assert [i.id for i in Timeline([], audio=[late, early]).active_audio()] == \
            [early.id, late.id]


class TestImportedFiles:
    def test_edl_keeps_imported_and_lifted_lane_events_at_their_positions(self, output_dir, tmp_path):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, audio=[
            AudioClip(origin=Imported(file="bed.wav", seconds=20, file_in=5, file_out=7),
                      start=3, gain_db=-6, fade_in=0.25, fade_out=0.5),
            AudioClip(origin=Lifted(source_id=source.id, source_in=1, source_out=2), start=1),
        ])
        clips = export.resolve(project, [take], output_dir)
        sounds = export.resolve_sound(project, [take], output_dir, tmp_path / "audio")
        text = export.edl(clips, "Cut", sounds)
        assert "* FROM CLIP NAME: bed.wav" in text
        events = [line.split() for line in text.splitlines() if line[:3].isdigit()]
        audio = [event for event in events if event[2] == "A"]
        assert len(audio) == 2
        assert audio[0][4:] == ["00:00:01:00", "00:00:02:00", "00:00:01:00", "00:00:02:00"]
        assert audio[1][4:] == ["00:00:05:00", "00:00:07:00", "00:00:03:00", "00:00:05:00"]
        assert "gain -6 dB" in text and "fade in 0.25s" in text
        assert "separate tracks" in text

    def test_a_file_needs_somewhere_to_be_before_it_can_be_mixed(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, audio=[
            AudioClip(origin=Imported(file="bed.wav", seconds=30.0), start=2.0),
        ])
        # No audio folder given, so there is no honest path to hand ffmpeg.
        assert export.resolve_lane(project, [take], output_dir) == []

    def test_with_a_folder_it_becomes_one_placed_piece(self, output_dir, tmp_path):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, audio=[
            AudioClip(origin=Imported(file="bed.wav", seconds=30.0, file_in=5.0,
                                      file_out=9.0),
                      start=2.0, gain_db=-6.0),
        ])
        lane = export.resolve_lane(project, [take], output_dir, tmp_path / "audio")
        assert len(lane) == 1
        assert lane[0].path == tmp_path / "audio" / "bed.wav"
        assert (lane[0].start, lane[0].length) == (2.0, 4.0)
        assert (lane[0].source_in, lane[0].source_out) == (5.0, 9.0)
        assert lane[0].gain_db == -6.0

    def test_no_out_point_means_the_rest_of_the_file(self):
        assert Imported(file="bed.wav", seconds=30.0, file_in=2.0).width() == 28.0

    def test_an_out_point_past_the_end_of_the_file_is_the_end_of_the_file(self):
        assert Imported(file="bed.wav", seconds=10.0, file_out=99.0).width() == 10.0


class TestStorage:
    def test_a_project_saved_before_the_lane_existed_loads_with_an_empty_one(self):
        timeline = Timeline.from_dict({"clips": [{"source_id": "src_1"}]})
        assert timeline.mutes == []
        assert timeline.audio == []

    def test_mutes_and_the_lane_round_trip(self):
        before = Timeline(
            [Clip(source_id="src_1")],
            mutes=[Mute(clip_id="clip_1", start=1.0, end=2.0)],
            audio=[
                AudioClip(origin=Lifted(source_id="src_1", source_out=2.0), start=3.0),
                AudioClip(origin=Imported(file="bed.wav", seconds=30.0), start=4.0),
            ],
        )
        after = Timeline.from_dict(before.to_dict())
        assert after.to_dict() == before.to_dict()
        assert isinstance(after.audio[0].origin, Lifted)
        assert isinstance(after.audio[1].origin, Imported)

    def test_an_origin_this_build_does_not_know_opens_as_an_empty_lineage(self):
        item = AudioClip.from_dict({"origin": {"kind": "something_new"}})
        assert isinstance(item.origin, Lifted)
        assert item.timeline_seconds() == 0.0

    def test_a_copied_item_shares_nothing_with_its_original(self):
        item = AudioClip(origin=Lifted(source_id="src_1", path_override=["take_1"]))
        copy = item.copy()
        copy.origin.path_override.append("take_2")
        assert item.origin.path_override == ["take_1"]
        assert copy.id != item.id
