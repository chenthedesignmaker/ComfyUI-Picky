"""What a clip is heard as: muting, borrowing another take, and the mix.

The interesting cases are the ones where picture and sound disagree, because
that is the only time the two resolvers can drift apart.
"""

from __future__ import annotations

import pytest

from video_prompter import Clip, Project, Segment, Sound, Source, Timeline, VideoPrompt
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


def a_project(*sources: Source, clips: list[Clip] | None = None) -> Project:
    made = clips if clips is not None else [Clip(source_id=s.id) for s in sources]
    return Project(name="Cut", sources=list(sources), timeline=Timeline(made))


def two_takes(source: Source) -> tuple[Take, Take]:
    """One take whose picture is used, one whose sound is."""
    seen = a_take(source, filename="seen.mp4")
    heard = a_take(source, filename="heard.mp4", prompt_id="run-2")
    source.set_path([seen.id])
    return seen, heard


@pytest.fixture
def output_dir(tmp_path):
    directory = tmp_path / "output"
    directory.mkdir()
    return directory


def graph_of(command: list[str]) -> str:
    return command[command.index("-filter_complex") + 1]


class TestSoundOnAClip:
    def test_a_fresh_clip_plays_its_own_sound(self):
        assert Clip().audio == Sound()
        assert not Clip().audio.borrowed

    def test_a_clip_written_before_sound_existed_loads_unchanged(self):
        clip = Clip.from_dict({"source_id": "src_1", "source_in": 2.0})
        assert clip.audio == Sound()

    def test_naming_a_take_or_a_source_is_what_borrowing_means(self):
        assert Sound(path_override=["take_2"]).borrowed
        assert Sound(source_id="src_9").borrowed
        # Muting is a different question from where the sound comes from.
        assert not Sound(muted=True).borrowed

    def test_a_split_gives_each_half_its_own_sound(self):
        clip = Clip(source_id="src_1", audio=Sound(path_override=["take_2"]))
        head, tail = Timeline([clip]).split(clip.id, 4.0, 8.0)
        tail.audio.muted = True
        tail.audio.path_override.append("take_3")
        assert head.audio.muted is False
        assert head.audio.path_override == ["take_2"]

    def test_a_duplicate_gives_the_copy_its_own_sound(self):
        clip = Clip(source_id="src_1", audio=Sound(gain_db=-6.0))
        copy = Timeline([clip]).duplicate(clip.id)
        copy.audio.gain_db = 0.0
        assert clip.audio.gain_db == -6.0

    def test_gain_outside_the_useful_range_is_a_typo(self):
        assert Sound.from_dict({"gain_db": 400}).gain_db == 6.0
        assert Sound.from_dict({"gain_db": -900}).gain_db == -60.0
        assert Sound.from_dict({"gain_db": "loud"}).gain_db == 0.0

    def test_it_survives_a_round_trip(self):
        sound = Sound(muted=True, gain_db=-3.0, source_id="src_2",
                      path_override=["take_1"], offset=0.5)
        assert Sound.from_dict(sound.to_dict()) == sound


class TestResolvingSound:
    def test_its_own_sound_resolves_exactly_like_its_picture(self):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id, source_in=2.0, source_out=6.0)
        bin = edit.read([source], [take])
        assert edit.resolve_audio(clip, source, bin) == edit.resolve_clip(clip, source, bin)

    def test_a_muted_clip_resolves_to_nothing(self):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clip = Clip(source_id=source.id, audio=Sound(muted=True))
        assert edit.resolve_audio(clip, source, edit.read([source], [take])) == []

    def test_a_borrowed_take_is_read_over_the_picture_s_window(self):
        source = a_source()
        seen, heard = two_takes(source)
        clip = Clip(source_id=source.id, source_in=2.0, source_out=5.0,
                    audio=Sound(source_id=source.id, path_override=[heard.id]))
        pieces = edit.resolve_audio(clip, source, edit.read([source], [seen, heard]))
        assert [piece.take_id for piece in pieces] == [heard.id]
        assert (pieces[0].source_from, pieces[0].source_to) == (2.0, 5.0)

    def test_an_offset_slides_the_window_without_changing_its_width(self):
        source = a_source()
        seen, heard = two_takes(source)
        clip = Clip(source_id=source.id, source_in=1.0, source_out=4.0,
                    audio=Sound(path_override=[heard.id], offset=0.5))
        pieces = edit.resolve_audio(clip, source, edit.read([source], [seen, heard]))
        assert (pieces[0].source_from, pieces[0].source_to) == (1.5, 4.5)

    def test_a_borrowed_lineage_that_runs_out_just_ends(self):
        source = a_source()
        seen = a_take(source, seconds=8.0, filename="seen.mp4")
        heard = a_take(source, seconds=3.0, filename="heard.mp4", prompt_id="run-2")
        source.set_path([seen.id])
        clip = Clip(source_id=source.id, audio=Sound(path_override=[heard.id]))
        pieces = edit.resolve_audio(clip, source, edit.read([source], [seen, heard]))
        # Three seconds of sound under eight of picture. Short, not broken.
        assert sum(piece.seconds for piece in pieces) == 3.0


class TestTheMix:
    def one(self, output_dir, **audio):
        """One clip, two takes, sound configured however the test wants."""
        source = a_source()
        seen, heard = two_takes(source)
        if audio.pop("borrow", False):
            audio["path_override"] = [heard.id]
        clip = Clip(source_id=source.id, audio=Sound(**audio))
        project = a_project(source, clips=[clip])
        takes = [seen, heard]
        return (
            export.resolve(project, takes, output_dir),
            export.resolve_sound(project, takes, output_dir),
        )

    def test_a_plain_clip_carries_its_own_sound_into_the_mix(self, output_dir):
        _, sounds = self.one(output_dir)
        assert [sound.path.name for sound in sounds] == ["seen.mp4"]
        assert not sounds[0].borrowed

    def test_a_muted_clip_contributes_nothing(self, output_dir):
        clips, sounds = self.one(output_dir, muted=True)
        assert sounds == []
        assert export.summary(clips, sounds)["silent"] == ["One"]

    def test_a_borrowed_clip_is_heard_from_the_other_take_s_file(self, output_dir):
        _, sounds = self.one(output_dir, borrow=True)
        assert [sound.path.name for sound in sounds] == ["heard.mp4"]
        assert sounds[0].borrowed

    def test_the_graph_mixes_rather_than_concatenates(self, output_dir):
        clips, sounds = self.one(output_dir)
        command = export.ffmpeg_command(clips, output_dir / "out.mp4", sounds)
        assert "amix=inputs=1:normalize=0" in graph_of(command)
        assert "[aout]" in command
        assert "aac" in command

    def test_a_silent_edit_maps_no_audio_at_all(self, output_dir):
        clips, sounds = self.one(output_dir, muted=True)
        command = export.ffmpeg_command(clips, output_dir / "out.mp4", sounds)
        assert "[aout]" not in command
        assert "-c:a" not in command

    def test_export_without_a_mix_is_what_it_always_was(self, output_dir):
        clips, _ = self.one(output_dir)
        assert export.ffmpeg_command(clips, output_dir / "out.mp4") == \
            export.ffmpeg_command(clips, output_dir / "out.mp4", [])

    def test_every_piece_is_delayed_to_its_place_on_the_timeline(self, output_dir):
        first, second = a_source("A"), a_source("B")
        takes = [a_take(first, 6.0), a_take(second, 4.0, prompt_id="run-2", filename="b.mp4")]
        first.set_path([takes[0].id])
        second.set_path([takes[1].id])
        project = a_project(first, second)
        clips = export.resolve(project, takes, output_dir)
        sounds = export.resolve_sound(project, takes, output_dir)
        graph = graph_of(export.ffmpeg_command(clips, output_dir / "out.mp4", sounds))
        assert "adelay=0:all=1" in graph
        assert "adelay=6000:all=1" in graph

    def test_audio_inputs_come_after_every_video_input(self, output_dir):
        clips, sounds = self.one(output_dir)
        # One video input at index 0, so the sound is input 1.
        assert "[1:a]atrim=" in graph_of(
            export.ffmpeg_command(clips, output_dir / "out.mp4", sounds)
        )

    def test_a_re_timed_clip_re_times_its_sound_too(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, clips=[Clip(source_id=source.id, speed=3.0)])
        clips = export.resolve(project, [take], output_dir)
        sounds = export.resolve_sound(project, [take], output_dir)
        # One atempo only stretches to 2x, so 3x has to be a chain of two.
        assert "atempo=2.0,atempo=1.5" in graph_of(
            export.ffmpeg_command(clips, output_dir / "o.mp4", sounds)
        )

    def test_gain_is_left_out_of_the_graph_when_it_is_unity(self, output_dir):
        clips, sounds = self.one(output_dir)
        assert "volume=" not in graph_of(
            export.ffmpeg_command(clips, output_dir / "o.mp4", sounds)
        )

    def test_gain_appears_in_decibels_when_it_is_not(self, output_dir):
        clips, sounds = self.one(output_dir, gain_db=-6.0)
        assert "volume=-6.0dB" in graph_of(
            export.ffmpeg_command(clips, output_dir / "o.mp4", sounds)
        )


class TestAtempoChain:
    @pytest.mark.parametrize("speed", [0.1, 0.5, 0.75, 1.5, 2.0, 3.0, 10.0])
    def test_the_chain_multiplies_back_to_the_speed_asked_for(self, speed):
        product = 1.0
        for part in export._atempo_chain(speed):
            product *= float(part.split("=")[1])
        assert product == pytest.approx(speed, rel=1e-4)

    def test_unity_needs_no_filter_at_all(self):
        assert export._atempo_chain(1.0) == []

    def test_every_link_is_inside_what_atempo_accepts(self):
        for part in export._atempo_chain(0.1) + export._atempo_chain(10.0):
            assert export.ATEMPO_MIN <= float(part.split("=")[1]) <= export.ATEMPO_MAX


class TestUnreadableSound:
    def test_a_file_that_is_not_there_is_dropped_rather_than_mapped(self, output_dir):
        source = a_source()
        take = a_take(source, filename="gone.mp4")
        source.set_path([take.id])
        sounds = export.resolve_sound(a_project(source), [take], output_dir)
        keep, dropped = export.playable_sounds(sounds)
        assert keep == []
        assert dropped == ["One"]


class TestListings:
    def resolved(self, output_dir, project, takes):
        return (
            export.resolve(project, takes, output_dir),
            export.resolve_sound(project, takes, output_dir),
        )

    def test_the_edl_marks_a_muted_clip_as_picture_only(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        project = a_project(source, clips=[Clip(source_id=source.id, audio=Sound(muted=True))])
        clips, sounds = self.resolved(output_dir, project, [take])
        text = export.edl(clips, "Cut", sounds)
        assert "  V     C" in text
        assert "  B     C" not in text

    def test_the_edl_marks_an_ordinary_clip_as_picture_and_sound(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clips, sounds = self.resolved(output_dir, a_project(source), [take])
        assert "  B     C" in export.edl(clips, "Cut", sounds)

    def test_a_borrowed_track_gets_an_event_of_its_own(self, output_dir):
        source = a_source()
        seen, heard = two_takes(source)
        clip = Clip(source_id=source.id, audio=Sound(path_override=[heard.id]))
        clips, sounds = self.resolved(
            output_dir, a_project(source, clips=[clip]), [seen, heard]
        )
        text = export.edl(clips, "Cut", sounds)
        assert "  A     C" in text
        assert "heard.mp4" in text
        assert "(sound only)" in text

    def test_the_clip_list_says_where_borrowed_sound_came_from(self, output_dir):
        source = a_source()
        seen, heard = two_takes(source)
        clip = Clip(source_id=source.id, audio=Sound(path_override=[heard.id]))
        clips, sounds = self.resolved(
            output_dir, a_project(source, clips=[clip]), [seen, heard]
        )
        assert "sound: from heard.mp4" in export.clip_list(clips, "Cut", sounds)

    def test_the_clip_list_stays_quiet_about_ordinary_sound(self, output_dir):
        source = a_source()
        take = a_take(source)
        source.set_path([take.id])
        clips, sounds = self.resolved(output_dir, a_project(source), [take])
        assert "sound:" not in export.clip_list(clips, "Cut", sounds)
