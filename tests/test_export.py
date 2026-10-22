from __future__ import annotations

import pytest

from video_prompter import Clip, Project, Segment, Source, Timeline, VideoPrompt
from video_prompter import export
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


def chosen(source: Source, *takes: Take) -> Source:
    source.set_path([take.id for take in takes])
    return source


def a_project(*sources: Source, clips: list[Clip] | None = None) -> Project:
    made = clips if clips is not None else [Clip(source_id=s.id) for s in sources]
    return Project(name="Cut", sources=list(sources), timeline=Timeline(made))


@pytest.fixture
def output_dir(tmp_path):
    directory = tmp_path / "output"
    directory.mkdir()
    return directory


class TestResolve:
    def test_it_pairs_clips_with_the_takes_on_their_path(self, output_dir):
        source = a_source()
        take = a_take(source)
        resolved = export.resolve(a_project(chosen(source, take)), [take], output_dir)
        assert [piece.take.filename for piece in resolved] == ["a.mp4"]

    def test_a_clip_with_nothing_to_play_stops_the_export(self, output_dir):
        with pytest.raises(export.ExportError, match="Rooftop"):
            export.resolve(a_project(a_source("Rooftop")), [], output_dir)

    def test_a_disabled_clip_is_not_required_to_have_a_take(self, output_dir):
        ready, waiting = a_source("Ready"), a_source("Waiting")
        take = a_take(ready)
        project = a_project(
            chosen(ready, take), waiting,
            clips=[Clip(source_id=ready.id), Clip(source_id=waiting.id, enabled=False)],
        )
        assert len(export.resolve(project, [take], output_dir)) == 1

    def test_an_empty_timeline_is_refused(self, output_dir):
        with pytest.raises(export.ExportError):
            export.resolve(Project(name="Cut"), [], output_dir)

    def test_starts_follow_the_real_take_lengths(self, output_dir):
        first, second = a_source("A"), a_source("B")
        takes = [a_take(first, 15.083), a_take(second, 8.0, prompt_id="run-2")]
        project = a_project(chosen(first, takes[0]), chosen(second, takes[1]))
        resolved = export.resolve(project, takes, output_dir)
        assert [piece.start for piece in resolved] == [0.0, 15.083]

    def test_the_take_subfolder_is_part_of_the_path(self, output_dir):
        source = a_source()
        take = a_take(source, subfolder="video")
        piece = export.resolve(a_project(chosen(source, take)), [take], output_dir)[0]
        assert piece.path == output_dir / "video" / "a.mp4"


class TestClipsThatCrossAJoin:
    """A clip is a range of lineage time, so one clip can span several takes."""

    def _chain(self):
        source = a_source(segments=2)
        root = a_take(source, 5.167, "root.mp4")
        child = a_take(source, 4.25, "child.mp4", parent=root, prompt_id="run-2")
        return chosen(source, root, child), [root, child]

    def test_a_whole_chain_resolves_to_one_piece_per_generation(self, output_dir):
        source, takes = self._chain()
        resolved = export.resolve(a_project(source), takes, output_dir)
        assert [piece.take.filename for piece in resolved] == ["root.mp4", "child.mp4"]

    def test_the_pieces_run_back_to_back_on_the_timeline(self, output_dir):
        source, takes = self._chain()
        resolved = export.resolve(a_project(source), takes, output_dir)
        assert [piece.start for piece in resolved] == [0.0, 5.167]

    def test_a_trim_across_the_join_cuts_both_files(self, output_dir):
        source, takes = self._chain()
        project = a_project(
            source, clips=[Clip(source_id=source.id, source_in=4.0, source_out=7.0)]
        )
        resolved = export.resolve(project, takes, output_dir)
        # 1.167s off the end of the root, then 1.833s into its continuation.
        assert [(p.source_in, p.source_out) for p in resolved] == [
            (4.0, 5.167), (0.0, 1.833)
        ]

    def test_a_trim_inside_one_generation_stays_one_piece(self, output_dir):
        source, takes = self._chain()
        project = a_project(
            source, clips=[Clip(source_id=source.id, source_in=1.0, source_out=3.0)]
        )
        assert len(export.resolve(project, takes, output_dir)) == 1

    def test_a_pinned_clip_plays_its_own_branch(self, output_dir):
        source, takes = self._chain()
        project = a_project(
            source,
            clips=[Clip(source_id=source.id, path_override=[takes[0].id])],
        )
        resolved = export.resolve(project, takes, output_dir)
        assert [piece.take.filename for piece in resolved] == ["root.mp4"]


class TestFfmpegCommand:
    def build(self, output_dir, tmp_path, clip_kwargs=None, **source_kwargs):
        source = a_source(**source_kwargs)
        take = a_take(source)
        project = a_project(
            chosen(source, take),
            clips=[Clip(source_id=source.id, **(clip_kwargs or {}))],
        )
        resolved = export.resolve(project, [take], output_dir)
        return export.ffmpeg_command(resolved, tmp_path / "out.mp4")

    def test_it_trims_at_the_in_and_out_points(self, output_dir, tmp_path):
        command = self.build(
            output_dir, tmp_path, clip_kwargs={"source_in": 1.5, "source_out": 6.0}
        )
        graph = command[command.index("-filter_complex") + 1]
        assert "trim=start=1.5:end=6.0" in graph

    def test_an_open_out_point_runs_to_the_end_of_source(self, output_dir, tmp_path):
        command = self.build(output_dir, tmp_path)
        graph = command[command.index("-filter_complex") + 1]
        assert "trim=start=0.0:end=8.0" in graph

    def test_speed_becomes_a_setpts_step(self, output_dir, tmp_path):
        command = self.build(output_dir, tmp_path, clip_kwargs={"speed": 2.0})
        graph = command[command.index("-filter_complex") + 1]
        assert "setpts=PTS/2.0" in graph

    def test_normal_speed_adds_no_retiming_step(self, output_dir, tmp_path):
        command = self.build(output_dir, tmp_path)
        graph = command[command.index("-filter_complex") + 1]
        assert "PTS/" not in graph

    def test_it_concats_every_input(self, output_dir, tmp_path):
        sources = [a_source("A"), a_source("B"), a_source("C")]
        takes = [
            a_take(source, prompt_id=f"run-{i}", filename=f"{i}.mp4")
            for i, source in enumerate(sources)
        ]
        for source, take in zip(sources, takes):
            chosen(source, take)

        command = export.ffmpeg_command(
            export.resolve(a_project(*sources), takes, output_dir), tmp_path / "out.mp4"
        )
        graph = command[command.index("-filter_complex") + 1]
        assert "concat=n=3:v=1:a=0" in graph
        assert command.count("-i") == 3

    def test_the_frame_rate_is_pinned_to_the_models(self, output_dir, tmp_path):
        command = self.build(output_dir, tmp_path)
        assert command[command.index("-r") + 1] == "24"


class TestEdl:
    def edl_of(self, output_dir, **clip_kwargs):
        source = a_source()
        take = a_take(source)
        project = a_project(
            chosen(source, take), clips=[Clip(source_id=source.id, **clip_kwargs)]
        )
        return export.edl(export.resolve(project, [take], output_dir), "Cut")

    def test_it_carries_the_title(self, output_dir):
        assert self.edl_of(output_dir).startswith("TITLE: Cut")

    def test_source_timecodes_come_from_the_trim(self, output_dir):
        # 1.5s at 24fps is 00:00:01:12.
        assert "00:00:01:12" in self.edl_of(output_dir, source_in=1.5, source_out=6.0)

    def test_the_record_side_starts_at_zero(self, output_dir):
        assert "00:00:00:00" in self.edl_of(output_dir, source_in=1.5)

    def test_a_speed_change_emits_an_m2(self, output_dir):
        assert "M2" in self.edl_of(output_dir, speed=0.5)

    def test_normal_speed_emits_no_m2(self, output_dir):
        assert "M2" not in self.edl_of(output_dir)


class TestClipList:
    def test_it_names_the_source_and_the_file(self, output_dir):
        source = a_source("Rooftop")
        take = a_take(source)
        text = export.clip_list(
            export.resolve(a_project(chosen(source, take)), [take], output_dir), "Cut"
        )
        assert "Rooftop" in text
        assert "a.mp4" in text
        assert "1 pieces, 8.0s" in text

    def test_a_clip_label_wins_over_the_source_name(self, output_dir):
        source = a_source("Rooftop")
        take = a_take(source)
        project = a_project(
            chosen(source, take),
            clips=[Clip(source_id=source.id, label="The leap")],
        )
        text = export.clip_list(export.resolve(project, [take], output_dir), "Cut")
        assert "The leap" in text


class TestShortSources:
    def resolved(self, output_dir, name: str = "One", **clip_kwargs):
        source = a_source(name)
        take = a_take(source)
        project = a_project(
            chosen(source, take), clips=[Clip(source_id=source.id, **clip_kwargs)]
        )
        return export.resolve(project, [take], output_dir)

    def test_a_file_that_is_not_there_is_left_to_the_existence_check(self, output_dir):
        assert export.short_sources(self.resolved(output_dir)) == []

    def test_a_short_file_is_named_with_both_lengths(self, output_dir, monkeypatch):
        (output_dir / "a.mp4").write_bytes(b"stand-in")
        monkeypatch.setattr(export, "probe_duration", lambda path: 5.06)
        found = export.short_sources(self.resolved(output_dir, name="Rooftop"))
        assert len(found) == 1
        assert "Rooftop" in found[0] and "5.06" in found[0]

    def test_a_long_enough_file_is_quiet(self, output_dir, monkeypatch):
        (output_dir / "a.mp4").write_bytes(b"stand-in")
        monkeypatch.setattr(export, "probe_duration", lambda path: 8.0)
        assert export.short_sources(self.resolved(output_dir)) == []

    def test_an_unprobeable_file_is_not_guessed_at(self, output_dir, monkeypatch):
        (output_dir / "a.mp4").write_bytes(b"stand-in")
        monkeypatch.setattr(export, "probe_duration", lambda path: None)
        assert export.short_sources(self.resolved(output_dir)) == []

    def test_a_trim_inside_a_short_file_is_fine(self, output_dir, monkeypatch):
        (output_dir / "a.mp4").write_bytes(b"stand-in")
        monkeypatch.setattr(export, "probe_duration", lambda path: 5.06)
        assert export.short_sources(self.resolved(output_dir, source_out=4.0)) == []


class TestTimecode:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "00:00:00:00"),
            (1.5, "00:00:01:12"),
            (60, "00:01:00:00"),
            (3661, "01:01:01:00"),
            (-5, "00:00:00:00"),
        ],
    )
    def test_it_matches_the_edl_form(self, seconds, expected):
        assert export._timecode(seconds) == expected
