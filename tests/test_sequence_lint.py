from __future__ import annotations

import pytest

from video_prompter import Clip, Project, Segment, Source, Timeline, VideoPrompt
from video_prompter.compose import compose
from video_prompter.gallery import Take
from video_prompter.lint import Severity
from video_prompter.sequence_lint import lint_sequence


def a_source(name: str = "One", duration: int = 8, segments: int = 1, **kwargs) -> Source:
    made = []
    for _ in range(segments):
        prompt = VideoPrompt.blank(duration)
        prompt.style = "A cinematic look."
        made.append(Segment(prompt=prompt))
    return Source(name=name, segments=made, **kwargs)


def a_take(source: Source, depth: int = 0, seconds: float = 8.0, **kwargs) -> Take:
    return Take(
        prompt_id=kwargs.pop("prompt_id", "run-1"),
        filename=kwargs.pop("filename", f"take{depth}.mp4"),
        shot=source.id,
        seconds=seconds,
        prompt_text=kwargs.pop("prompt_text", compose(source.segments[depth].prompt)),
        **kwargs,
    )


def a_project(*sources: Source, clips: list[Clip] | None = None) -> Project:
    """One clip per source unless the test wants something else."""
    made = clips if clips is not None else [Clip(source_id=s.id) for s in sources]
    return Project(name="Test", sources=list(sources), timeline=Timeline(made))


def rules(project: Project, takes: list[Take] | None = None) -> list[str]:
    return [finding.rule for finding in lint_sequence(project, takes or [])]


class TestPaths:
    def test_a_path_naming_nothing_is_an_error(self):
        source = a_source(path=["run-9:gone.mp4"])
        assert "clip.path_broken" in rules(a_project(source))

    def test_a_path_whose_links_do_not_hold_up_is_an_error(self):
        # The check that replaces every heuristic: step two must record step one
        # as its parent. Nothing is inferred and nothing is quietly repaired.
        source = a_source(segments=2)
        root = a_take(source, 0)
        stray = a_take(source, 1, filename="stray.mp4")  # no parent_take
        source.path = [root.id, stray.id]
        assert "clip.path_broken" in rules(a_project(source), [root, stray])

    def test_a_sound_path_is_quiet(self):
        source = a_source()
        take = a_take(source)
        source.path = [take.id]
        assert "clip.path_broken" not in rules(a_project(source), [take])

    def test_a_take_from_an_older_prompt_is_flagged(self):
        source = a_source()
        take = a_take(source, prompt_text="something else entirely")
        source.path = [take.id]
        assert "source.take_stale" in rules(a_project(source), [take])

    def test_editing_the_prompt_after_a_render_goes_stale(self):
        source = a_source()
        take = a_take(source)
        source.path = [take.id]
        assert "source.take_stale" not in rules(a_project(source), [take])

        source.segments[0].prompt.style = "A completely different look."
        assert "source.take_stale" in rules(a_project(source), [take])

    def test_takes_with_nothing_chosen_warn(self):
        source = a_source()
        take = a_take(source)
        # An unchosen source falls back to a default path, so say it is a guess.
        assert "source.no_path" in rules(a_project(source), [take])

    def test_no_takes_at_all_is_only_information(self):
        found = lint_sequence(a_project(a_source()), [])
        no_takes = next(f for f in found if f.rule == "source.no_takes")
        assert no_takes.severity is Severity.INFO

    def test_a_discarded_take_does_not_count_as_one(self):
        source = a_source()
        assert "source.no_takes" in rules(a_project(source), [a_take(source, discarded=True)])

    def test_a_chain_only_half_rendered_is_noted(self):
        source = a_source(segments=2)
        take = a_take(source)
        source.path = [take.id]
        assert "source.path_short" in rules(a_project(source), [take])


class TestTrim:
    def test_starting_past_the_end_of_the_lineage_is_an_error(self):
        source = a_source()
        project = a_project(source, clips=[Clip(source_id=source.id, source_in=20.0)])
        assert "clip.trim_past_source" in rules(project)

    def test_an_out_point_past_the_lineage_warns(self):
        source = a_source()
        project = a_project(source, clips=[Clip(source_id=source.id, source_out=20.0)])
        clipped = next(
            f for f in lint_sequence(project, []) if f.rule == "clip.trim_past_source"
        )
        assert clipped.severity is Severity.WARNING

    def test_the_chosen_take_decides_how_much_source_there_is(self):
        # Planned 8s, but the take really ran 15.083s, so a 12s out is fine.
        source = a_source()
        take = a_take(source, seconds=15.083)
        source.path = [take.id]
        project = a_project(source, clips=[Clip(source_id=source.id, source_out=12.0)])
        assert "clip.trim_past_source" not in rules(project, [take])

    def test_a_trim_that_leaves_nothing_warns(self):
        source = a_source()
        project = a_project(
            source, clips=[Clip(source_id=source.id, source_in=7.9, source_out=7.95)]
        )
        assert "clip.too_short" in rules(project)

    def test_a_disabled_clip_is_not_nagged_about_length(self):
        source = a_source()
        project = a_project(
            source,
            clips=[Clip(source_id=source.id, source_in=7.9, source_out=7.95, enabled=False)],
        )
        assert "clip.too_short" not in rules(project)


class TestBudget:
    @pytest.mark.parametrize("duration", [0, -1, 20])
    def test_a_budget_outside_the_model_range_is_an_error(self, duration):
        assert "segment.duration_invalid" in rules(a_project(a_source(duration=duration)))

    @pytest.mark.parametrize("duration", [1, 2, 5])
    def test_short_budgets_are_allowed(self, duration):
        found = rules(a_project(a_source(duration=duration)))
        assert "segment.duration_invalid" not in found


class TestTimeline:
    def test_an_empty_timeline_says_so(self):
        assert "sequence.empty" in rules(a_project())

    def test_a_clip_pointing_at_a_deleted_source_is_an_error(self):
        assert "clip.source_missing" in rules(
            a_project(clips=[Clip(source_id="src_ghost")])
        )

    def test_a_source_nothing_uses_is_only_information(self):
        source = a_source()
        found = lint_sequence(a_project(source, clips=[]), [])
        unused = next(f for f in found if f.rule == "source.unused")
        assert unused.severity is Severity.INFO

    def test_findings_are_worst_first(self):
        found = lint_sequence(a_project(a_source(path=["run-9:gone.mp4"])), [])
        assert found[0].severity is Severity.ERROR

    def test_each_finding_names_its_clip(self):
        found = lint_sequence(a_project(a_source("Rooftop"), a_source("Landing")), [])
        assert any("Landing" in finding.where for finding in found)

    def test_a_clip_label_wins_over_the_source_name(self):
        source = a_source("Rooftop")
        project = a_project(source, clips=[Clip(source_id=source.id, label="The leap")])
        assert any("The leap" in finding.where for finding in lint_sequence(project, []))
