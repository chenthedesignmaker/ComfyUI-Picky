"""Folding a pre-sources chain back into one source.

The rule these tests exist to hold: **the edit does not move.** Every case below
checks the timeline's duration across the fold as well as its shape, because a
fold that quietly lengthens or shortens the cut would be worse than not folding
at all.
"""

from __future__ import annotations

from video_prompter import Clip, Segment, Source, Timeline, VideoPrompt, edit, fold
from video_prompter.gallery import Take
from video_prompter.projects import Project


def take(name: str, *, source: str, parent: str = "", seconds: float = 5.0,
         rating: int = 0) -> Take:
    return Take(
        prompt_id=f"p_{name}",
        filename=f"{name}.mp4",
        shot=source,
        seconds=seconds,
        parent_take=f"p_{parent}:{parent}.mp4" if parent else "",
        rating=rating,
        created=f"2026-01-01T00:00:{ord(name[-1]) % 60:02d}Z",
    )


def a_source(source_id: str, *, path: list[str], segments: int = 1,
             name: str = "") -> Source:
    return Source(
        id=source_id,
        name=name or source_id,
        segments=[Segment(prompt=VideoPrompt.blank(8)) for _ in range(segments)],
        path=list(path),
    )


def a_project(sources: list[Source], clips: list[Clip]) -> Project:
    return Project(name="Fold", sources=list(sources), timeline=Timeline(clips=list(clips)))


def a_chain(*, lengths: tuple[float, ...] = (5.0, 4.0)) -> tuple[Project, list[Take]]:
    """One chain the renderer recorded, split across one source per link.

    This is what every project made before the Source model landed looks like on
    disk after the v1 migration: each shot one source, one segment, one take.
    """
    takes: list[Take] = []
    sources: list[Source] = []
    clips: list[Clip] = []
    for index, seconds in enumerate(lengths):
        name = chr(ord("a") + index)
        parent = chr(ord("a") + index - 1) if index else ""
        source_id = f"src_{name}"
        made = take(name, source=source_id, parent=parent, seconds=seconds)
        takes.append(made)
        sources.append(a_source(source_id, path=[made.id], name=f"Aviator {index + 1}"))
        clips.append(Clip(id=f"clip_{name}", source_id=source_id))
    return a_project(sources, clips), takes


def run(project: Project, takes: list[Take]) -> fold.Result:
    return fold.fold(project, edit.read(project.sources, takes))


def duration(project: Project, takes: list[Take]) -> float:
    bin = edit.read(project.sources, takes)
    return project.timeline.duration(edit.lengths(project.sources, project.clips, bin))


class TestFindingTheRun:
    def test_a_two_link_chain_is_one_run(self):
        project, takes = a_chain()
        plan = fold.plan(project, edit.read(project.sources, takes))
        assert [r.source_ids for r in plan.runs] == [["src_a", "src_b"]]

    def test_the_run_is_ordered_by_the_take_record_not_by_bin_order(self):
        project, takes = a_chain(lengths=(5.0, 4.0, 3.0))
        project.sources.reverse()
        plan = fold.plan(project, edit.read(project.sources, takes))
        assert [r.source_ids for r in plan.runs] == [["src_a", "src_b", "src_c"]]

    def test_sources_with_no_link_between_them_are_left_alone(self):
        takes = [take("a", source="src_a"), take("b", source="src_b")]
        project = a_project(
            [a_source("src_a", path=[takes[0].id]), a_source("src_b", path=[takes[1].id])],
            [Clip(source_id="src_a"), Clip(source_id="src_b")],
        )
        assert not fold.plan(project, edit.read(project.sources, takes)).runs

    def test_a_fork_used_by_two_sources_is_not_folded(self):
        # Both continue the same take. Picking one arm is the author's call.
        takes = [
            take("a", source="src_a"),
            take("b", source="src_b", parent="a"),
            take("c", source="src_c", parent="a"),
        ]
        project = a_project(
            [a_source("src_a", path=[takes[0].id]),
             a_source("src_b", path=[takes[1].id]),
             a_source("src_c", path=[takes[2].id])],
            [Clip(source_id="src_a"), Clip(source_id="src_b"), Clip(source_id="src_c")],
        )
        assert not fold.plan(project, edit.read(project.sources, takes)).runs

    def test_two_sources_standing_on_the_same_take_are_not_folded(self):
        # A duplicated bin row: which one does the continuation belong to?
        takes = [take("a", source="src_a"), take("b", source="src_c", parent="a")]
        project = a_project(
            [a_source("src_a", path=[takes[0].id]),
             a_source("src_b", path=[takes[0].id]),
             a_source("src_c", path=[takes[1].id])],
            [Clip(source_id="src_a"), Clip(source_id="src_b"), Clip(source_id="src_c")],
        )
        assert not fold.plan(project, edit.read(project.sources, takes)).runs

    def test_a_source_planning_more_than_it_rendered_mid_run_is_skipped(self):
        # Its unrendered segment may well *be* the next source's generation, and
        # nothing on disk says whether it is.
        project, takes = a_chain(lengths=(5.0, 4.0, 3.0))
        project.sources[1].segments.append(Segment(prompt=VideoPrompt.blank(6)))
        plan = fold.plan(project, edit.read(project.sources, takes))
        assert not plan.runs
        assert "has rendered 1" in plan.skipped[0].reason

    def test_an_unrendered_plan_on_the_last_source_is_fine(self):
        project, takes = a_chain()
        project.sources[-1].segments.append(Segment(prompt=VideoPrompt.blank(6)))
        assert fold.plan(project, edit.read(project.sources, takes)).runs


class TestWhatFoldingLeaves:
    def test_ten_bin_rows_become_one(self):
        project, takes = a_chain(lengths=tuple([4.25] * 10))
        run(project, takes)
        assert len(project.sources) == 1
        assert project.sources[0].id == "src_a"

    def test_the_segments_are_concatenated_in_chain_order(self):
        project, takes = a_chain(lengths=(5.0, 4.0, 3.0))
        run(project, takes)
        assert project.sources[0].depth == 3

    def test_the_path_is_stored_rather_than_left_to_the_default(self):
        # The default follows the best-rated child, which is not necessarily the
        # take each of these rows was picked onto.
        project, takes = a_chain()
        run(project, takes)
        assert project.sources[0].path == [takes[0].id, takes[1].id]

    def test_the_absorbed_name_is_kept_on_the_generation_it_named(self):
        project, takes = a_chain()
        run(project, takes)
        assert project.sources[0].segments[1].notes == "Aviator 2"

    def test_the_takes_of_an_absorbed_source_are_handed_over(self):
        project, takes = a_chain()
        assert run(project, takes).reassign == {"src_b": "src_a"}

    def test_folding_twice_does_nothing_the_second_time(self):
        project, takes = a_chain(lengths=(5.0, 4.0, 3.0))
        assert run(project, takes).folds
        assert not run(project, takes).folds

    def test_a_source_with_nothing_on_the_timeline_still_folds(self):
        project, takes = a_chain()
        project.timeline.clips = [c for c in project.clips if c.source_id == "src_a"]
        run(project, takes)
        assert len(project.sources) == 1
        assert project.sources[0].depth == 2


class TestTheEditDoesNotMove:
    def test_untrimmed_clips_in_order_become_one_clip(self):
        project, takes = a_chain(lengths=(5.0, 4.0, 3.0))
        before = duration(project, takes)
        run(project, takes)
        assert len(project.clips) == 1
        assert (project.clips[0].source_in, project.clips[0].source_out) == (0.0, 12.0)
        assert duration(project, takes) == before

    def test_the_joined_clip_is_pinned_shut_rather_than_left_open(self):
        # An open out point means "to the end of the lineage", and after a fold
        # that end is the end of the whole chain.
        project, takes = a_chain()
        run(project, takes)
        assert project.clips[0].source_out == 9.0

    def test_a_trimmed_clip_keeps_its_trim_and_does_not_join(self):
        project, takes = a_chain()
        project.clips[1].source_in = 1.0
        before = duration(project, takes)
        run(project, takes)
        assert len(project.clips) == 2
        assert (project.clips[1].source_in, project.clips[1].source_out) == (6.0, 9.0)
        assert duration(project, takes) == before

    def test_clips_used_out_of_order_are_rebased_but_not_joined(self):
        project, takes = a_chain()
        project.timeline.clips.reverse()
        before = duration(project, takes)
        run(project, takes)
        assert [(c.source_in, c.source_out) for c in project.clips] == [(5.0, 9.0), (0.0, 5.0)]
        assert duration(project, takes) == before

    def test_a_hidden_clip_stays_hidden_and_separate(self):
        project, takes = a_chain()
        project.clips[1].enabled = False
        before = duration(project, takes)
        run(project, takes)
        assert len(project.clips) == 2
        assert project.clips[1].enabled is False
        assert duration(project, takes) == before

    def test_a_clip_with_its_own_label_is_not_folded_away(self):
        project, takes = a_chain()
        project.clips[1].label = "the landing"
        run(project, takes)
        assert [c.label for c in project.clips] == ["", "the landing"]

    def test_a_pinned_clip_gains_the_generations_now_above_it(self):
        # An override is a whole path from depth 0, so folding deepens it.
        project, takes = a_chain()
        project.clips[1].path_override = [takes[1].id]
        run(project, takes)
        assert project.clips[1].path_override == [takes[0].id, takes[1].id]

    def test_the_same_source_used_twice_keeps_both_uses(self):
        project, takes = a_chain()
        project.timeline.add(Clip(id="clip_again", source_id="src_b"))
        before = duration(project, takes)
        run(project, takes)
        assert len(project.clips) == 2
        assert duration(project, takes) == before

    def test_a_planned_tail_on_the_last_source_still_reaches_the_clip(self):
        project, takes = a_chain()
        project.sources[-1].segments.append(Segment(prompt=VideoPrompt.blank(6)))
        before = duration(project, takes)
        run(project, takes)
        assert duration(project, takes) == before
