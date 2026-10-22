"""Checks against the local node's own arithmetic.

These mirror `align_frame_count`, `adapt_canvas` and the schema defaults in
ComfyUI's `comfy_extras/nodes_minimax_h3.py`. If a future ComfyUI release
changes them, these fail first.
"""

from __future__ import annotations

import pytest

from video_prompter import h3_spec as spec


class TestFrameGrid:
    @pytest.mark.parametrize("frames", [5, 22, 39, 124, 192, 362])
    def test_grid_members_are_left_alone(self, frames):
        assert spec.align_frames(frames) == frames

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [(1, 5), (6, 22), (23, 39), (120, 124), (191, 192), (360, 362)],
    )
    def test_everything_else_snaps_up_never_down(self, requested, expected):
        assert spec.align_frames(requested) == expected

    def test_every_aligned_count_satisfies_the_congruence(self):
        for n in range(1, 400):
            assert spec.align_frames(n) % spec.FRAME_GRID == spec.FRAME_OFFSET

    def test_alignment_never_shortens_a_request(self):
        for n in range(spec.MIN_FRAMES, 400):
            assert spec.align_frames(n) >= n


class TestDuration:
    def test_the_trained_range_maps_onto_five_to_fifteen_seconds(self):
        frames = spec.frames_for_seconds
        assert frames(spec.TRAINED_MIN_DURATION_S) == spec.TRAINED_MIN_FRAMES
        assert frames(spec.TRAINED_MAX_DURATION_S) == spec.TRAINED_MAX_FRAMES

    def test_the_allowed_floor_sits_below_the_trained_one(self):
        # Short clips are an editorial choice, so they are allowed and warned
        # about rather than refused.
        assert spec.MIN_DURATION_S < spec.TRAINED_MIN_DURATION_S
        assert spec.is_valid_duration(spec.MIN_DURATION_S)
        frames = spec.frames_for_seconds(spec.MIN_DURATION_S)
        assert frames >= spec.MIN_FRAMES
        assert not spec.is_trained_length(frames)

    def test_four_seconds_falls_under_the_trained_minimum(self):
        assert not spec.is_trained_length(spec.frames_for_seconds(4))

    def test_eight_seconds_is_the_only_exact_whole_second_budget(self):
        exact = [
            duration
            for duration in range(spec.MIN_DURATION_S, spec.MAX_DURATION_S + 1)
            if spec.seconds_for_frames(spec.frames_for_seconds(duration)) == duration
        ]
        assert exact == [8]

    def test_a_ten_second_budget_really_runs_ten_and_an_eighth(self):
        assert spec.frames_for_seconds(10) == 243
        assert spec.seconds_for_frames(243) == 10.125

    @pytest.mark.parametrize("duration", [1, 2, 4, 5, 8, 15])
    def test_the_supported_range_validates(self, duration):
        assert spec.is_valid_duration(duration)

    @pytest.mark.parametrize("duration", [0, -1, 16, 7.5, True, "8", None])
    def test_everything_else_does_not(self, duration):
        assert not spec.is_valid_duration(duration)


class TestCanvas:
    def test_sixteen_by_nine_gives_the_node_defaults(self):
        assert spec.canvas_for_ratio("16:9") == (spec.DEFAULT_WIDTH, spec.DEFAULT_HEIGHT)

    @pytest.mark.parametrize("ratio", spec.ASPECT_RATIOS)
    def test_every_supported_ratio_lands_on_the_grid_and_under_the_cap(self, ratio):
        width, height = spec.canvas_for_ratio(ratio)
        assert width % spec.CANVAS_MULTIPLE == 0
        assert height % spec.CANVAS_MULTIPLE == 0
        assert width * height <= spec.MAX_PIXELS

    def test_portrait_and_landscape_are_mirrors(self):
        assert spec.canvas_for_ratio("9:16") == spec.canvas_for_ratio("16:9")[::-1]

    def test_a_square_sits_at_the_base_short_edge(self):
        assert spec.canvas_for_ratio("1:1") == (spec.BASE_SHORT_EDGE, spec.BASE_SHORT_EDGE)

    def test_an_oversized_canvas_is_scaled_under_the_cap(self):
        width, height = spec.adapt_canvas(3840, 2160)
        assert width * height <= spec.MAX_PIXELS

    @pytest.mark.parametrize("ratio", ["", "16", "16:0", "a:b", "-1:2"])
    def test_a_malformed_ratio_is_rejected(self, ratio):
        with pytest.raises(ValueError):
            spec.canvas_for_ratio(ratio)


class TestEditorVocabulariesMatchTheSpec:
    """The editor hardcodes these lists; drift turns a picker into a typo machine.

    They are mirrored rather than fetched because they never change at runtime —
    but a mirror nobody checks is how the task-type picker shipped offering two
    values H3 has never heard of, which the linter then flagged on every save.
    """

    @staticmethod
    def _array(name: str) -> list[str]:
        import re
        from pathlib import Path

        source = (Path(__file__).parent.parent / "js" / "standalone" / "main.js").read_text(
            encoding="utf-8"
        )
        match = re.search(rf"const {name} = \[(.*?)\];", source, re.S)
        assert match, f"{name} is not declared in main.js"
        return re.findall(r'"([^"]+)"', match.group(1))

    def test_reference_task_types(self):
        assert self._array("REF_TASK_TYPES") == list(spec.REF_TASK_TYPES)

    def test_visual_retention_markers(self):
        assert self._array("VISUAL_RETENTION") == list(spec.VISUAL_RETENTION)

    def test_audio_retention_markers(self):
        assert self._array("AUDIO_RETENTION") == list(spec.AUDIO_RETENTION)

    def test_transitions(self):
        assert self._array("TRANSITIONS") == list(spec.TRANSITIONS)

    def test_camera_motions(self):
        assert self._array("CAMERA_MOTIONS") == list(spec.CAMERA_MOTIONS)

    def test_camera_amplitudes(self):
        assert self._array("CAMERA_AMPLITUDES") == list(spec.CAMERA_AMPLITUDES)

    def test_camera_speeds(self):
        assert self._array("CAMERA_SPEEDS") == list(spec.CAMERA_SPEEDS)

    def test_every_motion_has_a_verb_form(self):
        import re
        from pathlib import Path

        source = (Path(__file__).parent.parent / "js" / "standalone" / "main.js").read_text(
            encoding="utf-8"
        )
        block = re.search(r"const CAMERA_VERBS = \{(.*?)\n\};", source, re.S)
        assert block
        keyed = set(re.findall(r'"([^"]+)":', block.group(1)))
        assert keyed == set(spec.CAMERA_MOTIONS)
