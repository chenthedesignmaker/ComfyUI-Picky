from __future__ import annotations

import pytest

from video_prompter import (
    AudioCue,
    Line,
    Reference,
    ReferenceSet,
    Severity,
    Beat,
    Storyboard,
    VideoPrompt,
    errors,
    lint,
)


def rules(prompt: VideoPrompt) -> set[str]:
    return {finding.rule for finding in lint(prompt)}


class TestCleanPrompt:
    def test_a_well_formed_prompt_raises_no_errors(self, clean_prompt):
        assert errors(lint(clean_prompt)) == []

    def test_findings_come_back_worst_first(self):
        prompt = VideoPrompt(storyboard=Storyboard(3, [Beat(0, 1.5), Beat(1, 2.5)]))
        severities = [finding.severity for finding in lint(prompt)]
        assert severities == sorted(severities, reverse=True)


class TestSettings:
    @pytest.mark.parametrize("duration", [0, -1, 16, 100])
    def test_duration_the_node_cannot_generate_is_an_error(self, clean_prompt, duration):
        clean_prompt.storyboard.duration = duration
        assert "duration.invalid" in rules(clean_prompt)

    def test_a_short_clip_is_allowed_but_flagged_as_untrained(self, clean_prompt):
        # 2s aligns to 56 frames, under the trained minimum of 124. It renders;
        # the warning is there so the quality cliff is a choice, not a surprise.
        clean_prompt.storyboard = clean_prompt.storyboard.refit(2)
        found = rules(clean_prompt)
        assert "duration.invalid" not in found
        assert "duration.untrained" in found

    def test_fractional_duration_is_rejected(self, clean_prompt):
        clean_prompt.storyboard.duration = 7.5
        assert "duration.invalid" in rules(clean_prompt)

    @pytest.mark.parametrize("duration", [1, 2, 5, 10, 15])
    def test_the_supported_range_passes(self, clean_prompt, duration):
        clean_prompt.storyboard = clean_prompt.storyboard.refit(duration)
        assert "duration.invalid" not in rules(clean_prompt)

    def test_the_real_clip_length_is_reported_when_it_snaps(self, clean_prompt):
        clean_prompt.storyboard = clean_prompt.storyboard.refit(10)
        snapped = next(f for f in lint(clean_prompt) if f.rule == "duration.snapped")
        assert "243 frames" in snapped.message
        assert "10.125s" in snapped.message

    def test_eight_seconds_is_the_one_budget_that_does_not_snap(self, clean_prompt):
        # 8s = 192 frames exactly; every other whole second in 5-15 runs long.
        clean_prompt.storyboard = clean_prompt.storyboard.refit(8)
        assert "duration.snapped" not in rules(clean_prompt)

    @pytest.mark.parametrize("duration", [5, 6, 7, 9, 10, 11, 12, 13, 14, 15])
    def test_every_other_budget_reports_its_real_length(self, clean_prompt, duration):
        clean_prompt.storyboard = clean_prompt.storyboard.refit(duration)
        assert "duration.snapped" in rules(clean_prompt)

    def test_an_off_grid_canvas_is_an_error(self, clean_prompt):
        clean_prompt.width = 1000
        assert "canvas.off_grid" in rules(clean_prompt)

    def test_an_oversized_canvas_warns_with_what_the_node_would_use(self, clean_prompt):
        clean_prompt.width, clean_prompt.height = 1920, 1088
        finding = next(f for f in lint(clean_prompt) if f.rule == "canvas.too_large")
        assert "1344x768" in finding.message

    def test_the_default_canvas_is_clean(self, clean_prompt):
        assert not {"canvas.off_grid", "canvas.too_large", "canvas.too_small"} & rules(
            clean_prompt
        )

    def test_set_ratio_produces_a_canvas_the_linter_accepts(self, clean_prompt):
        for ratio in ("21:9", "16:9", "1:1", "9:16"):
            clean_prompt.set_ratio(ratio)
            assert not {"canvas.off_grid", "canvas.too_large"} & rules(clean_prompt), ratio


class TestStoryboard:
    def test_overlap_is_an_error_and_names_both_shots(self):
        prompt = VideoPrompt(storyboard=Storyboard(5, [Beat(0, 1.5, "a", "x"), Beat(1, 2.5, "b", "y")]))
        overlap = next(f for f in lint(prompt) if f.rule == "storyboard.overlap")
        assert overlap.severity is Severity.ERROR
        assert "Shots 1 and 2" in overlap.message
        assert "0.5s" in overlap.message

    def test_gap_is_a_warning(self, clean_prompt):
        clean_prompt.storyboard.beats = [Beat(0, 2, "a", "x"), Beat(5, 10, "b", "y")]
        gap = next(f for f in lint(clean_prompt) if f.rule == "storyboard.gap")
        assert gap.severity is Severity.WARNING
        assert gap.where == "[2s-5s]"

    def test_running_past_the_budget_is_an_error(self, clean_prompt):
        clean_prompt.storyboard.beats = [Beat(0, 14, "a", "x")]
        assert "storyboard.overrun" in rules(clean_prompt)

    def test_no_beats_warns_about_stretching_one_idea(self, clean_prompt):
        clean_prompt.storyboard.beats = []
        assert "storyboard.empty" in rules(clean_prompt)

    def test_make_contiguous_clears_the_overlap_finding(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(5, [Beat(0, 1.5, "a", "x"), Beat(1, 2.5, "b", "y")]),
            style="a look",
        )
        prompt.storyboard = prompt.storyboard.make_contiguous()
        assert "storyboard.overlap" not in rules(prompt)
        assert "storyboard.gap" not in rules(prompt)


class TestBeats:
    def test_a_reversed_span_is_an_error(self, clean_prompt):
        clean_prompt.storyboard.beats = [Beat(4, 2, "a", "x")]
        assert "beat.invalid_span" in rules(clean_prompt)

    def test_a_flash_frame_warns(self, clean_prompt):
        clean_prompt.storyboard.beats = [Beat(0, 0.3, "a", "x"), Beat(0.3, 10, "b", "y")]
        assert "beat.too_short" in rules(clean_prompt)

    def test_a_shot_with_no_action_warns(self, clean_prompt):
        clean_prompt.storyboard.beats = [Beat(0, 10, "wide", "")]
        assert "beat.no_action" in rules(clean_prompt)

    def test_too_many_cuts_per_second_warns(self):
        beats = [Beat(i * 0.5, (i + 1) * 0.5, "a", "x") for i in range(8)]
        prompt = VideoPrompt(storyboard=Storyboard(5, beats))
        assert "beat.density" in rules(prompt)

    def test_a_roomy_cut_does_not_warn(self, clean_prompt):
        assert "beat.density" not in rules(clean_prompt)


class TestDialogue:
    def test_a_line_with_no_speaker_warns(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [Line(text="Who is there?")]
        assert "dialogue.no_speaker" in rules(clean_prompt)

    def test_a_named_speaker_with_no_words_warns(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [Line(speakers=["ghost"])]
        finding = next(f for f in lint(clean_prompt) if f.rule == "dialogue.empty")
        assert finding.severity is Severity.WARNING

    def test_a_line_with_only_trimmings_warns(self, clean_prompt):
        """describe / verb / delivery compose to nothing without the words."""
        clean_prompt.storyboard.beats[0].lines = [
            Line(describe="the driver", verb="shouts", delivery="over the engine")
        ]
        finding = next(f for f in lint(clean_prompt) if f.rule == "dialogue.empty")
        assert finding.severity is Severity.WARNING

    def test_an_untouched_blank_line_is_only_info(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [Line()]
        finding = next(f for f in lint(clean_prompt) if f.rule == "dialogue.empty")
        assert finding.severity is Severity.INFO

    def test_a_seam_tagged_on_one_side_only_warns(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [
            Line(speakers=["m"], text="I still remember", continuity="into")
        ]
        assert "dialogue.seam_unpaired" in rules(clean_prompt)

    def test_a_seam_tagged_on_both_sides_is_clean(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [
            Line(speakers=["m"], text="I still remember", continuity="into")
        ]
        clean_prompt.storyboard.beats[1].lines = [
            Line(speakers=["m"], text="every turn of it.", continuity="from")
        ]
        assert "dialogue.seam_unpaired" not in rules(clean_prompt)

    def test_a_well_formed_line_is_clean(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [Line(speakers=["w"], text="Hello.")]
        assert not {"dialogue.no_speaker", "dialogue.empty"} & rules(clean_prompt)


class TestAudio:
    def test_a_silent_prompt_is_flagged_as_a_wasted_track(self, clean_prompt):
        clean_prompt.audio = ""
        clean_prompt.storyboard.cues = []
        assert "audio.missing" in rules(clean_prompt)

    def test_a_cue_past_the_budget_warns(self, clean_prompt):
        clean_prompt.storyboard.cues = [AudioCue(14, description="late hit")]
        assert "audio.cue_out_of_range" in rules(clean_prompt)

    def test_a_cue_inside_the_budget_is_fine(self, clean_prompt):
        clean_prompt.storyboard.cues = [AudioCue(4, description="hit")]
        assert "audio.cue_out_of_range" not in rules(clean_prompt)


class TestBlocks:
    def test_a_missing_style_contract_warns(self, clean_prompt):
        clean_prompt.style = ""
        assert "style.missing" in rules(clean_prompt)

    def test_screen_text_no_longer_conflicts_with_negatives(self, clean_prompt):
        # Negatives are not composed, so on-screen text cannot contradict them.
        clean_prompt.screen_text = ["MIDNIGHT LINE"]
        clean_prompt.negatives = ["no text, subtitles, logos or watermarks"]
        assert "text.conflict" not in rules(clean_prompt)

    def test_a_diegetic_source_in_the_score_is_flagged(self, clean_prompt):
        clean_prompt.music = "a song playing from the car radio"
        finding = next(f for f in lint(clean_prompt) if f.rule == "music.diegetic")
        assert finding.severity is Severity.INFO

    def test_a_clean_score_is_not_flagged(self, clean_prompt):
        assert "music.diegetic" not in rules(clean_prompt)


class TestReferences:
    def test_more_than_nine_images_is_an_error(self, clean_prompt):
        clean_prompt.references = ReferenceSet(
            [Reference("image", f"{i}.png", "mood") for i in range(10)]
        )
        assert "references.too_many" in rules(clean_prompt)

    def test_a_soundtrack_without_its_video_is_an_error(self, clean_prompt):
        clean_prompt.references = ReferenceSet(
            [Reference("video_audio", "track.wav", "the soundtrack of video 1")]
        )
        assert "references.orphan_soundtrack" in rules(clean_prompt)

    def test_a_soundtrack_paired_with_its_video_is_allowed(self, clean_prompt):
        clean_prompt.references = ReferenceSet(
            [
                Reference("video", "clip.mp4", "the camera move"),
                Reference("video_audio", "track.wav", "the soundtrack of video 1"),
            ]
        )
        assert "references.orphan_soundtrack" not in rules(clean_prompt)

    def test_a_reference_never_named_in_the_prompt_warns(self, clean_prompt):
        clean_prompt.references = ReferenceSet([Reference("image", "a.png")])
        assert "references.unused" in rules(clean_prompt)

    def test_a_reference_named_in_the_prompt_body_is_fine(self, clean_prompt):
        clean_prompt.references = ReferenceSet([Reference("image", "a.png")])
        clean_prompt.overview = "the character from <Picture 1> crosses the plaza"
        assert "references.unused" not in rules(clean_prompt)

    def test_an_unknown_task_type_warns(self, clean_prompt):
        clean_prompt.references = ReferenceSet([Reference("image", "a.png", "the hero")])
        clean_prompt.ref_task = ["reference generation", "make it cool"]
        assert "ref.task_invalid" in rules(clean_prompt)

    def test_a_valid_task_type_is_fine(self, clean_prompt):
        clean_prompt.references = ReferenceSet([Reference("image", "a.png", "the hero")])
        clean_prompt.ref_task = ["video continuation"]
        assert "ref.task_invalid" not in rules(clean_prompt)

    def test_a_retention_marker_off_vocab_warns(self, clean_prompt):
        clean_prompt.references = ReferenceSet(
            [Reference("image", "a.png", "the hero", retention="mostly_kept")]
        )
        assert "ref.retention_invalid" in rules(clean_prompt)

    def test_an_audio_marker_from_the_visual_set_warns(self, clean_prompt):
        # fully_preserved is a visual marker; audio uses fully_copy/reference/...
        clean_prompt.references = ReferenceSet(
            [Reference("audio", "v.wav", "the voice", retention="fully_preserved")]
        )
        assert "ref.retention_invalid" in rules(clean_prompt)

    def test_a_ref2v_shot_with_no_summary_warns(self, clean_prompt):
        clean_prompt.references = ReferenceSet([Reference("image", "a.png", "the hero")])
        clean_prompt.ref_summary = ""
        assert "ref.summary_missing" in rules(clean_prompt)

    def test_the_warning_names_the_line_it_would_compose_to(self, clean_prompt):
        clean_prompt.references = ReferenceSet(
            [Reference("image", "a.png", "the hero"), Reference("image", "b.png", "the set")]
        )
        clean_prompt.ref_summary = ""
        message = next(
            f.message for f in lint(clean_prompt) if f.rule == "ref.summary_missing"
        )
        assert "<Picture 1>, <Picture 2>" in message

    def test_a_written_summary_says_nothing(self, clean_prompt):
        clean_prompt.references = ReferenceSet([Reference("image", "a.png", "the hero")])
        clean_prompt.ref_summary = "Hold the face from <Picture 1> as she crosses the room."
        assert "ref.summary_missing" not in rules(clean_prompt)

    def test_whitespace_is_not_a_summary(self, clean_prompt):
        clean_prompt.references = ReferenceSet([Reference("image", "a.png", "the hero")])
        clean_prompt.ref_summary = '  \n\t  '
        assert "ref.summary_missing" in rules(clean_prompt)

    def test_a_shot_with_no_references_is_not_asked_for_one(self, clean_prompt):
        clean_prompt.ref_summary = ""
        assert "ref.summary_missing" not in rules(clean_prompt)


class TestLength:
    def test_past_the_api_ceiling_is_an_error(self, clean_prompt):
        clean_prompt.overview = "x" * 7_100
        assert "prompt.too_long" in rules(clean_prompt)

    def test_past_the_recommended_length_warns(self, clean_prompt):
        clean_prompt.overview = "x" * 5_500
        assert "prompt.long" in rules(clean_prompt)

    def test_a_normal_prompt_says_nothing_about_length(self, clean_prompt):
        assert not {"prompt.long", "prompt.too_long"} & rules(clean_prompt)


class TestCuts:
    """A cut has to bring something the shot before it did not have."""

    def test_reframing_the_same_thing_is_not_a_cut(self, clean_prompt):
        clean_prompt.storyboard = Storyboard(
            duration=10,
            beats=[
                Beat(0, 5, "a medium shot", "a baker kneads dough on a floured counter"),
                Beat(5, 10, "a close-up", "the baker kneads the dough"),
            ],
        )
        assert "beat.weak_cut" in rules(clean_prompt)

    def test_a_new_subject_makes_it_a_real_cut(self, clean_prompt):
        clean_prompt.storyboard = Storyboard(
            duration=10,
            beats=[
                Beat(0, 5, "a medium shot", "a baker kneads dough on a floured counter"),
                Beat(5, 10, "a close-up", "a customer pushes the shop door open"),
            ],
        )
        assert "beat.weak_cut" not in rules(clean_prompt)

    def test_the_opening_shot_is_never_a_weak_cut(self, clean_prompt):
        clean_prompt.storyboard = Storyboard(
            duration=10, beats=[Beat(0, 10, "a wide shot", "the city wakes")]
        )
        assert "beat.weak_cut" not in rules(clean_prompt)

    def test_an_empty_shot_is_left_to_beat_no_action(self, clean_prompt):
        clean_prompt.storyboard = Storyboard(
            duration=10,
            beats=[Beat(0, 5, "a medium shot", "a baker kneads dough"), Beat(5, 10)],
        )
        found = rules(clean_prompt)
        assert "beat.no_action" in found
        assert "beat.weak_cut" not in found


class TestAesthetics:
    """Quality tokens are a diffusion habit; H3 conditions on behaviour."""

    @pytest.mark.parametrize(
        "text",
        [
            "shot in 4k, highly detailed",
            "a masterpiece of neon light",
            "cinematic quality, award-winning",
            "photorealistic, don't look AI",
        ],
    )
    def test_quality_language_warns(self, clean_prompt, text):
        clean_prompt.style = text
        assert "prompt.aesthetic" in rules(clean_prompt)

    def test_a_behavioural_prompt_says_nothing(self, clean_prompt):
        assert "prompt.aesthetic" not in rules(clean_prompt)

    def test_a_character_may_call_something_a_masterpiece(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [
            Line(speakers=["critic"], text="It is a masterpiece.")
        ]
        assert "prompt.aesthetic" not in rules(clean_prompt)


class TestSpeechFits:
    """Words are spoken in time, and a beat only has so much of it."""

    LONG = (
        "I have been waiting here for a very long time and I do not think "
        "anybody is coming to help either of us tonight at all"
    )

    def test_more_words_than_the_beat_can_hold_warns(self, clean_prompt):
        clean_prompt.storyboard.beats[2].lines = [
            Line(speakers=["her"], text=self.LONG)
        ]
        assert "dialogue.overrun" in rules(clean_prompt)

    def test_speech_that_fills_the_beat_is_flagged_as_tight(self, clean_prompt):
        # 8 words is ~3.2s at 150 wpm, against a 3s... beat 2 runs 4s-7s.
        clean_prompt.storyboard.beats[1].lines = [
            Line(speakers=["her"], text="One two three four five six seven")
        ]
        assert "dialogue.tight" in rules(clean_prompt)

    def test_a_line_that_fits_says_nothing(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [
            Line(speakers=["her"], text="It is starting.")
        ]
        assert not {"dialogue.overrun", "dialogue.tight"} & rules(clean_prompt)

    def test_a_line_that_runs_across_the_cut_is_exempt(self, clean_prompt):
        clean_prompt.storyboard.beats[2].lines = [
            Line(speakers=["her"], text=self.LONG, continuity="into")
        ]
        assert "dialogue.overrun" not in rules(clean_prompt)

    def test_a_cut_off_line_is_exempt(self, clean_prompt):
        clean_prompt.storyboard.beats[2].lines = [
            Line(speakers=["her"], text=self.LONG, cutoff=True)
        ]
        assert "dialogue.overrun" not in rules(clean_prompt)


class TestAudioCues:
    """A cue is our own extension, so the linter keeps it to what it is for."""

    def test_a_cue_pinned_to_a_moment_passes_quietly(self, clean_prompt):
        clean_prompt.storyboard.cues = [AudioCue(3.5, description="a door slams")]
        assert "audio.cue_covers_clip" not in rules(clean_prompt)
        assert "audio.cue_is_dialogue" not in rules(clean_prompt)

    def test_a_cue_spanning_the_whole_clip_is_the_bed_restated(self, clean_prompt):
        clean_prompt.storyboard.cues = [AudioCue(0, 10, "rain and distant traffic")]
        assert "audio.cue_covers_clip" in rules(clean_prompt)

    def test_narration_parked_in_a_cue_is_flagged(self, clean_prompt):
        clean_prompt.storyboard.cues = [
            AudioCue(2, 7, "the narrator says: Dawn breaks over the savanna")
        ]
        assert "audio.cue_is_dialogue" in rules(clean_prompt)

    def test_a_sound_that_merely_reads_like_speech_is_left_alone(self, clean_prompt):
        # Narrow on purpose: plenty of real sounds borrow the vocabulary.
        clean_prompt.storyboard.cues = [AudioCue(2, description="the wind whispers")]
        assert "audio.cue_is_dialogue" not in rules(clean_prompt)


class TestOpeningLines:
    """A line can only open a shot that has a framing clause to open on."""

    def test_a_shot_with_a_composition_takes_the_opening_line_quietly(
        self, clean_prompt
    ):
        clean_prompt.storyboard.beats[0].lines = [
            Line(speakers=["her"], text="Not yet.", placement="open")
        ]
        assert "dialogue.no_opening" not in rules(clean_prompt)

    def test_a_shot_with_no_composition_says_the_front_slot_is_unavailable(
        self, clean_prompt
    ):
        beat = clean_prompt.storyboard.beats[0]
        beat.angle = ""
        beat.lines = [Line(speakers=["her"], text="Not yet.", placement="open")]
        assert "dialogue.no_opening" in rules(clean_prompt)

    def test_a_default_line_never_raises_it(self, clean_prompt):
        beat = clean_prompt.storyboard.beats[0]
        beat.angle = ""
        beat.lines = [Line(speakers=["her"], text="Not yet.")]
        assert "dialogue.no_opening" not in rules(clean_prompt)


class TestCameraLoad:
    """One primary move per shot; stacked moves compete."""

    def test_three_moves_in_one_shot_is_noted(self, clean_prompt):
        clean_prompt.storyboard.beats[0].camera = (
            "pushes in with small amplitude while orbiting left and tilting up"
        )
        assert "beat.camera_complex" in rules(clean_prompt)

    def test_one_move_is_fine(self, clean_prompt):
        clean_prompt.storyboard.beats[0].camera = (
            "pushes in with small amplitude at slow speed toward the door"
        )
        assert "beat.camera_complex" not in rules(clean_prompt)

    def test_a_held_frame_is_fine(self, clean_prompt):
        clean_prompt.storyboard.beats[0].camera = "holds a static shot"
        assert "beat.camera_complex" not in rules(clean_prompt)


class TestNoCutBeats:
    """Pacing rules count shots; a no-cut beat is a moment inside one."""

    def _prompt(self, count: int, *, cut: bool, span: float = 1.0) -> VideoPrompt:
        beats = [
            Beat(index * span, (index + 1) * span, f"framing {index}",
                 f"a thing {index} happens", camera="holds a static shot",
                 cut=cut or index == 0)
            for index in range(count)
        ]
        return VideoPrompt(
            storyboard=Storyboard(duration=round(count * span), beats=beats),
            style="Live-action, cinematic.",
            audio="room tone",
        )

    def test_continuations_are_not_counted_as_cuts_per_second(self):
        # Eight half-second beats in four seconds: two cuts a second if they all
        # cut, and one unhurried shot if they do not.
        assert "beat.density" in rules(self._prompt(8, cut=True, span=0.5))
        assert "beat.density" not in rules(self._prompt(8, cut=False, span=0.5))

    def test_a_short_continuation_is_not_a_flash_frame(self):
        # Nothing cuts, so nothing flashes; the floor is about cutting.
        prompt = self._prompt(2, cut=False)
        prompt.storyboard.beats[1].start = 0.9
        prompt.storyboard.beats[0].end = 0.9
        assert "beat.too_short" not in rules(prompt)

    def test_a_rushed_move_inside_a_take_is_flagged_instead(self):
        prompt = self._prompt(2, cut=False)
        prompt.storyboard.beats[0].end = 1.6
        prompt.storyboard.beats[1].start = 1.6
        prompt.storyboard.beats[1].camera = "pushes in with large amplitude"
        assert "beat.move_rushed" in rules(prompt)

    def test_a_continuation_that_does_nothing_is_flagged(self):
        prompt = self._prompt(2, cut=False)
        prompt.storyboard.beats[1].action = ""
        prompt.storyboard.beats[1].camera = ""
        assert "beat.empty_continuation" in rules(prompt)

    def test_a_pure_camera_move_needs_no_action(self):
        prompt = self._prompt(2, cut=False)
        prompt.storyboard.beats[1].action = ""
        assert "beat.no_action" not in rules(prompt)
        assert "beat.empty_continuation" not in rules(prompt)

    def test_a_weak_cut_is_not_raised_against_a_continuation(self):
        # Clearing the cut is this rule's own advice, so taking it must silence
        # it rather than repeat it.
        beats = [
            Beat(0, 3, "a medium shot", "she waits by the window",
                 camera="holds a static shot"),
            Beat(3, 6, "a close-up", "she waits by the window",
                 camera="pushes in with small amplitude at slow speed"),
        ]
        prompt = VideoPrompt(
            storyboard=Storyboard(duration=6, beats=beats),
            style="Live-action, cinematic.",
            audio="room tone",
        )
        assert "beat.weak_cut" in rules(prompt)
        prompt.storyboard.beats[1].cut = False
        assert "beat.weak_cut" not in rules(prompt)
