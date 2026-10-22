from __future__ import annotations

from video_prompter import (
    AudioCue,
    Beat,
    Line,
    Severity,
    Storyboard,
    VideoPrompt,
    compose,
    lint,
    parse,
)


class TestRoundTrip:
    def test_the_core_fields_survive_a_round_trip(self, clean_prompt):
        # Not an exact string round-trip — the composer weaves angle, action and
        # camera into one shot sentence — but the fields recover.
        reparsed = parse(compose(clean_prompt), duration=clean_prompt.duration)
        assert reparsed.music == clean_prompt.music
        assert reparsed.audio.startswith("Rain, distant traffic")
        assert len(reparsed.storyboard.beats) == 3

    def test_screen_text_survives(self, clean_prompt):
        clean_prompt.screen_text = ["MIDNIGHT LINE", "STARRING"]
        reparsed = parse(compose(clean_prompt), duration=10)
        assert reparsed.screen_text == ["MIDNIGHT LINE", "STARRING"]


class TestRooftopExample:
    """The rooftop chase, rewritten in the official format and read back."""

    def test_all_four_shots_are_recovered(self, rooftop_text):
        starts = [beat.start for beat in parse(rooftop_text).storyboard.ordered_beats()]
        assert starts == [0.0, 4.0, 8.0, 12.0]

    def test_duration_is_inferred_from_the_last_cut(self, rooftop_text):
        # Last cut at 12s, so the final shot needs a second to run: 13s.
        assert parse(rooftop_text).duration == 13

    def test_the_opening_comes_back_as_style_and_overview(self, rooftop_text):
        prompt = parse(rooftop_text)
        assert prompt.style == (
            "Realistic live-action cinematic look, anamorphic lens, film grain."
        )
        assert prompt.overview.startswith("A post-rain dusk metropolis")
        assert prompt.storyboard.beats[0].action.startswith("A high side angle")

    def test_an_opening_folded_into_shot_one_is_still_read(self):
        # Not every prompt in the wild states the style ahead of [Shot 1]; when
        # it is inside the shot there is nothing to split, so it stays there.
        prompt = parse(
            "integrated_multimodal_description: [Shot 1] Live-action, a door opens."
        )
        assert prompt.style == ""
        assert prompt.storyboard.beats[0].action == "Live-action, a door opens."

    def test_cut_transitions_are_recovered(self, rooftop_text):
        beats = parse(rooftop_text).storyboard.ordered_beats()
        assert beats[1].transition == "the camera cuts to"
        assert beats[3].transition == "the shot switches to"

    def test_the_soundscape_bed_and_cue_split_apart(self, rooftop_text):
        prompt = parse(rooftop_text)
        assert prompt.audio.startswith("wind, rapid footsteps")
        assert prompt.storyboard.cues[0].start == 12.0
        assert "score bursts" in prompt.storyboard.cues[0].description

    def test_the_score_lands_in_music(self, rooftop_text):
        assert parse(rooftop_text).music.startswith("A driving percussive score")

    def test_the_recovered_storyboard_is_contiguous_and_error_free(self, rooftop_text):
        prompt = parse(rooftop_text)
        assert prompt.storyboard.is_contiguous()
        assert [f for f in lint(prompt) if f.severity is Severity.ERROR] == []


class TestHeuristics:
    def test_music_na_is_read_as_empty(self):
        prompt = parse(
            "integrated_multimodal_description: [Shot 1] A door opens.\n\n"
            "non_diegetic_music: N/A"
        )
        assert prompt.music == ""
        assert prompt.storyboard.beats[0].action == "A door opens."

    def test_reference_roles_are_recovered(self):
        prompt = parse(
            "<Picture 1> is the mood reference. <Video 1> is the camera move.\n\n"
            "integrated_multimodal_description: [Shot 1] a door opens."
        )
        assert [(r.kind, r.role) for r in prompt.references.items] == [
            ("image", "the mood reference"),
            ("video", "the camera move"),
        ]

    def test_a_prompt_with_only_a_soundscape_still_parses(self):
        prompt = parse("overall_soundscape: wind and distant rain")
        assert prompt.audio == "wind and distant rain"
        assert prompt.storyboard.beats == []

    def test_a_single_shot_lands_its_prose_in_action(self):
        prompt = parse("integrated_multimodal_description: [Shot 1] the camera holds on the door.")
        assert prompt.storyboard.beats[0].start == 0.0
        assert prompt.storyboard.beats[0].action == "the camera holds on the door."


class TestDialogue:
    def test_dialogue_is_pulled_off_the_action_into_lines(self):
        prompt = parse(
            "integrated_multimodal_description: [Shot 1] a kitchen. "
            "The baker (S1) says: <d>[English] First batch.</d>"
        )
        beat = prompt.storyboard.beats[0]
        assert beat.action == "a kitchen."
        assert beat.lines[0].speakers == ["S1"]
        assert beat.lines[0].text == "First batch."

    def test_a_group_and_a_voiceover_recover(self):
        prompt = parse(
            "integrated_multimodal_description: [Shot 1] a hall. "
            "(S1,S2) shout together: <d>[English] Wait!</d> "
            "(S3) says in an off-screen voiceover: <d>[English] I remember.</d> "
            "while their lips remain completely closed."
        )
        lines = prompt.storyboard.beats[0].lines
        assert lines[0].speakers == ["S1", "S2"]
        assert lines[1].speakers == ["S3"] and lines[1].voiceover is True

    def test_reference_document_round_trips(self):
        from video_prompter import Reference, ReferenceSet, compose

        original = VideoPrompt(
            storyboard=Storyboard(8, [
                Beat(0, 4, "a medium shot", "<Picture 1> the shop"),
                Beat(4, 8, "a close-up", "the dog lunges"),
            ]),
            style="Realistic sitcom",
            references=ReferenceSet([
                Reference("image", "shop.png", "the coffee-shop environment"),
                Reference("audio", "v.wav", "the voice reference", retention="reference"),
            ]),
            ref_task=["reference generation", "audio reference"],
            ref_summary="The woman eats a cookie.",
        )
        reparsed = parse(compose(original), duration=8)
        assert reparsed.ref_task == ["reference generation", "audio reference"]
        assert reparsed.ref_summary == "The woman eats a cookie."
        assert reparsed.style == "Realistic sitcom."  # composer ends the style sentence
        roles = [(r.role, r.retention) for r in reparsed.references.items]
        assert ("the coffee-shop environment", "fully_preserved") in roles
        assert ("the voice reference", "reference") in roles
        assert len(reparsed.storyboard.beats) == 2

    def test_dialogue_survives_a_compose_round_trip(self):
        original = VideoPrompt(
            storyboard=Storyboard(
                6,
                [Beat(0, 6, "a shot", "she waits",
                      lines=[Line(speakers=["w"], text="Over here.", verb="calls")])],
            ),
            style="Cinematic",
        )
        from video_prompter import compose

        reparsed = parse(compose(original), duration=6)
        line = reparsed.storyboard.beats[0].lines[0]
        assert line.text == "Over here." and line.verb == "calls"

    def test_an_opening_line_comes_back_in_its_slot(self):
        from video_prompter import compose

        original = VideoPrompt(
            storyboard=Storyboard(
                6,
                [Beat(0, 6, "a close-up", "she waits", camera="pushes in",
                      lines=[Line(speakers=["w"], text="Over here.", placement="open"),
                             Line(speakers=["w"], text="Please.")])],
            ),
            style="Cinematic",
        )
        reparsed = parse(compose(original), duration=6)
        slots = [line.placement for line in reparsed.storyboard.beats[0].lines]
        assert slots == ["open", ""]

    def test_the_identity_guard_is_not_read_as_the_last_shot_s_action(self):
        from video_prompter import compose

        original = VideoPrompt(
            storyboard=Storyboard(
                6,
                [Beat(0, 6, "a two-shot", "they face each other",
                      lines=[Line(speakers=["a"], text="You came."),
                             Line(speakers=["b"], text="I did.")])],
            ),
            style="Cinematic",
            guard_identities=True,
        )
        reparsed = parse(compose(original), duration=6)
        assert "must not merge" not in reparsed.storyboard.beats[0].action
        assert reparsed.guard_identities


class TestCueDialect:
    """Cues compose in the shots' MM:SS.mmm; older prompts used bare seconds."""

    def test_a_timecoded_cue_survives_a_compose_round_trip(self):
        from video_prompter import compose

        original = VideoPrompt(
            storyboard=Storyboard(
                10,
                [Beat(0, 10, "a wide shot", "a room")],
                [AudioCue(3.5, description="a door slams below"),
                 AudioCue(6, 10, "a kettle rises to a whistle")],
            ),
            style="Cinematic",
            audio="rain drums on the glass",
        )
        reparsed = parse(compose(original), duration=10)
        assert reparsed.audio == "Rain drums on the glass."
        assert [(c.start, c.end, c.description) for c in reparsed.storyboard.cues] == [
            (3.5, None, "a door slams below"),
            (6.0, 10.0, "a kettle rises to a whistle"),
        ]

    def test_the_older_bare_second_form_still_reads(self):
        prompt = parse(
            "overall_soundscape: Wind and traffic. At 3.5s, a door slams. "
            "At 6s-10s, a kettle rises."
        )
        assert prompt.audio == "Wind and traffic."
        assert [(c.start, c.end) for c in prompt.storyboard.cues] == [
            (3.5, None),
            (6.0, 10.0),
        ]


class TestNoCutBeats:
    """A continuous take reads back as beats, not as one blob with times in it."""

    def _prompt(self) -> VideoPrompt:
        return VideoPrompt(
            storyboard=Storyboard(
                duration=9,
                beats=[
                    Beat(0, 3, "a medium shot of a woman at a counter",
                         "she stands still", camera="holds a static shot"),
                    Beat(3, 6, "", "she lifts a red mug and drinks",
                         camera="pushes in with small amplitude at slow speed",
                         cut=False),
                    Beat(6, 9, "", "she turns toward the doorway",
                         camera="pans right with small amplitude at slow speed",
                         cut=False),
                ],
            ),
            style="Live-action, cinematic.",
            audio="a quiet kitchen room tone",
        )

    def test_timed_moments_inside_a_shot_come_back_as_no_cut_beats(self):
        back = parse(compose(self._prompt()), duration=9)
        beats = back.storyboard.ordered_beats()
        assert [beat.cut for beat in beats] == [True, False, False]
        assert [beat.start for beat in beats] == [0.0, 3.0, 6.0]

    def test_the_text_it_read_is_the_text_it_writes(self):
        # Camera and action stay joined in `action` precisely so this holds:
        # a parse that guessed where the move ended would not survive it.
        text = compose(self._prompt())
        assert compose(parse(text, duration=9)) == text

    def test_a_prompt_with_no_internal_times_is_unchanged(self, clean_prompt):
        beats = parse(compose(clean_prompt), duration=10).storyboard.ordered_beats()
        assert all(beat.cut for beat in beats)
        assert len(beats) == 3
