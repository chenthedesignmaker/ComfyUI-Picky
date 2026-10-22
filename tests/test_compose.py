from __future__ import annotations

from video_prompter import (
    AudioCue,
    Beat,
    Line,
    Reference,
    ReferenceSet,
    Storyboard,
    VideoPrompt,
    compose,
)


def _field(text: str, name: str) -> str:
    """The one composed field block starting with `name:`."""
    return next(block for block in text.split("\n\n") if block.startswith(name))


class TestCoreFields:
    def test_the_three_core_fields_appear_in_official_order(self, clean_prompt):
        text = compose(clean_prompt)
        positions = [
            text.index("integrated_multimodal_description:"),
            text.index("overall_soundscape:"),
            text.index("non_diegetic_music:"),
        ]
        assert positions == sorted(positions)

    def test_the_style_opens_shot_one_from_inside_it(self, clean_prompt):
        # Guide 4.1: the base modes state the style and initial composition
        # after the [Shot 1] marker, not ahead of it. Only the full-reference
        # detailed_description lifts the opening onto its own line.
        desc = _field(compose(clean_prompt), "integrated_multimodal_description:")
        assert desc.startswith(
            "integrated_multimodal_description: [Shot 1] Realistic live-action "
            "cinematic look, anamorphic lens, film grain, wide establishing, "
            "the city wakes under low cloud."
        )

    def test_the_overview_takes_the_clause_after_the_style(self, clean_prompt):
        clean_prompt.overview = "a post-rain dusk metropolis"
        text = compose(clean_prompt)
        assert "[Shot 1] Realistic live-action cinematic look, anamorphic lens, "                "film grain, a post-rain dusk metropolis. Wide establishing, "                "the city wakes under low cloud." in text

    def test_the_overview_reaches_the_reference_document_too(self, clean_prompt):
        clean_prompt.overview = "A post-rain dusk metropolis."
        clean_prompt.references = ReferenceSet([Reference("image", "lead.png", role="the lead")])
        detail = _field(compose(clean_prompt), "detailed_description:")
        assert "A post-rain dusk metropolis." in detail
        assert detail.index("A post-rain dusk metropolis.") < detail.index("[Shot 1]")

    def test_music_defaults_to_na_when_empty(self, clean_prompt):
        clean_prompt.music = ""
        assert _field(compose(clean_prompt), "non_diegetic_music:") == "non_diegetic_music: N/A"

    def test_music_is_carried_through(self, clean_prompt):
        assert "non_diegetic_music: Sparse piano" in compose(clean_prompt)

    def test_negatives_never_reach_the_prompt(self, clean_prompt):
        # H3 has no negative input and the official format has no negative field.
        assert "watermark" not in compose(clean_prompt).lower()

    def test_an_empty_prompt_composes_to_nothing(self):
        assert compose(VideoPrompt(storyboard=Storyboard(10))) == ""


class TestShots:
    def test_shots_are_numbered_in_time_order(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(6, [Beat(3, 6, "b", "second"), Beat(0, 3, "a", "first")])
        )
        text = compose(prompt)
        assert "[Shot 1] A, first." in text
        assert "[Shot 2] At 00:03.000, the camera cuts to b, second." in text
        assert text.index("[Shot 1]") < text.index("[Shot 2]")

    def test_the_first_shot_carries_no_timecode(self):
        prompt = VideoPrompt(storyboard=Storyboard(6, [Beat(0, 3, "a", "x"), Beat(3, 6, "b", "y")]))
        text = compose(prompt)
        assert "[Shot 1] At" not in text

    def test_later_cut_times_use_mm_ss_mmm(self):
        prompt = VideoPrompt(storyboard=Storyboard(10, [Beat(0, 3.5, "a", "x"), Beat(3.5, 10, "b", "y")]))
        assert "[Shot 2] At 00:03.500," in compose(prompt)

    def test_camera_motion_is_woven_as_its_own_sentence(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(
                6,
                [Beat(0, 6, "a wide shot", "the door stays shut",
                      camera="pushes in with small amplitude at slow speed toward the door")],
            )
        )
        assert (
            "The camera pushes in with small amplitude at slow speed toward the door."
            in compose(prompt)
        )

    def test_a_custom_transition_replaces_the_default_cut(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(
                6,
                [Beat(0, 3, "a", "x"), Beat(3, 6, "b", "y", transition="the shot dissolves to")],
            )
        )
        assert "[Shot 2] At 00:03.000, the shot dissolves to b, y." in compose(prompt)


class TestSoundscape:
    def test_cues_follow_the_bed_in_time_order(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(
                5,
                [Beat(0, 5)],
                [AudioCue(4, description="the score bursts"), AudioCue(1, description="a hit")],
            ),
            audio="wind, footsteps",
        )
        # Guide 4.6: one paragraph of sentences, not a comma splice. Times are
        # the shots' own MM:SS.mmm, so the prompt speaks one dialect throughout.
        assert _field(compose(prompt), "overall_soundscape:") == (
            "overall_soundscape: Wind, footsteps. At 00:01.000, a hit. "
            "At 00:04.000, the score bursts."
        )

    def test_sustained_cues_render_as_a_span(self):
        # Plain English, because `6s-10s` is a compact form the guide never uses.
        prompt = VideoPrompt(
            storyboard=Storyboard(10, [Beat(0, 10)], [AudioCue(6, 10, "bass groove carries under")])
        )
        assert (
            "From 00:06.000 to 00:10.000, bass groove carries under." in compose(prompt)
        )

    def test_a_silent_soundscape_is_omitted(self):
        prompt = VideoPrompt(storyboard=Storyboard(5, [Beat(0, 5, "wide", "a door")]))
        text = compose(prompt)
        assert "overall_soundscape:" not in text
        assert "non_diegetic_music: N/A" in text


class TestDialogue:
    def _spoken(self, *lines: Line, **beat_kw) -> str:
        prompt = VideoPrompt(
            storyboard=Storyboard(6, [Beat(0, 6, "a shot", "she pauses", lines=list(lines), **beat_kw)])
        )
        return compose(prompt)

    def test_a_line_wraps_words_in_a_d_tag_outside_the_speaker_phrase(self):
        text = self._spoken(
            Line(speakers=["woman"], describe="the young woman with a breathy voice",
                 text="I get off at the next station.")
        )
        assert (
            "The young woman with a breathy voice (S1) says: "
            "<d>[English] I get off at the next station.</d>"
        ) in text

    def test_speaker_ids_are_assigned_by_first_appearance(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(
                8,
                [
                    Beat(0, 4, "a", "x", lines=[Line(speakers=["man"], text="Hello.")]),
                    Beat(4, 8, "b", "y", lines=[
                        Line(speakers=["woman"], text="Hi."),
                        Line(speakers=["man"], text="Bye."),
                    ]),
                ],
            )
        )
        text = compose(prompt)
        assert "(S1) says: <d>[English] Hello.</d>" in text  # man = S1
        assert "(S2) says: <d>[English] Hi.</d>" in text     # woman = S2
        assert "(S1) says: <d>[English] Bye.</d>" in text    # man keeps S1

    def test_a_group_line_uses_a_compound_tag(self):
        text = self._spoken(
            Line(speakers=["a", "b"], describe="the two children", text="Wait for us!",
                 verb="shout together")
        )
        assert "The two children (S1,S2) shout together: <d>[English] Wait for us!</d>" in text

    def test_a_voiceover_states_the_lips_stay_closed(self):
        text = self._spoken(
            Line(speakers=["man"], text="I still remember that road.", voiceover=True)
        )
        assert (
            "(S1) says in an off-screen voiceover: "
            "<d>[English] I still remember that road.</d> "
            "while their lips remain completely closed."
        ) in text

    def test_delivery_and_language_are_honoured(self):
        text = self._spoken(
            Line(speakers=["w"], text="Bonjour.", language="French", verb="whispers",
                 delivery="with a faint smile")
        )
        assert "(S1) whispers with a faint smile: <d>[French] Bonjour.</d>" in text

    def test_a_voiceover_can_name_whose_lips_stay_closed(self):
        text = self._spoken(
            Line(speakers=["man"], text="I still remember that road.", voiceover=True,
                 possessive="his")
        )
        assert "</d> while his lips remain completely closed." in text

    def test_a_line_with_no_words_is_dropped(self):
        assert "<d>" not in self._spoken(Line(speakers=["x"], text="   "))


class TestSeams:
    """A line crossing a cut is tagged at the connecting point in both halves."""

    def _crossing(self, first: Line, second: Line) -> str:
        return compose(
            VideoPrompt(
                storyboard=Storyboard(
                    10,
                    [
                        Beat(0, 5, "a wide shot", "he walks", lines=[first]),
                        Beat(5, 10, "a close-up", "his hands", lines=[second]),
                    ],
                )
            )
        )

    def test_both_halves_carry_the_tag_and_say_what_continues(self):
        text = self._crossing(
            Line(speakers=["m"], text="I still remember", continuity="into"),
            Line(speakers=["m"], text="every turn of it.", continuity="from"),
        )
        assert (
            "<d>[English] I still remember</d> <scenetrans> "
            "The speech continues uninterrupted into the next shot."
        ) in text
        assert (
            "<scenetrans> (S1) says: <d>[English] every turn of it.</d> "
            "The speech carries over from the previous shot."
        ) in text

    def test_cutoff_marks_speech_the_video_ends_on(self):
        text = self._crossing(
            Line(speakers=["m"], text="Listen,"),
            Line(speakers=["m"], text="there is one more thing I", cutoff=True),
        )
        assert (
            "<d>[English] there is one more thing I</d> <cutoff> "
            "The speech is cut off by the end of the video."
        ) in text

    def test_the_tags_never_land_inside_the_d_block(self):
        # Only the language tag and the verbatim words may go inside <d>.
        text = self._crossing(
            Line(speakers=["m"], text="I still remember", continuity="into"),
            Line(speakers=["m"], text="every turn.", continuity="from", cutoff=True),
        )
        for spoken in text.split("<d>")[1:]:
            inside = spoken.split("</d>")[0]
            assert "<scenetrans>" not in inside and "<cutoff>" not in inside


class TestScreenText:
    def test_strings_are_quoted_inside_the_opening_shot(self, clean_prompt):
        # Guide 4.5 states visible text in the shot it appears in. The flat list
        # has no shot of its own, so it lands in [Shot 1] rather than trailing
        # after the last shot, where it would read as a closing card.
        clean_prompt.screen_text = ["MIDNIGHT LINE", "STARRING"]
        desc = _field(compose(clean_prompt), "integrated_multimodal_description:")
        assert 'On-screen text reads "MIDNIGHT LINE", "STARRING".' in desc
        assert desc.index("On-screen text") < desc.index("[Shot 2]")


class TestKeyframeModes:
    def test_i2v_leads_with_the_first_frame_instruction(self, clean_prompt):
        clean_prompt.first_frame = "video_prompter/hero.png"
        assert compose(clean_prompt).startswith(
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )

    def test_fl2v_names_both_anchor_frames(self, clean_prompt):
        clean_prompt.first_frame = "start.png"
        clean_prompt.last_frame = "end.png"
        text = compose(clean_prompt)
        assert text.startswith("How the reference pictures align with the target video —")
        assert "Picture 1 (from Shot 1)" in text
        assert "Picture 2 (from Shot 3)" in text  # clean_prompt has three shots

    def test_l2v_anchors_the_last_shot(self, clean_prompt):
        clean_prompt.last_frame = "end.png"
        text = compose(clean_prompt)
        assert text.startswith("How the reference pictures align with the target video —")
        assert "<Picture 1> (from [Shot 3])" in text

    def test_t2v_has_no_alignment_instruction(self, clean_prompt):
        assert not compose(clean_prompt).startswith("For the target video")
        assert not compose(clean_prompt).startswith("How the reference pictures")


class TestReferences:
    def test_roles_become_subject_definitions(self, clean_prompt):
        clean_prompt.references = ReferenceSet(
            [
                Reference("image", "mood.png", "the overall mood and style reference"),
                Reference("image", "hero.png", "the lead character reference"),
            ]
        )
        assert compose(clean_prompt).startswith(
            "subject_definitions:\n"
            "<Picture 1> is the overall mood and style reference.\n"
            "<Picture 2> is the lead character reference."
        )

    def test_references_without_roles_have_no_subject_line(self, clean_prompt):
        clean_prompt.references = ReferenceSet([Reference("image", "mood.png")])
        # No role -> no subject_definitions section, so the doc opens elsewhere.
        assert not compose(clean_prompt).startswith("subject_definitions:")

    def test_tags_number_within_each_kind(self):
        references = ReferenceSet(
            [
                Reference("image", "a.png", "mood"),
                Reference("video", "b.mp4", "the camera move"),
                Reference("image", "c.png", "the character"),
            ]
        )
        assert references.render_roles() == (
            "<Picture 1> is mood. <Video 1> is the camera move. <Picture 2> is the character."
        )

    def test_identical_references_still_get_distinct_tags(self):
        references = ReferenceSet([Reference("image", ""), Reference("image", "")])
        assert [references.tag(item) for item in references.items] == [
            "<Picture 1>", "<Picture 2>"]

    def test_references_sharing_a_role_still_get_distinct_tags(self):
        references = ReferenceSet(
            [Reference("image", "a.png", "the mood"), Reference("image", "b.png", "the mood")]
        )
        assert references.tag(references.items[1]) == "<Picture 2>"

    def test_soundtracks_are_numbered_before_standalone_audio(self):
        references = ReferenceSet(
            [
                Reference("audio", "score.wav", "the music bed"),
                Reference("video_audio", "clip1.wav", "the soundtrack of video 1"),
            ]
        )
        assert references.tag(references.items[1]) == "<Audio 1>"
        assert references.tag(references.items[0]) == "<Audio 2>"


class TestReferenceDocument:
    def _ref_prompt(self) -> VideoPrompt:
        return VideoPrompt(
            storyboard=Storyboard(8, [
                Beat(0, 4, "a medium shot", "<Picture 1> the coffee shop, the woman holds a cookie"),
                Beat(4, 8, "a close-up", "the man enters with the dog"),
            ]),
            style="Realistic multi-camera sitcom with warm lighting",
            audio="soft coffee-shop room tone",
            references=ReferenceSet([
                Reference("image", "shop.png", "the coffee-shop environment"),
                Reference("audio", "voice.wav", "the voice-timbre reference for the woman",
                          retention="reference"),
            ]),
            ref_task=["reference generation", "audio reference"],
            ref_summary="The woman eats a cookie as a man enters with a dog.",
        )

    def test_the_six_sections_appear_in_order(self):
        text = compose(self._ref_prompt())
        order = [
            text.index("subject_definitions:"),
            text.index("summary:"),
            text.index("retention_analysis:"),
            text.index("detailed_description:"),
            text.index("overall_soundscape:"),
            text.index("non_diegetic_music:"),
        ]
        assert order == sorted(order)

    def test_the_summary_carries_the_task_type_prefix(self):
        assert "[reference generation + audio reference]" in compose(self._ref_prompt())

    def test_retention_uses_authored_and_default_markers(self):
        text = compose(self._ref_prompt())
        assert "<Audio 1>: reference -" in text                 # authored
        # The picture is woven into Shot 1's action, so it is placed there.
        assert "<Picture 1> (appears in [Shot 1]): fully_preserved -" in text

    def test_retention_states_where_each_item_appears(self):
        """Guide 4: the section records where an item lands, not only how much
        of it survives."""
        prompt = self._ref_prompt()
        # Audio is a layer rather than a thing in frame, so it stays bare.
        assert "<Audio 1>: reference -" in compose(prompt)

    def test_an_unplaceable_reference_gets_no_invented_scope(self):
        prompt = self._ref_prompt()
        prompt.storyboard.beats[0].action = "the woman holds a cookie"  # tag removed
        text = compose(prompt)
        assert "<Picture 1>: fully_preserved -" in text

    def test_the_style_leads_detailed_description_before_shot_one(self):
        text = compose(self._ref_prompt())
        detail = text.split("detailed_description:\n", 1)[1]
        assert detail.startswith("Realistic multi-camera sitcom with warm lighting.\n[Shot 1]")

    def test_default_tasks_when_none_are_set(self):
        prompt = self._ref_prompt()
        prompt.ref_task = []
        assert "[reference generation + audio reference]" in compose(prompt)


class TestIdentityGuard:
    """Two people in one shot is where H3 blends or swaps faces."""

    @staticmethod
    def _two_speakers(prompt):
        prompt.storyboard.beats[0].lines = [
            Line(speakers=["detective"], text="You saw who did it?"),
            Line(speakers=["informant"], text="I saw enough."),
        ]
        return prompt

    def test_speakers_sharing_a_shot_are_declared_distinct(self, clean_prompt):
        text = compose(self._two_speakers(clean_prompt))
        assert "Keep (S1) and (S2) visually distinct" in text

    def test_it_trails_the_description_rather_than_the_soundscape(self, clean_prompt):
        text = compose(self._two_speakers(clean_prompt))
        description = _field(text, "integrated_multimodal_description")
        assert "visually distinct" in description

    def test_one_speaker_needs_no_guard(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [
            Line(speakers=["her"], text="It is starting.")
        ]
        assert "visually distinct" not in compose(clean_prompt)

    def test_speakers_who_never_share_a_shot_are_not_guarded(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [
            Line(speakers=["her"], text="It is starting.")
        ]
        clean_prompt.storyboard.beats[2].lines = [
            Line(speakers=["him"], text="Not yet.")
        ]
        assert "visually distinct" not in compose(clean_prompt)

    def test_the_guard_can_be_switched_off(self, clean_prompt):
        prompt = self._two_speakers(clean_prompt)
        prompt.guard_identities = False
        assert "visually distinct" not in compose(prompt)

    def test_the_reference_document_states_it_above_shot_one(self, clean_prompt):
        prompt = self._two_speakers(clean_prompt)
        prompt.references = ReferenceSet(
            [Reference("image", "hero.png", "the detective")]
        )
        text = compose(prompt)
        detail = _field(text, "detailed_description")
        assert detail.index("visually distinct") < detail.index("[Shot 1]")


class TestLinePlacement:
    """Where a line sits inside its shot — the only thing that times speech."""

    def _prompt(self, placement: str) -> VideoPrompt:
        return VideoPrompt(
            storyboard=Storyboard(
                duration=10,
                beats=[
                    Beat(0, 5, "a medium-wide shot", "she stands at the window",
                         camera="holds level"),
                    Beat(5, 10, "a close-up", "she picks up the letter",
                         camera="pushes in",
                         lines=[Line(speakers=["w"], text="But here we are.",
                                     placement=placement)]),
                ],
            ),
            style="Live-action, cinematic.",
        )

    def test_by_default_a_line_trails_the_action_and_the_camera(self):
        desc = _field(compose(self._prompt("")), "integrated_multimodal_description:")
        assert (
            "[Shot 2] At 00:05.000, the camera cuts to a close-up, she picks up "
            "the letter. The camera pushes in. (S1) says: "
            "<d>[English] But here we are.</d>"
        ) in desc

    def test_open_puts_the_line_after_the_framing_and_before_the_action(self):
        desc = _field(compose(self._prompt("open")), "integrated_multimodal_description:")
        assert (
            "[Shot 2] At 00:05.000, the camera cuts to a close-up. (S1) says: "
            "<d>[English] But here we are.</d> She picks up the letter. "
            "The camera pushes in."
        ) in desc

    def test_shot_one_still_opens_on_the_style_not_on_speech(self):
        # The format has no slot ahead of the style, so an opening line goes
        # second there too.
        prompt = self._prompt("")
        prompt.storyboard.beats[0].lines = [
            Line(speakers=["w"], text="I never meant to stay.", placement="open")
        ]
        desc = _field(compose(prompt), "integrated_multimodal_description:")
        assert desc.startswith(
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic, "
            "a medium-wide shot. (S1) says: <d>[English] I never meant to stay.</d> "
            "She stands at the window. The camera holds level."
        )

    def test_an_opening_line_and_a_trailing_one_keep_their_own_slots(self):
        prompt = self._prompt("open")
        prompt.storyboard.beats[1].lines.append(
            Line(speakers=["m"], text="We are.")
        )
        desc = _field(compose(prompt), "integrated_multimodal_description:")
        assert (
            "a close-up. (S1) says: <d>[English] But here we are.</d> She picks "
            "up the letter. The camera pushes in. (S2) says: <d>[English] We are.</d>"
        ) in desc

    def test_a_shot_with_no_framing_clause_moves_the_line_only_as_far_as_it_can(self):
        # Nothing to open on: the transition phrase needs the action behind it,
        # so the action cannot move. The line still comes forward of the camera
        # sentence, which is as early as the shot allows. `dialogue.no_opening`
        # is what tells the author the front slot was not available.
        prompt = self._prompt("open")
        prompt.storyboard.beats[1].angle = ""
        assert (
            "[Shot 2] At 00:05.000, the camera cuts to she picks up the letter. "
            "(S1) says: <d>[English] But here we are.</d> The camera pushes in."
        ) in _field(compose(prompt), "integrated_multimodal_description:")

    def test_a_line_with_no_words_does_not_split_the_prose(self):
        prompt = self._prompt("")
        prompt.storyboard.beats[1].lines = [Line(speakers=["w"], placement="open")]
        assert (
            "a close-up, she picks up the letter."
            in _field(compose(prompt), "integrated_multimodal_description:")
        )


class TestNoCutBeats:
    """A beat with `cut` cleared is a moment inside the shot before it.

    The camera-led word order is the load-bearing part and is asserted
    literally: measured at one seed on 2026-08-31, the same move written behind
    its action rendered no camera movement at all, so a refactor that quietly
    reverses the clause is a real regression, not a wording change.
    """

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

    def test_continuations_fold_into_one_shot_marker(self):
        desc = _field(compose(self._prompt()), "integrated_multimodal_description:")
        assert desc.count("[Shot ") == 1
        assert "[Shot 2]" not in desc

    def test_a_continuation_leads_with_the_camera(self):
        desc = _field(compose(self._prompt()), "integrated_multimodal_description:")
        assert (
            "At 00:03.000, the camera pushes in with small amplitude at slow "
            "speed as she lifts a red mug and drinks."
        ) in desc

    def test_a_continuation_never_takes_a_cut_phrase(self):
        prompt = self._prompt()
        prompt.storyboard.beats[1].transition = "the shot cuts to"
        desc = _field(compose(prompt), "integrated_multimodal_description:")
        assert "the shot cuts to" not in desc

    def test_an_angle_on_a_continuation_is_where_the_move_arrives(self):
        prompt = self._prompt()
        prompt.storyboard.beats[2].angle = "a wider view of the same kitchen"
        desc = _field(compose(prompt), "integrated_multimodal_description:")
        assert (
            "the camera pans right with small amplitude at slow speed into a "
            "wider view of the same kitchen as she turns toward the doorway."
        ) in desc

    def test_a_continuation_without_a_move_leads_with_the_action(self):
        prompt = self._prompt()
        prompt.storyboard.beats[1].camera = ""
        desc = _field(compose(prompt), "integrated_multimodal_description:")
        assert "At 00:03.000, she lifts a red mug and drinks." in desc

    def test_dialogue_on_a_continuation_keeps_both_slots(self):
        prompt = self._prompt()
        prompt.storyboard.beats[1].lines = [
            Line(speakers=["w"], text="You are late.", placement="open"),
            Line(speakers=["w"], text="Again."),
        ]
        desc = _field(compose(prompt), "integrated_multimodal_description:")
        assert (
            "At 00:03.000, the camera pushes in with small amplitude at slow "
            "speed. (S1) says: <d>[English] You are late.</d> She lifts a red "
            "mug and drinks. (S1) says: <d>[English] Again.</d>"
        ) in desc

    def test_a_cut_after_a_continuation_opens_the_next_shot(self):
        prompt = self._prompt()
        prompt.storyboard.beats[2].cut = True
        prompt.storyboard.beats[2].angle = "a close-up of the doorway"
        desc = _field(compose(prompt), "integrated_multimodal_description:")
        assert "[Shot 2] At 00:06.000, the camera cuts to a close-up" in desc

    def test_a_keyframe_anchors_the_last_shot_not_the_last_beat(self):
        prompt = self._prompt()
        prompt.last_frame = "end.png"
        # Three beats, one shot: the end anchor is [Shot 1], not [Shot 3].
        assert "(from [Shot 1])" in compose(prompt)
