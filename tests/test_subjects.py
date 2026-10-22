"""`<Subject N>`: what the video contains, as opposed to which files it was given.

The guide's distinction (2.1-2.4) is the whole point of these tests. A picture
that only *shows* a character is not a referenced item in its own right — the
character is — so it must be cited inside that character's line and must not
get a `<Picture N>` definition or a retention marker of its own.
"""

from __future__ import annotations

from video_prompter import Reference, ReferenceSet, Subject, SubjectSet, VideoPrompt, compose
from video_prompter.cast import Cast, Member, resolve
from video_prompter.parse import parse
from video_prompter.serialize import prompt_from_dict, prompt_to_dict
from video_prompter.storyboard import Beat, Line, Storyboard


def _field(text: str, name: str) -> str:
    return next(block for block in text.split("\n\n") if block.startswith(name))


def _mara() -> Member:
    return Member(
        id="cast_11111111",
        name="Mara",
        kind="character",
        appearance="a woman in her thirties, dark hair, a grey hoodie",
        references=[Reference("image", "mara_face.png"), Reference("image", "mara_body.png")],
    )


def _bedroom() -> Member:
    return Member(
        id="cast_22222222",
        name="Bedroom",
        kind="environment",
        appearance="a corner bedroom with a window wall",
        references=[Reference("image", "bedroom.png")],
    )


def _speaking(prompt: VideoPrompt, member_id: str = "cast_11111111") -> VideoPrompt:
    prompt.storyboard.beats = [
        Beat(0, 8, action="she looks up",
             lines=[Line(speakers=[member_id], text="You're early.")])
    ]
    return prompt


class TestSubjectSet:
    def test_tags_number_in_order(self):
        subjects = SubjectSet([Subject(role="a woman"), Subject(role="a room")])
        assert subjects.tags() == ["<Subject 1>", "<Subject 2>"]

    def test_tags_are_matched_by_identity_not_value(self):
        # Two subjects that say the same thing must still be told apart.
        subjects = SubjectSet([Subject(role="a woman"), Subject(role="a woman")])
        assert subjects.tag(subjects.items[1]) == "<Subject 2>"

    def test_citing_finds_the_subject_a_file_defines(self):
        subject = Subject(role="a woman", sources=["face.png"])
        subjects = SubjectSet([subject])
        assert subjects.citing("face.png") is subject
        assert subjects.citing("other.png") is None
        assert subjects.citing("") is None


class TestComposedDefinitions:
    def _prompt(self) -> VideoPrompt:
        prompt = VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="she looks up")]),
            references=ReferenceSet([Reference("image", "face.png")]),
            subjects=SubjectSet([
                Subject(role="a woman in her thirties", sources=["face.png"], noun="person")
            ]),
        )
        return prompt

    def test_a_subject_cites_its_source_inside_its_own_line(self):
        text = _field(compose(self._prompt()), "subject_definitions")
        assert "<Subject 1> is the person in <Picture 1>, a woman in her thirties." in text

    def test_a_cited_picture_gets_no_line_of_its_own(self):
        prompt = self._prompt()
        prompt.references.items[0].role = "her full-body sheet"
        text = _field(compose(prompt), "subject_definitions")
        assert "<Picture 1> is" not in text

    def test_a_cited_picture_gets_no_retention_marker_of_its_own(self):
        text = _field(compose(self._prompt()), "retention_analysis")
        assert "<Subject 1>: fully_preserved - a woman in her thirties." in text
        assert "<Picture 1>:" not in text

    def test_an_uncited_picture_keeps_its_own_line(self):
        prompt = self._prompt()
        prompt.references.items.append(Reference("image", "plate.png", "the first frame of Shot 1"))
        text = _field(compose(prompt), "subject_definitions")
        assert "<Picture 2> is the first frame of Shot 1." in text

    def test_several_sources_are_listed_in_one_sentence(self):
        prompt = self._prompt()
        prompt.references.items.append(Reference("image", "body.png"))
        prompt.subjects.items[0].sources.append("body.png")
        text = _field(compose(prompt), "subject_definitions")
        assert "is the person in <Picture 1> and <Picture 2>," in text

    def test_a_subject_without_a_noun_reads_the_other_way_round(self):
        prompt = self._prompt()
        prompt.subjects.items[0].noun = ""
        text = _field(compose(prompt), "subject_definitions")
        assert "<Subject 1> is a woman in her thirties, shown in <Picture 1>." in text

    def test_a_sourceless_subject_states_itself_plainly(self):
        prompt = self._prompt()
        prompt.subjects.items[0].sources = []
        text = _field(compose(prompt), "subject_definitions")
        assert "<Subject 1> is a woman in her thirties." in text

    def test_subjects_lead_the_default_summary(self):
        prompt = self._prompt()
        text = _field(compose(prompt), "summary")
        assert "using <Subject 1>." in text  # not <Picture 1>, which has no line


class TestAudioBacklink:
    def _prompt(self) -> VideoPrompt:
        prompt = VideoPrompt(
            storyboard=Storyboard(8, []),
            references=ReferenceSet([
                Reference("image", "face.png"),
                Reference("audio", "voice.wav", "the voice-timbre reference"),
            ]),
            subjects=SubjectSet([
                Subject(role="a woman", sources=["face.png", "voice.wav"],
                        noun="person", member="cast_11111111")
            ]),
        )
        return _speaking(prompt)

    def test_audio_keeps_its_own_line_and_names_its_subject(self):
        text = _field(compose(self._prompt()), "subject_definitions")
        assert "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)." in text

    def test_the_speaker_id_is_dropped_when_the_subject_never_speaks(self):
        prompt = self._prompt()
        prompt.storyboard.beats = []
        text = _field(compose(prompt), "subject_definitions")
        assert "<Audio 1> is the voice-timbre reference for <Subject 1>." in text

    def test_an_authored_role_that_already_names_a_subject_is_left_alone(self):
        prompt = self._prompt()
        prompt.references.items[1].role = "the timbre of <Subject 1>"
        text = _field(compose(prompt), "subject_definitions")
        assert "<Audio 1> is the timbre of <Subject 1>." in text

    def test_audio_is_not_listed_among_the_visual_sources(self):
        text = _field(compose(self._prompt()), "subject_definitions")
        assert "is the person in <Picture 1>, a woman." in text


class TestCastResolution:
    def test_a_member_on_screen_becomes_one_subject_not_three_pictures(self):
        prompt = _speaking(VideoPrompt.blank(8))
        text = compose(prompt, Cast([_mara()]))
        definitions = _field(text, "subject_definitions")
        assert definitions == (
            "subject_definitions:\n<Subject 1> is the person in <Picture 1> and <Picture 2>, "
            "a woman in her thirties, dark hair, a grey hoodie."
        )

    def test_each_kind_is_cited_by_its_own_noun(self):
        prompt = _speaking(VideoPrompt.blank(8))
        prompt.cast = ["cast_22222222"]
        text = _field(compose(prompt, Cast([_mara(), _bedroom()])), "subject_definitions")
        assert "<Subject 2> is the setting in <Picture 3>, a corner bedroom" in text

    def test_the_members_files_still_reach_the_node(self):
        prompt = _speaking(VideoPrompt.blank(8))
        resolved = resolve(prompt, Cast([_mara()]))
        assert [item.path for item in resolved.references.items] == [
            "mara_face.png", "mara_body.png"]

    def test_a_member_with_no_identity_files_gets_no_subject(self):
        member = _mara()
        member.references = []
        resolved = resolve(_speaking(VideoPrompt.blank(8)), Cast([member]))
        assert resolved.subjects.items == []

    def test_an_audio_identity_file_defaults_to_a_timbre_role_not_an_appearance(self):
        member = _mara()
        member.references.append(Reference("audio", "mara_voice.wav"))
        text = _field(compose(_speaking(VideoPrompt.blank(8)), Cast([member])),
                      "subject_definitions")
        assert "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)." in text

    def test_a_linked_row_overrides_the_wording_for_this_segment_only(self):
        prompt = _speaking(VideoPrompt.blank(8))
        prompt.subjects = SubjectSet([
            Subject(role="the same woman, now soaked through", member="cast_11111111",
                    retention="partially_preserved")
        ])
        text = compose(prompt, Cast([_mara()]))
        assert "<Subject 1> is the person in <Picture 1> and <Picture 2>, the same woman, " \
            "now soaked through." in text
        assert "<Subject 1> (appears in [Shot 1]): partially_preserved -" in text

    def test_a_link_to_a_deleted_member_is_dropped(self):
        prompt = _speaking(VideoPrompt.blank(8))
        prompt.subjects = SubjectSet([Subject(role="a ghost", member="cast_99999999")])
        resolved = resolve(prompt, Cast([_mara()]))
        assert [subject.member for subject in resolved.subjects.items] == ["cast_11111111"]

    def test_a_linked_subject_puts_its_member_on_screen(self):
        prompt = VideoPrompt.blank(8)
        prompt.subjects = SubjectSet([Subject(member="cast_22222222")])
        resolved = resolve(prompt, Cast([_bedroom()]))
        assert [item.path for item in resolved.references.items] == ["bedroom.png"]

    def test_the_segments_own_subjects_stay_ahead_of_the_casts(self):
        prompt = _speaking(VideoPrompt.blank(8))
        prompt.references = ReferenceSet([Reference("image", "rain.png")])
        prompt.subjects = SubjectSet([Subject(role="heavy rain", sources=["rain.png"])])
        resolved = resolve(prompt, Cast([_mara()]))
        assert [subject.role for subject in resolved.subjects.items][0] == "heavy rain"
        assert resolved.subjects.items[1].member == "cast_11111111"


class TestRoundTrip:
    def test_subjects_survive_a_save(self):
        prompt = VideoPrompt.blank(8)
        prompt.subjects = SubjectSet([
            Subject(role="a woman", retention="attribute_transfer", sources=["face.png"],
                    member="cast_11111111", noun="person", label="Mara")
        ])
        back = prompt_from_dict(prompt_to_dict(prompt))
        assert back.subjects.items == prompt.subjects.items

    def test_a_prompt_written_before_subjects_existed_still_loads(self):
        data = prompt_to_dict(VideoPrompt.blank(8))
        del data["subjects"]
        assert prompt_from_dict(data).subjects.items == []

    def test_an_empty_row_is_dropped_rather_than_stored(self):
        data = prompt_to_dict(VideoPrompt.blank(8))
        data["subjects"] = [{"role": "  ", "sources": [], "member": ""}]
        assert prompt_from_dict(data).subjects.items == []

    def test_parse_reads_subject_lines_back(self):
        text = (
            "subject_definitions:\n"
            "<Subject 1> is the person in <Picture 1>, a woman in her thirties\n"
            "<Picture 2> is the first frame of Shot 1.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: attribute_transfer - the woman.\n"
        )
        prompt = parse(text)
        assert [subject.role for subject in prompt.subjects.items] == [
            "the person in <Picture 1>, a woman in her thirties"
        ]
        assert prompt.subjects.items[0].retention == "attribute_transfer"
        # The `<Picture 2>` line is a reference, not a subject.
        assert [item.role for item in prompt.references.items] == ["the first frame of Shot 1"]


class TestLinting:
    def test_a_file_a_subject_speaks_for_is_not_reported_unused(self):
        from video_prompter.lint import lint

        prompt = VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="she looks up")]),
            references=ReferenceSet([Reference("image", "face.png")]),
            subjects=SubjectSet([Subject(role="a woman", sources=["face.png"])]),
            ref_summary="Hold her face.",
        )
        codes = [finding.rule for finding in lint(prompt)]
        assert "references.unused" not in codes

class TestSpeakingSubjects:
    """Guide 5.4: a speaking subject keeps both labels — `<Subject N> (Sx)`."""

    def _prompt(self) -> VideoPrompt:
        return _speaking(VideoPrompt.blank(8))

    def test_a_speaking_subject_leads_with_its_label(self):
        text = compose(self._prompt(), Cast([_mara()]))
        assert "<Subject 1> (S1)" in text

    def test_the_appearance_rides_as_apposition_on_the_first_line(self):
        text = compose(self._prompt(), Cast([_mara()]))
        assert (
            "<Subject 1> (S1), a woman in her thirties, dark hair, a grey hoodie, says:"
        ) in text

    def test_a_later_line_is_just_the_two_tags(self):
        prompt = self._prompt()
        prompt.storyboard.beats.append(
            Beat(8, 8, action="she stands",
                 lines=[Line(speakers=["cast_11111111"], text="I know.")])
        )
        prompt.storyboard.duration = 12
        text = compose(prompt, Cast([_mara()]))
        assert "<Subject 1> (S1) says: <d>[English] I know.</d>" in text

    def test_base_mode_is_untouched(self):
        # No references, so no subjects and no labels to lead with.
        prompt = self._prompt()
        text = compose(prompt, Cast([Member(id="cast_11111111", name="Mara",
                                            appearance="a woman in her thirties")]))
        assert "<Subject" not in text
        assert "A woman in her thirties (S1) says:" in text

    def test_a_line_with_one_undefined_voice_falls_back_to_prose(self):
        prompt = self._prompt()
        prompt.storyboard.beats[0].lines[0].speakers = ["cast_11111111", "narrator"]
        text = compose(prompt, Cast([_mara()]))
        # Half-labelling a group would read as though the rest were unknown.
        assert "<Subject 1> (S1,S2)" not in text


class TestRetentionScope:
    """Guide 4: the section records WHERE each item appears."""

    def test_a_speaking_member_is_placed_by_their_lines(self):
        prompt = _speaking(VideoPrompt.blank(12))
        prompt.storyboard.beats.append(Beat(8, 12, action="the room is empty"))
        text = _field(compose(prompt, Cast([_mara()])), "retention_analysis")
        assert "<Subject 1> (appears in [Shot 1]):" in text

    def test_a_member_in_every_shot_is_written_as_throughout(self):
        prompt = _speaking(VideoPrompt.blank(12))
        prompt.cast = ["cast_22222222"]
        prompt.storyboard.beats.append(Beat(8, 12, action="the room is empty"))
        text = _field(compose(prompt, Cast([_mara(), _bedroom()])), "retention_analysis")
        assert "<Subject 2> (appears throughout):" in text

    def test_a_woven_tag_places_a_standalone_reference(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(8, [
                Beat(0, 4, action="the plate <Picture 1> fills the frame"),
                Beat(4, 8, action="she turns away"),
            ]),
            references=ReferenceSet([Reference("image", "plate.png", "the opening plate")]),
        )
        text = _field(compose(prompt), "retention_analysis")
        assert "<Picture 1> (appears in [Shot 1]):" in text

    def test_audio_entries_stay_bare(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="she waits")]),
            references=ReferenceSet([Reference("audio", "v.wav", "the timbre reference")]),
        )
        text = _field(compose(prompt), "retention_analysis")
        assert "<Audio 1>: reference -" in text

    def test_nothing_is_invented_for_an_unplaceable_item(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="she waits")]),
            references=ReferenceSet([Reference("image", "mood.png", "the mood")]),
        )
        text = _field(compose(prompt), "retention_analysis")
        assert "<Picture 1>: fully_preserved - the mood." in text


class TestSummaryAndLayout:
    def test_a_video_edit_opens_with_the_sentence_the_guide_names(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="the cut plays on")]),
            references=ReferenceSet([Reference("video", "source.mp4", "the source cut")]),
            ref_task=["video editing"],
            ref_summary="The grade is warmed and the middle beat is trimmed.",
        )
        text = _field(compose(prompt), "summary")
        assert text.startswith(
            "summary:\n[video editing] The target video is an edited version of <Video 1>."
        )

    def test_it_is_not_repeated_when_the_author_said_it(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="the cut plays on")]),
            references=ReferenceSet([Reference("video", "source.mp4", "the source cut")]),
            ref_task=["video editing"],
            ref_summary="The target video is an edited version of <Video 1>, warmed.",
        )
        assert _field(compose(prompt), "summary").count("edited version of") == 1

    def test_the_task_prefix_is_a_label_not_a_sentence(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="she waits")]),
            references=ReferenceSet([Reference("image", "a.png", "the mood")]),
            ref_summary="Hold the mood.",
        )
        assert "[reference generation] Hold the mood." in compose(prompt)

    def test_the_last_two_sections_sit_on_their_own_lines(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="she waits")]),
            references=ReferenceSet([Reference("image", "a.png", "the mood")]),
            audio="quiet room tone",
        )
        text = compose(prompt)
        assert "overall_soundscape:\nQuiet room tone." in text
        assert "non_diegetic_music:\nN/A" in text

    def test_base_mode_keeps_the_inline_form(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="she waits")]),
            audio="quiet room tone",
        )
        assert "overall_soundscape: Quiet room tone." in compose(prompt)


class TestSpokenPunctuation:
    """Guide 5.4: complete utterances end on `.`, `?` or `!` inside <d>."""

    def _line_text(self, line: Line) -> str:
        prompt = VideoPrompt(storyboard=Storyboard(8, [Beat(0, 8, action="he waits",
                                                            lines=[line])]))
        return compose(prompt)

    def test_a_bare_statement_is_completed(self):
        assert "<d>[English] You are early.</d>" in self._line_text(
            Line(speakers=["m"], text="You are early"))

    def test_an_existing_mark_is_left_alone(self):
        assert "<d>[English] You are early?</d>" in self._line_text(
            Line(speakers=["m"], text="You are early?"))

    def test_a_cut_off_line_stays_unfinished(self):
        # `<cutoff>` says the sentence never lands; a full stop would deny it.
        assert "<d>[English] there is one more thing I</d>" in self._line_text(
            Line(speakers=["m"], text="there is one more thing I", cutoff=True))

    def test_a_line_running_into_the_next_shot_stays_unfinished(self):
        assert "<d>[English] I still remember</d>" in self._line_text(
            Line(speakers=["m"], text="I still remember", continuity="into"))

    def test_a_comma_is_punctuation_and_is_respected(self):
        assert "<d>[English] Listen,</d>" in self._line_text(
            Line(speakers=["m"], text="Listen,"))


class TestGuideLints:
    def _lint(self, prompt: VideoPrompt, cast=None) -> list[str]:
        from video_prompter.lint import lint

        return [finding.rule for finding in lint(prompt, cast)]

    def _base(self) -> VideoPrompt:
        return VideoPrompt(
            storyboard=Storyboard(8, [Beat(0, 8, action="she waits")]),
            references=ReferenceSet([Reference("image", "a.png", "the mood")]),
            ref_summary="Hold the mood of <Picture 1>.",
        )

    def test_an_undefined_label_in_the_summary_is_flagged(self):
        prompt = self._base()
        prompt.ref_summary = "Hold <Picture 1> and the walk of <Video 2>."
        assert "ref.summary_undefined_tag" in self._lint(prompt)

    def test_a_defined_label_is_not(self):
        assert "ref.summary_undefined_tag" not in self._lint(self._base())

    def test_a_speaker_id_in_a_role_is_flagged(self):
        prompt = self._base()
        prompt.subjects = SubjectSet([Subject(role="the woman (S1)", sources=["a.png"])])
        assert "ref.retention_speaker_id" in self._lint(prompt)

    def test_an_audio_role_may_carry_its_speaker_id(self):
        prompt = self._base()
        prompt.references.items.append(
            Reference("audio", "v.wav", "a low, dry voice (S1)"))
        assert "ref.retention_speaker_id" not in self._lint(prompt)

    def test_a_subject_the_description_never_names_is_flagged(self):
        prompt = self._base()
        prompt.subjects = SubjectSet([Subject(role="heavy rain", sources=["a.png"])])
        assert "subjects.unused" in self._lint(prompt)

    def test_a_speaking_member_needs_no_hand_written_tag(self):
        prompt = _speaking(VideoPrompt.blank(8))
        prompt.ref_summary = "Hold her face."
        assert "subjects.unused" not in self._lint(prompt, Cast([_mara()]))

    def test_a_thin_description_is_noted_but_not_for_an_edit(self):
        assert "ref.description_length" in self._lint(self._base())
        edit = self._base()
        edit.ref_task = ["video editing"]
        assert "ref.description_length" not in self._lint(edit)

    def test_an_invalid_subject_retention_marker_is_flagged(self):
        prompt = self._base()
        prompt.subjects = SubjectSet([
            Subject(role="the woman in <Picture 1>", sources=["a.png"], retention="mostly")
        ])
        assert "ref.retention_invalid" in self._lint(prompt)
