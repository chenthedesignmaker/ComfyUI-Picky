from __future__ import annotations

import pytest

from video_prompter import (
    Beat,
    Cast,
    Line,
    Member,
    Reference,
    ReferenceSet,
    Storyboard,
    Subject,
    SubjectSet,
    VideoPrompt,
    compose,
    lint,
)
from video_prompter.cast import appearing, looks_like_id, mint_id, resolve, unknown_ids
from video_prompter.serialize import prompt_from_dict, prompt_to_dict


@pytest.fixture
def mara() -> Member:
    return Member(
        id="cast_11111111",
        name="Mara",
        kind="character",
        appearance="a tall woman in a rain-darkened red coat",
        voice="a low, unhurried voice",
    )


@pytest.fixture
def ivo() -> Member:
    return Member(
        id="cast_22222222",
        name="Ivo",
        kind="character",
        appearance="a wiry man with a shaved head",
    )


@pytest.fixture
def alley() -> Member:
    return Member(
        id="cast_33333333",
        name="The alley",
        kind="environment",
        appearance="a narrow brick alley under a broken sodium lamp",
    )


def _speaking(*keys: str) -> VideoPrompt:
    return VideoPrompt(
        storyboard=Storyboard(
            duration=10,
            beats=[
                Beat(
                    0,
                    5,
                    "a two-shot",
                    "they face each other",
                    lines=[
                        Line(speakers=[key], text=f"Line from {key}.") for key in keys
                    ],
                ),
                Beat(5, 10, "a close-up", "rain runs off the gutter"),
            ],
        ),
        style="Live-action, cinematic",
        audio="rain on brick",
    )


class TestIds:
    def test_a_minted_id_is_recognised(self):
        assert looks_like_id(mint_id())

    @pytest.mark.parametrize("key", ["baker", "", "cast_", "cast_xyz", "cast_1111111"])
    def test_a_hand_written_key_is_not_an_id(self, key):
        assert not looks_like_id(key)

    def test_ids_are_unique(self):
        assert len({mint_id() for _ in range(200)}) == 200


class TestIdentity:
    def test_appearance_and_voice_join(self, mara):
        assert mara.identity() == (
            "a tall woman in a rain-darkened red coat, a low, unhurried voice"
        )

    def test_appearance_alone_is_enough(self, ivo):
        assert ivo.identity() == "a wiry man with a shaved head"

    def test_an_empty_member_identifies_as_nothing(self):
        assert Member(name="Nobody").identity() == ""


class TestAppearing:
    def test_speakers_come_in_dialogue_order(self, mara, ivo):
        cast = Cast([ivo, mara])  # stored the other way round on purpose
        prompt = _speaking("cast_11111111", "cast_22222222")
        assert [m.name for m in appearing(prompt, cast)] == ["Mara", "Ivo"]

    def test_silent_members_follow_the_speakers(self, mara, alley):
        prompt = _speaking("cast_11111111")
        prompt.cast = ["cast_33333333"]
        cast = Cast([mara, alley])
        assert [m.name for m in appearing(prompt, cast)] == ["Mara", "The alley"]

    def test_a_member_named_twice_appears_once(self, mara):
        prompt = _speaking("cast_11111111")
        prompt.cast = ["cast_11111111"]
        assert len(appearing(prompt, Cast([mara]))) == 1

    def test_no_cast_means_nobody(self, mara):
        assert appearing(_speaking("cast_11111111"), None) == []

    def test_a_wordless_line_does_not_put_anyone_on_screen(self, mara):
        prompt = _speaking("cast_11111111")
        prompt.storyboard.beats[0].lines[0].text = ""
        assert appearing(prompt, Cast([mara])) == []


class TestResolve:
    def test_the_first_line_is_identified_from_the_member(self, mara):
        text = compose(_speaking("cast_11111111"), Cast([mara]))
        assert "tall woman in a rain-darkened red coat, a low, unhurried voice" in text

    def test_a_later_line_is_not_identified_again(self, mara):
        prompt = _speaking("cast_11111111")
        prompt.storyboard.beats[1].lines = [
            Line(speakers=["cast_11111111"], text="Still here.")
        ]
        text = compose(prompt, Cast([mara]))
        assert text.count("tall woman in a rain-darkened red coat") == 1

    def test_a_describe_the_author_wrote_wins(self, mara):
        prompt = _speaking("cast_11111111")
        prompt.storyboard.beats[0].lines[0].describe = "a stranger in the doorway"
        text = compose(prompt, Cast([mara]))
        assert "stranger in the doorway" in text
        assert "rain-darkened red coat" not in text

    def test_a_silent_member_is_stated_in_the_opening(self, alley):
        prompt = _speaking()
        prompt.cast = ["cast_33333333"]
        text = compose(prompt, Cast([alley]))
        description = text.split("\n\n")[0]
        # Its own sentence in the opening, so it is capitalised like one.
        assert "A narrow brick alley under a broken sodium lamp" in description
        assert description.index("sodium lamp") < description.index("[Shot 1]") + len(
            "[Shot 1]"
        ) + 200

    def test_a_speaking_member_is_not_also_stated_silently(self, mara):
        prompt = _speaking("cast_11111111")
        prompt.cast = ["cast_11111111"]
        text = compose(prompt, Cast([mara]))
        assert text.count("rain-darkened red coat") == 1

    def test_identity_files_reach_the_references(self, mara):
        mara.references = [Reference("image", "mara_front.png", retention="fully_preserved")]
        prompt = _speaking("cast_11111111")
        resolved = resolve(prompt, Cast([mara]))
        assert [r.path for r in resolved.references.items] == ["mara_front.png"]
        assert resolved.references.items[0].role == mara.identity()

    def test_a_reference_the_prompt_already_has_keeps_its_own_role(self, mara):
        mara.references = [Reference("image", "mara_front.png", role="from the cast")]
        prompt = _speaking("cast_11111111")
        prompt.references = ReferenceSet(
            [Reference("image", "mara_front.png", role="this shot only")]
        )
        resolved = resolve(prompt, Cast([mara]))
        assert len(resolved.references.items) == 1
        assert resolved.references.items[0].role == "this shot only"

    def test_resolution_never_mutates_the_caller(self, mara):
        prompt = _speaking("cast_11111111")
        resolve(prompt, Cast([mara]))
        assert prompt.storyboard.beats[0].lines[0].describe == ""

    def test_a_castless_prompt_is_returned_untouched(self, mara):
        prompt = _speaking("baker")
        assert resolve(prompt, Cast([mara])) is prompt

    def test_editing_a_member_changes_the_composed_text(self, mara):
        prompt = _speaking("cast_11111111")
        before = compose(prompt, Cast([mara]))
        mara.appearance = "a tall woman in a bottle-green parka"
        after = compose(prompt, Cast([mara]))
        assert "bottle-green parka" in after
        assert "red coat" not in after
        assert before != after


class TestUndescribed:
    """Keeping a member's identity files without their establishing paragraph."""

    def test_a_silent_member_is_normally_stated(self, alley):
        prompt = _speaking()
        prompt.cast = ["cast_33333333"]
        assert "narrow brick alley" in compose(prompt, Cast([alley]))

    def test_listing_them_undescribed_drops_the_prose(self, alley):
        prompt = _speaking()
        prompt.cast = ["cast_33333333"]
        prompt.undescribed = ["cast_33333333"]
        assert "narrow brick alley" not in compose(prompt, Cast([alley]))

    def test_their_identity_files_still_reach_the_model(self, alley):
        alley.references = [Reference("image", "alley.png", role="the alley plate")]
        prompt = _speaking()
        prompt.cast = ["cast_33333333"]
        prompt.undescribed = ["cast_33333333"]
        resolved = resolve(prompt, Cast([alley]))
        assert [r.path for r in resolved.references.items] == ["alley.png"]

    def test_the_authors_own_overview_survives(self, alley):
        prompt = _speaking()
        prompt.overview = "a chase at dusk"
        prompt.cast = ["cast_33333333"]
        prompt.undescribed = ["cast_33333333"]
        assert "a chase at dusk" in compose(prompt, Cast([alley]))

    def test_one_member_can_be_quiet_while_another_speaks_up(self, mara, alley):
        prompt = _speaking()
        prompt.cast = ["cast_11111111", "cast_33333333"]
        prompt.undescribed = ["cast_33333333"]
        text = compose(prompt, Cast([mara, alley]))
        assert "rain-darkened red coat" in text
        assert "narrow brick alley" not in text

    def test_a_speaker_is_unaffected_by_the_listing(self, mara):
        """Speakers are identified on their line, not in the opening."""
        prompt = _speaking("cast_11111111")
        prompt.undescribed = ["cast_11111111"]
        assert "rain-darkened red coat" in compose(prompt, Cast([mara]))

    def test_it_round_trips(self, alley):
        prompt = _speaking()
        prompt.undescribed = ["cast_33333333"]
        assert prompt_from_dict(prompt_to_dict(prompt)).undescribed == ["cast_33333333"]

    def test_a_prompt_written_before_it_existed_describes_as_before(self, alley):
        assert prompt_from_dict({"cast": ["cast_33333333"]}).undescribed == []

    # -- the same tick, for a member whose files make them a subject ---------
    #
    # The three places a description lands are the opening, the subject line
    # and the retention note. Silencing only the first is what made the tick
    # look broken: the words came straight back, twice.

    def _pictured(self, alley: Member) -> tuple[VideoPrompt, Cast]:
        alley.references = [
            Reference("image", "alley_front.png"),
            Reference("image", "alley_side.png"),
        ]
        prompt = _speaking()
        prompt.cast = ["cast_33333333"]
        prompt.undescribed = ["cast_33333333"]
        return prompt, Cast([alley])

    def test_the_subject_line_loses_the_description_too(self, alley):
        prompt, cast = self._pictured(alley)
        assert "narrow brick alley" not in compose(prompt, cast)

    def test_but_the_subject_is_still_defined(self, alley):
        prompt, cast = self._pictured(alley)
        assert "<Subject 1> is the setting in <Picture 1> and <Picture 2>." in compose(
            prompt, cast)

    def test_the_retention_marker_survives_the_silence(self, alley):
        """The marker is the instruction; the note only says what it applies to."""
        assert "fully_preserved - the setting." in compose(*self._pictured(alley))

    def test_a_role_written_for_this_segment_still_wins(self, alley):
        prompt, cast = self._pictured(alley)
        prompt.subjects = SubjectSet([
            Subject(member="cast_33333333", role="the alley, seen from the mouth"),
        ])
        text = compose(prompt, cast)
        assert "the alley, seen from the mouth" in text
        assert "narrow brick alley" not in text

    def test_describing_them_again_puts_every_place_back(self, alley):
        prompt, cast = self._pictured(alley)
        prompt.undescribed = []
        text = compose(prompt, cast)
        assert text.count("narrow brick alley") == 3


class TestLinkedReferences:
    """A media row that points at a member's identity file instead of copying it."""

    @pytest.fixture
    def mara(self, mara) -> Member:
        mara.references = [
            Reference("image", "mara_front.png", role="her full-body sheet",
                      retention="fully_preserved"),
            Reference("image", "mara_face.png", retention="fully_preserved"),
        ]
        return mara

    def _linked(self, member_id: str, slot: int, **over) -> VideoPrompt:
        prompt = _speaking("cast_11111111")
        prompt.references = ReferenceSet(
            [Reference("image", over.pop("path", ""), member=member_id, slot=slot, **over)]
        )
        return prompt

    def test_the_retention_set_here_wins_over_the_members(self, mara):
        prompt = self._linked("cast_11111111", 1, retention="attribute_transfer")
        resolved = resolve(prompt, Cast([mara]))
        assert [(r.path, r.retention) for r in resolved.references.items] == [
            ("mara_front.png", "fully_preserved"),
            ("mara_face.png", "attribute_transfer"),
        ]

    def test_an_empty_override_inherits_the_members_own(self, mara):
        resolved = resolve(self._linked("cast_11111111", 0), Cast([mara]))
        first = resolved.references.items[0]
        assert (first.role, first.retention) == ("her full-body sheet", "fully_preserved")

    def test_a_link_holds_its_place_so_tags_do_not_renumber(self, mara):
        plain = resolve(_speaking("cast_11111111"), Cast([mara]))
        linked = resolve(self._linked("cast_11111111", 1, role="just her face"), Cast([mara]))
        assert [r.path for r in plain.references.items] == [
            r.path for r in linked.references.items
        ]
        assert linked.references.tag(linked.references.items[1]) == "<Picture 2>"

    def test_swapping_the_members_file_flows_through_the_link(self, mara):
        prompt = self._linked("cast_11111111", 0, path="mara_front.png")
        mara.references[0].path = "mara_front_v2.png"
        resolved = resolve(prompt, Cast([mara]))
        assert [r.path for r in resolved.references.items] == [
            "mara_front_v2.png",
            "mara_face.png",
        ]

    def test_a_link_puts_its_member_on_screen_without_a_cast_entry(self, mara):
        prompt = VideoPrompt(Storyboard(duration=5), references=ReferenceSet(
            [Reference("image", "", member="cast_11111111", slot=0)]
        ))
        assert [m.id for m in appearing(prompt, Cast([mara]))] == ["cast_11111111"]
        assert "rain-darkened red coat" in compose(prompt, Cast([mara]))

    def test_a_link_to_a_deleted_member_is_dropped(self, mara):
        prompt = self._linked("cast_99999999", 0, path="ghost.png")
        assert [r.path for r in resolve(prompt, Cast([mara])).references.items] == [
            "mara_front.png",
            "mara_face.png",
        ]

    def test_a_link_to_a_deleted_member_is_reported_as_unknown(self, mara):
        prompt = self._linked("cast_99999999", 0)
        assert unknown_ids(prompt, Cast([mara])) == ["cast_99999999"]

    def test_a_link_past_the_end_of_the_members_files_is_dropped(self, mara):
        prompt = self._linked("cast_11111111", 7, path="ghost.png")
        assert [r.path for r in resolve(prompt, Cast([mara])).references.items] == [
            "mara_front.png",
            "mara_face.png",
        ]

    def test_a_stale_link_alone_leaves_nothing_behind(self, mara):
        """No member on screen, so the dead row must not still force ref2v."""
        prompt = VideoPrompt(Storyboard(duration=5), references=ReferenceSet(
            [Reference("image", "ghost.png", member="cast_99999999", slot=0)]
        ))
        assert resolve(prompt, Cast([mara])).references.items == []

    def test_a_plain_upload_still_comes_first_and_still_overrides_by_path(self, mara):
        prompt = _speaking("cast_11111111")
        prompt.references = ReferenceSet(
            [Reference("image", "mara_face.png", role="this shot only")]
        )
        resolved = resolve(prompt, Cast([mara]))
        assert [(r.path, r.role) for r in resolved.references.items] == [
            ("mara_face.png", "this shot only"),
            ("mara_front.png", "her full-body sheet"),
        ]

    def test_a_link_survives_the_round_trip_through_json(self, mara):
        prompt = self._linked("cast_11111111", 1, retention="weak_reference")
        back = prompt_from_dict(prompt_to_dict(prompt)).references.items[0]
        assert (back.member, back.slot, back.retention) == (
            "cast_11111111", 1, "weak_reference",
        )

    def test_a_reference_written_before_links_existed_loads_unlinked(self):
        back = prompt_from_dict(
            {"references": [{"kind": "image", "path": "old.png", "role": "r"}]}
        ).references.items[0]
        assert not back.is_linked
        assert (back.member, back.slot) == ("", 0)


class TestSpeakerNumbering:
    def test_members_still_map_to_s_ids_in_dialogue_order(self, mara, ivo):
        text = compose(_speaking("cast_11111111", "cast_22222222"), Cast([mara, ivo]))
        assert text.index("(S1)") < text.index("(S2)")

    def test_names_never_reach_the_prompt(self, mara, ivo):
        text = compose(_speaking("cast_11111111", "cast_22222222"), Cast([mara, ivo]))
        assert "Mara" not in text and "Ivo" not in text

    def test_the_identity_guard_still_fires_for_cast_speakers(self, mara, ivo):
        text = compose(_speaking("cast_11111111", "cast_22222222"), Cast([mara, ivo]))
        assert "Keep (S1) and (S2) visually distinct" in text


class TestUnknownIds:
    def test_a_deleted_member_is_reported(self, mara):
        prompt = _speaking("cast_11111111")
        assert unknown_ids(prompt, Cast()) == ["cast_11111111"]

    def test_a_hand_written_speaker_key_is_not_reported(self):
        assert unknown_ids(_speaking("baker"), Cast()) == []

    def test_a_silent_member_that_is_gone_is_reported(self):
        prompt = _speaking()
        prompt.cast = ["cast_33333333"]
        assert unknown_ids(prompt, Cast()) == ["cast_33333333"]

    def test_lint_warns_about_it(self, mara):
        prompt = _speaking("cast_11111111")
        rules = {finding.rule for finding in lint(prompt, Cast())}
        assert "cast.unknown" in rules

    def test_lint_without_a_cast_says_nothing(self, mara):
        prompt = _speaking("cast_11111111")
        rules = {finding.rule for finding in lint(prompt)}
        assert "cast.unknown" not in rules

    def test_a_member_that_exists_says_nothing(self, mara):
        prompt = _speaking("cast_11111111")
        rules = {finding.rule for finding in lint(prompt, Cast([mara]))}
        assert "cast.unknown" not in rules


class TestStorage:
    def test_a_member_round_trips(self, mara):
        mara.references = [Reference("image", "a.png", "the hero", "fully_preserved")]
        assert Member.from_dict(mara.to_dict()) == mara

    def test_a_cast_round_trips(self, mara, alley):
        cast = Cast([mara, alley])
        assert Cast.from_dict(cast.to_dict()) == cast

    def test_an_unknown_kind_falls_back_to_character(self):
        assert Member.from_dict({"kind": "villain"}).kind == "character"

    def test_a_member_without_an_id_gets_one(self):
        assert looks_like_id(Member.from_dict({"name": "Mara"}).id)

    def test_the_prompt_cast_list_round_trips(self):
        prompt = VideoPrompt.blank()
        prompt.cast = ["cast_11111111", "cast_33333333"]
        assert prompt_from_dict(prompt_to_dict(prompt)).cast == prompt.cast

    def test_an_old_prompt_loads_with_an_empty_cast(self):
        assert prompt_from_dict({"duration": 10}).cast == []
