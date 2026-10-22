from __future__ import annotations

import pytest

from video_prompter import (
    Beat,
    Cast,
    Line,
    Member,
    Reference,
    Storyboard,
    VideoPrompt,
    compose,
    lint,
)
from video_prompter import mentions
from video_prompter.cast import appearing, mention_label, mentioned, resolve


@pytest.fixture
def mara() -> Member:
    return Member(
        id="cast_11111111",
        name="Mara",
        kind="character",
        appearance="a tall woman in a rain-darkened red coat",
        references=[Reference(kind="image", path="mara.png")],
    )


@pytest.fixture
def ivo() -> Member:
    return Member(
        id="cast_22222222",
        name="Ivo",
        kind="character",
        appearance="a wiry man with a shaved head",
        references=[Reference(kind="image", path="ivo.png")],
    )


@pytest.fixture
def alley() -> Member:
    return Member(
        id="cast_33333333",
        name="The alley",
        kind="environment",
        appearance="a narrow brick alley under a broken sodium lamp",
        references=[Reference(kind="image", path="alley.png")],
    )


def _prompt(action: str, **fields: object) -> VideoPrompt:
    prompt = VideoPrompt(
        storyboard=Storyboard(
            duration=8,
            beats=[Beat(0, 8, angle="a medium shot", action=action)],
        )
    )
    for key, value in fields.items():
        setattr(prompt, key, value)
    return prompt


def _names(*members: Member) -> list[tuple[str, str]]:
    return [(member.name, member.id) for member in members]


# -- the syntax ------------------------------------------------------------


def test_find_matches_a_name_case_insensitively(mara):
    hits = mentions.find("@mara waits.", _names(mara))
    assert [(hit.start, hit.end, hit.key) for hit in hits] == [(0, 5, mara.id)]


def test_find_prefers_the_longest_name(mara):
    chen = Member(id="cast_44444444", name="Mara Chen")
    hits = mentions.find("@Mara Chen waits.", _names(mara, chen))
    assert [hit.key for hit in hits] == [chen.id]
    assert hits[0].end == len("@Mara Chen")


def test_find_stops_at_a_word_boundary(mara):
    """`@Mara` must not match inside `@Marathon`."""
    assert mentions.find("@Marathon day.", _names(mara)) == []
    assert len(mentions.find("@Mara's coat.", _names(mara))) == 1
    assert len(mentions.find("(@Mara)", _names(mara))) == 1


def test_find_skips_an_email_address(mara):
    assert mentions.find("write to mara@mara.com", _names(mara)) == []


def test_rewrite_replaces_only_the_mention(mara):
    text = "Mara waits. @Mara turns."
    assert mentions.rewrite(text, _names(mara), lambda hit: "<Subject 1>") == (
        "Mara waits. <Subject 1> turns."
    )


def test_unknown_reports_an_at_word_nobody_answers_to(mara):
    assert mentions.unknown("@Mara and @Mars.", _names(mara)) == ["Mars"]
    assert mentions.unknown("mail sales@example.com", _names(mara)) == []


def test_fields_skips_spoken_words_and_screen_text(mara):
    """A character saying another's name must not become a routing label."""
    prompt = _prompt("@Mara turns.", screen_text=["@Mara"])
    prompt.storyboard.beats[0].lines = [Line(speakers=["x"], text="@Mara, wait!")]
    addresses = {field.address for field in mentions.fields(prompt)}
    assert "beat.0.action" in addresses
    assert not [address for address in addresses if address.startswith("screen_text")]
    assert not [address for address in addresses if address.endswith(".text")]


# -- what a mention becomes ------------------------------------------------


def test_a_mention_composes_to_the_members_subject_tag(mara, ivo):
    prompt = _prompt("@Mara sets the mug down as @Ivo looks away.")
    text = compose(prompt, Cast([mara, ivo]))
    assert "<Subject 1> sets the mug down as <Subject 2> looks away." in text
    assert "@Mara" not in text


def test_the_tag_follows_the_shot_when_somebody_joins_it(mara, ivo):
    """The whole point: renumbering costs nothing, because no number was typed.

    Ivo is the only one in the shot, so he is `<Subject 1>`. Put Mara on screen
    ahead of him and every number after her moves — and the prose moves with it,
    which is exactly what a typed `<Subject 1>` would not have done.
    """
    prompt = _prompt("@Ivo waits.")
    assert "<Subject 1> waits." in compose(prompt, Cast([mara, ivo]))
    prompt.cast = [mara.id]
    assert "<Subject 2> waits." in compose(prompt, Cast([mara, ivo]))


def test_a_speaker_with_no_files_composes_to_their_id(mara):
    voice = Member(id="cast_55555555", name="Narrator", kind="voice")
    prompt = _prompt("@Narrator is heard over @Mara.")
    prompt.storyboard.beats[0].lines = [Line(speakers=[voice.id], text="It began here.")]
    text = compose(prompt, Cast([mara, voice]))
    assert "the speaker (S1) is heard over <Subject 1>." in text


def test_a_member_with_no_label_falls_back_to_their_noun(alley):
    bare = Member(id="cast_66666666", name="Ivo", appearance="a wiry man")
    prompt = _prompt("@Ivo steps into @The alley.")
    text = compose(prompt, Cast([bare, alley]))
    assert "the person steps into <Subject 1>." in text


def test_an_unknown_mention_is_left_exactly_as_written(mara):
    prompt = _prompt("@Mars waits for @Mara.")
    assert "@Mars waits for <Subject 1>." in compose(prompt, Cast([mara]))


def test_a_mention_reaches_every_prose_field(mara):
    prompt = _prompt("nothing", overview="Rain on @Mara.", audio="@Mara breathes.")
    text = compose(prompt, Cast([mara]))
    assert "Rain on <Subject 1>" in text
    assert "<Subject 1> breathes" in text


def test_mentioning_a_member_puts_them_on_screen(mara):
    """A mention is a pointer at a label, so it has to mint one."""
    prompt = _prompt("@Mara waits.")
    assert appearing(prompt, Cast([mara])) == [mara]
    assert mentioned(prompt, Cast([mara])) == [mara]
    assert "mara.png" in [item.path for item in resolve(prompt, Cast([mara])).references.items]


def test_a_mention_does_not_renumber_a_prompt_that_already_had_a_cast(mara, ivo):
    """Mentioned members are appended, so existing tags keep their meaning."""
    prompt = _prompt("@Ivo waits.", cast=[mara.id])
    resolved = resolve(prompt, Cast([mara, ivo]))
    assert resolved.subjects.tag(resolved.subjects.for_member(mara.id)) == "<Subject 1>"
    assert resolved.subjects.tag(resolved.subjects.for_member(ivo.id)) == "<Subject 2>"


def test_mention_label_reports_what_the_editor_should_show(mara, ivo):
    resolved = resolve(_prompt("@Mara waits."), Cast([mara, ivo]))
    assert mention_label(resolved, mara) == "<Subject 1>"
    assert mention_label(resolved, ivo) == "the person"


def test_resolving_twice_is_stable(mara):
    """Nothing is left for a second pass to find, and nothing is double-expanded."""
    cast = Cast([mara])
    once = resolve(_prompt("@Mara waits."), cast)
    assert resolve(once, cast).storyboard.beats[0].action == once.storyboard.beats[0].action


# -- renaming --------------------------------------------------------------


def test_rename_moves_the_mentions(mara):
    prompt = _prompt("@Mara waits.", overview="Rain on @Mara.")
    assert mentions.rename(prompt, "Mara", "Marisol") == 2
    assert prompt.storyboard.beats[0].action == "@Marisol waits."
    assert prompt.overview == "Rain on @Marisol."


def test_rename_leaves_the_bare_word_alone(mara):
    prompt = _prompt("Mara waits, and @Mara turns.")
    mentions.rename(prompt, "Mara", "Marisol")
    assert prompt.storyboard.beats[0].action == "Mara waits, and @Marisol turns."


def test_rename_is_a_no_op_without_a_change():
    prompt = _prompt("@Mara waits.")
    assert mentions.rename(prompt, "Mara", "Mara") == 0
    assert mentions.rename(prompt, "", "Marisol") == 0


# -- the findings ----------------------------------------------------------


def _rules(prompt: VideoPrompt, cast: Cast) -> set[str]:
    return {finding.rule for finding in lint(prompt, cast)}


def test_lint_flags_a_mention_that_names_nobody(mara):
    assert "mentions.unknown" in _rules(_prompt("@Mars waits."), Cast([mara]))
    assert "mentions.unknown" not in _rules(_prompt("@Mara waits."), Cast([mara]))


def test_lint_flags_a_mention_with_no_label_to_point_at():
    bare = Member(id="cast_66666666", name="Ivo", appearance="a wiry man")
    assert "mentions.untagged" in _rules(_prompt("@Ivo waits."), Cast([bare]))


def test_lint_flags_two_members_sharing_a_name(mara):
    twin = Member(id="cast_77777777", name="Mara", references=[Reference(kind="image", path="b.png")])
    assert "mentions.ambiguous" in _rules(_prompt("@Mara waits."), Cast([mara, twin]))


def test_lint_reports_a_repeated_mention_once(mara):
    bare = Member(id="cast_66666666", name="Ivo", appearance="a wiry man")
    prompt = _prompt("@Ivo waits. @Ivo turns. @Ivo leaves.")
    findings = [f for f in lint(prompt, Cast([bare])) if f.rule == "mentions.untagged"]
    assert len(findings) == 1
