"""Image prompt history, and writing one with a language model.

The two belong in one file because neither is much use without the other: a
rewrite of an image prompt replaces the whole box — there are no fields to
merge within, the way `llm.merge` protects a segment prompt — so the undo *is*
the safety net, and testing the rewrite without it would be testing half a
feature.
"""

from __future__ import annotations

import pytest

from video_prompter import PresetLibrary, ProfileStore, ProjectStore, bundled_root
from video_prompter import llm, paths
from video_prompter.api import Api, ApiError
from video_prompter.image_prompts import REVISION_LIMIT, ImagePromptStore


class Fake:
    """A provider that answers from a script, and remembers what it was asked."""

    name = "fake"
    model = "fake-1"

    def __init__(self, *answers: dict) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str, dict]] = []

    def complete(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append((system, user, schema))
        if not self.answers:
            raise AssertionError("The provider was asked one more time than scripted.")
        return self.answers.pop(0)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Keep "is a key configured?" from depending on the machine. See test_llm."""
    monkeypatch.setenv("VIDEO_PROMPTER_HOME", str(tmp_path / "home"))
    for name in llm.KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(llm.MODEL_VAR, raising=False)


@pytest.fixture
def store(tmp_path):
    projects = ProjectStore(tmp_path / "projects")
    projects.create("Rooftop", [])
    return ImagePromptStore(projects)


@pytest.fixture
def api(tmp_path):
    def build(provider=None):
        built = Api(
            projects=ProjectStore(tmp_path / "projects"),
            profiles=ProfileStore(tmp_path / "profiles"),
            presets=PresetLibrary(tmp_path / "presets", bundled_root()),
            provider=provider,
        )
        built.projects.create("Rooftop", [])
        return built

    return build


def status_of(fn, *args) -> int:
    with pytest.raises(ApiError) as raised:
        fn(*args)
    return raised.value.status


class TestTheTrail:
    def test_a_new_project_has_no_history(self, store):
        assert store.list("Rooftop") == []

    def test_filing_then_listing(self, store):
        store.file("Rooftop", "a rain-slicked alley")
        assert [r.text for r in store.list("Rooftop")] == ["a rain-slicked alley"]

    def test_newest_first(self, store):
        store.file("Rooftop", "first")
        store.file("Rooftop", "second")
        assert [r.text for r in store.list("Rooftop")] == ["second", "first"]

    def test_the_note_and_target_ride_through(self, store):
        store.file("Rooftop", "an alley", note="fake-1: colder", target="first_frame")
        filed = store.list("Rooftop")[0]
        assert filed.note == "fake-1: colder"
        assert filed.target == "first_frame"

    def test_an_empty_box_files_nothing(self, store):
        assert store.file("Rooftop", "   ") is None
        assert store.list("Rooftop") == []

    def test_the_same_words_twice_file_once(self, store):
        assert store.file("Rooftop", "an alley") is not None
        # A sweep of eight seeds from one prompt must not push eight entries.
        assert store.file("Rooftop", "an alley") is None
        assert store.file("Rooftop", "  an alley  ") is None
        assert len(store.list("Rooftop")) == 1

    def test_the_same_words_again_later_do_file(self, store):
        """Only the newest is compared: coming back to a prompt is a real edit."""
        store.file("Rooftop", "an alley")
        store.file("Rooftop", "a rooftop")
        assert store.file("Rooftop", "an alley") is not None
        assert [r.text for r in store.list("Rooftop")] == ["an alley", "a rooftop", "an alley"]

    def test_the_oldest_goes_first_at_the_limit(self, store):
        for n in range(REVISION_LIMIT + 5):
            store.file("Rooftop", f"prompt {n}")
        kept = store.list("Rooftop")
        assert len(kept) == REVISION_LIMIT
        assert kept[0].text == f"prompt {REVISION_LIMIT + 4}"
        assert kept[-1].text == "prompt 5"

    def test_a_revision_is_findable_by_id(self, store):
        filed = store.file("Rooftop", "an alley")
        assert store.find("Rooftop", filed.id).text == "an alley"
        assert store.find("Rooftop", "img_nope") is None

    def test_clearing_drops_the_trail(self, store):
        store.file("Rooftop", "one")
        store.file("Rooftop", "two")
        assert store.clear("Rooftop") == 2
        assert store.list("Rooftop") == []

    def test_a_damaged_file_reads_as_empty(self, store):
        store.file("Rooftop", "an alley")
        store.path_for("Rooftop").write_text("{not json", encoding="utf-8")
        # A broken side-car must never make a project unopenable.
        assert store.list("Rooftop") == []


class TestTheRoutes:
    def test_listing_comes_back_through_the_api(self, api):
        built = api()
        built.image_prompts.file("Rooftop", "an alley", note="by hand")
        answer = built.image_prompt_revisions("Rooftop")
        assert [r["text"] for r in answer["revisions"]] == ["an alley"]
        assert answer["revisions"][0]["note"] == "by hand"

    def test_filing_through_the_api(self, api):
        built = api()
        answer = built.file_image_prompt(
            {"project": "Rooftop", "text": "an alley", "target": "first_frame"}
        )
        assert answer["revision"]["text"] == "an alley"
        assert answer["revision"]["target"] == "first_frame"

    def test_filing_nothing_answers_null_rather_than_failing(self, api):
        built = api()
        assert built.file_image_prompt({"project": "Rooftop", "text": ""})["revision"] is None

    def test_filing_needs_a_project(self, api):
        assert status_of(api().file_image_prompt, {"text": "an alley"}) == 400


class TestWritingOne:
    def test_the_schema_holds_a_prompt_and_nothing_else(self, api):
        """The one guard rail that survives: it cannot answer with commentary."""
        fake = Fake({"prompt": "a rain-slicked alley at night"})
        api(fake).llm_draft_image({"instruction": "an alley at night"})
        _, _, schema = fake.calls[0]
        assert list(schema["properties"]) == ["prompt"]
        assert schema["required"] == ["prompt"]

    def test_writing_from_an_empty_box(self, api):
        fake = Fake({"prompt": "a rain-slicked alley at night"})
        answer = api(fake).llm_draft_image({"instruction": "an alley at night"})
        assert answer["prompt"] == "a rain-slicked alley at night"
        assert answer["before"] == ""
        assert answer["changed"] is True
        _, turn, _ = fake.calls[0]
        assert "The box is empty" in turn

    def test_rewriting_shows_the_model_what_is_there(self, api):
        fake = Fake({"prompt": "a rain-slicked alley at dawn"})
        answer = api(fake).llm_draft_image(
            {"prompt": "a rain-slicked alley at night", "instruction": "make it dawn"}
        )
        assert answer["before"] == "a rain-slicked alley at night"
        assert answer["prompt"] == "a rain-slicked alley at dawn"
        _, turn, _ = fake.calls[0]
        assert "a rain-slicked alley at night" in turn
        assert "make it dawn" in turn

    def test_the_look_control_is_named_so_the_model_leaves_the_grade_alone(self, api):
        fake = Fake({"prompt": "an alley"})
        api(fake).llm_draft_image({"instruction": "an alley", "style": "neon-noir"})
        _, turn, _ = fake.calls[0]
        assert "neon-noir" in turn
        assert "Leave the grade to it" in turn

    def test_an_unchanged_answer_says_so(self, api):
        same = "a rain-slicked alley"
        fake = Fake({"prompt": same})
        answer = api(fake).llm_draft_image({"prompt": same, "instruction": "tighten it"})
        assert answer["changed"] is False

    def test_an_empty_instruction_is_refused_before_the_model_is_asked(self, api):
        fake = Fake()
        assert status_of(api(fake).llm_draft_image, {"instruction": "  "}) == 400
        assert fake.calls == []

    def test_a_missing_instruction_is_refused(self, api):
        assert status_of(api(Fake()).llm_draft_image, {"prompt": "an alley"}) == 400

    def test_an_empty_answer_is_an_error_not_an_empty_box(self, api):
        """Wiping the box because the model shrugged would be the worst outcome."""
        assert status_of(
            api(Fake({"prompt": "   "})).llm_draft_image, {"instruction": "an alley"}
        ) == 502

    def test_no_model_key_says_so(self, api):
        assert status_of(api().llm_draft_image, {"instruction": "an alley"}) == 503

    def test_one_turn_only(self, api):
        """No linter for an image prompt, so a repair turn has nothing to check."""
        fake = Fake({"prompt": "an alley"})
        api(fake).llm_draft_image({"instruction": "an alley"})
        assert len(fake.calls) == 1


class TestTheRewriteIsUndoable:
    def test_naming_a_project_files_the_outgoing_prompt(self, api):
        built = api(Fake({"prompt": "a rain-slicked alley at dawn"}))
        answer = built.llm_draft_image({
            "project": "Rooftop",
            "prompt": "a rain-slicked alley at night",
            "instruction": "make it dawn",
            "target": "first_frame",
        })
        assert answer["filed"]["text"] == "a rain-slicked alley at night"
        kept = built.image_prompts.list("Rooftop")
        assert [r.text for r in kept] == ["a rain-slicked alley at night"]
        # The note says where the replacement came from, not what this text said.
        assert kept[0].note == "fake-1: make it dawn"
        assert kept[0].target == "first_frame"

    def test_leaving_the_project_off_files_nothing(self, api):
        """An agent comparing options should not litter the history."""
        built = api(Fake({"prompt": "at dawn"}))
        answer = built.llm_draft_image({"prompt": "at night", "instruction": "dawn"})
        assert "filed" not in answer
        assert built.image_prompts.list("Rooftop") == []

    def test_writing_into_an_empty_box_files_nothing(self, api):
        """There was no previous version to lose."""
        built = api(Fake({"prompt": "a rain-slicked alley"}))
        built.llm_draft_image({"project": "Rooftop", "instruction": "an alley"})
        assert built.image_prompts.list("Rooftop") == []

    def test_an_unchanged_rewrite_files_nothing(self, api):
        same = "a rain-slicked alley"
        built = api(Fake({"prompt": same}))
        built.llm_draft_image(
            {"project": "Rooftop", "prompt": same, "instruction": "tighten"}
        )
        assert built.image_prompts.list("Rooftop") == []

    def test_a_rewrite_can_be_walked_back_to(self, api):
        """The whole point: two rewrites, and the original is still reachable."""
        built = api(Fake({"prompt": "at dawn"}, {"prompt": "at noon"}))
        built.llm_draft_image(
            {"project": "Rooftop", "prompt": "at night", "instruction": "dawn"}
        )
        built.llm_draft_image(
            {"project": "Rooftop", "prompt": "at dawn", "instruction": "noon"}
        )
        assert [r.text for r in built.image_prompts.list("Rooftop")] == ["at dawn", "at night"]
