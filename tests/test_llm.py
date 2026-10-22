"""Writing a prompt with a language model: the schema, the scope, the repair."""

from __future__ import annotations

import io
import json

import pytest

from video_prompter import PresetLibrary, ProfileStore, ProjectStore, VideoPrompt, bundled_root
from video_prompter import llm, paths
from video_prompter.api import Api, ApiError
from video_prompter.serialize import prompt_to_dict


class Fake:
    """A provider that answers from a script, and remembers what it was asked.

    Standing in for the network keeps these tests about the parts that are ours
    — the schema, the merge and the repair loop — rather than about Google.
    """

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


def a_prompt(duration: int = 10) -> dict:
    from video_prompter import Beat

    prompt = VideoPrompt.blank(duration)
    prompt.style = "A cinematic look."
    prompt.audio = "Rain taps the window."
    prompt.storyboard.beats = [
        Beat(0, 5, "wide", "a door opens"),
        Beat(5, duration, "close-up", "a hand on the latch"),
    ]
    return prompt_to_dict(prompt)


def an_answer(prompt: dict | None = None, **overrides) -> dict:
    """A model reply: the authored fields of a prompt, plus any overrides."""
    base = prompt or a_prompt()
    answer = {name: base.get(name) for name in llm.AUTHORED_FIELDS}
    answer.update(overrides)
    return answer


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point the settings file at a scratch directory, for every test here.

    `configured()` reads a real file now, so without this the answer to "is a
    key configured?" would depend on whether the machine running the tests
    happens to have one — passing on a laptop and failing in a checkout, or the
    other way round.
    """
    monkeypatch.setenv("VIDEO_PROMPTER_HOME", str(tmp_path / "home"))
    for name in llm.KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(llm.MODEL_VAR, raising=False)
    monkeypatch.delenv(llm.LOCAL_URL_VAR, raising=False)
    monkeypatch.delenv(llm.LOCAL_MODEL_VAR, raising=False)


def write_settings(contents: str) -> None:
    """Put a settings file where `configured()` will look for it."""
    path = paths.settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


@pytest.fixture
def api(tmp_path):
    def build(provider=None):
        return Api(
            projects=ProjectStore(tmp_path / "projects"),
            profiles=ProfileStore(tmp_path / "profiles"),
            presets=PresetLibrary(tmp_path / "presets", bundled_root()),
            provider=provider,
        )

    return build


def status_of(fn, *args) -> int:
    with pytest.raises(ApiError) as raised:
        fn(*args)
    return raised.value.status


class TestTheSchemaIsTheGuardRail:
    def test_only_the_authored_fields_can_come_back(self):
        # Media paths, the canvas and the unused negatives block are not in the
        # shape the model can answer in, so no instruction can reach them.
        properties = llm.response_schema()["properties"]
        assert set(properties) == set(llm.AUTHORED_FIELDS)
        for off_limits in ("references", "first_frame", "width", "negatives"):
            assert off_limits not in properties

    def test_every_authored_field_is_required(self):
        # A partial answer would merge as "delete the music"; an empty string
        # says that far more clearly than an absent key.
        assert set(llm.response_schema()["required"]) == set(llm.AUTHORED_FIELDS)

    def test_a_keyframe_survives_a_whole_prompt_rewrite(self, api):
        current = a_prompt()
        current["first_frame"] = "video_prompter/hero.png"
        fake = Fake(an_answer(style="Something new."))

        result = api(fake).llm_draft({"prompt": current, "instruction": "new look"})

        assert result["prompt"]["first_frame"] == "video_prompter/hero.png"
        assert result["prompt"]["style"] == "Something new."

    def test_references_survive_a_whole_prompt_rewrite(self, api):
        current = a_prompt()
        current["references"] = [
            {"kind": "image", "path": "x.png", "role": "the hero", "retention": ""}
        ]
        fake = Fake(an_answer(style="Something new."))

        result = api(fake).llm_draft({"prompt": current, "instruction": "new look"})

        assert result["prompt"]["references"][0]["role"] == "the hero"

    def test_the_briefing_is_built_from_the_published_contract(self):
        # Written out by hand it would be a second copy of the field rules, and
        # a second copy is a bug with a delay on it.
        from video_prompter import contract

        briefing = llm.system_prompt()
        rule = contract.schema()["lint_rules"][0]["rule"]
        assert rule in briefing


class TestScope:
    def test_all_reads_as_the_whole_document(self):
        assert llm.Scope.parse("all").fields == llm.AUTHORED_FIELDS
        assert llm.Scope.parse(None).fields == llm.AUTHORED_FIELDS

    def test_a_field_name_scopes_to_that_field(self):
        assert llm.Scope.parse("audio").fields == ("audio",)

    def test_a_list_scopes_to_those_fields(self):
        assert llm.Scope.parse(["audio", "music"]).fields == ("audio", "music")

    def test_a_beat_scope_names_the_shot_the_way_a_person_counts(self):
        scope = llm.Scope.parse("beat:1")
        assert scope.beat == 1
        assert "shot 2" in scope.describe()

    def test_a_field_that_cannot_be_written_is_refused(self):
        for bad in ("first_frame", "references", "nonsense"):
            with pytest.raises(llm.LlmError) as raised:
                llm.Scope.parse(bad)
            assert raised.value.status == 400

    def test_an_unreadable_beat_index_is_refused(self):
        with pytest.raises(llm.LlmError):
            llm.Scope.parse("beat:second")


class TestScopeIsEnforcedAtTheMerge:
    def test_a_narrow_scope_takes_only_that_field(self):
        # The model is shown everything, because "make this faster" may need the
        # spans as well as the phrasing — but only what was in scope lands.
        current = a_prompt()
        answered = an_answer(audio="A new soundscape.", style="A different look.")

        merged = llm.merge(current, answered, llm.Scope.parse("audio"))

        assert merged["audio"] == "A new soundscape."
        assert merged["style"] == "A cinematic look."

    def test_a_single_beat_scope_leaves_its_neighbours_alone(self):
        current = a_prompt()
        rewritten = [dict(b) for b in current["beats"]]
        rewritten[0]["action"] = "should not land"
        rewritten[1]["action"] = "a hand hesitates on the latch"
        answered = an_answer(beats=rewritten)

        merged = llm.merge(current, answered, llm.Scope.parse("beat:1"))

        assert merged["beats"][0]["action"] == "a door opens"
        assert merged["beats"][1]["action"] == "a hand hesitates on the latch"

    def test_a_single_beat_answer_lands_in_the_scoped_slot(self):
        # A model that answers with just the shot it was aimed at is obliging,
        # not wrong.
        current = a_prompt()
        answered = an_answer(beats=[{"start": 5, "end": 10, "angle": "close",
                                     "action": "the latch turns", "camera": "",
                                     "transition": "", "lines": []}])

        merged = llm.merge(current, answered, llm.Scope.parse("beat:1"))

        assert merged["beats"][1]["action"] == "the latch turns"
        assert merged["beats"][0]["action"] == "a door opens"

    def test_a_beat_scope_past_the_end_is_refused(self):
        with pytest.raises(llm.LlmError) as raised:
            llm.merge(a_prompt(), an_answer(), llm.Scope.parse("beat:9"))
        assert raised.value.status == 400

    def test_an_instant_cue_comes_back_as_the_open_end_the_tool_stores(self):
        # The wire schema has no nullable number, so the flag is said out loud.
        merged = llm.merge(
            a_prompt(),
            an_answer(cues=[{"start": 2, "end": 0, "instant": True, "description": "a hit"}]),
            llm.Scope.everything(),
        )
        assert merged["cues"][0]["end"] is None

    def test_a_sustained_cue_keeps_its_end(self):
        merged = llm.merge(
            a_prompt(),
            an_answer(cues=[{"start": 2, "end": 6, "description": "a bed of rain"}]),
            llm.Scope.everything(),
        )
        assert merged["cues"][0]["end"] == 6


class TestTheRepairLoop:
    def test_a_clean_draft_is_asked_for_once(self, api):
        fake = Fake(an_answer())
        result = api(fake).llm_draft({"prompt": a_prompt(), "instruction": "tighten it"})

        assert result["attempts"] == 1
        assert result["findings"] == []

    def test_a_draft_that_trips_the_linter_is_handed_back_with_the_findings(self, api):
        # The beats no longer tile the budget, which is an error the tool would
        # refuse at render time — so it is worth one more turn here.
        broken = an_answer(beats=[{"start": 0, "end": 3, "angle": "wide",
                                   "action": "a door opens", "camera": "",
                                   "transition": "", "lines": []}])
        fake = Fake(broken, an_answer())

        result = api(fake).llm_draft({"prompt": a_prompt(), "instruction": "fix"})

        assert result["attempts"] == 2
        assert result["findings"] == []
        assert "does not pass" in fake.calls[1][1]

    def test_a_draft_that_never_comes_clean_is_returned_with_its_findings(self, api):
        broken = an_answer(beats=[{"start": 0, "end": 3, "angle": "wide",
                                   "action": "a door opens", "camera": "",
                                   "transition": "", "lines": []}])
        fake = Fake(broken, broken, broken)

        result = api(fake).llm_draft({"prompt": a_prompt(), "instruction": "fix"})

        # Handed over honestly rather than passed off as clean: the editor can
        # show what is still wrong.
        assert result["attempts"] == llm.REPAIR_ATTEMPTS + 1
        assert result["findings"]

    def test_the_reply_says_which_fields_moved(self, api):
        fake = Fake(an_answer(music="A slow upright bass at a walking tempo."))
        result = api(fake).llm_draft({"prompt": a_prompt(), "instruction": "score it"})

        assert result["changed"] == ["music"]


class TestDraftingOverTheApi:
    def test_an_empty_instruction_is_a_bad_request(self, api):
        assert status_of(
            api(Fake()).llm_draft, {"prompt": a_prompt(), "instruction": "  "}
        ) == 400

    def test_a_draft_with_nothing_to_work_from_is_a_bad_request(self, api):
        assert status_of(api(Fake()).llm_draft, {"instruction": "a chase"}) == 400

    def test_no_key_configured_is_told_apart_from_a_bad_request(self, api):
        assert status_of(
            api(None).llm_draft, {"prompt": a_prompt(), "instruction": "a chase"}
        ) == 503

    def test_a_model_that_fails_keeps_the_status_it_chose(self, api):
        class Refuses:
            name, model = "fake", "fake-1"

            def complete(self, system, user, schema):
                raise llm.LlmError("Could not reach Gemini: timed out", 504)

        assert status_of(
            api(Refuses()).llm_draft, {"prompt": a_prompt(), "instruction": "x"}
        ) == 504

    def test_drafting_without_a_source_saves_nothing(self, api):
        result = api(Fake(an_answer())).llm_draft(
            {"prompt": a_prompt(), "instruction": "tighten it"}
        )
        assert "project" not in result


class TestApplyingToASource:
    @pytest.fixture
    def cut(self, api):
        def build(provider):
            built = api(provider)
            built.create_project({"name": "Cut"})
            built.add_source({"project": "Cut", "name": "Rooftop"})
            return built

        return build

    def source_of(self, built) -> dict:
        return built.load_project("Cut")["project"]["sources"][0]

    def test_naming_a_source_applies_the_draft(self, cut):
        built = cut(Fake(an_answer(style="A neon-noir look.")))
        source_id = self.source_of(built)["id"]

        built.llm_draft({
            "project": "Cut", "source": source_id, "instruction": "neon noir",
        })

        assert self.source_of(built)["segments"][0]["prompt"]["style"] == "A neon-noir look."

    def test_the_prompt_to_work_from_is_read_off_the_segment(self, cut):
        # The editor should not have to post the document back just to have it
        # rewritten; the server already has it.
        fake = Fake(an_answer())
        built = cut(fake)
        source_id = self.source_of(built)["id"]
        built.llm_draft({"project": "Cut", "source": source_id, "instruction": "x"})

        assert "There is no prompt yet" in fake.calls[0][1]

    def test_applying_files_a_revision_so_it_can_be_walked_back(self, cut):
        built = cut(Fake(an_answer(style="A neon-noir look.")))
        source_id = self.source_of(built)["id"]

        built.llm_draft({
            "project": "Cut", "source": source_id, "instruction": "make it neon noir",
        })

        history = built.segment_revisions("Cut", source_id)["revisions"]
        assert len(history) == 1
        assert history[0]["note"].startswith("fake-1: make it neon noir")

    def test_the_revision_holds_the_prompt_that_was_replaced(self, cut):
        built = cut(Fake(an_answer(style="First pass."), an_answer(style="Second pass.")))
        source_id = self.source_of(built)["id"]
        built.llm_draft({"project": "Cut", "source": source_id, "instruction": "one"})
        built.llm_draft({"project": "Cut", "source": source_id, "instruction": "two"})

        history = built.segment_revisions("Cut", source_id)["revisions"]
        assert history[0]["prompt"]["style"] == "First pass."

    def test_applying_answers_with_the_updated_project(self, cut):
        built = cut(Fake(an_answer(style="A neon-noir look.")))
        source_id = self.source_of(built)["id"]

        result = built.llm_draft({
            "project": "Cut", "source": source_id, "instruction": "neon noir",
        })

        assert result["project"]["sources"][0]["segments"][0]["prompt"]["style"] == (
            "A neon-noir look."
        )

    def test_a_source_that_is_not_there_is_a_not_found(self, cut):
        built = cut(Fake(an_answer()))
        assert status_of(
            built.llm_draft,
            {"project": "Cut", "source": "src_nope", "instruction": "x"},
        ) == 404


class TestStatus:
    def test_no_key_says_where_one_could_go(self, api):
        # The only useful thing to say when there is no key is where to put one.
        status = api(None).llm_status()

        assert status["configured"] is False
        assert "VIDEO_PROMPTER_GEMINI_KEY" in status["key_vars"]
        assert status["settings_file"].endswith("settings.json")
        assert status["settings_key"] == "gemini_key"

    def test_a_configured_provider_names_its_model(self, api):
        status = api(Fake()).llm_status()

        assert status["configured"] is True
        assert status["model"] == "fake-1"
        assert "audio" in status["scopes"]

    def test_the_model_list_is_not_fetched_unless_asked(self, api):
        class Counting(Fake):
            listed = 0

            def models(self):
                Counting.listed += 1
                return ["gemini-flash-latest"]

        provider = Counting()
        api(provider).llm_status()
        assert Counting.listed == 0
        assert api(provider).llm_status(probe=True)["models"] == ["gemini-flash-latest"]

    def test_a_key_in_the_environment_is_found(self, api, monkeypatch):
        monkeypatch.setenv("VIDEO_PROMPTER_GEMINI_KEY", "sk-not-a-real-key")
        assert api(None).llm_status()["configured"] is True

    def test_the_model_can_be_named_by_the_environment(self, monkeypatch):
        monkeypatch.setenv("VIDEO_PROMPTER_GEMINI_KEY", "sk-not-a-real-key")
        monkeypatch.setenv("VIDEO_PROMPTER_GEMINI_MODEL", "gemini-something-else")
        assert llm.provider_from_settings().model == "gemini-something-else"

    def test_no_key_anywhere_means_no_provider(self):
        assert llm.provider_from_settings() is None


class TestReadingGeminisReply:
    def build(self) -> llm.Gemini:
        return llm.Gemini(api_key="sk-not-a-real-key")

    def wrapped(self, document: dict, **extra) -> dict:
        return {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps(document)}]}, **extra}
            ]
        }

    def test_the_document_is_pulled_out_of_the_candidate(self):
        assert self.build()._read(self.wrapped({"style": "x"})) == {"style": "x"}

    def test_a_blocked_prompt_says_why(self):
        with pytest.raises(llm.LlmError) as raised:
            self.build()._read({"promptFeedback": {"blockReason": "SAFETY"}})
        assert "SAFETY" in str(raised.value)

    def test_no_candidates_at_all_is_an_error(self):
        with pytest.raises(llm.LlmError):
            self.build()._read({})

    def test_running_out_of_room_says_that_rather_than_bad_json(self):
        # The difference matters: one is retryable with a narrower scope, the
        # other is not retryable at all.
        truncated = {
            "candidates": [
                {"content": {"parts": [{"text": '{"style": "half a doc'}]},
                 "finishReason": "MAX_TOKENS"}
            ]
        }
        with pytest.raises(llm.LlmError) as raised:
            self.build()._read(truncated)
        assert "ran out of room" in str(raised.value)

    def test_an_answer_that_is_not_a_document_is_refused(self):
        with pytest.raises(llm.LlmError):
            self.build()._read(self.wrapped(["not", "a", "document"]))

    def test_the_key_travels_in_a_header_not_the_url(self):
        # A URL ends up in logs and proxies, and would take the secret with it.
        assert "sk-not-a-real-key" not in self.build()._url("models")


class TestWhichFindingsAreWorthATurn:
    def test_a_warning_the_model_caused_is_repaired(self):
        # A storyboard with a seven-second hole in it passes as a warning and
        # would still waste a render.
        after = [{"rule": "storyboard.gap", "message": "Nothing is scheduled for 7.0s."}]
        assert llm.actionable([], after) == after

    def test_a_finding_that_was_already_there_is_not_the_models_to_fix(self):
        # Handing it back with "change nothing else" burns a turn the real
        # mistakes needed.
        inherited = [{"rule": "style.missing", "message": "No style contract."}]
        assert llm.actionable(inherited, list(inherited)) == []

    def test_a_finding_about_a_field_the_model_cannot_write_is_left_alone(self):
        for rule in ("mode.conflict", "canvas.too_large", "references.unused"):
            assert llm.actionable([], [{"rule": rule, "message": "x"}]) == []

    def test_an_inherited_problem_still_comes_back_in_the_reply(self, api):
        # Not repaired, but not hidden either: the editor should say what is
        # wrong even when the rewrite is not what made it wrong.
        current = a_prompt()
        current["first_frame"] = "hero.png"
        current["references"] = [{"kind": "image", "path": "x.png", "role": "hero"}]

        result = api(Fake(an_answer()))\
            .llm_draft({"prompt": current, "instruction": "tighten"})

        assert result["attempts"] == 1
        assert any(f["rule"] == "mode.conflict" for f in result["findings"])


class TestTheSettingsFile:
    """The everyday path: ComfyUI's own Restart re-executes in place and keeps
    the environment it already had, so a key added to the launcher afterwards
    stays invisible. A file is read at each use, so a Restart is enough."""

    def test_a_key_in_the_file_is_found(self):
        write_settings('{"gemini_key": "sk-from-the-file"}')
        found = llm.configured()

        assert found.key == "sk-from-the-file"
        assert found.source == "settings file"

    def test_the_model_can_be_named_in_the_file(self):
        write_settings('{"gemini_key": "k", "gemini_model": "gemini-from-file"}')
        assert llm.provider_from_settings().model == "gemini-from-file"

    def test_the_environment_wins_where_both_are_set(self, monkeypatch):
        # The deliberate override beats the standing one, as everywhere else.
        write_settings('{"gemini_key": "sk-from-the-file"}')
        monkeypatch.setenv("VIDEO_PROMPTER_GEMINI_KEY", "sk-from-the-environment")
        found = llm.configured()

        assert found.key == "sk-from-the-environment"
        assert found.source == "environment"

    def test_status_says_which_of_the_two_is_in_force(self, api):
        write_settings('{"gemini_key": "sk-from-the-file"}')
        assert api(None).llm_status()["source"] == "settings file"

    def test_no_file_at_all_is_the_normal_case_not_an_error(self):
        assert llm.read_settings() == {}
        assert llm.configured().key == ""

    def test_a_malformed_file_costs_the_model_not_the_editor(self):
        # Losing the whole node to a stray comma in an optional file would be a
        # far worse trade than losing the model.
        write_settings("{ this is not json")
        assert llm.read_settings() == {}
        assert llm.provider_from_settings() is None

    def test_a_file_holding_something_other_than_an_object_is_ignored(self):
        write_settings('["a list"]')
        assert llm.read_settings() == {}

    def test_a_blank_key_in_the_file_counts_as_no_key(self):
        write_settings('{"gemini_key": "   "}')
        assert llm.provider_from_settings() is None

    def test_the_default_model_stands_when_the_file_names_none(self):
        write_settings('{"gemini_key": "k"}')
        assert llm.configured().model == llm.DEFAULT_MODEL


class TestTheSchemaTheApiWillAccept:
    """Constraints the Generative Language API enforces on `responseSchema`.

    Found the way these things are always found: a 400 from the real endpoint,
    with the whole feature already written.
    """

    def walk(self, node):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from self.walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from self.walk(item)

    def test_no_enum_offers_an_empty_string(self):
        # "cannot be empty" — the API rejects the whole request over it, so an
        # optional field is expressed by leaving it out, never by listing "".
        for node in self.walk(llm.response_schema()):
            assert "" not in (node.get("enum") or []), node

    def test_a_line_with_no_seam_still_reads(self):
        # The freedom that replaced the enum: anything but the two markers is
        # dropped on the way in rather than refused.
        from video_prompter.serialize import prompt_from_dict

        prompt = prompt_from_dict({
            "duration": 5,
            "beats": [{"start": 0, "end": 5, "angle": "wide", "action": "a door",
                       "lines": [{"speakers": ["a"], "text": "hello", "continuity": ""}]}],
        })
        assert prompt.storyboard.beats[0].lines[0].continuity == ""


class TestTransientFailures:
    """A busy upstream is not an error worth showing anyone."""

    def build(self, monkeypatch, *codes):
        """A Gemini whose transport fails with `codes` in order, then succeeds."""
        import urllib.error

        gemini = llm.Gemini(api_key="sk-not-a-real-key")
        left = list(codes)
        opened = []

        def fake_urlopen(request, timeout=None):
            opened.append(request)
            if left:
                raise urllib.error.HTTPError(
                    "https://x", left.pop(0), "busy", {}, io.BytesIO(b"{}")
                )
            return io.BytesIO(json.dumps({"ok": True}).encode())

        monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(llm.time, "sleep", lambda _: None)
        return gemini, opened

    def test_a_busy_upstream_is_tried_again(self, monkeypatch):
        gemini, opened = self.build(monkeypatch, 503)
        assert gemini._get("models") == {"ok": True}
        assert len(opened) == 2

    def test_rate_limiting_is_tried_again_too(self, monkeypatch):
        gemini, opened = self.build(monkeypatch, 429, 429)
        assert gemini._get("models") == {"ok": True}
        assert len(opened) == 3

    def test_it_gives_up_rather_than_hammering(self, monkeypatch):
        gemini, opened = self.build(monkeypatch, 503, 503, 503, 503)
        with pytest.raises(llm.LlmError):
            gemini._get("models")
        assert len(opened) == llm.RETRIES

    def test_a_refused_key_is_not_retried(self, monkeypatch):
        # Repeating it would not change the answer, only the wait.
        gemini, opened = self.build(monkeypatch, 403)
        with pytest.raises(llm.LlmError) as raised:
            gemini._get("models")
        assert len(opened) == 1
        assert "refused the key" in str(raised.value)

    def test_an_unknown_model_says_how_to_find_a_real_one(self, monkeypatch):
        gemini, _ = self.build(monkeypatch, 404)
        with pytest.raises(llm.LlmError) as raised:
            gemini._get("models")
        assert "probe=1" in str(raised.value)


class TestTheDurationIsASettingNotASuggestion:
    def test_an_existing_prompt_is_told_to_keep_its_length(self, api):
        # The schema requires `duration`, so left alone the model re-picks one
        # to suit the scene and quietly overrides the editor's number.
        fake = Fake(an_answer())
        api(fake).llm_draft({"prompt": a_prompt(12), "instruction": "more dramatic"})

        assert "Keep the duration at 12s" in fake.calls[0][1]

    def test_a_blank_prompt_is_left_to_choose(self, api):
        # Nothing has been decided yet, so there is nothing to override.
        fake = Fake(an_answer())
        api(fake).llm_draft({"prompt": {"duration": 10}, "instruction": "a chase"})

        assert "Keep the duration" not in fake.calls[0][1]


class TestTheRepairLoopStaysInScope:
    """The bug this class exists for: a scoped rewrite asking, over and over,
    for a fix the merge was always going to throw away."""

    def test_a_beat_finding_is_unreachable_from_an_audio_scope(self):
        # scope=audio keeps only `audio`; a rewritten storyboard is discarded,
        # so the gap comes straight back and the turn bought nothing.
        gap = [{"rule": "storyboard.gap", "message": "Nothing scheduled for 4s."}]
        assert llm.actionable([], gap, llm.Scope.parse("audio")) == []

    def test_the_same_finding_is_worth_a_turn_when_beats_are_in_scope(self):
        gap = [{"rule": "storyboard.gap", "message": "Nothing scheduled for 4s."}]
        assert llm.actionable([], gap, llm.Scope.everything()) == gap

    def test_a_soundscape_finding_is_reachable_from_an_audio_scope(self):
        empty = [{"rule": "audio.missing", "message": "No soundscape."}]
        assert llm.actionable([], empty, llm.Scope.parse("audio")) == empty

    def test_a_whole_document_rule_is_reachable_from_anywhere(self):
        # An over-long prompt can be shortened from whichever field is in scope.
        long = [{"rule": "prompt.long", "message": "Over the ceiling."}]
        assert llm.actionable([], long, llm.Scope.parse("style")) == long

    def test_a_single_beat_scope_still_reaches_beat_rules(self):
        short = [{"rule": "beat.too_short", "message": "0.4s."}]
        assert llm.actionable([], short, llm.Scope.parse("beat:1")) == short

    def test_a_scoped_draft_does_not_spin_on_what_it_cannot_fix(self, api):
        # End to end: the storyboard is broken, the scope is `audio`, and the
        # model must be asked exactly once rather than three times.
        current = a_prompt()
        current["beats"] = [
            {"start": 0, "end": 2, "angle": "wide", "action": "a door opens",
             "camera": "", "transition": "", "lines": []}
        ]
        current["audio"] = ""
        answered = an_answer(audio="Rain on a tin roof, steady and close.")
        answered["beats"] = current["beats"]
        fake = Fake(answered)

        result = api(fake).llm_draft(
            {"prompt": current, "instruction": "add rain", "scope": "audio"}
        )

        assert result["attempts"] == 1
        assert result["prompt"]["audio"] == "Rain on a tin roof, steady and close."
        # Still reported, just not chased.
        assert any(f["rule"] == "storyboard.gap" for f in result["findings"])


class TestTheDraftHasAWallClock:
    def test_repairs_stop_once_the_budget_is_spent(self, api, monkeypatch):
        # Three repair turns each costing three transport retries at a 60s
        # timeout is nine minutes of spinner. The budget is the backstop.
        broken = an_answer(beats=[{"start": 0, "end": 3, "angle": "wide",
                                   "action": "a door opens", "camera": "",
                                   "transition": "", "lines": []}])
        # started, then the check after the first answer: already over budget.
        clock = iter([0.0, 1000.0, 2000.0, 3000.0])
        monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock))
        fake = Fake(broken, broken, broken)

        result = api(fake).llm_draft({"prompt": a_prompt(), "instruction": "fix"})

        assert result["attempts"] == 1
        assert result["findings"]


class TestTheModelWritesOnlyWhatWasAsked:
    """The fix for both halves of "slow, and it flattened my prompt": asking
    for the whole document back meant regenerating eight fields to change one."""

    def test_a_scoped_schema_asks_for_that_field_alone(self):
        assert list(llm.response_schema(llm.Scope.parse("audio"))["properties"]) == ["audio"]

    def test_a_beat_scope_asks_only_for_beats(self):
        assert list(llm.response_schema(llm.Scope.parse("beat:1"))["properties"]) == ["beats"]

    def test_a_whole_document_scope_still_asks_for_everything(self):
        every = llm.response_schema(llm.Scope.everything())["properties"]
        assert set(every) == set(llm.AUTHORED_FIELDS)

    def test_what_is_asked_for_is_what_is_required(self):
        for scope in ("all", "audio", "beat:0", "style"):
            schema = llm.response_schema(llm.Scope.parse(scope))
            assert set(schema["required"]) == set(schema["properties"])

    def test_the_draft_sends_the_narrowed_schema(self, api):
        fake = Fake(an_answer())
        api(fake).llm_draft(
            {"prompt": a_prompt(), "instruction": "grittier", "scope": "audio"}
        )
        assert list(fake.calls[0][2]["properties"]) == ["audio"]

    def test_the_turn_names_the_fields_it_wants(self, api):
        fake = Fake(an_answer())
        api(fake).llm_draft(
            {"prompt": a_prompt(), "instruction": "grittier", "scope": "audio"}
        )
        assert "Answer with `audio` and nothing else." in fake.calls[0][1]

    def test_a_beat_scope_asks_for_that_shot_alone(self, api):
        fake = Fake(an_answer())
        api(fake).llm_draft(
            {"prompt": a_prompt(), "instruction": "handheld", "scope": "beat:1"}
        )
        assert "holding only shot 2" in fake.calls[0][1]


class TestExistingWorkIsNotADraft:
    def test_a_written_prompt_is_defended_in_the_turn(self, api):
        fake = Fake(an_answer())
        api(fake).llm_draft({"prompt": a_prompt(), "instruction": "grittier"})
        said = fake.calls[0][1]

        assert "written deliberately" in said
        assert "Do not summarise" in said

    def test_a_blank_prompt_is_not(self, api):
        # There is nothing to preserve, and telling it to preserve nothing
        # would only crowd the instruction that matters.
        fake = Fake(an_answer())
        api(fake).llm_draft({"prompt": {"duration": 10}, "instruction": "a chase"})

        assert "written deliberately" not in fake.calls[0][1]

    def test_the_repair_turn_does_not_re_widen_the_scope(self, api):
        # `style.missing` is a warning the style scope can actually answer, so
        # a repair turn really fires and can be inspected.
        fake = Fake(an_answer(style=""), an_answer(style="A neon-noir look."))
        api(fake).llm_draft(
            {"prompt": a_prompt(), "instruction": "neon noir", "scope": "style"}
        )

        assert len(fake.calls) == 2
        assert "whole document" not in fake.calls[-1][1]
        assert "same fields as before" in fake.calls[-1][1]


class TestTheModelTier:
    def test_a_whole_document_write_can_use_a_different_model(self):
        write_settings('{"gemini_key": "k", "gemini_model": "fast-one",'
                       ' "gemini_model_whole": "strong-one"}')
        assert llm.provider_from_settings(llm.Scope.everything()).model == "strong-one"

    def test_a_scoped_tweak_stays_on_the_quick_one(self):
        # The tweak is the one someone is sitting there waiting for.
        write_settings('{"gemini_key": "k", "gemini_model": "fast-one",'
                       ' "gemini_model_whole": "strong-one"}')
        assert llm.provider_from_settings(llm.Scope.parse("audio")).model == "fast-one"

    def test_a_single_beat_counts_as_a_tweak(self):
        write_settings('{"gemini_key": "k", "gemini_model": "fast-one",'
                       ' "gemini_model_whole": "strong-one"}')
        assert llm.provider_from_settings(llm.Scope.parse("beat:0")).model == "fast-one"

    def test_unset_it_is_simply_the_same_model(self):
        # A second default nobody chose would surprise on both bill and latency.
        write_settings('{"gemini_key": "k", "gemini_model": "fast-one"}')
        assert llm.provider_from_settings(llm.Scope.everything()).model == "fast-one"


class TestALocalModel:
    """An OpenAI-compatible server on this machine: Ollama, LM Studio, llama.cpp."""

    @pytest.fixture
    def server(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen: dict = {"posts": []}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, payload: dict) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._reply({"data": [{"id": "nomic-embed-text"}, {"id": "qwen-local"}, {"id": "a-other"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen["posts"].append(body)
                content = seen.get("content", '{"audio": "Rain."}')
                self._reply({"choices": [{"message": {"content": content},
                                          "finish_reason": "stop"}]})

        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        seen["url"] = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        yield seen
        httpd.shutdown()

    def test_a_local_url_wins_over_a_gemini_key(self):
        write_settings('{"gemini_key": "k", "llm_url": "http://127.0.0.1:11434/v1/"}')
        provider = llm.provider_from_settings()
        assert isinstance(provider, llm.OpenAICompatible)
        assert provider.base_url == "http://127.0.0.1:11434/v1"

    def test_the_environment_can_name_one_too(self, monkeypatch):
        monkeypatch.setenv(llm.LOCAL_URL_VAR, "http://localhost:1234/v1")
        monkeypatch.setenv(llm.LOCAL_MODEL_VAR, "mistral")
        provider = llm.provider_from_settings()
        assert (provider.base_url, provider.model, provider.source) == (
            "http://localhost:1234/v1", "mistral", "environment")

    def test_without_a_url_gemini_still_answers(self):
        write_settings('{"gemini_key": "k"}')
        assert isinstance(llm.provider_from_settings(), llm.Gemini)

    def test_it_asks_with_the_schema_and_picks_the_first_model(self, server):
        provider = llm.OpenAICompatible(base_url=server["url"])
        schema = llm.response_schema(llm.Scope.parse("audio"))
        assert provider.complete("sys", "user", schema) == {"audio": "Rain."}
        sent = server["posts"][0]
        assert sent["model"] == "qwen-local"
        assert sent["response_format"]["json_schema"]["schema"] == schema
        assert [m["role"] for m in sent["messages"]] == ["system", "user"]

    def test_thinking_and_fences_are_not_the_document(self, server):
        server["content"] = '<think>hmm</think>\n```json\n{"audio": "Wind."}\n```'
        provider = llm.OpenAICompatible(base_url=server["url"], model="m")
        assert provider.complete("s", "u", {}) == {"audio": "Wind."}

    def test_a_server_that_is_down_says_so(self):
        provider = llm.OpenAICompatible(base_url="http://127.0.0.1:9/v1", model="m")
        with pytest.raises(llm.LlmError) as raised:
            provider.complete("s", "u", {})
        assert raised.value.status == 504
        assert "running" in str(raised.value)

    def test_status_reports_the_local_provider(self, api):
        write_settings('{"llm_url": "http://127.0.0.1:11434/v1", "llm_model": "qwen3"}')
        status = api(None).llm_status()
        assert (status["configured"], status["provider"], status["model"]) == (True, "local", "qwen3")
