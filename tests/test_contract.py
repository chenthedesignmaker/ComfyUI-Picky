from __future__ import annotations

import pytest

from video_prompter import (
    Beat,
    PresetLibrary,
    ProfileStore,
    ProjectStore,
    VideoPrompt,
    bundled_root,
    h3_spec,
)
from video_prompter import contract
from video_prompter.api import Api, ApiError
from video_prompter.lint import lint


@pytest.fixture
def api(tmp_path):
    return Api(
        projects=ProjectStore(tmp_path / "projects"),
        profiles=ProfileStore(tmp_path / "profiles"),
        presets=PresetLibrary(tmp_path / "presets", bundled_root()),
    )


def status_of(fn, *args) -> int:
    with pytest.raises(ApiError) as raised:
        fn(*args)
    return raised.value.status


PROMPT_KEYS = [
    "duration", "beats", "cues", "style", "overview", "camera", "audio",
    "screen_text", "negatives", "references", "width", "height",
]


class TestSchema:
    def test_describes_every_prompt_block(self):
        fields = contract.schema()["prompt_fields"]
        for key in PROMPT_KEYS:
            assert key in fields, key

    def test_limits_track_h3_spec(self):
        limits = contract.schema()["limits"]
        assert limits["duration_s"] == [h3_spec.MIN_DURATION_S, h3_spec.MAX_DURATION_S]
        assert limits["fps"] == h3_spec.FPS
        assert limits["canvas_multiple"] == h3_spec.CANVAS_MULTIPLE
        assert limits["max_pixels"] == h3_spec.MAX_PIXELS
        assert limits["aspect_ratios"] == list(h3_spec.ASPECT_RATIOS)
        assert limits["max_prompt_chars"] == h3_spec.MAX_PROMPT_CHARS
        assert limits["recommended_prompt_chars"] == h3_spec.RECOMMENDED_PROMPT_CHARS

    def test_reference_caps_are_per_kind(self):
        caps = contract.schema()["limits"]["reference_caps"]
        assert caps["image"] == h3_spec.MAX_REFERENCE_IMAGES
        assert caps["video"] == h3_spec.MAX_REFERENCE_VIDEOS

    def test_every_rule_has_a_valid_severity_and_meaning(self):
        for entry in contract.schema()["lint_rules"]:
            assert entry["rule"] and entry["meaning"]
            assert entry["severity"] in {"info", "warning", "error"}

    def test_catalogue_and_severities_agree(self):
        assert set(contract.LINT_RULES) == set(contract._SEVERITIES)

    def test_operations_list_the_authoring_loop(self):
        paths = {op["path"].split("?")[0] for op in contract.schema()["operations"]}
        assert "/preview" in paths
        assert "/source/update" in paths

    def test_findings_the_linter_emits_are_all_documented(self):
        # Drift guard: a prompt that trips several rules at once. Every rule the
        # linter returns must be in the catalogue an agent reads.
        prompt = VideoPrompt.blank(10)
        prompt.width = 30  # too small and off the 32 grid
        prompt.style = ""
        prompt.storyboard.beats = [
            Beat(0, 6, "wide", "a door opens"),
            Beat(4, 10, "close", "a hand reaches"),  # overlaps the first
        ]
        rules = {finding.rule for finding in lint(prompt)}
        assert rules  # the prompt really does trip something
        undocumented = rules - set(contract.LINT_RULES)
        assert not undocumented, undocumented

    def test_api_describe_schema_serves_the_contract(self, api):
        assert api.describe_schema() == contract.schema()


class TestGetSource:
    def _seed_one_source(self, api) -> str:
        api.create_project({"name": "P"})
        api.add_source({"project": "P", "name": "One", "duration": 8})
        return api.load_project("P")["project"]["sources"][0]["id"]

    def test_returns_the_source_and_its_first_prompt(self, api):
        source_id = self._seed_one_source(api)
        got = api.get_source("P", source_id)
        assert got["source"]["id"] == source_id
        assert "style" in got["prompt"]
        assert got["prompt"] == got["source"]["segments"][0]["prompt"]

    def test_missing_project_is_not_found(self, api):
        assert status_of(api.get_source, "ghost", "src_x") == 404

    def test_missing_source_is_not_found(self, api):
        self._seed_one_source(api)
        assert status_of(api.get_source, "P", "src_nope") == 404
