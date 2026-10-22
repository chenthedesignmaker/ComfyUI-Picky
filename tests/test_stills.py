"""The still library: generated images recorded, adopted and re-filed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_prompter import ProjectStore
from video_prompter.api import Api, ApiError
from video_prompter.gallery import EXTRA_DATA_KEY
from video_prompter.profiles import ProfileStore
from video_prompter.stills import (
    Still,
    StillStore,
    is_t2i_entry,
    stills_from_history,
)


def history_entry(
    project="Rooftop",
    filename="VideoPrompter_t2i_00001_.png",
    mode="t2i",
    **meta,
):
    """A finished image generation, the way ComfyUI stores one."""
    return {
        "prompt": [
            1,
            "img-1",
            {"save": {"class_type": "SaveImage", "inputs": {}}},
            {
                EXTRA_DATA_KEY: {
                    "project": project,
                    "mode": mode,
                    "prompt": "a rain-slicked alley",
                    "seed": 4242,
                    "steps": 8,
                    "width": 1920,
                    "height": 1088,
                    **meta,
                }
            },
            ["save"],
        ],
        "outputs": {
            "save": {"images": [{"filename": filename, "subfolder": "image", "type": "output"}]}
        },
        "status": {"completed": True},
    }


@pytest.fixture
def store(tmp_path):
    projects = ProjectStore(tmp_path / "projects")
    projects.create("Rooftop", [])
    return StillStore(projects)


class TestHistoryParsing:
    def test_input_image_preview_is_not_a_generated_still(self):
        entry = history_entry()
        entry["outputs"]["load"] = {"images": [{"filename": "reference.png", "type": "input"}]}
        assert [s.filename for s in stills_from_history("img-1", entry)] == ["VideoPrompter_t2i_00001_.png"]

    def test_an_image_generation_is_read(self):
        found = stills_from_history("img-1", history_entry())
        assert len(found) == 1
        assert found[0].filename == "VideoPrompter_t2i_00001_.png"
        assert found[0].subfolder == "image"

    def test_the_recipe_rides_through(self):
        still = stills_from_history("img-1", history_entry(style="noir"))[0]
        assert still.prompt_text == "a rain-slicked alley"
        assert (still.seed, still.steps, still.style) == (4242, 8, "noir")
        assert (still.width, still.height) == (1920, 1088)

    def test_the_segment_it_was_made_for_rides_through(self):
        still = stills_from_history("img-1", history_entry(shot="src_a1b2"))[0]
        assert still.source == "src_a1b2"

    def test_the_cast_member_it_was_made_for_rides_through(self):
        still = stills_from_history("img-1", history_entry(member="cast_12345678"))[0]
        assert still.member == "cast_12345678"

    def test_a_video_render_is_not_a_still(self):
        # The marker is the stamped mode, not the file type — a render that
        # saved a preview frame must not land in the keyframe library.
        assert stills_from_history("abc", history_entry(mode="Render")) == []

    def test_the_editors_own_label_counts_as_t2i(self):
        assert is_t2i_entry(history_entry(mode="Keyframe (T2I)"))

    def test_a_run_from_another_tool_is_not_a_still(self):
        assert not is_t2i_entry({"prompt": [1, "x", {}, {}, []], "outputs": {}})

    def test_a_video_output_inside_a_t2i_run_is_skipped(self):
        entry = history_entry()
        entry["outputs"]["save"]["images"].append(
            {"filename": "clip.mp4", "subfolder": "video", "type": "output"}
        )
        assert [still.filename for still in stills_from_history("img-1", entry)] == [
            "VideoPrompter_t2i_00001_.png"
        ]

    @pytest.mark.parametrize(
        "entry",
        [{}, {"outputs": None}, {"outputs": {"1": None}}, {"prompt": "nonsense", "outputs": {}}],
    )
    def test_a_malformed_entry_yields_nothing(self, entry):
        assert stills_from_history("img-1", entry) == []


class TestStore:
    def test_recording_then_listing(self, store):
        store.record("Rooftop", [Still(prompt_id="a", filename="one.png")])
        assert [still.filename for still in store.list("Rooftop")] == ["one.png"]

    def test_recording_the_same_image_twice_adds_it_once(self, store):
        still = Still(prompt_id="a", filename="one.png")
        assert len(store.record("Rooftop", [still])) == 1
        assert store.record("Rooftop", [still]) == []
        assert len(store.list("Rooftop")) == 1

    def test_newest_first(self, store):
        store.record("Rooftop", [
            Still(prompt_id="a", filename="old.png", created="2026-01-01T00:00:00+00:00"),
            Still(prompt_id="b", filename="new.png", created="2026-02-01T00:00:00+00:00"),
        ])
        assert [still.filename for still in store.list("Rooftop")] == ["new.png", "old.png"]

    def test_listing_can_narrow_to_one_source(self, store):
        store.record("Rooftop", [
            Still(prompt_id="a", filename="mine.png", source="src_1"),
            Still(prompt_id="b", filename="other.png", source="src_2"),
        ])
        assert [s.filename for s in store.list("Rooftop", "src_1")] == ["mine.png"]

    def test_annotating_flags_and_refiles(self, store):
        store.record("Rooftop", [Still(prompt_id="a", filename="one.png")])
        found = store.annotate("Rooftop", "a:one.png", favorite=True, member="cast_1")
        assert found.favorite and found.member == "cast_1"
        assert store.list("Rooftop")[0].favorite

    def test_annotating_something_that_is_not_there(self, store):
        assert store.annotate("Rooftop", "nope:x.png", favorite=True) is None

    def test_forgetting_drops_the_record_only(self, store, tmp_path):
        store.record("Rooftop", [Still(prompt_id="a", filename="one.png")])
        assert store.forget("Rooftop", "a:one.png") == 1
        assert store.list("Rooftop") == []

    def test_a_damaged_file_reads_as_empty(self, store):
        path = store.path_for("Rooftop")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert store.list("Rooftop") == []


def _api(tmp_path: Path) -> Api:
    projects = ProjectStore(tmp_path / "projects")
    projects.create("Rooftop", [])
    return Api(projects=projects, profiles=ProfileStore(tmp_path / "profiles"))


@pytest.fixture
def comfy_dirs(tmp_path, monkeypatch):
    """Stand-in ComfyUI output and input directories."""
    output = tmp_path / "output" / "image"
    output.mkdir(parents=True)
    (tmp_path / "input").mkdir()
    monkeypatch.setenv("VIDEO_PROMPTER_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("VIDEO_PROMPTER_INPUT_DIR", str(tmp_path / "input"))
    (output / "VideoPrompter_t2i_00001_.png").write_bytes(b"fake png")
    return tmp_path


class TestApi:
    def test_recording_adopts_the_image_into_the_input_dir(self, tmp_path, comfy_dirs):
        api = _api(tmp_path)
        result = api.record_stills({
            "project": "Rooftop",
            "history": {"img-1": history_entry()},
        })
        assert result["count"] == 1
        adopted = result["added"][0]["path"]
        assert adopted.startswith("video_prompter/")
        assert (comfy_dirs / "input" / adopted).is_file()

    def test_recording_twice_is_idempotent(self, tmp_path, comfy_dirs):
        api = _api(tmp_path)
        payload = {"project": "Rooftop", "history": {"img-1": history_entry()}}
        api.record_stills(payload)
        assert api.record_stills(payload)["count"] == 0

    def test_another_projects_run_is_ignored(self, tmp_path, comfy_dirs):
        api = _api(tmp_path)
        result = api.record_stills({
            "project": "Rooftop",
            "history": {"img-1": history_entry(project="Somewhere Else")},
        })
        assert result["count"] == 0

    def test_a_still_whose_file_is_gone_is_still_recorded(self, tmp_path, comfy_dirs):
        # The recipe is the part that cannot be recovered; without a file it
        # simply cannot be attached to a prompt.
        api = _api(tmp_path)
        result = api.record_stills({
            "project": "Rooftop",
            "history": {"img-1": history_entry(filename="cleaned_up.png")},
        })
        assert result["count"] == 1
        assert result["added"][0]["path"] == ""

    def test_listing_comes_back_through_the_api(self, tmp_path, comfy_dirs):
        api = _api(tmp_path)
        api.record_stills({"project": "Rooftop", "history": {"img-1": history_entry()}})
        assert len(api.list_stills("Rooftop")["stills"]) == 1

    def test_annotating_a_missing_still_is_404(self, tmp_path):
        with pytest.raises(ApiError, match="No still"):
            _api(tmp_path).annotate_still({
                "project": "Rooftop", "still": "nope:x.png", "favorite": True,
            })

    def test_forgetting_reports_what_it_dropped(self, tmp_path, comfy_dirs):
        api = _api(tmp_path)
        api.record_stills({"project": "Rooftop", "history": {"img-1": history_entry()}})
        still_id = api.list_stills("Rooftop")["stills"][0]["id"]
        assert api.forget_still({"project": "Rooftop", "still": still_id})["forgotten"] == 1

    def test_adopting_the_same_output_twice_keeps_both_files(self, tmp_path, comfy_dirs):
        api = _api(tmp_path)
        first = api.adopt_output_image({
            "filename": "VideoPrompter_t2i_00001_.png", "subfolder": "image",
        })
        second = api.adopt_output_image({
            "filename": "VideoPrompter_t2i_00001_.png", "subfolder": "image",
        })
        assert first["filename"] != second["filename"]


class TestBatch:
    def test_a_sweep_runs_consecutive_seeds(self, tmp_path):
        result = _api(tmp_path).prepare_t2i_batch({
            "prompt": "a rain-slicked alley", "count": 3, "seed": 100,
        })
        assert [entry["seed"] for entry in result["renders"]] == [100, 101, 102]

    def test_every_workflow_carries_the_prompt(self, tmp_path):
        result = _api(tmp_path).prepare_t2i_batch({"prompt": "a cat", "count": 2})
        for entry in result["renders"]:
            assert entry["workflow"]["positive"]["inputs"]["text"] == "a cat"

    def test_an_unpinned_sweep_still_gets_consecutive_seeds(self, tmp_path):
        seeds = [
            entry["seed"]
            for entry in _api(tmp_path).prepare_t2i_batch({"prompt": "a cat", "count": 4})["renders"]
        ]
        assert seeds == list(range(seeds[0], seeds[0] + 4))

    def test_a_silly_count_is_refused(self, tmp_path):
        with pytest.raises(ApiError, match="batch"):
            _api(tmp_path).prepare_t2i_batch({"prompt": "a cat", "count": 999})

    def test_an_empty_prompt_is_still_refused_in_a_sweep(self, tmp_path):
        with pytest.raises(ApiError, match="prompt"):
            _api(tmp_path).prepare_t2i_batch({"prompt": "  ", "count": 2})
