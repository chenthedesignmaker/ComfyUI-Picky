"""Showing a take's file in the desktop file manager.

The one endpoint here hands a path to the operating system, so the tests care
much more about which paths it refuses than about the window it opens — the
opening itself is stubbed out everywhere below.
"""

from __future__ import annotations

import pytest

from video_prompter.api import Api, ApiError
from video_prompter.profiles import ProfileStore
from video_prompter.projects import ProjectStore


@pytest.fixture
def api(tmp_path):
    return Api(
        projects=ProjectStore(tmp_path / "projects"),
        profiles=ProfileStore(tmp_path / "profiles"),
    )


@pytest.fixture
def comfy_dirs(tmp_path, monkeypatch):
    """Stand-in ComfyUI directories holding one rendered take."""
    output = tmp_path / "output"
    (output / "video").mkdir(parents=True)
    (tmp_path / "input").mkdir()
    (output / "take_00001_.mp4").write_bytes(b"fake mp4")
    (output / "video" / "nested_00001_.mp4").write_bytes(b"fake mp4")
    (tmp_path / "secret.txt").write_text("not yours")
    monkeypatch.setenv("VIDEO_PROMPTER_OUTPUT_DIR", str(output))
    monkeypatch.setenv("VIDEO_PROMPTER_INPUT_DIR", str(tmp_path / "input"))
    return tmp_path


@pytest.fixture
def revealed(monkeypatch):
    """Record what would have been shown instead of opening a window."""
    shown: list[str] = []
    monkeypatch.setattr(
        "video_prompter.api._reveal",
        lambda target: (shown.append(str(target)), True)[1],
    )
    return shown


class TestReveal:
    def test_reveals_a_take_in_the_output_dir(self, api, comfy_dirs, revealed):
        result = api.reveal_file({"filename": "take_00001_.mp4"})
        assert result["opened"] is True
        assert result["path"] == str(comfy_dirs / "output" / "take_00001_.mp4")
        assert result["folder"] == str(comfy_dirs / "output")
        assert revealed == [result["path"]]

    def test_follows_a_subfolder(self, api, comfy_dirs, revealed):
        result = api.reveal_file(
            {"filename": "nested_00001_.mp4", "subfolder": "video"}
        )
        assert result["path"] == str(
            comfy_dirs / "output" / "video" / "nested_00001_.mp4"
        )

    def test_refuses_to_escape_the_output_dir(self, api, comfy_dirs, revealed):
        with pytest.raises(ApiError) as error:
            api.reveal_file({"filename": "../secret.txt"})
        assert error.value.status == 400
        assert revealed == []

    def test_missing_file_is_a_404(self, api, comfy_dirs, revealed):
        with pytest.raises(ApiError) as error:
            api.reveal_file({"filename": "never_rendered.mp4"})
        assert error.value.status == 404

    def test_unknown_type_is_refused(self, api, comfy_dirs, revealed):
        with pytest.raises(ApiError) as error:
            api.reveal_file({"filename": "take_00001_.mp4", "type": "elsewhere"})
        assert error.value.status == 400

    def test_a_headless_box_still_answers_with_the_path(
        self, api, comfy_dirs, monkeypatch
    ):
        monkeypatch.setattr("video_prompter.api._reveal", lambda target: False)
        result = api.reveal_file({"filename": "take_00001_.mp4"})
        assert result["opened"] is False
        assert result["path"].endswith("take_00001_.mp4")
