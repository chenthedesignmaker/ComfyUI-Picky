from __future__ import annotations

import pytest

from video_prompter import PresetError, PresetLibrary, VideoPrompt, bundled_root
from video_prompter.presets import KINDS, sanitize_name, split_negatives


@pytest.fixture
def library(tmp_path):
    return PresetLibrary(tmp_path)


@pytest.fixture
def bundled():
    return PresetLibrary(bundled_root())


class TestNames:
    def test_spaces_survive_because_people_read_these(self):
        assert sanitize_name("Anamorphic night city") == "Anamorphic night city"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("../etc/passwd", "etcpasswd"), ("a/b", "ab"), ("a\\b", "ab"), ("dot.name", "dotname")],
    )
    def test_separators_and_dots_are_stripped(self, raw, expected):
        assert sanitize_name(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "...", "/"])
    def test_a_name_that_reduces_to_nothing_is_rejected(self, raw):
        with pytest.raises(PresetError):
            sanitize_name(raw)


class TestStorage:
    def test_write_then_read(self, library):
        library.write("style", "My look", "A cinematic look.")
        assert library.read("style", "My look") == "A cinematic look."

    def test_listing_is_case_insensitively_sorted(self, library):
        for name in ("zebra", "Alpha", "moose"):
            library.write("camera", name, "x")
        assert library.list("camera") == ["Alpha", "moose", "zebra"]

    def test_an_unknown_kind_is_rejected(self, library):
        with pytest.raises(PresetError):
            library.write("lighting", "x", "y")

    def test_reading_a_missing_preset_raises(self, library):
        with pytest.raises(PresetError):
            library.read("style", "nope")

    def test_delete_reports_whether_anything_went(self, library):
        library.write("style", "temp", "x")
        assert library.delete("style", "temp") is True
        assert library.delete("style", "temp") is False

    def test_listing_an_empty_library_is_empty_not_an_error(self, library):
        assert library.list("style") == []


class TestStackedRoots:
    def test_names_from_every_root_are_listed(self, tmp_path):
        local, shared = tmp_path / "local", tmp_path / "shared"
        PresetLibrary(local).write("style", "Local look", "local")
        PresetLibrary(shared).write("style", "Shared look", "shared")

        stacked = PresetLibrary(local, shared)
        assert stacked.list("style") == ["Local look", "Shared look"]

    def test_the_earlier_root_shadows_the_later_one(self, tmp_path):
        local, shared = tmp_path / "local", tmp_path / "shared"
        PresetLibrary(local).write("style", "Look", "the local one")
        PresetLibrary(shared).write("style", "Look", "the shared one")

        assert PresetLibrary(local, shared).read("style", "Look") == "the local one"

    def test_writes_land_in_the_primary_root_only(self, tmp_path):
        local, shared = tmp_path / "local", tmp_path / "shared"
        shared.mkdir()
        PresetLibrary(local, shared).write("style", "New", "x")
        assert (local / "style" / "New.txt").exists()
        assert not (shared / "style" / "New.txt").exists()

    def test_delete_never_touches_a_shared_root(self, tmp_path):
        local, shared = tmp_path / "local", tmp_path / "shared"
        PresetLibrary(shared).write("style", "Shared", "x")

        stacked = PresetLibrary(local, shared)
        assert stacked.delete("style", "Shared") is False
        assert stacked.read("style", "Shared") == "x"

    def test_a_library_needs_at_least_one_root(self):
        with pytest.raises(PresetError):
            PresetLibrary()


class TestApply:
    def test_style_lands_in_the_style_block(self, library):
        library.write("style", "Look", "A cinematic look.")
        prompt = library.apply(VideoPrompt.blank(), "style", "Look")
        assert prompt.style == "A cinematic look."

    def test_negatives_split_one_clause_per_line(self, library):
        library.write("negatives", "Bans", "no text\nno watermarks\nno cartoon rendering")
        prompt = library.apply(VideoPrompt.blank(), "negatives", "Bans")
        assert prompt.negatives == ["no text", "no watermarks", "no cartoon rendering"]

    def test_a_single_line_of_negatives_splits_on_commas(self):
        assert split_negatives("no text, no watermarks.") == ["no text", "no watermarks"]

    def test_a_storyboard_preset_refits_onto_the_current_budget(self, library):
        library.write(
            "storyboard",
            "Three beats",
            "[0s-5s] wide: one\n[5s-10s] medium: two\n[10s-15s] close: three",
        )
        prompt = library.apply(VideoPrompt.blank(duration=6), "storyboard", "Three beats")

        assert prompt.storyboard.duration == 6
        assert prompt.storyboard.is_contiguous()
        assert [shot.action for shot in prompt.storyboard.beats] == ["one", "two", "three"]

    def test_a_storyboard_preset_with_no_timecodes_is_rejected(self, library):
        library.write("storyboard", "Prose", "the hero runs across a rooftop")
        with pytest.raises(PresetError):
            library.apply(VideoPrompt.blank(), "storyboard", "Prose")


class TestBundled:
    def test_every_kind_ships_with_something(self, bundled):
        for kind in KINDS:
            assert bundled.list(kind), f"no bundled presets for {kind}"

    def test_bundled_storyboards_all_parse_and_refit(self, bundled):
        for name in bundled.list("storyboard"):
            prompt = bundled.apply(VideoPrompt.blank(duration=8), "storyboard", name)
            assert prompt.storyboard.beats, name
            assert prompt.storyboard.is_contiguous(), name

    def test_a_prompt_built_only_from_bundled_presets_lints_clean(self, bundled):
        from video_prompter import errors, lint

        prompt = VideoPrompt.blank(duration=15)
        bundled.apply(prompt, "style", "Anamorphic night city")
        bundled.apply(prompt, "camera", "Hard cuts")
        bundled.apply(prompt, "audio", "Trailer score")
        bundled.apply(prompt, "negatives", "No text or watermarks")
        bundled.apply(prompt, "storyboard", "Trailer escalation")

        assert errors(lint(prompt)) == []
