from __future__ import annotations

import io

import pytest

from video_prompter.cli import main


@pytest.fixture
def prompt_file(tmp_path, rooftop_text):
    path = tmp_path / "rooftop.txt"
    path.write_text(rooftop_text, encoding="utf-8")
    return str(path)


class TestExitStatus:
    def test_an_error_level_finding_exits_nonzero(self, prompt_file, capsys):
        # 20s is past what the node generates, so duration.invalid is an error.
        assert main([prompt_file, "-d", "20"]) == 1

    def test_a_clean_prompt_exits_zero(self, prompt_file, capsys):
        assert main([prompt_file]) == 0

    def test_a_malformed_preset_spec_exits_two(self, capsys):
        assert main(["--preset", "no-colon-here"]) == 2
        assert "error:" in capsys.readouterr().err


class TestOutput:
    def test_the_report_precedes_the_prompt(self, prompt_file, capsys):
        main([prompt_file])
        out = capsys.readouterr().out
        assert out.index("MiniMax-H3 local") < out.index("integrated_multimodal_description:")

    def test_quiet_prints_only_the_prompt(self, prompt_file, capsys):
        main([prompt_file, "--quiet"])
        out = capsys.readouterr().out
        assert "storyboard." not in out
        assert out.startswith("integrated_multimodal_description:")

    def test_output_is_written_to_the_named_file(self, prompt_file, tmp_path, capsys):
        target = tmp_path / "out.txt"
        main([prompt_file, "-o", str(target)])
        assert "integrated_multimodal_description:" in target.read_text(encoding="utf-8")

    def test_the_header_reports_frames_and_canvas(self, prompt_file, capsys):
        main([prompt_file, "-d", "15", "--ratio", "21:9"])
        out = capsys.readouterr().out
        # 15s aligns to 362 frames, the top of the trained range.
        assert "362 frames (15.083s @ 24fps)" in out
        # 21:9 at a 768 short edge exceeds the pixel cap, so it scales to this.
        assert "1536x672" in out


class TestStdin(object):
    def test_a_prompt_can_arrive_on_stdin(self, monkeypatch, rooftop_text, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO(rooftop_text))
        assert main([]) == 0
        assert "Realistic live-action" in capsys.readouterr().out


class TestPresets:
    def test_listing_covers_every_kind(self, capsys):
        assert main(["--list-presets"]) == 0
        out = capsys.readouterr().out
        for kind in ("style", "camera", "audio", "negatives", "storyboard"):
            assert f"{kind} (" in out

    def test_presets_alone_compose_a_clean_prompt(self, capsys):
        status = main(
            [
                "-d", "12",
                "-p", "style:Cinematic live-action",
                "-p", "camera:Hard cuts",
                "-p", "audio:Trailer score",
                "-p", "negatives:No text or watermarks",
                "-p", "storyboard:Reveal",
            ]
        )
        out = capsys.readouterr().out
        assert status == 0
        assert "clean" in out
        assert "[Shot 1]" in out

    def test_presets_layer_over_a_file(self, prompt_file, capsys):
        main([prompt_file, "--fix", "-p", "camera:Locked off"])
        assert "Locked off, static frame throughout." in capsys.readouterr().out

    def test_an_extra_root_shadows_the_bundled_library(self, tmp_path, capsys):
        root = tmp_path / "mine"
        (root / "camera").mkdir(parents=True)
        (root / "camera" / "Hard cuts.txt").write_text("My own rule.", encoding="utf-8")

        main(["-d", "8", "--preset-root", str(root), "-p", "camera:Hard cuts"])
        assert "My own rule." in capsys.readouterr().out

    def test_a_missing_preset_exits_two(self, capsys):
        assert main(["-p", "style:Does not exist"]) == 2
