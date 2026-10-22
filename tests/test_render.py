from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_prompter import (
    Beat,
    Cast,
    Member,
    Reference,
    ReferenceSet,
    Storyboard,
    VideoPrompt,
)
from video_prompter.profiles import ROLE_DURATION_S, capture
from video_prompter.render import (
    RenderBlocked,
    attach_latent_save,
    prepare_render,
    prepare_reshoot,
    preview,
)


@pytest.fixture
def profile():
    path = Path(__file__).parent / "fixtures" / "h3_i2v_real.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    return capture("real", workflow).bind(ROLE_DURATION_S, "105:111", "value")


@pytest.fixture
def good():
    return VideoPrompt(
        storyboard=Storyboard(10, [Beat(0, 5, "wide", "one"), Beat(5, 10, "close", "two")]),
        style="A cinematic look.",
        audio="rain on glass",
        negatives=["no watermarks"],
    )


@pytest.fixture
def broken(good):
    # Overlapping shots are an error-level finding.
    good.storyboard.beats = [Beat(0, 6, "wide", "one"), Beat(4, 10, "close", "two")]
    return good


class TestPreview:
    def test_it_returns_text_and_findings(self, good):
        result = preview(good)
        assert result["prompt"].startswith(
            "integrated_multimodal_description: [Shot 1] A cinematic look, wide, one."
        )
        assert isinstance(result["findings"], list)

    def test_it_reports_the_real_frame_count(self, good):
        result = preview(good)
        assert result["frames"] == 243
        assert result["seconds"] == 10.125

    def test_it_carries_the_spans_the_editor_tints_with(self, good):
        result = preview(good)
        text = result["prompt"]
        assert any(
            span["kind"] == "style" and text[span["start"] : span["end"]].startswith("A cinematic")
            for span in result["spans"]
        )

    def test_it_does_not_raise_on_a_broken_prompt(self, broken):
        result = preview(broken)
        assert any(f["rule"] == "storyboard.overlap" for f in result["findings"])

    def test_findings_are_json_serializable(self, broken):
        json.dumps(preview(broken))

    def test_a_linked_row_comes_back_named_so_the_editor_can_match_it(self, good):
        """Resolution reorders the list, so position alone no longer identifies a row."""
        member = Member(
            id="cast_11111111",
            name="Mara",
            kind="character",
            appearance="a tall woman in a red coat",
            references=[
                Reference("image", "mara_front.png"),
                Reference("image", "mara_face.png"),
            ],
        )
        good.references = ReferenceSet(
            [Reference("image", "", member="cast_11111111", slot=1,
                       retention="attribute_transfer")]
        )
        rows = preview(good, cast=Cast([member]))["references"]
        assert [(r["tag"], r["path"], r["member"], r["slot"]) for r in rows] == [
            ("<Picture 1>", "mara_front.png", "", 0),
            ("<Picture 2>", "mara_face.png", "cast_11111111", 1),
        ]


class TestPrepareRender:
    def test_a_good_prompt_produces_a_workflow(self, good, profile):
        prepared = prepare_render(good, profile, seed=7)
        assert prepared.workflow["105:104"]["inputs"]["prompt"].startswith(
            "integrated_multimodal_description: [Shot 1] A cinematic look, wide, one."
        )
        assert prepared.workflow["105:15"]["inputs"]["noise_seed"] == 7

    def test_seconds_reach_the_graphs_own_primitive(self, good, profile):
        prepared = prepare_render(good, profile, seed=1)
        assert prepared.workflow["105:111"]["inputs"]["value"] == 10.0

    def test_the_reported_frames_match_the_spec(self, good, profile):
        prepared = prepare_render(good, profile, seed=1)
        assert (prepared.frames, prepared.seconds) == (243, 10.125)

    def test_warnings_ride_along_without_blocking(self, good, profile):
        good.style = ""  # style.missing is a warning
        prepared = prepare_render(good, profile, seed=1)
        assert any(f.rule == "style.missing" for f in prepared.findings)

    def test_errors_block_before_a_generation_is_spent(self, broken, profile):
        with pytest.raises(RenderBlocked) as raised:
            prepare_render(broken, profile, seed=1)
        assert any(f.rule == "storyboard.overlap" for f in raised.value.findings)

    def test_the_block_can_be_overridden(self, broken, profile):
        prepared = prepare_render(broken, profile, seed=1, strict=False)
        assert prepared.workflow

    def test_the_payload_is_json_clean(self, good, profile):
        json.dumps(prepare_render(good, profile, seed=1).to_dict())

    def test_what_renders_is_what_was_reported(self, good, profile):
        prepared = prepare_render(good, profile, seed=1)
        assert prepared.workflow["105:104"]["inputs"]["prompt"] == prepared.prompt_text


class TestQualityKnobs:
    """megapixels + steps flow from the shot's quality preset into the graph."""

    def test_steps_reach_the_scheduler(self, good, profile):
        prepared = prepare_render(good, profile, seed=1, steps=8)
        assert prepared.workflow["105:9"]["inputs"]["steps"] == 8

    def test_megapixels_reach_the_resolution_nodes(self, good, profile):
        prepared = prepare_render(good, profile, seed=1, megapixels=0.2)
        assert prepared.workflow["115"]["inputs"]["megapixels"] == 0.2
        assert prepared.workflow["119"]["inputs"]["megapixels"] == 0.2

    def test_the_reported_resolution_follows_the_budget(self, good, profile):
        # This graph does not bind width/height, but a take must still record
        # the size it truly rendered at, derived from the megapixel budget.
        full = prepare_render(good, profile, seed=1)
        preview_render = prepare_render(good, profile, seed=1, megapixels=0.2)
        assert preview_render.width * preview_render.height < full.width * full.height

    def test_no_quality_leaves_the_graph_untouched(self, good, profile):
        prepared = prepare_render(good, profile, seed=1)
        assert prepared.workflow["105:9"]["inputs"]["steps"] == 8
        assert prepared.workflow["115"]["inputs"]["megapixels"] == 0.2


class TestLatentSave:
    """Every AV render also saves its latent, so any take can be extended on the
    cleaner latent path later."""

    def test_it_wires_a_save_to_the_same_latent_the_audio_decode_reads(self):
        wf = {
            "s": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
            "a": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["s", 0]}},
        }
        path = attach_latent_save(wf, seed=42)
        saves = [n for n in wf.values() if n["class_type"] == "MiniMaxH3MotionContextSaveLatent"]
        assert len(saves) == 1
        assert saves[0]["inputs"]["latent"] == ["s", 0]
        prefix = saves[0]["inputs"]["filename_prefix"]
        assert prefix.startswith("h3_context/take42_")
        assert path == f"{prefix}_00001.safetensors"

    def test_a_workflow_with_no_audio_is_left_untouched(self):
        wf = {"s": {"class_type": "SamplerCustomAdvanced", "inputs": {}}}
        before = dict(wf)
        assert attach_latent_save(wf, seed=1) is None
        assert wf == before

    def test_it_does_not_double_up_an_existing_save(self):
        wf = {
            "a": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["s", 0]}},
            "keep": {"class_type": "MiniMaxH3MotionContextSaveLatent", "inputs": {}},
        }
        assert attach_latent_save(wf, seed=1) is None
        saves = [n for n in wf.values() if n["class_type"] == "MiniMaxH3MotionContextSaveLatent"]
        assert len(saves) == 1

    def test_prepare_render_reports_the_latent_path_for_the_take(self, good, profile):
        prepared = prepare_render(good, profile, seed=7)
        assert prepared.latent_path.startswith("h3_context/take7_")
        assert prepared.to_dict()["latent_path"] == prepared.latent_path
        assert any(
            node["class_type"] == "MiniMaxH3MotionContextSaveLatent"
            for node in prepared.workflow.values()
        )

    def test_reshoots_at_the_same_seed_keep_separate_context(self, profile):
        renders = [prepare_reshoot(
            profile, prompt_text=text, width=1344, height=768, frames=192,
            seconds=8.0, seed=99,
        ) for text in ("first take", "revised take", "first take")]
        assert len({r.latent_path for r in renders}) == 3
        for render in renders:
            save = next(n for n in render.workflow.values()
                        if n["class_type"] == "MiniMaxH3MotionContextSaveLatent")
            assert render.latent_path == save["inputs"]["filename_prefix"] + "_00001.safetensors"


class TestReshoot:
    """Re-rendering a take from its own frozen values, not a structured prompt."""

    def test_the_takes_own_text_is_injected_verbatim(self, profile):
        prepared = prepare_reshoot(
            profile,
            prompt_text="frozen take text",
            width=1344,
            height=768,
            frames=192,
            seconds=8.0,
            seed=99,
        )
        assert prepared.workflow["105:104"]["inputs"]["prompt"] == "frozen take text"
        assert prepared.prompt_text == "frozen take text"

    def test_the_passed_seed_is_reproduced(self, profile):
        prepared = prepare_reshoot(
            profile, prompt_text="x", width=1344, height=768, frames=192,
            seconds=8.0, seed=4242,
        )
        assert prepared.seed == 4242
        assert prepared.workflow["105:15"]["inputs"]["noise_seed"] == 4242

    def test_no_seed_mints_a_fresh_one(self, profile):
        seeds = {
            prepare_reshoot(
                profile, prompt_text="x", width=1344, height=768, frames=192,
                seconds=8.0,
            ).seed
            for _ in range(8)
        }
        assert len(seeds) > 1

    def test_fine_scales_the_canvas_to_the_budget(self, profile):
        quick = prepare_reshoot(
            profile, prompt_text="x", width=1344, height=768, frames=192,
            seconds=8.0, seed=1, megapixels=0.2, steps=8,
        )
        fine = prepare_reshoot(
            profile, prompt_text="x", width=1344, height=768, frames=192,
            seconds=8.0, seed=1, megapixels=1.0, steps=20,
        )
        assert quick.width * quick.height < fine.width * fine.height
        assert fine.workflow["115"]["inputs"]["megapixels"] == 1.0
        assert fine.workflow["105:9"]["inputs"]["steps"] == 20

    def test_the_frozen_length_reaches_the_graph(self, profile):
        # This graph derives frames from a duration primitive; a re-shoot pins
        # the take's own seconds so the clip length matches.
        prepared = prepare_reshoot(
            profile, prompt_text="x", width=1344, height=768, frames=192,
            seconds=8.0, seed=1,
        )
        assert prepared.workflow["105:111"]["inputs"]["value"] == 8.0
        assert prepared.frames == 192
