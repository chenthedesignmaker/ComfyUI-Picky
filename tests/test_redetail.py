"""Tests for ReDetail 2.0 (LTX-2.5 Refine-Details / Pixel-Spatial) integration."""

import math

import pytest
from video_prompter.redetail import (
    solve_dims,
    fit_source,
    output_canvas,
    build_workflow,
    N_VIDEO,
    N_CANVAS,
    N_GUIDE_RESIZE,
    N_SAMPLER,
    N_GUIDE,
    N_SAVE,
    N_NOISE_PIXEL,
    N_REF_RESIZE,
    N_FRAME,
)
from video_prompter.refine import RefineSpec, prepare_refine


def test_solve_dims_refine_grid_32():
    # 1312x768 at 1.5x on 32-grid
    w, h, aerr, serr = solve_dims(1312, 768, 1.5, grid=32)
    assert w % 32 == 0
    assert h % 32 == 0
    assert (w, h) == (1920, 1120)
    assert aerr < 0.01  # Under 1% aspect error


def test_solve_dims_pixel_grid_64():
    # 1312x768 at 1.5x on 64-grid
    w, h, aerr, serr = solve_dims(1312, 768, 1.5, grid=64)
    assert w % 64 == 0
    assert h % 64 == 0
    assert aerr < 0.02


def test_solve_dims_invalid_input():
    with pytest.raises(ValueError, match="positive"):
        solve_dims(0, 768, 1.5)
    with pytest.raises(ValueError, match="positive"):
        solve_dims(1280, 0, 1.5)
    with pytest.raises(ValueError, match="positive"):
        solve_dims(1280, 720, -1.0)


def test_fit_source_behavior():
    # Exact /64 dimensions require no crop/resize
    assert fit_source(1280, 704) == ("none", 1280, 704)
    # Small remainder crops
    mode, cw, ch = fit_source(1280, 714)
    assert mode == "crop"
    assert cw % 64 == 0
    assert ch % 64 == 0


def test_output_canvas_presets():
    # Float scale
    w, h = output_canvas(1312, 768, 1.5, mode="refine")
    assert (w, h) == (1920, 1120)

    # "original"
    w, h = output_canvas(1280, 720, "original", mode="refine")
    assert w % 32 == 0 and h % 32 == 0

    # "1080p"
    w, h = output_canvas(1280, 720, "1080p", mode="refine")
    assert w % 32 == 0 and h % 32 == 0
    assert min(w, h) >= 1080

    # "4k"
    w, h = output_canvas(1280, 720, "4k", mode="refine")
    assert w % 32 == 0 and h % 32 == 0
    assert abs(min(w, h) - 2160) <= 32

    # Unknown preset
    with pytest.raises(ValueError, match="Unknown resolution"):
        output_canvas(1280, 720, "invalid_preset")


def test_build_workflow_refine():
    wf = build_workflow(
        source_video="takes/shot01.mp4 [output]",
        source_width=1312,
        source_height=768,
        scale=1.5,
        mode="refine",
        seed=12345,
        guide_strength=0.9,
        filename_prefix="Test_Refine",
    )

    # Video & Save prefix
    assert wf[N_VIDEO]["inputs"]["file"] == "takes/shot01.mp4 [output]"
    assert wf[N_SAVE]["inputs"]["filename_prefix"] == "Test_Refine"

    # Canvas & Guide dimensions
    assert wf[N_CANVAS]["inputs"]["width"] == 1920
    assert wf[N_CANVAS]["inputs"]["height"] == 1120
    assert wf[N_GUIDE_RESIZE]["inputs"]["width"] == 1920
    assert wf[N_GUIDE_RESIZE]["inputs"]["height"] == 1120

    # Sampler tile geometry and guide strength
    assert wf[N_SAMPLER]["inputs"]["seed"] == 12345
    assert wf[N_SAMPLER]["inputs"]["tile_width"] == 1024
    assert wf[N_SAMPLER]["inputs"]["tile_height"] == 576
    assert wf[N_GUIDE]["inputs"]["strength"] == 0.9


def test_build_workflow_pixel():
    wf = build_workflow(
        source_video="takes/shot02.mp4 [output]",
        source_width=1280,
        source_height=720,
        scale=2.0,
        mode="pixel",
        seed=999,
        filename_prefix="Test_Pixel",
    )

    assert wf[N_VIDEO]["inputs"]["file"] == "takes/shot02.mp4 [output]"
    assert wf[N_SAVE]["inputs"]["filename_prefix"] == "Test_Pixel"
    assert wf[N_NOISE_PIXEL]["inputs"]["noise_seed"] == 999
    assert wf[N_REF_RESIZE]["inputs"]["resize_type.shorter_size"] == 720
    assert wf[N_CANVAS]["inputs"]["width"] % 64 == 0
    assert wf[N_CANVAS]["inputs"]["height"] % 64 == 0


def test_refine_integration_refine_mode():
    spec = RefineSpec.from_request({
        "enhancement_mode": "redetail",
        "redetail_scale": 1.5,
        "guide_strength": 0.85,
    })
    prepared = prepare_refine(
        source_video="take.mp4",
        source_take_id="orig_01",
        source_width=1312,
        source_height=768,
        spec=spec,
        seed=42,
    )

    assert prepared.width == 1920
    assert prepared.height == 1120
    assert prepared.source_take_id == "orig_01"
    assert prepared.refinement == ["redetail", "refine", "upscale"]

    wf = prepared.workflow
    assert wf[N_GUIDE]["inputs"]["strength"] == 0.85
    assert wf[N_SAMPLER]["inputs"]["seed"] == 42


def test_refine_integration_pixel_mode():
    spec = RefineSpec.from_request({
        "enhancement_mode": "redetail_pixel",
        "redetail_scale": 1.0,
    })
    prepared = prepare_refine(
        source_video="take.mp4",
        source_take_id="orig_02",
        source_width=1280,
        source_height=720,
        spec=spec,
        seed=101,
    )

    assert prepared.refinement == ["redetail", "pixel"]
    wf = prepared.workflow
    assert wf[N_NOISE_PIXEL]["inputs"]["noise_seed"] == 101


def test_pixel_anchor_comes_from_the_source_video_without_external_image():
    wf = build_workflow("source.mp4 [output]", 1280, 720, mode="pixel")
    anchor = wf[N_FRAME]
    assert anchor["class_type"] == "ImageFromBatch"
    assert anchor["inputs"]["batch_index"] == 0
    assert anchor["inputs"]["length"] == 1
    components = wf[anchor["inputs"]["image"][0]]
    assert components["class_type"] == "GetVideoComponents"
    assert components["inputs"]["video"] == [N_VIDEO, 0]
    assert not any(node["class_type"] == "LoadImage" for node in wf.values())
    assert not any(isinstance(value, str) and value.startswith("__")
                   for node in wf.values() for value in node["inputs"].values())


@pytest.mark.parametrize("mode", ["refine", "pixel"])
def test_transformer_override_changes_the_model_that_is_loaded(mode):
    wf = build_workflow("source.mp4", 1280, 720, mode=mode,
                        model_overrides={"transformer": "alternate.safetensors"})
    loader = wf["5004:5602"]
    assert loader["class_type"] == "UNETLoader"
    assert loader["inputs"]["unet_name"] == "alternate.safetensors"
    assert "ckpt_name" not in loader["inputs"]
    # A later default render must not inherit the previous request's override.
    default = build_workflow("source.mp4", 1280, 720, mode=mode)
    assert default["5004:5602"]["inputs"]["unet_name"] != "alternate.safetensors"


@pytest.mark.parametrize("mode", ["refine", "pixel"])
def test_api_workflows_cannot_succeed_with_only_a_text_preview(mode):
    wf = build_workflow("source.mp4", 608, 320, mode=mode)
    assert wf[N_SAVE]["class_type"] == "SaveVideo"
    assert not any(node["class_type"] == "PreviewAny" for node in wf.values())
    # Dropping UI-only outputs must not leave dangling graph links.
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                assert value[0] in wf


@pytest.mark.parametrize("mode", ["refine", "pixel"])
def test_redetail_pads_before_sampling_and_restores_source_timing(mode):
    wf = build_workflow("source.mp4", 608, 320, mode=mode)
    count = ["5548:5029", 2]
    assert wf["5548:5029"]["inputs"]["image"] == ["5548:9006", 0]
    length = wf[wf[N_CANVAS]["inputs"]["length"][0]]["inputs"]
    assert length["values.a"] == count
    for original, padded in [(1, 1), (8, 9), (9, 9), (121, 121), (124, 129), (129, 129), (240, 241)]:
        assert eval(length["expression"], {"__builtins__": {}}, {"ceil": math.ceil, "a": original}) == padded
    pad = wf["vp_padded_frames"]["inputs"]
    assert pad["length"] == wf[N_CANVAS]["inputs"]["length"]
    assert wf["vp_last_frame"]["inputs"] == {
        "image": ["5548:9006", 0], "batch_index": -1, "length": 1}
    assert wf["vp_tail_frames"]["inputs"]["amount"] == 7
    if mode == "refine":
        assert wf[N_GUIDE_RESIZE]["inputs"]["image"] == ["vp_padded_frames", 0]
    else:
        assert wf[N_REF_RESIZE]["inputs"]["input"] == ["vp_padded_frames", 0]
    video = wf["5518:4849"]["inputs"]
    trim = wf[video["images"][0]]["inputs"]
    assert trim == {"image": ["5518:5538", 0], "batch_index": 0, "length": count}
    assert video["fps"] == ["5548:9006", 2]
    assert video["audio"] == ["5548:9006", 1]
