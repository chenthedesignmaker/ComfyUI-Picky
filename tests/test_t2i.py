"""Text-to-image workflow builder and API tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from video_prompter import t2i
from video_prompter.t2i import build_t2i_workflow, prepare_t2i


# -- workflow shape ----------------------------------------------------------


def test_build_t2i_workflow_has_required_nodes():
    wf = build_t2i_workflow("a cat sitting on a windowsill")
    # Must have all the core nodes
    for key in ("unet", "clip", "vae", "positive", "neg_zero",
                "latent", "sampler", "decode", "save"):
        assert key in wf, f"missing node {key!r}"


def test_build_t2i_workflow_wiring():
    """The sampler reads from the loader outputs and the latent canvas."""
    # supersample=1.0 so save reads straight from decode (no downscale node)
    wf = build_t2i_workflow("test prompt", supersample=1.0)
    inputs = wf["sampler"]["inputs"]

    # Model comes from UNETLoader
    assert inputs["model"] == ["unet", 0]
    # Positive from CLIPTextEncode
    assert inputs["positive"] == ["positive", 0]
    # Negative from ConditioningZeroOut
    assert inputs["negative"] == ["neg_zero", 0]
    # Latent from EmptySD3LatentImage
    assert inputs["latent_image"] == ["latent", 0]

    # Decode wired to sampler output and VAE
    assert wf["decode"]["inputs"]["samples"] == ["sampler", 0]
    assert wf["decode"]["inputs"]["vae"] == ["vae", 0]

    # Save wired to decode output
    assert wf["save"]["inputs"]["images"] == ["decode", 0]


def test_build_t2i_workflow_prompt_injected():
    wf = build_t2i_workflow("neon city at night")
    assert wf["positive"]["inputs"]["text"] == "neon city at night"


def test_build_t2i_workflow_default_canvas():
    wf = build_t2i_workflow("test")
    # Default supersample 1.5: render at 1920×1088 × 1.5 = 2880×1632, then
    # downscale to the 1920×1088 delivery size.
    latent = wf["latent"]["inputs"]
    assert latent["width"] == 2880
    assert latent["height"] == 1632
    ds = wf["downscale"]["inputs"]
    assert ds["width"] == 1920
    assert ds["height"] == 1088
    assert wf["save"]["inputs"]["images"] == ["downscale", 0]


def test_build_t2i_workflow_custom_canvas():
    # supersample=1.0 so the latent canvas equals the requested delivery size
    wf = build_t2i_workflow("test", width=1024, height=1024, supersample=1.0)
    latent = wf["latent"]["inputs"]
    assert latent["width"] == 1024
    assert latent["height"] == 1024


def test_build_t2i_workflow_seed():
    wf = build_t2i_workflow("test", seed=42)
    assert wf["sampler"]["inputs"]["seed"] == 42


def test_build_t2i_workflow_turbo_defaults():
    """Turbo model: few steps, cfg 1, euler/simple."""
    wf = build_t2i_workflow("test")
    inputs = wf["sampler"]["inputs"]
    assert inputs["steps"] == 8
    assert inputs["cfg"] == 1.0
    assert inputs["sampler_name"] == "euler"
    assert inputs["scheduler"] == "simple"
    assert inputs["denoise"] == 1.0


# -- supersampling -----------------------------------------------------------


def test_build_t2i_workflow_supersample_renders_larger_and_downscales():
    wf = build_t2i_workflow("test", width=1344, height=768, supersample=2.0)
    # Latent canvas is the supersampled (render) size
    assert wf["latent"]["inputs"]["width"] == 2688
    assert wf["latent"]["inputs"]["height"] == 1536
    # A downscale node brings it back to delivery size before saving
    ds = wf["downscale"]["inputs"]
    assert ds["width"] == 1344
    assert ds["height"] == 768
    assert ds["upscale_method"] == "lanczos"
    assert ds["image"] == ["decode", 0]
    assert wf["save"]["inputs"]["images"] == ["downscale", 0]


def test_build_t2i_workflow_supersample_snaps_to_16():
    # 1344*1.5=2016 (already /16); 768*1.5=1152 (/16). Use an odd factor.
    wf = build_t2i_workflow("test", width=1000, height=1000, supersample=1.3)
    # 1000*1.3=1300 -> nearest /16 is 1296
    assert wf["latent"]["inputs"]["width"] == 1296
    assert wf["latent"]["inputs"]["height"] == 1296


def test_build_t2i_workflow_supersample_one_is_off():
    wf = build_t2i_workflow("test", width=1344, height=768, supersample=1.0)
    assert wf["latent"]["inputs"]["width"] == 1344
    assert "downscale" not in wf


def test_build_t2i_workflow_custom_steps_cfg():
    wf = build_t2i_workflow("test", steps=8, cfg=3.5)
    inputs = wf["sampler"]["inputs"]
    assert inputs["steps"] == 8
    assert inputs["cfg"] == 3.5


def test_build_t2i_workflow_clip_type_krea2():
    """The CLIPLoader must use type 'krea2', not 'flux' or 'flux2'."""
    wf = build_t2i_workflow("test")
    assert wf["clip"]["inputs"]["type"] == "krea2"


def test_build_t2i_workflow_model_overrides():
    wf = build_t2i_workflow("test", models={"unet": "custom_model.safetensors"})
    assert wf["unet"]["inputs"]["unet_name"] == "custom_model.safetensors"
    # Others keep defaults
    assert wf["clip"]["inputs"]["clip_name"] == t2i.DEFAULT_MODELS["clip"]


def test_build_t2i_workflow_save_prefix():
    wf = build_t2i_workflow("test", image_prefix="custom/prefix")
    assert wf["save"]["inputs"]["filename_prefix"] == "custom/prefix"


def test_build_t2i_workflow_is_json_serialisable():
    wf = build_t2i_workflow("a mountain landscape at sunset")
    # Must be serialisable — ComfyUI's /prompt receives it as JSON.
    text = json.dumps(wf)
    assert isinstance(json.loads(text), dict)


# -- cinematic style presets -------------------------------------------------


def test_styled_prompt_appends_bundle():
    out = t2i.styled_prompt("a lone figure on a rooftop", "noir")
    assert out.startswith("a lone figure on a rooftop, ")
    assert t2i.STYLE_PRESETS["noir"] in out


def test_styled_prompt_none_is_unchanged():
    assert t2i.styled_prompt("a cat", None) == "a cat"
    assert t2i.styled_prompt("a cat", "") == "a cat"


def test_styled_prompt_unknown_style_raises():
    with pytest.raises(ValueError, match="Unknown style"):
        t2i.styled_prompt("a cat", "does-not-exist")


def test_style_names_sorted_and_nonempty():
    names = t2i.style_names()
    assert names == sorted(names)
    assert "kodak-2383" in names


def test_build_t2i_workflow_style_reaches_prompt_node():
    wf = build_t2i_workflow("a lone figure on a rooftop", style="noir")
    text = wf["positive"]["inputs"]["text"]
    assert text.startswith("a lone figure on a rooftop, ")
    assert t2i.STYLE_PRESETS["noir"] in text


# -- LoRA stacking -----------------------------------------------------------


def test_build_t2i_workflow_no_loras_wires_loaders_directly():
    """Without LoRAs the prompt/sampler read straight from the loaders."""
    wf = build_t2i_workflow("test")
    assert wf["positive"]["inputs"]["clip"] == ["clip", 0]
    assert wf["sampler"]["inputs"]["model"] == ["unet", 0]
    assert not any(k.startswith("lora_") for k in wf)


def test_build_t2i_workflow_single_lora_chain():
    wf = build_t2i_workflow("test", loras=[
        {"name": "film_grade.safetensors", "strength_model": 0.8, "strength_clip": 0.5},
    ])
    lora = wf["lora_0"]["inputs"]
    # Patches off the loaders
    assert lora["model"] == ["unet", 0]
    assert lora["clip"] == ["clip", 0]
    assert lora["lora_name"] == "film_grade.safetensors"
    assert lora["strength_model"] == 0.8
    assert lora["strength_clip"] == 0.5
    # Downstream reads from the LoRA, not the loaders
    assert wf["sampler"]["inputs"]["model"] == ["lora_0", 0]
    assert wf["positive"]["inputs"]["clip"] == ["lora_0", 1]


def test_build_t2i_workflow_lora_defaults_and_chaining():
    wf = build_t2i_workflow("test", loras=[
        {"name": "a.safetensors"},
        {"name": "b.safetensors", "strength_model": 0.3},
    ])
    # Defaults fill in
    assert wf["lora_0"]["inputs"]["strength_model"] == 1.0
    assert wf["lora_0"]["inputs"]["strength_clip"] == 1.0
    # Second LoRA chains off the first
    assert wf["lora_1"]["inputs"]["model"] == ["lora_0", 0]
    assert wf["lora_1"]["inputs"]["clip"] == ["lora_0", 1]
    # Sampler + prompt read from the last in the chain
    assert wf["sampler"]["inputs"]["model"] == ["lora_1", 0]
    assert wf["positive"]["inputs"]["clip"] == ["lora_1", 1]


def test_build_t2i_workflow_disabled_lora_dropped():
    wf = build_t2i_workflow("test", loras=[
        {"name": "a.safetensors", "enabled": False},
        {"name": "b.safetensors"},
    ])
    # Only the enabled one survives, renumbered from zero
    assert "lora_0" in wf
    assert "lora_1" not in wf
    assert wf["lora_0"]["inputs"]["lora_name"] == "b.safetensors"


def test_build_t2i_workflow_lora_missing_name_raises():
    with pytest.raises(ValueError, match="name"):
        build_t2i_workflow("test", loras=[{"strength_model": 0.5}])


# -- prepare_t2i ------------------------------------------------------------


def test_prepare_t2i_returns_all_fields():
    result = prepare_t2i("a cat")
    assert result["mode"] == "t2i"
    assert result["prompt"] == "a cat"
    assert isinstance(result["workflow"], dict)
    assert isinstance(result["seed"], int)
    assert result["width"] == 1920
    assert result["height"] == 1088
    assert result["steps"] == 8
    # Supersample 1.5 by default: render larger, deliver at 1920×1088
    assert result["supersample"] == 1.5
    assert result["render_width"] == 2880
    assert result["render_height"] == 1632


def test_prepare_t2i_supersample_reports_render_size():
    result = prepare_t2i("a cat", width=1344, height=768, supersample=2.0)
    assert result["supersample"] == 2.0
    assert result["render_width"] == 2688
    assert result["render_height"] == 1536
    assert result["width"] == 1344  # delivery unchanged


def test_prepare_t2i_random_seed():
    """Without a seed, each call picks a random one."""
    a = prepare_t2i("cat")
    b = prepare_t2i("cat")
    # In theory these *could* collide, but the range is 2^53 so practically never.
    assert a["seed"] != b["seed"]


def test_prepare_t2i_fixed_seed():
    result = prepare_t2i("cat", seed=123)
    assert result["seed"] == 123
    assert result["workflow"]["sampler"]["inputs"]["seed"] == 123


# -- Api.prepare_t2i --------------------------------------------------------


def test_api_prepare_t2i(tmp_path: Path):
    from video_prompter.api import Api

    api = Api(
        projects=__import__("video_prompter.projects", fromlist=["ProjectStore"]).ProjectStore(tmp_path / "p"),
        profiles=__import__("video_prompter.profiles", fromlist=["ProfileStore"]).ProfileStore(tmp_path / "pr"),
    )

    result = api.prepare_t2i({"prompt": "sunset over ocean"})
    assert result["mode"] == "t2i"
    assert result["prompt"] == "sunset over ocean"


def test_api_prepare_t2i_rejects_empty_prompt(tmp_path: Path):
    from video_prompter.api import Api, ApiError

    api = Api(
        projects=__import__("video_prompter.projects", fromlist=["ProjectStore"]).ProjectStore(tmp_path / "p"),
        profiles=__import__("video_prompter.profiles", fromlist=["ProfileStore"]).ProfileStore(tmp_path / "pr"),
    )

    with pytest.raises(ApiError, match="prompt"):
        api.prepare_t2i({"prompt": ""})


def _api(tmp_path: Path):
    from video_prompter.api import Api

    return Api(
        projects=__import__("video_prompter.projects", fromlist=["ProjectStore"]).ProjectStore(tmp_path / "p"),
        profiles=__import__("video_prompter.profiles", fromlist=["ProfileStore"]).ProfileStore(tmp_path / "pr"),
    )


def test_api_prepare_t2i_with_style(tmp_path: Path):
    result = _api(tmp_path).prepare_t2i({"prompt": "a rooftop", "style": "noir"})
    assert result["style"] == "noir"
    assert t2i.STYLE_PRESETS["noir"] in result["workflow"]["positive"]["inputs"]["text"]


def test_api_prepare_t2i_unknown_style_is_400(tmp_path: Path):
    from video_prompter.api import ApiError

    with pytest.raises(ApiError, match="Unknown style"):
        _api(tmp_path).prepare_t2i({"prompt": "a cat", "style": "nope"})


def test_api_prepare_t2i_with_loras(tmp_path: Path):
    result = _api(tmp_path).prepare_t2i({
        "prompt": "a cat",
        "loras": [{"name": "grade.safetensors", "strength_model": 0.7}],
    })
    assert result["loras"] == [
        {"name": "grade.safetensors", "strength_model": 0.7, "strength_clip": 1.0}
    ]
    assert result["workflow"]["lora_0"]["inputs"]["lora_name"] == "grade.safetensors"


def test_api_prepare_t2i_bad_lora_is_400(tmp_path: Path):
    from video_prompter.api import ApiError

    with pytest.raises(ApiError, match="name"):
        _api(tmp_path).prepare_t2i({"prompt": "a cat", "loras": [{"strength_model": 1}]})


def test_api_prepare_t2i_loras_not_a_list_is_400(tmp_path: Path):
    from video_prompter.api import ApiError

    with pytest.raises(ApiError, match="list"):
        _api(tmp_path).prepare_t2i({"prompt": "a cat", "loras": "grade"})


def test_api_t2i_styles_lists_presets(tmp_path: Path):
    result = _api(tmp_path).t2i_styles()
    names = [s["name"] for s in result["styles"]]
    assert names == sorted(names)
    assert "kodak-2383" in names
    assert all(s["prompt"] for s in result["styles"])


# -- Api.adopt_output_image -------------------------------------------------


def test_adopt_output_image(tmp_path: Path):
    from video_prompter.api import Api

    # Set up fake output and input dirs
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "input"
    output_dir.mkdir()
    input_dir.mkdir()

    # Place a fake generated image in the output dir
    (output_dir / "image").mkdir()
    src = output_dir / "image" / "VideoPrompter_t2i_00001_.png"
    src.write_bytes(b"fake png data")

    import os
    os.environ["VIDEO_PROMPTER_OUTPUT_DIR"] = str(output_dir)
    os.environ["VIDEO_PROMPTER_INPUT_DIR"] = str(input_dir)
    try:
        api = Api(
            projects=__import__("video_prompter.projects", fromlist=["ProjectStore"]).ProjectStore(tmp_path / "p"),
            profiles=__import__("video_prompter.profiles", fromlist=["ProfileStore"]).ProfileStore(tmp_path / "pr"),
        )

        result = api.adopt_output_image({
            "filename": "VideoPrompter_t2i_00001_.png",
            "subfolder": "image",
        })

        # Returns a path relative to the input dir
        assert result["filename"].startswith("video_prompter/")
        assert result["filename"].endswith(".png")

        # The file was actually copied
        adopted = input_dir / result["filename"]
        assert adopted.is_file()
        assert adopted.read_bytes() == b"fake png data"
    finally:
        os.environ.pop("VIDEO_PROMPTER_OUTPUT_DIR", None)
        os.environ.pop("VIDEO_PROMPTER_INPUT_DIR", None)
