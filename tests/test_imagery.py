from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_prompter import PresetLibrary, ProfileStore, ProjectStore, bundled_root
from video_prompter.api import Api, ApiError
from video_prompter.document import Reference, ReferenceSet, VideoPrompt
from video_prompter.h3_spec import NODE_IMAGE_TO_VIDEO, NODE_REFERENCE_TO_VIDEO
from video_prompter.imagery import (
    augment_i2v,
    build_ref2v_profile,
    build_ref2v_workflow,
    models_from_profile,
    profile_for,
    render_mode,
)
from video_prompter.lint import lint
from video_prompter.profiles import ProfileError, StaleProfileError, capture
from video_prompter.serialize import prompt_from_dict, prompt_to_dict


@pytest.fixture
def base_profile():
    workflow = json.loads(
        (Path(__file__).parent / "fixtures" / "h3_t2v.json").read_text(encoding="utf-8")
    )
    return capture("H3 local t2v", workflow)


def a_prompt() -> VideoPrompt:
    prompt = VideoPrompt.blank(10)
    prompt.style = "A cinematic look."
    prompt.audio = "rain"
    return prompt


class TestRenderMode:
    def test_nothing_attached_is_t2v(self):
        assert render_mode(a_prompt()) == "t2v"

    def test_a_first_frame_is_i2v(self):
        prompt = a_prompt()
        prompt.first_frame = "hero.png"
        assert render_mode(prompt) == "i2v"

    def test_a_last_frame_alone_is_i2v(self):
        prompt = a_prompt()
        prompt.last_frame = "end.png"
        assert render_mode(prompt) == "i2v"

    def test_references_are_ref2v(self):
        prompt = a_prompt()
        prompt.references = ReferenceSet([Reference("image", "ref.png", "the hero")])
        assert render_mode(prompt) == "ref2v"

    def test_a_keyframe_wins_over_references(self):
        # both set is a lint error; the mode still resolves to one node
        prompt = a_prompt()
        prompt.first_frame = "hero.png"
        prompt.references = ReferenceSet([Reference("image", "ref.png")])
        assert render_mode(prompt) == "i2v"


class TestAugmentI2v:
    def test_first_frame_wires_a_load_image(self, base_profile):
        graph = augment_i2v(base_profile.api_format, "hero.png")
        loaders = [(nid, n) for nid, n in graph.items() if n["class_type"] == "LoadImage"]
        assert len(loaders) == 1
        lid, loader = loaders[0]
        assert loader["inputs"]["image"] == "hero.png"
        h3 = next(n for n in graph.values() if n["class_type"] == NODE_IMAGE_TO_VIDEO)
        assert h3["inputs"]["first_frame"] == [lid, 0]

    def test_both_keyframes_wire_two_loaders(self, base_profile):
        graph = augment_i2v(base_profile.api_format, "start.png", "end.png")
        loaders = [n for n in graph.values() if n["class_type"] == "LoadImage"]
        assert len(loaders) == 2
        h3 = next(n for n in graph.values() if n["class_type"] == NODE_IMAGE_TO_VIDEO)
        assert "first_frame" in h3["inputs"] and "last_frame" in h3["inputs"]

    def test_it_does_not_mutate_the_source_workflow(self, base_profile):
        before = json.dumps(base_profile.api_format, sort_keys=True)
        augment_i2v(base_profile.api_format, "hero.png")
        assert json.dumps(base_profile.api_format, sort_keys=True) == before

    def test_a_workflow_without_the_image_node_refuses(self):
        with pytest.raises(StaleProfileError):
            augment_i2v({"1": {"class_type": "UNETLoader", "inputs": {}}}, "hero.png")


class TestModelsFromProfile:
    def test_names_come_from_the_loaders(self, base_profile):
        models = models_from_profile(base_profile)
        assert models["unet"].startswith("minimax_h3")
        assert "qwen3vl" in models["clip"]

    def test_the_audio_vae_is_the_one_feeding_the_audio_decode(self, base_profile):
        models = models_from_profile(base_profile)
        assert "audio" in models["audio_vae"]
        assert "video" in models["video_vae"]
        assert models["audio_vae"] != models["video_vae"]

    def test_none_profile_falls_back_to_defaults(self):
        models = models_from_profile(None)
        assert set(models) == {"unet", "clip", "video_vae", "audio_vae"}


class TestBuildRef2v:
    def test_the_reference_node_and_audio_vae_are_present(self):
        prompt = a_prompt()
        prompt.references = ReferenceSet([Reference("image", "ref.png", "the hero")])
        wf = build_ref2v_workflow(prompt)
        ref = wf["ref"]
        assert ref["class_type"] == NODE_REFERENCE_TO_VIDEO
        assert ref["inputs"]["audio_vae"] == ["avae", 0]

    def test_each_kind_wires_a_zero_based_slot(self):
        prompt = a_prompt()
        prompt.references = ReferenceSet([
            Reference("image", "img.png"),
            Reference("video", "clip.mp4"),
            Reference("video_audio", "clip.wav"),
            Reference("audio", "score.wav"),
        ])
        wf = build_ref2v_workflow(prompt)
        ref_inputs = wf["ref"]["inputs"]
        # keys are the executor's dotted <group>.<slot> so they nest correctly
        # image straight from LoadImage
        assert ref_inputs["ref_images.ref_image_0"] == ["refimg_0", 0]
        assert wf["refimg_0"]["class_type"] == "LoadImage"
        # video frames come through GetVideoComponents' first output
        assert ref_inputs["ref_videos.ref_video_0"] == ["refvidc_0", 0]
        assert wf["refvidc_0"]["class_type"] == "GetVideoComponents"
        # soundtrack and standalone audio each load audio into their own slot
        assert ref_inputs["ref_video_audios.ref_video_audio_0"] == ["refvidaud_0", 0]
        assert ref_inputs["ref_audios.ref_audio_0"] == ["refaud_0", 0]

    def test_multiple_images_get_consecutive_slots(self):
        prompt = a_prompt()
        prompt.references = ReferenceSet([
            Reference("image", "a.png"), Reference("image", "b.png"),
        ])
        ref_inputs = build_ref2v_workflow(prompt)["ref"]["inputs"]
        assert ref_inputs["ref_images.ref_image_0"] == ["refimg_0", 0]
        assert ref_inputs["ref_images.ref_image_1"] == ["refimg_1", 0]

    def test_the_profile_discovers_the_ordinary_roles(self):
        prompt = a_prompt()
        prompt.references = ReferenceSet([Reference("image", "ref.png", "the hero")])
        profile = build_ref2v_profile(prompt)
        for role in ("prompt", "length", "width", "height", "seed"):
            assert profile.controls(role), f"missing {role}"

    def test_references_are_scaled_to_the_render_by_default(self):
        prompt = a_prompt()
        prompt.references = ReferenceSet([Reference("image", "ref.png")])
        assert build_ref2v_workflow(prompt)["ref"]["inputs"]["ref_image_size"] == "match"

    def test_the_tier_can_buy_a_full_size_reference(self):
        prompt = a_prompt()
        prompt.references = ReferenceSet([Reference("image", "ref.png")])
        wf = build_ref2v_workflow(prompt, reference_size="max")
        assert wf["ref"]["inputs"]["ref_image_size"] == "max"

    def test_the_size_is_baked_into_the_profile_not_injected(self):
        """It shapes the graph, so it has to be right before the profile is built."""
        prompt = a_prompt()
        prompt.references = ReferenceSet([Reference("image", "ref.png")])
        profile = build_ref2v_profile(prompt, reference_size="max")
        node = next(
            n for n in profile.api_format.values()
            if n["class_type"] == NODE_REFERENCE_TO_VIDEO
        )
        assert node["inputs"]["ref_image_size"] == "max"


class TestProfileFor:
    def test_t2v_returns_the_base_untouched(self, base_profile):
        mode, profile = profile_for(a_prompt(), base_profile)
        assert mode == "t2v"
        assert profile is base_profile

    def test_i2v_builds_from_the_base(self, base_profile):
        prompt = a_prompt()
        prompt.first_frame = "hero.png"
        mode, profile = profile_for(prompt, base_profile)
        assert mode == "i2v"
        assert any(n["class_type"] == "LoadImage" for n in profile.api_format.values())

    def test_ref2v_builds_fresh(self, base_profile):
        prompt = a_prompt()
        prompt.references = ReferenceSet([Reference("image", "ref.png", "the hero")])
        mode, profile = profile_for(prompt, base_profile)
        assert mode == "ref2v"
        assert any(n["class_type"] == NODE_REFERENCE_TO_VIDEO for n in profile.api_format.values())

    def test_ref2v_carries_the_tiers_reference_size_through(self, base_profile):
        prompt = a_prompt()
        prompt.references = ReferenceSet([Reference("image", "ref.png", "the hero")])
        _, profile = profile_for(prompt, base_profile, "max")
        node = next(
            n for n in profile.api_format.values()
            if n["class_type"] == NODE_REFERENCE_TO_VIDEO
        )
        assert node["inputs"]["ref_image_size"] == "max"

    def test_no_profile_is_an_error(self):
        with pytest.raises(ProfileError):
            profile_for(a_prompt(), None)

    def test_i2v_without_a_base_is_an_error(self):
        prompt = a_prompt()
        prompt.first_frame = "hero.png"
        with pytest.raises(ProfileError):
            profile_for(prompt, None)


class TestSerializeKeyframes:
    def test_keyframes_round_trip(self):
        prompt = a_prompt()
        prompt.first_frame = "video_prompter/start.png"
        prompt.last_frame = "video_prompter/end.png"
        restored = prompt_from_dict(prompt_to_dict(prompt))
        assert restored.first_frame == "video_prompter/start.png"
        assert restored.last_frame == "video_prompter/end.png"

    def test_missing_keyframes_default_to_empty(self):
        restored = prompt_from_dict({"duration": 8})
        assert restored.first_frame == "" and restored.last_frame == ""


class TestMediaLint:
    def test_keyframe_and_references_together_is_an_error(self):
        prompt = a_prompt()
        prompt.first_frame = "hero.png"
        prompt.references = ReferenceSet([Reference("image", "ref.png", "the hero")])
        rules = {f.rule for f in lint(prompt)}
        assert "mode.conflict" in rules

    def test_a_lone_last_frame_is_flagged(self):
        prompt = a_prompt()
        prompt.last_frame = "end.png"
        rules = {f.rule for f in lint(prompt)}
        assert "keyframe.orphan" in rules

    def test_a_paired_first_and_last_frame_is_clean(self):
        prompt = a_prompt()
        prompt.first_frame = "start.png"
        prompt.last_frame = "end.png"
        rules = {f.rule for f in lint(prompt)}
        assert "keyframe.orphan" not in rules
        assert "mode.conflict" not in rules


class TestUpload:
    @pytest.fixture
    def api(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VIDEO_PROMPTER_INPUT_DIR", str(tmp_path / "input"))
        return Api(
            projects=ProjectStore(tmp_path / "projects"),
            profiles=ProfileStore(tmp_path / "profiles"),
            presets=PresetLibrary(tmp_path / "presets", bundled_root()),
        )

    def test_save_writes_into_the_input_dir(self, api, tmp_path):
        result = api.save_reference("Hero Shot.png", b"\x89PNGdata")
        assert result["kind"] == "image"
        assert result["filename"] == "video_prompter/Hero Shot.png"
        assert (tmp_path / "input" / result["filename"]).read_bytes() == b"\x89PNGdata"

    def test_a_name_collision_is_suffixed(self, api):
        first = api.save_reference("a.png", b"1")
        second = api.save_reference("a.png", b"2")
        assert first["filename"] != second["filename"]

    def test_concurrent_uploads_preserve_each_files_bytes(self, api, tmp_path):
        from concurrent.futures import ThreadPoolExecutor
        payloads = [str(index).encode() for index in range(16)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda data: api.save_reference("same.png", data), payloads))
        assert len({result["filename"] for result in results}) == len(payloads)
        for result, payload in zip(results, payloads):
            assert (tmp_path / "input" / result["filename"]).read_bytes() == payload

    def test_an_unsupported_extension_is_rejected(self, api):
        with pytest.raises(ApiError) as raised:
            api.save_reference("notes.txt", b"x")
        assert raised.value.status == 400

    def test_kind_follows_the_extension(self, api):
        assert api.save_reference("v.mp4", b"x")["kind"] == "video"
        assert api.save_reference("a.wav", b"x")["kind"] == "audio"

    def test_no_input_dir_is_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VIDEO_PROMPTER_INPUT_DIR", raising=False)
        # No ComfyUI import path either, so there is nowhere to write.
        api = Api(
            projects=ProjectStore(tmp_path / "projects"),
            profiles=ProfileStore(tmp_path / "profiles"),
            presets=PresetLibrary(tmp_path / "presets", bundled_root()),
        )
        with pytest.raises(ApiError) as raised:
            api.save_reference("a.png", b"x")
        assert raised.value.status == 503

    def test_fetch_rejects_non_http_schemes(self, api):
        with pytest.raises(ApiError) as raised:
            api.fetch_reference({"url": "file:///etc/passwd", "filename": "x.png"})
        assert raised.value.status == 400

    def test_fetch_downloads_and_saves(self, api, tmp_path, monkeypatch):
        import contextlib
        import urllib.request

        @contextlib.contextmanager
        def fake_urlopen(request, timeout=0):
            class _R:
                def read(self):
                    return b"downloaded-bytes"
            yield _R()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        result = api.fetch_reference({"url": "https://example.com/pic.jpg"})
        assert result["kind"] == "image"
        assert result["source"] == "https://example.com/pic.jpg"
        assert (tmp_path / "input" / result["filename"]).read_bytes() == b"downloaded-bytes"
