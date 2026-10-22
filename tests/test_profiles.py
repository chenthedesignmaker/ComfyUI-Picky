from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from video_prompter import VideoPrompt, h3_spec
from video_prompter.profiles import (
    ROLE_DURATION_S,
    ROLE_HEIGHT,
    ROLE_LENGTH,
    ROLE_MEGAPIXELS,
    ROLE_PROMPT,
    ROLE_SEED,
    ROLE_STEPS,
    ROLE_TURBO_STRENGTH,
    ROLE_WIDTH,
    Binding,
    ProfileError,
    ProfileStore,
    RenderProfile,
    StaleProfileError,
    backfill_quality_roles,
    capture,
    discover_bindings,
    inject,
    validate,
)


def t2v_workflow() -> dict:
    """A local H3 text-to-video graph in ComfyUI API format.

    Shaped after the official local template: the H3 conditioning node emits
    `positive` plus the joint AV latent, the negative is a zeroed copy, and the
    two VAEs decode picture and sound separately.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "minimax_h3_bf16.safetensors", "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen3vl_32b_minimax_h3.safetensors", "type": "minimax_h3"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "5": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": "",
                "width": 1344,
                "height": 768,
                "length": 124,
            },
        },
        "6": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": ["1", 0], "shift_video": 5.0, "shift_audio": 5.0},
        },
        "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["5", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 1],
                "seed": 0,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "11": {
            "class_type": "SaveVideo",
            "inputs": {"images": ["9", 0], "audio": ["10", 0], "filename_prefix": "h3/clip"},
        },
    }


@pytest.fixture
def workflow():
    return t2v_workflow()


@pytest.fixture
def profile(workflow):
    return capture("H3 local t2v", workflow)


class TestDiscovery:
    def test_the_prompt_input_is_found(self, workflow):
        assert discover_bindings(workflow)[ROLE_PROMPT] == [Binding("5", "prompt")]

    def test_geometry_is_found_on_the_conditioning_node(self, workflow):
        found = discover_bindings(workflow)
        assert found[ROLE_LENGTH] == [Binding("5", "length")]
        assert found[ROLE_WIDTH] == [Binding("5", "width")]
        assert found[ROLE_HEIGHT] == [Binding("5", "height")]

    def test_the_sampler_seed_is_found(self, workflow):
        assert discover_bindings(workflow)[ROLE_SEED] == [Binding("8", "seed")]

    def test_the_sampler_steps_are_found(self, workflow):
        assert discover_bindings(workflow)[ROLE_STEPS] == [Binding("8", "steps")]

    def test_noise_seed_counts_as_a_seed(self, workflow):
        workflow["12"] = {"class_type": "SamplerCustom", "inputs": {"noise_seed": 7}}
        assert Binding("12", "noise_seed") in discover_bindings(workflow)[ROLE_SEED]

    def test_a_wired_input_is_left_alone(self, workflow):
        # A width driven by another node was wired deliberately; do not claim it.
        workflow["5"]["inputs"]["width"] = ["13", 0]
        assert ROLE_WIDTH not in discover_bindings(workflow)

    def test_the_empty_latent_node_is_also_geometry(self):
        workflow = {
            "1": {
                "class_type": "EmptyMiniMaxH3LatentAV",
                "inputs": {"width": 1344, "height": 768, "length": 124},
            },
            "2": {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0], "prompt": ""},
            },
        }
        found = discover_bindings(workflow)
        assert found[ROLE_LENGTH] == [Binding("1", "length")]
        assert found[ROLE_PROMPT] == [Binding("2", "prompt")]

    def test_a_megapixels_input_is_claimed_as_a_quality_knob(self):
        workflow = t2v_workflow()
        workflow["13"] = {
            "class_type": "ResolutionSelector",
            "inputs": {"aspect_ratio": "16:9", "megapixels": 1.0, "multiple": 32},
        }
        assert discover_bindings(workflow)[ROLE_MEGAPIXELS] == [Binding("13", "megapixels")]

    def _with_turbo_lora(self, title: str) -> dict:
        workflow = t2v_workflow()
        workflow["14"] = {
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": title},
            "inputs": {
                "model": ["1", 0],
                "lora_name": "minimax_h3_turbo.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        }
        return workflow

    def test_the_turbo_lora_is_found_by_its_node_title(self):
        found = discover_bindings(self._with_turbo_lora("Turbo LoRA"))
        assert Binding("14", "strength_model") in found[ROLE_TURBO_STRENGTH]
        assert Binding("14", "strength_clip") in found[ROLE_TURBO_STRENGTH]

    def test_a_lora_without_the_turbo_hint_is_not_claimed(self):
        found = discover_bindings(self._with_turbo_lora("Detail Enhancer"))
        assert ROLE_TURBO_STRENGTH not in found

    def test_unrelated_literals_are_not_claimed(self, workflow):
        flat = [b for targets in discover_bindings(workflow).values() for b in targets]
        assert not any(b.input_name in ("cfg", "filename_prefix") for b in flat)

    def test_an_empty_workflow_discovers_nothing(self):
        assert discover_bindings({}) == {}


class TestValidation:
    def test_a_captured_profile_is_valid(self, profile):
        validate(profile)

    def test_a_workflow_with_no_prompt_input_is_rejected(self):
        with pytest.raises(ProfileError, match="no prompt binding"):
            capture("headless", {"1": {"class_type": "KSampler", "inputs": {"seed": 0}}})

    def test_an_empty_workflow_is_rejected(self):
        with pytest.raises(ProfileError, match="no workflow"):
            capture("nothing", {})

    def test_a_binding_to_a_deleted_node_is_stale(self, profile):
        del profile.api_format["8"]
        with pytest.raises(StaleProfileError, match="node 8"):
            validate(profile)

    def test_a_binding_to_a_renamed_input_is_stale(self, profile):
        del profile.api_format["5"]["inputs"]["length"]
        with pytest.raises(StaleProfileError, match="'length'"):
            validate(profile)

    def test_an_unknown_role_is_rejected(self, profile):
        profile.bindings["lighting"] = [Binding("5", "prompt")]
        with pytest.raises(ProfileError, match="unknown role"):
            validate(profile)


class TestInject:
    def test_the_prompt_text_lands_in_the_workflow(self, profile):
        workflow = inject(profile, VideoPrompt.blank(10), "a cinematic look", seed=1)
        assert workflow["5"]["inputs"]["prompt"] == "a cinematic look"

    def test_length_is_the_aligned_frame_count_not_the_seconds(self, profile):
        workflow = inject(profile, VideoPrompt.blank(10), "x", seed=1)
        assert workflow["5"]["inputs"]["length"] == 243

    def test_the_canvas_is_carried_across(self, profile):
        prompt = VideoPrompt.blank(8)
        prompt.set_ratio("9:16")
        workflow = inject(profile, prompt, "x", seed=1)
        assert (workflow["5"]["inputs"]["width"], workflow["5"]["inputs"]["height"]) == (768, 1344)

    def test_the_seed_is_written_where_asked(self, profile):
        workflow = inject(profile, VideoPrompt.blank(8), "x", seed=4242)
        assert workflow["8"]["inputs"]["seed"] == 4242

    def test_no_seed_means_a_fresh_one(self, profile):
        seeds = {
            inject(profile, VideoPrompt.blank(8), "x")["8"]["inputs"]["seed"]
            for _ in range(8)
        }
        assert len(seeds) > 1

    def test_the_template_is_never_mutated(self, profile):
        before = copy.deepcopy(profile.api_format)
        inject(profile, VideoPrompt.blank(15), "something else", seed=9)
        assert profile.api_format == before

    def test_untouched_inputs_survive(self, profile):
        workflow = inject(profile, VideoPrompt.blank(8), "x", seed=1)
        assert workflow["8"]["inputs"]["steps"] == 30
        assert workflow["11"]["inputs"]["filename_prefix"] == "h3/clip"

    def test_wiring_survives(self, profile):
        workflow = inject(profile, VideoPrompt.blank(8), "x", seed=1)
        assert workflow["8"]["inputs"]["latent_image"] == ["5", 1]

    def test_injecting_a_stale_profile_raises_rather_than_guessing(self, profile):
        del profile.api_format["5"]
        with pytest.raises(StaleProfileError):
            inject(profile, VideoPrompt.blank(8), "x", seed=1)

    def test_steps_are_written_only_when_the_quality_asks(self, profile):
        left = inject(profile, VideoPrompt.blank(8), "x", seed=1)
        assert left["8"]["inputs"]["steps"] == 30
        driven = inject(profile, VideoPrompt.blank(8), "x", seed=1, steps=8)
        assert driven["8"]["inputs"]["steps"] == 8

    def test_turbo_strength_drives_a_titled_lora_and_bypasses_at_zero(self):
        workflow = t2v_workflow()
        workflow["14"] = {
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "Turbo LoRA"},
            "inputs": {
                "model": ["1", 0],
                "lora_name": "minimax_h3_turbo.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        }
        profile = capture("turbo", workflow)
        # Untouched without a value; both strengths follow the knob when given.
        left = inject(profile, VideoPrompt.blank(8), "x", seed=1)
        assert left["14"]["inputs"]["strength_model"] == 1.0
        bypassed = inject(profile, VideoPrompt.blank(8), "x", seed=1, turbo_strength=0.0)
        assert bypassed["14"]["inputs"]["strength_model"] == 0.0
        assert bypassed["14"]["inputs"]["strength_clip"] == 0.0

    def test_a_megapixel_budget_shrinks_the_canvas(self, profile):
        prompt = VideoPrompt.blank(8)
        prompt.set_ratio("16:9")
        full = inject(profile, prompt, "x", seed=1)
        preview = inject(profile, prompt, "x", seed=1, megapixels=0.2)
        full_pixels = full["5"]["inputs"]["width"] * full["5"]["inputs"]["height"]
        preview_pixels = preview["5"]["inputs"]["width"] * preview["5"]["inputs"]["height"]
        assert preview_pixels < full_pixels
        # Near the budget, and the ratio is preserved.
        assert 0.15e6 < preview_pixels < 0.25e6

    def test_a_megapixel_budget_feeds_a_resolution_node(self):
        workflow = t2v_workflow()
        workflow["13"] = {
            "class_type": "ResolutionSelector",
            "inputs": {"aspect_ratio": "16:9", "megapixels": 1.0, "multiple": 32},
        }
        result = inject(
            capture("mp", workflow), VideoPrompt.blank(8), "x", seed=1, megapixels=0.2
        )
        assert result["13"]["inputs"]["megapixels"] == 0.2

    def test_length_and_duration_can_be_overridden(self, profile):
        # A re-shoot drives these from the take's frozen numbers, not a prompt.
        workflow = inject(
            profile, VideoPrompt.blank(10), "x", seed=1, length=209, duration_s=8.7
        )
        assert workflow["5"]["inputs"]["length"] == 209

    def test_two_prompt_nodes_both_receive_the_text(self, workflow):
        workflow["12"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": ""},
        }
        result = inject(capture("two", workflow), VideoPrompt.blank(8), "shared", seed=1)
        assert result["5"]["inputs"]["prompt"] == "shared"
        assert result["12"]["inputs"]["prompt"] == "shared"


class TestRoundTrip:
    def test_a_profile_survives_serialization(self, profile):
        restored = RenderProfile.from_dict(profile.to_dict())
        assert restored.bindings == profile.bindings
        assert restored.api_format == profile.api_format

    def test_the_canvas_workflow_is_kept_for_recapture(self, workflow):
        canvas = {"nodes": [{"id": 5, "type": "MiniMaxH3ImageToVideo"}], "links": []}
        profile = capture("with canvas", workflow, canvas_workflow=canvas)
        assert RenderProfile.from_dict(profile.to_dict()).canvas_workflow == canvas


class TestBackfill:
    """Profiles captured before the quality roles existed still drive them."""

    def _legacy(self, workflow):
        # A profile as it was stored before steps was a role: discovered, then
        # the quality roles stripped back out.
        profile = capture("legacy", workflow)
        profile.bindings.pop("steps", None)
        profile.bindings.pop("megapixels", None)
        return profile

    def test_loading_backfills_a_missing_steps_binding(self, tmp_path, workflow):
        store = ProfileStore(tmp_path)
        # Save the stripped-down form to disk, bypassing capture's discovery.
        path = store.path_for("legacy")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._legacy(workflow).to_dict()), encoding="utf-8")

        loaded = store.load("legacy")
        assert loaded.targets("steps") == [Binding("8", "steps")]

    def test_backfill_leaves_an_existing_binding_alone(self, workflow):
        profile = capture("fresh", workflow)
        profile.bind("steps", "8", "steps")
        # Pretend the user pointed steps somewhere odd; discovery must not
        # second-guess a binding that is already there.
        profile.bindings["steps"] = [Binding("999", "steps")]
        backfill_quality_roles(profile)
        assert profile.targets("steps") == [Binding("999", "steps")]

    def test_backfill_adds_nothing_a_workflow_lacks(self, workflow):
        # The t2v graph has no megapixels node, so backfill cannot invent one.
        profile = self._legacy(workflow)
        backfill_quality_roles(profile)
        assert not profile.controls("megapixels")


class TestRealWorkflow:
    """Against a workflow actually exported from ComfyUI 0.30.0.

    Exercises what a hand-built graph really looks like: subgraph-prefixed node
    ids, `SamplerCustomAdvanced`/`RandomNoise` in place of `KSampler`, and
    geometry derived from a loaded image rather than typed in.
    """

    @pytest.fixture
    def real(self):
        path = Path(__file__).parent / "fixtures" / "h3_i2v_real.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_prompt_is_found_through_a_subgraph_id(self, real):
        assert discover_bindings(real)[ROLE_PROMPT] == [Binding("105:104", "prompt")]

    def test_the_seed_is_found_on_random_noise(self, real):
        assert discover_bindings(real)[ROLE_SEED] == [Binding("105:15", "noise_seed")]

    def test_wired_geometry_is_not_claimed(self, real):
        # width/height come from a resolution node, length from a math
        # expression. Injecting into those would break the graph.
        found = discover_bindings(real)
        assert ROLE_WIDTH not in found
        assert ROLE_HEIGHT not in found
        assert ROLE_LENGTH not in found

    def test_the_megapixel_knobs_are_found(self, real):
        # This graph sizes by megapixels: a canvas selector and an image
        # resizer both carry the knob, and both should be driven together.
        found = discover_bindings(real)
        assert Binding("115", "megapixels") in found[ROLE_MEGAPIXELS]
        assert Binding("119", "megapixels") in found[ROLE_MEGAPIXELS]

    def test_such_a_profile_is_still_valid_and_renderable(self, real):
        profile = capture("real i2v", real)
        assert profile.controls(ROLE_PROMPT)
        assert not profile.controls(ROLE_LENGTH)

    def test_describe_says_what_the_workflow_keeps(self, real):
        described = capture("real i2v", real).describe()
        assert "controls prompt, seed" in described
        assert "the workflow owns: duration_s, length, width, height" in described

    def test_duration_can_be_pinned_to_the_seconds_primitive(self, real):
        profile = capture("real i2v", real)
        profile.bind(ROLE_DURATION_S, "105:111", "value")

        workflow = inject(profile, VideoPrompt.blank(12), "x", seed=1)
        assert workflow["105:111"]["inputs"]["value"] == 12.0

    def test_the_workflows_own_frame_maths_matches_the_spec(self, real):
        # The graph aligns frames itself with:
        #   max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
        expression = real["105:107"]["inputs"]["expression"]
        assert "% 17" in expression

        def graph_maths(seconds: float) -> int:
            n = max(5, round(seconds * 24))
            return n + (5 - (n % 17)) % 17

        for seconds in range(1, 21):
            assert graph_maths(seconds) == h3_spec.frames_for_seconds(seconds), seconds

    def test_injecting_leaves_the_derived_inputs_wired(self, real):
        profile = capture("real i2v", real).bind(ROLE_DURATION_S, "105:111", "value")
        workflow = inject(profile, VideoPrompt.blank(10), "x", seed=1)
        assert workflow["105:104"]["inputs"]["width"] == ["115", 0]
        assert workflow["105:104"]["inputs"]["length"] == ["105:107", 1]


class TestShippedT2VWorkflow:
    """The text-to-video graph in `examples/`, which is the supported path.

    Unlike a hand-built image-to-video graph, this one keeps geometry as
    literals so the editor drives everything with no manual binding.
    """

    @pytest.fixture
    def t2v(self):
        path = Path(__file__).parent / "fixtures" / "h3_t2v.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_it_is_shipped_alongside_the_fixture(self):
        shipped = Path(__file__).parent.parent / "examples" / "h3_t2v.json"
        fixture = Path(__file__).parent / "fixtures" / "h3_t2v.json"
        assert json.loads(shipped.read_text(encoding="utf-8")) == json.loads(
            fixture.read_text(encoding="utf-8")
        )

    def test_every_role_is_discovered_without_hand_binding(self, t2v):
        profile = capture("t2v", t2v)
        for role in (ROLE_PROMPT, ROLE_LENGTH, ROLE_WIDTH, ROLE_HEIGHT, ROLE_SEED):
            assert profile.controls(role), role

    def test_it_carries_no_image_conditioning(self, t2v):
        classes = {node["class_type"] for node in t2v.values()}
        assert "LoadImage" not in classes
        assert not any("first_frame" in node["inputs"] for node in t2v.values())

    def test_it_uses_the_fl2va_checkpoint_shared_by_t2v_and_i2v(self, t2v):
        loaders = [n for n in t2v.values() if n["class_type"] == "UNETLoader"]
        assert "fl2va" in loaders[0]["inputs"]["unet_name"]

    def test_it_decodes_picture_and_sound_separately(self, t2v):
        classes = [node["class_type"] for node in t2v.values()]
        assert "VAEDecode" in classes
        assert "VAEDecodeAudio" in classes

    def test_every_wire_points_at_a_node_that_exists(self, t2v):
        for node_id, node in t2v.items():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2:
                    assert str(value[0]) in t2v, f"{node_id}.{name} -> {value[0]}"

    def test_the_default_length_is_on_the_frame_grid(self, t2v):
        length = next(
            n["inputs"]["length"] for n in t2v.values()
            if n["class_type"] == h3_spec.NODE_IMAGE_TO_VIDEO
        )
        assert h3_spec.align_frames(length) == length
        assert h3_spec.is_trained_length(length)

    def test_the_editor_drives_it_end_to_end(self, t2v):
        profile = capture("t2v", t2v)
        prompt = VideoPrompt.blank(15)
        prompt.set_ratio("9:16")
        workflow = inject(profile, prompt, "a cinematic look", seed=7)

        h3 = next(
            node for node in workflow.values()
            if node["class_type"] == h3_spec.NODE_IMAGE_TO_VIDEO
        )
        assert h3["inputs"]["prompt"] == "a cinematic look"
        assert h3["inputs"]["length"] == 362
        assert (h3["inputs"]["width"], h3["inputs"]["height"]) == (768, 1344)

    def test_the_canvas_it_reports_is_the_canvas_it_renders(self, t2v):
        # The point of binding width/height: no ResolutionSelector in between
        # quietly choosing something else.
        profile = capture("t2v", t2v)
        prompt = VideoPrompt.blank(10)
        workflow = inject(profile, prompt, "x", seed=1)
        h3 = next(
            node for node in workflow.values()
            if node["class_type"] == h3_spec.NODE_IMAGE_TO_VIDEO
        )
        assert (h3["inputs"]["width"], h3["inputs"]["height"]) == (prompt.width, prompt.height)


class TestManualBinding:
    def test_an_unknown_role_is_refused(self, profile):
        with pytest.raises(ProfileError, match="Unknown role"):
            profile.bind("lighting", "5", "prompt")

    def test_binding_twice_does_not_duplicate(self, profile):
        profile.bind(ROLE_DURATION_S, "5", "length")
        profile.bind(ROLE_DURATION_S, "5", "length")
        assert len(profile.targets(ROLE_DURATION_S)) == 1

    def test_bind_returns_the_profile_for_chaining(self, profile):
        assert profile.bind(ROLE_DURATION_S, "5", "length") is profile

    def test_a_hand_pinned_binding_survives_serialization(self, profile):
        profile.bind(ROLE_DURATION_S, "5", "length")
        restored = RenderProfile.from_dict(profile.to_dict())
        assert restored.targets(ROLE_DURATION_S) == [Binding("5", "length")]

    def test_a_hand_pinned_binding_is_still_validated(self, profile):
        profile.bind(ROLE_DURATION_S, "nonexistent", "value")
        with pytest.raises(StaleProfileError):
            validate(profile)


class TestStore:
    @pytest.fixture
    def store(self, tmp_path):
        return ProfileStore(tmp_path / "profiles")

    def test_save_then_load(self, store, profile):
        store.save(profile)
        assert store.load("H3 local t2v").bindings == profile.bindings

    def test_listing_is_empty_before_anything_is_saved(self, store):
        assert store.list() == []

    def test_listing_finds_saved_profiles(self, store, profile):
        store.save(profile)
        assert store.list() == ["H3 local t2v"]

    def test_loading_a_missing_profile_raises(self, store):
        with pytest.raises(ProfileError):
            store.load("nope")

    def test_delete_reports_whether_anything_went(self, store, profile):
        store.save(profile)
        assert store.delete("H3 local t2v") is True
        assert store.delete("H3 local t2v") is False

    def test_an_invalid_profile_is_not_written(self, store, profile):
        profile.bindings = {}
        with pytest.raises(ProfileError):
            store.save(profile)
        assert store.list() == []
