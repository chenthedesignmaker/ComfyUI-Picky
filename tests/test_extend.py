from __future__ import annotations

import json

import pytest

from video_prompter import extend, imagery, twostage
from video_prompter.document import Reference, VideoPrompt
from video_prompter.extend import (
    ExtendError,
    augment_extend,
    build_base_workflow,
    build_extend_profile,
    build_extend_workflow,
)
from video_prompter.profiles import (
    ROLE_CONTEXT_LATENT,
    ROLE_CONTEXT_LENGTH,
    ROLE_CONTEXT_VIDEO,
    ROLE_PROMPT,
    ROLE_SEED,
    ROLE_STEPS,
    Binding,
    capture,
)
from video_prompter.render import prepare_extend


def graph_and_ids(**kwargs):
    """The fallback graph with Motion Context wired in, plus where it landed."""
    return augment_extend(build_base_workflow(), **kwargs)


def motion_context(workflow, ids):
    return workflow[ids["motion_context"]]["inputs"]


def node_of(workflow, class_type):
    return next(n for n in workflow.values() if n["class_type"] == class_type)


class TestProfile:
    def test_the_ordinary_roles_are_discovered(self):
        profile = build_extend_profile()
        for role in (ROLE_PROMPT, "length", "width", "height", ROLE_SEED, ROLE_STEPS):
            assert profile.controls(role), f"missing {role}"

    def test_the_source_clip_is_pinned_to_load_video(self):
        profile = build_extend_profile()
        binding = profile.targets(ROLE_CONTEXT_VIDEO)[0]
        assert profile.api_format[binding.node_id]["class_type"] == "LoadVideo"
        assert binding.input_name == "file"

    def test_the_video_path_carries_sound_from_the_decoded_source(self):
        # no latent: Motion Context reads audio + audio VAE, not a latent
        wf, ids = graph_and_ids(latent_backed=False)
        mc = motion_context(wf, ids)
        assert mc["context_audio"] == [ids["components"], 1]
        assert mc["audio_vae"] == ["avae", 0]
        assert "context_latent" not in mc
        assert "latent" not in ids

    def test_the_latent_path_swaps_the_audio_source(self):
        wf, ids = graph_and_ids(latent_backed=True)
        mc = motion_context(wf, ids)
        assert mc["context_latent"] == [ids["latent"], 0]
        assert "context_audio" not in mc
        assert wf[ids["latent"]]["class_type"] == "MiniMaxH3MotionContextLoadLatent"

    def test_the_latent_profile_pins_the_latent_path_role(self):
        assert build_extend_profile(latent_backed=True).controls(ROLE_CONTEXT_LATENT)
        assert not build_extend_profile(latent_backed=False).controls(ROLE_CONTEXT_LATENT)

    def test_context_length_drives_both_the_video_and_audio_windows(self):
        profile = build_extend_profile()
        mc_id = extend.motion_context_ids(profile)["motion_context"]
        assert set(profile.targets(ROLE_CONTEXT_LENGTH)) == {
            Binding(mc_id, "context_length"),
            Binding(mc_id, "audio_context_length"),
        }


class TestAugmenting:
    """A continuation is the shot's own graph with Motion Context wired in."""

    def test_the_guider_reads_motion_context_not_the_h3_node(self):
        wf, ids = graph_and_ids()
        assert wf["guider"]["inputs"]["conditioning"] == [ids["motion_context"], 0]

    def test_the_empty_latent_still_comes_from_the_h3_node(self):
        # Motion Context returns conditioning and a trim count, never a latent
        wf, ids = graph_and_ids()
        assert wf["samp"]["inputs"]["latent_image"] == ["i2v", 1]
        assert motion_context(wf, ids)["latent"] == ["i2v", 1]

    def test_the_carried_head_is_trimmed_before_the_video_is_assembled(self):
        wf, ids = graph_and_ids()
        assert wf["cvid"]["inputs"]["images"] == [ids["trim"], 0]
        assert wf["cvid"]["inputs"]["audio"] == [ids["trim"], 1]
        trim = wf[ids["trim"]]["inputs"]
        assert trim["images"] == ["vdec", 0]
        assert trim["trim_frames"] == [ids["motion_context"], 1]

    def test_the_video_vae_is_found_by_wire_not_by_name(self):
        # whichever VAE feeds the picture decode encodes the pinned frames
        graph = build_base_workflow()
        graph["vvae"]["inputs"]["vae_name"] = "some_other_name.safetensors"
        wf, ids = augment_extend(graph)
        assert motion_context(wf, ids)["vae"] == ["vvae", 0]

    def test_a_ref2v_shot_keeps_its_references(self):
        # the whole point: continuing a cast shot must not drop the cast
        prompt = VideoPrompt.blank()
        prompt.references.items.append(Reference("image", "face.png"))
        base = imagery.build_ref2v_profile(prompt)

        profile = build_extend_profile(base=base)
        graph = profile.api_format
        assert node_of(graph, "MiniMaxH3ReferenceToVideo")
        assert node_of(graph, "LoadImage")["inputs"]["image"] == "face.png"
        assert node_of(graph, "MiniMaxH3MotionContext")

    def test_a_ref2v_continuation_hangs_off_the_reference_node(self):
        prompt = VideoPrompt.blank()
        prompt.references.items.append(Reference("image", "face.png"))
        graph, ids = augment_extend(imagery.build_ref2v_workflow(prompt))
        assert motion_context(graph, ids)["conditioning"] == ["ref", 0]
        assert graph["guider"]["inputs"]["conditioning"] == [ids["motion_context"], 0]

    def test_the_bases_own_weights_and_loras_carry_over(self):
        # a continuation that swapped the checkpoint would not look like the
        # clip it continues, however well the motion carried
        base_graph = build_base_workflow()
        base_graph["unet"]["inputs"]["unet_name"] = "true_hybrid.safetensors"
        base = capture("mine", base_graph)

        graph = build_extend_profile(base=base).api_format
        assert graph["unet"]["inputs"]["unet_name"] == "true_hybrid.safetensors"

    def test_a_graph_with_no_h3_node_is_refused(self):
        with pytest.raises(ExtendError, match="H3 conditioning node"):
            augment_extend({"a": {"class_type": "LoadImage", "inputs": {}}})

    def test_a_graph_with_no_audio_is_refused(self):
        graph = build_base_workflow()
        del graph["adec"]
        with pytest.raises(ExtendError, match="VAEDecodeAudio"):
            augment_extend(graph)


class TestContextLength:
    def test_the_flag_reaches_both_motion_context_inputs(self):
        profile = build_extend_profile()
        prepared = prepare_extend(
            profile, "x", "s.mp4",
            width=608, height=352, frames=124, seconds=5.0, seed=1, context_length=5,
        )
        mc = motion_context(prepared.workflow, extend.motion_context_ids(profile))
        assert mc["context_length"] == 5
        assert mc["audio_context_length"] == 5

    def test_omitting_it_leaves_the_tested_default(self):
        profile = build_extend_profile()
        prepared = prepare_extend(
            profile, "x", "s.mp4",
            width=608, height=352, frames=124, seconds=5.0, seed=1,
        )
        mc = motion_context(prepared.workflow, extend.motion_context_ids(profile))
        assert mc["context_length"] == 22

    def test_the_pinned_length_is_recorded_on_the_prepared_render(self):
        # so the take remembers it and a re-roll carries over the same amount
        prepared = prepare_extend(
            build_extend_profile(), "x", "s.mp4",
            width=608, height=352, frames=124, seconds=5.0, seed=1, context_length=39,
        )
        assert prepared.context_length == 39
        assert prepared.to_dict()["context_length"] == 39

    def test_an_extend_that_did_not_choose_records_the_default(self):
        prepared = prepare_extend(
            build_extend_profile(), "x", "s.mp4",
            width=608, height=352, frames=124, seconds=5.0, seed=1,
        )
        assert prepared.context_length == 22


class TestContextGrid:
    def test_the_grid_matches_the_node_formula(self):
        # the video VAE's max(1, (n - 5) // 17 * 5 + 2) only distinguishes these
        assert extend.CONTEXT_GRID == (1, 5) + tuple(range(22, 244, 17))

    def test_an_off_grid_request_snaps_down_like_the_node_does(self):
        # the node snaps DOWN and only logs it; snapping here keeps the number
        # we report equal to the number actually pinned
        assert extend.snap_context(30) == 22
        assert extend.snap_context(38) == 22
        assert extend.snap_context(39) == 39

    def test_a_request_past_the_top_of_the_grid_lands_on_the_top(self):
        assert extend.snap_context(1000) == 243

    def test_a_request_below_the_grid_lands_on_the_smallest_run(self):
        assert extend.snap_context(0) == 1


class TestPrepareExtend:
    def test_the_source_and_prompt_reach_the_graph(self):
        profile = build_extend_profile()
        prepared = prepare_extend(
            profile, "she turns and walks away", "video/take_00007_.mp4 [output]",
            width=608, height=352, frames=124, seconds=124 / 24, seed=5,
        )
        wf = prepared.workflow
        source = profile.targets(ROLE_CONTEXT_VIDEO)[0].node_id
        assert wf[source]["inputs"]["file"] == "video/take_00007_.mp4 [output]"
        assert wf["i2v"]["inputs"]["prompt"] == "she turns and walks away"
        assert wf["i2v"]["inputs"]["length"] == 124
        assert wf["noise"]["inputs"]["noise_seed"] == 5

    def test_steps_reach_the_scheduler(self):
        prepared = prepare_extend(
            build_extend_profile(), "x", "s.mp4",
            width=608, height=352, frames=124, seconds=5.0, seed=1, steps=6,
        )
        assert prepared.workflow["sched"]["inputs"]["steps"] == 6

    def test_the_latent_path_injects_the_latent_file(self):
        profile = build_extend_profile(latent_backed=True)
        prepared = prepare_extend(
            profile, "x", "s.mp4 [output]",
            width=608, height=352, frames=124, seconds=5.0, seed=1,
            source_latent="h3_context/clip_00001.safetensors",
        )
        binding = profile.targets(ROLE_CONTEXT_LATENT)[0]
        assert prepared.workflow[binding.node_id]["inputs"]["latent_path"] == (
            "h3_context/clip_00001.safetensors"
        )

    def test_the_continuation_saves_a_latent_of_its_own(self):
        # not the parent's: a chain that reported its parent's latent would keep
        # feeding the next link its grandparent's sound
        prepared = prepare_extend(
            build_extend_profile(latent_backed=True), "x", "s.mp4",
            width=608, height=352, frames=124, seconds=5.0, seed=77,
            source_latent="h3_context/parent_00001.safetensors",
        )
        assert prepared.latent_path.startswith("h3_context/take77_")
        assert prepared.latent_path.endswith("_00001.safetensors")
        assert node_of(prepared.workflow, "MiniMaxH3MotionContextSaveLatent")

    def test_a_single_pass_tier_keeps_the_parents_canvas(self):
        # the megapixel budget is dropped: a continuation cannot deliver a
        # resolution the clip it continues never had
        prepared = prepare_extend(
            build_extend_profile(), "x", "s.mp4",
            width=1344, height=768, frames=124, seconds=5.0, seed=1, megapixels=0.2,
        )
        assert (prepared.width, prepared.height) == (1344, 768)
        assert prepared.workflow["i2v"]["inputs"]["width"] == 1344

    def test_a_missing_seed_is_still_reproducible(self):
        prepared = prepare_extend(
            build_extend_profile(), "x", "s.mp4",
            width=608, height=352, frames=124, seconds=5.0,
        )
        # the chosen seed is recorded and matches what went into the graph
        assert prepared.seed == prepared.workflow["noise"]["inputs"]["noise_seed"]

    def test_the_payload_is_json_clean(self):
        prepared = prepare_extend(
            build_extend_profile(), "x", "s.mp4",
            width=608, height=352, frames=124, seconds=5.0, seed=1,
        )
        json.dumps(prepared.to_dict())


class TestTwoPass:
    """A two-pass continuation proves the motion small and finishes it big."""

    def prepared(self):
        return prepare_extend(
            build_extend_profile(), "x", "s.mp4",
            width=1344, height=768, frames=124, seconds=5.0, seed=1,
            steps=20, megapixels=0.2, refine=(15, 1.0),
        )

    def test_pass_one_runs_at_the_tiers_canvas_and_the_clip_finishes_at_the_parents(self):
        prepared = self.prepared()
        assert prepared.workflow["i2v"]["inputs"]["width"] < 1344
        assert (prepared.width, prepared.height) == (1344, 768)
        assert prepared.pass1_steps == 15

    def test_pass_two_gets_a_motion_context_of_its_own(self):
        # H3 packs continuation guides into the target's spatial grid, so a
        # guide encoded at pass 1's canvas cannot condition pass 2's
        prepared = self.prepared()
        wf = prepared.workflow
        sampler2 = next(nid for nid, n in wf.items()
                        if (n.get("_meta") or {}).get("title", "").startswith("Sampler (pass 2"))
        guider2 = wf[sampler2]["inputs"]["guider"][0]
        mc2 = wf[guider2]["inputs"]["conditioning"][0]
        concat = next(nid for nid, n in wf.items()
                      if n["class_type"] == twostage.CONCAT_AV_NODE)

        assert wf[mc2]["class_type"] == "MiniMaxH3MotionContext"
        assert wf[mc2]["inputs"]["latent"] == [concat, 0]

    def test_both_passes_pin_the_same_run_of_the_same_clip(self):
        prepared = self.prepared()
        wf = prepared.workflow
        contexts = [n["inputs"] for n in wf.values()
                    if n["class_type"] == "MiniMaxH3MotionContext"]
        assert len(contexts) == 2
        assert len({c["context_frames"][0] for c in contexts}) == 1
        assert len({c["context_length"] for c in contexts}) == 1

    def test_the_saved_latent_is_the_finished_one(self):
        prepared = self.prepared()
        wf = prepared.workflow
        save = next(nid for nid, n in wf.items()
                    if n["class_type"] == "MiniMaxH3MotionContextSaveLatent")
        sampler2 = next(nid for nid, n in wf.items()
                        if (n.get("_meta") or {}).get("title", "").startswith("Sampler (pass 2"))
        assert wf[save]["inputs"]["latent"] == [sampler2, 0]


class TestBackwardsCompatibleGraph:
    def test_the_fallback_still_builds_a_whole_continuation(self):
        wf = build_extend_workflow()
        classes = {n["class_type"] for n in wf.values()}
        assert {"MiniMaxH3MotionContext", "MiniMaxH3MotionContextTrim",
                "LoadVideo", "GetVideoComponents"} <= classes
