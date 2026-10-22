from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from video_prompter import extend, imagery, twostage
from video_prompter.document import VideoPrompt, Reference


def single_pass() -> dict:
    """A minimal AV graph in the shape every H3 render path produces."""
    return {
        "sched": {"class_type": "BasicScheduler", "inputs": {"steps": 20}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 7}},
        "samp": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["ksel", 0],
                "sigmas": ["sched", 0],
                "latent_image": ["ref", 1],
            },
        },
        "vdec": {"class_type": "VAEDecode", "inputs": {"samples": ["samp", 0], "vae": ["vvae", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["samp", 0], "vae": ["avae", 0]}},
    }


class TestApply:
    def test_it_cuts_the_schedule_where_asked(self):
        graph = single_pass()
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        assert graph[ids["split"]]["inputs"]["step"] == 15
        # Pass 1 takes the head of the split, pass 2 the tail.
        assert graph["samp"]["inputs"]["sigmas"] == [ids["split"], 0]
        assert graph[ids["sampler2"]]["inputs"]["sigmas"] == [ids["split"], 1]

    def test_the_split_reads_the_schedule_the_sampler_used(self):
        graph = single_pass()
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        assert graph[ids["split"]]["inputs"]["sigmas"] == ["sched", 0]

    def test_the_upscaler_gets_the_clean_latent_not_the_noisy_one(self):
        # Slot 1 is `denoised_output`. Slot 0 still carries working noise, and
        # upscaling that would magnify it.
        graph = single_pass()
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        assert graph[ids["separate"]]["inputs"]["av_latent"] == ["samp", 1]

    def test_the_audio_half_skips_the_upscaler(self):
        graph = single_pass()
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        concat = graph[ids["concat"]]["inputs"]
        assert concat["video_latent"] == [ids["upscale"], 0]
        assert concat["audio_latent"] == [ids["separate"], 1]

    def test_the_finished_size_is_written_as_exact_dimensions(self):
        graph = single_pass()
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        upscale = graph[ids["upscale"]]["inputs"]
        # "target dimensions" is the only mode that honours width/height; the
        # node ignores its megapixels input there.
        assert upscale["mode"] == "target dimensions"
        assert (upscale["width"], upscale["height"]) == (1312, 768)

    def test_both_decodes_move_to_the_second_pass(self):
        graph = single_pass()
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        assert graph["vdec"]["inputs"]["samples"] == [ids["sampler2"], 0]
        assert graph["adec"]["inputs"]["samples"] == [ids["sampler2"], 0]

    def test_pass_two_reuses_the_same_noise_and_guider(self):
        # Same seed, same conditioning: the second pass finishes the first one's
        # picture rather than starting a different render.
        graph = single_pass()
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        second = graph[ids["sampler2"]]["inputs"]
        assert second["noise"] == ["noise", 0]
        assert second["guider"] == ["guider", 0]
        assert second["latent_image"] == [ids["concat"], 0]

    def test_an_extend_latent_save_follows_the_second_pass(self):
        # Otherwise a chain would carry the small first-pass latent forward while
        # the video it sits beside is the upscaled one.
        graph = single_pass()
        graph["savelat"] = {
            "class_type": "MiniMaxH3MotionContextSaveLatent",
            "inputs": {"latent": ["samp", 0], "filename_prefix": "h3_context/x"},
        }
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        assert graph["savelat"]["inputs"]["latent"] == [ids["sampler2"], 0]


class TestItRefuses:
    def test_a_graph_with_no_audio(self):
        graph = single_pass()
        del graph["adec"]
        with pytest.raises(twostage.TwoStageError, match="VAEDecodeAudio"):
            twostage.apply(graph, pass1_steps=15, width=1312, height=768)

    def test_a_sampler_it_cannot_read_a_clean_latent_from(self):
        graph = single_pass()
        graph["samp"]["class_type"] = "KSampler"
        with pytest.raises(twostage.TwoStageError, match="SamplerCustomAdvanced"):
            twostage.apply(graph, pass1_steps=15, width=1312, height=768)

    def test_a_schedule_that_is_not_wired(self):
        graph = single_pass()
        graph["samp"]["inputs"]["sigmas"] = 1.0
        with pytest.raises(twostage.TwoStageError, match="schedule"):
            twostage.apply(graph, pass1_steps=15, width=1312, height=768)

    def test_applying_it_twice(self):
        graph = single_pass()
        twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        with pytest.raises(twostage.TwoStageError, match="already"):
            twostage.apply(graph, pass1_steps=15, width=1312, height=768)

    def test_a_pass_one_with_no_steps(self):
        with pytest.raises(twostage.TwoStageError, match="at least one step"):
            twostage.apply(single_pass(), pass1_steps=0, width=1312, height=768)


class TestEveryRenderMode:
    """The transform finds its nodes by class, so node ids never matter."""

    def test_the_real_captured_t2v_workflow(self):
        # This fixture's ids contain colons ("105:14") and are nothing like the
        # ones the code-built graphs use.
        path = Path(__file__).parent / "fixtures" / "h3_i2v_real.json"
        graph = json.loads(path.read_text(encoding="utf-8"))
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        assert graph["105:10"]["inputs"]["samples"] == [ids["sampler2"], 0]
        assert graph[ids["separate"]]["inputs"]["av_latent"] == ["105:14", 1]

    def test_a_ref2v_graph(self):
        prompt = VideoPrompt.blank(8)
        prompt.references.items.append(Reference(kind="image", path="face.png", role="subject"))
        graph = imagery.build_ref2v_workflow(prompt)
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        assert graph["adec"]["inputs"]["samples"] == [ids["sampler2"], 0]

    def test_an_extend_graph_keeps_its_trim(self):
        # The trim reads decoded frames, which stay where they are; both decodes
        # move to the finished pass. (What the second pass *conditions* on is
        # `extend.rebind_two_stage`'s job, tested in test_extend.)
        graph = extend.build_extend_workflow()
        trim = next(nid for nid, node in graph.items()
                    if node["class_type"] == "MiniMaxH3MotionContextTrim")
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        assert graph["adec"]["inputs"]["samples"] == [ids["sampler2"], 0]
        assert graph[trim]["inputs"]["images"] == ["vdec", 0]

    def test_it_does_not_collide_with_existing_node_ids(self):
        graph = single_pass()
        graph["vp_split"] = {"class_type": "Note", "inputs": {}}
        ids = twostage.apply(graph, pass1_steps=15, width=1312, height=768)
        assert ids["split"] != "vp_split"
        assert graph["vp_split"]["class_type"] == "Note"


class TestSingularityTwoPass:
    def test_apply_singularity_2pass_on_minimal_graph(self):
        graph = single_pass()
        ids = twostage.apply_singularity_2pass(
            graph, pass1_steps=20, width=1920, height=1088, pass2_steps=4, denoise2=0.35
        )
        # Pass 1 scheduler
        assert graph["sched"]["inputs"]["steps"] == 20
        assert graph["sched"]["inputs"]["denoise"] == 1.0

        # Upscaler chain reads denoised slot 1
        assert graph[ids["separate"]]["inputs"]["av_latent"] == ["samp", 1]
        upscale = graph[ids["upscale"]]["inputs"]
        assert upscale["mode"] == "target dimensions"
        assert (upscale["width"], upscale["height"]) == (1920, 1088)
        concat = graph[ids["concat"]]["inputs"]
        assert concat["video_latent"] == [ids["upscale"], 0]
        assert concat["audio_latent"] == [ids["separate"], 1]

        # Pass 2 turbo LoRA & scheduler
        assert graph[ids["lora2"]]["class_type"] == "LoraLoaderModelOnly"
        assert graph[ids["lora2"]]["inputs"]["strength_model"] == 1.0
        assert graph[ids["sched2"]]["inputs"]["steps"] == 4
        assert graph[ids["sched2"]]["inputs"]["denoise"] == 0.35

        # Pass 2 sampler & repointing
        assert graph[ids["sampler2"]]["inputs"]["sigmas"] == [ids["sched2"], 0]
        assert graph[ids["sampler2"]]["inputs"]["latent_image"] == [ids["concat"], 0]
        assert graph["vdec"]["inputs"]["samples"] == [ids["sampler2"], 0]
        assert graph["adec"]["inputs"]["samples"] == [ids["sampler2"], 0]

    def test_it_swaps_unet_loader_to_singularity_checkpoint(self):
        graph = single_pass()
        graph["unet"] = {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "stock_unet.safetensors"},
        }
        graph["shift"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": ["unet", 0], "shift": 12.0},
        }
        graph["guider"] = {
            "class_type": "BasicGuider",
            "inputs": {"model": ["shift", 0], "conditioning": ["ref", 0]},
        }
        ids = twostage.apply_singularity_2pass(
            graph, pass1_steps=20, width=1920, height=1088
        )
        assert graph["unet"]["inputs"]["unet_name"] == twostage.DEFAULT_SINGULARITY_UNET
        assert graph[ids["lora2"]]["inputs"]["model"] == ["unet", 0]
        assert graph[ids["shift2"]]["inputs"]["model"] == [ids["lora2"], 0]
        assert graph[ids["guider2"]]["inputs"]["model"] == [ids["shift2"], 0]
        assert graph[ids["sched2"]]["inputs"]["model"] == [ids["shift2"], 0]

    def test_it_bypasses_existing_pass1_turbo_lora(self):
        graph = single_pass()
        graph["unet"] = {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "stock_unet.safetensors"},
        }
        graph["turbo"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["unet", 0], "strength_model": 1.0},
        }
        graph["shift"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": ["turbo", 0], "shift": 12.0},
        }
        graph["guider"] = {
            "class_type": "BasicGuider",
            "inputs": {"model": ["shift", 0], "conditioning": ["ref", 0]},
        }
        ids = twostage.apply_singularity_2pass(
            graph, pass1_steps=20, width=1920, height=1088
        )
        # Pass 1 turbo LoRA bypassed
        assert graph["turbo"]["inputs"]["strength_model"] == 0.0
        # Pass 2 turbo LoRA wired to base UNET
        assert graph[ids["lora2"]]["inputs"]["model"] == ["unet", 0]

    def test_real_captured_workflow_singularity(self):
        path = Path(__file__).parent / "fixtures" / "h3_i2v_real.json"
        graph = json.loads(path.read_text(encoding="utf-8"))
        ids = twostage.apply_singularity_2pass(
            graph, pass1_steps=20, width=1920, height=1088
        )
        assert graph["105:10"]["inputs"]["samples"] == [ids["sampler2"], 0]
        assert graph[ids["separate"]]["inputs"]["av_latent"] == ["105:14", 1]
        assert graph["105:6"]["inputs"]["unet_name"] == twostage.DEFAULT_SINGULARITY_UNET

    def test_ref2v_workflow_singularity(self):
        prompt = VideoPrompt.blank(8)
        prompt.references.items.append(Reference(kind="image", path="face.png", role="subject"))
        graph = imagery.build_ref2v_workflow(prompt)
        ids = twostage.apply_singularity_2pass(
            graph, pass1_steps=20, width=1920, height=1088
        )
        assert graph["adec"]["inputs"]["samples"] == [ids["sampler2"], 0]
        assert graph[ids["sched2"]]["inputs"]["steps"] == 4
        assert graph[ids["sched2"]]["inputs"]["denoise"] == 0.35

    def test_it_handles_workflow_without_shift_or_lora(self):
        graph = single_pass()
        graph["unet"] = {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "stock_unet.safetensors"},
        }
        # Guider directly connected to UNETLoader (like H3 Hybrid local t2v profile)
        graph["guider"] = {
            "class_type": "BasicGuider",
            "inputs": {"model": ["unet", 0], "conditioning": ["ref", 0]},
        }
        ids = twostage.apply_singularity_2pass(
            graph, pass1_steps=20, width=1920, height=1088
        )
        assert graph["unet"]["inputs"]["unet_name"] == twostage.DEFAULT_SINGULARITY_UNET
        assert graph[ids["lora2"]]["inputs"]["model"] == ["unet", 0]
        # Shift2 must have required shift_video and shift_audio inputs
        shift2_inputs = graph[ids["shift2"]]["inputs"]
        assert shift2_inputs["model"] == [ids["lora2"], 0]
        assert shift2_inputs["shift_video"] == 12.0
        assert shift2_inputs["shift_audio"] == 3.0
        assert "shift" not in shift2_inputs
        assert graph[ids["guider2"]]["inputs"]["model"] == [ids["shift2"], 0]

