from __future__ import annotations

import pytest

from video_prompter import selflift
from video_prompter.selflift import SelfLiftError


def mock_av_graph() -> dict:
    """A minimal AV graph in the shape every H3 render path produces."""
    return {
        "sched": {"class_type": "BasicScheduler", "inputs": {"steps": 8}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "ksel": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "guider": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["unet", 0], "conditioning": ["pos", 0]},
        },
        "samp": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["ksel", 0],
                "sigmas": ["sched", 0],
                "latent_image": ["empty_latent", 0],
            },
        },
        "vdec": {"class_type": "VAEDecode", "inputs": {"samples": ["samp", 0], "vae": ["vvae", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["samp", 0], "vae": ["avae", 0]}},
        "empty_latent": {"class_type": "EmptyLatentImage", "inputs": {"width": 640, "height": 384}},
    }


class TestSelfLiftApply:
    def test_it_transforms_workflow_into_selflift_nodes(self):
        graph = mock_av_graph()
        ids = selflift.apply(
            graph,
            pass1_steps=6,
            width=1920,
            height=1088,
            lowres_scale=0.5,
            tst_tau=0.2,
        )

        assert ids["tst"] in graph
        assert ids["sampler"] in graph

        # KSamplerSelect must be forced to euler for SelfLift
        assert graph["ksel"]["inputs"]["sampler_name"] == "euler"

        # TST node assertions
        tst = graph[ids["tst"]]
        assert tst["class_type"] == selflift.SELFLIFT_TST_NODE
        assert tst["inputs"]["model"] == ["unet", 0]
        assert tst["inputs"]["tau"] == 0.2

        # SelfLift sampler assertions
        sampler = graph[ids["sampler"]]
        assert sampler["class_type"] == selflift.SELFLIFT_SAMPLER_NODE
        assert sampler["inputs"]["model"] == [ids["tst"], 0]
        assert sampler["inputs"]["sampler"] == ["ksel", 0]
        assert sampler["inputs"]["transition_step"] == 6
        assert sampler["inputs"]["lowres_scale"] == 0.5
        assert sampler["inputs"]["seed"] == 42
        assert sampler["inputs"]["upscaler_model"] == selflift.DEFAULT_UPSCALER_MODEL

        # Downstream decoders must be repointed to SelfLiftH3Sampler
        assert graph["vdec"]["inputs"]["samples"] == [ids["sampler"], 0]
        assert graph["adec"]["inputs"]["samples"] == [ids["sampler"], 0]

        # Geometry resized to target width x height
        assert graph["empty_latent"]["inputs"]["width"] == 1920
        assert graph["empty_latent"]["inputs"]["height"] == 1088

    def test_it_sizes_a_canvas_wired_from_a_megapixel_node(self):
        # "H3 T2V w Turbo" sizes its latent through ResolutionSelector, which
        # still holds pass 1's small budget. Following the wire rendered the
        # finished take at preview size while reporting 1312x768.
        graph = mock_av_graph()
        graph["res"] = {"class_type": "ResolutionSelector", "inputs": {"megapixels": 0.2}}
        graph["empty_latent"]["inputs"].update(width=["res", 0], height=["res", 1])
        selflift.apply(graph, pass1_steps=6, width=1920, height=1088)
        assert graph["empty_latent"]["inputs"]["width"] == 1920
        assert graph["empty_latent"]["inputs"]["height"] == 1088

    def test_it_leaves_other_sized_nodes_alone(self):
        graph = mock_av_graph()
        graph["crop"] = {"class_type": "ImageResizeKJv2", "inputs": {"width": 512, "height": 512}}
        selflift.apply(graph, pass1_steps=6, width=1920, height=1088)
        assert graph["crop"]["inputs"]["width"] == 512

    def test_it_rejects_a_latent_without_a_canvas(self):
        graph = mock_av_graph()
        graph["empty_latent"]["inputs"] = {}
        with pytest.raises(SelfLiftError, match="width and height"):
            selflift.apply(graph, pass1_steps=6, width=1920, height=1088)

    def test_1080p_tiers_finish_at_1080p(self):
        from video_prompter import quality
        from video_prompter.render import split_into_two_passes
        from video_prompter.document import VideoPrompt
        from video_prompter.storyboard import Beat, Storyboard

        prompt = VideoPrompt(storyboard=Storyboard(5, [Beat(0, 5, "a field", "a fox trots")]))
        for tier in ("selflift_fhd", "selflift_final"):
            refine = quality.REFINE_PRESETS[tier]
            assert split_into_two_passes(mock_av_graph(), prompt, refine, is_selflift=True) == (1920, 1088)
        # The classic two-stage split keeps the 768-edge ceiling.
        assert split_into_two_passes(mock_av_graph(), prompt, (6, 2.1)) == (1344, 768)

    def test_it_rejects_graph_missing_audio_decode(self):
        graph = mock_av_graph()
        del graph["adec"]
        with pytest.raises(SelfLiftError, match="found no VAEDecodeAudio"):
            selflift.apply(graph, pass1_steps=6, width=1280, height=720)

    def test_it_rejects_graph_with_unwired_guider(self):
        graph = mock_av_graph()
        del graph["guider"]
        with pytest.raises(SelfLiftError, match="not wired to a guider"):
            selflift.apply(graph, pass1_steps=6, width=1280, height=720)
