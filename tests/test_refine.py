"""Tests for take refinement workflow generation and character reference injection."""

import pytest
from video_prompter import refine
from video_prompter.api import Api
from video_prompter.cast import Member
from video_prompter.document import Reference
from video_prompter.presets import PresetLibrary, bundled_root
from video_prompter.profiles import ProfileStore
from video_prompter.projects import ProjectStore
from video_prompter.sources import blank_source


@pytest.fixture
def api(tmp_path):
    return Api(
        projects=ProjectStore(tmp_path / "projects"),
        profiles=ProfileStore(tmp_path / "profiles"),
        presets=PresetLibrary(tmp_path / "presets", bundled_root()),
    )


def test_refine_spec_defaults():
    spec = refine.RefineSpec.from_request({})
    assert spec.guided_resample is True
    assert spec.face_refine is True
    assert spec.face_denoise == 0.25
    assert spec.face_target == "small_faces"
    assert spec.reference_face is None
    assert spec.face_prompt is None
    assert spec.adaptive_face_scale is True
    assert spec.min_magnification == 1.20
    assert spec.max_magnification == 1.60


def test_seedvr_api_records_mode_and_exact_output_without_cast_identity(api):
    shot = _cast_project(api, "SeedVRCast", in_shot=1)
    p = _refine(api, "SeedVRCast", shot, enhancement_mode="seedvr2_faces",
                enhancement_resolution="original", width=1280, height=720)
    assert p["reference_face"] is None
    assert p["refinement"] == ["seedvr2", "face_blend"]
    assert (p["width"], p["height"]) == (1280, 720)
    assert p["refined_from"] == "take123"


def test_refine_spec_custom_overrides():
    spec = refine.RefineSpec.from_request({
        "guided_resample": False,
        "face_refine": True,
        "face_refine_denoise": 0.20,
        "reference_face": "video_prompter/face2.png",
        "face_prompt": "Custom face prompt",
        "adaptive_face_scale": False,
        "min_magnification": 1.30,
        "max_magnification": 1.75,
    })
    assert spec.guided_resample is False
    assert spec.face_refine is True
    assert spec.face_denoise == 0.20
    assert spec.reference_face == "video_prompter/face2.png"
    assert spec.face_prompt == "Custom face prompt"
    assert spec.adaptive_face_scale is False
    assert spec.min_magnification == 1.30
    assert spec.max_magnification == 1.75


def test_build_refine_workflow_without_reference_face():
    spec = refine.RefineSpec(guided_resample=False, face_refine=True)
    wf = refine.build_refine_workflow(
        source_video="video/test.mp4 [output]",
        source_width=1248,
        source_height=768,
        spec=spec,
    )
    face_nodes = [n for n in wf.values() if n.get("class_type") == "VideoPrompterSmallFaceRefine"]
    assert len(face_nodes) == 1
    node = face_nodes[0]
    assert "reference_face" not in node["inputs"]
    assert node["inputs"]["denoise"] == 0.25
    assert node["inputs"]["min_face_size"] == 8
    assert node["inputs"]["max_face_size"] == 192
    assert "Picture 1" not in node["inputs"]["prompt"]


def test_build_refine_workflow_with_reference_face():
    spec = refine.RefineSpec(
        guided_resample=False,
        face_refine=True,
        reference_face="video_prompter/face2.png",
        face_target="reference_identity",
    )
    wf = refine.build_refine_workflow(
        source_video="video/test.mp4 [output]",
        source_width=1248,
        source_height=768,
        spec=spec,
    )
    face_nodes = [n for n in wf.values() if n.get("class_type") == "FeiHouEasyH3RHFaceRefine"]
    load_img_nodes = [n for n in wf.values() if n.get("class_type") == "LoadImage"]

    assert len(face_nodes) == 1
    assert len(load_img_nodes) == 1

    node = face_nodes[0]
    assert "reference_face" in node["inputs"]
    ref_link = node["inputs"]["reference_face"]
    assert wf[ref_link[0]]["class_type"] == "LoadImage"
    assert wf[ref_link[0]]["inputs"]["image"] == "video_prompter/face2.png"
    assert "Picture 1" in node["inputs"]["prompt"]
    assert "youthful" not in node["inputs"]["prompt"]
    assert "wrinkles" not in node["inputs"]["prompt"]


def _character(name: str, face: str) -> Member:
    return Member(
        name=name,
        kind="character",
        appearance="a young woman with short textured hair",
        references=[
            Reference(kind="image", path=f"video_prompter/{name}_sheet.jpg", role="full body sheet"),
            Reference(kind="image", path=face, role=f"a close study of {name}'s face"),
        ],
    )


def _cast_project(api: Api, name: str, *, in_shot: int, extras: int = 0) -> str:
    """A project with one source; `in_shot` characters cast on it, `extras` not."""
    api.create_project({"name": name})
    project = api.projects.load(name)
    members = [project.cast.add(_character(f"m{i}", f"video_prompter/face{i}.png"))
               for i in range(in_shot + extras)]
    source = project.add_source(blank_source("Shot"))
    source.segments[0].prompt.cast = [m.id for m in members[:in_shot]]
    api.projects.save(project)
    return source.id


def _refine(api: Api, project: str, shot: str | None, **extra) -> dict:
    return api.prepare_refine({
        "project": project,
        "source": "video/test.mp4 [output]",
        "take_id": "take123",
        "shot": shot,
        "width": 1280,
        "height": 720,
        "face_refine": True,
        "guided_resample": False,
        **extra,
    })


def test_api_prepare_refine_uses_two_characters_in_a_shot_as_no_reference(api: Api):
    # Face refine works on the largest face; either picture could be the wrong person.
    shot = _cast_project(api, "RefineTwo", in_shot=2)
    assert _refine(api, "RefineTwo", shot)["reference_face"] is None


def test_api_prepare_refine_never_borrows_a_face_from_outside_the_shot(api: Api):
    shot = _cast_project(api, "RefineNobody", in_shot=0, extras=1)
    assert _refine(api, "RefineNobody", shot)["reference_face"] is None


def test_small_faces_never_borrows_single_cast_face(api: Api):
    shot = _cast_project(api, "SmallFaces", in_shot=1)
    prepared = _refine(api, "SmallFaces", shot)
    assert prepared["reference_face"] is None
    assert any(n["class_type"] == "VideoPrompterSmallFaceRefine" for n in prepared["workflow"].values())


def test_reference_mode_requires_a_reference_and_crowd_mode_rejects_shared_identity():
    for spec in (refine.RefineSpec(face_target="reference_identity"),
                 refine.RefineSpec(reference_face="face.png"),
                 refine.RefineSpec(face_target="all_faces_typo")):
        with pytest.raises(ValueError):
            refine.build_refine_workflow(source_video="test.mp4", source_width=1280, source_height=720, spec=spec)


def test_explicit_reference_selects_identity_mode():
    spec = refine.RefineSpec.from_request({"reference_face": "face.png"})
    assert spec.face_target == "reference_identity"
    assert spec.face_denoise == 0.18


def test_api_prepare_refine_records_the_prompt_and_the_takes_length(api: Api):
    # An empty prompt_text makes the refined take impossible to re-shoot.
    api.create_project({"name": "RefinePrompt"})
    prepared = _refine(api, "RefinePrompt", None, prompt_text="A fox crosses the snow.", seconds=9.5)
    assert prepared["prompt_text"] == "A fox crosses the snow."
    assert prepared["seconds"] == 9.5


def test_refine_spec_keeps_a_zero_face_denoise():
    assert refine.RefineSpec.from_request({"face_refine_denoise": 0}).face_denoise == 0.0


def test_api_prepare_refine_autoresolves_character_face_for_single_face_mode(api: Api):
    project_name = "RefineCastTest"
    shot = _cast_project(api, project_name, in_shot=1, extras=1)

    prepared = _refine(api, project_name, shot, face_refine_target="largest_face")

    assert prepared["reference_face"] == "video_prompter/face0.png"
    assert prepared["quality"] == "refined"

    wf = prepared["workflow"]
    face_nodes = [n for n in wf.values() if n.get("class_type") == "FeiHouEasyH3RHFaceRefine"]
    assert len(face_nodes) == 1
    prompt = face_nodes[0]["inputs"]["prompt"]
    assert "Picture 1" in prompt
    assert "youthful" not in prompt
    assert "wrinkles" not in prompt


def test_build_refine_workflow_upscale_1080p():
    spec = refine.RefineSpec(
        guided_resample=True,
        face_refine=True,
        guide_megapixels=2.1,
    )
    prepared = refine.prepare_refine(
        source_video="video/test.mp4 [output]",
        source_take_id="take123",
        source_width=1248,
        source_height=768,
        spec=spec,
    )
    assert prepared.width == 1856
    assert prepared.height == 1152
    assert "guided_resample" in prepared.refinement
    assert "face_refine" in prepared.refinement
    assert "upscale" in prepared.refinement

    wf = prepared.workflow
    # Guided resample resize node should use native aligned dimensions (1248x768)
    resize_nodes = [n for n in wf.values() if n.get("class_type") == "ImageResizeKJv2"]
    assert len(resize_nodes) == 1  # guide input only; repaired crops go straight into the delivery canvas
    assert resize_nodes[0]["inputs"]["width"] == 1248
    assert resize_nodes[0]["inputs"]["height"] == 768

    face_id, face = next((nid, n) for nid, n in wf.items() if n["class_type"] == "VideoPrompterSmallFaceRefine")
    assert face["inputs"]["output_width"] == 1856
    assert face["inputs"]["output_height"] == 1152
    assert face["inputs"]["tiled_detection"] is True
    upscaler_id = face["inputs"]["upscale_model"][0]
    assert wf[upscaler_id]["class_type"] == "UpscaleModelLoader"
    assert wf[upscaler_id]["inputs"]["model_name"] == "RealESRGAN_x4plus.safetensors"

    # Preview override should be absent
    assert not any(n.get("class_type") == "ModelPreviewOverrideKJ" for n in wf.values())

    # VHS_VideoCombine should receive the upscaled frames
    combine_node = next(n for n in wf.values() if n.get("class_type") == "VHS_VideoCombine")
    assert combine_node["inputs"]["images"][0] == face_id


def test_face_preupscale_is_optional_and_only_loaded_for_small_face_enlargement():
    for target, megapixels, override in (
        ("small_faces", None, None),
        ("largest_face", 2.1, None),
        ("small_faces", 2.1, {"face_upscale_model": ""}),
    ):
        wf = refine.build_refine_workflow(source_video="test.mp4", source_width=1280,
            source_height=720, spec=refine.RefineSpec(guided_resample=False,
            face_target=target, guide_megapixels=megapixels), model_overrides=override)
        assert not any(n["class_type"] == "UpscaleModelLoader" for n in wf.values())


def test_build_refine_workflow_native_resolution():
    spec = refine.RefineSpec(
        guided_resample=True,
        face_refine=True,
        guide_megapixels=None,
    )
    prepared = refine.prepare_refine(
        source_video="video/test.mp4 [output]",
        source_take_id="take123",
        source_width=1248,
        source_height=768,
        spec=spec,
    )
    assert prepared.width == 1248
    assert prepared.height == 768
    assert "guided_resample" in prepared.refinement
    assert "face_refine" in prepared.refinement
    assert "upscale" not in prepared.refinement

    wf = prepared.workflow
    # Only 1 resize node for the guide input
    resize_nodes = [n for n in wf.values() if n.get("class_type") == "ImageResizeKJv2"]
    assert len(resize_nodes) == 1
    assert resize_nodes[0]["inputs"]["width"] == 1248
    assert resize_nodes[0]["inputs"]["height"] == 768
