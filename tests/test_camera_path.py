import json
from pathlib import Path

import pytest

from video_prompter import Beat, Storyboard, VideoPrompt
from video_prompter.camera_path import validate_path
from video_prompter.compose import compose
from video_prompter.provenance import spans
from video_prompter.serialize import prompt_from_dict, prompt_to_dict
from video_prompter.lint import lint
from video_prompter.profiles import capture
from video_prompter.render import prepare_render, RenderBlocked


def path():
    return dict(target="the kitchen island", interpolation="smooth", freeze=False, keyframes=[
        dict(time=0, azimuth=0, elevation=0, distance=1),
        dict(time=.5, azimuth=45, elevation=10, distance=.8),
        dict(time=1, azimuth=90, elevation=0, distance=1),
    ])


def prompt():
    return VideoPrompt(storyboard=Storyboard(5, [Beat(0, 5, "a wide kitchen view", "the kettle steams",
                       camera="holds still", camera_path=path())]), style="Live-action", audio="A kettle hisses.")


def test_round_trip_and_retime_preserve_coordinates_and_manual_camera():
    source = prompt()
    encoded = prompt_to_dict(source)
    restored = prompt_from_dict(json.loads(json.dumps(encoded)))
    assert restored.storyboard.beats[0].camera_path == path()
    assert restored.storyboard.beats[0].camera == "holds still"
    text = compose(restored)
    assert "camera's right" in text and "00:05.125" in text
    assert "the kettle steams" in text and "holds still" not in text
    assert "00:02.562" in text
    assert prompt_to_dict(restored) == encoded  # composing never destroys saved prose
    restored.storyboard = restored.storyboard.refit(10)
    text = compose(restored)
    assert "00:10.083" in text and "00:05.042" in text
    assert restored.storyboard.beats[0].camera_path == path()
    restored.storyboard.beats[0].camera_path = None
    assert "The camera holds still." in compose(restored)


def test_no_cut_paths_stay_inside_their_beat_and_are_tinted_as_camera():
    source = prompt()
    source.storyboard.beats = [Beat(0, 2, "wide", "the room is quiet"),
                              Beat(2, 5, action="the kettle steams", cut=False, camera_path=path())]
    text = compose(source)
    assert text.count("[Shot 1]") == 1 and "[Shot 2]" not in text
    assert "from 00:02.000" in text
    colored = spans(source)
    assert any(s["field"] == "beat.1.camera" and "orbits" in text[s["start"]:s["end"]] for s in colored)
    assert source.storyboard.beats[1].camera_path == path()


@pytest.mark.parametrize("invalid", [False, [], {}, {"keyframes": []},
    {**path(), "interpolation": "bounce"},
    {**path(), "keyframes": [{**p, "time": 0} for p in path()["keyframes"]]},
    {**path(), "keyframes": [path()["keyframes"][0], dict(time=1,azimuth=float("nan"),elevation=0,distance=1)]},
    {**path(), "keyframes": [path()["keyframes"][0], dict(time=1,azimuth=90,elevation=0,distance=0)]},
])
def test_invalid_paths_are_lint_errors_without_crashing_preview(invalid):
    source = prompt()
    source.storyboard.beats[0].camera_path = invalid
    assert validate_path(invalid)
    assert any(f.rule == "camera_path.invalid" and str(f.severity) == "error" for f in lint(source))
    assert compose(source)


def test_render_injects_planned_motion_and_rejects_bad_points():
    graph = json.loads((Path(__file__).parents[1] / "examples/h3_t2v.json").read_text(encoding="utf-8"))
    profile = capture("test", graph)
    source = prompt()
    rendered = prepare_render(source, profile, seed=20260911)
    assert "camera's right" in rendered.prompt_text
    assert any(rendered.prompt_text == value for node in rendered.workflow.values() for value in node["inputs"].values())
    assert rendered.frames == 124
    source.storyboard.beats[0].camera_path["keyframes"][-1]["distance"] = -1
    with pytest.raises(RenderBlocked):
        prepare_render(source, profile, seed=20260911)


def test_full_turn_keeps_unwrapped_direction_without_claiming_loop_anchor():
    source = prompt()
    source.storyboard.beats[0].camera_path["keyframes"] = [path()["keyframes"][0], dict(time=1, azimuth=-360, elevation=0, distance=1)]
    text = compose(source)
    assert "orbits 360 degrees toward the camera's left" in text
    assert "last frame" not in text


def test_old_documents_do_not_gain_a_camera_path_key():
    payload = dict(duration=5, beats=[dict(start=0,end=5,camera="holds still")])
    assert "camera_path" not in prompt_to_dict(prompt_from_dict(payload))["beats"][0]


def test_motion_pass_resolves_media_without_changing_saved_anchors_and_tags():
    from video_prompter.camera_path import camera_media
    from video_prompter.document import Reference
    from video_prompter.imagery import build_ref2v_workflow, render_mode
    from video_prompter.profiles import ProfileError
    source = prompt()
    source.first_frame, source.last_frame = "first.png", "last.png"
    source.references.items.append(Reference("video", "other.mp4", "a style example"))
    source.storyboard.beats[0].camera_path.update(frame_mode="motion", reference_video="action.mp4")
    saved = prompt_to_dict(source)
    assert render_mode(source) == "ref2v"
    effective = camera_media(source)
    assert not effective.first_frame and not effective.last_frame
    assert len(camera_media(effective).references.items) == 2  # repeated resolution is idempotent
    assert "preserves <Video 2>'s action order" in compose(source)
    assert not [f for f in lint(source) if f.rule == "mode.conflict"]
    with pytest.raises(ProfileError, match="Ref2VA"):
        build_ref2v_workflow(source)
    graph = build_ref2v_workflow(source, {"unet": "h3_ref2va.safetensors"})
    prep = graph["vp_camera_frames_1"]["inputs"]
    assert prep == dict(images=["refvidc_1", 0], source_fps=["refvidc_1", 2])
    assert graph["ref"]["inputs"]["ref_videos.ref_video_1"] == ["vp_camera_frames_1", 0]
    assert not any("ref_video_audio" in k for k in graph["ref"]["inputs"])
    assert prompt_to_dict(source) == saved
    source.storyboard.beats[0].camera_path["frame_mode"] = "scene"
    assert camera_media(source).first_frame == "first.png"


def test_loop_closure_wires_real_endpoint_only_when_compatible():
    from video_prompter.camera_path import loop_status
    from video_prompter.imagery import build_i2v_profile
    source = prompt()
    source.first_frame = "first.png"
    path = source.storyboard.beats[0].camera_path
    path.update(freeze=True, loop_closure="auto")
    path["keyframes"][-1]["azimuth"] = 360
    assert loop_status(source)[0]
    graph = json.loads((Path(__file__).parents[1] / "examples/h3_t2v.json").read_text())
    profile = build_i2v_profile(source, capture("test", graph))
    node = next(n for n in profile.api_format.values() if n["class_type"] == "MiniMaxH3ImageToVideo")
    for slot in ("first_frame", "last_frame"):
        assert profile.api_format[node["inputs"][slot][0]]["inputs"]["image"] == "first.png"
    assert source.last_frame == ""
    for value in (359, 720):
        path["keyframes"][-1]["azimuth"] = value
        assert not loop_status(source)[0]
    path["keyframes"][-1]["azimuth"] = -360
    source.last_frame = "custom.png"
    assert not loop_status(source)[0]
    assert any(f.rule == "camera_path.loop_disabled" for f in lint(source))


@pytest.mark.parametrize("fps,count,expected", [(24,124,124),(30,150,107),(60,300,107),(12,60,107)])
def test_motion_reference_resampling_uses_source_fps_and_h3_grid(fps, count, expected):
    from video_prompter.camera_frames import frame_indices
    indices = frame_indices(count, fps)
    assert len(indices) == expected
    assert indices[0] == 0 and indices[-1] < count
    assert len(indices) % 17 == 5
    assert abs(indices[-1]/fps - (len(indices)-1)/24) < 1/fps + 1e-8


@pytest.mark.parametrize("fps,count", [(0,124),(float("nan"),124),(24,20),(24,361)])
def test_motion_reference_rejects_bad_duration_or_fps(fps, count):
    from video_prompter.camera_frames import frame_indices
    with pytest.raises(ValueError):
        frame_indices(count, fps)


def test_subject_box_experiments_and_motion_scope_validation():
    source = prompt()
    path = source.storyboard.beats[0].camera_path
    path.update(subject_box=dict(left=.2, top=.1, width=.4, height=.6), experiment="compact")
    assert "[L=0.2000,T=0.1000,W=0.4000,H=0.6000]" in compose(source)
    assert "Camera keyframes:" in compose(source)
    path["experiment"] = "numbers"
    assert "physically orbits" not in compose(source)
    path["subject_box"]["width"] = .9
    assert validate_path(path)
    path.pop("subject_box")
    path.update(frame_mode="motion", reference_video="action.mp4")
    source.storyboard.beats.append(Beat(2, 5, action="a second shot"))
    assert any(f.rule == "camera_path.invalid" for f in lint(source))


@pytest.mark.parametrize("distance,direction,change,size", [(.3, "straight toward", "decreases", "larger"),
                                                          (2, "straight away from", "increases", "smaller")])
def test_pure_dolly_compiles_a_single_direction_without_orbit_instruction(distance, direction, change, size):
    source = prompt()
    source.storyboard.beats[0].camera_path["keyframes"] = [
        dict(time=0,azimuth=0,elevation=0,distance=1), dict(time=1,azimuth=0,elevation=0,distance=distance)]
    text = compose(source)
    assert direction in text and f"distance {change} continuously" in text
    assert f"progressively {size}" in text
    assert "path around" not in text


def test_dolly_with_authored_reversal_does_not_claim_one_way_movement():
    source = prompt()
    source.storyboard.beats[0].camera_path["keyframes"] = [
        dict(time=0,azimuth=0,elevation=0,distance=1),dict(time=.5,azimuth=0,elevation=0,distance=1.5),
        dict(time=1,azimuth=0,elevation=0,distance=.3)]
    text = compose(source)
    assert "toward and away" in text
    assert "moves only closer" not in text
