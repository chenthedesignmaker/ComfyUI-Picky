"""Regressions for small faces, crossings, cuts and compositing exclusion."""
from video_prompter.small_faces import FaceTrack, plan_tracks, track_pick, restrict_transform


def box(x, height=30):
    return [x, 20, x + height * 0.7, 20 + height]


def test_all_small_faces_selected_and_closeups_excluded():
    tracks, total = plan_tracks([[box(0), box(100, 50), box(400, 300)] for _ in range(8)])
    assert total == 3
    assert len(tracks) == 2
    assert [t.boxes[0][0] for t in tracks] == [0, 100]


def test_detection_order_does_not_change_people():
    detections = [[box(i), box(100 - i)][::(-1 if i % 2 else 1)] for i in range(8)]
    tracks, _ = plan_tracks(detections)
    assert len(tracks) == 2
    assert [tracks[0].boxes[i][0] for i in range(8)] == list(range(8))


def test_cut_resets_tracks_even_at_same_position():
    tracks, _ = plan_tracks([[box(0)] for _ in range(12)], cuts=[6])
    assert [list(t.boxes) for t in tracks] == [list(range(6)), list(range(6, 12))]


def test_dropout_and_late_arrival_are_independent():
    detections = [[box(0)] for _ in range(6)] + [[]] + [[box(0), box(100)] for _ in range(6)]
    tracks, _ = plan_tracks(detections)
    assert len(tracks) == 2
    assert 6 not in tracks[0].boxes
    assert min(tracks[1].boxes) == 7


def test_ambiguous_crossing_does_not_duplicate_or_swap_tracks():
    detections = [[box(10 * i), box(100 - 10 * i)] for i in range(11)]
    tracks, _ = plan_tracks(detections)
    assert len(tracks) == 2
    assert 5 not in tracks[0].boxes and 5 not in tracks[1].boxes
    assert tracks[0].boxes[10][0] == 100
    assert tracks[1].boxes[10][0] == 0


def test_tiny_faces_and_brief_false_positives_are_not_refined():
    tracks, _ = plan_tracks([[box(0, 6)]] * 10 + [[box(100)]] * 2)
    assert tracks == []


def test_pick_uses_only_one_tracks_original_boxes_and_local_frame_numbers():
    track = FaceTrack({5: box(0), 7: box(2)})
    pick = track_pick(track, 5, 8, 1344, 736)
    assert pick["boxes"] == [[box(0)], [], [box(2)]]
    assert pick["frames"] == 3 and pick["src_size"] == (1344, 736)


def test_stitch_weights_zero_missing_and_large_faces_even_after_smoothing():
    track = FaceTrack({5: box(0), 7: box(2, 300)})
    transform = {"weights": [1.0, 0.7, 1.0], "source": [0, 1, 2]}
    restrict_transform(transform, track, 5, 16, 192)
    assert transform["weights"] == [1.0, 0.0, 0.0]


def test_adapter_refines_every_track_from_original_and_accumulates_composites(monkeypatch):
    import sys
    import threading
    from dataclasses import dataclass
    from types import ModuleType, SimpleNamespace
    import pytest
    torch = pytest.importorskip("torch")
    from video_prompter import small_faces

    @dataclass(frozen=True)
    class Context:
        fps: float = 24.0
        reference_image_1: object = "wrong shared identity"

    backend_module = ModuleType("test_face_backend.sampling")
    backend_module._LOCK = threading.RLock()
    tracker = ModuleType("test_face_backend.face_refine_tracking")
    tracker._DETECTOR_CACHE = {}
    tracker._REC_CACHE = {}
    calls = []

    class Crop:
        def run(self, **kwargs):
            assert torch.count_nonzero(kwargs["images"]) == 0  # never crop the previous composite
            pick = kwargs["face_pick"]
            assert all(len(boxes) == 1 for boxes in pick["boxes"])
            assert pick["picks"][0]["segment"] == (0, 5)
            calls.append(pick["boxes"][0][0][0])
            transform = {"weights": [1.] * 5, "source": list(range(5)),
                         "src_size": (160, 64), "boxes": [(0, 0, 32, 32)] * 5}
            return kwargs["images"], transform, None, "ok", 160, 64, 5

    tracker.H3FaceTrackCrop = Crop

    class Backend:
        def refine(self, images, h3_context, **kwargs):
            assert h3_context.reference_image_1 is None
            assert kwargs["reference_face"] is None and kwargs["audio"] is None
            assert kwargs["adaptive_face_scale"] is False
            return self._run(tracker, images)

        def _run(self, tr, images):
            tr.H3FaceTrackCrop().run(images=images)
            result = images.clone()
            result[:, 20, int(calls[-1]), :] = 1
            return result, None, 24., "ok"

    Backend.__module__ = backend_module.__name__
    backend_module.__package__ = "test_face_backend"
    fake_comfy = ModuleType("comfy")
    fake_mm = ModuleType("comfy.model_management")
    fake_mm.throw_exception_if_processing_interrupted = lambda: None
    fake_comfy.model_management = fake_mm
    fake_utils = ModuleType("comfy.utils")
    fake_utils.ProgressBar = lambda n: SimpleNamespace(update=lambda n: None)
    for name, value in {
        "nodes": SimpleNamespace(NODE_CLASS_MAPPINGS={"FeiHouEasyH3RHFaceRefine": Backend}),
        backend_module.__name__: backend_module, tracker.__name__: tracker,
        "comfy": fake_comfy, "comfy.model_management": fake_mm, "comfy.utils": fake_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(small_faces, "detect_faces", lambda *args, **kwargs: ([[box(0), box(100)]] * 5, []))
    images = torch.zeros(5, 64, 160, 3)
    audio = {"test": "original audio"}
    context = Context()
    output = small_faces.SmallFaceRefine().refine(images, object(), context, audio=audio)
    result, result_audio, fps, report = output["result"]
    assert calls == [0, 100]
    assert torch.count_nonzero(images) == 0
    assert torch.count_nonzero(result) == 5 * 3 * 2
    assert result_audio is audio and fps == 24
    assert context.reference_image_1 == "wrong shared identity"
    assert "Processed 2 face tracks" in report
    assert output["ui"]["face_refine_report"] == [report]

    calls.clear()
    output = small_faces.SmallFaceRefine().refine(images, object(), context, audio=audio,
                                                 output_width=320, output_height=128)
    assert output["result"][0].shape == (5, 128, 320, 3)
    assert output["result"][1] is audio
    assert calls == [0, 100]

    assert "Repaired 10/10 detected small-face observations" in report
    monkeypatch.setattr(small_faces, "detect_faces", lambda *args, **kwargs: ([[]] * 5, []))
    output = small_faces.SmallFaceRefine().refine(images, object(), context, audio=audio)
    assert output["result"][0] is images
    assert "No eligible small faces" in output["result"][3]


def test_small_persistent_faces_are_included_below_old_cutoff():
    tracks, _ = plan_tracks([[box(0, 9), box(30, 13)]] * 3)
    assert len(tracks) == 2


def test_shrinking_box_prediction_stays_valid():
    t = FaceTrack({0: [0, 0, 40, 40], 1: [10, 10, 30, 30]})
    p = t.predict(4)
    assert p[2] > p[0] and p[3] > p[1]


def test_tile_duplicates_removed_without_merging_neighbour():
    from video_prompter.small_faces import merge_detections
    candidates = [[10, 10, 30, 40, .8], [11, 11, 30, 40, .9], [35, 10, 55, 40, .7]]
    assert merge_detections(candidates) == [[11, 11, 30, 40], [35, 10, 55, 40]]


def test_delivery_transform_scales_source_geometry_only():
    from video_prompter.small_faces import scale_transform
    transform = {"src_size": (100, 50), "boxes": [(10, 5, 20, 20)],
                 "canvas": (768, 768), "face_rect": [(200, 200, 300, 300)]}
    scale_transform(transform, 200, 100)
    assert transform["src_size"] == (200, 100)
    assert transform["boxes"] == [(20, 10, 40, 40)]
    assert transform["canvas"] == (768, 768)
    assert transform["face_rect"] == [(200, 200, 300, 300)]


def test_uniform_nonface_hits_rejected_without_rejecting_dark_detail():
    import numpy as np
    from video_prompter.small_faces import has_face_detail
    blank = np.full((40, 30, 3), 5, dtype=np.uint8)
    assert not has_face_detail(blank, [0, 0, 30, 40])
    blank[10:25, 8:20] = 22
    assert has_face_detail(blank, [0, 0, 30, 40])


def test_tiled_detector_maps_coordinates_and_merges_overlap():
    from types import SimpleNamespace
    import numpy as np
    from video_prompter.small_faces import detect_frame
    calls = []
    class Detector:
        def predict(self, image, **kwargs):
            calls.append(image.shape)
            detections = [
                [[630, 100, 650, 130]],
                [[630, 100, 650, 130]],
                [[54, 100, 74, 130], [424, 100, 444, 130], [0, 10, 15, 30]],
            ][len(calls) - 1]
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=np.array(detections),
                                                          conf=np.full(len(detections), .8)))]
    frame = np.random.default_rng(0).integers(0, 255, size=(736, 1344, 3), dtype=np.uint8)
    result = detect_frame(Detector(), frame, .25)
    assert calls == [(736, 1344, 3), (736, 768, 3), (736, 768, 3)]
    assert result == [[630, 100, 650, 130], [1000, 100, 1020, 130]]


def test_learned_upscale_uses_source_crop_size_and_preserves_detailed_crops(monkeypatch):
    import sys
    import pytest
    from types import ModuleType, SimpleNamespace
    torch = pytest.importorskip("torch")
    from video_prompter.small_faces import learned_face_crops
    calls, unloaded = [], []
    class Upscale:
        @staticmethod
        def execute(model, frames):
            calls.append(tuple(frames.shape))
            return (torch.nn.functional.interpolate(frames.movedim(-1, 1), scale_factor=4,
                     mode="nearest").movedim(1, -1),)
    mm = SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None,
        unload_model_and_clones=lambda model, **kwargs: unloaded.append(model))
    comfy = ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setitem(sys.modules, "comfy_extras", ModuleType("comfy_extras"))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_upscale_model", SimpleNamespace(ImageUpscaleWithModel=Upscale))
    model = SimpleNamespace(scale=4, patcher=object())
    crops = torch.full((5, 768, 768, 3), .5)
    result, used = learned_face_crops(crops, {"boxes": [(0, 0, 78, 78)] * 5}, model)
    assert used and calls == [(4, 80, 80, 3), (1, 80, 80, 3)]
    assert result.shape == crops.shape and torch.allclose(result, crops)
    assert unloaded == [model.patcher]
    result, used = learned_face_crops(crops, {"boxes": [(0, 0, 400, 400)] * 5}, model)
    assert not used and result is crops  # don't discard existing source detail
