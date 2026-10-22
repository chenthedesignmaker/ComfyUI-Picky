"""Multi-face planning and an adapter to Easy H3's temporal crop sampler.

Detection is shared by all tracks. Crops always come from the original video;
only the feathered composites accumulate. No global upstream monkey patches.
Heavy ComfyUI dependencies are imported only during execution.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field, replace
from types import SimpleNamespace


@dataclass
class FaceTrack:
    boxes: dict[int, list[float]] = field(default_factory=dict)

    def predict(self, frame):
        frames = list(self.boxes)
        last = frames[-1]
        box = self.boxes[last]
        if len(frames) < 2:
            return box
        prev = frames[-2]
        dt = min(frame - last, 3) / (last - prev)
        predicted = [v + (v - p) * dt for v, p in zip(box, self.boxes[prev])]
        # A turning/shrinking head must not extrapolate to an inverted box.
        cx, cy = (predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2
        w = max((box[2] - box[0]) * 0.5, predicted[2] - predicted[0])
        h = max((box[3] - box[1]) * 0.5, predicted[3] - predicted[1])
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _distance(a, b):
    ah, bh = a[3] - a[1], b[3] - b[1]
    aw, bw = a[2] - a[0], b[2] - b[0]
    ratio = max(ah / bh, bh / ah, aw / bw, bw / aw)
    if ratio > 1.8:
        return math.inf
    distance = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2,
                          (a[1] + a[3] - b[1] - b[3]) / 2) / max(ah, bh)
    return distance + abs(math.log(ah / bh)) if distance < 0.75 else math.inf


def plan_tracks(detections, cuts=(), *, min_size=8, max_size=192, min_frames=3, max_gap=3):
    """One-to-one motion association, reset at cuts; ambiguous matches are omitted.

    Track all face sizes so a size change cannot switch the selected person.
    Filter for repairable small-face observations only after association.
    """
    tracks, active = [], []
    cuts = set(cuts)
    for frame, candidates in enumerate(detections):
        boxes = [list(b) for b in candidates if len(b) == 4
                 and all(math.isfinite(v) for v in b) and b[2] > b[0] and b[3] > b[1]]
        active = [] if frame in cuts else [t for t in active if frame - max(t.boxes) <= max_gap + 1]
        costs = [[_distance(t.predict(frame), b) for b in boxes] for t in active]
        used = set()
        for i, track in enumerate(active):
            ranked = sorted((c, j) for j, c in enumerate(costs[i]) if math.isfinite(c))
            if not ranked:
                continue
            cost, j = ranked[0]
            rivals = sorted((row[j], k) for k, row in enumerate(costs) if math.isfinite(row[j]))
            # Mutual best plus a margin: don't swap people at a crossing.
            if rivals[0][1] != i or (len(ranked) > 1 and ranked[1][0] - cost < 0.15):
                continue
            if len(rivals) > 1 and rivals[1][0] - cost < 0.15:
                continue
            track.boxes[frame] = boxes[j]
            used.add(j)
        for j, box in enumerate(boxes):
            # A contested detection is not a new person. Wait for a clear match.
            if j in used or any(math.isfinite(row[j]) for row in costs):
                continue
            track = FaceTrack({frame: box})
            tracks.append(track)
            active.append(track)
    eligible = [t for t in tracks if sum(min_size <= b[3] - b[1] <= max_size
                                         for b in t.boxes.values()) >= min_frames]
    return eligible, len(tracks)


def _tile_starts(length, size=768):
    if length <= size:
        return [0]
    return sorted(set([*range(0, length - size + 1, size * 3 // 4), length - size]))


def merge_detections(candidates):
    """Suppress duplicate full-frame/tile boxes, keeping the most confident one."""
    kept = []
    for candidate in sorted(candidates, key=lambda c: c[4], reverse=True):
        a = candidate[:4]
        area = (a[2] - a[0]) * (a[3] - a[1])
        if a[2] <= a[0] or a[3] <= a[1] or not all(math.isfinite(v) for v in candidate):
            continue
        duplicate = False
        for b in kept:
            other_area = (b[2] - b[0]) * (b[3] - b[1])
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if overlap / (area + other_area - overlap) > 0.45 or (
                overlap / min(area, other_area) > 0.8 and max(area, other_area) / min(area, other_area) < 4
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(a)
    return kept


def has_face_detail(bgr, box):
    """Reject featureless detector hits, e.g. a completely black back of a head.

    Test contrast, not skin colour or average brightness. A uniform patch has
    no facial evidence to repair, even if the detector assigns it a box.
    """
    import numpy as np
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * 0.2, (y2 - y1) * 0.2
    centre = bgr[max(0, int(y1 + dy)):int(y2 - dy), max(0, int(x1 + dx)):int(x2 - dx)]
    if not centre.size:
        return False
    lo, hi = np.percentile(centre.astype(float).mean(axis=-1), (5, 95))
    return hi - lo >= 4.0


def detect_frame(model, bgr, confidence, detection_size=1280, tiled=True):
    """Enlarge overlapping tiles for detection, returning original-pixel coordinates."""
    height, width = bgr.shape[:2]
    regions = [(0, 0, width, height)]
    if tiled and max(width, height) > 768:
        regions += [(x, y, min(x + 768, width), min(y + 768, height))
                    for y in _tile_starts(height) for x in _tile_starts(width)]
    candidates = []
    for x, y, right, bottom in regions:
        prediction = model.predict(bgr[y:bottom, x:right], conf=confidence, imgsz=detection_size,
                                   device="cpu", verbose=False)[0]
        if prediction.boxes is None:
            continue
        for box, score in zip(prediction.boxes.xyxy.tolist(), prediction.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            # Don't introduce clipped half-faces at internal tile boundaries.
            if ((x and x1 < 3) or (y and y1 < 3) or
                (right < width and x2 > right - x - 3) or
                (bottom < height and y2 > bottom - y - 3)):
                continue
            candidates.append([x1 + x, y1 + y, x2 + x, y2 + y, float(score)])
    return [box for box in merge_detections(candidates) if has_face_detail(bgr, box)]


def detect_faces(images, tracker, detector, confidence, detection_size=1280, tiled=True, fps=24.):
    """Keep source-pixel boxes and scene boundaries; never detect on refined pixels."""
    import comfy.model_management as mm
    from comfy.utils import ProgressBar
    from scenedetect import FrameTimecode
    from scenedetect.detectors import AdaptiveDetector

    model = tracker._load_detector(detector)
    cut_detector = AdaptiveDetector(adaptive_threshold=3.0)
    boxes, cuts = [], []
    progress = ProgressBar(len(images))
    for i, frame in enumerate(images):
        mm.throw_exception_if_processing_interrupted()
        bgr = tracker._to_bgr_u8(frame)
        cuts.extend(int(t.get_frames()) for t in cut_detector.process_frame(FrameTimecode(i, fps=float(fps)), bgr))
        boxes.append(detect_frame(model, bgr, confidence, detection_size, tiled))
        progress.update(1)
    return boxes, cuts


def track_pick(track, first, last, width, height):
    """Feed upstream only this track's boxes, so its selector cannot choose a neighbour."""
    boxes = [[track.boxes[i]] if i in track.boxes else [] for i in range(first, last)]
    return {"frames": last - first, "src_size": (width, height), "boxes": boxes,
            "confs": [[1.0] if b else [] for b in boxes], "segments": [(0, last - first)],
            "picks": [{"frame": 0, "box": 0, "segment": (0, last - first)}],
            "detector": "shared small-face detection"}


def restrict_transform(transform, track, first, min_size, max_size):
    """Never paste into missing/ambiguous detections, or onto a close-up."""
    source = transform.get("source", range(len(transform["weights"])))
    for k, local in enumerate(source):
        box = track.boxes.get(first + int(local))
        if box is None or not min_size <= box[3] - box[1] <= max_size:
            transform["weights"][k] = 0.0


def scale_transform(transform, width, height):
    """Paste high-resolution crops directly into the delivery canvas.

    face_rect and canvas stay in crop pixels; only source-space geometry scales.
    """
    old_w, old_h = transform["src_size"]
    sx, sy = width / old_w, height / old_h
    transform["boxes"] = [(x * sx, y * sy, w * sx, h * sy) for x, y, w, h in transform["boxes"]]
    transform["src_size"] = (width, height)


def delivery_frames(images, width, height):
    """Bound resize temporaries while retaining all repaired detail at delivery size."""
    import torch
    import torch.nn.functional as F
    if tuple(images.shape[1:3]) == (height, width):
        return images.clone()
    result = torch.empty((len(images), height, width, 3), dtype=images.dtype, device="cpu")
    for start in range(0, len(images), 4):
        result[start:start + 4] = F.interpolate(images[start:start + 4].movedim(-1, 1),
            size=(height, width), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0, 1)
    return result


def learned_face_crops(crops, transform, upscale_model):
    """Restore tiny crops near source resolution before temporal H3 sampling.

    Don't run a 4x model on already enlarged 768px inputs, or downsample a
    detailed source face just to fit it through the upscaler.
    """
    import torch
    import torch.nn.functional as F
    import comfy.model_management as mm
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

    native_side = max(max(b[2], b[3]) for b in transform["boxes"])
    input_side = max(16, math.ceil(native_side / 8) * 8)
    if input_side * upscale_model.scale > min(crops.shape[1:3]):
        return crops, False
    result = torch.empty_like(crops)
    try:
        for first in range(0, len(crops), 4):
            mm.throw_exception_if_processing_interrupted()
            low = F.interpolate(crops[first:first + 4].movedim(-1, 1),
                size=(input_side, input_side), mode="bicubic", align_corners=False, antialias=True)
            restored = ImageUpscaleWithModel.execute(upscale_model, low.movedim(1, -1).clamp(0, 1))[0].cpu()
            result[first:first + 4] = F.interpolate(restored.movedim(-1, 1), size=crops.shape[1:3],
                mode="bicubic", align_corners=False).movedim(1, -1).clamp(0, 1)
        return result, True
    finally:
        mm.unload_model_and_clones(upscale_model.patcher, unload_additional_models=False)


class SmallFaceRefine:
    CATEGORY = "Video Prompter"
    FUNCTION = "refine"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "model": ("MODEL",), "h3_context": ("MINIMAX_H3_CONTEXT",),
            "detector": ("STRING", {"default": "face_yolov8m.pt"}),
            "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "canvas_size": (["512", "768"], {"default": "768"}),
            "confidence": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.95}),
            "min_face_size": ("INT", {"default": 8, "min": 8, "max": 128}),
            "max_face_size": ("INT", {"default": 192, "min": 16, "max": 512}),
            "prompt": ("STRING", {"multiline": True}),
        }, "optional": {"audio": ("AUDIO",),
            "output_width": ("INT", {"default": 0, "min": 0, "max": 8192}),
            "output_height": ("INT", {"default": 0, "min": 0, "max": 8192}),
            "upscale_model": ("UPSCALE_MODEL",),
            "tiled_detection": ("BOOLEAN", {"default": True})}}

    def refine(self, images, model, h3_context, detector="face_yolov8m.pt", denoise=0.25,
               steps=8, seed=42, canvas_size="768", confidence=0.25, min_face_size=8,
               max_face_size=192, prompt="", audio=None, output_width=0, output_height=0,
               tiled_detection=True, upscale_model=None):
        import nodes
        import torch
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        if images.ndim != 4 or not len(images) or images.shape[-1] != 3:
            raise ValueError("Small-face refinement needs a nonempty RGB video.")
        fps = float(h3_context.fps)
        if abs(fps - 24) > 0.01:
            raise ValueError("Small-face refinement supports native 24 FPS video; refine before interpolation.")
        if min_face_size > max_face_size:
            raise ValueError("Minimum face size must not exceed maximum face size.")
        if bool(output_width) != bool(output_height):
            raise ValueError("Set both output dimensions, or leave both zero for native resolution.")
        width, height = int(output_width or images.shape[2]), int(output_height or images.shape[1])
        if width < images.shape[2] or height < images.shape[1]:
            raise ValueError("Small-face output must be native size or larger.")
        if denoise <= 0:
            result = images if (width, height) == (images.shape[2], images.shape[1]) else delivery_frames(images.cpu(), width, height)
            return self._result(result, audio, fps, "Small-face refinement skipped: strength is zero.")
        backend = nodes.NODE_CLASS_MAPPINGS.get("FeiHouEasyH3RHFaceRefine")
        if backend is None:
            raise RuntimeError("Small-face refinement requires the Easy H3 Face Refine node.")
        module = importlib.import_module(backend.__module__)
        tracker = importlib.import_module(module.__package__ + ".face_refine_tracking")
        # Reuse the vendor's lock and sampler, without modifying its module or files.
        with module._LOCK, torch.inference_mode():
            try:
                original = images.cpu()
                detections, cuts = detect_faces(original, tracker, detector, confidence, tiled=tiled_detection)
                tracks, total = plan_tracks(detections, cuts, min_size=min_face_size, max_size=max_face_size)
                candidates = sum(min_face_size <= b[3] - b[1] <= max_face_size for frame in detections for b in frame)
                summary = (f"Small faces: {len(tracks)} eligible tracks of {total} detected tracks; "
                           f"{min_face_size}–{max_face_size}px source face height; {len(cuts)} cuts. "
                           f"{'Tiled + full-frame' if tiled_detection else 'Full-frame'} detection; output {width}×{height}. ")
                if not tracks:
                    result = images if (width, height) == (images.shape[2], images.shape[1]) else delivery_frames(original, width, height)
                    return self._result(result, audio, fps, summary + f"No eligible small faces; 0/{candidates} detected small-face observations repaired.")
                print("[VideoPrompter] " + summary, flush=True)
                # Never apply one cast/reference image to every person in a crowd.
                context = replace(h3_context, reference_image_1=None)
                result = delivery_frames(original, width, height)
                progress = ProgressBar(len(tracks))
                processed = 0
                applied_observations = 0
                restored_tracks = 0
                for index, track in enumerate(tracks):
                    mm.throw_exception_if_processing_interrupted()
                    first, last = min(track.boxes), max(track.boxes) + 1
                    source = original[first:last]
                    pick = track_pick(track, first, last, original.shape[2], original.shape[1])
                    applied = []
                    pre_upscaled = []

                    class SelectedCrop:
                        def run(self, **kwargs):
                            kwargs.update(images=source, face_pick=pick, identity_track=False,
                                          identity_reference=None, smooth_window=7, size_smooth_window=15)
                            crops, transform, preview, report, cw, ch, count = tracker.H3FaceTrackCrop().run(**kwargs)
                            restrict_transform(transform, track, first, min_face_size, max_face_size)
                            applied.append(sum(w > 1e-4 for w in transform["weights"]))
                            if upscale_model is not None:
                                crops, used = learned_face_crops(crops, transform, upscale_model)
                                pre_upscaled.append(used)
                            scale_transform(transform, width, height)
                            return crops, transform, preview, report, cw, ch, count

                    proxy = SimpleNamespace(**vars(tracker))
                    proxy.H3FaceTrackCrop = SelectedCrop

                    class SelectedRefine(backend):
                        def _run(self, tr, *args, **kwargs):
                            return super()._run(proxy, *args, **kwargs)

                    print(f"[VideoPrompter] Small face {index + 1}/{len(tracks)}, frames {first}:{last}", flush=True)
                    refined, _, _, _ = SelectedRefine().refine(
                        images=result[first:last], model=model, h3_context=context, detector=detector,
                        target="largest_face", denoise=denoise, steps=steps, seed=(int(seed) + index) % 2**64,
                        canvas_size=canvas_size, confidence=confidence, crop_factor=2.5,
                        chunk_frames="120", prompt=prompt, reference_face=None, audio=None,
                        # Size filtering above already protects close-ups. A second
                        # magnification gate could silently bypass eligible faces
                        # when the caller selects the smaller 512px canvas.
                        force_offload=True, clean_second_model=True, adaptive_face_scale=False)
                    result[first:last] = refined
                    if sum(applied):
                        processed += 1
                    applied_observations += sum(applied)
                    restored_tracks += int(any(pre_upscaled))
                    progress.update(1)
                return self._result(result, audio, fps, summary + f"Processed {processed} face tracks. "
                                    f"Learned upscaling before H3: {restored_tracks} tiny-face tracks. "
                                    f"Repaired {applied_observations}/{candidates} detected small-face observations "
                                    "(counted across frames, not unique people; undetected faces are not counted). "
                                    "Missing detections, faces outside the size range and original audio preserved.")
            finally:
                for det in tracker._DETECTOR_CACHE.values():
                    det.to("cpu")
                tracker._DETECTOR_CACHE.clear()
                tracker._REC_CACHE.clear()

    @staticmethod
    def _result(images, audio, fps, report):
        print("[VideoPrompter] " + report, flush=True)
        return {"ui": {"face_refine_report": [report]}, "result": (images, audio, fps, report)}
