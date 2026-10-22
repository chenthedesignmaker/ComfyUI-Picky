"""Whole-scene restoration and optional source-preserving face compositing."""
from __future__ import annotations
import math


def output_canvas(width, height, resolution):
    if width < 1 or height < 1:
        raise ValueError("Enhancement requires positive source dimensions.")
    if resolution not in ("original", "1080p"):
        raise ValueError("Enhancement resolution must be original or 1080p.")
    if resolution == "original":
        return width, height
    # Match the tested short-edge target and long-edge cap, never downsize.
    scale = max(1., min(1080 / min(width, height), 2048 / max(width, height)))
    return max(width, round(width * scale / 2) * 2), max(height, round(height * scale / 2) * 2)


def build_workflow(source, width, height, spec, seed, overrides=None):
    if spec.reference_face:
        raise ValueError("SeedVR2 enhancement does not use a reference identity.")
    if spec.enhancement_mode == "seedvr2_faces" and spec.enhancement_resolution != "original":
        raise ValueError("Face Blend currently keeps the original resolution.")
    tw, th = output_canvas(width, height, spec.enhancement_resolution)
    models = overrides or {}
    wf = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": source}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
            "model": models.get("seedvr2_dit", "seedvr2_ema_3b_fp8_e4m3fn.safetensors"),
            "device": "cuda:0", "blocks_to_swap": 16, "swap_io_components": True,
            "offload_device": "cpu", "cache_model": False, "attention_mode": "sdpa"}},
        "4": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model": models.get("seedvr2_vae", "ema_vae_fp16.safetensors"), "device": "cuda:0",
            "encode_tiled": True, "encode_tile_size": 512, "encode_tile_overlap": 64,
            "decode_tiled": True, "decode_tile_size": 512, "decode_tile_overlap": 64,
            "tile_debug": "false", "offload_device": "cpu", "cache_model": False}},
        "5": {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image": ["2", 0], "dit": ["3", 0], "vae": ["4", 0], "seed": seed,
            "resolution": max(1080, min(width, height)),
            # 4n+1 context, 16-frame stride, and a fractional overlap crossfade.
            # The old 5/2 preset switched batches every three frames; overlap 2
            # has weights [1, 0] in SeedVR2 and cannot smooth those switches.
            "max_resolution": max(2048, max(width, height)), "batch_size": 21,
            "uniform_batch_size": True, "color_correction": "lab", "temporal_overlap": 5,
            "prepend_frames": 4, "input_noise_scale": 0., "latent_noise_scale": 0.,
            "offload_device": "cpu", "enable_debug": False}},
        "6": {"class_type": "ImageScale", "inputs": {
            "image": ["5", 0], "upscale_method": "lanczos", "width": tw, "height": th,
            "crop": "disabled"}},
    }
    image_id = "6"
    if spec.enhancement_mode == "seedvr2_faces":
        wf["7"] = {"class_type": "VideoPrompterSeedVR2FaceBlend", "inputs": {
            "original": ["2", 0], "enhanced": ["6", 0], "fps": ["2", 2],
            "confidence": .25, "min_face_size": 8, "max_face_size": 192}}
        image_id = "7"
    wf["8"] = {"class_type": "VHS_VideoCombine", "inputs": {
        "images": [image_id, 0], "audio": ["2", 1], "frame_rate": ["2", 2],
        "save_output": True, "filename_prefix": "VideoPrompter_SeedVR2_" + (
            "FaceBlend" if image_id == "7" else spec.enhancement_resolution),
        "loop_count": 0, "pix_fmt": "yuv420p", "save_metadata": True, "crf": 16,
        "trim_to_audio": False, "format": "video/h264-mp4", "no_preview": False,
        "pingpong": False}}
    return wf


def smoothed_boxes(detections, cuts=(), min_size=8, max_size=192):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from .small_faces import plan_tracks
    tracks, total = plan_tracks(detections, cuts, min_size=min_size, max_size=max_size)
    boxes = [[] for _ in detections]
    for track in tracks:
        runs = []
        for i in sorted(k for k, b in track.boxes.items() if min_size <= b[3] - b[1] <= max_size):
            if not runs or i != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(i)
        for run in runs:
            smoothed = gaussian_filter1d(np.array([track.boxes[i] for i in run]), 1., axis=0, mode="nearest")
            for i, box in zip(run, smoothed):
                boxes[i].append(box)
    return boxes, len(tracks), total


def face_mask(boxes, width, height):
    """The tight, supersampled feathered mask from the reviewed experiment."""
    import cv2
    import numpy as np
    out = np.zeros((height, width), np.float32)
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        ax, ay = (x2-x1)*.65, (y2-y1)*.65
        if ax <= 0 or ay <= 0:
            continue
        left, right = max(0, math.floor(cx-ax-1)), min(width, math.ceil(cx+ax+1))
        top, bottom = max(0, math.floor(cy-ay-1)), min(height, math.ceil(cy+ay+1))
        if left >= right or top >= bottom:
            continue
        yy, xx = np.mgrid[top:bottom:.25, left:right:.25]
        r = np.sqrt(((xx+.125-cx)/ax)**2 + ((yy+.125-cy)/ay)**2)
        a = np.clip((1-r)/.35, 0, 1)
        a = a*a*(3-2*a)
        a = cv2.resize(a.astype(np.float32), (right-left, bottom-top), interpolation=cv2.INTER_AREA)
        out[top:bottom, left:right] = np.maximum(out[top:bottom, left:right], a)
    return out


class SeedVR2FaceBlend:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "Video Prompter"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "enhanced": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24., "min": 1.}),
            "confidence": ("FLOAT", {"default": .25, "min": .01, "max": 1.}),
            "min_face_size": ("INT", {"default": 8, "min": 1}),
            "max_face_size": ("INT", {"default": 192, "min": 1})}}

    def blend(self, original, enhanced, fps=24., confidence=.25, min_face_size=8, max_face_size=192):
        import torch
        import folder_paths
        import comfy.model_management as mm
        from ultralytics import YOLO
        from types import SimpleNamespace
        from pathlib import Path
        from .small_faces import detect_faces
        if original.shape != enhanced.shape or original.ndim != 4 or original.shape[-1] != 3 or not len(original):
            raise ValueError("Face Blend requires matching RGB frames and source dimensions.")
        if min_face_size > max_face_size:
            raise ValueError("Minimum face size must not exceed maximum face size.")
        path = Path(folder_paths.models_dir) / "ultralytics" / "bbox" / "face_yolov8m.pt"
        if not path.is_file():
            raise FileNotFoundError("Face Blend needs models/ultralytics/bbox/face_yolov8m.pt.")
        detector = YOLO(str(path))
        try:
            original = original.detach().cpu()
            enhanced = enhanced.detach().cpu()
            bridge = SimpleNamespace(_load_detector=lambda _: detector,
                _to_bgr_u8=lambda x: (x.numpy()[..., ::-1]*255).round().clip(0, 255).astype("uint8"))
            detections, cuts = detect_faces(original, bridge, "face_yolov8m.pt", confidence, fps=fps)
            boxes, eligible, total = smoothed_boxes(detections, cuts, min_face_size, max_face_size)
            result = original.clone()
            height, width = original.shape[1:3]
            for i, frame_boxes in enumerate(boxes):
                mm.throw_exception_if_processing_interrupted()
                mask = torch.from_numpy(face_mask(frame_boxes, width, height)).to(original.dtype).unsqueeze(-1)
                result[i] = original[i]*(1-mask) + enhanced[i]*mask
            report = (f"SeedVR2 Face Blend: {eligible}/{total} detected tracks; "
                      f"{sum(map(len, boxes))} detected face observations blended. "
                      "Tight soft masks; original pixels preserved outside masks before encoding. "
                      "Missing detections are not filled; undetected faces are not counted.")
            return {"ui": {"face_refine_report": [report]}, "result": (result,)}
        finally:
            detector.to("cpu")
