"""Prepare a camera-pass reference for native H3 Ref2VA; no sampler patches."""
import math


def frame_indices(count, source_fps):
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Motion Frame needs a finite positive source FPS.")
    duration = count / source_fps
    if count < 2 or not 2 <= duration <= 15:
        raise ValueError("Motion Frame needs a reference clip between 2 and 15 seconds. Trim the clip before uploading.")
    target = round(duration * 24)
    aligned = 5 + 17 * ((target - 5) // 17)
    return [min(count - 1, int(i * source_fps / 24)) for i in range(aligned)]


class CameraFrames:
    CATEGORY = "Picky"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reference_frames",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "source_fps": ("FLOAT", {"default": 24.0, "min": 0.01})}}

    def prepare(self, images, source_fps):
        import torch.nn.functional as F
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("Motion Frame expects an RGB video frame sequence.")
        frames = images[frame_indices(len(images), source_fps)]
        # Bound reference VAE cost on a local GPU. Never upscale a source.
        h, w = frames.shape[1:3]
        scale = min(1.0, math.sqrt(262144 / (h * w)))
        th, tw = max(32, int(h * scale) // 32 * 32), max(32, int(w * scale) // 32 * 32)
        if (th, tw) != (h, w):
            frames = F.interpolate(frames.movedim(-1, 1), size=(th, tw), mode="bilinear", align_corners=False).movedim(1, -1)
        return (frames,)
