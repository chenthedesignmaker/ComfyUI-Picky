"""ReDetail 2.0 integration for Video Prompter.

Provides generative video re-detailing and upscaling using Lightricks LTX-2.5 22B
and the Refine-Details / Pixel-Spatial IC-LoRAs.

Can be used in two ways:
1. As an enhancement mode inside ``video_prompter.refine``:
   ``spec.enhancement_mode = "redetail"`` or ``"redetail_pixel"``
2. As a standalone runner or CLI:
   ``python -m video_prompter.redetail input.mp4 --scale 1.5``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Sequence

# Node ids in the shipped API workflows
N_FRAME = "2004"
N_VIDEO = "5001"
N_SAVE = "4852"
N_REF_RESIZE = "5548:5026"
N_CANVAS = "9002:3059"
N_GUIDE_RESIZE = "9002:9201"
N_GUIDE = "9002:5012"
N_SAMPLER = "5516:9100"
N_NOISE_PIXEL = "5516:4832"

GRID = {"pixel": 64, "refine": 32}
FMP_BUDGET = {"pixel": 850.0, "refine": 500.0}
SAVE_CRF = 12
MIN_SEC = 2.0

WORKFLOWS_DIR = Path(__file__).parent / "workflows"
WF_REFINE = WORKFLOWS_DIR / "ltx25_refine_API.json"
WF_PIXEL = WORKFLOWS_DIR / "ltx25_upscale_API.json"


# -----------------------------------------------------------------------------
# Geometry calculations
# -----------------------------------------------------------------------------


def solve_dims(w: int, h: int, scale: float, grid: int = 32) -> tuple[int, int, float, float]:
    """Find the nearest on-grid target to the requested scale that maintains aspect ratio.

    Returns:
        (target_w, target_h, aspect_error, scale_error)
    """
    if w <= 0 or h <= 0:
        raise ValueError("Source dimensions must be positive.")
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    ar = w / h
    best = None
    lo = max(grid, int(scale * h * 0.85) // grid * grid)
    hi = int(scale * h * 1.15) // grid * grid + grid + 1
    for th in range(lo, hi, grid):
        tw = max(grid, round(th * ar / grid) * grid)
        aerr = abs((tw / th) / ar - 1)
        if aerr > 0.02:
            continue
        serr = ((tw / w) + (th / h)) / 2 / scale - 1
        score = aerr * 8 + abs(serr) * (1.3 if serr > 0 else 1.0)
        if best is None or score < best[0]:
            best = (score, tw, th, aerr, serr)

    if best is None:
        # Fallback to direct grid snapping
        tw = max(grid, round(w * scale / grid) * grid)
        th = max(grid, round(h * scale / grid) * grid)
        return tw, th, abs((tw / th) / ar - 1), ((tw / w) + (th / h)) / 2 / scale - 1

    return best[1], best[2], best[3], best[4]


def fit_source(w: int, h: int, max_crop: float = 0.03) -> tuple[str, int, int]:
    """Get the source onto the /64 grid for pixel mode."""
    cw, ch = w // 64 * 64, h // 64 * 64
    if (cw, ch) == (w, h):
        return "none", w, h
    if max(1 - cw / w, 1 - ch / h) <= max_crop:
        return "crop", cw, ch
    ar, best = w / h, None
    for b in range(1, 65):
        fh = b * 64
        fw = max(64, round(fh * ar / 64) * 64)
        if abs((fw / fh) / ar - 1) > 0.005:
            continue
        if not 0.65 <= fh / h <= 2.0:
            continue
        d = abs(fh / h - 1)
        if best is None or d < best[0]:
            best = (d, fw, fh)
    return ("resize", best[1], best[2]) if best else ("crop", cw, ch)


def target_for(
    orig_w: int, orig_h: int, fit_w: int, fit_h: int, scale: float, grid: int = 32
) -> tuple[int, int, float, float]:
    """Calculate the target output canvas dimensions."""
    eff = (orig_h * scale) / fit_h
    return solve_dims(fit_w, fit_h, eff, grid)


def output_canvas(
    source_width: int, source_height: int, scale_or_res: float | str, mode: str = "refine"
) -> tuple[int, int]:
    """Compute output canvas size for Video Prompter API integration.

    Supports float scales (e.g. 1.0, 1.5, 2.0) and named presets ("original", "1080p", "4k").
    """
    grid = GRID.get(mode, 32)
    if isinstance(scale_or_res, str):
        if scale_or_res == "original":
            scale = 1.0
        elif scale_or_res == "1080p":
            scale = max(1.0, 1080.0 / min(source_width, source_height))
        elif scale_or_res == "4k":
            scale = max(1.0, 2160.0 / min(source_width, source_height))
        else:
            try:
                scale = float(scale_or_res)
            except ValueError:
                raise ValueError(f"Unknown resolution/scale preset: {scale_or_res}")
    else:
        scale = float(scale_or_res)

    tw, th, _, _ = solve_dims(source_width, source_height, scale, grid=grid)
    return tw, th


# -----------------------------------------------------------------------------
# Workflow Builder
# -----------------------------------------------------------------------------


def build_workflow(
    source_video: str,
    source_width: int,
    source_height: int,
    spec: Any = None,
    seed: int = 42,
    model_overrides: dict[str, str] | None = None,
    scale: float | None = None,
    mode: str | None = None,
    guide_strength: float = 1.0,
    filename_prefix: str = "VideoPrompter_ReDetail",
) -> dict[str, Any]:
    """Construct a ComfyUI workflow dictionary using ReDetail LTX-2.5 graphs."""
    eff_mode = mode or "refine"
    eff_scale = scale or 1.5
    eff_strength = guide_strength

    if spec is not None:
        if hasattr(spec, "enhancement_mode"):
            if "pixel" in spec.enhancement_mode:
                eff_mode = "pixel"
            else:
                eff_mode = "refine"
        if hasattr(spec, "redetail_scale"):
            eff_scale = spec.redetail_scale
        if hasattr(spec, "guide_strength"):
            eff_strength = spec.guide_strength

    grid = GRID[eff_mode]
    wf_path = WF_PIXEL if eff_mode == "pixel" else WF_REFINE
    if not wf_path.exists():
        raise FileNotFoundError(f"Missing workflow template: {wf_path}")

    with open(wf_path, encoding="utf-8") as f:
        wf = json.load(f)

    # API jobs must succeed only when the video branch is valid. Independent
    # PreviewAny outputs let ComfyUI accept a broken video graph and report a
    # successful text-only execution (for example when a LoRA is missing).
    wf = {nid: node for nid, node in wf.items() if node["class_type"] != "PreviewAny"}

    # LTX decodes 8n+1 frames. Pad the guide with its last frame to the next
    # valid length, then trim after decoding so H3's 124-frame clips retain
    # their full duration. Keep the original audio rather than VAE-roundtrip it.
    source_images = ["5548:9006", 0]
    source_count = ["5548:5029", 2]
    wf["5548:5029"]["inputs"]["image"] = source_images
    wf.update({
        "vp_frame_length": {
            "class_type": "ComfyMathExpression",
            "inputs": {"expression": "ceil((a - 1) / 8) * 8 + 1", "values.a": source_count},
        },
        "vp_last_frame": {
            "class_type": "ImageFromBatch",
            "inputs": {"image": source_images, "batch_index": -1, "length": 1},
        },
        "vp_tail_frames": {
            "class_type": "RepeatImageBatch",
            "inputs": {"image": ["vp_last_frame", 0], "amount": 7},
        },
        "vp_extended_frames": {
            "class_type": "ImageBatch",
            "inputs": {"image1": source_images, "image2": ["vp_tail_frames", 0]},
        },
        "vp_padded_frames": {
            "class_type": "ImageFromBatch",
            "inputs": {"image": ["vp_extended_frames", 0], "batch_index": 0,
                       "length": ["vp_frame_length", 1]},
        },
        "vp_trim_frames": {
            "class_type": "ImageFromBatch",
            "inputs": {"image": ["5518:5538", 0], "batch_index": 0, "length": source_count},
        },
    })
    wf[N_CANVAS]["inputs"]["length"] = ["vp_frame_length", 1]
    if eff_mode == "refine":
        wf[N_GUIDE_RESIZE]["inputs"]["image"] = ["vp_padded_frames", 0]
        wf["9002:9200"]["inputs"]["frame_count"] = ["vp_frame_length", 1]
        wf["9002:9200"]["inputs"]["image"] = ["vp_padded_frames", 0]
    else:
        wf[N_REF_RESIZE]["inputs"]["input"] = ["vp_padded_frames", 0]
    wf["5518:4849"]["inputs"]["images"] = ["vp_trim_frames", 0]
    wf["5518:4849"]["inputs"]["audio"] = ["5548:9006", 1]

    tw, th, _, _ = solve_dims(source_width, source_height, eff_scale, grid=grid)

    # 1. Video source input & output filename
    if N_VIDEO in wf:
        wf[N_VIDEO]["inputs"]["file"] = source_video
    if N_SAVE in wf:
        wf[N_SAVE]["inputs"]["filename_prefix"] = filename_prefix

    # 2. Canvas geometry
    if N_CANVAS in wf:
        wf[N_CANVAS]["inputs"]["width"] = tw
        wf[N_CANVAS]["inputs"]["height"] = th

    # 3. Mode-specific adjustments
    if eff_mode == "refine":
        if N_GUIDE_RESIZE in wf:
            wf[N_GUIDE_RESIZE]["inputs"]["width"] = tw
            wf[N_GUIDE_RESIZE]["inputs"]["height"] = th
        if N_SAMPLER in wf:
            wf[N_SAMPLER]["inputs"]["seed"] = seed
            wf[N_SAMPLER]["inputs"]["tile_width"] = 576 if th > tw else 1024
            wf[N_SAMPLER]["inputs"]["tile_height"] = 1024 if th > tw else 576
        if N_GUIDE in wf:
            wf[N_GUIDE]["inputs"]["strength"] = eff_strength
    else:
        # Derive the image anchor from the same video as the guide. The
        # exported template's LoadImage placeholder is not an installed file.
        wf[N_FRAME] = {
            "class_type": "ImageFromBatch",
            "inputs": {"image": ["5548:9006", 0], "batch_index": 0, "length": 1},
            "_meta": {"title": "Source video's first frame"},
        }
        if N_NOISE_PIXEL in wf:
            wf[N_NOISE_PIXEL]["inputs"]["noise_seed"] = seed
        if N_REF_RESIZE in wf:
            wf[N_REF_RESIZE]["inputs"]["resize_type.shorter_size"] = min(source_width, source_height)

    # Apply any model overrides if supplied
    if model_overrides:
        if "transformer" in model_overrides and "5004:5602" in wf:
            wf["5004:5602"]["inputs"]["unet_name"] = model_overrides["transformer"]

    return wf


# -----------------------------------------------------------------------------
# Standalone CLI & Full Video Processing Pipeline
# -----------------------------------------------------------------------------


class ComfyClient:
    """Minimal client for ComfyUI HTTP interaction."""

    def __init__(self, url: str = "http://127.0.0.1:8188"):
        self.url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def upload_file(self, filepath: str) -> str:
        with open(filepath, "rb") as f:
            data = f.read()
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{os.path.basename(filepath)}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
        sub = resp.get("subfolder", "")
        return f"{sub}/{resp['name']}" if sub else resp["name"]

    def submit_prompt(self, workflow: dict[str, Any]) -> str:
        req = urllib.request.Request(
            f"{self.url}/prompt",
            data=json.dumps({"prompt": workflow, "client_id": self.client_id}).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["prompt_id"]

    def wait_for_result(self, prompt_id: str, save_node_id: str = N_SAVE, poll_interval: float = 5.0) -> dict[str, Any] | None:
        while True:
            time.sleep(poll_interval)
            req = urllib.request.Request(f"{self.url}/history/{prompt_id}")
            h = json.loads(urllib.request.urlopen(req, timeout=30).read())
            if prompt_id in h:
                st = h[prompt_id].get("status", {})
                if st.get("status_str") != "success":
                    return None
                outputs = h[prompt_id].get("outputs", {}).get(save_node_id, {})
                for k in ("video", "videos", "gifs", "images"):
                    for item in outputs.get(k) or []:
                        if item.get("type", "output") == "output":
                            return item
                return None

    def download_output(self, filename: str, subfolder: str = "", dest_path: str = "") -> str:
        url = f"{self.url}/view?filename={urllib.parse.quote(filename)}&subfolder={urllib.parse.quote(subfolder)}&type=output"
        urllib.request.urlretrieve(url, dest_path)
        return dest_path


def redetail_video(
    input_path: str,
    output_path: str | None = None,
    scale: float = 1.5,
    model: str = "refine",
    guide_strength: float = 1.0,
    comfy_url: str = "http://127.0.0.1:8188",
) -> str:
    """Run ReDetail 2.0 on a video file through ComfyUI."""
    src = os.path.abspath(input_path)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Input video not found: {src}")

    out = output_path or os.path.splitext(src)[0] + "_redetail.mp4"

    # 1. Probe input
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames", "-of", "json", src],
        capture_output=True, text=True, check=True
    )
    info = json.loads(r.stdout)["streams"][0]
    w, h = int(info["width"]), int(info["height"])

    # 2. Upload file to ComfyUI
    client = ComfyClient(comfy_url)
    remote_name = client.upload_file(src)

    # 3. Build workflow
    wf = build_workflow(
        source_video=remote_name,
        source_width=w,
        source_height=h,
        scale=scale,
        mode=model,
        guide_strength=guide_strength,
        filename_prefix="VideoPrompter_ReDetail",
    )

    # 4. Submit to ComfyUI
    prompt_id = client.submit_prompt(wf)
    print(f"Submitted to ComfyUI. Prompt ID: {prompt_id}")

    # 5. Wait and retrieve
    res = client.wait_for_result(prompt_id)
    if not res:
        raise RuntimeError(f"ComfyUI prompt {prompt_id} failed or returned no output.")

    fn = res.get("filename")
    sub = res.get("subfolder", "")
    client.download_output(fn, sub, out)
    print(f"ReDetail output saved to: {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="ReDetail 2.0 Video Upscaler for Video Prompter")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--scale", type=float, default=1.5, help="Scale factor (default: 1.5)")
    parser.add_argument("--model", choices=["refine", "pixel"], default="refine", help="Model mode")
    parser.add_argument("--strength", type=float, default=1.0, help="Guide strength for refine mode")
    parser.add_argument("--out", default=None, help="Output file path")
    parser.add_argument("--comfy", default="http://127.0.0.1:8188", help="ComfyUI URL")
    args = parser.parse_args()

    redetail_video(
        input_path=args.input,
        output_path=args.out,
        scale=args.scale,
        model=args.model,
        guide_strength=args.strength,
        comfy_url=args.comfy,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
