"""Hard limits and vocabulary for MiniMax H3 (Hailuo 3.0), **local weights**.

Taken from ComfyUI's `comfy_extras/nodes_minimax_h3.py` (Comfy-Org/ComfyUI
#15224, merged 2026-08-03), not from the hosted API docs — the two differ in
ways that matter. Local generation is frame-based on a 17k+5 grid, has no
negative-prompt input, and sizes by width/height rather than a resolution
preset. Everything the linter enforces lives here.
"""

from __future__ import annotations

import math
from typing import Final

MODEL_NAME: Final = "MiniMax-H3"

# ComfyUI release that first carries the local nodes.
MIN_COMFYUI_VERSION: Final = "0.30.0"

NODE_EMPTY_LATENT: Final = "EmptyMiniMaxH3LatentAV"
NODE_IMAGE_TO_VIDEO: Final = "MiniMaxH3ImageToVideo"
NODE_REFERENCE_TO_VIDEO: Final = "MiniMaxH3ReferenceToVideo"
NODE_SIGMA_SHIFT: Final = "MiniMaxH3SigmaShift"

FPS: Final = 24

# Frame counts must satisfy `n % 17 == 5`; the node snaps up to the next one.
FRAME_GRID: Final = 17
FRAME_OFFSET: Final = 5

# What the node accepts...
MIN_FRAMES: Final = 5
MAX_FRAMES: Final = 3600
# ...and what the model was actually trained on. Outside this it still runs,
# but the node's own tooltip calls it untested.
TRAINED_MIN_FRAMES: Final = 124
TRAINED_MAX_FRAMES: Final = 362

# The trained frame range lands exactly on these whole-second budgets.
TRAINED_MIN_DURATION_S: Final = 5
TRAINED_MAX_DURATION_S: Final = 15

# What a prompt is allowed to ask for. The trained floor is a quality cliff,
# not a wall: the node aligns and runs any count from MIN_FRAMES up, so a
# one-second beat generates. Short is a real editorial choice — a two-second
# reaction can beat a padded five — so the floor is the shortest whole second
# and `duration.untrained` warns instead of the linter blocking the render.
MIN_DURATION_S: Final = 1
MAX_DURATION_S: Final = TRAINED_MAX_DURATION_S

# Canvas sizing, mirroring the node's `adapt_canvas`.
CANVAS_MULTIPLE: Final = 32
BASE_SHORT_EDGE: Final = 768
MAX_PIXELS: Final = 768 * 1344
DEFAULT_WIDTH: Final = 1344
DEFAULT_HEIGHT: Final = 768

ASPECT_RATIOS: Final = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")

# Native stereo, generated in the same pass as the picture.
AUDIO_SAMPLE_RATE_HZ: Final = 32_000
AUDIO_LATENT_FPS: Final = 40

# Omni-reference slots on MiniMaxH3ReferenceToVideo. `video_audio` is the
# soundtrack of the same-numbered reference video; `audio` is standalone.
MAX_REFERENCE_IMAGES: Final = 9
MAX_REFERENCE_VIDEOS: Final = 3
MAX_REFERENCE_VIDEO_AUDIOS: Final = 3
MAX_REFERENCE_AUDIO: Final = 3

# Reference videos are aligned to the same grid and need at least a few frames.
MIN_REFERENCE_VIDEO_FRAMES: Final = 5
MAX_REFERENCE_VIDEO_S: Final = 15

# Full-reference (ref2v) controlled vocabulary, verbatim from the official
# VIDEO_PROMPT_WRITING_GUIDE_ref_en. The summary's task-type prefix draws from
# REF_TASK_TYPES (combined with " + "); retention_analysis uses one marker per
# label — the visual set for pictures/videos/subjects, the audio set for audio.
REF_TASK_TYPES: Final = (
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
)
VISUAL_RETENTION: Final = (
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
)
AUDIO_RETENTION: Final = ("fully_copy", "partially_copy", "reference", "weak_reference")
DEFAULT_VISUAL_RETENTION: Final = "fully_preserved"
DEFAULT_AUDIO_RETENTION: Final = "reference"

# --- Rewrite-format vocabulary, from VIDEO_PROMPT_WRITING_GUIDE_base_en -----

# Camera motion is written as natural English inside the shot: motion type
# first, then amplitude and speed only when they carry meaning (medium
# amplitude and normal speed are left unsaid).
CAMERA_MOTIONS: Final = (
    "Zoom In",
    "Zoom Out",
    "Push In",
    "Pull Out",
    "Pan Left",
    "Pan Right",
    "Truck Left",
    "Truck Right",
    "Tilt Up",
    "Tilt Down",
    "Pedestal Up",
    "Pedestal Down",
    "Arc Shot",
    "Tracking Shot",
    "Static Shot",
    "Shake Slightly",
    "Shake Strongly",
    "POV",
    "Roll Clockwise",
    "Roll Counterclockwise",
)
CAMERA_AMPLITUDES: Final = ("with small amplitude", "with large amplitude")
CAMERA_SPEEDS: Final = ("at slow speed", "at fast speed")

# The cut phrases a later shot may open with.
TRANSITIONS: Final = (
    "the camera cuts to",
    "the shot cuts to",
    "the shot transitions to",
    "the shot changes to",
    "the shot switches to",
)
DEFAULT_TRANSITION: Final = "the camera cuts to"

# Voiceover is a fixed phrase, and every voiceover <d> block is followed by a
# note that the matching on-screen character's lips stay shut.
VOICEOVER_VERB: Final = "says in an off-screen voiceover"
DEFAULT_POSSESSIVE: Final = "their"

# Dialogue spanning a cut is tagged at the seam in *both* parts. The tags sit
# outside <d>, because only the language tag and the verbatim words may go
# inside it.
SCENE_TRANSITION_TAG: Final = "<scenetrans>"
CUTOFF_TAG: Final = "<cutoff>"
CONTINUITY_PHRASES: Final = {
    "into": "continues uninterrupted into the next shot",
    "from": "carries over from the previous shot",
}
CONTINUITY_ALTERNATIVES: Final = (
    "continues seamlessly across the cut",
    "continues uninterrupted into the next shot",
    "carries over from the previous shot",
    "remains audible across the transition",
)
CUTOFF_PHRASE: Final = "is cut off by the end of the video"

# There is no separate prompt ceiling on the local path — the text encoder's
# context is the real limit — but long prompts still lose their tail, so the
# hosted API's guidance is kept as a soft target.
MAX_PROMPT_CHARS: Final = 7_000
RECOMMENDED_PROMPT_CHARS: Final = 5_000

# Pacing heuristics, not model limits.
MIN_SHOT_S: Final = 0.5
COMFORTABLE_MIN_SHOT_S: Final = 1.0
MAX_SHOTS_PER_SECOND: Final = 1.0

# A no-cut beat is a moment inside a shot, not a shot, so the flash-frame floor
# above does not apply to it — nothing flashes when nothing cuts. What does
# apply is that a camera move needs time to travel: measured 2026-08-31, an
# event directed inside a continuous take lands roughly where it was asked for
# but the action still has to physically get there, so a move given much less
# than this either snaps or arrives late.
MIN_CONTINUATION_S: Final = 0.75

# A multi-shot storyboard crammed into a short budget is usually a habit
# carried over from models that cap out at 5s. H3 does not.
HEADROOM_DURATION_S: Final = 7
HEADROOM_MIN_SHOTS: Final = 3

# Unhurried screen delivery. Used to estimate whether a written line can be
# spoken inside the beat it sits on; performance speed varies, so treat what it
# produces as an estimate rather than a limit.
SPEECH_WORDS_PER_MINUTE: Final = 150
# Below this fraction of the beat there is no room left for a breath or a pause.
SPEECH_COMFORTABLE_FILL: Final = 0.8


def align_frames(count: int) -> int:
    """Snap a frame count up to the model's 17k+5 grid.

    Mirrors `align_frame_count` in the node, which increments until the count
    is congruent — so a request is never silently shortened.
    """
    count = max(MIN_FRAMES, int(count))
    while count % FRAME_GRID != FRAME_OFFSET:
        count += 1
    return count


def frames_for_seconds(seconds: float) -> int:
    """The frame count the node will actually use for a requested duration."""
    return align_frames(math.ceil(seconds * FPS))


def seconds_for_frames(frames: int) -> float:
    """Real clip length, which is rarely the round number that was asked for."""
    return round(frames / FPS, 3)


def is_trained_length(frames: int) -> bool:
    return TRAINED_MIN_FRAMES <= frames <= TRAINED_MAX_FRAMES


def is_valid_duration(seconds: object) -> bool:
    """True when `seconds` is a whole-second budget the node will generate.

    Wider than the trained range at the bottom: `is_trained_length` is what
    says whether the frame count is one the model was taught.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, int):
        return False
    return MIN_DURATION_S <= seconds <= MAX_DURATION_S


def adapt_canvas(
    width: float, height: float, max_pixels: float = MAX_PIXELS,
    *, upscale: bool = False,
) -> tuple[int, int]:
    """768-short-edge canvas with an area cap, per-axis rounded to 32.

    A copy of the node's own sizing, so the editor can show the dimensions
    that will really be used rather than the ones that were typed.

    `max_pixels` defaults to the node's own cap but can be lowered to size a
    render to a megapixel budget: the same aspect ratio, scaled down to hit the
    target area. A budget below the 768-short-edge nominal always bites, so a
    preview at e.g. 0.2 MP comes out proportionally smaller.

    A budget above the nominal is ignored unless `upscale` is set. Only a
    progressive (SelfLift) finish asks for that: it composes at low resolution
    and writes its final canvas straight onto the latent node, so the 768 edge
    is not a limit it has to respect. Without the flag a "1080p" budget would
    silently come back as 1344x768.
    """
    ratio = width / height
    if ratio >= 1.0:
        nominal_w, nominal_h = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nominal_w, nominal_h = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio

    area = nominal_w * nominal_h
    if area > max_pixels or (upscale and area < max_pixels):
        scale = math.sqrt(max_pixels / area)
        nominal_w, nominal_h = nominal_w * scale, nominal_h * scale

    return (
        max(CANVAS_MULTIPLE, round(nominal_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def canvas_for_ratio(ratio: str) -> tuple[int, int]:
    """Width and height for an aspect ratio like `16:9`."""
    try:
        width, height = (float(part) for part in ratio.split(":", 1))
    except ValueError as error:
        raise ValueError(f"Not an aspect ratio: {ratio!r}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"Not an aspect ratio: {ratio!r}")
    return adapt_canvas(width, height)
