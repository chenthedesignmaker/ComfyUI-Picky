from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_prompter import Beat, Storyboard, VideoPrompt  # noqa: E402

# The rooftop-chase prompt this project started from. Deliberately kept in the
# shape `compose` used to emit — the opening stated ahead of `[Shot 1]` and the
# soundscape cue comma-spliced onto the bed — so the parser stays able to read
# prompts written that way, whether by an older build or by hand. New output
# folds the opening into `[Shot 1]` and writes the soundscape as sentences.
ROOFTOP_PROMPT = """\
integrated_multimodal_description: Realistic live-action cinematic look, \
anamorphic lens, film grain. A post-rain dusk metropolis with flying-car traffic \
between the towers. [Shot 1] A high side angle frames the protagonist sprinting at \
the roof edge as pursuers appear in the rooftop doorway behind him. The camera pans \
right with large amplitude at fast speed. [Shot 2] At 00:04.000, the camera cuts to the \
protagonist leaping across the gap between buildings, his body stretching mid-air \
with towers and light trails behind him. [Shot 3] At 00:08.000, the camera cuts to \
a low-angle shot as he lands, rolls, rises and keeps running through the fog. \
[Shot 4] At 00:12.000, the shot switches to a held silhouette at the edge of the \
next roof as he launches into the jump.

overall_soundscape: wind, rapid footsteps and city ambience with an accent hit on \
each leap, at 12s: the score bursts open.

non_diegetic_music: A driving percussive score at a fast tempo with low brass \
stabs, swelling into the final leap and cutting out on the silhouette."""


@pytest.fixture
def rooftop_text() -> str:
    return ROOFTOP_PROMPT


@pytest.fixture
def clean_prompt() -> VideoPrompt:
    """A well-formed 10s prompt: contiguous shots, every block filled."""
    return VideoPrompt(
        storyboard=Storyboard(
            duration=10,
            beats=[
                Beat(0, 4, "wide establishing", "the city wakes under low cloud"),
                Beat(4, 7, "medium tracking", "she crosses the empty plaza"),
                Beat(7, 10, "close-up", "she looks up as the rain starts"),
            ],
        ),
        style="Realistic live-action cinematic look, anamorphic lens, film grain.",
        camera="Clean hard cuts, no dissolves.",
        audio="rain, distant traffic, a low sustained string bed",
        music="Sparse piano at a slow tempo under sustained low strings.",
        negatives=["no watermarks", "no cartoon rendering"],
    )
