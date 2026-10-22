"""Render-quality presets: a megapixel budget, a step count and a turbo-LoRA
strength per level.

Quality is orthogonal to a shot's aspect ratio. The ratio is the creative
choice; the quality level is how hard to work on it. The knobs feed the
workflow through the `megapixels`, `steps` and `turbo_strength` roles (see
`profiles.py`), so a profile that binds any of them gets driven by whichever
preset the shot is set to.

`preview` is not only the cheap tier — it is the *obedient* one. H3 follows
reference, keyframe and shot direction markedly better at a small canvas than
at 720p, because every extra pixel is another visual token competing with the
text for the model's attention. Steps do not buy that back: more steps at the
same seed polish the image without recovering an instruction the model already
dropped. So a prompt is proved at `preview` and only then paid for at `final`;
going straight to `final` to "see it properly" is how a bad prompt gets
expensive.

Five tiers:
- `preview`  — 0.2 MP, 8 steps, turbo LoRA on. The adherence pass: prove that
  the cuts, dialogue, timing and references land before spending on framing.
- `turbo_hd` — 1.0 MP (720p), 8 steps, turbo LoRA on (fast HD drafts/sweeps),
  for seed-hunting once the prompt itself is settled.
- `turbo_2pass` — the same eight turbo steps as `turbo_hd`, spent differently:
  6 at 0.2 MP where the model obeys, then the clean latent upscaled to 1.0 MP
  and 2 more for texture. Deliberately step-for-step identical to `turbo_hd`,
  because the only question it exists to answer is whether *where* the steps
  are spent beats spending them all at 720p. See `REFINE_PRESETS` for why the
  cut is at 6 and not in the middle.
- `final`    — 1.0 MP (720p), 25 steps, turbo LoRA **bypassed** for the keeper's
  full-base quality (strength 0.0).
- `final_2pass` — the keeper that refuses the trade-off above. 20 base-model
  steps split 15 + 5: the first fifteen at 0.2 MP where the model actually
  obeys, then the clean latent upscaled to 1.0 MP and five more steps spent on
  texture. Costs about what `final` costs, and keeps the adherence of
  `preview`. See `twostage.py` for the mechanism.

A tier with a *refine* entry renders in two passes; its `megapixels` is then the
first pass's canvas and `REFINE_PRESETS` carries where the schedule is cut and
what the finished size is.

The built-ins ship as defaults; a project may override the numbers (not the set
of names) via `Project.settings["quality_presets"]`.
"""

from __future__ import annotations

from typing import Any

# (megapixels, steps, turbo_strength). The turbo tiers keep the LoRA at 1.0; the
# keeper bypasses it (0.0) so the full base model renders. Numbers are the
# defaults; a project can override them.
QUALITY_PRESETS: dict[str, tuple[float, int, float]] = {
    "preview": (0.2, 8, 1.0),
    "turbo_hd": (1.0, 8, 1.0),
    # Same eight steps as `turbo_hd`, split across two canvases. The step count
    # is held equal on purpose: a fair A/B of *where* the schedule is spent.
    "turbo_2pass": (0.2, 8, 1.0),
    # SelfLift progressive sampling tiers
    "selflift_hd": (0.2, 8, 1.0),
    "selflift_fhd": (0.5, 8, 1.0),
    "selflift_final": (0.5, 20, 0.0),
    "final": (1.0, 25, 0.0),
    # Two-pass: `megapixels` here is pass 1's canvas, and `steps` the whole
    # schedule across both passes. See REFINE_PRESETS for the rest.
    "final_2pass": (0.2, 20, 0.0),
    # Singularity 2-pass: 20 base steps at 0.5 MP (no turbo LoRA) on Singularity
    # Ref2va UNET, 3D latent upscale to 2.1 MP (1080p), finished with 4 steps
    # @ 0.35 denoise using the 8-step turbo LoRA. Total steps = 24 (20 + 4).
    "singularity_2pass": (0.5, 24, 0.0),
}

# (pass1_steps, output megapixels) for the tiers that render in two passes.
# `pass1_steps` cuts the tier's own step count, so 20 split at 15 gives 15 + 5;
# the output megapixels is the finished canvas, not pass 1's.
#
# 15/5 is not a free parameter. Fewer than five finishing steps does not lay down
# texture; many more re-invents the fine detail pass 1 got right, which is how
# small text and clock digits get mangled.
REFINE_PRESETS: dict[str, tuple[int, float]] = {
    # 6 + 2, and the 6 is not negotiable on this stack. What the latent
    # upscaler needs is a nearly-clean latent, and the sigma the schedule has
    # reached is what decides that — not how many steps have run. At shift 12
    # the curve is violently back-loaded, so an 8-step run cut at 4 hands over
    # at sigma 0.923 (the frame is still 92% noise) and the upscaler magnifies
    # a guess: measured output is heavy speckle and a blown-out display. Cut at
    # 6 and the handover is sigma 0.800 — the same point `final_2pass` hands
    # over at, and the one that demonstrably works. 5 + 3 (sigma 0.878) is
    # visibly better than 4 + 4 but still carries blocky edge artefacts.
    #
    # Published workflows claim 4 + 4 works; every setting distinguishing them
    # from ours was tested and ruled out (`euler` for `res_multistep`, a
    # hand-written pass-2 sigma list for `SplitSigmas.low_sigmas`, and their
    # 4-step turbo LoRA for the 8-step one) — each left 4 + 4 exactly as
    # broken, alone and together. What differs is the UNET: they load stock
    # weights, this profile runs the `true_hybrid` checkpoint.
    "turbo_2pass": (6, 1.0),
    "selflift_hd": (6, 1.0),
    "selflift_fhd": (6, 2.1),
    "selflift_final": (15, 2.1),
    "final_2pass": (15, 1.0),
    "singularity_2pass": (20, 2.1),
}

SELFLIFT_TIERS: set[str] = {"selflift_hd", "selflift_fhd", "selflift_final"}
SINGULARITY_TIERS: set[str] = {"singularity_2pass"}

# How each tier hands identity references to the ref2v node.
#
# `match` scales every reference down to the *render's own* pixel area, which at
# 0.2 MP means a face sheet arrives about 450px on a side — so the cheap tier
# was quietly the one with the least identity to work from, and a character who
# drifts at `preview` is not necessarily a prompt that needs rewriting. `max`
# uses the reference pipeline's 2048px short edge instead. Reference tokens ride
# through every sampling step, so `max` costs real time; the tiers that exist to
# be cheap keep `match`, and the keepers pay for the face.
#
# Only ref2v reads this: t2v has no references and i2v keyframes are a different
# node with its own sizing.
REFERENCE_SIZES: dict[str, str] = {
    "preview": "match",
    "turbo_hd": "match",
    # `max`, unlike its single-pass twin. `match` sizes references to the
    # tier's OWN canvas, which for a two-pass tier is pass 1's 0.2 MP — a face
    # sheet arriving ~450px across. Against `turbo_hd`'s references at 1.0 MP
    # that would lose the comparison for a reason that has nothing to do with
    # the split schedule. It costs time; override it per project if the shot
    # has no faces in it.
    "turbo_2pass": "max",
    "selflift_hd": "max",
    "selflift_fhd": "max",
    "selflift_final": "max",
    "final": "max",
    "final_2pass": "max",
    "singularity_2pass": "max",
}
REFERENCE_SIZE_CHOICES: tuple[str, ...] = ("match", "max")

DEFAULT_QUALITY = "preview"


def levels() -> list[str]:
    return list(QUALITY_PRESETS)


def reference_size(
    name: str | None, project_settings: dict[str, Any] | None = None
) -> str:
    """`match` or `max`: how big this tier lets reference images arrive.

    Kept out of `resolve`'s tuple because it drives graph *construction* rather
    than a scalar the injector writes, the same reason `refine` sits apart.
    """
    level = name if name in QUALITY_PRESETS else DEFAULT_QUALITY
    size = REFERENCE_SIZES.get(level, "match")

    override = ((project_settings or {}).get("quality_presets") or {}).get(level)
    if isinstance(override, dict):
        chosen = override.get("reference_size")
        if chosen in REFERENCE_SIZE_CHOICES:
            size = chosen
    return size


def refine(
    name: str | None, project_settings: dict[str, Any] | None = None
) -> tuple[int, float] | None:
    """(pass1_steps, output megapixels) for a two-pass tier, else None.

    None is the answer for every single-pass tier, so a caller can ask this of
    any level and branch on the result. Unknown names fall back the way
    `resolve` does, so a stale tier renders cheaply and in one pass.
    """
    level = name if name in QUALITY_PRESETS else DEFAULT_QUALITY
    preset = REFINE_PRESETS.get(level)
    if preset is None:
        return None

    pass1_steps, megapixels = preset
    override = ((project_settings or {}).get("quality_presets") or {}).get(level)
    if isinstance(override, dict):
        pass1_steps = _positive_int(override.get("pass1_steps"), pass1_steps)
        megapixels = _positive_float(override.get("refine_megapixels"), megapixels)

    # A cut at or past the whole schedule would leave pass 2 with no sigmas to
    # run, so it is clamped rather than trusted: an override that overshoots
    # should cost a thin second pass, not a failed render.
    total_steps = resolve(level, project_settings)[1]
    return min(pass1_steps, max(total_steps - 1, 1)), megapixels


def resolve(
    name: str | None, project_settings: dict[str, Any] | None = None
) -> tuple[float, int, float]:
    """(megapixels, steps, turbo_strength) for a level, honouring overrides.

    An unknown or missing name falls back to `DEFAULT_QUALITY` rather than
    raising: a stale name on an old shot should still render, just cheaply.
    """
    level = name if name in QUALITY_PRESETS else DEFAULT_QUALITY
    megapixels, steps, turbo_strength = QUALITY_PRESETS[level]

    overrides = (project_settings or {}).get("quality_presets") or {}
    override = overrides.get(level)
    if isinstance(override, dict):
        megapixels = _positive_float(override.get("megapixels"), megapixels)
        steps = _positive_int(override.get("steps"), steps)
        # A bypass is strength 0.0, so this knob permits zero — unlike the others.
        turbo_strength = _non_negative_float(
            override.get("turbo_strength"), turbo_strength
        )

    return megapixels, steps, turbo_strength


def _positive_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if result > 0 else fallback


def _non_negative_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if result >= 0 else fallback


def _positive_int(value: Any, fallback: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return fallback
    return result if result > 0 else fallback


def is_selflift(name: str | None) -> bool:
    """True when this quality tier renders via the SelfLift progressive sampler."""
    return name in SELFLIFT_TIERS


def is_singularity_2pass(name: str | None) -> bool:
    """True when this quality tier renders via the Singularity 2-pass recipe."""
    return name in SINGULARITY_TIERS

