# Quality tiers, in depth

The one-line choice lives in SKILL.md. Read this when choosing between the
two-pass or SelfLift tiers, when refining a finished take, when an extend refuses a tier, when a face drifts only at
`preview`, or when render time looks wrong.

## The tiers

- `Q` (quality tier) — a `(megapixels, steps, turbo_strength)` preset. Quality
  is orthogonal to aspect ratio: the canvas is the shot's ratio resized to the
  megapixel budget, so the resolutions below assume a ~16:9 shot.
  - **`preview`** (0.2 MP ≈ 608 wide, 8 steps): the **adherence** pass,
    ~1.5 min/take. Not merely the cheap tier — the *obedient* one. H3 follows
    reference, keyframe and shot direction markedly better at a small canvas
    than at 720p, because every extra pixel is another visual token competing
    with your text. Steps do not buy that back: more steps at the same seed
    polish the picture without recovering an instruction the model already
    dropped. **Prove every prompt here first** — cuts, dialogue, timing,
    references — and only then pay for framing. If a shot is being ignored at
    720p, drop to `preview` and see whether it lands at all; if it does, the
    prompt is fine and the resolution was the problem.
  - **`turbo_hd`** (1.0 MP ≈ 1344×736, 8 steps): fast HD drafting & sweeps,
    ~4.5 min/take. For seed-hunting once the prompt itself is settled.
  - **`turbo_2pass`** (8 steps, split 6 + 2): `turbo_hd`'s budget spent
    differently — six steps composing at 0.2 MP, then the clean latent
    upscaled to 1.0 MP and two more for texture. Deliberately step-for-step
    identical to `turbo_hd` so the two can be compared honestly; the only
    variable is *where* the schedule is spent. Measured on a clock-digit
    close-up it beat one-pass `turbo_hd` on adherence at the same cost —
    `7:29:58` legible with its colons, where one-pass dropped the colons and
    showed the wrong second. Reach for it when a turbo draft keeps missing
    direction that `preview` lands. Unlike `turbo_hd` it sends references at
    `max` — `match` on a two-pass tier would size faces to pass 1's small
    canvas — so it costs more on a ref2v shot; drop it to `match` per
    project on shots with no faces.
    - **The 6 is not a free parameter, and is not the middle.** What the
      latent upscaler needs is a nearly-clean latent, and the *sigma* the
      schedule has reached decides that, not the step count. At shift 12 the
      curve is violently back-loaded: an 8-step run sits at sigma 0.923
      after four steps — 92% noise — and the upscaler magnifies a guess.
      Measured output at 4 + 4 is heavy speckle and a blown-out display.
      Cutting at 6 reaches sigma 0.800, the same handover `final_2pass`
      uses, and comes back clean. 5 + 3 (0.878) still carries blocky edge
      artefacts; 7 + 1 hands over cleanly but leaves too few steps to
      resolve the digits, which come back ghosted.
    - Published H3 two-pass workflows do claim 4 + 4 works, and every
      setting that distinguishes them from ours was tested and **ruled
      out**: `euler` instead of `res_multistep`; a hand-written pass-2 sigma
      list instead of `SplitSigmas.low_sigmas`; and their **4-step** turbo
      LoRA instead of the 8-step one. Each left 4 + 4 exactly as broken,
      individually and all together. The one difference left standing is the
      UNET — those workflows load stock `*_pruned_int8_convrot` weights,
      while this profile runs the `true_hybrid` checkpoint, which the turbo
      LoRAs were not distilled against. Do not copy a 4 + 4 split off a
      workflow found online without re-testing it on this profile.
  - **`final`** (1.0 MP ≈ 1344×736, 25 steps): the keeper — the same model run
    harder for maximum micro-detail, natural physics and audio clarity;
    ~20 min/take.
  - **`final_2pass`** (20 base-model steps, split 15 + 5): the keeper that
    refuses the trade-off between the two tiers above. The first 15 steps run
    at `preview`'s 0.2 MP, where H3 actually obeys; the sampler's **clean**
    latent (`denoised_output`) is then upscaled to 1.0 MP and the last 5 steps
    are spent on skin, cloth and grain at full size. You get a preview's
    adherence at a final's resolution.
    - Reach for it when a shot has something small that must survive — clock
      digits, a sign, a face at distance, precise hand action. Those are
      exactly what a one-pass 720p render loses.
    - The split is not a free parameter. Five finishing steps is about right;
      many more and the model starts re-inventing the fine detail pass 1 got
      right rather than sharpening it. Do not raise it to "add quality".
    - Nothing to install or re-capture: the graph is rewritten at prepare
      time (`video_prompter/twostage.py`) by finding nodes by class, so t2v,
      i2v and ref2v all get it. It needs the workflow to sample with
      `SamplerCustomAdvanced` and to have a `VAEDecodeAudio` — a **422**
      says which of those is missing.
    - On an extend, `preview` and `turbo_hd` are the **same render**. They
      differ only in a megapixel budget, and a continuation drops it (see
      below), so the two build a byte-identical graph — 8 turbo steps at the
      parent's canvas either way. The editor's extend picker folds them into
      one entry, keeping whichever name matches the canvas that will really
      be delivered. Nothing stops the API taking either name; they just mean
      the same thing there.
    - `/render/extend` takes them too, with one twist. A continuation
      *finishes* on the canvas of the clip it continues (the pinned frames
      come from there), so a one-pass tier's megapixel budget is dropped and
      only its step count applies. A two-pass tier still means something:
      pass 1 runs at the tier's small canvas and pass 2 finishes at the
      parent's. That only pays when the parent is at least twice the tier's
      first pass in pixels — extending a 608x352 draft at `final_2pass` is
      refused with a **400** rather than rendering one pass under a two-pass
      name. Pass 2 gets a Motion Context node of its own, because H3 packs
      continuation guides into the spatial grid of the latent being sampled,
      and pass 2's is bigger.
  - **`singularity_2pass`** (24 steps total, 20 base @ 0.5 MP + 4 turbo @ 2.1 MP / 1080p):
    the ultra-fidelity keeper using the community **Singularity Ref2va** checkpoint.
    Pass 1 runs 20 pure base-model steps at half resolution (0.5 MP ≈ 960×544 for 16:9)
    with Turbo LoRA bypassed. Between passes, the clean latent is upscaled to 1080p
    (2.1 MP ≈ 1920×1088) via `MinimaxH3LatentUpscaler3D`. Pass 2 attaches the 8-step
    Turbo LoRA and finishes with 4 steps at denoise 0.35.
    - Vastly superior specular lighting, skin texture, and hand/limb articulation
      over stock weights, eliminating waxy sheen and plastic skin.
    - Like `final_2pass`, reference sizing defaults to `max` (2048px short edge).
    - Note: `/render/extend` currently does not support `singularity_2pass` (returns 400);
      use `turbo_hd` or `final_2pass` for continuations.
  - **SelfLift tiers** (`selflift_hd`, `selflift_fhd`, `selflift_final`):
    progressive sampling through the installed `comfyui-SelfLift` nodes. The
    first steps run at half resolution inside `SelfLiftH3Sampler`
    (`lowres_scale` 0.5), the latent is lifted by the 3D latent upscaler, and
    the rest finish at full size, with `SelfLiftH3TST` patched onto the model
    for frame-to-frame consistency. The rewrite is `video_prompter/selflift.py`.

    | tier | finished canvas (16:9) | steps (low-res + full) | turbo |
    |---|---|---|---|
    | `selflift_hd` | 1344×736 | 8 (6 + 2) | on |
    | `selflift_fhd` | **1920×1088** | 8 (6 + 2) | on |
    | `selflift_final` | **1920×1088** | 20 (15 + 5) | bypassed |

    - These are the only tiers that go **above** the 768-short-edge canvas.
      SelfLift writes the finished size straight onto the node that makes the
      sampler's latent, so it works whether a profile sizes by width/height or
      through a megapixel node. The 1080p tiers need a lot of GPU memory; on a
      24 GB card keep them to short clips (see
      `references/long-form-and-upscale.md` for the 1088p limit).
    - The first number in the tier's `quality_presets` row (0.2 / 0.5 MP) is
      not used by SelfLift; the pass split and the finished size are.
    - **Not for continuations.** `/render/extend` refuses a SelfLift tier with
      a **400** (its graph only knows the classic two-stage split). Use
      `turbo_2pass` or `final_2pass` there. The editor hides them for extends.
    - Tuning per project: `settings.selflift_settings` takes `lowres_scale`,
      `tst_tau`, `upscaler_model`, `rho`, `w_min`, `w_max`.
  - **Refine & Upscale** is not a tier; it post-processes a finished take.
    `POST /video_prompter/render/refine` with `project`, `source` (the take's
    `"subfolder/file.mp4 [output]"` reference), `take_id`, `width`, `height`
    and `prompt_text`, plus `guided_resample`, `face_refine` and
    `guided_resample_megapixels` (2.1 → 1920×1088, `null` keeps the size).
    It returns a workflow to submit like any other; the take comes back with
    `quality: "refined"` and `refined_from` pointing at the original.
    - Face refine works on the **largest face in frame**. A reference face is
      picked automatically only when the take's shot has exactly one cast
      character; with two or more, or none, it refines without one. Pass
      `reference_face` yourself when you know who it is.
    - `seconds`/`frames` default to the take's own recorded length — don't
      send a guess, the guided pass crops its output to it.
  - Each tier also fixes **how large identity references arrive** on a ref2v
    shot (`quality.reference_size`, baked into the graph as the node's
    `ref_image_size`). `match` scales every reference down to that tier's own
    canvas; `max` sends a 2048px short edge. `preview` and `turbo_hd` use
    `match`; `turbo_2pass`, the SelfLift tiers, `final` and `final_2pass`
    use `max`.
    - This matters more than it looks: at `preview`'s 0.2 MP a full-body
      sheet arrives about 450px across, so the tier you prove prompts on is
      also the one with the least identity to work from. A face that drifts
      only at `preview` may be a sizing artefact, not a prompt fault —
      don't rewrite a working prompt to chase it.
    - A project can buy it back per tier via
      `settings.quality_presets.<tier>.reference_size` (the editor's settings
      panel has a **refs** dropdown per row). Reference tokens ride through
      every sampling step, so `max` costs real time.
  - There are three profiles, all sampling with `res_multistep` (checked
    2026-09-20):

    | profile | UNET | turbo LoRA |
    |---|---|---|
    | `H3 Hybrid local t2v` | `true_hybrid` checkpoint | none |
    | `H3 local t2v` | `pruned_int8_convrot` | none |
    | `H3 T2V w Turbo` | `pruned_int8_convrot` | 8-step, strength 1.0 |

    On the two without a LoRA the single-pass tiers differ only by megapixel
    budget and step count — both already driven by the preset, no re-capture
    needed. The
    preset also carries a `turbo_strength` (1.0 on the draft tiers, 0.0 on
    `final`), but it only does anything if a graph actually loads a LoRA whose
    node is **titled "turbo"** — then `final` zeroes it. Wire that in and
    re-capture only if you later put a turbo LoRA on a full base model.

## Render time: what is and is not worth engineering around

Do **not** try to free VRAM first. It is a tempting fix and it is measurably
wrong — 16 controlled runs at 576x352 / 8 steps / refs=match, interleaved:

| change | no free | `/free` first |
|---|---|---|
| seed only   | **47.8s** median | 53.9s |
| prompt      | **60.3s** median | 73.9s |

The distributions barely overlap, so freeing reliably *costs* 6-14s and buys
nothing. Renders that began with 0.8-1.3GiB free VRAM showed no penalty at
all: ComfyUI's own swapping is already good, and pre-emptying the card just
makes it reload. Do not diagnose from Windows' "shared GPU memory" counter
either — it reads ~25GB with the GPU idle at 0% and torch holding 0.1GB, so
it tracks reserved address space, not thrashing.

What *is* real: a prompt change costs ~12s more than a seed change, because
it invalidates `ref` and `MiniMaxH3ReferenceToVideo` then has to re-run the
clip and both VAEs (it takes `clip`, `vae` and `audio_vae` as inputs to the
one node). Modest, and the price of changing the words — not something to
engineer around.

Renders do sometimes spike 3-8x (39.8s against 328s, same graph). Measured
2026-08-30, that is a **GPU stall, not slowness and not memory**: free RAM
27 GB, process disk and CPU idle, the GPU at 95 W of 450 W with its memory
controller at 0% while `nvidia-smi` reads "100% utilization". `POST /interrupt`
cannot cancel it — ComfyUI checks the flag between sampler steps, and the
stalled step never returns — so a render stuck far past its tier's usual time
has to be killed with the process. Launching with `--cache-ram 16` cut the
median `turbo_2pass` total from 268s to 121s; **never add `--fast-disk`**, which
hung outright on its second render. Do not attribute any of this to VRAM.
