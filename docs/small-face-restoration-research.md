# Improving small-face restoration

Research checked 2026-10-03 against primary papers, author repositories, the installed ComfyUI node schema, and our office-video experiments. SeedVR2 and a six-track KEEP benchmark were subsequently completed; see the experiment results below. Other candidate models remain untested on this clip.

The next priority is the restoration model. Enlarging, restoring, and shrinking a face is a reasonable way to improve its appearance at the original size. It does not require recovering an authentic high-resolution identity. However, reconstruction must remain consistent over time and preserve the original pose and expression.

Our current 56-frame crowd test processed 30 tracks and 1,609/1,611 accepted face observations, yet several faces stayed soft or became overly smooth. That coverage denominator excludes undetected faces. Person-guided detection, including learned enlargement of head regions, added no tracks persisting for at least three observations in this excerpt. These results favor testing restoration quality before another generic person-detector expansion.

## Recommended candidates

| Priority | Candidate | Why test it | Practical limitation |
|---|---|---|---|
| 1 | SeedVR2 whole-frame video restoration | Processes the scene without requiring a face box, using temporal information | May alter details throughout the scene; memory and hallucination must be measured |
| 2 | KEEP on tracked face sequences | Designed for video face super-resolution, with information propagated across frames | Needs correctly aligned per-person crops; default script handles only one selected face |
| 3 | BFVR-STC on the same face sequences | Video face enhancement with spatial/temporal codebooks and de-flickering | Separate environment/integration and aligned-crop preparation |
| Diagnostic baseline | CodeFormer or GFPGAN | Face-specific restoration can reveal whether a facial prior beats general H3 refinement | Frame-by-frame output can flicker or change identity; not automatically a production video solution |

SeedVR2 is a one-step video restoration model with released 3B/7B variants. Its original research implementation has much larger published GPU requirements than our machine, so use the memory-managed ComfyUI integration rather than assume the research defaults will fit. [Author repository](https://github.com/ByteDance-Seed/SeedVR)

The installed server exposes SeedVR2VideoUpscaler and DiT/VAE loaders. Locally verified weights are `seedvr2_ema_3b_fp8_e4m3fn.safetensors` and `ema_vae_fp16.safetensors`. Its live schema supports frame batches of 4n+1, overlap, padding, color correction, VAE tiling and model offloading. A proposed first test uses batch 5, overlap 2, zero added input/latent noise, padding of the final batch, and a 1080-pixel short edge; increase to batch 9 only if memory permits. These are starting settings, not a demonstrated optimum. The integration documents at least five frames for temporal processing. [ComfyUI maintainer documentation](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

KEEP propagates information from previously restored frames to regulate the current restoration. Its released models and inference code make it an actionable face-specific comparison. [Author project](https://jnjaby.github.io/projects/KEEP/) · [Code and models](https://github.com/jnjaby/KEEP)

An important integration detail: KEEP's inference script defaults to selecting the center face, also requests the largest face, and calls landmark detection with an eye-distance threshold of five. Passing our whole crowd through that default script would repeat the original single-face mistake. Feed each tracked, enlarged and aligned face sequence separately through the aligned-input path. Do not invent precise landmarks when an extreme profile or tiny face does not support them. [Inference implementation](https://github.com/jnjaby/KEEP/blob/main/inference_keep.py)

BFVR-STC explicitly targets video face restoration and de-flickering; code and pretrained models are released. Its inference script resizes input frames to 512×512, so it should receive prepared face sequences, not a whole office scene. Results on portrait benchmarks are not proof that it will fix eight-pixel crowd faces. [Author repository](https://github.com/Dixin-Lab/BFVR-STC) · [Inference script](https://github.com/Dixin-Lab/BFVR-STC/blob/main/scripts/infer_bfvr.py)

CodeFormer offers a fidelity control; higher values favor fidelity over reconstruction quality. Use it as a controlled comparator rather than assuming that accepting video input provides a temporal restoration model. Its inference processes faces frame by frame and also applies the five-pixel eye-distance filter. [Author documentation](https://github.com/sczhou/CodeFormer) · [Inference implementation](https://github.com/sczhou/CodeFormer/blob/master/inference_codeformer.py)

GFPGAN provides a second face-specific baseline. The authors themselves list identity changes and different sharpness/naturalness tradeoffs among versions. [Author repository](https://github.com/TencentARC/GFPGAN)

## Detection and tracking changes worth testing separately

Compare SCRFD-10G with landmarks, or RetinaFace, against our current face_yolov8m detector using identical crops and human-reviewed frames. SCRFD provides ONNX inference and landmark variants. Its benchmark scores should not be treated as predicted recall on this generated video. [SCRFD implementation and benchmarks](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)

Try 384/512-pixel overlapping tiles in the distant crowd region in addition to our current 768-pixel tiles. At a 1280-pixel detector input, a 384-pixel tile makes an eight-pixel face about 27 detector-input pixels tall, compared with about 13 from a 768-pixel tile. This is a proposed geometry change, not extra source detail or guaranteed recall. Sliced inference is an established small-object detection approach. [SAHI](https://github.com/obss/sahi)

Keep confidence scores and use weak detections to continue an established track, instead of accepting every low-score box as a new face. ByteTrack demonstrates this association strategy. For longer dropouts, evaluate motion/point tracking with visibility checks; never paste a front-facing repair across a turn to the back of the head. [ByteTrack](https://github.com/FoundationVision/ByteTrack)

Our current adapter zeroes the composite on missing detections. Carefully validated gap handling may reduce changes between refined and original frames, but it is lower priority for this excerpt, where accepted-observation coverage was already high.

## Implementation and evaluation plan

1. Compare the existing H3 result with SeedVR2-only restoration on the same 56 frames. Export both native-size and 1080p versions. Keep audio and timing unchanged.
2. Select six face tracks spanning 8–12, 13–24 and 25–64 source-pixel heights. Compare KEEP and a CodeFormer baseline using identical input crops. Test BFVR-STC if these do not produce a clear improvement.
3. Review playback and matched-frame crops for facial plausibility, expression changes, flicker, mask edges and false-face generation. Count manually verified visible faces rather than calling accepted detections total coverage.
4. Avoid unnecessary resize cycles. The current learned pre-upscale helper shrinks an already enlarged backend crop toward source size before upscaling it again. A new adapter should read the original crop directly, restore it, and perform one final downsample/composite. This is an implementation hypothesis to test, not a measured cause of the softness.
5. Promote only the winning path. Do not stack H3, CodeFormer, KEEP and SeedVR2 by default: a stronger-looking result from one stage does not establish that another stage will improve it further.

DVFace is a relevant newer research direction, but the repository inspected still lists testing code and pretrained-model release as TODO items. It is not the immediate implementation candidate. [Author repository](https://github.com/zhengchen1999/DVFace)

## Completed experiment: SeedVR2 and KEEP

Recorded 2026-10-04. Input was the same 56-frame office excerpt (original frames 24–79), 1344×736 at 24 fps, 2.33 seconds. Native-size comparisons preserve the original frame count and audio. These are diagnostic results, not a new production default.

### SeedVR2

ComfyUI prompt `0b5475f4-93f6-4e19-8c48-37d7f35020dd` succeeded in 149.973 seconds according to execution history. Used the installed 3B FP8 model, FP16 VAE, batch 5, temporal overlap 2, seed 42, no added noise and LAB color correction. Saved 1972×1080 and Lanczos-downsampled 1344×736 outputs. The entire scene was restored without face detection. This provides scene coverage, not evidence that every visible face was repaired.

Matched-frame review favors SeedVR2 overall for this excerpt: 11–13-pixel profile faces have clearer contours and foreground faces have cleaner features after downsampling. The approximately eight-pixel, partly occluded face remains indistinct and its apparent orientation changes. It also changes hair, clothing, and background detail. Five sampled time points were inspected; the saved video is available for motion review. No objective fidelity ground truth exists for these generated faces.

### KEEP

Used the official `KEEP-b76feb75.pth` checkpoint with strict state-dictionary loading and original model architecture. Dependencies were isolated in `tmp/keep-test-deps`; no ComfyUI package versions were changed. A small import bootstrap loads only required inference modules to avoid unrelated training and diffusion-pipeline dependencies.

Selected six spatially separated, slow-moving tracks using prior face detections, with median source face heights 8.6, 11.2, 13.5, 19.1, 37.5, and 50.6 pixels. This selection is a controlled benchmark and is not a production crowd tracker. RetinaFace ran on enlarged local crops, with center/size gates against the existing face boxes. It found five-point landmarks in 335 of 336 observations: one missing frame on the smallest face. Landmarks were smoothed with sigma 3; the single missing observation was interpolated for model input but skipped during final compositing.

Each face sequence was aligned to 512×512 directly from original pixels, restored independently in eight-frame chunks using float32, and composited with a feathered elliptical mask. Inverse warps were supersampled 4× and area-downsampled to avoid aliasing. All six sequences completed; inference after model loading took 31.35 seconds for 336 aligned frames. This timing excludes detection, alignment, loading, and export and is not directly comparable to whole-job H3/SeedVR2 timings.

KEEP substantially sharpens the two larger aligned faces. On the smallest inputs it invents conspicuous structures, including beard-like detail on the 11-pixel profile and distorted features around the occluded eight-pixel face. Downsampling conceals some artifacts but does not produce a clear overall improvement over SeedVR2. Sampled-frame review is insufficient to establish flicker-free playback, particularly at eight-frame chunk boundaries and the skipped landmark frame. Do not promote this KEEP configuration as a general small-face fix.

### Artifacts and recommendation

- Four-column comparison: `tmp/restoration-comparison.mp4` (original / H3 / SeedVR2 / KEEP six-face subset).
- Native-pixel crop sheet: `tmp/restoration-face-details.png`; five-time-point sheet: `tmp/restoration-temporal-check.jpg`.
- KEEP aligned input/output diagnostic: `tmp/keep-aligned-preview.jpg`; native composite: `tmp/keep-office-native.mp4`.
- SeedVR2 submission/history: `tmp/seedvr-office-prompt.json`, `tmp/seedvr-office-history.json`.
- KEEP coverage and timing: `tmp/keep-office/landmark-audit.json`, `inference-timing.json`, `composite-audit.json`.
- Reproducible scripts: `tmp/test_seedvr_office.py`, `tmp/test_keep_office.py`, `tmp/keep_bootstrap.py`, `tmp/compare_keep.py`.

SeedVR2 is the strongest candidate from this test for a broader clip evaluation. Neither method establishes that almost every small face can be fixed. Keep the existing production behavior pending review of the comparison; further validation should include movement, occlusion, and longer sequences before integrating a new default.

## Face-only SeedVR2 compositing experiment

User requested inspection of blend quality before integration. Reused the completed native-size SeedVR2 output and the original 56 frames; no additional AI restoration run was required. Cached original-frame detections from the person-fallback audit were associated using the current `plan_tracks`: 29 eligible tracks out of 31 detected tracks, with 1,599 eligible observations. These cached detections are a different audit from the earlier H3 run's 30/32 track count; neither count measures all visible people.

Two variants copy SeedVR2 pixels into the original with tracked elliptical masks. The tight mask has horizontal/vertical radii of 0.65 times face-box width/height; the wider variant uses 0.90/0.85. Both use a smooth transition over the outer 35% of their radius, four-times mask supersampling, and one-frame-sigma box smoothing within contiguous detections. Missing observations are not filled. Overlapping masks combine with a maximum rather than repeatedly applying the enhancement. These are box-derived soft masks, not semantic face segmentation.

The tight and wide masks touch an average of 1.389% and 2.467% of frame pixels. The script asserted exact equality with the original outside mask support on every frame before encoding. H.264 re-encoding can still change pixels throughout the exported picture. Both native results and the four-column comparison were fully decoded and verified as 56 frames, 24 fps, 2.33 seconds with original audio stream copied into the exports.

Visual inspection of matched crops and five time points shows no conspicuous hard seam on the sampled larger faces. Tight masks preserve softer original hair around sharper facial features; the wider variant transitions farther into head surroundings. It also transfers more of SeedVR2's changed detail. Tiny-face geometry changes remain present in both. The masks have not yet been validated on fast motion, occlusion, or extended tracking dropouts, and sampled stills do not establish flicker-free playback.

Artifacts:
- `tmp/seedvr-faces-only-comparison.mp4`: original / whole-scene SeedVR2 / tight face mask / wider face mask.
- `tmp/seedvr-faces-only.mp4`, `tmp/seedvr-faces-only-wide.mp4`: full native-size results.
- `tmp/seedvr-faces-only-details.png`, `tmp/seedvr-faces-only-temporal.jpg`: close-ups and time samples.
- `tmp/seedvr-faces-mask-overlay.png`: yellow overlay of tight-mask coverage at frame 24.
- `tmp/seedvr-faces-only-audit.json`, `tmp/seedvr-faces-only-media-check.json`: coverage and verification.
- `tmp/test_seedvr_faces_only.py`: reproducible experiment, without production changes.

## Full-frame difference maps

Compared all 56 synchronized native-size frames with the original using the mean absolute difference of the three 8-bit RGB channels at each pixel. All heatmaps use the same linear 0–24 scale, clipping larger changes to the brightest color. These are pixel-change measurements, not restoration-quality scores. No image alignment or independent auto-contrast was applied.

| Variant versus original | Mean absolute RGB change / 255 | Pixels with mean RGB change > 5 |
|---|---:|---:|
| Whole-scene SeedVR2 saved render | 4.7985 | 27.643% |
| Tight blend before final export | 0.07612 | 0.4925% |
| Wider blend before final export | 0.13238 | 0.8367% |
| Tight blend saved video | 2.0281 | 1.3725% |
| Wider blend saved video | 2.0716 | 1.6958% |

The pre-export composites were reconstructed from the same masks and saved SeedVR2 render. Frame 24 matches the earlier lossless composite PNGs exactly. All reconstructed frames preserve source pixels outside mask support exactly. Whole-scene SeedVR2 differences include its existing H.264 encoding and downsampling; this does not isolate neural processing from those steps. The saved blend's additional change includes final color conversion, chroma subsampling, and compression. Its export-only difference averages 1.9751 channel levels; this explains widespread faint background signal in saved-video heatmaps.

Maps visibly locate whole-scene changes across blinds, desks, clothing and ceiling edges. Tight blending removes these changes outside its small face regions. This supports the scope-preservation benefit of masking, not a claim that every retained facial change is good or that a whole-scene change is bad.

- `tmp/seedvr-difference-maps.png`: matched frame 24, whole-versus-original, tight-blend-versus-original, and whole-versus-tight-blend.
- `tmp/seedvr-difference-average.png`: mean absolute-change maps over all 56 frames, whole/tight/wide.
- `tmp/seedvr-difference-export.png`: saved-video maps with an export-only control.
- `tmp/seedvr-difference-map-video.mp4`: whole/tight/wide results above their synchronized difference maps, with audio.
- `tmp/seedvr-difference-report.json`: definitions, per-frame metrics, aggregate measurements, paths and limitations.
- `tmp/seedvr_difference_maps.py`: reproducible analysis script.


## Temporal flashing investigation (2026-10-04)

The user's full enhancement (`5e852e7b-8b33-4ac1-9cb5-12b9277e7806`) processed all 294 frames of `VideoPrompter_H3_00240_.mp4` with 3B FP8, 1080 short edge, LAB color correction, zero added noise, batch size 5, overlap 2, and no prepended warm-up. It completed in 946 seconds. Source cuts are at frames 134 and 183.

The installed SeedVR2 implementation advances each batch by `batch_size - temporal_overlap`, hence three frames for the initial preset. Its overlap weights for a two-frame overlap are exactly `[1, 0]`, providing no fractional crossfade. A five-frame overlap includes a fractional middle weight. Residual-change measurements show repeated three-frame spikes, consistent with batch-boundary instability. This is evidence for a contributor, not proof that every visible flash shares this cause.

A close-up-only test (`7e867abb-7985-4c8f-ab60-fe531a7687fa`) used 21-frame batches, 5-frame overlap, and 4 prepended warm-up frames, retaining other settings. All 111 source frames matched the corresponding original decoded images at the diagnostic resolution. It completed in 209 seconds; reported peak allocated GPU memory was 8.94 GB, reserved 13.80 GB. The extraction had a container-reported rate of 24.218 fps, inherited by its output; the comparison explicitly plays all variants at the original 24 fps. The extraction script now specifies CFR 24 for future runs.

Mean absolute adjacent change of the restoration residual (Gaussian-blurred luma, measured at 672 x 368) decreased from 0.3474 to 0.2838, about 18.3%. The 95th percentile decreased from 0.6829 to 0.5372. This diagnostic includes movement, altered geometry, resizing and encoding; it is not a pure perceptual flicker score. The close-up test also resets context at the cut, unlike the full original run, so it cannot isolate the parameter effect alone. A full-clip comparison is required before promoting the preset.

Reproducible evidence: `tmp/audit_seedvr_flashing.py`, `tmp/seedvr-flash-audit.json`, `tmp/test_seedvr_temporal.py`, `tmp/compare_seedvr_temporal.py`, `tmp/seedvr-temporal-comparison.json`, `tmp/seedvr-temporal-comparison.mp4`. Official parameter guidance: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler .

### Full-clip validation and default update

Full test `9e1c0ea5-e5f7-4959-9209-0ccdc0aea09d` completed in 548.14 seconds with the same full source, seed and non-temporal settings. It preserves all 294 frames at 24 fps, with a 1972 x 1080 output and source audio connection. Mean residual change decreased from 0.24449 to 0.20724 (15.2%); the 95th percentile decreased from 0.60346 to 0.47763. Excluding cut neighborhoods, means decreased in the wide shot (0.14308 to 0.13513), monitor shot (0.12291 to 0.11872), and close-up (0.34485 to 0.29723). Selected background regions slightly worsened: ceiling 0.11696 to 0.12453, close-up blinds 0.14340 to 0.14533. This is a modest overall improvement, not a universal reduction or proof of flicker-free playback.

Matched frames around both cuts reveal cross-shot ghosting: previous-shot details appear in the following scene. The longer preset reduces some artifacts, but faint horizontal structures remain over the first close-up frame. The workflow currently does not segment shots before restoration. Shot-by-shot processing is the appropriate follow-up for that separate failure mode; it has not been implemented in this update.

Promoted the tested 21/5/4 temporal preset for both enhancement modes, with LAB, noise, resolution and model unchanged. All 26 targeted SeedVR2/refinement tests pass, including regression checks for the new preset in all three delivery choices. README documents memory and scene-cut limitations. Artifacts: `tmp/seedvr-temporal-full-comparison.mp4`, `.png`, `.json`, `tmp/seedvr-temporal-cut-*.jpg`, and `tmp/compare_seedvr_temporal_full.py`. The three columns are original / old preset / new preset. The comparison plays at the original 24 fps.
