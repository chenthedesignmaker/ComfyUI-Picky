# video prompter

A **timeline editor** for **MiniMax H3 (Hailuo 3.0)** on **local open weights
in ComfyUI**. You fill a bin with generations, write each one's prompt, render
as many candidates as you like, choose the one that worked, and cut the result
together.

The working loop is: *add a source → write it → sweep a few seeds →
double-click to compare the candidates → choose one → trim its clip → add the
next source.*

## SeedVR2 enhancement

Open a video take's sparkle-button menu to choose:

- **Enhance Whole Video — Original Size** restores the full scene with SeedVR2,
  then returns to the exact source dimensions. This is also the sparkle button's
  primary action.
- **Enhance Whole Video — Upscale** saves the enhanced scene at a 1080-pixel
  short-edge target, capped at 2048 pixels on the long edge. Larger sources are
  retained at their original dimensions.
- **Enhance Faces Only — Blend** runs the same full-scene restoration, then
  blends detected faces 8–192 source pixels tall back into the original using
  the tighter soft masks from our comparison. Output stays at source size.

Each option creates a new take linked to its source and keeps the source audio,
frame count and frame rate. Whole-video enhancement can change clothing,
backgrounds and facial details. Face Blend preserves pixels outside its masks
before encoding, but missed faces remain unchanged and moving mask edges need
review. Face Blend remains experimental; both modes are available for comparison.

Both modes use 21-frame temporal batches, 5-frame overlap and 4 prepended warm-up
frames. This replaces the initial 5-frame/2-frame-overlap preset to reduce
batch-boundary flashing. It needs more memory per batch and does not guarantee
flicker-free output; review movement and scene cuts. SeedVR2 still processes the
entire take without automatically splitting it into shots.

These options use SeedVR2 alone, without stacking H3 refinement. They require
the SeedVR2 ComfyUI nodes, `seedvr2_ema_3b_fp8_e4m3fn.safetensors` and
`ema_vae_fp16.safetensors`. Face Blend also requires
`models/ultralytics/bbox/face_yolov8m.pt`, ultralytics and scenedetect.
Project `refine_models` may override `seedvr2_dit` and `seedvr2_vae` filenames.
Restart ComfyUI after installation to register `VideoPrompterSeedVR2FaceBlend`,
then refresh the editor.

API: send `enhancement_mode` as `seedvr2` or `seedvr2_faces` and
`enhancement_resolution` as `original` or `1080p` to `/render/refine`.
Face Blend currently requires `original`. Omitting `enhancement_mode` preserves
the existing H3 API behavior. The take's Refinements field records SeedVR2,
Face Blend and upscaling; the Face Refine Report records blend detection coverage.

## H3 small-face refinement

Use the take's refinement menu → **Fix Small Faces (1080p)**. The face pass
detects on the full frame and enlarged overlapping tiles, tracks multiple faces,
and restores tiny source crops with RealESRGAN before H3 refinement. Repaired
crops are blended directly into the larger delivery canvas, retaining detail
that would be lost by shrinking them to native size before upscaling. The
**Native** option keeps the original output resolution.
It preserves the original audio. The combined texture/upscale actions use the
same small-face pass. **Refine Largest Face Only** remains a separate targeting
option within that menu.

By default, faces must be 8–192 source pixels tall and detected in at least
three frames. Tracking resets at camera cuts; missing or ambiguous detections
and close-ups are excluded from the face composite. Very tiny, obscured or
undetected faces can remain unchanged. A crowded shot takes longer because
each tracked face gets its own temporal sampling pass.

The take's **Face Refine Report** records detected/eligible tracks and repaired
face observations across frames. These counts measure detected coverage, not
unique people or guaranteed visual improvement; undetected faces cannot be
counted. Each person uses their own source crop as the reference;
the small-face mode never copies one cast member's face onto everyone. API
clients can select `face_refine_target` as `small_faces` (default),
`largest_face`, or `reference_identity` with `reference_face`.

This requires the installed Easy H3 Face Refine backend, its face detector and
PySceneDetect. The 1080p small-face action also requires
`RealESRGAN_x4plus.safetensors` in ComfyUI's `models/upscale_models` folder;
the native action does not load this model. Workflow builders can override
`face_upscale_model` (or set it to an empty string to disable pre-upscaling).
Upscaling improves detail but cannot recover a reliable identity from a few
pixels; H3 may still change expressions. Restart ComfyUI after installing this update to register the
`VideoPrompterSmallFaceRefine` node, then refresh the editor.

## The three types

A single word — "shot" — used to do three unrelated jobs at once: what to
generate, which take won, and how the edit uses it. Each disagreed with the
others eventually. They are three types now, and this is the whole model:

| | What it is | Where it lives |
|---|---|---|
| **Source** | a bin item: one lineage of generations, its `segments` and one ordered `path` of take ids | `sources.py` |
| **Segment** | one generation link — a prompt and a quality, at depth 0, 1, 2 … | `sources.py` |
| **Clip** | one use of a source on the timeline: in and out points, speed, enabled | `sequence.py` |
| **Take** | one rendered file | `gallery.py` |
| **Beat** | one `[0s-4s] Shot 1: …` line *inside* a single generation's storyboard | `storyboard.py` |

A plain un-extended source is a source with **one segment**. There is no "is
this a chain or a single?" branch anywhere — that special case is what the old
model kept tripping over.

The word *shot* survives in exactly one place, and it is not this model: the
prompt text H3 reads says `Shot N` for a beat. That is MiniMax's format and it
is unchanged. (`Take.shot` also keeps its name, because every `takes.json` on
disk carries it; what it holds is a source id.)

**Which take follows which** lives in one place: `path`, an ordered list of take
ids, one per generation. It is sound when each take's recorded `parent_take` is
the take before it — one line, checked, and it can say "this link is broken"
instead of quietly substituting something that looks about right.

## Two clocks

**You cannot generate a two-second cut.** The frame grid snaps to 17k+5 and the
trained floor is 124 frames, so the shortest generation is a hair over five
seconds. A short cut is therefore a *trimmed* one, and two clocks meet:

- **lineage time** — what H3 made: a source's takes laid end to end. What the
  prompt's beats address, and what the linter checks.
- **timeline time** — `source_in`..`source_out` over `speed`. What the edit
  uses.

So the idiom is **generate long, cut short**. Trimming and re-timing never
touch the prompt, which is why the frame grid and the editing layer never
have to know about each other.

A clip's in and out points are seconds into *lineage* time, not into any one
file. That is what lets a clip start inside generation two and run across the
join into generation three without the edit knowing there was a join —
`lineage.resolve` turns a range of that clock into the real files covering it,
and the monitor, the previews and the export all go through it, so a join
cannot behave one way on screen and another in the file.

Clip *order* is the edit. Positions are derived by running total, so gaps and
overlaps cannot be represented, let alone go wrong. It is a cuts-only
assembly, which is an honest description of a pile of separate generations.

## Status

Working end to end: composition core, the bin and timeline with their edit
operations, Motion Context chains, preset library, project store, render
profiles, seed-sweep batches, export, the ComfyUI node, the HTTP layer and a
standalone timeline editor.

Project files are **schema v2** (sources and clips). A v1 file is migrated as it
is read — one shot becomes one source with one segment plus one clip — so
nothing on disk needs converting and the timeline is unchanged frame for frame.

Install by putting this directory in `ComfyUI/custom_nodes/`. The node is
**Video Prompt (MiniMax H3)**; the editor opens from its *Open editor* button
or directly at `/video_prompter/app`.

**Scope: text-to-video.** `examples/h3_t2v.json` is a ready workflow — load it
in ComfyUI, capture a profile from the node, and the editor drives prompt,
frame count, canvas and seed with nothing to bind by hand.

Image-to-video works too (same `MiniMaxH3ImageToVideo` node, same FL2VA
checkpoint) but a hand-built i2v graph usually derives geometry from the loaded
image, so the profile controls less — see *Render profiles* below.

Reference-to-video is deliberately **not** supported yet.
`MiniMaxH3ReferenceToVideo` runs a different checkpoint (Ref2VA, not FL2VA), so
it is a second model rather than extra inputs. The prompt model underneath
already handles reference roles and `<Picture i>` tags, and the linter checks
them, but nothing is surfaced in the editor.

**Requires ComfyUI 0.30.0 or later** — the local H3 nodes landed in
[Comfy-Org/ComfyUI#15224](https://github.com/Comfy-Org/ComfyUI/pull/15224) on
2026-08-03.

## Local, not the hosted API

The two paths differ in ways that change the tool, and everything here follows
the local nodes (`comfy_extras/nodes_minimax_h3.py`), not the API docs:

| | Hosted API | Local weights |
|---|---|---|
| Duration | 4–15s, integer seconds | frame count on a **17k+5 grid**, snapped up |
| Trained range | — | 124–362 frames = **5–15s** |
| Allowed here | — | **1–15s**; under 5s renders but warns as untrained |
| Negative prompt | — | **none**; a single `prompt` string |
| Sizing | `768P` / `2K` preset | `width`/`height`, multiples of 32, 768·1344 pixel cap |
| References | "Image 1 is the mood reference" | **`<Picture i>` / `<Video k>` / `<Audio j>` tags** |

Two consequences worth knowing:

- **A 10s clip is not 10s.** It becomes 243 frames, which is 10.125s. **8s is
  the only whole-second budget in the trained range that lands exactly on the
  grid** — every other one runs long. The linter tells you the real length.
- **4s does not work.** It aligns to 107 frames, under the trained minimum of
  124. The floor is 5s.

## The six-block model

MiniMax's official prompts share a consistent architecture, which `VideoPrompt`
mirrors one field per block:

| Block | Field | Notes |
|---|---|---|
| Reference roles | `references` | omni-reference: 9 images, 3 videos, 3 video soundtracks, 3 standalone audio |
| Style contract | `style` | medium, texture, palette, era — stated before `[Shot 1]`, fixes the look across shots |
| Scene overview | `overview` | optional; the second sentence of that opening, still before `[Shot 1]` |
| Storyboard | `storyboard.beats` | `[Shot N] At MM:SS.mmm, <cut> composition, action` |
| Camera | `camera` | cross-shot rules, including explicit refusals |
| Audio | `audio` + `storyboard.cues` | generated in the same pass as picture, so cue timing belongs here |
| On-screen text | `screen_text` | literal strings, quoted, with a spelling guard |
| Negative list | `negatives` | banned transitions, objects, clichés — part of the prompt text, since there is no negative input |

References are declared by tag, and the linter warns when one is attached but
never named in the prompt:

```
<Picture 1> is the overall mood and style reference. <Picture 2> is the lead character.
```

## Usage

Lint and preview a prompt:

```bash
python -m video_prompter examples/rooftop_chase.txt
```

Refit onto the full 15s budget and close any gaps or overlaps:

```bash
python -m video_prompter examples/rooftop_chase.txt --fix -d 15 -o out.txt
```

Build one from presets alone — no input file needed:

```bash
python -m video_prompter -d 12 -p style:"Anamorphic night city" -p camera:"Hard cuts" -p storyboard:Chase
```

See what's available:

```bash
python -m video_prompter --list-presets
```

Exit status is 1 when any error-level finding is present and 2 on a bad
preset reference, so it drops into a pre-generation check.

As a library:

```python
from video_prompter import compose, lint, parse

prompt = parse(open("examples/rooftop_chase.txt", encoding="utf-8").read())
prompt.storyboard = prompt.storyboard.refit(15).make_contiguous()
print(compose(prompt))
for finding in lint(prompt):
    print(finding)
```

## Presets

Plain text files at `<root>/<kind>/<name>.txt`, one preset per file, so the
library stays hand-editable and diffs cleanly. Kinds are `style`, `camera`,
`audio`, `negatives` and `storyboard`.

Roots stack — the first is the writable one, later ones are read-only
fallbacks:

```python
from video_prompter import PresetLibrary, bundled_root

library = PresetLibrary("my/presets", "../comfy prompt node/presets", bundled_root())
library.apply(prompt, "style", "Anamorphic night city")
```

That stacking is how this library and the image tool's can share the blocks
that transfer between them without either project depending on the other.

Storyboard presets are stored as ordinary timecoded shot lists at whatever
duration they were authored, and refit onto the prompt's current budget when
applied — so a five-beat pattern written at 15s drops onto a 6s clip with its
shape intact.

## Projects: the bin and the timeline

A project is two lists. `sources` is what was generated; `timeline.clips` is how
the edit uses them, in order.

```python
from video_prompter import Clip, ProjectStore, blank_source

store = ProjectStore("projects")
store.create("Rooftop", [blank_source("Opening", 12)])

def add_landing(project):
    source = project.add_source(blank_source("Landing", 8))
    project.timeline.add(Clip(source_id=source.id))

store.update("Rooftop", add_landing)
```

`create` places one clip per source for you; the two steps are separate above
because they are separate ideas — a source can sit in the bin unused, or back
several clips at different trims.

Sources, clips and takes carry **stable ids**. `Take.shot` holds the id of the
source it rendered for rather than its name, which is the thing that stops
renaming orphaning a render history; `Take.id` reuses the
`(prompt_id, filename)` pair that recording already dedupes on, so it needs no
storage of its own.

Storage is carried over from `comfy prompt node`, which earned this shape:

- writes go through a temp file and `os.replace`, so a crashed save cannot
  leave a truncated `project.json`
- every save snapshots the previous state into `.backups/` across three tiers —
  a per-save ring that churns fast, plus hourly and daily tiers that survive
  save bursts
- a save that drops more than 30% of a project's sources or beats is *red
  flagged* into a fourth, longer-lived tier; it still saves, it just leaves a
  rope behind
- backups are best-effort and never block a save
- `update()` does load-mutate-save under a per-project lock, which is what
  keeps concurrent edits from dropping each other's sources

### Folding a chain back together

Every extend used to be authored as its own shot, so a ten-generation chain
landed on disk as ten bin rows and ten clips, and the v1 migration deliberately
kept it that way rather than guessing which rows belonged together.

Opening a project folds those rows back into the one source they always were.
It is not a guess: the renderer wrote down which take each continuation
continues, so when one row's path ends exactly where the next row's path
begins, they are the same lineage. Where that link is ambiguous — a fork two
rows both continue, or two rows standing on the same take — the run is left
alone.

**The edit does not move.** Each absorbed clip is rebased onto the surviving
source by its offset in lineage time, and only clips that already ran back to
back, untrimmed and in order, collapse into one. A trimmed clip does not meet
its neighbour, so a trim can never be folded away. Verified across every project
on disk: 22 runs folded, 70 rows absorbed, and not one frame of drift.

The takes follow — `Take.shot` is re-stamped onto the source that now owns them
— and the absorbed row's name is kept on the segment it named. It runs once:
a folded project has no runs left to find. `.backups/` holds the state from
before, as with any other save.

### Project defaults

A project holds default `style`, `camera` and `negatives`. New segments copy
them on creation; there is no live inheritance, because it would have to be
unpicked in `compose` and would fight the per-segment edits that are the point.
An explicit *match all* overwrites every segment of every source — which is what
you actually want when source nine has drifted from source one.

## Edit operations

Trim, speed, enable/disable, reorder, delete, duplicate and split.

**Reordering belongs to the timeline, not the bin.** A clip drags along the
track to a new cut, and a source drags out of the bin onto a cut to become
another use of that lineage. The bin has no order to change: it lists what was
generated, the timeline says how it is used, and a source used twice has no
single position a bin row could stand for.

**Split** is the interesting one, and it got simpler with sources: both halves
point at the same source, so there is no prompt to deep-copy and no pick to keep
in step. They are one lineage seen through two windows — splitting spends no
render, and the export references one file twice.

Clamping happens on the server, against how much lineage there **really is**,
not the prompt's budget. A length is *measured where takes exist and planned
beyond them*: the path's real durations, plus the prompt budget of every segment
past the end of the path. So an unrendered source still holds its place on the
ruler, a finished one is all measurement, and planning a third generation widens
its clip immediately rather than after the render.

**Flipping a source's branch updates its clips.** They follow by default, which
is what makes comparing two branches against the edit worth doing at all;
*pinning* freezes one clip on the branch it is showing, and is the deliberate
act. Dead branches are never auto-archived — off-path takes stay visible and
playable so they can be compared, and go only by hand.

## Candidates

Comparing needs candidates, and the cheap way to get them is the same prompt at
different seeds. **How many seeds** is a control of its own in the render bar,
not a button: set it above ×1 and the buttons beside it queue that many renders
at consecutive seeds instead of one. So candidates are not stuck at draft
resolution, and an overnight hold can be a sweep. It sits next to the seed pin
because the two answer one question together: the pin is where the seeds start,
the count is how many follow. Pinned, the whole sweep is reproducible from that
one number; unpinned, the base is rolled with headroom for the run.

Both controls follow you down a chain. At generation 1 or deeper the render bar
is an *Extend*, and it sweeps too — same parent, same pinned run, same words, so
the takes differ only in what the model invented past the context, which is the
only part of a continuation that was ever a guess. The pin there sets the
continuation's own seed, not the parent's: what the parent rolled is already
baked into the frames being carried over.

Seeds are capped at 2⁵³−1 rather than ComfyUI's 2⁶⁴−1. A seed makes the whole
round trip through the browser, and JavaScript numbers are float64: past 2⁵³,
consecutive seeds collapse onto the same value and a sweep quietly becomes N
renders of one seed. A pinned base is clamped to leave room for the sweep it
starts, for the same reason.

Double-clicking opens the candidates for the generation you are authoring. Rate
them, discard the failures, choose one — or select two to four and compare them
over one transport, looping the trimmed range so you are judging the cut you
would actually use.

The takes on offer are only the children of whatever the path stands on at the
depth above, so a take belonging to a branch you are not on is never drawn. That
is the invariant the old parent/child guessing kept breaking, and it is now
structural rather than checked.

The take sheet's **Lineage** view shows every chain the source has, one family
per row with generations running left to right. Families are the long axis — a
source commonly has twenty of them and almost none run past two generations — so
they get the vertical scroll, and a take sits beside its own continuation.
Adopting a chain there sets the source's whole path in one write.

## Export

The timeline has to become a file, or it is a mood board with a ruler.

- **ffmpeg** — one concat graph that trims, re-times, and fits every piece to a
  common canvas.
- **EDL** — CMX3600, so a round trip into an NLE lands on the same frames.
- **clip list** — plain text, for when the question is which takes went in.

Export resolves each clip through the same `lineage.resolve` the monitor plays
through, so a clip that crosses a generation join comes out as two pieces, back
to back, cut at exactly the frames you were watching.

Sound is resolved separately and **mixed**, not concatenated: each piece is
trimmed, re-timed, delayed to its place on the timeline and summed. It has to be
a mix rather than a join, because a clip may be muted or may be heard from a
take whose picture was rejected, so the audio pieces do not line up one-to-one
with the video ones. A clip whose sound cannot be found exports silent rather
than blocking a picture that is finished; the export reports which ones.

Nothing here copies video; takes already point into ComfyUI's output directory.

Two things fail the export rather than being skipped, because both produce a
result that looks finished and is not: a clip with **nothing to play**, and a
take file that is **shorter than the trim expects** (checked with `ffprobe`
before anything runs — trimming past the end of a file does not error, it just
comes up short).

Set `VIDEO_PROMPTER_OUTPUT_DIR` when ComfyUI's output directory cannot be
introspected.

## Render profiles

The editor never serializes a live canvas. You capture a workflow once; the
profile records which `[node_id, input_name]` pairs carry the prompt, frame
count, canvas and seeds, and rendering writes values into a copy of that
template.

```python
from video_prompter import capture, inject, compose

profile = capture("H3 local t2v", api_format_workflow)
workflow = inject(profile, prompt, compose(prompt))   # ready for POST /prompt
```

Bindings are discovered by reading the graph, and deliberately narrowly: only
*literal* inputs on recognised nodes. An input already wired to another node is
left alone, because the wire means the author meant it to come from elsewhere.

Real workflows route values through nodes of their own, and discovery cannot
infer those — a float is just a float. Pin them by hand:

```python
# duration lives in a PrimitiveFloat feeding a frame-alignment expression
profile.bind("duration_s", "105:111", "value")
print(profile.describe())
# your i2v graph: controls prompt, duration_s, seed
#   the workflow owns: length, width, height
```

`duration_s` carries seconds for graphs that align frames themselves; `length`
carries the already-aligned frame count for graphs that don't. A profile can
use either.

The point of the indirection is that the editor knows nothing about H3's node
graph. Swap the sampler, add a LoRA, move from `MiniMaxH3ImageToVideo` to
`MiniMaxH3ReferenceToVideo` — re-capture, and the editor is unchanged. When a
binding goes missing, `StaleProfileError` says re-capture rather than guessing
at a repair.

`inject` writes `length` as the **aligned frame count**, not the requested
seconds, so what renders is what the linter reported.

### What a profile does not control

`describe()` reports which roles a profile drives and which the workflow keeps,
and the editor shows the same thing next to the profile picker. This matters
more than it sounds: ComfyUI's official H3 template sizes the canvas with a
`ResolutionSelector` at 0.4 MP, which for 16:9 gives **864×480** — not the
1344×768 that `adapt_canvas` would report. When a profile does not bind
`width`/`height`, the editor brackets the figure to say so rather than showing
a number the render will disagree with.

The shipped `examples/h3_t2v.json` avoids the problem by keeping width, height
and length as literals, so the editor owns them outright.

### Continuations run on the shot's own graph

An Extend is not a second pipeline. `extend.augment_extend` is a *transform over
a finished workflow*, the same way `twostage` is: it takes whatever the shot
would have rendered as — t2v, i2v, or ref2v with the cast wired in — finds the
H3 conditioning node, the two VAEs and the decodes by class and by wire, and
inserts Motion Context between the conditioning and the guider. So a
continuation inherits the checkpoint, the LoRAs, the sampler and every reference
image the shot itself renders with.

It was a graph of its own for a while, and a fixed graph can only ever be one
mode on one set of weights. Continuations of a `<Subject 1>` shot went out with
no reference node in them at all and on a checkpoint nobody had chosen: the
model got the description, never got the face, and invented one.

Two canvas rules follow from the pinned frames coming out of the parent clip. A
continuation *finishes* on the parent's canvas, so a one-pass tier's megapixel
budget is dropped and only its step count applies. A two-pass tier still earns
its second pass — pass 1 at the tier's small obedient canvas, pass 2 back up at
the parent's — and it gets a **second Motion Context node** to do it, because H3
packs continuation guides into the spatial grid of the latent being sampled
(`PackedLayout` in `comfy/ldm/minimax/model.py`) and pass 2's grid is bigger.
When the parent is not at least twice the tier's first pass in pixels there is
nothing for pass 2 to add, and the request is refused rather than rendered as
one pass under a two-pass name.

A consequence worth stating out loud: **Quick and Turbo HD are the same render
on an extend.** They differ only in that megapixel budget, and a single-pass
continuation drops it, so both build a byte-identical graph — eight turbo steps
at the parent's canvas. The extend picker folds tiers that come out identical
into one entry and keeps the name that matches the canvas actually delivered, so
a 720p parent offers Turbo HD and a 608×352 one offers Quick. Every option there
also carries its real canvas and step split in the label, because "Quick" on a
720p parent costs what Turbo HD costs and that is better said than discovered.

That collision is also why a take now records **the tier it was asked for, by
name** (`Take.quality`, carried in the queue metadata). The gallery used to read
the tier back off the canvas and the step count, which is exact for a fresh
generation — every tier has its own signature — and cannot work for a
continuation rendering at somebody else's canvas: 1.0 MP at 8 steps *is*
`turbo_hd`, whatever was asked for. Takes made before the name was recorded still
fall back to reading the numbers.

## The node

`Video Prompt (MiniMax H3)` takes a project and a source — by id or by name,
defaulting to the first — and emits `prompt`, `length`, `width`, `height`,
`seconds` from its first segment. (Its widget is still called `shot`: the name
is saved inside every workflow anyone has built, and renaming it would blank
them all.) A later segment is a continuation, rendered through the extend graph
rather than this node.

`length` is the frame count **already aligned to the 17k+5 grid**, from the
same table the linter uses. A hand-built graph usually carries a seconds
primitive feeding a maths expression to do that alignment; this replaces it,
and guarantees the number in the graph is the number the editor reported.

## The editor

Served at `/video_prompter/app`. The **bin** down the left, a program monitor,
and a **timeline** of clips sized by their timeline length. The bin has two
views: **Shots**, one row per source with its lineage length, its takes and how
many clips use it; and **Images**, every picture generated for the project, with
the button that makes more. Double-clicking
either opens the editor: the form for the generation being authored, one tab per
generation when there is a chain, a live preview, findings, and the candidates
for that depth.

A clip spanning a generation join draws the join as a dashed seam, because that
is a normal thing here rather than damage, and the seams are snap points for the
playhead like any other cut.

### Authoring tabs

The tabs follow the shape of the composed prompt rather than the shape of the
data model, because the writer is proof-reading one document:

| Tab | Holds | Composes to |
| --- | --- | --- |
| **Source** | name, duration, aspect, clip trim and speed | nothing — the clip, not the prompt |
| **Scene** | the opening (style, overview), the fallback camera, the shots, on-screen text, negatives | `integrated_multimodal_description`, in that order |
| **Audio** | the diegetic bed, its timed cues, and the score | `overall_soundscape` and `non_diegetic_music` |
| **Media** | keyframes, references, generated images, ref2v task types and summary | the alignment directive, or the six-section reference document |
| **Findings** | the lint | — |
| **Prompt** | the composed text | — |

A shot is a **card**, not a table row: composition, action and camera are prose,
and prose in a 100px column cannot be proof-read. **Spoken lines are nested in
the shot that speaks them**, since that is where H3 reads them — authoring them
in a separate table means checking one sentence against another across two tabs.
Fields inside a card run in the order the shot composes: how the cut into it
reads → composition → subjects / environment / action → camera.

A line carries no timecode, so the only thing saying *when* it lands is where
its sentence sits among the shot's others — which is why the card gives each of
the format's two slots **its own place and its own add button**: `placement:
"open"` between the framing clause and the action, the default after the camera
move. A line is authored in the slot it composes into, and the dropdown on a
line moves it to the other one. An opening line needs both a composition and an
action to sit between; without them it falls back to the front of the camera
sentence and `dialogue.no_opening` says so.

The **Prompt** tab shows the composed text **tinted by which field wrote each
part** — hover names the field, clicking jumps to it, and a legend chip mutes a
kind. The offsets come from `/video_prompter/preview` as `spans` (see
`provenance.py`); tinting is read-only, and the *Edit as text* button switches to
the plain editable pane, where a hand edit overrides `compose()` for the next
render exactly as before.

Render goes `POST /video_prompter/render/prepare` → `POST /prompt`. Preparing
is where the linting happens: **error-level findings return 422 and the render
never reaches the queue**, because a prompt with overlapping beats will cost a
generation to discover. 409 means the profile's bindings no longer match its
workflow — re-capture.

The editor holds no model rules of its own, and no timeline arithmetic either.
Contiguity, refitting and canvas sizing go through
`POST /video_prompter/transform`; trim clamping, clip lengths, start positions,
the branch tree per depth and the resolved playlist all come back from
`GET /video_prompter/sequence`. The frame grid, `adapt_canvas`, the two-clock
arithmetic and the join each exist in exactly one place.

## Generated images

A keyframe does not have to come from somewhere else. The bin's **Images** view
is the library and the front door — *Generate images* there, **Generate** in the
Media tab, the pick button on either keyframe slot, and **Generate** beside a
cast member's identity files all open the same image studio: a prompt, a
cinematic look preset, a seed, a step count, a canvas, a detail factor and a
count. It sweeps consecutive seeds through Krea 2 Turbo — a few seconds each —
and every image that lands is kept in the project's library.

The destination is a control rather than three separate sheets, because writing
an image, hunting a seed for it and choosing the winner is one activity and the
only thing that differs is where the picked image goes:

| Destination | What it sets | The segment then renders as |
| --- | --- | --- |
| First / last frame | `first_frame` / `last_frame` | image-to-video |
| Reference | a `<Picture N>` reference | reference-to-video |
| Identity file | a cast member's `references` | whatever names that member |

Stills live in `stills.json`, beside `takes.json` and for the same reasons — but
they are deliberately **not** takes. A take is a delivered clip: cut into the
timeline, rated, extended, exported. A still is an *input*, and mixing the two
would put a PNG in the take grid where it could be picked into an edit.

What a still records is the **recipe**, not the pixels: the prompt, seed, style,
steps and canvas, plus which segment or cast member it was made for. That is the
part no finished file carries and the part you want when a composition was nearly
right — the ↻ button loads a recipe back into the form so one thing can be
varied. Every image is **adopted into ComfyUI's input directory as it is
recorded**, because an image `LoadImage` cannot name is not usable as a keyframe,
and one the author has to adopt by hand is one they will forget to adopt.
*Forget* drops the library record and leaves the file alone, so anything already
pointing at it keeps working.

Clicking any image — in the bin or in the studio — opens it **in a lightbox**
rather than a browser tab, which is a one-way trip out of the editor. The arrow
keys walk whichever set it was opened from, so comparing two seeds is two
keypresses, and the destination control comes with it: you judge a picture and
use it from the same place. Escape steps back one layer at a time — lightbox,
then studio, then the editor.

Recording is keyed on the queue metadata: a run is an image generation because it
was stamped `mode: "t2i"` (or `"Keyframe (T2I)"`), never because its output
happens to be a PNG — a video render that saves a preview frame must not land in
the keyframe library. An agent that stamps the same mode gets its images filed
here too, picked up on the next page load like any other finished run.

## Takes

Finished renders are recorded per project in `takes.json`, beside
`project.json` rather than inside it — output history should not trip the
project store's shrink heuristic, and authored content stays separable from
results.

A take **references** ComfyUI's output file rather than copying it. A 15s clip
is megabytes, ComfyUI already owns the file, and it serves it from `/view`.
*Forget* drops the record and leaves the file alone.

Two things keep the list honest:

- The editor mints its own `prompt_id`, so a render is tracked from the moment
  it is sent. Execution events are routed to the submitting socket, so a tab
  only ever sees its own runs.
- On load the page replays ComfyUI's history, which picks up anything that
  finished while it was closed. Recording is **idempotent on
  `(prompt_id, filename)`**, so replaying costs nothing.

One detail worth knowing if you extend this: `SaveVideo` reports through
`ui.PreviewVideo`, whose `as_dict` is `{"images": [...], "animated": True}` —
so **video outputs arrive under the `images` key**, told apart from stills by
extension. When a workflow saves both, the clip wins.

## Layout

| File | Role |
|---|---|
| `h3_spec.py` | every hard limit and heuristic in one place, mirroring the node's own arithmetic |
| `storyboard.py` | `Beat`, `AudioCue`, `Storyboard`; refit, contiguity, snapping — inside one generation |
| `sources.py` | `Source`, `Segment`; the bin — a lineage, its plan, and its chosen `path` |
| `sequence.py` | `Clip`, `Timeline`; the edit, its two clocks, trim, split, reorder |
| `lineage.py` | the continuation forest from `parent_take` alone, and `resolve()` |
| `edit.py` | where the bin, the edit and the take record meet: lengths, branches, paths |
| `migrate.py` | a v1 project dict → v2, run on every read |
| `fold.py` | a pre-sources chain spread across sibling sources → one, run once at open |
| `document.py` | `VideoPrompt` and the omni-reference set |
| `compose.py` | structure → prompt text |
| `parse.py` | prompt text → structure |
| `provenance.py` | which authored field wrote each stretch of the composed text |
| `lint.py` | the checks on one prompt, graded info / warning / error |
| `sequence_lint.py` | the checks that need the edit: broken chains, stale takes, trims past the lineage |
| `serialize.py` | structure ↔ plain dicts, for disk and wire |
| `presets.py` | the preset library over stacked roots |
| `projects.py` | project storage, defaults, atomic writes, tiered backups, locks |
| `profiles.py` | render profiles: capture, discover, validate, inject |
| `render.py` | compose + lint + inject, seed sweeps, and the block on error-level findings |
| `extend.py` | Motion Context as a transform over a render graph: continue a take on its own weights, mode and references |
| `gallery.py` | finished takes, ratings, and parsing them out of ComfyUI history |
| `t2i.py` | the Krea 2 Turbo image graph: look presets, LoRA stack, supersampling |
| `stills.py` | generated images per project — the recipe, the adopted file, the library |
| `export.py` | ffmpeg graph, EDL, clip list, and the short-source check |
| `api.py` | dicts in, dicts out — the editor's whole surface, no HTTP |
| `paths.py` | storage roots, inside ComfyUI or standalone |
| `comfy_node.py` | the ComfyUI node |
| `comfy_routes.py` | aiohttp routes, a thin adapter over `api.py` |
| `cli.py` | lint, preview, refit, apply presets |
| `../js/video_prompter.js` | canvas launcher and profile capture |
| `../js/standalone/` | the editor page |

## Notable behaviour

- `make_contiguous()` snaps cut points to a 0.5s grid so timecodes read like a
  person wrote them, falling back to exact proportions when the grid is too
  coarse to give every beat a slot.
- `refit()` rescales beats and cues proportionally: beat *proportions* are
  authorial, absolute times are not.
- A **stale take** is caught by comparing the take's stored `prompt_text`
  against the current `compose()` — so editing a prompt after a render says so
  rather than leaving you to notice.
- An un-rendered source still holds its place on the timeline, using its
  segments' budget, so a whole sequence can be laid out before anything is
  generated. Rendering resizes it to what really came out.
- A take's recorded duration is the length that was *asked for*; the extend
  graph trims 22 pinned context frames before saving, so a continuation delivers
  0.917s less. Durations are re-read from each container's own `mvhd` header at
  load and corrected in memory. `takes.json` is never rewritten.
- `parent_take` was not always recorded, but the MP4 keeps the answer: an extend
  loads its parent through `LoadVideo` and the whole graph is in the container's
  `prompt` tag. Lineage is recovered from there, also in memory only.
- **The plan sets the depth; the path only decides which take fills each level.**
  A default path stops at the number of segments a source plans — following a
  chain further would hand one source seconds belonging to the next, and the
  edit would play them twice.
- `parse()` recovers audio cues in the `at 4s: ...` form that `compose` writes.
  Timing mentioned in prose (`"the score bursting at 4s"`) stays prose — it is
  not promoted to a cue.
- Deserialization is tolerant by design: unknown keys are ignored, missing ones
  take defaults, a reference of an unrecognised kind is dropped rather than
  failing the whole load, and a source with no id is given one.

## Next

1. **EDL across a join.** One event per resolved piece is right for ffmpeg and
   arguably wrong for a round trip into Resolve, where a clip that crosses a
   join might read better as one event against a merged source.
2. LLM assist: logline → timed storyboard, beats derived from the segments.
3. Audio lanes. A clip's own sound is settled — it can be muted, or heard from
   another take of the same lineage (`sequence.Sound`) — but sound that belongs
   to *nothing on screen*, a music bed or room tone running under three cuts,
   has no picture to hang off. That needs a positioned lane of its own, which
   would be the first thing on this timeline with an explicit start time. The
   export's mixer is already built for it.
4. Clips that are not generations: black slugs for a pause, an imported ending
   card. Those fit the picture track as it stands — a clip resolves to pieces,
   and where a piece comes from is `resolve_clip`'s business — but a title *over*
   picture needs real compositing, which the concat graph cannot do.

## Testing

```bash
python -m pytest
```

When Node.js is on PATH, pytest also runs the client save, refresh and queue
transport regression tests. They can be run directly with
`node --test tests/frontend_save.test.cjs`.

Held renders submit to the configured ComfyUI server address, independently of
browser Host headers. For a TLS endpoint or deployment that needs another
trusted submission URL, set `VIDEO_PROMPTER_COMFY_URL` in the server environment
(for example, `https://comfy.example.com`). New generations save motion context
under a unique path per preparation; reusing a seed no longer replaces an older
take's latent. Existing take paths continue to work as recorded.

EDL exports include audio-lane events after the picture edit. They include
comments for mix details that need manual reconstruction in the receiving NLE,
including overlapping audio tracks, gain and fades. Use the MP4 export for the
rendered mix.

`tests/fixtures/h3_i2v_real.json` is a workflow exported from ComfyUI 0.30.0
rather than a hand-built one, so discovery is tested against what a real graph
looks like: subgraph-prefixed node ids (`105:104`), `SamplerCustomAdvanced` and
`RandomNoise` in place of `KSampler`, and geometry derived from a loaded image.

`tests/test_h3_spec.py` mirrors the node's own arithmetic — `align_frame_count`
and `adapt_canvas`. If a future ComfyUI release changes either, those fail
first.

## Video LoRA controls

The shot editor includes **Shot LoRAs** above Visual style: inherit project defaults,
use a custom stack, or turn additional LoRAs off. Each stack supports installed-file
selection, weights, enable/disable and ordering. Project settings holds optional
shared defaults. See [Video LoRAs](video-loras.md) for usage,
trigger words, continuation inheritance and the API contract.
