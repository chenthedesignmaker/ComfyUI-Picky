---
name: video-prompt
description: >-
  Author and revise MiniMax-H3 video prompts for this project's NLE from plain
  language. Use when the user describes a scene ("neon alley chase, 12s,
  handheld"), asks to fill in or rewrite a source's fields, or wants to tweak
  part of one verbally ("make the second clip handheld", "punch up the audio",
  "I dislike the ending, redo it"). Drives the running editor's HTTP API so
  edits appear live in the open editor, and renders on the local GPU.
---

# Authoring H3 video prompts

You are the "brain" for this tool: the user describes a scene in natural
language and you produce or revise the **structured** prompt document for a
segment — never raw prompt text. The tool composes and lints; you fill in the
fields, fix what the linter flags, and then **render them yourself** and report
back with the results — the user does not want to press Render.

## The three types, and why the word "shot" is gone

The tool is an NLE and its model says so. `Shot` used to do three unrelated jobs
at once and was split:

- **Source** — a bin item: one lineage of generations, with the `segments` that
  planned it and one ordered `path` of take ids saying which take follows which.
- **Segment** — one generation link: a prompt and a quality, at depth 0, 1, 2 …
  **This is the thing you author.** A plain un-extended source is one segment.
- **Clip** — one use of a source on the timeline: `source_in`/`source_out` in
  seconds of *lineage time* (the source's takes end to end), plus speed and
  enabled. Several clips can share one source.

A take records the source it rendered for in its `shot` field. That field name
is unchanged because every `takes.json` on disk carries it; what it holds is a
source id.

"Shot" survives in exactly one place, and it is not this model: H3's own prompt
format emits `[Shot N] At MM:SS.mmm` for the **beats inside one generation**.
Those are still called shots below, because that is what the model reads.

## The one rule that matters

A prompt is a **structured document**, not free text. You edit the `serialize`
dict (the fields below); the server composes it into H3's official rewrite
format — an optional image-alignment line, then the three literal core fields
`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`.
You never write those fields directly: the storyboard becomes the
`[Shot N] At MM:SS.mmm` description, `audio` becomes the soundscape, `music`
becomes the score. Always validate with `/preview` and clear every **error**
finding before saving.

The format follows MiniMax's own guides (`docs/VIDEO_PROMPT_WRITING_GUIDE_*` on
the H3 repo). Two things it insists on: camera motion is **woven into each
shot** (type + amplitude + speed), never a trailing label; and diegetic sound
(what the scene makes) is kept apart from the audience-only score.

## Where the tool is

The editor's backend is mounted under `/video_prompter` on whatever server is
running:

- Real ComfyUI: `http://127.0.0.1:8188/video_prompter/...` — **use this.** It
  has the GPU, so it is the only one that can actually render, and it is what
  the user runs. Note it loads the node's Python at startup, so backend code
  changes need a ComfyUI restart to go live.
- GPU-free devserver (`python tools/devserver.py`): `http://127.0.0.1:8899/...`
  — a stand-in for developing the tool without a GPU; it cannot render. The
  user does not run it, so default to `:8188` unless it is down.

Drive it with `curl`. Do **not** send `origin` in your writes — that field is
how an open browser tab ignores its *own* edits; omitting it means your writes
live-update the editor the user is watching.

## First, orient yourself

```bash
curl -s http://127.0.0.1:8188/video_prompter/schema
```

This returns the full contract: every prompt field with its type and intent,
the model's hard limits (duration 1–15 whole seconds, canvas multiples of 32,
aspect ratios, reference caps, prompt-character ceiling), the **lint rule
catalogue** (what "good" means), and the operations below. Read it before
drafting so your output passes the linter the first time.

## Read further only when you need it

This file is the core. The rest lives beside it in `references/`; open a file
when the task reaches it, not before.

| When the task involves… | Read |
|---|---|
| `references`, `subjects`, `cast`, `@Name`, anything ref2v | `references/ref2v-and-cast.md` |
| choosing between tiers, two-pass, SelfLift, Refine & Upscale, slow or stuck renders | `references/quality-tiers.md` |
| generating a keyframe or identity image, attaching a file, restoring an old prompt | `references/images-and-media.md` |
| 1080p chains and memory limits, multi-scene chain graphs, SeedVR2 upscaling | `references/long-form-and-upscale.md` |
| a render the script cannot express, or debugging one | `references/rendering-by-hand.md` |

## The fields you write

- `duration` — whole seconds, 1–15 (the storyboard's total budget). Under 5s
  falls outside the model's trained range: it renders, and the linter says so
  with `duration.untrained` rather than blocking it. Reach for a 1–4s budget
  when the shot really is a beat — a reaction, a cutaway, an impact — not to
  save render time.
- `beats` — the timed storyboard, a list of
  `{start, end, angle, action, camera, transition, cut}`. `angle` is framing,
  `action` is what happens. Spans should **tile the budget with no gaps or
  overlaps** — only each beat's *start* is emitted, as `[Shot N] At MM:SS.mmm`
  (the first shot is un-timed), so contiguity is what keeps the cuts sensible.
  A beat is **not** always a shot: see `cut` below.
  - `camera` — this shot's camera **motion**, phrased to follow "The camera …":
    motion type + amplitude (`with small/large amplitude`) + speed (`at
    slow/fast speed`), e.g. `pushes in with small amplitude at slow speed toward
    the door`. Woven into the shot, never a trailing label. Leave empty for a
    held frame.
  - `transition` — the cut phrase that opens a later shot (default `the camera
    cuts to`; also `the shot transitions to`, etc.). Ignored on the first shot,
    and ignored on a beat whose `cut` is false — nothing is cutting.
  - `cut` (default `true`) — **whether this beat opens a `[Shot N]` of its
    own.** Set it `false` and the beat folds into the shot before it as a timed
    moment inside one continuous take: no new marker, no cut phrase, just
    `At MM:SS.mmm, the camera pushes in with small amplitude at slow speed as
    she lifts the mug.` Several beats then compose into one `[Shot N]`, and
    every shot number the document quotes — the `(appears in [Shot 2])` scope,
    the keyframe alignment line — counts shots, not beats.
    - **Reach for it whenever only the camera or the framing changes.** That is
      the same case `beat.weak_cut` complains about; clearing the cut is the
      fix it is asking for. A real cut still needs new information.
    - **The camera must lead the sentence.** The composer writes it that way and
      you should think in it: measured at one seed (2026-08-31), the identical
      move written *behind* its action produced no camera movement at all,
      while the camera-led form rendered it. Put the move in `camera` and the
      performance in `action`; the composer joins them with `as`.
    - On a no-cut beat `angle` is **where the move arrives**, composed as
      `into a wider view of the same kitchen`. It is not a framing to cut to.
    - **Timestamps order events reliably and place them approximately.** In the
      same test the first directed moment landed dead on its mark and the
      second ran about a second late, because a continuous action has to
      physically get there — a cut can reset anywhere, a take cannot. Write the
      order you need; do not count on the exact second late in a take.
  - `lines` — dialogue spoken in the shot, a list of `{speakers, text, language,
    verb, delivery, describe, voiceover, possessive, continuity, cutoff,
    placement}`.
    `speakers` are stable author keys (e.g. `["baker"]`) mapped to `(S1)`,
    `(S2)`… by first vocal appearance across the whole prompt; a key keeps its
    ID across shots, and two keys on one line make a group `(S1,S2)`. `text` is
    the **verbatim** words — only these and the language land inside
    `<d>[Language] …</d>`; everything else (the `describe` identifying phrase
    for a first appearance, the `verb` says/shouts/sings, the `delivery`) is
    rendered outside the tag. Keep punctuation clean inside `text`.
    - `voiceover: true` — off-screen narration. Emits the format's fixed phrase
      `says in an off-screen voiceover`, then the note that the on-screen
      character's lips stay shut. `possessive` picks the word in that note
      (`his`, `her`, `the baker's`); it defaults to `their`.
    - A line that **runs across a cut** is written as two lines on adjacent
      beats: the outgoing half sets `continuity: "into"`, the incoming half
      `continuity: "from"`. Each emits `<scenetrans>` at the seam plus a
      sentence saying the audio keeps running. Tagging only one side is a
      `dialogue.seam_unpaired` warning.
    - `cutoff: true` — the end of the video truncates this line; emits
      `<cutoff>`.
    - `placement` — **where in the shot the line is spoken.** A line carries no
      timecode of its own, so the only thing that says when it lands is where
      its sentence sits among the shot's others. Default `""` puts it last,
      after the action and the camera move. `"open"` puts it directly after the
      framing clause and moves the action behind it, so the speech starts and
      the action follows:
      `[Shot 2] At 00:05.000, the camera cuts to a close-up. (S1) says:
      <d>[English] But here we are.</d> She turns and picks up the letter.`
      A shot may never *begin* on speech — the format opens every one with its
      style (Shot 1) or its cut time and transition — so `"open"` needs an
      `angle` to follow. Without one the line stays put and the linter raises
      `dialogue.no_opening`. Sentence order is a soft cue: if the speech must
      certainly precede the action, give it its own earlier beat or use
      `continuity: "from"` instead.
- `cues` — diegetic audio events `{start, end, description}`; `end: null` is an
  instant hit, a value is a sustained bed. Folded into the soundscape as
  `At MM:SS.mmm, …` / `From … to …, …` sentences, in the same timecode the shots
  use.
  - **Tested, and the timestamp does nothing.** A/B/C/D at one seed
    (2026-08-25): a cue at `00:03.500` produced the clip's *quietest* moment at
    3.5s, and the slam it named arrived at 6.1s — the same 6.1s an untimed
    mention of the slam produced, and the same 6.1s the slam produced when
    written into a beat's `action`. Naming a sound reliably creates it (all
    three were ~40 dB above the no-slam control); nothing tested placed it.
    Treat the time on a cue as documentation for yourself, not direction for
    the model, and write the sound into the beat it belongs to.
  - A cue is not a place for narration. Speech there is *described* rather than
    spoken and has been a source of burned-in captions; put it on the beat as a
    line with `voiceover: true`. The linter raises `audio.cue_is_dialogue`.
  - A cue spanning the whole clip is the bed restated — `audio.cue_covers_clip`.
- `style` — the style contract: medium, texture, palette, era. In the base
  modes it opens **`[Shot 1]` from inside it** and the next clause continues the
  same sentence, exactly as the guide's cases do:
  `[Shot 1] Live-action, cinematic, a medium-wide shot frames a baker…`. (Only
  the full-reference document lifts style and overview onto their own line above
  the marker.)
- `overview` — optional one-line scene overview, taking the clause right after
  the style. Write it lower-case and unpunctuated (`a rain-soaked alley at
  dusk`), not as a capitalised sentence.
- `camera` — a **global** fallback woven into Shot 1 only; prefer per-beat
  `beats[].camera`.
- `audio` — the **diegetic soundscape**: ambient, physical-action and
  non-verbal human sound (wind, footsteps, breathing), merged with `cues` into
  `overall_soundscape` as **1–4 sentences in one paragraph**. Write it as a
  sentence, not a comma list. Dialogue and score do **not** go here. H3 makes
  sound in the same pass, so never leave this empty.
- `music` — the **non-diegetic score** only the audience hears, emitted as
  `non_diegetic_music`: instrumentation, tempo, rhythm, dynamics — not mood
  words. Empty composes to `N/A`. Music a character can hear (radio, jukebox) is
  diegetic — put it in the shot instead.
- `cast` — ids of project cast members who are **on screen but silent** in this
  segment: a location, a character with no line. A member who speaks is named by
  a line's `speakers` instead, never here as well. See "The cast" below.
- `screen_text` — literal on-screen text, reproduced exactly in double quotes.
  Stated in `[Shot 1]`, since the list carries no shot of its own. For signage
  that appears in a **later** shot, write it into that beat's `action` in double
  quotes instead — that is what the format actually wants.
- `negatives` — **not composed.** H3 has no negative input and the official
  format has no negative section, so naming unwanted things only summons them.
  Leave empty; the field is kept only so the editor can still show old prompts.
- `references`, `subjects`, `ref_task`, `ref_summary` — the reference-to-video
  (ref2v) fields. Any reference turns the shot into MiniMax's six-section
  full-reference document, with its own vocabulary and its own ways to go
  wrong. **Read `references/ref2v-and-cast.md` before writing any of them.**
- `first_frame` / `last_frame` — i2v keyframe image names in the input dir.
  Setting `first_frame` makes the shot render as image-to-video (the frame is
  the start anchor); `last_frame` is an optional end anchor. Do **not** combine
  keyframes with `references` (they are different nodes — a `mode.conflict`
  error).
- `width` / `height` — multiples of 32.

## The cast

A project keeps reusable identities — characters, locations, props, voices —
and segments point at one by id instead of describing it again. That is how
source nine keeps looking like source one, so **check the cast before you
invent anybody**:

```bash
curl -s "http://127.0.0.1:8188/video_prompter/cast?project=NAME"
```

- A member who **speaks** goes in that line's `speakers`; one on screen and
  **silent** goes in `cast`. Never both.
- In prose, write `@Mara`, never a typed `<Subject 2>` — the number is
  positional and the composer fills it in.
- Names never reach the prompt; the tag is the identity.
- Editing a member re-composes **every** segment that names them. Say so when
  you do it.

Creating members, per-segment overrides, linked files and the full `@mention`
rules: `references/ref2v-and-cast.md`.

## Render modes (chosen automatically)

The tool picks the H3 task mode from what the shot carries — you do not select
it: **keyframe → image-to-video**, **references → reference-to-video**,
**neither → text-to-video**. All three render the same way; only the prompt
fields differ. Generating a keyframe with Krea, attaching a file, and prompt
history are in `references/images-and-media.md`.

## Workflow

### Draft a whole source from a logline

1. `GET /projects`, `GET /project/load?name=NAME` — the project is `sources`
   (the bin) and `timeline.clips` (the edit).
2. Build the target segment's `prompt` dict from the user's description: a style
   contract, a contiguous storyboard with per-beat camera motion, the diegetic
   `audio` soundscape, and the `music` score (or leave empty for `N/A`). Use the
   user's stated duration (default to a comfortable length within 5–15s; H3
   rewards longer, multi-shot ranges — do not cram). A two-beat dialogue prompt
   looks like this — note `camera`/`transition`/`lines` per beat, and speaker
   keys that the server turns into `(S1)`/`(S2)`:

   ```json
   {
     "duration": 10,
     "style": "Live-action, cinematic neo-noir, warm tungsten light, film grain",
     "beats": [
       {"start": 0, "end": 5, "angle": "a warm two-shot across a diner booth",
        "action": "a weary detective sits opposite a fidgeting informant, rain on the window",
        "camera": "holds a static shot",
        "lines": [{"speakers": ["detective"], "describe": "the weary detective with a gravelly voice",
                   "verb": "asks", "delivery": "quietly", "text": "You saw who did it, didn't you?"}]},
       {"start": 5, "end": 10, "angle": "a close-up on the informant",
        "transition": "the camera cuts to", "action": "his eyes dart to the door",
        "camera": "pushes in with small amplitude at slow speed on his face",
        "lines": [{"speakers": ["informant"], "describe": "the young informant with a breathy voice",
                   "verb": "answers", "text": "I saw enough to be scared."}]}
     ],
     "audio": "a low diner hum, a coffee machine hissing, rain tapping the window",
     "music": "a sparse upright-bass line at a slow tempo with brushed drums"
   }
   ```
3. Validate:
   ```bash
   curl -s -X POST http://127.0.0.1:8188/video_prompter/preview \
     -H 'Content-Type: application/json' \
     -d '{"prompt": { ...your dict... }}'
   ```
   Read `findings`. Resolve every `error`; address warnings when cheap.
4. For storyboard/canvas issues, don't hand-fix arithmetic — use `/transform`
   with ops `fix` (close gaps/overlaps), `refit` (`{"op":"refit","duration":N}`),
   or `ratio` (`{"op":"ratio","ratio":"16:9"}`). It returns the corrected prompt.
5. Save the source (send the whole source object, its segments carrying the
   new `prompt`):
   ```bash
   curl -s -X POST http://127.0.0.1:8188/video_prompter/source/update \
     -H 'Content-Type: application/json' \
     -d '{"project":"NAME","source": { ...source with segments... }}'
   ```
   The open editor updates live.

### Targeted verbal tweak ("make the second clip handheld")

1. `GET /source?project=NAME&source=SOURCE_ID` — fetch one source, its segments,
   and depth 0's prompt.
2. Change only what was asked (e.g. set a beat's `camera` motion, rewrite one
   beat's `action`, edit a `line`'s `delivery`, add a `cue`). Leave everything
   else intact.
3. `/preview` to re-lint, then `/source/update` to save.

### "I dislike the second clip, redo it"

Regenerate that segment's `prompt` fresh from the user's intent, re-lint, save. If
they dislike a specific block, regenerate only that block.

### "Break shots 2 and 3 out into their own clip"

One generation asked to hold three cuts often deserves to be two. `POST
/source/split` copies the picked shots — and the scene around them — into a new
bin item that has generated nothing:

```bash
curl -s -X POST http://127.0.0.1:8188/video_prompter/source/split   -H "Content-Type: application/json"   -d '{"project":"NAME","source":"SRC","beats":[1,2]}'
```

`beats` are 0-based positions in the segment's storyboard — the same numbering a
`beat:N` rewrite scope uses — so "shots 2 and 3" is `[1, 2]`. Each picked shot
keeps its own length, laid end to end from zero, on a budget rounded to the
whole second; style, sound, cast, references and subjects come across whole, and
the keyframes are dropped unless the pick still opens or closes where they
pointed. Add `"segment"` for a continuation, `"name"` to name it yourself, and
`"clip": false` to leave it off the timeline (by default the clip lands beside
the one it came out of). The reply's `source` is the new id.

It is a copy: the original keeps its shots. Trimming them out of it is a
separate `/source/update`, and worth doing only once the split has rendered
something the user likes.

### Render them (do this yourself — do not hand it back)

After authoring and saving, render what changed (a new or edited source, or the
whole sequence for a fresh project) and report the outcome. The user has said
they do not want to press Render, so treat rendering as part of fulfilling any
"make me a video" request — this is standing authorization for the **local GPU
only**, never a hosted API. Only skip it if the prompt still has an error finding
you could not clear, or the user explicitly said "just draft it".

**Use the committed script. Do not write your own render loop** — it already
prepares, queues on ComfyUI, waits, records the take and picks it, sending
exactly what the editor's Render button sends:

```bash
python tools/render_sources.py --project NAME --source SOURCE_ID
```

Run it in the background (renders take minutes). It queues everything first so
the GPU stays busy, prints progress to stderr, and prints one JSON summary to
stdout at the end — per render: `status`, `seed`, `mode`, `quality`, `takes`,
`picked`, `render_seconds`, `error`. Exit status is 0 only if every render
succeeded.

| Want | Flags |
|---|---|
| one or more sources | `--source ID_OR_NAME` (repeatable) |
| the whole bin | `--all` |
| a seed sweep | `--count N` (1-8; takes are recorded, **not** picked) |
| continue a take | `--extend TAKE_ID` |
| re-render a take's frozen text | `--reshoot TAKE_ID --quality final` |
| check it would prepare, without the GPU | `--dry-run` |
| a specific tier or seed | `--quality TIER`, `--seed N` |

- **Profile.** `--profile NAME`, else the project's `render_profile`, else the
  only one there is. With several and none pinned the script stops and lists
  them: ask the user which, and offer to pin it on the project so the question
  is asked once, not every session.
- **Tier.** Defaults to each segment's own `quality`. In short:

  | tier | canvas / steps | per take | use it for |
  |---|---|---|---|
  | `preview` | 0.2 MP / 8 | ~1.5 min | **proving every prompt first** — H3 obeys best at a small canvas |
  | `turbo_hd` | 1.0 MP / 8 | ~4.5 min | seed sweeps once the prompt is settled |
  | `turbo_2pass` | 6 + 2 steps | ~turbo_hd | when a turbo draft misses direction `preview` lands |
  | `final` | 1.0 MP / 25 | ~20 min | the keeper |
  | `final_2pass` | 15 + 5 steps | ~final | a keeper with small detail that must survive — digits, signs, distant faces |
  | `singularity_2pass` | 20 + 4 steps | ~15-20 min | 1080p Singularity Ref2va keeper: 20 base steps @ 0.5 MP, 3D latent upscale, 4 Turbo steps @ 0.35 denoise |

  If a shot is ignored at 720p, drop it to `preview`: if it lands there, the
  prompt is fine and the resolution was the problem. Why the splits are what
  they are, reference sizing, and extend rules: `references/quality-tiers.md`.
- **Extend.** Author the continuation first — `POST /source/segment/add` with
  its `prompt` — and the script composes that segment into the continuation
  text. It refuses to continue into a blank plan rather than render nothing.
- **When it fails.** `refused … lint errors` is a 422: fix the findings and
  rerun. A 409 means the profile drifted from its workflow; tell the user to
  re-capture it. A render running far past its tier's time may be the known
  GPU stall (`references/quality-tiers.md`). **Do not free VRAM before
  rendering** — measured, it only makes renders slower.

Report which sources rendered, their seeds and take ids, and any that failed.

## What H3 ignores, and why

The format being right is not the same as the model obeying it. These are the
places a prompt composes perfectly and still comes back wrong. The linter now
catches most of them, but write so it never has to.

- **A cut must introduce new information** — a new subject, place, state,
  viewpoint or jump in time. If the only difference between two shots is how
  tightly the same thing is framed, H3 merges them, which reads afterwards as
  the model having *skipped* a shot. Reaching for a tighter framing of the same
  subject in the same place is a push in, not a cut. (`beat.weak_cut`)
- **One action arc and one primary camera move per *beat*.** Verbose does not
  mean more events. Spend the complexity budget detailing the physics of one
  beat — orientation, blocking, material, light, how the action progresses —
  rather than stacking four. Two camera moves in one breath compete and the
  model usually resolves them into whichever it read first.
  (`beat.camera_complex`) A **shot** may now hold several moves, because a
  no-cut beat gives the second one its own timecode, and a timecode is the
  ordering signal a stacked pair never had — that is the whole point of `cut:
  false`. What still fails is two moves in one field with nothing to sequence
  them.
- **A timestamp inside a shot works; a timestamp on an audio cue does not.**
  Tested at one seed on 2026-08-31: the same three moments written as one
  continuous take landed the first on its mark (3.00s directed, 3.00s
  rendered), and with the clock removed drifted to 3.85s. Contrast the `cues`
  note above, where timing was measured to do nothing at all. So: time your
  *actions and camera moves*, and write sound into the beat it belongs to.
- **Quality words are wasted words.** "4k", "masterpiece", "highly detailed",
  "cinematic quality", "don't look AI" are diffusion-image habits: they describe
  your reaction, not anything a lens can be pointed at. H3 conditions on
  described behaviour. The look belongs in the `style` contract, never sprinkled
  through the action. (`prompt.aesthetic`)
- **Behaviour beats intent.** "Her feet, hips, shoulders and head remain
  oriented in the same direction" works; "realistic, natural-looking" does not.
  Name anatomy and physics.
- **Words take time to say.** Roughly 150 words a minute at unhurried delivery.
  A beat holding more speech than its running time gets a raced delivery or a
  dropped tail — lengthen the beat, cut the line, or split it across the cut
  with `continuity`. (`dialogue.overrun`)
- **Two people in one shot get blended.** The composer now appends one sentence
  naming the co-present speakers and saying they must not merge or swap; leave
  `guard_identities` alone unless it reads as noise on a particular prompt.
- **Turning away from camera is a known stress case**, especially in ref2v where
  the identity images usually show a face. Cutting to a shot where the character
  is *already* facing away is far more reliable than asking for the rotation.
  If you must rotate: name each body part and where it ends up (hips, shoulders
  and head together), and describe the end state the lens sees rather than the
  verb — "turns around" is ambiguous.
- **Negatives are soft, with one exception.** Prose prohibitions get dropped;
  this is why `negatives` is not composed and why an empty `music` becomes the
  literal `N/A` rather than a sentence asking for silence. The exception is an
  orientation lock, which works *because* it is paired with a positive
  anatomical statement: "her feet, hips, shoulders and head remain oriented in
  the same direction; she does not track the camera."
- **Change one thing per retry.** When a render misses, change a single variable
  — one clause, one retention marker, the quality tier — so you learn which one
  mattered. A multi-edit retry teaches you nothing.

## Notes

- The video model is **local weights only** — you are the authoring brain, but
  never route generation through a hosted video API.
- H3 **burns subtitles into the frame** for any dialogue or narration (`lines`),
  so a talky shot renders with baked-in captions to crop in post. If the user
  wants no captions, keep that shot silent; and split long narration across
  scenes so a single line does not get clipped.
- Keep the composed prompt under the recommended character budget (the linter
  warns past it); later blocks lose weight when a prompt runs long.
- If `/preview` still shows an error you cannot resolve, tell the user what the
  finding says rather than saving a prompt that would waste a generation.
- **Every save files a revision** server-side, so a save never loses the old
  prompt; `/source/update` takes an optional `note` saying what it was for.
  When the user asked to change one part, change that part and leave the rest
  byte-identical. Restoring: `references/images-and-media.md`.
