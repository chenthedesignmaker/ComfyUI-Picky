# References, subjects and the cast

Read this when a segment carries `references`, `subjects` or `cast`, or
names somebody with `@Name` — i.e. whenever it renders as reference-to-video
(ref2v) or reuses a project identity. Plain t2v and i2v shots never need it.

## The reference fields

- `references` — omni-reference files `{kind, path, role, retention, member,
  slot}` that **drive** reference-to-video (ref2v): each file is fed to the
  matching node slot, not just described. `path` is a name in ComfyUI's input
  dir (see "Attach media"). Any reference makes the shot render as the
  **six-section full-reference document** (`subject_definitions` → `summary` →
  `retention_analysis` → `detailed_description` → soundscape → music):
  - `role` becomes the reference's own `subject_definitions` line and its
    `<Picture N>`/`<Video N>`/`<Audio N>` identity; weave that same tag into the
    beats where it appears. **But only for a file that stands on its own** — a
    keyframe, a source video, an audio track. A picture that merely *shows* a
    person, a place or a prop is cited inside that thing's `<Subject N>` and
    writes no line of its own; see `subjects` below.
  - `retention` is the preservation marker for `retention_analysis`. Use the
    **controlled vocabulary** (see `/schema` `reference_vocabulary`): visual refs
    take `fully_preserved` / `partially_preserved` / `attribute_transfer` /
    `weak_reference`; audio refs take `fully_copy` / `partially_copy` /
    `reference` / `weak_reference`. Empty uses the per-kind default.
  - `member` + `slot` make the entry a **link** to a cast member's identity file
    rather than a copy of it: set `member` to the cast id and `slot` to the
    0-based index into that member's own `references`, and leave `path` alone —
    `kind` and `path` then resolve from the member at compose time. Use it when
    one segment needs its own `role` or `retention` for a file the whole project
    shares: those two are read from the entry, everything else from the member,
    so swapping their sheet still updates this segment. A linked entry also puts
    the member on screen, so do not also list them in `cast`.
- `subjects` — the `<Subject N>` layer: what the target video **contains**, as
  opposed to which files it was given. This is the distinction H3 actually
  tracks, and the one most easily got wrong: a character and their room are
  subjects; the pictures of them are only sources. The relationship is
  many-to-many — one subject can be cut from several files, one file can supply
  several subjects — which is why it is its own list.
  - Entries are `{role, retention, sources, noun, member, label}`. `role` is
    the prose after `<Subject N> is`; `noun` is the word the citation hangs on
    (`person`, `setting`, `object`, `speaker`, or empty); `sources` are
    `references[].path` values.
  - Composes as `<Subject 1> is the person in <Picture 1> and <Picture 2>, a
    woman in her thirties, dark hair.` A cited **visual** source loses its own
    `<Picture N>` line. A cited **audio** source keeps its `<Audio N>` line and
    gains a back-link: `<Audio 1> is the voice-timbre reference for <Subject 1>
    (S1).` — reusing the speaker id the dialogue already carries.
  - `retention` is the preservation marker, same visual vocabulary as a
    reference's, and it replaces the per-file markers in `retention_analysis`:
    what is preserved is the person, not the PNG.
  - **A cast member on screen who has identity files becomes a subject on its
    own** at compose time, cited from all of their files. Do not write one for
    them. To change only what *this* segment says about them, add an entry with
    `member` set (the cast id) and no `sources`: its `role`/`retention` win, and
    everything else still follows the member. Write a plain entry (no `member`)
    only for something with no cast identity behind it — a style, an action, a
    one-off prop.
  - `label` is for the editor and is never composed. Names never reach the
    prompt; the tag is the identity.
- **What full-reference mode writes for you** — do not hand-write these:
  - A speaking cast member composes as `<Subject 1> (S1), <appearance>, says:`
    on their first line and `<Subject 1> (S1) says:` after it. The guide (5.4)
    wants both labels — the subject says which referenced identity, the `(Sx)`
    which voice.
  - `retention_analysis` states where each item lands:
    `<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...`.
    A member who speaks is placed by their lines; one listed in `cast` is
    present for the whole segment; anything else is placed by the shots whose
    prose names its tag. An item that cannot be placed gets no parenthetical
    rather than a guessed one.
  - A `video editing` task opens the summary with "The target video is an
    edited version of `<Video 1>`." unless you have written it yourself.
  - Dialogue inside `<d>` gets a closing `.` when it has no punctuation at all
    — **except** a line marked `cutoff` or `continuity: "into"`, which the
    format says is deliberately unfinished.
- **What it cannot write for you**, and what the linter now says about it:
  - Weave every `<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>` into
    the shot where it first appears (`subjects.unused`, `references.unused`). A
    speaking member is the one exception — their label is written in for you.
  - Never put an `(Sx)` in a subject's or a picture's `role`: the role composes
    into `retention_analysis`, and the format forbids speaker ids there
    (`ref.retention_speaker_id`). An *audio* role may carry one, because guide
    2.4 asks for it in the definition.
  - Only use labels in `ref_summary` that something defines
    (`ref.summary_undefined_tag`).
  - Aim for a 350-500 word `detailed_description` on a generation task
    (`ref.description_length`, informational). Full-reference mode spends its
    budget on per-shot detail — composition, lighting, state changes, and where
    each reference actually takes effect — not on a plot summary.
- `ref_task` / `ref_summary` — ref2v only. `ref_task` is the list of task types
  for the summary's `[...]` prefix (from `reference_vocabulary.task_types`, e.g.
  `["reference generation", "audio reference"]`); `ref_summary` is the one
  paragraph after it, using the `<Tag N>` labels. Leaving `ref_task` empty
  infers a sensible prefix. **Always write `ref_summary`** on a ref2v shot:
  empty, `summary` still composes, to nothing but "The target video is generated
  using <Subject 1>, <Subject 2>." — the section that states the *task* says
  only that things exist. `subject_definitions` and `retention_analysis` are
  still full, so the document reads finished while the model is left to invent
  what to do with the references. `ref.summary_missing` warns about exactly
  this. Say which identity to hold, which tag is a set rather than a framing,
  and which tags are the same subject seen twice.

### The cast

A project keeps a list of reusable identities — characters, locations, props,
voices — and segments point at one by id rather than describing it again. This
is how source nine keeps looking like source one, so **check the cast before you
invent anybody**.

```bash
curl -s "http://127.0.0.1:8188/video_prompter/cast?project=NAME"
```

A member is `{id, name, kind, appearance, voice, references, notes}`. The
`appearance` is the load-bearing part: the canonical description every segment
inherits. `voice` is timbre and register, and joins the appearance to form the
identifying phrase a first vocal appearance needs. `references` are identity
files that get fed to the reference node whenever the member is on screen — and
on a ref2v shot the member resolves to **one `<Subject N>` citing all of them**,
not one `<Picture N>` line per file. Three sheets of the same face are one
person, and saying it three times is how a model starts treating them as three.

Write one with `POST /cast/save` (omit `id` to mint one):

```bash
curl -s -X POST http://127.0.0.1:8188/video_prompter/cast/save \
  -H 'Content-Type: application/json' \
  -d '{"project":"NAME","member":{"name":"Mara","kind":"character",
       "appearance":"a tall woman in a rain-darkened red coat",
       "voice":"a low, unhurried voice"}}'
```

Then put the returned id where the person goes:

- **They speak** — the id is a key in that line's `speakers`, exactly like a
  hand-written key. Leave `describe` empty and the member's identifying phrase
  fills it on their first line; write your own `describe` and yours wins, which
  is how you say "but in this shot she is soaked" without editing the member.
- **They are on screen and silent** — the id goes in the prompt's `cast` list.
  Their appearance is stated in the opening. Never list a speaker here too.
  Add the id to `undescribed` as well to keep their identity files *without*
  that opening description — the case that matters is a location in a close
  shot, whose appearance is a paragraph of window walls and doorways that asks
  the camera to go and show them.
- **One segment needs its own wording** — add an entry to `subjects` with
  `member` set to the cast id and nothing else but the `role` and/or
  `retention` you want here. That is how you say "but in this shot she is
  soaked" for the subject line, the way `describe` does it for a spoken line.
  A member on screen already sends every identity file they carry, so this adds
  no file; it only takes over what the document says about them.
- **One segment needs its own retention on a single *file*** — add a linked
  entry to `references` (`member` + `slot`, see above). Rarely what you want
  now: a file a subject is cut from has no line of its own to mark, so reach
  for this only for a member's file that stands alone.

Three things follow from resolution being live, at compose time:

- Editing a member **changes every segment that names them**, including ones you
  are not looking at. That is the point, and it is also the risk: a reworded
  appearance re-composes nine prompts. Say so when you do it.
- Rendered takes do not move. A take freezes its prompt text, so a cast edit
  reaches future renders only. `/render/reshoot` replays the frozen text.
- Deleting a member leaves its id on the prompts that used it, and they start
  raising `cast.unknown`. `GET /cast/usage?project=NAME&member=ID` says what a
  delete would strand — check it before deleting anything.

**Names never reach the prompt.** `name` is for the author and the editor; the
composed text refers to people as `(S1)` and `<Subject 1>`, because the tag is
the routing identity H3 tracks.

#### Point at a member with `@Name`, not with a typed tag

Do not type `<Subject 2>` into a beat's action. The number is *positional* — it
is the member's place in this segment's resolved subject list — so it moves the
moment the shot gains or loses somebody, and a number you typed does not move
with it. Write the mention instead and the composer fills the number in:

```json
{"action": "@Mara sets the mug down as @Ivo looks away."}
```

composes to `<Subject 1> sets the mug down as <Subject 2> looks away.` — and to
`<Subject 2> ... <Subject 3> ...` the moment a location joins the shot ahead of
them, with nothing to re-edit.

- **What it becomes.** The member's `<Subject N>` when they have identity files;
  `the <noun> (S1)` when they only speak; `the <noun>` when they have neither,
  which raises `mentions.untagged` to say why it reads that vaguely.
- **Mentioning somebody puts them on screen.** It has to: the label only exists
  because they resolved into this segment. So `@Mara` alone brings her identity
  files and her description, exactly as listing her in `cast` would — do not do
  both.
- **Matched by name**, whole and case-insensitively, longest name first, so a
  name may hold spaces (`@Mara Chen` beats `@Mara`). An `@` with a word
  character in front of it is an address, not a mention. A name nobody answers
  to composes literally and raises `mentions.unknown`.
- **Where it works**: `style`, `overview`, `camera`, `audio`, `music`,
  `ref_summary`, `beats[].angle` / `action` / `camera`,
  `beats[].lines[].describe` / `delivery`, and the `role` on a subject or a
  reference. **Not** in `beats[].lines[].text` or `screen_text` — a character
  saying somebody's name out loud, or a name painted on a sign, stays a name.
- **Renaming a member rewrites its mentions** across every segment of the
  project, since the name is the link. `POST /cast/save` reports how many fields
  moved as `renamed_mentions`.

Old prompts with hand-written speaker keys (`["baker"]`) keep working untouched:
a key is only held to the cast when it is shaped like a minted id, so nothing
that predates this warns. When you are writing a character who will appear more
than once, though, make a member instead of a bare key.
