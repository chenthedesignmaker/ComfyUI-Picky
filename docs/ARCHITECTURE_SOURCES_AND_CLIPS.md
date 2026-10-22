# Sources and Clips

Decided 2026-08-19, built 2026-08-19/20. **Phases 0 through 4 and 6 are done and
the tool works end to end.** Only Phase 5's tail is open, and most of that fell
out of Phase 2.

## Picking this up

Branch: **`lineage-sources-and-clips`** (off `main`). Three commits carry the
work: `0127e58` the switchover, `09f9c16` the editor, and the fold (Phase 6).

Read this section and *The decision* below; the phase log after them is history
and is only worth opening when you want to know why something is the shape it
is.

### Do this first

**Restart ComfyUI before opening the editor at :8188.** The Python routes
changed. A running ComfyUI holds the module it imported at start-up, so
`/source/*` and `/clip/*` 404 until it restarts — and because the bin is the
first thing the editor loads, *everything* looks broken. JS and CSS are still
live on refresh. This is by far the most likely thing to look like a bug and not
be one.

### What the model is now

`Shot` did three unrelated jobs at once and is gone. Three types replace it:

- **Source** (`sources.py`) — a bin item: one lineage. Holds `segments` and one
  ordered `path` of take ids.
- **Segment** (`sources.py`) — one generation link: a prompt and a quality, at
  depth 0, 1, 2 … A plain un-extended source is one segment.
- **Clip** (`sequence.py`) — one use of a source on the timeline. In and out
  points are seconds into **lineage time**, not into any one file.
- **Sound** (`sequence.py`) — what a clip is *heard* as, as a field on the clip
  rather than a track of its own. Left alone it is the clip's own generation;
  it can be muted, or pointed at a take whose picture was rejected. Because a
  clip addresses lineage seconds, a re-roll of the same shot is the same clock,
  so borrowing needs a take id and never a second position to keep aligned.

Two rules carry most of the design, and breaking either is how this went wrong
the first time:

1. **`path` is the single answer to "which take follows which."** It is sound
   when each take's recorded `parent_take` is the take before it. Nothing is
   inferred from names, ordering or position — and when the check fails the UI
   says the link is broken rather than substituting something plausible.
2. **The plan sets the depth; the path only decides which take fills each
   level.** A *default* path stops at the number of segments a source plans.
   Following a chain further would hand one source seconds belonging to the
   next, and the edit would play them twice. An explicit choice may run deeper,
   because somebody said so.

`edit.py` is the one place a source, a clip and the take record meet. Every
length, branch list and resolved playlist comes from there, so the timeline, the
linter and the export cannot disagree.

### What is left

- **Phase 5's tail** — an EDL question, not a phase. See Phase 5 below.
- `sweep_results/` is untracked and predates this branch. Not part of this work;
  deliberately left alone.

Merging siblings — the one piece this plan carried as unbuilt to the end — is
done, and it is not the author action the plan expected. See Phase 6.

### Running it

```
python -m pytest tests/ --basetemp=<a fresh temp dir>
```

911 tests. The `--basetemp` matters on this machine: a stale locked
`pytest-of-fried` directory raises `PermissionError` that looks like a test
failure and is not.

The editor is served by the real ComfyUI at **:8188** (`custom_nodes` is
symlinked to this repo). **JS and CSS are live on refresh; Python changes need a
ComfyUI restart.**

`tools/devserver.py --port 8899` runs the same routes with no GPU, and
`.claude/launch.json` has an `editor` entry for it. Use it when you need to
restart the server freely — the user's ComfyUI may have a render in flight. Its
home is `tools/.devhome/` (gitignored), holding copies of real projects:

- **THE PAPER AVIATOR** — a clean ten-link chain, ten sources as migration
  leaves it, **one source of ten generations the moment it is opened** (Phase
  6). Reads **43.417s** either way, which is the known-true number; if it reads
  51.66s the take files are out of reach and the recorded durations are being
  used instead.
- **PLANET OFFICE DRAFT** — 18 sources, 300+ takes, real 4-way forks, one
  genuinely stale pick that *should* be reported as broken. Opening it folds it
  to 10 sources; the stale row, *Romance B*, is one of the ones left alone.

Both copies on disk are the *pre-fold* state, so they still exercise it. Opening
either rewrites it — restore from `.backups/` to run the fold again.

The devserver has no `/ws` and no favicon, so a failed WebSocket and one 404 are
expected there and mean nothing.

### Traps already hit, worth not re-hitting

- `tools/devserver.py` imports the route module at start-up, so **Python edits
  need the devserver restarted too**, exactly like ComfyUI. An API change that
  appears to have no effect is usually this.
- **`where.endsWith(name)` with an empty name matches every finding.** One
  project on disk has an unnamed shot, and it silently claimed every error in
  the project. Drop blank names before matching.
- A grid/flex item defaults to `min-width: auto` and will not shrink below its
  content. The lineage rows have fixed-width cells, so every container between
  the sheet and the track needs `min-width: 0`; one missing declaration looked
  fine at 1280px and pushed the takes column 444px past the sheet at 900px.
- Verify against the right control before concluding a feature is broken. A
  browser check here selected the adopt button with `title*="whole chain"`,
  which matches *"Play this whole chain"* first — so the test clicked play,
  nothing was chosen, and the unchanged state read as a bug that was never
  there.
- Screenshots need the Browser pane actually displayed. When it is not,
  `read_page` and `javascript_tool` verify structure perfectly well and are what
  the Phase 3 checks ran on.
- The editor's own state is module-scoped, so `state` is not reachable from the
  console. Drive the real UI (`click()`, `dispatchEvent`) and read the DOM back;
  that tests the wiring rather than the internals anyway.
- **A `parent_take` is not always a take id.** Three shapes reach
  `gallery.list`: a real id, `file:NAME` written by the MP4 recovery, and
  `:NAME` — an id with an empty prompt_id — written by some older renders. The
  last two are resolved by filename in memory, and only when that filename is
  unique in the project. Reading `parent_take` straight out of `takes.json` and
  expecting an id is how two real chains stayed invisible.
- The devserver's editor is at **`/video_prompter/app`**, not at `/`. The root
  is a plain 404 and looks like the server failed to start.

## Why

Motion Context extends made chains a first-class thing, but the model never
grew a place to put them. Parent/child mismatches kept appearing. They were not
bugs in the linking code — they were the predictable result of the same fact
being stored in three places and *guessed* in a fourth.

Today "which take follows which" lives in:

1. `take.parent_take` — the exact upstream take id, written by the renderer at
   render time. **This one cannot be wrong.**
2. `shot.pick` — one chosen take per shot.
3. `shot.branch_picks` — a map of `parent_take_id -> child_take_id`.

And when those disagree, the editor guesses:

- `effective_continuation_of` (`api.py`) infers a parent shot from *shot names*
  — "part 1"/"part 2", "shot A"/"shot B".
- `get_child_takes_for_parent_take` (`api.py`) falls back to pairing takes by
  **chronological position**: the 3rd take of the parent pairs with the 3rd take
  of the child.

Neither can say "I don't know". They always return something, so a wrong pairing
is silent.

There is a second, structural conflict. Every extend today lands as a take on the
**same shot** as its parent — every `runExtend` call site passes
`targetShot = null`, so `destShot = shot`. A chain therefore physically exists as
a pile of sibling takes inside one shot. But `continuation_of` and `branch_picks`
model a chain as a *run of separate shots*. Two incompatible shapes for one
thing. `cascade()` (Python) and `cascadeClientPicks()` (JS) then try to keep the
picks in sync — the same logic written twice, in two languages, which drifts.

## The decision

`Shot` does three unrelated jobs. Split it into three types.

| Job | Fields today | Becomes |
|---|---|---|
| What to generate | `prompt`, `quality` | **Segment** |
| Which take won | `pick`, `branch_picks` | **Source** (one `path`) |
| How the edit uses it | `source_in/out`, `speed`, `enabled` | **Clip** |
| What it is heard as | `audio` (mute, gain, borrowed take) | **Clip** |

A source is a *bin item*: one lineage of generations. A clip is a *timeline
instance* pointing into a source. The same source can appear on the timeline
many times with different trims. This is the NLE split, and it is what the tool
already wanted to be.

### Vocabulary (fixed — use these names in code)

- **Source** — a bin item: one lineage. Holds segments, branches, active path.
- **Clip** — one timeline instance referencing a range of a source.
- **Sound** — a clip's audio: its own, muted, or borrowed from another take.
- **Segment** — one generation link in a lineage (depth 0, 1, 2 ...).
- **Take** — one rendered file.
- **Branch** — an alternative path through the lineage.

`Shot` retires. It is the ambiguous word that caused this.

### Shape

```
Project
  sources:  [Source]   # the bin — what you generated
  timeline: [Clip]     # the edit — ordered, cuts-only (unchanged)

Source
  id, name
  segments: [Segment]                     # depth 0,1,2… one per generation link
  path:     [take_id, …]                  # the active lineage
  remembered: {parent_take: child_take}   # advisory only, never authoritative
  legacy_continuation_of                  # a v1 guess, carried, never read

Segment
  id, prompt, quality, notes              # generation intent only

Clip
  id, source_id
  path_override: [take_id…] | null        # null = follow the source (default)
  source_in, source_out, speed, enabled, label
```

**A plain single shot is a Source with one segment.** No "is this a group or a
shot?" branch anywhere. That special case simply does not exist.

### Path replaces pick

`pick` + `branch_picks` collapse into one ordered list:

```
path = ["t_root", "t_child_b", "t_grand_a"]
```

One value, one place. Switching a child at depth 2 means truncating the list to
length 2 and re-extending downward. The parent's pick *is* `path[n-1]`, so parent
and child cannot disagree.

A segment's depth is its **position in the list**, not a stored number: a stored
one could disagree with the list it sits in, which is the class of bug this whole
rewrite is about.

Validation is one line:

```
valid  <=>  take(path[i]).parent_take == path[i-1]   for all i > 0
```

That single check retires every heuristic. When it fails the UI says "this link
is broken" instead of quietly substituting something.

Note the `i > 0`. A source's path is **not** required to start at a root, and
learning that cost an afternoon: every chain made before sources existed is
spread across sibling shots, so after migration the second shot's path
legitimately begins one link down. Eight sources in PLANET OFFICE DRAFT read as
broken until `lineage.validate` grew `must_start_at_root=False` for this case.
A *whole lineage* still has to start at a root; a source's own path does not.

`branch_picks` survives as `remembered`, demoted to advisory: it only rebuilds
the tail of a path when you switch branches. Stale data costs you a mediocre
default, never a broken timeline.

## Source time and the one function

A source has its own continuous clock: segments concatenated, ~5.2s each. A
3-link source is a ~15.5s source, and a clip may trim anywhere into it —
including across a join.

Everything runs through one function:

```
resolve(source, path, in, out) -> [{file, from, to}, …]
```

Give it a range in lineage-time, get back real files with in/out points.

| Feature | Call |
|---|---|
| Play the timeline | resolve each clip, concatenate |
| Play the whole lineage | `resolve(path, 0, end)` |
| Play only this segment | `resolve(path, seg_start, seg_end)` |
| Compare two branches | resolve twice with different paths |
| Export | resolve, then concat |

Trim points are plain seconds into lineage time. Sibling branches at the same
depth are normally the same length, so a branch switch does not slide content in
practice — but this is a convention, not a guarantee: a root generation is 124
frames while its continuations are 102, and 272-frame takes exist. Segment-
relative anchoring is deliberately not built; if branch switching is ever seen to
slide a trim, that is the fix to reach for.

## Decided behaviours

- **Flipping a source's branch updates its timeline clips.** Clips follow by
  default (`path_override: null`). A "pin this branch" action freezes one clip.
- **Dead branches are preserved.** Nothing is auto-archived. Off-path takes stay
  visible and playable so branches can be compared, and are deleted only by hand.
- **The bin panel ships in v1.** The existing shot-list panel becomes it.

## What does not change

The renderer and the extend backend. `extend.py` and `/render/extend` already
record `parent_take` correctly — they have been the reliable part all along.
"Extend" just gains a precise meaning: *append a segment at depth N+1 from this
take*. This is an editor and data-model refactor.

## Plan

### Phase 0 — resolve() and a playlist-driven player — **DONE**
`lineage.py` builds the forest from `parent_take` alone and resolves a range of
lineage time into playable pieces. `GET /lineage` and `POST /lineage/resolve`
expose it. The monitor now plays any playlist: `mode: "timeline"` drives the
playhead and follows the selection, `mode: "preview"` does neither, so reviewing
a branch leaves the edit untouched. Take cards gained *play this segment* and
*play the whole lineage*.

The take sheet covers the timeline, so a preview launched from a take card was
playing in the program monitor *behind* the sheet — running, but not watchable.
The monitor now **relocates**: `dockPreview` moves the `.stage` element into
`#preview-dock` at the top of the takes column for the duration of a preview and
puts it back on stop, on sheet close, or when the transport reclaims it. Moving
the one player beats standing up a second one, which would have meant two copies
of the buffering and two ways a join could behave. The move only ever happens
while the buffers are paused — re-inserting a playing `<video>` is what makes it
stall — so callers dock first and play second.

Two defects had to be fixed first, both of which the rest of the plan rests on.

**Durations were the length that was asked for, not delivered.** The extend graph
trims the 22 pinned context frames before saving (`MiniMaxH3MotionContextTrim`),
so a 124-frame request writes a 102-frame file — 0.917s shorter. `takes.json`
recorded the request. 87 of 540 takes across the projects on disk were wrong, and
every clip width, running total and trim default downstream of an extend
inherited the error: one 10-link chain read 51.66s when it is really 43.417s.
`gallery.measured_seconds` now reads the container's own `mvhd` header (no
subprocess, no decode) and corrects `seconds`/`frames` in memory at load, marking
the take `duration_source: "measured"`.

**Most lineage was never recorded.** `parent_take` only entered queue metadata
partway through this tool's life, and the ComfyUI history that could backfill it
is cleared long before a project is finished — of ten projects with extends, only
one had any parents recorded at all. The MP4 keeps the answer permanently: an
extend loads its parent through `LoadVideo`, and the whole graph is in the
container's `prompt` tag. `Gallery._recover_lineage` reads it. THE PAPER AVIATOR
went from ten apparently unrelated takes to one clean ten-link chain.

Neither repair writes to disk. `takes.json` stays exactly as it was, the way
prompt recovery already worked, so nothing is lost if the reading is ever wrong.

### Phase 0.5 — the lineage browser — **DONE**
Built ahead of the model deliberately: the columns are what prove the single
`path` is the right shape, and changing a UI is cheap where changing a shipped
schema plus migrated project files is not.

The take sheet gained a **Takes / Lineage** switch.

Lineage was first built as columns-per-generation, Finder-style. Real data killed
that layout immediately: a single shot in PLANET OFFICE DRAFT has **19 families,
18 of them exactly two generations**, and across the whole project 214 families
run no deeper than two. Columns put the long axis (families) vertically inside
one column and the short axis (generations) horizontally — so the first column
was a 2745px scroll of unrelated roots, and selecting one near the bottom drew
its only child at the *top* of the next column, a screen away.

**Families are the long axis, so they get the vertical scroll.** The layout is
one row per family with generations running left to right, which puts a take and
its own continuation permanently side by side. A deep chain (PAPER AVIATOR, ten
generations) is one 100px row whose track scrolls sideways, so no family can
stretch the others.

Alternatives at a step are pills under the cell rather than a column of their
own — about one family in twenty forks, so they do not deserve an axis. Each
family owns its own chosen path (`state.lineagePaths`, keyed by root take id), so
switching a branch in one leaves the rest alone.

The generation-columns idea is worth remembering if lineages ever get deep, but
it is the wrong default for how this tool is actually used.

**Ordering.** Most chains are settled — one take, one continuation, nothing to
decide — so they are reference, not work. Families sort by unmade choices first
(any node with more than one child), then by best rating in the family, then
newest. Two sticky headings, *Needs a choice* and *Single path*, split the list
so the ordering explains itself instead of looking arbitrary. On the sample shot
that puts the one branching family at the top and files the other eighteen
below it.

**Picking works from here too**, which turned out not to need the Source model:
every take already records the shot it belongs to, so a cell can set that shot's
`pick` directly. Each cell has *use this take*; each family header has *use this
whole chain*, which picks every take on the path, applied root first so the
existing cascade's guesses about downstream shots are overwritten by the explicit
choice rather than the reverse. One reload runs at the end instead of one per
take. Picked cells carry an accent border and a tick; a chain whose every
generation is picked marks its row as the current edit.

The one case that cannot work is a chain whose generations share a single shot —
a shot stands for exactly one take. That is precisely what the Source model
fixes, and until then `pickChain` says so rather than silently dropping takes.

Verified against PLANET OFFICE DRAFT: switching a four-way fork to its second
branch and adopting the chain stored both picks, and the row marked itself as the
current edit. Worth knowing that the root-first ordering was checked rather than
assumed — running the same two picks through the API with the cascade on and off
gives the same result, so no flag to suppress it is needed.

### Phase 1 — the model and the importer — **DONE**
`sources.py` holds `Source` and `Segment`; `sequence.py` gains `Clip` and
`Timeline` alongside the v1 `Shot`/`Sequence` they will replace. `migrate.py`
turns a v1 project dict into a v2 one. `export.Clip` was renamed `ResolvedClip`
first, so the new `Clip` had a free name.

A clip's in and out points are seconds into **lineage time**, not into any one
file, which is what lets a clip start inside generation two and run across the
join into generation three without the edit knowing there was a join.
`Timeline.split` got simpler in the process: both halves point at the same
source, so there is no prompt to deep-copy and no pick to keep in step.

The importer is deliberately dull — one shot becomes one source with one segment
plus one clip. Verified against every real project on disk: **24 projects, 171
shots, all 170 existing shot ids preserved** (the one shot without an id predates
shot ids and gets a minted one), and every trim, speed, enabled flag, pick and
quality carried across unchanged. Merging sibling shots into one multi-segment
source is *not* done here: it changes the edit, and choosing which shots to merge
is exactly the guessing this rewrite exists to delete. It becomes an explicit
action taken on a source with its lineage on screen.

**Not yet done, and the reason the schema version is still 1:** `Project` does
not hold `sources`/`timeline` yet, so nothing writes v2. Bumping the header
before the store actually writes that shape would just make the file lie. The
switchover is Phase 2's first job — 47 call sites, 28 of them in `api.py`.

### Phase 2 — the switchover and the server — **DONE**
`Project` holds `sources: list[Source]` and `timeline: Timeline`. `from_dict`
runs `migrate.to_v2` on the way in, so *every* route into a project — a load, a
restored backup, a dict an agent posts — lands on the same shape, and `to_dict`
writes v2. `SCHEMA_VERSION` is 2 now that something actually writes that shape.

**`edit.py` is the new join.** A source knows what it planned and which path it
chose; a clip knows which slice it uses; neither knows how long anything is,
because only the takes know that. `edit.read` indexes a project's takes once and
answers every length and branch question from there, so the timeline, the linter
and the export cannot disagree. `export.resolve` walks clips through
`lineage.resolve`, which is Phase 5 falling out as predicted: a clip that crosses
a generation join now exports as two pieces back to back, and the export goes
through the same function the monitor plays through.

**A length is measured where takes exist and planned beyond them** — the path's
real durations, plus the prompt budget of every segment past the end of the path.
An unrendered source is all plan, which is what a v1 shot's fallback to its
prompt length bought and what lets a sequence be laid out before a frame is
rendered; a finished one is all measurement. Planning a third generation widens
its clip immediately rather than after the render.

**Phase 4 fell out here too**, because it had to: `effective_continuation_of`,
`get_child_takes_for_parent_take`, `cascade()` and `Shot.continuation_of` all
read fields that no longer exist. `pick_take` survives as a convenience over
`set_source_path` — it walks up from the take through each take's recorded
parent, then carries the path on down the chain preferring what was remembered.
There is no cascade because there is nothing left to realign: the parent's choice
*is* `path[n-1]`. Only `cascadeClientPicks()` is left, in the JS, and Phase 3
deletes it.

`Shot`, `Sequence` and `blank_shot` are gone from `sequence.py`. `blank_source`
in `sources.py` replaces the last of them.

The routes, for Phase 3 to call:

| Bin | Timeline |
|---|---|
| `GET /source?project&source` | `POST /clip/add` `{project, source, at?}` |
| `POST /source/add` `{project, name?, at?, duration?, clip?}` | `POST /clip/delete` `{project, clip}` |
| `POST /source/update` `{project, source}` | `POST /clip/move` `{project, clip, to}` |
| `POST /source/delete` `{project, source}` | `POST /clip/edit` `{project, clip, …, pin?}` |
| `POST /source/duplicate` `{project, source}` | `POST /clip/split` `{project, clip, at}` |
| `POST /source/path` `{project, source, path}` | `POST /clip/duplicate` `{project, clip}` |
| `POST /source/segment/add` `{project, source, prompt?}` | |

`GET /sequence` now returns `duration`, `clips` (id, source_id, name, enabled,
`starts_at`, `source_seconds`, `source_in`/`source_out`, speed,
`timeline_seconds`, `path`, `pinned`, `rendered`), `sources` (name, `path`,
`chosen`, `seconds`, `take_count`, `problems`, `used_by`, `segments`, and
`branches` — what is on offer at each depth given the path chosen above it), and
`findings`. `POST /source/path` refuses a path whose links do not hold up with a
409 carrying `problems`, so a bad chain is caught at the write, not at export.

**A source's path is not required to start at a root.** That rule was in
`lineage.validate` and it was wrong for a source: every chain made before sources
existed is spread across sibling shots, so after the migration the second shot's
path legitimately begins one link down. Eight sources in PLANET OFFICE DRAFT read
as broken until `validate` grew `must_start_at_root=False` for this case. The
link rule — each take's recorded parent is the take before it — still holds
everywhere.

Verified read-only against every project on disk: **28 projects, 207 sources,
207 clips**, every shot id preserved, every v1 pick landing as `path[0]`, and one
genuinely stale pick correctly reported as pointing at a take that is gone. With
the real output directory in reach so durations measure, THE PAPER AVIATOR's ten
links total **43.417s** — the number Phase 0 established as the truth, arrived at
independently through the new lengths.

**`take.shot` keeps its name.** It holds the source id (the importer deliberately
kept shot ids), and renaming the field would orphan every `takes.json` on disk to
save one word. The node's `shot` widget keeps its name for the same reason:
it is a saved value inside every workflow anyone has built.

### Phase 3 — the editor and the skill — **DONE**
The shot-list panel is the **Bin**: one row per source, showing its lineage
length, its take count, how many clips use it, and whether anything uses it at
all. A source nothing uses is dimmed, not hidden — the point of a bin is that it
holds what the edit does not.

The timeline renders **clips**. A clip that spans a generation join draws the
join as a dashed seam, because that is a normal thing here rather than damage,
and the seams are snap points for the playhead like any other cut. A pinned clip
carries a dot and an outline.

**The Finder-style columns became segment tabs.** The plan called for columns by
depth; what the editor actually needed was one tab per generation above the
author form, because the form only ever edits *one* segment and the columns
would have duplicated the take grid below them. The invariant the columns were
for is kept exactly: `candidatesAtDepth` offers only the children of the take
the path stands on at the depth above, so a take from a branch you are not on is
never drawn, and the mismatch cannot be displayed. A source that has never been
extended shows no tabs at all — just a *Plan a continuation* button, so the
chain model stays out of the way until there is a chain.

The take card's **Use this** sets the whole path that ends at that take. The
lineage browser's **adopt** sets a whole chain in one write and grows the plan to
match. That is the case Phase 0.5 had to refuse — "a shot stands for exactly one
take" — and it now simply works.

`cascadeClientPicks` is gone, which was the last of the guessing: the JS half of
a cascade written twice in two languages. `state.selected` split into
`selectedClip` and `selectedSource`, because a bin item with nothing on the
timeline is a real thing to be looking at.

`GET /sequence` also carries the **playlist** now: the whole edit resolved into
playable pieces, server side. The monitor plays that list directly rather than
rebuilding it from clips and takes, so it cuts at exactly the frame the export
does. Doing that arithmetic in the browser again would have been a second
opinion about where a join falls.

The `video-prompt` skill was updated with it, and gained a short section naming
the three types — including the one place "shot" survives, which is H3's own
`[Shot N]` beats inside a single generation.

**Three real defects, found by running it rather than by reading it:**

- **A source with a blank name claimed every error in the project.** Findings
  are matched by `where.endsWith(name)`, and `endsWith("")` is true of every
  string there is. One project on disk has an unnamed shot, so the bug was
  visible the moment a real project loaded. Blank names are dropped before the
  match now, and an unnamed source draws as *Untitled*.
- **Pinning a clip silently did nothing** when its source was running on a
  default path: `pin` froze `source.path`, which is empty until somebody
  chooses, and an empty override reads as "follow". It pins the *resolved* path
  now, and refuses with a 409 when there is genuinely nothing to pin.
- **The default path ran the whole chain even when the source planned one
  generation**, which for a chain spread across sibling sources handed one
  source seconds belonging to the next — and the edit would have played them
  twice. The default stops at the planned depth now. The plan sets the depth;
  the path only decides which take fills each level. An explicit choice may
  still run deeper, because somebody said so.

The pre-existing `js/standalone/main.js` call to `/video_prompter/project?name=…`
— a route that never existed, 404ing silently behind a `.catch` on every project
open — is fixed in passing: it calls `/project/load`.

### Phase 4 — delete the guessing — **DONE**
`effective_continuation_of`, `get_child_takes_for_parent_take`, `cascade()` and
`Shot.continuation_of` went in Phase 2, because they read fields that no longer
exist. `cascadeClientPicks()` went in Phase 3 with the editor. Nothing infers a
parent from a name, and nothing pairs takes by chronological position, anywhere.

Keeping this as its own phase worked exactly as intended even though nothing was
left to do by the time it came round: naming it up front is what stopped the old
functions being quietly kept "just in case" during the switchover.

### Phase 5 — export through resolve() — **MOSTLY DONE**
`export.py` walks clips and resolves; a clip crossing a join exports as two
pieces, back to back, through the same function the monitor plays through. EDL
source timecodes come from the resolved take, as before.

What is left is EDL detail rather than a phase: an event per *piece* is correct
for ffmpeg but arguably wrong for a round trip into Resolve, where one clip that
crosses a join might be better carried as one event against a merged source.
Nothing has needed it yet.

### Phase 6 — folding the siblings back together — **DONE**
`fold.py`, run from `Api.load_project`. Every phase above assumed merging would
be an explicit author action on a source with its lineage on screen, and the
first design here was exactly that: multi-select in the bin, then a Merge
button. **That design was wrong, and saying why is the point of this phase.**

Asked to hand-pick the rows, the author's answer was: *in the bin those look
like unrelated segments — why do they need to merge?* Which is correct. The
sibling rows are not a thing anyone authored. They are a scar from the days
before the Source model, when every extend was its own shot. A chain is *already*
one bin item today, and the only rows in this state are old ones. Asking the
author to identify them is asking them to reverse-engineer a bookkeeping
accident.

And the app does not need to ask. **`take.parent_take` already says it.** When
one row's path ends exactly where the next row's path begins, they are one
lineage — a fact from the renderer's own record, in the same class as everything
else this rewrite trusts. The reason the plan called merging "guessing" was that
it had the old `continuation_of` heuristics in mind, which matched on names and
ordering. Matching on the recorded parent is not that, and conflating the two
kept a repair out of the tool for no reason.

So it folds by itself, once, when a project is opened. Two things keep it honest:

- **The link must be unique in both directions.** A take continued by two
  sources is a fork used twice; two sources ending on the same take is a
  duplicated row. Either way there is a choice, and making it here *would* be
  guessing. Ambiguous runs are left alone.
- **The edit does not move.** Every absorbed clip is rebased onto the surviving
  source by its offset in lineage time, and only clips that already ran back to
  back — untrimmed, in order, unlabelled, same speed, same pin — collapse into
  one. A trimmed clip does not meet its neighbour, so a trim cannot be folded
  away.

One more guard, and it is the subtle one: **a source part-way down the run that
plans more than it has rendered is refused.** Its unrendered segment may well
*be* the generation the next source rendered, and nothing on disk says whether
it is. Folding it would either double-count those planned seconds or throw the
segment away. Only the *last* source in a run may carry a planned tail, which is
what makes `source_seconds` come out identical before and after.

`Take.shot` is re-stamped onto the surviving source, because the take grid and
the take count filter on it and the absorbed rows' takes would otherwise vanish
from the source that now owns them. `parent_take` is never rewritten. The
absorbed row's name goes onto the segment it named, the way
`legacy_continuation_of` is kept.

**One defect found by running it on real projects:** twelve takes in PLANET
OFFICE DRAFT recorded a parent as `:FILENAME` — a take id with an empty
prompt_id, written by a render that knew which file it was continuing but not
the job that had made it. Those continuations read as unrelated roots, so two
real chains were invisible in the lineage browser and unfoldable. `gallery.
_resolve_parent_takes` now resolves that shape by filename alongside `file:`,
and *only when the filename belongs to exactly one take* — output names are not
unique within a project, and a parent picked out of two candidates is a
fabricated chain. That tightening applies to the `file:` path too, which
previously took whichever take came last.

Verified read-only across every project on disk before anything was written:
**24 projects, 171 sources, 22 runs folded, 70 rows absorbed, zero skipped, and
zero drift** — every timeline the same length after the fold as before it. Then
end to end through the devserver: THE PAPER AVIATOR's ten rows became one source
of ten generations behind one clip reading 43.417s, with nine dashed seams;
PLANET OFFICE DRAFT went 18 rows to 10, and *Romance B* — the known stale pick —
stayed separate and still flags itself broken.

## Migration

*This section originally planned for the importer to merge sibling chains. It
deliberately does not, and the reasons below are still right — but read Phase 6
after them, because the conclusion they were used to justify ("merging is an
explicit author action") turned out to be wrong. Merging happens by itself now.
What the migration must not do is merge **on a dict**, with no takes in reach:
the whole argument rests on the take record, and `Project.from_dict` cannot see
it.*

**Each existing shot becomes exactly one Source with exactly one Segment, plus
exactly one Clip** carrying its trims. The shot's id becomes the source's id:
takes record the shot they were rendered for, so a fresh id would orphan every
take in the project to save a naming convention. `continuation_of` was a guess,
so it is carried across as `legacy_continuation_of` rather than acted on or
dropped.

The timeline afterwards is the timeline before it, frame for frame.

Chains spread across sibling shots are **not** merged here, which the plan first
assumed they would be. Deciding *which* shots to merge from a dict alone would
be the guessing this rewrite exists to delete — the answer is in the take
record, which this layer cannot read. `fold.py` does it one layer up, at open,
where the takes are in reach; see Phase 6.

No heuristic runs here. The old inference functions were deleted outright rather
than given a farewell run, because a migration that silently writes a guess into
a stored `path` is worse than one that leaves the path empty and lets the
default show what it is.

Migration runs on **every read** (`Project.from_dict` → `migrate.to_v2`), not as
a one-off conversion pass, so a v1 file — including a restored backup or a dict
posted by an agent — lands on the same shape as everything else. Nothing on disk
needs converting.

Because lineage recovery is per-file and cached by mtime, a project's first load
after this change reads every take's header once — 300 takes took 0.23s.

Verified read-only across every project on disk: **28 projects, 207 sources, 207
clips**, every shot id preserved, every v1 pick landing as `path[0]`.

## Known snags

- ~~`export.py` already defines a class named `Clip`~~ — renamed `ResolvedClip`
  in Phase 1, and in Phase 2 it became one *piece*: a clip that crosses a join
  resolves to several.
- ~~`Sequence.split` gets simpler~~ — done, and it is `Timeline.split` now.
- ~~Tests touching this~~ — all ported in Phase 2, plus new `test_sources.py`
  and `test_edit.py`. Phase 1 shipped `Source`/`Clip`/`Timeline` with no unit
  tests of their own; they have them now.
- ~~`contract.py` and the `video-prompt` skill~~ — both updated, in Phases 2
  and 3. The skill now names the three types outright, because an agent reading
  it has no editor to learn them from.
- Restarting ComfyUI is required after any Python change, and the editor's new
  routes make that unmissable rather than subtle: the whole bin 404s.
- The lineage row's cells are fixed-width, which means every container between
  the sheet and the track needs `min-width: 0`. A grid/flex item defaults to
  `min-width: auto` and will not shrink below its content, so one missing
  declaration silently pushes the takes column wider than the sheet — it looked
  fine at 1280px and blew out at 900px.
