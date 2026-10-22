# Rendering by hand

`tools/render_sources.py` does everything below and is what you should run.
Read this only to do something the script does not: a custom `/prompt`
payload, setting a chain with `/source/path`, or debugging a render the
script reported as failed.

## The loop

1. **Prepare.** `POST /video_prompter/render/prepare` with
   `{"prompt": <segment.prompt>, "profile": PROFILE, "count": 1, "quality": Q,
   "project": NAME}`. Returns `{workflow, width, height, frames, seconds, seed,
   prompt, mode}` — `mode` is the auto-selected t2v/i2v/ref2v, and the `workflow`
   already has any keyframe/reference loaders baked in, so submission is
   identical for every mode. Always send the `profile` (the t2v one): i2v is
   built by augmenting it, ref2v reuses its model names. A **422** carries lint
   `findings` (fix them first — a bad render
   burns a generation); a **409** means the profile drifted from its workflow
   (tell the user to re-capture it).
   - `PROFILE`: `GET /video_prompter/profiles`; if exactly one, use it. If
     several, use the project's `render_profile` when set, else ask which.
   - `Q`: the tier. See `quality-tiers.md`.
2. **Queue on ComfyUI** (its root, not `/video_prompter`).

   Mint a UUID `prompt_id`, then `POST /prompt` with:
   ```json
   {"prompt": <prepared.workflow>, "client_id": "agent-render",
    "prompt_id": "<uuid>",
    "extra_data": {"video_prompter": {"project": NAME, "shot": <source.id>,
      "frames": <prepared.frames>, "seconds": <prepared.seconds>,
      "seed": <prepared.seed>, "prompt": <prepared.prompt>,
      "width": <prepared.width>, "height": <prepared.height>,
      "quality": <prepared.quality>, "steps": <prepared.steps>,
      "pass1_steps": <prepared.pass1_steps, omit when null>,
      "latent_path": <prepared.latent_path>,
      "parent_take": <prepared.parent_take, extends only>,
      "context_length": <prepared.context_length, extends only>}}}
   ```
   The `extra_data.video_prompter` block is what lets the take be recorded and
   matched to its source — the key is `shot`, the value is a source id, and it
   must always be included.

   Pass `quality` back verbatim from the prepare response. It is the tier **by
   name**, and it is how the take's quality badge is set. Without it the editor
   falls back to reading the tier off the canvas and the step count, which is
   right for a fresh generation and wrong for a continuation: an extend renders
   at the canvas of the clip it continues, so a `preview` continuation of a 720p
   parent is 1.0 MP at 8 steps — indistinguishable from `turbo_hd` by the
   numbers alone, and badged as such.
3. **Poll** `GET /history/{prompt_id}` until the entry has `outputs`. If its
   `status.status_str == "error"`, that render failed — surface it, don't record.
4. **Record** `POST /video_prompter/gallery/record` with
   `{"project": NAME, "history": {"<prompt_id>": <entry>}}` → `{added: [takes]}`.
5. **Choose** each returned take for its source:
   `POST /video_prompter/gallery/pick` with `{"project": NAME, "source":
   <source.id>, "take": <take.id>}`. That sets the source's whole `path` — the
   server walks up through each take's recorded `parent_take` to build it, then
   carries it on down the chain. There is no cascade to run afterwards: the
   parent's choice *is* `path[n-1]`.

   To set a chain outright, `POST /source/path` with `{"project", "source",
   "path": [take_id, …]}`. It is checked against what the renderer recorded and
   refused with a **409** carrying `problems` if the links do not hold up — so a
   chain that does not add up is caught at the write, not at export.

   To continue a source, plan the generation first with `POST
   /source/segment/add` `{"project", "source", "prompt"?}`, then render through
   `/render/extend` with the parent take as `source`.


## Batch endpoints

Three server-side conveniences prepare several workflows in one call (you still
queue each on ComfyUI yourself): `POST /video_prompter/render/batch` returns N
prepared renders of one prompt at consecutive seeds (seed sweeps); `POST
/video_prompter/render/extend/batch` does the same for a continuation — the
single extend payload plus `count`, returning `{"renders": [...]}` off one
parent take, one pinned run and one set of words, so the candidates differ only
in what the model invented past the context; `POST
/video_prompter/render/reshoot` re-prepares a take from its own frozen text,
canvas and length — the way to re-render a picked draft at `final` quality
without re-authoring.

Every batch endpoint caps `count` at 8 and takes an optional `seed` as the
*base* of the run — seeds come back consecutive, so a sweep is reproducible
from its first number. The base is clamped to leave room below the 2⁵³−1 seed
ceiling; past it consecutive seeds collapse onto one float64 and the sweep
becomes N copies of one clip.
