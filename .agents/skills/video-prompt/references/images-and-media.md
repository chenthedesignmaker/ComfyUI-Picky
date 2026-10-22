# Keyframe images, attached media and prompt history

Read this when generating a keyframe or identity image, attaching a file,
restoring an earlier prompt, or touching the project's image prompt.

### Generate a keyframe image (text-to-image → i2v pipeline)

When the user wants a specific look or composition as a starting frame, generate
one with Krea 2 Turbo (4 steps, ~seconds) then feed it into the i2v path:

1. **Generate.** `POST /video_prompter/render/t2i` with
   `{"prompt": "cinematic shot of a neon-lit alley at night, rain-slicked pavement"}`.
   Optional: `width`, `height` (delivery size, default **1920×1088**), `seed`,
   `steps` (default **8**), `cfg`, `style`, `loras`, `supersample`. Returns
   `{workflow, prompt, width, height, render_width, render_height, supersample, seed, steps, style, loras, mode: "t2i"}`.

   - **`supersample`** (default **1.5**) — renders at `width×height × factor`
     (snapped to /16) then lanczos-downscales to the delivery size, for sharper
     backgrounds. Note: the factor changes the latent size, so it **re-rolls the
     composition** — keep it fixed across seed-hunting and the final render. `1.0`
     turns it off (fastest, ~10s vs ~25s at 1.5).

   - **`style`** — a cinematic look bundle appended to the prompt (camera / film
     stock / lighting / grade / mood), e.g. `"kodak-2383"`, `"noir"`,
     `"neon-noir"`, `"golden-hour"`, `"anamorphic"`, `"bleach-bypass"`. The user's
     prompt is preserved and comes first. `GET /video_prompter/t2i/styles` lists
     every preset and the language it adds; an unknown name is a 400.
   - **`loras`** — a stack of `{name, strength_model, strength_clip, enabled}`
     objects patched onto the model/CLIP for grade/lens/character control.
     `strength_*` default to 1.0, `enabled` to true; disabled entries are dropped.
     Names must be Krea 2-compatible LoRA files in ComfyUI's `loras` dir.

   Several seeds at once: `POST /video_prompter/render/t2i/batch` takes the same
   payload plus `count` (1-8) and returns `{"renders": [...]}` at consecutive
   seeds. A `seed` you send is the base, so the whole sweep is reproducible from
   that one number.

2. **Queue on ComfyUI** the same way as a video render: `POST /prompt` with the
   returned `workflow`. Poll `/history/{prompt_id}` until `outputs` appears.
   The result is an image in ComfyUI's output dir (e.g. `image/VideoPrompter_t2i_00001_.png`).

   **Stamp the run as image generation** so the editor files it in the project's
   image library instead of ignoring it:

   ```json
   "extra_data": {"video_prompter": {
     "project": "Night run", "mode": "t2i",
     "shot": "<source id, if it is for one>", "member": "<cast id, if it is for one>",
     "prompt": "<the image prompt>", "seed": 4242, "steps": 8,
     "width": 1920, "height": 1088, "style": "neon-noir"
   }}
   ```

   `mode` is the marker — a video render that saves a preview frame must not land
   in the keyframe library, so the file type is never what decides. The rest is
   the recipe, which nothing on the finished PNG records and which is exactly
   what the author needs when a composition was nearly right.

3. **Adopt.** `POST /video_prompter/render/adopt` with
   `{"filename": "VideoPrompter_t2i_00001_.png", "subfolder": "image"}`.
   This copies the output image to the input dir and returns the path to set as
   `first_frame` (e.g. `"video_prompter/VideoPrompter_t2i_00001_.png"`).

   Or let the library do it: `POST /video_prompter/stills/record` with
   `{"project": "...", "history": {...}}` records every stamped image out of raw
   ComfyUI history **and adopts each one on the way in**, returning the stored
   `path` ready to put on a field. That is what the editor calls, so an image you
   record this way shows up in its studio too.

4. **Set the keyframe.** Update the shot's `first_frame` to the adopted filename
   and save. The shot now renders as **i2v** — the generated image anchors the
   start, H3 animates from it.

This is the recommended workflow for cinematic control: write a strong image
prompt for the key look, generate a few seeds to find the right composition,
adopt the winner as a keyframe, then render the video with a motion prompt.

A generated image is not only a keyframe. The same file can be a `<Picture N>`
reference on a segment, or a **cast member's identity file** — `references` on a
member, which every segment naming that member folds in — which is how one
generated portrait keeps a character looking the same across nine shots.

**The image library.** `GET /video_prompter/stills?project=...` (add `&source=`
to narrow) lists what a project has generated, newest first: the adopted `path`,
the recipe (`prompt_text`, `seed`, `style`, `steps`, `width`, `height`) and what
it was made for (`source`, `member`). Before generating a new picture, look here
— the shot you want may already exist. `POST /video_prompter/stills/annotate`
sets `favorite`/`discarded` or re-files a still against a `source` or `member`;
`POST /video_prompter/stills/forget` drops the record and leaves the file alone,
so anything already pointing at it keeps working.

The user can do all of this in the editor: the bin's **Images** view lists the
library and holds the *Generate images* button, and **Generate** in the Media tab
and beside a cast member's identity files opens the same generator over the same
library. So an image you record shows up for them without being told where to
look.

## Attach media (keyframes and references)

A keyframe or reference file must live in ComfyUI's input dir before it can drive
a render. Upload it once, then put the returned name on the prompt:

```bash
# fetch an online image server-side (or POST a multipart file part instead)
curl -s -X POST http://127.0.0.1:8188/video_prompter/reference/upload \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://picsum.photos/1024/576","filename":"hero.jpg"}'
# -> {"filename":"video_prompter/hero.jpg","kind":"image", ...}
```

Then set `first_frame: "video_prompter/hero.jpg"` (i2v) or add
`{"kind":"image","path":"video_prompter/hero.jpg","role":"the hero"}` to
`references` (ref2v) on the segment's prompt, and save the source as usual. The
open editor's Media tab shows the same attachments live.

## Prompt history, and the model in the editor

Two things landed that change what you can assume.

**Every save files a revision.** The server compares the incoming prompt against
what is on disk and, when it changed, keeps the old one on the segment. You do
not snapshot anything and you cannot lose someone's prompt by posting a source
you built from `/schema` — history is read off disk, never from your payload.
Segments in a project reply carry only a `revision_count`; the revisions
themselves come from their own route so an ordinary save does not ship them.

```bash
# one segment's past prompts, newest first, each with its composed text
curl -s "http://127.0.0.1:8188/video_prompter/source/revisions?project=NAME&source=SRC&segment=SEG"

# put one back (files the current one, so undoing an undo is another restore)
curl -s -X POST http://127.0.0.1:8188/video_prompter/source/revisions/restore \
  -H 'Content-Type: application/json' \
  -d '{"project":"NAME","source":"SRC","segment":"SEG","revision":"rev_1a2b3c4d"}'
```

Say what a save was for and it shows against the revision: `/source/update`
takes an optional `note` ("tightened shot 3", "reverted the score").

**The editor can write prompts itself**, through `POST /llm/draft`, when a
Gemini key is configured — either `VIDEO_PROMPTER_GEMINI_KEY` in ComfyUI's
environment, or a `gemini_key` in `settings.json` beside the projects (see
`GET /llm/status`, which reports which of the two is in force and where the
file lives). Prefer the file: ComfyUI's own Restart button re-executes in place
and inherits the environment it already had, so a key added to the launcher
after startup stays invisible, while the file is read at each use. You rarely need this — you *are* the
model — but two things about it matter to you:

- Prompts you did not write may carry revisions noted `gemini-…: <instruction>`.
  That is where they came from, not a corruption.
- The endpoint enforces scope at the merge and answers in a schema that has no
  `references`, `first_frame`, `width` or `negatives` in it, so an in-editor
  rewrite can never touch a keyframe or a canvas. Neither should yours when the
  user asked for one part: change what was asked and leave the rest byte-identical.

**The image prompt has its own history, and its own writer.** An image prompt is
one free-text string, so none of the protection above applies to it: there are no
fields, so no response schema to narrow, and nothing for `merge` to keep out of
scope. A rewrite replaces all of it. What makes that safe is the trail:

```bash
# past versions of a project's image prompt, newest first
curl -s "http://127.0.0.1:8188/video_prompter/image_prompt/revisions?project=NAME"

# keep the words you are about to replace (no-ops on an empty box, or on the
# same text the newest entry already holds)
curl -s -X POST http://127.0.0.1:8188/video_prompter/image_prompt/revisions/file   -H 'Content-Type: application/json'   -d '{"project":"NAME","text":"the prompt being replaced","note":"why","target":"first_frame"}'

# write or rewrite one; naming the project files the outgoing text for you
curl -s -X POST http://127.0.0.1:8188/video_prompter/llm/draft-image   -H 'Content-Type: application/json'   -d '{"project":"NAME","prompt":"a rain-slicked alley at night","instruction":"make it dawn","style":"neon-noir"}'
# -> {"prompt":"...","before":"...","changed":true,"model":"gemini-...","filed":{...}}
```

One trail per project, not one per destination — each entry records the `target`
it was written for so the list can say, but recovery is not made into a hunt.
There is no restore route: put the text back yourself, filing what it displaces
first, which is exactly what the editor does. Leave `project` off `/llm/draft-image`
to draft without filing anything, which is what you want when comparing options.

If **you** are replacing someone's image prompt, file the old one first. The
editor's undo and yours are the same trail.
