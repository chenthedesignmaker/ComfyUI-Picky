# Video LoRAs

Open a shot's editor and expand **Shot LoRAs**, just above **Visual style**.
Choose **Use project defaults**, **Custom stack**, or **Off**. A custom stack
replaces the project defaults; Off bypasses additional LoRAs for this shot.
Custom rows are retained when switching modes. LoRAs may control characters,
clothing, motion, or style as well as realism. The setting belongs to the whole
source/shot and applies to all of its generation segments, not individual
storyboard beats within one render. Duplicating or splitting a shot copies it.

Optional shared defaults remain under **Project settings → Default video LoRAs**.
Choose an installed `.safetensors` file,
click **Add LoRA**, and set its **Weight**. The checkbox bypasses a row without
forgetting it; weight 0 also bypasses it. Add more rows to stack adapters and use
the arrows to change their order. Settings save automatically with the project.

For the tested Realism People model, start at **0.6** and put `r34l1sm` in the
prompt's **Style** field. Cinematic Realism uses `DY`. Trigger words are not
silently added or removed. Files must be compatible with the selected video
model. The picker shows only files whose safetensors metadata identifies
MiniMax H3 (`base_model`, `ss_base_model_version`, or `modelspec.architecture`).
Other model families, files without identifiable metadata, and Turbo adapters
are hidden. This identifies the model family, not an assurance of visual quality
or suitability for every conditioning mode. Files are never deleted or moved.
Copy new models into ComfyUI's `models/loras` folder, then click **Refresh files**.
Renamed files still work when they retain their model metadata. An older running
backend shows only the two verified Realism People and Cinematic Realism filenames
until ComfyUI restarts and loads metadata detection. Existing saved stacks remain intact.

The stack applies to fresh text-to-video, image-to-video and reference-to-video
renders, re-renders, seed batches, and all sampling passes. With **Use project
defaults**, extends retain their parent take's recorded stack; legacy takes with
no stack record use the project settings. **Custom stack** and **Off** override
that inheritance, including on extends. The selected profile's existing LoRAs remain in place. Loading the
same file again is refused. Turbo belongs in the profile and stays controlled by
the quality tier, so it cannot be added as an extra stack row.

Each new take records the active filenames and weights and shows them in its
metadata. Held jobs keep the prepared graph and stack even if project settings
change before release. This feature does not alter post-processing/refine jobs
or the separate image-generation LoRA configuration.

## API

- `GET /video_prompter/video-loras` returns
  `{loras: [filename, ...], available, model_family: "minimax-h3", hidden_count}`.
  `available: false` means the server is running without ComfyUI's model catalog.
- `POST /video_prompter/video-loras` takes `project` and `loras` and saves only
  that project's LoRA setting, without replacing source or timeline edits.
- Video prepare, reshoot, batch and extend endpoints accept an optional `loras`
  override. An explicit `[]` bypasses the project/parent stack for that request.
- Send `source_id` with those requests to resolve the shot setting. It is
  separate from `source`, which is the video reference on an extend. The editor
  and render CLI send it automatically. Request `loras` overrides take priority.
- Save `source.video_loras` through `/source/update` as
  `{mode: "inherit" | "custom" | "off", loras: [...]}`. Missing/null means inherit.
- Each row is `{name, strength_model, enabled}`. Weight defaults to 1 and enabled
  to true. Weights must be finite numbers from -2 through 2; at most eight unique
  files are allowed. Negative strengths are supported but are not recommended
  starting points for the realism adapters.
- Prepared results return the active `loras` list. Clients submitting to ComfyUI
  must copy it into `extra_data.video_prompter.loras` so take history and extends
  retain it. The built-in editor does this automatically.

The implementation inserts model-only loaders after the shared model root,
before existing model patches, and preserves CLIP/VAE outputs. Profiles with
multiple or unsupported model roots fail preparation with an explicit error
instead of guessing a branch. Restart ComfyUI after installing this code change,
then refresh the editor tab.
