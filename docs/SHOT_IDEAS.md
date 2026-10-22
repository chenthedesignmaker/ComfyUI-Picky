# Shot Ideas

Open **Scene → Shot Ideas** at the top of the tab, beside **Focus**. Pick a feeling to browse two curated
approaches, then choose a storyboard shot (or a continuing move) in **For**.

- Select composition, color, lighting, and/or camera motion.
- Each approach includes two **Action snippets**, labeled by performance intensity:
  Subtle, Building, Strong, or Peak. Each snippet has its own **Copy** button. These
  are standalone examples to adapt to your characters and setting; copying never
  edits the storyboard, and visual-direction apply does not include them.
- Each composition, color, lighting, and camera description also has its own
  **Copy** button. Use these independently of the combined-direction checkboxes.
- **Copy direction** copies the selected qualities. On browsers without clipboard
  access, the drawer presents selectable text for manual copying.
- **References** opens Frame Set with the approach's search terms. Optional
  reference context, such as `office`, is appended and URL-encoded. FilmGrab and
  Eyecandy are also linked for browsing. No external images are fetched or stored.
- **Preview direction** shows current and proposed fields before applying.
  Composition replaces `angle`; camera replaces `camera`; color and lighting are
  appended to `action`. Existing action text, dialogue, cuts, timing, and scene
  defaults are preserved. Review any existing lighting descriptions for conflicts.
- **Undo last apply** restores the previous fields, until the storyboard changes
  again. Apply and undo mark the editor dirty; use the normal **Save** action.

The first version is a local cheat sheet, without an LLM or account requirement.
Scene context is displayed for the author; it does not automatically rewrite or
rank the curated approaches. Reference search terms are sent only when a link is
opened. These are creative options rather than fixed rules about emotions.

## Maintaining the feature

- `js/standalone/shot-ideas-data.js`: 12 feelings, 24 approaches, 48 action snippets, search terms,
  selective field patches, and preview freshness checks.
- `js/standalone/shot-ideas.js`: accessible modal drawer, copy, preview, and undo.
- `js/standalone/shot-ideas.css`: responsive drawer presentation.
- `main.js`: scene entry point and guarded commits through existing dirty/save flow.

Beat indices are positional. Preview and undo compare the entire beat list and
project/source/generation identity before committing, so an edit, reorder, or
source switch cannot silently redirect an old suggestion to another beat.

Run regression checks with `python -m pytest tests/test_frontend.py -q`.
