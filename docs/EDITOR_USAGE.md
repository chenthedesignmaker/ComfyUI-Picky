# Editor navigation and usage

The authoring views are Scene, Audio, Inputs, and Clip. Scene opens by default.
The internal panel keys `media` and `source` remain compatible with existing
field navigation; usage exports name them `inputs` and `clip`.

- Scene contains the AI writer, visual direction, storyboard, and dialogue.
- Audio contains generated soundscape, cues, and score.
- Inputs contains current-generation identities, keyframes, subjects, and references.
- Clip contains trim, speed, and playback sound.
- Project library, composed prompt, validation, and usage open in resizable panels.
- Name is editable in the header; generation duration/aspect stay near Render.

## Collection

`js/standalone/usage.js` collects aggregate counters in browser localStorage under
`vp:usage:v1:`. Nothing is sent to a service. It stores no project/source names,
identifiers, prompt text, field values, media paths, or individual keystrokes.
Data belongs to this browser origin; another browser/port has separate data.

Each page load starts a session. Returning after 30 minutes idle starts another.
Manual visits, contextual visits, and initial displays are separate counters.
Clicking an already-active tab does not add a visit. Navigation with arrow keys
counts as manual; programmatic prompt-to-field jumps count as contextual.
Source/generation changes count as initial displays, not manual visits.

Active time stops after 60 seconds without editor activity and excludes hidden
or unfocused windows. Counters flush every 15 seconds and on navigation/page exit;
an abrupt crash may lose the latest unflushed actions. Sessions older than 90 days
are removed when opening the usage report/export. Separate session keys prevent
two browser tabs from overwriting each other's totals. Reset and opt-out are shared
across tabs. Storage failures leave temporary in-memory counters instead.

Actions count committed field changes, one text-edit burst per focus interval,
and commands. They are activity measures, not proof that a save succeeded.
Render submission/rejection, still submission/rejection, render validation blocks,
take selection, and error clearance are separate outcomes. Identity reuse counts
successful video submissions using project identities, including repeated renders.
External HTTP/API edits and background refreshes are not user editing actions.

## Reading the data

Open **Usage** below the form for per-area manual visits, session reach, actions,
and active minutes. **Export JSON** produces a selectable preview and download link,
including contextual/initial visits, transitions and outcomes. **Collect usage**
pauses collection without deleting history. **Reset data** clears usage records
without affecting projects or other preferences.

The layout version is `four-tabs-v1`. There is no historical baseline for the old
seven-tab layout. Compare future versions by layout, session reach, task outcomes,
and switching patterns, not raw clicks alone. Render-submission counts count jobs,
so a seed sweep can produce several outcomes from one command. A validation-clear
event means previously displayed errors cleared in the same source/generation;
it is not proof that the user consciously fixed them.

## Validation

Run `node --test tests/frontend_usage.test.cjs` for navigation counts, idle timing,
session rollover, retention, export records, storage failures, cross-tab resets,
opt-out, and privacy allowlists. The suite is included in `tests/test_frontend.py`.
