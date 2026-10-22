"""The machine-readable contract an agent reads to drive this tool.

`GET /video_prompter/schema` serves `schema()` so any agent — Claude Code, an
embedded panel, or an external tool-caller like Hermes or OpenClaw — can
self-orient without a human pasting the field list. It describes the prompt
document, the model's hard limits, the lint rules that decide "good", and the
handful of HTTP operations that make up the authoring loop.

Kept free of `Api` and aiohttp so it can be unit-tested against `h3_spec`
directly and reused wherever the contract is published.
"""

from __future__ import annotations

from typing import Any

from . import h3_spec
from .document import REFERENCE_KINDS, ReferenceSet
from .lint import Severity

# The prompt document, block by block. Types are described the way an agent
# should send them in the `serialize` dict, not as Python types.
PROMPT_FIELDS: dict[str, dict[str, Any]] = {
    "duration": {
        "type": "int",
        "unit": "seconds",
        "note": "whole seconds; the storyboard's total budget",
    },
    "beats": {
        "type": "list",
        "of": {
            "start": "float",
            "end": "float",
            "angle": "str",
            "action": "str",
            "camera": "str",
            "camera_path": "object|null",
            "transition": "str",
            "cut": "bool",
            "lines": "list",
        },
        "note": "the timed storyboard; spans should tile the budget with no gap "
        "or overlap. `angle` is framing, `action` is what happens; they join with "
        "a comma into one clause, so write `action` as a phrase that can follow "
        "one ('steam rising from the bread'), not a predicate ('frames the "
        "baker'). `camera` is "
        "this shot's camera motion woven into a sentence (motion type + amplitude "
        "'with small/large amplitude' + speed 'at slow/fast speed'), e.g. 'pushes "
        "in with small amplitude at slow speed toward the door'. `transition` is "
        "the cut phrase opening a later shot (default 'the camera cuts to'); the "
        "first shot is un-timed and takes no transition. Only the shot start is "
        "emitted, as `[Shot N] At MM:SS.mmm`. `cut` (default true) says whether "
        "the beat opens a `[Shot N]` of its own; set it false and the beat folds "
        "into the shot before it as a timed moment inside one continuous take, "
        "composed camera-first as `At MM:SS.mmm, the camera pushes in ... as she "
        "lifts the mug` — the camera must lead, since a move written behind its "
        "action was measured not to render at all. On a no-cut beat `angle` is "
        "where the move arrives, written `into a wider view`, not a framing to "
        "cut to. Reach for it when only the framing or the camera changes; a "
        "real cut needs new information. `lines` are the shot's dialogue "
        "(see the lines schema).",
    },
    "beats[].camera_path": {
        "type": "object|null",
        "of": {"target": "str", "interpolation": "smooth|linear", "freeze": "bool",
               "frame_mode": "scene|motion", "reference_video": "str",
               "subject_box": "{left,top,width,height}|null", "loop_closure": "off|auto",
               "experiment": "off|compact|numbers",
               "keyframes": "list[{time, azimuth, elevation, distance}]"},
        "note": "Optional camera planner, used instead of beats[].camera while present. "
        "2–12 ordered keyframes; time is a fraction of the beat (0–1), azimuth is an "
        "unwrapped orbit offset (-720 to 720 degrees, positive camera-right), elevation "
        "is -89 to 89 degrees, distance is 0.05–4 times the opening radius. First "
        "keyframe is {time:0,azimuth:0,elevation:0,distance:1}; last time is 1. Timing "
        "follows the beat when retimed. This is prompt guidance, not geometric "
        "conditioning. Motion mode requires a video path and a single full-generation beat; it "
        "uses Ref2VA with saved still anchors inactive. Auto loop closure anchors the final frame "
        "only for one frozen ±360° path returning to its opening elevation/radius, a first image "
        "and no references or conflicting last image. Subject boxes use normalized coordinates. "
        "Experiments are text recipes only. Null restores handwritten camera text.",
    },
    "beats[].lines": {
        "type": "list",
        "of": {
            "speakers": "list[str]",
            "text": "str",
            "language": "str",
            "verb": "str",
            "delivery": "str",
            "describe": "str",
            "voiceover": "bool",
            "possessive": "str",
            "continuity": '"" | "into" | "from"',
            "cutoff": "bool",
            "placement": "str",
        },
        "note": "spoken/sung lines in a shot. `speakers` are stable author keys "
        "mapped to (S1),(S2)... by first vocal appearance across the whole prompt "
        "(two keys on one line make a group tag (S1,S2)); a key keeps its ID "
        "across shots. `text` is the VERBATIM words (only these and the language "
        "go inside <d>[Language] ...</d>). `describe` is the identifying phrase "
        "at first appearance (voice/age/on- or off-screen), `verb` is says/shouts"
        "/sings, `delivery` the manner. `voiceover: true` emits the fixed phrase "
        "'says in an off-screen voiceover' plus the note that the on-screen "
        "character's lips stay shut; `possessive` picks the word in that note "
        "(\"his\", \"her\", \"the baker's\" — default \"their\"). A line that "
        "crosses a cut is written as two lines on adjacent beats: the outgoing "
        "half sets `continuity: \"into\"`, the incoming half `\"from\"`, and each "
        "emits <scenetrans> at the seam. `cutoff: true` marks speech the end of "
        "the video truncates. `placement: \"open\"` moves the line to the front "
        "of the shot, right after the framing clause and ahead of the action — "
        "a line has no timecode, so where its sentence sits is the only thing "
        "that says when it lands. Default \"\" keeps it last.",
    },
    "cues": {
        "type": "list",
        "of": {"start": "float", "end": "float|null", "description": "str"},
        "note": "audio events pinned to the ruler; `end` null is an instant hit, "
        "set it for a sustained bed. Composed as `At MM:SS.mmm, ...` / `From ... "
        "to ..., ...` sentences in `overall_soundscape`. TESTED AND IT DOES NOT "
        "TIME (2026-08-25): a cue at 00:03.500 produced the clip's quietest "
        "moment at 3.5s, while the slam it named landed at 6.1s — the same "
        "6.1s an untimed mention produced. Naming a sound works; timing it "
        "does not. The official format times shots, not sounds, and a cue is "
        "our own extension. Write the sound into the beat it belongs to and "
        "accept that H3 places it.",
    },
    "style": {
        "type": "str",
        "note": "the style contract: medium, texture, palette, era. Fixes the look "
        "so it does not drift between shots.",
    },
    "overview": {
        "type": "str",
        "note": "optional one-line scene overview. In the base modes it "
        "continues the style's opening clause INSIDE [Shot 1]; only the "
        "full-reference document lifts style and overview above the marker. "
        "Write it as a lower-case clause ('a rain-soaked alley at dusk'), not a "
        "capitalised sentence.",
    },
    "camera": {
        "type": "str",
        "note": "global camera/cutting style, woven into [Shot 1] as a fallback. "
        "Prefer per-shot `beats[].camera` — the format wants motion woven into "
        "each shot, never stacked as a trailing label.",
    },
    "audio": {
        "type": "str",
        "note": "the DIEGETIC soundscape in prose: ambient, physical-action and "
        "non-verbal human sound (wind, footsteps, breathing). Merged with `cues` "
        "and emitted as `overall_soundscape`. Dialogue and score do NOT go here. "
        "H3 generates stereo in the same pass, so an empty track is wasted.",
    },
    "music": {
        "type": "str",
        "note": "non-diegetic score only the audience hears, emitted as "
        "`non_diegetic_music`. Describe instrumentation, tempo, rhythm and "
        "dynamics — not mood words. Empty composes to `N/A`. Music a character "
        "can hear (radio, jukebox) is diegetic and belongs in the shot instead.",
    },
    "screen_text": {
        "type": "list[str]",
        "note": "literal on-screen text, reproduced exactly in double quotes. "
        "Stated in [Shot 1], since the list carries no shot of its own — for "
        "signage that appears in a later shot, write it into that beat's "
        "`action` in double quotes instead, which is what the format wants.",
    },
    "negatives": {
        "type": "list[str]",
        "note": "kept for the editor but NOT composed into the prompt: H3 has no "
        "negative input and the official format has no negative section, so naming "
        "unwanted things would only summon them. Leave empty in new prompts.",
    },
    "references": {
        "type": "list",
        "of": {
            "kind": "|".join(REFERENCE_KINDS),
            "path": "str",
            "role": "str",
            "retention": "str",
            "member": "str",
            "slot": "int",
        },
        "note": "omni-reference files that DRIVE reference-to-video (ref2v): the "
        "kind's node slot is fed the file at `path` (a name in ComfyUI's input "
        "dir, from /reference/upload). The prompt renders as the six-section "
        "full-reference document. `role` becomes the file\'s own `<Picture N>`/"
        "`<Video N>`/`<Audio N>` subject_definitions line and `retention` its "
        "retention_analysis marker (see reference_vocabulary; empty uses the "
        "per-kind default) — but ONLY when the file stands on its own, as a "
        "keyframe, a source video or an audio track. A picture that merely "
        "shows a person, a place or a prop is named by a `subjects` entry "
        "instead and writes no line of its own. Setting any reference renders "
        "the segment as ref2v. "
        "Set `member` (a cast id) and `slot` (0-based index into that member's "
        "`references`) instead of `path` to LINK one of a member's identity "
        "files rather than copy it: `kind`/`path` then resolve from the member "
        "at compose time, so swapping their sheet updates every segment, while "
        "`role` and `retention` set here override the member's for this segment "
        "only. A linked row also puts that member on screen, so it needs no "
        "entry in `cast`. Leave `member` empty for an ordinary file reference.",
    },
    "subjects": {
        "type": "list",
        "of": {
            "role": "str",
            "retention": "str",
            "sources": "list[str]",
            "noun": "str",
            "member": "str",
            "label": "str",
        },
        "note": "the `<Subject N>` layer: what the target video CONTAINS, as "
        "opposed to which files it was given. H3 tracks subjects, so a person "
        "or a location is a subject and the pictures of them are only its "
        "sources. `role` is the prose after `<Subject N> is`; `noun` is the "
        "word the citation hangs on (person / setting / object / speaker) and "
        "may be empty; `sources` are `references[].path` values that define it "
        "— a visual one loses its own `<Picture N>` line, an audio one keeps "
        "its `<Audio N>` line and gains a `for <Subject N> (Sx)` back-link. "
        "A cast member on screen who has identity files becomes a subject "
        "automatically at compose time, so do NOT write one for them; set "
        "`member` (a cast id) with no `sources` to override just that "
        "member's `role`/`retention` for this segment. Write a plain entry "
        "(no `member`) only for something with no cast identity behind it. "
        "`label` is for the editor and is never composed.",
    },
    "ref_task": {
        "type": "list[str]",
        "note": "ref2v only: the summary's task types (see reference_vocabulary."
        "task_types), joined with ' + '. Empty infers a sensible default.",
    },
    "ref_summary": {
        "type": "str",
        "note": "ref2v only: the one-paragraph summary after the task-type "
        "prefix. Uses the <Tag N> labels; do not introduce new ones here.",
    },
    "first_frame": {
        "type": "str",
        "note": "keyframe image name in ComfyUI's input dir (from "
        "/reference/upload). Setting it renders the segment as image-to-video "
        "(i2v): the frame is the geometry anchor the clip starts from. Cannot "
        "be combined with `references` (different node).",
    },
    "last_frame": {
        "type": "str",
        "note": "optional end keyframe image name (input dir). Pairs with "
        "`first_frame`; the clip resolves toward it.",
    },
    "mentions": {
        "type": "syntax",
        "note": "`@Name` in any prose field is a link to a project cast member, "
        "resolved at compose time to whatever label routes to them: their "
        "`<Subject N>` when they have identity files, `the <noun> (Sx)` when "
        "they only speak, `the <noun>` when they have neither. Write `@Mara` "
        "rather than a typed `<Subject 2>` — the number is positional, so it "
        "moves whenever the shot gains or loses a member, and a typed one does "
        "not move with it. Mentioning a member also puts them on screen, the "
        "same way attaching their identity file does. Matched whole and "
        "case-insensitively, longest name first, so a name may hold spaces. "
        "Honoured in: style, overview, camera, audio, music, ref_summary, "
        "beats[].angle/action/camera, beats[].lines[].describe/delivery, and "
        "the `role` on a subject or reference. NOT honoured in "
        "beats[].lines[].text or screen_text — a character saying another's "
        "name out loud, or a name painted on a sign, must stay a name.",
    },
    "cast": {
        "type": "list[str]",
        "note": "project cast member ids that appear in this segment but "
        "never speak — a location, a silent character. A member who speaks "
        "is named by a line's `speakers` instead. The member's appearance, "
        "voice and identity files are folded in at compose time, so editing "
        "the member updates every segment naming them. See /cast.",
    },
    "undescribed": {
        "type": "list[str]",
        "note": "cast ids present in this segment whose `appearance` must NOT be "
        "stated in the opening clause. Being on screen normally writes that "
        "appearance in, which is right for an establishing shot and wrong for a "
        "close one — a location's appearance is a paragraph of window walls and "
        "doorways, and stating it asks the camera to show them. Use it to keep a "
        "member's identity files without the prose. Ignored for a speaker, who is "
        "identified on their line instead.",
    },
    "guard_identities": {
        "type": "bool",
        "note": "default true. When two speakers share a shot, appends one "
        "sentence telling the model they are different people and must not "
        "merge or swap. Co-present identities are where H3 blends faces; set "
        "false only when the sentence reads as noise.",
    },
    "width": {"type": "int", "note": "canvas width, a multiple of 32"},
    "height": {"type": "int", "note": "canvas height, a multiple of 32"},
}

# rule id -> one-line meaning. Mirrors the findings raised in `lint.py`; kept
# here as a flat catalogue so an agent knows what it is aiming at before it
# drafts, and what a returned finding means. Severities come from `_SEVERITIES`.
LINT_RULES: dict[str, str] = {
    "duration.invalid": "duration is not a whole number of seconds the node can generate",
    "duration.snapped": "the clip length snaps up to the model's frame grid",
    "duration.untrained": "the frame count is outside the model's trained range",
    "canvas.too_small": "width or height is below the 32px minimum",
    "canvas.off_grid": "width or height is not a multiple of 32",
    "canvas.too_large": "the canvas is over the pixel cap and would be scaled down",
    "storyboard.empty": "no beats; H3 will stretch one idea across the whole clip",
    "storyboard.overlap": "two beats claim the same time; pacing drifts",
    "storyboard.gap": "part of the clip has no beat scheduled",
    "storyboard.overrun": "the storyboard runs past the clip budget and is cut off",
    "beat.invalid_span": "a beat has an empty or reversed time span",
    "beat.too_short": "a beat is under the minimum and reads as a flash frame",
    "beat.tight": "a beat is short; little room for the action to land",
    "beat.no_action": "a beat describes no action",
    "beat.empty_continuation": "a no-cut beat neither moves the camera nor describes an action",
    "camera_path.invalid": "camera path keyframes or settings are invalid",
    "camera_path.freeze_dialogue": "a frozen scene also asks for a speaking performance",
    "camera_path.loop_disabled": "requested loop closure cannot anchor the final frame",
    "beat.move_rushed": "a no-cut camera move has too little time to travel",
    "beat.density": "too many cuts per second for the budget",
    "beat.weak_cut": "a cut introduces nothing the previous shot did not have, so H3 tends to merge the two",
    "beat.camera_complex": "one beat asks for several camera moves at once; only one usually lands — give the second its own no-cut beat",
    "dialogue.no_speaker": "a spoken line has no speaker to map to an (S) id",
    "dialogue.empty": "a speaker is named with no words to say",
    "dialogue.seam_unpaired": "a line crossing a cut is tagged on only one side of it",
    "dialogue.overrun": "a beat holds more words than can be spoken in its running time",
    "dialogue.tight": "speech nearly fills a beat, leaving no room for a breath or pause",
    "dialogue.no_opening": "a line set to open its shot has no composition to follow",
    "storyboard.headroom": "a short budget is crammed while H3 allows much longer",
    "audio.missing": "no audio direction, so the generated track is unspecified",
    "audio.cue_out_of_range": "an audio cue falls outside the clip",
    "audio.cue_covers_clip": "an audio cue spans the whole clip, so it pins nothing",
    "audio.cue_is_dialogue": "an audio cue holds speech, which belongs on a line",
    "style.missing": "no style contract, so the look is undefined",
    "music.diegetic": "non_diegetic_music names an in-world source the audience-only score should not have",
    "references.too_many": "more references of a kind than the node accepts",
    "references.orphan_soundtrack": "a reference-video soundtrack has no matching video",
    "references.unused": "a reference is attached but never referred to by tag",
    "ref.task_invalid": "a ref2v summary task type is not in the controlled vocabulary",
    "ref.summary_undefined_tag": "the ref2v summary names a label nothing defines",
    "ref.retention_speaker_id": "a retention_analysis line carries an (Sx), which the format forbids",
    "ref.description_length": "a ref2v detailed_description is outside the guide's 350-500 word range",
    "subjects.unused": "a subject is defined but never named in the description",
    "ref.retention_invalid": "a ref2v retention marker is not in the controlled vocabulary",
    "ref.summary_missing": "a ref2v shot has no summary, so nothing says what to do with its references",
    "cast.unknown": "the prompt names a cast member the project does not have, so their description is not composed in",
    "mode.conflict": "both a keyframe and omni-references are set; i2v and ref2v are different nodes",
    "keyframe.orphan": "a last-frame keyframe is set with no first frame",
    "prompt.too_long": "the composed prompt is over the hard character limit",
    "prompt.long": "the composed prompt is long; later blocks lose weight",
    "prompt.aesthetic": "quality language (4k, masterpiece) the camera cannot see; H3 conditions on described behaviour",
}

# Severity per rule prefix-free lookup, taken from lint.py's own choices.
_SEVERITIES: dict[str, Severity] = {
    "camera_path.invalid": Severity.ERROR,
    "camera_path.freeze_dialogue": Severity.WARNING,
    "camera_path.loop_disabled": Severity.WARNING,
    "duration.invalid": Severity.ERROR,
    "duration.snapped": Severity.INFO,
    "duration.untrained": Severity.WARNING,
    "canvas.too_small": Severity.ERROR,
    "canvas.off_grid": Severity.ERROR,
    "canvas.too_large": Severity.WARNING,
    "storyboard.empty": Severity.WARNING,
    "storyboard.overlap": Severity.ERROR,
    "storyboard.gap": Severity.WARNING,
    "storyboard.overrun": Severity.ERROR,
    "beat.invalid_span": Severity.ERROR,
    "beat.too_short": Severity.WARNING,
    "beat.tight": Severity.INFO,
    "beat.no_action": Severity.WARNING,
    "beat.empty_continuation": Severity.WARNING,
    "beat.move_rushed": Severity.INFO,
    "beat.density": Severity.WARNING,
    "beat.weak_cut": Severity.WARNING,
    "beat.camera_complex": Severity.INFO,
    "dialogue.no_speaker": Severity.WARNING,
    "dialogue.empty": Severity.INFO,
    "dialogue.seam_unpaired": Severity.WARNING,
    "dialogue.overrun": Severity.WARNING,
    "dialogue.tight": Severity.INFO,
    "dialogue.no_opening": Severity.WARNING,
    "storyboard.headroom": Severity.INFO,
    "audio.missing": Severity.INFO,
    "audio.cue_out_of_range": Severity.WARNING,
    "audio.cue_covers_clip": Severity.INFO,
    "audio.cue_is_dialogue": Severity.WARNING,
    "style.missing": Severity.WARNING,
    "music.diegetic": Severity.INFO,
    "references.too_many": Severity.ERROR,
    "references.orphan_soundtrack": Severity.ERROR,
    "references.unused": Severity.WARNING,
    "ref.task_invalid": Severity.WARNING,
    "ref.summary_undefined_tag": Severity.WARNING,
    "ref.retention_speaker_id": Severity.WARNING,
    "ref.description_length": Severity.INFO,
    "subjects.unused": Severity.WARNING,
    "ref.retention_invalid": Severity.WARNING,
    "ref.summary_missing": Severity.WARNING,
    "cast.unknown": Severity.WARNING,
    "mode.conflict": Severity.ERROR,
    "keyframe.orphan": Severity.INFO,
    "prompt.too_long": Severity.ERROR,
    "prompt.long": Severity.WARNING,
    "prompt.aesthetic": Severity.WARNING,
}

# The authoring loop, described so an agent can run it over HTTP. Paths are
# relative to the /video_prompter prefix the routes are mounted under.
OPERATIONS: list[dict[str, str]] = [
    {
        "step": "list installed video LoRAs",
        "method": "GET",
        "path": "/video-loras",
        "note": "Returns installed filenames and catalog availability. Use adapters compatible with the video model.",
    },
    {
        "step": "save project video LoRA stack",
        "method": "POST",
        "path": "/video-loras",
        "note": "{project, loras: [{name, strength_model, enabled}]}. Up to 8 unique files, finite weights -2..2. Video prepare/reshoot/batch/extend also accept a loras override; [] bypasses. Extends otherwise inherit the parent's recorded stack. Put trigger words in Style. Copy prepared.loras into extra_data.video_prompter.loras when submitting.",
    },
    {
        "step": "read projects",
        "method": "GET",
        "path": "/projects",
        "note": "`projects` is the active names; `entries` adds a row per "
        "project on every shelf, with its shelf, dates and `lineage` (the "
        "version family a duplicate shares with its original).",
    },
    {
        "step": "shelve a project",
        "method": "POST",
        "path": "/project/shelve",
        "note": "{name, shelf: active|archived|trashed}. Archiving hides a "
        "finished project; trashing is a recoverable delete. Restoring is the "
        "same call asking for `active`.",
    },
    {
        "step": "duplicate a project",
        "method": "POST",
        "path": "/project/duplicate",
        "note": "{name, new_name?} — copies takes and stills, inherits the "
        "lineage, and defaults to the next `vN` name in the family.",
    },
    {
        "step": "read a project",
        "method": "GET",
        "path": "/project/load?name=NAME",
        "note": "the whole project: `sources` is the bin (one lineage each, "
        "with the segments that planned it), `timeline.clips` is the edit.",
    },
    {
        "step": "read one source",
        "method": "GET",
        "path": "/source?project=NAME&source=SOURCE_ID",
        "note": "convenience getter for one source, its segments, and depth 0's "
        "prompt dict",
    },
    {
        "step": "validate a draft",
        "method": "POST",
        "path": "/preview",
        "note": "body {prompt}; returns composed text and lint findings. Resolve "
        "every error finding before saving.",
    },
    {
        "step": "fix the storyboard or canvas",
        "method": "POST",
        "path": "/transform",
        "note": "body {prompt, ops}; ops are fix / refit / ratio, so the frame "
        "grid and contiguity rules are not reimplemented by the caller.",
    },
    {
        "step": "save a source",
        "method": "POST",
        "path": "/source/update",
        "note": "body {project, source, note?}; the source carries its segments, "
        "each with a prompt. Optional source.video_loras is {mode: inherit|custom|off, loras: [...]}; "
        "custom replaces project defaults and off bypasses them. Send source_id on video prepare requests "
        "to resolve this setting; custom/off also override parent LoRAs on extends. "
        "Omit `origin` and any open editor live-updates. "
        "`note` says where the save came from (\"Gemini: made it faster\") and "
        "is shown against the revision this save files.",
    },
    {
        "step": "write or rewrite a prompt with a language model",
        "method": "POST",
        "path": "/llm/draft",
        "note": "body {instruction, prompt? | project+source(+segment?), "
        "scope?}. One operation for both ways in: a whole scene from a "
        "description, and a targeted tweak (\"make this faster\") on a prompt "
        "that already says something. `scope` is \"all\", a field name, a list "
        "of field names, or \"beat:N\" — the model is shown the whole "
        "document for context but is only asked for, and only allowed to "
        "change, the fields in scope. A tweak therefore cannot wander, and "
        "does not pay to regenerate the fields it was never about. An "
        "existing prompt is defended in the turn as finished work rather "
        "than a draft to improve. Fields outside the authored set (references, "
        "keyframes, canvas) are not in the shape the model can answer in and "
        "come through untouched. The draft is re-linted and handed back for "
        "repair up to twice. Naming `project` and `source` applies it and "
        "files a revision, so it is undoable rather than asked about.",
    },
    {
        "step": "check a model is configured",
        "method": "GET",
        "path": "/llm/status",
        "note": "whether a key is set, which model it names, and whether it "
        "came from the environment or the settings file (`settings_file` in "
        "the reply says where that is). `?probe=1` also lists the models the "
        "key can see. A 503 from /llm/draft means neither source has one.",
    },
    {
        "step": "read a segment's past prompts",
        "method": "GET",
        "path": "/source/revisions?project=NAME&source=SOURCE_ID&segment=SEG_ID",
        "note": "newest first, each with its composed text. The server files a "
        "revision itself whenever a save changes a prompt, so a rewrite is "
        "always reversible and no caller has to remember to snapshot. Segments "
        "in the ordinary project reply carry only a `revision_count`; the "
        "revisions themselves are served here so every save does not ship them.",
    },
    {
        "step": "undo a rewrite",
        "method": "POST",
        "path": "/source/revisions/restore",
        "note": "body {project, source, segment?, revision}; puts that past "
        "prompt back on one segment and files the current one, so undoing an "
        "undo is just another restore. Scoped to the segment — unlike "
        "`/project/restore`, which rolls back the whole bin.",
    },
    {
        "step": "put it on the timeline",
        "method": "POST",
        "path": "/clip/add",
        "note": "body {project, source, at?}; a clip is one use of a source. "
        "`/source/add` already makes one, so this is for a second use.",
    },
    {
        "step": "silence part of a clip",
        "method": "POST",
        "path": "/mute/add",
        "note": "body {project, clip, start, end} in *lineage* seconds — the "
        "clock `source_in`/`source_out` use, not the ruler. A mute is a "
        "decision about a moment of the footage, so trimming or moving the "
        "clip carries it. `/mute/edit` {project, mute, start?, end?} and "
        "`/mute/delete` {project, mute}. To silence a whole clip use "
        "`/clip/edit` instead.",
    },
    {
        "step": "mute a clip, or hear a different take",
        "method": "POST",
        "path": "/clip/edit",
        "note": "body {project, clip, audio:{muted?, take?, gain_db?, offset?}}. "
        "`take` names the take to *hear* — for a re-roll whose picture you "
        "rejected and whose sound you kept — and null gives the clip its own "
        "sound back. Picture and sound share the clip's trim, so nothing has "
        "to be re-aligned. Same route also does source_in / source_out / "
        "speed / label / enabled / pin.",
    },
    {
        "step": "choose which take a source plays",
        "method": "POST",
        "path": "/source/path",
        "note": "body {project, source, path}; `path` is one take id per "
        "generation, root first. Checked against each take's recorded parent "
        "and refused with 409 if the chain does not hold up. `/gallery/pick` "
        "does the same from a single take id.",
    },
    {
        "step": "split shots into their own source",
        "method": "POST",
        "path": "/source/split",
        "note": "body {project, source, beats, segment?, name?, clip?}; `beats` "
        "are 0-based positions in that segment's storyboard (the numbering "
        "`beat:N` uses). The picked shots become a new bin item that has "
        "rendered nothing, each keeping its own length laid end to end from "
        "zero, on a budget rounded to the whole second. Style, sound, cast, "
        "references and subjects come across whole — it is the same scene; "
        "`first_frame`/`last_frame` are dropped unless the pick still opens or "
        "closes where they pointed. A copy, not a move: the original keeps its "
        "shots. Use it when one generation is being asked to hold more cuts "
        "than it can, or when a shot deserves a whole generation to itself.",
    },
    {
        "step": "plan a continuation",
        "method": "POST",
        "path": "/source/segment/add",
        "note": "body {project, source, prompt?}; adds one generation to the "
        "end of a source's chain. Intent only — nothing renders here.",
    },
    {
        "step": "reuse a character or location across sources",
        "method": "POST",
        "path": "/cast/save",
        "note": "body {project, member:{id?, name, kind, appearance, voice?, "
        "references?, notes?}}; omit `id` to mint one. `kind` is character / "
        "environment / prop / voice. Point a prompt at the member by putting "
        "its id in a line's `speakers` (they speak) or in the prompt's `cast` "
        "list (they do not). The appearance, voice and identity files are "
        "resolved in at compose time, so editing the member updates every "
        "segment naming it. GET /cast?project=NAME lists them; "
        "GET /cast/usage?project=NAME&member=ID says which segments would be "
        "stranded by a delete; POST /cast/delete removes one.",
    },
    {
        "step": "attach media (keyframe or reference)",
        "method": "POST",
        "path": "/reference/upload",
        "note": "get an image/video/audio into ComfyUI's input dir so it can "
        "drive a render. Send JSON {url} to fetch an http(s) file server-side, "
        "or a multipart file part to upload bytes. Returns {filename}; put that "
        "on the segment prompt's `first_frame`/`last_frame` (i2v) or a "
        "`references[].path` (ref2v). Render mode is then chosen automatically: "
        "keyframe -> i2v, references -> ref2v, neither -> t2v; both is a "
        "mode.conflict error.",
    },
    {
        "step": "parse composed prompt text into structured fields",
        "method": "POST",
        "path": "/parse",
        "note": "body {text, duration?}; returns {prompt, structured} so a "
        "frozen composed prompt can be restored into editable beats, style, "
        "soundscape and music.",
    },
    {
        "step": "refine or upscale an existing take",
        "method": "POST",
        "path": "/render/refine",
        "note": "body {source, take_id?, width, height, seconds?, frames?, "
        "seed?, prompt_text?, enhancement_mode?: h3|seedvr2|seedvr2_faces|redetail|redetail_pixel, "
        "enhancement_resolution?: original|1080p, redetail_scale?, guide_strength?, "
        "guided_resample?, face_refine?}. Builds a standalone refinement workflow "
        "linked to the parent take.",
    },
]


def schema() -> dict[str, Any]:
    """The whole contract as one JSON-serialisable object."""
    empty = ReferenceSet()
    return {
        "model": h3_spec.MODEL_NAME,
        "prompt_fields": PROMPT_FIELDS,
        "limits": {
            "duration_s": [h3_spec.MIN_DURATION_S, h3_spec.MAX_DURATION_S],
            "trained_duration_s": [
                h3_spec.TRAINED_MIN_DURATION_S,
                h3_spec.TRAINED_MAX_DURATION_S,
            ],
            "fps": h3_spec.FPS,
            "canvas_multiple": h3_spec.CANVAS_MULTIPLE,
            "max_pixels": h3_spec.MAX_PIXELS,
            "aspect_ratios": list(h3_spec.ASPECT_RATIOS),
            "reference_caps": {kind: empty.limit_for(kind) for kind in REFERENCE_KINDS},
            "min_shot_s": h3_spec.MIN_SHOT_S,
            "max_prompt_chars": h3_spec.MAX_PROMPT_CHARS,
            "recommended_prompt_chars": h3_spec.RECOMMENDED_PROMPT_CHARS,
        },
        "camera_vocabulary": {
            "motion": list(h3_spec.CAMERA_MOTIONS),
            "amplitude": list(h3_spec.CAMERA_AMPLITUDES),
            "speed": list(h3_spec.CAMERA_SPEEDS),
            "transitions": list(h3_spec.TRANSITIONS),
            "note": "motion type + amplitude + speed, written as natural English "
            "inside the shot. Amplitude and speed are stated only when they carry "
            "meaning; medium amplitude and normal speed are left unsaid.",
        },
        "dialogue_vocabulary": {
            "voiceover_verb": h3_spec.VOICEOVER_VERB,
            "continuity_phrases": list(h3_spec.CONTINUITY_ALTERNATIVES),
            "seam_tags": [h3_spec.SCENE_TRANSITION_TAG, h3_spec.CUTOFF_TAG],
        },
        "reference_vocabulary": {
            "task_types": list(h3_spec.REF_TASK_TYPES),
            "visual_retention": list(h3_spec.VISUAL_RETENTION),
            "audio_retention": list(h3_spec.AUDIO_RETENTION),
        },
        "lint_rules": [
            {"rule": rule, "severity": str(_SEVERITIES[rule]), "meaning": meaning}
            for rule, meaning in LINT_RULES.items()
        ],
        "operations": OPERATIONS,
    }
