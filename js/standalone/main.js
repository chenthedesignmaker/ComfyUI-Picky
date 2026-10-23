import { setupEditorUX } from "./editor-ux.js";
let editorUX = null;
import { createShotIdeas } from "./shot-ideas.js";
import { previewMatches } from "./shot-ideas-data.js";
import { openCameraPlanner, cameraPathSummary } from "./camera-path.js";
import {
  PROMPT_FIELDS,
  promptSections,
  promptSectionView,
  diffMark,
  diffRun,
  sectionedDiff,
  diffWords,
  setPromptSectionToast,
} from "./prompt-sections.js";
import {
  getStageDescription,
  getNodeStage,
  resolveWorkflowNodeId,
  PIPELINE_GROUP_WEIGHTS,
  linkedWorkflowNodes,
  workflowAncestors,
  executableWorkflowNodes,
  pipelineGroupForNode,
  calculateOverallProgress,
  getJobStageInfo,
  applyProgressFill,
  stageMarkup,
} from "./pipeline-progress.js";
import {
  compareState,
  configureCompareModal,
  openCompareModal,
  closeCompareModal,
  stopSync,
  startSync,
  toggleComparePlay,
  stepCompareFrame,
  wireCompareModal,
  renderCompare,
} from "./compare-modal.js";

// Standalone editor. Talks only to /video_prompter/* plus ComfyUI's own
// /prompt, and imports nothing from the ComfyUI frontend so the same file
// works served from either place.
//
// Two things are deliberately *not* computed here: the frame grid and the
// timeline arithmetic. Both come back from the server — /transform for the
// prompt, /sequence for the edit — so there is exactly one copy of each.
//
// The layout is an NLE: the bin on the left — one row per source, which is one
// lineage — a program monitor for preview, and a timeline of clips for playback
// and fine-tuning. Authoring and generating happen in the editor window, opened
// by double-clicking either.

// uuid() requires a secure context (HTTPS or localhost). Over
// plain HTTP on a Tailscale/LAN hostname it is undefined, so fall back to a
// simple v4 UUID from the CSPRNG which is always available.
const uuid = typeof crypto.randomUUID === "function"
  ? () => crypto.randomUUID()
  : () => ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
      (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16));

const API = "/video_prompter";
const RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
const MAX_COMPARE = 4;

// Render-quality presets, mirroring `video_prompter/quality.py`. A segment renders
// cheap while a prompt is being found, then once at full quality for the keeper.
// These are the built-in defaults; a project can override the numbers (not the
// set of names) via `project.settings.quality_presets`.
const QUALITY_DEFAULTS = {
  preview:        { megapixels: 0.2, steps: 8,  turbo_strength: 1.0, reference_size: "match" },
  turbo_hd:       { megapixels: 1.0, steps: 8,  turbo_strength: 1.0, reference_size: "match" },
  // The same eight turbo steps as turbo_hd, spent across two canvases. The
  // count is held equal on purpose: it is an A/B of WHERE the steps go.
  turbo_2pass:    { megapixels: 0.2, steps: 8,  turbo_strength: 1.0, reference_size: "max" },
  // SelfLift progressive sampling tiers
  selflift_hd:    { megapixels: 0.2, steps: 8,  turbo_strength: 1.0, reference_size: "max" },
  selflift_fhd:   { megapixels: 0.5, steps: 8,  turbo_strength: 1.0, reference_size: "max" },
  selflift_final: { megapixels: 0.5, steps: 20, turbo_strength: 0.0, reference_size: "max" },
  final:          { megapixels: 1.0, steps: 25, turbo_strength: 0.0, reference_size: "max" },
  // Two-pass: `megapixels` is pass 1's canvas and `steps` the whole schedule
  // across both passes. REFINE_DEFAULTS carries the rest.
  final_2pass:    { megapixels: 0.2, steps: 20, turbo_strength: 0.0, reference_size: "max" },
  singularity_2pass: { megapixels: 0.5, steps: 24, turbo_strength: 0.0, reference_size: "max" },
};

// How large identity references may reach the model, per tier. `match` scales
// them to the tier's own canvas — which at 0.2 MP leaves very little face to
// work from — and `max` sends a 2048px short edge. Mirrors quality.py.
const REFERENCE_SIZES = ["match", "max"];

// Tiers that render in two passes: most of the schedule at the small canvas
// where H3 actually obeys the prompt, then the clean latent upscaled and the
// last few steps spent on texture. `pass1_steps` cuts the tier's own step count
// (20 split at 15 leaves 5 to finish with); `megapixels` is the finished size.
const REFINE_DEFAULTS = {
  // 6 + 2, not 4 + 4: the latent upscaler needs the handover to happen at a
  // low enough sigma, and at shift 12 an 8-step run only reaches 0.80 — the
  // point final_2pass hands over at — after six steps. Cut at four and the
  // upscaler magnifies a 92%-noise guess. See quality.py.
  turbo_2pass:    { pass1_steps: 6, megapixels: 1.0 },
  selflift_hd:    { pass1_steps: 6, megapixels: 1.0 },
  selflift_fhd:   { pass1_steps: 6, megapixels: 2.1 },
  selflift_final: { pass1_steps: 15, megapixels: 2.1 },
  final_2pass:    { pass1_steps: 15, megapixels: 1.0 },
  singularity_2pass: { pass1_steps: 20, megapixels: 2.1 },
};

/** Effective quality numbers for a preset level, with project overrides. */
function resolveQuality(level) {
  const fallback = QUALITY_DEFAULTS[level] ?? QUALITY_DEFAULTS.preview;
  const overrides = (state.project?.settings?.quality_presets ?? {})[level];
  if (!overrides || typeof overrides !== "object") return { ...fallback };
  const mp = parseFloat(overrides.megapixels);
  const st = parseInt(overrides.steps, 10);
  const ts = parseFloat(overrides.turbo_strength);
  const rs = overrides.reference_size;
  return {
    megapixels: mp > 0 ? mp : fallback.megapixels,
    steps: st > 0 ? st : fallback.steps,
    turbo_strength: ts >= 0 ? ts : fallback.turbo_strength,
    reference_size: REFERENCE_SIZES.includes(rs) ? rs : fallback.reference_size,
  };
}

/** Two-pass numbers for a level, or null when the level renders in one pass. */
function resolveRefine(level) {
  const fallback = REFINE_DEFAULTS[level];
  if (!fallback) return null;
  const overrides = (state.project?.settings?.quality_presets ?? {})[level];
  const split = parseInt(overrides?.pass1_steps, 10);
  const mp = parseFloat(overrides?.refine_megapixels);
  const total = resolveQuality(level).steps;
  return {
    // Clamped for the reason quality.py clamps it: a split at or past the whole
    // schedule leaves pass 2 with no sigmas to run.
    pass1_steps: Math.min(split > 0 ? split : fallback.pass1_steps, Math.max(total - 1, 1)),
    megapixels: mp > 0 ? mp : fallback.megapixels,
  };
}

/** The canvas a level's finished file lands at — pass 2's, for a two-pass tier. */
function outputMegapixels(level) {
  return resolveRefine(level)?.megapixels ?? resolveQuality(level).megapixels;
}

/** A tier's cost as a phrase: the canvas and the steps it will actually spend.
 *
 * A two-pass tier has two of each, and reporting only the first would read a
 * keeper as a draft — so it says both halves of the schedule. Written once here
 * because three places quote it now: the budget hint, the render bar's tier
 * picker and the take cards' re-render menu.
 */
function tierNumbers(level) {
  const preset = resolveQuality(level);
  const refine = resolveRefine(level);
  return refine
    ? `${preset.megapixels} MP × ${refine.pass1_steps} steps → ` +
      `${refine.megapixels} MP × ${preset.steps - refine.pass1_steps} steps`
    : `${preset.megapixels} MP · ${preset.steps} steps`;
}

// -- what a take was rendered at ------------------------------------------
//
// A take does not record the preset it was queued under. The name lives on the
// segment, and a segment's quality moves on the moment the next render is set
// up — so a stored name would go stale and lie. What the take *does* record is
// the canvas it rendered at and the steps it took, which are exactly the two
// knobs a preset turns. Reading the tier back off those is therefore not a
// guess: it is the render describing itself.
//
// Nearest preset in both knobs wins. Distance is measured as a ratio, not a
// difference: 0.4 MP is twice a preview and less than half a turbo, which is
// the comparison that matches how the knobs actually scale, and it puts a
// megapixel and a step count on one scale so they can be added. Ties go to the
// cheaper tier — with no step count recorded an HD take could be either turbo
// or final, and claiming the cheaper one is the honest under-read.
//
// Canvas and steps stopped being enough the moment two tiers spent the same
// budget: `turbo_2pass` and `turbo_hd` both finish at 1.0 MP after 8 steps, so
// on those two numbers alone they are the same render. What separates them is
// `pass1_steps` — recorded on the take, and non-null exactly when the schedule
// was cut in two. So it is used first, as a filter on which tiers are even
// candidates, and only then is the nearest of those chosen. A take from before
// the field existed (null on a render that really was two-pass) falls back to
// the old behaviour rather than being excluded.
const TIERS = [
  "preview", "turbo_hd", "turbo_2pass",
  "selflift_hd", "selflift_fhd", "selflift_final",
  "final", "final_2pass", "singularity_2pass",
];
const ALL_TIERS = [...TIERS, "refined"];
const TIER_LABELS = {
  preview: "Preview", turbo_hd: "Turbo HD", turbo_2pass: "Turbo 2-pass",
  selflift_hd: "SelfLift HD", selflift_fhd: "SelfLift FHD", selflift_final: "SelfLift Final",
  final: "Final", final_2pass: "Final 2-pass",
  singularity_2pass: "Singularity 2-pass",
  refined: "Refined",
};

// What each tier is FOR, in one line. The numbers say what a tier costs; these
// say when to reach for it, which is the part a megapixel count cannot tell
// you. Condensed from the reasoning in quality.py, and quoted wherever a tier
// has to be chosen rather than merely reported.
const TIER_BLURBS = {
  preview: "The obedient tier. A small canvas is where H3 actually follows the " +
    "prompt, so prove the cuts, dialogue and timing here before paying for framing.",
  turbo_hd: "Fast HD draft. For hunting a seed once the prompt itself is settled.",
  turbo_2pass: "Turbo HD's eight steps, spent half small and half large. Same " +
    "cost, composed where H3 obeys. The tier to reach for when a turbo draft " +
    "keeps missing the direction.",
  selflift_hd: "Fast 720p progressive lift. The first steps compose at half " +
    "resolution, where H3 obeys camera and timing, then finish at 720p with TST consistency.",
  selflift_fhd: "1080p (1920×1088) progressive lift. Composes at half resolution, " +
    "finishes at 1080p in 8 Turbo steps. Heavy on GPU memory for long clips.",
  selflift_final: "1080p (1920×1088) keeper with the base model (Turbo bypassed). " +
    "20 progressive steps with TST temporal coherence. Heavy on GPU memory for long clips.",
  final: "The keeper. Full base model at 720p, turbo LoRA bypassed.",
  final_2pass: "The keeper that refuses the trade-off: most of the schedule at the " +
    "small canvas where H3 obeys, then upscaled and finished for texture. " +
    "Costs about what Fine costs and keeps a preview's adherence.",
  singularity_2pass: "1080p Singularity Ref2va 2-pass recipe: 20 base steps at half-res, " +
    "clean 3D latent upscale, then 4 Turbo steps at 0.35 denoise for realism and fine texture.",
};

// The glyph each tier has always worn on the render bar. The four tier buttons
// have collapsed into one, and it wears the icon of whichever tier is chosen —
// so the shape you learned to reach for still means the same thing.
const TIER_ICONS = {
  preview: "sparkles", turbo_hd: "zap", turbo_2pass: "zoom-in",
  selflift_hd: "activity", selflift_fhd: "monitor", selflift_final: "layers",
  final: "maximize", final_2pass: "layers", singularity_2pass: "cpu",
  refined: "sparkles",
};

/** The preset a take rendered at, or null when it cannot be told.
 *
 * A recorded name wins outright. Reading the tier back off the canvas and the
 * step count works for a fresh generation — each tier has its own signature
 * there — and cannot work for a continuation, which renders at the canvas of
 * the clip it continues rather than its own tier's: a Quick extend of a 720p
 * parent is 1.0 MP at 8 steps, which is Turbo HD's signature exactly, and that
 * is what the badge used to say. Everything below is the fallback for takes
 * rendered before the name was recorded.
 */
function tierOf(take) {
  if (take?.quality === "refined" || (take?.refinement && take.refinement.length > 0)) return "refined";
  if (QUALITY_DEFAULTS[take?.quality]) return take.quality;

  const mp = take?.width > 0 && take?.height > 0 ? (take.width * take.height) / 1e6 : null;
  const steps = take?.steps > 0 ? take.steps : null;
  if (mp === null && steps === null) return null;

  // null means "not recorded", which cannot rule anything out; a number means
  // this render was two-pass, and 0/absent-but-recorded is not a thing we write.
  const split = take?.pass1_steps > 0 ? take.pass1_steps : null;
  const candidates = split === null
    ? TIERS
    : TIERS.filter((level) => Boolean(resolveRefine(level)));

  let best = null;
  let bestScore = Infinity;
  for (const level of candidates) {
    const preset = resolveQuality(level);
    // A two-pass take records the size it FINISHED at, not the canvas its first
    // pass composed on, so that is the number to compare against.
    const canvas = outputMegapixels(level);
    let score = 0;
    if (mp !== null) score += Math.abs(Math.log(mp / Math.max(canvas, 0.001)));
    if (steps !== null) score += Math.abs(Math.log(steps / Math.max(preset.steps, 1)));
    // Among two-pass tiers, where the cut fell is a third knob on the same
    // ratio scale — it is what tells a 4+4 turbo from a 15+5 keeper when the
    // step totals have been overridden close to one another.
    const refine = resolveRefine(level);
    if (split !== null && refine)
      score += Math.abs(Math.log(split / Math.max(refine.pass1_steps, 1)));
    if (score < bestScore) { bestScore = score; best = level; }
  }
  return best;
}

/** A take's schedule as a phrase: "8 steps", or "4 + 4 steps" when it was cut.
 *
 * `steps` on a two-pass take is the WHOLE schedule, so printing it bare would
 * read a 4 + 4 turbo as if it had spent all eight at full size — which is the
 * one thing the tier exists to be distinguished from.
 */
function stepsPhrase(take) {
  if (!(take?.steps > 0)) return null;
  const split = take?.pass1_steps > 0 && take.pass1_steps < take.steps
    ? take.pass1_steps : null;
  return split === null
    ? `${take.steps} steps`
    : `${split} + ${take.steps - split} steps`;
}

/** A delivery name for a take's size, by its short edge: 4K, 1080p, 736p. */
function outputSizeLabel(take) {
  const short = Math.min(take?.width || 0, take?.height || 0);
  if (!short) return "";
  if (short >= 2160) return "4K";
  if (short >= 1440) return "1440p";
  if (short >= 1080) return "1080p";
  return `${short}p`;
}

/** The tier as a sentence, with the numbers behind it. */
function tierTitle(take) {
  const tier = tierOf(take);
  const mp = take?.width > 0 && take?.height > 0
    ? `${take.width}×${take.height} (${((take.width * take.height) / 1e6).toFixed(2)} MP)`
    : "canvas not recorded";
  const steps = stepsPhrase(take) ?? "steps not recorded";
  if (!tier) return `Render quality unknown — ${mp}, ${steps}`;
  // Worth distinguishing, because a recorded tier and its numbers can honestly
  // disagree: a continuation renders at its parent's canvas, so a Quick extend
  // of a 720p clip really is 1.0 MP. Saying which one is being reported stops
  // that reading as a bug.
  const known = QUALITY_DEFAULTS[take?.quality]
    ? "as asked for" : "read from the numbers";
  return `${TIER_LABELS[tier]} render (${known}) — ${mp}, ${steps}`;
}

/** Does this take pass the tier filter? Unreadable takes are never hidden. */
function passesTierFilter(take) {
  const tier = tierOf(take);
  if (!tier) return true;
  if (tier === "refined") {
    return state.tierFilter.has("refined") || TIERS.every((t) => state.tierFilter.has(t));
  }
  return state.tierFilter.has(tier);
}

// The rating buckets, most stars first, with 0 standing for "not rated yet".
// Unrated is a bucket rather than an absence because it is the one a review
// actually starts from: the takes still to be judged.
const RATINGS = [5, 4, 3, 2, 1, 0];

const STAR = "★";
const HOLLOW = "☆";
const TICK = "✓";

/** A rating drawn as five stars, filled up to it. */
const ratingLabel = (rating) =>
  STAR.repeat(rating) + HOLLOW.repeat(5 - rating);

/** A rating as it reads inside a sentence. */
const ratingWords = (rating) =>
  rating ? `${rating} star${rating === 1 ? "" : "s"}` : "unrated";

/** The bucket a take falls in. A missing or zero rating is "unrated". */
const ratingOf = (take) => Math.min(5, Math.max(0, Math.round(take?.rating || 0)));

/** Does this take pass the rating filter? */
function passesStarFilter(take) {
  return state.starFilter.has(ratingOf(take));
}

/**
 * Both filters at once — quality AND rating.
 *
 * They narrow the same sheet along two axes, and every place that used to ask
 * about the tier alone wants the pair now. Kept as one call so a third axis is
 * a change here rather than a change at six call sites.
 */
function passesFilters(take) {
  return passesTierFilter(take) && passesStarFilter(take);
}

/** The weakest tier along a run of takes — what a chain is really worth. */
function weakestTier(takes) {
  let worst = null;
  for (const take of takes) {
    const tier = tierOf(take);
    if (!tier) continue;
    const rank = ALL_TIERS.indexOf(tier);
    if (rank < 0) continue;
    if (worst === null || rank < ALL_TIERS.indexOf(worst)) worst = tier;
  }
  return worst;
}

/** A coloured dot for a tier, for the places too tight for a whole pill. */
function tierDotHTML(tier) {
  return `<span class="tier-dot" data-tier="${tier ?? "unknown"}"></span>`;
}

const state = {
  projects: [],
  // Every project on every shelf, as the manager lists them. The picker still
  // only offers `projects` -- the active names -- so archiving something really
  // does take it out of the way.
  projectEntries: [],
  projectShelf: "active",
  projectQuery: "",
  project: null,
  report: null,
  takes: [],
  // The edit selects a clip; the bin selects a source. A clip always implies
  // its source, but a bin item with nothing on the timeline has no clip, so
  // the two are kept apart rather than derived from one another.
  selectedClip: null,
  selectedSource: null,
  // Shots ticked to be split off into a bin item of their own, as indices into
  // the open segment's storyboard. Session-only, and cleared whenever those
  // indices could have moved underneath it — a different source, a different
  // generation, a shot added or removed.
  splitPick: new Set(),
  // Which generation of the selected source the author form is editing. Zero
  // for everything that has not been extended, which is most things.
  depth: 0,
  // Frames of motion context the author last chose deliberately, or null while
  // they have not. The footer rebuilds on plenty of unrelated events, so the
  // choice lives here rather than in the control, which would reset with it.
  extendContext: null,
  // The tier the author last chose for a continuation, or null while they have
  // not chosen one. Same reasoning as `extendContext`: the controls are rebuilt
  // by plenty of unrelated events, so the choice cannot live in the control.
  extendQuality: null,
  // A seed the author has pinned, so the next render reuses its starting noise
  // instead of rolling fresh — the way to tweak a prompt while holding the
  // composition. Session-only on purpose: it is a scratch choice made while
  // iterating on one shot, not something the project should remember and
  // silently reapply days later. Null means "roll a fresh one", the default.
  seedPin: null,
  // How many seeds the next render covers. 1 is a single render; more is a
  // sweep, queued at consecutive seeds off one base. It rides beside the seed
  // pin rather than belonging to any one tier button, because "how many
  // candidates" is a separate question from "how hard to work on each" — which
  // is what let the old preview-only Sweep button be replaced by a modifier
  // every tier honours. Session-only for the same reason as the pin.
  sweepCount: 1,
  // Whether the next render goes to the overnight shelf instead of ComfyUI's
  // live queue. A modifier, like the two above, and for the same reason: hold
  // used to be one button locked to Fine, so a held sweep, a held continuation
  // or a held re-shoot were all unreachable. Read at the dispatch funnel, so
  // every render path honours it without being told.
  //
  // A sticky mode is forgettable, which is why nothing here is subtle: every
  // render button renames itself to "Hold ..." while this is on, and a job sent
  // to the live queue by mistake can be moved onto the shelf from the queue
  // tracker afterwards. Session-only, like the pin and the count.
  holdRenders: false,
  // The tier a take card's re-render button repeats. Turbo HD and Fine used to
  // be a button each on every card, which is why the two-pass keeper never got
  // one there: a tier per button means every card grows by one every time a
  // tier is added. One split button reads this instead, and the menu beside it
  // sets it. Starts at `final`, so the first press does exactly what the old
  // Fine button on that card did. Session-only, like the modifiers above.
  reshootTier: "final",
  // Who is in the open segment, as `renderCast` last drew it. The cast tiles
  // are only rebuilt when this changes.
  castSignature: "",
  // The member whose detail sheet is open, by id. Empty means no sheet.
  castDetail: "",
  profiles: [],
  presets: {},
  refKinds: [],
  // Generated images for the open project, and the look presets the generator
  // offers. The library is fetched when the studio opens rather than with the
  // project: it is a sheet you visit, not something on screen by default.
  stills: [],
  styles: [],
  // What the open image studio is aimed at, while it is open. Null otherwise.
  studio: null,
  // A draft image prompt per destination, so the words written for shot two's
  // keyframe are still there after a detour through the courier's portrait.
  // Keyed by the same string the destination control uses. Session-only: a
  // prompt that generated something is in the library, and one that did not is
  // a thought, not a document.
  studioDrafts: {},
  // How the library is narrowed, and which images are held for comparison.
  studioFav: false,
  studioQuery: "",
  studioCompare: new Set(),
  studioComparing: false,
  // Past versions of the image prompt, and which one the sheet has selected.
  // Unlike the segment history this is one trail per project, because an image
  // prompt is not owned by anything smaller.
  imageHistory: { revisions: [], picked: null },
  // Which half of the bin is showing: the sources the edit is cut from, or the
  // images generated for the project.
  binView: "shots",
  binQuery: "",
  // The set the lightbox is walking and where in it, while it is open.
  lightbox: null,
  // Whether the server has a model key, and which model it names. Asked once at
  // boot: the buttons need the yes/no to know whether to offer themselves, and
  // a key cannot appear without ComfyUI restarting anyway.
  llm: { configured: false, model: "" },
  // What the open writer overlay is aimed at, while it is open.
  asking: null,
  // Whether a rewrite queues a render behind it. Kept here rather than read
  // off a checkbox because there are two of them, in two places.
  llmRender: false,
  // The revisions last fetched for the open segment, and which one is selected.
  history: { revisions: [], current: null, picked: null },
  // Reference tags for the current segment, from the last /preview (server owns
  // the <Audio j> numbering, so the editor never recomputes it).
  refTags: [],
  // The `<Subject N>` list as the server resolved it — cast members folded in,
  // numbered. Owned there for the same reason the tags are: working it out in
  // two places is how the two drift.
  subjects: [],
  // Which member the "from the project" picture grid is filling, when it is up.
  picker: null,
  workflowOwnsCanvas: false,
  // Renders this tab submitted and is still waiting on.
  jobs: new Map(),
  runningJob: null,
  pendingJobs: [],
  heldJobs: [],
  jobHistory: [],
  inspectedJob: null,
  queueTicker: null,
  clientId: sessionStorage.getItem("vp_client_id") || uuid(),
  compare: new Set(),
  player: null,
  // Take sheet: "takes" is the flat candidate list, "lineage" the branch tree,
  // "inspect" one take read back. All three are the same column, so none of
  // them can cover the fields being written on the left.
  takesView: "takes",
  // The take the inspect view is showing, and the source it belongs to. Held
  // so switching back to Inspect returns to it instead of an empty pane.
  inspectedTake: null,
  inspectedSource: null,
  // Redraws the inspected take's diff against whatever the fields say now.
  inspectRefresh: null,
  // Whether the take itself is unfolded. Sticky across takes, so comparing
  // seeds or step counts one after another costs one click, not one each.
  inspectDetail: false,
  // Which of the inspect view's tabs is showing; also sticky across takes.
  inspectTab: "original",
  // Which render tiers the take sheet shows. Both views read it, so narrowing
  // to the finals in Takes leaves Lineage narrowed to the chains that have any.
  tierFilter: new Set(ALL_TIERS),
  // Which ratings the take sheet shows, by the same rule and read in the same
  // places. Comparing shots is mostly "show me the fours and fives" or "what
  // have I not judged yet", which the tier axis cannot answer.
  starFilter: new Set(RATINGS),
  // Narrow the take sheet to takes nobody has opened yet. Off by default and
  // never sticky across projects — it is a way to find the batch that just
  // landed, not a mode to work in.
  newOnly: false,
  // The bin's own new-only filter. Separate from `newOnly`, which narrows the
  // take sheet inside one source: these two answer different questions ("which
  // source got something?" against "which take of this one?") and are read at
  // different moments, so one switch driving both would surprise.
  binNewOnly: false,
  lineage: null,
  // Chosen path per family, keyed by root take id. Owned here so switching
  // branches never needs a round trip — adopting one writes it to the source.
  lineagePaths: {},
  dirty: false,
  editRevision: 0,
  projectEpoch: 0,
  sequenceRequest: 0,
  // Timeline view: zoom is a multiplier of the fit-to-window scale; playhead
  // is the current position in timeline seconds.
  zoom: 1,
  playheadSec: 0,
  editorOpen: false,
  inspectorOpen: false,
  reviewOpen: false,
  previewRequest: 0,
  queueTrackerOpen: false,
  jobInspectorOpen: false,
  queueActiveTab: "all",
  queueFilterQuery: "",
  // When the user manually edits the composed prompt preview, the edited text
  // overrides compose() for the next render. Reset on a change of segment or
  // any form edit.
  previewDirty: false,
  previewEditedText: "",
  // Prompt tab: tinted (a map of which field wrote what, read-only) or plain
  // (the editable text). Muted kinds keep their colour out of the way.
  promptTint: true,
  hasSpans: false,
  mutedKinds: new Set(),
  // Inspectors: show a finished take's recorded prompt as labelled sections
  // (the default) or as the one flat block of text it is stored as.
  promptSectionsOn: true,
};
sessionStorage.setItem("vp_client_id", state.clientId);

const $ = (id) => document.getElementById(id);

// Close an overlay when its backdrop is clicked -- but only when the press both
// started and ended there. A `click` fires on the nearest ancestor shared by
// mousedown and mouseup, so dragging a text selection out of the sheet and
// letting go over the backdrop otherwise reads as a backdrop click and throws
// the sheet (and the edit in progress) away.
function closeOnBackdrop(overlay, close) {
  let pressed = false;
  let released = false;
  overlay.addEventListener("mousedown", (event) => { pressed = event.target === overlay; });
  overlay.addEventListener("mouseup", (event) => { released = event.target === overlay; });
  overlay.addEventListener("click", (event) => {
    const onBackdrop = pressed && released && event.target === overlay;
    pressed = released = false;
    if (onBackdrop) close();
  });
}

// Inline icon markup that resolves against the <symbol> sprite in index.html.
// `iconHTML` for template strings; `label` pairs an icon with text so a button
// reads as glyph + word rather than one or the other.
const iconHTML = (name, cls = "") =>
  `<svg class="ico ${cls}" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
const label = (name, text, cls = "") => `${iconHTML(name, cls)}<span>${escapeHtml(text)}</span>`;

const BLOCKS = ["style", "overview", "camera", "audio", "music", "ref_summary"];
const LIST_BLOCKS = ["screen_text", "negatives"];

// ref2v retention markers, mirroring h3_spec. Visual for pictures/videos,
// audio for standalone audio and reference-video soundtracks.
const VISUAL_RETENTION = ["fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"];
const AUDIO_RETENTION = ["fully_copy", "partially_copy", "reference", "weak_reference"];

// What each marker actually asks for. The words are the model's, not English —
// "attribute_transfer" tells you nothing until someone says it means "borrow the
// qualities, not the thing", so the editor says it, on hover and in a line under
// the control.
const RETENTION_HELP = {
  fully_preserved: "Reproduce it exactly. For a face, a costume, a set that has "
    + "to match other shots.",
  partially_preserved: "Keep it recognisably the same, but let it move, re-light "
    + "and re-frame.",
  attribute_transfer: "Borrow its qualities — colour, material, layout, mood — "
    + "not the thing itself. For a floor plan, a palette, a texture.",
  weak_reference: "A loose hint only. The model may depart from it freely.",
  fully_copy: "Reproduce this audio as it is.",
  partially_copy: "Keep it recognisable, but let it vary.",
  reference: "Use it as a guide to the sound, not as the sound itself.",
};

// The cut phrases a later shot may open with (h3_spec.TRANSITIONS). Shot 1 has
// no cut into it, so its field composes nothing.
const TRANSITIONS = [
  "the camera cuts to",
  "the shot cuts to",
  "the shot transitions to",
  "the shot changes to",
  "the shot switches to",
];

// The camera vocabulary H3 was trained on (h3_spec). Motion is written into the
// shot as English, so these build a phrase rather than replacing the field:
// medium amplitude and normal speed are said by leaving them out.
const CAMERA_MOTIONS = [
  "Zoom In", "Zoom Out", "Push In", "Pull Out", "Pan Left", "Pan Right",
  "Truck Left", "Truck Right", "Tilt Up", "Tilt Down", "Pedestal Up",
  "Pedestal Down", "Arc Shot", "Tracking Shot", "Static Shot",
  "Shake Slightly", "Shake Strongly", "POV", "Roll Clockwise",
  "Roll Counterclockwise",
];
// The verb form each motion takes inside a sentence ("the camera pushes in …").
const CAMERA_VERBS = {
  "Zoom In": "zooms in", "Zoom Out": "zooms out",
  "Push In": "pushes in", "Pull Out": "pulls out",
  "Pan Left": "pans left", "Pan Right": "pans right",
  "Truck Left": "trucks left", "Truck Right": "trucks right",
  "Tilt Up": "tilts up", "Tilt Down": "tilts down",
  "Pedestal Up": "pedestals up", "Pedestal Down": "pedestals down",
  "Arc Shot": "arcs around the subject", "Tracking Shot": "tracks with the subject",
  "Static Shot": "holds still", "Shake Slightly": "shakes slightly",
  "Shake Strongly": "shakes strongly", "POV": "takes the subject's point of view",
  "Roll Clockwise": "rolls clockwise", "Roll Counterclockwise": "rolls counterclockwise",
};
const CAMERA_AMPLITUDES = ["with small amplitude", "with large amplitude"];
const CAMERA_SPEEDS = ["at slow speed", "at fast speed"];

// ref2v summary task types (h3_spec.REF_TASK_TYPES). A short fixed set, which is
// why mistyping one has its own lint rule — so pick, do not type.
const REF_TASK_TYPES = [
  "keyframe completion",
  "reference generation",
  "video editing",
  "video continuation",
  "audio reuse",
  "audio reference",
];

// No spec list — H3 takes the language name as written. These are the ones worth
// one click; the field stays free text for everything else.
const COMMON_LANGUAGES = [
  "English", "Mandarin", "Cantonese", "Japanese", "Korean", "Spanish",
  "French", "German", "Italian", "Portuguese", "Russian", "Hindi", "Arabic",
];

// -- transport ------------------------------------------------------------

async function call(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (typeof editorUX !== "undefined" && editorUX && options.method === "POST") {
    if (response.ok && path === `${API}/gallery/pick`) editorUX.tracker.outcome("take_chosen");
    if (!response.ok && response.status === 422 && path.startsWith(`${API}/render/`)) editorUX.tracker.outcome("render_blocked");
  }
  if (!response.ok) {
    const error = new Error(body.error || `${response.status} ${response.statusText}`);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return body;
}

// Every write carries this tab's client id as `origin`. The server echoes it
// on its change notification, so this tab can ignore the live-sync event that
// its own write triggered (an agent or another tab omits it, so their writes
// do reach us). Non-mutating routes ignore the extra key.
const post = (path, payload, options = {}) =>
  call(path, {
    method: "POST",
    body: JSON.stringify({ origin: state.clientId, ...payload }),
    ...options,
  });

function toast(message, bad = false) {
  const element = $("toast");
  if (!bad && toast.badUntil && Date.now() < toast.badUntil && element.classList.contains("bad")) {
    return;
  }
  element.textContent = message;
  element.className = bad ? "show bad" : "show";
  toast.badUntil = bad ? Date.now() + 4000 : 0;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => {
    element.className = "";
    toast.badUntil = 0;
  }, bad ? 8000 : 2500);
}
setPromptSectionToast(toast);

const guard = (fn) => (...args) =>
  Promise.resolve(fn(...args)).catch((error) => toast(error.message, true));

// Local edits that the server has not been told about yet. A dot in the header
// says so, and `beforeunload` refuses to let a stray tab-close drop them.
function markDirty() {
  state.editRevision++;
  if (state.dirty) return;
  state.dirty = true;
  $("dirty").classList.add("on");
}

function clearDirty() {
  state.dirty = false;
  $("dirty").classList.remove("on");
}

// -- model ----------------------------------------------------------------
//
// Two lists, and they are not the same list. `sources` is the bin: one lineage
// each, holding the segments that planned it and the `path` of take ids saying
// which take follows which. `timeline.clips` is the edit: windows into a
// source, in order. One source can back several clips.
//
// Nothing here infers a parent or pairs takes by position. A chain is whatever
// `path` says, and `path` is checked against what the renderer recorded.

const sources = () => state.project?.sources ?? [];
const clips = () => state.project?.timeline?.clips ?? [];

const sourceById = (id) =>
  sources().find((source) => source.id === id || source.name === id) ?? null;
const clipById = (id) => clips().find((clip) => clip.id === id) ?? null;

const currentSource = () => sourceById(state.selectedSource);
const currentClip = () => clipById(state.selectedClip);

/** The generation being authored. Depth 0 for anything never extended. */
function currentSegment() {
  const source = currentSource();
  if (!source?.segments?.length) return null;
  return source.segments[Math.min(state.depth, source.segments.length - 1)] ?? null;
}

const currentPrompt = () => currentSegment()?.prompt ?? null;

const clipReport = (id) => state.report?.clips?.find((entry) => entry.id === id) ?? null;

// -- mutes ----------------------------------------------------------------
//
// A mute is stored in lineage seconds so it rides the footage, and the server
// sends both clocks: `start`/`end` are the record to post back, `at`/`length`
// are where it falls inside the clip on the ruler. A band the current trim has
// pushed out of sight arrives with a null `at` — still on the record, nothing to
// draw — so every reader here filters on that rather than on the trim.

const drawableMutes = (clipId) =>
  (clipReport(clipId)?.mutes ?? []).filter((m) => m.at !== null && m.length > 0);

// Memoised against the report object itself: this is asked on every timeupdate,
// and a new report is the only thing that can change the answer.
let muteCache = { report: null, windows: [] };

/** Every drawable mute as an absolute range on the ruler. */
function muteWindows() {
  if (muteCache.report === state.report) return muteCache.windows;
  const windows = [];
  for (const entry of state.report?.clips ?? []) {
    if (entry.enabled === false) continue;
    for (const mute of entry.mutes ?? []) {
      if (mute.at === null || !(mute.length > 0)) continue;
      windows.push([entry.starts_at + mute.at, entry.starts_at + mute.at + mute.length]);
    }
  }
  muteCache = { report: state.report, windows };
  return windows;
}

/** Is this instant silenced by a band? The one question playback asks. */
const mutedAt = (sec) =>
  muteWindows().some(([from, to]) => sec >= from - 1e-6 && sec < to - 1e-6);
const sourceReport = (id) =>
  state.report?.sources?.find((entry) => entry.id === id || entry.name === id) ?? null;

/** Every clip backed by one source — what the bin needs to show usage. */
const clipsOf = (sourceId) => clips().filter((clip) => clip.source_id === sourceId);

/** The takes a source rendered. `take.shot` holds the source id; see api.py. */
const takesFor = (id) => {
  const source = sourceById(id);
  return state.takes.filter(
    (take) => take.shot === id || (source && (take.shot === source.id || take.shot === source.name)),
  );
};

const takeById = (id) => {
  if (!id) return null;
  return state.takes.find((take) => take.id === id || take.filename === id || take.id.endsWith(`:${id}`)) ?? null;
};

// -- what has been looked at ----------------------------------------------
//
// A render lands while attention is somewhere else, and a batch queued across
// ten sources comes back over ten minutes in no particular order. Without a
// mark, finding what is actually new means remembering what the bin looked
// like before — so the take itself carries the mark, server-side beside its
// rating, and the badge is drawn from that.
//
// Discarded takes are never new: rejecting one is looking at it.

/**
 * Has nobody opened this take yet?
 *
 * Strictly `false`, not merely falsy: a server too old to know about the flag
 * sends no `seen` at all, and reading that as new would badge every take in
 * the project until it was restarted.
 */
const isNewTake = (take) => Boolean(take) && take.seen === false && !take.discarded;

// Counted by id, not by record. A project's takes.json can hold the same take
// twice — older recording runs left duplicates behind — and a badge reading
// "4 new" over two takes is a badge that lies about how much there is to watch.
const countNew = (takes) => new Set(takes.filter(isNewTake).map((take) => take.id)).size;

/** How many takes on this source are still unopened. */
const newTakesFor = (id) => countNew(takesFor(id));

/** Every unopened take in the project — what the bin header counts. */
const newTakeCount = () => countNew(state.takes);

/**
 * Mark takes as looked at, on the server and on screen.
 *
 * The local flag is set first and the request is not awaited by the callers
 * that matter: marking is a side effect of clicking a card to watch it, and a
 * badge that lingers for a round trip after the click reads as a failed click.
 * A rejected write is reported and the flag rolls back.
 */
async function markSeen(ids, seen = true) {
  const wanted = new Set((Array.isArray(ids) ? ids : [ids]).filter(Boolean));
  // Every record carrying the id, not just the first: a duplicated take would
  // otherwise leave its twin unseen and the count stuck one above zero. The
  // server marks by id and does the same.
  const hits = state.takes.filter((take) => wanted.has(take.id) && take.seen !== seen);
  const changing = [...new Set(hits.map((take) => take.id))];
  if (!changing.length || !state.project) return 0;

  for (const take of hits) take.seen = seen;
  renderNewBadges();

  try {
    await post(`${API}/gallery/seen`, {
      project: state.project.name, takes: changing, seen,
    });
  } catch (error) {
    for (const take of hits) take.seen = !seen;
    renderNewBadges();
    toast(`Could not mark ${changing.length} take(s): ${error.message}`, true);
    return 0;
  }
  return changing.length;
}

/**
 * Redraw everywhere a new badge appears, without a round trip to the server.
 *
 * Take cards are patched in place rather than rebuilt: the usual way a take
 * stops being new is that someone pressed play on it, and rebuilding the grid
 * would replace the `<video>` under the click and stop the thing they came to
 * watch. For the same reason the new-only filter is not re-applied here — a
 * card that clears its badge stays put until the sheet is drawn again.
 */
function renderNewBadges() {
  renderBin();
  renderTimeline();
  for (const card of document.querySelectorAll(".candidate[data-take-id]")) {
    card.classList.toggle("is-new", isNewTake(takeById(card.dataset.takeId)));
  }
  for (const cell of document.querySelectorAll(".lin-cell[data-take-id]")) {
    cell.classList.toggle("is-new", isNewTake(takeById(cell.dataset.takeId)));
  }
  if (state.editorOpen || state.reviewOpen) renderNewFilter();
}

/** The lineage a source plays, as the server resolved it (choice or default). */
const pathOf = (sourceId) => sourceReport(sourceId)?.path ?? sourceById(sourceId)?.path ?? [];

/** The take a source stands on at one depth, or its last one. */
const takeAtDepth = (sourceId, depth) => takeById(pathOf(sourceId)[depth]);

const seconds = (value) => `${Number(value ?? 0).toFixed(2)}s`;
const fmtClock = (sec) => {
  sec = Math.max(0, sec || 0);
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};

// -- rendering the authoring form -----------------------------------------

// A shot is authored as a card, not a table row: composition, action and camera
// are prose, and prose in a 100px column is unreadable while you write it. The
// card carries its own dialogue too — H3 reads a spoken line inside the shot
// that speaks it, so authoring it anywhere else means jumping tabs to check
// one sentence against another.
//
// The fields run in the order the shot composes: the cut into it first — it is
// what `compose` writes right after the timecode — then composition, then
// subjects / environment / action, then camera.
// Disclosure is view state, kept outside the project and its prompt payload.
// Key by source + generation so opening a different scene never hides its text
// according to a previous scene's choices, and saves can rebind the same view.
const sceneOpenBySource = new Map();
let shotIdeas = null;

function shotIdeasSnapshot() {
  const prompt = currentPrompt();
  if (!prompt) return null;
  const numbers = shotNumbers(prompt.beats);
  const moves = moveNumbers(prompt.beats);
  return {
    key: JSON.stringify([state.projectEpoch, state.project?.name, state.selectedSource, state.depth]),
    name: currentSource()?.name || "",
    overview: prompt.overview || "",
    beats: structuredClone(prompt.beats),
    labels: prompt.beats.map((beat, index) => index > 0 && !cuts(beat)
      ? `Shot ${numbers[index]} · move ${moves[index]}` : `Shot ${numbers[index]}`),
  };
}

function commitShotIdea(preview) {
  if (!previewMatches(shotIdeasSnapshot(), preview)) return false;
  const beat = currentPrompt()?.beats[preview.index];
  if (!beat || !Object.entries(preview.patch).every(([key, value]) =>
    ["angle", "camera", "action"].includes(key) && typeof value === "string")) return false;
  Object.assign(beat, preview.patch);
  sceneOpenShots().add(preview.index);
  markDirty();
  renderBeats();
  refresh();
  return true;
}

function sceneOpenShots() {
  const key = `${state.project?.name}:${state.selectedSource}:${state.depth}`;
  if (!sceneOpenBySource.has(key)) sceneOpenBySource.set(key, new Set((currentPrompt()?.beats || []).map((_, i) => i)));
  return sceneOpenBySource.get(key);
}

function setSceneShotOpen(index, open) {
  const visible = sceneOpenShots();
  if (open) visible.add(index); else visible.delete(index);
  const card = $("beats").querySelector(`[data-beat="${index}"]`);
  if (card) {
    card.classList.toggle("expanded", open);
    card.querySelector(".beat-content").hidden = !open;
    card.querySelector(".beat-toggle").setAttribute("aria-expanded", String(open));
  }
  $("scene-navigator").querySelector(`[data-beat-jump="${index}"]`)
    ?.setAttribute("aria-expanded", String(open));
  const count = currentPrompt()?.beats.length || 0;
  $("scene-expand-all").textContent = count && Array.from({ length: count }, (_, i) => i).every(i => visible.has(i))
    ? "Collapse all" : "Expand all";
  if (open) refitGrowFields();
}

function jumpToSceneBeat(index) {
  setSceneShotOpen(index, true);
  const card = $("beats").querySelector(`[data-beat="${index}"]`);
  card?.scrollIntoView({ block: "start", behavior: "instant" });
  card?.querySelector(".beat-toggle").focus({ preventScroll: true });
}

function renderSceneNavigator(prompt) {
  const nav = $("scene-navigator");
  nav.textContent = "";
  const section = (name, target) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene-jump";
    button.textContent = name;
    button.onclick = () => $(target).scrollIntoView({ block: "start", behavior: "instant" });
    nav.append(button);
  };
  section("Foundation", "scene-overview");
  const beats = prompt?.beats || [];
  const numbers = shotNumbers(beats);
  const moves = moveNumbers(beats);
  beats.forEach((beat, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene-jump shot";
    button.dataset.beatJump = String(index);
    button.setAttribute("aria-controls", `scene-beat-${index}`);
    button.setAttribute("aria-expanded", String(sceneOpenShots().has(index)));
    button.textContent = index > 0 && !cuts(beat) ? `↳ Move ${moves[index]}` : `Shot ${numbers[index]}`;
    button.title = `${seconds(beat.start)}–${seconds(beat.end)} · ${beat.angle || beat.action || "Add direction"}`;
    button.onclick = () => jumpToSceneBeat(index);
    nav.append(button);
  });
  section("Frame text", "scene-output");
  $("scene-expand-all").disabled = !beats.length;
  $("scene-focus").disabled = !prompt;
  $("shot-ideas-open").disabled = !prompt;
  $("scene-expand-all").textContent = beats.length && beats.every((_, i) => sceneOpenShots().has(i))
    ? "Collapse all" : "Expand all";
}

function finishBeatCard(card, beat, index, shot, move, held) {
  card.classList.add("scene-beat");
  card.dataset.beat = String(index);
  const body = document.createElement("div");
  body.className = "beat-content";
  body.id = `scene-beat-${index}`;
  body.append(...card.childNodes);
  const header = document.createElement("div");
  header.className = "beat-heading";
  const grip = body.querySelector(":scope > .card-head > .card-grip");
  if (grip) header.append(grip);
  const toggle = document.createElement("button");
  toggle.className = "beat-toggle";
  toggle.type = "button";
  toggle.setAttribute("aria-controls", body.id);
  const title = document.createElement("span");
  title.className = "beat-name";
  title.textContent = held ? `${round3(beat.start)}s · Continuing move` : `Shot ${String(shot).padStart(2, "0")}`;
  const time = document.createElement("span");
  time.className = "beat-time";
  time.textContent = `${round3(beat.start)}–${round3(beat.end)}s · ${round3(beat.end - beat.start)}s`;
  const preview = document.createElement("span");
  preview.className = "beat-summary";
  const update = () => { preview.textContent = (beat.angle || beat.action || beat.camera || "Add framing, action and camera direction").trim(); };
  update();
  const arrow = document.createElement("span");
  arrow.className = "beat-chevron";
  arrow.innerHTML = iconHTML("chevron-down", "sm");
  toggle.append(title, time, preview, arrow);
  toggle.onclick = () => setSceneShotOpen(index, !sceneOpenShots().has(index));
  header.append(toggle);
  // The camera field can be folded away when empty. Keep its planner reachable
  // from the shot header, even when the entire shot body is collapsed.
  const pathButton = document.createElement("button");
  pathButton.type = "button";
  pathButton.className = "small ghost beat-camera-path";
  pathButton.textContent = "Camera path";
  pathButton.title = beat.camera_path ? "Edit this shot's camera path" : "Plan this shot's camera path";
  pathButton.setAttribute("aria-label", `Camera path for shot ${shot}${held ? ` move ${move}` : ""}`);
  pathButton.onclick = () => body.querySelector(".camera-path-tools button")?.click();
  header.append(pathButton);
  // Keep precision controls available without breaking the reading surface.
  const settings = document.createElement("div");
  settings.className = "beat-settings";
  settings.id = `scene-settings-${index}`;
  settings.hidden = true;
  const timing = body.querySelector(":scope > .card-head");
  const transition = body.querySelector(":scope > .card-field");
  settings.append(timing, transition);
  body.prepend(settings);
  const settingsButton = document.createElement("button");
  settingsButton.type = "button";
  settingsButton.className = "small ghost beat-settings-toggle";
  settingsButton.textContent = "Settings";
  settingsButton.setAttribute("aria-label", `Settings for shot ${shot}${held ? ` move ${move}` : ""}`);
  settingsButton.setAttribute("aria-controls", settings.id);
  settingsButton.setAttribute("aria-expanded", "false");
  settingsButton.onclick = () => {
    setSceneShotOpen(index, true);
    settings.hidden = !settings.hidden;
    settingsButton.setAttribute("aria-expanded", String(!settings.hidden));
    refitGrowFields();
  };
  header.append(settingsButton);
  const additions = document.createElement("details");
  additions.className = "scene-additions";
  const addTitle = document.createElement("summary");
  addTitle.textContent = "+ Add direction or dialogue";
  additions.append(addTitle);
  for (const field of body.querySelectorAll(":scope > .card-field")) {
    const control = field.querySelector("textarea");
    const optional = control?.dataset.field === `beat.${index}.camera`
      || (held && control?.dataset.field === `beat.${index}.angle`);
    if (!optional || control.value.trim()) continue;
    field.hidden = true;
    const add = addButton(field.querySelector("label").textContent, "Add this direction", () => {
      field.hidden = false;
      add.hidden = true;
      additions.open = false;
      refitGrowFields();
      control.focus();
    });
    additions.append(add);
  }
  for (const row of body.querySelectorAll(":scope > .line-add")) additions.append(row);
  body.append(additions);
  card.append(header, body);
  card.addEventListener("input", update);
  const open = sceneOpenShots().has(index);
  card.classList.toggle("expanded", open);
  body.hidden = !open;
  toggle.setAttribute("aria-expanded", String(open));
  return card;
}

function wireSceneEditor() {
  shotIdeas = createShotIdeas({ snapshot: shotIdeasSnapshot, commit: commitShotIdea });
  $("shot-ideas-open").onclick = () => shotIdeas.open();
  $("scene-expand-all").onclick = () => {
    const beats = currentPrompt()?.beats || [];
    const open = !beats.every((_, index) => sceneOpenShots().has(index));
    beats.forEach((_, index) => setSceneShotOpen(index, open));
  };
  $("scene-focus").onclick = () => {
    const editor = $("editor");
    const focused = editor.classList.contains("scene-focus");
    if (!state.editorOpen) openEditor(state.selectedSource);
    if (!focused) {
      stopSync();
      ensureTimeline();
      hushWithin($("review-pane"));
    }
    editor.classList.toggle("scene-focus", !focused);
    $("scene-focus").querySelector("span").textContent = focused ? "Focus" : "Split view";
    document.querySelector('#author-tabs [data-tab="scene"]').click();
    refitGrowFields();
  };
}

function renderBeats() {
  const prompt = currentPrompt();
  const body = $("beats");
  body.textContent = "";
  renderSplitButton();
  renderSceneNavigator(prompt);
  shotIdeas?.sync();
  if (!prompt) return;

  if (!prompt.beats.length) {
    body.innerHTML = `<div class="empty">No shots yet — the whole generation is one
      uncut take. Add one to cut inside it.</div>`;
    return;
  }

  tileBeats(prompt);
  wireBeatDrop(body, prompt);
  const shotOf = shotNumbers(prompt.beats);
  const moveOf = moveNumbers(prompt.beats);
  prompt.beats.forEach((beat, index) => body.append(beatCard(prompt, beat, index, shotOf, moveOf)));
  renderSceneNavigator(prompt);
}

// The shortest shot the editor will leave standing when a cut is moved onto
// its neighbour. Small enough never to get in the way, big enough that a shot
// cannot be squeezed to nothing by a typo.
const MIN_SHOT = 0.1;

const round3 = (value) => Math.round((Number(value) || 0) * 1000) / 1000;

// Shots tile the clip end to end: shot 1 opens it, the last one runs to the
// budget, and every boundary in between is a single number that the shot
// before it and the shot after it both read. Enforcing that here, on the way
// into the cards, is what makes "close the gaps" not a thing you have to ask
// for — a storyboard written elsewhere (by the prompting skill, or by an
// older project) reads as contiguous the moment it is opened.
//
// Every start stays exactly where its author put it and only the ends move,
// because an end is not really a number of its own — it is the next shot's
// start, which is the rule `parse` already applies reading a storyboard back.
// So a gap closes by growing the shot in front of it, and an overlap closes by
// trimming that shot back to the cut that overtook it. A shot left with no
// length at all keeps MIN_SHOT of it rather than vanishing from the list.
function tileBeats(prompt) {
  const beats = prompt.beats;
  if (!beats.length) return false;
  beats.sort((a, b) => (a.start - b.start) || (a.end - b.end));

  const limit = Math.max(Number(prompt.duration) || 0, 0);
  let changed = false;
  const set = (beat, key, value) => {
    if (beat[key] !== value) {
      beat[key] = value;
      changed = true;
    }
  };

  let cursor = 0;
  beats.forEach((beat, index) => {
    set(beat, "start", cursor);
    const next = beats[index + 1];
    cursor = next
      ? Math.min(Math.max(round3(next.start), round3(cursor + MIN_SHOT)), limit)
      : limit;
    set(beat, "end", cursor);
  });
  return changed;
}

// Move the cut that opens shot `index`. Cut 0 is the head of the clip and cut
// `beats.length` is its tail; neither belongs to a shot, so neither moves —
// the clip is as long as the budget says it is.
//
// This is a roll, not a ripple: the shot on one side of the cut grows by
// exactly what the shot on the other side loses, so the budget still adds up
// and no shot further down the storyboard is disturbed. Push a cut into a
// neighbour and it stops at that neighbour's far edge rather than eating it;
// to lose a shot entirely, remove the shot.
function setCut(prompt, index, value) {
  const beats = prompt.beats;
  if (index <= 0 || index >= beats.length) return;
  const floor = round3(beats[index - 1].start + MIN_SHOT);
  const ceiling = round3(beats[index].end - MIN_SHOT);
  const cut = round3(Math.min(Math.max(value, floor), Math.max(floor, ceiling)));
  beats[index - 1].end = cut;
  beats[index].start = cut;
}

// -- dragging a shot up or down the storyboard -----------------------------
//
// The list is ordered by time, so reordering it *is* retiming it: a card that
// moves carries its own length, and every cut after it rolls to suit. The
// budget is untouched — the shots still tile the generation end to end, in a
// new order.
//
// What moves is a shot, not a card. A held beat is a moment inside the shot
// above it, so dragging a shot head carries its moves with it and a continuous
// take arrives somewhere else intact. A held card drags alone, which is how one
// moment is handed to a different shot.
//
// A card is draggable only while its grip is held. The cards are mostly
// textareas, and a permanently draggable card is one you cannot select a
// sentence inside.

const DRAG_BEAT = "application/x-vp-beat";

// The run the pointer is carrying. Read during `dragover`, where `getData` is
// blocked, so it cannot live in the drag itself.
let draggingBeats = null;

/** True when this beat continues the one above rather than opening a shot. */
const heldBeat = (beats, index) => index > 0 && !cuts(beats[index]);

/** The run of beats one card drags: a shot and its moves, or one move alone. */
function beatSpan(beats, index) {
  if (heldBeat(beats, index)) return { at: index, count: 1, held: true };
  let end = index + 1;
  while (end < beats.length && !cuts(beats[end])) end += 1;
  return { at: index, count: end - index, held: false };
}

/** The insert points a span may land on, as indices into the beat list.
 *
 * A shot can only land between shots: dropping one into the middle of another
 * shot's moves would put a cut inside a take that is meant to be continuous. A
 * lone move can land anywhere but the head of the list, which has no shot above
 * it to be a moment of.
 */
function beatDropPoints(beats, span) {
  const points = beats.map((beat, index) => index).concat(beats.length);
  if (span.held) return points.slice(1);
  return points.filter((index) => index === beats.length || index === 0 || cuts(beats[index]));
}

/** Move a span to an insert point and re-lay the storyboard behind it. */
function moveBeats(prompt, span, to) {
  const beats = prompt.beats;
  const moving = beats.splice(span.at, span.count);
  // The insert point was read against the list with the span still in it, so
  // anything past the span's own place is that much lower once it is out.
  const target = span.at < to ? to - span.count : to;
  beats.splice(Math.max(0, Math.min(target, beats.length)), 0, ...moving);
  // Nothing opens the clip but the first beat, so it cannot be a continuation
  // of anything. Only reachable by dropping a shot above a held one.
  if (beats.length && !cuts(beats[0])) beats[0].cut = true;
  retimeBeats(prompt);
}

// Lay the shots out end to end in the order the list now stands in, each one
// keeping the length it had. Reordering never changes the budget, so the sum
// still lands on it and the last shot still ends where it did.
function retimeBeats(prompt) {
  const beats = prompt.beats;
  const limit = Math.max(Number(prompt.duration) || 0, 0);
  const lengths = beats.map((beat) => Math.max(MIN_SHOT, round3(beat.end - beat.start)));
  let cursor = 0;
  beats.forEach((beat, index) => {
    beat.start = round3(cursor);
    cursor = index === beats.length - 1
      ? limit
      : Math.min(round3(cursor + lengths[index]), limit);
    beat.end = round3(cursor);
  });
}

/** The grip that makes one card draggable, wired to the card it sits in. */
function beatGrip(card, beats, index) {
  card.dataset.beat = String(index);

  const grip = document.createElement("button");
  grip.type = "button";
  grip.className = "small ghost iconbtn card-grip";
  grip.innerHTML = iconHTML("grip", "sm");
  grip.title = heldBeat(beats, index)
    ? "Drag to move this moment into another shot"
    : "Drag to reorder this shot. Any moves held inside it come with it.";
  // Armed on the way down and disarmed the moment the button comes back up,
  // so the rest of the card is ordinary selectable text between drags. The
  // release is watched on the document rather than on the grip: a press that
  // wandered a few pixels without becoming a drag lets go somewhere else, and
  // a card left armed is one you cannot select a sentence inside.
  grip.onmousedown = () => {
    card.draggable = true;
    document.addEventListener("mouseup", () => { card.draggable = false; }, { once: true });
  };
  grip.onclick = (event) => event.preventDefault();

  card.ondragstart = (event) => {
    draggingBeats = beatSpan(beats, index);
    event.dataTransfer.setData(DRAG_BEAT, String(index));
    event.dataTransfer.effectAllowed = "move";
    card.classList.add("dragging");
  };
  card.ondragend = () => {
    draggingBeats = null;
    card.draggable = false;
    card.classList.remove("dragging");
    clearBeatCaret();
  };
  return grip;
}

// One handler for the whole list rather than one per card: the insert point is
// a position down the column, so the gap under the last shot is the same case
// as the boundary between two of them.
function wireBeatDrop(body, prompt) {
  const carrying = (event) =>
    [...(event.dataTransfer?.types ?? [])].includes(DRAG_BEAT);

  body.ondragover = (event) => {
    if (!carrying(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    showBeatCaret(body, beatDropAt(body, prompt, event.clientY));
  };
  body.ondragleave = (event) => {
    if (!body.contains(event.relatedTarget)) clearBeatCaret();
  };
  body.ondrop = (event) => {
    if (!carrying(event)) return;
    event.preventDefault();
    const span = draggingBeats;
    const to = beatDropAt(body, prompt, event.clientY);
    clearBeatCaret();
    // Both of a span's own edges resolve to where it already is.
    if (!span || to === null || (to >= span.at && to <= span.at + span.count)) return;
    moveBeats(prompt, span, to);
    // Every index from the move down just changed, so no tick still means what
    // it said. Clearing is the only honest answer, as it is on remove.
    clearSplitPick();
    markDirty();
    renderBeats();
    refresh();
  };
}

// Where a drop lands, as an index into the beat list. Cards carry their own
// index rather than being counted, and the raw answer is then snapped to the
// nearest point this particular span is allowed to occupy.
function beatDropAt(body, prompt, clientY) {
  if (!draggingBeats) return null;
  const cards = [...body.querySelectorAll(":scope > .card[data-beat]")];
  let raw = prompt.beats.length;
  for (const card of cards) {
    const box = card.getBoundingClientRect();
    if (clientY < box.top + box.height / 2) {
      raw = Number(card.dataset.beat);
      break;
    }
  }
  const points = beatDropPoints(prompt.beats, draggingBeats);
  if (!points.length) return null;
  return points.reduce((best, point) =>
    (Math.abs(point - raw) < Math.abs(best - raw) ? point : best));
}

// A rule drawn across the gap the shot would fall into. The cards are tall
// enough that a floating ghost would be off screen as often as not.
function showBeatCaret(body, index) {
  clearBeatCaret();
  if (index === null) return;
  const cards = [...body.querySelectorAll(":scope > .card[data-beat]")];
  const at = cards.find((card) => Number(card.dataset.beat) === index);
  if (at) at.classList.add("drop-above");
  else cards.at(-1)?.classList.add("drop-below");
}

function clearBeatCaret() {
  for (const node of document.querySelectorAll(".drop-above, .drop-below")) {
    node.classList.remove("drop-above", "drop-below");
  }
}

// -- splitting shots off into their own source -----------------------------
//
// One generation is often asked to hold more cuts than it can: three shots in
// twelve seconds, and the second one is the only one worth its frames. Ticking
// shots here and pressing the button in the Shots header copies them — and the
// scene around them — into a new bin item that has rendered nothing, so they
// can be generated at their own length and cut back in.
//
// A tick box rather than a mode: picking shots is something you do while
// reading them, and a mode you have to enter first is a mode you forget you
// are in. The box stays faint until it is used, and the button that acts on it
// is not on screen at all until there is a tick to act on.

function splitPickBox(index) {
  const box = document.createElement("input");
  box.type = "checkbox";
  box.className = "split-pick";
  box.checked = state.splitPick.has(index);
  box.title = "Tick to split this shot off into a source of its own";
  box.onchange = () => {
    if (box.checked) state.splitPick.add(index);
    else state.splitPick.delete(index);
    renderSplitButton();
  };
  return box;
}

function renderSplitButton() {
  const button = $("split-beats");
  if (!button) return;
  const count = state.splitPick.size;
  button.hidden = !count;
  if (!count) return;
  button.innerHTML = label("scissors", count === 1 ? "Split off 1 shot" : `Split off ${count} shots`, "sm");
  button.title = "Copy the ticked shots into a new bin item, timing and all. "
    + "This shot list keeps them.";
}

/** Forget the ticks. Called wherever the indices they name could have moved. */
function clearSplitPick() {
  if (!state.splitPick.size) return;
  state.splitPick.clear();
  renderSplitButton();
}

function beatCard(prompt, beat, index, shotOf, moveOf) {
  // A beat that does not cut is a moment inside the shot above it, so the card
  // reads as one: indented, badged with the shot it belongs to, and led by the
  // camera rather than by a framing nobody is cutting to.
  const held = index > 0 && !cuts(beat);

  const card = document.createElement("div");
  card.className = held ? "card beat-held" : "card";

  const head = document.createElement("div");
  head.className = "card-head";
  // A storyboard of one has nothing to reorder, so it is offered no handle.
  if (prompt.beats.length > 1) head.append(beatGrip(card, prompt.beats, index));
  head.append(
    badge(held ? `↳ Shot ${shotOf[index]} · move ${moveOf[index]}` : `Shot ${shotOf[index]}`),
    index === 0
      ? pinnedTime(beat.start, "Shot 1 opens the clip, so it always starts at 0.")
      : tiny(cutInput(prompt, index, beat.start,
          "The cut into this shot. Moving it moves the end of the shot before it.")),
    separator("→"),
    index === prompt.beats.length - 1
      ? pinnedTime(beat.end, "The last shot runs to the end of the clip. Change the budget to move it.")
      : tiny(cutInput(prompt, index + 1, beat.end,
          "The cut out of this shot. Moving it moves the start of the next shot.")),
    separator("s"),
    shotLength(beat),
    spacer(),
    splitPickBox(index),
    sparkButton(`beat:${index}`, `shot ${shotOf[index]}`),
    removeButton(() => {
      prompt.beats.splice(index, 1);
      // Every index after this one just moved, so no tick still means what it
      // said. Clearing is the only honest answer.
      clearSplitPick();
      markDirty();
      renderBeats();
      refresh();
    }),
  );

  card.append(head, labelled(
    index === 0 ? "Opening transition" : "Transition into shot",
    beginsWith(beat, index, index === 0)));

  if (held) {
    // Camera first, matching the sentence this composes into: `At 00:03.000,
    // the camera pushes in ... as she lifts the mug`. The move leads, and the
    // framing is where it arrives rather than something to cut to.
    card.append(
      labelled("Camera move", cameraField(beat, index)),
      labelled("Reframe",
        growText(beat.angle, (value) => (beat.angle = value), `beat.${index}.angle`,
          "composes as “into …” — e.g. a wider view of the same kitchen")),
      lineSlot(prompt, beat, index, shotOf, "open",
        "Line that opens the move", "A line spoken before the action, straight off the camera clause"),
      labelled("What happens at this moment", growText(beat.action,
        (value) => (beat.action = value), `beat.${index}.action`,
        "the action the move carries — e.g. she lifts a red mug and drinks")),
      lineSlot(prompt, beat, index, shotOf, "",
        "Line after the action", "A line spoken after the action"),
    );
    return finishBeatCard(card, beat, index, shotOf[index], moveOf[index], held);
  }

  card.append(
    labelled("Composition", growText(beat.angle, (value) => (beat.angle = value),
      `beat.${index}.angle`, "framing and shot scale — e.g. a low-angle medium shot")),
    lineSlot(prompt, beat, index, shotOf, "open",
      "Line that opens the shot", "A line spoken before the action, straight off the framing clause"),
    labelled("Action & environment", growText(beat.action,
      (value) => (beat.action = value), `beat.${index}.action`,
      "who and what is in frame, the light and materials, and the motion arc")),
    labelled("Camera", cameraField(beat, index)),
    lineSlot(prompt, beat, index, shotOf, "",
      "Line after the action", "A line spoken after the action and the camera move"),
  );
  return finishBeatCard(card, beat, index, shotOf[index], moveOf[index], held);
}

// The sentinel for "no cut" in the Begins with picker. Bracketed so it can
// never collide with a transition phrase an agent wrote.
const NO_CUT = "[no cut]";

/** How this beat begins: a cut — and which phrase — or no cut at all.
 *
 * One control, because from the author's chair it is one decision: the film
 * either cuts here or it keeps rolling. Picking "no cut" folds the beat into
 * the shot above as a timed camera move, which H3 renders as one continuous
 * take. Shot 1 has nothing to continue, so it is only ever offered a cut.
 */
function beginsWith(beat, index, isFirst) {
  const select = document.createElement("select");
  select.className = "role";
  select.dataset.field = `beat.${index}.cut`;

  const cut = document.createElement("optgroup");
  cut.label = "Cut to a new shot";
  cut.append(new Option("the camera cuts to (default)", ""));
  for (const phrase of TRANSITIONS) cut.append(new Option(phrase, phrase));
  // A phrase written before this was a picker — or by an agent — still shows.
  if (beat.transition && !TRANSITIONS.includes(beat.transition)) {
    cut.append(new Option(`${beat.transition} (custom)`, beat.transition));
  }
  select.append(cut);

  if (!isFirst) {
    const held = document.createElement("optgroup");
    held.label = "No cut";
    held.append(new Option("the shot continues — a timed camera move", NO_CUT));
    select.append(held);
  }

  select.value = !isFirst && !cuts(beat) ? NO_CUT : (beat.transition || "");
  select.title = "Whether the film cuts here, or keeps rolling and moves the "
    + "camera instead. A cut has to introduce new information; if only the "
    + "framing changes, hold the shot and move.";
  select.onchange = () => {
    if (select.value === NO_CUT) {
      beat.cut = false;
    } else {
      beat.cut = true;
      beat.transition = select.value;
    }
    markDirty();
    // The whole card changes shape — badge, indent, which field leads — so it
    // is redrawn rather than patched.
    renderBeats();
    refresh();
  };
  return select;
}

/** A blank line, in the slot it is being authored into. */
const blankLine = (placement) => ({
  speakers: [], text: "", language: "English", verb: "says",
  delivery: "", describe: "", voiceover: false,
  possessive: "", continuity: "", cutoff: false, placement,
});

// One of the two slots the format has room for: the lines already in it,
// followed by the button that adds another. A line carries no timecode, so
// where its sentence sits among the shot's other sentences is the only thing
// that says when it lands — which makes the slot a *place* in the card, not a
// dropdown on a block parked at the bottom. Adding a line here is choosing
// when it is spoken. The dropdown inside a line survives as the way to move an
// existing one between the two without retyping it.
function lineSlot(prompt, beat, index, shotOf, placement, label, title) {
  const slot = document.createDocumentFragment();
  (beat.lines ?? []).forEach((line, position) => {
    if ((line.placement ?? "") !== placement) return;
    slot.append(lineBlock(prompt, beat, line, index, position, shotOf));
  });

  const row = document.createElement("div");
  row.className = "line-add";
  row.append(addButton(label, title, () => {
    const position = (beat.lines ??= []).length;
    beat.lines.push(blankLine(placement));
    markDirty();
    renderBeats();
    refresh();
    focusField(`line.${index}.${position}.text`);
  }));
  slot.append(row);
  return slot;
}

/** A menu of cast members; picking one appends its id to the speaker field. */
function castPicker(line, field, names) {
  const picker = document.createElement("select");
  picker.className = "cast-pick";
  picker.title = "Add a cast member as a speaker on this line";
  picker.disabled = !castMembers().length;

  const head = document.createElement("option");
  head.textContent = picker.disabled ? "no cast" : "+ cast";
  head.value = "";
  picker.append(head);

  for (const member of castMembers()) {
    const option = document.createElement("option");
    option.value = member.id;
    option.textContent = member.name || member.id;
    picker.append(option);
  }

  picker.onchange = () => {
    const id = picker.value;
    picker.value = "";
    if (!id || (line.speakers || []).includes(id)) return;
    line.speakers = [...(line.speakers || []), id];
    field.value = line.speakers.join(", ");
    renderSpeakerNames(names, line);
    markDirty();
    renderCast();
    refresh();
  };
  return picker;
}

/** Who the keys on this line resolve to, so an id is not a dead end. */
function renderSpeakerNames(host, line) {
  const keys = (line.speakers || []).filter((key) => looksLikeMemberId(key));
  host.textContent = keys.length ? keys.map(memberLabel).join(", ") : "";
}

// One spoken line, nested in its shot. Everything except the words and the
// language sits outside `<d>`, which is why "describe / verb / delivery" are
// their own fields rather than one sentence.
function lineBlock(prompt, beat, line, beatIndex, position, shotOf) {
  const block = document.createElement("div");
  block.className = "line-block";

  const speakers = textInput((line.speakers ?? []).join(", "), (value) => {
    line.speakers = value.split(",").map((key) => key.trim()).filter(Boolean);
    renderSpeakerNames(names, line);
  });
  speakers.placeholder = "speaker key(s), comma-separated";
  speakers.title = "A stable key per speaker. Keys become (S1), (S2)… by first appearance. "
    + "A cast member's id here pulls in their appearance and identity files.";

  // Ids are unreadable, so what they resolve to is shown beside them. The
  // field still holds the raw keys, which is what an agent writes.
  const names = document.createElement("span");
  names.className = "cast-empty";
  renderSpeakerNames(names, line);

  const castPick = castPicker(line, speakers, names);

  const vo = document.createElement("input");
  vo.type = "checkbox";
  vo.checked = !!line.voiceover;
  vo.onchange = () => {
    line.voiceover = vo.checked;
    markDirty();
    refresh();
  };
  const voLabel = document.createElement("label");
  voLabel.title = "Off-screen voiceover — composes the lips-closed clause too";
  voLabel.append(vo, document.createTextNode("VO"));

  // The card already says which slot this line is in by where it sits, so this
  // is the way to move it to the other one without retyping it — the same job
  // `move` does across shots.
  const slot = document.createElement("select");
  slot.title = "Move this line to the other slot in the shot";
  [["", "after the action"], ["open", "opens the shot"]].forEach(([value, label]) => {
    const option = new Option(label, value);
    if ((line.placement ?? "") === value) option.selected = true;
    slot.append(option);
  });
  slot.onchange = () => {
    line.placement = slot.value;
    markDirty();
    refresh();
  };

  // A line that runs across a cut is written as two lines on adjacent shots,
  // each tagged <scenetrans> at the seam. Tagging only one side is a lint
  // warning, because the model then has half a seam.
  const seam = document.createElement("select");
  seam.title = "Does this line run across a cut?";
  [["", "no seam"], ["into", "→ into next shot"], ["from", "← from previous shot"]]
    .forEach(([value, label]) => {
      const option = new Option(label, value);
      if ((line.continuity ?? "") === value) option.selected = true;
      seam.append(option);
    });
  seam.onchange = () => {
    line.continuity = seam.value;
    markDirty();
    refresh();
  };

  const cut = document.createElement("input");
  cut.type = "checkbox";
  cut.checked = !!line.cutoff;
  cut.onchange = () => {
    line.cutoff = cut.checked;
    markDirty();
    refresh();
  };
  const cutLabel = document.createElement("label");
  cutLabel.title = "The end of the video truncates this line — composes <cutoff>";
  cutLabel.append(cut, document.createTextNode("cut off"));

  // Moving a line between shots is a real edit (a conversation crosses a cut),
  // so the shot it belongs to stays changeable from here.
  const move = document.createElement("select");
  move.title = "Move this line to another shot";
  const moveOf = moveNumbers(prompt.beats);
  prompt.beats.forEach((other_, other) => {
    // Several beats can share a shot number now, so the moment disambiguates.
    const label = cuts(other_) ? `Shot ${shotOf[other]}`
      : `Shot ${shotOf[other]} · move ${moveOf[other]}`;
    const option = new Option(label, String(other));
    if (other === beatIndex) option.selected = true;
    move.append(option);
  });
  move.onchange = () => {
    const target = parseInt(move.value, 10);
    if (target === beatIndex) return;
    beat.lines.splice(position, 1);
    (prompt.beats[target].lines ??= []).push(line);
    markDirty();
    renderBeats();
    refresh();
  };

  // The words are what compose() keys on: a line without them is dropped
  // whole. Without this note, filling in describe / verb / delivery first
  // changes nothing in the Prompt tab and reads as the editor being broken.
  const mute = document.createElement("span");
  mute.className = "line-mute";
  mute.textContent = "no words — not in the prompt";
  mute.title = "A line only reaches the prompt once Words has something in it.";
  const syncMute = () => (mute.hidden = !!(line.text ?? "").trim());
  syncMute();

  const head = document.createElement("div");
  head.className = "card-head";
  head.append(badge("Line"), move, slot, speakers, castPick, names, voLabel, seam, cutLabel, mute, spacer(),
    removeButton(() => {
    beat.lines.splice(position, 1);
    markDirty();
    renderBeats();
    refresh();
  }));

  const where = `line.${beatIndex}.${position}`;
  const row = document.createElement("div");
  row.className = "card-row";
  row.append(
    labelled("Describe", growText(line.describe, (value) => (line.describe = value),
      `${where}.describe`, "the young woman with a quiet, breathy voice")),
    labelled("Verb", growText(line.verb, (value) => (line.verb = value),
      `${where}.verb`, "says")),
    labelled("Delivery", growText(line.delivery, (value) => (line.delivery = value),
      `${where}.delivery`, "without turning around")),
    labelled("Lips (VO)", growText(line.possessive, (value) => (line.possessive = value),
      `${where}.possessive`, "their — whose lips stay closed under the voiceover")),
    // Composes as the tag inside <d>[Language] ...</d>. It had no control at
    // all, so every line in the editor was silently English.
    labelled("Language", suggestOne(COMMON_LANGUAGES, line.language, "English",
      `${where}.language`, (value) => {
        line.language = value;
        markDirty();
        refresh();
      })),
  );

  const delivery = document.createElement("details");
  delivery.className = "line-delivery";
  const deliveryTitle = document.createElement("summary");
  deliveryTitle.textContent = "Voice & placement";
  delivery.append(deliveryTitle, head, row);
  block.append(
    labelled("Dialogue", growText(line.text, (value) => {
      line.text = value;
      syncMute();
    }, `${where}.text`, "The exact words spoken in this moment…")), delivery);
  delivery.ontoggle = () => { if (delivery.open) refitGrowFields(); };
  return block;
}

/** Beat list positions mapped to the shot number compose will give them. */
/** Beat positions in time order — the order the prompt composes them in. */
function beatOrder(beats) {
  return beats.map((beat, index) => index)
    .sort((a, b) => (beats[a].start - beats[b].start) || (beats[a].end - beats[b].end));
}

/** True unless the author cleared the cut. Undefined is a cut: every beat
 * written before no-cut beats existed opened a shot of its own. */
const cuts = (beat) => beat.cut !== false;

/** The `[Shot N]` each beat composes into, by beat position.
 *
 * No longer one per beat: a beat whose cut is cleared shares the number of the
 * one before it, because that is what the prompt will say and therefore what
 * the card has to show. The first beat always opens a shot — there is nothing
 * ahead of it to continue.
 */
function shotNumbers(beats) {
  const numbers = [];
  let current = 0;
  beatOrder(beats).forEach((index, position) => {
    if (position === 0 || cuts(beats[index])) current += 1;
    numbers[index] = current;
  });
  return numbers;
}

/** Which moment a beat is *within* its shot: 1 is the beat that opens it. */
function moveNumbers(beats) {
  const numbers = [];
  let within = 0;
  beatOrder(beats).forEach((index, position) => {
    within = position === 0 || cuts(beats[index]) ? 1 : within + 1;
    numbers[index] = within;
  });
  return numbers;
}

function renderCues() {
  const prompt = currentPrompt();
  const body = $("cues");
  body.textContent = "";
  if (!prompt) return;

  if (!prompt.cues.length) {
    body.innerHTML = `<tr><td colspan="5"><div class="empty">No timed cues — the bed
      above carries the whole generation.</div></td></tr>`;
    return;
  }

  const shotOf = shotNumbers(prompt.beats);

  prompt.cues.forEach((cue, index) => {
    const row = document.createElement("tr");

    // A blank end is meaningful: it marks an instantaneous hit rather than a
    // sustained bed, so it maps to null instead of a number.
    const end = document.createElement("input");
    end.type = "number";
    end.step = "0.1";
    end.placeholder = "—";
    end.value = cue.end ?? "";
    end.oninput = () => {
      cue.end = end.value === "" ? null : parseFloat(end.value) || 0;
      markDirty();
      refresh();
    };

    const description = textInput(cue.description, (value) => (cue.description = value));
    description.dataset.field = `cue.${index}.description`;

    // Which shot the cue lands in. Read-only, and only a reading aid: a cue is
    // pinned to the ruler, not to a shot, and stays put when the shots move.
    const inShot = prompt.beats.findIndex(
      (beat) => cue.start >= beat.start && cue.start < beat.end,
    );
    const shot = document.createElement("span");
    shot.className = "stat";
    shot.textContent = inShot >= 0 ? `Shot ${shotOf[inShot]}` : "—";

    row.append(
      cell(numberInput(cue.start, (value) => (cue.start = value)), "num"),
      cell(end, "num"),
      cell(description),
      cell(shot, "num"),
      cell(removeButton(() => {
        prompt.cues.splice(index, 1);
        renderCues();
        refresh();
      })),
    );
    body.append(row);
  });
}

function addButton(text, title, onClick) {
  const button = document.createElement("button");
  button.className = "small ghost";
  button.title = title;
  button.innerHTML = label("plus", text, "sm");
  button.onclick = onClick;
  return button;
}

function removeButton(onClick) {
  const button = document.createElement("button");
  button.className = "small iconbtn ghost";
  button.title = "Remove";
  button.innerHTML = iconHTML("x", "sm");
  button.onclick = onClick;
  return button;
}

function cell(child, className) {
  const td = document.createElement("td");
  if (className) td.className = className;
  td.append(child);
  return td;
}

function numberInput(value, onChange) {
  const input = document.createElement("input");
  input.type = "number";
  input.step = "0.1";
  input.value = value;
  input.oninput = () => {
    onChange(parseFloat(input.value) || 0);
    markDirty();
    refresh();
  };
  return input;
}

function textInput(value, onChange) {
  const input = document.createElement("input");
  input.type = "text";
  input.value = value ?? "";
  input.oninput = () => {
    onChange(input.value);
    markDirty();
    refresh();
  };
  return input;
}

// -- card building blocks --------------------------------------------------

/** A wrapping field that grows to fit what is in it, so nothing is hidden.
    `address` is the name the composed prompt knows this field by, which is
    what lets a click in the Prompt tab land back here. */
function growText(value, onChange, address, placeholder) {
  const box = document.createElement("textarea");
  box.rows = 1;
  box.value = value ?? "";
  if (address) box.dataset.field = address;
  if (placeholder) box.placeholder = placeholder;
  box.oninput = () => {
    fitToText(box);
    onChange(box.value);
    markDirty();
    refresh();
  };
  // The element has no layout until it is in the document, so the first fit
  // waits for the frame after the card is appended.
  requestAnimationFrame(() => fitToText(box));
  // Only the fields the composer honours a mention in: an `@` in the verbatim
  // words of a line is somebody saying a name out loud, and highlighting it
  // would promise a substitution that must not happen (see mentions.fields).
  if (MENTION_FIELD.test(address || "")) wireMentions(box);
  return box;
}

// A textarea in a hidden tab has no layout and reports a scroll height of 0;
// leaving it on `auto` keeps it usable until the tab is shown and refits it.
function fitToText(box) {
  box.style.height = "auto";
  if (box.scrollHeight) box.style.height = `${box.scrollHeight}px`;
}

function refitGrowFields() {
  for (const box of document.querySelectorAll(".card-field textarea, textarea.grow")) {
    fitToText(box);
  }
  // Values set from code fired no `input`, so the highlights are a form behind.
  repaintMentions();
}

/* The shot cards have grown to fit their text since they were built — "a
 * wrapped field you can see all of is the point". The block fields (style,
 * overview, camera, the sound beds, a member's appearance) never did, so the
 * longest and most-reread prose in the editor was the prose you could see four
 * lines of. Same treatment, same reason. */
function makeGrow(box) {
  box.classList.add("grow");
  box.addEventListener("input", () => fitToText(box));
  requestAnimationFrame(() => fitToText(box));
  return box;
}

// -- @mentions -------------------------------------------------------------
//
// `@Mara` in a prose field is a link to a cast member. The composer turns it
// into whatever label routes to her at that moment (see mentions.py):
// `<Subject 2>` when she has identity files, `(S1)` when she only speaks. The
// number is never typed, so reordering the cast — or putting one more person on
// screen — cannot leave the prose pointing at the wrong person.
//
// Everything below is the presentation of that. The matcher is the server's
// rule written again, because the highlight has to cover exactly the run the
// server will replace; a highlight that disagreed would be worse than none.

/** The field addresses a mention is honoured in — `mentions.fields` in JS. */
const MENTION_FIELD = new RegExp(
  "^(style|overview|camera|audio|music|ref_summary"
  + "|beat\\.\\d+\\.(angle|action|camera)"
  + "|line\\.\\d+\\.\\d+\\.(describe|delivery))$");

/** `[name, id]` for every named member, longest name first.
 *
 *  Longest first is what lets a name hold a space without a delimiter: with
 *  both "Mara" and "Mara Chen" in the cast, `@Mara Chen` has to try the longer
 *  one before the shorter claims the first four letters. Same order as `_index`.
 */
const mentionNames = () =>
  castMembers()
    .map((member) => [String(member.name || "").trim(), member.id])
    .filter(([name]) => name)
    .sort((a, b) => b[0].length - a[0].length);

const isWordChar = (ch) => Boolean(ch) && /\w/.test(ch);

/** Every `@name` in `text`, left to right. Mirrors `mentions.find`. */
function findMentions(text, names = mentionNames()) {
  const hits = [];
  if (!text || !names.length) return hits;
  let cursor = 0;
  for (;;) {
    const at = text.indexOf("@", cursor);
    if (at < 0) return hits;
    // An `@` with a word character in front of it is the middle of an address.
    if (at && isWordChar(text[at - 1])) { cursor = at + 1; continue; }
    const rest = text.slice(at + 1);
    const found = names.find(([name]) =>
      rest.slice(0, name.length).toLowerCase() === name.toLowerCase()
      && !isWordChar(rest.charAt(name.length)));
    if (!found) { cursor = at + 1; continue; }
    hits.push({ start: at, end: at + 1 + found[0].length, member: found[1] });
    cursor = at + 1 + found[0].length;
  }
}

// A bare `@word` nobody in the cast answers to. Drawn as a warning rather than
// left looking like a mention that worked: it reaches the model as an at-sign.
// The same thing `lint.mentions.unknown` says, said where it is being typed.
const STRAY_MENTION = /(?<![\w@])@([A-Za-z][\w'’-]*)/g;

function findStrayMentions(text, hits) {
  const stray = [];
  for (const match of String(text || "").matchAll(STRAY_MENTION)) {
    if (hits.some((hit) => match.index >= hit.start && match.index < hit.end)) continue;
    stray.push({ start: match.index, end: match.index + match[0].length, member: "" });
  }
  return stray;
}

/** What picking this member would put in the prompt, as the server reported it.
 *
 *  Never worked out here: the label is a position in the resolved subject list,
 *  and a second implementation of that in JavaScript is how the editor's answer
 *  and the render's answer start to differ.
 */
function mentionBecomes(member) {
  const entry = (state.lastPreview?.mentions || []).find((row) => row.id === member.id);
  if (!entry) return "";
  if (entry.present) return entry.label;
  // Not in this shot yet. Mentioning them puts them in it and mints their tag,
  // so there is no number to quote — say what will happen instead of quoting
  // the fallback as though it were the answer.
  return (member.references || []).length ? "joins as a new subject" : entry.label;
}

// -- the highlight ---------------------------------------------------------

// Copied from the field so the mirror wraps its text on exactly the same
// characters. Anything that moves a glyph has to be here; anything that only
// paints (colour, background) must not be, or the mirror would draw twice.
const MIRROR_METRICS = [
  "fontFamily", "fontSize", "fontWeight", "fontStyle", "fontVariant",
  "letterSpacing", "wordSpacing", "lineHeight", "textIndent", "textTransform",
  "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
  "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
];

const mirrorOf = (box) =>
  box?.parentNode?.classList?.contains("mention-host")
    ? box.parentNode.querySelector(".mention-mirror")
    : null;

function mentionHost(box) {
  if (box.parentNode?.classList?.contains("mention-host")) return box.parentNode;
  const host = document.createElement("div");
  host.className = "mention-host";
  box.parentNode.insertBefore(host, box);
  host.append(box);
  const mirror = document.createElement("div");
  mirror.className = "mention-mirror";
  mirror.setAttribute("aria-hidden", "true");
  host.append(mirror);
  return host;
}

function syncMirrorMetrics(box) {
  const mirror = mirrorOf(box);
  if (!mirror) return;
  const style = getComputedStyle(box);
  for (const property of MIRROR_METRICS) mirror.style[property] = style[property];
}

function paintMentions(box) {
  const mirror = mirrorOf(box);
  if (!mirror) return;
  const text = box.value || "";
  const hits = findMentions(text);
  const runs = [...hits, ...findStrayMentions(text, hits)].sort((a, b) => a.start - b.start);
  mirror.textContent = "";
  let cursor = 0;
  for (const run of runs) {
    mirror.append(document.createTextNode(text.slice(cursor, run.start)));
    mirror.append(mentionMark(box, text.slice(run.start, run.end), run));
    cursor = run.end;
  }
  // A block does not lay out a trailing newline; a textarea does. Without this
  // the last line's highlight sits one line high on a field ending in a break.
  mirror.append(document.createTextNode(`${text.slice(cursor)}\n`));
  mirror.scrollTop = box.scrollTop;
}

function mentionMark(box, text, run) {
  const mark = document.createElement("mark");
  mark.className = run.member ? "mention" : "mention stray";
  mark.textContent = text;
  if (!run.member) {
    mark.title = `${text} names nobody in the cast, so it is sent to the model `
      + `literally. Check the spelling, or add the member.`;
    return mark;
  }
  // The mirror sits over the field, so a click here would otherwise never reach
  // it. Selecting the whole mention is the useful answer anyway: it is one
  // token, replaced rather than edited into.
  mark.onmousedown = (event) => {
    event.preventDefault();
    box.focus();
    box.setSelectionRange(run.start, run.end);
  };
  mark.onmouseenter = () => showMentionCard(mark, run.member);
  mark.onmouseleave = hideMentionCardSoon;
  return mark;
}

/** Repaint every wired field. Cheap, and the labels move under them: renaming a
 *  member or putting somebody new on screen changes what a mention means. */
function repaintMentions() {
  for (const box of document.querySelectorAll('[data-mention="on"]')) paintMentions(box);
}

// -- the hover card --------------------------------------------------------

let mentionCardTimer = 0;

function mentionCardEl() {
  let card = $("mention-card");
  if (card) return card;
  card = document.createElement("div");
  card.id = "mention-card";
  card.className = "mention-card";
  card.hidden = true;
  // Hovering the card keeps it open, so the button on it can be reached.
  card.onmouseenter = () => clearTimeout(mentionCardTimer);
  card.onmouseleave = hideMentionCardSoon;
  document.body.append(card);
  return card;
}

function hideMentionCardSoon() {
  clearTimeout(mentionCardTimer);
  mentionCardTimer = setTimeout(() => { mentionCardEl().hidden = true; }, 180);
}

function showMentionCard(anchor, id) {
  clearTimeout(mentionCardTimer);
  const member = memberById(id);
  const card = mentionCardEl();
  if (!member) { card.hidden = true; return; }

  card.textContent = "";
  card.append(mentionCardBody(member));
  card.hidden = false;

  const at = anchor.getBoundingClientRect();
  const size = card.getBoundingClientRect();
  const above = at.top > size.height + 14;
  card.style.left =
    `${Math.max(8, Math.min(at.left, window.innerWidth - size.width - 8))}px`;
  card.style.top = `${above ? at.top - size.height - 8 : at.bottom + 8}px`;
}

function mentionCardBody(member) {
  const body = document.createDocumentFragment();

  const head = document.createElement("div");
  head.className = "mc-head";

  const face = document.createElement("div");
  face.className = "mc-face";
  const shown = (member.references || [])
    .find((file) => file.kind === "image" || file.kind === "video");
  if (shown) {
    const node = document.createElement(shown.kind === "image" ? "img" : "video");
    node.src = mediaViewUrl(shown.path);
    if (shown.kind === "image") node.alt = "";
    else { node.muted = true; node.preload = "metadata"; }
    face.append(node);
  } else {
    face.innerHTML = iconHTML(CAST_KIND_ICONS[memberKind(member)] || "users");
  }

  const names = document.createElement("div");
  const kind = CAST_KINDS.find(([value]) => value === memberKind(member));
  const becomes = mentionBecomes(member);
  names.innerHTML =
    `<div class="mc-kind">${escapeHtml(kind ? kind[1] : "Character")}</div>`
    + `<div class="mc-name">${escapeHtml(member.name || "unnamed")}</div>`
    + (becomes ? `<span class="mc-tag" title="What this mention composes to in `
        + `this shot">${escapeHtml(becomes)}</span>` : "");
  head.append(face, names);
  body.append(head);

  if ((member.appearance || "").trim()) {
    const line = document.createElement("p");
    line.textContent = member.appearance.trim();
    body.append(line);
  }
  if ((member.voice || "").trim()) {
    const line = document.createElement("p");
    line.className = "mc-voice";
    line.textContent = member.voice.trim();
    body.append(line);
  }

  const open = document.createElement("button");
  open.className = "mc-open";
  open.innerHTML = label("maximize", "Open in Cast", "sm");
  open.onclick = () => {
    mentionCardEl().hidden = true;
    openCastDetail(member.id);
  };
  body.append(open);
  return body;
}

// -- the `@` menu ----------------------------------------------------------
//
// Anchored to the field rather than to the caret. The field is the context you
// are typing in, and a menu that chases the caret around a five-line paragraph
// is harder to aim at, not easier.

let mentionMenuBox = null;
let mentionActive = 0;

function mentionMenuEl() {
  let menu = $("mention-menu");
  if (menu) return menu;
  menu = document.createElement("div");
  menu.id = "mention-menu";
  menu.className = "mention-menu";
  menu.hidden = true;
  document.body.append(menu);
  return menu;
}

function closeMentionMenu() {
  mentionMenuBox = null;
  const menu = $("mention-menu");
  if (menu) menu.hidden = true;
}

/** The `@query` the caret is sitting in, or null. */
function mentionQuery(box) {
  if (box.selectionStart !== box.selectionEnd) return null;
  const caret = box.selectionStart;
  const text = box.value || "";
  const from = text.lastIndexOf("@", caret - 1);
  if (from < 0) return null;
  if (from && isWordChar(text[from - 1])) return null;
  const query = text.slice(from + 1, caret);
  // A name is a few words at most. A run longer than this, or one crossing a
  // line break, is prose that happens to hold an at-sign.
  if (query.length > 32 || /[\n\r]/.test(query)) return null;
  return { from, caret, query };
}

/** Members whose name the query is still the start of.
 *
 *  An empty query — the `@` just typed — offers the whole cast. A query that
 *  has run past every name means the mention is finished, or was never one, and
 *  the menu closes rather than sitting there matching nothing.
 */
function mentionMatches(query) {
  const folded = query.trim().toLowerCase();
  return castMembers().filter((member) => {
    const name = String(member.name || "").trim().toLowerCase();
    return name && name.startsWith(folded);
  });
}

function openMentionMenu(box, members) {
  const menu = mentionMenuEl();
  mentionMenuBox = box;
  mentionActive = 0;
  menu.textContent = "";
  members.forEach((member, index) => {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "mention-option";
    const becomes = mentionBecomes(member);
    option.innerHTML =
      iconHTML(CAST_KIND_ICONS[memberKind(member)] || "users", "sm")
      + `<span class="mention-name">${escapeHtml(member.name)}</span>`
      + (becomes ? `<span class="mention-tag">${escapeHtml(becomes)}</span>` : "");
    // The field must keep focus, or the caret we are about to write at is gone.
    option.onmousedown = (event) => event.preventDefault();
    option.onclick = () => acceptMention(box, member);
    option.onmouseenter = () => {
      mentionActive = index;
      markMentionActive();
    };
    menu.append(option);
  });
  menu.hidden = false;
  positionMentionMenu(box);
  markMentionActive();
}

function markMentionActive() {
  const menu = $("mention-menu");
  if (!menu) return;
  [...menu.children].forEach((option, index) => {
    option.setAttribute("aria-current", String(index === mentionActive));
    if (index === mentionActive) option.scrollIntoView({ block: "nearest" });
  });
}

function positionMentionMenu(box) {
  const menu = mentionMenuEl();
  const at = box.getBoundingClientRect();
  menu.style.minWidth = `${Math.min(Math.max(at.width, 180), 340)}px`;
  const size = menu.getBoundingClientRect();
  const below = window.innerHeight - at.bottom > size.height + 12;
  menu.style.left =
    `${Math.max(8, Math.min(at.left, window.innerWidth - size.width - 8))}px`;
  menu.style.top = `${below ? at.bottom + 4 : Math.max(8, at.top - size.height - 4)}px`;
}

function acceptMention(box, member) {
  const found = mentionQuery(box);
  if (!found) return closeMentionMenu();
  const text = box.value || "";
  const after = text.slice(found.caret);
  const written = `@${String(member.name || "").trim()}`
    + (after && !/^\s/.test(after) ? " " : "");
  box.value = text.slice(0, found.from) + written + after;
  const caret = found.from + written.length;
  box.setSelectionRange(caret, caret);
  closeMentionMenu();
  // The field's own handlers own everything downstream — the write into the
  // prompt, the dirty flag, the preview — so the edit is announced rather than
  // repeated here.
  box.dispatchEvent(new Event("input", { bubbles: true }));
  box.focus();
}

function onMentionInput(box) {
  paintMentions(box);
  const found = mentionQuery(box);
  const members = found ? mentionMatches(found.query) : [];
  if (!members.length) return closeMentionMenu();
  openMentionMenu(box, members);
}

function mentionKey(box, event) {
  const menu = $("mention-menu");
  if (!menu || menu.hidden || mentionMenuBox !== box || !menu.children.length) return;
  if (event.key === "Escape") {
    event.preventDefault();
    closeMentionMenu();
    return;
  }
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const step = event.key === "ArrowDown" ? 1 : -1;
    mentionActive = (mentionActive + step + menu.children.length) % menu.children.length;
    markMentionActive();
    return;
  }
  if (event.key === "Enter" || event.key === "Tab") {
    event.preventDefault();
    menu.children[mentionActive]?.click();
  }
}

/** Give one field the whole treatment: highlight, hover card, `@` menu.
 *
 *  Idempotent, because the shot cards are rebuilt often and a field wired twice
 *  would paint twice.
 */
function wireMentions(box) {
  if (!box || box.dataset.mention === "on") return box;
  box.dataset.mention = "on";

  box.addEventListener("input", () => onMentionInput(box));
  box.addEventListener("click", () => onMentionInput(box));
  box.addEventListener("keydown", (event) => mentionKey(box, event));
  box.addEventListener("blur", () => setTimeout(() => {
    if (mentionMenuBox === box) closeMentionMenu();
  }, 140));
  box.addEventListener("scroll", () => {
    const mirror = mirrorOf(box);
    if (mirror) mirror.scrollTop = box.scrollTop;
  });

  // Only a textarea gets the mirror: the prose lives in those, and a one-line
  // input scrolls its text sideways, which an overlay cannot follow honestly.
  // The role fields keep the menu, which is the half that saves the typing.
  if (box.tagName === "TEXTAREA") {
    // Wrapped a frame late, on purpose. `labelled` moves the field into its own
    // wrapper after this returns, and wrapping first would have that move take
    // the textarea straight back out of the host.
    requestAnimationFrame(() => {
      if (!box.isConnected) return;
      mentionHost(box);
      syncMirrorMetrics(box);
      paintMentions(box);
    });
  }
  return box;
}

function labelled(labelText, control) {
  const wrap = document.createElement("div");
  wrap.className = "card-field";
  const label = document.createElement("label");
  label.textContent = labelText;
  if (control.id || control.dataset.field) {
    label.htmlFor = control.id ||= `f-${control.dataset.field.replace(/\./g, "-")}`;
  }
  wrap.append(label, control);
  return wrap;
}

function badge(text) {
  const span = document.createElement("span");
  span.className = "card-n";
  span.textContent = text;
  return span;
}

function separator(text) {
  const span = document.createElement("span");
  span.className = "sep";
  span.textContent = text;
  return span;
}

function spacer() {
  const span = document.createElement("span");
  span.className = "spacer";
  return span;
}

function tiny(input) {
  input.classList.add("tiny");
  return input;
}

// One end of a shot, which is also one end of its neighbour. Commits on
// `change` rather than `input`, because the field on the far side of the cut
// follows this one: typing "12" over "6" would otherwise roll the cut to 1
// on the way past, and take the neighbouring shot with it.
function cutInput(prompt, cutIndex, value, title) {
  const input = document.createElement("input");
  input.type = "number";
  input.step = "0.1";
  input.value = round3(value);
  input.title = title;
  input.onchange = () => {
    setCut(prompt, cutIndex, parseFloat(input.value) || 0);
    markDirty();
    renderBeats();
    refresh();
  };
  return input;
}

/** A boundary the budget owns rather than the author: shown, not editable. */
function pinnedTime(value, title) {
  const span = document.createElement("span");
  span.className = "pinned";
  span.textContent = round3(value).toFixed(2);
  span.title = title;
  return span;
}

/** How long a shot runs — the number people actually think in. */
function shotLength(beat) {
  const span = document.createElement("span");
  span.className = "len";
  span.textContent = `${round3(beat.end - beat.start).toFixed(2)}s`;
  span.title = "How long this shot runs";
  return span;
}

function fillForm() {
  const source = currentSource();
  renderShotLoras();
  if (typeof editorUX !== "undefined") editorUX?.context(source?.id, state.depth);
  updateReviewContext();
  $("editor").classList.toggle("has-source", !!source);
  const clip = currentClip();
  const segment = currentSegment();
  const prompt = segment?.prompt ?? null;
  const disabled = !prompt;

  for (const key of BLOCKS) {
    $(key).value = prompt?.[key] ?? "";
    $(key).disabled = disabled;
  }
  for (const key of LIST_BLOCKS) {
    $(key).value = (prompt?.[key] ?? []).join("\n");
    $(key).disabled = disabled;
  }
  // Values set from code, so nothing fired `input`; refit them by hand.
  refitGrowFields();
  $("source-name").value = source?.name ?? "";
  // Defaults true on a prompt that predates the control, matching the document.
  $("guard_identities").checked = prompt?.guard_identities ?? true;
  $("guard_identities").disabled = disabled;
  $("duration").value = prompt?.duration ?? 10;
  // Trim belongs to the clip, not the source: two clips on one lineage are
  // trimmed differently on purpose.
  $("source-in").value = clip?.source_in ?? 0;
  $("source-out").value = clip?.source_out ?? "";
  $("speed").value = clip?.speed ?? 1;
  for (const id of ["source-in", "source-out", "speed"]) $(id).disabled = !clip;
  fillSound(clip);

  renderSegmentPicker(source);
  updateHistoryCount();

  // The budget belongs to the generation on screen, not to the source. With one
  // generation those are the same thing; with a chain it is the difference
  // between a 12s opener and the 8s beat after it, so the label says which.
  const durationLabel = document.querySelector('label[for="duration"]');
  if (durationLabel) {
    const count = source?.segments?.length ?? 0;
    const depth = Math.min(state.depth, Math.max(count - 1, 0));
    durationLabel.textContent = count > 1 ? `Gen ${depth + 1} (s)` : "Generate (s)";
    durationLabel.title = count > 1
      ? "How long this generation runs. Every generation has its own budget."
      : "";
  }

  for (const id of ["source-name", "duration", "ratio"]) $(id).disabled = disabled;
  renderQuality(segment);

  const entry = clipReport(state.selectedClip);
  const report = sourceReport(state.selectedSource);
  const isHidden = clip?.enabled === false;
  $("clip-stat").textContent = entry
    ? (isHidden
        ? `${seconds(entry.source_seconds)} of lineage · hidden from timeline`
        : `${seconds(entry.source_seconds)} of lineage → ${seconds(entry.timeline_seconds)} on the timeline`)
    : (report ? `${seconds(report.seconds)} of lineage · not on the timeline` : "");
  $("editor-title").textContent = source?.name ?? "Source";
  $("editor-stat").textContent = report
    ? `${seconds(entry?.timeline_seconds ?? report.seconds)} · ${report.take_count} take(s)` +
      (source.segments.length > 1 ? ` · ${source.segments.length} generations` : "") +
      (isHidden ? " · (hidden)" : "")
    : "";

  updateClipButtons();

  renderBeats();
  renderCues();
  renderMedia();
  renderCast();
}

// -- what a clip is heard as ----------------------------------------------

// One question with one answer: its own sound, nothing, or a take whose picture
// was rejected. Muting and borrowing are separate fields on the clip — they have
// to be, since a borrowed track can be muted too — but nobody thinks about it
// that way while cutting, so the control does not either.
const OWN_SOUND = "";
// A take id is always "prompt_id:filename", so this cannot collide with one.
const NO_SOUND = "__muted__";

/** What to call a take in a list: its seed if it has one, else its file. */
const takeLabel = (take) =>
  take.seed === null || take.seed === undefined ? take.filename : `Seed #${take.seed}`;

// The prefixes this editor and its post-processes write on their own. They say
// nothing the tier pill does not, so a take saved under one has no name worth
// showing; only a prefix a person (or a test script) chose does.
const HOUSE_PREFIXES = /^(MiniMax_H3|Minimax_H3_Refined|VideoPrompter_H3(ref)?|H3extend|VideoPrompter_ReDetail|VideoPrompter_SeedVR2(_\w+)?|ComfyUI)$/i;

/** The words of a take's filename that someone chose, or [] for a house name.
 *  `HerdB_shield_fromFixA_00002_.mp4` → ["HerdB", "shield", "fromFixA"]. */
function takeNameTokens(take) {
  const stem = String(take.filename || "")
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/-audio$/i, "")
    .replace(/_\d{4,}_?$/, "");
  if (!stem || HOUSE_PREFIXES.test(stem)) return [];
  return stem.split("_").filter(Boolean);
}

/** How many leading filename words every named take in a grid shares. A sweep
 *  named `HerdB_shield_…`, `HerdB_clean_…` needs only `shield`, `clean` on its
 *  cards — the shared `HerdB` is the same on every one and says nothing. */
function sharedNamePrefix(takes) {
  const named = takes.map(takeNameTokens).filter((tokens) => tokens.length);
  if (named.length < 2) return 0;
  let n = 0;
  while (named.every((tokens) => tokens.length > n + 1 && tokens[n] === named[0][n])) n += 1;
  return n;
}

function fillSound(clip) {
  const select = $("clip-audio-take");
  const gain = $("clip-audio-gain");
  const offset = $("clip-audio-offset");
  const sound = clipReport(clip?.id)?.audio ?? null;

  for (const control of [select, gain, offset]) control.disabled = !clip;
  select.textContent = "";
  if (!clip) { gain.value = ""; offset.value = ""; return; }

  select.append(
    new Option("Its own sound", OWN_SOUND),
    new Option("Muted", NO_SOUND),
  );

  // Discarded takes belong in this list. Discarding is how you reject a
  // *picture*, and a rejected picture with sound worth keeping is the whole
  // reason to be here.
  const onScreen = new Set(clipReport(clip.id)?.path ?? []);
  for (const take of takesFor(clip.source_id)) {
    const note = take.discarded ? " — discarded" : (onScreen.has(take.id) ? " — on screen" : "");
    select.append(new Option(`${takeLabel(take)}${note}`, take.id));
  }

  const wanted = sound?.muted ? NO_SOUND : (sound?.borrowed ? sound.take : OWN_SOUND);
  select.value = wanted ?? OWN_SOUND;
  // A borrowed take that has left the bin would leave the picker blank and then
  // quietly reset the clip on the next change. Say it is gone instead.
  if (select.value !== wanted) {
    select.append(new Option("(take no longer in the bin)", wanted));
    select.value = wanted;
  }

  gain.value = sound?.gain_db ?? 0;
  offset.value = sound?.offset ?? 0;
  // Nudging is only meaningful against a track that came from somewhere else.
  offset.disabled = !sound?.borrowed || Boolean(sound?.muted);
}

// One chip per generation, in the editor head rather than in the Source tab:
// every tab authors the generation this picks, so it has to outrank the tab bar.
// A source that has never been extended is one segment, which needs no chips —
// but it still gets the add button, or there would be no way to plan a
// continuation without first rendering one.
function renderSegmentPicker(source) {
  const host = $("segment-picker");
  if (!host) return;
  host.textContent = "";
  const segments = source?.segments ?? [];
  host.hidden = !segments.length;
  if (!segments.length) return;

  const path = pathOf(source.id);
  if (segments.length > 1) {
    segments.forEach((segment, depth) => {
      const button = document.createElement("button");
      button.className = "seg-tab" + (depth === state.depth ? " active" : "");
      const rendered = Boolean(path[depth]);
      button.innerHTML =
        `<span>Gen ${depth + 1}</span>` +
        (rendered ? "" : `<span class="seg-todo" title="Not rendered yet">·</span>`);
      button.title = (rendered
        ? `Generation ${depth + 1} — ${takeById(path[depth])?.filename ?? "chosen"}`
        : `Generation ${depth + 1} — planned, not rendered`) +
        "\nAlt+[ / Alt+] steps generations";
      button.onclick = () => selectDepth(depth);
      host.append(button);
    });
  }

  // Icon-only in both cases: the head is a row of buttons, not a place for a
  // sentence. The tooltip carries what the label used to say.
  const add = document.createElement("button");
  add.className = "seg-tab add";
  add.innerHTML = iconHTML("plus", "sm");
  add.title = "Plan one more generation on the end of this chain (Motion Context)";
  add.setAttribute("aria-label", "Plan a continuation");
  add.onclick = addSegment;
  host.append(add);

  // A long chain scrolls inside the head, so keep the generation you just
  // stepped to on screen. `nearest` moves as little as possible and leaves the
  // head itself alone, which does not scroll.
  host.querySelector(".seg-tab.active")
    ?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

/** Author a different generation of the open source. */
function selectDepth(depth) {
  if (state.editorOpen || state.inspectorOpen) readForm();
  state.depth = depth;
  clearSplitPick();
  // Fill the form BEFORE anything reads it back. `resetPreviewEdit` re-composes,
  // and composing starts with its own `readForm` — run first, it would write the
  // generation you were just looking at straight over the one you switched to,
  // silently flattening a chain into N copies of its first prompt.
  fillForm();
  resetPreviewEdit();
  renderCandidates();
  refresh();
}

function readForm() {
  const source = currentSource();
  const segment = currentSegment();
  if (!source || !segment) return;
  const prompt = segment.prompt;
  for (const key of BLOCKS) prompt[key] = $(key).value;
  for (const key of LIST_BLOCKS) {
    prompt[key] = $(key).value.split("\n").map((line) => line.trim()).filter(Boolean);
  }
  source.name = $("source-name").value;
  prompt.guard_identities = $("guard_identities").checked;
  const duration = parseInt($("duration").value, 10);
  if (!Number.isNaN(duration)) prompt.duration = duration;
}

// -- media: keyframes (i2v) and references (ref2v) ------------------------

// Which node H3 will run, mirroring `imagery.render_mode` on the server, so the
// editor can show it before a render is prepared.
//
// Counted on the resolved prompt, like the server does: a member on screen here
// brings their identity files with them, and a shot carrying nothing of its own
// still renders as ref2v because of them. Showing "Text-to-Video" there would be
// a lie the author only discovers in the finished clip.
function mediaMode(prompt) {
  if (!prompt) return "t2v";
  if (prompt.beats?.some(b=>b.camera_path?.frame_mode==="motion"&&b.camera_path?.reference_video)) return "ref2v";
  if ((prompt.first_frame || "").trim() || (prompt.last_frame || "").trim()) return "i2v";
  if (effectiveReferenceCount(prompt)) return "ref2v";
  return "t2v";
}

/** A closed list as a select: the value is one of `options`, or "" for default. */
function pickOne(options, value, blankLabel, field, onPick) {
  const select = document.createElement("select");
  select.className = "role";
  if (field) select.dataset.field = field;
  select.append(new Option(blankLabel, ""));
  for (const option of options) select.append(new Option(option, option));
  // A value written before this was a picker — or by an agent — still shows,
  // rather than silently reading as the default.
  if (value && !options.includes(value)) select.append(new Option(`${value} (custom)`, value));
  select.value = value || "";
  select.onchange = () => onPick(select.value);
  return select;
}

/** An open list: the common answers are one click, anything else can be typed. */
function suggestOne(options, value, placeholder, field, onPick) {
  const id = `list-${field.replace(/[^\w-]/g, "-")}`;
  const wrap = document.createElement("div");
  wrap.className = "suggest";

  const input = document.createElement("input");
  input.type = "text";
  input.className = "role";
  input.dataset.field = field;
  input.setAttribute("list", id);
  input.placeholder = placeholder;
  input.value = value || "";
  input.oninput = () => onPick(input.value);

  const list = document.createElement("datalist");
  list.id = id;
  for (const option of options) list.append(new Option(option, option));

  wrap.append(input, list);
  return wrap;
}

/** The camera prose field, with the trained vocabulary behind three pickers.
 *
 * The field stays free text because the format wants motion woven into the
 * shot's own sentence, not a label bolted on. The pickers write that sentence
 * for you: pick a move (and, only where it means something, an amplitude and a
 * speed) and the phrase is appended — medium and normal are said by leaving
 * them out, so both of those default to blank.
 */
function cameraField(beat, index) {
  const wrap = document.createElement("div");
  wrap.className = "camera-field";

  const text = growText(beat.camera, (value) => (beat.camera = value),
    `beat.${index}.camera`,
    "move + amplitude + speed — e.g. pushes in with small amplitude at slow speed");
  wrap.append(text);

  const pathTools = document.createElement("div");
  pathTools.className = "camera-path-tools";
  const plan = document.createElement("button");
  plan.type = "button";
  plan.className = "small ghost";
  plan.textContent = beat.camera_path ? "Edit camera path" : "Plan camera path";
  plan.onclick = () => {
    const prompt = currentPrompt();
    const source = state.selectedSource;
    const segment = currentSegment()?.id;
    const epoch = state.projectEpoch;
    const fingerprint = JSON.stringify(prompt?.beats[index]);
    openCameraPlanner({path: beat.camera_path, duration: beat.end - beat.start,
      title: `Shot direction ${index + 1}`,
      whole: prompt.beats.length===1&&beat.start===0&&beat.end===prompt.duration,
      hasFirst: !!prompt.first_frame, hasReferences: !!effectiveReferenceCount(prompt),
      hasDifferentLast: !!prompt.last_frame&&prompt.last_frame!==prompt.first_frame,
      videos: [...(prompt.references||[]).filter(r=>r.kind==="video").map(r=>({path:r.path,label:r.role||r.path})),
        ...state.takes.filter(t=>isVideoTake(t)&&!t.discarded&&(!t.seconds||(t.seconds>=2&&t.seconds<=15)))
          .map(t=>({path:extendSource(t),label:`Take · ${t.shot_name||t.filename}`}))],
      mediaURL: path=>{const tagged=/^(.*) \[(input|output|temp)\]$/.exec(path);if(!tagged)return mediaViewUrl(path);
        const slash=tagged[1].lastIndexOf("/");return `/view?${new URLSearchParams({filename:tagged[1].slice(slash+1),subfolder:slash<0?"":tagged[1].slice(0,slash),type:tagged[2]})}`;},
      uploadVideo: async()=>{const file=await pickFile("video/*");if(!file)return null;
        const uploaded=await uploadMedia(file);if(uploaded.kind!=="video")throw new Error("Choose a video file for Motion Frame.");
        return {path:uploaded.filename,label:file.name};},
      imageURL: prompt?.first_frame ? `/view?type=input&filename=${encodeURIComponent(prompt.first_frame)}` : "",
      apply: path => {
        const current = currentPrompt();
        if (state.projectEpoch !== epoch || state.selectedSource !== source
          || currentSegment()?.id !== segment || JSON.stringify(current?.beats[index]) !== fingerprint) return false;
        if (path) current.beats[index].camera_path = path;
        else delete current.beats[index].camera_path;
        markDirty(); renderBeats(); refresh(); return true;
      }});
  };
  pathTools.append(plan);
  if (beat.camera_path) {
    text.readOnly = true;
    text.value = cameraPathSummary(beat.camera_path);
    text.title = "The planned path supplies camera direction. Use written camera in the planner to restore your text.";
    const active = document.createElement("span"); active.textContent = "Path active"; pathTools.append(active);
  }
  wrap.append(pathTools);

  const strip = document.createElement("div");
  strip.className = "camera-picker";

  const motion = document.createElement("select");
  motion.title = "The camera moves H3 was trained on. Picking one writes it into "
    + "the sentence above.";
  motion.append(new Option("insert a move…", ""));
  for (const name of CAMERA_MOTIONS) motion.append(new Option(name, name));

  const amplitude = document.createElement("select");
  amplitude.title = "How far the move travels. Leave blank for medium — the format "
    + "says medium by not mentioning it.";
  amplitude.append(new Option("amplitude — medium", ""));
  for (const value of CAMERA_AMPLITUDES) {
    amplitude.append(new Option(value.replace("with ", ""), value));
  }

  const speed = document.createElement("select");
  speed.title = "How fast it travels. Leave blank for normal — again, said by "
    + "leaving it out.";
  speed.append(new Option("speed — normal", ""));
  for (const value of CAMERA_SPEEDS) speed.append(new Option(value.replace("at ", ""), value));

  motion.onchange = () => {
    if (!motion.value) return;
    const phrase = [CAMERA_VERBS[motion.value], amplitude.value, speed.value]
      .filter(Boolean).join(" ");
    const existing = (beat.camera || "").trim();
    beat.camera = existing ? `${existing.replace(/[.,]$/, "")}, then ${phrase}` : phrase;
    motion.value = "";
    markDirty();
    renderBeats();
    refresh();
  };

  strip.append(motion, amplitude, speed);
  const tools = document.createElement("details");
  tools.className = "camera-tools";
  const toolsTitle = document.createElement("summary");
  toolsTitle.textContent = "Insert camera move";
  tools.append(toolsTitle, strip);
  tools.hidden = !!beat.camera_path;
  wrap.append(tools);
  return wrap;
}

/** A retention picker plus the line that says what the chosen marker means.
 *
 * Built once and used by both the Cast tab and the Media tab, so the two places
 * a marker can be set explain it the same way. `defaultLabel` names what an
 * empty value falls back to; `onPick` receives the new value.
 */
function retentionControl(kind, value, defaultLabel, onPick) {
  const isAudio = kind === "audio" || kind === "video_audio";
  const fallback = isAudio ? "reference" : "fully_preserved";

  const wrap = document.createElement("div");
  wrap.className = "retention";

  const select = document.createElement("select");
  select.className = "role";
  const blank = new Option(defaultLabel ?? `retention — default (${fallback})`, "");
  blank.title = `Falls back to ${fallback}. ${RETENTION_HELP[fallback]}`;
  select.append(blank);
  for (const marker of isAudio ? AUDIO_RETENTION : VISUAL_RETENTION) {
    const option = new Option(marker, marker);
    option.title = RETENTION_HELP[marker] ?? "";
    select.append(option);
  }
  select.value = value || "";

  const hint = document.createElement("p");
  hint.className = "retention-hint";
  const explain = () => {
    const marker = select.value || fallback;
    hint.textContent = RETENTION_HELP[marker] ?? "";
    select.title = hint.textContent;
  };
  explain();

  select.onchange = () => {
    explain();
    onPick(select.value);
  };

  wrap.append(select, hint);
  return wrap;
}

/** The cast member and identity file a linked row points at, or null. */
function linkedTarget(ref) {
  if (!ref?.member) return null;
  const member = memberById(ref.member);
  const reference = member?.references?.[ref.slot || 0];
  return reference ? { member, reference } : null;
}

/** What a row really sends: the member's file for a link, its own otherwise. */
function referenceFile(ref) {
  const target = linkedTarget(ref);
  return target
    ? { kind: target.reference.kind, path: target.reference.path }
    : { kind: ref?.kind, path: ref?.path || "" };
}

/** The `<Picture N>` a row answers to, from the server's resolved list.
 *
 * Not by position: resolution puts the cast's files in that list too and orders
 * it by member, so a row's index here is not its index there. Linked rows are
 * matched on the member and slot they name, plain ones on the file they carry.
 */
function tagFor(ref) {
  const match = ref?.member
    ? (state.refTags || []).find(
        (item) => item.member === ref.member && (item.slot || 0) === (ref.slot || 0))
    : (state.refTags || []).find(
        (item) => !item.member && item.path === (ref?.path || ""));
  return match?.tag || "";
}

/** The resolved subject a file is cited by, or null.
 *
 * A picture that a subject is cut from writes no `subject_definitions` line of
 * its own, so its role field is inert and the row has to say why.
 */
function subjectCiting(path) {
  if (!path) return null;
  return (state.subjects || []).find((item) => (item.sources || []).includes(path)) || null;
}

/** The stored author row behind a resolved subject, or null.
 *
 * A cast member's subject usually has no stored row at all: resolution rebuilds
 * it from the member every time, and a row exists only once someone has
 * overridden the wording for this segment. Read-only — rendering must not
 * create one, or opening a segment would dirty it.
 */
function findSubjectRow(prompt, resolved) {
  const rows = prompt?.subjects || [];
  if (resolved.member) return rows.find((item) => item.member === resolved.member) ?? null;
  // Resolution keeps the segment's own subjects ahead of the cast's and in
  // order, so position identifies them.
  return rows.filter((item) => !item.member)[resolved.plainIndex] ?? null;
}

/** The same row, minted if it is not there yet. Only an edit may call this. */
function subjectRowFor(prompt, resolved) {
  if (!prompt) return null;
  const found = findSubjectRow(prompt, resolved);
  if (found || !resolved.member) return found;
  const row = {
    role: "", retention: "", sources: [], member: resolved.member, noun: "", label: "",
  };
  prompt.subjects = [...(prompt.subjects || []), row];
  return row;
}

/** Drop a link row that no longer overrides anything, so saves stay clean. */
function pruneSubjectRow(prompt, row) {
  if (!row?.member) return;
  if ((row.role || "").trim() || (row.retention || "").trim()) return;
  prompt.subjects = (prompt.subjects || []).filter((item) => item !== row);
}

/** Take a member off screen here, by every route that puts them on it.
 *
 * There are three, and the tick is only one: `cast` lists them, a linked
 * reference row carries their file, and a subject row overrides their wording
 * for this segment. Clearing the listing alone leaves the other two standing —
 * and a linked row is by itself enough to be on screen (`castReferencePaths`
 * counts one, matching `cast.appearing`), so the member comes back as a
 * subject anyway, now at the default retention instead of the one they were
 * ticked with. Whoever is going, goes by all three.
 *
 * A speaker is deliberately untouched: their line is what puts them on screen,
 * so the line is what has to go, and silently dropping their identity out from
 * under it would leave the line unattributed.
 */
function takeMemberOffScreen(prompt, id) {
  if (!prompt || !id) return;
  prompt.cast = (prompt.cast || []).filter((entry) => entry !== id);
  prompt.undescribed = (prompt.undescribed || []).filter((entry) => entry !== id);
  prompt.references = (prompt.references || [])
    .filter((item) => !(linkedTarget(item) && item.member === id));
  prompt.subjects = (prompt.subjects || []).filter((item) => item.member !== id);
}

/** Identity files the members on screen in this segment bring with them. */
function castReferencePaths(prompt) {
  if (!prompt) return [];
  const rows = prompt.references || [];
  // A linked row is itself a way to be on screen, so its member counts even
  // without a tick or a line — matching `cast.appearing`.
  const onScreen = [
    ...speakingMemberIds(prompt),
    ...(prompt.cast || []),
    ...rows.filter((item) => linkedTarget(item)).map((item) => item.member),
  ];
  const paths = [];
  for (const id of onScreen) {
    for (const reference of memberById(id)?.references || []) {
      if (reference.path && !paths.includes(reference.path)) paths.push(reference.path);
    }
  }
  // Deduped against the shot's own the way `cast.resolve` does, so the count
  // here is the count that reaches the model. A linked row resolves to the
  // member's file, so it must not make that file look like a second one.
  const own = new Set(rows.map((item) => referenceFile(item).path));
  return paths.filter((path) => !own.has(path));
}

/** How many references this shot really sends: its own plus the cast's. */
function effectiveReferenceCount(prompt) {
  return (prompt?.references || []).length + castReferencePaths(prompt).length;
}

const MODE_LABEL = {
  t2v: "Text-to-Video",
  i2v: "Image-to-Video",
  ref2v: "Reference-to-Video",
};

// What each reference kind's file picker should accept.
const KIND_ACCEPT = {
  image: "image/*",
  video: "video/*",
  video_audio: "audio/*",
  audio: "audio/*",
};

async function loadRefKinds() {
  state.refKinds = (await call(`${API}/reference_kinds`)).kinds ?? [];
}

// A stored media path is "<subfolder>/<name>"; ComfyUI serves it from /view.
function mediaViewUrl(path) {
  const slash = String(path || "").lastIndexOf("/");
  const subfolder = slash >= 0 ? path.slice(0, slash) : "";
  const filename = slash >= 0 ? path.slice(slash + 1) : path;
  const query = new URLSearchParams({ filename, subfolder, type: "input" });
  return `/view?${query}`;
}

// Send a picked file to the input dir; returns its stored path or throws.
async function uploadMedia(file) {
  const form = new FormData();
  form.append("file", file, file.name);
  const response = await fetch(`${API}/reference/upload`, { method: "POST", body: form });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Upload failed (${response.status})`);
  return body; // {filename, kind, bytes}
}

function pickFile(accept) {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept || "";
    input.onchange = () => resolve(input.files?.[0] ?? null);
    input.click();
  });
}

// A thumbnail element for a stored media path, by kind.
function mediaThumb(path, kind, set) {
  const url = mediaViewUrl(path);
  if (kind === "image") {
    const img = document.createElement("img");
    img.src = url;
    img.alt = "";
    return viewable(img, path, kind, set);
  }
  if (kind === "video") {
    const video = document.createElement("video");
    video.src = url;
    video.muted = true;
    video.preload = "metadata";
    return viewable(video, path, kind, set);
  }
  const glyph = document.createElement("span");
  glyph.innerHTML = iconHTML("music");
  return glyph;
}

/** Let a thumbnail be opened large. One click: nothing else claims a click on
 *  the picture itself -- the slot's own click only answers when the slot is
 *  empty, and the buttons over it are siblings rather than children. `set` is
 *  what the arrows walk from here — the other references on the segment, say —
 *  and defaults to this file alone. */
function viewable(node, path, kind, set) {
  node.classList.add("viewable");
  node.title = `${mediaName(path)} — click to see it large`;
  node.onclick = (event) => {
    event.stopPropagation();
    openMediaViewer(path, kind, set);
  };
  return node;
}

/** The file's own name, without the folder it was uploaded into. */
function mediaName(path) {
  return String(path || "").split("/").pop();
}

function renderMedia() {
  const prompt = currentPrompt();
  renderKeyframes(prompt);
  renderSubjects(prompt);
  renderReferences(prompt);
  renderRefTasks(prompt);
  renderModeChip(prompt);
  const count = prompt
    ? ((prompt.first_frame || "").trim() ? 1 : 0)
      + ((prompt.last_frame || "").trim() ? 1 : 0)
      + (prompt.references || []).length
    : 0;
  const badge = $("media-count");
  if (badge) badge.textContent = count ? String(count) : "";
}

// -- cast ------------------------------------------------------------------
//
// The project's reusable identities. A member is stored on the project, not on
// the prompt, and prompts point at one by id — so every write here goes
// straight to the server rather than riding along on the next source save: an
// appearance edit changes how *other* segments compose, and half of them are
// not on screen to be saved.

// The third entry heads that kind's section; the second still labels the
// select, where one member is being named rather than a group counted.
const CAST_KINDS = [
  ["character", "Character", "Characters"],
  ["environment", "Location", "Locations"],
  ["prop", "Prop", "Props"],
  ["voice", "Voice", "Voices"],
];

/* What kind of thing a member is, as a picture. Four cards down the panel all
 * said "Character" or "Location" in a select you had to read; the marker says
 * it before you read anything. */
const CAST_KIND_ICONS = {
  character: "users",
  environment: "layout",
  prop: "archive",
  voice: "volume",
};

const KNOWN_KINDS = new Set(CAST_KINDS.map(([value]) => value));

/* Which section a member files under. A kind the editor does not know about
 * has no section of its own to sit in, and a member who files nowhere would
 * drop off the tab altogether — so an unknown kind files with the characters
 * rather than vanishing. */
const memberKind = (member) => (KNOWN_KINDS.has(member.kind) ? member.kind : "character");

const castMembers = () => state.project?.cast || [];
const memberById = (id) => castMembers().find((member) => member.id === id) || null;

/** What the prompt calls this member, for a chip beside a speaker key. */
function memberLabel(id) {
  const member = memberById(id);
  if (member) return member.name || "unnamed";
  return looksLikeMemberId(id) ? `${id} (missing)` : id;
}

const looksLikeMemberId = (key) => /^cast_[0-9a-f]{8}$/.test(key || "");

/* The tab is a grid of tiles under a heading per kind, not one long column of
 * open forms. A cast grows past what a column can be read at: by twenty
 * members, "is she on screen in this shot?" means scrolling past nineteen
 * appearance paragraphs to find out. So a tile shows only what you scan for
 * — the face, the name, and the two decisions about this segment — and the
 * paragraphs move into a sheet the tile opens. */
function renderCast() {
  const prompt = currentPrompt();
  state.castSignature = JSON.stringify(
    [speakingMemberIds(prompt), prompt?.cast || [], prompt?.undescribed || []]);
  const list = $("castlist");
  const badge = $("cast-count");
  if (badge) badge.textContent = castMembers().length ? String(castMembers().length) : "";
  const inUse = new Set([...speakingMemberIds(prompt), ...(prompt?.cast || []),
    ...(prompt?.references || []).map(ref => ref.member).filter(Boolean)]);
  const current = $("inputs-castlist");
  if (current) {
    current.replaceChildren();
    const members = castMembers().filter(member => inUse.has(member.id));
    if (members.length) {
      const grid = document.createElement("div"); grid.className = "cast-grid";
      members.forEach(member => grid.append(castTile(member))); current.append(grid);
    } else {
      const empty = document.createElement("p"); empty.className = "input-empty";
      empty.textContent = "No identities used here yet. Open Project library to add characters, locations, props or voices.";
      current.append(empty);
    }
  }
  if (!list) return;

  list.textContent = "";
  if (!state.project) {
    list.innerHTML = `<div class="empty">Open a project to build its cast.</div>`;
    renderCastDetail();
    return;
  }
  if (!castMembers().length) {
    list.innerHTML = `<div class="empty">No cast yet. Add a character, location, prop or
      voice and reuse it across every source.</div>`;
    renderCastDetail();
    return;
  }
  for (const [kind, , plural] of CAST_KINDS) {
    const members = castMembers().filter((member) => memberKind(member) === kind);
    if (members.length) list.append(castSection(kind, plural, members));
  }
  // The open member's sheet is drawn from the same pass, so a tick, a rename
  // or a deletion never leaves the two disagreeing about who is on screen.
  renderCastDetail();
}

/** One kind's heading and its grid. A kind with no members draws nothing. */
function castSection(kind, plural, members) {
  const section = document.createElement("section");
  section.className = "cast-section";

  const heading = document.createElement("div");
  heading.className = "cast-sub";
  heading.innerHTML = iconHTML(CAST_KIND_ICONS[kind] || "users", "sm");
  heading.append(document.createTextNode(`${plural} · ${members.length}`));

  const grid = document.createElement("div");
  grid.className = "cast-grid";
  for (const member of members) grid.append(castTile(member));

  section.append(heading, grid);
  return section;
}

/** A member at a glance, and the two ticks that are about this segment. */
function castTile(member) {
  const prompt = currentPrompt();
  const speaks = speakingMemberIds(prompt).includes(member.id);
  const silent = (prompt?.cast || []).includes(member.id);

  const tile = document.createElement("div");
  tile.className = `cast-tile${speaks || silent ? " here" : ""}`;
  tile.tabIndex = 0;
  tile.ondblclick = () => openCastDetail(member.id);
  // Enter opens it from the keyboard, but only while the tile itself has the
  // focus: the ticks inside are focusable too, and a checkbox keeps its keys.
  tile.onkeydown = (event) => {
    if (event.target !== tile || event.key !== "Enter") return;
    event.preventDefault();
    openCastDetail(member.id);
  };

  const ticks = document.createElement("div");
  ticks.className = "cast-tile-ticks";
  ticks.append(castPresence(member, prompt, speaks, silent),
    castDescribed(member, prompt, speaks, silent));

  tile.append(castFace(member), castTileHead(member), ticks);
  return tile;
}

/** The member's first identity picture, or their kind's mark if they have none. */
function castFace(member) {
  const face = document.createElement("div");
  face.className = "cast-face";

  const files = member.references || [];
  const shown = files.find((file) => file.kind === "image" || file.kind === "video");
  if (shown) {
    // Not `mediaThumb`: that makes the picture open the lightbox on a click,
    // and on a tile a click has to belong to the tile.
    const node = document.createElement(shown.kind === "image" ? "img" : "video");
    node.src = mediaViewUrl(shown.path);
    if (shown.kind === "image") node.alt = "";
    else {
      node.muted = true;
      node.preload = "metadata";
    }
    face.append(node);
  } else {
    const mark = document.createElement("div");
    mark.className = "cast-face-mark";
    mark.innerHTML = iconHTML(CAST_KIND_ICONS[memberKind(member)] || "users");
    face.append(mark);
  }

  // Counted rather than shown: a member's other sheets are what the detail
  // sheet is for, and four thumbnails on a tile this size are four
  // unreadable ones.
  if (files.length) {
    const count = document.createElement("span");
    count.className = "cast-face-count";
    count.innerHTML = iconHTML("image", "sm");
    count.append(document.createTextNode(String(files.length)));
    count.title = `${files.length} identity file${files.length === 1 ? "" : "s"}`;
    face.append(count);
  }

  const open = document.createElement("button");
  open.className = "iconbtn small cast-open";
  open.title = "Open this member — appearance, voice and identity files";
  open.setAttribute("aria-label", `Open ${member.name || "this member"}`);
  open.innerHTML = iconHTML("maximize", "sm");
  open.onclick = (event) => {
    event.stopPropagation();
    openCastDetail(member.id);
  };
  face.append(open);

  return face;
}

/** Name, and one line of what they are, because names go vague. */
function castTileHead(member) {
  const head = document.createElement("div");
  head.className = "cast-tile-head";

  const name = document.createElement("div");
  name.className = "cast-tile-name";
  name.textContent = member.name || "unnamed";
  head.append(name);

  const said = (member.appearance || member.voice || "").trim();
  if (said) {
    const line = document.createElement("div");
    line.className = "cast-tile-said";
    line.textContent = said;
    head.append(line);
  }
  // One tooltip for the whole head rather than one per line: the name and the
  // description are both clipped, and both are wanted at once.
  head.title = said ? `${member.name || "unnamed"} — ${said}` : (member.name || "unnamed");
  return head;
}

// -- one member, opened ----------------------------------------------------
//
// What a member *is*, as opposed to what they do in this shot: appearance,
// voice, notes, and the identity files with the role sentence on each. All of
// it used to be on the tab for every member at once, which is what made the
// tab unreadable once a project had a real cast.

function openCastDetail(id) {
  if (!memberById(id)) return;
  state.castDetail = id;
  $("castdetail").classList.add("open");
  renderCastDetail();
  $("castdetail-head").querySelector(".cast-detail-close")?.focus();
}

function closeCastDetail() {
  state.castDetail = "";
  $("castdetail").classList.remove("open");
}

/** Redraw the open member — or shut the sheet, if they are gone. */
function renderCastDetail() {
  const sheet = $("castdetail");
  if (!sheet) return;
  const member = state.castDetail ? memberById(state.castDetail) : null;
  // Deleting the member you have open takes the sheet with them, rather than
  // leaving a form writing to an id the project no longer holds.
  if (!member) {
    if (state.castDetail) closeCastDetail();
    return;
  }

  const head = $("castdetail-head");
  head.textContent = "";
  head.append(castDetailHead(member));

  // The body is refilled rather than replaced, and its scroll is put back:
  // every field here saves on change, and a save redraws the sheet — which
  // would otherwise throw you to the top each time you finished a sentence.
  const body = $("castdetail-body");
  const scroll = body.scrollTop;
  body.textContent = "";
  body.append(castDetailBody(member));
  body.scrollTop = scroll;
}

/** The sheet's title row: kind mark, name, kind, delete, close. */
function castDetailHead(member) {
  const frag = document.createDocumentFragment();

  const marker = document.createElement("div");
  marker.innerHTML = iconHTML(CAST_KIND_ICONS[memberKind(member)] || "users");
  const markerIcon = marker.firstElementChild;
  markerIcon.classList.add("cast-kind-ico");

  const name = document.createElement("input");
  name.type = "text";
  name.className = "cast-name";
  name.value = member.name || "";
  name.placeholder = "Name";
  name.onchange = guard(() => saveMember({ ...member, name: name.value }));

  const kind = document.createElement("select");
  for (const [value, text] of CAST_KINDS) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    option.selected = value === member.kind;
    kind.append(option);
  }
  kind.onchange = guard(() => saveMember({ ...member, kind: kind.value }));

  const remove = document.createElement("button");
  remove.className = "iconbtn small danger";
  remove.title = "Delete this member";
  remove.setAttribute("aria-label", "Delete this member");
  remove.innerHTML = iconHTML("trash", "sm");
  remove.onclick = guard(() => deleteMember(member));

  const close = document.createElement("button");
  close.className = "iconbtn small cast-detail-close";
  close.title = "Close (Esc)";
  close.setAttribute("aria-label", "Close");
  close.innerHTML = iconHTML("x", "sm");
  close.onclick = closeCastDetail;

  frag.append(markerIcon, name, kind, remove, close);
  return frag;
}

/** Everything about the member that takes reading, plus this segment's ticks. */
function castDetailBody(member) {
  const prompt = currentPrompt();
  const speaks = speakingMemberIds(prompt).includes(member.id);
  const silent = (prompt?.cast || []).includes(member.id);

  const body = document.createDocumentFragment();

  // -- what this member does in *this* segment, kept apart from what the
  // member is. Two different questions, so two different places. The tile
  // carries the same two ticks, and both write the same field.
  const ticks = document.createElement("div");
  ticks.className = "cast-ticks";
  const tag = document.createElement("span");
  tag.className = "seg-tag";
  tag.textContent = "This segment";
  ticks.append(tag,
    castPresence(member, prompt, speaks, silent),
    castDescribed(member, prompt, speaks, silent));
  body.append(ticks);

  // -- the description every segment inherits
  body.append(
    castField("Appearance", member.appearance, "a tall woman in a rain-darkened red coat",
      (value) => saveMember({ ...member, appearance: value }), true),
    castField("Voice", member.voice, "a low, unhurried voice with a faint rasp",
      (value) => saveMember({ ...member, voice: value })),
  );

  // Notes never reach the model, so they are typeset as the aside they are
  // rather than as a third field of equal weight to the two that compose.
  const notes = castField("Notes", member.notes, "continuity reminders, casting intent",
    (value) => saveMember({ ...member, notes: value }));
  notes.classList.add("aside");
  notes.querySelector("label").append(mutedTag("never composed"));
  body.append(notes);

  body.append(castRefsHeading(member), castReferences(member));
  return body;
}

/** The small uppercase caveat that rides beside a label. */
function mutedTag(text) {
  const span = document.createElement("span");
  span.className = "muted-tag";
  span.textContent = text;
  return span;
}

/** Divides the member's own fields from the files that prove their identity. */
function castRefsHeading(member) {
  const heading = document.createElement("div");
  heading.className = "cast-sub";
  const count = (member.references || []).length;
  heading.innerHTML = iconHTML("image", "sm");
  heading.append(document.createTextNode(
    count ? `Identity files · ${count}` : "Identity files"));
  return heading;
}

/** The "in this segment" tick — only meaningful for a member who stays silent. */
function castPresence(member, prompt, speaks, silent) {
  const wrap = document.createElement("label");
  wrap.className = "cast-here";

  const box = document.createElement("input");
  box.type = "checkbox";
  // The tick is the silent listing, and nothing else. Someone who speaks is
  // already on screen, so the box reads as checked either way — but it only
  // locks when there is no listing behind it to clear, otherwise a member
  // listed first and given a line later would be stuck with a dead entry.
  box.checked = speaks || silent;
  box.disabled = !prompt || (speaks && !silent);
  box.title = !speaks
    ? "Put this member on screen in this segment without giving them a line."
    : silent
      ? "They already speak here, so this listing adds nothing — untick to tidy it away."
      : "Already in this segment as a speaker — untick by removing them from the line.";
  box.onchange = () => {
    if (!prompt) return;
    if (box.checked) prompt.cast = [...(prompt.cast || []), member.id];
    // Unticking has to clear every route on screen, not just the listing: a
    // linked reference row left behind puts the member straight back as a
    // subject, and the tick then reads as having done nothing.
    else takeMemberOffScreen(prompt, member.id);
    markDirty();
    renderCast();
    renderMedia();
    refresh();
  };

  wrap.append(box, document.createTextNode("On screen"));
  return wrap;
}

/** The "describe them" tick — whether their appearance opens the shot.
 *
 * On by default, and only meaningful for a member who is present and silent: a
 * speaker is identified on their own line instead. Untick it to keep a
 * member's identity files while dropping the paragraph they otherwise write
 * into the opening — which is what a close shot wants from a location, whose
 * appearance runs to window walls and doorways the camera would then go and
 * find.
 *
 * It needs identity files under it to mean anything. With no files there is no
 * `<Subject N>` for the tick to fall back to, so unticking would not trim the
 * member's description — it would delete the member from the prompt outright.
 * The box locks rather than hides in that case, and only in the direction that
 * enters the bad state, so a member left unticked before their files went away
 * can still be ticked back on.
 */
function castDescribed(member, prompt, speaks, silent) {
  const wrap = document.createElement("label");
  wrap.className = "cast-here";

  const described = !(prompt?.undescribed || []).includes(member.id);
  const files = (member.references || []).length;

  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = described;
  box.disabled = !prompt || !silent || speaks || (described && !files);
  box.title = speaks
    ? "They speak here, so they are identified on their line rather than in the opening."
    : !silent
      ? "Only applies to a member who is on screen in this segment."
      : described && !files
        ? "They have no identity files, so this description is the only thing that "
          + "puts them in the prompt at all. Attach a file first, and unticking then "
          + "keeps the file and drops the words."
        : "Their appearance opens the shot AND names them in subject_definitions. "
          + "Untick to keep their identity files without either — the tag still routes "
          + "to the files, and the camera stops being told to find every feature.";
  box.onchange = () => {
    if (!prompt) return;
    const listed = prompt.undescribed || [];
    prompt.undescribed = box.checked
      ? listed.filter((id) => id !== member.id)
      : [...listed, member.id];
    markDirty();
    renderCast();
    refresh();
  };

  wrap.append(box, document.createTextNode("Describe them"));
  return wrap;
}

function castField(labelText, value, placeholder, onSave, big = false) {
  const field = document.createElement("div");
  field.className = "field";
  const tag = document.createElement("label");
  tag.textContent = labelText;

  const box = makeGrow(document.createElement("textarea"));
  box.rows = big ? 2 : 1;
  box.value = value || "";
  box.placeholder = placeholder;
  // On change, not on input: every keystroke would be a project write, and a
  // project write reloads the form under the caret.
  box.onchange = guard(() => onSave(box.value));

  field.append(tag, box);
  return field;
}

function castReferences(member) {
  const wrap = document.createElement("div");
  wrap.className = "cast-refs";

  for (const [index, reference] of (member.references || []).entries()) {
    const row = document.createElement("div");
    row.className = "cast-ref-row";

    const slot = document.createElement("div");
    slot.className = "cast-ref";
    slot.title = reference.path;
    slot.append(mediaThumb(reference.path, reference.kind, member.references));

    // Same swap the segment's reference rows have, and for the same reason: an
    // identity file carries a role sentence and a retention marker that are
    // easy to lose and slow to retype, so dropping the entry to point at a
    // redrawn sheet would quietly throw away words worth keeping.
    const swap = document.createElement("button");
    swap.className = "iconbtn small cast-swap";
    swap.title = "Replace this file. The role and retention on it stay.";
    swap.innerHTML = iconHTML("rotate", "sm");
    swap.onclick = guard(async () => {
      const file = await pickFile(KIND_ACCEPT[reference.kind] || "image/*");
      if (!file) return;
      const { filename } = await uploadMedia(file);
      await saveMember({
        ...member,
        references: member.references.map(
          (each, at) => (at === index ? { ...each, path: filename } : each),
        ),
      });
    });
    slot.append(swap);

    const clear = document.createElement("button");
    clear.className = "iconbtn small kf-clear";
    clear.title = "Remove this identity file";
    clear.innerHTML = iconHTML("x", "sm");
    clear.onclick = guard(() => saveMember({
      ...member,
      references: member.references.filter((_, at) => at !== index),
    }));
    slot.append(clear);
    row.append(slot);

    // What this file *is*, and how much of it to keep. Both compose — `role`
    // becomes the file's `subject_definitions` sentence and `retention` its
    // `retention_analysis` marker — and until now neither had a control
    // anywhere, so the most load-bearing prose in a reference document was the
    // prose nobody in the editor could read, let alone change.
    const saveRef = (patch) => saveMember({
      ...member,
      references: member.references.map(
        (each, at) => (at === index ? { ...each, ...patch } : each),
      ),
    });

    const body = document.createElement("div");
    body.className = "cast-ref-body";

    const role = document.createElement("input");
    role.type = "text";
    role.className = "role";
    role.value = reference.role || "";
    role.placeholder = "role — what this picture is, e.g. her full-body identity sheet";
    role.title = "Composed as “<Picture N> is <role>.” in subject_definitions, so write "
      + "it as a noun phrase and do not start it with “is”. Empty falls back to the "
      + "member’s appearance and voice.";
    // On change, not on input: every save is a POST that rebuilds these cards,
    // which would take the caret with it mid-word.
    role.onchange = guard(() => saveRef({ role: role.value }));

    const retention = retentionControl(
      reference.kind, reference.retention, null,
      guard((marker) => saveRef({ retention: marker })),
    );

    body.append(role, retention);
    row.append(body);
    wrap.append(row);
  }

  const actions = document.createElement("div");
  actions.className = "cast-ref-actions";

  const make = document.createElement("button");
  make.className = "small fixed";
  make.innerHTML = label("sparkles", "Generate", "sm");
  make.title = "Write this member's identity image with Krea 2 Turbo, starting "
    + "from their appearance";
  make.onclick = () => openStudio({ kind: "member", member: member.id });
  actions.append(make);

  const existing = document.createElement("button");
  existing.className = "small fixed";
  existing.innerHTML = label("image", "From the project", "sm");
  existing.title = "Attach a picture this project already holds — a library still, "
    + "a keyframe, another member's reference sheet";
  existing.onclick = () => openImagePicker(member);
  actions.append(existing);

  const add = document.createElement("button");
  add.className = "small fixed";
  add.innerHTML = label("plus", "Upload a file", "sm");
  add.title = "An image, video or audio clip from disk that fixes what this member "
    + "looks or sounds like";
  add.onclick = guard(async () => {
    const file = await pickFile("image/*,video/*,audio/*");
    if (!file) return;
    const { filename, kind } = await uploadMedia(file);
    await saveMember({
      ...member,
      references: [...(member.references || []), { kind: kind || "image", path: filename, role: "" }],
    });
  });
  actions.append(add);

  if (!(member.references || []).length) {
    const hint = document.createElement("span");
    hint.className = "cast-empty";
    hint.textContent = "no identity files — the description alone carries this member";
    actions.append(hint);
  }
  wrap.append(actions);
  return wrap;
}

// -- attaching a picture the project already has ---------------------------
//
// The three ways an identity file used to arrive were "generate a new one" and
// "upload one from disk", which between them miss the common case: the picture
// is already here. It was the keyframe of the shot this member was written for,
// or a still from the library, or the sheet another member already points at.
// Uploading it again would put a second copy in the input directory, and two
// copies of one face drift apart the moment either is regenerated.

/** Every image the project already points at, newest intent first. */
function projectImages() {
  const seen = new Set();
  const found = [];
  const offer = (path, where) => {
    if (!path || seen.has(path)) return;
    seen.add(path);
    found.push({ path, where });
  };

  for (const still of state.stills || []) {
    if (still.discarded || !still.path) continue;
    const madeFor = still.member ? memberById(still.member)?.name : sourceById(still.source)?.name;
    offer(still.path, madeFor ? `library — for ${displayName(madeFor)}` : "library");
  }
  for (const source of sources()) {
    const whose = displayName(source.name);
    for (const segment of source.segments || []) {
      const prompt = segment.prompt || {};
      offer(prompt.first_frame, `first frame — ${whose}`);
      offer(prompt.last_frame, `last frame — ${whose}`);
      for (const reference of prompt.references || []) {
        if (reference.kind === "image") offer(reference.path, `reference — ${whose}`);
      }
    }
  }
  for (const member of castMembers()) {
    for (const reference of member.references || []) {
      if (reference.kind === "image") {
        offer(reference.path, `identity — ${member.name || "unnamed member"}`);
      }
    }
  }
  return found;
}

/** Open the grid, aimed at the member the picked file becomes an identity for. */
function openImagePicker(member) {
  if (!state.project) return toast("Open a project first.", true);
  state.picker = { member: member.id };
  $("picker-scope").textContent = `for ${member.name || "this member"}`;
  $("picker").classList.add("open");
  renderImagePicker();
  $("picker-close").focus();
  // The library is only fetched when the studio or the bin has asked for it,
  // and this is a third door onto the same shelf.
  if (!(state.stills || []).length) {
    reloadStills().then(renderImagePicker).catch((error) => toast(error.message, true));
  }
}

function closeImagePicker() {
  state.picker = null;
  $("picker").classList.remove("open");
}

function renderImagePicker() {
  const grid = $("picker-grid");
  if (!grid || !state.picker) return;
  const member = memberById(state.picker.member);
  if (!member) return closeImagePicker();

  grid.textContent = "";
  const already = new Set((member.references || []).map((reference) => reference.path));
  const offered = projectImages().filter((image) => !already.has(image.path));

  if (!offered.length) {
    grid.innerHTML = `<div class="empty">Nothing here yet that ${escapeHtml(
      member.name || "this member")} does not already have. Generate one, or
      upload a file.</div>`;
    return;
  }

  for (const image of offered) {
    const pick = document.createElement("button");
    pick.className = "picker-pick";
    pick.title = `${image.path}

Attach as an identity file`;

    const shot = document.createElement("img");
    shot.src = mediaViewUrl(image.path);
    shot.alt = mediaName(image.path);
    pick.append(shot);

    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = image.where;
    pick.append(tag);

    pick.onclick = guard(async () => {
      // Read again rather than closing over the card's copy: the sheet stays
      // open for a second pick, and the member has grown by one since.
      const current = memberById(state.picker.member);
      if (!current) return closeImagePicker();
      await saveMember({
        ...current,
        references: [...(current.references || []),
                     { kind: "image", path: image.path, role: "" }],
      });
      toast(`Attached to ${current.name || "the member"}.`);
      renderImagePicker();
    });
    grid.append(pick);
  }
}

/** Members with a spoken line somewhere in this prompt. */
function speakingMemberIds(prompt) {
  const found = [];
  for (const beat of prompt?.beats || []) {
    for (const line of beat.lines || []) {
      if (!(line.text || "").trim()) continue;
      for (const key of line.speakers || []) {
        if (memberById(key) && !found.includes(key)) found.push(key);
      }
    }
  }
  return found;
}

// A line gaining or losing words changes who counts as speaking, and the tick
// beside each member has to follow — but the cards are full of live inputs, so
// they are only rebuilt when the membership itself changed. Typing prose leaves
// them alone, which is what keeps the caret where it was.
function syncCastPresence() {
  if (!state.project) return;
  const prompt = currentPrompt();
  const signature = JSON.stringify(
    [speakingMemberIds(prompt), prompt?.cast || [], prompt?.undescribed || []]);
  if (signature === state.castSignature) return;
  state.castSignature = signature;
  renderCast();
}

async function saveMember(member) {
  if (!state.project) return;
  await apply(post(`${API}/cast/save`, { project: state.project.name, member }));
  renderCast();
}

async function addMember() {
  if (!state.project) return toast("Open a project first.", true);
  await saveMember({ name: `Member ${castMembers().length + 1}`, kind: "character" });
}

async function deleteMember(member) {
  if (!state.project) return;
  const { used_by: usedBy } = await call(
    `${API}/cast/usage?project=${encodeURIComponent(state.project.name)}` +
    `&member=${encodeURIComponent(member.id)}`);
  const also = usedBy.length
    ? `\n\n${usedBy.length} segment(s) name them and will start warning until you point ` +
      `those lines somewhere else.`
    : "";
  if (!window.confirm(`Delete ${member.name || "this member"}?${also}`)) return;
  await apply(post(`${API}/cast/delete`, {
    project: state.project.name, member: member.id,
  }));
  renderCast();
}

// -- the image studio ------------------------------------------------------
//
// Krea 2 Turbo, in the editor. The generator existed on the server long before
// this — an agent could write a keyframe and hand it back — but from inside the
// editor the only way to fill a keyframe slot was to upload a file made
// somewhere else. This is the same call with the knobs exposed.
//
// It is one sheet rather than three because writing an image, sweeping seeds
// for it and choosing the winner are one activity, and the only thing that
// differs between a first frame, a reference and a cast member's identity file
// is where the picked image lands. So the destination is a control, and every
// image ever generated stays in the project's library — a keyframe you liked in
// shot two is exactly what shot nine wants.

// Delivery sizes for a generated image. Every one is a multiple of 16 (the Wan
// latent grid Krea 2 shares) and generous: a keyframe is resized to the shot's
// canvas on the way into i2v, so the only cost of a bigger picture is time.
const STILL_SIZES = [
  { label: "1920 × 1088 — landscape", width: 1920, height: 1088 },
  { label: "1088 × 1920 — portrait", width: 1088, height: 1920 },
  { label: "1456 × 1088 — 4:3", width: 1456, height: 1088 },
  { label: "1088 × 1088 — square", width: 1088, height: 1088 },
];

const studioOpen = () => Boolean(state.studio);

async function loadStyles() {
  state.styles = (await call(`${API}/t2i/styles`)).styles ?? [];
}

async function reloadStills() {
  if (!state.project) {
    state.stills = [];
    return;
  }
  const name = encodeURIComponent(state.project.name);
  state.stills = (await call(`${API}/stills?project=${name}`)).stills ?? [];
}

/** Open the studio aimed at somewhere: a keyframe, a reference, a member. */
function openStudio(target = { kind: "none" }) {
  if (!state.project) return toast("Open a project first.", true);
  state.studio = { target, source: currentSource()?.id || "" };
  fillStudioForm();
  $("studio").classList.add("open");
  renderStudioGrid();
  paintStudioLlm();
  $("studio-ask").value = "";
  $("studio-ask-status").textContent = "";
  $("studio-prompt").focus();
  reloadStills().then(renderStudioGrid).catch((error) => toast(error.message, true));
  loadImageHistoryCount();
}

function closeStudio() {
  rememberDraft();
  state.studio = null;
  state.studioComparing = false;
  $("studio").classList.remove("open");
}

/** File whatever is in the box against the destination it was written for. */
function rememberDraft() {
  const box = $("studio-prompt");
  if (!box || !state.studio) return;
  const key = targetValue(state.studio.target);
  const text = box.value;
  if (text.trim()) state.studioDrafts[key] = text;
  else delete state.studioDrafts[key];
}

/** Every place a picked image can go, given what is open. */
function studioDestinations() {
  const prompt = currentPrompt();
  const options = [];
  if (prompt) {
    // Named, because this list is also read from the bin, where no segment is
    // on screen and "First frame" on its own does not say whose.
    const whose = currentSource()?.name ? ` — ${displayName(currentSource().name)}` : "";
    options.push({ value: "first_frame", label: `First frame${whose}` });
    options.push({ value: "last_frame", label: `Last frame${whose}` });
    options.push({ value: "reference", label: `Reference${whose}` });
  }
  for (const member of castMembers()) {
    options.push({
      value: `member:${member.id}`,
      label: `Identity file — ${member.name || "unnamed member"}`,
    });
  }
  options.push({ value: "none", label: "Nowhere — just build the library" });
  return options;
}

function targetValue(target) {
  if (!target) return "none";
  if (target.kind === "keyframe") return target.key;
  if (target.kind === "reference") return "reference";
  if (target.kind === "member") return `member:${target.member}`;
  return "none";
}

function targetFromValue(value) {
  if (value === "first_frame" || value === "last_frame") return { kind: "keyframe", key: value };
  if (value === "reference") return { kind: "reference" };
  if (String(value).startsWith("member:")) return { kind: "member", member: value.slice(7) };
  return { kind: "none" };
}

function fillStudioForm() {
  const { target } = state.studio;

  // The look presets, asked for once at boot. Their own language is the
  // tooltip, so choosing one is not a guess about what it will add.
  const styles = $("studio-style");
  if (!styles.options.length) {
    styles.append(new Option("none — my words only", ""));
    for (const style of state.styles) {
      const option = new Option(style.name, style.name);
      option.title = style.prompt;
      styles.append(option);
    }
  }

  // Sizes, with the one closest to the segment's own canvas marked and chosen.
  // Nearest ratio rather than an exact match: H3 canvases are 7:4-ish and none
  // of these are exactly that, so exact matching would silently offer nothing —
  // and shape, not size, is the part of a keyframe that has to agree with the
  // video, since the image is resized on the way into i2v anyway.
  const sizes = $("studio-size");
  const prompt = currentPrompt();
  const canvas = prompt?.width && prompt?.height ? prompt.width / prompt.height : null;
  const closest = canvas
    ? STILL_SIZES.reduce((best, size) =>
        Math.abs(size.width / size.height - canvas) < Math.abs(best.width / best.height - canvas)
          ? size : best)
    : null;
  sizes.textContent = "";
  for (const size of STILL_SIZES) {
    const text = size === closest
      ? `${size.label} — closest to this segment (${prompt.width}×${prompt.height})`
      : size.label;
    sizes.append(new Option(text, `${size.width}x${size.height}`));
  }
  if (closest) sizes.value = `${closest.width}x${closest.height}`;

  const counts = $("studio-count");
  if (!counts.options.length) {
    for (const count of SWEEP_COUNTS) counts.append(new Option(`×${count}`, String(count)));
    counts.value = "2";
  }

  const targets = $("studio-target");
  targets.textContent = "";
  for (const option of studioDestinations()) {
    targets.append(new Option(option.label, option.value));
  }
  targets.value = targetValue(target);
  // A destination that is not on offer (no segment open, member deleted) falls
  // back rather than leaving the control showing something it cannot do.
  if (!targets.value) targets.value = "none";
  state.studio.target = targetFromValue(targets.value);

  const scope = $("studio-scope");
  const source = currentSource();
  if (target.kind === "member") {
    const member = memberById(target.member);
    scope.textContent = `for ${member?.name || "a cast member"}`;
  } else if (source) {
    scope.textContent = `for ${source.name}`;
  } else {
    scope.textContent = state.project?.name || "";
  }

  $("studio-mine").checked = false;
  $("studio-mine").disabled = !state.studio.source;
  // The narrowing controls follow the state rather than resetting: a sheet
  // reopened to look at the starred images should still be showing them.
  $("studio-fav").checked = state.studioFav;
  $("studio-find").value = state.studioQuery;
  prefillStudioPrompt();
  renderStudioHold();
  $("studio-status").textContent = "";
}

// What the box should hold for the destination now aimed at. A draft written
// for that slot wins — the alley you wrote for shot two survives a detour
// through the courier's portrait and is still there when you come back. Only a
// slot with no draft falls through to a guess at what the picture is of, since
// a guess is a starting point, never something to overwrite work with.
function prefillStudioPrompt() {
  const box = $("studio-prompt");
  const { target } = state.studio;

  const draft = state.studioDrafts[targetValue(target)];
  if (draft && draft.trim()) {
    box.value = draft;
    return;
  }
  box.value = "";

  if (target.kind === "member") {
    const member = memberById(target.member);
    if (!member) return;
    // An identity file is a portrait of the description, so the description is
    // the prompt — with the framing an identity reference wants added.
    const description = (member.appearance || "").trim();
    if (!description) return;
    box.value = member.kind === "environment"
      ? `${description}, establishing wide shot, no people`
      : `${description}, clean full-body reference portrait, neutral background, even lighting`;
    return;
  }

  const prompt = currentPrompt();
  if (!prompt) return;
  const opening = (prompt.beats || []).find((beat) => (beat.description || "").trim());
  const bits = [opening?.description, prompt.style].map((bit) => (bit || "").trim());
  const suggestion = bits.filter(Boolean).join(" ");
  if (suggestion) box.value = suggestion;
}

/** The library as the grid shows it: what is still generating, then what landed. */
function renderStudioGrid() {
  const grid = $("studio-grid");
  if (!grid || !studioOpen()) return;
  renderRecall();
  updateCompareButton();
  grid.textContent = "";
  grid.classList.toggle("comparing", state.studioComparing);

  if (state.studioComparing) return renderStudioCompare(grid);

  const pending = [...state.jobs.values()].filter(isStillJob);
  for (const job of pending) grid.append(pendingStillCard(job));

  const stills = studioStills();

  if (!stills.length && !pending.length) {
    const narrowed = state.studioFav || state.studioQuery
      || ($("studio-mine").checked && state.studio?.source);
    grid.innerHTML = narrowed
      ? `<div class="empty">Nothing matches those filters. The library still has
         ${state.stills.filter((s) => !s.discarded).length} image(s) in it.</div>`
      : `<div class="empty">No images yet. Describe one above and press
         Generate — a couple of seeds at a time is the usual way to find a composition.</div>`;
    return;
  }
  for (const still of stills) grid.append(stillCard(still));
}

// How many images the compare view holds at once. Two is the gesture; four is
// the ceiling before each one is too small to judge on the same screen.
const MAX_STILL_COMPARE = 4;

/** Say what pressing Compare would do, and whether it can be pressed at all. */
function updateCompareButton() {
  const button = $("studio-compare");
  if (!button) return;
  const held = state.studioCompare.size;
  if (state.studioComparing) {
    button.classList.add("primary");
    button.title = "Back to the library";
    return;
  }
  button.classList.remove("primary");
  button.disabled = held < 2;
  button.title = held < 2
    ? "Tick two or more images with the compare button on their cards, "
      + "then press this to hold them side by side"
    : `Hold the ${held} ticked images side by side`;
}

/** The held images, large, beside each other. The point of the whole sheet. */
function renderStudioCompare(grid) {
  const chosen = [...state.studioCompare]
    .map((id) => state.stills.find((still) => still.id === id))
    .filter((still) => still && still.path);

  if (chosen.length < 2) {
    state.studioComparing = false;
    return renderStudioGrid();
  }

  for (const still of chosen) {
    const figure = document.createElement("figure");
    figure.className = "studio-cmp";

    const image = document.createElement("img");
    image.src = mediaViewUrl(still.path);
    image.alt = still.prompt_text || "generated image";
    image.title = "Look at it full size";
    image.onclick = () => openLightbox(chosen, chosen.indexOf(still));

    const caption = document.createElement("figcaption");
    const recipe = document.createElement("span");
    const bits = [`seed ${still.seed ?? "—"}`];
    if (still.style) bits.push(still.style);
    if (still.steps) bits.push(`${still.steps} steps`);
    recipe.textContent = bits.join(" · ");
    caption.append(recipe, Object.assign(document.createElement("span"), { className: "spacer" }));

    caption.append(favButton(still), copyButton(still));

    const use = document.createElement("button");
    use.className = "small primary";
    use.textContent = useLabel();
    use.disabled = state.studio?.target.kind === "none";
    use.onclick = guard(() => useStill(still));
    caption.append(use);

    const drop = document.createElement("button");
    drop.className = "small ghost iconbtn";
    drop.innerHTML = iconHTML("x", "sm");
    drop.title = "Take this one out of the comparison";
    drop.onclick = () => {
      state.studioCompare.delete(still.id);
      renderStudioGrid();
    };
    caption.append(drop);

    figure.append(image, caption);
    grid.append(figure);
  }
}

/** The star, wherever an image is shown. Reads and writes `favorite`. */
function favButton(still, className = "small ghost iconbtn") {
  const button = document.createElement("button");
  button.className = `${className}${still.favorite ? " on" : ""}`;
  button.textContent = still.favorite ? "★" : "☆";
  button.title = still.favorite
    ? "Starred — click to unstar"
    : "Star this image, so \"starred only\" finds it later";
  button.onclick = guard(async (event) => {
    event.stopPropagation();
    await annotateStill(still, { favorite: !still.favorite });
  });
  return button;
}

/** Copy an image's prompt. The thing the library exists to give back. */
function copyButton(still) {
  const button = document.createElement("button");
  button.className = "small ghost iconbtn";
  button.innerHTML = iconHTML("copy", "sm");
  button.title = "Copy this image's prompt to the clipboard";
  button.disabled = !(still.prompt_text || "").trim();
  button.onclick = (event) => {
    event.stopPropagation();
    copyPrompt(still.prompt_text);
  };
  return button;
}

/** One wording for every copy button on the image side. */
function copyPrompt(text) {
  const prompt = (text || "").trim();
  if (!prompt) return toast("No prompt recorded for this image.", true);
  navigator.clipboard.writeText(prompt);
  toast("Prompt copied to clipboard.");
}

// -- writing the image prompt with a model ---------------------------------
//
// The video side's writer merges the model's answer into named fields, so a
// rewrite of the music is unable to reach the storyboard. An image prompt is
// one string: there are no fields, so there is nothing to merge within, and a
// rewrite replaces the whole box. That is why the history below is not a nicety
// — it is the reason the answer can be applied straight away instead of
// arriving as a proposal to accept.
//
// The server files the outgoing text itself, so a rewrite is undoable even if
// this tab dies between the ask and the answer.

/** Grey the studio's model controls out, and say why, when there is no key. */
function paintStudioLlm() {
  const bar = $("studio-writer");
  if (!bar) return;
  const on = state.llm.configured;
  bar.classList.toggle("off", !on);
  for (const id of ["studio-ask", "studio-ask-go"]) {
    $(id).disabled = !on;
    $(id).title = llmWhy();
  }
}

/** Ask the model to write or rewrite what is in the box. */
async function askStudioWriter() {
  if (!state.llm.configured) return toast("No model key configured.", true);
  if (!state.project) return toast("Open a project first.", true);
  const said = $("studio-ask").value.trim();
  if (!said) return toast("Say what you want written or changed.", true);

  const box = $("studio-prompt");
  const status = $("studio-ask-status");
  const button = $("studio-ask-go");
  const stop = tickingStatus(status, `Asking ${state.llm.model}…`);
  const abort = new AbortController();
  const bell = setTimeout(() => abort.abort(), DRAFT_WAIT_MS);
  button.disabled = true;

  let result;
  try {
    result = await post(`${API}/llm/draft-image`, {
      project: state.project.name,
      prompt: box.value,
      instruction: said,
      style: $("studio-style").value || "",
      target: targetValue(state.studio?.target),
    }, { signal: abort.signal });
  } catch (error) {
    throw error.name === "AbortError"
      ? new Error(`${state.llm.model} did not answer within ${DRAFT_WAIT_MS / 1000}s. `
                  + "Nothing was changed.")
      : error;
  } finally {
    clearTimeout(bell);
    stop();
    button.disabled = !state.llm.configured;
  }

  if (!result.changed) {
    toast("The model gave back the same words.");
    return;
  }
  box.value = result.prompt;
  rememberDraft();
  $("studio-ask").value = "";
  await loadImageHistoryCount();
  toast("Rewritten. History has the version it replaced.");
  box.focus();
}

/** How many past versions there are, on the button that opens them. */
async function loadImageHistoryCount() {
  const label = $("studio-history-count");
  if (!label || !state.project) return;
  try {
    const { revisions = [] } = await call(
      `${API}/image_prompt/revisions?project=${encodeURIComponent(state.project.name)}`);
    state.imageHistory.revisions = revisions;
    label.textContent = revisions.length ? `History (${revisions.length})` : "History";
  } catch {
    // A count is decoration; failing to fetch it must not break the sheet.
  }
}

/** File what is in the box as a past version, before something replaces it. */
async function fileImagePrompt(text, note) {
  if (!state.project || !(text || "").trim()) return;
  try {
    await post(`${API}/image_prompt/revisions/file`, {
      project: state.project.name,
      text,
      note,
      target: targetValue(state.studio?.target),
    });
    await loadImageHistoryCount();
  } catch (error) {
    // Filing is the safety net, so say when it did not catch — silently losing
    // the previous version is the one outcome this whole path exists to avoid.
    toast(`Could not file the previous prompt: ${error.message}`, true);
  }
}

async function openImageHistory() {
  if (!state.project) return toast("Open a project first.", true);
  const { revisions = [] } = await call(
    `${API}/image_prompt/revisions?project=${encodeURIComponent(state.project.name)}`);
  state.imageHistory = { revisions, picked: revisions[0]?.id ?? null };
  $("imghist-stat").textContent = revisions.length
    ? `${revisions.length} past version(s) of the image prompt`
    : "";
  renderImageHistory();
  $("imghist").classList.add("open");
}

function closeImageHistory() {
  $("imghist").classList.remove("open");
}

function renderImageHistory() {
  const list = $("imghist-list");
  list.textContent = "";
  const { revisions, picked } = state.imageHistory;

  if (!revisions.length) {
    list.innerHTML = `<div class="empty" style="padding:14px">Nothing yet. Anything
      that replaces the words in the box — a rewrite, a reuse, a restore — files
      the version it displaced here first.</div>`;
    $("imghist-diff").textContent = "";
    $("imghist-restore").disabled = true;
    $("imghist-copy").disabled = true;
    return;
  }

  const now = document.createElement("div");
  now.className = "hist-row now";
  now.innerHTML = '<div class="why">As the box stands now</div>';
  list.append(now);

  for (const revision of revisions) {
    const row = document.createElement("div");
    row.className = `hist-row${revision.id === picked ? " active" : ""}`;
    const when = document.createElement("div");
    when.className = "when";
    // Which slot it was written for, when it says — the trail is one per
    // project, so "whose prompt was this" is worth a glance.
    when.textContent = [whenText(revision.at), imageTargetLabel(revision.target)]
      .filter(Boolean).join(" · ");
    const why = document.createElement("div");
    why.className = "why";
    why.textContent = revision.note || "replaced in the editor";
    row.append(when, why);
    row.onclick = () => {
      state.imageHistory.picked = revision.id;
      renderImageHistory();
    };
    list.append(row);
  }

  const chosen = revisions.find((entry) => entry.id === picked);
  $("imghist-restore").disabled = !chosen;
  $("imghist-copy").disabled = !chosen;
  renderImageHistoryDiff(chosen);
}

/** A short name for the destination a past prompt was aimed at. */
function imageTargetLabel(value) {
  if (!value || value === "none") return "";
  if (value === "first_frame") return "first frame";
  if (value === "last_frame") return "last frame";
  if (value === "reference") return "reference";
  if (String(value).startsWith("member:")) {
    return memberById(String(value).slice(7))?.name || "cast member";
  }
  return "";
}

/** That version against the words in the box, as one marked-up run. */
function renderImageHistoryDiff(revision) {
  const pane = $("imghist-diff");
  pane.textContent = "";
  if (!revision) return;
  const chunks = diffWords(revision.text || "", $("studio-prompt").value || "");
  for (const chunk of chunks) {
    if (chunk.type === "same") pane.append(chunk.text);
    else pane.append(diffMark(chunk.type, chunk.text));
  }
}

/** Put a past version back in the box, filing what it displaces on the way. */
async function restoreImagePrompt() {
  const { revisions, picked } = state.imageHistory;
  const chosen = revisions.find((entry) => entry.id === picked);
  if (!chosen) return;
  // Round trip on purpose, as on the video side: restoring is itself an edit,
  // so undoing an undo is another restore rather than a special case.
  await fileImagePrompt($("studio-prompt").value, `restored ${chosen.id}`);
  $("studio-prompt").value = chosen.text;
  rememberDraft();
  closeImageHistory();
  toast("Put that version back. It is itself undoable.");
  $("studio-prompt").focus();
}

function wireImageWriter() {
  makeGrow($("studio-ask"));
  $("studio-ask-go").onclick = guard(askStudioWriter);
  $("studio-ask").onkeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("studio-ask-go").click();
    }
  };
  $("studio-history").onclick = guard(openImageHistory);
  $("imghist-close").onclick = closeImageHistory;
  closeOnBackdrop($("imghist"), closeImageHistory);
  $("imghist-restore").onclick = guard(restoreImagePrompt);
  $("imghist-copy").onclick = () => {
    const { revisions, picked } = state.imageHistory;
    copyPrompt(revisions.find((entry) => entry.id === picked)?.text);
  };
}

/** The library as the studio is currently filtering it. */
function studioStills() {
  const mineOnly = $("studio-mine").checked && state.studio?.source;
  const starredOnly = state.studioFav;
  // Matched against the words, not the recipe: you remember what you asked for,
  // not which seed answered. Every term has to appear, in any order.
  const terms = state.studioQuery.toLowerCase().split(/\s+/).filter(Boolean);
  return state.stills.filter((still) => {
    if (still.discarded) return false;
    if (mineOnly && still.source !== state.studio.source) return false;
    if (starredOnly && !still.favorite) return false;
    if (terms.length) {
      const text = (still.prompt_text || "").toLowerCase();
      if (!terms.every((term) => text.includes(term))) return false;
    }
    return true;
  });
}

/** Set a flag on a still, and redraw wherever it is showing. */
async function annotateStill(still, patch) {
  const { still: updated } = await post(`${API}/stills/annotate`, {
    project: state.project.name, still: still.id, ...patch,
  });
  // Patch in place rather than refetching the library: the grid is redrawn
  // straight after, and a round trip per star makes starring feel like work.
  Object.assign(still, updated || patch);
  renderStudioGrid();
  // The bin's list is one element shared by both its views, so redrawing the
  // image half while the shots half is showing would wipe the shots.
  if (state.binView === "images") renderBinImages();
  if (state.lightbox) renderLightbox();
}

// Every prompt this project has already generated from, newest first and
// deduped. The library records `prompt_text` on every still, so this is a view
// of data already held rather than a second thing to keep in step.
function recentPrompts() {
  const seen = new Set();
  const out = [];
  for (const still of state.stills) {
    const text = (still.prompt_text || "").trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
    if (out.length >= 40) break;
  }
  return out;
}

/** Refill the reuse list. Called whenever the library changes underneath it. */
function renderRecall() {
  const picker = $("studio-recall");
  if (!picker) return;
  const prompts = recentPrompts();
  picker.textContent = "";
  picker.append(new Option(
    prompts.length ? `${prompts.length} earlier prompt${prompts.length === 1 ? "" : "s"}…` : "nothing generated yet",
    "",
  ));
  for (const text of prompts) {
    // One line in the list, all of it on hover — a prompt is a paragraph and
    // the control is a select.
    const option = new Option(text.length > 90 ? `${text.slice(0, 89)}…` : text, text);
    option.title = text;
    picker.append(option);
  }
  picker.value = "";
  picker.disabled = !prompts.length;
}

function pendingStillCard(job) {
  const card = document.createElement("div");
  card.className = "still pending";
  const box = document.createElement("div");
  box.className = "miss";
  const working = document.createElement("div");
  working.className = "working";
  working.textContent = job.status === "running" ? "generating…" : "queued";
  const meta = document.createElement("div");
  meta.className = "still-meta";
  const recipe = document.createElement("div");
  recipe.className = "still-recipe";
  recipe.textContent = `seed ${job.seed ?? "—"}`;
  meta.append(recipe);
  card.append(box, working, meta);
  return card;
}

function stillCard(still) {
  const card = document.createElement("div");
  card.className = "still";
  if (still.favorite) card.classList.add("favorite");
  if (state.studioCompare.has(still.id)) card.classList.add("chosen");

  if (still.path) {
    const shot = document.createElement("img");
    shot.className = "shot";
    shot.src = mediaViewUrl(still.path);
    shot.alt = still.prompt_text || "generated image";
    shot.title = "Look at it full size";
    shot.onclick = () => {
      const shown = [...document.querySelectorAll("#studio-grid .still")]
        .indexOf(card) - document.querySelectorAll("#studio-grid .still.pending").length;
      openLightbox(studioStills(), Math.max(0, shown));
    };
    card.append(shot);
  } else {
    const gone = document.createElement("div");
    gone.className = "miss";
    gone.textContent = "the file is no longer on disk — the recipe below still is";
    card.append(gone);
  }

  const meta = document.createElement("div");
  meta.className = "still-meta";

  const recipe = document.createElement("div");
  recipe.className = "still-recipe";
  const bits = [`seed ${still.seed ?? "—"}`];
  if (still.style) bits.push(still.style);
  if (still.width && still.height) bits.push(`${still.width}×${still.height}`);
  const madeFor = still.member
    ? memberById(still.member)?.name
    : sourceById(still.source)?.name;
  if (madeFor) bits.push(`for ${madeFor}`);
  for (const bit of bits) {
    const span = document.createElement("span");
    span.textContent = bit;
    recipe.append(span);
  }

  const text = document.createElement("div");
  text.className = "still-text";
  text.textContent = still.prompt_text || "";
  text.title = "Click to read the whole prompt";
  // Unfolds in place. A prompt you cannot read is a prompt you cannot reuse,
  // and reuse is what the library is for.
  text.onclick = () => {
    const open = text.classList.toggle("open");
    text.title = open ? "" : "Click to read the whole prompt";
  };
  meta.append(recipe, text);

  // The star is a verdict on the picture, so it sits on the picture.
  card.append(favButton(still, "fav"));

  const acts = document.createElement("div");
  acts.className = "still-acts";

  const use = document.createElement("button");
  use.className = "small primary";
  use.textContent = useLabel();
  use.disabled = !still.path || state.studio?.target.kind === "none";
  use.title = still.path
    ? "Send this image to the destination chosen above"
    : "The image file is gone, so it cannot be attached to anything";
  use.onclick = guard(() => useStill(still));

  const compare = document.createElement("button");
  compare.className = "small ghost iconbtn";
  const held = state.studioCompare.has(still.id);
  compare.innerHTML = iconHTML(held ? "check-square" : "columns", "sm");
  compare.title = held
    ? "Held for comparison — click to drop it"
    : "Hold this one for comparison, then press Compare above";
  compare.onclick = () => {
    if (state.studioCompare.has(still.id)) state.studioCompare.delete(still.id);
    else if (state.studioCompare.size < MAX_STILL_COMPARE) state.studioCompare.add(still.id);
    else toast(`Compare holds at most ${MAX_STILL_COMPARE} images at once.`);
    renderStudioGrid();
  };

  const again = document.createElement("button");
  again.className = "small ghost iconbtn";
  again.innerHTML = iconHTML("rotate", "sm");
  again.title = "Load this recipe back into the form — same words, same look, "
    + "same seed, ready to vary one of them";
  again.onclick = () => loadRecipe(still);

  const forget = document.createElement("button");
  forget.className = "small ghost iconbtn";
  forget.innerHTML = iconHTML("trash", "sm");
  forget.title = "Take this out of the library. The file itself is left alone, "
    + "so anything already pointing at it keeps working.";
  forget.onclick = guard(() => forgetStill(still));

  const gap = document.createElement("span");
  gap.className = "spacer";
  acts.append(use, gap, copyButton(still), compare, again, forget);
  card.append(meta, acts);
  return card;
}

function useLabel() {
  const target = state.studio?.target;
  if (!target || target.kind === "none") return "Pick a destination";
  if (target.kind === "keyframe") {
    return target.key === "first_frame" ? "Use as first frame" : "Use as last frame";
  }
  if (target.kind === "reference") return "Add as reference";
  return "Add as identity file";
}

/** Send a picked image wherever the destination control says. */
async function useStill(still) {
  const applied = await applyStill(still, state.studio?.target);
  // A keyframe is one decision and it is now made; a reference or an identity
  // file is usually several, so only that one closes the sheet behind itself.
  if (applied && state.studio?.target?.kind === "keyframe") closeStudio();
}

/** Put an image on a field. Shared by the studio's grid and the lightbox. */
async function applyStill(still, target) {
  if (!still.path || !target || target.kind === "none") return false;

  if (target.kind === "member") {
    const member = memberById(target.member);
    if (!member) {
      toast("That cast member is gone.", true);
      return false;
    }
    await saveMember({
      ...member,
      references: [...(member.references || []), { kind: "image", path: still.path, role: "" }],
    });
    toast(`Added to ${member.name || "the member"}.`);
    return true;
  }

  const prompt = currentPrompt();
  if (!prompt) {
    toast("Select the source it belongs to first.", true);
    return false;
  }

  if (target.kind === "reference") {
    if ((prompt.first_frame || "").trim() || (prompt.last_frame || "").trim()) {
      toast("Clear the keyframe first — a segment is image OR reference driven.", true);
      return false;
    }
    prompt.references = [...(prompt.references || []),
      { kind: "image", path: still.path, role: "" }];
    markDirty();
    renderMedia();
    refresh();
    toast("Added as a reference.");
    return true;
  }

  if (effectiveReferenceCount(prompt)) {
    toast(castReferencePaths(prompt).length && !(prompt.references || []).length
      ? "A cast member here carries identity files, so this segment is "
        + "reference-driven. Take them off it first."
      : "Remove the references first — a segment is image OR reference driven.", true);
    return false;
  }
  prompt[target.key] = still.path;
  markDirty();
  renderMedia();
  refresh();
  toast(target.key === "first_frame" ? "Set as the first frame." : "Set as the last frame.");
  return true;
}

/** Queue the sweep. Same prepare → /prompt → history path as a video render. */
async function generateStills() {
  const text = $("studio-prompt").value.trim();
  if (!text) return toast("Say what the picture shows first.", true);
  if (!state.project) return toast("Open a project first.", true);

  const [width, height] = String($("studio-size").value).split("x").map(Number);
  const body = {
    prompt: text,
    steps: Number($("studio-steps").value) || 8,
    width,
    height,
    supersample: Number($("studio-super").value),
    count: Number($("studio-count").value) || 1,
  };
  const style = $("studio-style").value;
  if (style) body.style = style;
  const seed = parseSeed($("studio-seed").value);
  if (seed != null) body.seed = seed;

  const status = $("studio-status");
  status.textContent = "Preparing…";
  const button = $("studio-generate");
  button.disabled = true;
  try {
    const { renders = [] } = await post(`${API}/render/t2i/batch`, body);
    // An image is a generation like any other, so the render bar's hold toggle
    // governs it too — the shelf takes the identical payload either way.
    const holding = state.holdRenders;
    let queued = 0;
    for (const prepared of renders) {
      const ok = holding ? await holdStill(prepared) : await submitStill(prepared);
      if (ok) queued++;
    }
    status.textContent = queued
      ? `${queued} image${queued === 1 ? "" : "s"} ${holding ? "held for later" : "queued"}.`
      : "Nothing queued.";
    renderStudioGrid();
  } finally {
    button.disabled = false;
  }
}

// How an image generation identifies itself in the queue and in ComfyUI's
// history. The recorder matches it case-insensitively, so an agent that stamps
// a plain "t2i" is understood too.
const STILL_JOB_MODE = "Keyframe (T2I)";
const isStillJob = (job) => /^(t2i|keyframe \(t2i\))$/i.test(String(job?.mode || ""));

// The queue metadata is what makes an image findable afterwards: `mode` is how
// the recorder tells a picture from a clip, and the rest is the recipe, which
// nothing on the finished file itself records.
// One recipe, two doors. `member` and `style` live only here, which is why the
// shelf stores this payload whole rather than rebuilding it on release.
function buildStillPayload(prepared, promptId = uuid()) {
  const target = state.studio?.target || { kind: "none" };
  const sourceId = state.studio?.source || "";
  return {
    prompt: prepared.workflow,
    client_id: state.clientId,
    prompt_id: promptId,
    extra_data: {
      video_prompter: {
        project: state.project?.name,
        mode: STILL_JOB_MODE,
        shot: sourceId,
        shot_name: sourceById(sourceId)?.name || "",
        member: target.kind === "member" ? target.member : "",
        prompt: prepared.prompt,
        seed: prepared.seed,
        steps: prepared.steps,
        width: prepared.width,
        height: prepared.height,
        style: prepared.style || "",
      },
    },
  };
}

/** Freeze an image outside ComfyUI's queue, the way a held clip is frozen. */
async function holdStill(prepared) {
  const payload = buildStillPayload(prepared);
  try {
    await post(`${API}/render/hold`, {
      payload,
      summary: payload.extra_data.video_prompter,
    });
    await syncQueue();
    return true;
  } catch (error) {
    toast(`Could not hold image: ${error.message}`, true);
    return false;
  }
}

async function submitStill(prepared) {
  const promptId = uuid();
  const sourceId = state.studio?.source || "";
  const payload = buildStillPayload(prepared, promptId);

  try {
    const response = await fetch("/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const answer = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(answer.error?.message || `Queue refused: ${response.status}`);
    }
    const actualPromptId = answer.prompt_id || promptId;
    state.jobs.set(actualPromptId, {
      promptId: actualPromptId,
      extraData: payload.extra_data,
      shot: sourceId,
      shotName: sourceById(sourceId)?.name || "Keyframe",
      mode: STILL_JOB_MODE,
      seed: prepared.seed,
      steps: prepared.steps,
      width: prepared.width,
      height: prepared.height,
      prompt: prepared.prompt,
      workflow: prepared.workflow,
      project: state.project?.name || "",
      queuedAt: Date.now(),
      startTime: null,
      status: "queued",
      progress: { value: 0, max: prepared.steps || 8 },
    });
    if (typeof editorUX !== "undefined") editorUX?.tracker.outcome("still_submitted");
    syncQueue();
    return true;
  } catch (error) {
    if (typeof editorUX !== "undefined") editorUX?.tracker.outcome("still_rejected");
    toast(error.message, true);
    return false;
  }
}

function wireStudio() {
  const close = () => closeStudio();
  $("studio-close").onclick = close;
  closeOnBackdrop($("studio"), close);
  $("studio-generate").onclick = guard(generateStills);
  $("studio-target").onchange = () => {
    if (!state.studio) return;
    // File the words against the slot they were written for before aiming
    // somewhere else, then show whatever that slot already holds.
    rememberDraft();
    state.studio.target = targetFromValue($("studio-target").value);
    prefillStudioPrompt();
    renderStudioGrid();
  };
  $("studio-mine").onchange = renderStudioGrid;
  $("studio-fav").onchange = () => {
    state.studioFav = $("studio-fav").checked;
    renderStudioGrid();
  };
  $("studio-find").oninput = () => {
    state.studioQuery = $("studio-find").value.trim();
    renderStudioGrid();
  };
  $("studio-copy").onclick = () => copyPrompt($("studio-prompt").value);
  $("studio-compare").onclick = () => {
    if (state.studioComparing) state.studioComparing = false;
    else if (state.studioCompare.size >= 2) state.studioComparing = true;
    else return toast("Hold two or more images first.", true);
    renderStudioGrid();
  };
  // Reuse puts the words back in the box and leaves the seed and look alone —
  // the point is to edit a prompt that worked, not to re-run it unchanged.
  $("studio-recall").onchange = guard(async () => {
    const text = $("studio-recall").value;
    if (!text) return;
    // Reuse replaces the box, so what it displaces is filed first — the same
    // rule every other replacement here follows.
    await fileImagePrompt($("studio-prompt").value, "replaced by a reused prompt");
    $("studio-prompt").value = text;
    rememberDraft();
    $("studio-recall").value = "";
    $("studio-prompt").focus();
  });
  $("studio-seed").oninput = () => {
    $("studio-seed").classList.toggle("pinned", parseSeed($("studio-seed").value) != null);
  };
  $("studio-reroll").onclick = () => {
    $("studio-seed").value = "";
    $("studio-seed").classList.remove("pinned");
  };
  const kf = $("kf-generate");
  if (kf) kf.onclick = () => openStudio({ kind: "keyframe", key: "first_frame" });
  const ref = $("ref-generate");
  if (ref) ref.onclick = () => openStudio({ kind: "reference" });
}

// -- the lightbox ----------------------------------------------------------
//
// A generated image used to be inspected by opening it in a browser tab, which
// is a one-way trip out of the editor. This is the same close look without
// leaving: the arrow keys walk whichever set it was opened from, so comparing
// two seeds is two keypresses, and the destination control comes along — you
// judge a picture and use it from the same place, rather than going back to a
// grid to find it again.
//
// It also answers a click on any thumbnail the editor merely holds — a
// reference, a keyframe, an identity file. Those have no recipe under them and
// nowhere to be sent, so the footer that offers both goes away and the picture
// takes the room; the file's name becomes the heading, since that is the one
// thing a 78-pixel thumbnail could not tell you.

/** Open the viewer on `stills[index]`, keeping the set for prev/next. */
function openLightbox(stills, index = 0) {
  const list = (stills || []).filter((still) => still.path);
  if (!list.length) return;
  // `at` so a habitual double-click cannot close what its own first click just
  // opened: the second click lands on the backdrop, which is a close.
  state.lightbox = { list, index: Math.max(0, Math.min(index, list.length - 1)),
                     at: Date.now() };

  // A set is all stills or all plain files, never a mix, so the first entry
  // answers for the set: a plain file has no recipe to reuse and nowhere to be
  // sent, and its footer is off, so there is no destination to fill in.
  if (!list[0].plain) {
    const targets = $("lb-target");
    targets.textContent = "";
    for (const option of studioDestinations()) {
      targets.append(new Option(option.label, option.value));
    }
    // Aimed where the studio is aimed when it is open — the sheet behind this
    // one already asked the question. From the bin it defaults to the first
    // frame of whatever is selected, which is named in the option itself.
    const wanted = state.studio ? targetValue(state.studio.target) : "first_frame";
    targets.value = wanted;
    if (!targets.value) targets.value = "none";
  }

  $("lightbox").classList.add("open");
  renderLightbox();
  $("lb-close").focus();
}

/** Open the viewer on a media file the editor merely holds — a reference, a
 *  keyframe, an identity file — rather than on a still it generated. The same
 *  close look and the same arrow keys; there is simply no recipe under it, so
 *  the footer that reuses one stays down. Audio is dropped: there is nothing
 *  to see, and the row it sits in already plays it. */
function openMediaViewer(path, kind, set) {
  const near = (set && set.length ? set : [{ path, kind }])
    .filter((item) => item.path && item.kind !== "audio")
    .map((item) => ({ ...item, plain: true }));
  const at = near.findIndex((item) => item.path === path);
  openLightbox(near, Math.max(0, at));
}

function closeLightbox() {
  stopLightboxVideo();
  state.lightbox = null;
  $("lightbox").classList.remove("open");
}

/** Put the entry on the stage. A still is always a picture, but a reference can
 *  be a video, which needs a player rather than an <img> showing nothing. */
function showLightboxMedia(entry) {
  const url = mediaViewUrl(entry.path);
  const moving = entry.kind === "video";
  $("lb-image").hidden = moving;
  $("lb-video").hidden = !moving;
  if (moving) {
    // Only when it actually changes: this runs again on every refresh while the
    // viewer is open, and re-assigning the same source restarts the clip.
    if ($("lb-video").getAttribute("src") !== url) $("lb-video").src = url;
    return;
  }
  stopLightboxVideo();
  $("lb-image").src = url;
  $("lb-image").alt = entry.plain ? mediaName(entry.path)
    : entry.prompt_text || "generated image";
}

function stopLightboxVideo() {
  const video = $("lb-video");
  if (!video || !video.getAttribute("src")) return;
  video.pause();
  video.removeAttribute("src");
  // Without this the player keeps buffering the source it no longer shows.
  video.load();
}

function lightboxStill() {
  const box = state.lightbox;
  return box ? box.list[box.index] : null;
}

function stepLightbox(by) {
  const box = state.lightbox;
  if (!box || box.list.length < 2) return;
  box.index = (box.index + by + box.list.length) % box.list.length;
  renderLightbox();
}

function renderLightbox() {
  const still = lightboxStill();
  if (!still) return closeLightbox();
  const { list, index } = state.lightbox;

  showLightboxMedia(still);
  $("lb-count").textContent = list.length > 1 ? `${index + 1} of ${list.length}` : "";
  $("lb-prev").hidden = list.length < 2;
  $("lb-next").hidden = list.length < 2;

  // A file the editor merely holds has no recipe, no star and nowhere to be
  // sent, so the whole footer goes and the picture takes the room. Its name is
  // the heading instead, which is the one thing the grid could not tell you.
  $("lb-foot").hidden = !!still.plain;
  $("lb-title").textContent = still.plain ? mediaName(still.path) : "Image";
  if (still.plain) return;

  const recipe = $("lb-recipe");
  recipe.textContent = "";
  const bits = [`seed ${still.seed ?? "—"}`];
  if (still.style) bits.push(still.style);
  if (still.steps) bits.push(`${still.steps} steps`);
  if (still.width && still.height) bits.push(`${still.width}×${still.height}`);
  const madeFor = still.member
    ? memberById(still.member)?.name
    : sourceById(still.source)?.name;
  if (madeFor) bits.push(`for ${madeFor}`);
  for (const bit of bits) {
    const span = document.createElement("span");
    span.textContent = bit;
    recipe.append(span);
  }
  // The whole prompt, selectable and scrolling if it is long — not a two-line
  // caption with the rest hidden in a tooltip.
  $("lb-text").textContent = still.prompt_text || "";
  $("lb-text").title = "";
  $("lb-use").disabled = $("lb-target").value === "none";
  $("lb-copy").disabled = !(still.prompt_text || "").trim();

  const star = $("lb-fav");
  star.textContent = still.favorite ? "★" : "☆";
  star.classList.toggle("on", !!still.favorite);
  star.title = still.favorite ? "Starred — click to unstar" : "Star this image";
}

/** Put a still's recipe back in the generator, opening it if it is not up. */
function loadRecipe(still) {
  // Opened from the viewer with the sheet down, the destination chosen there is
  // the one the generator should come up aimed at.
  if (!studioOpen()) openStudio(targetFromValue($("lb-target").value));
  // This replaces the box like any other reuse, so file what it displaces.
  // Not awaited: the recipe should land the instant it is asked for, and the
  // filing is a background safety net rather than a step in the gesture.
  fileImagePrompt($("studio-prompt").value, "replaced by a reloaded recipe");
  $("studio-prompt").value = still.prompt_text || "";
  $("studio-style").value = still.style || "";
  $("studio-seed").value = still.seed ?? "";
  $("studio-seed").classList.toggle("pinned", still.seed != null);
  if (still.steps) $("studio-steps").value = still.steps;
  // The box now holds these words on purpose, so file them as this slot's
  // draft — reopening must not offer to replace them with a guess.
  rememberDraft();
  closeLightbox();
  $("studio-prompt").focus();
}

async function forgetStill(still) {
  await post(`${API}/stills/forget`, { project: state.project.name, still: still.id });
  await reloadStills();
  renderStudioGrid();
  renderBin();
}

function wireLightbox() {
  $("lb-close").onclick = closeLightbox;
  closeOnBackdrop($("lightbox"), () => {
    if (Date.now() - (state.lightbox?.at ?? 0) < 400) return;
    closeLightbox();
  });
  $("lb-prev").onclick = () => stepLightbox(-1);
  $("lb-next").onclick = () => stepLightbox(1);
  $("lb-target").onchange = renderLightbox;
  $("lb-use").onclick = guard(async () => {
    const still = lightboxStill();
    if (!still) return;
    if (await applyStill(still, targetFromValue($("lb-target").value))) closeLightbox();
  });
  $("lb-copy").onclick = () => copyPrompt(lightboxStill()?.prompt_text);
  $("lb-fav").onclick = guard(async () => {
    const still = lightboxStill();
    if (still) await annotateStill(still, { favorite: !still.favorite });
  });
  $("lb-again").onclick = () => {
    const still = lightboxStill();
    if (still) loadRecipe(still);
  };
  $("lb-forget").onclick = guard(async () => {
    const still = lightboxStill();
    if (!still) return;
    const box = state.lightbox;
    await forgetStill(still);
    box.list.splice(box.index, 1);
    if (!box.list.length) return closeLightbox();
    box.index = Math.min(box.index, box.list.length - 1);
    renderLightbox();
  });
}

// -- the bin's image view ---------------------------------------------------

function setBinView(view) {
  state.binView = view;
  $("bin-shots").setAttribute("aria-pressed", String(view === "shots"));
  $("bin-images").setAttribute("aria-pressed", String(view === "images"));
  $("bin-search").placeholder = view === "images" ? "Search images…" : "Search sources…";
  $("bin-shots").classList.toggle("active", view === "shots");
  $("bin-images").classList.toggle("active", view === "images");
  renderBin();
  // The count is only right once the library has been read, and the bin is on
  // screen from the moment a project opens.
  if (view === "images" && !state.stills.length) {
    reloadStills().then(renderBin).catch((error) => toast(error.message, true));
  }
}

function renderBinImages() {
  const list = $("bin-list");
  list.textContent = "";

  const make = document.createElement("button");
  make.className = "small primary bin-make";
  make.innerHTML = label("sparkles", "Generate images", "sm");
  make.title = "Write a picture with Krea 2 Turbo. Where it lands is chosen in the sheet.";
  make.onclick = () => openStudio(
    currentPrompt() ? { kind: "keyframe", key: "first_frame" } : { kind: "none" });
  list.append(make);

  const query = (state.binQuery || "").trim().toLowerCase();
  const stills = state.stills.filter((still) => !still.discarded &&
    (!query || [still.prompt_text, still.style, still.seed, still.path].join(" ").toLowerCase().includes(query)));
  if (!stills.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.style.padding = "8px";
    empty.textContent = query
      ? "No images match your search. Try a different word or clear the search."
      : "No images yet. Generate a keyframe, reference or cast image to start your library.";
    list.append(empty);
    return;
  }

  const grid = document.createElement("div");
  grid.className = "bin-images";
  stills.forEach((still, index) => {
    const cell = document.createElement("div");
    cell.className = "bin-still";
    cell.title = still.prompt_text || "";
    cell.tabIndex = 0;
    cell.setAttribute("role", "button");
    cell.setAttribute("aria-label", `Open image: ${still.style || still.prompt_text || `seed ${still.seed ?? "unknown"}`}`);
    cell.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        openLightbox(stills, index);
      }
    };
    if (still.path) {
      const img = document.createElement("img");
      img.src = mediaViewUrl(still.path);
      img.alt = "";
      cell.append(img);
    } else {
      const gone = document.createElement("div");
      gone.className = "miss";
      gone.textContent = "file gone";
      cell.append(gone);
    }
    const tag = document.createElement("span");
    tag.className = "tag";
    // A star set in the studio has to read as set from here too, or the flag
    // looks like it belongs to one sheet rather than to the image.
    tag.textContent = (still.favorite ? "★ " : "") + (still.style || `seed ${still.seed ?? "—"}`);
    cell.append(tag);
    cell.onclick = () => openLightbox(stills, index);
    grid.append(cell);
  });
  list.append(grid);
}

function renderBinCount() {
  const badge = $("bin-image-count");
  if (badge) {
    const count = state.stills.filter((still) => !still.discarded).length;
    badge.textContent = count ? String(count) : "";
  }
  // What the tab says it counts. This badge used to carry the project's
  // unopened-take count, which sat beside the word "Shots" reading as a number
  // of shots and was wrong by a factor of three. Newness has its own chip
  // below, in its own colour, saying the word "new".
  const shots = $("bin-source-count");
  if (shots) {
    const count = sources().length;
    shots.textContent = count ? String(count) : "";
    shots.title = `${count} source(s) in the bin`;
  }
  renderBinNewFilter();
}

/**
 * The bin's "N new" chip: how much has landed, and where.
 *
 * A count on its own says a batch queued half an hour ago has started coming
 * back. In a bin of twenty-nine rows it does not say *which* rows, and finding
 * out meant scrolling the whole column looking for a badge. Pressing this
 * narrows the bin to the rows that got something.
 *
 * Drawn only while something is unopened — with nothing new the filter has
 * nothing to narrow to, so leaving it on would empty the bin. Switching it off
 * here is what makes clearing the last badge give the whole bin back.
 */
function renderBinNewFilter() {
  const host = $("bin-new-filter");
  if (!host) return;
  const count = newTakeCount();
  host.textContent = "";
  if (!count) {
    state.binNewOnly = false;
    return;
  }

  // Shown on the Images tab too — it is the one number that says a batch has
  // started landing while you are looking at something else — where pressing
  // it goes back to the Shots tab already narrowed.
  const onImages = state.binView === "images";
  const chip = document.createElement("button");
  chip.className = "new-chip";
  chip.setAttribute("aria-pressed", String(!onImages && state.binNewOnly));
  chip.title = onImages
    ? `Back to the ${count} take(s) you have not opened`
    : state.binNewOnly
      ? "Show every source again"
      : `Show only the sources holding the ${count} take(s) you have not opened`;
  chip.innerHTML = `${iconHTML("sparkles", "sm")}<span>${count} new</span>`;
  chip.onclick = () => {
    if (onImages) {
      state.binNewOnly = true;
      return setBinView("shots");
    }
    state.binNewOnly = !state.binNewOnly;
    renderBin();
  };
  host.append(chip);
}

function wireBinViews() {
  $("library-add").onclick = addSource;
  $("bin-search").oninput = () => {
    state.binQuery = $("bin-search").value;
    renderBin();
  };
  $("bin-search").onkeydown = (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      $("bin-search").value = "";
      state.binQuery = "";
      renderBin();
    }
  };
  $("monitor-fullscreen").onclick = guard(async () => {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $("player-stage").requestFullscreen();
  });
  $("bin-shots").onclick = () => setBinView("shots");
  $("bin-images").onclick = () => setBinView("images");
}

function renderModeChip(prompt) {
  const chip = $("mode-chip");
  if (!chip) return;
  const mode = mediaMode(prompt);
  chip.innerHTML = `${iconHTML("film", "sm")}<span>Renders as <b>${MODE_LABEL[mode]}</b></span>`;
}

function renderKeyframes(prompt) {
  const host = $("keyframes");
  if (!host) return;
  host.textContent = "";
  const fromCast = castReferencePaths(prompt).length;
  const disabled = !prompt || effectiveReferenceCount(prompt) > 0;
  const generate = $("kf-generate");
  if (generate) {
    generate.disabled = disabled;
    generate.title = !disabled
      ? "Write a picture with Krea 2 Turbo and use it as the first frame"
      : fromCast
        ? `A cast member on this segment carries ${fromCast} identity file(s), which `
          + "makes this shot reference-driven. Take them off the segment, or clear "
          + "their identity files, to use a keyframe here."
        : "Remove the references first — a segment is image OR reference driven.";
  }
  for (const [key, label] of [["first_frame", "First frame"], ["last_frame", "Last frame"]]) {
    host.append(keyframeSlot(prompt, key, label, disabled));
  }
}

function keyframeSlot(prompt, key, labelText, disabled) {
  const slot = document.createElement("div");
  slot.className = "kf-slot";
  const path = prompt?.[key] || "";

  if (path) {
    // Both keyframes, so the arrows walk first to last without going back out.
    const both = ["first_frame", "last_frame"]
      .map((which) => ({ path: prompt?.[which] || "", kind: "image" }))
      .filter((frame) => frame.path);
    slot.append(mediaThumb(path, "image", both));
    const clear = document.createElement("button");
    clear.className = "iconbtn small kf-clear";
    clear.title = `Clear ${labelText.toLowerCase()}`;
    clear.innerHTML = iconHTML("x", "sm");
    clear.onclick = (event) => {
      event.stopPropagation();
      prompt[key] = "";
      markDirty();
      renderMedia();
      refresh();
    };
    slot.append(clear);
  } else {
    const tag = document.createElement("span");
    tag.className = "kf-label";
    tag.textContent = labelText;
    const hint = document.createElement("span");
    hint.innerHTML = `${iconHTML("plus", "sm")} add image`;
    slot.append(tag, hint);
  }

  if (disabled && !path) {
    slot.style.opacity = ".45";
    slot.style.cursor = "not-allowed";
    slot.title = "Remove the references first — a segment is image OR reference driven.";
    return slot;
  }

  const pick = document.createElement("button");
  pick.className = "iconbtn small kf-pick";
  pick.innerHTML = iconHTML("image", "sm");
  pick.title = `Choose a generated image for the ${labelText.toLowerCase()}, `
    + "or write a new one";
  pick.onclick = (event) => {
    event.stopPropagation();
    openStudio({ kind: "keyframe", key });
  };
  slot.append(pick);

  const load = guard(async (file) => {
    if (!file || !prompt) return;
    const { filename } = await uploadMedia(file);
    prompt[key] = filename;
    markDirty();
    renderMedia();
    refresh();
  });

  slot.onclick = () => { if (!path) pickFile("image/*").then(load); };
  slot.ondragover = (event) => { event.preventDefault(); slot.classList.add("drag"); };
  slot.ondragleave = () => slot.classList.remove("drag");
  slot.ondrop = (event) => {
    event.preventDefault();
    slot.classList.remove("drag");
    const file = event.dataTransfer?.files?.[0];
    if (file) load(file);
  };
  return slot;
}

/** The ref2v task types, as ticks. A closed set with its own lint rule for
 *  mistyping one, which is the clearest sign it was never a text field. */
function renderRefTasks(prompt) {
  const host = $("ref_task");
  if (!host) return;
  host.textContent = "";
  const chosen = prompt?.ref_task ?? [];

  for (const task of REF_TASK_TYPES) {
    const wrap = document.createElement("label");
    wrap.className = "task-tick";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = chosen.includes(task);
    box.disabled = !prompt;
    box.onchange = () => {
      // Kept in the vocabulary's own order, so the composed prefix reads the
      // same whichever order they were ticked in.
      prompt.ref_task = REF_TASK_TYPES.filter(
        (name) => (name === task ? box.checked : (prompt.ref_task ?? []).includes(name)),
      );
      markDirty();
      refresh();
    };
    wrap.append(box, document.createTextNode(task));
    host.append(wrap);
  }

  // Anything an agent wrote that is not in the set still shows, so a value the
  // linter is about to complain about is visible rather than silently dropped.
  for (const task of chosen.filter((name) => !REF_TASK_TYPES.includes(name))) {
    const odd = document.createElement("span");
    odd.className = "task-odd";
    odd.textContent = `${task} — not an H3 task type`;
    host.append(odd);
  }
}

function renderSubjects(prompt) {
  const list = $("subjectlist");
  const adder = $("subject-add");
  if (!list || !adder) return;

  adder.textContent = "";
  const add = document.createElement("button");
  add.className = "small fixed";
  add.innerHTML = label("plus", "Subject", "sm");
  add.disabled = !prompt;
  add.title = "Define something the video contains that has no cast member behind it — "
    + "a style, an action, a one-off prop. Cast members on screen are already subjects.";
  add.onclick = guard(() => {
    if (!prompt) return;
    prompt.subjects = [...(prompt.subjects || []),
      { role: "", retention: "", sources: [], member: "", noun: "", label: "" }];
    markDirty();
    renderMedia();
    refresh();
  });
  adder.append(add);

  // Numbered by the server, so the tag on screen is the tag in the prompt.
  let plainIndex = 0;
  const resolved = (state.subjects || []).map((item) => ({
    ...item,
    plainIndex: item.member ? -1 : plainIndex++,
  }));

  list.textContent = "";
  if (!resolved.length) {
    list.innerHTML = `<div class="empty">No subjects. Put a cast member with identity
      files on screen, or add one here.</div>`;
    return;
  }
  for (const item of resolved) list.append(subjectRow(prompt, item));
}

function subjectRow(prompt, resolved) {
  const row = document.createElement("div");
  row.className = resolved.member ? "subject-row cast" : "subject-row";
  const stored = findSubjectRow(prompt, resolved);

  const head = document.createElement("div");
  head.className = "subject-head";
  const cited = (resolved.sources || [])
    .map((path) => (state.refTags || []).find((tag) => tag.path === path)?.tag)
    .filter(Boolean);
  head.innerHTML =
    `<span class="tag">${escapeHtml(resolved.tag || "")}</span>` +
    (resolved.member
      ? `<span class="ref-link" title="A cast member. The sources and the fallback wording` +
        ` follow them; a role typed here overrides it for this segment only.">` +
        `${iconHTML("users", "sm")}${escapeHtml(resolved.label || "unnamed")}</span>`
      : "") +
    (resolved.noun ? `<span>${escapeHtml(resolved.noun)}</span>` : "") +
    (cited.length
      ? `<span class="subject-from" title="Cited inside this subject's line, so these`
        + ` files write no definition of their own.">from ${escapeHtml(cited.join(" + "))}</span>`
      : `<span class="subject-from">no source files</span>`);
  row.append(head);

  const role = document.createElement("input");
  role.type = "text";
  role.className = "role";
  role.value = stored?.role || "";
  role.placeholder = resolved.member
    ? `role — from the cast: ${resolved.role || resolved.label || "the member"}`
    : "role — what this is, e.g. a woman in her thirties, dark hair, a grey hoodie";
  role.title = "Composed as \u201C<Subject N> is the <noun> in <Picture 1>, <role>.\u201D — "
    + "so write what the thing IS, not what to do with it. Type @ to name a cast member.";
  // The menu without the highlight: a one-line input scrolls its text sideways,
  // which an overlay cannot follow honestly. The composer still resolves an
  // `@name` written here (see mentions.fields).
  wireMentions(role);
  role.onchange = guard(() => {
    const target = stored || subjectRowFor(prompt, resolved);
    if (!target) return;
    target.role = role.value;
    pruneSubjectRow(prompt, target);
    markDirty();
    refresh();
  });
  row.append(role);

  const retention = retentionControl(
    "image", stored?.retention,
    resolved.member
      ? `retention — from the cast: ${resolved.retention || "fully_preserved"}`
      : null,
    guard((marker) => {
      const target = stored || subjectRowFor(prompt, resolved);
      if (!target) return;
      target.retention = marker;
      pruneSubjectRow(prompt, target);
      markDirty();
      refresh();
    }),
  );
  row.append(retention);

  // A cast subject's sources are the member's files — changing them here would
  // mean changing the member, which is the Cast tab's job. Only a hand-written
  // subject picks what it is cut from.
  if (!resolved.member && stored) {
    const noun = document.createElement("input");
    noun.type = "text";
    noun.className = "role";
    noun.value = stored.noun || "";
    noun.placeholder = "noun — person / setting / object / style (optional)";
    noun.title = "The word the citation hangs on: \u201Cis the person in <Picture 1>\u201D. "
      + "Leave it empty and the line reads \u201Cis <role>, shown in <Picture 1>\u201D.";
    noun.onchange = guard(() => {
      stored.noun = noun.value;
      markDirty();
      refresh();
    });
    row.append(noun);

    const picker = document.createElement("div");
    picker.className = "cast-ticks";
    const tag = document.createElement("span");
    tag.className = "seg-tag";
    tag.textContent = "cut from";
    picker.append(tag);
    for (const reference of prompt?.references || []) {
      const file = referenceFile(reference);
      if (!file.path) continue;
      const tick = document.createElement("label");
      tick.className = "cast-here";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = (stored.sources || []).includes(file.path);
      box.onchange = guard(() => {
        stored.sources = box.checked
          ? [...(stored.sources || []), file.path]
          : (stored.sources || []).filter((path) => path !== file.path);
        markDirty();
        renderMedia();
        refresh();
      });
      tick.append(box, document.createTextNode(
        `${tagFor(reference) || file.kind} ${file.path.split("/").pop()}`));
      picker.append(tick);
    }
    row.append(picker);
  }

  // Taking a subject out of this segment is not editing the member, so it does
  // not belong behind the guard above — the noun and the source picker are the
  // Cast tab's job, this is not. Sharing their condition left a cast subject
  // with no way out of the block at all, which is how a member unticked in the
  // Cast tab could go on showing up here.
  const speaks = resolved.member && speakingMemberIds(prompt).includes(resolved.member);
  if (resolved.member || stored) {
    const remove = document.createElement("button");
    remove.className = "iconbtn small";
    remove.disabled = Boolean(speaks);
    remove.title = speaks
      ? "They speak here, so their line is what puts them on screen — remove the line instead."
      : resolved.member
        ? "Take this cast member off screen in this segment. The member itself is untouched."
        : "Remove subject";
    remove.innerHTML = iconHTML("trash", "sm");
    remove.onclick = guard(() => {
      if (resolved.member) takeMemberOffScreen(prompt, resolved.member);
      else prompt.subjects = (prompt.subjects || []).filter((item) => item !== stored);
      markDirty();
      renderCast();
      renderMedia();
      refresh();
    });
    head.append(remove);
  }

  return row;
}

function renderReferences(prompt) {
  const adder = $("ref-add");
  const list = $("reflist");
  if (!adder || !list) return;

  const references = prompt?.references || [];
  const blockedByKeyframe = !prompt
    || (prompt.first_frame || "").trim() || (prompt.last_frame || "").trim();

  const generate = $("ref-generate");
  if (generate) {
    generate.disabled = Boolean(blockedByKeyframe);
    generate.title = blockedByKeyframe
      ? "Remove the keyframe first — a segment is image OR reference driven."
      : "Generate a picture and add it as a reference";
  }

  // Add buttons, one per kind, disabled at the cap or when keyframes are set.
  adder.textContent = "";
  for (const { kind, label: kindLabel, limit } of state.refKinds) {
    const used = references.filter((ref) => ref.kind === kind).length;
    const button = document.createElement("button");
    button.className = "small fixed";
    button.innerHTML = label("plus", kindLabel, "sm");
    button.disabled = !prompt || blockedByKeyframe || used >= limit;
    button.title = blockedByKeyframe
      ? "Remove the keyframe first — a segment is image OR reference driven."
      : used >= limit ? `At most ${limit} ${kindLabel.toLowerCase()} references.`
      : `Add a ${kindLabel.toLowerCase()} reference`;
    button.onclick = guard(async () => {
      const file = await pickFile(KIND_ACCEPT[kind]);
      if (!file || !prompt) return;
      const { filename } = await uploadMedia(file);
      prompt.references = [...(prompt.references || []), { kind, path: filename, role: "" }];
      markDirty();
      renderMedia();
      refresh();
    });
    adder.append(button);
  }

  // A member's identity files already reach the model when they are on screen;
  // what they cannot do from the Cast tab is carry a retention marker chosen
  // for *this* shot. This adds a row per file that says so — a link, not a
  // copy, so the file itself still follows the member.
  const linked = new Set(
    references.filter((item) => item.member).map((item) => `${item.member}#${item.slot || 0}`),
  );
  const linkable = castMembers().filter(
    (member) => (member.references || []).some((_, slot) => !linked.has(`${member.id}#${slot}`)),
  );

  const fromCast = document.createElement("select");
  fromCast.className = "small fixed ref-from-cast";
  fromCast.append(new Option("+ From cast…", ""));
  for (const member of linkable) {
    const count = (member.references || []).filter(
      (_, slot) => !linked.has(`${member.id}#${slot}`),
    ).length;
    fromCast.append(new Option(`${member.name || "unnamed"} (${count})`, member.id));
  }
  fromCast.disabled = !prompt || Boolean(blockedByKeyframe) || !linkable.length;
  fromCast.title = blockedByKeyframe
    ? "Remove the keyframe first — a segment is image OR reference driven."
    : !linkable.length
      ? "Every cast identity file is already a row here, or no member has one yet."
      : "Bring a cast member's identity files in as rows, so you can set the role and "
        + "retention for this segment. The file stays linked to the member.";
  fromCast.onchange = () => {
    const member = memberById(fromCast.value);
    fromCast.value = "";
    if (!member || !prompt) return;
    const rows = (member.references || [])
      .map((reference, slot) => ({ reference, slot }))
      .filter(({ slot }) => !linked.has(`${member.id}#${slot}`))
      .map(({ reference, slot }) => ({
        kind: reference.kind,
        // The last known path, so a preview with no cast to hand still shows
        // something. Resolution ignores it and re-reads the member.
        path: reference.path,
        role: "",
        retention: "",
        member: member.id,
        slot,
      }));
    prompt.references = [...(prompt.references || []), ...rows];
    markDirty();
    renderMedia();
    refresh();
  };
  adder.append(fromCast);

  list.textContent = "";
  if (!references.length) {
    list.innerHTML = `<div class="empty">No references. Add an image, video or audio the
      model should honour.</div>`;
    return;
  }
  references.forEach((ref, index) => list.append(referenceRow(prompt, ref, index)));
}

function referenceRow(prompt, ref, index) {
  const row = document.createElement("div");
  row.className = "ref-row";

  // A linked row owns no file of its own: everything but role and retention is
  // read back off the member, so the row shows what will actually be sent
  // rather than whatever path was current when the link was made.
  const target = linkedTarget(ref);
  const file = referenceFile(ref);
  if (target) row.classList.add("linked");

  const thumb = document.createElement("div");
  thumb.className = "thumb";
  thumb.append(mediaThumb(file.path, file.kind, prompt.references));
  row.append(thumb);

  const body = document.createElement("div");
  body.className = "ref-body";
  const head = document.createElement("div");
  head.className = "ref-head";
  const tag = tagFor(ref);
  const kindLabel = state.refKinds.find((k) => k.kind === file.kind)?.label || file.kind;
  head.innerHTML =
    `<span class="tag">${escapeHtml(tag)}</span>` +
    `<span>${escapeHtml(kindLabel)}</span>` +
    (target
      ? `<span class="ref-link" title="Linked to a cast member — the file follows them.` +
        ` Edit it in the Cast tab.">${iconHTML("users", "sm")}` +
        `${escapeHtml(target.member.name || "unnamed")}</span>`
      : "") +
    `<span class="name" title="${escapeHtml(file.path)}">${escapeHtml(file.path.split("/").pop())}</span>`;

  const inherited = target ? target.reference : null;

  // A file a subject is cut from has no definition line of its own, so its role
  // has nowhere to compose to. Say so on the row rather than leaving a field
  // that looks live and silently does nothing.
  const isSound = file.kind === "audio" || file.kind === "video_audio";
  const owner = subjectCiting(file.path);
  if (owner && !isSound) {
    head.insertAdjacentHTML("beforeend",
      `<span class="ref-in-subject" title="Cited inside ${escapeHtml(owner.tag)}, so this`
      + ` picture writes no subject_definitions line of its own — that subject speaks for it.">`
      + `${iconHTML("link", "sm")}in ${escapeHtml(owner.tag)}</span>`);
  }

  const role = document.createElement("input");
  role.className = "role";
  role.dataset.field = `ref.${index}.role`;
  role.type = "text";
  wireMentions(role);
  role.placeholder = owner && !isSound
    ? `role — not composed; ${owner.tag} speaks for this file`
    : owner
      ? `role — composed as \u201Cis <role> for ${owner.tag}\u201D`
      : target
        ? `role — from the cast: ${inherited.role || target.member.name || "the member"}`
        : "role — e.g. the hero's face (or name its tag in the prompt)";
  role.value = ref.role || "";
  role.oninput = () => { ref.role = role.value; markDirty(); refresh(); };

  const isAudio = file.kind === "audio" || file.kind === "video_audio";
  const fallback = isAudio ? "reference" : "fully_preserved";
  const retention = retentionControl(
    file.kind, ref.retention,
    target ? `retention — from the cast: ${inherited.retention || fallback}` : null,
    (marker) => { ref.retention = marker; markDirty(); refresh(); },
  );
  retention.querySelector("select").dataset.field = `ref.${index}.retention`;

  body.append(head, role, retention);
  row.append(body);

  // Swapping the file is its own gesture rather than remove-then-add, because
  // everything else on the row survives a swap: the role sentence, the
  // retention marker, and the place in the list that decides which tag the
  // prompt calls it by. Removing and re-adding loses all three, and an updated
  // reference -- a redrawn character sheet, a re-shot plate -- is the same
  // reference, not a new one.
  // On a linked row the same button means something else: the file is the
  // member's, so replacing it here would either edit every other segment or
  // silently strand the link. Cutting the link first makes the choice explicit,
  // and leaves a plain row that swaps the ordinary way.
  const swap = document.createElement("button");
  swap.className = "iconbtn small";
  swap.innerHTML = iconHTML(target ? "unlink" : "rotate", "sm");
  swap.title = target
    ? `Cut the link to ${target.member.name || "the member"} and keep the file as this `
      + `segment's own, so it stops following them. The role and retention stay.`
    : `Replace this ${kindLabel.toLowerCase()} with another file. The role, the `
      + `retention and the ${tag || "tag"} it answers to all stay.`;
  swap.onclick = guard(async () => {
    if (target) {
      ref.kind = target.reference.kind;
      ref.path = target.reference.path;
      ref.role = ref.role || target.reference.role || "";
      ref.retention = ref.retention || target.reference.retention || "";
      ref.member = "";
      ref.slot = 0;
      markDirty();
      renderMedia();
      refresh();
      return;
    }
    const file = await pickFile(KIND_ACCEPT[ref.kind]);
    if (!file) return;
    const { filename } = await uploadMedia(file);
    ref.path = filename;
    markDirty();
    renderMedia();
    refresh();
  });

  const remove = document.createElement("button");
  remove.className = "iconbtn small";
  remove.title = "Remove reference";
  remove.innerHTML = iconHTML("trash", "sm");
  remove.onclick = guard(() => {
    prompt.references = (prompt.references || []).filter((_, i) => i !== index);
    markDirty();
    renderMedia();
    refresh();
  });
  row.append(swap, remove);
  return row;
}

// The quality preset drives the megapixel budget and step count for renders of
// this segment's current prompt.
const QUALITY_LABELS = {
  preview: "Quick", turbo_hd: "Turbo HD", turbo_2pass: "Turbo 2-pass",
  selflift_hd: "SelfLift HD", selflift_fhd: "SelfLift FHD", selflift_final: "SelfLift Final",
  final: "Fine", final_2pass: "Fine 2-pass",
  singularity_2pass: "Singularity 2-pass",
};

function renderQuality(segment) {
  const level = segment?.quality ?? "preview";
  const text = segment ? (QUALITY_LABELS[level] ?? "Quick") : "";
  $("quality-hint").textContent = segment ? `${text} · ${tierNumbers(level)}` : "";
  renderGenActions(segment);
}

// A seed typed by hand, or null for "roll a fresh one". Anything that is not a
// whole number in range reads as unpinned rather than as an error: the box is a
// convenience, and a half-typed number must not block the render buttons beside
// it. The ceiling is MAX_SEED in profiles.py — the point past which float64
// stops telling consecutive seeds apart.
function parseSeed(text) {
  const trimmed = String(text ?? "").trim();
  if (!trimmed) return null;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

// Pin a seed and the next render reuses that starting noise instead of rolling
// fresh, which is what lets a prompt be tweaked while the composition holds.
// Empty is the normal case, so this stays visually inert until it is filled.
//
// It is deliberately not part of the author form: a seed is a property of a
// render, not of the prompt being authored, and persisting it per segment would
// mean a stale seed quietly steering a render weeks later.
function seedPinControl() {
  const box = document.createElement("input");
  box.type = "text";
  box.inputMode = "numeric";
  box.id = "seed-pin";
  box.className = "seed-pin";
  box.placeholder = "Seed: random";
  box.title =
    "Reuse a seed for the next render. Empty rolls a fresh one." +
    "\n\n" +
    "The seed fixes the starting noise, so a small prompt tweak usually keeps " +
    "the composition — but it only reproduces at the SAME quality tier. " +
    "Different steps or canvas give a different clip from the same seed.";
  box.value = state.seedPin == null ? "" : String(state.seedPin);
  box.classList.toggle("pinned", state.seedPin != null);
  box.oninput = () => {
    state.seedPin = parseSeed(box.value);
    box.classList.toggle("pinned", state.seedPin != null);
  };
  return box;
}

// How many seeds one press of a render button covers. Mirrors render.MAX_BATCH
// on the server, which refuses anything larger.
const MAX_SWEEP = 8;
const SWEEP_COUNTS = [1, 2, 3, 4, 5, 6, 8];

// A sweep used to be its own button, locked to draft quality, so the only way
// to get candidates was at 0.2 MP — and comparing HD takes meant pressing Turbo
// HD by hand N times, rolling a fresh unrelated seed each time. Making the
// count a modifier instead fixes all of it: every tier sweeps, a held run sweeps
// into the overnight queue, and a continuation sweeps too — which was the one
// thing no button could do, because Extend has no tier button of its own. The
// hold toggle beside it is the same argument, applied a second time.
//
// It sits next to the seed pin because the two answer the same question
// together — the pin is where the seeds start, this is how many follow.
function sweepCountControl() {
  const el = document.createElement("select");
  el.id = "sweep-count";
  el.className = "sweep-count";
  el.title =
    "How many seeds the render button beside this one covers \u2014 Extend " +
    "included, on a continuation." +
    "\n\n" +
    "\u00d71 renders once. More queues that many renders of the same prompt at " +
    "consecutive seeds, so they differ only in their starting noise and are " +
    "worth comparing against each other." +
    "\n\n" +
    "With a seed pinned the sweep starts there and counts up, so the batch is " +
    "reproducible from that one number. Unpinned, the base is random.";
  for (const count of SWEEP_COUNTS) {
    const option = document.createElement("option");
    option.value = String(count);
    option.textContent = count === 1 ? "×1 seed" : `×${count} seeds`;
    el.append(option);
  }
  el.value = String(sweepCount());
  el.classList.toggle("sweeping", sweepCount() > 1);
  el.onchange = () => {
    state.sweepCount = Number(el.value) || 1;
    // The buttons carry the count in their labels, so they have to be redrawn
    // — which replaces this very control, so hand the focus back to its
    // successor rather than dropping it on the body.
    renderGenActions(currentSegment());
    $("sweep-count")?.focus();
  };
  return el;
}

/** The chosen sweep size, clamped to what the server will accept. */
function sweepCount() {
  const wanted = Number(state.sweepCount) || 1;
  return Math.min(Math.max(Math.round(wanted), 1), MAX_SWEEP);
}

// Send the next render to the overnight shelf instead of ComfyUI's live queue.
// The third modifier, beside the seed pin and the sweep count, and the one that
// replaced the old Hold Fine button: hold used to be welded to the Fine tier,
// so a held draft, a held sweep and a held continuation had no way to be asked
// for. Answered once at `dispatchRenders`, so every render path obeys it.
//
// Being a sticky mode, it is forgettable — which is the whole design problem
// here. The mitigation is not this control but what it does to everything else:
// while it is armed, every render button renames itself "Hold ...", so the mode
// is written on the thing being pressed rather than on a light beside it. The
// cure for forgetting anyway is "Shelve it" on a queued job.
function holdToggleControl(id = "hold-toggle") {
  const el = document.createElement("button");
  el.id = id;
  el.type = "button";
  el.className = `fixed hold-toggle${state.holdRenders ? " armed" : ""}`;
  el.setAttribute("aria-pressed", String(state.holdRenders));
  el.innerHTML = label("clock", state.holdRenders ? "Holding" : "Hold", "sm");
  el.title = state.holdRenders
    ? "Armed: the next render is frozen outside ComfyUI's queue to run later " +
      "or overnight — every tier, sweeps and continuations included.\n\n" +
      "Press to send renders straight to the live queue again."
    : "Freeze renders outside ComfyUI's queue so they can run later or " +
      "overnight, instead of rendering now.\n\n" +
      "Stays on until pressed again; while it is on, every render button says " +
      "\"Hold\" so it cannot be pressed by accident.";
  el.onclick = () => {
    state.holdRenders = !state.holdRenders;
    syncHoldControls();
    $(id)?.focus();
    toast(state.holdRenders
      ? "Holding: generations go to the shelf until you press Hold again."
      : "Live: generations go straight to ComfyUI's queue.");
  };
  return el;
}

// Everywhere the armed state is written down. Three places wear it — the render
// bar, the take cards' icon buttons and the image studio — and they are drawn
// by different code, so flipping the mode has to visit all three or one of them
// would keep offering a press that no longer does what it says.
function syncHoldControls() {
  // `renderCandidates` redraws the render bar itself, but it returns early with
  // no source open, so the bar is asked directly afterwards too.
  renderCandidates();
  renderGenActions(currentSegment());
  renderStudioHold();
}

// The studio's own copy. Images are generations like any other, so they honour
// the same toggle — but the studio opens as a sheet over the editor, which
// hides the render bar. A mode you cannot see or change from where you are
// pressing is worse than no mode, so the control comes with it.
function renderStudioHold() {
  const slot = $("studio-hold-slot");
  if (slot) {
    slot.textContent = "";
    slot.append(holdToggleControl("studio-hold-toggle"));
  }
  const go = $("studio-generate");
  if (go) {
    go.innerHTML = label("sparkles", `${holdPrefix()}Generate`, "sm");
    go.title = state.holdRenders
      ? "Freeze these images outside ComfyUI's queue, to run later or overnight"
      : "Render these images now";
  }
}

/** What every render button is called right now — the mode, worn on the press. */
const holdPrefix = () => (state.holdRenders ? "Hold " : "");

// The footer renders the CURRENT prompt: Quick to draft, Fine for the keeper.
// Re-shooting a specific take lives on the take card instead, because that is
// where the frozen prompt and seed are.
//
// At depth 1 or deeper the segment is a continuation, so the first button is an
// extend from whatever take the path stands on at the depth above — not a fresh
// generation, which would throw the chain away.
function renderGenActions(segment) {
  const host = $("gen-actions");
  if (!host) return;
  host.textContent = "";
  // While renders are being held the whole bar changes colour, on top of every
  // button renaming itself. One signal is a light you learn to stop seeing; two
  // that disagree with the resting state are hard to miss on a small screen.
  host.classList.toggle("holding", Boolean(state.holdRenders));
  const ready = segment && state.profiles.length && $("profile-picker").value;
  if (!ready) return;

  const button = (icon, text, title, primary, onclick) => {
    const el = document.createElement("button");
    el.className = primary ? "fixed primary" : "fixed";
    el.title = title;
    el.innerHTML = label(icon, text);
    el.onclick = guard(onclick);
    return el;
  };

  const source = currentSource();
  const parentTake = state.depth > 0 ? takeAtDepth(source?.id, state.depth - 1) : null;

  if (state.depth > 0) {
    if (parentTake) {
      // How much of the generation above carries into this one. The ceiling is
      // THIS segment's length (what the footer's stats report), not the
      // parent's — a short beat after a long opener can pin less.
      const context = contextSelect(parentTake, state.lastPreview?.frames);
      const tier = qualitySelect(extendQualityFor(source, state.depth), parentTake);
      // A continuation is guesswork past the pinned run, so it wants candidates
      // for the same reason a fresh generation does — and the same two controls
      // give them. The pin sets where its own seed starts, not the parent's:
      // what the parent rolled is already baked into the frames being carried.
      const extendCount = sweepCount();
      const extendTimes = extendCount > 1 ? ` ×${extendCount}` : "";
      host.append(
        seedPinControl(),
        sweepCountControl(),
        holdToggleControl(),
        context,
        tier,
        button("rotate", `${holdPrefix()}Extend${extendTimes}`,
          `Continue generation ${state.depth} from ` +
          `${parentTake.seed != null ? `seed #${parentTake.seed}` : parentTake.filename}` +
          (extendCount > 1
            ? ` — ${extendCount} candidates, at consecutive seeds of their own`
            : "") +
          (state.holdRenders ? " — held for later, not rendered now" : ""),
          true,
          async () => {
            readForm();
            const text = composedText() || `Continuation of ${source.name}`;
            await runExtend(parentTake, source, text,
                            { depth: state.depth, context: chosenContext(context),
                              quality: chosenQuality(tier),
                              count: extendCount, seed: state.seedPin });
          }),
      );
      // Nothing else belongs here. The buttons below render a FRESH clip from
      // the prompt, which at depth 1+ would land a take that continues nothing
      // — the chain thrown away by a button sitting next to the one that keeps
      // it. The tier control above is how this generation picks its quality.
      return;
    } else {
      const warn = document.createElement("span");
      warn.className = "stat";
      warn.innerHTML =
        `${iconHTML("alert-triangle", "sm")} Generation ${state.depth} has no chosen take yet — ` +
        `render and choose one first.`;
      host.append(warn);
      return;
    }
  }

  // Every tier button below renders the sweep size, so the count belongs in
  // their labels and tooltips rather than only in the control that sets it.
  const count = sweepCount();
  const times = count > 1 ? ` ×${count}` : "";
  const seeds = count > 1 ? ` × ${count} seeds` : "";
  // Hold used to be a fourth button, and being a button it could only hold the
  // one tier it was welded to. As a modifier it renames these three instead, so
  // a held draft and a held sweep exist and there is no button left that could
  // be pressed while forgetting the mode.
  const held = state.holdRenders
    ? " — held outside ComfyUI's queue, to run later or overnight"
    : "";

  // One tier picker and one Render button, where four tier buttons used to sit.
  // The tier is a value, so it belongs in a control: as buttons it was a
  // property of WHICH ONE was pressed, which meant every tier added cost a
  // button here AND another on every take card — four tiers, eight buttons, and
  // a fifth tier would have cost two more. It also settles an inconsistency the
  // bar had with itself, since the extend branch above has always been a tier
  // dropdown and one button.
  const level = renderTier(segment);
  host.append(
    seedPinControl(),
    sweepCountControl(),
    holdToggleControl(),
    renderTierSelect(segment),
    button(TIER_ICONS[level] ?? "sparkles", `${holdPrefix()}Render${times}`,
      `Render the current prompt at ${QUALITY_LABELS[level]} — ` +
      `${tierNumbers(level)}${seeds}${held}\n\n${TIER_BLURBS[level] ?? ""}`,
      true,
      // Read live off the control rather than off this closure: the tier can
      // change without the bar being redrawn, and a button that renders a stale
      // tier is the one failure this whole change exists to prevent.
      () => generate({ quality: renderTier(currentSegment()) })),
  );
}

/** The tier the render bar is set to — the segment's own, defaulted and validated. */
function renderTier(segment) {
  const level = segment?.quality;
  return QUALITY_LABELS[level] ? level : "preview";
}

// The tier picker. What it edits is the segment's OWN `quality` field, not a
// separate mode of the bar: the four tier buttons wrote that field on press
// anyway, so reading it back means the picker, the budget hint above it and the
// render that follows agree by construction rather than by being kept in step.
function renderTierSelect(segment) {
  const el = document.createElement("select");
  el.id = "render-tier";
  el.className = "render-tier";
  el.title = "How hard to work on this prompt. A prompt is proved cheap and only " +
    "then paid for — going straight to a keeper to \"see it properly\" is how a " +
    "bad prompt gets expensive.\n\n" +
    TIERS.map((t) => `${QUALITY_LABELS[t]} — ${tierNumbers(t)}\n  ${TIER_BLURBS[t]}`)
      .join("\n\n");
  for (const tier of TIERS) {
    const option = document.createElement("option");
    option.value = tier;
    option.textContent = `${QUALITY_LABELS[tier]} · ${tierNumbers(tier)}`;
    el.append(option);
  }
  el.value = renderTier(segment);
  el.onchange = () => {
    const chosen = QUALITY_LABELS[el.value] ? el.value : "preview";
    const target = currentSegment();
    if (!target || target.quality === chosen) return;
    target.quality = chosen;
    markDirty();
    // Redraws the hint, and through it this very control — the Render button
    // beside it wears the tier's icon and numbers, so it has to be rebuilt. Hand
    // the focus to the successor rather than dropping it on the body.
    renderQuality(target);
    refresh();
    $("render-tier")?.focus();
  };
  return el;
}

// -- settings panel -------------------------------------------------------

// Which input ids belong to which tier. The panel used to name every id in
// every function, so a fifth tier meant a fifth copy of four blocks of code;
// this is the same information said once. A two-pass tier owns an extra
// `-split` box, and its `-mp` means the FINISHED size, not pass 1's canvas —
// composing small is the point of those tiers, not a preference.
const SETTINGS_ROWS = [
  { level: "preview",        prefix: "sq" },
  { level: "turbo_hd",       prefix: "st" },
  { level: "turbo_2pass",    prefix: "sy" },
  { level: "selflift_hd",    prefix: "sh" },
  { level: "selflift_fhd",   prefix: "si" },
  { level: "selflift_final", prefix: "sk" },
  { level: "final",          prefix: "sf" },
  { level: "final_2pass",    prefix: "sx" },
  { level: "singularity_2pass", prefix: "ss" },
];

/** Fill the settings inputs from the current project. */
function fillSettings() {
  renderVideoLoraRows();
  loadVideoLoraFiles();
  for (const { level, prefix } of SETTINGS_ROWS) {
    const preset = resolveQuality(level);
    const refine = resolveRefine(level);
    $(`${prefix}-mp`).value = refine ? refine.megapixels : preset.megapixels;
    $(`${prefix}-steps`).value = preset.steps;
    $(`${prefix}-ref`).value = preset.reference_size;
    if (refine) $(`${prefix}-split`).value = refine.pass1_steps;
  }
}

/** Read the inputs back into `state.project.settings` and push to the server. */
async function pushSettings() {
  if (!state.project) return;
  const presets = {};

  // Only overrides that differ from the defaults are stored. The numbers are
  // written as a unit — one of them moving means the tier is no longer on the
  // default set — while the reference size is its own decision and rides along
  // whether or not the numbers were touched.
  for (const { level, prefix } of SETTINGS_ROWS) {
    const defaults = QUALITY_DEFAULTS[level];
    const refineDefaults = REFINE_DEFAULTS[level];
    const over = {};

    const mp = parseFloat($(`${prefix}-mp`).value);
    const steps = parseInt($(`${prefix}-steps`).value, 10);
    if (refineDefaults) {
      const split = parseInt($(`${prefix}-split`).value, 10);
      // `refine_megapixels` is the finished size; plain `megapixels` would be
      // pass 1's canvas, which this row deliberately does not offer.
      if (mp > 0 && steps > 1 && split > 0 &&
          (mp !== refineDefaults.megapixels || steps !== defaults.steps ||
           split !== refineDefaults.pass1_steps))
        Object.assign(over, {
          steps, pass1_steps: split, refine_megapixels: mp,
        });
    } else if (mp > 0 && steps > 0 &&
               (mp !== defaults.megapixels || steps !== defaults.steps)) {
      Object.assign(over, { megapixels: mp, steps });
    }

    const size = $(`${prefix}-ref`).value;
    if (REFERENCE_SIZES.includes(size) && size !== defaults.reference_size)
      over.reference_size = size;

    if (Object.keys(over).length) presets[level] = over;
  }

  state.project.settings = state.project.settings || {};
  if (Object.keys(presets).length)
    state.project.settings.quality_presets = presets;
  else
    delete state.project.settings.quality_presets;

  await post(`${API}/project/update`, {
    name: state.project.name,
    project: state.project,
  });
  // Refresh any visible quality labels with the new numbers.
  renderQuality(currentSegment());
  toast("Settings saved");
}

/** Every input id the panel owns, in row order. */
function settingsInputIds() {
  const ids = [];
  for (const { level, prefix } of SETTINGS_ROWS) {
    ids.push(`${prefix}-mp`, `${prefix}-steps`, `${prefix}-ref`);
    if (REFINE_DEFAULTS[level]) ids.push(`${prefix}-split`);
  }
  return ids;
}

function wireSettings() {
  wireShotLoras();
  const btn = $("settings-btn");
  const panel = $("settings-panel");
  $("video-lora-refresh").onclick = loadVideoLoraFiles;
  $("video-lora-add-model").onchange = () => {
    $("video-lora-add").disabled = !$("video-lora-add-model").value || !state.project;
  };
  $("video-lora-add").onclick = guard(async () => {
    if (!state.project) return;
    const name = $("video-lora-add-model").value;
    if (!name) return;
    const rows = [...(state.project.settings?.video_loras || [])];
    if (rows.some(row => row.name === name)) throw new Error("This LoRA is already in the stack.");
    if (rows.length >= 8) throw new Error("Use at most 8 LoRAs.");
    rows.push({ name, strength_model: name.includes("realism-people") ? 0.6 : 1.0, enabled: true });
    await saveVideoLoras(rows);
  });

  btn.onclick = () => {
    const opening = !panel.classList.contains("open");
    panel.classList.toggle("open");
    if (opening) fillSettings();
  };
  // Close on outside click.
  document.addEventListener("mousedown", (event) => {
    if (panel.classList.contains("open") &&
        !panel.contains(event.target) && event.target !== btn) {
      panel.classList.remove("open");
    }
  });

  // Save on change of any input.
  for (const id of settingsInputIds()) $(id).onchange = guard(pushSettings);

  $("settings-defaults").onclick = guard(async () => {
    for (const { level, prefix } of SETTINGS_ROWS) {
      const refine = REFINE_DEFAULTS[level];
      $(`${prefix}-mp`).value = refine ? refine.megapixels : QUALITY_DEFAULTS[level].megapixels;
      $(`${prefix}-steps`).value = QUALITY_DEFAULTS[level].steps;
      $(`${prefix}-ref`).value = QUALITY_DEFAULTS[level].reference_size;
      if (refine) $(`${prefix}-split`).value = refine.pass1_steps;
    }
    await pushSettings();
  });
}

// -- preview --------------------------------------------------------------

let videoLoraSave = Promise.resolve();

function videoLoraPickerNames(result) {
  if (result.model_family === "minimax-h3") return result.loras || [];
  // A still-running older backend lists the entire shared folder. Until its
  // next restart, expose only the two adapters whose H3 metadata we verified.
  const knownH3 = new Set(["h3-realism-people-t2v-i2v-r2v.safetensors", "minimax_h3_cinematic_realism_v01.safetensors"]);
  return (result.loras || []).filter(name => knownH3.has(name.replaceAll("\\", "/").split("/").pop().toLowerCase()));
}

async function loadVideoLoraFiles(prefix = "video-lora") {
  if (typeof prefix !== "string") prefix = "video-lora";
  const picker = $(`${prefix}-add-model`);
  try {
    const result = await call(`${API}/video-loras`);
    const names = videoLoraPickerNames(result);
    const selected = picker.value;
    picker.replaceChildren(new Option("Choose an installed LoRA…", ""));
    for (const name of names) picker.append(new Option(name, name));
    if ([...picker.options].some(o => o.value === selected)) picker.value = selected;
    $(`${prefix}-add`).disabled = !picker.value || !state.project;
    $(`${prefix}-status`).textContent = result.available
      ? `${names.length} MiniMax H3 LoRAs. ${result.hidden_count ?? ((result.loras || []).length - names.length)} unrelated, unverified or Turbo files hidden.` +
        (result.model_family === "minimax-h3" ? " Compatibility is identified from model metadata." : " Restart ComfyUI to detect other H3 adapters by metadata.")
      : "LoRA files are available when connected to ComfyUI.";
  } catch (error) {
    $(`${prefix}-status`).textContent = `Cannot load LoRA files: ${error.message}. If you just updated the tool, restart ComfyUI.`;
  }
}

async function saveVideoLoras(rows) {
  const project = state.project;
  if (!project) return;
  project.settings ||= {};
  project.settings.video_loras = rows;
  renderVideoLoraRows();
  renderShotLoras();
  const snapshot = JSON.parse(JSON.stringify(rows));
  videoLoraSave = videoLoraSave.catch(() => {}).then(() => post(`${API}/video-loras`, {
    project: project.name, loras: snapshot,
  }));
  await videoLoraSave;
  toast("LoRA settings saved");
}

function renderVideoLoraRows(prefix = "video-lora", rows = state.project?.settings?.video_loras || [], saveRows = saveVideoLoras) {
  const host = $(`${prefix}-rows`);
  host.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "note"; empty.textContent = "No additional LoRAs."; host.append(empty);
  }
  rows.forEach((row, index) => {
    const box = document.createElement("div"); box.className = "video-lora-row";
    const name = document.createElement("span"); name.className = "video-lora-name";
    name.textContent = row.name; box.append(name);
    const controls = document.createElement("div"); controls.className = "video-lora-controls";
    const enabledLabel = document.createElement("label");
    const enabled = document.createElement("input"); enabled.type = "checkbox"; enabled.checked = row.enabled !== false;
    enabled.setAttribute("aria-label", `Enable ${row.name}`);
    enabledLabel.append(enabled, "Enabled");
    const weightLabel = document.createElement("label"); weightLabel.append("Weight");
    const weight = document.createElement("input"); weight.type = "number"; weight.min = "-2"; weight.max = "2"; weight.step = "0.05";
    weight.value = row.strength_model ?? 1; weight.setAttribute("aria-label", `Weight for ${row.name}`);
    weightLabel.append(weight);
    const update = guard(async () => {
      if (!weight.reportValidity() || weight.value === "") return;
      const copy = rows.map(r => ({ ...r }));
      copy[index] = { ...row, enabled: enabled.checked, strength_model: Number(weight.value) };
      await saveRows(copy);
    });
    enabled.onchange = update; weight.onchange = update;
    controls.append(enabledLabel, weightLabel);
    for (const [text, offset] of [["↑", -1], ["↓", 1]]) {
      const move = document.createElement("button"); move.className = "small ghost"; move.textContent = text;
      move.title = `Move ${row.name} ${offset < 0 ? "up" : "down"}`;
      move.disabled = index + offset < 0 || index + offset >= rows.length;
      move.onclick = guard(async () => { const copy = [...rows]; [copy[index], copy[index + offset]] = [copy[index + offset], copy[index]]; await saveRows(copy); });
      controls.append(move);
    }
    const remove = document.createElement("button"); remove.className = "small ghost"; remove.textContent = "Remove";
    remove.onclick = guard(() => saveRows(rows.filter((_, i) => i !== index)));
    controls.append(remove); box.append(controls); host.append(box);
  });
}

function renderShotLoras() {
  const source = currentSource();
  const setting = source?.video_loras || { mode: "inherit", loras: [] };
  $("shot-lora-mode").value = setting.mode;
  $("shot-lora-mode").disabled = !source;
  $("shot-lora-custom").hidden = setting.mode !== "custom";
  const defaults = state.project?.settings?.video_loras || [];
  const describe = rows => rows.filter(r => r.enabled !== false && r.strength_model !== 0)
    .map(r => `${r.name.split(/[\\/]/).pop()} × ${r.strength_model ?? 1}`).join(", ") || "No additional LoRAs";
  $("shot-lora-summary").textContent = setting.mode === "custom" ? `· Custom (${setting.loras.length})`
    : setting.mode === "off" ? "· Off" : "· Project defaults";
  $("shot-lora-note").textContent = setting.mode === "custom"
    ? "This stack replaces project defaults, including on extends. Changes save automatically."
    : setting.mode === "off" ? "Additional LoRAs are off for this shot, including extends. The quality tier still controls Turbo."
    : `Project: ${describe(defaults)}. Extends retain the parent take’s recorded stack; choose Custom or Off to override it.`;
  renderVideoLoraRows("shot-lora", setting.loras, rows => saveShotLoras({ mode: "custom", loras: rows }));
}

async function saveShotLoras(setting) {
  const source = currentSource();
  if (!source) return;
  readForm();
  source.video_loras = structuredClone(setting);
  markDirty();
  renderShotLoras();
  // pushSource captures this source now; render preparation waits for its save.
  videoLoraSave = Promise.all([videoLoraSave.catch(() => {}), pushSource()]);
  await videoLoraSave;
  toast("Shot LoRAs saved");
}

function wireShotLoras() {
  $("shot-lora-open").onclick = () => {
    const panel = $("shot-lora-panel");
    panel.open = true;
    panel.scrollIntoView({ block: "start", behavior: "instant" });
    $("shot-lora-mode").focus({ preventScroll: true });
  };
  $("shot-lora-panel").ontoggle = () => {
    if ($("shot-lora-panel").open) loadVideoLoraFiles("shot-lora");
  };
  $("shot-lora-mode").onchange = guard(() => saveShotLoras({
    mode: $("shot-lora-mode").value, loras: currentSource()?.video_loras?.loras || [],
  }));
  $("shot-lora-refresh").onclick = () => loadVideoLoraFiles("shot-lora");
  $("shot-lora-add-model").onchange = () => {
    $("shot-lora-add").disabled = !$("shot-lora-add-model").value || !currentSource();
  };
  $("shot-lora-add").onclick = guard(async () => {
    const name = $("shot-lora-add-model").value;
    if (!name || !currentSource()) return;
    const rows = [...(currentSource().video_loras?.loras || [])];
    if (rows.some(row => row.name === name)) throw new Error("This LoRA is already in the stack.");
    if (rows.length >= 8) throw new Error("Use at most 8 LoRAs.");
    rows.push({ name, strength_model: name.includes("realism-people") ? 0.6 : 1, enabled: true });
    await saveShotLoras({ mode: "custom", loras: rows });
  });
}

let previewTimer = null;

function refresh() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(runPreview, 250);
}

async function runPreview() {
  readForm();
  const prompt = currentPrompt();
  if (!prompt) {
    renderComposed("", []);
    $("findings").textContent = "";
    $("stats").textContent = "";
    updateFindingsCount([]);
    return;
  }

  try {
    const quality = currentSegment()?.quality;
    showPreview(await post(`${API}/preview`, { prompt, quality, project: state.project?.name }));
  } catch (error) {
    toast(error.message, true);
  }
}

function showPreview(result) {
  syncCastPresence();
  // Stash the frame-aligned values so generate() can use them when the user
  // edits the composed text (the reshoot path needs explicit frames/seconds).
  const lengthChanged = state.lastPreview?.frames !== result.frames;
  state.lastPreview = result;
  // The extend control's ceiling is this clip's length, so a changed duration
  // has to rebuild it. Only on a real change — a preview runs on every edit,
  // and rebuilding the footer under every keystroke is needless churn.
  if (lengthChanged) renderGenActions(currentSegment());
  // Don't overwrite the user's manual edit; only update when clean.
  if (!state.previewDirty) {
    renderComposed(result.prompt, result.spans);
  }
  const displayLen = state.previewDirty ? state.previewEditedText.length : result.prompt.length;
  const canvas = state.workflowOwnsCanvas
    ? `<span title="the render profile does not bind the canvas">(${result.width}×${result.height})</span>`
    : `${result.width}×${result.height}`;
  $("stats").innerHTML =
    `Source <b>${result.frames}</b>f · <b>${result.seconds}s</b> · ` +
    `${canvas} · ${displayLen} chars`;
  showFindings(result.findings);

  // The server owns the reference tags (the <Audio j> numbering spans video
  // soundtracks and standalone audio together). Update them in place rather
  // than rebuilding the rows, so a role field being typed into keeps focus.
  state.refTags = result.references ?? [];
  const tagSpans = document.querySelectorAll("#reflist .ref-row .ref-head .tag");
  (currentPrompt()?.references || []).forEach((ref, index) => {
    if (tagSpans[index]) tagSpans[index].textContent = tagFor(ref);
  });

  // Subjects come back resolved too. Their rows are rebuilt only when the shape
  // of the list changes — a member arriving, a file being cited — because a
  // rebuild under every keystroke would take the cursor out of the role being
  // typed, and typing a role never changes the shape.
  const shape = (list) => JSON.stringify(
    (list || []).map((item) => [item.tag, item.member, item.label, item.noun, item.sources]));
  const changed = shape(state.subjects) !== shape(result.subjects);
  state.subjects = result.subjects ?? [];
  if (changed) renderMedia();
  renderModeChip(currentPrompt());
}

// -- tinting the composed prompt by where each part came from --------------
//
// The server hands back the offsets (see provenance.py); this wraps each one in
// a span so the writer can see that this sentence is the style contract and
// that one is Shot 2's action, and can click through to the field that wrote
// it. `textContent` is still exactly the composed text, so everything
// downstream — char count, render, copy — is unaffected.

// Legend order: picture first, then sound, then the reference document.
const SPAN_KINDS = [
  ["style", "Style"],
  ["overview", "Overview"],
  ["angle", "Angle"],
  ["action", "Action"],
  ["camera", "Camera"],
  ["transition", "Transition"],
  ["dialogue", "Dialogue"],
  ["screen", "On-screen text"],
  ["sound", "Soundscape"],
  ["music", "Music"],
  ["reference", "Reference"],
];

function renderComposed(text, spans) {
  const pane = $("preview");
  pane.textContent = "";

  let cursor = 0;
  for (const span of spans ?? []) {
    // The server sorts and de-overlaps these; the guard is only so a stale or
    // unexpected span can never scramble the text itself.
    if (span.start < cursor || span.end > text.length) continue;
    if (span.start > cursor) pane.append(text.slice(cursor, span.start));
    const mark = document.createElement("span");
    mark.dataset.field = span.field;
    mark.dataset.kind = span.kind;
    mark.title = span.label;
    mark.textContent = text.slice(span.start, span.end);
    pane.append(mark);
    cursor = span.end;
  }
  pane.append(text.slice(cursor));

  // No spans is a legitimate answer — an empty prompt, or a compose the marker
  // pass could not vouch for — so the tint simply has nothing to offer and the
  // pane goes back to being plain editable text.
  state.hasSpans = cursor > 0;
  renderLegend(spans ?? []);
  applyMutedKinds();
  applyPromptMode();
}

function renderLegend(spans) {
  const bar = $("prompt-legend");
  if (!bar) return;
  bar.textContent = "";
  const present = new Set(spans.map((span) => span.kind));
  for (const [kind, name] of SPAN_KINDS) {
    if (!present.has(kind)) continue;
    const button = document.createElement("button");
    button.dataset.kind = kind;
    button.textContent = name;
    button.title = `Mute or unmute ${name.toLowerCase()}`;
    button.onclick = () => {
      if (state.mutedKinds.has(kind)) state.mutedKinds.delete(kind);
      else state.mutedKinds.add(kind);
      applyMutedKinds();
    };
    bar.append(button);
  }
}

function bindCastControls() {
  const add = $("cast-add");
  if (add) add.onclick = guard(addMember);
}

// Hovering a tinted run lights that kind's legend chip. The chip is already
// on screen and already the right colour, so nothing in the toolbar changes
// size: printing the field's name here instead re-wrapped the toolbar and
// shoved the text down under the cursor mid-selection. The name itself is the
// mark's `title`, which the browser shows at the pointer anyway.
function litLegendKind(kind) {
  for (const button of document.querySelectorAll("#prompt-legend button"))
    button.classList.toggle("lit", button.dataset.kind === kind);
}

function applyMutedKinds() {
  for (const button of document.querySelectorAll("#prompt-legend button"))
    button.classList.toggle("off", state.mutedKinds.has(button.dataset.kind));
  for (const mark of document.querySelectorAll("#preview [data-field]"))
    mark.classList.toggle("muted", state.mutedKinds.has(mark.dataset.kind));
}

/** Tinted is a read-only map of the composed text; plain is yours to edit. */
function setPromptTint(on) {
  state.promptTint = on;
  applyPromptMode();
}

// The preference only takes effect when there is something to show and the text
// is still compose()'s — a hand edit has no provenance, and neither does a pass
// that produced no spans.
function applyPromptMode() {
  const on = state.promptTint && state.hasSpans && !state.previewDirty;
  const pane = $("preview");
  pane.classList.toggle("plain", !on);
  // Editing a tinted pane would leave marks pointing at text that moved, so
  // the two modes are exclusive — and the toggle is how you get to editing.
  pane.contentEditable = on ? "false" : "plaintext-only";
  $("prompt-legend").hidden = !on;
  litLegendKind(null);
  $("prompt-tint-note").hidden = !on;

  const button = $("prompt-tint-toggle");
  button.hidden = !state.hasSpans;
  button.disabled = state.previewDirty;
  button.textContent = on ? "Edit as text" : "Show colours";
  button.title = on
    ? "Turn the tint off and edit the composed text by hand"
    : "Colour the text by which field wrote each part";
}

// Prompt section parsing, markedText, sectionCard, promptSectionView,
// diffMark, diffRun, sectionedDiff, and diffWords live in ./prompt-sections.js.

/** One side of the side-by-side view: its own sections when they can be read
    and the preference is on, the flat word diff when they cannot. */
function diffPane(title, text, chunks, keep) {
  const pane = document.createElement("div");
  pane.className = "diff-pane";
  const head = document.createElement("div");
  head.className = "diff-pane-title";
  head.textContent = title;

  const box = document.createElement("div");
  box.className = "diff-box";
  const sectioned = state.promptSectionsOn ? sectionedDiff(text, chunks, keep) : null;
  if (sectioned) {
    box.classList.add("sectioned");
    box.append(sectioned);
  } else {
    box.append(diffRun(chunks, keep));
  }

  pane.append(head, box);
  return pane;
}

/** Fill `host` with `text`, sectioned when the preference and the text allow. */
function showRecordedPrompt(host, text, empty) {
  host.textContent = "";
  if (!(text || "").trim()) {
    host.textContent = empty;
    return;
  }
  const view = state.promptSectionsOn ? promptSectionView(text) : null;
  if (view) host.append(view);
  else host.textContent = text;
}

/** Point a toggle at `text`, hiding it when that prompt has no structure. */
function setPromptSectionToggle(button, text, redraw) {
  if (!button) return;
  button.hidden = !promptSections(text);
  button.textContent = state.promptSectionsOn ? "Plain text" : "Show sections";
  button.title = state.promptSectionsOn
    ? "Show the recorded prompt as the one flat block it was sent as"
    : "Break the recorded prompt into labelled, separately copyable sections";
  button.onclick = () => {
    state.promptSectionsOn = !state.promptSectionsOn;
    redraw();
  };
}

/** Jump from a run of composed text to the field that wrote it. */
function focusField(address) {
  const target = findField(address);
  if (!target) return;
  const panel = target.closest(".tabpanel");
  if (panel) $("author-tabs").querySelector(`button[data-tab="${panel.dataset.panel}"]`)?.click();
  const shot = target.closest(".scene-beat");
  if (shot) setSceneShotOpen(Number(shot.dataset.beat), true);
  for (let parent = target.parentElement; parent && parent !== panel; parent = parent.parentElement) {
    if (parent.tagName === "DETAILS") parent.open = true;
    if (parent.classList?.contains("card-field")) parent.hidden = false;
    if (parent.classList?.contains("beat-settings")) {
      parent.hidden = false;
      shot?.querySelector(".beat-settings-toggle")?.setAttribute("aria-expanded", "true");
    }
  }
  refitGrowFields();
  target.scrollIntoView({ block: "center", behavior: "smooth" });
  target.focus({ preventScroll: true });
  target.classList.remove("flash");
  void target.offsetWidth; // restart the animation rather than skip it
  target.classList.add("flash");
}

/** `screen_text.0` is one line of the `screen_text` box, so an address that
    nothing renders exactly falls back to its parent.

    The search skips the Prompt panel on purpose: the tinted marks carry the
    same `data-field` addresses as the fields they stand for, and a mark is
    never what a jump is looking for. */
function findField(address) {
  const parts = address.split(".");
  while (parts.length) {
    const found = document.querySelector(
      `.author .tabpanel:not([data-panel="prompt"]) [data-field="${CSS.escape(parts.join("."))}"]`,
    );
    if (found) return found;
    parts.pop();
  }
  return null;
}

// The composed prompt the user wants to render. When they manually edit the
// preview pane, the edited text takes priority over compose(). Otherwise the
// text comes from the latest preview (server-composed).
function composedText() {
  if (state.previewDirty) return state.previewEditedText;
  return ($("preview")?.textContent || "").trim();
}

// Forget any manual edit of the composed text. The tint can come back once the
// text is compose()'s again — while it is the user's, the marks would point at
// words that have moved.
function clearPreviewEdit() {
  state.previewDirty = false;
  state.previewEditedText = "";
  $("preview").classList.remove("dirty");
  $("preview-dirty-hint").hidden = true;
  applyPromptMode();
}

// Clear the manual edit and re-compose from the structured fields.
function resetPreviewEdit() {
  clearPreviewEdit();
  setPromptTint(true);
  runPreview();
}
// Exposed for the inline onclick in the HTML hint.
window.resetPreviewEdit = resetPreviewEdit;

// The frame grid, the canvas rules and contiguity all live on the server.
// Asking it to apply them is one round trip and keeps exactly one copy.
async function transform(...ops) {
  readForm();
  const segment = currentSegment();
  if (!segment) return;
  try {
    const result = await post(`${API}/transform`, { prompt: segment.prompt, ops });
    segment.prompt = result.prompt;
    fillForm();
    showPreview(result.preview);
  } catch (error) {
    toast(error.message, true);
  }
}

function showFindings(findings) {
  const container = $("findings");
  container.textContent = "";

  // Prompt-level findings, then the ones that only exist because this source
  // sits in an edit — a broken chain, a trim past the end of the lineage.
  const source = currentSource();
  const sequence = source ? findingsFor(source) : [];
  const all = [...findings, ...sequence];
  updateFindingsCount(all);

  if (!all.length) {
    container.innerHTML = `<div class="finding info">` +
      `<span class="mark">${iconHTML("check")}</span>` +
      `<span>Nothing to flag.</span></div>`;
    return;
  }
  const marks = { error: "alert-triangle", warning: "alert-circle", info: "info" };
  for (const finding of all) {
    const row = document.createElement("div");
    row.className = `finding ${finding.severity}`;
    row.innerHTML =
      `<span class="mark">${iconHTML(marks[finding.severity])}</span>` +
      `<span>${escapeHtml(finding.message)} ` +
      `<span class="rule">${escapeHtml(finding.rule)}` +
      `${finding.where ? " · " + escapeHtml(finding.where) : ""}</span></span>`;
    if (finding.where && findField(finding.where)) {
      const jump = document.createElement("button");
      jump.className = "small ghost"; jump.textContent = "Go to field";
      jump.onclick = () => focusField(finding.where);
      row.append(jump);
    }
    container.append(row);
  }
}

// The Findings tab wears its own count so a blocking error is visible even
// while you are reading the Prompt or Scene tab next to it.
function updateFindingsCount(all) {
  const badge = $("findings-count");
  if (!badge) return;
  const errors = all.filter((finding) => finding.severity === "error").length;
  badge.textContent = all.length ? String(all.length) : "";
  badge.classList.toggle("bad", errors > 0);
  if (typeof editorUX !== "undefined") editorUX?.findings(all);
}

const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// -- the bin (what was generated) -----------------------------------------
//
// One row per source: one lineage, with the segments that planned it and the
// takes that came out. This is the panel the shot list became. It is *not* the
// edit — that is the timeline below, and a source can appear there more than
// once or not at all, which the row says outright.

function renderBin() {
  const list = $("bin-list");
  // Deferred to after this redraw has settled, so it is asked on the way in
  // rather than chased down each of the ways out.
  queueThumbSweep();
  renderBinCount();
  if (state.binView === "images") return renderBinImages();
  list.textContent = "";
  const all = sources();

  if (!all.length) {
    list.innerHTML = `<div class="library-empty">${iconHTML("film")}<h3>Your story starts here</h3><p>Add a source, describe your scene, then choose your favorite take.</p></div>`;
  }

  const query = (state.binQuery || "").trim().toLowerCase();
  let visibleCount = 0;
  all.forEach((source, index) => {
    if (query && !displayName(source.name).toLowerCase().includes(query)) return;
    // The filter hides rows; it does not renumber them. A row's number is its
    // place in the bin, and shifting it under a toggle would make the same
    // source read as a different one depending on what is switched on.
    if (state.binNewOnly && !newTakesFor(source.id)) return;
    visibleCount++;
    const report = sourceReport(source.id);
    const uses = clipsOf(source.id);
    const onTimeline = uses.length > 0;
    const hidden = onTimeline && uses.every((clip) => clip.enabled === false);
    const path = report?.path ?? source.path ?? [];

    const row = document.createElement("div");
    row.className = "bin-row";
    row.dataset.id = source.id;
    row.tabIndex = 0;
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", `${displayName(source.name)}. Press Enter to edit.`);
    row.onkeydown = (event) => {
      if (event.target !== row) return;
      if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        openEditor(source.id);
      }
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        event.stopPropagation();
        const rows = [...list.querySelectorAll(".bin-row")];
        const next = rows.indexOf(row) + (event.key === "ArrowDown" ? 1 : -1);
        rows[next]?.focus();
      }
    };
    row.draggable = true;
    if (source.id === state.selectedSource) row.classList.add("selected");
    if (hidden) row.classList.add("disabled");
    if (!path.length) row.classList.add("unrendered");
    if (!onTimeline) row.classList.add("unused");
    // Every row, not only the chained ones: a title wider than the column is
    // cut with an ellipsis, and this is where the rest of it went.
    row.title = source.segments.length > 1
      ? `${displayName(source.name)} — ${source.segments.length} generations`
      : displayName(source.name);
    if (source.segments.length > 1) row.classList.add("chained");

    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = index + 1;

    // Which rows came back with something. Queue a batch across ten sources
    // and this is the only thing in the bin that says which of them landed
    // while you were reading a different one.
    const fresh = newTakesFor(source.id);
    if (fresh) row.classList.add("has-new");

    const first = takeById(path[0]);
    const thumbwrap = document.createElement("div");
    thumbwrap.className = "thumbwrap";
    const thumb = first
      ? heldThumb(`bin:${source.id}`, `${first.url}#t=0.1`, "thumb")
      : document.createElement("div");
    thumb.className = "thumb";
    thumbwrap.append(thumb);
    if (fresh) {
      const dot = document.createElement("span");
      dot.className = "new-dot";
      dot.title = `${fresh} new take(s) you have not opened`;
      thumbwrap.append(dot);
    }

    const who = document.createElement("div");
    who.className = "who";
    const nm = document.createElement("div");
    nm.className = "nm";
    if (source.segments.length > 1) {
      nm.innerHTML =
        `${escapeHtml(displayName(source.name))} ` +
        `<span class="chain-badge" title="${source.segments.length} generations">` +
        `${iconHTML("link", "sm")}${source.segments.length}</span>`;
    } else {
      nm.textContent = displayName(source.name);
    }
    if (!source.name) nm.classList.add("untitled");

    const sub = document.createElement("div");
    sub.className = "sub";
    const count = report?.take_count ?? 0;
    // First in the line, not last. Hovering a row floats its action buttons
    // over the line's right-hand end — so the one stat you are scanning for
    // goes where nothing lands on it.
    sub.innerHTML =
      (fresh
        ? `<span class="new-pill" title="${fresh} new take(s) here that you have not opened">` +
          `${iconHTML("sparkles", "sm")}${fresh}</span>`
        : "") +
      `<span>${iconHTML("clock")}${seconds(report?.seconds)}</span>` +
      `<span>${iconHTML("layers")}${count || "0"}</span>` +
      (uses.length > 1 ? `<span title="used ${uses.length} times">${iconHTML("copy", "sm")}×${uses.length}</span>` : "") +
      (!onTimeline ? `<span class="hidden-pill spare" title="In the bin, not on the timeline">${iconHTML("eye-off", "sm")}</span>` : "") +
      (hidden ? `<span class="hidden-pill" title="On the timeline, but hidden">${iconHTML("eye-off", "sm")}</span>` : "");
    who.append(nm, sub);

    // Colour the status spine: a chosen chain is settled (teal), takes with no
    // choice is in progress (gold), a blocking finding overrides both (red).
    const errors = findingsFor(source).filter((f) => f.severity === "error").length;
    if (report?.chosen) row.classList.add("picked");
    else if (count) row.classList.add("has-takes");
    if (errors || report?.problems?.length) row.classList.add("has-error");

    const flags = document.createElement("div");
    flags.className = "flags";
    if (errors || report?.problems?.length) {
      const bad = document.createElement("span");
      bad.className = "err";
      bad.title = report?.problems?.length
        ? report.problems.join("\n")
        : `${errors} blocking issue(s)`;
      bad.innerHTML = iconHTML("alert-triangle", "sm");
      flags.append(bad);
    }

    const actions = document.createElement("div");
    actions.className = "bin-actions";

    if (onTimeline) {
      const toggleBtn = document.createElement("button");
      toggleBtn.className = "bin-action-btn toggle-visibility" + (hidden ? " active-off" : "");
      toggleBtn.title = hidden ? "Show on the timeline" : "Hide from the timeline";
      toggleBtn.innerHTML = iconHTML(hidden ? "eye-off" : "eye", "sm");
      toggleBtn.onclick = guard(async (e) => {
        e.stopPropagation();
        await toggleClipsOf(source.id, hidden);
      });
      actions.append(toggleBtn);
    } else {
      const useBtn = document.createElement("button");
      useBtn.className = "bin-action-btn";
      useBtn.title = `Put ${displayName(source.name)} on the timeline`;
      useBtn.innerHTML = iconHTML("plus", "sm");
      useBtn.onclick = guard(async (e) => {
        e.stopPropagation();
        await addClipFor(source.id);
      });
      actions.append(useBtn);
    }

    const dupBtn = document.createElement("button");
    dupBtn.className = "bin-action-btn duplicate-source";
    dupBtn.title = `Duplicate ${displayName(source.name)} — same prompt, no takes`;
    dupBtn.innerHTML = iconHTML("copy", "sm");
    dupBtn.onclick = guard(async (e) => {
      e.stopPropagation();
      await duplicateSourceById(source.id);
    });

    const delBtn = document.createElement("button");
    delBtn.className = "bin-action-btn delete-source danger";
    delBtn.title = `Delete ${displayName(source.name)}`;
    delBtn.innerHTML = iconHTML("trash", "sm");
    delBtn.onclick = guard(async (e) => {
      e.stopPropagation();
      await deleteSourceById(source.id);
    });

    actions.append(dupBtn, delBtn);

    row.append(idx, thumbwrap, who, flags, actions);
    row.onclick = () => selectSource(source.id);
    row.ondblclick = () => openEditor(source.id);
    wireSourceDrag(row);
    list.append(row);
  });

  if (all.length && !visibleCount) {
    const empty = document.createElement("div");
    empty.className = "library-empty";
    empty.innerHTML = `${iconHTML("search")}<h3>No matching sources</h3><p>Try a different name or clear your filters.</p>`;
    list.append(empty);
  }

  if (state.binNewOnly) {
    const hint = document.createElement("div");
    hint.className = "list-hint";
    hint.textContent = "showing only sources with new takes";
    list.append(hint);
    return;
  }

  const add = document.createElement("button");
  add.className = "add-source-tile";
  add.innerHTML = label("plus", "Add source", "sm");
  add.onclick = addSource;
  list.append(add);

  const hint = document.createElement("div");
  hint.className = "list-hint";
  hint.textContent = "Double-click to edit · drag to timeline";
  list.append(hint);
}

// Findings the server filed against a source or one of its clips. The server
// writes `where` as "<n>. <name>", so matching is by suffix — and a blank name
// is dropped first, because `endsWith("")` is true of every finding there is
// and would hand an unnamed source the whole project's errors.
function findingsFor(source) {
  const names = [source.name, ...clipsOf(source.id).map((clip) => clip.label)].filter(Boolean);
  if (!names.length) return [];
  return (state.report?.findings ?? []).filter(
    (finding) => finding.where && names.some((name) => finding.where.endsWith(name)),
  );
}

/** What to call a source that was never named. */
const displayName = (name) => (name || "").trim() || "Untitled";

// A bin row drags onto the timeline, and that is all it does. The bin lists
// what was generated and the timeline says how it is used, so the two orders
// are unrelated and a row cannot stand in for a clip: dragging one used twice
// would have to pick a use, and there is no honest way to pick. Reordering the
// cut belongs to the track, where every use is visible and separately grabbable.
function wireSourceDrag(row) {
  row.ondragstart = (event) => {
    event.dataTransfer.setData(DRAG_SOURCE, row.dataset.id);
    event.dataTransfer.setData("text/plain", row.dataset.id);
    event.dataTransfer.effectAllowed = "copy";
    row.classList.add("dragging");
  };
  row.ondragend = () => {
    row.classList.remove("dragging");
    clearDropCaret();
  };
}

const addSource = guard(async () => {
  if (!state.project) return;
  await pushSource();
  const result = await apply(post(`${API}/source/add`, { project: state.project.name }));
  state.binQuery = "";
  state.binNewOnly = false;
  $("bin-search").value = "";
  setBinView("shots");
  const added = result.project.sources.at(-1);
  if (added) selectSource(added.id);
});

/** Put another use of a source on the end of the timeline. */
async function addClipFor(sourceId) {
  await placeSourceAt(sourceId, clips().length);
}

async function duplicateSourceById(sourceId) {
  if (!state.project) return;
  const source = sourceById(sourceId || state.selectedSource);
  if (!source) return;
  await pushSource();
  const result = await apply(post(`${API}/source/duplicate`, {
    project: state.project.name, source: source.id,
  }));
  const all = result.project?.sources ?? sources();
  const copy = all[all.findIndex((s) => s.id === source.id) + 1] ?? null;
  if (copy) selectSource(copy.id);
}

async function deleteSourceById(sourceId) {
  if (!state.project) return;
  const source = sourceById(sourceId || state.selectedSource);
  if (!source) return;
  const uses = clipsOf(source.id).length;
  const also = uses ? ` and ${uses} clip(s) using it` : "";
  if (!window.confirm(`Delete ${displayName(source.name)}${also}? Its takes stay on disk.`)) return;

  const all = sources();
  const index = all.findIndex((s) => s.id === source.id);
  const next = all[index + 1] || all[index - 1] || null;

  await apply(post(`${API}/source/delete`, {
    project: state.project.name, source: source.id,
  }));

  if (next && sourceById(next.id)) selectSource(next.id);
  else if (!sources().length) {
    state.selectedSource = null;
    state.selectedClip = null;
    if (state.editorOpen) closeEditor();
    fillForm();
    renderBin();
    renderTimeline();
    refresh();
  }
}

/** Take a clip out of the edit. The source stays in the bin. */
async function deleteClipById(clipId) {
  if (!state.project) return;
  const clip = clipById(clipId || state.selectedClip);
  if (!clip) return;
  const source = sourceById(clip.source_id);
  const name = clip.label || (source ? displayName(source.name) : "this clip");
  if (!window.confirm(`Remove ${name} from the timeline? It stays in the bin.`)) return;

  const order = clips();
  const index = order.findIndex((c) => c.id === clip.id);
  const next = order[index + 1] || order[index - 1] || null;

  await apply(post(`${API}/clip/delete`, {
    project: state.project.name, clip: clip.id,
  }));

  if (next && clipById(next.id)) select(next.id);
  else if (!clips().length) {
    state.selectedClip = null;
    fillForm();
    renderTimeline();
    refresh();
  }
}

/** Show or hide every clip a source backs, so the bin row toggles as one. */
async function toggleClipsOf(sourceId, show) {
  if (!state.project) return;
  await pushSource();
  // One write per clip, then a single reload: reloading between them would
  // reflow the timeline once per clip and flicker through wrong intermediates.
  for (const clip of clipsOf(sourceId)) {
    const result = await post(`${API}/clip/edit`, {
      project: state.project.name, clip: clip.id, enabled: Boolean(show),
    });
    if (result.project) state.project = result.project;
  }
  restoreSelection();
  await reloadSequence();
  fillForm();
  refresh();
  updateClipButtons();
}

async function toggleClipVisibility(clipId) {
  if (!state.project) return;
  const clip = clipById(clipId || state.selectedClip);
  if (!clip) return;
  await pushSource();
  await apply(post(`${API}/clip/edit`, {
    project: state.project.name, clip: clip.id, enabled: clip.enabled === false,
  }));
  updateClipButtons();
}

/** Silence the selected clip, or give it its voice back.

 * Muting lives on the transport bar as well as in the inspector because it is
 * something you reach for *while cutting*, next to hide and delete — not a
 * property you open a sheet to set. Where the sound comes from is the sheet's
 * business; whether you hear it at all is the timeline's.
 */
async function toggleClipMute(clipId) {
  if (!state.project) return;
  const clip = clipById(clipId || state.selectedClip);
  if (!clip) return;
  await pushSource();
  await apply(post(`${API}/clip/edit`, {
    project: state.project.name, clip: clip.id,
    audio: { muted: !(clipReport(clip.id)?.audio?.muted ?? clip.audio?.muted ?? false) },
  }));
  updateClipButtons();
}

/** Add one more generation to the open source's plan. */
const addSegment = guard(async () => {
  const source = currentSource();
  if (!source || !state.project) return;
  await pushSource();
  const result = await apply(post(`${API}/source/segment/add`, {
    project: state.project.name, source: source.id,
  }));
  const grown = result.project?.sources?.find((s) => s.id === source.id);
  if (grown) selectDepth(grown.segments.length - 1);
});

function updateClipButtons() {
  const clip = currentClip();
  const source = currentSource();
  const hidden = clip?.enabled === false;

  for (const id of ["toggle-clip", "editor-toggle-clip"]) {
    const button = $(id);
    if (!button) continue;
    button.innerHTML = iconHTML(hidden ? "eye-off" : "eye");
    button.title = hidden ? "Show on the timeline" : "Hide from the timeline";
    button.classList.toggle("active-off", hidden);
    button.disabled = !clip;
  }

  const heard = clip ? clipReport(clip.id)?.audio ?? clip.audio ?? null : null;
  const mute = $("mute-clip");
  if (mute) {
    mute.innerHTML = iconHTML(heard?.muted ? "volume-x" : "volume");
    mute.title = heard?.muted
      ? "Unmute this clip"
      : (heard?.borrowed
          ? "Mute this clip — it is currently heard from another take"
          : "Mute this clip (M)");
    mute.classList.toggle("active-off", Boolean(heard?.muted));
    mute.classList.toggle("sounding", Boolean(heard?.borrowed) && !heard?.muted);
    mute.disabled = !clip;
  }

  for (const id of ["split-clip", "delete-clip", "mute-range"]) {
    const button = $(id);
    if (button) button.disabled = !clip;
  }
  for (const id of ["duplicate-source", "editor-delete-source"]) {
    const button = $(id);
    if (button) button.disabled = !source;
  }
}

// -- the timeline ---------------------------------------------------------

// Fit the whole sequence to the dock at zoom 1; zooming in scrolls. The scale
// is clamped so a huge project never collapses to nothing and a tiny one never
// stretches absurdly wide.
const MIN_PPS = 0.5;
const MAX_PPS = 400;

function fitPixelsPerSecond(total) {
  const avail = Math.max(60, $("tl-scroll").clientWidth - 8);
  return total > 0 ? avail / total : 40;
}

function pixelsPerSecond() {
  const total = state.report?.duration ?? 0;
  const fit = fitPixelsPerSecond(total);
  return Math.min(MAX_PPS, Math.max(MIN_PPS, fit * state.zoom));
}

function renderTimeline() {
  const track = $("track");
  const all = clips();
  queueThumbSweep();
  track.textContent = "";
  // Wired before the empty cases return, so an empty timeline is still
  // somewhere a source can be dropped.
  wireTrackDrop(track);

  const blank = (message) => {
    track.innerHTML = `<div class="tl-empty">${message}</div>`;
    track.style.width = "";
    $("ruler").textContent = "";
    $("ruler").style.width = "";
    $("playhead").style.display = "none";
    updateSequenceStat();
  };

  if (!all.length) return blank("Nothing on the timeline — drag a source in from the bin.");

  const active = all.filter((clip) => clip.enabled !== false);
  if (!active.length) return blank("Every clip is hidden — unhide one from the bin, or add a source.");

  const total = state.report?.duration ?? 0;
  const pps = pixelsPerSecond();
  const contentWidth = Math.max(total * pps, 60);
  track.style.width = `${contentWidth + 40}px`;
  $("ruler").style.width = `${contentWidth + 40}px`;

  for (const clip of active) {
    const entry = clipReport(clip.id);
    const source = sourceById(clip.source_id);
    const name = clip.label || (source ? displayName(source.name) : "(missing source)");
    const start = entry?.starts_at ?? 0;
    const length = entry?.timeline_seconds ?? 0;
    const path = entry?.path ?? [];

    // What the clip plays and at what render quality. A chain is worth no
    // more than its weakest link — the same reading the bin's chain head
    // takes — so that is the tier the block reports when it names one.
    const pieces = clipPieces(clip, entry);
    const worstTier = weakestTier(pieces.map((piece) => piece.take).filter(Boolean));
    const mixedTiers =
      new Set(pieces.map((piece) => tierOf(piece.take)).filter(Boolean)).size > 1;

    const block = document.createElement("div");
    block.className = "clip";
    block.dataset.id = clip.id;
    block.dataset.tier = worstTier ?? "unknown";
    block.draggable = true;
    if (clip.id === state.selectedClip) block.classList.add("selected");
    if (!entry?.rendered) block.classList.add("unrendered");
    if (entry?.pinned) block.classList.add("pinned");

    // Status rail, matching the bin: chosen / has-takes / has-error.
    const errors = source
      ? findingsFor(source).filter((f) => f.severity === "error").length
      : 1;
    const report = sourceReport(clip.source_id);
    if (report?.chosen) block.classList.add("picked");
    else if (report?.take_count) block.classList.add("has-takes");
    if (errors) block.classList.add("has-error");

    // Same mark as the bin row, so a batch landing is visible from the
    // timeline without reading the bin. A corner dot rather than a pill: a
    // clip can be six pixels wide and has no room for a count.
    const fresh = newTakesFor(clip.source_id);
    if (fresh) {
      block.classList.add("has-new");
      // A real element rather than a pseudo: `::before` is the status rail and
      // `::after` is the pin marker, and both are already spoken for.
      const dot = document.createElement("span");
      dot.className = "new-dot";
      dot.title = `${fresh} new take(s) you have not opened`;
      block.append(dot);
    }

    block.style.left = `${start * pps}px`;
    block.style.width = `${Math.max(6, length * pps)}px`;
    block.title =
      `${name} — ${seconds(length)}` +
      (fresh ? `
${fresh} new take(s) you have not opened` : "") +
      (path.length > 1 ? `\n${path.length} generations` : "") +
      (entry?.pinned ? "\npinned to this branch" : "") +
      (pieces.length
        ? "\n" + (worstTier
            ? (mixedTiers
                ? `Mixed render qualities — the weakest piece is ${TIER_LABELS[worstTier]}`
                : `${TIER_LABELS[worstTier]} render`)
            : "Render quality unknown")
        : "") +
      `\nDouble-click to edit · drag to move · drag edges to trim`;

    const title = document.createElement("div");
    title.className = "name";
    title.textContent = name;

    const first = takeById(path[0]);
    const poster = first
      ? heldThumb(`clip:${clip.id}`, `${first.url}#t=${(clip.source_in || 0) + 0.1}`, "poster")
      : document.createElement("div");
    poster.className = "poster";

    const len = document.createElement("div");
    len.className = "len";
    len.innerHTML = (pieces.length ? tierDotHTML(worstTier) : "") +
      `${iconHTML("clock")}${seconds(length)}`;

    const badge = document.createElement("span");
    badge.className = "badge";
    // A chain says how many generations it plays; a plain clip says how many
    // takes its source has to choose between.
    badge.innerHTML = path.length > 1
      ? `${iconHTML("link")}${path.length}`
      : `${iconHTML("layers")}${report?.take_count ?? 0}`;

    // A clip whose sound is not its own picture's says so on the block. There
    // is nothing else on screen that could tell you — that is the cost of sound
    // being part of the clip rather than a track under it.
    const heard = entry?.audio;
    if (heard?.muted || heard?.borrowed) {
      const mark = document.createElement("span");
      mark.className = `sound-mark${heard.muted ? " muted" : ""}`;
      mark.innerHTML = iconHTML(heard.muted ? "volume-x" : "volume", "sm");
      const from = heard.take ? takeById(heard.take) : null;
      mark.title = heard.muted
        ? "Muted"
        : `Sound from ${from ? takeLabel(from) : "another take"}`;
      block.append(mark);
    }

    // The joins inside this clip, drawn where the generations meet. A clip that
    // runs across one is a normal thing here, so it should look like one.
    for (const seam of seamsIn(clip, entry)) {
      const mark = document.createElement("span");
      mark.className = "seam";
      mark.style.left = `${seam * pps}px`;
      mark.title = "Generation join";
      block.append(mark);
    }

    // Render quality along the bottom edge, in the hues the take list uses.
    // Drawn piece by piece rather than as one bar because a chained clip can
    // be part Final and part Preview, and the weak stretch is the one worth
    // finding — which is the whole reason the colour is here at all.
    if (pieces.length) {
      const rail = document.createElement("div");
      rail.className = "tier-rail";
      for (const piece of pieces) {
        const seg = document.createElement("span");
        seg.className = "tier-seg";
        seg.dataset.tier = tierOf(piece.take) ?? "unknown";
        seg.style.left = `${piece.start * pps}px`;
        seg.style.width = `${Math.max(1, piece.length * pps)}px`;
        seg.title = piece.take ? tierTitle(piece.take) : "Render quality unknown";
        rail.append(seg);
      }
      block.append(rail);
    }

    const left = document.createElement("div");
    left.className = "handle left";
    left.title = "Trim in — Alt-drag to rate-stretch";
    const right = document.createElement("div");
    right.className = "handle right";
    right.title = "Trim out — Alt-drag to rate-stretch";

    block.append(title, poster, len, badge, left, right);

    // The silences, over everything else on the block. A mute is a decision
    // about this shot rather than a property of it, and it has to be grabbable.
    for (const mute of drawableMutes(clip.id)) {
      block.append(muteBand(mute, clip, entry, pps));
    }

    block.tabIndex = 0;
    block.setAttribute("role", "button");
    block.setAttribute("aria-label", `${name}, ${seconds(length)}`);
    block.onclick = () => { if (!suppressClick && clip.id !== state.selectedClip) select(clip.id); };
    block.ondblclick = () => openEditor(clip.source_id, clip.id);
    block.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        openEditor(clip.source_id, clip.id);
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const allClips = [...track.querySelectorAll(".clip")];
        const idx = allClips.indexOf(block);
        const nextBlock = allClips[idx + (event.key === "ArrowRight" ? 1 : -1)];
        if (nextBlock) {
          select(nextBlock.dataset.id);
          nextBlock.focus();
        }
      }
    });
    // Shift-drag draws a new band. Before `wireTrim`, so an edge grab inside an
    // existing band still resizes it rather than starting a second one.
    block.addEventListener("mousedown", (event) => {
      if (!event.shiftKey || event.button !== 0) return;
      if (event.target.closest(".mute-band") || event.target.closest(".handle")) return;
      event.preventDefault();
      event.stopPropagation();
      startMuteDraw(event, block, clip);
    });
    wireTrim(block, clip);
    wireClipDrag(block, clip);
    track.append(block);
  }

  // The trailing tile adds a source at the end of the sequence.
  const add = document.createElement("button");
  add.className = "add-clip-tile";
  add.style.left = `${total * pps + 4}px`;
  add.innerHTML = iconHTML("plus");
  add.title = "Add a source";
  add.onclick = addSource;
  track.append(add);

  renderRuler(total, pps);
  positionPlayhead();
  updateSequenceStat();
}

/** Where the generation joins fall inside one clip, in its own timeline time. */
function seamsIn(clip, entry) {
  const pieces = (state.report?.playlist ?? []).filter((piece) => piece.clip === clip.id);
  if (pieces.length < 2 || !entry) return [];
  const base = entry.starts_at ?? 0;
  return pieces.slice(1).map((piece) => piece.start - base);
}

/** What one clip plays, piece by piece, with the take behind each piece.
 *
 * A clip is a range of lineage time, so it can cross generations that were
 * rendered at different qualities. The playlist already says where each piece
 * starts and how long it runs — the same numbers `seamsIn` reads — so a tier
 * drawn from this lines up with the joins instead of guessing at them.
 */
function clipPieces(clip, entry) {
  const base = entry?.starts_at ?? 0;
  return (state.report?.playlist ?? [])
    .filter((piece) => piece.clip === clip.id)
    .map((piece) => ({
      take: takeById(piece.take_id),
      start: piece.start - base,
      length: piece.length ?? 0,
    }));
}

function updateSequenceStat() {
  const all = clips();
  const active = all.filter((clip) => clip.enabled !== false);
  const bad = (state.report?.findings ?? []).filter((f) => f.severity === "error").length;
  const countStr = active.length === all.length
    ? `<b>${all.length}</b>`
    : `<b>${active.length}</b>/${all.length}`;
  const spare = sources().length - new Set(all.map((c) => c.source_id)).size;
  const stat = $("sequence-stat");
  stat.innerHTML = all.length || sources().length
    ? `${countStr} clip${all.length === 1 ? "" : "s"} · <b>${seconds(state.report?.duration)}</b>` +
      (spare > 0 ? ` · ${spare} unused` : "") +
      (bad ? ` · <span style="color:var(--error);display:inline-flex;align-items:center;gap:2px">` +
             `${iconHTML("alert-triangle", "sm")}${bad}</span>` : "")
    : "—";
  // The line is one column wide and clips from the right. The tooltip is where
  // the tail goes when a long one does not fit.
  stat.title = stat.textContent.replace(/\s+/g, " ").trim();
}

function renderRuler(total, pps) {
  const ruler = $("ruler");
  ruler.textContent = "";
  if (!total) return;

  const target = 70 / pps;
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  const step = steps.find((s) => s >= target) ?? 3600;

  for (let at = 0; at <= total + 1e-6; at += step) {
    const tick = document.createElement("span");
    tick.className = "tick";
    tick.style.left = `${at * pps}px`;
    ruler.append(tick);

    const mark = document.createElement("span");
    mark.style.left = `${at * pps}px`;
    mark.textContent = step >= 60 ? fmtClock(at) : `${Math.round(at)}s`;
    ruler.append(mark);
  }
}

// -- playhead + seeking ---------------------------------------------------

function positionPlayhead() {
  const head = $("playhead");
  const total = state.report?.duration ?? 0;
  if (!total) { head.style.display = "none"; return; }
  const pps = pixelsPerSecond();
  head.style.display = "block";
  head.style.left = `${Math.min(state.playheadSec, total) * pps}px`;
  $("tl-time").innerHTML = `<b>${fmtClock(state.playheadSec)}</b> / ${fmtClock(total)}`;
}

// Which playable piece sits under a timeline position, and how far into its
// own file that lands. A hidden clip, or one with nothing chosen, leaves a gap
// the monitor cannot fill, so those return null.
function locate(sec) {
  const list = state.player?.list ?? [];
  for (let i = 0; i < list.length; i++) {
    const entry = list[i];
    if (sec >= entry.start - 1e-6 && sec < entry.start + entry.length - 1e-6) {
      const into = (sec - entry.start) * entry.speed;
      return { index: i, source: entry.from + into };
    }
  }
  return null;
}

// The monitor is double-buffered: two <video> elements, one visible and one
// idle. `active` (0 or 1) indexes the visible one.
function playerVideos() {
  return [$("player-video"), $("player-video-b")];
}

function showBuffer(active) {
  const videos = playerVideos();
  videos[active].classList.add("active");
  videos[1 - active].classList.remove("active");
}

// Point a buffer at an entry and park it on the right frame, reusing the source
// when it is already loaded so consecutive pieces off one take never reload.
function primeVideo(video, entry, source) {
  if (!video || !entry) return;
  if (video.dataset.url !== entry.url) {
    video.src = entry.url;
    video.dataset.url = entry.url;
    video.load();
  }
  video.playbackRate = entry.speed || 1;
  const from = source ?? entry.from ?? 0;
  const apply = () => (video.currentTime = from);
  if (video.readyState >= 1) apply();
  else video.onloadedmetadata = apply;
}

// -- borrowed sound -------------------------------------------------------
//
// The monitor plays one file at a time, and normally that file carries the sound
// as well as the picture. A muted clip, or one heard from a take whose picture
// was rejected, breaks that: the buffers go silent and a third element carries
// the sound, nudged back into step whenever it drifts.
//
// Whether to silence the buffers is decided from the *clip*, never from whether
// a borrowed piece happens to be playing. A borrowed track that runs out short
// has to leave silence under the rest of the shot — not let the rejected take's
// own sound wander back in halfway through.

// A little under two frames at 24fps. Correcting more eagerly than this clicks;
// correcting less leaves a slip you can hear against the picture.
const SOUND_DRIFT = 0.08;

const gainToVolume = (db) => Math.min(1, Math.max(0, Math.pow(10, (db || 0) / 20)));

function pauseBorrowedSound() {
  const track = $("player-audio");
  if (track && !track.paused) track.pause();
}

/** The borrowed piece under a timeline position, and how far into its file. */
function soundAt(sec) {
  for (const entry of state.player?.audio ?? []) {
    if (sec >= entry.start - 1e-6 && sec < entry.start + entry.length - 1e-6) {
      return { entry, into: entry.from + (sec - entry.start) * entry.speed };
    }
  }
  return null;
}

/** Point the monitor's speakers at whatever this position is heard as. */
function applySound(sec, playing = false) {
  const track = $("player-audio");
  const player = state.player;
  if (!track || !player) return;

  // A preview auditions one lineage on its own. The edit's sound choices are
  // not part of what is being auditioned, so it plays the file as it is.
  if (player.mode !== "timeline") {
    for (const video of playerVideos()) video.muted = false;
    pauseBorrowedSound();
    return;
  }

  const heard = clipReport(player.list[player.index]?.id)?.audio ?? null;
  // A band is a window on the ruler, so the same check covers a clip heard as
  // itself and one heard from another take: silence the buffers, and stop the
  // borrowed track rather than letting it play under the hole.
  const silent = Boolean(heard?.muted) || mutedAt(sec);
  for (const video of playerVideos()) video.muted = silent || Boolean(heard?.borrowed);

  const hit = silent ? null : soundAt(sec);
  if (!hit) return pauseBorrowedSound();

  if (track.dataset.url !== hit.entry.url) {
    track.src = hit.entry.url;
    track.dataset.url = hit.entry.url;
    track.load();
  }
  track.playbackRate = hit.entry.speed || 1;
  track.volume = gainToVolume(hit.entry.gain_db);

  const chase = () => {
    if (Math.abs(track.currentTime - hit.into) > SOUND_DRIFT) track.currentTime = hit.into;
  };
  if (track.readyState >= 1) chase();
  else track.onloadedmetadata = chase;

  if (playing && track.paused) track.play().catch(() => {});
  if (!playing && !track.paused) track.pause();
}

// Load the clip that follows `index` onto the idle buffer so the next cut is a
// swap, not a reload. The reload happens off-screen, where it can't flash.
function preloadNext(index) {
  const player = state.player;
  const next = player.list[index + 1];
  if (!next) return;
  const idle = playerVideos()[1 - player.active];
  primeVideo(idle, next, null);
  idle.pause();
}

function seekTo(sec) {
  ensureTimeline();
  const total = state.report?.duration ?? 0;
  state.playheadSec = Math.max(0, Math.min(sec, total));
  positionPlayhead();

  const hit = locate(state.playheadSec);
  if (!hit) { pauseBorrowedSound(); return; }
  const player = state.player;
  const entry = player.list[hit.index];
  player.index = hit.index;
  primeVideo(playerVideos()[player.active], entry, hit.source);
  showBuffer(player.active);
  preloadNext(hit.index);
  applySound(state.playheadSec, player.playing);
  $("player-position").textContent = `${hit.index + 1}/${player.list.length} · ${entry.name}`;
}

// Pressing the ruler, an empty stretch of track, or the playhead's grab strip
// scrubs: the playhead follows the pointer until release. A press without a
// drag is just a seek. Clips and their handles keep their own behaviour.
function startScrub(event) {
  if (event.target.closest(".clip") || event.target.closest(".add-clip-tile")) return;
  event.preventDefault();
  stopPlayer();
  const scroll = $("tl-scroll");
  const at = (e) => {
    const x = e.clientX - scroll.getBoundingClientRect().left + scroll.scrollLeft;
    seekTo(snapSeconds(x / pixelsPerSecond(), event.altKey));
  };
  at(event);
  const up = () => {
    document.removeEventListener("mousemove", at);
    document.removeEventListener("mouseup", up);
  };
  document.addEventListener("mousemove", at);
  document.addEventListener("mouseup", up);
}

// Snap to a cut — a clip's start or end, a generation join inside one, or the
// sequence bounds — when the pointer lands within a few pixels. Alt scrubs
// freely past the magnets.
function snapSeconds(sec, free) {
  if (free) return sec;
  const pps = pixelsPerSecond();
  const threshold = 6 / pps;
  const points = [0, state.report?.duration ?? 0];
  for (const entry of state.report?.clips ?? []) {
    if (entry.enabled === false) continue;
    points.push(entry.starts_at, entry.starts_at + entry.timeline_seconds);
  }
  for (const piece of state.report?.playlist ?? []) points.push(piece.start);
  let best = sec;
  let bestDistance = threshold;
  for (const point of points) {
    const distance = Math.abs(point - sec);
    if (distance < bestDistance) { bestDistance = distance; best = point; }
  }
  return best;
}

// -- trim + rate-stretch --------------------------------------------------

let suppressClick = false;

// -- dragging on the track -------------------------------------------------
//
// Two drags land here and both mean "put something at this cut": a clip being
// moved within the edit, and a source arriving from the bin as a new use. The
// kind has to be readable while the pointer is still moving, and `getData` is
// blocked until the drop, so each drag carries its own MIME type and `types`
// answers the question during dragover.

const DRAG_CLIP = "application/x-vp-clip";
const DRAG_SOURCE = "application/x-vp-source";

function wireClipDrag(block, clip) {
  block.ondragstart = (event) => {
    event.dataTransfer.setData(DRAG_CLIP, clip.id);
    event.dataTransfer.setData("text/plain", clip.id);
    event.dataTransfer.effectAllowed = "move";
    block.classList.add("dragging");
  };
  block.ondragend = () => {
    block.classList.remove("dragging");
    clearDropCaret();
  };
}

// One handler for the whole track rather than one per clip: the insert point is
// a position along the track, so the gap past the last clip and the empty
// timeline are the same case as landing on a block, not special ones.
function wireTrackDrop(track) {
  track.ondragover = (event) => {
    const kind = dragKind(event);
    if (!kind) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = kind === DRAG_CLIP ? "move" : "copy";
    showDropCaret(track, dropTargetAt(track, event.clientX).x);
  };
  track.ondragleave = (event) => {
    if (!track.contains(event.relatedTarget)) clearDropCaret();
  };
  track.ondrop = guard(async (event) => {
    const kind = dragKind(event);
    if (!kind) return;
    event.preventDefault();
    clearDropCaret();
    const id = event.dataTransfer.getData(kind);
    if (!id) return;
    const { index } = dropTargetAt(track, event.clientX);
    if (kind === DRAG_SOURCE) await placeSourceAt(id, index);
    else await moveClipTo(id, index);
  });
}

/** Which of the two drags this is. Readable during dragover, unlike the data. */
function dragKind(event) {
  const types = [...(event.dataTransfer?.types ?? [])];
  if (types.includes(DRAG_CLIP)) return DRAG_CLIP;
  if (types.includes(DRAG_SOURCE)) return DRAG_SOURCE;
  return null;
}

// Where a drop lands, as an index into the *whole* clip list and a pixel to
// draw the caret on. The index cannot come from counting blocks: hidden clips
// are not rendered, so a drop after the second block is not a drop at index 2
// whenever one before it is hidden. Each block carries its clip id, so the
// position is looked up in the real list instead.
function dropTargetAt(track, clientX) {
  const x = clientX - track.getBoundingClientRect().left;
  const order = clips();
  const blocks = [...track.querySelectorAll(".clip")];
  if (!blocks.length) return { index: order.length, x: 0 };

  const px = (block, prop) => parseFloat(block.style[prop]) || 0;
  const positionOf = (block) => order.findIndex((clip) => clip.id === block.dataset.id);

  for (const block of blocks) {
    const left = px(block, "left");
    if (x < left + px(block, "width") / 2) {
      const index = positionOf(block);
      return { index: index < 0 ? order.length : index, x: left };
    }
  }
  const last = blocks.at(-1);
  const index = positionOf(last);
  return {
    index: index < 0 ? order.length : index + 1,
    x: px(last, "left") + px(last, "width"),
  };
}

/** Reorder the edit. `to` is an insert point in the list as it stands now. */
async function moveClipTo(clipId, to) {
  if (!state.project) return;
  const from = clips().findIndex((clip) => clip.id === clipId);
  if (from < 0) return;
  // `Timeline.move` pops the clip before inserting it, so an insert point past
  // its own place is one lower once it is out of the list. Both of its own
  // edges then resolve to where it already is, which is not a move.
  const target = from < to ? to - 1 : to;
  if (target === from) return;
  await pushSource();
  await apply(post(`${API}/clip/move`, {
    project: state.project.name, clip: clipId, to: target,
  }));
}

/** Put a use of a source into the edit at a given cut. */
async function placeSourceAt(sourceId, at) {
  if (!state.project) return;
  await pushSource();
  const result = await apply(post(`${API}/clip/add`, {
    project: state.project.name, source: sourceId, at,
  }));
  const added = result.project?.timeline?.clips?.[at];
  if (added) select(added.id);
}

function showDropCaret(track, x) {
  let caret = track.querySelector(".drop-caret");
  if (!caret) {
    caret = document.createElement("div");
    caret.className = "drop-caret";
    track.append(caret);
  }
  caret.style.left = `${x}px`;
}

function clearDropCaret() {
  for (const node of document.querySelectorAll(".drop-caret")) node.remove();
}

// -- mute bands -----------------------------------------------------------
//
// A band is drawn from the ruler clock and written back in the lineage one, and
// this is the only place that conversion happens in the browser: the clip's left
// edge sits at lineage second `source_in`, and one ruler second is `speed`
// lineage seconds. Everything else here is pointer arithmetic.

const muteBase = (clip) => Math.max(0, clip.source_in || 0);

/** Ask for a new band over one clip. Both ends are lineage seconds. */
const addMute = (clip, start, end) => apply(post(`${API}/mute/add`, {
  project: state.project.name,
  clip: clip.id,
  start: Number(start.toFixed(3)),
  end: Number(end.toFixed(3)),
}));

/** One band, drawn inside its clip's block and wired for move and resize. */
function muteBand(mute, clip, entry, pps) {
  const band = document.createElement("div");
  band.className = "mute-band";
  band.dataset.id = mute.id;
  band.style.left = `${mute.at * pps}px`;
  band.style.width = `${Math.max(2, mute.length * pps)}px`;
  band.title =
    `Silent for ${seconds(mute.length)}\n` +
    `Drag to move · drag an edge to resize · double-click to remove`;

  const left = document.createElement("span");
  left.className = "edge left";
  const right = document.createElement("span");
  right.className = "edge right";
  band.append(left, right);

  // The block underneath opens the source on a double-click and selects on a
  // single one. Neither is what a press on the band means.
  band.onclick = (event) => event.stopPropagation();
  band.ondblclick = guard(async (event) => {
    event.stopPropagation();
    await apply(post(`${API}/mute/delete`, {
      project: state.project.name, mute: mute.id,
    }));
  });
  band.onmousedown = (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const edge = event.target.closest(".edge");
    startMuteDrag(
      event, band, clip, entry, mute,
      edge ? (edge.classList.contains("left") ? "left" : "right") : "move",
    );
  };
  return band;
}

function startMuteDrag(event, band, clip, entry, mute, side) {
  const pps = pixelsPerSecond();
  const speed = clip.speed || 1;
  const base = muteBase(clip);
  const full = entry.source_seconds;
  const startX = event.clientX;
  const block = band.parentElement;

  // A press on a band must not become a clip reorder on the way out.
  block.draggable = false;
  band.classList.add("dragging");

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
  const MIN = 0.05;
  let moved = 0;
  let next = { start: mute.start, end: mute.end };

  const onMove = (e) => {
    const dx = e.clientX - startX;
    moved = Math.abs(dx);
    const slid = (dx / pps) * speed;
    if (side === "move") {
      const width = mute.end - mute.start;
      const start = clamp(mute.start + slid, 0, Math.max(0, full - width));
      next = { start, end: start + width };
    } else if (side === "left") {
      next = { start: clamp(mute.start + slid, 0, mute.end - MIN), end: mute.end };
    } else {
      next = { start: mute.start, end: clamp(mute.end + slid, mute.start + MIN, full) };
    }
    // Drawn against the clip's own left edge rather than re-asking the server,
    // so the band tracks the pointer even where it runs past the trim.
    band.style.left = `${((next.start - base) / speed) * pps}px`;
    band.style.width = `${Math.max(2, ((next.end - next.start) / speed) * pps)}px`;
    $("tl-time").textContent = `silent ${seconds((next.end - next.start) / speed)}`;
  };

  const onUp = guard(async () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    band.classList.remove("dragging");
    block.draggable = true;
    // A press that went nowhere leaves the band exactly where it was drawn, so
    // there is nothing to redraw — and redrawing would replace this element
    // between the two clicks of a double-click, which is how you lose the
    // gesture that removes a band.
    if (moved <= 3) return;
    suppressClick = true;
    setTimeout(() => (suppressClick = false), 0);
    await apply(post(`${API}/mute/edit`, {
      project: state.project.name,
      mute: mute.id,
      start: Number(next.start.toFixed(3)),
      end: Number(next.end.toFixed(3)),
    }));
  });

  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

/** Shift-drag across a clip: the band is drawn where the pointer went. */
function startMuteDraw(event, block, clip) {
  const pps = pixelsPerSecond();
  const speed = clip.speed || 1;
  const rect = block.getBoundingClientRect();
  // `clientLeft` is the border: children are placed against the padding box and
  // the rect is the border box, and a pixel out is a pixel of the wrong sound.
  const at = (e) => e.clientX - rect.left - block.clientLeft;
  const fromX = at(event);
  let toX = fromX;

  const ghost = document.createElement("div");
  ghost.className = "mute-band drawing";
  ghost.style.left = `${fromX}px`;
  ghost.style.width = "0px";
  block.append(ghost);
  block.draggable = false;

  const onMove = (e) => {
    toX = at(e);
    ghost.style.left = `${Math.min(fromX, toX)}px`;
    ghost.style.width = `${Math.abs(toX - fromX)}px`;
    $("tl-time").textContent = `silent ${seconds(Math.abs(toX - fromX) / pps)}`;
  };

  const onUp = guard(async () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    ghost.remove();
    block.draggable = true;
    suppressClick = true;
    setTimeout(() => (suppressClick = false), 0);

    const lo = Math.max(0, Math.min(fromX, toX)) / pps;
    const hi = Math.max(0, Math.max(fromX, toX)) / pps;
    // A shift-click is not a zero-length band; it is a miss.
    if (hi - lo < 0.05) { renderTimeline(); return; }
    const base = muteBase(clip);
    await addMute(clip, base + lo * speed, base + hi * speed);
  });

  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

function wireTrim(block, clip) {
  block.querySelectorAll(".handle").forEach((handle) => {
    handle.onmousedown = (event) => {
      event.preventDefault();
      event.stopPropagation();
      // A trim and a move both start with a press on the clip. Taking the block
      // out of the drag model for the duration is what stops an edge grab from
      // turning into a reorder; `startTrim` puts it back on mouse-up.
      block.draggable = false;
      startTrim(event, block, clip, handle.classList.contains("left") ? "left" : "right");
    };
  });
}

function startTrim(event, block, clip, side) {
  const entry = clipReport(clip.id);
  if (!entry) return;
  const pps = pixelsPerSecond();
  const startX = event.clientX;
  const in0 = clip.source_in || 0;
  const out0 = entry.source_out;                 // effective out, in lineage seconds
  const speed0 = clip.speed || 1;
  const fullLen = entry.source_seconds;          // the whole lineage available
  const left0 = parseFloat(block.style.left) || 0;
  let moved = 0;
  let pending = null;

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));

  const onMove = (e) => {
    const dx = e.clientX - startX;
    moved = Math.abs(dx);
    const dTimeline = dx / pps;

    if (side === "right" && e.altKey) {
      // Rate-stretch: keep the source range, change speed to fill the new width.
      block.classList.add("stretching");
      const newLen = Math.max(0.2, (out0 - in0) / speed0 + dTimeline);
      const speed = clamp((out0 - in0) / newLen, 0.1, 10);
      block.style.width = `${((out0 - in0) / speed) * pps}px`;
      pending = { speed };
      $("tl-time").textContent = `speed ${speed.toFixed(2)}×`;
    } else if (side === "right") {
      const newOut = clamp(out0 + dTimeline * speed0, in0 + 0.1, fullLen);
      block.style.width = `${((newOut - in0) / speed0) * pps}px`;
      pending = { source_out: Number(newOut.toFixed(3)) };
      $("tl-time").textContent = `out ${seconds(newOut)}`;
    } else {
      const newIn = clamp(in0 + dTimeline * speed0, 0, out0 - 0.1);
      block.style.left = `${left0 + (newIn - in0) / speed0 * pps}px`;
      block.style.width = `${((out0 - newIn) / speed0) * pps}px`;
      pending = { source_in: Number(newIn.toFixed(3)) };
      $("tl-time").textContent = `in ${seconds(newIn)}`;
    }
  };

  const onUp = guard(async () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    block.classList.remove("stretching");
    block.draggable = true;
    if (moved > 3) {
      suppressClick = true;
      setTimeout(() => (suppressClick = false), 0);
    }
    if (!pending || moved <= 3) { renderTimeline(); return; }
    await apply(post(`${API}/clip/edit`, {
      project: state.project.name,
      clip: clip.id,
      source_in: "source_in" in pending ? pending.source_in : in0,
      source_out: "source_out" in pending ? pending.source_out : out0,
      speed: "speed" in pending ? pending.speed : speed0,
    }));
  });

  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

// -- selection ------------------------------------------------------------
//
// Two things can be selected and they are not interchangeable. Picking a clip
// on the timeline picks its source too, because a clip always has one. Picking
// a source in the bin picks its first clip when it has one, and leaves the
// timeline selection alone when it does not — a bin item nobody has used yet is
// a real thing to be looking at.

function select(clipId) {
  const clip = clipById(clipId);
  if (!clip) return;
  applySelection(clip.source_id, clip.id);
  const entry = clipReport(clip.id);
  if (entry && !state.player?.playing) seekTo(entry.starts_at);
}

function selectSource(sourceId) {
  if (sourceId === state.selectedSource) return;
  const first = clipsOf(sourceId)[0] ?? null;
  applySelection(sourceId, first?.id ?? null);
  const entry = first ? clipReport(first.id) : null;
  if (entry && !state.player?.playing) seekTo(entry.starts_at);
}

function applySelection(sourceId, clipId) {
  if (state.editorOpen || state.inspectorOpen) readForm();
  const changedSource = sourceId !== state.selectedSource;
  state.selectedSource = sourceId;
  state.selectedClip = clipId;
  // A different source starts at its first generation, and composes fresh.
  if (changedSource) state.depth = 0;
  // The ticks name shots in the storyboard that was open, so they mean nothing
  // in another one.
  if (changedSource) {
    clearSplitPick();
    stopSync();
    state.compare.clear();
    state.newOnly = false;
    clearTakeInspector();
    setTakesView("takes");
    ensureTimeline();
  }
  clearPreviewEdit();
  fillForm();
  renderBin();
  renderTimeline();
  if (state.editorOpen || state.reviewOpen) renderCandidates();
  refresh();
}

// -- projects -------------------------------------------------------------

async function loadProjects(select) {
  takeListing(await call(`${API}/projects`));
  const picker = $("project-picker");
  picker.textContent = "";
  for (const name of state.projects) {
    picker.append(new Option(name, name));
  }
  if (!state.projects.length) {
    picker.append(new Option("(no projects)", ""));
    return;
  }
  const last = localStorage.getItem("vp:lastProject");
  const initial = select && state.projects.includes(select)
    ? select
    : last && state.projects.includes(last)
      ? last
      : state.projects[0];
  picker.value = initial;
  await openProject(picker.value);
}

async function openProject(name) {
  if (!name) return;
  const epoch = ++state.projectEpoch;
  localStorage.setItem("vp:lastProject", name);
  const opened = await call(`${API}/project/load?name=${encodeURIComponent(name)}`);
  if (epoch !== state.projectEpoch) return;
  state.project = opened.project;
  if (state.profiles?.length) selectProjectProfile();
  restoreSelection();
  state.playheadSec = 0;
  // The studio's drafts, filters and comparison all name things in the project
  // being left behind, so none of them survive the switch.
  state.studioDrafts = {};
  state.studioCompare.clear();
  state.studioComparing = false;
  state.studioFav = false;
  state.studioQuery = "";
  state.imageHistory = { revisions: [], picked: null };
  stopSync();
  state.compare.clear();
  state.newOnly = false;
  clearTakeInspector();
  setTakesView("takes");
  state.binNewOnly = false;
  state.binQuery = "";
  $("bin-search").value = "";
  clearDirty();
  await reconcile();
  await reloadStills().catch(() => {});
  await reloadSequence();
  fillForm();
  fillSettings();
  refresh();
  // Show the first playable frame in the monitor rather than opening on black.
  const first = clipReport(state.selectedClip);
  if (first) seekTo(first.starts_at);
  // Last, so the load's own work cannot eat the message's few seconds.
  reportFold(opened.folded);
}

// Every project route that changes the shelves hands the whole listing back,
// so one place takes it on and nothing has to re-fetch to stay in step.
function takeListing(listing) {
  state.projects = listing.projects ?? [];
  state.projectEntries = listing.entries ?? [];
  if ($("projects").classList.contains("open")) renderProjects();
}

// -- the project manager --------------------------------------------------
//
// The header picker is for switching between the handful of projects in play.
// This is for the rest of them: the finished ones worth keeping out of the
// way, the ones being thrown out, and the version chains that record how a cut
// got from its first pass to its current one.

const SHELF_LABEL = { active: "Active", archived: "Archived", trashed: "Trash" };

const SHELF_NOTE = {
  active: "Duplicating starts the next version of a project and carries its "
    + "takes and images across. Archiving keeps a project but takes it out of "
    + "the picker.",
  archived: "Archived projects are complete and untouched — they just do not "
    + "clutter the picker. Everything about them still works.",
  trashed: "Nothing here is gone yet. Emptying the trash deletes the project "
    + "file and its backups; rendered videos live in ComfyUI's output folder "
    + "and are never touched.",
};

const EMPTY_NOTE = {
  active: "No active projects. Make one with New.",
  archived: "Nothing archived yet.",
  trashed: "The trash is empty.",
};

function openProjects(shelf) {
  if (shelf) state.projectShelf = shelf;
  $("projects").classList.add("open");
  renderProjects();
  $("proj-search").focus();
}

/** Gather a shelf's projects into version families, oldest version first. */
function familiesOn(shelf, query) {
  const wanted = query.trim().toLowerCase();
  const rows = state.projectEntries
    .filter((entry) => entry.shelf === shelf)
    .filter((entry) => !wanted || entry.name.toLowerCase().includes(wanted));

  const families = new Map();
  for (const entry of rows) {
    // A project written before version families existed has no lineage id. It
    // is a family of itself rather than being lumped in with every other
    // unstamped project, so it is keyed by its own name.
    const key = entry.lineage || `solo:${entry.name}`;
    if (!families.has(key)) families.set(key, []);
    families.get(key).push(entry);
  }
  for (const members of families.values()) {
    members.sort((a, b) => (a.created || "").localeCompare(b.created || "")
      || a.name.localeCompare(b.name));
  }
  // Families sort by their most recent activity, so what you touched last is
  // at the top rather than whatever happens to sort first alphabetically.
  return [...families.values()].sort((a, b) =>
    latestTouch(b).localeCompare(latestTouch(a)));
}

const latestTouch = (members) =>
  members.reduce((newest, one) => (one.modified > newest ? one.modified : newest), "");

function renderProjects() {
  const shelf = state.projectShelf;
  for (const one of Object.keys(SHELF_LABEL)) {
    const count = state.projectEntries.filter((entry) => entry.shelf === one).length;
    $(`proj-n-${one}`).textContent = count ? String(count) : "";
    $("proj-tabs").querySelector(`[data-shelf="${one}"]`)
      .classList.toggle("active", one === shelf);
  }
  $("proj-empty-trash").style.display = shelf === "trashed" ? "" : "none";
  $("proj-empty-trash").disabled =
    !state.projectEntries.some((entry) => entry.shelf === "trashed");
  $("proj-note").textContent = SHELF_NOTE[shelf];

  const body = $("proj-list");
  body.textContent = "";
  const families = familiesOn(shelf, state.projectQuery);
  if (!families.length) {
    const empty = document.createElement("div");
    empty.className = "proj-empty";
    empty.textContent = state.projectQuery
      ? "Nothing here matches that."
      : EMPTY_NOTE[shelf];
    body.append(empty);
    return;
  }
  for (const members of families) body.append(familyBlock(members));
}

function familyBlock(members) {
  const block = document.createElement("div");
  block.className = "proj-family";
  // A single version is just a project. Drawing a one-link chain around it
  // would suggest a progression that has not happened yet.
  if (members.length > 1) {
    block.classList.add("proj-chain");
    const head = document.createElement("div");
    head.className = "proj-family-head";
    head.innerHTML = `${iconHTML("git-branch", "sm")}`
      + `<span>${escapeHtml(familyStem(members))}</span>`
      + `<span>&middot; ${members.length} versions</span>`;
    block.append(head);
  }
  members.forEach((entry, index) =>
    block.append(projectRow(entry, members.length > 1 ? index + 1 : 0)));
  return block;
}

/** The name the family is known by: its oldest version's, without the number. */
const familyStem = (members) => members[0].name.replace(/[ _]v\d+$/i, "");

/** The version chip for a row, or "" when the name already carries one.
 *
 * Most versions are named `... v2` by the duplicate itself, and repeating that
 * in a chip beside the name reads as "v2v2". The chip is for the ones named
 * something else entirely — a hand-named `Director cut` sitting third in its
 * family still wants to say where in the chain it is.
 */
function versionTag(name, position) {
  if (!position || /[ _]v\d+$/i.test(name)) return "";
  return `<span class="proj-vtag">v${position}</span>`;
}

function projectRow(entry, version) {
  const row = document.createElement("div");
  row.className = "proj-row";
  if (entry.name === state.project?.name) row.classList.add("current");

  const main = document.createElement("div");
  main.className = "proj-main";
  const tag = versionTag(entry.name, version);
  const from = entry.derived_from
    ? ` &middot; from ${escapeHtml(entry.derived_from)}`
    : "";
  main.innerHTML = `<div class="proj-name">${escapeHtml(entry.name)}${tag}</div>`
    + `<div class="proj-meta">${entry.sources} source${entry.sources === 1 ? "" : "s"}`
    + ` &middot; made <span class="when">${escapeHtml(whenText(entry.created))}</span>`
    + ` &middot; edited <span class="when">${escapeHtml(whenText(entry.modified))}</span>`
    + `${from}</div>`;

  const acts = document.createElement("div");
  acts.className = "proj-acts";
  for (const action of rowActions(entry)) acts.append(action);

  row.append(main, acts);
  return row;
}

function actionButton(text, icon, title, onclick, cls = "small ghost") {
  const button = document.createElement("button");
  button.className = cls;
  button.title = title;
  button.innerHTML = label(icon, text, "sm");
  button.onclick = guard(onclick);
  return button;
}

function rowActions(entry) {
  const name = entry.name;
  if (entry.shelf === "trashed") {
    return [
      actionButton("Restore", "corner-up-left", "Put it back with the active projects",
        () => shelveProject(name, "active")),
      actionButton("Delete forever", "trash", "Erase the project file and its backups",
        () => purgeProject(entry), "small danger"),
    ];
  }
  const actions = [
    actionButton("Open", "film", "Switch to this project", () => switchTo(name)),
    actionButton("Duplicate", "copy", "Start the next version from this one",
      () => duplicateProject(name)),
  ];
  actions.push(entry.shelf === "archived"
    ? actionButton("Unarchive", "corner-up-left", "Put it back in the picker",
      () => shelveProject(name, "active"))
    : actionButton("Archive", "archive", "Keep it, but take it out of the picker",
      () => shelveProject(name, "archived")));
  actions.push(actionButton("Trash", "trash",
    "Move to the trash — undoable until the trash is emptied",
    () => shelveProject(name, "trashed")));
  return actions;
}

async function switchTo(name) {
  if (name === state.project?.name) {
    $("projects").classList.remove("open");
    return;
  }
  await pushSource();
  $("project-picker").value = name;
  await openProject(name);
  $("projects").classList.remove("open");
}

/** Move a project between shelves, keeping the open one open if it moved. */
async function shelveProject(name, shelf) {
  const isOpen = name === state.project?.name;
  // The form holds edits the server has not seen. Shelving moves the directory
  // underneath it, so anything unsent has to go first or it is written to a
  // path that is no longer there.
  if (isOpen) await pushSource();
  const result = await post(`${API}/project/shelve`, { name, shelf });
  takeListing(result);
  await refreshPicker(result.name);
  toast(shelf === "active"
    ? `${result.name} is active again.`
    : `${result.name} moved to ${SHELF_LABEL[shelf].toLowerCase()}.`);
  renderProjects();
}

async function duplicateProject(name) {
  const result = await post(`${API}/project/duplicate`, { name });
  takeListing(result);
  state.projectShelf = "active";
  await refreshPicker(result.project.name);
  renderProjects();
  toast(`${result.project.name} started from ${name}.`);
}

async function purgeProject(entry) {
  if (!window.confirm(
    `Delete ${entry.name} forever?\n\n`
    + "This erases the project file and its backups — the edit, the prompts and "
    + "the list of takes.\n\n"
    + "The rendered videos themselves are not touched: they live in ComfyUI's "
    + "output folder, and this only ever held pointers to them.")) return;
  takeListing(await post(`${API}/project/purge`, { name: entry.name }));
  await refreshPicker(null);
  renderProjects();
  toast(`${entry.name} deleted.`);
}

async function emptyTrash() {
  const waiting = state.projectEntries.filter((entry) => entry.shelf === "trashed");
  if (!waiting.length) return;
  if (!window.confirm(
    `Delete ${waiting.length} project(s) forever?\n\n`
    + waiting.map((entry) => `  ${entry.name}`).join("\n")
    + "\n\nThis erases their project files and backups. Rendered videos in "
    + "ComfyUI's output folder are not touched.")) return;
  const result = await post(`${API}/project/empty-trash`, {});
  takeListing(result);
  await refreshPicker(null);
  renderProjects();
  toast(`Emptied the trash — ${result.purged.length} project(s) deleted.`);
}

/** Rebuild the header picker from the active names, without reloading a project.
 *
 * `loadProjects` opens whatever it selects, which is right at boot and wrong
 * here: archiving a project you are not looking at must not yank you out of the
 * one you are. Only a move that took the open project off the active shelf
 * reopens anything.
 */
async function refreshPicker(prefer) {
  const picker = $("project-picker");
  picker.textContent = "";
  for (const name of state.projects) picker.append(new Option(name, name));
  if (!state.projects.length) {
    picker.append(new Option("(no projects)", ""));
    return;
  }
  const open = state.project?.name;
  if (open && state.projects.includes(open)) {
    picker.value = open;
    return;
  }
  const target = prefer && state.projects.includes(prefer) ? prefer : state.projects[0];
  picker.value = target;
  await openProject(target);
}

// Opening a project folds any chain that was authored before sources existed —
// one bin row per generation — back into the single source the renderer says it
// always was. See `fold.py`. It happens once and it changes the file, so it is
// said out loud rather than discovered later.
function reportFold(report) {
  const folds = report?.folds ?? [];
  if (!folds.length) return;
  const rows = folds.reduce((total, one) => total + one.absorbed.length, 0);
  const what = folds.length === 1
    ? `${displayName(folds[0].name)} — ${folds[0].generations} generations`
    : `${folds.length} chains`;
  toast(`Folded ${rows} continuation row(s) into ${what}. The edit is unchanged.`);
}

/** Fall back to the first clip, or to the first bin item when nothing is used. */
function restoreSelection() {
  if (!clipById(state.selectedClip)) state.selectedClip = clips()[0]?.id ?? null;
  const clip = currentClip();
  if (clip) state.selectedSource = clip.source_id;
  else if (!sourceById(state.selectedSource)) state.selectedSource = sources()[0]?.id ?? null;
  const source = currentSource();
  if (source) state.depth = Math.min(state.depth, source.segments.length - 1);
}

// Anything that changes the project server-side hands the whole thing back, so
// the client never has to guess what the operation did.
async function apply(request) {
  const epoch = state.projectEpoch;
  const revision = state.editRevision;
  const result = await request;
  if (epoch !== state.projectEpoch) return result;
  if (result.project && revision === state.editRevision) {
    adoptProject(result.project);
  }
  restoreSelection();
  await reloadSequence();
  fillForm();
  refresh();
  return result;
}

// Take on a project document the server just handed back.
//
// Every prompt inside it is a fresh object, and the shot cards on screen write
// through closures over the exact `beat` they were built from — nothing reads
// them back out of the DOM the way `readForm` reads the blocks above them. So
// a card left bound to the copy that was just replaced still *shows* what you
// type into it and never delivers it: the render goes out with the words from
// before the edit, and the next `fillForm` paints those same old words back
// over yours. Rebinding the form is what keeps the cards pointing at the
// document the rest of the editor is reading.
function adoptProject(project) {
  if (!project) return;
  state.project = project;
  // Before the form is rebuilt, not after: `fillForm` draws whatever is
  // selected, and a selection that did not survive the swap would have it draw
  // the wrong source, or none.
  restoreSelection();
  if (state.editorOpen || state.inspectorOpen) keepingCaret(fillForm);
}

// `fillForm` throws the shot cards away and builds new ones, which takes the
// caret with them. A render landing is enough to trigger that, and it lands
// whenever it lands — so the field is found again by the address it is drawn
// with and the caret put back where it was, rather than mid-sentence typing
// being interrupted by something that finished in the background.
function keepingCaret(redraw) {
  const was = document.activeElement;
  const address = was?.dataset?.field || was?.id || "";
  const start = was?.selectionStart ?? null;
  const end = was?.selectionEnd ?? null;

  redraw();

  if (!address) return;
  const back = findField(address) || document.getElementById(address);
  if (!back) return;
  // Also when the element survived: `fillForm` assigns `.value` on the blocks
  // it does not rebuild, which parks the caret at the end of the box.
  back.focus({ preventScroll: true });
  if (start !== null && typeof back.setSelectionRange === "function") {
    const limit = back.value?.length ?? 0;
    try {
      back.setSelectionRange(Math.min(start, limit), Math.min(end ?? start, limit));
    } catch {
      // A field that does not carry a text selection (a checkbox, a select).
    }
  }
}

async function reloadSequence() {
  if (!state.project) return;
  const epoch = state.projectEpoch;
  const requestId = ++state.sequenceRequest;
  // The copy on disk predates anything typed since the last save, so reloading
  // over the top of local edits silently undoes them. A finished render is the
  // usual way in here — it arrives on its own schedule, in the middle of
  // whatever is being written — so what is on screen goes up first and comes
  // back down as part of the reload.
  if (state.dirty) {
    try {
      await pushSource();
    } catch (error) {
      toast(`Could not save before reloading: ${error.message}`, true);
      return;
    }
  }
  if (epoch !== state.projectEpoch || requestId !== state.sequenceRequest || state.dirty) return;
  const revision = state.editRevision;
  const name = encodeURIComponent(state.project.name);
  const [report, takes, projRes] = await Promise.all([
    call(`${API}/sequence?project=${name}`),
    call(`${API}/gallery?project=${name}`),
    call(`${API}/project/load?name=${name}`).catch(() => null),
  ]);
  if (epoch !== state.projectEpoch || requestId !== state.sequenceRequest
      || revision !== state.editRevision || state.dirty) return;
  state.report = report;
  // Older servers may still return a LoadVideo input preview as a take.
  // Filtering here also fixes existing cards before a server restart.
  state.takes = takes.takes.filter(take => take.type !== "input");
  if (projRes?.project) adoptProject(projRes.project);
  restoreSelection();
  updateReviewContext();
  buildPlaylist();
  renderBin();
  renderTimeline();
  if (state.editorOpen || state.reviewOpen) renderCandidates();
  if (!state.player?.playing) {
    seekTo(state.playheadSec || 0);
  }
}

// Local edits live in `state.project` until something needs the server to have
// them. Pushing one source rather than the whole project keeps two tabs from
// overwriting each other's other sources.
let sourceSaveQueue = Promise.resolve();

async function pushSource() {
  const project = state.project?.name;
  const sourceId = currentSource()?.id;
  const epoch = state.projectEpoch;
  if (!project || !sourceId) return;
  // Read a fresh snapshot only when earlier saves have finished. Otherwise a
  // queued older snapshot could overwrite a newer one that already succeeded.
  const save = sourceSaveQueue.catch(() => {}).then(async () => {
    while (epoch === state.projectEpoch && state.project?.name === project) {
      if (currentSource()?.id !== sourceId) return;
      if (state.editorOpen || state.inspectorOpen) readForm();
      const revision = state.editRevision;
      const result = await post(`${API}/source/update`, {
        project, source: structuredClone(currentSource()),
      });
      if (epoch !== state.projectEpoch || state.project?.name !== project
          || currentSource()?.id !== sourceId) return;
      // Typing continued while saving. Keep the local objects and persist the
      // newer revision before allowing a waiting project/source switch through.
      if (revision !== state.editRevision) continue;
      adoptProject(result.project);
      clearDirty();
      return;
    }
  });
  sourceSaveQueue = save;
  return save;
}

async function save() {
  if (!state.project) return;
  await pushSource();
  await reloadSequence();
  toast("Saved");
}

// -- presets --------------------------------------------------------------

async function loadPresets() {
  state.presets = (await call(`${API}/presets`)).presets;
  for (const span of document.querySelectorAll("[data-preset]")) {
    const kind = span.dataset.preset;
    const select = document.createElement("select");
    select.style.width = "auto";
    select.append(new Option("preset…", ""));
    for (const name of state.presets[kind] ?? []) select.append(new Option(name, name));
    select.onchange = async () => {
      if (!select.value || !currentPrompt()) return;
      await applyPreset(kind, select.value);
      select.value = "";
    };
    span.append(select);
  }
}

async function applyPreset(kind, name) {
  const { text } = await call(
    `${API}/preset?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}`);
  const prompt = currentPrompt();
  if (kind === "negatives") {
    prompt.negatives = text.split("\n").map((line) => line.trim()).filter(Boolean);
  } else {
    prompt[kind] = text;
  }
  markDirty();
  fillForm();
  refresh();
}

// -- rendering ------------------------------------------------------------

async function loadProfiles() {
  state.profiles = (await call(`${API}/profiles`)).profiles;
  const picker = $("profile-picker");
  picker.textContent = "";
  if (!state.profiles.length) {
    picker.append(new Option("(no render profile)", ""));
    renderGenActions(currentSegment());
    return;
  }
  for (const name of state.profiles) picker.append(new Option(name, name));
  selectProjectProfile();
  await showCoverage();
  renderGenActions(currentSegment());
}

// A project that pins a render profile renders on it, not on whatever the
// picker was last left at — otherwise a reroll can land on another checkpoint.
function selectProjectProfile() {
  const pinned = state.project?.render_profile;
  const picker = $("profile-picker");
  if (pinned && state.profiles.includes(pinned) && picker.value !== pinned) {
    picker.value = pinned;
    showCoverage();
  }
}

// A profile only drives the inputs it is bound to; the rest the workflow keeps
// for itself. Saying so is the difference between a stat you can trust and one
// that quietly disagrees with what renders.
async function showCoverage() {
  const name = $("profile-picker").value;
  const label = $("profile-coverage");
  if (!name) {
    label.textContent = "";
    return;
  }
  try {
    const { profile } = await call(`${API}/profile?name=${encodeURIComponent(name)}`);
    const owned = ["width", "height"].filter((role) => !profile.bindings?.[role]);
    state.workflowOwnsCanvas = owned.length > 0;
    label.textContent = owned.length ? `workflow sets ${owned.join("/")}` : "";
    label.title = owned.length
      ? "This profile does not bind the canvas, so the size shown here is what " +
        "the prompt asks for, not what the workflow will use."
      : "";
  } catch {
    label.textContent = "";
    state.workflowOwnsCanvas = false;
  }
  refresh();
}

// Render the current segment's prompt at a quality. This is how a source gets
// its first take, and how an edited prompt gets a fresh one. Re-shooting a
// specific take you already like is a take-level action (see `reshoot`).
//
// A pinned seed rides along, which is the one combination the take-level
// actions cannot offer: `reshoot` holds the seed but re-sends the take's frozen
// text, so the words cannot be tweaked there. Here the text is whatever is
// being authored, and the seed is held only when asked.
//
// A render is filed against the *source*, because that is what a take belongs
// to; the prompt comes from the segment being authored.
function renderTarget() {
  const source = currentSource();
  const segment = currentSegment();
  if (!source || !segment) return null;
  return { id: source.id, name: source.name, segment, prompt: segment.prompt };
}

// One press of a render button, at whatever sweep size is set. One seed is a
// render; more is a sweep of the same prompt at consecutive seeds, and the
// difference between the two is only how many workflows come back — the tier,
// the pin and the hold all mean the same thing either way.
// `hold` is left undefined rather than false so an unspecified press inherits
// the render bar's toggle at the dispatch funnel. Defaulting it here would mean
// every button on the bar quietly overriding the mode it is displaying.
async function generate({ quality, hold } = {}) {
  readForm();
  const target = renderTarget();
  const profile = $("profile-picker").value;
  if (!target || !profile) return;
  const segment = target.segment;
  const count = sweepCount();

  if (quality && quality !== segment.quality) {
    segment.quality = quality;
    renderQuality(segment);
    refresh();
  }

  const renders = await prepareRenders(target, profile, segment.quality, count);
  if (!renders.length) return;

  if (!await saveBeforeRender()) return;

  await dispatchRenders(renders, target, QUALITY_LABELS[segment.quality] ?? "Quick",
                        { hold });
}

// Queue — or hold — every workflow one press produced, and report once at the
// end rather than per render. Shared by fresh generations and continuations,
// which differ in how their workflows were built and in nothing after that.
//
// Returns how many actually made it, since a queue can refuse midway.
async function dispatchRenders(renders, target, label, { hold } = {}) {
  const sweeping = renders.length > 1;
  // The one place the shelf-or-queue question is answered. Callers that do not
  // care say nothing and inherit the toggle, which is how a re-shoot, a reroll
  // and a continuation all gained holding without a line of their own.
  const held = hold ?? state.holdRenders;

  let done = 0;
  for (const prepared of renders) {
    // A sweep labels each take with its own seed, because that is the only
    // thing telling them apart in the candidate strip.
    const mode = sweeping ? `${label} sweep #${prepared.seed}` : label;
    const ok = held
      ? await holdPreparedRender(prepared, target, mode)
      : await submit(prepared, target, mode);
    if (ok) done++;
  }
  if (!done) return 0;

  const first = renders[0];
  const canvas = `${first.width}×${first.height}, ${first.frames}f (${first.seconds}s)`;
  if (sweeping) {
    toast(held
      ? `${label}: held ${done} seeds for ${target.name} — ${canvas}`
      : `${label}: queued ${done} seeds for ${target.name} — ${canvas}`);
  } else {
    toast(held
      ? `${label}: held outside ComfyUI's queue for later`
      : `${label}: queued ${canvas}`);
  }
  syncQueue();
  return done;
}

// The workflows for one press: `count` of them, from whichever endpoint suits
// the text being sent. Returns an empty list when the server refused — the
// toast has already been shown by then.
//
// A dirty preview is the awkward case. The batch endpoint composes the prompt
// itself, which would throw away the text the author edited by hand, so the
// only way to keep their words is N separate re-shoots. Those roll their own
// seeds server-side, so a pinned base has to be counted out here to stop the
// sweep being N copies of one seed.
async function prepareRenders(target, profile, quality, count) {
  const body = {
    source_id: target.id,
    profile,
    project: state.project?.name,
    quality,
  };

  if (state.previewDirty) {
    const lp = state.lastPreview || {};
    const one = (offset) => preparePost(`${API}/render/reshoot`, {
      ...body,
      // The hand-edited text is what renders, but the shot's media still has to
      // reach the graph: `prompt` is sent for its references and keyframe only,
      // never re-composed (see `prepare_reshoot`).
      prompt: target.prompt,
      prompt_text: state.previewEditedText,
      width: lp.width || target.prompt.width,
      height: lp.height || target.prompt.height,
      frames: lp.frames || target.prompt.frames || 124,
      seconds: lp.seconds || target.prompt.seconds || 5.167,
      ...(state.seedPin != null ? { seed: state.seedPin + offset } : {}),
    });
    const renders = [];
    for (let i = 0; i < count; i++) {
      const prepared = await one(i);
      // The first failure is the informative one; the rest would repeat it.
      if (!prepared) break;
      renders.push(prepared);
    }
    return renders;
  }

  if (count === 1) {
    const prepared = await preparePost(`${API}/render/prepare`, {
      ...body,
      prompt: target.prompt,
      count: 1,
      ...(state.seedPin != null ? { seed: state.seedPin } : {}),
    });
    return prepared ? [prepared] : [];
  }

  // A pinned seed becomes the batch's base, so the whole sweep is reproducible
  // from that one number; unpinned, the server rolls a base with headroom for
  // the run. Either way the seeds are consecutive.
  const result = await preparePost(`${API}/render/batch`, {
    ...body,
    prompt: target.prompt,
    count,
    ...(state.seedPin != null ? { seed: state.seedPin } : {}),
  });
  return Array.isArray(result?.renders) ? result.renders : [];
}

// Re-shoot a take from its own frozen prompt, canvas and length — not the
// segment's live prompt, which may have drifted. Reroll gives a fresh seed;
// Fine passes the take's seed to reproduce the same composition at full
// quality. Old takes with no recorded canvas fall back to the current one.
async function reshoot(take, target, { quality, seed } = {}) {
  const profile = $("profile-picker").value;
  if (!target || !profile) return;

  const body = {
    source_id: target.id,
    profile,
    project: state.project?.name,
    quality,
    // Frozen text, live media. A take records the words it rendered from and
    // nothing else, so the shot's references and keyframe have to come from the
    // shot; without them a re-shoot of a `<Picture 1>` prompt renders t2v and
    // the model invents a stranger.
    prompt: target.prompt,
    prompt_text: take.prompt_text,
    width: take.width || target.prompt.width,
    height: take.height || target.prompt.height,
    frames: take.frames,
    seconds: take.seconds,
  };
  if (seed !== undefined && seed !== null) body.seed = seed;

  const prepared = await preparePost(`${API}/render/reshoot`, body);
  if (prepared) await queueTake(prepared, target, QUALITY_LABELS[quality] ?? "Reroll");
}

// Refine into a new take, retaining the source for comparison.
async function runRefine(take, target, { guided_resample = true, face_refine = true, megapixels = 2.1, denoise = null, reference_face = null, face_target = "small_faces", enhancement_mode = "h3", enhancement_resolution = "original" } = {}) {
  if (!target) return;

  const body = {
    project: state.project?.name,
    source: extendSource(take),
    take_id: take.id,
    parent_take: take.parent_take || undefined,
    refined_from: take.id,
    prompt_text: take.prompt_text || "",
    width: take.width || target.prompt?.width || 1344,
    height: take.height || target.prompt?.height || 768,
    // Left unset when unknown: the server reads them off the take rather than
    // cropping a longer clip to a guessed five seconds.
    frames: take.frames || undefined,
    seconds: take.seconds || undefined,
    seed: take.seed != null ? take.seed : 42,
    guided_resample: guided_resample,
    face_refine: face_refine,
    face_refine_target: face_target,
    guided_resample_megapixels: megapixels,
    face_refine_denoise: denoise != null ? denoise : undefined,
    reference_face: reference_face || undefined,
    enhancement_mode,
    enhancement_resolution,
  };

  const prepared = await preparePost(`${API}/render/refine`, body);
  if (prepared) await queueTake(prepared, target, enhancement_mode === "seedvr2_faces" ? "SeedVR2 Face Blend" : enhancement_mode === "seedvr2" ? "SeedVR2 Enhance" : "Refine");
}


// Extend a take: generate a NEW clip that continues it, motion and sound
// carrying across the join (H3 Motion Context). The take's own video is the
// source; the user types what happens next. Unlike reshoot, this needs no
// profile — the extend graph is fixed server-side — but it does need the take's
// ComfyUI-visible video reference and its canvas/length to match.
const isVideoTake = (take) => /\.(mp4|webm|mov|mkv)$/i.test(take?.filename || "");

// Frames of the parent clip H3 can pin as motion context. Only these lengths
// are distinct to the video VAE — anything between two of them is snapped DOWN
// by the node, so offering a free number would quietly carry over less than it
// said. Mirrors CONTEXT_GRID in video_prompter/extend.py.
const CONTEXT_GRID = [1, 5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, 209, 226, 243];
const DEFAULT_CONTEXT = 22;

// Longer context is not free: in head mode the pinned run is regenerated at the
// front of the new clip and then trimmed off, so the delivered clip comes back
// shorter by roughly this much. Worth saying in the option text.
const contextLabel = (frames) => {
  const carried = (frames / 24).toFixed(2);
  if (frames === DEFAULT_CONTEXT) return `${carried}s carried over (default)`;
  if (frames === 39) return `${carried}s carried over (same shot)`;
  return `${carried}s carried over`;
};

// The grid values a continuation can actually use. Two ceilings, not one: the
// pinned run has to be shorter than the NEW clip (the node refuses otherwise),
// and no longer than the parent actually has to give (the node would quietly
// pin fewer). Never returns empty, so the control always has a value.
const contextChoices = (clipFrames, availableFrames) => {
  const clip = clipFrames || 124;
  const available = availableFrames || clip;
  const fits = CONTEXT_GRID.filter((g) => g < clip && g <= available);
  return fits.length ? fits : [CONTEXT_GRID[0]];
};

// The control itself. Both places an extend can start — the take card's inline
// field and the footer button — build it the same way, so the two cannot drift.
// Defaults to what `parentTake` itself carried over, keeping a chain consistent
// unless the choice is changed on purpose.
function contextSelect(parentTake, clipFrames) {
  const el = document.createElement("select");
  el.title = "Frames of the previous generation pinned as motion context. More "
    + "carries motion and sound across more strongly, but the pinned run is "
    + "trimmed off the front of the result, so the new clip comes back shorter.";
  const choices = contextChoices(clipFrames, parentTake?.frames);
  for (const frames of choices) {
    const option = document.createElement("option");
    option.value = String(frames);
    option.textContent = contextLabel(frames);
    el.append(option);
  }
  // A deliberate choice outranks the parent's: it is the more recent act, and
  // it is the one still visible in the control. Either may be too long for this
  // clip, so fall back to the default and then to the longest that does fit.
  const wanted = state.extendContext || parentTake?.context_length || DEFAULT_CONTEXT;
  el.value = String(
    choices.includes(wanted) ? wanted
      : choices.includes(DEFAULT_CONTEXT) ? DEFAULT_CONTEXT
      : choices[choices.length - 1]);
  el.onchange = () => { state.extendContext = chosenContext(el); };
  return el;
}

const chosenContext = (el) => Number(el?.value) || DEFAULT_CONTEXT;

// How hard to work on a continuation. Unlike a fresh generation this cannot
// choose its canvas: the pinned frames come from the parent clip, so an extend
// *finishes* on that clip's pixels and a single-pass tier's megapixel budget is
// dropped server-side. A two-pass tier still means something, because it can
// spend its first pass at the small obedient canvas and its last steps back up
// at the parent's — the same trade as a fresh two-pass render, ending where the
// parent ended. That only pays when the parent is meaningfully bigger than the
// first pass, so those tiers are offered only when it is; the server refuses
// them otherwise rather than quietly rendering one pass under a two-pass name.
const QUALITY_LEVELS = ["preview", "turbo_hd", "turbo_2pass", "final", "final_2pass"];

// A two-pass continuation is worth its second pass only when the clip it
// continues has at least twice the pixels of the tier's first pass. Mirrors
// `Api._check_extend_two_pass`, so the control never offers what the server
// would refuse.
function extendTierFits(level, take) {
  const refine = resolveRefine(level);
  if (!refine) return true;
  const pixels = (take?.width || 0) * (take?.height || 0);
  return pixels >= 2 * resolveQuality(level).megapixels * 1e6;
}

// Everything about a continuation that a tier can still change. The canvas is
// not on the list: it is the parent's whichever tier is picked, which is what
// makes `preview` and `turbo_hd` the *same render* here — they differ only in a
// megapixel budget an extend drops. Offered as two options they were one choice
// wearing two names, and whichever you picked, the badge read back whichever
// name the numbers happened to match.
const extendPlanKey = (level) => {
  const preset = resolveQuality(level);
  const refine = resolveRefine(level);
  return JSON.stringify([preset.steps, preset.turbo_strength,
                         preset.reference_size, refine?.pass1_steps ?? null]);
};

// Of two tiers that build the identical graph, the honest name is the one whose
// own canvas is nearest the one this extend will really deliver — the parent's.
// So Quick survives on a small parent and Turbo HD on a 720p one, and the tier
// on the badge is never a claim about a resolution nobody rendered at.
function distinctExtendTiers(parentTake) {
  const parentMp = ((parentTake?.width || 0) * (parentTake?.height || 0)) / 1e6;
  const distance = (level) => Math.abs(Math.log(
    Math.max(outputMegapixels(level), 0.001) / Math.max(parentMp, 0.001)));

  const kept = new Map();
  for (const level of QUALITY_LEVELS.filter((l) => extendTierFits(l, parentTake))) {
    const key = extendPlanKey(level);
    const rival = kept.get(key);
    if (rival === undefined || (parentMp > 0 && distance(level) < distance(rival)))
      kept.set(key, level);
  }
  return kept;
}

function qualitySelect(inherited, parentTake) {
  const el = document.createElement("select");
  el.title = "How hard to work on the continuation. A continuation finishes on "
    + "the canvas of the clip it continues, so a one-pass tier here only changes "
    + "the step count; a two-pass tier proves the motion at a small canvas first "
    + "and spends its last steps back up at the parent's size. Tiers that would "
    + "build the identical graph are listed once.";
  const kept = distinctExtendTiers(parentTake);
  const levels = QUALITY_LEVELS.filter((level) => kept.get(extendPlanKey(level)) === level);
  // The canvas is in every label because it is the surprising part: an extend
  // cannot go smaller than the clip it continues, so "Quick" on a 720p parent
  // costs what Turbo HD costs. Better said in the option than discovered after.
  const canvas = parentTake?.width > 0 && parentTake?.height > 0
    ? `${parentTake.width}×${parentTake.height}, ` : "";
  for (const level of levels) {
    const option = document.createElement("option");
    option.value = level;
    const refine = resolveRefine(level);
    const steps = resolveQuality(level).steps;
    option.textContent = refine
      ? `${QUALITY_LABELS[level]} · ${canvas}${refine.pass1_steps}+${steps - refine.pass1_steps} steps`
      : `${QUALITY_LABELS[level]} · ${canvas}${steps} steps`;
    el.append(option);
  }
  // A deliberate choice outranks the chain's own tier, as with context. The
  // inherited tier is the honest default: it is what an extend with nobody
  // choosing would have rendered at anyway. When that tier was folded into an
  // identical one, land on the survivor — it is the same render, under the name
  // this extend will actually be badged with.
  const wanted = state.extendQuality || inherited;
  el.value = levels.includes(wanted)
    ? wanted
    : (kept.get(extendPlanKey(wanted)) ?? levels[0]);
  el.onchange = () => { state.extendQuality = chosenQuality(el); };
  return el;
}

const chosenQuality = (el) => (QUALITY_LABELS[el?.value] ? el.value : "preview");

// What an extend renders at when nobody says: the segment it will fulfil, or —
// when it will plan a new one — the generation it continues. Mirrors the
// server's own fallback, so the control opens on the truth rather than on a
// tier the render would then ignore.
function extendQualityFor(source, depth) {
  const segments = source?.segments ?? [];
  if (!segments.length) return "preview";
  const wanted = depth ?? segments.length;
  return segments[Math.min(wanted, segments.length - 1)]?.quality || "preview";
}

// A take's ComfyUI-visible reference, the form LoadVideo accepts.
const extendSource = (take) =>
  `${take.subfolder ? `${take.subfolder}/` : ""}${take.filename} [${take.type || "output"}]`;

// The new take lands on the same source as its parent — that is what a chain
// is now, one source deepening — and `depth` says which segment's plan it
// fulfils. The source grows a segment first when the extend runs past its plan,
// so the intent and the result stay the same length.
async function runExtend(take, source, text,
                         { seed, count = 1, quality, depth,
                           label: modeLabel, context } = {}) {
  if (!source) return;
  const wanted = depth ?? source.segments.length;
  if (wanted >= source.segments.length) {
    // A new continuation segment inherits the tier of the one before it unless
    // this says otherwise — the server's own rule, overridden here rather than
    // worked around afterwards.
    await apply(post(`${API}/source/segment/add`, {
      project: state.project.name, source: source.id, quality: quality || undefined,
    }));
    source = sourceById(source.id) ?? source;
  }
  let segment = source.segments[Math.min(wanted, source.segments.length - 1)];
  // An existing segment keeps its tier until someone says otherwise here, the
  // same way the footer's render buttons work for a fresh generation. Writing
  // it down rather than passing it once keeps the quality badge, a later
  // reroll and the next extend agreeing with what actually rendered.
  if (quality && segment && quality !== segment.quality) {
    segment.quality = quality;
    await apply(post(`${API}/source/update`, {
      project: state.project.name, source,
    }));
    source = sourceById(source.id) ?? source;
    segment = source.segments[Math.min(wanted, source.segments.length - 1)] ?? segment;
  }

  const body = {
    source: extendSource(take),
    parent_take: take.id,
    source_id: source.id,
    prompt_text: text,
    project: state.project?.name,
    quality: quality || segment?.quality || "preview",
    width: take.width || segment?.prompt?.width || 608,
    height: take.height || segment?.prompt?.height || 352,
    // A continuation renders through the shot's OWN graph with Motion Context
    // wired in — same checkpoint, same LoRAs, same cast references. Both of
    // these are what tell the server which graph that is: the profile carries
    // the weights, the segment's prompt carries its media. Without them a
    // continuation of a `<Subject 1>` shot rendered plain t2v on stock weights,
    // and the model, given a description and no face, invented a stranger.
    profile: $("profile-picker").value,
    prompt: segment?.prompt,
  };
  // Length is the *segment's* budget, not the parent take's: a 12s opener can
  // be followed by an 8s beat, and the "Generate (s)" field on this generation
  // is where that is said. Seconds alone — the server snaps it to the frame
  // grid. Only a segment with no budget of its own matches what it continues.
  if (segment?.prompt?.duration) {
    body.seconds = segment.prompt.duration;
  } else {
    body.frames = take.frames || 124;
    body.seconds = take.seconds || 5.167;
  }
  if (seed !== undefined && seed !== null) body.seed = seed;
  // Frames of the parent pinned as motion context. Omitted means the server's
  // default; the server snaps anything off-grid and reports back what it
  // really pinned, so the take records the honest number.
  if (context) body.context = context;
  // Ride the cleaner latent path when this take saved an AV latent at render
  // time; older takes have none and fall back to the video path server-side.
  if (take.latent_path) body.latent = take.latent_path;

  // One seed or several. The batch endpoint takes the identical payload plus a
  // count, so the only thing a sweep changes is how many workflows come back;
  // a pinned seed is the base either way, and unpinned the server rolls one
  // with room for the run.
  const seeds = Math.max(1, Math.round(count) || 1);
  let renders;
  if (seeds === 1) {
    const prepared = await preparePost(`${API}/render/extend`, body);
    renders = prepared ? [prepared] : [];
  } else {
    const result = await preparePost(`${API}/render/extend/batch`,
                                     { ...body, count: seeds });
    renders = Array.isArray(result?.renders) ? result.renders : [];
  }
  if (!renders.length) return;

  // Every take in the sweep continues the same parent — that is what makes
  // them comparable, and what keeps the chain intact whichever one is chosen.
  for (const prepared of renders) prepared.parent_take = take.id;

  if (!await saveBeforeRender()) return;
  await dispatchRenders(
    renders,
    { id: source.id, name: source.name, segment, prompt: segment.prompt },
    modeLabel || "Extend",
  );
}

// Inline "what happens next" field, opened from the Extend button — typed in
// place on the take card rather than through a browser prompt dialog. Enter
// extends, Escape closes; only one field open per card.
function openExtendField(take, source, card) {
  const existing = card.querySelector(".extend-field");
  if (existing) {
    existing.querySelector("input")?.focus();
    return;
  }
  const row = document.createElement("div");
  row.className = "extend-field";
  row.style.cssText = "display:flex;gap:6px;margin-top:6px;width:100%";

  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Continue this take — what happens next?";
  input.style.cssText = "flex:1;min-width:0";

  // This path adds a fresh segment, so the continuation is the same length as
  // the take it continues — one ceiling serves as both.
  const context = contextSelect(take, take.frames);
  context.className = "small";
  context.style.cssText = "flex:0 0 auto;max-width:11rem";

  // The tier this would have inherited, offered rather than assumed — a Fine
  // parent is exactly where a cheap draft continuation is worth having, and a
  // Quick one exactly where a keeper is.
  const tier = qualitySelect(extendQualityFor(source), take);
  tier.className = "small";
  tier.style.cssText = "flex:0 0 auto;max-width:9.5rem";

  const go = document.createElement("button");
  go.className = "small primary";
  go.innerHTML = label("chevron-right", "Extend", "sm");

  const cancel = document.createElement("button");
  cancel.className = "small ghost iconbtn";
  cancel.title = "Cancel";
  cancel.innerHTML = iconHTML("x", "sm");

  const close = () => row.remove();
  const submit = guard(async () => {
    const text = input.value.trim();
    if (!text) return input.focus();
    const frames = chosenContext(context);
    const level = chosenQuality(tier);
    close();
    await runExtend(take, source, text, { context: frames, quality: level });
  });

  go.onclick = submit;
  cancel.onclick = close;
  input.onkeydown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      submit();
    } else if (event.key === "Escape") {
      event.preventDefault();
      close();
    }
  };

  row.append(input, context, tier, go, cancel);
  card.append(row);
  input.focus();
}

// Post a prepare request, turning the known error statuses into toasts. Returns
// the prepared render, or null when it could not be prepared.
async function preparePost(path, body) {
  try {
    await videoLoraSave;
    return await post(path, body);
  } catch (error) {
    if (error.status === 422) {
      showFindings(error.body.findings ?? []);
      toast("Fix the errors first, or they will cost you a generation.", true);
    } else if (error.status === 409) {
      toast(`${error.message} Re-capture the profile.`, true);
    } else {
      toast(error.message, true);
    }
    return null;
  }
}

// Save the source, so a take has something to attach to. A batch saves once
// for the whole run rather than once per render — the source does not change
// between them, so the rest would be identical writes.
async function saveBeforeRender() {
  try {
    await pushSource();
    return true;
  } catch (error) {
    toast(`Could not save before rendering: ${error.message}`, true);
    return false;
  }
}

// Save, then queue a lone render. The sweep of one, kept as its own name
// because a re-shoot really is always one clip.
// `hold` defaults to undefined rather than false on purpose: passing an
// explicit false here would override the render bar's toggle, and a re-shoot is
// exactly one of the presses that toggle is meant to catch.
async function queueTake(prepared, target, mode, { hold } = {}) {
  if (!await saveBeforeRender()) return;
  await dispatchRenders([prepared], target, mode, { hold });
}

// `shot` in the queue metadata is the source id. The name is what every
// takes.json on disk already carries, so it stays.
function buildPromptPayload(prepared, target, mode, promptId = uuid()) {
  const promptText = prepared.prompt_text || prepared.prompt || "";
  return {
    prompt: prepared.workflow,
    client_id: state.clientId,
    prompt_id: promptId,
    extra_data: {
      video_prompter: {
        project: state.project?.name,
        shot: target.id,
        shot_name: target.name,
        mode,
        frames: prepared.frames,
        seconds: prepared.seconds,
        seed: prepared.seed,
        prompt: promptText,
        width: prepared.width,
        height: prepared.height,
        // The tier by NAME. `steps` and the canvas cannot always be read
        // backwards into one — an extend renders at its parent's canvas, so a
        // Quick continuation of a 720p clip looks exactly like a Turbo HD one.
        quality: prepared.quality || undefined,
        steps: prepared.steps,
        // Only present on a two-pass render, which is what makes its absence
        // meaningful: the tier readback treats a number here as proof the
        // schedule was cut.
        pass1_steps: prepared.pass1_steps || undefined,
        loras: prepared.loras,
        latent_path: prepared.latent_path,
        parent_take: prepared.parent_take || undefined,
        context_length: prepared.context_length || undefined,
        refinement: prepared.refinement || undefined,
        refined_from: prepared.refined_from || undefined,
        reference_face: prepared.reference_face || undefined,
      },
    },
  };
}

async function holdPreparedRender(prepared, target, mode = "Render") {
  const payload = buildPromptPayload(prepared, target, mode);
  try {
    await post(`${API}/render/hold`, {
      payload,
      summary: payload.extra_data.video_prompter,
    });
    await syncQueue();
    return true;
  } catch (error) {
    toast(`Could not hold render: ${error.message}`, true);
    return false;
  }
}

// Submit to ComfyUI itself. Everything above only prepared the workflow. The
// id is minted here so the job can be tracked from the moment it is sent,
// rather than from whenever the response comes back.
async function submit(prepared, target, mode = "Render") {
  const promptId = uuid();
  const promptText = prepared.prompt_text || prepared.prompt || "";
  const payload = buildPromptPayload(prepared, target, mode, promptId);
  try {
    const response = await fetch("/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error?.message || `Queue refused: ${response.status}`);

    const actualPromptId = body.prompt_id || promptId;
    const jobInfo = {
      promptId: actualPromptId,
      // Kept verbatim so this job can be moved onto the shelf afterwards
      // without its metadata being reconstructed from the fields below — some
      // of which (a keyframe's cast member, its style) have no field here.
      extraData: payload.extra_data,
      shot: target.id,
      shotName: target.name,
      mode: mode,
      frames: prepared.frames,
      seconds: prepared.seconds,
      seed: prepared.seed,
      steps: prepared.steps || 20,
      width: prepared.width,
      height: prepared.height,
      quality: prepared.quality || null,
      prompt: promptText,
      workflow: prepared.workflow,
      latent_path: prepared.latent_path || null,
      parent_take: prepared.parent_take || null,
      context_length: prepared.context_length || null,
      project: state.project?.name || "",
      queuedAt: Date.now(),
      startTime: null,
      status: "queued",
      progress: { value: 0, max: prepared.steps || 20 },
    };
    state.jobs.set(actualPromptId, jobInfo);
    if (typeof editorUX !== "undefined" && editorUX) {
      editorUX.tracker.outcome("render_submitted");
      if (target.prompt && ((target.prompt.cast || []).length
          || (target.prompt.references || []).some(ref => ref.member)
          || speakingMemberIds(target.prompt).length)) editorUX.tracker.outcome("identity_reused");
    }
    syncQueue();
    return true;
  } catch (error) {
    if (typeof editorUX !== "undefined") editorUX?.tracker.outcome("render_rejected");
    toast(error.message, true);
    return false;
  }
}

// -- queue tracker & live jobs --------------------------------------------

function formatDuration(seconds) {
  if (!seconds || isNaN(seconds) || seconds < 0) return "00:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function calcAspectRatio(width, height) {
  const w = Math.round(Number(width));
  const h = Math.round(Number(height));
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return "";
  let a = w;
  let b = h;
  while (b !== 0) [a, b] = [b, a % b];
  const rw = w / a;
  const rh = h / a;
  if ((rw === 16 && rh === 9) || (rw === 9 && rh === 16) || (rw === 4 && rh === 3) || (rw === 1 && rh === 1)) {
    return `${rw}:${rh}`;
  }
  const ratio = w / h;
  if (Math.abs(ratio - 16 / 9) < 0.02) return "16:9";
  if (Math.abs(ratio - 21 / 9) < 0.04) return "21:9";
  if (Math.abs(ratio - 4 / 3) < 0.02) return "4:3";
  if (Math.abs(ratio - 1) < 0.02) return "1:1";
  if (Math.abs(ratio - 9 / 16) < 0.02) return "9:16";
  if (Math.abs(ratio - 3 / 4) < 0.02) return "3:4";
  return `${rw}:${rh}`;
}

function extractJobFromQueueEntry(entry, index = 0, isRunning = false) {
  if (!Array.isArray(entry) || entry.length < 2) return null;
  const num = entry[0];
  const promptId = String(entry[1] || "");
  const workflow = entry[2] || {};
  const extraData = entry[3] || {};
  const vpMeta = extraData.video_prompter || {};

  const existing = state.jobs.get(promptId) || {};

  let shotId = vpMeta.shot || existing.shot || "";
  let shotName = vpMeta.shot_name || existing.shotName || "";
  if (!shotName && shotId) {
    const found = sourceById(shotId);
    if (found) shotName = found.name;
  }
  if (!shotName) {
    shotName = isRunning ? "Active Render" : `Queued Job #${index + 1}`;
  }

  let promptText = vpMeta.prompt || existing.prompt || "";
  if (!promptText && typeof workflow === "object") {
    for (const node of Object.values(workflow)) {
      if (node?.inputs?.prompt && typeof node.inputs.prompt === "string") {
        promptText = node.inputs.prompt;
        break;
      } else if (node?.inputs?.text && typeof node.inputs.text === "string") {
        promptText = node.inputs.text;
        break;
      }
    }
  }

  let seed = vpMeta.seed ?? existing.seed;
  let steps = vpMeta.steps ?? existing.steps;
  let width = vpMeta.width ?? existing.width;
  let height = vpMeta.height ?? existing.height;
  let frames = vpMeta.frames ?? existing.frames;
  let seconds = vpMeta.seconds ?? existing.seconds;
  let mode = vpMeta.mode || existing.mode || "";

  if (typeof workflow === "object") {
    for (const node of Object.values(workflow)) {
      const cls = String(node?.class_type || "");
      if (!mode) {
        if (cls.includes("ImageToVideo")) mode = "Image-to-Video";
        else if (cls.includes("ReferenceToVideo")) mode = "Reference-to-Video";
        else if (cls.includes("MotionContext")) mode = "Extend";
        else if (cls.includes("TextToVideo")) mode = "Text-to-Video";
        else if (cls.includes("T2I") || cls.includes("Krea")) mode = "Keyframe (T2I)";
      }
      if (seed == null) {
        if (node?.inputs?.seed != null && typeof node.inputs.seed === "number") seed = node.inputs.seed;
        else if (node?.inputs?.noise_seed != null && typeof node.inputs.noise_seed === "number") seed = node.inputs.noise_seed;
      }
      if (steps == null && node?.inputs?.steps != null && typeof node.inputs.steps === "number") {
        steps = node.inputs.steps;
      }
      if ((!width || !height) && node?.inputs?.width && node?.inputs?.height) {
        width = node.inputs.width;
        height = node.inputs.height;
      }
      if (node?.inputs?.length != null &&
          (cls.includes("ImageToVideo") || cls.includes("ReferenceToVideo") || cls.includes("TextToVideo"))) {
        frames = node.inputs.length;
      } else if (!frames && node?.inputs?.length != null && !cls.includes("ImageFromBatch")) {
        frames = node.inputs.length;
      }
    }
  }
  if (!mode) mode = vpMeta.parent_take ? "Extend" : "Render";

  const job = {
    promptId,
    number: num,
    // ComfyUI hands back what was submitted, so a job read off the live queue
    // after a reload can still be shelved with its original metadata intact.
    extraData: (entry[3] && Object.keys(extraData).length) ? extraData : existing.extraData,
    shot: shotId,
    shotName,
    mode,
    prompt: promptText,
    workflow,
    width: width || 832,
    height: height || 480,
    frames: frames || 124,
    seconds: seconds || (frames ? Math.round((frames / 24) * 1000) / 1000 : 5.17),
    seed: seed != null ? seed : null,
    steps: steps || 20,
    quality: vpMeta.quality || existing.quality || null,
    latent_path: vpMeta.latent_path || existing.latent_path || null,
    parent_take: vpMeta.parent_take || existing.parent_take || null,
    context_length: vpMeta.context_length || existing.context_length || null,
    project: vpMeta.project || existing.project || state.project?.name || "",
    queuedAt: existing.queuedAt || extraData?.create_time || Date.now(),
    // The clock a job shows counts rendering, not waiting. Starting it at
    // submit time made every job in a deep queue look like it had been
    // rendering since the batch was sent.
    startTime: existing.startTime || (isRunning ? Date.now() : null),
    status: isRunning ? "running" : "queued",
    progress: existing.progress || { value: 0, max: steps || 20 },
    currentNode: existing.currentNode || null,
    nodeProgress: existing.nodeProgress || {},
    cachedNodes: existing.cachedNodes || [],
    outputNodes: Array.isArray(entry[4]) ? entry[4].map(String) : (existing.outputNodes || []),
    overallProgress: Number(existing.overallProgress) || 0,
    failed: Boolean(existing.failed),
  };

  state.jobs.set(promptId, job);
  return job;
}

function extractJobFromHistoryEntry(promptId, entry) {
  if (!entry || typeof entry !== "object") return null;
  const promptArr = Array.isArray(entry.prompt) ? entry.prompt : [];
  const num = promptArr[0] || 0;
  const workflow = promptArr[2] || {};
  const extraData = promptArr[3] || {};
  const vpMeta = extraData.video_prompter || {};

  let shotId = vpMeta.shot || "";
  let shotName = vpMeta.shot_name || "";
  if (!shotName && shotId) {
    const found = sourceById(shotId);
    if (found) shotName = found.name;
  }

  let promptText = vpMeta.prompt || "";
  if (!promptText && typeof workflow === "object") {
    for (const node of Object.values(workflow)) {
      if (node?.inputs?.prompt && typeof node.inputs.prompt === "string") {
        promptText = node.inputs.prompt;
        break;
      } else if (node?.inputs?.text && typeof node.inputs.text === "string") {
        promptText = node.inputs.text;
        break;
      }
    }
  }

  let seed = vpMeta.seed;
  let steps = vpMeta.steps;
  let width = vpMeta.width;
  let height = vpMeta.height;
  let frames = vpMeta.frames;
  let seconds = vpMeta.seconds;
  let mode = vpMeta.mode || "";

  if (typeof workflow === "object") {
    for (const node of Object.values(workflow)) {
      const cls = String(node?.class_type || "");
      if (!mode) {
        if (cls.includes("ImageToVideo")) mode = "Image-to-Video";
        else if (cls.includes("ReferenceToVideo")) mode = "Reference-to-Video";
        else if (cls.includes("MotionContext")) mode = "Extend";
        else if (cls.includes("TextToVideo")) mode = "Text-to-Video";
        else if (cls.includes("T2I") || cls.includes("Krea")) mode = "Keyframe (T2I)";
      }
      if (seed == null) {
        if (node?.inputs?.seed != null && typeof node.inputs.seed === "number") seed = node.inputs.seed;
        else if (node?.inputs?.noise_seed != null && typeof node.inputs.noise_seed === "number") seed = node.inputs.noise_seed;
      }
      if (steps == null && node?.inputs?.steps != null && typeof node.inputs.steps === "number") {
        steps = node.inputs.steps;
      }
      if ((!width || !height) && node?.inputs?.width && node?.inputs?.height) {
        width = node.inputs.width;
        height = node.inputs.height;
      }
      if (node?.inputs?.length != null &&
          (cls.includes("ImageToVideo") || cls.includes("ReferenceToVideo") || cls.includes("TextToVideo"))) {
        frames = node.inputs.length;
      } else if (!frames && node?.inputs?.length != null && !cls.includes("ImageFromBatch")) {
        frames = node.inputs.length;
      }
    }
  }
  if (!mode) mode = vpMeta.parent_take ? "Extend" : "Render";

  // Extract output files from entry.outputs
  let filename = "";
  let subfolder = "";
  let type = "output";
  let url = null;
  let isVideo = false;

  if (entry.outputs && typeof entry.outputs === "object") {
    for (const nodeOutput of Object.values(entry.outputs)) {
      const fileList = ["images", "videos", "gifs"].flatMap(key =>
        Array.isArray(nodeOutput[key]) ? nodeOutput[key] : []);
      if (Array.isArray(fileList) && fileList.length > 0) {
        const fileItem = fileList.find(item => item?.filename && item.type !== "input");
        if (fileItem && typeof fileItem === "object") {
          filename = fileItem.filename || "";
          subfolder = fileItem.subfolder || "";
          type = fileItem.type || "output";
          if (filename) {
            url = `/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder)}&type=${encodeURIComponent(type)}`;
            isVideo = /\.(mp4|webm|mkv|mov|avi|gif)$/i.test(filename);
            break;
          }
        }
      }
    }
  }

  // Extract duration and timestamps
  let duration = null;
  let completedAt = extraData?.create_time || Date.now();
  let startTs = null;
  let endTs = null;

  if (entry.status?.messages && Array.isArray(entry.status.messages)) {
    for (const msg of entry.status.messages) {
      if (Array.isArray(msg) && msg.length >= 2) {
        const evt = msg[0];
        const payload = msg[1];
        if (evt === "execution_start" && payload?.timestamp) startTs = payload.timestamp;
        if (evt === "execution_success" && payload?.timestamp) endTs = payload.timestamp;
      }
    }
  }
  if (startTs && endTs) {
    duration = Math.round(((endTs - startTs) / 1000) * 10) / 10;
    completedAt = endTs;
  } else if (endTs) {
    completedAt = endTs;
  }

  if (!shotName) {
    if (filename) {
      shotName = filename.replace(/\.(mp4|webm|mkv|mov|png|jpg)$/i, "");
    } else {
      shotName = `Render #${num || promptId.slice(0, 6)}`;
    }
  }

  const statusStr = entry.status?.status_str || (entry.status?.completed ? "completed" : "failed");

  return {
    promptId,
    number: num,
    shot: shotId,
    shotName,
    mode,
    prompt: promptText,
    filename,
    subfolder,
    type,
    url,
    isVideo,
    width: width || 832,
    height: height || 480,
    frames: frames || 124,
    seconds: seconds || (frames ? Math.round((frames / 24) * 1000) / 1000 : 5.17),
    seed: seed != null ? seed : null,
    steps: steps || 20,
    quality: vpMeta.quality || null,
    duration,
    latent_path: vpMeta.latent_path || null,
    parent_take: vpMeta.parent_take || null,
    context_length: vpMeta.context_length || null,
    project: vpMeta.project || state.project?.name || "",
    status: statusStr === "success" ? "completed" : statusStr,
    queuedAt: extraData?.create_time || null,
    completedAt,
    workflow,
  };
}

function getCombinedHistory() {
  const historyMap = new Map();

  // 1. Session & ComfyUI history items
  for (const job of state.jobHistory || []) {
    const key = job.promptId || job.url || job.filename || Math.random();
    historyMap.set(key, job);
  }

  // 2. Project takes (saved renders tied to output files)
  for (const take of state.takes || []) {
    const key = take.prompt_id || take.id || take.filename;
    const found = sourceById(take.shot);
    const shotName = found?.name || take.shot || "Source";
    const existing = historyMap.get(key) || {};

    const url = take.url || (take.filename ? `/view?filename=${encodeURIComponent(take.filename)}&subfolder=${encodeURIComponent(take.subfolder || "")}&type=${encodeURIComponent(take.type || "output")}` : null);

    historyMap.set(key, {
      ...existing,
      promptId: take.prompt_id || existing.promptId || key,
      shot: take.shot || existing.shot,
      shotName: existing.shotName && existing.shotName !== "Source" ? existing.shotName : shotName,
      mode: take.parent_take ? "Extend" : (existing.mode || "Render"),
      prompt: take.prompt_text || existing.prompt || "",
      filename: take.filename || existing.filename,
      subfolder: take.subfolder || existing.subfolder || "",
      type: take.type || existing.type || "output",
      url: url || existing.url,
      width: take.width || existing.width || 832,
      height: take.height || existing.height || 480,
      frames: take.frames || existing.frames || 124,
      seconds: take.seconds || existing.seconds || 5.17,
      seed: take.seed != null ? take.seed : existing.seed,
      steps: take.steps || existing.steps || 20,
      quality: take.quality || existing.quality || null,
      duration: take.generation_time || existing.duration || null,
      latent_path: take.latent_path || existing.latent_path || null,
      parent_take: take.parent_take || existing.parent_take || null,
      context_length: take.context_length || existing.context_length || null,
      status: existing.status || "completed",
      completedAt: take.created ? new Date(take.created).getTime() : (existing.completedAt || Date.now()),
    });
  }

  // Sort by completedAt descending
  return Array.from(historyMap.values()).sort((a, b) => (b.completedAt || 0) - (a.completedAt || 0));
}

function applyProgressState(data) {
  const promptId = data?.prompt_id;
  if (state.runningJob && promptId && promptId !== state.runningJob.promptId && !state.jobs.has(promptId)) {
    return null; // Ignore a just-finished registry snapshot during queue turnover.
  }
  let target = (state.runningJob && (!promptId || promptId === state.runningJob.promptId))
    ? state.runningJob
    : (promptId && state.jobs.has(promptId) ? state.jobs.get(promptId) : null);
  if (!target && promptId) {
    target = {
      promptId,
      status: "running",
      startTime: Date.now(),
      workflow: {},
      nodeProgress: {},
    };
    state.jobs.set(promptId, target);
    state.runningJob = target;
  }
  if (!target || !data?.nodes || typeof data.nodes !== "object") return null;

  const normalized = {};
  let active = null;
  for (const [reportedId, raw] of Object.entries(data.nodes)) {
    if (!raw || typeof raw !== "object") continue;
    const nodeId = resolveWorkflowNodeId(
      target.workflow,
      raw.display_node_id,
      raw.real_node_id,
      raw.node_id,
      reportedId,
    );
    const progress = { ...raw, node_id: nodeId || String(reportedId) };
    normalized[progress.node_id] = progress;
    if (progress.state === "running") active = progress;
  }
  target.nodeProgress = normalized;
  if (active) {
    target.currentNode = active.node_id;
    target.progress = {
      value: active.value,
      max: active.max,
      node: active.node_id,
      display_node: active.display_node_id,
    };
  }
  return target;
}

function normalizeHeldJob(job) {
  const summary = job?.summary || job?.payload?.extra_data?.video_prompter || {};
  return {
    ...summary,
    heldId: job?.id,
    shot: summary.shot || "",
    shotName: summary.shot_name || summary.shotName || "Source",
    promptId: job?.payload?.prompt_id || "",
    workflow: job?.payload?.prompt || {},
    status: job?.status || "held",
    releaseAt: job?.release_at || null,
    createdAt: job?.created_at || null,
    attempts: job?.attempts || 0,
    error: job?.last_error || "",
  };
}

async function syncQueue() {
  try {
    const [queueRes, historyRes, progressRes, heldRes] = await Promise.all([
      fetch("/queue").catch(() => null),
      fetch("/history?max_items=100").catch(() => null),
      fetch(`${API}/render/progress`).catch(() => null),
      fetch(`${API}/render/held`).catch(() => null),
    ]);

    if (queueRes && queueRes.ok) {
      const data = await queueRes.json().catch(() => ({}));
      const runningEntries = data.queue_running || [];
      const pendingEntries = data.queue_pending || [];

      if (runningEntries.length > 0) {
        state.runningJob = extractJobFromQueueEntry(runningEntries[0], 0, true);
      } else {
        state.runningJob = null;
      }

      state.pendingJobs = pendingEntries
        .map((entry, idx) => extractJobFromQueueEntry(entry, idx, false))
        .filter(Boolean);
    }

    if (progressRes && progressRes.ok) {
      const progressData = await progressRes.json().catch(() => ({}));
      applyProgressState(progressData);
    }

    if (heldRes && heldRes.ok) {
      const heldData = await heldRes.json().catch(() => ({}));
      state.heldJobs = Array.isArray(heldData.jobs) ? heldData.jobs.map(normalizeHeldJob) : [];
    }

    if (historyRes && historyRes.ok) {
      const historyData = await historyRes.json().catch(() => ({}));
      for (const [promptId, entry] of Object.entries(historyData)) {
        const job = extractJobFromHistoryEntry(promptId, entry);
        if (job) {
          const existingIdx = state.jobHistory.findIndex((h) => h.promptId === promptId);
          if (existingIdx >= 0) {
            state.jobHistory[existingIdx] = { ...job, ...state.jobHistory[existingIdx] };
          } else {
            state.jobHistory.push(job);
          }
        }
      }
    }

    renderQueueUI();
  } catch {
    renderQueueUI();
  }
}

function startQueueTicker() {
  if (state.queueTicker) clearInterval(state.queueTicker);
  state.queueTicker = setInterval(() => {
    if (state.runningJob) {
      updateRunningTimeAndProgressUI();
    }
  }, 500);
}

// Pipeline stage & progress calculation imported from ./pipeline-progress.js

function updateRunningTimeAndProgressUI() {
  const job = state.runningJob;
  if (!job) return;

  const elapsed = job.startTime ? (Date.now() - job.startTime) / 1000 : 0;
  const elapsedStr = formatDuration(elapsed);
  const stage = getJobStageInfo(job);

  // Mini progress bar in header
  const miniFill = $("queue-mini-progress-fill");
  applyProgressFill(miniFill, stage);

  const updateCardEl = (containerSelector) => {
    const card = document.querySelector(`${containerSelector} .job-card.running`);
    if (!card) return;
    const timeEl = card.querySelector(".job-time-ticker");
    if (timeEl) timeEl.textContent = `${elapsedStr} elapsed`;
    const fillEl = card.querySelector(".job-progress-fill");
    applyProgressFill(fillEl, stage);
    const stepEl = card.querySelector(".job-step-text");
    if (stepEl) stepEl.innerHTML = stageMarkup(stage);
    const detailEl = card.querySelector(".job-stage-detail");
    if (detailEl) detailEl.textContent = `Overall ${stage.overallPct}% · ${stage.detail}`;
  };

  updateCardEl("#queue-running-container");
  updateCardEl("#in-flight");

  // Inside Job Inspector Modal if active
  if (state.jobInspectorOpen && state.inspectedJob?.promptId === job.promptId) {
    const bannerSub = $("job-inspector-status-sub");
    if (bannerSub) {
      bannerSub.textContent = `Elapsed: ${elapsedStr} · Overall ${stage.overallPct}% · ${stage.label}${stage.metric ? ` · ${stage.metric}` : ""}`;
    }
  }
}

function renderQueueUI() {
  renderQueueButton();
  renderQueueModal();
  showJobs();
  // The studio's own grid shows the images still generating, so it follows the
  // queue for the same reason every other job view does.
  renderStudioGrid();
}

function renderQueueButton() {
  const btn = $("queue-tracker-btn");
  const dot = $("header-queue-dot");
  const labelEl = $("queue-btn-label");
  const miniBar = $("queue-mini-progress");
  const miniFill = $("queue-mini-progress-fill");
  if (!btn || !dot || !labelEl) return;

  const isRunning = Boolean(state.runningJob);
  const pendingCount = state.pendingJobs.length;
  const heldCount = state.heldJobs.length;

  if (isRunning) {
    btn.classList.add("active");
    dot.className = "queue-status-dot active";
    labelEl.textContent = `1 Rendering${pendingCount > 0 ? ` (+${pendingCount})` : ""}${heldCount > 0 ? ` · ${heldCount} Held` : ""}`;
    if (miniBar) miniBar.classList.add("visible");
    applyProgressFill(miniFill, getJobStageInfo(state.runningJob));
  } else if (pendingCount > 0) {
    btn.classList.remove("active");
    dot.className = "queue-status-dot pending";
    labelEl.textContent = `${pendingCount} Queued${heldCount > 0 ? ` · ${heldCount} Held` : ""}`;
    if (miniBar) miniBar.classList.remove("visible");
  } else {
    btn.classList.remove("active");
    dot.className = "queue-status-dot";
    labelEl.textContent = heldCount > 0 ? `Queue clear · ${heldCount} Held` : "Queue (0)";
    if (miniBar) miniBar.classList.remove("visible");
  }
}

function renderQueueModal() {
  if (!state.queueTrackerOpen) return;

  const runningCount = state.runningJob ? 1 : 0;
  const pendingCount = state.pendingJobs.length;
  const heldCount = state.heldJobs.length;
  const fullHistory = getCombinedHistory();
  const historyCount = fullHistory.length;

  const activeTotal = runningCount + pendingCount;
  const allTotal = activeTotal + heldCount + historyCount;

  // Filter text query
  const q = (state.queueFilterQuery || "").trim().toLowerCase();
  const matchesQuery = (job) => {
    if (!q) return true;
    return (
      (job.shotName && job.shotName.toLowerCase().includes(q)) ||
      (job.prompt && job.prompt.toLowerCase().includes(q)) ||
      (job.filename && job.filename.toLowerCase().includes(q)) ||
      (job.mode && job.mode.toLowerCase().includes(q)) ||
      (job.promptId && job.promptId.toLowerCase().includes(q))
    );
  };

  const filteredHistory = fullHistory.filter(matchesQuery);

  const summaryStat = $("queue-summary-stat");
  if (summaryStat) {
    summaryStat.textContent = `${runningCount} rendering · ${pendingCount} queued · ${heldCount} held/scheduled · ${historyCount} historical renders`;
  }

  const tabAllCount = $("tab-count-all");
  if (tabAllCount) tabAllCount.textContent = String(allTotal);
  const tabActiveCount = $("tab-count-active");
  if (tabActiveCount) tabActiveCount.textContent = String(activeTotal);
  const tabHeldCount = $("tab-count-held");
  if (tabHeldCount) tabHeldCount.textContent = String(heldCount);
  const tabHistoryCount = $("tab-count-history");
  if (tabHistoryCount) tabHistoryCount.textContent = String(historyCount);

  const runningBadge = $("queue-running-count");
  if (runningBadge) runningBadge.textContent = String(runningCount);
  const pendingBadge = $("queue-pending-count");
  if (pendingBadge) pendingBadge.textContent = String(pendingCount);
  const heldBadge = $("queue-held-count");
  if (heldBadge) heldBadge.textContent = String(heldCount);
  const historyBadge = $("queue-history-count");
  if (historyBadge) historyBadge.textContent = String(filteredHistory.length);

  const interruptBtn = $("queue-interrupt-btn");
  if (interruptBtn) interruptBtn.style.display = runningCount > 0 ? "inline-flex" : "none";

  const clearPendingBtn = $("queue-clear-pending-btn");
  if (clearPendingBtn) clearPendingBtn.style.display = pendingCount > 0 ? "inline-flex" : "none";

  const shelvePendingBtn = $("queue-shelve-pending-btn");
  if (shelvePendingBtn) shelvePendingBtn.style.display = pendingCount > 0 ? "inline-flex" : "none";
  const releaseAllHeldBtn = $("queue-release-all-held-btn");
  if (releaseAllHeldBtn) releaseAllHeldBtn.style.display = heldCount > 0 ? "inline-flex" : "none";
  const heldBulk = $("queue-held-bulk");
  if (heldBulk) heldBulk.style.display = heldCount > 0 ? "inline-flex" : "none";
  const unscheduleAllBtn = $("queue-unschedule-all-held-btn");
  if (unscheduleAllBtn) {
    const anyScheduled = state.heldJobs.some((job) => job.status === "scheduled");
    unscheduleAllBtn.style.display = anyScheduled ? "inline-flex" : "none";
  }

  // Tab display switching
  const tab = state.queueActiveTab || "all";
  const runningSec = $("queue-running-section");
  const pendingSec = $("queue-pending-section");
  const heldSec = $("queue-held-section");
  const historySec = $("queue-history-section");

  if (runningSec) runningSec.style.display = (tab === "history" || tab === "held") ? "none" : "block";
  if (pendingSec) pendingSec.style.display = (tab === "history" || tab === "held") ? "none" : "block";
  if (heldSec) heldSec.style.display = (tab === "active" || tab === "history") ? "none" : "block";
  if (historySec) historySec.style.display = (tab === "active" || tab === "held") ? "none" : "block";

  // 1. Currently Running Container
  const runningContainer = $("queue-running-container");
  if (runningContainer && runningSec && runningSec.style.display !== "none") {
    runningContainer.textContent = "";
    if (state.runningJob && matchesQuery(state.runningJob)) {
      const job = state.runningJob;
      const card = createJobCard(job, { isRunning: true });
      runningContainer.append(card);
    } else if (state.runningJob && !matchesQuery(state.runningJob)) {
      runningContainer.innerHTML = `<div class="queue-empty">Active render does not match filter "${escapeHtml(q)}".</div>`;
    } else {
      runningContainer.innerHTML = `<div class="queue-empty">No render currently executing. Engine is idle.</div>`;
    }
  }

  // 2. Upcoming Queued Jobs Container
  const pendingContainer = $("queue-pending-container");
  if (pendingContainer && pendingSec && pendingSec.style.display !== "none") {
    pendingContainer.textContent = "";
    const filteredPending = state.pendingJobs.filter(matchesQuery);
    if (filteredPending.length > 0) {
      const list = document.createElement("div");
      list.className = "queue-list";
      filteredPending.forEach((job, idx) => {
        const card = createJobCard(job, { isPending: true, queueIndex: idx + 1 });
        list.append(card);
      });
      pendingContainer.append(list);
    } else if (state.pendingJobs.length > 0) {
      pendingContainer.innerHTML = `<div class="queue-empty">No queued jobs match filter "${escapeHtml(q)}".</div>`;
    } else {
      pendingContainer.innerHTML = `<div class="queue-empty">No upcoming jobs waiting in queue.</div>`;
    }
  }

  // 3. Held and scheduled jobs (not present in ComfyUI yet)
  const heldContainer = $("queue-held-container");
  if (heldContainer && heldSec && heldSec.style.display !== "none") {
    heldContainer.textContent = "";
    const filteredHeld = state.heldJobs.filter(matchesQuery);
    if (filteredHeld.length > 0) {
      const list = document.createElement("div");
      list.className = "queue-list";
      filteredHeld.forEach((job) => list.append(createJobCard(job, { isHeld: true })));
      heldContainer.append(list);
    } else if (state.heldJobs.length > 0) {
      heldContainer.innerHTML = `<div class="queue-empty">No held jobs match filter "${escapeHtml(q)}".</div>`;
    } else {
      heldContainer.innerHTML = `<div class="queue-empty">No held jobs. Arm Hold in the render bar, or shelve a queued job, to keep overnight work out of ComfyUI's live queue.</div>`;
    }
  }

  // 4. Render History Container (Tied to output files)
  const historyContainer = $("queue-history-container");
  if (historyContainer && historySec && historySec.style.display !== "none") {
    releaseWithin(historyContainer);
    historyContainer.textContent = "";
    if (filteredHistory.length > 0) {
      const list = document.createElement("div");
      list.className = "queue-list";
      filteredHistory.forEach((job) => {
        const card = createJobCard(job, { isHistory: true });
        list.append(card);
      });
      historyContainer.append(list);
    } else if (fullHistory.length > 0) {
      historyContainer.innerHTML = `<div class="queue-empty">No historical renders match filter "${escapeHtml(q)}".</div>`;
    } else {
      historyContainer.innerHTML = `<div class="queue-empty">No completed renders recorded yet.</div>`;
    }
  }
}

function createJobCard(job, { isRunning = false, isPending = false, isHeld = false, isHistory = false, queueIndex = 0 } = {}) {
  const card = document.createElement("div");
  const hasThumb = Boolean(isHistory && job.url);
  card.className = `job-card ${hasThumb ? "job-card-with-thumb" : ""} ${isRunning ? "running" : isPending ? "pending" : isHeld ? "held" : job.status || "history"}`;

  // If historical take with output video, render the video thumbnail
  if (hasThumb) {
    const thumbBox = document.createElement("div");
    thumbBox.className = "job-thumb-box";
    thumbBox.title = "Click to inspect prompt & watch output";

    const vid = lazyThumb(job.url);

    // Hover snippet play
    thumbBox.onmouseenter = () => { vid.play().catch(() => {}); };
    thumbBox.onmouseleave = () => { vid.pause(); vid.currentTime = 0; };
    thumbBox.onclick = (e) => {
      e.stopPropagation();
      openJobInspector(job);
    };

    const playOverlay = document.createElement("div");
    playOverlay.className = "job-thumb-play";
    playOverlay.innerHTML = iconHTML("play", "sm");

    thumbBox.append(vid, playOverlay);
    card.append(thumbBox);
  }

  const contentCol = document.createElement("div");
  contentCol.className = "job-card-content";
  contentCol.style.display = "flex";
  contentCol.style.flexDirection = "column";
  contentCol.style.gap = "7px";
  contentCol.style.flex = "1 1 auto";
  contentCol.style.minWidth = "0";

  const top = document.createElement("div");
  top.className = "job-card-top";

  if (isPending) {
    const posTag = document.createElement("span");
    posTag.className = "job-tag pos";
    posTag.textContent = `#${queueIndex} in queue`;
    top.append(posTag);
  }

  if (isHeld) {
    const holdTag = document.createElement("span");
    holdTag.className = `job-tag ${job.status === "error" ? "status-failed" : "mode"}`;
    if (job.status === "scheduled" && job.releaseAt) {
      holdTag.textContent = `Scheduled ${new Date(job.releaseAt).toLocaleString()}`;
    } else if (job.status === "error") {
      holdTag.textContent = "Release failed";
      holdTag.title = job.error || "Release failed";
    } else {
      holdTag.textContent = "Held";
    }
    top.append(holdTag);
  }

  const shotBadge = document.createElement("span");
  shotBadge.className = "job-shot-badge";
  shotBadge.innerHTML = `${iconHTML("film", "sm")} ${escapeHtml(job.shotName || "Source")}`;
  top.append(shotBadge);

  if (job.mode) {
    const modeTag = document.createElement("span");
    modeTag.className = "job-tag mode";
    modeTag.textContent = job.mode;
    top.append(modeTag);
  }

  if (job.seed != null) {
    const seedTag = document.createElement("span");
    seedTag.className = "job-tag seed";
    seedTag.textContent = `Seed #${job.seed}`;
    top.append(seedTag);
  }

  if (job.width && job.height) {
    const resTag = document.createElement("span");
    resTag.className = "job-tag";
    resTag.textContent = `${job.width}×${job.height} (${calcAspectRatio(job.width, job.height)})`;
    top.append(resTag);
  }

  if (job.seconds && job.frames) {
    const durTag = document.createElement("span");
    durTag.className = "job-tag";
    durTag.textContent = `${job.seconds}s (${job.frames}f)`;
    top.append(durTag);
  }

  if (isRunning) {
    const statusTag = document.createElement("span");
    statusTag.className = "job-tag status-running";
    statusTag.textContent = "● Rendering";
    top.append(statusTag);

    const timeTicker = document.createElement("span");
    timeTicker.className = "stat job-time-ticker";
    const elapsed = job.startTime ? (Date.now() - job.startTime) / 1000 : 0;
    timeTicker.textContent = `${formatDuration(elapsed)} elapsed`;
    top.append(timeTicker);
  } else if (isHistory) {
    const statusTag = document.createElement("span");
    statusTag.className = `job-tag status-${job.status || "done"}`;
    const durStr = job.duration ? ` in ${job.duration.toFixed(1)}s` : "";
    statusTag.textContent = job.status === "completed" ? `✓ Finished${durStr}` : `✕ ${job.status || "Stopped"}`;
    top.append(statusTag);
  }

  if (job.filename) {
    const fileTag = document.createElement("span");
    fileTag.className = "job-tag file";
    fileTag.title = job.filename;
    fileTag.innerHTML = `${iconHTML("file", "sm")} ${escapeHtml(job.filename.split("/").pop())}`;
    top.append(fileTag);
  }

  const actions = document.createElement("div");
  actions.className = "job-card-actions";

  const inspectBtn = document.createElement("button");
  inspectBtn.className = "small ghost fixed";
  inspectBtn.innerHTML = label("file-text", "Inspect", "sm");
  inspectBtn.title = "Inspect prompt text, output video and generation settings";
  inspectBtn.onclick = (e) => {
    e.stopPropagation();
    openJobInspector(job);
  };
  actions.append(inspectBtn);

  if (isRunning) {
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "small danger fixed";
    cancelBtn.innerHTML = label("stop-circle", "Interrupt", "sm");
    cancelBtn.title = "Interrupt currently running generation";
    cancelBtn.onclick = (e) => {
      e.stopPropagation();
      interruptRunningJob();
    };
    actions.append(cancelBtn);
  } else if (isPending) {
    // The way back off the live queue. Its opposite, "Run now", has always sat
    // on the held cards; this is the same round trip finally closed, and the
    // undo for a render started with the hold toggle forgotten.
    const shelveBtn = document.createElement("button");
    shelveBtn.className = "small ghost fixed";
    shelveBtn.innerHTML = label("clock", "Shelve", "sm");
    shelveBtn.title = "Move this job out of ComfyUI's queue and onto the shelf, "
      + "to run later or overnight";
    shelveBtn.onclick = (e) => {
      e.stopPropagation();
      shelveQueuedJob(job);
    };
    actions.append(shelveBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "small ghost danger fixed";
    deleteBtn.innerHTML = iconHTML("trash", "sm");
    deleteBtn.title = "Remove job from queue";
    deleteBtn.onclick = (e) => {
      e.stopPropagation();
      deleteQueuedJob(job.promptId);
    };
    actions.append(deleteBtn);
  } else if (isHeld) {
    const runBtn = document.createElement("button");
    runBtn.className = "small primary fixed";
    runBtn.innerHTML = label("play", "Run now", "sm");
    runBtn.title = "Release this job into ComfyUI's live queue now";
    runBtn.onclick = (event) => {
      event.stopPropagation();
      releaseHeldJob(job.heldId);
    };
    actions.append(runBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "small ghost danger fixed";
    deleteBtn.innerHTML = iconHTML("trash", "sm");
    deleteBtn.title = "Delete this held job without rendering it";
    deleteBtn.onclick = (event) => {
      event.stopPropagation();
      deleteHeldJob(job.heldId);
    };
    actions.append(deleteBtn);
  }

  top.append(actions);
  contentCol.append(top);

  if (isHeld) {
    const scheduleRow = document.createElement("div");
    scheduleRow.className = "held-schedule-row";
    const scheduleLabel = document.createElement("span");
    scheduleLabel.className = "stat";
    scheduleLabel.textContent = "Run at";
    const input = document.createElement("input");
    input.type = "datetime-local";
    input.className = "held-schedule-input";
    if (job.releaseAt) {
      const date = new Date(job.releaseAt);
      const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
      input.value = local.toISOString().slice(0, 16);
    }
    const save = document.createElement("button");
    save.className = "small ghost fixed";
    save.textContent = input.value ? "Update time" : "Schedule";
    save.onclick = (event) => {
      event.stopPropagation();
      scheduleHeldJob(job.heldId, input.value);
    };
    const clear = document.createElement("button");
    clear.className = "small ghost fixed";
    clear.textContent = "Keep held";
    clear.style.display = job.releaseAt ? "inline-flex" : "none";
    clear.onclick = (event) => {
      event.stopPropagation();
      scheduleHeldJob(job.heldId, "");
    };
    scheduleRow.append(scheduleLabel, input, save, clear);
    contentCol.append(scheduleRow);
  }

  if (isRunning) {
    const pRow = document.createElement("div");
    pRow.className = "job-progress-row";

    const pTrack = document.createElement("div");
    pTrack.className = "job-progress-track";
    const pFill = document.createElement("div");
    pFill.className = "job-progress-fill";
    const stage = getJobStageInfo(job);
    applyProgressFill(pFill, stage);
    pTrack.append(pFill);

    const pInfo = document.createElement("div");
    pInfo.className = "job-progress-info";
    pInfo.innerHTML = `<span class="job-step-text">${stageMarkup(stage)}</span><span class="job-stage-detail">Overall ${stage.overallPct}% · ${escapeHtml(stage.detail)}</span>`;

    pRow.append(pTrack, pInfo);
    contentCol.append(pRow);
  }

  if (job.prompt) {
    const promptExcerpt = document.createElement("div");
    promptExcerpt.className = "job-prompt-excerpt";
    promptExcerpt.textContent = job.prompt;
    promptExcerpt.title = "Click to inspect prompt and settings";
    promptExcerpt.onclick = () => openJobInspector(job);
    contentCol.append(promptExcerpt);
  }

  card.append(contentCol);
  return card;
}

function openQueueTracker() {
  state.queueTrackerOpen = true;
  $("queue-modal").classList.add("open");
  syncQueue();
}

function closeQueueTracker() {
  state.queueTrackerOpen = false;
  $("queue-modal").classList.remove("open");
  releaseWithin($("queue-modal"));
}

function openJobInspector(job) {
  state.inspectedJob = job;
  state.jobInspectorOpen = true;
  $("job-inspector-modal").classList.add("open");
  renderJobInspector(job);
}

function closeJobInspector() {
  state.jobInspectorOpen = false;
  state.inspectedJob = null;
  const vid = $("job-inspector-video");
  if (vid) {
    vid.pause();
    vid.removeAttribute("src");
    vid.style.display = "none";
  }
  $("job-inspector-modal").classList.remove("open");
}

function renderJobInspector(job) {
  if (!job) return;

  $("job-inspector-shot-name").textContent = `${job.shotName || "Source"} (${job.promptId ? job.promptId.slice(0, 8) : "—"}...)`;

  // Video Output Player
  const videoEl = $("job-inspector-video");
  if (videoEl) {
    if (job.url) {
      videoEl.src = job.url;
      videoEl.style.display = "block";
      videoEl.load();
    } else {
      videoEl.pause();
      videoEl.removeAttribute("src");
      videoEl.style.display = "none";
    }
  }

  // Status banner
  const banner = $("job-inspector-banner");
  const bannerTitle = $("job-inspector-status-title");
  const bannerSub = $("job-inspector-status-sub");
  if (banner && bannerTitle && bannerSub) {
    if (job.status === "running" || state.runningJob?.promptId === job.promptId) {
      banner.className = "job-status-banner running";
      bannerTitle.innerHTML = `${iconHTML("activity", "sm")} Status: Currently Rendering`;
      const elapsed = job.startTime ? (Date.now() - job.startTime) / 1000 : 0;
      const stage = getJobStageInfo(job);
      bannerSub.textContent = `Elapsed: ${formatDuration(elapsed)} · Overall ${stage.overallPct}% · ${stage.label}${stage.metric ? ` · ${stage.metric}` : ""}`;
    } else if (job.status === "completed") {
      banner.className = "job-status-banner";
      bannerTitle.innerHTML = `${iconHTML("check", "sm")} Status: Completed`;
      bannerSub.textContent = job.duration ? `Finished in ${job.duration.toFixed(1)}s` : "Render finished successfully";
    } else if (job.status === "held" || job.status === "scheduled") {
      banner.className = "job-status-banner";
      bannerTitle.innerHTML = `${iconHTML("clock", "sm")} Status: ${job.status === "scheduled" ? "Scheduled" : "Held"}`;
      bannerSub.textContent = job.releaseAt
        ? `Will enter ComfyUI at ${new Date(job.releaseAt).toLocaleString()}`
        : "Stored outside ComfyUI until you release or schedule it";
    } else if (job.status === "failed" || job.status === "interrupted" || job.status === "error") {
      banner.className = "job-status-banner";
      bannerTitle.innerHTML = `${iconHTML("alert-triangle", "sm")} Status: ${job.status === "interrupted" ? "Interrupted" : "Failed"}`;
      bannerSub.textContent = job.error || "Generation stopped";
    } else {
      banner.className = "job-status-banner";
      bannerTitle.innerHTML = `${iconHTML("clock", "sm")} Status: Queued`;
      bannerSub.textContent = "Waiting in ComfyUI queue to begin rendering";
    }
  }

  // Properties grid
  const props = $("job-inspector-props");
  if (props) {
    const rows = [
      ["Source", escapeHtml(job.shotName || "—")],
      ["Mode", escapeHtml(job.mode || "Render")],
      ["Resolution", `${job.width} × ${job.height} (${calcAspectRatio(job.width, job.height)})`],
      ["Length", `${job.seconds}s (${job.frames} frames)`],
      ["Seed", job.seed != null ? `#${job.seed}` : "Random / Auto"],
      ["Steps", `${job.steps || 20} steps`],
      ["Output File", job.filename ? `<a href="${job.url}" target="_blank" download="${escapeHtml(job.filename)}" style="color:var(--accent);text-decoration:underline">${escapeHtml(job.filename)}</a>` : "None yet"],
      ["Latent Path", job.latent_path ? escapeHtml(job.latent_path) : "None (Clean Video Path)"],
      ["Parent Take", job.parent_take ? escapeHtml(job.parent_take) : "None (Root Clip)"],
      ["Project", escapeHtml(job.project || state.project?.name || "—")],
      ["Prompt ID", escapeHtml(job.promptId || "—")],
      ["Queued Time", (job.queuedAt || job.completedAt) ? new Date(job.queuedAt || job.completedAt).toLocaleTimeString() : "—"],
    ];
    props.innerHTML = rows
      .map(([k, v]) => `<div class="props-key">${k}</div><div class="props-val">${v}</div>`)
      .join("");
  }

  // Prompt box
  const promptBox = $("job-inspector-prompt-box");
  const promptStats = $("job-prompt-stats");
  const promptText = job.prompt || "";
  if (promptBox) {
    const drawPrompt = () => {
      showRecordedPrompt(
        promptBox, promptText, "(No composed prompt text available for this job)",
      );
      setPromptSectionToggle($("job-prompt-sections"), promptText, drawPrompt);
    };
    drawPrompt();
  }
  if (promptStats) {
    const chars = promptText.length;
    const words = promptText.trim() ? promptText.trim().split(/\s+/).length : 0;
    promptStats.textContent = `${chars} chars · ${words} words`;
  }

  // Action buttons
  const copyBtn = $("job-copy-prompt-btn");
  const copyTopBtn = $("job-copy-prompt-top-btn");
  const copyAction = () => {
    if (!promptText) return;
    navigator.clipboard.writeText(promptText);
    toast("Composed prompt copied to clipboard!");
  };
  if (copyBtn) copyBtn.onclick = copyAction;
  if (copyTopBtn) copyTopBtn.onclick = copyAction;

  const restoreBtn = $("job-restore-prompt-btn");
  if (restoreBtn) {
    if (promptText && currentSegment()) {
      restoreBtn.style.display = "inline-flex";
      restoreBtn.onclick = guard(async () => {
        const segment = currentSegment();
        if (!segment) return;
        await restorePromptToSegment(segment, promptText);
        fillForm();
        markDirty();
        await runPreview();
        toast(`Restored prompt into ${currentSource()?.name ?? "this source"}`);
        closeJobInspector();
      });
    } else {
      restoreBtn.style.display = "none";
    }
  }

  const openShotBtn = $("job-open-shot-btn");
  if (openShotBtn) {
    if (job.shot) {
      openShotBtn.style.display = "inline-flex";
      openShotBtn.onclick = () => {
        closeJobInspector();
        closeQueueTracker();
        openEditor(job.shot);
      };
    } else {
      openShotBtn.style.display = "none";
    }
  }

  const cancelBtn = $("job-cancel-btn");
  if (cancelBtn) {
    if (job.status === "running" || state.runningJob?.promptId === job.promptId) {
      cancelBtn.style.display = "inline-flex";
      cancelBtn.innerHTML = label("stop-circle", "Interrupt Active Render", "sm");
      cancelBtn.onclick = () => {
        interruptRunningJob();
        closeJobInspector();
      };
    } else if (job.heldId) {
      cancelBtn.style.display = "inline-flex";
      cancelBtn.innerHTML = label("trash", "Delete Held Job", "sm");
      cancelBtn.onclick = () => {
        deleteHeldJob(job.heldId);
        closeJobInspector();
      };
    } else if (job.status === "queued" || state.pendingJobs.some((p) => p.promptId === job.promptId)) {
      cancelBtn.style.display = "inline-flex";
      cancelBtn.innerHTML = label("trash", "Remove from Queue", "sm");
      cancelBtn.onclick = () => {
        deleteQueuedJob(job.promptId);
        closeJobInspector();
      };
    } else {
      cancelBtn.style.display = "none";
    }
  }
}

async function interruptRunningJob() {
  try {
    await call("/interrupt", { method: "POST" });
    toast("Interrupt signal dispatched to ComfyUI.");
    syncQueue();
  } catch (err) {
    toast(`Could not interrupt: ${err.message}`, true);
  }
}

async function deleteQueuedJob(promptId) {
  try {
    await call("/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delete: [promptId] }),
    });
    toast("Job removed from queue.");
    state.jobs.delete(promptId);
    syncQueue();
  } catch (err) {
    toast(`Could not delete queued job: ${err.message}`, true);
  }
}

async function clearPendingQueue() {
  try {
    await call("/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clear: true }),
    });
    toast("Pending render queue cleared.");
    syncQueue();
  } catch (err) {
    toast(`Could not clear queue: ${err.message}`, true);
  }
}

// Move a job that is already waiting in ComfyUI's queue onto the shelf. The
// cure for the render bar's hold toggle being a mode you can forget: press Fine
// by mistake at 11pm and this is one tap, instead of a wasted overnight slot.
// It is also the mirror of `releaseHeldJob`, which has always gone the other
// way — the queue and the shelf are now both round trips.
//
// Held first, deleted second. The other order risks a job existing in neither
// place if the hold is refused; this order risks it existing in both, which is
// visible and fixable.
async function shelveQueuedJob(job, { quiet = false } = {}) {
  const payload = heldPayloadFromJob(job);
  if (!payload) {
    if (!quiet) toast("That job's workflow is not available to shelve.", true);
    return false;
  }
  try {
    await post(`${API}/render/hold`, {
      payload,
      summary: payload.extra_data.video_prompter,
    });
  } catch (error) {
    if (!quiet) toast(`Could not shelve: ${error.message}`, true);
    return false;
  }
  try {
    await call("/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delete: [job.promptId] }),
    });
    state.jobs.delete(job.promptId);
  } catch (error) {
    if (!quiet) {
      toast(`Shelved, but it is still in ComfyUI's queue — remove it there: ${error.message}`,
            true);
    }
    await syncQueue();
    return false;
  }
  if (!quiet) {
    toast(`Shelved${job.seed != null ? ` seed #${job.seed}` : ""} — it will not render until you release it.`);
    await syncQueue();
  }
  return true;
}

/** Every job still waiting, in one press. A sweep of 8 is one tap, not eight. */
async function shelveAllPending() {
  const pending = [...state.pendingJobs];
  if (!pending.length) return;
  let done = 0;
  for (const job of pending) {
    if (await shelveQueuedJob(job, { quiet: true })) done++;
  }
  toast(done === pending.length
    ? `Shelved ${done} queued job${done === 1 ? "" : "s"}.`
    : `Shelved ${done} of ${pending.length} — the rest are still queued.`,
    done !== pending.length);
  await syncQueue();
}

// The payload the shelf stores, rebuilt for a job that is already in flight.
// The original `extra_data` is reused whenever it is known, because it carries
// fields no job record has — which cast member a keyframe was rendered for, the
// style it used — and losing those would misfile the result on release.
//
// A fresh `prompt_id` is minted either way: the old one names a job ComfyUI has
// been told to forget, so reusing it would collide with its own record.
function heldPayloadFromJob(job) {
  if (!job?.workflow || typeof job.workflow !== "object") return null;
  const known = job.extraData?.video_prompter;
  const summary = known && typeof known === "object" ? { ...known } : {
    project: job.project || state.project?.name,
    shot: job.shot || "",
    shot_name: job.shotName || "",
    mode: job.mode || "Render",
    frames: job.frames,
    seconds: job.seconds,
    seed: job.seed,
    prompt: job.prompt || "",
    width: job.width,
    height: job.height,
    quality: job.quality || undefined,
    steps: job.steps,
    latent_path: job.latent_path || undefined,
    parent_take: job.parent_take || undefined,
    context_length: job.context_length || undefined,
  };
  return {
    prompt: job.workflow,
    client_id: state.clientId,
    prompt_id: uuid(),
    extra_data: { ...(job.extraData || {}), video_prompter: summary },
  };
}

async function releaseHeldJob(heldId) {
  try {
    const result = await post(`${API}/render/held/release`, { id: heldId });
    toast(`Released into ComfyUI queue${result.prompt_id ? ` · ${result.prompt_id.slice(0, 8)}` : ""}`);
    await syncQueue();
  } catch (error) {
    toast(error.message, true);
    await syncQueue();
  }
}

async function scheduleHeldJob(heldId, localValue) {
  if (!localValue) {
    try {
      await post(`${API}/render/held/schedule`, { id: heldId, release_at: null });
      toast("Job will stay held until you release it.");
      await syncQueue();
    } catch (error) {
      toast(error.message, true);
    }
    return;
  }
  const release = new Date(localValue);
  if (Number.isNaN(release.getTime())) {
    toast("Choose a valid run date and time.", true);
    return;
  }
  try {
    await post(`${API}/render/held/schedule`, { id: heldId, release_at: release.toISOString() });
    toast(`Scheduled for ${release.toLocaleString()}`);
    await syncQueue();
  } catch (error) {
    toast(error.message, true);
  }
}

async function deleteHeldJob(heldId) {
  try {
    await post(`${API}/render/held/delete`, { id: heldId });
    toast("Held job deleted.");
    await syncQueue();
  } catch (error) {
    toast(error.message, true);
  }
}

/** The next time today's clock reads `clock` ("01:00"): tonight, else tomorrow.
 *
 * A shelf is scheduled by time of day, not by date — "one AM" means the one
 * coming up, and asking for a full calendar date to say that is friction.
 */
function nextOccurrenceOf(clock) {
  const parts = /^(\d{1,2}):(\d{2})$/.exec(String(clock || "").trim());
  if (!parts) return null;
  const hours = Number(parts[1]);
  const minutes = Number(parts[2]);
  if (hours > 23 || minutes > 59) return null;
  const when = new Date();
  when.setHours(hours, minutes, 0, 0);
  if (when.getTime() <= Date.now()) when.setDate(when.getDate() + 1);
  return when;
}

async function scheduleAllHeldJobs() {
  const count = state.heldJobs.length;
  if (!count) return;
  const release = nextOccurrenceOf($("queue-held-bulk-time")?.value);
  if (!release) {
    toast("Choose a time of day to run the held jobs.", true);
    return;
  }
  if (!window.confirm(`Run all ${count} held jobs starting ${release.toLocaleString()}?`)) return;
  try {
    const result = await post(`${API}/render/held/schedule_all`, { release_at: release.toISOString() });
    const done = result.count ?? 0;
    toast(`${done} job${done === 1 ? "" : "s"} scheduled for ${release.toLocaleString()}.`
      + (done < count ? " Jobs already being dispatched were left alone." : ""));
    await syncQueue();
  } catch (error) {
    toast(error.message, true);
  }
}

async function unscheduleAllHeldJobs() {
  if (!state.heldJobs.length) return;
  try {
    const result = await post(`${API}/render/held/schedule_all`, { release_at: null });
    toast(`${result.count ?? 0} job(s) back on hold until you release them.`);
    await syncQueue();
  } catch (error) {
    toast(error.message, true);
  }
}

async function releaseAllHeldJobs() {
  const count = state.heldJobs.length;
  if (!count) return;
  if (!window.confirm(`Release all ${count} held jobs into ComfyUI now?`)) return;
  try {
    const result = await post(`${API}/render/held/release_all`, {});
    const failures = result.errors?.length || 0;
    toast(failures
      ? `Released ${result.released?.length || 0}; ${failures} could not be queued.`
      : `Released ${result.released?.length || 0} held jobs into ComfyUI.`, failures > 0);
    await syncQueue();
  } catch (error) {
    toast(error.message, true);
  }
}

function clearJobHistory() {
  state.jobHistory = [];
  renderQueueModal();
  toast("Finished job history cleared.");
}

// -- tracking -------------------------------------------------------------

function openSocket() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  let socket;
  try {
    socket = new WebSocket(`${scheme}//${location.host}/ws?clientId=${state.clientId}`);
  } catch {
    return;
  }

  socket.onmessage = (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return; // Binary previews arrive on the same socket.
    }
    handleEvent(message);
  };

  socket.onclose = () => setTimeout(openSocket, 3000);
  socket.onerror = () => socket.close();
}

function handleEvent(message) {
  if (message?.type === "video_prompter.changed") {
    onProjectChanged(message.data ?? {});
    return;
  }
  if (message?.type === "video_prompter.held_changed") {
    syncQueue();
    return;
  }

  const data = message?.data ?? {};
  const promptId = data.prompt_id;

  if (message.type === "status") {
    syncQueue();
    return;
  }

  if (message.type === "execution_start") {
    if (promptId) {
      if (state.jobs.has(promptId)) {
        const job = state.jobs.get(promptId);
        job.status = "running";
        job.startTime = Date.now();
        state.runningJob = job;
      } else if (!state.runningJob) {
        state.runningJob = { promptId, status: "running", startTime: Date.now(), progress: { value: 0, max: 20 } };
      }
    }
    syncQueue();
    return;
  }

  if (message.type === "execution_cached") {
    const target = (promptId && state.jobs.has(promptId))
      ? state.jobs.get(promptId)
      : state.runningJob;
    if (target) target.cachedNodes = Array.isArray(data.nodes) ? data.nodes.map(String) : [];
    return;
  }

  // ComfyUI 0.30+ reports authoritative per-node progress in one message.
  // `display_node_id` maps dynamic/subgraph nodes back to IDs present in the
  // submitted API workflow, which is what the stage classifier understands.
  if (message.type === "progress_state") {
    if (applyProgressState(data)) updateRunningTimeAndProgressUI();
    return;
  }

  if (message.type === "progress") {
    const target = (state.runningJob && (!promptId || promptId === state.runningJob.promptId))
      ? state.runningJob
      : (promptId && state.jobs.has(promptId) ? state.jobs.get(promptId) : state.runningJob);

    if (target) {
      const nodeId = resolveWorkflowNodeId(target.workflow, data.display_node, data.node);
      target.progress = { value: data.value, max: data.max, node: nodeId, display_node: data.display_node };
      if (nodeId) target.currentNode = nodeId;
      updateRunningTimeAndProgressUI();
    } else if (data.value != null && data.max != null) {
      state.runningJob = {
        promptId: promptId || "active",
        status: "running",
        startTime: Date.now(),
        progress: { value: data.value, max: data.max, node: data.node },
        currentNode: data.node,
      };
      updateRunningTimeAndProgressUI();
    }
    return;
  }

  if (message.type === "executing") {
    if (data.node !== null) {
      const target = (state.runningJob && (!promptId || promptId === state.runningJob.promptId))
        ? state.runningJob
        : (promptId && state.jobs.has(promptId) ? state.jobs.get(promptId) : state.runningJob);

      if (target) {
        const nodeId = resolveWorkflowNodeId(target.workflow, data.display_node, data.node);
        if (nodeId !== target.currentNode) {
          // Do not show the previous node's completed progress under this new
          // stage while waiting for its first progress_state update.
          target.progress = null;
        }
        target.currentNode = nodeId;
        updateRunningTimeAndProgressUI();
      }
    }
    return;
  }

  if (message.type === "execution_success") {
    if (promptId) {
      const job = state.jobs.get(promptId) || state.runningJob || {};
      const duration = job.startTime ? (Date.now() - job.startTime) / 1000 : null;
      state.jobHistory.unshift({
        ...job,
        promptId,
        status: "completed",
        duration,
        completedAt: Date.now(),
      });
      if (state.jobHistory.length > 100) state.jobHistory.pop();
      collect(promptId);
    }
    state.runningJob = null;
    syncQueue();
    return;
  }

  if (message.type === "execution_error" || message.type === "execution_interrupted") {
    if (promptId) {
      const job = state.jobs.get(promptId) || state.runningJob || {};
      const isInterrupted = message.type === "execution_interrupted";
      state.jobHistory.unshift({
        ...job,
        promptId,
        status: isInterrupted ? "interrupted" : "failed",
        error: data.exception_message || (isInterrupted ? "Execution interrupted" : "Execution error"),
        completedAt: Date.now(),
      });
      if (state.jobHistory.length > 100) state.jobHistory.pop();
      toast(data.exception_message || (isInterrupted ? "Render interrupted." : "Render failed."), true);
    }
    state.runningJob = null;
    syncQueue();
    return;
  }
}

// A change made outside this tab — an agent driving the tool, or another
// editor. Refresh the open project so its sources and prompts update live.
async function onProjectChanged(data) {
  if (!data || data.origin === state.clientId) return;
  if (!state.project || data.project !== state.project.name) return;

  if (state.dirty) {
    toast("This project changed elsewhere. Save or reopen to merge.", true);
    return;
  }

  try {
    const epoch = state.projectEpoch;
    const revision = state.editRevision;
    const name = encodeURIComponent(state.project.name);
    const refreshed = (await call(`${API}/project/load?name=${name}`)).project;
    if (epoch !== state.projectEpoch || revision !== state.editRevision || state.dirty) return;
    if (!refreshed) return;
    state.project = refreshed;
    restoreSelection();
    await reloadSequence();
    if (state.editorOpen || state.inspectorOpen) fillForm();
    refresh();
    toast("Updated from an external change");
  } catch (error) {
    toast(`Could not sync external change: ${error.message}`, true);
  }
}

async function collect(promptId) {
  const job = state.jobs.get(promptId);
  const genTime = job?.startTime ? Math.round((Date.now() - job.startTime) / 100) / 10 : undefined;
  const wasStill = isStillJob(job);
  state.jobs.delete(promptId);
  showJobs();
  if (!state.project) return;

  // A finished image is a library entry, not a take: nothing about the edit
  // changed, so none of the sequence machinery runs for one.
  if (wasStill) {
    try {
      const history = await (await fetch(`/history/${promptId}`)).json();
      await post(`${API}/stills/record`, { project: state.project.name, history });
      await reloadStills();
      renderStudioGrid();
      renderBin();
    } catch (error) {
      toast(`Generated, but could not file it: ${error.message}`, true);
    }
    return;
  }

  try {
    const history = await (await fetch(`/history/${promptId}`)).json();
    await post(`${API}/gallery/record`, {
      project: state.project.name,
      history,
      generation_times: genTime ? { [promptId]: genTime } : undefined,
    });
    await reloadSequence();
    renderCandidates();
  } catch (error) {
    toast(`Finished, but could not record it: ${error.message}`, true);
  }
}

// Renders that finished while this page was closed are still in ComfyUI's history.
async function reconcile() {
  if (!state.project) return;
  try {
    const history = await (await fetch("/history?max_items=200")).json();
    const result = await post(`${API}/gallery/record`, {
      project: state.project.name,
      history,
    });
    if (result.count) toast(`Picked up ${result.count} finished render(s)`);
    // Images generated while this tab was closed — by an agent, or by another
    // editor — belong in the library for the same reason.
    await post(`${API}/stills/record`, { project: state.project.name, history });
  } catch {
    // No ComfyUI behind us, or it has no history yet.
  }
}

function showJobs() {
  const container = $("in-flight");
  if (!container) return;
  container.textContent = "";

  const source = currentSource();
  if (!source) return;

  // Show the running job when it belongs to this source
  const mine = (job) => job.shot === source.id || job.shot === source.name;
  if (state.runningJob && mine(state.runningJob)) {
    const card = createJobCard(state.runningJob, { isRunning: true });
    container.append(card);
  }

  const matchingPending = state.pendingJobs.filter(mine);
  matchingPending.forEach((job, idx) => {
    const card = createJobCard(job, { isPending: true, queueIndex: idx + 1 });
    container.append(card);
  });
}

// -- the source editor: author, generate, choose --------------------------

// One form and one gallery, relocated between the workspace and expanded
// editor. Keeping their DOM nodes preserves drafts, handlers and media state.
function updateReviewContext() {
  const source = currentSource();
  const count = source ? candidatesAtDepth(source, state.depth).length : 0;
  $("review-count").textContent = String(count);
  $("review-source").textContent = source
    ? `${displayName(source.name)} · Generation ${state.depth + 1}`
    : "Select a source to review its takes";
  $("review-takes").disabled = !source;
  $("editor-expand").disabled = !source;
}

function reviewHeightLimits() {
  // Leave room for the header, monitor and compact sequence. Short displays
  // scroll the gallery instead of letting it push the monitor off screen.
  return { min: 180, max: Math.max(180, Math.min(650, window.innerHeight - 550)) };
}

function setReviewHeight(height, persist = false) {
  const { min, max } = reviewHeightLimits();
  const value = Math.round(Math.max(min, Math.min(max, height)));
  $("review-content").style.height = `${value}px`;
  const grip = $("review-resize");
  grip.setAttribute("aria-valuemin", String(min));
  grip.setAttribute("aria-valuemax", String(max));
  grip.setAttribute("aria-valuenow", String(value));
  if (persist) localStorage.setItem("vp:reviewHeight", String(value));
}

function setReviewOpen(open) {
  if (open && !currentSource()) return;
  state.reviewOpen = open;
  document.body.classList.toggle("review-open", open);
  $("review-content").hidden = !open;
  $("review-resize").hidden = !open;
  $("review-close").hidden = !open;
  $("review-timeline").classList.toggle("active", !open);
  $("review-takes").classList.toggle("active", open);
  $("review-timeline").setAttribute("aria-pressed", String(!open));
  $("review-takes").setAttribute("aria-pressed", String(open));
  if (open) {
    const saved = Number(localStorage.getItem("vp:reviewHeight"));
    setReviewHeight(saved > 0 ? saved : 360);
    setTakesView("takes");
    renderCandidates();
    showJobs();
  } else {
    stopSync();
    ensureTimeline();
    hushWithin($("review-pane"));
  }
  renderTimeline();
}

function wireWorkspaceLayout() {
  $("review-content").append($("review-pane"));
  $("editor-expand").onclick = () => openEditor(state.selectedSource);
  $("review-timeline").onclick = () => setReviewOpen(false);
  $("review-takes").onclick = () => setReviewOpen(true);
  $("review-close").onclick = () => {
    setReviewOpen(false);
    $("review-takes").focus();
  };
  $("monitor-return").onclick = ensureTimeline;
  const grip = $("review-resize");
  grip.onpointerdown = (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    grip.setPointerCapture(event.pointerId);
    const start = event.clientY;
    const height = $("review-content").getBoundingClientRect().height;
    grip.onpointermove = (move) => setReviewHeight(height + start - move.clientY);
    const finish = () => {
      setReviewHeight($("review-content").getBoundingClientRect().height, true);
      grip.onpointermove = null;
      grip.onpointerup = null;
      grip.onpointercancel = null;
    };
    grip.onpointerup = finish;
    grip.onpointercancel = finish;
  };
  grip.onkeydown = (event) => {
    if (!["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    const { min, max } = reviewHeightLimits();
    const height = $("review-content").getBoundingClientRect().height;
    setReviewHeight(event.key === "Home" ? min : event.key === "End" ? max
      : height + (event.key === "ArrowUp" ? 30 : -30), true);
  };
  window.addEventListener("resize", () => {
    if (state.reviewOpen) setReviewHeight(parseInt($("review-content").style.height, 10) || 360);
  });
}

let editorReturnFocus = null;

function openEditor(sourceId, clipId = null) {
  if (clipId) select(clipId);
  else if (sourceId) selectSource(sourceId);
  if (!currentSource()) return;
  state.editorOpen = true;
  editorReturnFocus = document.activeElement;
  $("editor").classList.add("open");
  $("editor").setAttribute("role", "dialog");
  $("editor").setAttribute("aria-modal", "true");
  document.querySelector("#editor .panes").append($("review-pane"));
  fillForm();
  // The tree and the inspected take belong to the source that was open, so
  // opening another one starts back on its own takes rather than showing the
  // previous source's branches or prompt. Dropped before the takes are drawn,
  // or they would be marked up for a take that is no longer among them.
  clearTakeInspector();
  // The new-only chip belongs to the source it was pressed on. Carrying it
  // into the next source would open that one already narrowed, on a question
  // nobody asked there.
  state.newOnly = false;
  renderCandidates();
  setTakesView("takes");
  showJobs();
  $("editor-close").focus();
}

function closeEditor() {
  stopSync();
  // The member sheet belongs to the Cast tab, so it cannot outlive the editor
  // the tab is in.
  closeCastDetail();
  // Hand the monitor back before the sheet goes away, or the stage is left
  // inside a hidden dialog with nothing on screen playing.
  ensureTimeline();
  if (state.editorOpen || state.inspectorOpen) readForm();
  state.editorOpen = false;
  $("editor").classList.remove("open", "scene-focus");
  $("scene-focus").querySelector("span").textContent = "Focus";
  $("editor").setAttribute("role", "region");
  $("editor").removeAttribute("aria-modal");
  $("review-content").append($("review-pane"));
  dockPreview(false);
  if (!state.reviewOpen) releaseWithin($("review-pane"));
  // A take left playing is still heard once the sheet is down, so the sheet
  // takes its sound with it -- and its players, which it has no use for while
  // it is down and the bin behind it does.
  releaseWithin($("editor"));
  state.compare.clear();
  if (state.reviewOpen) renderCandidates();
  if (editorReturnFocus?.focus) editorReturnFocus.focus();
  editorReturnFocus = null;
}

// The takes on offer for the generation being authored.
//
// At depth 0 that is every take that starts a chain. Deeper, it is only the
// children of whatever the path stands on at the depth above — so a take
// belonging to a branch you are not on is not drawn at all, and the parent /
// child mismatch the old editor kept displaying has nowhere to appear.
function candidatesAtDepth(source, depth) {
  const mine = takesFor(source.id).filter((take) => !take.discarded);
  if (depth <= 0) {
    return mine.filter((take) => {
      if (take.refined_from) {
        const origin = takeById(take.refined_from);
        if (origin) return !origin.parent_take || !takeById(origin.parent_take);
      }
      return !take.parent_take || !takeById(take.parent_take);
    });
  }
  const parent = takeAtDepth(source.id, depth - 1);
  if (!parent) return [];
  // A continuation can be filed against another source when the chain was made
  // before sources existed, so children are found across the whole project.
  return state.takes.filter((take) => {
    if (take.discarded) return false;
    if (take.refined_from) {
      const origin = takeById(take.refined_from);
      if (origin) return origin.parent_take === parent.id;
    }
    return take.parent_take === parent.id;
  });
}

function renderCandidates() {
  // Any open card menu is anchored to a button in the grid about to be
  // rebuilt. The grid redraws on its own — a render landing, a rating saved —
  // so without this a menu can be left floating over the page pointing at a
  // card that no longer exists. The filter menus above the grid are spared:
  // ticking one is what triggers this redraw in the first place.
  closeCardMenu({ volatileOnly: true });
  const source = currentSource();
  updateReviewContext();
  const body = $("candidates");
  // The grid is about to be thrown away. A card playing when that happens is
  // detached rather than stopped, and a detached <video> keeps its sound -- and
  // its player, until the collector comes round.
  releaseWithin(body);
  if (!source) { body.textContent = ""; return; }
  const segment = currentSegment();
  renderGenActions(segment);

  const depth = state.depth;
  const offered = candidatesAtDepth(source, depth);
  const chosen = pathOf(source.id)[depth] ?? null;
  const parent = depth > 0 ? takeAtDepth(source.id, depth - 1) : null;
  renderTakeFilters();
  renderNewFilter();

  // The take the edit plays survives the filters whatever it was rendered at
  // and however it is rated — hiding what is currently on the timeline would
  // misreport the edit. It is dimmed instead, so it is plainly the exception
  // rather than a match.
  const kept = offered.filter((take) => passesFilters(take) || take.id === chosen);
  // The new-only chip is a stricter question — "what came back that I have not
  // watched" — so nothing escapes it, not even the take in the edit.
  const visible = state.newOnly ? kept.filter(isNewTake) : kept;
  const hidden = offered.length - kept.length;
  const notNew = kept.length - visible.length;

  $("modal-hint").textContent = offered.length
    ? `${visible.length} candidate(s) at generation ${depth + 1}` +
      (hidden ? ` · ${hidden} hidden by filters` : "") +
      (notNew ? ` · ${notNew} already seen` : "") +
      ` · select up to ${MAX_COMPARE} to compare`
    : "nothing rendered yet";
  $("modal-compare").disabled = state.compare.size < 2;
  $("modal-compare").innerHTML = label("columns", `Compare${state.compare.size ? ` (${state.compare.size})` : ""}`, "sm");

  body.textContent = "";

  if (depth > 0) {
    const banner = document.createElement("div");
    banner.className = "branch-banner";
    if (parent) {
      const name = parent.seed != null ? `Seed #${parent.seed}` : parent.filename;
      banner.innerHTML =
        `<div class="title">${iconHTML("link")} Continuing generation <b>${depth}</b></div>` +
        `<div class="parent-info">From ${escapeHtml(name)} ★${parent.rating || 0}</div>` +
        `<span class="spacer"></span>`;
      const tier = qualitySelect(extendQualityFor(source, depth), parent);
      tier.className = "small";
      const again = document.createElement("button");
      again.className = "small primary";
      again.innerHTML = label("rotate", "Extend again", "sm");
      again.title = "Render another continuation from the same parent take";
      again.onclick = guard(async () => {
        readForm();
        const text = composedText() || `Continuation of ${source.name}`;
        await runExtend(parent, source, text, { depth, quality: chosenQuality(tier) });
      });
      banner.append(tier, again);
    } else {
      banner.innerHTML =
        `<div class="title" style="color:var(--warning)">${iconHTML("alert-triangle")} Generation ${depth} has no take</div>` +
        `<div class="parent-info">Choose a take at generation ${depth} before continuing from it.</div>`;
    }
    body.append(banner);
  }

  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = notNew
      ? `Every take at this generation has been opened already —
         switch <b>new</b> off above to see them.`
      : hidden
      ? `All ${hidden} take(s) here are filtered out by the quality and rating
         filters — widen one of them above to see them.`
      : depth > 0
        ? `Nothing continues that take yet — <b>Extend again</b> above renders one.`
        : `Render this source, or sweep a few seeds, and the candidates land here.`;
    body.append(empty);
    return;
  }

  const grid = document.createElement("div");
  grid.className = "grid";
  const sharedName = sharedNamePrefix(visible);
  for (const take of visible) grid.append(candidateCard(take, source, chosen, sharedName));
  body.append(grid);

  // Takes filed against this source that are not on offer here belong to other
  // depths or other branches. They stay visible and playable — dead branches
  // are never hidden — but out of the way.
  const shown = new Set(offered.map((take) => take.id));
  const rest = takesFor(source.id)
    .filter((take) => !shown.has(take.id))
    .filter(passesFilters)
    .filter((take) => !state.newOnly || isNewTake(take));
  if (rest.length) {
    const fold = document.createElement("details");
    fold.style.marginTop = "16px";
    const summary = document.createElement("summary");
    summary.className = "branch-section-title";
    summary.style.cursor = "pointer";
    summary.innerHTML = `${iconHTML("layers", "sm")} Other takes on this source (${rest.length})`;
    fold.append(summary);
    const other = document.createElement("div");
    other.className = "grid";
    other.style.marginTop = "8px";
    const restShared = sharedNamePrefix(rest);
    for (const take of rest) other.append(candidateCard(take, source, chosen, restShared));
    fold.append(other);
    body.append(fold);
  }

  // The cards are new objects, so the mark on the one being inspected has to
  // be put back — and the inspect view walks this grid's order, which a
  // re-render may have changed under it.
  markInspectedCard();
  if (state.inspectedTake) renderInspectBar(state.inspectedTake);
}

function formatGenTime(seconds) {
  if (!seconds || seconds <= 0) return "0s";
  if (seconds < 60) return `${Math.round(seconds * 10) / 10}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

function getAspectRatio(w, h) {
  if (!w || !h) return "";
  const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
  const g = gcd(w, h);
  const rw = w / g;
  const rh = h / g;
  if ((rw === 16 && rh === 9) || (rw === 9 && rh === 16) || (rw === 4 && rh === 3) || (rw === 1 && rh === 1) || (rw === 21 && rh === 9)) {
    return `${rw}:${rh}`;
  }
  const ratio = (w / h).toFixed(2);
  return `${ratio}:1`;
}

/** Leave the inspect view. The takes it was reached from come back. */
function closeTakeInspector() {
  setTakesView("takes");
}

/**
 * Forget the inspected take and fall back to the pane's empty state. Called
 * when the editor moves to another source, whose takes this one is not among.
 */
function clearTakeInspector() {
  state.inspectedTake = null;
  state.inspectedSource = null;
  state.inspectRefresh = null;
  $("inspect-empty").hidden = false;
  $("inspect-bar").hidden = true;
  $("inspect-top").hidden = true;
  $("inspect-prompt").hidden = true;
  const vid = $("take-inspector-video");
  if (vid) {
    vid.pause();
    vid.removeAttribute("src");
    vid.load();
  }
}

/**
 * The takes the inspect view can walk, in the order the takes grid is drawing
 * them. Read off the cards rather than recomputed so the two can never
 * disagree — a filtered-out or folded-away take is not stepped onto.
 */
function inspectSiblings() {
  return [...document.querySelectorAll("#candidates .candidate")]
    .map((card) => card.dataset.takeId)
    .filter(Boolean);
}

/** Move the inspect view onto the take `delta` places along the takes grid. */
function inspectStep(delta) {
  const ids = inspectSiblings();
  const at = ids.indexOf(state.inspectedTake?.id);
  if (at < 0) return;
  const next = takeById(ids[at + delta]);
  if (next) openTakeInspector(next, state.inspectedSource);
}

/** Show in the takes grid which card the inspect view is reading, so flipping
    back to Takes lands on the take that was being read rather than nothing. */
function markInspectedCard() {
  const id = state.inspectedTake?.id;
  for (const card of document.querySelectorAll("#candidates .candidate")) {
    card.classList.toggle("inspecting", !!id && card.dataset.takeId === id);
  }
}

/** The one-line identity of the inspected take, and where it sits in the grid. */
function renderInspectBar(take) {
  const tier = tierOf(take);
  const facts = [];
  if (tier) facts.push(`<b style="color:var(--tier-${tier})">${TIER_LABELS[tier]}</b>`);
  if (take.width && take.height) facts.push(`${take.width}×${take.height}${tier === "refined" ? ` (${outputSizeLabel(take)})` : ""}`);
  if (take.seconds) facts.push(`${take.seconds}s`);
  if (take.seed != null) facts.push(`seed ${take.seed}`);
  const schedule = stepsPhrase(take);
  if (schedule) facts.push(schedule);
  $("inspect-facts").innerHTML = facts.join(" · ") || "—";

  const ids = inspectSiblings();
  const at = ids.indexOf(take.id);
  $("inspect-pos").textContent = at >= 0 ? `${at + 1} / ${ids.length}` : "";
  $("inspect-prev").disabled = at <= 0;
  $("inspect-next").disabled = at < 0 || at >= ids.length - 1;
  markInspectedCard();
}

/**
 * Point the player at the inspected take, but only once it is on screen —
 * folded away it would fetch a video per ‹ › press that nobody is watching.
 */
function syncInspectVideo() {
  const vid = $("take-inspector-video");
  const want = state.inspectedTake?.url;
  if (!vid) return;
  if (!want || !state.inspectDetail) {
    vid.pause();
    return;
  }
  if (vid.getAttribute("src") === want) return;
  vid.src = want;
  vid.load();
}

/** Fold the take itself in or out. The prompt keeps whatever is left. */
function setInspectDetail(open) {
  state.inspectDetail = open;
  $("inspect-top").hidden = !open || !state.inspectedTake;
  const button = $("inspect-detail");
  button.classList.toggle("active", open);
  button.setAttribute("aria-expanded", String(open));
  syncInspectVideo();
}

function openTakeInspector(take, shot) {
  state.inspectedTake = take;
  state.inspectedSource = shot;
  // Reading a take's prompt is looking at it. Marked here as well as on the
  // card because ‹ › walk the grid without anyone clicking the cards they
  // walk onto.
  markSeen(take.id);
  // Dropped before the view flips, or switching in would redraw the take that
  // was inspected last only to overwrite it a moment later.
  state.inspectRefresh = null;
  $("inspect-empty").hidden = true;
  $("inspect-bar").hidden = false;
  $("inspect-prompt").hidden = false;
  setInspectDetail(state.inspectDetail);
  setTakesView("inspect");
  renderInspectBar(take);

  // The source is named in the editor's own title bar, so the line here only
  // has to say which of its takes this is.
  $("take-inspector-filename").textContent = take.filename;
  $("take-inspector-filename").title = `${take.filename} · ${shot.name || "Source"}`;

  const propsEl = $("take-inspector-props");
  propsEl.innerHTML = "";
  const addProp = (key, val) => {
    const k = document.createElement("span");
    k.className = "props-key";
    k.textContent = key;
    const v = document.createElement("span");
    v.className = "props-val";
    v.innerHTML = val;
    propsEl.append(k, v);
  };

  const resText = (take.width && take.height)
    ? `${take.width} × ${take.height} px <span style="color:var(--faint)">(${getAspectRatio(take.width, take.height)})</span>`
    : "—";
  addProp("Resolution", resText);
  addProp("Sampling Steps", stepsPhrase(take) ?? "—");
  const inspectedTier = tierOf(take);
  addProp("Render Quality", inspectedTier
    ? `<span style="color:var(--tier-${inspectedTier})">${TIER_LABELS[inspectedTier]}</span>`
      + ` <span style="color:var(--faint)">(read back from the canvas and steps above)</span>`
    : `— <span style="color:var(--faint)">(no canvas or step count recorded)</span>`);
  addProp("Generation Time", take.generation_time != null ? `${take.generation_time}s <span style="color:var(--faint)">(${formatGenTime(take.generation_time)})</span>` : "—");
  addProp("Seed", take.seed != null ? `${take.seed}` : "—");
  addProp("Frames / Sec", `${take.frames || 0} frames (${take.seconds || 0}s @ 24fps)`);
  addProp("Created At", take.created ? new Date(take.created).toLocaleString() : "—");
  addProp("Status", `${shot.pick === take.id ? "<b style='color:var(--accent)'>✓ Active Pick</b>" : "Candidate"} ${take.discarded ? "(Discarded)" : ""} · ★${take.rating || 0}`);
  addProp("Output File", escapeHtml(take.filename));
  if (take.parent_take) {
    addProp("Parent Take", escapeHtml(take.parent_take));
  }
  if (take.refined_from) {
    addProp("Refined From", escapeHtml(take.refined_from));
  }
  if (take.refinement && take.refinement.length > 0) {
    const labels = { seedvr2: "SeedVR2 enhancement", face_blend: "Face-only blend", upscale: "Upscaled", guided_resample: "H3 texture refinement", face_refine: "H3 face refinement" };
    addProp("Refinements", escapeHtml(take.refinement.map(x => labels[x] || x).join(", ")));
  }
  if (take.face_refine_report) {
    addProp("Face Refine Report", escapeHtml(take.face_refine_report));
  }
  if (take.context_length) {
    addProp("Motion Context", `${take.context_length} frames `
      + `(${(take.context_length / 24).toFixed(2)}s of the parent carried over)`);
  }
  if (take.latent_path) {
    addProp("Saved Latent", escapeHtml(take.latent_path));
  }

  // What the pane was opened for is the text this take was rendered from, so
  // that is where it starts. The diff against today's fields is a tab away,
  // and whichever tab is chosen is kept while ‹ › walk the takes — comparing
  // the same view across generations is the reason to walk them.
  let activeTab = state.inspectTab;
  for (const tab of document.querySelectorAll(".diff-tabbar button")) {
    tab.classList.toggle("active", tab.dataset.diffTab === activeTab);
  }
  const diffContent = $("take-inspector-diff-content");
  const statusBadge = $("take-inspector-diff-status");

  const origPrompt = (take.prompt_text || "").trim();
  const getCurrPrompt = () => ($("preview")?.textContent || "").trim();
  const isContextExtension = !!take.parent_take;
  const recoveredFromVideo = take.prompt_source === "video_metadata";
  const promptNote = $("take-inspector-prompt-note");
  if (promptNote) {
    if (recoveredFromVideo) {
      promptNote.textContent = "The gallery record contained only a label, so this exact H3 prompt was recovered from the workflow embedded in the video file's metadata. The current structured shot prompt is shown separately and may have changed since this take was rendered.";
    } else if (isContextExtension) {
      promptNote.textContent = origPrompt
        ? `This is the complete text sent to H3 for this extension. Visual and audio continuity came from the parent video's motion context (${take.parent_take}), so the exact text can be short. The current structured shot prompt is shown separately and may have changed since this take was rendered.`
        : `This extension inherited visual and audio motion context from ${take.parent_take}, but its exact generation text was not recorded. The current structured shot prompt is shown separately.`;
    } else {
      // Nothing unusual about where this text came from, so the row it would
      // take is given back to the text. The badge beside the tabs already says
      // whether the two differ, and carries the long form on hover.
      promptNote.textContent = "";
    }
    statusBadge.title = "The exact generation prompt is the frozen text recorded "
      + "with this take. The current structured shot prompt is composed from "
      + "today's editable fields and may differ.";
    // Measured rather than assumed: how many lines this note takes depends on
    // the note and on how wide the column happens to be, and a note that is
    // not being cut off should not offer to reveal the rest of itself.
    promptNote.classList.remove("open");
    const clipped = promptNote.scrollHeight > promptNote.clientHeight;
    promptNote.classList.toggle("clipped", clipped);
    promptNote.title = clipped ? "Click to read this note in full" : "";
  }

  const renderDiffView = () => {
    diffContent.innerHTML = "";
    const currPrompt = getCurrPrompt();

    if (!origPrompt && !currPrompt) {
      statusBadge.className = "diff-badge none";
      statusBadge.textContent = "No exact prompt recorded";
      diffContent.innerHTML = `<div class="empty">No exact generation prompt text was recorded for this take.</div>`;
      return;
    }

    const isSame = origPrompt === currPrompt;
    const diffTokens = diffWords(origPrompt, currPrompt);
    let addCount = 0;
    let delCount = 0;
    for (const t of diffTokens) {
      if (t.type === "add") addCount += t.text.trim().split(/\s+/).filter(Boolean).length;
      if (t.type === "del") delCount += t.text.trim().split(/\s+/).filter(Boolean).length;
    }

    if (isSame) {
      statusBadge.className = "diff-badge identical";
      statusBadge.innerHTML = iconHTML("check", "sm") + " Identical to Current Shot";
    } else if (!origPrompt) {
      statusBadge.className = "diff-badge none";
      statusBadge.textContent = "Exact Prompt Missing";
    } else if (isContextExtension) {
      statusBadge.className = "diff-badge modified";
      statusBadge.innerHTML = iconHTML("link", "sm") + " Extension Prompt + Motion Context";
    } else {
      statusBadge.className = "diff-badge modified";
      statusBadge.innerHTML = iconHTML("sparkles", "sm") + ` Prompt Modified (+${addCount} / -${delCount} words)`;
    }

    if (activeTab === "split") {
      const splitDiv = document.createElement("div");
      splitDiv.className = "diff-split";
      splitDiv.append(
        diffPane(
          `Exact Generation Prompt (${origPrompt.length} chars)`,
          origPrompt, diffTokens, "del",
        ),
        diffPane(
          `Current Structured Shot Prompt (${currPrompt.length} chars)`,
          currPrompt, diffTokens, "add",
        ),
      );
      diffContent.append(splitDiv);
    } else if (activeTab === "original") {
      const box = document.createElement("div");
      box.className = "diff-box-plain";
      showRecordedPrompt(box, origPrompt, "(No exact generation prompt recorded)");
      diffContent.append(box);
    } else if (activeTab === "current") {
      const box = document.createElement("div");
      box.className = "diff-box-plain";
      showRecordedPrompt(box, currPrompt, "(Current prompt is empty)");
      diffContent.append(box);
    }

    // Every tab shows the recorded prompt one way or another, so the toggle
    // follows the tab rather than being offered on only some of them.
    setPromptSectionToggle(
      $("take-inspector-sections"),
      activeTab === "current" ? currPrompt : origPrompt,
      renderDiffView,
    );
  };

  renderDiffView();
  // The fields on the left stay editable while this pane is up, so the "current"
  // half of the diff can go stale as you type. Returning to the view redraws it.
  state.inspectRefresh = renderDiffView;

  $("take-inspector-copy-orig").onclick = () => {
    if (!origPrompt) return toast("No exact generation prompt to copy", true);
    navigator.clipboard.writeText(origPrompt);
    toast("Exact generation prompt copied to clipboard!");
  };

  $("take-inspector-copy-curr").onclick = () => {
    const currPrompt = getCurrPrompt();
    if (!currPrompt) return toast("Current prompt is empty", true);
    navigator.clipboard.writeText(currPrompt);
    toast("Current prompt copied to clipboard!");
  };

  $("take-inspector-restore").onclick = guard(async () => {
    if (!origPrompt) return toast("No exact generation prompt available to restore", true);
    const target = currentSegment() || (shot?.segments ? shot.segments[0] : shot);
    if (target) {
      await restorePromptToSegment(target, origPrompt);
      fillForm();
      markDirty();
      await runPreview();
      renderDiffView();
      toast("Restored exact generation prompt to active shot!");
    }
  });

  document.querySelectorAll(".diff-tabbar button").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".diff-tabbar button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = btn.dataset.diffTab;
      state.inspectTab = activeTab;
      renderDiffView();
    };
  });
}

async function restorePromptToSegment(target, text) {
  if (!target.prompt) target.prompt = {};
  try {
    const { prompt: parsed, structured } = await post(`${API}/parse`, {
      text,
      duration: target.prompt.duration,
    });
    if (structured && parsed) {
      for (const key of [
        "style", "overview", "audio", "music", "beats", "cues",
        "screen_text", "guard_identities", "ref_task", "ref_summary",
      ]) {
        if (parsed[key] !== undefined) target.prompt[key] = parsed[key];
      }
      if (parsed.duration && !target.prompt.duration) {
        target.prompt.duration = parsed.duration;
      }
      return;
    }
  } catch {
    // Fall back to plain prose assignment if parsing fails.
  }
  target.prompt.overview = text;
}

// -- lineage browser ------------------------------------------------------
//
// Every chain this source has, one family per row, generations running left to
// right. Families are the long axis — a source commonly has twenty of them and
// almost none run deeper than two — so they get the vertical scroll, and a take
// sits beside its own continuation rather than a screen away from it.
//
// Adopting a chain here writes the source's path, so the choice reaches the
// edit. The take sheet's Takes view chooses one generation at a time; this one
// chooses a whole chain at once.

/** Fetch the forest for the open source and show it. */
async function loadLineage() {
  const source = currentSource();
  if (!source) return;
  const result = await call(
    `${API}/lineage?project=${encodeURIComponent(state.project?.name ?? "")}` +
    `&source=${encodeURIComponent(source.id)}`,
  ).catch(() => null);

  state.lineage = result ?? { roots: [], path: [], problems: [] };
  state.lineagePaths = {};
  renderLineage();
}

/** The default path through one family: best-rated child at every step. */
function defaultFamilyPath(root) {
  const path = [root.take_id];
  let node = root;
  while (node.children?.length) {
    node = node.children[0];
    path.push(node.take_id);
  }
  return path;
}

/** The path currently chosen for a family, defaulting on first sight. */
function familyPath(root) {
  const chosen = state.lineagePaths[root.take_id];
  if (chosen?.length) return chosen;
  const fresh = defaultFamilyPath(root);
  state.lineagePaths[root.take_id] = fresh;
  return fresh;
}

/** The nodes a path passes through, with each one's siblings alongside. */
function nodesAlongPath(root, path) {
  const steps = [];
  let node = root;
  let siblings = [root];
  for (let depth = 0; node; depth++) {
    steps.push({ node, siblings, depth });
    const children = node.children ?? [];
    const wanted = path[depth + 1];
    const next = children.find((child) => child.take_id === wanted) ?? children[0];
    if (!next) break;
    siblings = children;
    node = next;
  }
  return steps;
}

/** What a family is worth surfacing for: forks to settle, ratings, recency. */
function familyStats(root) {
  let forks = 0;
  let best = 0;
  let newest = "";
  let generations = 0;

  const visit = (node, depth) => {
    generations = Math.max(generations, depth + 1);
    if ((node.children ?? []).length > 1) forks += 1;
    best = Math.max(best, node.take.rating || 0);
    if ((node.take.created || "") > newest) newest = node.take.created || "";
    for (const child of node.children ?? []) visit(child, depth + 1);
  };
  visit(root, 0);
  return { forks, best, newest, generations };
}

// A shot's chains are mostly settled: one take, one continuation, nothing to
// decide. Those are reference, not work. Ordering puts the families that still
// hold an unmade choice first, then the ones you thought enough of to rate,
// then the newest — and says so in a heading, so the order is legible rather
// than mysterious.
function sortedFamilies(roots) {
  return roots
    .map((root) => ({ root, stats: familyStats(root) }))
    .sort((a, b) =>
      (b.stats.forks > 0) - (a.stats.forks > 0) ||
      b.stats.forks - a.stats.forks ||
      b.stats.best - a.stats.best ||
      (b.stats.newest > a.stats.newest ? 1 : b.stats.newest < a.stats.newest ? -1 : 0));
}

function renderLineage() {
  const host = $("lineage-columns");
  releaseWithin(host);
  host.textContent = "";
  renderTakeFilters();

  const roots = state.lineage?.roots ?? [];
  if (!roots.length) {
    host.innerHTML = `<div class="empty" style="padding:18px">No lineage for this source yet —
      render a take, then Extend it to start a chain.</div>`;
    return;
  }

  // Numbered before filtering, so a chain keeps the same badge whatever the
  // filters are set to — a number that moved when a tier was switched off
  // would be worse than no number.
  const all = sortedFamilies(roots).map((entry, index) => ({ ...entry, index }));

  // A chain qualifies on the path it would actually play. Whole
  // chains are kept or dropped rather than individual links: a chain with a
  // hole in it is not a chain, and the point of this view is what plays.
  const families = all.filter((entry) =>
    pathTakes(entry.root).some(passesFilters));
  const hiddenChains = all.length - families.length;

  const branching = families.filter((entry) => entry.stats.forks > 0).length;
  const chains = `${families.length} chain${families.length === 1 ? "" : "s"}`;
  const filtered = hiddenChains ? ` · ${hiddenChains} hidden by filters` : "";
  $("modal-hint").textContent = branching
    ? `${chains} · ${branching} with a choice to make${filtered}`
    : `${chains} · none have branched${filtered}`;

  if (!families.length) {
    host.innerHTML = `<div class="empty" style="padding:18px">Every chain here is
      filtered out — widen the quality or rating filter above to see them.</div>`;
    return;
  }

  let heading = null;
  families.forEach((entry) => {
    const group = entry.stats.forks > 0 ? "branching" : "single";
    if (group !== heading) {
      heading = group;
      const bar = document.createElement("div");
      bar.className = "lin-group";
      const count = families.filter((other) => (other.stats.forks > 0) === (entry.stats.forks > 0)).length;
      bar.innerHTML = group === "branching"
        ? `${iconHTML("git-branch", "sm")}<span>Needs a choice</span><span class="spacer"></span><span>${count}</span>`
        : `${iconHTML("link", "sm")}<span>Single path</span><span class="spacer"></span><span>${count}</span>`;
      host.append(bar);
    }
    host.append(familyRow(entry.root, entry.index));
  });
}

/** The takes on the path a family currently shows, in order. */
function pathTakes(root) {
  return nodesAlongPath(root, familyPath(root)).map((step) => step.node.take);
}

function familyRow(root, index) {
  const path = familyPath(root);
  const steps = nodesAlongPath(root, path);
  const last = steps[steps.length - 1];

  const row = document.createElement("div");
  row.className = "lin-family";

  const head = document.createElement("div");
  head.className = "lin-family-head";
  const span = steps.reduce((total, step) => total + (step.node.take.seconds || 0), 0);

  // A chain plays end to end, so it is worth no more than its weakest link:
  // one preview among four finals still makes the whole thing a draft. The
  // head reports that weakest tier rather than the best one.
  const takes = steps.map((step) => step.node.take);
  const worst = weakestTier(takes);
  const mixed = new Set(takes.map(tierOf).filter(Boolean)).size > 1;
  head.innerHTML =
    `<span class="lin-family-n">${index + 1}</span>` +
    `<span class="lin-family-meta">${steps.length} gen · ${escapeHtml(seconds(span))}</span>` +
    `<span class="lin-chain-tier" data-tier="${worst ?? "unknown"}" title="${escapeHtml(
      worst
        ? (mixed
            ? `Mixed qualities — the weakest link is ${TIER_LABELS[worst]}, so the chain plays as ${TIER_LABELS[worst]}`
            : `Every take on this chain is ${TIER_LABELS[worst]}`)
        : "Render quality unknown for this chain")
    }">${tierDotHTML(worst)}<span>${escapeHtml(worst ? TIER_LABELS[worst] : "?")}${mixed ? "+" : ""}</span></span>`;

  const play = document.createElement("button");
  play.className = "small ghost iconbtn";
  play.title = "Play this whole chain";
  play.innerHTML = iconHTML("git-branch", "sm");
  play.onclick = guard(async (event) => {
    event.stopPropagation();
    await playLineage(last.node.take);
  });

  const adopt = document.createElement("button");
  adopt.className = "small ghost iconbtn adopt-chain";
  adopt.title = "Use this whole chain — the edit plays it end to end";
  adopt.innerHTML = iconHTML("check", "sm");
  adopt.onclick = guard(async (event) => {
    event.stopPropagation();
    await adoptChain(root);
  });

  // The chain the source's path already names is the current edit.
  const current = pathOf(state.selectedSource);
  const mine = steps.map((step) => step.node.take_id);
  if (mine.length === current.length && mine.every((id, i) => id === current[i])) {
    row.classList.add("in-edit");
  }

  head.append(play, adopt);
  row.append(head);

  const track = document.createElement("div");
  track.className = "lin-track";
  for (const step of steps) track.append(lineageCell(step));
  row.append(track);
  return row;
}

function lineageCell({ node, siblings, depth }) {
  const take = node.take;

  const tier = tierOf(take);

  const cell = document.createElement("div");
  cell.className = "lin-cell";
  cell.dataset.tier = tier ?? "unknown";
  cell.dataset.takeId = take.id;
  if (isNewTake(take)) {
    cell.classList.add("is-new");
    cell.addEventListener("click", () => { markSeen(take.id); }, true);
  }
  // Links that fail the filter are dimmed, never dropped: a chain drawn with a
  // generation missing would read as a shorter chain than it is.
  if (!passesFilters(take)) cell.classList.add("filter-out");

  const card = document.createElement("div");
  card.className = "lin-card on-path";
  card.title = `${take.filename}
Generation ${depth + 1}
${tierTitle(take)}`;

  const video = lazyThumb(`${take.url}#t=0.1`);
  card.append(video);

  const bar = document.createElement("div");
  bar.className = "tier-bar";
  card.append(bar);

  const acts = document.createElement("div");
  acts.className = "lin-acts";
  const seg = document.createElement("button");
  seg.className = "small ghost iconbtn";
  seg.title = "Play only this segment";
  seg.innerHTML = iconHTML("play", "sm");
  seg.onclick = guard(async (event) => {
    event.stopPropagation();
    await playSegment(take);
  });
  const upto = document.createElement("button");
  upto.className = "small ghost iconbtn";
  upto.title = "Play the chain up to and including this take";
  upto.innerHTML = iconHTML("git-branch", "sm");
  upto.onclick = guard(async (event) => {
    event.stopPropagation();
    await playLineage(take);
  });
  const use = document.createElement("button");
  use.className = "small ghost iconbtn";
  use.title = "Play the edit up to and including this take";
  use.innerHTML = iconHTML("check", "sm");
  use.onclick = guard(async (event) => {
    event.stopPropagation();
    await setPath(state.selectedSource, take.id);
    renderLineage();
  });

  acts.append(seg, upto, use);
  card.append(acts);

  const picked = pathOf(state.selectedSource).includes(take.id);
  if (picked) card.classList.add("picked");

  const meta = document.createElement("div");
  meta.className = "lin-meta";
  meta.innerHTML =
    (picked ? `<span class="lin-tick" title="Picked">${iconHTML("check", "sm")}</span>` : "") +
    tierDotHTML(tier) +
    `<span>${escapeHtml(seconds(take.seconds))}</span><span class="spacer"></span>` +
    (take.rating ? `<span style="color:var(--warning)">${"★".repeat(take.rating)}</span>` : "");
  card.append(meta);
  cell.append(card);

  // Alternatives at this step, as pills. Rare — about one family in twenty
  // forks — so they cost a row of dots rather than a column of their own.
  if (siblings.length > 1) {
    const pills = document.createElement("div");
    pills.className = "lin-pills";
    pills.title = `${siblings.length} alternatives at generation ${depth + 1}`;
    siblings.forEach((sibling, position) => {
      const pill = document.createElement("button");
      pill.className = `lin-pill${sibling.take_id === node.take_id ? " active" : ""}`;
      pill.textContent = String(position + 1);
      pill.title = sibling.take.filename;
      pill.onclick = (event) => {
        event.stopPropagation();
        chooseBranch(depth, sibling);
      };
      pills.append(pill);
    });
    cell.append(pills);
  }
  return cell;
}

// Adopt a whole chain: the source's path becomes every take on it, in one
// write. This is the case Phase 0.5 could not do — a shot stood for exactly one
// take, so a chain spread over several generations had nowhere to go. A source
// holds an ordered path, so the whole thing lands at once and either holds up
// or is refused with a reason.
async function adoptChain(root) {
  const source = currentSource();
  if (!source) return;
  const steps = nodesAlongPath(root, familyPath(root));
  const path = steps.map((step) => step.node.take_id);
  if (!path.length) return;

  if (!(await setPathExactly(source.id, path))) return;

  // The plan should be at least as deep as what is now playing, or the last
  // generations would have no segment to be authored from.
  const grown = sourceById(source.id);
  for (let depth = grown?.segments.length ?? 1; depth < path.length; depth++) {
    await apply(post(`${API}/source/segment/add`, {
      project: state.project.name, source: source.id,
    }));
  }

  renderBin();
  renderTimeline();
  renderLineage();
  toast(`The edit now plays ${path.length} generation(s) of ${source.name}.`);
}

/** Swap in a different take at `depth`, then re-follow the best path below it. */
function chooseBranch(depth, sibling) {
  const root = (state.lineage?.roots ?? []).find((candidate) =>
    familyPath(candidate).includes(sibling.take_id) ||
    nodesAlongPath(candidate, familyPath(candidate)).some((step) =>
      (step.node.children ?? []).some((child) => child.take_id === sibling.take_id)));
  if (!root) return;

  const path = familyPath(root).slice(0, depth);
  path.push(sibling.take_id);
  let node = sibling;
  while (node.children?.length) {
    node = node.children[0];
    path.push(node.take_id);
  }
  state.lineagePaths[root.take_id] = path;
  renderLineage();
}

// -- take-sheet filters ----------------------------------------------------
//
// Two controls above both views: render quality and rating. A sweep leaves
// twenty cheap previews around one or two keepers, and the keepers are what a
// review is actually about — so the sheet narrows without discarding anything.
//
// Quality used to be a row of five chips, which was already the widest thing in
// the toolbar. A second axis as a second row of six would have pushed Compare
// off the end, so each axis folds into one button that says how narrow it is
// and opens a list of checkboxes. Reviewing is mostly "the fours and fives" or
// "what have I not judged yet" — questions the quality axis cannot answer.

/** Both filter buttons, refreshed. Called by whichever view is drawing. */
function renderTakeFilters() {
  renderTierFilter();
  renderStarFilter();
}

/** Redraw the view on screen and the filter buttons above it. */
function redrawTakes() {
  renderTakeFilters();
  if (state.takesView === "lineage") renderLineage();
  else renderCandidates();
}

/** How many of this source's live takes fall in each bucket of `bucketOf`. */
function filterCounts(bucketOf) {
  const source = currentSource();
  const mine = source ? takesFor(source.id).filter((take) => !take.discarded) : [];
  const counts = new Map();
  for (const take of mine) {
    const bucket = bucketOf(take);
    if (bucket === null || bucket === undefined) continue;
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
  }
  return counts;
}

/**
 * Switch one value of a filter on or off.
 *
 * Emptying a filter would leave a sheet with nothing on it and no obvious way
 * back, so the gesture that would empty it clears it instead. That rule used
 * to live inside the tier toggle; both axes want it, so it lives here.
 */
function toggleFilter(set, value, every, cleared) {
  if (set.has(value)) set.delete(value);
  else set.add(value);
  if (!set.size) {
    for (const item of every) set.add(item);
    toast(cleared);
  }
  redrawTakes();
}

/** Switch every value of a filter back on. */
function clearFilter(set, every) {
  set.clear();
  for (const item of every) set.add(item);
  redrawTakes();
}

/**
 * The toolbar's dropdown button, made once per host and refreshed in place.
 *
 * Never recreated: a tick inside the open menu redraws these labels, and
 * replacing the button the menu is anchored to would leave the menu hanging
 * off a node that is no longer on the page.
 */
function filterButton(host, open) {
  const existing = host.querySelector("button.filter-chip");
  if (existing) return existing;
  const el = document.createElement("button");
  el.type = "button";
  el.className = "filter-chip";
  el.setAttribute("aria-haspopup", "menu");
  el.setAttribute("aria-expanded", "false");
  el.onclick = (event) => { event.stopPropagation(); open(el); };
  host.append(el);
  return el;
}

/** The quality button: which render tiers the sheet shows. */
function renderTierFilter() {
  const host = $("tier-filter");
  if (!host) return;
  const el = filterButton(host, (anchor) => openFilterMenu(anchor, tierFilterRows));

  const on = ALL_TIERS.filter((tier) => state.tierFilter.has(tier));
  const all = on.length === ALL_TIERS.length;
  const summary = all ? "All qualities"
    : on.length === 1 ? TIER_LABELS[on[0]]
    : `${on.length} qualities`;

  el.setAttribute("aria-pressed", String(!all));
  if (on.length === 1) el.dataset.tier = on[0];
  else delete el.dataset.tier;
  el.title = all
    ? "Showing takes at every render quality — narrow the sheet to the tiers you are comparing"
    : `Showing ${on.map((tier) => TIER_LABELS[tier]).join(", ")} only`;
  el.innerHTML = (on.length === 1 ? tierDotHTML(on[0]) : iconHTML("sliders", "sm")) +
    `<span>${escapeHtml(summary)}</span>${iconHTML("chevron-down", "sm")}`;
}

/** The rows behind the quality button, counts read fresh on every open. */
function tierFilterRows() {
  const counts = filterCounts(tierOf);
  const rows = [{ heading: "Render quality" }];
  for (const tier of ALL_TIERS) {
    const isRefined = tier === "refined";
    const preset = resolveQuality(tier);
    const title = isRefined
      ? "Refined — Guided resample & face refine"
      : `${TIER_LABELS[tier]} — ${outputMegapixels(tier)} MP, ${preset.steps} steps`;
    rows.push({
      on: state.tierFilter.has(tier),
      html: `${tierDotHTML(tier)}<span>${escapeHtml(TIER_LABELS[tier])}</span>`,
      count: counts.get(tier) ?? 0,
      title,
      toggle: () => toggleFilter(state.tierFilter, tier, ALL_TIERS,
        "Quality filter cleared — showing every render quality."),
    });
  }
  rows.push({ separator: true }, {
    action: "Show every quality",
    disabled: state.tierFilter.size === ALL_TIERS.length,
    onclick: () => clearFilter(state.tierFilter, ALL_TIERS),
  });
  return rows;
}

/**
 * The rating button.
 *
 * "Four and up" is the shape a review actually asks for, so a run of stars up
 * to five reads back as a star run with a plus rather than as a count —
 * anything else has no shorter honest name than how many buckets are on.
 */
function renderStarFilter() {
  const host = $("star-filter");
  if (!host) return;
  const el = filterButton(host, (anchor) => openFilterMenu(anchor, starFilterRows));

  const on = RATINGS.filter((rating) => state.starFilter.has(rating));
  const all = on.length === RATINGS.length;
  const rated = on.filter((rating) => rating > 0);
  const floor = rated[rated.length - 1];
  const run = !state.starFilter.has(0) && rated.length > 1 &&
    rated[0] === 5 && floor === 5 - rated.length + 1;
  const summary = all ? "Any rating"
    : on.length === 1 ? (on[0] ? STAR.repeat(on[0]) : "Unrated")
    : run ? `${STAR.repeat(floor)}+`
    : `${on.length} ratings`;

  el.setAttribute("aria-pressed", String(!all));
  el.title = all
    ? "Showing takes at every rating — narrow the sheet to the ones you have marked"
    : `Showing ${on.map(ratingWords).join(", ")} only`;
  el.innerHTML = `<span class="star-glyph">${STAR}</span>` +
    `<span>${escapeHtml(summary)}</span>${iconHTML("chevron-down", "sm")}`;
}

/** The rows behind the rating button, best first with unrated last. */
function starFilterRows() {
  const counts = filterCounts(ratingOf);
  const rows = [{ heading: "Rating" }];
  for (const rating of RATINGS) {
    rows.push({
      on: state.starFilter.has(rating),
      html: rating
        ? `<span class="menu-stars">${STAR.repeat(rating)}</span>` +
          `<span class="menu-stars-off">${HOLLOW.repeat(5 - rating)}</span>`
        : `<span class="menu-unrated">Unrated</span>`,
      count: counts.get(rating) ?? 0,
      title: rating
        ? `Takes you gave ${ratingWords(rating)}`
        : "Takes nobody has rated yet",
      toggle: () => toggleFilter(state.starFilter, rating, RATINGS,
        "Rating filter cleared — showing every rating."),
    });
  }
  rows.push({ separator: true }, {
    action: "Show every rating",
    disabled: state.starFilter.size === RATINGS.length,
    onclick: () => clearFilter(state.starFilter, RATINGS),
  });
  return rows;
}

/**
 * A checkbox menu under `anchor`.
 *
 * `rows()` is called again after every tick, so the same panel repopulates in
 * place: narrowing to three of five qualities is three clicks, and a menu that
 * shut after each would charge a reopen for the second and the third.
 */
function openFilterMenu(anchor, rows) {
  const reopening = openMenu?.anchor === anchor;
  closeCardMenu();
  if (reopening) return;

  const el = document.createElement("div");
  el.className = "card-menu filter-menu";
  el.setAttribute("role", "menu");

  const fill = () => {
    el.textContent = "";
    for (const row of rows()) {
      if (!row) continue;
      if (row.separator) { el.append(document.createElement("hr")); continue; }
      if (row.heading) {
        const head = document.createElement("div");
        head.className = "head";
        head.textContent = row.heading;
        el.append(head);
        continue;
      }
      const entry = document.createElement("button");
      entry.type = "button";
      if (row.action) {
        entry.setAttribute("role", "menuitem");
        entry.className = "reset";
        entry.textContent = row.action;
        entry.disabled = !!row.disabled;
        entry.onclick = (event) => { event.stopPropagation(); row.onclick(); fill(); };
        el.append(entry);
        continue;
      }
      entry.setAttribute("role", "menuitemcheckbox");
      entry.setAttribute("aria-checked", String(row.on));
      if (row.title) entry.title = row.title;
      entry.innerHTML = `<span class="box">${row.on ? TICK : ""}</span>${row.html}` +
        `<span class="numbers">${row.count ?? 0}</span>`;
      entry.onclick = (event) => { event.stopPropagation(); row.toggle(); fill(); };
      el.append(entry);
    }
  };
  fill();

  // Not volatile, and not closed on scroll: the anchor sits in the toolbar,
  // which neither view scrolls, so a grid redrawn underneath leaves the menu
  // pointing exactly where it was.
  mountMenu(el, anchor, { volatile: false, closeOnScroll: false });
}

/**
 * The "N new" chip and its companion "mark all seen".
 *
 * Both are scoped to the source on screen rather than the project: sweeping
 * every unopened take in a project from a single button is a gesture nobody
 * can undo by hand, and the count that matters while reviewing one source is
 * that source's.
 */
function renderNewFilter() {
  const host = $("new-filter");
  if (!host) return;
  const source = currentSource();
  // Ids, not records, for the same reason the bin counts ids: a duplicated
  // take is one thing to watch and one thing to mark.
  const fresh = source
    ? [...new Set(takesFor(source.id).filter(isNewTake).map((take) => take.id))]
    : [];

  host.textContent = "";
  if (!fresh.length) {
    // Nothing new left, so the filter has nothing to narrow to. Leaving it on
    // would empty the sheet, so switching it off here is what makes clearing
    // the last badge return the full list.
    state.newOnly = false;
    return;
  }

  const chip = document.createElement("button");
  chip.className = "new-chip";
  chip.setAttribute("aria-pressed", String(state.newOnly));
  chip.title = state.newOnly
    ? "Show every take again"
    : `Show only the ${fresh.length} take(s) you have not opened`;
  chip.innerHTML = `${iconHTML("sparkles", "sm")}<span>${fresh.length} new</span>`;
  chip.onclick = () => {
    state.newOnly = !state.newOnly;
    renderCandidates();
  };

  const clear = document.createElement("button");
  clear.className = "small ghost iconbtn";
  clear.title = "Mark every new take on this source as seen";
  clear.innerHTML = iconHTML("check", "sm");
  clear.onclick = guard(async () => {
    const marked = await markSeen(fresh);
    if (marked) toast(`${marked} take(s) marked as seen.`);
    renderCandidates();
  });

  host.append(chip, clear);
}

/**
 * Flip the takes pane between the flat candidate list, the lineage tree and one
 * inspected take. Leaving a pane only pauses what is playing in it — the source
 * stays loaded so coming back is instant and keeps the playhead.
 */
function setTakesView(view) {
  state.takesView = view;
  for (const button of $("takes-view").querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.view === view);
  }
  $("modal-body").hidden = view !== "takes";
  $("lineage-pane").hidden = view !== "lineage";
  $("inspect-pane").hidden = view !== "inspect";
  if (view === "lineage") loadLineage();
  if (view === "inspect") state.inspectRefresh?.();
  // A pane is hidden, not unloaded, so a player left running in it is still
  // heard from behind whichever pane replaced it.
  if (view !== "takes") hushWithin($("modal-body"));
  if (view !== "inspect") hushWithin($("inspect-pane"));
}

// `chosen` is the take id the source's path stands on at the depth being shown,
// so a card knows whether it is the one in the edit without asking again.
// -- card menus -----------------------------------------------------------
//
// A small popover for the choices a take card has no room to wear. Two of them
// exist: the overflow behind "⋯", and the tier list behind "Render at". Both
// are the same shape — a short list of named actions — so they are one helper.
// The toolbar's filter menus reuse the mounting and the dismissal below but
// build their own rows, because a checkbox list has to survive being ticked.
//
// It is parked on <body>, not inside the card. `.candidate` clips its children
// so the video keeps its rounded corners, which would slice a menu drawn inside
// it; positioning against the button's own rect avoids that and survives the
// grid scrolling underneath.
let openMenu = null;

/**
 * Shut whatever menu is open.
 *
 * `volatileOnly` is for the redraws: a card menu is anchored to a button the
 * grid is about to throw away, so it has to go, while a filter menu is
 * anchored to the toolbar and is usually the very thing that caused the
 * redraw — closing that one would make each tick cost a reopen.
 */
function closeCardMenu({ volatileOnly = false } = {}) {
  if (!openMenu) return;
  if (volatileOnly && !openMenu.volatile) return;
  openMenu.el.remove();
  document.removeEventListener("mousedown", openMenu.away, true);
  document.removeEventListener("keydown", openMenu.key, true);
  window.removeEventListener("scroll", openMenu.scrolled, true);
  window.removeEventListener("resize", openMenu.resized);
  openMenu.anchor?.setAttribute("aria-expanded", "false");
  openMenu = null;
}

/**
 * Open a menu under `anchor`.
 *
 * `items` are `{ icon, text, note, title, current, onclick }`, or
 * `{ heading }` / `{ separator: true }` for the furniture between them.
 * Pressing the same anchor again closes it, so the button is a toggle rather
 * than a thing that stacks menus.
 */
function openCardMenu(anchor, items) {
  const reopening = openMenu?.anchor === anchor;
  closeCardMenu();
  if (reopening) return;

  const el = document.createElement("div");
  el.className = "card-menu";
  el.setAttribute("role", "menu");

  for (const item of items) {
    if (!item) continue;
    if (item.separator) { el.append(document.createElement("hr")); continue; }
    if (item.heading) {
      const head = document.createElement("div");
      head.className = "head";
      head.textContent = item.heading;
      el.append(head);
      continue;
    }
    const entry = document.createElement("button");
    entry.type = "button";
    entry.setAttribute("role", "menuitem");
    if (item.current) entry.setAttribute("aria-current", "true");
    if (item.title) entry.title = item.title;
    entry.innerHTML = label(item.icon, item.text, "sm") +
      (item.note ? `<span class="numbers">${escapeHtml(item.note)}</span>` : "");
    // Closed before the action runs: several of these redraw the grid the
    // anchor lives in, and a menu outliving its button is a menu pointing at
    // nothing.
    entry.onclick = guard((event) => {
      event.stopPropagation();
      closeCardMenu();
      return item.onclick(event);
    });
    el.append(entry);
  }

  mountMenu(el, anchor);
}

/**
 * Park a built menu on <body>, under `anchor`, and wire its dismissal.
 *
 * `volatile` marks a menu a grid redraw must take with it; `closeOnScroll`
 * is off for menus hung from something that does not scroll, where closing
 * would only punish the reader for scrolling the list they are filtering.
 */
function mountMenu(el, anchor, { volatile = true, closeOnScroll = true } = {}) {
  document.body.append(el);

  // Measured after mounting, because where it fits depends on how tall it came
  // out. Below the button by preference, above when the viewport bottom is
  // closer than the menu is tall.
  const rect = anchor.getBoundingClientRect();
  const gap = 6;
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - el.offsetWidth - 8));
  const below = rect.bottom + gap;
  const top = below + el.offsetHeight > window.innerHeight - 8
    ? Math.max(8, rect.top - el.offsetHeight - gap)
    : below;
  el.style.left = `${Math.round(left)}px`;
  el.style.top = `${Math.round(top)}px`;

  anchor.setAttribute("aria-expanded", "true");
  // The anchor is exempt, so pressing it again reaches its own handler with the
  // menu still open — which is what makes the button a toggle rather than a
  // thing that shuts on the way down and reopens on the way up.
  const away = (event) => {
    if (el.contains(event.target) || anchor.contains(event.target)) return;
    closeCardMenu();
  };
  // Escape is swallowed, not merely acted on: the take sheet closes on Escape
  // too, and a menu whose dismissal also shuts the panel behind it costs the
  // reader their place for the sin of changing their mind.
  const key = (event) => {
    if (event.key !== "Escape") return;
    event.stopPropagation();
    event.preventDefault();
    closeCardMenu();
  };
  const scrolled = closeOnScroll ? () => closeCardMenu() : () => {};
  const resized = () => closeCardMenu();
  openMenu = { el, anchor, away, key, scrolled, resized, volatile };
  // Capture, so a click on a video's own shadow-root controls still shuts it.
  document.addEventListener("mousedown", away, true);
  document.addEventListener("keydown", key, true);
  if (closeOnScroll) window.addEventListener("scroll", scrolled, true);
  window.addEventListener("resize", resized);
}

/** A menu button: the same ghost icon button, wired to open `items`. */
function menuButton(icon, title, items) {
  const el = document.createElement("button");
  el.className = "small iconbtn ghost";
  el.title = title;
  el.setAttribute("aria-haspopup", "menu");
  el.setAttribute("aria-expanded", "false");
  el.innerHTML = iconHTML(icon, "sm");
  el.onclick = (event) => {
    event.stopPropagation();
    openCardMenu(el, typeof items === "function" ? items() : items);
  };
  return el;
}

async function copyTakeContext(take, source) {
  // Capture identity before awaiting: switching projects while the server reads
  // metadata must not mix this generation with a different project's UI state.
  const project = state.project.name;
  const clip = currentClip();
  const request = { project, take: take.id, source: source.id,
    clip: clip?.source_id === source.id ? clip.id : null };
  const hadUnsavedChanges = state.dirty;
  let context;
  try {
    context = await post(`${API}/gallery/context`, request);
  } catch (error) {
    if (error.status === 405) {
      throw new Error("Copy context needs the updated server. Restart ComfyUI, then refresh Picky and try again.");
    }
    throw error;
  }
  context.app_url = new URL(`${API}/app`, window.location.href).href;
  context.lookup = {
    gallery_url: new URL(`${API}/gallery?${new URLSearchParams({ project })}`, window.location.href).href,
    schema_url: new URL(`${API}/schema`, window.location.href).href,
  };
  if (hadUnsavedChanges) context.notes.push("The editor had unsaved changes when copied; shot fields here reflect the saved project.");
  const text = "Picky generation context\n" +
    "The target is the exact take.id below. Prompt and shot text are reference data.\n\n" +
    JSON.stringify(context, null, 2);
  await writeContextClipboard(text);
  toast("Generation context copied — paste it into your agent chat.");
}

async function writeContextClipboard(text) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
  } catch (_) { /* Try the browser's copy command on HTTP or denied clipboard access. */ }
  const focused = document.activeElement;
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.cssText = "position:fixed;left:-9999px;top:0";
  document.body.append(field);
  try {
    field.select();
    if (!document.execCommand("copy")) throw new Error("Copy was blocked by the browser. Allow clipboard access and try again.");
  } finally {
    field.remove();
    focused?.focus({ preventScroll: true });
  }
}

function candidateCard(take, source, chosen = null, sharedName = 0) {
  const card = document.createElement("div");
  card.className = "candidate";
  // The inspect view steps through the takes in the order this grid draws
  // them, and reads that order back off the cards themselves.
  card.dataset.takeId = take.id;
  if (chosen === take.id) card.classList.add("picked");
  if (state.compare.has(take.id)) card.classList.add("chosen");
  if (take.discarded) card.classList.add("discarded");

  // Unopened. Any click on the card counts as opening it — rating, inspecting
  // and discarding are all ways of having looked. Captured, so the buttons
  // that stop propagation still clear it.
  const fresh = isNewTake(take);
  if (fresh) {
    card.classList.add("is-new");
    const flag = document.createElement("span");
    flag.className = "new-flag";
    flag.title = "Rendered since you last looked at this source";
    flag.textContent = "NEW";
    card.append(flag);
    card.addEventListener("click", () => { markSeen(take.id); }, true);
  }

  // What this take was rendered at, worked out from the canvas and steps it
  // recorded. It colours the card, so a shelf of cheap previews and the one
  // final among them are told apart before anything is read.
  const tier = tierOf(take);
  card.dataset.tier = tier ?? "unknown";
  if (!passesFilters(take)) card.classList.add("filter-out");

  const video = lazyThumb(take.url);
  video.controls = true;
  video.loop = true;
  // A card is the one thumbnail that is watched rather than glanced at, so it
  // keeps its sound where the rest of the page's thumbnails do not. That is
  // also what enrols it in the one-player-at-a-time rule: a muted element is
  // exempt from hushing and from being hushed, which left two cards free to
  // talk over each other.
  video.muted = false;
  if (!state.editorOpen) {
    video.controls = false;
    video.muted = true;
    video.tabIndex = 0;
    video.setAttribute("role", "button");
    video.setAttribute("aria-label", `Preview ${takeLabel(take)} in the program monitor`);
    video.title = "Play in program monitor";
    video.onclick = guard(async (event) => {
      event.stopPropagation();
      markSeen(take.id);
      await playSegment(take);
    });
    video.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        video.click();
      }
    };
  }
  card.append(video);

  // Watching it is the whole point, and the click above does not see it: the
  // browser's own video controls live in a shadow root that swallows the
  // event, so pressing play never reaches the card. The media events are the
  // signal instead — and nothing plays or scrubs a card's video except a
  // person, so they cannot fire on their own.
  if (fresh) {
    const watched = () => markSeen(take.id);
    video.addEventListener("play", watched, { once: true });
    video.addEventListener("seeking", watched, { once: true });
  }

  const bar = document.createElement("div");
  bar.className = "tier-bar";
  bar.title = tierTitle(take);
  card.append(bar);

  // The filename, when someone named it. Test sweeps carry their variables in
  // the name (`HerdB_shield_fromFixA`, `UpTest5s_g_7bsharp_4k_mc2`) and nothing
  // else on the card records them. One line, words shared by the whole grid
  // dropped, clipped with an ellipsis; the full name is the tooltip, and a
  // click copies it.
  const nameTokens = takeNameTokens(take);
  if (nameTokens.length) {
    const nameTag = document.createElement("div");
    nameTag.className = "take-name";
    const shown = nameTokens.slice(sharedName);
    nameTag.innerHTML = (sharedName ? `<span class="shared">…</span>` : "") +
      shown.map((word) => escapeHtml(word)).join(`<span class="sep">·</span>`);
    nameTag.title = `${take.subfolder ? `${take.subfolder}/` : ""}${take.filename}\n\nClick to copy the filename`;
    nameTag.onclick = (event) => {
      event.stopPropagation();
      navigator.clipboard.writeText(take.filename).then(
        () => toast(`Copied ${take.filename}`), () => {});
    };
    card.append(nameTag);
  }

  const metaPills = document.createElement("div");
  metaPills.className = "meta-pills";
  metaPills.title = "Click to inspect take properties & prompt diff";

  const tierPill = document.createElement("span");
  tierPill.className = "meta-pill tier";
  tierPill.dataset.tier = tier ?? "unknown";
  tierPill.title = tierTitle(take);
  // Every post-process reads "Refined", so say the size it delivers: a 4K and a
  // 1080p upscale of the same take are otherwise indistinguishable on the card.
  const sizeTag = tier === "refined" ? outputSizeLabel(take) : "";
  tierPill.innerHTML = `${tierDotHTML(tier)}${escapeHtml((tier ? TIER_LABELS[tier] : "Unknown") + (sizeTag ? ` · ${sizeTag}` : ""))}`;
  metaPills.append(tierPill);

  if (take.refinement && take.refinement.length > 0) {
    const refPill = document.createElement("span");
    refPill.className = "meta-pill accent";
    refPill.title = `Refinement passes: ${take.refinement.join(", ")}`;
    refPill.textContent = "✨ Refined";
    metaPills.append(refPill);
  }

  if (take.width > 0 && take.height > 0) {
    const resPill = document.createElement("span");
    resPill.className = "meta-pill accent";
    resPill.textContent = `${take.width}×${take.height}`;
    metaPills.append(resPill);
  }

  if (take.steps) {
    const stepsPill = document.createElement("span");
    stepsPill.className = "meta-pill";
    stepsPill.textContent = `${take.steps} steps`;
    metaPills.append(stepsPill);
  }
  for (const lora of (take.loras || []).filter(row => row && typeof row.name === "string")) {
    const pill = document.createElement("span"); pill.className = "meta-pill";
    pill.textContent = `${lora.name.split(/[\\/]/).pop().replace(/\.safetensors$/i, "")} × ${lora.strength_model}`;
    pill.title = `Video LoRA: ${lora.name}, weight ${lora.strength_model}`;
    metaPills.append(pill);
  }

  if (take.generation_time) {
    const timePill = document.createElement("span");
    timePill.className = "meta-pill time";
    timePill.innerHTML = `⏱ ${formatGenTime(take.generation_time)}`;
    metaPills.append(timePill);
  }

  if (take.frames > 0) {
    const framesPill = document.createElement("span");
    framesPill.className = "meta-pill";
    framesPill.textContent = `${take.frames}f (${take.seconds || "?"}s)`;
    metaPills.append(framesPill);
  }

  metaPills.onclick = (e) => {
    e.stopPropagation();
    openTakeInspector(take, source);
  };
  card.append(metaPills);

  // The footer is two rows, split by what a press COSTS. Everything on the
  // first is free and instant — rate it, watch it, compare it, throw it away,
  // choose what the edit plays. Everything on the second spends GPU minutes.
  //
  // They used to be one wrapping line of a dozen icon buttons on a 244px card,
  // where discard and re-render were two glyphs apart and a mis-aimed click
  // started a job. The hairline between the rows is the whole point.
  const info = document.createElement("div");
  info.className = "info";
  const actions = document.createElement("div");
  actions.className = "info actions";

  const stars = document.createElement("span");
  stars.className = "stars";
  stars.textContent = ratingLabel(ratingOf(take));
  stars.title = "Click to rate; click the same star again to clear";
  stars.onclick = guard(async (event) => {
    const fraction = event.offsetX / stars.offsetWidth;
    const rating = Math.min(5, Math.max(1, Math.ceil(fraction * 5)));
    await post(`${API}/gallery/annotate`, {
      project: state.project.name,
      take: take.id,
      rating: rating === take.rating ? 0 : rating,
    });
    await reloadSequence();
    renderCandidates();
  });
  info.append(stars);

  const seedLabel = document.createElement("span");
  seedLabel.className = "seed";
  seedLabel.innerHTML = take.seed === null || take.seed === undefined
    ? `${iconHTML("clock", "sm")}${take.seconds || "?"}s`
    : `${iconHTML("sparkles", "sm")}${take.seed}`;
  info.append(seedLabel, Object.assign(document.createElement("span"), { className: "spacer" }));

  // Keep segment and lineage playback together for one-click review. Both
  // use the timeline resolver, so joins play exactly as they will in the edit.
  const playSeg = document.createElement("button");
  playSeg.className = "small iconbtn ghost";
  playSeg.title = "Play only this segment";
  playSeg.innerHTML = iconHTML("play", "sm");
  playSeg.onclick = guard(async (e) => {
    e.stopPropagation();
    await playSegment(take);
  });
  info.append(playSeg);

  if (lineageTo(take).length > 1 || hasContinuations(take)) {
    const playChain = document.createElement("button");
    playChain.className = "small iconbtn ghost";
    playChain.title = "Play lineage — the whole chain up to and including this take";
    playChain.setAttribute("aria-label", "Play lineage");
    playChain.innerHTML = iconHTML("git-branch", "sm");
    playChain.onclick = guard(async (e) => {
      e.stopPropagation();
      await playLineage(take);
    });
    info.append(playChain);
  }

  // Compare stays on the card rather than going into the overflow: it is a
  // selection toggle used in bursts across a whole sweep, and a toggle behind a
  // menu has to be opened once per card to do the one job it has.
  const compare = document.createElement("button");
  compare.className = "small iconbtn ghost";
  const selected = state.compare.has(take.id);
  compare.title = selected ? "Deselect" : "Select to compare";
  compare.innerHTML = iconHTML(selected ? "check-square" : "columns", "sm");
  compare.onclick = () => {
    if (state.compare.has(take.id)) state.compare.delete(take.id);
    else if (state.compare.size < MAX_COMPARE) state.compare.add(take.id);
    else toast(`Compare takes at most ${MAX_COMPARE} at once.`);
    renderCandidates();
  };
  info.append(compare);

  // The occasional ones, given words instead of glyphs. Inspect is the least
  // missed of them: the pill row above the footer already opens the same
  // inspector, and does it with a target the size of the card.
  info.append(menuButton("more", "More actions for this take", () => {
    const items = [
      { icon: "file-text", text: "Inspect",
        title: "Properties & prompt diff",
        onclick: () => openTakeInspector(take, source) },
      { icon: "copy", text: "Copy context for agent",
        title: "Copy project, shot, clip, file location, prompt and generation details",
        onclick: () => copyTakeContext(take, source) },
      // The way out of here. A take bound for Resolve or Premiere is already a
      // file on the same disk ComfyUI wrote it to, so the shortest export is
      // the file manager opened on it, ready to drag.
      { icon: "folder-open", text: "Show in Explorer",
        title: "Open this take's folder with the file selected",
        onclick: async () => {
          const { path, opened } = await post(`${API}/reveal`, {
            filename: take.filename,
            subfolder: take.subfolder || "",
            type: take.type || "output",
          });
          // No file manager to open — a ComfyUI on another machine, typically.
          // The path is still the answer; hand it over as text.
          if (!opened) {
            await navigator.clipboard.writeText(path).catch(() => {});
            toast(`No file manager here — path copied: ${path}`);
          }
        } },
    ];
    // Copy this take's seed into the render bar's pin, so the next render of
    // the AUTHORED prompt starts from the same noise. This is the bridge the
    // re-render actions otherwise lack: they all re-send this take's frozen
    // text, so none of them can carry a seed onto an edited prompt.
    if (take.seed !== null && take.seed !== undefined) {
      items.push({ separator: true }, {
        icon: "link", text: `Pin seed ${take.seed}`,
        title: "Reuse this seed for the next render of the authored prompt — " +
          "keeps this composition while you tweak the prompt",
        onclick: () => {
          state.seedPin = take.seed;
          renderGenActions(currentSegment());
          toast(`Seed ${take.seed} pinned — the next render will reuse it`);
        },
      });
    }
    return items;
  }));

  // Hear this take under whatever picture the edit plays. This is the gesture
  // the whole feature exists for: a re-roll you rejected for its picture and
  // kept for its sound. Only offered while a clip on this source is selected —
  // sound belongs to a use of a lineage, not to the lineage.
  const editing = currentClip();
  if (editing && editing.source_id === source.id) {
    const heard = clipReport(editing.id)?.audio ?? null;
    const hearing = Boolean(heard?.borrowed) && heard.take === take.id;
    const sound = document.createElement("button");
    sound.className = `small iconbtn ghost${hearing ? " sounding" : ""}`;
    sound.title = hearing
      ? "Stop hearing the clip from this take — back to its own sound"
      : "Use this take's sound under the picture the edit plays";
    sound.innerHTML = iconHTML(hearing ? "volume-x" : "volume", "sm");
    sound.onclick = guard(async (event) => {
      event.stopPropagation();
      await editSound({ muted: false, take: hearing ? null : take.id });
    });
    info.append(sound);
  }

  const discard = document.createElement("button");
  discard.className = "small iconbtn ghost";
  discard.title = take.discarded ? "Restore" : "Discard";
  discard.innerHTML = iconHTML(take.discarded ? "rotate" : "trash", "sm");
  discard.onclick = guard(async () => {
    await post(`${API}/gallery/annotate`, {
      project: state.project.name, take: take.id, discarded: !take.discarded,
    });
    await reloadSequence();
    renderCandidates();
  });
  info.append(discard);

  // Choosing a take sets the whole path that ends at it, carried on down the
  // chain. Choosing the one already chosen clears it.
  //
  // Only the chosen one is filled. A grid of twenty takes used to be a grid of
  // twenty identical solid buttons, which made the choice itself invisible —
  // the card already in the edit looked like every card offering to be. Here
  // the offer is a quiet outline and the state wears the colour.
  const pick = document.createElement("button");
  pick.className = `small pick${chosen === take.id ? " in-edit" : ""}`;
  pick.innerHTML = chosen === take.id
    ? `${iconHTML("check", "sm on")}${iconHTML("x", "sm off")}<span>In the edit</span>`
    : label("check", "Use this", "sm");
  pick.title = chosen === take.id
    ? "This is the take the edit plays — click to clear the choice"
    : "Make this the take the edit plays";
  pick.onclick = guard(async () => {
    await setPath(source.id, chosen === take.id ? "" : take.id);
    renderCandidates();
  });
  info.append(pick);

  // -- row two: the presses that spend GPU minutes -------------------------
  //
  // Re-shoot this take from its own frozen prompt (not the segment's, which may
  // have drifted). Again explores a fresh seed; the split button beside it
  // reproduces this exact take at a tier you choose — same composition, more
  // megapixels and steps.
  //
  // A continuation needs the extend path instead: a plain reshoot would drop the
  // motion context from its parent, so the parent take is the source video and
  // this take's frozen prompt is the continuation text. The parent comes from
  // `take.parent_take` alone — the renderer's own record, never a guess.
  const parentTake = take.parent_take ? takeById(take.parent_take) : null;
  const isExtensionTake = parentTake && isVideoTake(parentTake);
  const takeDepth = Math.max(0, lineageTo(take).length - 1);

  // These re-render, so the render bar's hold toggle governs them exactly as it
  // governs the Render button — they share the dispatch funnel. Icon-only
  // buttons cannot rename themselves the way that one does, so they take the
  // armed ring and say it in words in their tooltip instead.
  const heldNote = state.holdRenders
    ? " — held for later (Hold is on), not rendered now"
    : "";
  const armed = (el) => {
    el.classList.toggle("hold-armed", Boolean(state.holdRenders));
    return el;
  };

  // A reroll keeps the tier the take rendered at: rerolling a Fine keeper must
  // not quietly hand back a Quick draft. Only a take with no real tier recorded
  // (an old take, a refine) falls back to Quick.
  const rerollTier = QUALITY_LABELS[take.quality] ? take.quality : "preview";
  const reroll = document.createElement("button");
  reroll.className = "small iconbtn ghost";
  reroll.title = (isExtensionTake
    ? `Again — same prompt and parent, fresh seed, at ${QUALITY_LABELS[rerollTier]}`
    : `Again — same prompt, a fresh seed, at ${QUALITY_LABELS[rerollTier]}`) + heldNote;
  armed(reroll);
  reroll.innerHTML = iconHTML("rotate", "sm");
  reroll.onclick = guard(() => {
    if (isExtensionTake) {
      return runExtend(parentTake, source, take.prompt_text, { quality: rerollTier, depth: takeDepth, label: "Reroll", context: take.context_length });
    }
    return reshoot(take, renderTarget(), { quality: rerollTier });
  });
  actions.append(reroll);

  // Re-render this take at a chosen tier. Turbo HD and Fine used to be a button
  // each here, which is why the two-pass keeper never got one: a tier per button
  // means every card grows by one every time a tier is added. As a menu the
  // fourth tier costs nothing, and the tiers arrive with their names and their
  // real cost written out rather than encoded in a glyph.
  //
  // An extend cannot run the two-pass transform yet (see QUALITY_LEVELS), so a
  // continuation is offered the three tiers its own endpoint can honour — the
  // menu must not queue a single-pass render under a two-pass name.
  const rerenderTiers = isExtensionTake ? QUALITY_LEVELS : TIERS;
  const stickyTier = rerenderTiers.includes(state.reshootTier) ? state.reshootTier : "final";
  const seedNote = take.seed !== null && take.seed !== undefined
    ? `same prompt and seed ${take.seed}` : "same prompt (no seed recorded)";

  const rerenderAt = (level) => {
    if (isExtensionTake) {
      return runExtend(parentTake, source, take.prompt_text,
                       { seed: take.seed, quality: level, depth: takeDepth,
                         label: QUALITY_LABELS[level] ?? "Fine",
                         context: take.context_length });
    }
    return reshoot(take, renderTarget(), { quality: level, seed: take.seed });
  };

  // A split button, not a plain menu: the escalation people actually make is
  // "this one, properly", and burying the commonest press one click deeper to
  // reach the rarer ones would be a bad trade. The left half repeats whatever
  // tier was last chosen here, which starts at Fine — exactly what the old Fine
  // button on this card did.
  const split = document.createElement("span");
  split.className = `splitbtn${state.holdRenders ? " hold-armed" : ""}`;
  const rerenderBtn = document.createElement("button");
  rerenderBtn.className = "small iconbtn ghost";
  rerenderBtn.title = `Render this take again at ${QUALITY_LABELS[stickyTier]} — ` +
    `${seedNote}, ${tierNumbers(stickyTier)}${heldNote}` +
    `\n\n${TIER_BLURBS[stickyTier] ?? ""}`;
  rerenderBtn.innerHTML = iconHTML(TIER_ICONS[stickyTier] ?? "maximize", "sm");
  rerenderBtn.onclick = guard(async () => {
    await rerenderAt(stickyTier);
  });
  split.append(rerenderBtn, menuButton("chevron-down", "Render this take at another tier", () => [
    { heading: "Render again at" },
    ...rerenderTiers.map((level) => ({
      icon: TIER_ICONS[level] ?? "maximize",
      text: QUALITY_LABELS[level] ?? level,
      note: tierNumbers(level),
      title: `${seedNote}.\n\n${TIER_BLURBS[level] ?? ""}${heldNote}`,
      current: level === stickyTier,
      onclick: async () => {
        state.reshootTier = level;
        await rerenderAt(level);
        renderCandidates();
      },
    })),
  ]));
  actions.append(split);

  // Extend: a new clip that continues this one, motion and sound carrying over.
  // Only offered on video takes — there is nothing to continue from a still.
  if (isVideoTake(take)) {
    const extend = document.createElement("button");
    extend.className = "small iconbtn ghost";
    extend.title = "Extend — generate a continuation of this take" + heldNote;
    armed(extend);
    extend.innerHTML = iconHTML("chevron-right", "sm");
    extend.onclick = guard(() => openExtendField(take, source, card));
    actions.append(extend);

    // SeedVR2 enhancement, with face blending and existing H3 options in the menu.
    const refineSplit = document.createElement("span");
    refineSplit.className = `splitbtn${state.holdRenders ? " hold-armed" : ""}`;
    const refineBtn = document.createElement("button");
    refineBtn.className = "small iconbtn ghost";
    refineBtn.title = "Enhance Whole Video — Original Size (SeedVR2). Changes the full scene and saves a new take." + heldNote;
    armed(refineBtn);
    refineBtn.innerHTML = iconHTML("sparkles", "sm");
    refineBtn.onclick = guard(() => runRefine(take, renderTarget(), { enhancement_mode: "seedvr2", enhancement_resolution: "original", guided_resample: false, face_refine: false, megapixels: null }));
    refineSplit.append(refineBtn, menuButton("chevron-down", "Refinement options", () => [
      { heading: "SeedVR2 Enhancement" },
      {
        icon: "sparkles",
        text: "Enhance Whole Video — Original Size",
        note: "SeedVR2 · changes the whole scene",
        title: "Restore the full scene with SeedVR2, then return to the original dimensions. Can change faces, clothing and background detail. Saves a new take with original audio and timing.",
        onclick: () => runRefine(take, renderTarget(), { enhancement_mode: "seedvr2", enhancement_resolution: "original", guided_resample: false, face_refine: false, megapixels: null }),
      },
      {
        icon: "maximize",
        text: "Enhance Whole Video — Upscale",
        note: "SeedVR2 · 1080p target, keeps larger sources",
        title: "Restore the full scene and enlarge toward a 1080-pixel short edge, capped at 2048 on the long edge. Keeps larger sources at their original size. Saves a new take with original audio and timing.",
        onclick: () => runRefine(take, renderTarget(), { enhancement_mode: "seedvr2", enhancement_resolution: "1080p", guided_resample: false, face_refine: false, megapixels: null }),
      },
      {
        icon: "sparkles",
        text: "Enhance Faces Only — Blend",
        note: "SeedVR2 · original size · experimental",
        title: "Run SeedVR2 on the full scene, then blend only detected small faces into the original using soft masks. Preserves surrounding pixels before video encoding. Missed faces stay unchanged; blend edges and motion need review. Saves a new take.",
        onclick: () => runRefine(take, renderTarget(), { enhancement_mode: "seedvr2_faces", enhancement_resolution: "original", guided_resample: false, face_refine: false, megapixels: null }),
      },
      { heading: "H3 Refinement" },
      {
        icon: "sparkles",
        text: "Refine & Upscale to 1080p (Full)",
        note: "1080p + small-face repair",
        title: "Enhance textures, detect faces on enlarged tiles, upscale tiny face crops before H3 repair, then blend directly into the 2.1 MP output. Processing time increases with the number of faces.",
        onclick: () => runRefine(take, renderTarget(), { guided_resample: true, face_refine: true, megapixels: 2.1 }),
      },
      {
        icon: "maximize",
        text: "Refine at Native Resolution",
        note: "Textures + small-face repair",
        title: "Enhance textures and refine small faces across the scene at source resolution. Each face is tracked separately; close-ups are left out of the face pass.",
        onclick: () => runRefine(take, renderTarget(), { guided_resample: true, face_refine: true, megapixels: null }),
      },
      {
        icon: "zoom-in",
        text: "Guided Texture only (1080p)",
        note: "Upscale 1080p",
        title: "Upscale to 1080p with guided texture enhancement (no face detection pass).",
        onclick: () => runRefine(take, renderTarget(), { guided_resample: true, face_refine: false, megapixels: 2.1 }),
      },
      {
        icon: "sparkles",
        text: "Fix Small Faces (1080p)",
        note: "Upscale tiny faces before repair",
        title: "Detect small faces on enlarged overlapping tiles, restore tiny crops with RealESRGAN before H3 refinement, then blend into a 2.1 MP output. Preserves audio. Very tiny or obscured faces may still be missed.",
        onclick: () => runRefine(take, renderTarget(), { guided_resample: false, face_refine: true, megapixels: 2.1 }),
      },
      {
        icon: "sparkles",
        text: "Fix Small Faces (Native)",
        note: "Track and repair each small face",
        title: "Detect faces on enlarged tiles and refine faces 8–192 source pixels tall at native output resolution. Preserves audio and skips close-ups. See the Face Refine Report for detected coverage.",
        onclick: () => runRefine(take, renderTarget(), { guided_resample: false, face_refine: true, megapixels: null }),
      },
      {
        icon: "sparkles",
        text: "Refine Largest Face Only (Native)",
        note: "Single-face mode",
        title: "Refine just the largest face, with adaptive close-up protection.",
        onclick: () => runRefine(take, renderTarget(), { guided_resample: false, face_refine: true, megapixels: null, face_target: "largest_face" }),
      },
    ]));
    actions.append(refineSplit);
  }

  if (parentTake) {
    const parentLabel = parentTake.seed != null ? `Seed #${parentTake.seed}` : parentTake.filename;
    const line = document.createElement("div");
    line.className = "lineage";
    line.innerHTML = `${iconHTML("link", "sm")}<span>From ${escapeHtml(parentLabel)}</span>`;
    line.title = `Continues ${take.parent_take}`;
    actions.append(line);
  } else if (take.parent_take) {
    // The parent was recorded but is not among this project's takes — deleted,
    // or filed elsewhere. Say so rather than drawing the chain as unbroken.
    const line = document.createElement("div");
    line.className = "lineage";
    line.innerHTML = `${iconHTML("alert-triangle", "sm")}<span>Parent take is missing</span>`;
    line.title = `Continues ${take.parent_take}, which is not recorded here`;
    actions.append(line);
  }

  card.append(info, actions);
  return card;
}

/** Change what the selected clip is heard as. Picture and trim are untouched. */
async function editSound(patch) {
  const clip = currentClip();
  if (!clip) return;
  await apply(post(`${API}/clip/edit`, {
    project: state.project.name, clip: clip.id, audio: patch,
  }));
}

// Choose the lineage a source plays. `takeId` names one take; the server walks
// up through each take's recorded parent to build the whole path, then carries
// it on down the chain preferring what was remembered. An empty id clears it.
//
// `cascadeClientPicks` used to live here, mirroring a Python cascade of the
// same name — the same guessing written twice, in two languages, drifting apart
// between them. Both are gone: a path names its own parent at every step, so
// there is nothing downstream to realign.
async function setPath(sourceId, takeId) {
  const result = await post(`${API}/gallery/pick`, {
    project: state.project.name, source: sourceId, take: takeId,
  });
  state.project = result.project;
  await reloadSequence();
  fillForm();
}

/** Set a source's path outright, as an ordered list of take ids. */
async function setPathExactly(sourceId, path) {
  try {
    const result = await post(`${API}/source/path`, {
      project: state.project.name, source: sourceId, path,
    });
    state.project = result.project;
  } catch (error) {
    // A 409 carries the reasons the chain does not hold up. Showing them beats
    // storing a path the export would choke on.
    const problems = error.body?.problems;
    toast(problems?.length ? problems[0] : error.message, true);
    return false;
  }
  await reloadSequence();
  fillForm();
  return true;
}

// -- full-screen take compare (extracted to ./compare-modal.js) -----------

// Step through the edit, not the bin: next means the next clip on the timeline,
// which is the order the author is thinking in. A source with no clip is
// reachable from the bin instead.
function editorNav(delta) {
  const order = clips();
  const index = order.findIndex((clip) => clip.id === state.selectedClip);
  const next = order[index + delta];
  if (next) openEditor(next.source_id, next.id);
}

// -- one player at a time -------------------------------------------------
//
// Several parts of the editor own a <video>: the monitor, the take cards, the
// inspect view, the queue's job sheet, the lightbox. Nothing stopped one when
// another started, so two takes talked over each other, and a sheet closed
// mid-clip went on being heard from behind the page — a media element keeps
// playing after it is hidden, and after it is removed from the document.
//
// The rule is that starting a sound stops every other sound. Silent players are
// exempt in both directions: the hover previews and the thumbnail loops are
// moving pictures rather than audio, and freezing the whole grid every time
// something plays would take away more than it fixes.

// The monitor is three elements playing as one — two picture buffers and the
// borrowed sound track — so one of them starting must not silence its partners.
// Its buffers also run muted whenever the sound is borrowed, which is why
// membership is decided by id and not by whether the element makes a noise.
const MONITOR_MEDIA = new Set(["player-video", "player-video-b", "player-audio"]);

const isMonitorMedia = (node) => MONITOR_MEDIA.has(node.id);

/** Pause every player except `keep` — and, when `keep` is one of the monitor's,
 *  the rest of the monitor along with it. */
function hushOthers(keep) {
  const sparing = Boolean(keep && isMonitorMedia(keep));
  // The monitor is a transport, not just an element: pausing its buffers behind
  // its back would leave the button reading "pause" and the playhead armed.
  if (!sparing && state.player?.playing) stopPlayer();
  for (const node of document.querySelectorAll("video, audio")) {
    if (node === keep || node.paused) continue;
    if (isMonitorMedia(node)) {
      if (!sparing) node.pause();
      continue;
    }
    if (node.muted) continue;
    node.pause();
  }
}

/** Pause every player inside a panel that is closing, or about to be redrawn.
 *  Hiding the panel or emptying it does not stop what is playing in it. */
function hushWithin(root) {
  if (!root) return;
  for (const node of root.querySelectorAll("video, audio")) {
    if (!node.paused) node.pause();
  }
}

// -- thumbnails that are not always there ---------------------------------
//
// Nearly every <video> on this page is showing a still frame. The bin has one
// per source, the timeline one per clip, the take sheet one per candidate, the
// queue one per historical render -- and a working project is forty sources,
// thirty clips and eight hundred takes. Built plainly that is well past the
// seventy-odd media players a browser will hand out to a single page, and the
// ones past the limit do not fail loudly: they come back blank. A bin full of
// black rectangles while a render was running was this, not a slow disk.
//
// Two rules keep the count down. A thumbnail loads only while it is on screen,
// and one that leaves the screen hands its player back. Handing it back has to
// be said out loud -- dropping the element out of the document is not enough,
// the player outlives it until the collector runs, so a redraw doubles the
// count all on its own for as long as that takes.

/** Give a media element's player back to the browser. */
function releaseMedia(node) {
  if (!node || !node.getAttribute("src")) return;
  try {
    node.pause();
    node.removeAttribute("src");
    node.load();
  } catch { /* already torn down */ }
}

/** Empty a panel's players rather than merely quieting them. For the places
 *  that are about to wipe their contents, where `hushWithin` leaves behind
 *  exactly the players this page cannot spare. */
function releaseWithin(root) {
  if (!root) return;
  for (const node of root.querySelectorAll("video, audio")) {
    if (isMonitorMedia(node)) { if (!node.paused) node.pause(); continue; }
    releaseMedia(node);
  }
}

// A screen's worth of margin either way, so scrolling the bin finds pictures
// already there rather than watching them arrive.
const thumbSeen = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const node = entry.target;
    if (entry.isIntersecting) {
      const want = node.dataset.src;
      if (want && node.getAttribute("src") !== want) {
        node.setAttribute("src", want);
        // Back to where it was left. A take sheet is scrolled through while a
        // card is half-watched, and coming back to it rewound would be a worse
        // trade than the player it cost to avoid.
        const at = Number(node.dataset.at);
        if (at > 0) {
          node.addEventListener("loadedmetadata", () => {
            try { node.currentTime = at; } catch { /* seek refused */ }
          }, { once: true });
        }
      }
      continue;
    }
    // A redraw takes these out of the document for a moment before putting
    // them back, and the callback for that can land after they are already
    // home. Only a thumbnail still on the page has genuinely scrolled away.
    if (!node.isConnected) continue;
    // Never take the picture out from under someone watching it: the take
    // sheet's cards carry real controls, and scrolling a playing one half out
    // of view is an ordinary thing to do.
    if (!node.paused) continue;
    // One that was watched and paused gives its player back like any other,
    // and is noted down so it can be put back where it was. Keeping it loaded
    // instead would mean a sheet of a hundred and sixty takes creeping back
    // over the limit one played card at a time.
    if (node.dataset.keep === "1" && node.currentTime > 0) {
      node.dataset.at = String(node.currentTime);
    }
    releaseMedia(node);
  }
}, { rootMargin: "300px" });

/** A thumbnail that fetches nothing until it is looked at. */
function lazyThumb(url, className) {
  const node = document.createElement("video");
  if (className) node.className = className;
  node.preload = "metadata";
  node.muted = true;
  node.playsInline = true;
  node.dataset.src = url;
  // Noting that this one has been watched, which is what decides whether its
  // position is worth remembering when it scrolls away.
  node.addEventListener("play", () => { node.dataset.keep = "1"; }, { once: true });
  thumbSeen.observe(node);
  return node;
}

// The bin and the timeline redraw for reasons that have nothing to do with
// their pictures -- a badge landing, a selection moving -- and both rebuild
// every row from scratch when they do. Keyed by what the thumbnail is OF, so a
// redraw is handed back the element already holding that picture instead of
// starting seventy downloads over.
const heldThumbs = new Map();

function heldThumb(key, url, className) {
  const held = heldThumbs.get(key);
  if (held && held.url === url) {
    thumbSeen.observe(held.node);
    return held.node;
  }
  if (held) { thumbSeen.unobserve(held.node); releaseMedia(held.node); }
  const node = lazyThumb(url, className);
  heldThumbs.set(key, { url, node });
  return node;
}

/** Let go of every held thumbnail the last redraw did not put back. */
function sweepThumbs() {
  for (const [key, held] of heldThumbs) {
    if (held.node.isConnected) continue;
    thumbSeen.unobserve(held.node);
    releaseMedia(held.node);
    heldThumbs.delete(key);
  }
}

let sweepQueued = false;
/** After a redraw, once the page has settled and it is clear what stayed.
 *
 *  A timer rather than an animation frame: a batch of renders landing while
 *  the editor sits in a background tab still redraws the bin over and over,
 *  and a tab that is not being painted is handed no frames at all -- so the
 *  sweep would never run, and coming back to the tab would mean coming back to
 *  every player the last hour of redraws had left behind. */
function queueThumbSweep() {
  if (sweepQueued) return;
  sweepQueued = true;
  setTimeout(() => { sweepQueued = false; sweepThumbs(); }, 0);
}

/** Media events do not bubble, so this has to listen on the way down. */
function watchForSound() {
  document.addEventListener("play", (event) => {
    const node = event.target;
    if (!(node instanceof HTMLMediaElement)) return;
    if (node.muted && !isMonitorMedia(node)) return;
    hushOthers(node);
  }, true);
}

// -- the sequence player --------------------------------------------------

// One <video> stepped through the resolved pieces in order, driven by the
// timeline playhead. A visible seam at a cut is honest — this is an assembly,
// not the export.
// The monitor plays whatever list it is handed. "timeline" is the edit itself —
// it drives the playhead and follows the selection. "preview" is a throwaway
// list (one lineage, one segment) that must not move either, so reviewing a
// branch never disturbs the edit you are reviewing it against.
function loadPlaylist(list, { mode = "timeline", label = "", audio = [] } = {}) {
  state.player = { list, index: 0, playing: false, active: 0, mode, label, audio };
  showBuffer(0);
  // Whatever the last list was heard as does not apply to this one.
  pauseBorrowedSound();
  const has = list.length > 0;
  $("tl-play").disabled = !has;
  $("tl-play-clip").disabled = !has;
  if (!has) $("player-position").textContent = "—";
  return list;
}

// The edit, as files to play. Resolved on the server and handed over whole:
// a clip is a range of lineage time, so one clip can span two takes, and the
// monitor must cut at exactly the frame the export does. Doing that arithmetic
// here again would be a second opinion about where a join falls.
function buildPlaylist() {
  const list = (state.report?.playlist ?? []).map((piece) => ({
    id: piece.clip,
    source: piece.source,
    name: piece.name,
    takeId: piece.take_id,
    url: piece.url,
    from: piece.from,
    to: piece.to,
    speed: piece.speed || 1,
    length: piece.length,
    start: piece.start,
  }));

  // The server sends only the sound that differs from the picture — a clip
  // playing its own generation is already being decoded for the screen.
  loadPlaylist(list, { mode: "timeline", audio: state.report?.audio ?? [] });
  const covered = list.reduce((total, entry) => total + entry.length, 0);
  const total = state.report?.duration ?? 0;
  const played = new Set(list.map((entry) => entry.id)).size;
  $("player-stat").textContent = list.length
    ? `${played}/${clips().length} clips · ${seconds(covered)} of ${seconds(total)}`
    : "nothing to play yet";
  return list;
}

// Move the one program monitor into the take sheet and back out again. The
// sheet covers the timeline, so a preview played in place would run behind it;
// relocating the stage beats standing up a second player, which would mean two
// copies of the buffering and two ways a join could behave.
//
// The move happens while the buffers are paused — re-inserting a playing
// <video> is what makes it stall — so callers dock before they play.
function dockPreview(open) {
  const stage = $("player-stage");
  const dock = $("preview-dock");
  const monitor = document.querySelector(".monitor");
  if (!stage || !dock || !monitor) return;

  if (open) {
    if (stage.parentElement !== dock) dock.insertBefore(stage, dock.firstChild);
    dock.classList.add("open");
    applyDockHeight(savedDockHeight());
  } else {
    if (stage.parentElement !== monitor) {
      monitor.insertBefore(stage, monitor.querySelector(".monitor-bar"));
    }
    dock.classList.remove("open");
    // The docked height is an inline style on the shared stage, so it has to
    // come off or the program monitor inherits the dock's size.
    stage.style.height = "";
  }
}

// How tall the docked preview is. Floor keeps it watchable; ceiling keeps at
// least a couple of take cards on screen, which is the point of docking here.
const DOCK_MIN_PX = 140;
const dockCeiling = () => Math.round(window.innerHeight * 0.72);
const clampDock = (px) => Math.max(DOCK_MIN_PX, Math.min(Math.round(px), dockCeiling()));

function savedDockHeight() {
  const saved = parseInt(localStorage.getItem("vp:previewDockHeight") ?? "", 10);
  return Number.isFinite(saved) ? saved : Math.min(340, Math.round(window.innerHeight * 0.38));
}

function applyDockHeight(px) {
  const stage = $("player-stage");
  if (stage) stage.style.height = `${clampDock(px)}px`;
}

function startDockResize(event) {
  event.preventDefault();
  const stage = $("player-stage");
  const startY = event.clientY;
  const startHeight = stage.getBoundingClientRect().height;

  const move = (e) => applyDockHeight(startHeight + (e.clientY - startY));
  const up = () => {
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
    localStorage.setItem(
      "vp:previewDockHeight",
      String(Math.round(stage.getBoundingClientRect().height)),
    );
  };
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up);
}

// Play a one-off list, then hand the monitor back. Preview playback is a
// detour, so the edit it interrupted has to be exactly where it was left.
function playPreview(list, label) {
  if (!list.length) return toast("Nothing to play there yet.", true);
  stopPlayer();
  if ($("editor").classList.contains("open")) dockPreview(true);
  $("preview-dock-label").textContent = label;
  $("monitor-mode").textContent = "Take preview";
  $("monitor-return").hidden = false;
  loadPlaylist(list, { mode: "preview", label });
  playFrom(0);
}

/** Put the timeline back under the monitor if a preview has taken it over. */
function ensureTimeline() {
  // Cancel even an unresolved preview: collapsing review or seeking the edit
  // must win over a late lineage response that has not begun playing yet.
  state.previewRequest = (state.previewRequest || 0) + 1;
  if (state.player?.mode !== "preview") return;
  stopPlayer();
  dockPreview(false);
  buildPlaylist();
  // The readout still names the preview that just finished; leaving it there
  // would label the edit with a detour it is no longer playing.
  $("player-position").textContent = "—";
  $("monitor-mode").textContent = "Sequence preview";
  $("monitor-return").hidden = true;
  seekTo(state.playheadSec || 0);
}

// The chain ending at `take`, oldest first, walked from recorded parents only.
// Nothing here is inferred: if a link was never written down, the walk stops
// rather than reaching for the take that looks about right.
function lineageTo(take) {
  const path = [];
  const seen = new Set();
  let current = take;
  while (current && !seen.has(current.id)) {
    seen.add(current.id);
    path.unshift(current.id);
    current = current.parent_take ? takeById(current.parent_take) : null;
  }
  return path;
}

const hasContinuations = (take) =>
  (state.takes ?? []).some((other) => other.parent_take === take.id && !other.discarded);

async function playLineage(take) {
  const request = state.previewRequest = (state.previewRequest || 0) + 1;
  const epoch = state.projectEpoch;
  const path = lineageTo(take);
  const pieces = await resolveLineage(path);
  if (request !== state.previewRequest || epoch !== state.projectEpoch) return;
  playPreview(pieces, `Lineage ${path.length}×`);
}

async function playSegment(take) {
  // Resolved in the context of its whole lineage, then narrowed to this take:
  // one code path decides where a segment starts and ends, everywhere.
  const epoch = state.projectEpoch;
  const source = state.selectedSource;
  const request = state.previewRequest = (state.previewRequest || 0) + 1;
  const pieces = await resolveLineage(lineageTo(take));
  if (epoch !== state.projectEpoch || source !== state.selectedSource || request !== state.previewRequest) return;
  const mine = pieces.filter((piece) => piece.id === take.id);
  const base = mine[0]?.start ?? 0;
  playPreview(
    mine.map((piece) => ({ ...piece, start: piece.start - base })),
    "Segment",
  );
}

/** Ask the server which files cover a range of a lineage, as a playlist. */
async function resolveLineage(path, { start, end } = {}) {
  const body = { project: state.project?.name, path };
  if (start !== undefined) body.start = start;
  if (end !== undefined) body.end = end;
  const result = await post(`${API}/lineage/resolve`, body);
  if (!result) return [];
  if (result.problems?.length) toast(result.problems[0], true);
  return (result.playlist ?? []).map((piece, index) => ({
    id: piece.take_id,
    name: `${index + 1} · depth ${piece.depth}`,
    url: piece.url,
    from: piece.from,
    to: piece.to,
    speed: 1,
    length: piece.length,
    start: piece.start,
  }));
}

function playFrom(index, source) {
  const player = state.player;
  if (!player || !player.list.length) return;
  if (index >= player.list.length) {
    stopPlayer();
    // A finished preview gives the monitor back rather than leaving the edit
    // stranded behind a list the viewer has already watched.
    if (player.mode === "preview") ensureTimeline();
    return;
  }

  player.index = index;
  player.playing = true;
  const entry = player.list[index];
  const videos = playerVideos();

  // At a natural cut (no explicit source) the idle buffer already holds this
  // clip, decoded and parked on its in-point — cut to it. A fresh start or a
  // seek-resume has no such buffer, so load into the active one.
  const idle = videos[1 - player.active];
  const useIdle = source == null && idle.dataset.url === entry.url;
  const incoming = useIdle ? idle : videos[player.active];

  primeVideo(incoming, entry, source);
  incoming.play().catch(() => {});
  if (useIdle) {
    videos[player.active].pause();
    player.active = 1 - player.active;
  }
  showBuffer(player.active);
  preloadNext(index);
  // Where the picture is actually starting, which is not the entry's start when
  // a seek dropped us into the middle of one.
  applySound(
    entry.start + Math.max(0, ((source ?? entry.from) - entry.from) / (entry.speed || 1)),
    true,
  );
  $("tl-play").innerHTML = iconHTML("pause", "fill");
  const prefix = player.mode === "preview" ? `${player.label} — ` : "";
  const readout = `${prefix}${index + 1}/${player.list.length} · ${entry.name}`;
  $("player-position").textContent = readout;
  // While docked the monitor bar is off-screen with the timeline, so the dock
  // carries the readout instead.
  if (player.mode === "preview") $("preview-dock-label").textContent = readout;

  // A preview's entries are takes, not clips: selecting one would drag the
  // editor to a source the viewer never asked to open. Playing the edit does
  // follow along, but only when the playhead crosses into a different clip —
  // the pieces inside one clip are a join, not a cut.
  if (player.mode === "timeline" && entry.id !== state.selectedClip) {
    state.selectedClip = entry.id;
    state.selectedSource = entry.source;
    fillForm();
    renderBin();
    renderTimeline();
  }
}

function stopPlayer() {
  playerVideos().forEach((video) => video.pause());
  pauseBorrowedSound();
  if (state.player) state.player.playing = false;
  $("tl-play").innerHTML = iconHTML("play", "fill");
}

function togglePlay() {
  let player = state.player;
  if (player?.playing) return stopPlayer();
  // The transport button belongs to the edit: pressing it after a preview
  // resumes the timeline, not the detour.
  ensureTimeline();
  player = state.player;
  if (!player?.list.length) return;
  // Resume from the playhead when it sits on a playable shot, else from start.
  const hit = locate(state.playheadSec);
  if (hit) playFrom(hit.index, hit.source);
  else playFrom(0);
}

function watchPlayer() {
  for (const video of playerVideos()) {
    video.ontimeupdate = () => {
      const player = state.player;
      if (!player) return;
      // Only the visible buffer drives the playhead; the idle one is seeking
      // its preload and must not hijack the clock.
      if (video !== playerVideos()[player.active]) return;
      const entry = player.list[player.index];
      if (!entry) return;
      // Track the playhead against the running total whether playing or
      // scrubbing — but only for the edit. A preview borrows the monitor, not
      // the timeline, so it must leave the playhead where it found it.
      if (player.mode === "timeline") {
        const into = (video.currentTime - entry.from) / entry.speed;
        state.playheadSec = entry.start + Math.max(0, into);
        positionPlayhead();
        // Two elements playing at the same rate do not stay together on their
        // own — different decode paths, different keyframes — so the picture's
        // clock pulls the borrowed track back whenever it has slipped.
        applySound(state.playheadSec, player.playing);
      }

      if (!player.playing) return;
      // The out point is the cut. Waiting for `ended` would play the tail the
      // edit trims away.
      if (entry.to !== null && video.currentTime >= entry.to - 0.03) {
        playFrom(player.index + 1);
      }
    };
    video.onended = () => {
      const player = state.player;
      if (player?.playing && video === playerVideos()[player.active]) {
        playFrom(player.index + 1);
      }
    };
  }
}

// -- zoom -----------------------------------------------------------------

function setZoom(next) {
  const total = state.report?.duration ?? 0;
  const fit = fitPixelsPerSecond(total);
  // Keep the resulting scale inside the clamp so the buttons never no-op
  // silently at the extremes.
  state.zoom = Math.max(MIN_PPS / fit, Math.min(MAX_PPS / fit, next));
  const pps = pixelsPerSecond();
  $("zoom-label").textContent = Math.abs(state.zoom - 1) < 0.02 ? "fit" : `${(pps / fit).toFixed(1)}×`;
  renderTimeline();
}

// -- export ---------------------------------------------------------------

async function exportSequence() {
  if (!state.project) return;
  await pushSource();

  let plan;
  try {
    plan = await post(`${API}/export`, {
      project: state.project.name, format: "ffmpeg", dry_run: true,
    });
  } catch (error) {
    if (error.status === 409) {
      toast(error.message, true);
      return;
    }
    throw error;
  }

  const missing = plan.clips.filter((clip) => !clip.exists).length;
  const blocked = !plan.ffmpeg_available || missing > 0;
  const short = plan.short_sources ?? [];
  // Silence is a legible result — you can hear it — so it is said, not refused.
  const silent = plan.silent ?? [];
  const unreadable = plan.unreadable_audio ?? [];
  const lines = [
    `${plan.pieces} piece(s) across ${clips().length} clip(s), ${seconds(plan.duration)}.`,
    plan.ffmpeg_available ? "" : "ffmpeg is not on PATH.",
    missing ? `${missing} take file(s) are not where the record says.` : "",
    short.length ? `Trimmed past the end of the file:\n  ${short.join("\n  ")}` : "",
    silent.length ? `Silent (muted, or nothing resolved):\n  ${silent.join("\n  ")}` : "",
    unreadable.length
      ? `Sound file has no audio track, or is missing:\n  ${unreadable.join("\n  ")}`
      : "",
    "",
    blocked
      ? "OK writes an EDL and a clip list instead."
      : "OK renders the video with ffmpeg. Cancel writes an EDL and a clip list instead.",
  ].filter(Boolean);

  if (!window.confirm(lines.join("\n")) || blocked) {
    const edl = await post(`${API}/export`, { project: state.project.name, format: "edl" });
    await post(`${API}/export`, { project: state.project.name, format: "shotlist" });
    toast(`Wrote ${edl.destination}`);
    return;
  }

  toast("Exporting — ffmpeg runs over every piece, so this takes a while.");
  const result = await post(`${API}/export`, {
    project: state.project.name, format: "ffmpeg",
  });
  toast(result.ok ? `Wrote ${result.destination}` : result.message, !result.ok);
}

// -- wiring ---------------------------------------------------------------

// A tab bar and its panels share the same container. Clicking a button shows
// the panel whose data-panel matches its data-tab.
function setupTabs(barId) {
  const bar = $(barId);
  const container = bar.parentElement;
  const buttons = bar.querySelectorAll("button[data-tab]");
  buttons.forEach((button) => {
    button.setAttribute("role", "tab");
    button.id = `author-tab-${button.dataset.tab}`;
    button.setAttribute("aria-controls", `author-panel-${button.dataset.tab}`);
    button.setAttribute("aria-selected", String(button.classList.contains("active")));
    button.tabIndex = button.classList.contains("active") ? 0 : -1;
    const targetPanel = container.querySelector(`.tabpanel[data-panel="${button.dataset.tab}"]`);
    targetPanel.id = `author-panel-${button.dataset.tab}`;
    targetPanel.setAttribute("role", "tabpanel");
    targetPanel.setAttribute("aria-labelledby", button.id);
    button.onclick = (event) => {
      buttons.forEach((other) => {
        other.classList.toggle("active", other === button);
        other.setAttribute("aria-selected", String(other === button));
        other.tabIndex = other === button ? 0 : -1;
      });
      editorUX?.navigate(button.dataset.tab, event?.isTrusted ? "manual" : "contextual");
      container.querySelectorAll(".tabpanel").forEach((panel) =>
        panel.classList.toggle("active", panel.dataset.panel === button.dataset.tab));
      if (barId === "author-tabs" && button.dataset.tab !== "scene") {
        $("editor").classList.remove("scene-focus");
        $("scene-focus").querySelector("span").textContent = "Focus";
      }
      // Fields that were hidden had no height to grow into; now they do.
      refitGrowFields();
    };
    button.onkeydown = (event) => {
      const order = [...buttons];
      const index = order.indexOf(button);
      const next = event.key === "Home" ? order[0] : event.key === "End" ? order.at(-1)
        : event.key === "ArrowRight" ? order[(index + 1) % order.length]
        : event.key === "ArrowLeft" ? order[(index + order.length - 1) % order.length] : null;
      if (next) { event.preventDefault(); next.focus(); next.onclick(event); }
    };
  });
}

/* -- the help text under each section heading ------------------------------
 *
 * Every section carries a paragraph saying what its fields are for and why
 * they exist. That prose is worth keeping — a lot of it is the only written
 * record of how H3 actually reads a field — but at full length between every
 * pair of inputs it buried the form it was explaining, and the author panel
 * read as one grey wall.
 *
 * So it folds. Each section gets a toggle in its own heading, the tab bar gets
 * one that flips them all, and the choice is remembered per section. Sections
 * open closed by default; nothing is lost, it is one click away, and the
 * default is what someone who has read it once wants.
 */
const HELP_KEY = "vp:sectionHelp";

const helpState = () => {
  try { return JSON.parse(localStorage.getItem(HELP_KEY) || "{}") || {}; }
  catch { return {}; }
};

const rememberHelp = (name, on) => {
  const state = helpState();
  state[name] = on;
  try { localStorage.setItem(HELP_KEY, JSON.stringify(state)); } catch { /* private mode */ }
};

/** A stable name for a section, so its choice survives a reload.
 *
 * The heading's own words only — `textContent` would drag in the label of
 * whatever button sits in the heading, and renaming "Add member" would then
 * quietly forget the choice. */
const helpName = (group) => {
  const title = group.querySelector(".block-title");
  if (!title) return "";
  return [...title.childNodes]
    .filter((node) => node.nodeType === Node.TEXT_NODE)
    .map((node) => node.textContent)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 40);
};

function setHelp(group, on) {
  group.classList.toggle("help-on", on);
  group.querySelector(".block-title .help-t")?.classList.toggle("on", on);
}

/** True when every section that has help to show is showing it. */
const allHelpOn = (groups) => groups.length > 0
  && groups.every((group) => group.classList.contains("help-on"));

function wireSectionHelp() {
  const groups = [...document.querySelectorAll(".author .group")]
    .filter((group) => group.querySelector(".group-body .note"));
  const remembered = helpState();
  const all = $("help-all");

  for (const group of groups) {
    const title = group.querySelector(".block-title");
    if (!title) continue;

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "help-t small";
    toggle.title = "Show or hide what this section is for";
    toggle.innerHTML = iconHTML("info", "sm");
    // The heading's spacer, if it has one, already pushes its own buttons
    // right; without one the toggle needs to push itself.
    if (!title.querySelector(".spacer")) title.append(spacer());
    title.append(toggle);

    toggle.onclick = () => {
      const on = !group.classList.contains("help-on");
      setHelp(group, on);
      rememberHelp(helpName(group), on);
      all?.classList.toggle("on", allHelpOn(groups));
      refitGrowFields();
    };

    setHelp(group, remembered[helpName(group)] === true);
  }

  if (all) {
    all.classList.toggle("on", allHelpOn(groups));
    all.onclick = () => {
      const on = !allHelpOn(groups);
      for (const group of groups) {
        setHelp(group, on);
        rememberHelp(helpName(group), on);
      }
      all.classList.toggle("on", on);
      refitGrowFields();
    };
  }
}

// -- writing with a model -------------------------------------------------
//
// Two ways in, one operation behind them. The bar at the top of the Source tab
// writes a whole prompt from a description; the small buttons beside each block
// and each shot rewrite one part of it from a note. Both post to /llm/draft
// with a scope, and the server merges only that scope back, so a tweak aimed at
// the soundscape cannot wander into the storyboard.
//
// The result applies straight away rather than arriving as a proposal to
// accept. That is only safe because every write files a revision server-side:
// the History button beside the bar is the undo, and it is per-segment, so
// walking one prompt back does not roll back the bin the way a project backup
// would.

async function loadLlmStatus() {
  try {
    const status = await call(`${API}/llm/status`);
    state.llm = { configured: !!status.configured, model: status.model || "" };
  } catch {
    state.llm = { configured: false, model: "" };
  }
  paintLlmAvailability();
}

/** Why a model control is on or off. A disabled button that will not say why
    is the same as a broken one, so this is the only wording either path uses. */
const llmWhy = () => (state.llm.configured
  ? `Writes with ${state.llm.model}`
  : "No model configured. Set llm_url (a local model) or gemini_key in Picky's settings file (user/video_prompter/settings.json), then restart ComfyUI.");

/** Grey the model controls out, and say why, when there is no key. */
function paintLlmAvailability() {
  const on = state.llm.configured;
  $("writer-bar").classList.toggle("off", !on);
  for (const id of ["writer-go", "writer-seed"]) {
    $(id).disabled = !on;
    $(id).title = llmWhy();
  }
  for (const button of document.querySelectorAll("button.spark-btn")) {
    button.disabled = !on;
    if (!on) button.title = llmWhy();
  }
  paintStudioLlm();
}

/** A small button that opens the writer aimed at one part of the prompt. */
function sparkButton(scope, what) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "small spark-btn";
  button.innerHTML = '<svg class="ico sm" aria-hidden="true"><use href="#i-zap"></use></svg>';
  button.append(document.createTextNode("Rewrite"));
  button.disabled = !state.llm.configured;
  button.title = button.disabled ? llmWhy() : `Ask the model to rewrite ${what}`;
  button.onclick = (event) => {
    event.preventDefault();
    openWriter(scope, what);
  };
  return button;
}

function openWriter(scope, what) {
  if (!state.llm.configured) return toast("No model key configured.", true);
  if (!currentSegment()) return toast("Open a source first.", true);
  state.asking = { scope, what };
  $("writer-scope").textContent = scope === "all" ? "whole prompt" : scope;
  $("writer-note").textContent =
    `Say what should change about ${what}. Everything else is left exactly as it is.`;
  $("writer-status").textContent = "";
  $("writer-input").value = "";
  $("writer").classList.add("open");
  $("writer-input").focus();
}

function closeWriter() {
  $("writer").classList.remove("open");
  state.asking = null;
}

// How long the editor waits before deciding the model is not coming back. The
// server has its own budget well inside this; the ceiling is here so a stalled
// connection cannot leave a spinner running for the rest of the afternoon.
const DRAFT_WAIT_MS = 150000;

/** A status line that counts, so a long wait reads as working, not frozen. */
function tickingStatus(element, label) {
  if (!element) return () => {};
  const began = Date.now();
  const paint = () => {
    const seconds = Math.round((Date.now() - began) / 1000);
    element.textContent = `${label} ${seconds}s`;
  };
  paint();
  const timer = setInterval(paint, 1000);
  return () => {
    clearInterval(timer);
    element.textContent = "";
  };
}

/** Post one draft and adopt what comes back. Shared by both ways in. */
async function runDraft(scope, instruction, statusEl) {
  const source = currentSource();
  const segment = currentSegment();
  if (!source || !segment || !state.project) return;
  if (!instruction.trim()) return toast("Say what you want written.", true);

  // The server reads the prompt off the segment, so anything typed and not yet
  // saved would be invisible to it. Push first rather than rewrite a stale one.
  if (state.dirty) await pushSource();

  const stop = tickingStatus(statusEl, `Asking ${state.llm.model}…`);
  const abort = new AbortController();
  const bell = setTimeout(() => abort.abort(), DRAFT_WAIT_MS);
  let result;
  try {
    result = await post(`${API}/llm/draft`, {
      project: state.project.name,
      source: source.id,
      segment: segment.id,
      instruction,
      scope,
    }, { signal: abort.signal });
  } catch (error) {
    // An abort is this timeout firing, not a server fault; say which.
    throw error.name === "AbortError"
      ? new Error(`${state.llm.model} did not answer within ${DRAFT_WAIT_MS / 1000}s. Nothing was changed.`)
      : error;
  } finally {
    clearTimeout(bell);
    stop();
  }

  state.project = result.project;
  restoreSelection();
  await reloadSequence();
  fillForm();
  refresh();

  const moved = (result.changed || []).length;
  const stillWrong = (result.findings || []).filter((f) => f.severity === "error").length;
  toast(
    moved
      ? `Rewrote ${(result.changed || []).join(", ")}${stillWrong ? ` — ${stillWrong} error left` : ""}`
      : "The model changed nothing.",
    stillWrong > 0,
  );

  // Write and go, when that was asked for. Not on an error finding: prepare
  // would refuse it anyway, and a render that does go out on a broken prompt
  // spends minutes of GPU proving the linter was right.
  if (!state.llmRender) return;
  if (stillWrong) {
    toast(`Not rendering — ${stillWrong} error still in the prompt.`, true);
    return;
  }
  if (statusEl) statusEl.textContent = "Queueing the render…";
  await generate({});
}

// -- prompt history --------------------------------------------------------
//
// The undo for everything above. Takes already freeze the text they rendered,
// but only for prompts that reached the GPU; this covers the edits in between,
// which is where the version you actually want usually is.

function updateHistoryCount() {
  const segment = currentSegment();
  const count = segment?.revision_count ?? 0;
  $("history-count").textContent = count ? `History (${count})` : "History";
  $("history-open").disabled = !segment;
}

async function openHistory() {
  const source = currentSource();
  const segment = currentSegment();
  if (!source || !segment || !state.project) return;
  if (state.dirty) await pushSource();

  const query = new URLSearchParams({
    project: state.project.name, source: source.id, segment: segment.id,
  });
  const fetched = await call(`${API}/source/revisions?${query}`);
  state.history = {
    revisions: fetched.revisions || [],
    current: fetched.current || null,
    picked: fetched.revisions?.[0]?.id ?? null,
  };
  $("history-stat").textContent = state.history.revisions.length
    ? `${state.history.revisions.length} past version(s) of this generation`
    : "";
  renderHistory();
  $("history").classList.add("open");
}

function renderHistory() {
  const list = $("history-list");
  list.textContent = "";
  const { revisions, picked } = state.history;

  if (!revisions.length) {
    list.innerHTML = `<div class="empty" style="padding:14px">Nothing yet. Every
      edit that changes this prompt — yours or the model's — files a version here.</div>`;
    $("history-diff").textContent = "";
    $("history-restore").disabled = true;
    return;
  }

  const now = document.createElement("div");
  now.className = "hist-row now";
  now.innerHTML = '<div class="why">As it stands now</div>';
  list.append(now);

  for (const revision of revisions) {
    const row = document.createElement("div");
    row.className = `hist-row${revision.id === picked ? " active" : ""}`;
    const when = document.createElement("div");
    when.className = "when";
    when.textContent = whenText(revision.at);
    const why = document.createElement("div");
    why.className = "why";
    // A hand edit files no note; saying so beats an empty row.
    why.textContent = revision.note || "edited in the editor";
    row.append(when, why);
    row.onclick = () => {
      state.history.picked = revision.id;
      renderHistory();
    };
    list.append(row);
  }

  const chosen = revisions.find((entry) => entry.id === picked);
  $("history-restore").disabled = !chosen;
  renderHistoryDiff(chosen);
}

/** That version against the prompt as it stands, as one marked-up run. */
function renderHistoryDiff(revision) {
  const pane = $("history-diff");
  pane.textContent = "";
  if (!revision) return;
  const chunks = diffWords(revision.text || "", state.history.current?.text || "");
  for (const chunk of chunks) {
    if (chunk.type === "same") pane.append(chunk.text);
    else pane.append(diffMark(chunk.type, chunk.text));
  }
}

function whenText(stamp) {
  if (!stamp) return "";
  const at = new Date(stamp);
  return Number.isNaN(at.getTime()) ? stamp : at.toLocaleString();
}

async function restorePicked() {
  const source = currentSource();
  const segment = currentSegment();
  const id = state.history.picked;
  if (!source || !segment || !id || !state.project) return;

  const result = await post(`${API}/source/revisions/restore`, {
    project: state.project.name,
    source: source.id,
    segment: segment.id,
    revision: id,
  });
  state.project = result.project;
  restoreSelection();
  await reloadSequence();
  fillForm();
  refresh();
  $("history").classList.remove("open");
  toast("Restored that version. It is itself undoable.");
}

function wireWriter() {
  // A whole shot described in a sentence is often three lines of one, and the
  // bar used to show the first and clip the rest.
  makeGrow($("writer-seed"));

  // Two checkboxes, one setting: whichever is clicked, both show it.
  for (const id of ["writer-render", "writer-render-2"]) {
    $(id).onchange = () => {
      state.llmRender = $(id).checked;
      for (const other of ["writer-render", "writer-render-2"]) {
        $(other).checked = state.llmRender;
      }
    };
  }

  $("writer-go").onclick = guard(async () => {
    const seed = $("writer-seed");
    await runDraft("all", seed.value, $("writer-status"));
    seed.value = "";
  });
  $("history-open").onclick = guard(openHistory);
  $("history-close").onclick = () => $("history").classList.remove("open");
  $("history-restore").onclick = guard(restorePicked);

  $("writer-close").onclick = closeWriter;
  $("writer-cancel").onclick = closeWriter;
  $("writer-submit").onclick = guard(async () => {
    const aim = state.asking;
    if (!aim) return;
    const input = $("writer-input");
    const said = input.value;
    $("writer-submit").disabled = true;
    try {
      await runDraft(aim.scope, said, $("writer-status"));
      closeWriter();
    } finally {
      $("writer-submit").disabled = false;
    }
  });

  // Enter sends, shift+Enter breaks the line — the shape of every other box
  // that takes one instruction.
  for (const [id, run] of [["writer-input", () => $("writer-submit").click()],
                           ["writer-seed", () => $("writer-go").click()]]) {
    $(id).onkeydown = (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        run();
      }
    };
  }

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if ($("writer").classList.contains("open")) closeWriter();
    else if ($("history").classList.contains("open")) $("history").classList.remove("open");
    else if ($("projects").classList.contains("open")) $("projects").classList.remove("open");
  });

  // A rewrite button beside every block the model can write. Added here rather
  // than in the markup so one list decides which fields are offered, and it is
  // the same list the server enforces.
  const blocks = {
    style: "the style contract", overview: "the scene overview",
    camera: "the fallback camera line", audio: "the diegetic soundscape",
    music: "the score",
  };
  for (const [id, what] of Object.entries(blocks)) {
    const label = $(id).closest(".field")?.querySelector("label");
    if (label) label.append(document.createTextNode(" "), sparkButton(id, what));
  }
}

function wire() {
  wireWorkspaceLayout();
  wireSceneEditor();
  for (const ratio of RATIOS) $("ratio").append(new Option(ratio, ratio));
  $("ratio").value = "16:9";

  setupTabs("author-tabs");
  editorUX = setupEditorUX({ refit: refitGrowFields, uuid, renderLibrary: renderCast });
  wireSectionHelp();
  wireWriter();
  wireStudio();
  wireImageWriter();
  wireLightbox();
  configureCompareModal({
    $,
    getState: () => state,
    currentClip,
    clipReport,
    currentSource,
    pathOf,
    stopPlayer,
    label,
    iconHTML,
    ratingOf,
    ratingLabel,
    guard,
    toast,
    setPath,
    renderCandidates,
    takeById,
  });
  wireCompareModal();
  wireBinViews();
  bindCastControls();

  for (const key of [...BLOCKS, ...LIST_BLOCKS]) {
    makeGrow($(key));
    // `screen_text` and `negatives` are the LIST_BLOCKS, and neither takes a
    // mention: screen text is drawn on a sign in the frame, and negatives never
    // reach the prompt at all.
    if (BLOCKS.includes(key)) wireMentions($(key));
    $(key).oninput = () => {
      // Editing a structured field invalidates any manual edit of the composed
      // text — the user is back to authoring through the form.
      clearPreviewEdit();
      markDirty();
      refresh();
    };
  }
  $("guard_identities").onchange = () => {
    clearPreviewEdit();
    markDirty();
    refresh();
  };
  $("source-name").oninput = () => {
    const source = currentSource();
    if (!source) return;
    source.name = $("source-name").value;
    markDirty();
    $("editor-title").textContent = source.name;
    renderBin();
    renderTimeline();
  };

  // Delegated double-click on the track and shot list. Native `dblclick` is
  // unreliable on macOS trackpads and across DOM rebuilds (select() recreates
  // elements, so the second click lands on a new node). Instead we detect two
  // rapid clicks ourselves and delegate from a surviving parent.
  function delegatedDblClick(container, selector, handler) {
    let lastClick = { id: null, time: 0 };
    container.addEventListener("click", (e) => {
      const el = e.target.closest(selector);
      if (!el?.dataset.id) return;
      const now = Date.now();
      if (el.dataset.id === lastClick.id && now - lastClick.time < 500) {
        lastClick = { id: null, time: 0 };
        handler(el.dataset.id);
      } else {
        lastClick = { id: el.dataset.id, time: now };
      }
    });
    // Keep native dblclick as a belt-and-suspenders fallback.
    container.addEventListener("dblclick", (e) => {
      const el = e.target.closest(selector);
      if (el?.dataset.id) handler(el.dataset.id);
    });
  }
  delegatedDblClick($("track"), ".clip", (id) => {
    const clip = clipById(id);
    if (clip) openEditor(clip.source_id, clip.id);
  });
  delegatedDblClick($("bin-list"), ".bin-row", (id) => openEditor(id));

  // The track is sized to its dock, so a resize changes the fit scale.
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderTimeline, 120);
  });

  // A stray tab-close should not silently drop edits the server has not seen.
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });

  // Committing a new duration refits the beats: their proportions are
  // authorial, their absolute times are not. On `change`, not `input`, so it
  // does not fire on every keystroke.
  $("duration").oninput = () => { markDirty(); refresh(); };
  $("duration").onchange = () => {
    const duration = parseInt($("duration").value, 10);
    if (!Number.isNaN(duration) && currentPrompt()?.beats.length) {
      transform({ op: "refit", duration });
    }
  };

  $("ratio").onchange = () => transform({ op: "ratio", ratio: $("ratio").value });

  // The Prompt tab has two modes: tinted, where every run of text is clickable
  // back to the field that wrote it, and plain, where the text is editable.
  $("prompt-tint-toggle").onclick = () => setPromptTint(!state.promptTint);
  setPromptTint(true);

  $("preview").onmousemove = (event) => {
    const mark = state.promptTint && event.target.closest?.("[data-field]");
    litLegendKind(mark ? mark.dataset.kind : null);
  };
  $("preview").onmouseleave = () => litLegendKind(null);
  $("preview").onclick = (event) => {
    if (!state.promptTint) return;
    const mark = event.target.closest?.("[data-field]");
    if (mark) focusField(mark.dataset.field);
  };

  // Let the user edit the composed prompt directly. When they do, the edited
  // text overrides compose() for the next render/reroll.
  $("preview").oninput = () => {
    state.previewDirty = true;
    state.previewEditedText = $("preview").textContent;
    $("preview").classList.add("dirty");
    $("preview-dirty-hint").hidden = false;
    // The text is no longer compose()'s, so it has no provenance to show.
    $("prompt-tint-toggle").disabled = true;
    // Update the char count in the stats line.
    const len = state.previewEditedText.length;
    const statsEl = $("stats");
    if (statsEl) statsEl.innerHTML = statsEl.innerHTML.replace(/\d+ chars/, `${len} chars`);
  };

  // Trim and speed go through the server, which clamps them against however
  // much lineage the clip really has to cut from.
  const editClip = guard(async () => {
    const clip = currentClip();
    if (!clip) return;
    const out = $("source-out").value;
    await apply(post(`${API}/clip/edit`, {
      project: state.project.name,
      clip: clip.id,
      source_in: parseFloat($("source-in").value) || 0,
      source_out: out === "" ? null : parseFloat(out) || 0,
      speed: parseFloat($("speed").value) || 1,
    }));
  });
  for (const id of ["source-in", "source-out", "speed"]) $(id).onchange = editClip;

  // Sound goes through the same route. The server works the path to a take out
  // of the take record, so nothing here assembles one — that would be a second
  // opinion about which take follows which.
  $("clip-audio-take").onchange = guard(async (event) => {
    const chosen = event.target.value;
    await editSound(
      chosen === NO_SOUND
        ? { muted: true }
        : { muted: false, take: chosen === OWN_SOUND ? null : chosen },
    );
  });
  $("clip-audio-gain").onchange = guard(() =>
    editSound({ gain_db: parseFloat($("clip-audio-gain").value) || 0 }));
  $("clip-audio-offset").onchange = guard(() =>
    editSound({ offset: parseFloat($("clip-audio-offset").value) || 0 }));

  // Switching projects without pushing first would drop whatever is in the
  // form. If the push fails, stay put and snap the picker back.
  $("project-picker").onchange = guard(async (event) => {
    const target = event.target.value;
    try {
      await pushSource();
    } catch (error) {
      event.target.value = state.project?.name ?? "";
      throw error;
    }
    await openProject(target);
  });

  $("new-project").onclick = guard(async () => {
    const name = window.prompt("New project name");
    if (!name) return;
    await post(`${API}/project/create`, { name });
    await post(`${API}/source/add`, { project: name, name: "Source 1" });
    await loadProjects(name);
  });

  $("manage-projects").onclick = guard(async () => {
    // The listing is a read per project, so it is fetched when the manager is
    // opened rather than kept warm behind a panel nobody has looked at.
    takeListing(await call(`${API}/projects`));
    openProjects();
  });
  $("projects-close").onclick = () => $("projects").classList.remove("open");
  $("proj-tabs").onclick = (event) => {
    const tab = event.target.closest("[data-shelf]");
    if (!tab) return;
    state.projectShelf = tab.dataset.shelf;
    renderProjects();
  };
  $("proj-search").oninput = (event) => {
    state.projectQuery = event.target.value;
    renderProjects();
  };
  $("proj-empty-trash").onclick = guard(emptyTrash);

  $("duplicate-source").onclick = guard(async () => {
    await duplicateSourceById();
  });

  // The bin's split, which is not the timeline's below it. That one cuts one
  // lineage into two windows on the same takes; this one copies words into a
  // source that has generated nothing yet, so the shots can be generated
  // again, at their own length, and cut back in.
  $("split-beats").onclick = guard(async () => {
    const source = currentSource();
    const segment = currentSegment();
    if (!source || !segment || !state.splitPick.size) return;
    const beats = [...state.splitPick].sort((a, b) => a - b);

    // The server splits the *stored* prompt, so the words on screen have to
    // reach disk first — otherwise a shot rewritten a minute ago splits off
    // with the sentence it used to have.
    await pushSource();
    state.splitPick.clear();
    const result = await apply(post(`${API}/source/split`, {
      project: state.project.name,
      source: source.id,
      segment: segment.id,
      beats,
    }));
    if (result.source) selectSource(result.source);
    renderSplitButton();
    toast(beats.length === 1 ? "Split off 1 shot." : `Split off ${beats.length} shots.`);
  });

  // Splitting is free: both halves are the same lineage seen through two
  // windows, so it spends no render and copies no prompt.
  $("split-clip").onclick = guard(async () => {
    const clip = currentClip();
    const entry = clipReport(state.selectedClip);
    if (!clip || !entry) return;
    if (clip.enabled === false) {
      toast("That clip is hidden from the timeline.", true);
      return;
    }
    // Split at the playhead when it sits inside the selected clip; otherwise
    // ask, so the button always does something predictable.
    let at = state.playheadSec - entry.starts_at;
    if (!(at > 0.05 && at < entry.timeline_seconds - 0.05)) {
      const answer = window.prompt(
        `Split ${entry.name} how many seconds in? (0–${entry.timeline_seconds.toFixed(2)})`,
        (entry.timeline_seconds / 2).toFixed(2));
      if (answer === null) return;
      at = parseFloat(answer);
    }
    await pushSource();
    await apply(post(`${API}/clip/split`, {
      project: state.project.name, clip: clip.id, at,
    }));
  });

  $("mute-clip").onclick = guard(async () => {
    await toggleClipMute();
  });

  // The discoverable way in. Shift-drag is faster once you know it exists, and
  // this button's tooltip is where you find out.
  const muteRange = $("mute-range");
  if (muteRange) {
    muteRange.onclick = guard(async () => {
      const clip = currentClip();
      const entry = clipReport(state.selectedClip);
      if (!clip || !entry) return;
      if (!(entry.timeline_seconds > 0)) {
        toast("Nothing to silence yet — that clip has no rendered length.", true);
        return;
      }
      // A second from the playhead when it is inside this clip, else from its
      // head, so the button always does something you can see.
      const inside = state.playheadSec - entry.starts_at;
      const at = inside > 0 && inside < entry.timeline_seconds ? inside : 0;
      const width = Math.min(1, entry.timeline_seconds - at);
      if (width < 0.05) {
        toast("No room for a band there.", true);
        return;
      }
      const speed = clip.speed || 1;
      const base = muteBase(clip);
      await addMute(clip, base + at * speed, base + (at + width) * speed);
    });
  }

  $("toggle-clip").onclick = guard(async () => {
    await toggleClipVisibility();
  });

  const edToggle = $("editor-toggle-clip");
  if (edToggle) {
    edToggle.onclick = guard(async () => {
      await toggleClipVisibility();
    });
  }

  $("delete-clip").onclick = guard(async () => {
    await deleteClipById();
  });

  const edDel = $("editor-delete-source");
  if (edDel) {
    edDel.onclick = guard(async () => {
      await deleteSourceById();
    });
  }

  const dupClip = $("duplicate-clip");
  if (dupClip) {
    dupClip.onclick = guard(async () => {
      const clip = currentClip();
      if (!clip) return;
      await apply(post(`${API}/clip/duplicate`, {
        project: state.project.name, clip: clip.id,
      }));
    });
  }

  // Pin a clip to the branch it is showing, so a later branch switch in the bin
  // leaves it alone. Following is the default; pinning is the deliberate act.
  const pinBtn = $("pin-clip");
  if (pinBtn) {
    pinBtn.onclick = guard(async () => {
      const clip = currentClip();
      if (!clip) return;
      await apply(post(`${API}/clip/edit`, {
        project: state.project.name, clip: clip.id, pin: clip.path_override === null,
      }));
      toast(clip.path_override === null ? "Pinned to this branch." : "Following the source again.");
    });
  }

  $("push-defaults").onclick = guard(async () => {
    const prompt = currentPrompt();
    if (!prompt) return;
    if (!window.confirm(
      "Copy this segment's style, camera and negatives onto every segment of " +
      "every source, replacing what they say now?")) return;
    readForm();
    await apply(post(`${API}/defaults`, {
      project: state.project.name,
      defaults: {
        style: prompt.style, camera: prompt.camera, negatives: prompt.negatives,
      },
      push: true,
    }));
    toast("Every segment now matches this one.");
  });

  $("add-beat").onclick = () => {
    const prompt = currentPrompt();
    if (!prompt) return;
    // A new shot is a new cut, not new time — the clip is already as long as
    // the budget says. Halving the last shot is the one place a cut always
    // fits, however full the storyboard already is.
    const last = prompt.beats.at(-1);
    if (last && round3(last.end - last.start) < MIN_SHOT * 2) {
      toast("The last shot is too short to cut in half.", true);
      return;
    }
    const start = last ? round3((last.start + last.end) / 2) : 0;
    prompt.beats.push({
      start, end: prompt.duration,
      angle: "", action: "", camera: "", transition: "", cut: true, lines: [],
    });
    sceneOpenShots().add(prompt.beats.length - 1);
    // The new shot is sorted into the list by `tileBeats`, so anything after it
    // shifts by one and no tick still names the shot it was put on.
    clearSplitPick();
    markDirty();
    renderBeats();
    refresh();
    jumpToSceneBeat(prompt.beats.length - 1);
  };

  $("add-cue").onclick = () => {
    const prompt = currentPrompt();
    if (!prompt) return;
    prompt.cues.push({ start: 0, end: null, description: "" });
    markDirty();
    renderCues();
    refresh();
  };

  $("profile-picker").onchange = () => showCoverage();

  // Settings panel: toggle, inputs, reset.
  wireSettings();

  $("save").onclick = guard(save);
  $("export").onclick = guard(exportSequence);

  // Editor.
  $("editor-close").onclick = closeEditor;
  $("preview-dock-close").onclick = ensureTimeline;
  $("dock-resize").onmousedown = startDockResize;
  for (const button of $("takes-view").querySelectorAll("button")) {
    button.onclick = () => setTakesView(button.dataset.view);
  }
  $("editor-prev").onclick = () => editorNav(-1);
  $("editor-next").onclick = () => editorNav(1);

  // Inspect view.
  $("inspect-prev").onclick = () => inspectStep(-1);
  $("inspect-next").onclick = () => inspectStep(1);
  $("inspect-detail").onclick = () => setInspectDetail(!state.inspectDetail);
  $("take-inspector-prompt-note").onclick = (event) => {
    const note = event.currentTarget;
    if (note.classList.contains("clipped")) note.classList.toggle("open");
  };
  $("modal-compare").onclick = renderCompare;
  closeOnBackdrop($("editor"), closeEditor);

  // Queue Tracker & Live Jobs.
  $("queue-tracker-btn").onclick = openQueueTracker;
  $("queue-modal-close").onclick = closeQueueTracker;
  closeOnBackdrop($("queue-modal"), closeQueueTracker);
  $("queue-refresh-btn").onclick = () => {
    syncQueue();
    toast("Queue refreshed");
  };
  $("queue-interrupt-btn").onclick = interruptRunningJob;
  $("queue-clear-pending-btn").onclick = clearPendingQueue;
  $("queue-shelve-pending-btn").onclick = shelveAllPending;
  $("queue-schedule-all-held-btn").onclick = scheduleAllHeldJobs;
  $("queue-unschedule-all-held-btn").onclick = unscheduleAllHeldJobs;
  $("queue-release-all-held-btn").onclick = releaseAllHeldJobs;
  $("queue-clear-history-btn").onclick = clearJobHistory;

  document.querySelectorAll(".queue-tabbar button[data-queue-tab]").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".queue-tabbar button[data-queue-tab]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.queueActiveTab = btn.dataset.queueTab || "all";
      renderQueueModal();
    };
  });

  const filterInput = $("queue-filter-input");
  if (filterInput) {
    filterInput.oninput = (e) => {
      state.queueFilterQuery = e.target.value;
      renderQueueModal();
    };
  }

  // Job Inspector.
  $("job-inspector-close").onclick = closeJobInspector;
  closeOnBackdrop($("job-inspector-modal"), closeJobInspector);

  // The cast's "from the project" picture grid, and the member sheet that
  // opens it. The sheet's own close button is wired where it is built, since
  // the head is redrawn on every save.
  $("picker-close").onclick = closeImagePicker;
  closeOnBackdrop($("picker"), closeImagePicker);
  closeOnBackdrop($("castdetail"), closeCastDetail);

  // Timeline transport + zoom + scrubbing.
  $("tl-play").onclick = togglePlay;
  $("tl-play-clip").onclick = () => {
    const index = state.player?.list.findIndex((entry) => entry.id === state.selectedClip) ?? -1;
    if (index >= 0) playFrom(index);
    else toast("This clip has nothing to play yet.");
  };
  $("zoom-in").onclick = () => setZoom(state.zoom * 1.6);
  $("zoom-out").onclick = () => setZoom(state.zoom / 1.6);
  $("zoom-fit").onclick = () => setZoom(1);
  $("tl-scroll").onmousedown = startScrub;
  watchPlayer();
  watchForSound();

  const MODAL_STACK = [
    () => {
      const cmp = $("compare-modal");
      return cmp && !cmp.hidden ? cmp : null;
    },
    () => ($("job-inspector-modal")?.classList.contains("open") ? $("job-inspector-modal") : null),
    () => ($("picker")?.classList.contains("open") ? $("picker") : null),
    () => ($("imghist")?.classList.contains("open") ? $("imghist") : null),
    () => ($("lightbox")?.classList.contains("open") ? $("lightbox") : null),
    () => ($("writer")?.classList.contains("open") ? $("writer") : null),
    () => ($("history")?.classList.contains("open") ? $("history") : null),
    () => ($("projects")?.classList.contains("open") ? $("projects") : null),
    () => ($("studio")?.classList.contains("open") ? $("studio") : null),
    () => ($("queue-modal")?.classList.contains("open") ? $("queue-modal") : null),
    () => ($("castdetail")?.classList.contains("open") ? $("castdetail") : null),
    () => ($("editor")?.classList.contains("open") ? $("editor") : null),
  ];

  function activeModalDialog() {
    if (document.querySelector("dialog[open]")) return null;
    for (const resolve of MODAL_STACK) {
      const el = resolve();
      if (el) return el;
    }
    return null;
  }

  function trapModalTab(event) {
    if (event.key !== "Tab" || event.altKey || event.ctrlKey || event.metaKey) return false;
    const modal = activeModalDialog();
    if (!modal) return false;
    const selector = [
      "a[href]",
      "button:not(:disabled)",
      "input:not(:disabled):not([type='hidden'])",
      "select:not(:disabled)",
      "textarea:not(:disabled)",
      "[tabindex]:not([tabindex='-1'])",
    ].join(",");
    const focusables = Array.from(modal.querySelectorAll(selector)).filter(
      (el) => !el.closest("[hidden]") && el.offsetParent !== null
    );
    if (!focusables.length) {
      event.preventDefault();
      return true;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;
    if (event.shiftKey) {
      if (active === first || !modal.contains(active)) {
        event.preventDefault();
        last.focus();
        return true;
      }
    } else if (active === last || !modal.contains(active)) {
      event.preventDefault();
      first.focus();
      return true;
    }
    return false;
  }

  document.addEventListener("keydown", (event) => {
    if (trapModalTab(event)) return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName ?? "");
    if ((event.ctrlKey || event.metaKey) && event.key === "s") {
      event.preventDefault();
      guard(save)();
      return;
    }
    // Step through a chain's generations. Deliberately above the `typing`
    // bail below: switching is most wanted mid-sentence, and `selectDepth`
    // saves the open form before it swaps. `code` rather than `key` so the
    // binding survives a non-US layout.
    if (event.altKey && (event.code === "BracketLeft" || event.code === "BracketRight")) {
      if ((state.editorOpen || state.inspectorOpen) && !state.queueTrackerOpen && !state.jobInspectorOpen) {
        const count = currentSource()?.segments?.length ?? 0;
        const next = state.depth + (event.code === "BracketRight" ? 1 : -1);
        if (count > 1 && next >= 0 && next < count) {
          event.preventDefault();
          selectDepth(next);
        }
      }
      return;
    }
    if (state.lightbox) {
      // The arrows walk the set, unless the destination control has focus —
      // there they belong to the select.
      if (!typing && event.key === "ArrowLeft") { event.preventDefault(); stepLightbox(-1); return; }
      if (!typing && event.key === "ArrowRight") { event.preventDefault(); stepLightbox(1); return; }
      if (event.key === "Escape") { closeLightbox(); return; }
    }
    if (compareState.open) {
      if (event.key === "Escape") {
        event.preventDefault();
        closeCompareModal();
        return;
      }
      if (!typing && (event.code === "Space" || event.key === " ")) {
        event.preventDefault();
        toggleComparePlay();
        return;
      }
      if (!typing && event.key === "ArrowLeft") {
        event.preventDefault();
        stepCompareFrame(-1);
        return;
      }
      if (!typing && event.key === "ArrowRight") {
        event.preventDefault();
        stepCompareFrame(1);
        return;
      }
      if (!typing && (event.key === "f" || event.key === "F")) {
        event.preventDefault();
        const vidB = $("cmp-diff-vid-b");
        if (vidB) {
          vidB.style.visibility = vidB.style.visibility === "hidden" ? "visible" : "hidden";
        }
        return;
      }
      return;
    }
    if (event.key === "Escape") {
      if (document.querySelector("dialog[open]")) return;
      // The picture grid is opened from a cast card, which is behind every
      // other sheet, so it unwinds before any of them.
      if (state.picker) { closeImagePicker(); return; }
      // The history sheet sits over the studio, so it unwinds first.
      if ($("imghist").classList.contains("open")) { closeImageHistory(); return; }
      // Compare is a view of the studio's grid, not a sheet of its own, so
      // Escape steps back to the library before it closes the studio — the
      // same two-stage back the take sheet's Inspect view has.
      if (state.studio && state.studioComparing) {
        state.studioComparing = false;
        renderStudioGrid();
        return;
      }
      if (state.studio) { closeStudio(); return; }
      if (state.jobInspectorOpen) { closeJobInspector(); return; }
      if (state.queueTrackerOpen) { closeQueueTracker(); return; }
      // A member's sheet sits over the editor and can raise the studio and the
      // picture grid, so it unwinds after those two and before the editor.
      if (state.castDetail) { closeCastDetail(); return; }
      if (!$("utility-cast").hidden) { $("utility-cast").querySelector("[data-close-utility]").click(); return; }
      // Inspect is a view of the takes column rather than a sheet, so Escape
      // steps back to the takes first and only then closes the editor.
      if (state.editorOpen && state.takesView === "inspect") { closeTakeInspector(); return; }
      if (state.editorOpen) { closeEditor(); return; }
    }
    if (typing || document.activeElement?.isContentEditable || state.queueTrackerOpen || state.jobInspectorOpen) return;
    if (event.key === "Escape" && state.reviewOpen) { setReviewOpen(false); return; }
    if (document.activeElement?.closest("#utility-cast")) return;
    if (document.activeElement?.closest("#editor, #review-content") && !state.editorOpen) return;

    // Clip and bin shortcuts (when not typing in form inputs). Delete takes
    // the clip out of the edit; the source stays in the bin, which is deleted
    // from the bin row itself so it cannot happen by a stray keypress.
    if (event.key === "Delete" || event.key === "Backspace") {
      if (!state.editorOpen) {
        event.preventDefault();
        guard(deleteClipById)();
        return;
      }
    }
    if ((event.key === "h" || event.key === "H") && !event.ctrlKey && !event.metaKey && !event.altKey) {
      if (!state.editorOpen) {
        event.preventDefault();
        guard(toggleClipVisibility)();
        return;
      }
    }
    if ((event.key === "m" || event.key === "M") && !event.ctrlKey && !event.metaKey && !event.altKey) {
      if (!state.editorOpen) {
        event.preventDefault();
        guard(toggleClipMute)();
        return;
      }
    }
    if ((event.key === "d" || event.key === "D") && !event.ctrlKey && !event.metaKey && !event.altKey) {
      if (!state.editorOpen) {
        event.preventDefault();
        guard(duplicateSourceById)();
        return;
      }
    }

    if (state.editorOpen) return;

    // Transport by keyboard, the way an NLE expects: space plays, the arrows
    // nudge the playhead (a bigger step with Shift), Home/End jump the ends.
    const total = state.report?.duration ?? 0;
    if (event.key === " ") {
      event.preventDefault();
      togglePlay();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      stopPlayer();
      seekTo(state.playheadSec - (event.shiftKey ? 5 : 1));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      stopPlayer();
      seekTo(state.playheadSec + (event.shiftKey ? 5 : 1));
    } else if (event.key === "Home") {
      event.preventDefault();
      stopPlayer();
      seekTo(0);
    } else if (event.key === "End") {
      event.preventDefault();
      stopPlayer();
      seekTo(total);
    }
  });
}

async function boot() {
  wire();
  openSocket();
  startQueueTicker();
  try {
    // Settle all three independently — a preset or profile failure should not
    // block loading projects (important for remote/Tailscale access).
    const results = await Promise.allSettled([
      loadPresets(), loadProfiles(), loadRefKinds(), loadLlmStatus(), loadStyles(),
    ]);
    for (const r of results) {
      if (r.status === "rejected") console.warn("boot sub-task failed:", r.reason);
    }
    await loadProjects();
    await syncQueue();
  } catch (error) {
    toast(`Could not reach the server: ${error.message}`, true);
  }
}

boot();
