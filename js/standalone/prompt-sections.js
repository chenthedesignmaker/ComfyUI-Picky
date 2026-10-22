// -- reading the structure back out of a finished take's prompt -----------
//
// The tint in the live editor needs the structured document, and a finished
// take does not keep one: all it stores is the exact text that went to the
// model. That text is still a labelled document, though, so its sections can
// be read straight back out of it — the `field:` headers `compose` writes, the
// `[Shot N]` markers inside the description, the `<d>` dialogue tags. Nothing
// here re-composes or rewrites anything, so every block shown is character for
// character what was sent, which is what makes it safe to copy one out of an
// old take and paste it into a new prompt.
//
// This is coarser than the live tint, and unavoidably so: `compose` weaves a
// shot's angle, action and camera into a single sentence, and that split
// cannot be recovered from the finished text. Sections and shots can.
//
// Three shapes turn up in a gallery that has been rendered into for a while:
// the official H3 rewrite format, the labelled blocks this tool composed
// before that rewrite, and hand-written continuation prompts that are one run
// of prose. The first two are read exactly. The third is only ever guessed at,
// and says so.

let notifyToast = () => {};

/** Register the host app's toast handler for section copy notifications. */
export function setPromptSectionToast(fn) {
  if (typeof fn === "function") notifyToast = fn;
}

// `[header, what to call it, colour group]`, sharing SPAN_KINDS' hues. The
// official rewrite fields first, then the labels the older composer wrote.
export const PROMPT_FIELDS = [
  ["integrated_multimodal_description", "Description", "action"],
  ["detailed_description", "Detailed description", "action"],
  ["overall_soundscape", "Soundscape", "sound"],
  ["non_diegetic_music", "Music", "music"],
  ["subject_definitions", "Subject definitions", "reference"],
  ["summary", "Summary", "overview"],
  ["retention_analysis", "Retention analysis", "reference"],
  ["scene overview", "Scene overview", "overview"],
  ["camera", "Camera", "camera"],
  ["audio", "Audio", "sound"],
];

// The fields whose body is a run of `[Shot N]` segments rather than prose.
const DESCRIPTION_FIELDS = new Set([
  "integrated_multimodal_description",
  "detailed_description",
]);

// Official-format fields run until the next header, blank lines and all: hand
// editing puts paragraph breaks inside a description often enough that ending
// one at the first blank line would scatter it. The older format's labels are
// one paragraph each, and the block after the last of them is the negatives.
const SPANNING_FIELDS = new Set([
  ...DESCRIPTION_FIELDS,
  "overall_soundscape",
  "non_diegetic_music",
  "subject_definitions",
  "summary",
  "retention_analysis",
  "storyboard",
]);

const FIELD_HEAD_RE = new RegExp(
  `^(${PROMPT_FIELDS.map(([name]) => name).join("|")}):[ \\t]*`,
  "i",
);
const SHOT_SPLIT_RE = /(?=\[Shot\s+\d+\])/i;
const SHOT_MARKER_RE = /^\[Shot\s+(\d+)\]\s*/i;

// The older composer's storyboard block: a header line ending in a colon, then
// one `[0s-4s] Shot 1: ...` line per beat.
const BEAT_LINE_RE = /^\[\s*(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\s*\]\s*(?:Shot\s+(\d+)\s*:\s*)?/i;
const STORYBOARD_HEAD_RE = /^[^:\n]{3,140}:$/;
const SCREEN_TEXT_HEAD_RE = /^On-screen text, spelled exactly:\s*/i;
// What that composer wrote ahead of the style contract, when a shot had refs.
const ROLE_BLOCK_RE = /^<?(?:Picture|Image|Video|Audio)\s+\d+>?\s+is\s/i;
// How the official format opens when the shot renders from a keyframe.
const ALIGNMENT_RE = /^(?:For the target video|How the reference pictures)/i;

// The cut phrases the reader will treat as a shot boundary inside a prompt
// that has no markers at all. The same list `parse.py` reads back.
const CUT_PHRASES = [
  "the camera cuts to",
  "the shot cuts to",
  "the shot transitions to",
  "the shot changes to",
  "the shot switches to",
];
const CUT_PHRASE_RE = new RegExp(CUT_PHRASES.join("|"), "i");
const SENTENCE_END_RE = /[.!?]\s+/g;

// Runs inside a section that are still identifiable from the text alone: a cut
// time (only ever at the head of a shot), the dialogue tags and their speaker
// ids, the scene tags, and the reference handles.
const INLINE_RE = new RegExp(
  [
    "^At\\s+\\d{1,2}:\\d{2}\\.\\d{3}\\s*,",
    "<d>[\\s\\S]*?</d>",
    "\\(S\\d+\\)",
    "<scenetrans>",
    "<cutoff>",
    "<(?:Picture|Image|Video|Audio)\\s+\\d+>",
  ].join("|"),
  "gi",
);

function inlineKind(text) {
  if (/^At\s/i.test(text)) return "transition";
  if (/^<(?:scenetrans|cutoff)>$/i.test(text)) return "transition";
  if (/^<(?:Picture|Image|Video|Audio)\s/i.test(text)) return "reference";
  return "dialogue";
}

/** The header this line opens a section with, or null if it opens nothing. */
function headerAt(lines, index) {
  const line = lines[index];

  const field = line.match(FIELD_HEAD_RE);
  if (field) {
    const name = field[1].toLowerCase();
    const [, label, kind] = PROMPT_FIELDS.find(([header]) => header === name);
    return { name, label, kind, offset: field[0].length };
  }

  const screen = line.match(SCREEN_TEXT_HEAD_RE);
  if (screen)
    return { name: "screen_text", label: "On-screen text", kind: "screen", offset: screen[0].length };

  // The older storyboard header carries no fixed wording — it is whatever the
  // shot's `storyboard_header` said — so it is recognised by what follows it.
  if (STORYBOARD_HEAD_RE.test(line) && BEAT_LINE_RE.test((lines[index + 1] ?? "").trim()))
    return { name: "storyboard", label: "Storyboard", kind: "overview", offset: 0 };

  return null;
}

/** The prompt cut at its header lines, so a field that was hand-edited into
    several paragraphs still comes back as one block. */
function promptBlocks(text) {
  const lines = text.split("\n");
  const blocks = [];
  let current = null;

  lines.forEach((line, index) => {
    const head = headerAt(lines, index);
    // A blank line closes anything that is not a spanning field: the older
    // format separated its blocks that way, and its unlabelled negative list
    // has no header of its own to be found by.
    const closed = current && !line.trim() && !SPANNING_FIELDS.has(current.head?.name);
    if (head || !current || closed) {
      current = { head, lines: [] };
      blocks.push(current);
    }
    current.lines.push(line);
  });

  return blocks
    .map((block) => ({ head: block.head, text: block.lines.join("\n").trim() }))
    .filter((block) => block.text);
}

/** Split a recorded prompt into labelled sections, or `null` if it has none.

    Every section carries `start`/`end` into the trimmed prompt, so a caller
    that is drawing something else over the same text — the word diff, say —
    can line its own marks up with them. */
export function promptSections(text) {
  const whole = (text || "").trim();
  if (!whole) return null;

  const blocks = promptBlocks(whole);
  const sections = [];
  let labelled = 0;

  blocks.forEach((block, index) => {
    const head = block.head;
    if (!head) {
      sections.push({ text: block.text, ...unlabelled(block.text, index, blocks.length, labelled) });
      return;
    }

    labelled += 1;
    if (head.name === "storyboard") {
      sections.push(...beatSections(block.text));
      return;
    }
    const value = block.text.slice(head.offset).trim();
    if (!value) return;
    if (DESCRIPTION_FIELDS.has(head.name)) sections.push(...shotSections(value, head.label));
    else sections.push({ kind: head.kind, label: head.label, text: value });
  });

  // Nothing carried a header, so this is one run of prose — a hand-written
  // continuation, most often. The only boundary left to go on is a cut phrase.
  const found = labelled ? sections : prosePartSections(whole);
  if (!found || found.length < 2) return null;
  return withOffsets(found, whole);
}

/** Where each section sits in the prompt it came out of.

    Every section is a verbatim slice and they are in document order, so a
    cursor that only moves forward finds each one exactly once. A miss means
    the reader rewrote something it should not have, and the whole result is
    dropped rather than shown against the wrong offsets. */
function withOffsets(sections, whole) {
  let cursor = 0;
  for (const section of sections) {
    const at = whole.indexOf(section.text, cursor);
    if (at < 0) return null;
    section.start = at;
    section.end = at + section.text.length;
    cursor = section.end;
  }
  return sections;
}

/** Naming a block that carries no header, from where it sits. */
function unlabelled(body, index, count, labelledSoFar) {
  if (ROLE_BLOCK_RE.test(body)) return { label: "Reference roles", kind: "reference" };
  // Before any header comes the style contract in the older format, and the
  // keyframe alignment instruction in the official one — both open the prompt.
  if (!labelledSoFar) {
    return ALIGNMENT_RE.test(body)
      ? { label: "Keyframe alignment", kind: "screen" }
      : { label: "Style", kind: "style" };
  }
  // The older format closed with the negative list and nothing else.
  if (index === count - 1) return { label: "Negatives", kind: "screen" };
  return { label: "Unlabelled block", kind: "screen" };
}

/** A description body split at its `[Shot N]` markers. */
function shotSections(body, label) {
  const parts = body.split(SHOT_SPLIT_RE).map((part) => part.trim()).filter(Boolean);
  if (!parts.some((part) => SHOT_MARKER_RE.test(part)))
    return [{ kind: "action", label, text: body }];

  return parts.map((part) => {
    const marker = part.match(SHOT_MARKER_RE);
    // Text ahead of `[Shot 1]` is the opening the full-reference document
    // lifts out of shot one. The marker itself becomes the label and is left
    // out of the copied text — pasting `[Shot 2]` into a field is never wanted.
    return marker
      ? { kind: "action", label: `Shot ${marker[1]}`, text: part.slice(marker[0].length).trim() }
      : { kind: "overview", label: "Opening", text: part };
  });
}

/** The older format's storyboard block, one section per beat line. */
function beatSections(body) {
  const lines = body.split("\n").map((line) => line.trim()).filter(Boolean);
  const sections = BEAT_LINE_RE.test(lines[0])
    ? []
    : [{ kind: "overview", label: "Storyboard", text: lines[0].replace(/:$/, "") }];

  for (const line of lines) {
    const beat = line.match(BEAT_LINE_RE);
    if (!beat) continue;
    const span = `${beat[1]}s–${beat[2]}s`;
    sections.push({
      kind: "action",
      label: beat[3] ? `Shot ${beat[3]} · ${span}` : span,
      text: line.slice(beat[0].length).trim(),
    });
  }
  return sections;
}

/** One run of prose cut at its cut phrases. Guesswork, and flagged as such.

    Cut by index rather than by splitting, so each part stays an exact slice
    of the prompt and can still be lined up against it. */
function prosePartSections(text) {
  const starts = [0];
  for (const match of text.matchAll(SENTENCE_END_RE)) starts.push(match.index + match[0].length);

  const parts = [];
  starts.forEach((start, index) => {
    const end = index + 1 < starts.length ? starts[index + 1] : text.length;
    const sentence = text.slice(start, end);
    if (!sentence.trim()) return;
    // A sentence that names a cut opens a new part; every other sentence joins
    // the part it follows.
    if (!parts.length || CUT_PHRASE_RE.test(sentence)) parts.push({ start, end });
    else parts[parts.length - 1].end = end;
  });
  if (parts.length < 2) return null;

  return parts.map((part, index) => ({
    kind: "screen",
    label: `Part ${index + 1}`,
    text: text.slice(part.start, part.end).trimEnd(),
    inferred: true,
  }));
}

/** One section's text with its identifiable runs marked. */
export function markedText(text) {
  const holder = document.createDocumentFragment();
  let cursor = 0;
  for (const match of text.matchAll(INLINE_RE)) {
    if (match.index > cursor) holder.append(text.slice(cursor, match.index));
    const mark = document.createElement("span");
    mark.dataset.kind = inlineKind(match[0]);
    mark.textContent = match[0];
    holder.append(mark);
    cursor = match.index + match[0].length;
  }
  holder.append(text.slice(cursor));
  return holder;
}

/** One section as a labelled, separately copyable card around `body`. */
export function sectionCard(section, body, onCopy = notifyToast) {
  const card = document.createElement("div");
  card.className = "prompt-section";
  card.dataset.kind = section.kind;

  const head = document.createElement("div");
  head.className = "prompt-section-head";
  const name = document.createElement("span");
  name.className = "prompt-section-name";
  name.textContent = section.label;
  head.append(name);

  if (section.inferred) {
    const flag = document.createElement("span");
    flag.className = "prompt-section-flag";
    flag.textContent = "inferred";
    flag.title = "This prompt carried no section headers, so the split was "
      + "guessed at the cut phrases. The text itself is untouched.";
    head.append(flag);
  }

  const spacer = document.createElement("span");
  spacer.className = "spacer";
  const stat = document.createElement("span");
  stat.className = "prompt-section-stat";
  stat.textContent = `${section.text.length} chars`;
  const copy = document.createElement("button");
  copy.className = "small ghost";
  copy.textContent = "Copy";
  copy.title = `Copy ${section.label} exactly as it was sent`;
  copy.onclick = () => {
    navigator.clipboard.writeText(section.text);
    onCopy(`${section.label} copied to clipboard`);
  };
  head.append(spacer, stat, copy);

  const holder = document.createElement("div");
  holder.className = "prompt-section-body";
  holder.append(body);

  card.append(head, holder);
  return card;
}

/** A recorded prompt as a stack of labelled, separately copyable sections. */
export function promptSectionView(text) {
  const sections = promptSections(text);
  if (!sections) return null;

  const wrap = document.createElement("div");
  wrap.className = "prompt-sections";
  for (const section of sections) wrap.append(sectionCard(section, markedText(section.text)));
  return wrap;
}

// -- the same sections, over one side of the word diff ---------------------

export function diffMark(type, text) {
  const mark = document.createElement(type === "del" ? "del" : "ins");
  mark.className = type === "del" ? "diff-del" : "diff-add";
  mark.textContent = text;
  return mark;
}

/** The `keep` side of a word diff as one flat run of marked-up text. */
export function diffRun(chunks, keep) {
  const frag = document.createDocumentFragment();
  for (const chunk of chunks) {
    if (chunk.type === "same") frag.append(chunk.text);
    else if (chunk.type === keep) frag.append(diffMark(chunk.type, chunk.text));
  }
  return frag;
}

/** The `keep` side of a word diff, cut into that side's own sections.

    Returns null when this text has no sections to cut it into, so the caller
    can fall back to the flat run. */
export function sectionedDiff(text, chunks, keep) {
  const whole = (text || "").trim();
  const sections = promptSections(whole);
  if (!sections) return null;

  // Where each kept chunk lands in this side's text. Dropping the other
  // side's chunks is what makes the running total line up with `text`.
  const pieces = [];
  let cursor = 0;
  for (const chunk of chunks) {
    if (chunk.type !== "same" && chunk.type !== keep) continue;
    pieces.push({ type: chunk.type, text: chunk.text, start: cursor, end: cursor + chunk.text.length });
    cursor += chunk.text.length;
  }
  // Offsets are measured against the trimmed prompt; the diff runs over what
  // the caller passed, so anything trimmed off the front has to be added back.
  const lead = text.length - text.trimStart().length;

  const wrap = document.createElement("div");
  wrap.className = "prompt-sections";
  for (const section of sections) {
    const from = section.start + lead;
    const to = section.end + lead;
    const body = document.createDocumentFragment();
    for (const piece of pieces) {
      if (piece.end <= from || piece.start >= to) continue;
      const slice = piece.text.slice(
        Math.max(from, piece.start) - piece.start,
        Math.min(to, piece.end) - piece.start,
      );
      if (!slice) continue;
      body.append(piece.type === "same" ? slice : diffMark(piece.type, slice));
    }
    wrap.append(sectionCard(section, body));
  }
  return wrap;
}

export function diffWords(oldText, newText) {
  if (oldText === newText) {
    return [{ type: "same", text: oldText }];
  }
  const tokenize = (str) => str.match(/(\s+|[^\s\w]+|\w+)/g) || [];
  const a = tokenize(oldText);
  const b = tokenize(newText);
  const n = a.length;
  const m = b.length;

  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      if (a[i] === b[j]) {
        dp[i + 1][j + 1] = dp[i][j] + 1;
      } else {
        dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  let i = n;
  let j = m;
  const chunks = [];
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      chunks.push({ type: "same", text: a[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      chunks.push({ type: "add", text: b[j - 1] });
      j--;
    } else if (i > 0 && (j === 0 || dp[i][j - 1] < dp[i - 1][j])) {
      chunks.push({ type: "del", text: a[i - 1] });
      i--;
    }
  }
  chunks.reverse();

  const merged = [];
  for (const chunk of chunks) {
    if (merged.length && merged[merged.length - 1].type === chunk.type) {
      merged[merged.length - 1].text += chunk.text;
    } else {
      merged.push({ type: chunk.type, text: chunk.text });
    }
  }
  return merged;
}
