"""Read an existing prompt back into structured form.

Lets you paste a prompt you already have — yours or one of MiniMax's official
ones — and get it on the timeline instead of retyping it. Targets the official
H3 rewrite format: an optional alignment instruction, `integrated_multimodal_
description` with `[Shot N]` markers, `overall_soundscape`, and
`non_diegetic_music`.

Best-effort by design: a pasted shot's prose lands in `action` (the composer
weaves angle, action and camera together, so that split cannot be recovered
exactly), but the opening before `[Shot 1]`, shot boundaries, cue timing, sound
fields, and reference roles all come back.
"""

from __future__ import annotations

import math
import re

from .document import Reference, ReferenceSet, Subject, SubjectSet, VideoPrompt
from .storyboard import AudioCue, Beat, Line

# The labelled fields, as literal `field: value` headers — the three base core
# fields plus the extra ref2v sections.
_FIELD_RE = re.compile(
    r"^(?P<field>integrated_multimodal_description|detailed_description|"
    r"overall_soundscape|non_diegetic_music|subject_definitions|summary|"
    r"retention_analysis)"
    r":\s*(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)

_TASK_PREFIX_RE = re.compile(r"^\[(?P<tasks>[^\]]+)\]\s*(?P<rest>.*)$", re.DOTALL)
# `<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...`. The
# parenthetical is where the item lands (guide 4) and is read past: the tag
# and the marker are the two things this restores.
_RETENTION_RE = re.compile(
    r"^<(?P<tag>[^>]+)>\s*(?:\([^)]*\)\s*)?:\s*(?P<marker>[a-z_]+)\s*-", re.IGNORECASE
)

# The gap between two sentences, for splitting the opening back into its parts.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# `[Shot 3]` markers that open each shot in the description.
_SHOT_SPLIT_RE = re.compile(r"(?=\[Shot\s+\d+\])", re.IGNORECASE)
_SHOT_RE = re.compile(
    r"^\[Shot\s+(?P<index>\d+)\]\s*(?P<body>.*)$", re.IGNORECASE | re.DOTALL
)

# A later shot opens with `At MM:SS.mmm, <cut phrase> ...`.
_CUT_RE = re.compile(
    r"^At\s+(?P<mm>\d{1,2}):(?P<ss>\d{2})\.(?P<mmm>\d{3})\s*,\s*(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_TRANSITIONS = (
    "the camera cuts to",
    "the shot cuts to",
    "the shot transitions to",
    "the shot changes to",
    "the shot switches to",
)

# Definition lines: `<Subject 1> is the woman in <Picture 1>...`, `<Picture 1> is
# the mood.`, and the legacy `Image 1 is ...`. The role stops at the first
# period *or* the end of its line: the block is one definition per line, and a
# line left without a full stop must not swallow the one below it.
_ROLE_RE = re.compile(
    r"<?(?P<kind>Subject|Picture|Image|Video|Audio)\s+(?P<index>\d+)>?\s+is\s+"
    r"(?P<role>[^.\n]+)",
    re.IGNORECASE,
)
_KIND_FROM_TAG = {"picture": "image", "image": "image", "video": "video", "audio": "audio"}

# A timed cue, as `compose` writes it: its own sentence in the soundscape, timed
# in the same `MM:SS.mmm` the shots use.
_TIMECODE = r"\d{1,2}:\d{2}\.\d{3}"
_CUE_TIMECODE_RE = re.compile(
    rf"^(?:At\s+(?P<at>{_TIMECODE})"
    rf"|From\s+(?P<from>{_TIMECODE})\s+to\s+(?P<to>{_TIMECODE}))"
    r"\s*,\s*(?P<body>.*?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
# Prompts written before cues spoke the shots' dialect timed them in bare
# seconds, `At 3.5s,` and `At 6s-10s,`.
_CUE_SENTENCE_RE = re.compile(
    r"^At\s+(?P<start>\d+(?:\.\d+)?)s(?:-(?P<end>\d+(?:\.\d+)?)s)?\s*,\s*"
    r"(?P<body>.*?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
# Prompts written before cues became sentences comma-spliced them instead.
_CUE_SPLIT_RE = re.compile(
    r",\s*(?=at\s+\d+(?:\.\d+)?s(?:-\d+(?:\.\d+)?s)?\s*:)", re.IGNORECASE
)
_CUE_RE = re.compile(
    r"^at\s+(?P<start>\d+(?:\.\d+)?)s(?:-(?P<end>\d+(?:\.\d+)?)s)?\s*:\s*(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)

# `On-screen text reads "A", "B".`, and the older `..., reproduced exactly:`.
_SCREEN_TEXT_RE = re.compile(
    r'On-screen text\s*(?:reads|,\s*reproduced exactly\s*:)\s*'
    r'(?P<body>"[^"]+"(?:\s*,\s*"[^"]+")*)\s*\.?',
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r'"([^"]+)"')

# A composed dialogue line, with everything the format hangs off it:
# `<scenetrans> The woman (S1) says: <d>[English] Hi.</d> while her lips remain
# completely closed. <cutoff> The speech carries over from the previous shot.`
_LINE_RE = re.compile(
    r"(?P<lead><scenetrans>\s*)?"
    r"(?P<describe>[^.<>]*?)\((?P<ids>S\d+(?:,S\d+)*)\)\s+(?P<verb>[^:<]+?):\s*"
    r"<d>\[(?P<lang>[^\]]+)\]\s*(?P<text>.*?)</d>"
    r"(?P<vo>\s+while\s+(?P<possessive>[^.<>]{1,40}?)\s+lips remain completely closed)?"
    r"\.?(?P<tags>(?:\s*(?:<scenetrans>|<cutoff>))*)"
    r"(?P<notes>(?:\s*The speech\s[^.]+\.)*)",
    re.IGNORECASE | re.DOTALL,
)

# `compose._identity_guard` trails the last shot with a whole-video instruction.
# It is not that shot's action, so it comes off before the prose is read.
_GUARD_RE = re.compile(
    r"\s*Keep\s+\(S\d+\)[^.]*identities must not merge or swap\s*\.",
    re.IGNORECASE,
)


def parse(text: str, *, duration: int | None = None) -> VideoPrompt:
    """Best-effort structural read of an official-format prompt.

    `duration` overrides the budget; by default it is inferred from the last cut
    time, rounded up to a whole second.
    """
    prompt = VideoPrompt.blank()
    fields: dict[str, str] = {}
    roles: list[Reference] = []

    for block in _blocks(text):
        match = _FIELD_RE.match(block)
        if match is not None:
            fields[match["field"].lower()] = match["body"].strip()
            continue
        roles.extend(_roles(block))

    # The full-reference block, when present, owns both lists: its asset lines
    # are the references and its `<Subject N>` lines the subjects.
    definitions = fields.get("subject_definitions", "")
    declared = _roles(definitions)
    if declared:
        prompt.references = ReferenceSet(declared)
    elif roles:
        prompt.references = ReferenceSet(roles)
    prompt.subjects = SubjectSet(_subjects(definitions))

    _absorb_summary(fields.get("summary", ""), prompt)
    _absorb_retention(fields.get("retention_analysis", ""), prompt)

    description = fields.get("integrated_multimodal_description", "") or fields.get(
        "detailed_description", ""
    )
    if description:
        _absorb_shots(_absorb_opening(description, prompt), prompt)

    _absorb_soundscape(fields.get("overall_soundscape", ""), prompt)
    music = fields.get("non_diegetic_music", "").strip()
    prompt.music = "" if music.upper() == "N/A" else music

    prompt.storyboard.duration = (
        duration if duration is not None else _infer_duration(prompt)
    )
    _close_ends(prompt)
    return prompt


def _blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]


def _absorb_shots(description: str, prompt: VideoPrompt) -> None:
    beats: list[Beat] = []
    for chunk in _SHOT_SPLIT_RE.split(description):
        chunk = chunk.strip()
        shot = _SHOT_RE.match(chunk)
        if shot is None:
            continue
        body = shot["body"].strip()
        start, transition, body = _shot_timing(body, first=not beats)
        # Screen text is a sentence inside the shot, not part of its action.
        screen = _SCREEN_TEXT_RE.search(body)
        if screen is not None:
            prompt.screen_text = _QUOTED_RE.findall(screen["body"])
            body = " ".join((body[: screen.start()] + body[screen.end() :]).split())
        body, guarded = _strip_guard(body)
        prompt.guard_identities = prompt.guard_identities or guarded

        # One `[Shot N]` may hold several timed moments — a continuous take with
        # camera moves inside it. Each becomes a beat of its own that does not
        # cut, so the structure survives the round trip instead of collapsing
        # into one wall of text with timecodes buried in it.
        lead, *held = _split_continuations(body)

        lead, lines = _extract_lines(lead)
        beats.append(
            Beat(start=start, end=start, action=lead, transition=transition, lines=lines)
        )
        for moment, rest in held:
            rest, held_lines = _extract_lines(rest)
            # Camera and action are left joined in `action`: the composer writes
            # `At MM:SS.mmm, <the whole clause>` for a beat with no `camera` of
            # its own, so this recomposes to the very text it was read from.
            # Guessing where the move ends and the action begins would not.
            beats.append(
                Beat(
                    start=moment,
                    end=moment,
                    action=rest,
                    cut=False,
                    lines=held_lines,
                )
            )

    prompt.storyboard.beats = beats


# A moment inside a shot: a sentence that opens `At MM:SS.mmm,`.
_HELD_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=At\s+\d{1,2}:\d{2}\.\d{3}\s*,)", re.IGNORECASE
)


def _split_continuations(body: str) -> list[object]:
    """A shot body as `[lead, (seconds, text), ...]`.

    The lead is whatever opens the shot; each pair after it is a timed moment
    the shot holds without cutting. A body with no internal timecode comes back
    as a one-item list, which is every prompt written before no-cut beats.
    """
    head, *rest = _HELD_SPLIT_RE.split(body)
    parts: list[object] = [head.strip()]
    for chunk in rest:
        match = _CUT_RE.match(chunk.strip())
        if match is None:  # the lookahead matched, so this cannot happen
            parts[0] = f"{parts[0]} {chunk.strip()}".strip()
            continue
        seconds = int(match["mm"]) * 60 + int(match["ss"]) + int(match["mmm"]) / 1000
        parts.append((round(seconds, 3), match["rest"].strip()))
    return parts


def _strip_guard(body: str) -> tuple[str, bool]:
    """Take the identity-guard sentence off a shot body, and say if it was there."""
    cleaned, count = _GUARD_RE.subn(" ", body)
    return " ".join(cleaned.split()), bool(count)


def _extract_lines(body: str) -> tuple[str, list[Line]]:
    """Pull dialogue clauses out of a shot body, leaving the narration behind.

    The speaker key is the composed `S1`/`S2` id itself — enough to keep the
    line on the beat and recompose it with the same tags.
    """
    lines: list[Line] = []
    for match in _LINE_RE.finditer(body):
        verb = " ".join(match["verb"].split())
        voiceover = bool(match["vo"]) or "voiceover" in verb.lower()
        tags = (match["tags"] or "").lower()
        if match["lead"]:
            continuity = "from"
        elif "<scenetrans>" in tags:
            continuity = "into"
        else:
            continuity = ""
        possessive = (match["possessive"] or "").strip()
        lines.append(
            Line(
                speakers=[key for key in match["ids"].split(",") if key],
                text=match["text"].strip(),
                language=match["lang"].strip() or "English",
                verb="says" if voiceover else verb,
                describe=match["describe"].strip(),
                voiceover=voiceover,
                possessive="" if possessive.lower() == "their" else possessive,
                continuity=continuity,
                cutoff="<cutoff>" in tags,
                # A trailing line has nothing but other lines behind it; prose
                # still to come is what marks a line as opening the shot.
                placement="open" if _prose_follows(body, match.end()) else "",
            )
        )
    cleaned = " ".join(_LINE_RE.sub(" ", body).split())
    return cleaned, lines


def _prose_follows(body: str, position: int) -> bool:
    """Is there narration left in `body` after `position`, ignoring other lines?"""
    return bool(_LINE_RE.sub(" ", body[position:]).strip())


def _shot_timing(body: str, *, first: bool) -> tuple[float, str, str]:
    """Pull the cut time and transition phrase off a shot body."""
    if first:
        return 0.0, "", body
    cut = _CUT_RE.match(body)
    if cut is None:
        return 0.0, "", body
    seconds = int(cut["mm"]) * 60 + int(cut["ss"]) + int(cut["mmm"]) / 1000
    rest = cut["rest"].strip()
    transition = ""
    for phrase in _TRANSITIONS:
        if rest.lower().startswith(phrase):
            transition = phrase
            rest = rest[len(phrase) :].strip()
            break
    return round(seconds, 3), transition, rest


def _close_ends(prompt: VideoPrompt) -> None:
    """Give each beat an end: the next beat's start, the last one the budget."""
    beats = prompt.storyboard.ordered_beats()
    duration = float(prompt.storyboard.duration)
    for index, beat in enumerate(beats):
        beat.end = beats[index + 1].start if index + 1 < len(beats) else duration
        if beat.end <= beat.start:
            beat.end = beat.start


def _absorb_soundscape(body: str, prompt: VideoPrompt) -> None:
    """Split the paragraph into the diegetic bed and the cues pinned to a time."""
    if not body.strip() or body.strip().upper() == "N/A":
        return
    bed: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(body.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        timed = _CUE_TIMECODE_RE.match(sentence)
        if timed is not None:
            prompt.storyboard.cues.append(_timecode_cue(timed))
            continue
        cue = _CUE_SENTENCE_RE.match(sentence)
        if cue is not None:
            prompt.storyboard.cues.append(_cue(cue))
            continue
        # Fall back to the older comma-spliced shape.
        legacy = [part.strip() for part in _CUE_SPLIT_RE.split(sentence)]
        for part in legacy:
            old = _CUE_RE.match(part)
            if old is None:
                if part:
                    bed.append(part)
            else:
                prompt.storyboard.cues.append(_cue(old))
    prompt.audio = " ".join(bed).strip()


def _timecode_cue(match: re.Match[str]) -> AudioCue:
    """A cue written in the shots' `MM:SS.mmm`, instant or spanning."""
    if match["at"]:
        return AudioCue(
            start=_seconds(match["at"]),
            description=match["body"].strip().rstrip("."),
        )
    return AudioCue(
        start=_seconds(match["from"]),
        end=_seconds(match["to"]),
        description=match["body"].strip().rstrip("."),
    )


def _seconds(timecode: str) -> float:
    minutes, _, rest = timecode.partition(":")
    seconds, _, millis = rest.partition(".")
    return round(int(minutes) * 60 + int(seconds) + int(millis) / 1000, 3)


def _cue(match: re.Match[str]) -> AudioCue:
    return AudioCue(
        start=float(match["start"]),
        end=float(match["end"]) if match["end"] else None,
        description=match["body"].strip().rstrip("."),
    )


def _absorb_opening(description: str, prompt: VideoPrompt) -> str:
    """Peel the sentences before `[Shot 1]` back into style and overview.

    Both descriptions carry the same opening: the composer writes the style
    first and the scene overview second, so that is how they come apart. A
    hand-written opening whose style runs to two sentences will split in the
    wrong place — but the two fields recompose in the same order, so the prompt
    text comes back identical either way.
    """
    marker = description.find("[Shot")
    if marker == 0:
        return description.strip()

    opening = (description if marker < 0 else description[:marker]).strip()
    if opening:
        sentences = _SENTENCE_SPLIT_RE.split(opening, maxsplit=1)
        prompt.style = sentences[0].strip()
        if len(sentences) > 1:
            prompt.overview = sentences[1].strip()
    return "" if marker < 0 else description[marker:].strip()


def _absorb_summary(body: str, prompt: VideoPrompt) -> None:
    if not body.strip():
        return
    match = _TASK_PREFIX_RE.match(body.strip())
    if match is None:
        prompt.ref_summary = body.strip()
        return
    prompt.ref_task = [task.strip() for task in match["tasks"].split("+") if task.strip()]
    prompt.ref_summary = match["rest"].strip()


def _absorb_retention(body: str, prompt: VideoPrompt) -> None:
    """Set retention markers on references matched by their `<Tag N>`."""
    if not body.strip():
        return
    by_tag: dict[str, Reference | Subject] = {
        prompt.subjects.tag(subject): subject for subject in prompt.subjects.items
    }
    by_tag.update({prompt.references.tag(item): item for item in prompt.references.items})
    for line in body.splitlines():
        match = _RETENTION_RE.match(line.strip())
        if match is None:
            continue
        item = by_tag.get(f"<{match['tag'].strip()}>")
        if item is not None:
            item.retention = match["marker"].strip()


def _roles(block: str) -> list[Reference]:
    """The `<Picture N>` / `<Video N>` / `<Audio N>` definitions in a block."""
    return [
        Reference(
            kind=_KIND_FROM_TAG[match["kind"].lower()],  # type: ignore[arg-type]
            path="",
            role=match["role"].strip(),
        )
        for match in _ROLE_RE.finditer(block)
        if match["kind"].lower() != "subject"
    ]


def _subjects(block: str) -> list[Subject]:
    """The `<Subject N>` definitions in a block.

    `sources` cannot be recovered here — the text cites tags, and a parsed
    document has no files behind them — so the whole sentence is kept as the
    role. Recomposing writes it back unchanged, which is the honest round trip:
    the citation was authored prose by the time it reached the page.
    """
    return [
        Subject(role=match["role"].strip())
        for match in _ROLE_RE.finditer(block)
        if match["kind"].lower() == "subject"
    ]


def _infer_duration(prompt: VideoPrompt) -> int:
    """Whole-second budget from the last cut, always leaving the final shot room."""
    starts = [beat.start for beat in prompt.storyboard.beats]
    starts += [cue.end or cue.start for cue in prompt.storyboard.cues]
    if not starts:
        return prompt.storyboard.duration
    last = max(starts)
    inferred = math.ceil(last)
    if inferred <= last:  # last cut lands on a whole second; give it a beat to run
        inferred += 1
    return inferred
