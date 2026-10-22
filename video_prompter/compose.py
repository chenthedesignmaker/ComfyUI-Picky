"""Render a `VideoPrompt` into the text MiniMax H3 was trained to read.

H3's rewrite format is a rigid labelled document, not free prose. Every base
mode (T2VA / I2VA / FL2VA / L2VA) produces the same shape:

    <image-alignment instruction, for keyframe modes only>

    integrated_multimodal_description: <opening> [Shot 1] ... [Shot 2] At MM:SS.mmm, ...

    overall_soundscape: ...

    non_diegetic_music: ...

The overall style and the initial composition open `[Shot 1]` itself, right
after the marker (guide 4.1). Only the full-reference `detailed_description`
lifts that opening onto its own line *ahead* of `[Shot 1]` — the reference
guide's 5.2 table names that as one of the two modes' differences. Later shots
carry a strictly increasing
`At MM:SS.mmm` cut time; camera motion is woven into each shot as a sentence
rather than stacked as a trailing label. Diegetic sound goes in
`overall_soundscape`, the audience-only score in `non_diegetic_music`.

Reference-to-video (the six-section full-reference document) is composed
separately; until that lands, references fall back to a role preamble here so
the local node still receives its tags.
"""

from __future__ import annotations

from . import h3_spec
from .camera_path import materialize_paths
from .cast import Cast, resolve
from .document import Reference, Subject, VideoPrompt
from .storyboard import Beat, Line, format_timecode

_SENTENCE_ENDINGS = ".!?"
_CLOSING_TAGS = ("</d>", h3_spec.SCENE_TRANSITION_TAG, h3_spec.CUTOFF_TAG)


def compose(prompt: VideoPrompt, cast: Cast | None = None) -> str:
    """Assemble the full prompt in the official H3 rewrite format.

    A `cast` is folded in first, so everything below sees one flat document
    with the members' descriptions and identity files already written into it.
    Callers that have no project behind them — the CLI, the bare ComfyUI node —
    pass nothing and compose exactly as before.
    """
    prompt = materialize_paths(resolve(prompt, cast))
    if mode(prompt) == "ref2v":
        return _compose_reference(prompt)

    blocks: list[str] = []

    instruction = _alignment_instruction(prompt)
    if instruction:
        blocks.append(instruction)

    description = _description(prompt)
    if description:
        blocks.append(f"integrated_multimodal_description: {description}")

    soundscape = _soundscape(prompt)
    if soundscape:
        blocks.append(f"overall_soundscape: {soundscape}")

    if description or soundscape or prompt.music.strip():
        blocks.append(f"non_diegetic_music: {_music(prompt)}")

    return "\n\n".join(blocks)


def mode(prompt: VideoPrompt) -> str:
    """The H3 task mode a shot renders as, from what media it carries."""
    if prompt.references.items:
        return "ref2v"
    has_first = bool(prompt.first_frame.strip())
    has_last = bool(prompt.last_frame.strip())
    if has_first and has_last:
        return "fl2v"
    if has_first:
        return "i2v"
    if has_last:
        return "l2v"
    return "t2v"


def _alignment_instruction(prompt: VideoPrompt) -> str:
    """Part One: how the keyframe(s) line up with the target video.

    T2VA and ref2v have no keyframe instruction. The wording is taken verbatim
    from the official guide, including its `<Picture 1>` / `S.SS` conventions.
    """
    task = mode(prompt)
    last = _last_shot_index(prompt)
    seconds = f"{prompt.rendered_seconds:.2f}"

    if task == "i2v":
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if task == "fl2v":
        return (
            "How the reference pictures align with the target video — Picture 1 "
            "(from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {last}) aligns with the {seconds}-second mark "
            "of the target video."
        )
    if task == "l2v":
        return (
            "How the reference pictures align with the target video — <Picture 1> "
            f"(from [Shot {last}]) aligns with the {seconds}-second mark of the "
            "target video."
        )
    return ""


def _last_shot_index(prompt: VideoPrompt) -> int:
    """The number of the closing `[Shot N]` — shots, not beats.

    A keyframe anchors the end of the *video*, and the end of the video is the
    end of the last shot however many timed moments that shot holds.
    """
    return max(1, len(prompt.storyboard.shots()))


def _description(prompt: VideoPrompt) -> str:
    """The `integrated_multimodal_description` body: shots, opening folded in.

    In the base modes the style and the initial composition are the first thing
    *inside* `[Shot 1]`, so there is nothing ahead of the marker.
    """
    shots = prompt.storyboard.shots()
    ids = _speaker_ids(prompt)

    if not shots:
        # Nothing to number, so there is no shot marker to open: the opening
        # sentences, any global camera rule and the screen text are all of it.
        return _join(
            [prompt.style, prompt.overview, prompt.camera, _screen_text_clause(prompt)]
        )

    body = " ".join(
        _shot(prompt, index, beats, ids, is_first=index == 1)
        for index, beats in enumerate(shots, start=1)
    )
    # No slot exists ahead of [Shot 1] in the base modes — the style opens it —
    # so a whole-video instruction can only trail the shots.
    return _join([body, _identity_guard(prompt, ids)])


def _opening(prompt: VideoPrompt) -> str:
    """The one or two sentences that precede `[Shot 1]` — full-reference only."""
    return _join([prompt.style, prompt.overview])


def _shot_one_opening(prompt: VideoPrompt, prose: str) -> str:
    """`Live-action, cinematic, a medium-wide shot frames ...` — style first.

    The style contract opens the first sentence and the next clause continues
    it after a comma, the shape every case in the guide uses. A scene overview,
    when there is one, takes that slot and the shot's own composition follows
    as its own sentence.
    """
    style = prompt.style.strip().rstrip(".")
    pieces = [piece for piece in (prompt.overview.strip(), prose.strip()) if piece]

    if not pieces:
        return _ensure_period(_capitalize(style)) if style else ""

    head, *rest = pieces
    opening = f"{_capitalize(style)}, {head}" if style else _capitalize(head)
    return _join([opening, *rest]) if rest else opening


def _shot(
    prompt: VideoPrompt,
    index: int,
    beats: list[Beat],
    ids: dict[str, str],
    *,
    is_first: bool,
) -> str:
    """One `[Shot N] ...` segment: the beat that cuts to it, plus its continuations.

    A shot is a group of beats now, not one. The first opens the marker the way
    it always did; any no-cut beats after it are folded in behind as timed
    sentences, so the whole group renders as one uninterrupted take.
    """
    lead, *rest = beats
    opening, trailing = _split_lines(lead)
    head, action = _split_prose(lead, split=bool(opening))

    if is_first:
        # Shot 1 carries no cut time: the style opening leads it instead.
        text = _shot_one_opening(prompt, head)
    else:
        timecode = format_timecode(lead.start)
        transition = lead.transition.strip() or h3_spec.DEFAULT_TRANSITION
        text = f"At {timecode}, {transition} {head}".strip()

    text = _join([text, _dialogue(opening, ids), action])
    text = _weave_camera(text, lead, prompt, is_first=is_first)
    parts = [f"[Shot {index}] {text}".strip()]
    if is_first:
        parts.append(_screen_text_clause(prompt))
    parts.append(_dialogue(trailing, ids))
    parts += [_continuation(beat, ids) for beat in rest]
    return _ensure_period(_join(parts))


def _continuation(
    beat: Beat, ids: dict[str, str], subjects: dict[str, str] | None = None
) -> str:
    """A no-cut beat, as a timed sentence inside the shot it continues.

    Leads with the camera — `At 00:03.000, the camera pushes in ... as she
    lifts the mug` — mirroring the way a cut sentence leads with its transition
    phrase. The order is load-bearing, not stylistic: measured at one seed on
    2026-08-31, the same move written *after* its action as a trailing sentence
    produced no camera movement at all, while the camera-led form rendered it.

    With no motion the timecode simply leads the action, which is how a beat
    that only turns the performance is written.

    `angle` means something different here than on a cut. Nobody is cutting to
    a new framing, so it is where the move *arrives* — written `into a wider
    view` behind the motion, never comma-glued to the action, which would read
    as a sentence with two subjects.
    """
    opening, trailing = _split_lines(beat)
    timecode = format_timecode(beat.start)
    motion = beat.camera.strip()
    angle = beat.angle.strip()
    action = beat.action.strip()

    if motion:
        lead = f"At {timecode}, {_camera_clause(motion)}"
        if angle:
            lead = f"{lead} into {angle}"
        if action and not opening:
            # One sentence, camera first: the order the model actually obeys.
            lead, action = f"{lead} as {action}", ""
    else:
        # No camera to lead with, so the prose does, glued exactly as a cut's is.
        prose = _beat_prose(beat) if not opening else angle
        lead = f"At {timecode}, {prose}" if prose else f"At {timecode}"
        if not opening:
            action = ""

    return _join(
        [
            lead,
            _dialogue(opening, ids, subjects),
            action,
            _dialogue(trailing, ids, subjects),
        ]
    )


# -- ref2v: the six-section full-reference document -----------------------


def _compose_reference(prompt: VideoPrompt) -> str:
    """The full-reference rewrite: six labelled sections in the official order."""
    blocks: list[str] = []

    subjects = _subject_definitions(prompt)
    if subjects:
        blocks.append(f"subject_definitions:\n{subjects}")

    summary = _summary(prompt)
    if summary:
        blocks.append(f"summary:\n{summary}")

    retention = _retention(prompt)
    if retention:
        blocks.append(f"retention_analysis:\n{retention}")

    detail = _detailed_description(prompt)
    if detail:
        blocks.append(f"detailed_description:\n{detail}")

    # The full-reference example lays every section out the same way: the label
    # on its own line, the body beneath it. Base mode keeps the inline form its
    # own guide shows.
    soundscape = _soundscape(prompt)
    if soundscape:
        blocks.append(f"overall_soundscape:\n{soundscape}")

    blocks.append(f"non_diegetic_music:\n{_music(prompt)}")
    return "\n\n".join(blocks)


def _subject_definitions(prompt: VideoPrompt) -> str:
    """Every referenced item, one line each — subjects first, then bare assets.

    The split is the guide's own: `<Subject N>` names content the target video
    contains, `<Picture N>` / `<Video N>` / `<Audio N>` name the files it was
    given. A picture that only *shows* a subject writes no line of its own
    (guide 2.2) — it is cited inside that subject's sentence — so a character
    with three identity sheets is defined once rather than three times over.
    """
    ids = _speaker_ids(prompt)
    lines = [
        line
        for line in (_subject_line(subject, prompt) for subject in prompt.subjects.items)
        if line
    ]
    for item in prompt.references.items:
        if prompt.subject_owns(item):
            continue
        line = _asset_line(item, prompt, ids)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _subject_line(subject: Subject, prompt: VideoPrompt) -> str:
    """`<Subject 1> is the person in <Picture 1>, dark hair and a grey hoodie.`

    The description is optional, and dropping it is a deliberate setting rather
    than an omission: a subject the author has asked not to describe still has
    to be *defined*, because `summary` and the prose refer to its tag. What is
    left is the routing sentence alone — `<Subject 1> is the setting in
    <Picture 1>` — which says which files the tag stands for and stops there,
    instead of listing features the camera then goes and finds.

    Empty is returned only for a subject that has nothing at all to say: no
    description, no noun and no visual source.
    """
    tag = prompt.subjects.tag(subject)
    role = subject.role.strip().rstrip(".")
    noun = subject.noun.strip()
    tags = _source_tags(subject, prompt)
    if not tags:
        if role:
            return f"{tag} is {role}."
        return f"{tag} is the {noun}." if noun else ""
    cited = _and_list(tags)
    if noun:
        return f"{tag} is the {noun} in {cited}, {role}." if role else             f"{tag} is the {noun} in {cited}."
    return f"{tag} is {role}, shown in {cited}." if role else f"{tag} is shown in {cited}."


def _asset_line(item: Reference, prompt: VideoPrompt, ids: dict[str, str]) -> str:
    """A standalone file's definition line, or empty when it has nothing to say."""
    role = item.role.strip().rstrip(".")
    owner = _audio_owner(item, prompt, ids)
    if owner:
        role = role or "the voice reference"
        if owner.startswith("("):
            # A speaker with no subject of their own: the description carries
            # the id directly, per guide 2.4.
            if owner not in role:
                role = f"{role} {owner}"
        elif "<Subject" not in role:
            role = f"{role} for {owner}"
    if not role:
        return ""
    return f"{prompt.references.tag(item)} is {role}."


def _audio_owner(item: Reference, prompt: VideoPrompt, ids: dict[str, str]) -> str:
    """`<Subject 2> (S1)` for an audio track that belongs to a subject.

    The guide asks an audio definition to reuse the target video's global
    speaker id rather than mint one of its own, so the `(Sx)` here is the same
    number the dialogue already carries — and is omitted for a subject who
    never speaks, since there is then no id to reuse.
    """
    if not item.is_audio:
        return ""
    subject = prompt.subjects.citing(item.path)
    if subject is not None:
        tag = prompt.subjects.tag(subject)
        speaker = ids.get(subject.member, "")
        return f"{tag} ({speaker})" if speaker else tag
    # No subject, but the row names a member who speaks: the guide still wants
    # the global speaker id, carried by the voice description instead of a
    # label. The id is never minted here — only reused.
    speaker = ids.get(item.member, "")
    return f"({speaker})" if speaker else ""


def _source_tags(subject: Subject, prompt: VideoPrompt) -> list[str]:
    """The visual files this subject is cited from, in reference order.

    Audio is left out: a voice is not something a subject is *shown* in, and it
    introduces itself on its own `<Audio N>` line instead.
    """
    tags: list[str] = []
    for item in prompt.references.items:
        if item.is_audio or item.path not in subject.sources:
            continue
        tag = prompt.references.tag(item)
        if tag not in tags:
            tags.append(tag)
    return tags


def _and_list(tags: list[str]) -> str:
    if len(tags) < 2:
        return "".join(tags)
    return f"{', '.join(tags[:-1])} and {tags[-1]}"




def _summary(prompt: VideoPrompt) -> str:
    tasks = [task.strip() for task in prompt.ref_task if task.strip()] or _default_tasks(prompt)
    prefix = f"[{' + '.join(tasks)}]"
    body = prompt.ref_summary.strip()
    if not body:
        tags = ", ".join(prompt.defined_tags())
        body = f"The target video is generated using {tags}." if tags else ""
    # The prefix is a label, not a sentence, so it is not joined as one.
    rest = _join([_edit_opening(prompt), body])
    return _ensure_period(f"{prefix} {rest}".strip()) if rest else prefix


def _edit_opening(prompt: VideoPrompt) -> str:
    """The sentence the guide (3) puts first on a video-editing summary.

    Written for the author rather than demanded of them, and skipped when they
    have already said it themselves — the point is that the model is told which
    video is being edited, not that the sentence appears twice.
    """
    if "video editing" not in [task.strip() for task in prompt.ref_task]:
        return ""
    videos = prompt.references.of("video")
    if not videos:
        return ""
    tag = prompt.references.tag(videos[0])
    if "edited version of" in prompt.ref_summary:
        return ""
    return f"The target video is an edited version of {tag}."


def _default_tasks(prompt: VideoPrompt) -> list[str]:
    """A sensible task-type prefix when the author hasn't set one."""
    tasks = ["reference generation"]
    if any(item.is_audio for item in prompt.references.items):
        tasks.append("audio reference")
    return tasks


def _retention(prompt: VideoPrompt) -> str:
    """One marker per referenced item, keyed the way `subject_definitions` is.

    Which is the point: what survives into the target video is the *person*,
    not the PNG. Keyed on files, this said "this face sheet is fully preserved"
    once per sheet and never named the character at all.
    """
    refs = prompt.references
    subjects = prompt.subjects
    lines = []
    for subject in subjects.items:
        note = _subject_note(subject)
        if not note:
            continue
        marker = subject.retention.strip() or h3_spec.DEFAULT_VISUAL_RETENTION
        lines.append(f"{subjects.tag(subject)}{_scope(prompt, subject)}: {marker} - {note}.")
    for item in refs.items:
        if prompt.subject_owns(item):
            continue
        marker = item.retention.strip() or _default_retention(item)
        note = item.role.strip().rstrip(".") or "as defined"
        lines.append(f"{refs.tag(item)}{_scope(prompt, item)}: {marker} - {note}.")
    return "\n".join(lines)


def _subject_note(subject: Subject) -> str:
    """What a retention marker is *about*, in as few words as it has.

    The marker is the load-bearing half of the line — `fully_preserved` is the
    instruction, the note only says what it applies to — so a subject with its
    description turned off keeps its entry and names itself by noun. Dropping
    the row instead would quietly stop asking for the reference to be preserved
    at all, which is the opposite of what turning off a *description* means.
    """
    role = subject.role.strip().rstrip(".")
    if role:
        return role
    noun = subject.noun.strip()
    return f"the {noun}" if noun else ""


def _scope(prompt: VideoPrompt, item: object) -> str:
    """` (appears in [Shot 1], [Shot 3])`, the guide's section-4 parenthetical.

    The section is meant to record *where* each item lands, not only how much
    of it survives. Two things can tell us: a subject whose member speaks is in
    every shot they speak in, and any label woven into a shot's own prose is in
    that shot. A cast member present for the whole segment is in all of them.

    Nothing is asserted that cannot be shown — an item we cannot place gets no
    parenthetical rather than a guessed one, and `subjects.unplaced` says so.
    Audio is skipped: it is a layer, not a thing in the frame, and the guide's
    own examples leave its entries bare.
    """
    if isinstance(item, Reference):
        if item.is_audio:
            return ""
        tag = prompt.references.tag(item)
        member = item.member
    else:
        tag = prompt.subjects.tag(item)
        member = item.member

    # Numbered by the shot a beat composes into, not by its position: a beat
    # that continues shot 2 without cutting appears in [Shot 2], and there is
    # no [Shot 3] for the reader to look at.
    beats = prompt.storyboard.ordered_beats()
    numbers = prompt.storyboard.shot_numbers()
    shots: list[int] = []
    for number, beat in zip(numbers, beats):
        spoken = any(
            member and member in line.speakers for line in beat.lines if line.text.strip()
        )
        woven = tag in " ".join([beat.angle, beat.action, beat.camera])
        if (spoken or woven) and number not in shots:
            shots.append(number)

    total = len(prompt.storyboard.shots())
    if member and member in prompt.cast and total:
        shots = list(range(1, total + 1))  # listed as present for the segment
    if not shots:
        return ""
    if total > 1 and len(shots) == total:
        return " (appears throughout)"
    return f" (appears in {_and_list([f'[Shot {n}]' for n in shots])})"


def _default_retention(item: Reference) -> str:
    return (
        h3_spec.DEFAULT_AUDIO_RETENTION
        if item.is_audio
        else h3_spec.DEFAULT_VISUAL_RETENTION
    )


def _detailed_description(prompt: VideoPrompt) -> str:
    """Like the base description, but the opening leads on its own line before
    [Shot 1] and each shot sits on its own line (per the full-reference
    example)."""
    shots = prompt.storyboard.shots()
    ids = _speaker_ids(prompt)
    lines: list[str] = []
    opening = _join([_opening(prompt), _identity_guard(prompt, ids)])
    if opening:
        lines.append(opening)
    for index, beats in enumerate(shots, start=1):
        lines.append(_ref_shot(prompt, index, beats, ids, is_first=index == 1))
    if not shots:  # no [Shot 1] to carry it
        screen_text = _screen_text_clause(prompt)
        if screen_text:
            lines.append(screen_text)
    return "\n".join(lines)


def _ref_shot(
    prompt: VideoPrompt,
    index: int,
    beats: list[Beat],
    ids: dict[str, str],
    *,
    is_first: bool,
) -> str:
    subjects = _subject_tags_by_speaker(prompt)
    lead, *rest = beats
    opening, trailing = _split_lines(lead)
    head, action = _split_prose(lead, split=bool(opening))
    if is_first:
        text = head
    else:
        timecode = format_timecode(lead.start)
        transition = lead.transition.strip() or h3_spec.DEFAULT_TRANSITION
        text = f"At {timecode}, {transition} {head}".strip()
    text = _join([text, _dialogue(opening, ids, subjects), action])
    text = _weave_camera(text, lead, prompt, is_first=False)  # per-shot camera only
    parts = [f"[Shot {index}] {text}".strip()]
    if is_first:
        parts.append(_screen_text_clause(prompt))
    parts.append(_dialogue(trailing, ids, subjects))
    parts += [_continuation(beat, ids, subjects) for beat in rest]
    return _ensure_period(_join(parts))


# -- speakers and dialogue (shared by base and ref) -----------------------


def _speaker_ids(prompt: VideoPrompt) -> dict[str, str]:
    """Map each speaker key to `S1`, `S2`, ... in order of first vocal appearance."""
    return prompt.storyboard.speaker_ids()


def _identity_guard(prompt: VideoPrompt, ids: dict[str, str]) -> str:
    """Name the people who share a shot and say they are not the same person.

    Two characters present at once is where H3 blends faces or swaps who is
    who, and the format has no field that says otherwise — so the instruction
    goes in as prose. Only speakers that actually appear together in one shot
    are listed: two people who never share a frame cannot be swapped in it.
    """
    if not prompt.guard_identities:
        return ""

    together: list[str] = []
    for beat in prompt.storyboard.ordered_beats():
        present = [
            key
            for line in beat.lines
            if line.text.strip()
            for key in line.speakers
            if key in ids
        ]
        if len(set(present)) < 2:
            continue
        together += [key for key in present if key not in together]

    if len(together) < 2:
        return ""

    tags = [f"({ids[key]})" for key in sorted(together, key=lambda k: ids[k])]
    named = f"{', '.join(tags[:-1])} and {tags[-1]}"
    return (
        f"Keep {named} visually distinct throughout; they are different people "
        f"and their identities must not merge or swap."
    )


def _split_lines(beat: Beat) -> tuple[list[Line], list[Line]]:
    """The shot's spoken lines, split into the ones that open it and the rest.

    Lines with no words are dropped here rather than in `_dialogue`, so an
    empty line marked `open` cannot force the prose apart for nothing.
    """
    spoken = [line for line in beat.lines if line.text.strip()]
    return (
        [line for line in spoken if line.placement == "open"],
        [line for line in spoken if line.placement != "open"],
    )


def _split_prose(beat: Beat, *, split: bool) -> tuple[str, str]:
    """The beat's prose as `(what opens the shot, what follows the line)`.

    A line can never be the first thing in a shot — the format opens every one
    with its style (Shot 1) or its cut time and transition phrase, and both need
    a composition to land on. So an opening line goes *second*: the framing
    clause holds the front, the action moves behind the speech.

    Without an opening line — or without a framing clause to open on — angle and
    action stay glued as one clause, exactly as they compose today. In that
    second case the line still lands ahead of the camera sentence, which is as
    far forward as the shot allows; `lint.dialogue.no_opening` is what says the
    front slot was not available.
    """
    angle = beat.angle.strip()
    action = beat.action.strip()
    if not split or not angle or not action:
        return _beat_prose(beat), ""
    return angle, action


def _dialogue(
    lines: list[Line], ids: dict[str, str], subjects: dict[str, str] | None = None
) -> str:
    """The given spoken lines, each its own sentence."""
    return " ".join(_line(line, ids, subjects) for line in lines)


def _line(line: Line, ids: dict[str, str], subjects: dict[str, str] | None = None) -> str:
    """`The young woman (S1) says: <d>[English] Hi.</d>` and its variants.

    Only the language tag and the verbatim words may go inside `<d>`, so the
    voiceover lips-closed note and the `<scenetrans>` / `<cutoff>` seam markers
    all sit outside it.

    In full-reference mode a speaker who is a defined subject leads with their
    label — `<Subject 1> (S1), a woman in her thirties, says: ...` — because the
    guide (5.4) asks for both: `<Subject N>` says which referenced identity this
    is, `(Sx)` says which voice. The appearance stays as apposition, so the
    first mention still describes them and later ones are just the two tags.
    """
    tag = ",".join(ids.get(key, "S?") for key in line.speakers if key)
    id_tag = f"({tag})" if tag else ""
    verb = h3_spec.VOICEOVER_VERB if line.voiceover else (line.verb.strip() or "says")
    label = _subject_lead(line, subjects)
    describe = line.describe.strip()
    if label:
        lead = f"{label} {id_tag}".strip()
        if describe:
            lead = f"{lead}, {describe.rstrip('.,')},"
        intro = " ".join(part for part in (lead, verb) if part)
    else:
        intro = " ".join(part for part in (describe, id_tag, verb) if part)
    if line.delivery.strip():
        intro = f"{intro} {line.delivery.strip()}"
    language = line.language.strip() or "English"

    spoken = f"{_capitalize(intro)}: <d>[{language}] {_spoken(line)}</d>"
    if line.voiceover:
        # The guide states this immediately after the voiceover block, as a
        # lower-case continuation of the same sentence.
        possessive = line.possessive.strip() or h3_spec.DEFAULT_POSSESSIVE
        spoken = f"{spoken} while {possessive} lips remain completely closed"

    pieces = [_ensure_period(spoken)]
    if line.continuity == "into":
        pieces.append(h3_spec.SCENE_TRANSITION_TAG)
    if line.cutoff:
        pieces.append(h3_spec.CUTOFF_TAG)
    pieces += [_ensure_period(part) for part in _continuity_statements(line)]

    rendered = " ".join(pieces)
    # The incoming half is tagged at its start, which is where it connects.
    if line.continuity == "from":
        rendered = f"{h3_spec.SCENE_TRANSITION_TAG} {rendered}"
    return rendered


def _subject_lead(line: Line, subjects: dict[str, str] | None) -> str:
    """The `<Subject N>` label(s) this line's speakers answer to, if any.

    All-or-nothing per line: a chorus where only some voices are defined
    subjects would read as though the rest were unidentified, so it falls back
    to plain prose rather than half-labelling the group.
    """
    if not subjects:
        return ""
    keys = [key for key in line.speakers if key]
    tags = [subjects.get(key, "") for key in keys]
    if not tags or not all(tags):
        return ""
    return _and_list(list(dict.fromkeys(tags)))


def _subject_tags_by_speaker(prompt: VideoPrompt) -> dict[str, str]:
    """Speaker key -> `<Subject N>`, for the members that have one."""
    return {
        subject.member: prompt.subjects.tag(subject)
        for subject in prompt.subjects.items
        if subject.member
    }


# Anything that already ends the line's punctuation, terminal or not.
_PUNCTUATED = ".?!\u2026,;:\u2014\u2013-"


def _spoken(line: Line) -> str:
    """The verbatim line, ended on `.`, `?` or `!` as the format requires.

    Only *complete* utterances get the mark. A line the video cuts off, or one
    that runs on into the next shot, is unfinished on purpose — that is exactly
    what `<cutoff>` and `<scenetrans>` say — and closing it with a full stop
    would contradict the tag sitting beside it. A line that already ends on any
    punctuation is left as typed.

    The words are never touched either way: `<d>` is the one place the original
    wording and language have to survive intact, so this completes a line
    rather than rewriting one.
    """
    body = line.text.strip()
    if not body or body[-1] in _PUNCTUATED:
        return body
    if line.cutoff or line.continuity == "into":
        return body
    return f"{body}."


def _continuity_statements(line: Line) -> list[str]:
    """The seam wants the tag *and* an explicit statement of what carries over."""
    statements = []
    phrase = h3_spec.CONTINUITY_PHRASES.get(line.continuity)
    if phrase:
        statements.append(f"The speech {phrase}")
    if line.cutoff:
        statements.append(f"The speech {h3_spec.CUTOFF_PHRASE}")
    return statements


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _beat_prose(beat: Beat) -> str:
    """Angle and action as one clause, e.g. `a close-up, steam rising`."""
    angle = beat.angle.strip()
    action = beat.action.strip()
    if angle and action:
        return f"{angle}, {action}"
    return angle or action


def _weave_camera(text: str, beat: Beat, prompt: VideoPrompt, *, is_first: bool) -> str:
    """Append the shot's camera motion as its own sentence.

    Per-shot `beat.camera` is a motion phrase and gets a `The camera ...`
    subject; the global `prompt.camera` is only a shot-one fallback and is
    appended verbatim (it usually describes cutting style, not motion).
    """
    motion = beat.camera.strip()
    if motion:
        return f"{_ensure_period(text)} {_camera_sentence(motion)}" if text else _camera_sentence(motion)
    if is_first and prompt.camera.strip():
        fallback = _ensure_period(prompt.camera.strip())
        return f"{_ensure_period(text)} {fallback}" if text else fallback
    return text


def _camera_clause(motion: str) -> str:
    """`pushes in ... toward the door` -> `the camera pushes in ... toward the door`.

    The subject supplied, nothing else: a clause that can sit mid-sentence
    after a timecode, or be capitalised into a sentence of its own.
    """
    lowered = motion.lower()
    if lowered.startswith("the camera"):
        return motion
    if lowered.startswith("camera"):
        return f"the {motion}"
    return f"the camera {motion}"


def _camera_sentence(motion: str) -> str:
    """`pushes in ... toward the door` -> `The camera pushes in ... toward the door.`

    Always its own sentence, so it is capitalised even when the author already
    wrote the subject in lower case ("the camera holds level").
    """
    return _ensure_period(_capitalize(_camera_clause(motion)))


def _music(prompt: VideoPrompt) -> str:
    """The audience-only score. Empty really is `N/A` — no score is a choice."""
    music = prompt.music.strip()
    return _ensure_period(music) if music else "N/A"


def _soundscape(prompt: VideoPrompt) -> str:
    """Diegetic bed plus any timed cues, as sentences. Empty if unwritten.

    The field is one continuous paragraph of English sentences, so the bed and
    each cue are capitalised and closed rather than comma-spliced. Sound is
    generated in the same pass as the picture, so cue timing lives in the
    prompt; dialogue and score are excluded by construction (they live in the
    description and `non_diegetic_music`).

    An unwritten soundscape composes to nothing rather than `N/A`: `N/A` claims
    the video is silent throughout, which is a claim only the author can make.
    """
    parts = [prompt.audio.strip()] if prompt.audio.strip() else []
    parts += [
        cue.sentence() for cue in sorted(prompt.storyboard.cues, key=lambda c: c.start)
    ]
    return _join(parts)


def _screen_text_clause(prompt: VideoPrompt) -> str:
    """Visible text, quoted verbatim in English double quotes (guide 4.5).

    The guide states on-screen text inside the shot it appears in — signage
    written straight into a beat's `action` already does that. The flat
    `screen_text` list has no shot of its own, so it is stated in `[Shot 1]`,
    where establishing text nearly always lives, rather than trailing after the
    final shot, where the model would read it as a closing card.
    """
    entries = [text.strip() for text in prompt.screen_text if text.strip()]
    if not entries:
        return ""
    quoted = ", ".join(f'"{text}"' for text in entries)
    return f"On-screen text reads {quoted}."


def _join(items: list[str]) -> str:
    """Space-join non-empty pieces, each capitalised and closed as a sentence."""
    return " ".join(
        _ensure_period(_capitalize(item.strip())) for item in items if item.strip()
    )


def _ensure_period(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    # A dialogue tag closes its own sentence; the words inside carry the
    # period. The seam markers are punctuation of their own too.
    if text.endswith(_CLOSING_TAGS) or text[-1] in _SENTENCE_ENDINGS:
        return text
    return f"{text}."
