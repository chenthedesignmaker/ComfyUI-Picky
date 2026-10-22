"""Checks that run against a prompt before you spend a generation on it.

Two kinds of rule live here. Hard ones mirror the API and fail the request if
violated (duration, resolution, reference counts, prompt length). Soft ones
encode what MiniMax's official prompts do and yours probably should —
storyboard contiguity above all, since overlapping timecodes are the single
most common way a hand-written storyboard drifts.

A third kind — `_check_adherence` — encodes how H3 is known to *misread* a
prompt rather than how the format is spelled: cuts it merges, quality words it
cannot see, speech that will not fit the beat holding it. These never block a
render; they are the difference between a prompt that parses and one the model
actually follows.

This lints one generation. Checks that span a whole edit — a shot trimmed
past its source, a pick left behind by a prompt edit — live in `sequence`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum

from . import h3_spec, mentions
from .camera_path import validate_path, camera_media, loop_status
from .cast import SUBJECT_NOUNS, Cast, resolve, unknown_ids
from .compose import compose
from .document import REFERENCE_KINDS, VideoPrompt
from .storyboard import Beat, Line, format_span


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: Severity
    message: str
    where: str = ""

    def __str__(self) -> str:
        prefix = f"{self.severity}: "
        suffix = f" ({self.where})" if self.where else ""
        return f"{prefix}{self.message}{suffix}"


def lint(prompt: VideoPrompt, cast: Cast | None = None) -> list[Finding]:
    """Every finding for `prompt`, worst first.

    Linting happens on the *resolved* prompt, so a member's identity files
    count toward the reference caps and their prose counts toward the length —
    the limits apply to what the model is actually sent.
    """
    missing = unknown_ids(prompt, cast) if cast is not None else []
    # Which references the author attached here, before the cast adds its own.
    # `_check_media` needs the difference: a shot cannot be told to remove
    # references it does not appear to have.
    authored = {item.path for item in prompt.references.items}
    resolved = camera_media(resolve(prompt, cast))
    # Mentions are checked against both documents at once: the `@name` only
    # exists on the authored one, and whether it has a label to become only
    # shows on the resolved one.
    mention_findings = _check_mentions(prompt, resolved, cast)
    prompt = resolved
    findings = [
        *_check_camera_paths(prompt),
        *_check_cast(missing),
        *mention_findings,
        *_check_settings(prompt),
        *_check_storyboard(prompt),
        *_check_beats(prompt),
        *_check_dialogue(prompt),
        *_check_audio(prompt),
        *_check_blocks(prompt),
        *_check_adherence(prompt),
        *_check_references(prompt),
        *_check_reference_doc(prompt),
        *_check_media(prompt, authored),
        *_check_length(prompt),
    ]
    return sorted(findings, key=lambda f: -f.severity)


def _check_camera_paths(prompt: VideoPrompt) -> list[Finding]:
    found = []
    for index, beat in enumerate(prompt.storyboard.beats):
        issue = validate_path(beat.camera_path)
        if issue:
            found.append(Finding("camera_path.invalid", Severity.ERROR, issue, f"beat.{index}.camera"))
        elif isinstance(beat.camera_path, dict) and beat.camera_path.get("frame_mode") == "motion":
            if len(prompt.storyboard.beats) != 1 or beat.start != 0 or beat.end != prompt.duration:
                found.append(Finding("camera_path.invalid", Severity.ERROR,
                    "Motion Frame needs one camera path covering the whole generation. Use a separate source for this camera pass.", f"beat.{index}.camera"))
        elif beat.camera_path is not None and beat.camera_path.get("freeze") and beat.lines:
            found.append(Finding("camera_path.freeze_dialogue", Severity.WARNING,
                                 "Freeze scene holds the subject still; turn it off for a speaking performance.",
                                 f"beat.{index}.camera"))
    enabled, reason = loop_status(prompt)
    if not enabled and any(isinstance(b.camera_path, dict) and b.camera_path.get("loop_closure") == "auto" for b in prompt.storyboard.beats):
        found.append(Finding("camera_path.loop_disabled", Severity.WARNING, reason, "camera"))
    return found


def errors(findings: list[Finding]) -> list[Finding]:
    """Just the findings that would break a real request."""
    return [f for f in findings if f.severity is Severity.ERROR]


def _check_cast(missing: list[str]) -> list[Finding]:
    """Members the prompt names that the project no longer has.

    A deleted member leaves its id behind on every prompt that used it, and the
    symptom — a character who silently loses their description — is invisible
    in the composed text. Naming the dangling id is the only way to see it.
    """
    if not missing:
        return []
    return [
        Finding(
            "cast.unknown",
            Severity.WARNING,
            f"This prompt names {len(missing)} cast member(s) the project does "
            f"not have ({', '.join(missing)}). They were probably deleted; the "
            f"appearance and identity files they carried are no longer composed "
            f"in. Point the line at a member that exists, or write the "
            f"description on the prompt itself.",
        )
    ]


def _check_mentions(
    authored: VideoPrompt, resolved: VideoPrompt, cast: Cast | None
) -> list[Finding]:
    """`@name` pointers that will not land the way they read.

    Three ways a mention disappoints, none of them visible in the composed text
    — which is the point of checking: an `@` that resolved to nothing looks
    exactly like prose that was written that way on purpose.

    Reported once per member per rule rather than once per mention: a character
    named eight times in a shot has one problem, not eight, and eight copies of
    the same sentence is how a findings list stops being read.
    """
    if cast is None or not len(cast):
        return []

    names = [(member.name, member.id) for member in cast]
    speakers = authored.storyboard.speaker_ids()
    sharing: dict[str, list[str]] = {}
    for member in cast:
        folded = member.name.strip().casefold()
        if folded:
            sharing.setdefault(folded, []).append(member.id)

    found: list[Finding] = []
    seen: set[tuple[str, str]] = set()

    def note(rule: str, key: str, severity: Severity, message: str, where: str) -> None:
        if (rule, key) in seen:
            return
        seen.add((rule, key))
        found.append(Finding(rule, severity, message, where))

    for field in mentions.fields(authored):
        for word in mentions.unknown(field.text, names):
            note(
                "mentions.unknown",
                word.casefold(),
                Severity.WARNING,
                f"`@{word}` names nobody in the cast, so it composes as the "
                f"literal text `@{word}` and the model reads an at-sign. Check "
                f"the spelling, or add the member — cast names are matched "
                f"whole and case-insensitively.",
                field.label,
            )
        for key in mentions.keys(field.text, names):
            member = cast.get(key)
            if member is None:
                continue
            others = sharing.get(member.name.strip().casefold(), [])
            if len(others) > 1:
                note(
                    "mentions.ambiguous",
                    member.name.strip().casefold(),
                    Severity.WARNING,
                    f"{len(others)} cast members are called '{member.name}', and a "
                    f"mention is matched by name — so every `@{member.name}` in this "
                    f"prompt resolves to whichever of them the cast lists first. "
                    f"Give them distinct names.",
                    field.label,
                )
            if resolved.subjects.for_member(member.id) is not None:
                continue
            if member.id in speakers:
                continue
            noun = SUBJECT_NOUNS.get(member.kind, "person")
            note(
                "mentions.untagged",
                member.id,
                Severity.INFO,
                f"`@{member.name}` composes as 'the {noun}', because they have no "
                f"identity file to carry a <Subject N> and no line to carry an "
                f"(Sx). That is all the label there is to point at. Attach a "
                f"reference to them, or give them a line, and the mention "
                f"sharpens on its own.",
                field.label,
            )

    return found


def _check_settings(prompt: VideoPrompt) -> list[Finding]:
    found: list[Finding] = []
    duration = prompt.storyboard.duration

    if not h3_spec.is_valid_duration(duration):
        found.append(
            Finding(
                "duration.invalid",
                Severity.ERROR,
                f"Duration must be a whole number of seconds from "
                f"{h3_spec.MIN_DURATION_S} to {h3_spec.MAX_DURATION_S}; got {duration!r}. "
                f"Anything under {h3_spec.TRAINED_MIN_DURATION_S}s still generates, "
                f"but it is outside the trained range and warns.",
            )
        )
    else:
        frames = prompt.frames
        rendered = prompt.rendered_seconds
        if rendered != duration:
            found.append(
                Finding(
                    "duration.snapped",
                    Severity.INFO,
                    f"{duration}s becomes {frames} frames at {h3_spec.FPS} fps, so the "
                    f"clip runs {rendered}s. Frame counts snap up to the model's "
                    f"{h3_spec.FRAME_GRID}k+{h3_spec.FRAME_OFFSET} grid.",
                )
            )
        if not h3_spec.is_trained_length(frames):
            found.append(
                Finding(
                    "duration.untrained",
                    Severity.WARNING,
                    f"{frames} frames is outside the trained range of "
                    f"{h3_spec.TRAINED_MIN_FRAMES}-{h3_spec.TRAINED_MAX_FRAMES}. "
                    f"It will run, but the node's own tooltip calls this untested.",
                )
            )

    found += _check_canvas(prompt)
    return found


def _check_canvas(prompt: VideoPrompt) -> list[Finding]:
    """Width and height as the local nodes want them: multiples of 32, capped."""
    found: list[Finding] = []
    width, height = prompt.width, prompt.height

    for name, value in (("width", width), ("height", height)):
        if value < h3_spec.CANVAS_MULTIPLE:
            found.append(
                Finding(
                    "canvas.too_small",
                    Severity.ERROR,
                    f"{name} is {value}; the minimum is {h3_spec.CANVAS_MULTIPLE}.",
                )
            )
        elif value % h3_spec.CANVAS_MULTIPLE:
            found.append(
                Finding(
                    "canvas.off_grid",
                    Severity.ERROR,
                    f"{name} is {value}, which is not a multiple of "
                    f"{h3_spec.CANVAS_MULTIPLE}.",
                )
            )

    if width >= h3_spec.CANVAS_MULTIPLE and height >= h3_spec.CANVAS_MULTIPLE:
        if width * height > h3_spec.MAX_PIXELS:
            adapted = h3_spec.adapt_canvas(width, height)
            found.append(
                Finding(
                    "canvas.too_large",
                    Severity.WARNING,
                    f"{width}x{height} is {width * height} pixels, over the "
                    f"{h3_spec.MAX_PIXELS} cap. The node would use "
                    f"{adapted[0]}x{adapted[1]}.",
                )
            )

    return found


def _check_storyboard(prompt: VideoPrompt) -> list[Finding]:
    storyboard = prompt.storyboard
    beats = storyboard.ordered_beats()

    if not beats:
        return [
            Finding(
                "storyboard.empty",
                Severity.WARNING,
                "No beats. H3 reads prompts as timelines — without a storyboard "
                "it will stretch one idea across the whole clip.",
            )
        ]

    found: list[Finding] = []

    for left, right in storyboard.overlaps:
        a, b = beats[left - 1], beats[right - 1]
        amount = round(min(a.end, b.end) - max(a.start, b.start), 3)
        found.append(
            Finding(
                "storyboard.overlap",
                Severity.ERROR,
                f"Shots {left} and {right} overlap by {amount}s. Official H3 "
                f"prompts use contiguous blocks; overlapping spans make pacing drift.",
                f"{format_span(a.start, a.end)} vs {format_span(b.start, b.end)}",
            )
        )

    for start, end in storyboard.gaps:
        found.append(
            Finding(
                "storyboard.gap",
                Severity.WARNING,
                f"Nothing is scheduled for {round(end - start, 3)}s of the clip.",
                format_span(start, end),
            )
        )

    last = max(beat.end for beat in beats)
    if last > storyboard.duration:
        found.append(
            Finding(
                "storyboard.overrun",
                Severity.ERROR,
                f"The storyboard runs to {last}s but the clip is "
                f"{storyboard.duration}s. Anything past the budget is dropped.",
            )
        )

    return found


def _check_beats(prompt: VideoPrompt) -> list[Finding]:
    """Per-beat sanity, then the pacing rules — which count *shots*, not beats.

    A no-cut beat is a moment inside the shot before it, so it is not a shot
    for any of the length or density heuristics: five timed moves in one
    continuous take is not five cuts per second, and a 0.6s camera move is not
    a flash frame. The rules that judge cutting are applied to the groups
    `Storyboard.shots()` returns; the rules that judge a moment stay per beat.
    """
    storyboard = prompt.storyboard
    beats = storyboard.ordered_beats()
    numbers = storyboard.shot_numbers()
    shots = storyboard.shots()
    found: list[Finding] = []

    for number, beat in zip(numbers, beats):
        where = format_span(beat.start, beat.end)
        names = f"Shot {number}" if beat.cut else f"The move at {where} in shot {number}"

        if beat.start < 0 or beat.end <= beat.start:
            found.append(
                Finding(
                    "beat.invalid_span",
                    Severity.ERROR,
                    f"{names} has an empty or reversed span.",
                    where,
                )
            )
            continue

        if not beat.cut:
            if beat.duration < h3_spec.MIN_CONTINUATION_S and (beat.camera.strip() or beat.camera_path is not None):
                found.append(
                    Finding(
                        "beat.move_rushed",
                        Severity.INFO,
                        f"{names} gives the camera {beat.duration}s to travel. "
                        f"Under {h3_spec.MIN_CONTINUATION_S}s a move inside a "
                        f"continuous take tends to snap or arrive late — the "
                        f"shot cannot cut to where it is going.",
                        where,
                    )
                )
            if not beat.action.strip() and not beat.camera.strip() and beat.camera_path is None:
                found.append(
                    Finding(
                        "beat.empty_continuation",
                        Severity.WARNING,
                        f"{names} neither moves the camera nor describes an "
                        f"action, so it adds a timecode and nothing to happen "
                        f"at it. Give it a move, give it an action, or fold it "
                        f"into the beat before it.",
                        where,
                    )
                )
            continue

        if beat.duration < h3_spec.MIN_SHOT_S:
            found.append(
                Finding(
                    "beat.too_short",
                    Severity.WARNING,
                    f"Shot {number} is {beat.duration}s — under "
                    f"{h3_spec.MIN_SHOT_S}s it reads as a flash frame.",
                    where,
                )
            )
        elif beat.duration < h3_spec.COMFORTABLE_MIN_SHOT_S:
            found.append(
                Finding(
                    "beat.tight",
                    Severity.INFO,
                    f"Shot {number} is {beat.duration}s. Workable for a trailer "
                    f"cut, but there is little room for the action to land.",
                    where,
                )
            )

        if not beat.action.strip():
            found.append(
                Finding(
                    "beat.no_action",
                    Severity.WARNING,
                    f"Shot {number} has no action described.",
                    where,
                )
            )

    if shots and h3_spec.is_valid_duration(storyboard.duration):
        density = len(shots) / storyboard.duration
        if density > h3_spec.MAX_SHOTS_PER_SECOND:
            found.append(
                Finding(
                    "beat.density",
                    Severity.WARNING,
                    f"{len(shots)} shots in {storyboard.duration}s is "
                    f"{density:.1f} cuts per second. H3 goes to "
                    f"{h3_spec.MAX_DURATION_S}s — consider more room per shot.",
                )
            )
        elif (
            storyboard.duration <= h3_spec.HEADROOM_DURATION_S
            and len(shots) >= h3_spec.HEADROOM_MIN_SHOTS
        ):
            average = round(storyboard.duration / len(shots), 2)
            found.append(
                Finding(
                    "storyboard.headroom",
                    Severity.INFO,
                    f"{len(shots)} shots share a {storyboard.duration}s budget "
                    f"({average}s each) while H3 allows "
                    f"{h3_spec.MAX_DURATION_S}s. Multi-shot range is one of the "
                    f"things this model is for — the extra seconds are free.",
                )
            )

    return found


def _check_dialogue(prompt: VideoPrompt) -> list[Finding]:
    """Spoken lines: every voiced line needs a speaker, every speaker some words."""
    found: list[Finding] = _check_seams(prompt) + _check_speech_fits(prompt)
    for index, beat in enumerate(prompt.storyboard.ordered_beats(), start=1):
        where = format_span(beat.start, beat.end)
        for line in beat.lines:
            has_text = bool(line.text.strip())
            has_speaker = any(key.strip() for key in line.speakers)
            if has_text and not has_speaker:
                found.append(
                    Finding(
                        "dialogue.no_speaker",
                        Severity.WARNING,
                        f"Shot {index} has a spoken line with no speaker. Give the "
                        f"voice a stable key so it maps to (S1), (S2), ...",
                        where,
                    )
                )
            # An opening line moves the action behind the speech, which needs a
            # framing clause left in front to open on — the format never starts
            # a shot with dialogue. Without one the line quietly stays put.
            if (
                has_text
                and line.placement == "open"
                and beat.action.strip()
                and not beat.angle.strip()
            ):
                found.append(
                    Finding(
                        "dialogue.no_opening",
                        Severity.WARNING,
                        f"Shot {index} has a line set to open the shot, but the "
                        f"shot has no composition for the line to follow, so it "
                        f"lands after the action rather than in front of it. "
                        f"Fill in Composition to give the line something to "
                        f"open on.",
                        where,
                    )
                )
            # The words are what compose() keys on: a line without them is
            # dropped whole, taking its speaker, describe, verb and delivery
            # with it. Say so, or the dropped work looks like a broken editor.
            if not has_text:
                found.append(
                    Finding(
                        "dialogue.empty",
                        Severity.WARNING if _line_has_trimmings(line, has_speaker)
                        else Severity.INFO,
                        f"Shot {index} has a spoken line with no words, so none of "
                        f"it reaches the prompt. Fill in Words, or remove the line.",
                        where,
                    )
                )
    return found


def _check_speech_fits(prompt: VideoPrompt) -> list[Finding]:
    """Can the words on a beat actually be said in the time the beat has?

    Nothing in the format stops you writing forty words over a two-second shot;
    H3 simply races the delivery or drops the tail. Lines are spoken one after
    another, so a beat's speech time is the sum of its lines. Lines that are
    *meant* to outrun the beat are exempt: a `continuity` half runs across the
    cut by design, and a `cutoff` line is truncated on purpose.
    """
    found: list[Finding] = []
    for index, beat in enumerate(prompt.storyboard.ordered_beats(), start=1):
        if beat.duration <= 0:
            continue
        words = sum(
            len(line.text.split())
            for line in beat.lines
            if not line.cutoff and not line.continuity
        )
        if not words:
            continue

        seconds = round(words * 60 / h3_spec.SPEECH_WORDS_PER_MINUTE, 1)
        where = format_span(beat.start, beat.end)
        if seconds > beat.duration:
            found.append(
                Finding(
                    "dialogue.overrun",
                    Severity.WARNING,
                    f"Shot {index} holds {words} words but runs {beat.duration}s. "
                    f"At an unhurried {h3_spec.SPEECH_WORDS_PER_MINUTE} words per "
                    f"minute that is about {seconds}s of speech. Shorten the line, "
                    f"lengthen the shot, or direct a faster delivery.",
                    where,
                )
            )
        elif seconds > beat.duration * h3_spec.SPEECH_COMFORTABLE_FILL:
            found.append(
                Finding(
                    "dialogue.tight",
                    Severity.INFO,
                    f"Shot {index} is {beat.duration}s and about {seconds}s of it "
                    f"is speech, leaving almost no room for a breath or a pause.",
                    where,
                )
            )
    return found


def _check_seams(prompt: VideoPrompt) -> list[Finding]:
    """A line crossing a cut is tagged in *both* halves, or in neither.

    `<scenetrans>` marks the connecting point on each side, so an "into" with
    no "from" waiting in the next shot (or the reverse) leaves the model half a
    seam and no continuous audio.
    """
    beats = prompt.storyboard.ordered_beats()
    found: list[Finding] = []
    for index, beat in enumerate(beats, start=1):
        following = beats[index] if index < len(beats) else None
        for line in beat.lines:
            if line.continuity == "into" and not _has_continuity(following, "from"):
                found.append(
                    Finding(
                        "dialogue.seam_unpaired",
                        Severity.WARNING,
                        f"Shot {index} hands a line into the next shot, but no "
                        f"line there carries it over. Tag both halves, or "
                        f"neither.",
                        format_span(beat.start, beat.end),
                    )
                )
            if line.continuity == "from" and not _has_continuity(
                beats[index - 2] if index > 1 else None, "into"
            ):
                found.append(
                    Finding(
                        "dialogue.seam_unpaired",
                        Severity.WARNING,
                        f"Shot {index} carries a line over from the previous "
                        f"shot, but nothing there hands it across.",
                        format_span(beat.start, beat.end),
                    )
                )
    return found


def _has_continuity(beat: Beat | None, marker: str) -> bool:
    return beat is not None and any(line.continuity == marker for line in beat.lines)


def _line_has_trimmings(line: Line, has_speaker: bool) -> bool:
    """True when a wordless line still carries work that is being thrown away.

    `verb` defaults to "says", so only a verb the author changed counts.
    """
    return (
        has_speaker
        or bool(line.voiceover)
        or any(part.strip() for part in (line.describe, line.delivery))
        or line.verb.strip() not in ("", "says")
    )


# How close to the clip edges a cue may sit and still count as covering it.
_CUE_EDGE_S = 0.001

# Cue text that is really a spoken line. Deliberately narrow: "the wind
# whispers" is a sound, "the narrator says" is dialogue in the wrong field.
_SPEECH_RE = re.compile(
    r"\b(?:says?|said|narrat(?:es|or)|voice-?over)\b", re.IGNORECASE
)


def _check_audio(prompt: VideoPrompt) -> list[Finding]:
    storyboard = prompt.storyboard
    found: list[Finding] = []

    if not prompt.audio.strip() and not storyboard.cues:
        found.append(
            Finding(
                "audio.missing",
                Severity.INFO,
                "No audio direction. H3 generates stereo sound in the same pass "
                "as picture, so an unspecified track is a wasted one.",
            )
        )

    for cue in storyboard.cues:
        end = cue.start if cue.is_instant else cue.end
        if cue.start < 0 or end > storyboard.duration:
            found.append(
                Finding(
                    "audio.cue_out_of_range",
                    Severity.WARNING,
                    f"Audio cue falls outside the {storyboard.duration}s clip.",
                    cue.render(),
                )
            )
        # A cue exists to pin a sound to a moment. One that runs the whole clip
        # pins it to nothing: it is the bed, restated as a second sentence.
        if (
            not cue.is_instant
            and cue.start <= _CUE_EDGE_S
            and cue.end >= storyboard.duration - _CUE_EDGE_S
        ):
            found.append(
                Finding(
                    "audio.cue_covers_clip",
                    Severity.INFO,
                    f"This cue spans the whole {storyboard.duration}s clip, so it "
                    f"pins the sound to no particular moment. That is what the "
                    f"soundscape bed is for — fold it into Audio.",
                    cue.render(),
                )
            )
        if _SPEECH_RE.search(cue.description):
            found.append(
                Finding(
                    "audio.cue_is_dialogue",
                    Severity.WARNING,
                    f"This cue reads as speech. Narration in the soundscape is "
                    f"described rather than spoken, and has been a source of "
                    f"burned-in captions — put it on the shot as a line with "
                    f"voiceover instead, where it composes inside <d>.",
                    cue.render(),
                )
            )

    return found


def _check_blocks(prompt: VideoPrompt) -> list[Finding]:
    found: list[Finding] = []

    if not prompt.style.strip():
        found.append(
            Finding(
                "style.missing",
                Severity.WARNING,
                "No style contract. The style opens [Shot 1] and fixes medium, "
                "texture, palette and era — without it the look drifts between shots.",
            )
        )

    if prompt.music.strip() and prompt.music.strip().upper() != "N/A":
        leaked = [word for word in _DIEGETIC_WORDS if word in prompt.music.lower()]
        if leaked:
            found.append(
                Finding(
                    "music.diegetic",
                    Severity.INFO,
                    "non_diegetic_music reads like a diegetic source "
                    f"({leaked[0]}). The score is audience-only; sound the "
                    "characters can hear belongs in the shot or the soundscape.",
                )
            )

    return found


# Words that usually mean an in-world sound source rather than an audience score.
_DIEGETIC_WORDS = ("radio", "jukebox", "car stereo", "television", "phone", "speaker")


def _check_adherence(prompt: VideoPrompt) -> list[Finding]:
    """How H3 is known to misread a prompt, as opposed to how the format works.

    None of these break a request. They are the cases where a prompt composes
    perfectly and the model still does something else: a cut it merges away,
    quality words it has no way to see, a camera plan with two jobs in it.
    """
    return [
        *_check_cuts(prompt),
        *_check_aesthetics(prompt),
        *_check_camera_load(prompt),
    ]


def _check_cuts(prompt: VideoPrompt) -> list[Finding]:
    """A cut has to introduce something the previous shot did not have.

    New subject, new place, new state, new viewpoint, new time. When the only
    thing that changes is how tightly the same thing is framed, H3 tends to
    merge the two shots — which reads afterwards as the model having skipped
    one. That case wants a camera move, not a cut.

    Kept deliberately blunt: it fires only when a shot names *nothing* the one
    before it did not, framing vocabulary excluded. A shot that adds even one
    new noun is left alone, because judging whether that noun is enough is not
    something a word list can do.

    Only real cuts are judged. A no-cut beat is the fix this rule recommends,
    so firing on one would tell an author who took the advice to take it again.
    """
    shots = prompt.storyboard.shots()
    found: list[Finding] = []

    def words(beats: list[Beat]) -> set[str]:
        """Everything the whole shot names, continuations included."""
        return _content_words(*[part for b in beats for part in (b.angle, b.action)])

    for position, beats in enumerate(shots[1:], start=1):
        subject = words(beats)
        # An empty shot is beat.no_action's problem, not this rule's.
        if not subject or subject - words(shots[position - 1]):
            continue
        lead = beats[0]
        found.append(
            Finding(
                "beat.weak_cut",
                Severity.WARNING,
                f"Shot {position + 1} names nothing that shot {position} did not. "
                f"A cut has to introduce new information — a subject, a place, a "
                f"state, a viewpoint or a jump in time. If only the framing "
                f"changes, H3 usually merges the two shots; clear its cut so it "
                f"becomes a camera move inside shot {position} instead.",
                format_span(lead.start, lead.end),
            )
        )

    return found


def _check_aesthetics(prompt: VideoPrompt) -> list[Finding]:
    """Quality tokens are a diffusion-image habit and cost adherence here.

    "4k", "masterpiece", "cinematic quality" describe the viewer's reaction,
    not anything a lens can be pointed at. H3 conditions on described
    behaviour, so the words are spent for nothing and crowd out detail that
    would have landed. Dialogue is exempt — a character is allowed to call
    something a masterpiece.
    """
    fields: list[tuple[str, str]] = [
        ("style", prompt.style),
        ("overview", prompt.overview),
        ("camera", prompt.camera),
        ("audio", prompt.audio),
        ("music", prompt.music),
        ("summary", prompt.ref_summary),
    ]
    numbers = prompt.storyboard.shot_numbers()
    for number, beat in zip(numbers, prompt.storyboard.ordered_beats()):
        fields.append(
            (f"Shot {number}", " ".join((beat.angle, beat.action, beat.camera)))
        )

    hits: list[tuple[str, str]] = []
    for where, text in fields:
        if not text.strip():
            continue
        hits += [
            (where, label)
            for label, pattern in _AESTHETIC_WORDS
            if pattern.search(text)
        ]

    if not hits:
        return []

    tokens = sorted({label for _, label in hits})
    return [
        Finding(
            "prompt.aesthetic",
            Severity.WARNING,
            f"Quality language ({', '.join(tokens)}) describes your reaction, not "
            f"anything the camera can see, and H3 conditions on described "
            f"behaviour. Spend the words on observable physics instead — "
            f"orientation, blocking, material, light. The look belongs in the "
            f"style contract, not sprinkled through the action.",
            hits[0][0],
        )
    ]


def _check_camera_load(prompt: VideoPrompt) -> list[Finding]:
    """One primary camera move per *beat*; two in one compete and neither lands.

    Per beat, not per shot: a shot may now hold several moves and that is the
    point of a no-cut beat. What was measured to fail is two moves stacked in
    one breath, with nothing to say which comes first. Giving the second its
    own timed beat is exactly the fix, so the rule looks no wider than the
    field it can see a collision in.
    """
    found: list[Finding] = []
    numbers = prompt.storyboard.shot_numbers()
    for number, beat in zip(numbers, prompt.storyboard.ordered_beats()):
        if beat.camera_path is not None:
            continue
        moves = sorted(
            {name for name, pattern in _CAMERA_MOVES if pattern.search(beat.camera)}
        )
        if len(moves) < 2:
            continue
        where = format_span(beat.start, beat.end)
        names = f"Shot {number}" if beat.cut else f"The move at {where} in shot {number}"
        found.append(
            Finding(
                "beat.camera_complex",
                Severity.INFO,
                f"{names} asks the camera to do {len(moves)} things at once "
                f"({', '.join(moves)}). One main action arc and one primary move "
                f"at a time is what the model holds onto; stacked moves usually "
                f"resolve into whichever one it read first. Give the second one "
                f"its own no-cut beat and the timecode will order them.",
                where,
            )
        )
    return found


def _content_words(*parts: str) -> set[str]:
    """The nouns and verbs a shot introduces, with the framing vocabulary gone.

    Framing words are stripped precisely because they are what a weak cut
    changes: leaving "close-up" in would make every re-framing look like new
    information.
    """
    words = re.findall(r"[a-z0-9]+", " ".join(parts).lower())
    return {
        word
        for word in words
        if len(word) > 2 and word not in _STOPWORDS and word not in _FRAMING_WORDS
    }


_STOPWORDS = frozenset(
    "and but the for from into onto with are was were being been this that these "
    "those his her their its him them they she has have had not very just really "
    "already while during over under across then than there here who whom which "
    "what when where how all any both each more most other some such own same too "
    "can will would could should may might must".split()
)

# How a shot is framed, not what is in it. See `_content_words`.
_FRAMING_WORDS = frozenset(
    "shot angle frame frames framing view views viewing viewpoint closeup close "
    "extreme wide medium tight loose full insert detail single two twoshot "
    "shoulder reverse low high overhead level eye waist chest profile front rear "
    "behind pov centred centered symmetrical composition foreground background "
    "left right side above below".split()
)

# Diffusion-image quality tokens, matched on word boundaries.
_AESTHETIC_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("4k", r"\b4k\b"),
        ("8k", r"\b8k\b"),
        ("ultra HD", r"\bultra[ -]?hd\b|\buhd\b"),
        ("masterpiece", r"\bmasterpiece\b"),
        ("best quality", r"\b(?:best|high|highest|professional) quality\b"),
        ("highly detailed", r"\b(?:hyper|highly|ultra)[ -]?detailed\b"),
        ("award-winning", r"\baward[ -]?winning\b"),
        ("cinematic quality", r"\b(?:cinematic|movie|film|photorealistic) quality\b"),
        ("stunning", r"\b(?:stunning|breathtaking|gorgeous)\b"),
        ("not AI-generated", r"\b(?:don[\u2019']?t|do not|not)\s+(?:look\s+)?ai\b"),
        ("realistic-looking", r"\brealistic[ -]?looking\b|\blooks realistic\b"),
    )
)

# The camera's primary jobs. Two in one shot is a plan with two authorities.
_CAMERA_MOVES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("push in", r"\bpush(?:es|ing)?\s+in\b|\bdoll(?:y|ies|ying)\s+in\b"),
        ("pull out", r"\bpull(?:s|ing)?\s+(?:out|back)\b|\bdoll(?:y|ies|ying)\s+(?:out|back)\b"),
        ("pan", r"\bpan(?:s|ning)?\b"),
        ("tilt", r"\btilt(?:s|ing)?\b"),
        ("truck", r"\btruck(?:s|ing)?\b|\bslides?\s+(?:left|right)\b"),
        ("crane", r"\bcranes?\b|\bjib\b|\bpedestals?\b"),
        ("orbit", r"\borbit(?:s|ing)?\b|\barc(?:s|ing)?\s+(?:around|left|right)\b"),
        ("track", r"\btrack(?:s|ing)?\b|\bfollow(?:s|ing)?\b"),
        ("zoom", r"\bzoom(?:s|ing)?\b"),
        ("roll", r"\brolls?\b|\brolling\b"),
        ("rack focus", r"\brack(?:s|ing)?\s+focus\b"),
    )
)


def _check_references(prompt: VideoPrompt) -> list[Finding]:
    references = prompt.references
    found: list[Finding] = []

    for kind in REFERENCE_KINDS:
        items = references.of(kind)
        limit = references.limit_for(kind)
        if len(items) > limit:
            found.append(
                Finding(
                    "references.too_many",
                    Severity.ERROR,
                    f"{len(items)} {kind} references attached; "
                    f"{h3_spec.NODE_REFERENCE_TO_VIDEO} accepts {limit}.",
                )
            )

    # A soundtrack belongs to the same-numbered reference video, so it cannot
    # outnumber them.
    soundtracks = len(references.of("video_audio"))
    videos = len(references.of("video"))
    if soundtracks > videos:
        found.append(
            Finding(
                "references.orphan_soundtrack",
                Severity.ERROR,
                f"{soundtracks} reference video soundtracks but only {videos} "
                f"reference videos. Each soundtrack pairs with the video of the "
                f"same number.",
            )
        )

    # A reference is "used" if some subject is defined from it, if it has a role
    # of its own (its `subject_definitions` line), or if its tag is woven into
    # the authored narrative. The retention_analysis line is generated for every
    # reference, so the composed doc is not the right haystack.
    narrative = _narrative(prompt)
    cited = prompt.subjects.cited_paths()
    for item in references.items:
        tag = references.tag(item)
        if item.path in cited:
            continue
        if not item.role.strip() and tag not in narrative:
            found.append(
                Finding(
                    "references.unused",
                    Severity.WARNING,
                    f"{tag} is attached but never referred to in the prompt. The "
                    f"local nodes expect references to be called out by tag.",
                    item.path or item.kind,
                )
            )

    return found


# Any `<Tag N>` label, and the `(S1)` speaker ids the format keeps out of
# retention_analysis.
_TAG_RE = re.compile(r"<(?:Subject|Picture|Video|Audio)\s+\d+>")
_SPEAKER_ID_RE = re.compile(r"\(S\d+\)")


def _narrative(prompt: VideoPrompt) -> str:
    """The authored narrative text, where a reference tag would be woven in."""
    parts = [prompt.overview, prompt.style, prompt.ref_summary, *prompt.screen_text]
    for beat in prompt.storyboard.beats:
        parts += [beat.angle, beat.action, beat.camera]
        parts += [line.text for line in beat.lines]
        parts += [line.describe for line in beat.lines]
    return " ".join(part for part in parts if part)


def _check_reference_doc(prompt: VideoPrompt) -> list[Finding]:
    """Full-reference (ref2v) fields against the official controlled vocabulary."""
    refs = prompt.references
    if not refs.items:
        return []

    found: list[Finding] = []

    # `summary` is the section that states the *task* — what the target video
    # does with the pictures. Left empty it still composes, to a bare line
    # naming the tags and nothing else, which is the one failure here that reads
    # as a finished document: subject_definitions and retention_analysis are
    # still full, so nothing looks wrong until the model starts inventing what
    # to do with three unexplained references.
    if not prompt.ref_summary.strip():
        # Named the way the document names them: a face sheet cited by a
        # subject has no tag of its own to talk about.
        tags = ", ".join(prompt.defined_tags())
        found.append(
            Finding(
                "ref.summary_missing",
                Severity.WARNING,
                f"No reference summary, so `summary` composes to nothing but "
                f"'The target video is generated using {tags}.' Say in one "
                f"paragraph what each tag is and what the shot does with it — "
                f"which identity to hold, which is a set rather than a framing, "
                f"and which tags are the same subject seen twice.",
            )
        )

    for task in prompt.ref_task:
        if task.strip() and task.strip() not in h3_spec.REF_TASK_TYPES:
            found.append(
                Finding(
                    "ref.task_invalid",
                    Severity.WARNING,
                    f"'{task}' is not an H3 task type. Use one of: "
                    f"{', '.join(h3_spec.REF_TASK_TYPES)}.",
                )
            )

    for item in refs.items:
        marker = item.retention.strip()
        if not marker:
            continue
        valid = h3_spec.AUDIO_RETENTION if item.is_audio else h3_spec.VISUAL_RETENTION
        if marker not in valid:
            found.append(
                Finding(
                    "ref.retention_invalid",
                    Severity.WARNING,
                    f"{refs.tag(item)} has retention marker '{marker}', which is "
                    f"not one of: {', '.join(valid)}.",
                    item.path or item.kind,
                )
            )

    for subject in prompt.subjects.items:
        marker = subject.retention.strip()
        if marker and marker not in h3_spec.VISUAL_RETENTION:
            found.append(
                Finding(
                    "ref.retention_invalid",
                    Severity.WARNING,
                    f"{prompt.subjects.tag(subject)} has retention marker "
                    f"'{marker}', which is not one of: "
                    f"{', '.join(h3_spec.VISUAL_RETENTION)}.",
                    subject.label or subject.role[:40],
                )
            )

    found += _check_summary_tags(prompt)
    found += _check_speaker_ids_in_retention(prompt)
    found += _check_subjects_used(prompt)
    found += _check_description_length(prompt)
    return found


def _check_summary_tags(prompt: VideoPrompt) -> list[Finding]:
    """The summary may only use labels `subject_definitions` already defined.

    "Do not introduce new reference labels in this section" (guide 3). A tag
    invented here points at nothing, and the model has to guess which of the
    real labels was meant.
    """
    known = set(prompt.defined_tags())
    # A cited file keeps its number even without a line of its own, so naming it
    # in the summary is a numbering mistake, not an invention. Both are wrong,
    # but only the truly undefined ones are worth a finding.
    known |= {prompt.references.tag(item) for item in prompt.references.items}
    unknown = [tag for tag in _TAG_RE.findall(prompt.ref_summary) if tag not in known]
    if not unknown:
        return []
    return [
        Finding(
            "ref.summary_undefined_tag",
            Severity.WARNING,
            f"The summary names {', '.join(sorted(set(unknown)))}, which nothing "
            f"defines. Use only the labels from subject_definitions: "
            f"{', '.join(sorted(known)) or 'none yet'}.",
        )
    ]


def _check_speaker_ids_in_retention(prompt: VideoPrompt) -> list[Finding]:
    """`retention_analysis` never carries a speaker id (guide 5.4, last para).

    The section is about how much of a reference survives, not about who talks;
    an `(S1)` written into a role composes straight into that section.
    """
    found: list[Finding] = []
    rows: list[tuple[str, str]] = [
        (prompt.subjects.tag(subject), subject.role) for subject in prompt.subjects.items
    ]
    rows += [
        (prompt.references.tag(item), item.role)
        for item in prompt.references.items
        # An audio definition may name `(Sx)`; only its retention line may not,
        # and that line reuses the same role text, so the audio rows are left
        # alone rather than made to choose.
        if not item.is_audio
    ]
    for tag, role in rows:
        if _SPEAKER_ID_RE.search(role or ""):
            found.append(
                Finding(
                    "ref.retention_speaker_id",
                    Severity.WARNING,
                    f"{tag}'s role carries a speaker id, which composes into "
                    f"retention_analysis — the format does not allow one there. "
                    f"Name the speaker in the description instead.",
                    tag,
                )
            )
    return found


def _check_subjects_used(prompt: VideoPrompt) -> list[Finding]:
    """A subject the description never names is a definition doing no work.

    The guide (5.2, 5.3) asks for every label to be inserted at its first
    appearance and wherever its role applies. A speaking cast member is written
    in for you; anything else has to be woven in by hand.
    """
    if not prompt.subjects.items:
        return []
    narrative = _narrative(prompt)
    speaking = {
        key
        for beat in prompt.storyboard.ordered_beats()
        for line in beat.lines
        if line.text.strip()
        for key in line.speakers
    }
    found: list[Finding] = []
    for subject in prompt.subjects.items:
        tag = prompt.subjects.tag(subject)
        if tag in narrative or (subject.member and subject.member in speaking):
            continue
        found.append(
            Finding(
                "subjects.unused",
                Severity.WARNING,
                f"{tag} is defined but never named in the description. Write the "
                f"tag into the shot where it first appears, so the model knows "
                f"which part of the frame the reference is for.",
                subject.label or (subject.role[:40] if subject.role else tag),
            )
        )
    return found


def _check_description_length(prompt: VideoPrompt) -> list[Finding]:
    """`detailed_description` runs 350-500 English words for a generation task.

    Informational, not a rule: the guide exempts dialogue-dense content and
    video edits, and both are common here. It is there because a full-reference
    document that composes to eighty words is usually a storyboard that was
    never filled in, not a deliberately terse one.
    """
    if "video editing" in [task.strip() for task in prompt.ref_task]:
        return []
    words = len(_narrative(prompt).split())
    if 350 <= words <= 500 or not prompt.storyboard.ordered_beats():
        return []
    direction = "short of" if words < 350 else "past"
    return [
        Finding(
            "ref.description_length",
            Severity.INFO,
            f"The authored description is about {words} words, {direction} the "
            f"guide's 350-500 for a generation task. Detail per shot is what "
            f"full-reference mode spends its budget on — composition, lighting, "
            f"state changes, and where each reference actually takes effect.",
        )
    ]


def _check_media(prompt: VideoPrompt, authored: set[str] | None = None) -> list[Finding]:
    """Keyframes and references live on different H3 nodes, so a shot picks one.

    Keyframes (`first_frame`/`last_frame`) drive `MiniMaxH3ImageToVideo` (i2v);
    references drive `MiniMaxH3ReferenceToVideo` (ref2v). Both set means two
    contradictory modes, which the renderer can only resolve by dropping one.

    `authored` is the paths the shot itself attached. Anything else came from a
    cast member, and the message has to say so: the Media tab shows no
    references at all in that case, so "remove the references" would send the
    author looking somewhere they are not.
    """
    found: list[Finding] = []
    has_keyframe = bool(prompt.first_frame.strip() or prompt.last_frame.strip())

    if has_keyframe and prompt.references.items:
        from_cast = [
            item.path
            for item in prompt.references.items
            if authored is not None and item.path not in authored
        ]
        blame = (
            (
                f" {len(from_cast)} of them "
                + ("is an identity file" if len(from_cast) == 1 else "are identity files")
                + " carried by a cast member on screen here, not something this "
                "shot attached — take the member off the shot, or drop the "
                "keyframe."
            )
            if from_cast
            else ""
        )
        found.append(
            Finding(
                "mode.conflict",
                Severity.ERROR,
                "This shot has both a keyframe and omni-references. H3 uses "
                "different nodes for image-to-video and reference-to-video — "
                "keep one or the other." + blame,
            )
        )

    if prompt.last_frame.strip() and not prompt.first_frame.strip():
        found.append(
            Finding(
                "keyframe.orphan",
                Severity.INFO,
                "A last-frame keyframe is set with no first frame. The end anchor "
                "works best paired with a start frame.",
            )
        )

    return found


def _check_length(prompt: VideoPrompt) -> list[Finding]:
    length = len(compose(prompt))

    if length > h3_spec.MAX_PROMPT_CHARS:
        return [
            Finding(
                "prompt.too_long",
                Severity.ERROR,
                f"Prompt is {length} characters; the API limit is "
                f"{h3_spec.MAX_PROMPT_CHARS}.",
            )
        ]

    if length > h3_spec.RECOMMENDED_PROMPT_CHARS:
        return [
            Finding(
                "prompt.long",
                Severity.WARNING,
                f"Prompt is {length} characters. MiniMax recommends staying near "
                f"{h3_spec.RECOMMENDED_PROMPT_CHARS} — past that, later blocks "
                f"lose weight.",
            )
        ]

    return []
