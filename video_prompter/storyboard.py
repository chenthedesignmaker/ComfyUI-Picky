"""One generation's internal ruler: beats and audio cues against one budget.

A *beat* is a moment inside a single H3 generation — the `[0s-4s] Shot 1: ...`
lines of a storyboard. It is not a shot in the edit; that lives in `sequence`,
where a `Shot` is one whole generation with its own prompt and takes.

The vocabulary is split this way because the words collide badly otherwise: the
text H3 reads calls these lines "Shot N", and so does `Beat.render`, but inside
this tool a shot is a unit of the timeline and a beat is a unit of a prompt.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Iterable

from . import h3_spec

# Accepts the bracket forms seen in MiniMax's official prompts:
#   [0s-1.5s]  [0s - 1.5s]  [0-2s]  [0 to 2 seconds]  [2s to 4s]
_SPAN_RE = re.compile(
    r"""\[\s*
        (?P<start>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*
        (?:-|–|—|to)\s*
        (?P<end>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*
    \]""",
    re.VERBOSE | re.IGNORECASE,
)

# "[0s-1.5s] Shot 1: high side angle: the protagonist sprinting..."
# The "Shot N:" prefix is optional; angle is whatever precedes the next colon.
_BEAT_LINE_RE = re.compile(
    r"""^\s*(?P<span>\[[^\]]+\])\s*
        (?:Shot\s*(?P<index>\d+)\s*:\s*)?
        (?P<body>.*)$""",
    re.VERBOSE | re.IGNORECASE,
)


def format_seconds(value: float) -> str:
    """Render a time the way the legacy bracket storyboard did: `0s`, `1.5s`."""
    rounded = round(value, 3)
    if rounded == int(rounded):
        return f"{int(rounded)}s"
    return f"{rounded:g}s"


def format_timecode(value: float) -> str:
    """Render a cut time as the official guide's `MM:SS.mmm`, e.g. `00:03.500`.

    H3's rewrite format labels every shot after the first with `At MM:SS.mmm`;
    milliseconds are always three digits and minutes at least two.
    """
    total_ms = round(max(0.0, value) * 1000)
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_span(start: float, end: float) -> str:
    return f"[{format_seconds(start)}-{format_seconds(end)}]"


def parse_span(text: str) -> tuple[float, float] | None:
    """Pull `(start, end)` out of a bracketed timecode, or None."""
    match = _SPAN_RE.search(text)
    if match is None:
        return None
    return float(match["start"]), float(match["end"])


@dataclass
class Line:
    """One spoken (or sung) line inside a shot.

    Speaker identity is a whole-prompt concept: `speakers` holds one or more
    stable author keys, which the composer maps to `(S1)`, `(S2)`, ... in order
    of first vocal appearance (a group line becomes `(S1,S2)`). Everything that
    is not the verbatim words — the identifying phrase (`describe`), the ID, the
    verb, the delivery — is rendered outside the `<d>` tag; only the language and
    the exact words go inside it.

    `continuity` and `cutoff` carry the format's seam tags. A line split across
    a cut is written as two `Line`s on adjacent beats: the outgoing half sets
    `continuity="into"`, the incoming half `continuity="from"`, and each emits
    `<scenetrans>` at the seam plus the phrase that says the audio keeps
    running. `cutoff` marks speech the end of the video truncates.

    `placement` says where in the shot the line is spoken. The format gives a
    line no timecode of its own, so the only thing that says *when* it lands is
    where its sentence sits among the shot's other sentences: by default last,
    after the action and the camera move. `placement="open"` puts it directly
    after the framing clause instead, so the speech starts and the action
    follows it.
    """

    speakers: list[str] = field(default_factory=list)
    text: str = ""
    language: str = "English"
    verb: str = "says"
    delivery: str = ""
    describe: str = ""
    voiceover: bool = False
    # Possessive used in the voiceover lips-closed note ("his", "her", "the
    # baker's"). Empty falls back to "their".
    possessive: str = ""
    # "" | "into" (continues into the next shot) | "from" (carries over from
    # the previous one).
    continuity: str = ""
    cutoff: bool = False
    # "" (after the action, the format's default) | "open" (right after the
    # shot's framing clause, ahead of the action).
    placement: str = ""


@dataclass
class Beat:
    """One moment inside a generation. `angle` is framing, `action` is what happens.

    `camera` is the shot's camera motion woven into its own sentence — the
    official format wants motion type + amplitude + speed inside the shot, never
    stacked as a trailing label. `transition` is the cut phrase that introduces
    this shot ("the camera cuts to"); it is ignored for the opening shot, which
    the guide leaves un-timed and un-transitioned. `lines` are the dialogue or
    lyrics spoken during the shot.

    `cut` says whether this beat *opens* a `[Shot N]` of its own. Cleared, the
    beat is a timed moment inside the shot before it — the camera moves, or the
    action turns, without the film cutting. Several beats then compose into one
    `[Shot N]` block, each later one leading with `At MM:SS.mmm, the camera ...`.
    The first beat always opens a shot whatever this says, since there is no
    earlier one to continue.
    """

    start: float
    end: float
    angle: str = ""
    action: str = ""
    camera: str = ""
    transition: str = ""
    cut: bool = True
    lines: list[Line] = field(default_factory=list)

    # Optional normalized trajectory, separate from handwritten camera prose.
    camera_path: dict | None = None

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def overlaps(self, other: "Beat") -> bool:
        """True when two beats claim the same instant. Touching is not overlap."""
        return self.start < other.end and other.start < self.end

    def render(self, index: int, *, label_beats: bool = True) -> str:
        """The storyboard line. Still emits `Shot N:` — that is H3's own format."""
        head = format_span(self.start, self.end)
        if label_beats:
            head = f"{head} Shot {index}:"
        parts = [part for part in (self.angle.strip(), self.action.strip()) if part]
        return f"{head} {': '.join(parts)}" if parts else head

    @classmethod
    def parse(cls, line: str) -> "Beat | None":
        """Read back a rendered storyboard line. Returns None if it isn't one."""
        match = _BEAT_LINE_RE.match(line)
        if match is None:
            return None
        span = parse_span(match["span"])
        if span is None:
            return None

        body = match["body"].strip()
        angle, _, action = body.partition(":")
        if not action:
            # No second colon, so the whole body is the action.
            angle, action = "", body
        return cls(span[0], span[1], angle.strip(), action.strip())


@dataclass
class AudioCue:
    """A sound event pinned to the ruler.

    `end` is None for an instantaneous hit ("an accent hit at 4s"); set it for
    a sustained bed ("[6s-15s] the bass groove carries under").
    """

    start: float
    end: float | None = None
    description: str = ""

    @property
    def is_instant(self) -> bool:
        return self.end is None

    def render(self) -> str:
        when = (
            format_seconds(self.start)
            if self.is_instant
            else f"{format_seconds(self.start)}-{format_seconds(self.end)}"
        )
        text = self.description.strip()
        return f"at {when}: {text}" if text else f"at {when}"

    def sentence(self) -> str:
        """The cue as a sentence, for `overall_soundscape`.

        The field is one continuous paragraph of English sentences, so the
        compact `at 3s: ...` label `render` produces would not sit in it.

        Times are written the way the format writes every other time — the
        shots' `MM:SS.mmm` — rather than in a second notation of our own. A span
        reads as plain English `From ... to ...` for the same reason: `6s-10s`
        is a compact form nothing in the guide uses, and a prompt that speaks
        one dialect throughout is the only version worth testing.
        """
        text = self.description.strip()
        if not text:
            return ""
        if self.is_instant:
            return f"At {format_timecode(self.start)}, {text}"
        return (
            f"From {format_timecode(self.start)} to {format_timecode(self.end)}, "
            f"{text}"
        )


@dataclass
class Storyboard:
    """Beats plus audio cues against a fixed integer-second budget."""

    duration: int
    beats: list[Beat] = field(default_factory=list)
    cues: list[AudioCue] = field(default_factory=list)

    @property
    def covered(self) -> float:
        """Total wall time claimed by beats, counting overlaps only once."""
        total = 0.0
        for start, end in self._merged_spans():
            total += end - start
        return round(total, 3)

    @property
    def gaps(self) -> list[tuple[float, float]]:
        """Stretches of the budget no beat covers."""
        result: list[tuple[float, float]] = []
        cursor = 0.0
        for start, end in self._merged_spans():
            if start > cursor:
                result.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < self.duration:
            result.append((cursor, float(self.duration)))
        return result

    @property
    def overlaps(self) -> list[tuple[int, int]]:
        """Index pairs (1-based, as rendered) whose beats collide."""
        ordered = self.ordered_beats()
        found: list[tuple[int, int]] = []
        for i, beat in enumerate(ordered):
            for j in range(i + 1, len(ordered)):
                if beat.overlaps(ordered[j]):
                    found.append((i + 1, j + 1))
                else:
                    break
        return found

    def ordered_beats(self) -> list[Beat]:
        return sorted(self.beats, key=lambda b: (b.start, b.end))

    def shots(self) -> list[list[Beat]]:
        """Beats grouped into the `[Shot N]` blocks they compose into.

        Each group opens with a beat that cuts and carries any no-cut beats
        that follow it. One beat per group is the old behaviour and still the
        common case; a group of three is one continuous take with two timed
        moves inside it.
        """
        groups: list[list[Beat]] = []
        for beat in self.ordered_beats():
            if groups and not beat.cut:
                groups[-1].append(beat)
            else:
                groups.append([beat])
        return groups

    def shot_numbers(self) -> list[int]:
        """The 1-based `[Shot N]` each ordered beat lands in, parallel to
        `ordered_beats()`.

        This is what everything outside the composer should number by: a beat's
        position is no longer its shot, and `(appears in [Shot 2])` has to mean
        the shot the reader will count.
        """
        numbers: list[int] = []
        current = 0
        for beat in self.ordered_beats():
            if not numbers or beat.cut:
                current += 1
            numbers.append(current)
        return numbers

    def speaker_ids(self) -> dict[str, str]:
        """Each speaker key mapped to `S1`, `S2`, ... by first vocal appearance.

        Lives here rather than in the composer because it is a fact about the
        storyboard, and two other places need it: `cast` has to know what an
        `@mention` of a speaker composes to, and the linter has to know whether
        a mentioned member has any label at all.
        """
        order: list[str] = []
        for beat in self.ordered_beats():
            for line in beat.lines:
                for key in line.speakers:
                    if key and key not in order:
                        order.append(key)
        return {key: f"S{position}" for position, key in enumerate(order, start=1)}

    def is_contiguous(self) -> bool:
        """True when beats tile the budget end to end with no gap or overlap."""
        return not self.gaps and not self.overlaps

    def refit(self, duration: int | None = None) -> "Storyboard":
        """Rescale beats and cues proportionally onto a new budget.

        Used when the duration slider moves: beat *proportions* are authorial,
        absolute times are not, so we stretch rather than truncate.
        """
        target = self.duration if duration is None else duration
        span = max((beat.end for beat in self.beats), default=0.0)
        if span <= 0:
            return replace(self, duration=target, beats=[], cues=list(self.cues))

        scale = target / span

        def at(value: float) -> float:
            return round(value * scale, 3)

        return Storyboard(
            duration=target,
            beats=[replace(b, start=at(b.start), end=at(b.end)) for b in self.beats],
            cues=[
                replace(c, start=at(c.start), end=None if c.end is None else at(c.end))
                for c in self.cues
            ],
        )

    def make_contiguous(self, snap: float | None = 0.5) -> "Storyboard":
        """Close gaps and overlaps by butting each beat against the next.

        Preserves beat order and relative weight, then pins the last beat to
        the budget. This is the fix for the overlapping-timecode problem.

        Boundaries land on a `snap`-second grid so the storyboard reads like
        one a person wrote; pass None to keep exact proportions.
        """
        ordered = self.ordered_beats()
        if not ordered:
            return replace(self, beats=[])

        weights = [max(beat.duration, _EPSILON) for beat in ordered]
        total = sum(weights)

        cuts = [0.0]
        for weight in weights[:-1]:
            cuts.append(cuts[-1] + (weight / total) * self.duration)
        cuts.append(float(self.duration))

        if snap:
            cuts = _snap_cuts(cuts, snap, self.duration)

        result = [
            replace(beat, start=round(start, 3), end=round(end, 3))
            for beat, start, end in zip(ordered, cuts, cuts[1:])
        ]
        return replace(self, beats=result)

    def extract(self, indices: Iterable[int]) -> "Storyboard":
        """The named beats as a storyboard of their own, rebased to zero.

        This is how a scene written as one generation becomes two: pick the
        shots, and what comes back is a ruler that starts where they start.

        Each kept beat keeps its own length and they run back to back, so a
        contiguous pick is the same stretch of the clip slid down to zero, and
        a scattered one simply closes the gaps between what was kept — there is
        nothing to play in a gap whose shot was left behind. The budget is that
        total rounded to the whole second H3 counts in, and the spans are then
        refitted onto it so they still tile it exactly.

        A cue travels with the beat its start falls inside. One that starts in
        a beat nobody kept has nothing left to hang off and is dropped, rather
        than being pinned to a moment that no longer exists.

        Indices are positions in `beats`, the same 0-based numbering a `beat:N`
        rewrite scope uses. Order does not matter — the result reads in time
        order — and anything out of range is ignored.
        """
        kept = sorted({int(i) for i in indices if 0 <= int(i) < len(self.beats)})
        if not kept:
            return replace(self, beats=[], cues=[])

        beats: list[Beat] = []
        # Original index -> how far its content moved, so a cue can follow it.
        shift: dict[int, float] = {}
        cursor = 0.0
        for index in kept:
            beat = self.beats[index]
            length = max(0.0, beat.duration)
            shift[index] = round(cursor - beat.start, 3)
            beats.append(
                replace(
                    beat,
                    start=round(cursor, 3),
                    end=round(cursor + length, 3),
                    # Whatever it continued has been left behind: the first beat
                    # of a new generation is its Shot 1 and has to open one.
                    cut=True if not beats else beat.cut,
                    lines=deepcopy(beat.lines),
                )
            )
            cursor = round(cursor + length, 3)

        cues: list[AudioCue] = []
        for cue in self.cues:
            home = next(
                (i for i in kept if self.beats[i].start <= cue.start < self.beats[i].end),
                None,
            )
            if home is None:
                continue
            start = round(cue.start + shift[home], 3)
            end = None if cue.end is None else round(min(cue.end + shift[home], cursor), 3)
            # A bed whose tail was cut away is a hit, not a span of no length.
            if end is not None and end <= start:
                end = None
            cues.append(replace(cue, start=start, end=end))

        budget = min(
            max(round(cursor), h3_spec.MIN_DURATION_S), h3_spec.MAX_DURATION_S
        )
        board = Storyboard(duration=budget, beats=beats, cues=cues)
        # Rounding to a whole second moved the tail off the budget by up to half
        # a second. Refitting puts it back without disturbing the proportions —
        # and with nothing to scale, there is nothing to refit.
        return board.refit() if cursor > 0 else board

    def _merged_spans(self) -> list[tuple[float, float]]:
        """Beat spans with overlaps collapsed, in order."""
        merged: list[tuple[float, float]] = []
        for beat in self.ordered_beats():
            if merged and beat.start <= merged[-1][1]:
                last_start, last_end = merged[-1]
                merged[-1] = (last_start, max(last_end, beat.end))
            else:
                merged.append((beat.start, beat.end))
        return merged


_EPSILON = 1e-6


def _snap_cuts(cuts: list[float], snap: float, duration: float) -> list[float]:
    """Round interior cut points onto a grid, keeping them strictly increasing.

    Falls back to the exact cuts when the grid is too coarse to give every beat
    a slot — better a fractional timecode than a zero-length beat.
    """
    beat_count = len(cuts) - 1
    if beat_count * snap > duration:
        return cuts

    snapped = [0.0]
    for index, value in enumerate(cuts[1:-1], start=1):
        remaining = beat_count - index
        # Leave room for every later beat to keep at least one grid step.
        ceiling = duration - remaining * snap
        candidate = round(value / snap) * snap
        candidate = min(max(candidate, snapped[-1] + snap), ceiling)
        snapped.append(round(candidate, 3))

    snapped.append(float(duration))
    return snapped
