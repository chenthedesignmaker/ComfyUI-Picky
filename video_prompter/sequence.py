"""The edit: an ordered run of clips, each a window into one source's lineage.

Two clocks meet here, and keeping them apart is most of what this module does.

**Lineage time** is what H3 made: a source's takes laid end to end. It is not
free — the frame grid snaps to 17k+5 and the trained floor is 124 frames, so the
shortest thing that can be generated is a hair over five seconds. A two-second
cut is not a short generation, it is a trimmed one. A clip's in and out points
are seconds into this clock, *not* into any one file, which is what lets a clip
run across the join between two generations without the edit knowing.

**Timeline time** is what the edit uses — `source_in` to `source_out`, divided
by `speed`. Trimming and remapping never touch a prompt, so `h3_spec` and the
linter stay where they are and the editing layer above stays metadata until
export.

Clip order *is* the edit. There are no explicit start times, so gaps and
overlaps cannot be represented, let alone go wrong; `starts_at` derives
positions by running total. That makes this a cuts-only assembly, which is what
a pile of separate generations honestly is.

Picture and sound are one clip, not two, and that is deliberate. A clip already
addresses lineage seconds rather than a file, so a rejected re-roll of the same
shot is the *same clock* — borrowing its audio needs a take id, not a second
position to keep aligned. :class:`Sound` is where that is said. Sound that
belongs to nothing on screen — a music bed, room tone under three cuts — has no
picture to hang off and is not this; that is a lane of its own, and it does not
exist yet.

`Shot` used to live here, doing three unrelated jobs at once. It is gone: what
to generate is a :class:`~.sources.Segment`, which take won is a
:class:`~.sources.Source`'s `path`, and how the edit uses it is a `Clip`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

# Speeds outside this are a typo, not an intent.
MIN_SPEED = 0.1
MAX_SPEED = 10.0

# A borrowed track can be lifted or dropped, but not by an amount that is
# clearly a slip of the keyboard. Boosting is capped low because the monitor
# plays through an HTML element, which cannot amplify past unity — a +12 dB
# setting would sound one way in the preview and another in the export.
MIN_GAIN_DB = -60.0
MAX_GAIN_DB = 6.0


def new_clip_id() -> str:
    return f"clip_{uuid.uuid4().hex[:8]}"


@dataclass
class Sound:
    """What a clip is *heard* as, which is not always what it is seen as.

    A generation carries its own sound, so the honest default — everything here
    left alone — means "play my own audio", and that is what every clip written
    before this existed loads as.

    The interesting case is a re-roll whose picture you rejected and whose sound
    you did not. Because a clip addresses **lineage seconds** rather than a file,
    a rejected take of the same shot is the same clock: naming it here reads the
    audio from that take over exactly the window the picture already plays. Trim
    the clip and the sound follows, with no second position to keep in step. That
    is the whole reason this is a field on a clip and not a clip of its own.

    `source_id` empty means the clip's own source, which is the usual case —
    both takes are re-rolls of one shot. Naming another source is allowed and
    resolves the same way, but nothing keeps the two lineages the same length,
    so a short one simply runs out and the tail is silent.
    """

    # Contribute nothing to the mix. Not the same as a gain of -60: a muted clip
    # is a decision, a quiet one is a level.
    muted: bool = False
    gain_db: float = 0.0
    # Where the sound comes from. Both empty means "my own".
    source_id: str = ""
    path_override: list[str] | None = None
    # Slides the read window inside the borrowed lineage. Positive starts later.
    offset: float = 0.0

    @property
    def borrowed(self) -> bool:
        """True when the sound comes from somewhere other than the picture."""
        return bool(self.source_id or self.path_override)

    def copy(self) -> "Sound":
        """A separate one, for when a clip is split or duplicated.

        `dataclasses.replace` copies the *reference*, so without this the two
        halves of a split would share a path list and muting one would mute
        both.
        """
        return Sound(
            muted=self.muted,
            gain_db=self.gain_db,
            source_id=self.source_id,
            path_override=None if self.path_override is None else list(self.path_override),
            offset=self.offset,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "muted": self.muted,
            "gain_db": self.gain_db,
            "source_id": self.source_id,
            "path_override": None if self.path_override is None else list(self.path_override),
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Sound":
        data = data or {}
        override = data.get("path_override")
        return cls(
            muted=bool(data.get("muted", False)),
            gain_db=_clamp_gain(data.get("gain_db")),
            source_id=str(data.get("source_id") or ""),
            path_override=[str(x) for x in override] if isinstance(override, list) else None,
            offset=_float_or(data.get("offset"), 0.0),
        )


def new_mute_id() -> str:
    return f"mute_{uuid.uuid4().hex[:8]}"


def new_audio_id() -> str:
    return f"aud_{uuid.uuid4().hex[:8]}"


# A hard splice on a waveform clicks, and that click is indistinguishable from a
# render artifact when you go hunting for one. Short enough to be inaudible as a
# fade, long enough to kill the edge.
DEFAULT_FADE = 0.04
MAX_FADE = 5.0


@dataclass
class Mute:
    """A stretch of one clip that is heard as silence.

    Stored in **lineage seconds** — the clock `source_in` and `source_out`
    already use — and that is the whole design. A mute is a decision about a
    moment of the *footage*: the door slam, the line being pulled early.
    Anchoring it to the ruler instead would leave it behind the moment anything
    upstream was trimmed, which is exactly when nobody is looking at it.

    Everything wanted follows from that clock for free. Trimming the clip's head
    slides the mute across the screen, because the picture slid. Moving the clip
    carries it along. Changing speed rescales it, since lineage-to-timeline is
    always a divide by `speed`. Trimming past a mute clips it off and trimming
    back brings it whole again — the record keeps its full extent and only the
    reading of it narrows, so nothing here is destructive.

    A band dragged across a cut is two of these, one per clip. There is no single
    clock spanning two clips, so there is no honest way to store it as one.

    It silences whatever the clip is *heard* as, borrowed sound included: the
    window is over the clip, not over a particular file.
    """

    id: str = field(default_factory=new_mute_id)
    clip_id: str = ""
    start: float = 0.0
    end: float = 0.0

    def __post_init__(self) -> None:
        # A band dragged right-to-left is the same band.
        if self.end < self.start:
            self.start, self.end = self.end, self.start

    @property
    def seconds(self) -> float:
        return round(max(0.0, self.end - self.start), 3)

    @property
    def empty(self) -> bool:
        return self.seconds <= 0

    def clamped(self, low: float, high: float) -> "Mute | None":
        """This mute narrowed to a range of lineage seconds, or None if outside.

        The clip's trim is the range that matters, and a mute may sit wholly or
        partly beyond it. Callers narrow rather than delete, which is what makes
        over-trimming reversible.
        """
        start = max(self.start, low)
        end = min(self.end, high)
        if end - start <= 0:
            return None
        return replace(self, start=round(start, 3), end=round(end, 3))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "clip_id": self.clip_id,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mute":
        return cls(
            id=str(data.get("id") or new_mute_id()),
            clip_id=str(data.get("clip_id") or ""),
            start=_float_or(data.get("start"), 0.0),
            end=_float_or(data.get("end"), 0.0),
        )


@dataclass
class Lifted:
    """Lane sound read out of a rendered lineage, in that lineage's own seconds.

    The same address a clip uses, for the same reason: a range of lineage time
    can cross a generation join, so this resolves to however many files cover it
    rather than to one.
    """

    source_id: str = ""
    path_override: list[str] | None = None
    source_in: float = 0.0
    source_out: float = 0.0

    def width(self, available: float | None = None) -> float:
        """How much sound there is, once the lineage's real length is known."""
        end = self.source_out if available is None else min(self.source_out, available)
        return round(max(0.0, end - max(0.0, self.source_in)), 3)

    def copy(self) -> "Lifted":
        return replace(
            self,
            path_override=None if self.path_override is None else list(self.path_override),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "lifted",
            "source_id": self.source_id,
            "path_override": None if self.path_override is None else list(self.path_override),
            "source_in": self.source_in,
            "source_out": self.source_out,
        }


@dataclass
class Imported:
    """Lane sound from a file brought in from outside: music, room tone, a hit.

    `file` is a name inside the project's own audio folder, not wherever the file
    was dragged from — a project that stops playing because something moved on
    disk is not a project. `seconds` is the probed length of the whole file,
    stored so the lane can be drawn before anything is decoded.
    """

    file: str = ""
    seconds: float = 0.0
    file_in: float = 0.0
    file_out: float | None = None

    def out_point(self) -> float:
        if self.file_out is None:
            return self.seconds
        return min(self.file_out, self.seconds) if self.seconds else self.file_out

    def width(self, available: float | None = None) -> float:
        return round(max(0.0, self.out_point() - max(0.0, self.file_in)), 3)

    def copy(self) -> "Imported":
        return replace(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "file",
            "file": self.file,
            "seconds": self.seconds,
            "file_in": self.file_in,
            "file_out": self.file_out,
        }


Origin = Lifted | Imported


def origin_from_dict(data: dict[str, Any] | None) -> Origin:
    """A lane item's provenance, read back by its own `kind`.

    An unknown kind falls back to an empty :class:`Lifted` rather than raising: a
    project written by a newer build should still open, minus the part this one
    cannot play.
    """
    data = data or {}
    if data.get("kind") == "file":
        out = data.get("file_out")
        return Imported(
            file=str(data.get("file") or ""),
            seconds=_float_or(data.get("seconds"), 0.0),
            file_in=_float_or(data.get("file_in"), 0.0),
            file_out=None if out is None else _float_or(out, 0.0),
        )
    override = data.get("path_override")
    return Lifted(
        source_id=str(data.get("source_id") or ""),
        path_override=[str(x) for x in override] if isinstance(override, list) else None,
        source_in=_float_or(data.get("source_in"), 0.0),
        source_out=_float_or(data.get("source_out"), 0.0),
    )


@dataclass
class AudioClip:
    """Sound on a lane of its own, placed at a time it chose.

    `start` is the first absolute position in this project. Picture clips have
    none — order *is* the edit, and `starts_at` derives the rest — but a lane item
    exists precisely to sit somewhere its picture does not, so it has to say
    where. That is the break from the cuts-only rule, and it is deliberate.

    Placement and provenance are separate on purpose. Everything about *where*
    and *how loud* is here; everything about *what* is in `origin`, which is a
    :class:`Lifted` or an :class:`Imported` rather than a flag, because the two do
    not resolve alike — one can span a generation join, the other is always a
    single file.

    The fades default to something rather than nothing. A lifted item is a splice
    out of continuous sound, and a splice with square edges clicks.
    """

    id: str = field(default_factory=new_audio_id)
    origin: Origin = field(default_factory=Lifted)
    start: float = 0.0
    gain_db: float = 0.0
    fade_in: float = DEFAULT_FADE
    fade_out: float = DEFAULT_FADE
    enabled: bool = True
    label: str = ""

    @property
    def is_lifted(self) -> bool:
        return isinstance(self.origin, Lifted)

    def timeline_seconds(self, available: float | None = None) -> float:
        """How long this sits on the ruler. Lane items play at natural rate."""
        return self.origin.width(available)

    def ends_at(self, available: float | None = None) -> float:
        return round(self.start + self.timeline_seconds(available), 3)

    def copy(self) -> "AudioClip":
        """A separate one, origin included — see :meth:`Sound.copy` for why."""
        return replace(self, id=new_audio_id(), origin=self.origin.copy())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "origin": self.origin.to_dict(),
            "start": self.start,
            "gain_db": self.gain_db,
            "fade_in": self.fade_in,
            "fade_out": self.fade_out,
            "enabled": self.enabled,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioClip":
        return cls(
            id=str(data.get("id") or new_audio_id()),
            origin=origin_from_dict(data.get("origin")),
            start=max(0.0, _float_or(data.get("start"), 0.0)),
            gain_db=_clamp_gain(data.get("gain_db")),
            fade_in=_clamp_fade(data.get("fade_in")),
            fade_out=_clamp_fade(data.get("fade_out")),
            enabled=bool(data.get("enabled", True)),
            label=str(data.get("label") or ""),
        )


@dataclass
class Clip:
    """One use of a source in the edit: which lineage, and which slice of it.

    The in and out points are seconds into **lineage time** — the source's takes
    laid end to end — not into any one file. That is what lets a clip start
    mid-way through generation two and run across the join into generation three
    without the edit knowing there was a join at all.

    A source can back several clips at different trims, which is the whole point
    of separating the two: the bin holds what was generated, the timeline holds
    how it is used.
    """

    id: str = field(default_factory=new_clip_id)
    source_id: str = ""
    # Take ids to use instead of the source's own path. `None` means follow the
    # source, which is the default so that switching a branch in the bin updates
    # the edit; pinning is the deliberate act, not following.
    path_override: list[str] | None = None
    source_in: float = 0.0
    source_out: float | None = None
    speed: float = 1.0
    enabled: bool = True
    # Overrides the source's name on this instance only. Empty means inherit.
    label: str = ""
    # What this clip is heard as. The default is its own generation's sound, so
    # a clip written before sound was separable behaves exactly as it did.
    audio: Sound = field(default_factory=Sound)

    def path(self, source_path: list[str]) -> list[str]:
        """The lineage this clip plays: its own pin, or the source's."""
        return list(self.path_override) if self.path_override else list(source_path)

    def source_length(self, lineage_seconds: float | None = None) -> float:
        """How much source there is to cut from, in lineage time.

        Zero when nothing is rendered yet — unlike a shot, a clip cannot fall
        back to a prompt's planned length, because a lineage's length is the sum
        of however many generations actually exist.
        """
        return round(lineage_seconds, 3) if lineage_seconds and lineage_seconds > 0 else 0.0

    def out_point(self, lineage_seconds: float | None = None) -> float:
        available = self.source_length(lineage_seconds)
        if self.source_out is None:
            return available
        return min(self.source_out, available)

    def trimmed_seconds(self, lineage_seconds: float | None = None) -> float:
        start = max(0.0, self.source_in)
        return round(max(0.0, self.out_point(lineage_seconds) - start), 3)

    def timeline_seconds(self, lineage_seconds: float | None = None) -> float:
        speed = self.speed if self.speed > 0 else 1.0
        return round(self.trimmed_seconds(lineage_seconds) / speed, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "path_override": None if self.path_override is None else list(self.path_override),
            "source_in": self.source_in,
            "source_out": self.source_out,
            "speed": self.speed,
            "enabled": self.enabled,
            "label": self.label,
            "audio": self.audio.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Clip":
        source_out = data.get("source_out")
        override = data.get("path_override")
        return cls(
            id=str(data.get("id") or new_clip_id()),
            source_id=str(data.get("source_id") or ""),
            path_override=[str(x) for x in override] if isinstance(override, list) else None,
            source_in=float(data.get("source_in") or 0.0),
            source_out=None if source_out is None else float(source_out),
            speed=_clamp_speed(data.get("speed")),
            enabled=bool(data.get("enabled", True)),
            label=data.get("label", ""),
            audio=Sound.from_dict(data.get("audio")),
        )


@dataclass
class Timeline:
    """The clips in order, the mutes over them, and the sound beside them.

    The picture is still cuts-only and still positionless: `starts_at` derives
    every clip position by running total, so a gap or an overlap in the *picture*
    cannot be represented, let alone go wrong.

    Sound is where that stops being enough, and the two new lists say so in
    different ways. A :class:`Mute` is anchored to a clip and carried by it, so it
    inherits the positionless rule and needs no coordinate of its own. An
    :class:`AudioClip` cannot: it exists to sit somewhere its picture does not, so
    it carries an absolute `start` and may overlap anything, including another
    lane item. That is the one place gaps and overlaps are real here, and it is
    the price of being able to pull a sound early.
    """

    clips: list[Clip] = field(default_factory=list)
    # Stretches of a clip heard as silence, in that clip's lineage seconds.
    mutes: list[Mute] = field(default_factory=list)
    # Sound with no picture to hang off, at absolute timeline positions.
    audio: list[AudioClip] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self) -> Iterator[Clip]:
        return iter(self.clips)

    def active(self) -> list[Clip]:
        return [clip for clip in self.clips if clip.enabled]

    def index_of(self, clip_id: str) -> int:
        for index, clip in enumerate(self.clips):
            if clip.id == clip_id:
                return index
        raise KeyError(clip_id)

    def get(self, clip_id: str) -> Clip | None:
        return next((clip for clip in self.clips if clip.id == clip_id), None)

    def using(self, source_id: str) -> list[Clip]:
        """Every clip backed by one source — what a bin needs to show usage."""
        return [clip for clip in self.clips if clip.source_id == source_id]

    def add(self, clip: Clip, at: int | None = None) -> Clip:
        if at is None:
            self.clips.append(clip)
        else:
            self.clips.insert(max(0, min(at, len(self.clips))), clip)
        return clip

    def remove(self, clip_id: str) -> bool:
        """Ripple delete. The source stays in the bin — removing a use of a
        lineage from the edit is not throwing the lineage away.

        Its mutes go with it, because a mute is a window over a clip and there is
        nothing left to be a window over. Lane items do not: one lifted from this
        clip is sound the edit is using somewhere else, and deleting the picture
        it came from is not a reason to delete it.
        """
        before = len(self.clips)
        self.clips = [clip for clip in self.clips if clip.id != clip_id]
        if len(self.clips) == before:
            return False
        self.mutes = [mute for mute in self.mutes if mute.clip_id != clip_id]
        return True

    def forget_source(self, source_id: str) -> None:
        """Drop everything the edit holds for one source, in one place.

        Its clips, their mutes, and any lane item lifted from its lineage. The
        lane exception in :meth:`remove` does not apply here: that item reads a
        lineage which is about to stop existing, so keeping it would leave a
        silent block on the timeline pointing at nothing.
        """
        gone = {clip.id for clip in self.clips if clip.source_id == source_id}
        self.clips = [clip for clip in self.clips if clip.source_id != source_id]
        self.mutes = [mute for mute in self.mutes if mute.clip_id not in gone]
        self.audio = [
            item for item in self.audio
            if not (isinstance(item.origin, Lifted) and item.origin.source_id == source_id)
        ]

    def move(self, clip_id: str, to: int) -> None:
        index = self.index_of(clip_id)
        clip = self.clips.pop(index)
        self.clips.insert(max(0, min(to, len(self.clips))), clip)

    def split(
        self, clip_id: str, at: float, lineage_seconds: float | None = None
    ) -> tuple[Clip, Clip]:
        """Cut one clip in two. Both halves point at the same source.

        Simpler than it was for shots: there is no prompt to deep-copy and no
        pick to keep in step, because both halves are the same lineage seen
        through two windows.
        """
        index = self.index_of(clip_id)
        clip = self.clips[index]

        length = clip.timeline_seconds(lineage_seconds)
        if not 0 < at < length:
            raise ValueError(
                f"Split point must fall inside the clip's {length}s, got {at}s."
            )

        speed = clip.speed if clip.speed > 0 else 1.0
        cut = round(clip.source_in + at * speed, 3)

        tail = replace(
            clip,
            id=new_clip_id(),
            source_in=cut,
            source_out=clip.out_point(lineage_seconds),
            path_override=None if clip.path_override is None else list(clip.path_override),
            audio=clip.audio.copy(),
        )
        clip.source_out = cut

        # A mute is in lineage seconds and both halves are the same lineage, so
        # dividing them is a comparison against the cut and nothing more. The
        # halves are *not* clamped to either trim: a mute reaching past an edit
        # point survives there, so re-extending the trim brings it back whole.
        divided: list[Mute] = []
        for mute in self.mutes:
            if mute.clip_id != clip.id:
                divided.append(mute)
                continue
            if mute.start < cut:
                divided.append(replace(mute, end=min(mute.end, cut)))
            if mute.end > cut:
                divided.append(
                    replace(mute, id=new_mute_id(), clip_id=tail.id,
                            start=max(mute.start, cut))
                )
        self.mutes = divided

        self.clips.insert(index + 1, tail)
        return clip, tail

    def duplicate(self, clip_id: str) -> Clip:
        """Another use of the same lineage, trimmed the same way."""
        index = self.index_of(clip_id)
        source = self.clips[index]
        copy = replace(
            source,
            id=new_clip_id(),
            path_override=None if source.path_override is None else list(source.path_override),
            audio=source.audio.copy(),
        )
        # "Trimmed the same way" has to include the silences, or a duplicated
        # clip plays sound the original had deliberately taken out.
        self.mutes += [
            replace(mute, id=new_mute_id(), clip_id=copy.id)
            for mute in self.mutes if mute.clip_id == source.id
        ]
        self.clips.insert(index + 1, copy)
        return copy

    # -- mutes ------------------------------------------------------------

    def mutes_for(self, clip_id: str) -> list[Mute]:
        """One clip's mutes, earliest first. Empty for a clip with none."""
        found = [mute for mute in self.mutes if mute.clip_id == clip_id]
        return sorted(found, key=lambda mute: mute.start)

    def mute(self, mute_id: str) -> Mute | None:
        return next((mute for mute in self.mutes if mute.id == mute_id), None)

    def add_mute(self, mute: Mute) -> Mute:
        self.mutes.append(mute)
        return mute

    def remove_mute(self, mute_id: str) -> bool:
        before = len(self.mutes)
        self.mutes = [mute for mute in self.mutes if mute.id != mute_id]
        return len(self.mutes) != before

    # -- the audio lane ----------------------------------------------------

    def active_audio(self) -> list[AudioClip]:
        """The lane items that play, earliest first.

        Sorted by `start` rather than left in insertion order: unlike clips,
        where order *is* the edit, a lane item's order carries no meaning, so the
        only useful arrangement is the one you see on the ruler.
        """
        return sorted(
            (item for item in self.audio if item.enabled),
            key=lambda item: item.start,
        )

    def audio_clip(self, audio_id: str) -> AudioClip | None:
        return next((item for item in self.audio if item.id == audio_id), None)

    def add_audio(self, item: AudioClip) -> AudioClip:
        self.audio.append(item)
        return item

    def remove_audio(self, audio_id: str) -> bool:
        before = len(self.audio)
        self.audio = [item for item in self.audio if item.id != audio_id]
        return len(self.audio) != before

    def starts_at(self, lengths: dict[str, float] | None = None) -> dict[str, float]:
        """Each clip's timeline start, by running total over the active ones."""
        cursor = 0.0
        result: dict[str, float] = {}
        for clip in self.clips:
            result[clip.id] = round(cursor, 3)
            if clip.enabled:
                cursor += clip.timeline_seconds(_length_of_clip(lengths, clip))
        return result

    def duration(self, lengths: dict[str, float] | None = None) -> float:
        return round(
            sum(
                clip.timeline_seconds(_length_of_clip(lengths, clip))
                for clip in self.active()
            ),
            3,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "clips": [clip.to_dict() for clip in self.clips],
            "mutes": [mute.to_dict() for mute in self.mutes],
            "audio": [item.to_dict() for item in self.audio],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Timeline":
        """Read a timeline. The two sound lists are optional on the way in, so a
        project saved before the lane existed loads with an empty one rather than
        needing a migration."""
        return cls(
            clips=[Clip.from_dict(entry) for entry in data.get("clips") or []],
            mutes=[Mute.from_dict(entry) for entry in data.get("mutes") or []],
            audio=[AudioClip.from_dict(entry) for entry in data.get("audio") or []],
        )


def _length_of_clip(lengths: dict[str, float] | None, clip: Clip) -> float | None:
    return None if lengths is None else lengths.get(clip.id)


def _float_or(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _clamp_fade(value: Any) -> float:
    return min(max(_float_or(value, DEFAULT_FADE), 0.0), MAX_FADE)


def _clamp_gain(value: Any) -> float:
    return min(max(_float_or(value, 0.0), MIN_GAIN_DB), MAX_GAIN_DB)


def _clamp_speed(value: Any) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0
    if speed <= 0:
        return 1.0
    return min(max(speed, MIN_SPEED), MAX_SPEED)
