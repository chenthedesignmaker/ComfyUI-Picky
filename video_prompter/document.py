"""The structured prompt document.

This is the authoring model. `compose` turns it into the text MiniMax H3 was
trained on: an optional image-alignment instruction, then the three core fields
`integrated_multimodal_description`, `overall_soundscape`, and
`non_diegetic_music`. The split between the two sound fields is load-bearing —
the model treats diegetic sound (ambient, action, non-verbal human sound) and
the audience-only score as separate channels — so `audio` holds the former and
`music` the latter.

There is no negative-prompt input on the local nodes — `MiniMaxH3ImageToVideo`
and `MiniMaxH3ReferenceToVideo` each take a single `prompt` string — and the
official format has no negative section, so `negatives` is retained on the model
for the editor but is not composed into the prompt text (naming unwanted things
in the positive stream tends to summon them).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, Literal

from . import h3_spec
from .storyboard import Storyboard

ReferenceKind = Literal["image", "video", "video_audio", "audio"]

# The tags the local nodes expect the prompt to use. Video soundtracks and
# standalone audio share one <Audio j> numbering, soundtracks first.
_KIND_TAGS: dict[ReferenceKind, str] = {
    "image": "Picture",
    "video": "Video",
    "video_audio": "Audio",
    "audio": "Audio",
}

_KIND_LIMITS: dict[ReferenceKind, int] = {
    "image": h3_spec.MAX_REFERENCE_IMAGES,
    "video": h3_spec.MAX_REFERENCE_VIDEOS,
    "video_audio": h3_spec.MAX_REFERENCE_VIDEO_AUDIOS,
    "audio": h3_spec.MAX_REFERENCE_AUDIO,
}

REFERENCE_KINDS: tuple[ReferenceKind, ...] = ("image", "video", "video_audio", "audio")


@dataclass
class Reference:
    """One omni-reference file plus the role it plays.

    `role` is prose describing what the reference is for; the prompt refers to
    it by tag, e.g. `<Picture 2>`. `retention` is the ref2v relationship marker
    (from `h3_spec.VISUAL_RETENTION` / `AUDIO_RETENTION`); empty composes to the
    per-kind default.

    The role only becomes a `subject_definitions` line when the file stands on
    its own — a keyframe, a source video, an audio track. A file that merely
    *shows* something (a face sheet, a location plate) is cited inside that
    thing's `Subject` instead and writes no line of its own, per guide 2.2.

    `member`/`slot` make the row a *link* to one of a cast member's identity
    files rather than a copy of it. A linked row exists so the segment can say
    what that file means *here* — its own `role` and `retention` win — while
    `kind` and `path` keep resolving from the member, so swapping the member's
    face sheet flows through to every segment instead of stranding copies.
    Both empty (`member == ""`) is a plain reference, which is every reference
    written before this existed.
    """

    kind: ReferenceKind
    path: str
    role: str = ""
    retention: str = ""
    member: str = ""
    slot: int = 0

    @property
    def is_audio(self) -> bool:
        return self.kind in ("audio", "video_audio")

    @property
    def is_linked(self) -> bool:
        """True when this row defers to a cast member for its file."""
        return bool(self.member)


@dataclass
class Subject:
    """One unit of content the target video will actually contain.

    The guide's `<Subject N>`: the person, place, prop or style the model has to
    track, as opposed to the file it was cut from. `<Picture N>` names a file;
    `<Subject N>` names the thing. Kept as its own list rather than a flag on
    `Reference` because the relationship is many-to-many — one subject can be
    defined by several files ("the face from <Picture 1>, the walk from
    <Video 1>") and one file can supply several subjects.

    `role` is the prose after `<Subject N> is`; `noun` is the word the citation
    hangs on ("the *person* in <Picture 1>") and may be left empty. `sources`
    are the reference paths that define it. A *visual* source cited here loses
    the standalone `<Picture N>`/`<Video N>` line it would otherwise get, which
    is what the guide asks for; an audio source keeps its `<Audio N>` line and
    gains a back-link to this subject instead.

    `member` ties the subject to a cast member, and then it is a *link* in the
    same sense a linked `Reference` is: `cast.resolve` keeps `sources`, `noun`
    and the default `role` in step with the member, and only the `role` and
    `retention` written here override, for this segment alone.
    """

    role: str = ""
    retention: str = ""
    sources: list[str] = field(default_factory=list)
    member: str = ""
    noun: str = ""
    # Author-facing name, for the editor. Never composed: the tag is the
    # routing identity the model tracks, and a name in the prose is noise.
    label: str = ""

    @property
    def is_linked(self) -> bool:
        """True when this subject defers to a cast member for its sources."""
        return bool(self.member)


@dataclass
class SubjectSet:
    items: list[Subject] = field(default_factory=list)

    def ordinal(self, subject: Subject) -> int:
        """1-based number in the `<Subject N>` space, matched by identity."""
        for position, item in enumerate(self.items, start=1):
            if item is subject:
                return position
        return self.items.index(subject) + 1

    def tag(self, subject: Subject) -> str:
        return f"<Subject {self.ordinal(subject)}>"

    def tags(self) -> list[str]:
        return [self.tag(item) for item in self.items]

    def for_member(self, member_id: str) -> Subject | None:
        return next((item for item in self.items if item.member == member_id), None)

    def citing(self, path: str) -> Subject | None:
        """The first subject that draws on `path`, if any."""
        if not path:
            return None
        return next((item for item in self.items if path in item.sources), None)

    def cited_paths(self) -> set[str]:
        """Every reference path some subject already speaks for."""
        return {path for item in self.items for path in item.sources if path}

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class ReferenceSet:
    items: list[Reference] = field(default_factory=list)

    def of(self, kind: ReferenceKind) -> list[Reference]:
        return [item for item in self.items if item.kind == kind]

    def limit_for(self, kind: ReferenceKind) -> int:
        return _KIND_LIMITS[kind]

    def ordinal(self, reference: Reference) -> int:
        """1-based number within this reference's tag space.

        Matched by identity, not equality: `Reference` compares by value, so
        two references that are still blank — or that happen to share a role —
        would otherwise both resolve to the first one's number.
        """
        if reference.kind in ("video_audio", "audio"):
            # Soundtracks are presented before standalone audio and share the
            # <Audio j> sequence with it.
            sequence = self.of("video_audio") + self.of("audio")
        else:
            sequence = self.of(reference.kind)

        for position, item in enumerate(sequence, start=1):
            if item is reference:
                return position
        # Not one of ours: fall back to value matching so external callers
        # passing an equal copy still get a sensible answer.
        return sequence.index(reference) + 1

    def tag(self, reference: Reference) -> str:
        """The token the prompt uses to point at this reference."""
        return f"<{_KIND_TAGS[reference.kind]} {self.ordinal(reference)}>"

    def tags(self) -> list[str]:
        return [self.tag(item) for item in self.items]

    def render_roles(self) -> str:
        """The role-declaration block, or empty when nothing is described."""
        described = [item for item in self.items if item.role.strip()]
        if not described:
            return ""
        return " ".join(
            f"{self.tag(item)} is {item.role.strip().rstrip('.')}." for item in described
        )

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class VideoPrompt:
    """A complete H3 prompt, still structured.

    `storyboard` owns the duration in whole seconds; the frame count the node
    will really use is derived, because it snaps to a 17k+5 grid.
    """

    storyboard: Storyboard
    style: str = ""
    overview: str = ""
    camera: str = ""
    # Diegetic soundscape: ambient, physical-action, and non-verbal human sound.
    # Rendered as `overall_soundscape`. Dialogue and score do NOT belong here.
    audio: str = ""
    # Non-diegetic score only the audience hears. Rendered as `non_diegetic_music`;
    # empty composes to `N/A`.
    music: str = ""
    screen_text: list[str] = field(default_factory=list)
    # Project cast members on screen who never speak — a location, a silent
    # character. Speaking members are named by `Line.speakers` instead, so a
    # member listed here as well is simply resolved once. Ids only; the
    # descriptions live on the project and are folded in by `cast.resolve`.
    cast: list[str] = field(default_factory=list)
    # Members present here whose `appearance` must NOT be stated in the opening.
    # Being on screen normally writes that appearance into the clause after the
    # style contract, which is right for an establishing shot and wrong for a
    # close one: a location's appearance is a paragraph of window walls and
    # doorways, and stating it asks the camera to show them. Listing the member
    # here keeps their identity files — the reason they were attached — without
    # the prose. Ids only, and ignored for a member who speaks, since a speaker
    # is identified on their line instead.
    undescribed: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    references: ReferenceSet = field(default_factory=ReferenceSet)
    # The `<Subject N>` layer over those files: what the video actually
    # contains, as opposed to which files it was given. Cast members on screen
    # resolve into this list; author-written rows sit ahead of them. A subject
    # citing a picture is what suppresses that picture's own definition line.
    subjects: SubjectSet = field(default_factory=SubjectSet)
    # ref2v (full-reference) authoring. `ref_task` are the summary's task types
    # (from h3_spec.REF_TASK_TYPES, joined with " + "); `ref_summary` is the one
    # paragraph after that prefix. Both only matter when references are attached.
    ref_task: list[str] = field(default_factory=list)
    ref_summary: str = ""
    # i2v keyframes, filenames in ComfyUI's input dir. Unlike `references`
    # (omni-refs on MiniMaxH3ReferenceToVideo) these are the first/last-frame
    # image inputs on MiniMaxH3ImageToVideo, so a shot uses one node or the
    # other, never both. `first_frame` is the geometry anchor; `last_frame` an
    # optional end keyframe. Their presence is what makes a shot render as i2v.
    first_frame: str = ""
    last_frame: str = ""
    width: int = h3_spec.DEFAULT_WIDTH
    height: int = h3_spec.DEFAULT_HEIGHT
    # When two speakers share a shot, state plainly that they are different
    # people. H3 blends co-present identities often enough that the sentence
    # earns its characters; set False for a prompt where it reads as noise.
    guard_identities: bool = True
    storyboard_header: str = (
        "Storyboard (each shot a separate scene, rapid cuts, "
        "all landing on the musical beats)"
    )
    label_beats: bool = True

    @property
    def duration(self) -> int:
        """The requested budget, in whole seconds."""
        return self.storyboard.duration

    @property
    def frames(self) -> int:
        """What the node's `length` input will be set to."""
        return h3_spec.frames_for_seconds(self.storyboard.duration)

    @property
    def rendered_seconds(self) -> float:
        """The real clip length, which the frame grid rarely leaves round."""
        return h3_spec.seconds_for_frames(self.frames)

    def subject_owns(self, reference: Reference) -> bool:
        """True when a subject already accounts for this file.

        Such a file writes no definition line of its own: the guide asks for a
        picture that merely *shows* a character or a room to be cited inside
        that subject instead (2.2). Visual files only — an `<Audio N>` keeps
        its line and names the subject it belongs to (2.4).
        """
        if reference.is_audio:
            return False
        return self.subjects.citing(reference.path) is not None

    def defined_tags(self) -> list[str]:
        """Every label the composed document defines, subjects first.

        Not the same as every attached file: the face sheets behind a subject
        have no tag of their own to talk about.
        """
        return self.subjects.tags() + [
            self.references.tag(item)
            for item in self.references.items
            if not self.subject_owns(item)
        ]

    def for_beats(self, indices: Iterable[int]) -> "VideoPrompt":
        """This prompt with only the named shots kept — the scene around them intact.

        Everything that describes the *scene* rather than a moment inside it
        comes across whole: the style contract, both sound fields, the cast,
        the references and the subjects standing on them. That is the point —
        two shots lifted out to be generated on their own still happen in the
        same place, to the same people, in the same light, and a copy that
        dropped all that would be a different scene that merely shares its
        words.

        The keyframes are the exception. `first_frame` fixes the geometry of
        the opening frame and `last_frame` the closing one, so each survives
        only a pick that still opens, or still closes, where it was pointed.

        Speaker identity needs no repair here: `(S1)`, `(S2)` … are assigned at
        compose time by first vocal appearance, so they simply renumber. What a
        pick *can* leave behind is the `describe` phrase that introduced a
        speaker in a shot nobody kept — the linter is the right place to notice
        that, on a prompt that has become its own document.
        """
        beats = self.storyboard.beats
        kept = sorted({int(i) for i in indices if 0 <= int(i) < len(beats)})
        made = deepcopy(self)
        made.storyboard = self.storyboard.extract(kept)
        if not kept or kept[0] != 0:
            made.first_frame = ""
        if not kept or kept[-1] != len(beats) - 1:
            made.last_frame = ""
        return made

    def set_ratio(self, ratio: str) -> None:
        """Size the canvas from an aspect ratio, the way the node would."""
        self.width, self.height = h3_spec.canvas_for_ratio(ratio)

    @classmethod
    def blank(cls, duration: int = 10) -> "VideoPrompt":
        return cls(storyboard=Storyboard(duration=duration))
