"""The project cast: identities that outlive the prompt that mentions them.

A character used to exist three times over, unlinked — as an opaque speaker key
in `Line.speakers`, as a `describe` phrase retyped on every first appearance,
and as a `role` string beside a reference path in whichever sources happened to
attach the same file. Keeping source nine looking like source one meant
retyping a description from memory.

A `Member` is that identity written once. Prompts point at one by id, and
`resolve` folds the description, the voice and the identity files back into the
prompt on the way to `compose`.

Resolution is **live**, deliberately unlike `projects.Defaults`, which copies
its blocks onto a segment at creation and lets them diverge. Defaults are a
starting point and should drift; an identity is the opposite — "the same person
looks the same in every shot" is the whole reason the list exists, so editing a
member has to reach every segment that names them. Past renders are unaffected
either way: a take freezes its prompt text, and `/render/reshoot` replays from
that text rather than recomposing.

Names never reach the prompt. MiniMax's own guidance is to write `(S1)` or
`<Picture 2>` in instructional prose rather than "Mara", because the tag is the
routing identity the model tracks; `name` is for the author and the editor.
"""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import mentions
from .document import Reference, ReferenceKind, Subject, VideoPrompt

KINDS = ("character", "environment", "prop", "voice")

# How a member's `<Subject N>` line names them before the citation: "is the
# *person* in <Picture 1>". A word rather than the kind itself, because the
# sentence is prose the model reads, not a schema field.
SUBJECT_NOUNS: dict[str, str] = {
    "character": "person",
    "environment": "setting",
    "prop": "object",
    "voice": "speaker",
}

ID_PREFIX = "cast_"
_ID_RE = re.compile(rf"^{ID_PREFIX}[0-9a-f]{{8}}$")


class CastError(ValueError):
    """Raised for an unknown kind or a member that isn't there."""


def mint_id() -> str:
    return f"{ID_PREFIX}{uuid.uuid4().hex[:8]}"


def looks_like_id(key: str) -> bool:
    """True for a string minted by `mint_id`.

    Speaker keys are free-form — `"baker"` is a perfectly good one — so a key
    is only held to the cast when it is shaped like a member id. That is what
    lets a prompt written before any of this existed keep linting clean.
    """
    return bool(_ID_RE.match(key or ""))


@dataclass
class Member:
    """One reusable identity: a character, a location, a prop, or a voice.

    `appearance` is the load-bearing field — the canonical description every
    segment inherits. `voice` is timbre, accent and register; it joins the
    appearance to form the identifying phrase a first vocal appearance needs.
    `references` are this member's identity files, fed to the reference node.
    On screen, the member resolves to one `<Subject N>` that cites those files
    rather than to one `<Picture N>` line per file — the model tracks subjects,
    and three sheets of the same face are one person, not three.
    """

    id: str = field(default_factory=mint_id)
    name: str = ""
    kind: str = "character"
    appearance: str = ""
    voice: str = ""
    references: list[Reference] = field(default_factory=list)
    notes: str = ""

    def identity(self) -> str:
        """The identifying phrase, for `Line.describe` and a reference role.

        Appearance and voice are both author-written prose, so they are joined
        with a comma and nothing cleverer. A member with neither returns empty
        and resolution leaves the line alone rather than writing a blank in.
        """
        parts = [part.strip().rstrip(".") for part in (self.appearance, self.voice)]
        return ", ".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "appearance": self.appearance,
            "voice": self.voice,
            "notes": self.notes,
            "references": [
                {
                    "kind": item.kind,
                    "path": item.path,
                    "role": item.role,
                    "retention": item.retention,
                }
                for item in self.references
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Member":
        kind = str(data.get("kind") or "character")
        return cls(
            id=str(data.get("id") or mint_id()),
            name=str(data.get("name") or ""),
            kind=kind if kind in KINDS else "character",
            appearance=str(data.get("appearance") or ""),
            voice=str(data.get("voice") or ""),
            references=[
                reference
                for reference in map(_reference, data.get("references") or [])
                if reference is not None
            ],
            notes=str(data.get("notes") or ""),
        )


def _reference(entry: Any) -> Reference | None:
    if not isinstance(entry, dict):
        return None
    kind = str(entry.get("kind") or "image")
    if kind not in ("image", "video", "video_audio", "audio"):
        return None
    path = str(entry.get("path") or "")
    if not path:
        return None
    return Reference(
        kind=kind,  # type: ignore[arg-type]
        path=path,
        role=str(entry.get("role") or ""),
        retention=str(entry.get("retention") or ""),
    )


@dataclass
class Cast:
    """A project's members, in author order."""

    members: list[Member] = field(default_factory=list)

    def get(self, member_id: str) -> Member | None:
        return next((m for m in self.members if m.id == member_id), None)

    def named(self, name: str) -> Member | None:
        folded = (name or "").casefold().strip()
        return next((m for m in self.members if m.name.casefold().strip() == folded), None)

    def add(self, member: Member) -> Member:
        self.members.append(member)
        return member

    def remove(self, member_id: str) -> bool:
        before = len(self.members)
        self.members = [m for m in self.members if m.id != member_id]
        return len(self.members) != before

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self):
        return iter(self.members)

    def to_dict(self) -> list[dict[str, Any]]:
        return [member.to_dict() for member in self.members]

    @classmethod
    def from_dict(cls, data: Any) -> "Cast":
        if not isinstance(data, list):
            return cls()
        return cls([Member.from_dict(entry) for entry in data if isinstance(entry, dict)])


def check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise CastError(f"Unknown cast kind {kind!r}; expected one of {', '.join(KINDS)}.")
    return kind


# -- resolution ------------------------------------------------------------


def appearing(prompt: VideoPrompt, cast: "Cast | None") -> list[Member]:
    """Every member this prompt puts on screen, in order of first appearance.

    Speakers come first because they are ordered by the dialogue itself, which
    is the same order `compose` numbers `(S1)`, `(S2)` in; the silent members
    named in `prompt.cast` follow in the order they were listed.
    """
    if cast is None or not len(cast):
        return []

    found: list[Member] = []

    def note(member: Member | None) -> None:
        if member is not None and member not in found:
            found.append(member)

    for beat in prompt.storyboard.ordered_beats():
        for line in beat.lines:
            if not line.text.strip():
                continue
            for key in line.speakers:
                note(cast.get(key))
    for member_id in prompt.cast:
        note(cast.get(member_id))
    # A linked reference row is itself a statement that the member is in this
    # shot — attaching their face sheet and then having to tick them present as
    # well would be a second switch for one decision.
    for item in prompt.references.items:
        if item.is_linked:
            note(cast.get(item.member))
    # A per-segment subject row says the same thing about the same member.
    for subject in prompt.subjects.items:
        if subject.is_linked:
            note(cast.get(subject.member))
    # And so does writing `@Mara` into the action. A mention is a pointer at a
    # label, and the only way there is a label to point at is for the member to
    # be resolved into this shot — so mentioning somebody puts them in it,
    # exactly as attaching their face sheet does. Last, so that adding a
    # mention to a prompt that already had a cast cannot renumber it.
    for member in mentioned(prompt, cast):
        note(member)

    return found


def mentioned(prompt: VideoPrompt, cast: "Cast | None") -> list[Member]:
    """Members named by an `@mention` anywhere in the prose, in reading order."""
    if cast is None or not len(cast):
        return []
    names = [(member.name, member.id) for member in cast]
    found: list[Member] = []
    for field in mentions.fields(prompt):
        for key in mentions.keys(field.text, names):
            member = cast.get(key)
            if member is not None and member not in found:
                found.append(member)
    return found


def resolve(prompt: VideoPrompt, cast: "Cast | None") -> VideoPrompt:
    """A copy of `prompt` with every member it names written into it.

    Four things happen, and each one is skipped where the author has already
    said something: a first vocal appearance with no `describe` gets the
    member's identifying phrase; a member on screen who never speaks has their
    appearance stated in the opening; every member's identity files are
    appended to `references` so the reference node actually receives them; and
    every `@mention` in the prose becomes the label that now routes to them.

    `prompt.undescribed` cuts across the last two: a member listed there keeps
    their files and their `<Subject N>`, and loses the description in *both*
    places it would otherwise land — the opening and the subject line, which
    carries `retention_analysis` with it. Silencing only one of the three was
    the whole reason the tick appeared not to work.

    Returns the prompt untouched when there is nothing to resolve, so the
    common path — a cast-free prompt, the CLI, the bare ComfyUI node — costs a
    single emptiness check. A prompt whose only cast trace is a linked
    reference to a member since deleted still resolves, so the dead row is
    cleared rather than rendering off a path nothing points at any more.
    """
    members = appearing(prompt, cast)
    stale = cast is not None and (
        any(item.is_linked for item in prompt.references.items)
        or any(subject.is_linked for subject in prompt.subjects.items)
    )
    if not members and not stale:
        return prompt
    assert cast is not None  # `appearing` returns nothing without one

    resolved = copy.deepcopy(prompt)
    spoke = _fill_describes(resolved, cast)
    _state_silent_members(resolved, [m for m in members if m.id not in spoke])
    _attach_references(resolved, members)
    _attach_subjects(resolved, members)
    # Last, because a mention composes to a *number* and the numbers do not
    # exist until the two lists above have been rebuilt.
    _expand_mentions(resolved, cast)
    return resolved


def _expand_mentions(prompt: VideoPrompt, cast: "Cast") -> None:
    """Turn every `@name` into the label the model actually routes on.

    Three answers, in the order the format prefers them. A member with identity
    files has a `<Subject N>`, and that is the strongest pointer there is. A
    member who only speaks has the global speaker id the dialogue already
    carries, written as `the person (S1)` — the id alone would be a sentence
    with a parenthesis for a subject, and the guide's own prose pairs the two.
    A member with neither has nothing to be pointed *at*: their appearance
    reached the prompt as prose in the opening and nothing there is labelled,
    so the mention falls back to the bare noun and `lint.mentions.untagged`
    says why it reads as "the person".

    A mention of a name the cast no longer has is left exactly as written. It
    is not ours to guess at, and `lint.mentions.unknown` has already said so.
    """
    names = [(member.name, member.id) for member in cast]
    if not names:
        return
    speakers = prompt.storyboard.speaker_ids()

    def label(hit: mentions.Hit) -> str:
        member = cast.get(hit.key)
        if member is None:
            return f"{mentions.MARK}{hit.name}"
        return mention_label(prompt, member, speakers)

    for field in mentions.fields(prompt):
        updated = mentions.rewrite(field.text, names, label)
        if updated != field.text:
            field.write(updated)


def mention_label(
    prompt: VideoPrompt, member: Member, speakers: dict[str, str] | None = None
) -> str:
    """What `@member` composes to against this prompt. See `_expand_mentions`.

    Public because the editor shows it: a mention is only worth typing if you
    can see what it became, and working that out a second time in JavaScript is
    how the two answers start to differ.

    Read against a *resolved* prompt — the subject list a mention points into is
    the rebuilt one.
    """
    subject = prompt.subjects.for_member(member.id)
    if subject is not None:
        return prompt.subjects.tag(subject)
    if speakers is None:
        speakers = prompt.storyboard.speaker_ids()
    noun = SUBJECT_NOUNS.get(member.kind, "person")
    speaker = speakers.get(member.id, "")
    return f"the {noun} ({speaker})" if speaker else f"the {noun}"


def _fill_describes(prompt: VideoPrompt, cast: "Cast") -> set[str]:
    """Identify each speaking member once, the first time they are heard.

    Returns the ids that spoke, so the caller knows which members still need
    stating in prose. The format identifies a voice on its first appearance and
    refers to it by `(S1)` afterwards, so later lines are left alone even when
    they too have an empty `describe`.
    """
    seen: set[str] = set()
    for beat in prompt.storyboard.ordered_beats():
        for line in beat.lines:
            if not line.text.strip():
                continue
            for key in line.speakers:
                member = cast.get(key)
                if member is None or member.id in seen:
                    continue
                seen.add(member.id)
                if not line.describe.strip():
                    line.describe = member.identity()
    return seen


def _state_silent_members(prompt: VideoPrompt, members: list[Member]) -> None:
    """Describe the members who are present but never speak.

    They ride on `overview`, which composes into the clause immediately after
    the style contract — where an establishing description of the scene and who
    is in it belongs. A member with no appearance written is skipped rather
    than contributing an empty sentence.

    A member the prompt lists as `undescribed` is skipped too. That is the
    escape hatch for a shot that wants a member's identity files without the
    establishing paragraph they normally bring: a location's appearance runs to
    window walls and doorways, and stating it in a close shot asks the camera
    to go and show them.
    """
    statements = [
        member.identity()
        for member in members
        if member.identity() and member.id not in prompt.undescribed
    ]
    if not statements:
        return
    # Each statement becomes its own sentence in the opening, so each one is
    # capitalised. Joined raw, the second onward came out mid-sentence —
    # "...a wooden coffee table. thick white fur, pointed ears" — which reads
    # as a fragment the model has no reason to trust as a new subject.
    parts = [prompt.overview.strip().rstrip(".")] + [
        statement[:1].upper() + statement[1:] for statement in statements
    ]
    prompt.overview = ". ".join(part for part in parts if part)


def _attach_references(prompt: VideoPrompt, members: list[Member]) -> None:
    """Feed each member's identity files to the reference node.

    The list is rebuilt rather than appended to, in one fixed order: the shot's
    own uploads first, then the cast's files in member order. That ordering is
    what keeps `<Picture 2>` meaning the same thing from one save to the next —
    tags are positional, so a row that slid in ahead of another would silently
    repoint every tag already written into the prose.

    A *linked* row (one carrying `member`/`slot`) is not a file of its own but
    a place to say what that member's file means here: its `role` and
    `retention` win, while `kind` and `path` still come from the member, so
    swapping the member's sheet flows through instead of stranding a copy. It
    resolves into the position the member's file would have taken anyway, and
    is dropped when the member or the slot is gone.

    Plain rows are still deduped by path, which is how a segment overrode a
    member before links existed; those projects keep working unchanged. A
    member reference with no role of its own inherits the member's identifying
    phrase, which is what `subject_definitions` will state.
    """
    by_id = {member.id: member for member in members}
    plain: list[Reference] = []
    linked: dict[tuple[str, int], Reference] = {}
    for item in prompt.references.items:
        if not item.is_linked:
            plain.append(item)
            continue
        member = by_id.get(item.member)
        if member is None or not 0 <= item.slot < len(member.references):
            continue  # the member, or that identity file, is no longer there
        linked.setdefault((item.member, item.slot), item)

    resolved = list(plain)
    attached = {item.path for item in plain}
    for member in members:
        for slot, reference in enumerate(member.references):
            if reference.path in attached:
                continue
            attached.add(reference.path)
            row = linked.get((member.id, slot))
            resolved.append(
                Reference(
                    kind=reference.kind,
                    path=reference.path,
                    role=(row.role.strip() if row else "")
                    or reference.role.strip()
                    or _default_role(member, reference),
                    retention=(row.retention.strip() if row else "") or reference.retention,
                    member=member.id if row else "",
                    slot=slot if row else 0,
                )
            )

    prompt.references.items = resolved


def _default_role(member: Member, reference: Reference) -> str:
    """What an identity file says about itself when nothing was written on it.

    An appearance is the wrong sentence for a soundtrack: the file carries a
    voice, not a face, and the `<Subject N>` line it hangs off already says
    whose voice it is. The visual case keeps the member's identifying phrase,
    though a subject citing that file normally speaks for it instead.
    """
    if reference.kind in ("audio", "video_audio"):
        return "the voice-timbre reference"
    return member.identity()


def _attach_subjects(prompt: VideoPrompt, members: list[Member]) -> None:
    """Give every member on screen a `<Subject N>` that cites their files.

    This is the guide's distinction made real: `<Picture N>` names a file,
    `<Subject N>` names the thing in the video. A face sheet is only ever the
    second, so the member's files are cited *inside* one subject sentence and
    lose the standalone picture entries they used to get — one per file, each
    restating the same person to a model that was already tracking them.

    Rebuilt in one fixed order, the way the reference list is: the segment's
    own subjects first, then the cast in member order, so `<Subject 2>` keeps
    meaning the same thing between saves. A row carrying `member` is a *link* —
    somewhere to say what this member is in *this* shot — and contributes only
    its `role` and `retention`; sources, noun and the fallback role always come
    from the member, and the row is dropped when the member is gone.

    A member with no identity file attached gets no subject: `subject_definitions`
    is a list of *referenced* content, and their appearance already reaches the
    prompt through the opening (see :func:`_state_silent_members`) or their own
    first spoken line. Which is also why the editor will not let such a member
    be marked `undescribed`: with no files to route to and no paragraph to
    write, turning the description off would erase them from the prompt.
    """
    attached = {item.path for item in prompt.references.items}
    plain: list[Subject] = []
    linked: dict[str, Subject] = {}
    for subject in prompt.subjects.items:
        if subject.is_linked:
            linked.setdefault(subject.member, subject)
        else:
            plain.append(subject)

    resolved = list(plain)
    for member in members:
        sources = [
            reference.path
            for reference in member.references
            if reference.path in attached
        ]
        if not sources:
            continue
        row = linked.get(member.id)
        # `undescribed` is a statement about the words, not about the files:
        # the subject keeps its tag, its sources and its retention marker, and
        # loses only the description that would otherwise be repeated here and
        # in `retention_analysis` after the opening has already dropped it. A
        # role typed on this segment's own row still wins — an author who wrote
        # a sentence for this shot means it.
        described = member.id not in prompt.undescribed
        resolved.append(
            Subject(
                role=(row.role.strip() if row else "")
                or (member.identity() if described else ""),
                retention=(row.retention.strip() if row else ""),
                sources=sources,
                member=member.id,
                noun=SUBJECT_NOUNS.get(member.kind, ""),
                label=member.name,
            )
        )

    prompt.subjects.items = resolved


def unknown_ids(prompt: VideoPrompt, cast: "Cast | None") -> list[str]:
    """Ids the prompt points at that the cast does not have.

    Speaker keys are only judged when they are shaped like a minted id — a
    hand-written key such as `"baker"` is not a broken reference, it is a
    prompt that predates the cast. Entries in `prompt.cast` are always ids, so
    every one of those that misses is reported.
    """
    known = {member.id for member in (cast or Cast())}
    missing: list[str] = []

    def note(key: str) -> None:
        if key and key not in known and key not in missing:
            missing.append(key)

    for beat in prompt.storyboard.ordered_beats():
        for line in beat.lines:
            for key in line.speakers:
                if looks_like_id(key):
                    note(key)
    for member_id in prompt.cast:
        note(member_id)
    for item in prompt.references.items:
        if item.is_linked:
            note(item.member)
    for subject in prompt.subjects.items:
        if subject.is_linked:
            note(subject.member)

    return missing
