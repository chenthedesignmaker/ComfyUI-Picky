"""`@Mara` — naming a cast member from inside the prose.

A shot with two people in it refers to them in nearly every field: the opening
states who is there, the action says which of them moves, the retention section
says which one to hold. The format has a label for that — `<Subject 2>` — and
the label is *positional*, so it belongs to the member's place in the cast
rather than to the member. Reorder the cast, or attach a face sheet to somebody
who did not have one, and every number after them shifts. The prose does not
shift with it, because the prose is a string: what was `<Subject 2>` is now
`<Subject 3>`, and nothing says so.

A mention is that same pointer written the way it is thought about. `@Mara` in
any prose field is a link to the member, and `cast.resolve` turns it into
whatever label routes to her *at that moment* — her `<Subject N>` if she has
identity files, her `(Sx)` if she only speaks. Renumbering then costs nothing,
because the number was never typed.

Mentions are stored by **name**, not by id. An id is stable and unreadable, and
these fields are read constantly — by the author in the editor, by an agent
through the API, by anyone opening the JSON. So the readable form is the stored
form, and a rename is handled by rewriting the mentions (see `rename`) rather
than by making every field unreadable to guard against one.

This module owns the *syntax* only: where a mention starts and ends, and which
fields may hold one. What a mention means — which label it composes to — is
`cast`'s, because that is where members live.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from .document import VideoPrompt

MARK = "@"

_WORD = re.compile(r"\w")

# A bare `@word`, so a mention that names nobody can be *noticed* rather than
# composed silently as itself. Deliberately one word: a multi-word member is
# found by name in `find`, and what `find` did not claim is either a typo or an
# address somebody pasted.
_LOOSE = re.compile(r"(?<![\w@])@([A-Za-z][\w'’-]*)")


@dataclass(frozen=True)
class Hit:
    """One `@name` inside a field: where it sits, and who it names.

    `name` is the text as written, which may differ from the member's own name
    in case alone — matching is case-insensitive, so `@mara` still lands.
    """

    start: int
    end: int
    key: str
    name: str


@dataclass
class Field:
    """One piece of authored prose a mention may appear in.

    `write` puts a new value back where `text` came from, so a caller can walk
    the prompt once and rewrite in place without knowing its shape.
    """

    address: str
    label: str
    text: str
    write: Callable[[str], None]


Names = Iterable[tuple[str, str]]


def _index(names: Names) -> list[tuple[str, str, str]]:
    """`(name, folded name, key)`, longest first.

    Longest first is what makes multi-word names work without a delimiter: with
    both "Mara" and "Mara Chen" in the cast, `@Mara Chen` has to try the longer
    one before the shorter one claims the first four letters and leaves "Chen"
    stranded in the prose.
    """
    entries = [
        (clean, clean.casefold(), key)
        for clean, key in ((str(name or "").strip(), key) for name, key in names)
        if clean
    ]
    entries.sort(key=lambda entry: -len(entry[0]))
    return entries


def find(text: str, names: Names) -> list[Hit]:
    """Every mention in `text`, left to right.

    An `@` with a word character in front of it is skipped: that is the middle
    of an email address, not the start of a mention.
    """
    entries = _index(names)
    if not text or not entries:
        return []

    hits: list[Hit] = []
    cursor = 0
    while True:
        at = text.find(MARK, cursor)
        if at < 0:
            return hits
        if at and _WORD.match(text[at - 1]):
            cursor = at + 1
            continue
        rest = text[at + 1 :]
        for clean, folded, key in entries:
            length = len(clean)
            if rest[:length].casefold() != folded:
                continue
            # The character after the name has to end a word, or `@Mara` would
            # match inside `@Marathon`.
            after = rest[length : length + 1]
            if after and _WORD.match(after):
                continue
            hits.append(Hit(at, at + 1 + length, key, rest[:length]))
            break
        cursor = hits[-1].end if hits and hits[-1].start == at else at + 1


def keys(text: str, names: Names) -> list[str]:
    """The members mentioned in `text`, deduped, in reading order."""
    found: list[str] = []
    for hit in find(text, names):
        if hit.key not in found:
            found.append(hit.key)
    return found


def rewrite(text: str, names: Names, replace: Callable[[Hit], str]) -> str:
    """`text` with every mention run through `replace`.

    Rebuilt from the slices between the hits rather than by repeated
    replacement, so a member whose name also appears as an ordinary word only
    loses the mention.
    """
    hits = find(text, names)
    if not hits:
        return text
    pieces: list[str] = []
    cursor = 0
    for hit in hits:
        pieces.append(text[cursor : hit.start])
        pieces.append(replace(hit))
        cursor = hit.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def unknown(text: str, names: Names) -> list[str]:
    """`@word`s that name nobody in the cast, deduped, in reading order.

    These are not errors — a sign that says `sales@`, an address in a note, a
    member not added yet — so they compose exactly as written. They are
    reported so a misspelt mention is noticed while it can still be fixed,
    rather than reaching the model as the literal text `@Mars`.
    """
    covered = [(hit.start, hit.end) for hit in find(text, names)]
    found: list[str] = []
    for match in _LOOSE.finditer(text or ""):
        if any(start <= match.start() < end for start, end in covered):
            continue
        if match.group(1) not in found:
            found.append(match.group(1))
    return found


# -- where a mention may live ---------------------------------------------


def fields(prompt: VideoPrompt) -> Iterator[Field]:
    """Every prose field of `prompt` a mention is honoured in.

    Two authored fields are deliberately left out. `Line.text` is the verbatim
    words a character speaks, and people say each other's *names* out loud —
    turning "@Mara, wait!" into "<Subject 2>, wait!" would put a routing label
    in somebody's mouth. `screen_text` is the same problem painted on a sign.
    Both keep an `@` exactly as typed.
    """

    def block(name: str, label: str) -> Field:
        return Field(
            name,
            label,
            getattr(prompt, name),
            lambda value, name=name: setattr(prompt, name, value),
        )

    yield block("style", "Style contract")
    yield block("overview", "Scene overview")
    yield block("camera", "Camera")
    yield block("audio", "Soundscape")
    yield block("music", "Non-diegetic music")
    yield block("ref_summary", "Reference summary")

    # Beats are authored in list order and composed in time order; the address
    # points at the row being edited, the label names the shot it reads as —
    # the same split `provenance` makes, for the same reason.
    order = {
        id(beat): number
        for number, beat in enumerate(prompt.storyboard.ordered_beats(), start=1)
    }
    for index, beat in enumerate(prompt.storyboard.beats):
        shot = f"Shot {order.get(id(beat), index + 1)}"
        if isinstance(beat.camera_path, dict) and isinstance(beat.camera_path.get("target"), str):
            yield Field(f"beat.{index}.camera", f"{shot} · Camera target", beat.camera_path["target"],
                        lambda value, beat=beat: beat.camera_path.__setitem__("target", value))
        for name, title in (("angle", "Angle"), ("action", "Action"), ("camera", "Camera")):
            yield Field(
                f"beat.{index}.{name}",
                f"{shot} · {title}",
                getattr(beat, name),
                lambda value, beat=beat, name=name: setattr(beat, name, value),
            )
        for position, line in enumerate(beat.lines):
            where = f"{shot} · Line {position + 1}"
            for name, title in (("describe", "Describe"), ("delivery", "Delivery")):
                yield Field(
                    f"line.{index}.{position}.{name}",
                    f"{where} · {title}",
                    getattr(line, name),
                    lambda value, line=line, name=name: setattr(line, name, value),
                )

    for index, subject in enumerate(prompt.subjects.items):
        yield Field(
            f"subject.{index}.role",
            f"{prompt.subjects.tag(subject)} role",
            subject.role,
            lambda value, subject=subject: setattr(subject, "role", value),
        )

    for index, item in enumerate(prompt.references.items):
        yield Field(
            f"ref.{index}.role",
            f"{prompt.references.tag(item)} role",
            item.role,
            lambda value, item=item: setattr(item, "role", value),
        )


def rename(prompt: VideoPrompt, old: str, new: str) -> int:
    """Point this prompt's mentions of `old` at `new`. Returns how many fields moved.

    The cost of storing a mention by name: the name *is* the link, so changing
    it has to reach the prose. Doing that here, on save, is what keeps a rename
    from quietly turning every `@Mara` into text that names nobody.
    """
    old, new = str(old or "").strip(), str(new or "").strip()
    if not old or not new or old == new:
        return 0
    names = [(old, "renamed")]
    moved = 0
    for field in fields(prompt):
        updated = rewrite(field.text, names, lambda hit: f"{MARK}{new}")
        if updated != field.text:
            field.write(updated)
            moved += 1
    return moved
