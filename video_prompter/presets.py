"""The preset library: reusable blocks stored as plain text files.

Layout is `<root>/<kind>/<name>.txt`, one file per preset, so the library stays
hand-editable and diffs cleanly in git. Several roots can be stacked — a
project-local one over a shared one — which is how this library and the image
tool's can share the blocks that transfer between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .document import VideoPrompt
from .storyboard import Beat

KINDS = ("style", "camera", "audio", "negatives", "storyboard")

_UNSAFE = re.compile(r"[^\w \-]", re.UNICODE)


class PresetError(ValueError):
    """Raised for an unknown kind, a bad name, or a preset that isn't there."""


def sanitize_name(name: str) -> str:
    """Reduce a preset name to something that cannot escape its directory.

    Spaces survive — preset names are read by people — but separators, dots and
    anything exotic do not.
    """
    cleaned = _UNSAFE.sub("", name or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise PresetError(f"Not a usable preset name: {name!r}")
    return cleaned


@dataclass(frozen=True)
class Preset:
    kind: str
    name: str
    text: str


class PresetLibrary:
    """Presets across one or more roots; the first root is the writable one."""

    def __init__(self, *roots: Path | str) -> None:
        if not roots:
            raise PresetError("A preset library needs at least one root.")
        self.roots = [Path(root) for root in roots]

    @property
    def primary(self) -> Path:
        return self.roots[0]

    def list(self, kind: str) -> list[str]:
        """Every preset name of `kind` across all roots, deduped and sorted."""
        _check_kind(kind)
        names = {
            path.stem
            for root in self.roots
            for path in sorted((root / kind).glob("*.txt"))
            if path.is_file()
        }
        return sorted(names, key=str.casefold)

    def read(self, kind: str, name: str) -> str:
        """Preset body. Earlier roots shadow later ones."""
        path = self.find(kind, name)
        if path is None:
            raise PresetError(f"No {kind} preset named {name!r}.")
        return path.read_text(encoding="utf-8").strip()

    def find(self, kind: str, name: str) -> Path | None:
        _check_kind(kind)
        safe = sanitize_name(name)
        for root in self.roots:
            candidate = root / kind / f"{safe}.txt"
            if candidate.is_file():
                return candidate
        return None

    def write(self, kind: str, name: str, text: str) -> Path:
        """Save into the primary root, creating the kind directory as needed."""
        _check_kind(kind)
        safe = sanitize_name(name)
        target = self.primary / kind / f"{safe}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.strip() + "\n", encoding="utf-8")
        return target

    def delete(self, kind: str, name: str) -> bool:
        """Remove from the primary root only; shared roots are never touched."""
        _check_kind(kind)
        target = self.primary / kind / f"{sanitize_name(name)}.txt"
        if not target.is_file():
            return False
        target.unlink()
        return True

    def all(self, kind: str) -> list[Preset]:
        return [Preset(kind, name, self.read(kind, name)) for name in self.list(kind)]

    def apply(self, prompt: VideoPrompt, kind: str, name: str) -> VideoPrompt:
        """Write a preset into `prompt` in place, and return it.

        Storyboard presets are refit onto whatever budget the prompt already
        has, so a pattern authored at 10s drops onto a 6s clip unchanged in
        shape.
        """
        text = self.read(kind, name)

        if kind == "style":
            prompt.style = text
        elif kind == "camera":
            prompt.camera = text
        elif kind == "audio":
            prompt.audio = text
        elif kind == "negatives":
            prompt.negatives = split_negatives(text)
        elif kind == "storyboard":
            _apply_storyboard(prompt, text)

        return prompt


def split_negatives(text: str) -> list[str]:
    """One clause per line, or comma-separated when it is a single line."""
    lines = [line.strip().rstrip(".") for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [clause.strip().rstrip(".") for clause in text.split(",") if clause.strip()]


def _apply_storyboard(prompt: VideoPrompt, text: str) -> None:
    beats = [beat for beat in map(Beat.parse, text.splitlines()) if beat is not None]
    if not beats:
        raise PresetError("Storyboard preset contains no timecoded shot lines.")

    budget = prompt.storyboard.duration
    prompt.storyboard.beats = beats
    prompt.storyboard = prompt.storyboard.refit(budget).make_contiguous()


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise PresetError(f"Unknown preset kind {kind!r}; expected one of {', '.join(KINDS)}.")


def bundled_root() -> Path:
    """The presets that ship with this package."""
    return Path(__file__).resolve().parent.parent / "presets"
