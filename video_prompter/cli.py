"""Command-line front end: lint a prompt, preview it, refit it.

Stands in for the editor's live preview until the UI layer exists, and stays
useful afterwards for checking prompts in bulk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import h3_spec
from .compose import compose
from .document import VideoPrompt
from .lint import Severity, lint
from .parse import parse
from .presets import KINDS, PresetError, PresetLibrary, bundled_root

_MARKS = {Severity.ERROR: "x", Severity.WARNING: "!", Severity.INFO: "-"}


def main(argv: list[str] | None = None) -> int:
    # Prompts are full of typographic punctuation; the Windows console is not
    # UTF-8 by default.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = _parser().parse_args(argv)
    library = PresetLibrary(*(args.preset_root or []), bundled_root())

    if args.list_presets:
        _list_presets(library)
        return 0

    try:
        prompt = _load(args, library)
    except PresetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.duration is not None:
        prompt.storyboard = prompt.storyboard.refit(args.duration)
    if args.fix:
        prompt.storyboard = prompt.storyboard.make_contiguous()

    if args.ratio:
        prompt.set_ratio(args.ratio)
    if args.width:
        prompt.width = args.width
    if args.height:
        prompt.height = args.height

    rendered = compose(prompt)
    findings = lint(prompt)

    if not args.quiet:
        _report(prompt, rendered, findings)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    elif args.quiet:
        print(rendered)

    return 1 if any(f.severity is Severity.ERROR for f in findings) else 0


def _load(args: argparse.Namespace, library: PresetLibrary) -> VideoPrompt:
    """Start from a file, stdin, or nothing, then layer presets over the top."""
    if args.file:
        prompt = parse(Path(args.file).read_text(encoding="utf-8"), duration=args.duration)
    elif args.preset:
        # Presets alone are enough to build a prompt; do not wait on stdin.
        prompt = VideoPrompt.blank(args.duration or h3_spec.MAX_DURATION_S)
    else:
        prompt = parse(sys.stdin.read(), duration=args.duration)

    for spec in args.preset or []:
        kind, separator, name = spec.partition(":")
        if not separator:
            raise PresetError(f"Expected --preset KIND:NAME, got {spec!r}.")
        library.apply(prompt, kind.strip(), name.strip())

    return prompt


def _list_presets(library: PresetLibrary) -> None:
    for kind in KINDS:
        names = library.list(kind)
        print(f"{kind} ({len(names)})")
        for name in names:
            print(f"  {name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_prompter",
        description="Lint and preview a MiniMax H3 prompt.",
    )
    parser.add_argument("file", nargs="?", help="prompt file; omit to read stdin")
    parser.add_argument(
        "-d", "--duration", type=int, help=f"refit onto an N-second budget "
        f"({h3_spec.MIN_DURATION_S}-{h3_spec.MAX_DURATION_S})"
    )
    parser.add_argument(
        "--fix", action="store_true", help="close gaps and overlaps in the storyboard"
    )
    parser.add_argument(
        "-p", "--preset", action="append", metavar="KIND:NAME",
        help="apply a preset; repeatable, applied in order",
    )
    parser.add_argument(
        "--preset-root", action="append", metavar="DIR",
        help="extra preset directory, searched before the bundled one; repeatable",
    )
    parser.add_argument("--list-presets", action="store_true", help="list presets and exit")
    parser.add_argument("-o", "--output", help="write the composed prompt here")
    parser.add_argument("--ratio", choices=h3_spec.ASPECT_RATIOS, help="size the canvas")
    parser.add_argument("--width", type=int, help="canvas width, multiple of 32")
    parser.add_argument("--height", type=int, help="canvas height, multiple of 32")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the prompt")
    return parser


def _report(prompt, rendered: str, findings: list) -> None:
    storyboard = prompt.storyboard
    print(
        f"{h3_spec.MODEL_NAME} local  {prompt.frames} frames "
        f"({prompt.rendered_seconds}s @ {h3_spec.FPS}fps)  {prompt.width}x{prompt.height}"
    )
    print(
        f"{len(storyboard.beats)} shots, {len(storyboard.cues)} audio cues, "
        f"{len(rendered)} chars"
    )

    print()
    if findings:
        for finding in findings:
            where = f"  {finding.where}" if finding.where else ""
            print(f"  {_MARKS[finding.severity]} {finding.rule}: {finding.message}{where}")
    else:
        print("  clean")

    print()
    print("-" * 72)
    print(rendered)
