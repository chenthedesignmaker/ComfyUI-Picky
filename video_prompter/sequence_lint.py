"""Checks that only exist once a source has takes and a clip has a place in an edit.

`lint` grades one prompt against the model's limits. Nothing there can see that a
clip was trimmed past the end of what actually rendered, or that the take a
source stands on was made from a prompt that has since been rewritten — those
need the take record and the timeline, which is why they live apart.

The check this module gained with sources is the honest one: a path is sound when
each take's recorded parent is the take before it. When it is not, the finding
says the link is broken instead of quietly substituting a take that looks about
right, which is what the old pairing heuristics did.

Findings reuse `lint.Finding`, so the editor renders both lists the same way.
"""

from __future__ import annotations

from . import edit, h3_spec
from .compose import compose
from .edit import Bin
from .gallery import Take
from .lint import Finding, Severity
from .projects import Project
from .sequence import Clip
from .sources import Source

# Below this there is nothing left to see.
MIN_TIMELINE_S = 0.2


def lint_sequence(project: Project, takes: list[Take]) -> list[Finding]:
    """Every timeline-level finding for `project`, worst first."""
    bin = edit.read(project.sources, takes)

    counts: dict[str, int] = {}
    for take in takes:
        if not take.discarded:
            counts[take.shot] = counts.get(take.shot, 0) + 1

    findings: list[Finding] = []
    if not project.clips:
        findings.append(
            Finding(
                "sequence.empty",
                Severity.INFO,
                "Nothing on the timeline yet. Add a clip to start the edit.",
            )
        )

    used: set[str] = set()
    for index, clip in enumerate(project.clips, start=1):
        source = project.source_of(clip)
        if source is None:
            findings.append(
                Finding(
                    "clip.source_missing",
                    Severity.ERROR,
                    f"Clip {index} points at a source that is gone "
                    f"({clip.source_id}). Delete it, or restore the source.",
                    f"{index}. {clip.label or 'clip'}",
                )
            )
            continue
        used.add(source.id)
        findings += _check_clip(clip, source, index, bin, counts.get(source.id, 0))

    for source in project.sources:
        if source.id not in used:
            findings.append(
                Finding(
                    "source.unused",
                    Severity.INFO,
                    f"{source.name} is in the bin but not on the timeline.",
                    source.name,
                )
            )
        findings += _check_segments(source)

    return sorted(findings, key=lambda finding: -finding.severity)


def _check_clip(
    clip: Clip, source: Source, index: int, bin: Bin, take_count: int
) -> list[Finding]:
    name = clip.label or source.name
    where = f"{index}. {name}"
    found: list[Finding] = []

    path = clip.path(bin.path_of(source))
    on_path = bin.takes_of(path)

    for problem in bin.problems_of(path):
        found.append(Finding("clip.path_broken", Severity.ERROR, f"{name}: {problem}", where))

    for depth, take in enumerate(on_path):
        segment = source.segment_at(depth)
        if segment is None:
            continue
        if take.prompt_text and take.prompt_text != compose(segment.prompt):
            found.append(
                Finding(
                    "source.take_stale",
                    Severity.WARNING,
                    f"{name}'s take at depth {depth + 1} was rendered from an older "
                    f"version of that segment's prompt. Re-render if the edits were "
                    f"meant to land.",
                    where,
                )
            )

    available = edit.source_seconds(source, on_path)
    if clip.source_in >= available:
        found.append(
            Finding(
                "clip.trim_past_source",
                Severity.ERROR,
                f"{name} starts at {clip.source_in}s but the lineage only runs "
                f"{available}s, so nothing plays.",
                where,
            )
        )
    elif clip.source_out is not None and clip.source_out > available + 1e-6:
        found.append(
            Finding(
                "clip.trim_past_source",
                Severity.WARNING,
                f"{name} is trimmed out at {clip.source_out}s but the lineage is "
                f"{available}s. The tail is silently clipped.",
                where,
            )
        )

    length = clip.timeline_seconds(available)
    if clip.enabled and length < MIN_TIMELINE_S:
        found.append(
            Finding("clip.too_short", Severity.WARNING,
                    f"{name} occupies {length}s of the timeline.", where)
        )

    if take_count == 0 and not on_path:
        found.append(
            Finding(
                "source.no_takes",
                Severity.INFO,
                f"{name} has not been rendered yet; it is holding {length}s from "
                f"its segments' budget.",
                where,
            )
        )
    elif not source.path and clip.path_override is None:
        # The source has takes but nobody has said which chain is the one, so
        # `path_of` picked a default. It plays, and it is still a guess.
        found.append(
            Finding(
                "source.no_path",
                Severity.WARNING,
                f"{name} has {take_count} take(s) but no chosen chain, so the edit "
                f"is running on a best-guess default.",
                where,
            )
        )
    elif len(on_path) < len(source.segments):
        # A half-rendered chain is normal mid-work, so this is information, not
        # a complaint: the clip is holding the unrendered tail's budget.
        found.append(
            Finding(
                "source.path_short",
                Severity.INFO,
                f"{name} plans {len(source.segments)} generations but only "
                f"{len(on_path)} has been rendered and chosen.",
                where,
            )
        )

    return found


def _check_segments(source: Source) -> list[Finding]:
    found: list[Finding] = []
    for depth, segment in enumerate(source.segments):
        duration = segment.prompt.storyboard.duration
        if not h3_spec.is_valid_duration(duration):
            found.append(
                Finding(
                    "segment.duration_invalid",
                    Severity.ERROR,
                    f"{source.name} segment {depth + 1} asks for {duration}s; H3 "
                    f"generates {h3_spec.MIN_DURATION_S}-{h3_spec.MAX_DURATION_S}s "
                    f"in whole seconds. Generate longer and trim.",
                    source.name,
                )
            )
    return found
