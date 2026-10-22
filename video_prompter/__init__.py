"""Structured prompt authoring for MiniMax H3 (Hailuo 3.0).

The core is deliberately free of any ComfyUI dependency so it can be tested and
driven on its own; the node and editor layers sit on top.
"""

from __future__ import annotations

from . import h3_spec
from .compose import compose
from .cast import Cast, Member
from .document import Reference, ReferenceSet, Subject, SubjectSet, VideoPrompt
from .lint import Finding, Severity, errors, lint
from .parse import parse
from .presets import Preset, PresetError, PresetLibrary, bundled_root
from .profiles import (
    Binding,
    ProfileError,
    ProfileStore,
    RenderProfile,
    StaleProfileError,
    capture,
    discover_bindings,
    inject,
)
from .projects import Defaults, Project, ProjectError, ProjectStore
from .sequence import (
    AudioClip,
    Clip,
    Imported,
    Lifted,
    Mute,
    Sound,
    Timeline,
    new_audio_id,
    new_clip_id,
    new_mute_id,
)
from .sources import Revision, Segment, Source, blank_source, new_source_id
from .serialize import prompt_from_dict, prompt_to_dict
from .storyboard import AudioCue, Beat, Line, Storyboard

__all__ = [
    "AudioCue",
    "Beat",
    "Binding",
    "AudioClip",
    "Clip",
    "Imported",
    "Lifted",
    "Mute",
    "Line",
    "Defaults",
    "Finding",
    "Preset",
    "PresetError",
    "PresetLibrary",
    "ProfileError",
    "ProfileStore",
    "Project",
    "ProjectError",
    "ProjectStore",
    "Cast",
    "Member",
    "Reference",
    "ReferenceSet",
    "Subject",
    "SubjectSet",
    "Revision",
    "Segment",
    "RenderProfile",
    "Severity",
    "Sound",
    "Source",
    "StaleProfileError",
    "Storyboard",
    "Timeline",
    "VideoPrompt",
    "blank_source",
    "bundled_root",
    "capture",
    "compose",
    "discover_bindings",
    "errors",
    "h3_spec",
    "inject",
    "lint",
    "new_audio_id",
    "new_clip_id",
    "new_mute_id",
    "new_source_id",
    "parse",
    "prompt_from_dict",
    "prompt_to_dict",
]
