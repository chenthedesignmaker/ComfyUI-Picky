"""Where things live on disk.

Resolves storage under ComfyUI's user directory when running inside ComfyUI,
and falls back to a local directory otherwise so the package stays runnable and
testable on its own.
"""

from __future__ import annotations

import os
from pathlib import Path

from .presets import bundled_root

# Overrides everything. Set by tests, and useful for running several libraries
# side by side.
HOME_ENV_VAR = "VIDEO_PROMPTER_HOME"

# Where rendered files really are, when ComfyUI cannot be asked.
OUTPUT_ENV_VAR = "VIDEO_PROMPTER_OUTPUT_DIR"

# Where uploaded keyframe/reference media lands — ComfyUI's input dir, which its
# LoadImage/LoadVideo/LoadAudio nodes read from. Overridable for the devserver.
INPUT_ENV_VAR = "VIDEO_PROMPTER_INPUT_DIR"

PACKAGE_NAME = "video_prompter"

# The one file for things that belong to this install rather than to a project:
# API keys, chiefly. It sits beside the projects, not in the repo, so pulling an
# update never touches it and no key is ever a candidate for a commit.
SETTINGS_FILE = "settings.json"


def comfy_user_dir() -> Path | None:
    """ComfyUI's user directory, or None when we are not inside ComfyUI."""
    try:
        import folder_paths  # type: ignore[import-not-found]
    except ImportError:
        return None

    getter = getattr(folder_paths, "get_user_directory", None)
    if getter is None:
        return None
    try:
        return Path(getter())
    except Exception:
        return None


def storage_root() -> Path:
    """The directory holding projects, profiles and user presets."""
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        return Path(override)

    user_dir = comfy_user_dir()
    if user_dir is not None:
        return user_dir / PACKAGE_NAME

    # Standalone: keep state beside the package rather than in the repo root.
    return Path(__file__).resolve().parent.parent / "user_data"


def settings_path() -> Path:
    """Where install-level settings live. May not exist; that is not an error."""
    return storage_root() / SETTINGS_FILE


def projects_root() -> Path:
    return storage_root() / "projects"


def profiles_root() -> Path:
    return storage_root() / "profiles"


def comfy_output_dir() -> Path | None:
    """ComfyUI's output directory — where takes' files really are.

    The override matters for more than tests: the editor can be pointed at a
    ComfyUI whose output directory it cannot introspect, and export needs the
    real path to hand ffmpeg.
    """
    override = os.environ.get(OUTPUT_ENV_VAR)
    if override:
        return Path(override)

    try:
        import folder_paths  # type: ignore[import-not-found]
    except ImportError:
        return None

    getter = getattr(folder_paths, "get_output_directory", None)
    if getter is None:
        return None
    try:
        return Path(getter())
    except Exception:
        return None


def comfy_input_dir() -> Path | None:
    """ComfyUI's input directory — where LoadImage/LoadVideo/LoadAudio look.

    Uploaded keyframes and references must land here to be nameable in a
    workflow. The override lets the GPU-free devserver keep its own input
    sandbox rather than writing into a real ComfyUI.
    """
    override = os.environ.get(INPUT_ENV_VAR)
    if override:
        return Path(override)

    try:
        import folder_paths  # type: ignore[import-not-found]
    except ImportError:
        return None

    getter = getattr(folder_paths, "get_input_directory", None)
    if getter is None:
        return None
    try:
        return Path(getter())
    except Exception:
        return None


def comfy_temp_dir() -> Path | None:
    """ComfyUI's temp directory — where preview nodes write.

    A take is nearly always an `output`, but a workflow that ended in a preview
    node records its file as a `temp`. Only "show me this file" needs to know;
    nothing else in here looks in temp.
    """
    try:
        import folder_paths  # type: ignore[import-not-found]
    except ImportError:
        return None

    getter = getattr(folder_paths, "get_temp_directory", None)
    if getter is None:
        return None
    try:
        return Path(getter())
    except Exception:
        return None


def exports_root() -> Path:
    return storage_root() / "exports"


def user_presets_root() -> Path:
    """The writable preset root. Shadows the bundled library."""
    return storage_root() / "presets"


def preset_roots() -> list[Path]:
    """Preset search order: the user's own, then anything shared, then bundled.

    A `shared` directory next to the user's presets is a convenient place to
    point at another tool's library — the image-side `comfy prompt node` keeps
    style and negative blocks that transfer unchanged.
    """
    roots = [user_presets_root()]
    shared = storage_root() / "shared_presets"
    if shared.is_dir():
        roots.append(shared)
    roots.append(bundled_root())
    return roots
