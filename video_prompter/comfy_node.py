"""The ComfyUI node.

Two jobs. It is the launcher for the editor, and it emits one of a project's
sources as the values an H3 graph needs: the composed prompt, and the frame count already
aligned to the model's 17k+5 grid.

That second job matters more than it looks. A hand-built graph usually carries
a seconds primitive feeding a maths expression to do the alignment; this node
does the same arithmetic from the same table the linter uses, so the number in
the graph and the number the editor reported are the same number.
"""

from __future__ import annotations

import logging

from . import h3_spec, paths
from .compose import compose
from .document import VideoPrompt
from .projects import ProjectStore

logger = logging.getLogger("ComfyUI.VideoPrompter")

NO_PROJECT = "(none)"


def _project_names() -> list[str]:
    try:
        return [NO_PROJECT, *ProjectStore(paths.projects_root()).list()]
    except Exception as error:  # pragma: no cover - defensive, listed at import
        logger.warning(f"Could not list video prompter projects: {error}")
        return [NO_PROJECT]


class VideoPromptNode:
    """Emits a source's prompt text and geometry for a MiniMax H3 graph.

    The `shot` widget still carries that name: it is a saved value inside every
    workflow anyone has built, and renaming it would silently blank them all.
    What it holds is a source id or name.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": (_project_names(),),
                "shot": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                # Overrides the stored budget without editing the project, for
                # sweeping a source across lengths.
                "duration_override": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": h3_spec.MAX_DURATION_S,
                        "tooltip": (
                            f"0 uses the segment's own budget. Otherwise "
                            f"{h3_spec.MIN_DURATION_S}-{h3_spec.MAX_DURATION_S} seconds."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("prompt", "length", "width", "height", "seconds")
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Composes a MiniMax H3 prompt from a video prompter project and emits the "
        "frame count aligned to the model's 17k+5 grid."
    )

    def build(self, project: str, shot: str, duration_override: int = 0):
        prompt = self._load(project, shot)

        if duration_override:
            prompt.storyboard = prompt.storyboard.refit(duration_override)

        return (
            compose(prompt),
            prompt.frames,
            prompt.width,
            prompt.height,
            prompt.rendered_seconds,
        )

    @classmethod
    def IS_CHANGED(cls, project: str, shot: str, duration_override: int = 0):
        """Re-run whenever the stored project changes on disk."""
        if project in ("", NO_PROJECT):
            return f"{project}|{shot}|{duration_override}"
        try:
            stamp = (ProjectStore(paths.projects_root()).path_for(project) / "project.json").stat()
            return f"{project}|{shot}|{duration_override}|{stamp.st_mtime_ns}"
        except OSError:
            return f"{project}|{shot}|{duration_override}|missing"

    @staticmethod
    def _load(project: str, shot: str):
        """Find a source by id or by name; blank when no project is selected.

        Ids are accepted first because they survive a rename, but a graph
        author typing one in by hand would be unreasonable — so a name works
        too, and the first source is the default.

        The prompt handed back is depth 0's. A later segment is a continuation,
        which is rendered through the extend graph rather than this node.
        """
        if project in ("", NO_PROJECT):
            return VideoPrompt.blank(h3_spec.TRAINED_MIN_DURATION_S)

        store = ProjectStore(paths.projects_root())
        if not store.exists(project):
            raise ValueError(f"No video prompter project named {project!r}.")

        loaded = store.load(project)
        if not loaded.sources:
            raise ValueError(f"Project {project!r} has no sources.")

        wanted = shot.strip()
        if not wanted:
            return loaded.sources[0].segments[0].prompt

        found = loaded.source(wanted) or loaded.source_named(wanted)
        if found is None:
            available = ", ".join(entry.name for entry in loaded.sources)
            raise ValueError(f"Project {project!r} has no source {wanted!r}. Have: {available}.")
        return found.segments[0].prompt
