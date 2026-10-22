"""Dicts in, dicts out. No HTTP, no ComfyUI.

Everything the editor can ask for lives here, so it can be tested directly and
the aiohttp layer above stays a thin translation of payloads and status codes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import (
    contract, edit, export, extend, fold, h3_spec, imagery, lineage, llm, mentions,
    migrate, parse, paths, quality, refine, render, selflift, t2i, twostage, video_loras,
)
from . import cast as cast_module
from .cast import Cast, Member
from .compose import compose
from .document import REFERENCE_KINDS, ReferenceSet, VideoPrompt
from .gallery import Gallery, Take, project_of, takes_from_history
from .image_prompts import ImagePromptStore
from .presets import KINDS, PresetError, PresetLibrary
from .profiles import (
    ProfileError,
    ProfileStore,
    RenderProfile,
    StaleProfileError,
    capture,
    validate,
)
from .projects import (
    ACTIVE, PROJECT_JSON, SHELVES, TRASHED,
    Defaults, Project, ProjectError, ProjectStore, utc_now_iso,
)
from .sequence import MAX_GAIN_DB, MAX_SPEED, MIN_GAIN_DB, MIN_SPEED, Clip, Mute
from .sequence_lint import lint_sequence
from .sources import Segment, Source, blank_source, split_off
from .stills import Still, StillStore, stills_from_history
from .serialize import prompt_from_dict, prompt_to_dict


class ApiError(Exception):
    """An error with an HTTP status already decided."""

    def __init__(self, status: int, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.message, **self.extra}


# `effective_continuation_of` and `get_child_takes_for_parent_take` used to live
# here. Both guessed which take continued which — one from shot names, one by
# pairing takes chronologically — and neither could say "I don't know", so a
# wrong answer was silent. A source's `path` is now the single record of what
# follows what, checked against the renderer's own `parent_take`, so there is
# nothing left for them to do.


class Api:
    """The editor's whole surface, over a set of stores."""

    def __init__(
        self,
        projects: ProjectStore | None = None,
        profiles: ProfileStore | None = None,
        presets: PresetLibrary | None = None,
        provider: llm.Provider | None = None,
    ) -> None:
        self.projects = projects or ProjectStore(paths.projects_root())
        self.profiles = profiles or ProfileStore(paths.profiles_root())
        self.presets = presets or PresetLibrary(*paths.preset_roots())
        self.gallery = Gallery(self.projects)
        self.stills = StillStore(self.projects)
        self.image_prompts = ImagePromptStore(self.projects)
        # Project name -> the (project.json, takes.json) mtimes that were last
        # found to have nothing left to fold. See `_fold_chains`.
        self._folded: dict[str, tuple[int, int]] = {}
        # Injected by tests. Otherwise left unset and read from the environment
        # at each use, so nothing here caches a key that was never there.
        self._provider = provider

    # -- contract ---------------------------------------------------------

    def describe_schema(self) -> dict[str, Any]:
        """The machine-readable contract an agent reads to drive the tool.

        The prompt document, the model's hard limits, the lint catalogue and
        the authoring loop, so any agent can self-orient without a human
        pasting the field list.
        """
        return contract.schema()

    # -- projects ---------------------------------------------------------

    def list_projects(self) -> dict[str, Any]:
        """Active project names, plus a row per project on every shelf.

        `projects` stays exactly what it always was — the active names, which
        is what the picker and every script already read. `entries` is the
        richer listing the project manager needs: shelf, dates and the lineage
        that gathers a version family into one chain.
        """
        return {
            "projects": self.projects.list(),
            "entries": [entry.to_dict() for entry in self.projects.entries()],
        }

    def shelve_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Move a project between active, archived and trashed.

        One verb for all six moves, restoring included: putting something back
        is just asking for `active`. Nothing is deleted here, so every one of
        them is undone by asking for the shelf it came from.
        """
        name = _required(payload, "name")
        shelf = _required(payload, "shelf")
        if shelf not in SHELVES:
            raise ApiError(400, f"shelf must be one of {', '.join(SHELVES)}.")
        with _translated():
            if self.projects.shelf_of(name) is None:
                raise ApiError(404, f"No project named {name!r}.")
            landed = self.projects.shelve(name, shelf)
        return {"name": landed, "shelf": shelf, **self.list_projects()}

    def purge_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Erase one trashed project for good. Refuses anything not trashed.

        What goes is the project's own JSON and its backups. Rendered videos
        live in ComfyUI's output directory and are only pointed at from
        `takes.json`, so no delete in this tool can reach them.
        """
        name = _required(payload, "name")
        with _translated():
            shelf = self.projects.shelf_of(name)
            if shelf is None:
                raise ApiError(404, f"No project named {name!r}.")
            if shelf != TRASHED:
                raise ApiError(409, f"{name!r} is not in the trash. Trash it first.")
            purged = self.projects.purge(name)
        return {"purged": purged, **self.list_projects()}

    def empty_trash(self) -> dict[str, Any]:
        with _translated():
            gone = self.projects.empty_trash()
        return {"purged": gone, **self.list_projects()}

    def duplicate_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Copy a project into the next version of the same family.

        Without a `new_name` the copy is named for where it sits in the family
        — `PLANET OFFICE` duplicates to `PLANET OFFICE v2` — and it inherits
        the lineage id that makes the two of them a chain a picker can show in
        order. Its takes and stills come along, so it opens with the footage
        the original had rather than an empty gallery.
        """
        name = _required(payload, "name")
        wanted = str(payload.get("new_name") or "").strip() or None
        with _translated():
            if self.projects.shelf_of(name) is None:
                raise ApiError(404, f"No project named {name!r}.")
            copy = self.projects.duplicate(name, wanted)
        return {"project": copy.to_dict(history=False), **self.list_projects()}

    def create_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _required(payload, "name")
        # Active only: a name sitting in the trash is free to reuse, and the
        # store keeps the two apart.
        if self.projects.exists(name, ACTIVE):
            raise ApiError(409, f"Project {name!r} already exists.")
        entries = payload.get("sources") or payload.get("shots") or []
        sources = [_source(entry) for entry in entries]
        with _translated():
            return {"project": self.projects.create(name, sources).to_dict(history=False)}

    def load_project(self, name: str) -> dict[str, Any]:
        """Open a project, folding any pre-sources chain on the way in.

        The fold is the one repair that runs by itself, because it is not a
        guess: the take record says which source continues which. It happens at
        open rather than at read because it needs the takes, which
        `Project.from_dict` has no access to — and it is idempotent, so a folded
        project simply finds nothing on every open after the first.
        """
        with _translated():
            if not self.projects.exists(name):
                raise ApiError(404, f"No project named {name!r}.")
            folded = self._fold_chains(name)
            loaded = {"project": self.projects.load(name).to_dict(history=False)}
        return loaded if folded is None else {**loaded, "folded": folded}

    def _fold_chains(self, name: str) -> dict[str, Any] | None:
        """Fold sibling sources back into one, once. None when there is nothing.

        Detection runs twice: cheaply outside the lock to decide whether a save
        is warranted at all, then again inside it against the project as it
        actually is, so a concurrent edit cannot be folded on top of a stale
        plan. The takes are the same read both times.

        The editor reloads a project on every sequence refresh, and detection
        costs a take read plus a forest — a tenth of a second on a 300-take
        project, which is not worth paying to be told "nothing" again. A clean
        answer is remembered against the two files it was read from, so it is
        thrown away the moment either changes rather than trusted onwards.
        """
        stamp = self._fold_stamp(name)
        if stamp is not None and self._folded.get(name) == stamp:
            return None

        loaded = self.projects.load(name)
        takes = self.gallery.list(name)
        if len(loaded.sources) < 2 or not fold.plan(loaded, edit.read(loaded.sources, takes)):
            if stamp is not None:
                self._folded[name] = stamp
            return None

        done: fold.Result | None = None

        def mutate(project: Project) -> None:
            nonlocal done
            done = fold.fold(project, edit.read(project.sources, takes))

        self.projects.update(name, mutate)
        if done is None or not done:
            return None
        # After the project is safely written: a take pointing at a source that
        # is gone is recoverable, a project pointing at nothing is not.
        self.gallery.reassign(name, done.reassign)
        return done.to_dict()

    def _fold_stamp(self, name: str) -> tuple[int, int] | None:
        """The mtimes a fold answer depends on, or None when they can't be read."""
        try:
            project = (self.projects.path_for(name) / PROJECT_JSON).stat().st_mtime_ns
        except OSError:
            return None
        takes_file = self.gallery.path_for(name)
        takes = takes_file.stat().st_mtime_ns if takes_file.is_file() else 0
        return (project, takes)

    def update_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _required(payload, "name")
        data = payload.get("project")
        if not isinstance(data, dict):
            raise ApiError(400, "update needs a `project` object.")
        note = str(payload.get("note") or "")

        with _translated():
            if not self.projects.exists(name):
                raise ApiError(404, f"No project named {name!r}.")

            def overwrite(project: Project) -> None:
                try:
                    incoming = Project.from_dict(data)
                except ValueError as error:
                    raise ApiError(400, str(error)) from error
                kept = {source.id: source for source in project.sources}
                for source in incoming.sources:
                    was = kept.get(source.id)
                    if was is not None:
                        _carry_revisions(was, source, note)
                project.sources = incoming.sources
                project.timeline = incoming.timeline
                project.defaults = incoming.defaults
                # Older callers omit cast; full documents replace it, including
                # an explicit empty list. Do not erase it for legacy updates.
                if "cast" in data:
                    project.cast = incoming.cast
                project.settings = incoming.settings
                project.render_profile = incoming.render_profile

            return {"project": self.projects.update(name, overwrite).to_dict(history=False)}

    def delete_project(self, name: str) -> dict[str, Any]:
        with _translated():
            return {"deleted": self.projects.delete(name)}

    def project_backups(self, name: str) -> dict[str, Any]:
        with _translated():
            return {
                "backups": [
                    {
                        "tier": backup.tier,
                        "file": backup.path.name,
                        "modified": backup.modified,
                        "size": backup.size,
                    }
                    for backup in self.projects.list_backups(name)
                ]
            }

    def restore_backup(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _required(payload, "name")
        filename = _required(payload, "file")

        with _translated():
            match = next(
                (b for b in self.projects.list_backups(name) if b.path.name == filename),
                None,
            )
            if match is None:
                raise ApiError(404, f"No backup named {filename!r} for {name!r}.")
            return {"project": self.projects.restore(name, match.path).to_dict(history=False)}

    # -- sources (the bin) -------------------------------------------------

    def get_source(self, project: str, source_id: str) -> dict[str, Any]:
        """One source, its segments, and the prompt of the first of them.

        A convenience over `load_project`: an agent revising one thing does not
        have to fish it out of the whole bin. `prompt` is depth 0's, which is
        the one an un-extended source has.
        """
        with _translated():
            source = self._source_of(project, source_id)
            data = source.to_dict(history=False)
            segments = data.get("segments") or [{}]
            return {"source": data, "prompt": segments[0].get("prompt", {})}

    def add_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A new source in the bin, seeded from the project's defaults.

        A clip is placed on the timeline for it unless `clip` is false, because
        "add a shot" has always meant both, and a bin item nobody can see is a
        confusing thing to have just created.
        """
        name = _required(payload, "project")
        at = payload.get("at")
        duration = payload.get("duration")
        with_clip = payload.get("clip", True)

        with _translated():

            def mutate(project: Project) -> None:
                label = payload.get("name") or f"Source {len(project.sources) + 1}"
                source = blank_source(
                    str(label),
                    int(duration)
                    if isinstance(duration, (int, float))
                    else h3_spec.TRAINED_MIN_DURATION_S,
                )
                project.defaults.apply_to(source.segments[0])
                project.add_source(source)
                if with_clip:
                    project.timeline.add(
                        Clip(source_id=source.id),
                        None if at is None else int(at),
                    )

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def update_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Replace one source in place, leaving the rest of the bin alone.

        Saving a whole project works too, but an editor that round-trips the
        entire bin on every keystroke turns two tabs into a race.
        """
        name = _required(payload, "project")
        data = payload.get("source")
        if not isinstance(data, dict):
            raise ApiError(400, "update needs a `source` object.")
        incoming = _source(data)
        # An optional word about where this save came from — "Gemini: faster",
        # "restored" — shown against the revision it files.
        note = str(payload.get("note") or "")

        with _translated():

            def mutate(project: Project) -> None:
                index = next(
                    (i for i, s in enumerate(project.sources) if s.id == incoming.id),
                    None,
                )
                if index is None:
                    raise ApiError(404, f"No source {incoming.id!r} in {name!r}.")
                _carry_revisions(project.sources[index], incoming, note)
                project.sources[index] = incoming

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def delete_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Remove a source and every clip that used it.

        The takes are untouched: deleting a bin item is not deleting renders.
        """
        name = _required(payload, "project")
        source_id = _required(payload, "source")

        with _translated():

            def mutate(project: Project) -> None:
                if not project.remove_source(source_id):
                    raise ApiError(404, f"No source {source_id!r} in {name!r}.")

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def duplicate_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Copy a source's intent as a fresh bin item, with no takes of its own.

        Segments and prompts come across; the path does not, because the copy
        has not rendered anything. A clip is placed for it, same as a new source.
        """
        name = _required(payload, "project")
        source_id = _required(payload, "source")

        with _translated():

            def mutate(project: Project) -> None:
                original = project.source(source_id)
                if original is None:
                    raise ApiError(404, f"No source {source_id!r} in {name!r}.")
                copy = original.copy(name=_next_name(original.name))
                index = project.sources.index(original)
                project.sources.insert(index + 1, copy)
                project.timeline.add(Clip(source_id=copy.id))

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def split_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Break some of a segment's shots out as a bin item of their own.

        Body {project, source, beats, segment?, name?, clip?}. `beats` are
        0-based positions in that segment's storyboard — the same numbering a
        `beat:N` rewrite scope uses — and the new source keeps each picked
        shot's length, laid end to end from zero, so the timing survives the
        move. Everything that describes the scene rather than one moment in it
        comes across whole; only the keyframes are dropped when the pick no
        longer opens or closes where they pointed.

        Nothing is rendered and no take is carried: what comes back is intent,
        ready to generate on its own. The original keeps its shots — this is a
        copy, and deleting the ones it replaces is a separate decision.

        The clip lands next to the original's last one rather than at the end
        of the timeline, because a split-off shot belongs beside the thing it
        came out of, which is where it will be cut in.
        """
        name = _required(payload, "project")
        source_id = _required(payload, "source")
        beats = payload.get("beats")
        if not isinstance(beats, list) or not beats:
            raise ApiError(400, "`beats` must be a non-empty list of shot indices.")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in beats):
            raise ApiError(400, "Each entry in `beats` must be a whole number.")
        with_clip = payload.get("clip", True)
        made: dict[str, str] = {}

        with _translated():

            def mutate(project: Project) -> None:
                original = project.source(source_id)
                if original is None:
                    raise ApiError(404, f"No source {source_id!r} in {name!r}.")
                segment = _segment_in(original, payload.get("segment"))
                count = len(segment.prompt.storyboard.beats)
                stray = sorted({x for x in beats if not 0 <= x < count})
                if stray:
                    raise ApiError(
                        400,
                        f"{segment.id} has {count} shot(s), so "
                        f"{', '.join(str(x) for x in stray)} is not one of them.",
                    )

                fresh = split_off(
                    original, segment, beats, name=str(payload.get("name") or "")
                )
                project.sources.insert(project.sources.index(original) + 1, fresh)
                made["id"] = fresh.id

                if with_clip:
                    uses = project.timeline.using(original.id)
                    at = (
                        project.timeline.index_of(uses[-1].id) + 1
                        if uses else None
                    )
                    project.timeline.add(Clip(source_id=fresh.id), at)

            project = self.projects.update(name, mutate)
            return {"project": project.to_dict(history=False), "source": made.get("id", "")}

    def set_source_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Choose the lineage a source plays, as one ordered list of take ids.

        The single replacement for `pick` and `branch_picks`. It is checked
        against what the renderer recorded before it is stored, so a path whose
        links do not hold up is refused here rather than discovered at export.
        An empty list clears the choice.
        """
        name = _required(payload, "project")
        source_id = _required(payload, "source")
        raw = payload.get("path")
        if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
            raise ApiError(400, "`path` must be a list of take ids.")
        path = [x for x in raw if x]

        with _translated():
            bin = self._bin(name)

        problems = bin.problems_of(path)
        if problems:
            raise ApiError(409, "That path does not hold up.", problems=problems)

        with _translated():

            def mutate(project: Project) -> None:
                source = project.source(source_id)
                if source is None:
                    raise ApiError(404, f"No source {source_id!r} in {name!r}.")
                source.set_path(path)

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def add_segment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Plan one more generation on the end of a source's chain.

        Intent only — nothing is rendered here. The segment's prompt is the
        continuation text an extend will be given; leaving it blank and filling
        it in afterwards is normal.
        """
        name = _required(payload, "project")
        source_id = _required(payload, "source")
        data = payload.get("prompt")
        quality_name = payload.get("quality")
        duration = payload.get("duration")

        with _translated():

            def mutate(project: Project) -> None:
                source = project.source(source_id)
                if source is None:
                    raise ApiError(404, f"No source {source_id!r} in {name!r}.")

                if isinstance(data, dict):
                    prompt = self._read_prompt(data)
                else:
                    # A continuation runs as long as the generation before
                    # it unless told otherwise — the same reasoning as the
                    # quality below, and it makes the budget on the form real
                    # rather than a number nobody chose.
                    prompt = VideoPrompt.blank(
                        int(duration) if isinstance(duration, (int, float))
                        else source.segments[-1].prompt.duration
                    )
                segment = Segment(prompt=prompt)
                # A continuation is rendered the way the segment before it was,
                # or one chain comes out of the export at two grades.
                segment.quality = str(quality_name or source.segments[-1].quality)
                project.defaults.apply_to(segment)
                source.add_segment(segment)

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    # -- writing with a language model --------------------------------------

    def _llm(self, scope: "llm.Scope | None" = None) -> llm.Provider:
        provider = self._provider or llm.provider_from_settings(scope)
        if provider is None:
            raise ApiError(
                503,
                "No model configured. For a local model, put an `llm_url` "
                "(e.g. http://127.0.0.1:11434/v1 for Ollama) in "
                f"{paths.settings_path()}; for Gemini, a `gemini_key`. Then "
                "restart ComfyUI.",
            )
        return provider

    def llm_status(self, probe: bool = False) -> dict[str, Any]:
        """Whether a model is reachable, and which. The buttons ask this first.

        `probe` also lists the models the key can see. It costs a round trip to
        Google, so it is opt-in: the editor wants the yes/no on every load and
        the list only when someone is choosing.
        """
        provider = self._provider or llm.provider_from_settings()
        # Where to put a key is the only useful thing to say when there is not
        # one, so it is said whether or not one was found.
        where = {
            "key_vars": list(llm.KEY_VARS),
            "settings_file": str(paths.settings_path()),
            "settings_key": llm.SETTINGS_KEY,
            "local_url_var": llm.LOCAL_URL_VAR,
            "local_settings_key": llm.SETTINGS_LOCAL_URL,
        }
        if provider is None:
            return {"configured": False, **where}

        status: dict[str, Any] = {
            "configured": True,
            "provider": getattr(provider, "name", "provider"),
            "model": provider.model or "the loaded local model",
            "whole_model": (provider.model if isinstance(provider, llm.OpenAICompatible)
                            else llm.configured().whole_model),
            "source": getattr(provider, "source", ""),
            "scopes": list(llm.AUTHORED_FIELDS),
            **where,
        }
        if probe and hasattr(provider, "models"):
            with _llm_errors():
                status["models"] = provider.models()
        return status

    def llm_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write or rewrite a prompt, and apply it when a segment was named.

        One operation serves both ways in: describing a whole scene is the same
        call as "make this faster" with a narrower scope and a prompt that
        already says something. Splitting them would mean two paths to keep in
        step for no gain — the difference is entirely in the scope and what is
        already there.

        Naming `project` and `source` applies the result and files a revision,
        so a rewrite is undoable rather than asked about. Leaving them off
        drafts without saving, which is what an agent comparing options wants.
        """
        instruction = _required(payload, "instruction")
        scope = payload.get("scope")
        project_name = str(payload.get("project") or "")
        source_id = str(payload.get("source") or "")
        segment_id = payload.get("segment")

        current = payload.get("prompt")
        if current is None and project_name and source_id:
            current = prompt_to_dict(
                _segment_in(self._source_of(project_name, source_id), segment_id).prompt
            )
        if not isinstance(current, dict):
            raise ApiError(
                400,
                "Send a `prompt` to work from, or a `project` and `source` to "
                "read one from.",
            )

        with _llm_errors():
            aim = llm.Scope.parse(scope)
            drafted = llm.draft(self._llm(aim), current, instruction, aim)

        result = drafted.to_dict()
        if not (project_name and source_id):
            return result

        note = f"{drafted.model}: {instruction.strip()}"[:200]
        with _translated():

            def mutate(project: Project) -> None:
                source = project.source(source_id)
                if source is None:
                    raise ApiError(404, f"No source {source_id!r} in {project_name!r}.")
                _segment_in(source, segment_id).revise(
                    self._read_prompt(drafted.prompt), note
                )

            result["project"] = self.projects.update(
                project_name, mutate
            ).to_dict(history=False)
        return result

    def llm_draft_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write or rewrite the image prompt, filing the outgoing one.

        Naming a `project` files what was in the box before the rewrite, which
        is what makes applying the answer straight away safe: an image prompt
        has no scope to merge within, so a rewrite replaces all of it, and the
        undo is the whole safety net. Leaving `project` off drafts without
        filing anything, for a caller comparing options.

        The reply carries `before` as well as `prompt`, so the editor can show
        what moved without having kept a copy of its own.
        """
        instruction = _required(payload, "instruction")
        current = str(payload.get("prompt") or "")
        style = str(payload.get("style") or "")
        project_name = str(payload.get("project") or "")
        target = str(payload.get("target") or "")

        with _llm_errors():
            drafted = llm.draft_image(self._llm(), current, instruction, style)

        result = drafted.to_dict()
        if project_name and drafted.changed:
            note = f"{drafted.model}: {instruction.strip()}"[:200]
            with _translated():
                filed = self.image_prompts.file(
                    project_name, drafted.before, note=note, target=target
                )
            result["filed"] = filed.to_dict() if filed else None
        return result

    # -- prompt history -----------------------------------------------------

    def image_prompt_revisions(self, project: str) -> dict[str, Any]:
        """Every past image prompt for a project, newest first."""
        with _translated():
            return {
                "revisions": [
                    revision.to_dict()
                    for revision in self.image_prompts.list(project)
                ]
            }

    def file_image_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Keep an image prompt as a past version.

        The editor calls this with the text it is about to replace — on a
        generate, and before a hand edit that would otherwise lose it. An empty
        box, or the same words the newest entry already holds, file nothing and
        answer `{"revision": null}`; a caller sweeping eight seeds from one
        prompt must not push eight identical entries.
        """
        project = _required(payload, "project")
        text = str(payload.get("text") or "")
        with _translated():
            filed = self.image_prompts.file(
                project,
                text,
                note=str(payload.get("note") or ""),
                target=str(payload.get("target") or ""),
            )
        return {"revision": filed.to_dict() if filed else None}

    def segment_revisions(
        self, project: str, source_id: str, segment_id: str | None = None
    ) -> dict[str, Any]:
        """One segment's past prompts, newest first, each with its composed text.

        The text comes along because that is what the editor diffs, and asking
        for fifty of them one at a time to fill a history strip would be fifty
        round trips. `lint` is deliberately not run: a revision is not being
        written, it is being looked at.
        """
        source = self._source_of(project, source_id)
        segment = _segment_in(source, segment_id)

        with _translated():
            return {
                "source": source.id,
                "segment": segment.id,
                "revisions": [
                    {**revision.to_dict(), "text": compose(revision.prompt)}
                    for revision in reversed(segment.revisions)
                ],
                "current": {
                    "prompt": prompt_to_dict(segment.prompt),
                    "text": compose(segment.prompt),
                },
            }

    def restore_revision(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Put one segment's past prompt back, filing the current one on the way.

        Scoped to the segment on purpose. Restoring a project backup to recover
        a paragraph rolls back the whole bin, which is why nobody does it.
        """
        name = _required(payload, "project")
        source_id = _required(payload, "source")
        revision_id = _required(payload, "revision")
        segment_id = payload.get("segment")

        with _translated():

            def mutate(project: Project) -> None:
                source = project.source(source_id)
                if source is None:
                    raise ApiError(404, f"No source {source_id!r} in {name!r}.")
                segment = _segment_in(source, segment_id)
                if segment.restore(str(revision_id)) is None:
                    raise ApiError(404, f"No revision {revision_id!r} on {segment.id!r}.")

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    # -- clips (the timeline) ----------------------------------------------

    def add_clip(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Put another use of a source on the timeline."""
        name = _required(payload, "project")
        source_id = _required(payload, "source")
        at = payload.get("at")

        with _translated():

            def mutate(project: Project) -> None:
                if project.source(source_id) is None:
                    raise ApiError(404, f"No source {source_id!r} in {name!r}.")
                project.timeline.add(
                    Clip(source_id=source_id), None if at is None else int(at)
                )

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def delete_clip(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Take a clip out of the edit. Its source stays in the bin."""
        name = _required(payload, "project")
        clip_id = _required(payload, "clip")

        with _translated():

            def mutate(project: Project) -> None:
                if not project.timeline.remove(clip_id):
                    raise ApiError(404, f"No clip {clip_id!r} in {name!r}.")

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def move_clip(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Reorder. Position is derived from order, so this *is* the edit."""
        name = _required(payload, "project")
        clip_id = _required(payload, "clip")
        to = _int(payload, "to")

        with _translated():

            def mutate(project: Project) -> None:
                try:
                    project.timeline.move(clip_id, to)
                except KeyError as error:
                    raise ApiError(404, f"No clip {clip_id!r} in {name!r}.") from error

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def edit_clip(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Trim, speed, label and enablement — clamped to the lineage's length.

        Kept apart from `update_source` because these are the ones with rules:
        an in point past the out, or a speed of zero, is a clip that silently
        plays nothing. Clamping here means the timeline UI can be careless.

        `pin` freezes this clip on the branch it is currently showing, so a
        later branch switch in the bin leaves it alone; `pin: false` puts it
        back to following its source.

        `audio` is an object — `{muted, gain_db, offset, take}`. `take` is the
        friendly form of borrowing: name the take you want to *hear* and the
        path to it is worked out here, so a caller never assembles one by hand.
        `take: null` gives the clip its own sound back.
        """
        name = _required(payload, "project")
        clip_id = _required(payload, "clip")

        with _translated():
            loaded = self.projects.load(name)
            bin = edit.read(loaded.sources, self.gallery.list(name))
            lengths = edit.lengths(loaded.sources, loaded.clips, bin)

            def mutate(project: Project) -> None:
                clip = project.clip(clip_id)
                if clip is None:
                    raise ApiError(404, f"No clip {clip_id!r} in {name!r}.")

                available = clip.source_length(lengths.get(clip_id))

                if "pin" in payload:
                    source = project.source_of(clip)
                    if not payload["pin"] or source is None:
                        clip.path_override = None
                    else:
                        # Pin to what the clip is actually showing, which is the
                        # source's *resolved* path — a source running on a
                        # default has an empty stored one, and pinning to that
                        # would quietly leave the clip following.
                        showing = clip.path(bin.path_of(source))
                        if not showing:
                            raise ApiError(
                                409,
                                "There is nothing to pin — this clip has no take yet.",
                            )
                        clip.path_override = showing
                if "source_in" in payload:
                    clip.source_in = _clamp(payload["source_in"], 0.0, available)
                if "source_out" in payload:
                    value = payload["source_out"]
                    clip.source_out = (
                        None if value is None else _clamp(value, 0.0, available)
                    )
                if "speed" in payload:
                    clip.speed = _clamp(payload["speed"], MIN_SPEED, MAX_SPEED)
                if "enabled" in payload:
                    clip.enabled = bool(payload["enabled"])
                if "label" in payload:
                    clip.label = str(payload["label"])
                if "audio" in payload:
                    _apply_sound(clip, payload["audio"], bin)

                # An inverted trim is always a slip, and leaves nothing to play.
                if clip.source_out is not None and clip.source_out <= clip.source_in:
                    clip.source_out = None

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    # -- mutes -------------------------------------------------------------

    def add_mute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Silence a stretch of one clip. `start`/`end` are lineage seconds.

        Lineage rather than ruler seconds because that is what a mute *is*: a
        decision about a moment of the footage. Trimming or moving the clip
        afterwards carries it, and the editor never has to re-place it.
        """
        name = _required(payload, "project")
        clip_id = _required(payload, "clip")
        start = _number(payload, "start")
        end = _number(payload, "end")

        with _translated():
            lengths = self._clip_lengths(name)
            made: dict[str, Any] = {}

            def mutate(project: Project) -> None:
                clip = project.clip(clip_id)
                if clip is None:
                    raise ApiError(404, f"No clip {clip_id!r} in {name!r}.")
                low, high = _mute_span(start, end, clip.source_length(lengths.get(clip_id)))
                mute = project.timeline.add_mute(
                    Mute(clip_id=clip_id, start=low, end=high)
                )
                made["id"] = mute.id

            result = self.projects.update(name, mutate).to_dict(history=False)
            return {"project": result, "mute": made.get("id", "")}

    def edit_mute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Move or resize a band. Both ends are optional; give what changed."""
        name = _required(payload, "project")
        mute_id = _required(payload, "mute")

        with _translated():
            lengths = self._clip_lengths(name)

            def mutate(project: Project) -> None:
                mute = project.timeline.mute(mute_id)
                if mute is None:
                    raise ApiError(404, f"No mute {mute_id!r} in {name!r}.")
                clip = project.clip(mute.clip_id)
                available = clip.source_length(lengths.get(clip.id)) if clip else 0.0
                mute.start, mute.end = _mute_span(
                    payload.get("start", mute.start),
                    payload.get("end", mute.end),
                    available,
                )

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def delete_mute(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _required(payload, "project")
        mute_id = _required(payload, "mute")

        with _translated():

            def mutate(project: Project) -> None:
                if not project.timeline.remove_mute(mute_id):
                    raise ApiError(404, f"No mute {mute_id!r} in {name!r}.")

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def split_clip(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Cut a clip in two at `at` seconds into its timeline length.

        Free: both halves are the same lineage seen through two windows, so a
        split spends no render and copies no prompt.
        """
        name = _required(payload, "project")
        clip_id = _required(payload, "clip")
        at = payload.get("at")
        if isinstance(at, bool) or not isinstance(at, (int, float)):
            raise ApiError(400, "`at` must be a number of seconds.")

        with _translated():
            lengths = self._clip_lengths(name)

            def mutate(project: Project) -> None:
                if project.clip(clip_id) is None:
                    raise ApiError(404, f"No clip {clip_id!r} in {name!r}.")
                try:
                    project.timeline.split(clip_id, float(at), lengths.get(clip_id))
                except ValueError as error:
                    raise ApiError(400, str(error)) from error

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def duplicate_clip(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Another use of the same lineage, trimmed the same way."""
        name = _required(payload, "project")
        clip_id = _required(payload, "clip")

        with _translated():

            def mutate(project: Project) -> None:
                try:
                    project.timeline.duplicate(clip_id)
                except KeyError as error:
                    raise ApiError(404, f"No clip {clip_id!r} in {name!r}.") from error

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def set_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Set the project's default blocks, optionally pushing them everywhere.

        Pushing overwrites every segment of every source: it is the deliberate
        "make them all match again" action, so it does not tiptoe around the
        ones that have diverged.
        """
        name = _required(payload, "project")
        data = payload.get("defaults")
        if not isinstance(data, dict):
            raise ApiError(400, "Expected a `defaults` object.")
        push = bool(payload.get("push"))

        with _translated():

            def mutate(project: Project) -> None:
                project.defaults = _defaults(data)
                if not push:
                    return
                for source in project.sources:
                    for segment in source.segments:
                        segment.prompt.style = project.defaults.style
                        segment.prompt.camera = project.defaults.camera
                        segment.prompt.negatives = list(project.defaults.negatives)

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    # -- cast -------------------------------------------------------------

    def _cast(self, payload: dict[str, Any]) -> Cast | None:
        """The project's cast, when the caller said which project they are in.

        Composing and linting work without one — the CLI and the bare ComfyUI
        node have no project — so a missing or unknown `project` is not an
        error here, it just means nothing to resolve. A caller that wanted the
        cast and mistyped the name gets a `cast.unknown` finding on every
        member instead of silence.
        """
        name = str(payload.get("project") or "").strip()
        if not name:
            return None
        try:
            if not self.projects.exists(name):
                return None
            return self.projects.load(name).cast
        except (ProjectError, OSError):
            return None

    def list_cast(self, project: str) -> dict[str, Any]:
        with _translated():
            return {"cast": self.projects.load(project).cast.to_dict()}

    def save_member(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create or overwrite one member. A missing `id` mints a new one.

        Whole-member writes, deliberately: a member is small, and a partial
        patch would need a merge rule for `references` that nobody would guess
        right. Send the member back the way it came, changed.

        A **rename** reaches further than the member. `@mentions` in the prose
        are stored by name, so changing the name has to move them too, in every
        segment of the project — otherwise a rename would silently turn every
        `@Mara` into text that names nobody, on prompts that are not on screen
        to notice. `renamed_mentions` reports how many fields moved.
        """
        name = _required(payload, "project")
        data = payload.get("member")
        if not isinstance(data, dict):
            raise ApiError(400, "Expected a `member` object.")

        member = Member.from_dict(data)
        if not member.name.strip():
            raise ApiError(400, "A cast member needs a name.")
        try:
            cast_module.check_kind(member.kind)
        except cast_module.CastError as error:
            raise ApiError(400, str(error)) from error

        moved: list[int] = []

        with _translated():

            def mutate(project: Project) -> None:
                existing = project.cast.get(member.id)
                if existing is None:
                    project.cast.add(member)
                    return
                position = project.cast.members.index(existing)
                project.cast.members[position] = member
                moved.append(_rename_mentions(project, existing.name, member.name))

            project = self.projects.update(name, mutate)
            return {
                "member": member.to_dict(),
                "renamed_mentions": sum(moved),
                "project": project.to_dict(history=False),
            }

    def delete_member(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Remove a member. Prompts naming it keep the id and start warning.

        The id is left on the prompts on purpose. Scrubbing it would silently
        drop a speaker from lines that still have words, whereas a dangling id
        raises `cast.unknown` and can be pointed at a different member — the
        author gets to decide which.
        """
        name = _required(payload, "project")
        member_id = _required(payload, "member")

        with _translated():
            removed: list[bool] = []

            def mutate(project: Project) -> None:
                removed.append(project.cast.remove(member_id))

            project = self.projects.update(name, mutate)
            if not removed or not removed[0]:
                raise ApiError(404, f"No cast member {member_id!r} in {name!r}.")
            return {
                "deleted": member_id,
                "used_by": _members_used(project, member_id),
                "project": project.to_dict(history=False),
            }

    def member_usage(self, project: str, member_id: str) -> dict[str, Any]:
        """Which segments name this member — what a delete would strand."""
        with _translated():
            return {"used_by": _members_used(self.projects.load(project), member_id)}

    # -- presets ----------------------------------------------------------

    def list_presets(self, kind: str | None = None) -> dict[str, Any]:
        with _translated():
            kinds = [kind] if kind else list(KINDS)
            return {"presets": {one: self.presets.list(one) for one in kinds}}

    def read_preset(self, kind: str, name: str) -> dict[str, Any]:
        with _translated():
            return {"kind": kind, "name": name, "text": self.presets.read(kind, name)}

    def save_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = _required(payload, "kind")
        name = _required(payload, "name")
        text = payload.get("text", "")
        with _translated():
            self.presets.write(kind, name, text)
            return {"saved": True, "kind": kind, "name": name}

    def delete_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = _required(payload, "kind")
        name = _required(payload, "name")
        with _translated():
            return {"deleted": self.presets.delete(kind, name)}

    # -- render profiles --------------------------------------------------

    def list_profiles(self) -> dict[str, Any]:
        return {"profiles": self.profiles.list()}

    def load_profile(self, name: str) -> dict[str, Any]:
        with _translated():
            profile = self.profiles.load(name)
            return {"profile": profile.to_dict(), "describes": profile.describe()}

    def save_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _required(payload, "name")
        api_format = payload.get("api_format")
        if not isinstance(api_format, dict) or not api_format:
            raise ApiError(400, "Capturing a profile needs an `api_format` workflow.")

        with _translated():
            profile = capture(
                name,
                api_format,
                canvas_workflow=payload.get("canvas_workflow"),
                bindings=None,
            )
            for role, targets in (payload.get("bindings") or {}).items():
                for entry in targets:
                    profile.bind(role, entry[0], entry[1])
            validate(profile)
            self.profiles.save(profile)
            return {"profile": profile.to_dict(), "describes": profile.describe()}

    def delete_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _required(payload, "name")
        with _translated():
            return {"deleted": self.profiles.delete(name)}

    # -- reference vocabulary ---------------------------------------------

    def reference_kinds(self) -> dict[str, Any]:
        """The slots `MiniMaxH3ReferenceToVideo` offers, and their limits.

        Served so the editor does not carry its own copy of the numbers.
        """
        labels = {
            "image": "Image",
            "video": "Video",
            "video_audio": "Video soundtrack",
            "audio": "Audio",
        }
        empty = ReferenceSet()
        return {
            "kinds": [
                {
                    "kind": kind,
                    "label": labels[kind],
                    "limit": empty.limit_for(kind),
                }
                for kind in REFERENCE_KINDS
            ]
        }

    # -- reference / keyframe uploads -------------------------------------

    # Extensions the H3 loaders can actually read, grouped so a keyframe cannot
    # be a video and a reference audio cannot be a picture.
    _MEDIA_EXTENSIONS: dict[str, set[str]] = {
        "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
        "video": {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"},
        "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"},
    }
    # A subfolder inside ComfyUI's input dir, so uploads stay out of the way of
    # whatever else lives there. The value returned to the client is relative to
    # the input root, which is what the loaders resolve.
    _UPLOAD_SUBFOLDER = "video_prompter"

    def _media_kind(self, name: str) -> str:
        ext = Path(name).suffix.lower()
        for kind, extensions in self._MEDIA_EXTENSIONS.items():
            if ext in extensions:
                return kind
        allowed = ", ".join(sorted(e for s in self._MEDIA_EXTENSIONS.values() for e in s))
        raise ApiError(400, f"Unsupported media type {ext or name!r}. Allowed: {allowed}.")

    def _input_target(self, name: str) -> tuple[Path, str]:
        """A writable, collision-free path in the input dir and its workflow ref.

        Reserves an empty file and returns (absolute path, workflow filename).
        Exclusive creation prevents concurrent downloads/uploads from choosing
        the same filename before either has written its bytes. The
        loaders take a path relative to the input root, so the subfolder is part
        of the returned reference.
        """
        input_dir = paths.comfy_input_dir()
        if input_dir is None:
            raise ApiError(
                503,
                "No ComfyUI input directory available to upload into. Set "
                "VIDEO_PROMPTER_INPUT_DIR when running outside ComfyUI.",
            )
        from .presets import sanitize_name

        stem = sanitize_name(Path(name).stem) or "reference"
        ext = Path(name).suffix.lower()
        folder = input_dir / self._UPLOAD_SUBFOLDER
        folder.mkdir(parents=True, exist_ok=True)

        candidate = f"{stem}{ext}"
        n = 1
        while True:
            try:
                with (folder / candidate).open("xb"):
                    pass
                return folder / candidate, f"{self._UPLOAD_SUBFOLDER}/{candidate}"
            except FileExistsError:
                candidate = f"{stem}_{n}{ext}"
                n += 1

    def save_reference(self, filename: str, data: bytes) -> dict[str, Any]:
        """Write uploaded bytes into the input dir; return its workflow ref.

        The route hands raw bytes here from a multipart part. The returned
        `filename` is what goes on a segment's `first_frame` / `last_frame` or a
        `references[].path`.
        """
        kind = self._media_kind(filename)
        if not data:
            raise ApiError(400, "Empty upload.")
        target, ref = self._input_target(filename)
        target.write_bytes(data)
        return {"filename": ref, "kind": kind, "bytes": len(data)}

    def fetch_reference(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Download a media URL server-side into the input dir.

        Only http/https, so the editor and an agent can attach an online image
        without a manual download. The name is taken from the URL path; the
        extension decides the media kind, same as an upload.
        """
        import urllib.parse
        import urllib.request

        url = _required(payload, "url").strip()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ApiError(400, "Only http(s) URLs can be fetched.")

        name = payload.get("filename") or Path(parsed.path).name or "reference"
        kind = self._media_kind(name)  # reject bad extensions before the request
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "video-prompter"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
        except Exception as error:
            raise ApiError(502, f"Could not fetch {url!r}: {error}") from error

        target, ref = self._input_target(name)
        target.write_bytes(data)
        return {"filename": ref, "kind": kind, "bytes": len(data), "source": url}

    # -- gallery ----------------------------------------------------------

    def list_takes(self, project: str, source: str | None = None) -> dict[str, Any]:
        """This project's takes, optionally only the ones a source rendered.

        A take records the source it was rendered for in `shot`, the field name
        it has carried since before sources existed. Renaming it would orphan
        every `takes.json` on disk to save one word.
        """
        with _translated():
            return {"takes": [take.to_dict() for take in self.gallery.list(project, source)]}

    def take_lineage(self, project: str, source: str | None = None) -> dict[str, Any]:
        """The continuation forest for a project, and a path to start from.

        Built only from what the renderer recorded — `take.parent_take` — so
        nothing here is inferred from names, ordering, or how many takes happen
        to sit either side of a link.

        `source` narrows the roots to the chains that touch that source. Chains
        are not clipped to it: a lineage that runs on past the source it started
        in is one lineage, and cutting it at that boundary is what made chains
        hard to see in the first place.
        """
        with _translated():
            takes = self.gallery.list(project)

        roots = lineage.build_forest(takes)
        if source:
            roots = [
                root for root in roots
                if any(node.take.shot == source for node in root.walk())
            ]

        index = lineage.index_forest(roots)
        path = lineage.default_path(roots)
        on_path = lineage.path_takes(path, index)

        return {
            "roots": lineage.tree_to_dict(roots, path),
            "path": path,
            "problems": lineage.validate(path, index),
            "seconds": lineage.lineage_seconds(on_path),
            "segments": [
                {
                    "take_id": take.id,
                    "depth": depth,
                    "filename": take.filename,
                    "shot": take.shot,
                    "start": start,
                    "end": end,
                    "seconds": round(end - start, 3),
                }
                for depth, (take, (start, end)) in enumerate(
                    zip(on_path, lineage.segment_bounds(on_path))
                )
            ],
        }

    def take_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A read-only, portable reference to one exact generation and its edit."""
        name = _required(payload, "project")
        take_id = _required(payload, "take")
        with _translated():
            project = self.projects.load(name)
            takes = self.gallery.list(name)
        by_id = {take.id: take for take in takes}
        take = by_id.get(take_id)
        if take is None:
            raise ApiError(404, "That take is no longer in this project.")
        source = project.source(str(payload.get("source") or take.shot))
        recorded_source = project.source(take.shot)
        directory = self.projects.path_for(name).resolve()
        media: dict[str, Any] = {"path": None, "exists": None}
        try:
            target = _media_path(take.filename, take.subfolder, take.type, must_exist=False)
            media.update(path=str(target), exists=target.is_file())
        except ApiError as error:
            media["note"] = error.message

        def take_ref(item: Take) -> dict[str, Any]:
            return {"id": item.id, "filename": item.filename, "shot": item.shot,
                    "parent_take": item.parent_take, "seed": item.seed}

        ancestors = []
        seen = {take.id}
        parent = take.parent_take
        while parent in by_id and parent not in seen:
            seen.add(parent)
            ancestor = by_id[parent]
            ancestors.append(take_ref(ancestor))
            parent = ancestor.parent_take

        bin = edit.read(project.sources, takes)
        timeline = []
        related_clips = []
        for position, clip in enumerate(project.clips, 1):
            holder = project.source_of(clip)
            chosen_path = clip.path(bin.path_of(holder)) if holder else []
            summary = {"position": position, "id": clip.id,
                       "name": clip.label or (holder.name if holder else ""),
                       "source_id": clip.source_id,
                       "source_name": holder.name if holder else None,
                       "enabled": clip.enabled}
            timeline.append(summary)
            if (source and clip.source_id == source.id) or take.id in chosen_path:
                related_clips.append({**clip.to_dict(), **summary,
                                      "chosen_take_path": chosen_path,
                                      "target_in_chosen_path": take.id in chosen_path,
                                      "selected_in_ui": clip.id == payload.get("clip")})

        return {
            "format": "video-prompter-generation-context/v1",
            "copied_at": utc_now_iso(),
            "project": {"name": project.name, "directory": str(directory),
                        "project_file": str(directory / PROJECT_JSON),
                        "takes_file": str(directory / "takes.json"),
                        "render_profile": project.render_profile},
            "take": {**take.to_dict(), "file": media},
            "shot": source.to_dict(history=False) if source else None,
            "recorded_source": {"id": take.shot,
                                "name": recorded_source.name if recorded_source else None},
            "related_clips": related_clips,
            "timeline_order": timeline,
            "lineage": {"ancestors": list(reversed(ancestors)),
                        "unresolved_parent": parent or None,
                        "children": [take_ref(item) for item in takes
                                     if item.parent_take == take.id]},
            "notes": ["take.prompt_text is the exact recorded generation prompt; "
                      "shot contains the current saved authoring fields and may differ.",
                      "Related clips can use other takes; target_in_chosen_path does not "
                      "guarantee the take is visible after trimming.",
                      "File paths are on the server running Video Prompter."],
        }

    def resolve_lineage(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Turn a path and a range of lineage time into playable pieces.

        The monitor, a lineage preview, a single-segment preview and the export
        all come through here, so a join cannot behave one way in the player and
        another way in the file.
        """
        project = _required(payload, "project")
        raw_path = payload.get("path")
        if not isinstance(raw_path, list) or not all(isinstance(x, str) for x in raw_path):
            raise ApiError(400, "`path` must be a list of take ids.")

        with _translated():
            takes = self.gallery.list(project)

        index = lineage.index_forest(lineage.build_forest(takes, include_discarded=True))
        problems = lineage.validate(raw_path, index)
        on_path = lineage.path_takes(raw_path, index)

        start = float(payload.get("start") or 0.0)
        end = payload.get("end")
        pieces = lineage.resolve(on_path, start, None if end is None else float(end))

        return {
            "playlist": [
                {
                    "take_id": piece.take_id,
                    "url": piece.url,
                    "depth": piece.depth,
                    "from": piece.source_from,
                    "to": piece.source_to,
                    "start": piece.start,
                    "length": piece.seconds,
                }
                for piece in pieces
            ],
            "seconds": round(sum(piece.seconds for piece in pieces), 3),
            "problems": problems,
        }

    def pick_take(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Choose a take, as the whole path that ends at it. Empty clears it.

        A convenience over `set_source_path` for the one-click case: the path
        from the take's root down to it comes from each take's recorded parent,
        and it is then carried on down the chain — preferring what was
        remembered — so choosing at depth 1 of a three-segment source fills in
        depth 2 rather than leaving the clip short.

        There is no cascade any more. Nothing downstream needs realigning,
        because the parent's choice *is* `path[n-1]`.
        """
        name = _required(payload, "project")
        source_id = _required(payload, "source") if payload.get("source") else _required(payload, "shot")
        take_id = str(payload.get("take") or "")

        with _translated():
            bin = self._bin(name)
            source = self._source_of(name, source_id)

        path: list[str] = []
        if take_id:
            if take_id not in bin.index:
                raise ApiError(404, f"No take {take_id!r} in {name!r}.")
            reachable = {
                node.id for root in bin.roots_for(source_id) for node in root.walk()
            }
            if take_id not in reachable:
                raise ApiError(
                    400,
                    "That take is not part of any chain belonging to this source.",
                )
            path = edit.extend_path(
                bin,
                edit.path_to(bin, take_id),
                remembered=source.remembered,
                depth=len(source.segments),
            )

        with _translated():

            def mutate(project: Project) -> None:
                chosen = project.source(source_id)
                if chosen is None:
                    raise ApiError(404, f"No source {source_id!r} in {name!r}.")
                chosen.set_path(path)

            return {"project": self.projects.update(name, mutate).to_dict(history=False)}

    def annotate_take(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _required(payload, "project")
        take_id = _required(payload, "take")
        rating = payload.get("rating")
        discarded = payload.get("discarded")
        seen = payload.get("seen")

        with _translated():
            take = self.gallery.annotate(
                name,
                take_id,
                rating=None if rating is None else int(rating),
                discarded=None if discarded is None else bool(discarded),
                seen=None if seen is None else bool(seen),
            )
            if take is None:
                raise ApiError(404, f"No take {take_id!r} in {name!r}.")
            return {"take": take.to_dict()}

    def mark_takes_seen(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Clear (or restore) the new badge on a batch of takes.

        Ids that name nothing are ignored rather than refused: the editor
        sweeps whatever it has on screen, and a take forgotten between the
        draw and the click is not an error worth stopping the sweep for.
        """
        name = _required(payload, "project")
        wanted = payload.get("takes")
        if not isinstance(wanted, list):
            raise ApiError(400, "'takes' must be a list of take ids.")
        seen = bool(payload.get("seen", True))

        with _translated():
            changed = self.gallery.mark_seen(name, [str(one) for one in wanted], seen=seen)
            return {"changed": changed, "count": len(changed)}

    def _bin(self, project: str) -> edit.Bin:
        """This project's takes, read as lineages."""
        loaded = self.projects.load(project)
        return edit.read(loaded.sources, self.gallery.list(project))

    def _source_of(self, project: str, source_id: str) -> Source:
        if not self.projects.exists(project):
            raise ApiError(404, f"No project named {project!r}.")
        source = self.projects.load(project).source(source_id)
        if source is None:
            raise ApiError(404, f"No source {source_id!r} in {project!r}.")
        return source

    def _clip_lengths(self, project: str) -> dict[str, float]:
        """How much source each clip has to cut from, by clip id.

        Trim clamping needs to know how much lineage there really is, and only
        the take record knows — a segment's prompt is what was *asked* for.
        """
        loaded = self.projects.load(project)
        bin = edit.read(loaded.sources, self.gallery.list(project))
        return edit.lengths(loaded.sources, loaded.clips, bin)

    def sequence_report(self, project: str) -> dict[str, Any]:
        """The edit and the bin as the editor needs them, joined server side.

        Computed here because it is a join across the project and the take
        record, and doing that join in the browser would mean two copies of the
        trim arithmetic and two opinions about where a chain forks.
        """
        with _translated():
            if not self.projects.exists(project):
                raise ApiError(404, f"No project named {project!r}.")
            loaded = self.projects.load(project)
            takes = self.gallery.list(project)

        bin = edit.read(loaded.sources, takes)
        lengths = edit.lengths(loaded.sources, loaded.clips, bin)
        starts = loaded.timeline.starts_at(lengths)

        counts: dict[str, int] = {}
        for take in takes:
            if not take.discarded:
                counts[take.shot] = counts.get(take.shot, 0) + 1

        return {
            "duration": loaded.timeline.duration(lengths),
            "clips": [self._clip_report(clip, loaded, bin, lengths, starts)
                      for clip in loaded.clips],
            "sources": [self._source_report(source, loaded, bin, counts)
                        for source in loaded.sources],
            "playlist": self._playlist(loaded, bin, starts),
            "audio": self._sound_playlist(loaded, bin, starts),
            "findings": [
                render.finding_dict(finding)
                for finding in lint_sequence(loaded, takes)
            ],
        }

    def _playlist(
        self, project: Project, bin: edit.Bin, starts: dict[str, float]
    ) -> list[dict[str, Any]]:
        """The whole edit as files to play, in order.

        Resolved here rather than in the browser for the same reason the export
        is: a clip is a range of lineage time, so one clip can span two takes,
        and the monitor must cut at exactly the frame the export does. One
        entry per playable piece, with `start` in timeline seconds.
        """
        entries: list[dict[str, Any]] = []
        for clip in project.timeline.active():
            source = project.source_of(clip)
            if source is None:
                continue
            speed = clip.speed if clip.speed > 0 else 1.0
            cursor = starts.get(clip.id, 0.0)
            for piece in edit.resolve_clip(clip, source, bin):
                length = round(piece.seconds / speed, 3)
                entries.append({
                    "clip": clip.id,
                    "source": source.id,
                    "name": clip.label or source.name,
                    "take_id": piece.take_id,
                    "url": piece.url,
                    "depth": piece.depth,
                    "from": piece.source_from,
                    "to": piece.source_to,
                    "speed": speed,
                    "length": length,
                    "start": round(cursor, 3),
                })
                cursor += length
        return entries

    def _sound_playlist(
        self, project: Project, bin: edit.Bin, starts: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Only the sound that differs from the picture.

        A clip playing its own generation needs no entry — the monitor is
        already decoding that file for the picture, and a second copy of it
        would be two decoders and two opinions about the same frames. A muted
        clip needs no entry either; the clip report says so, and there is
        nothing to play. So this list is exactly the borrowed tracks.
        """
        entries: list[dict[str, Any]] = []
        for clip in project.timeline.active():
            source = project.source_of(clip)
            if source is None or clip.audio.muted or not clip.audio.borrowed:
                continue

            holder = project.source(clip.audio.source_id) or source
            speed = clip.speed if clip.speed > 0 else 1.0
            cursor = starts.get(clip.id, 0.0)
            for piece in edit.resolve_audio(clip, source, bin, holder):
                length = round(piece.seconds / speed, 3)
                entries.append({
                    "clip": clip.id,
                    "take_id": piece.take_id,
                    "url": piece.url,
                    "from": piece.source_from,
                    "to": piece.source_to,
                    "speed": speed,
                    "gain_db": clip.audio.gain_db,
                    "length": length,
                    "start": round(cursor, 3),
                })
                cursor += length
        return entries

    def _clip_report(
        self,
        clip: Clip,
        project: Project,
        bin: edit.Bin,
        lengths: dict[str, float],
        starts: dict[str, float],
    ) -> dict[str, Any]:
        source = project.source_of(clip)
        available = lengths.get(clip.id, 0.0)
        path = clip.path(bin.path_of(source)) if source else []
        return {
            "id": clip.id,
            "source_id": clip.source_id,
            "name": clip.label or (source.name if source else "(missing source)"),
            "label": clip.label,
            "enabled": clip.enabled,
            "starts_at": starts.get(clip.id, 0.0),
            "source_seconds": available,
            "source_in": clip.source_in,
            "source_out": clip.out_point(available),
            "speed": clip.speed,
            "timeline_seconds": clip.timeline_seconds(available),
            "path": path,
            # Pinned means "stay on this branch when the bin switches".
            "pinned": clip.path_override is not None,
            "mutes": self._mute_report(clip, project, available),
            "rendered": bool(bin.takes_of(path)),
            "audio": {
                "muted": clip.audio.muted,
                "gain_db": clip.audio.gain_db,
                "offset": clip.audio.offset,
                "borrowed": clip.audio.borrowed,
                "source_id": clip.audio.source_id,
                # The leaf of the borrowed path — the take the editor's picker
                # is showing as chosen. The rest of the path is how to get to it.
                "take": (clip.audio.path_override or [None])[-1],
                "path": list(clip.audio.path_override or []),
            },
        }

    def _mute_report(
        self, clip: Clip, project: Project, available: float
    ) -> list[dict[str, Any]]:
        """One clip's mutes, in the clock that stores them and the one that draws.

        `start`/`end` are lineage seconds — the record, and what an edit posts
        back. `at`/`length` are seconds into the clip as it sits on the ruler,
        which is the only thing a band can be drawn from, and they are null for a
        mute the current trim has pushed out of sight. Sending both means the
        browser never has to redo this arithmetic and get a different answer.
        """
        speed = clip.speed if clip.speed > 0 else 1.0
        low = max(0.0, clip.source_in)
        high = clip.out_point(available)

        entries: list[dict[str, Any]] = []
        for mute in project.timeline.mutes_for(clip.id):
            seen = mute.clamped(low, high) if high > low else None
            entries.append({
                "id": mute.id,
                "start": mute.start,
                "end": mute.end,
                "at": None if seen is None else round((seen.start - low) / speed, 3),
                "length": None if seen is None else round(
                    (seen.end - seen.start) / speed, 3
                ),
            })
        return entries

    def _source_report(
        self, source: Source, project: Project, bin: edit.Bin, counts: dict[str, int]
    ) -> dict[str, Any]:
        path = bin.path_of(source)
        on_path = bin.takes_of(path)
        return {
            "id": source.id,
            "name": source.name,
            "path": path,
            # False when the path is a default rather than a stored choice.
            "chosen": bool(source.path),
            "remembered": dict(source.remembered),
            "seconds": edit.source_seconds(source, on_path),
            "rendered_seconds": lineage.lineage_seconds(on_path),
            "take_count": counts.get(source.id, 0),
            "problems": bin.problems_of(path),
            "used_by": [clip.id for clip in project.timeline.using(source.id)],
            "segments": [
                {
                    "id": segment.id,
                    "depth": depth,
                    "quality": segment.quality,
                    "planned_seconds": segment.prompt.rendered_seconds,
                    "take_id": path[depth] if depth < len(path) else None,
                }
                for depth, segment in enumerate(source.segments)
            ],
            # What is on offer at each depth, given the path chosen above it.
            # A take whose parent is not on the path is not in this list at all.
            "branches": [
                {
                    "depth": depth,
                    "chosen": path[depth] if depth < len(path) else None,
                    "options": [
                        {
                            "take_id": node.id,
                            "filename": node.take.filename,
                            "rating": node.take.rating,
                            "seconds": node.take.seconds,
                            "children": len(node.children),
                        }
                        for node in nodes
                    ],
                }
                for depth, nodes in enumerate(bin.branches_of(source, path))
            ],
        }

    def record_takes(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record finished renders.

        Accepts either takes the client already parsed, or raw ComfyUI history
        entries — reconciliation on page load has the latter and should not
        have to understand output shapes.
        """
        project = _required(payload, "project")
        takes = [Take.from_dict(entry) for entry in payload.get("takes") or []]

        for prompt_id, entry in (payload.get("history") or {}).items():
            if not isinstance(entry, dict):
                continue
            # Only this project's runs; the queue is shared with everything else.
            if project_of(entry) != project:
                continue
            takes.extend(takes_from_history(str(prompt_id), entry))

        generation_times = payload.get("generation_times")
        if isinstance(generation_times, dict):
            for take in takes:
                if take.generation_time is None and str(take.prompt_id) in generation_times:
                    try:
                        take.generation_time = float(generation_times[str(take.prompt_id)])
                    except (ValueError, TypeError):
                        pass

        with _translated():
            added = self.gallery.record(project, takes)
            return {"added": [take.to_dict() for take in added], "count": len(added)}

    def forget_take(self, payload: dict[str, Any]) -> dict[str, Any]:
        project = _required(payload, "project")
        prompt_id = _required(payload, "prompt_id")
        with _translated():
            return {"forgotten": self.gallery.forget(project, prompt_id)}

    # -- composing and rendering ------------------------------------------

    def preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Live pane: composed text plus findings. Never fails on a bad prompt."""
        megapixels, _, _ = self._quality(payload)
        return render.preview(
            self._prompt(payload), megapixels=megapixels, cast=self._cast(payload)
        )

    def parse_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Read a composed prompt string back into structured prompt fields."""
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ApiError(400, "Expected non-empty `text` to parse.")
        raw_duration = payload.get("duration")
        duration = (
            int(raw_duration)
            if isinstance(raw_duration, (int, float)) and raw_duration > 0
            else None
        )
        parsed = parse(text, duration=duration)
        structured = bool(
            parsed.storyboard.beats
            or parsed.style
            or parsed.overview
            or parsed.audio
            or parsed.music
            or parsed.storyboard.cues
            or parsed.screen_text
        )
        if not structured:
            parsed.overview = text
        return {"prompt": prompt_to_dict(parsed), "structured": structured}

    def transform(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply storyboard and canvas operations, and hand back the result.

        Exists so the editor never has to re-implement the frame grid, the
        canvas rules or contiguity in JavaScript. One round trip returns both
        the new prompt and its preview.
        """
        prompt = self._prompt(payload)
        operations = payload.get("ops")
        if not isinstance(operations, list):
            raise ApiError(400, "Expected an `ops` list.")

        for operation in operations:
            if not isinstance(operation, dict):
                raise ApiError(400, "Each op must be an object.")
            name = operation.get("op")

            if name == "fix":
                prompt.storyboard = prompt.storyboard.make_contiguous()
            elif name == "refit":
                prompt.storyboard = prompt.storyboard.refit(_int(operation, "duration"))
            elif name == "ratio":
                try:
                    prompt.set_ratio(str(operation.get("ratio", "")))
                except ValueError as error:
                    raise ApiError(400, str(error)) from error
            else:
                raise ApiError(400, f"Unknown op {name!r}.")

        return {
            "prompt": prompt_to_dict(prompt),
            "preview": render.preview(prompt, cast=self._cast(payload)),
        }

    def prepare_render(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Compose and inject one render.

        The mode is decided on the *resolved* prompt: a cast member's identity
        files are references like any other, and a ref2v graph bakes its
        reference loaders in at build time. Choosing the mode before resolution
        would build a t2v graph for a shot whose composed text goes on to name
        `<Picture 1>`, and the member's face would never reach the model.
        """
        prompt = self._prompt(payload)
        cast = self._cast(payload)
        base = self._profile(payload)
        megapixels, steps, turbo_strength = self._quality(payload)

        with _translated(), _unblocked():
            mode, profile = imagery.profile_for(
                cast_module.resolve(prompt, cast), base, self._reference_size(payload)
            )
            prepared = render.prepare_render(
                prompt,
                profile,
                seed=payload.get("seed"),
                strict=bool(payload.get("strict", True)),
                megapixels=megapixels,
                steps=steps,
                turbo_strength=turbo_strength,
                cast=cast,
                refine=self._refine(payload),
                is_selflift=self._is_selflift(payload),
                selflift_settings=self._selflift_settings(payload),
                is_singularity_2pass=self._is_singularity_2pass(payload),
                singularity_settings=self._singularity_settings(payload),
            )

        return self._with_video_loras(prepared, payload, mode)

    def prepare_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """N renders of one prompt at consecutive seeds, for comparing.

        Returned as a list rather than queued here: submitting is the client's
        job, because it owns the socket the execution events come back on.

        The mode comes off the resolved prompt, for the reason `prepare_render`
        gives.
        """
        prompt = self._prompt(payload)
        cast = self._cast(payload)
        base = self._profile(payload)
        megapixels, steps, turbo_strength = self._quality(payload)

        count = _batch_count(payload)

        with _translated(), _unblocked():
            mode, profile = imagery.profile_for(
                cast_module.resolve(prompt, cast), base, self._reference_size(payload)
            )
            prepared = render.prepare_batch(
                prompt,
                profile,
                count,
                seed=payload.get("seed"),
                strict=bool(payload.get("strict", True)),
                megapixels=megapixels,
                steps=steps,
                turbo_strength=turbo_strength,
                cast=cast,
                refine=self._refine(payload),
                is_selflift=self._is_selflift(payload),
                selflift_settings=self._selflift_settings(payload),
                is_singularity_2pass=self._is_singularity_2pass(payload),
                singularity_settings=self._singularity_settings(payload),
            )

        tier = self._tier(payload)
        return {"renders": [self._with_video_loras(entry, payload, mode) for entry in prepared],
                "mode": mode, "quality": tier}

    def prepare_reshoot(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Re-render a take from its own frozen values.

        The client sends the take's stored prompt text, canvas, length and (for
        a fine re-shoot) seed, so the new render matches the take rather than the
        shot's live prompt, which may have drifted since.

        The *text* is frozen; the *media* is not, and cannot be. A take records
        the words it rendered from, never the reference images or keyframe that
        were wired beside them, so there is nothing frozen here to rebuild a
        ref2v graph out of. The shot's current `prompt` is therefore sent along
        purely for its media, and the mode is chosen off that — otherwise a
        re-shoot of a shot whose text says `<Picture 1>` renders through the
        bare t2v profile, and the model, given a description and no face,
        invents a new one. Omitting `prompt` still renders text-only, which is
        what a caller with no shot to hand (the CLI) wants.
        """
        base = self._profile(payload)
        cast = self._cast(payload)
        megapixels, steps, turbo_strength = self._quality(payload)

        prompt_text = payload.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ApiError(400, "A re-shoot needs the take's `prompt_text`.")

        with _translated(), _unblocked():
            mode, profile = "t2v", base
            if isinstance(payload.get("prompt"), dict):
                mode, profile = imagery.profile_for(
                    cast_module.resolve(self._prompt(payload), cast),
                    base,
                    self._reference_size(payload),
                )
            prepared = render.prepare_reshoot(
                profile,
                prompt_text=prompt_text,
                width=_int(payload, "width"),
                height=_int(payload, "height"),
                frames=_int(payload, "frames"),
                seconds=_num(payload, "seconds"),
                seed=payload.get("seed"),
                megapixels=megapixels,
                steps=steps,
                turbo_strength=turbo_strength,
                refine=self._refine(payload),
                is_selflift=self._is_selflift(payload),
                selflift_settings=self._selflift_settings(payload),
                is_singularity_2pass=self._is_singularity_2pass(payload),
                singularity_settings=self._singularity_settings(payload),
            )

        return self._with_video_loras(prepared, payload, mode)

    def prepare_refine(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Refine an existing take via guided resample and/or face refine.

        Takes a take reference (`source`, e.g. `"subfolder/take.mp4 [output]"`),
        the take's dimensions, prompt, and optional refinement settings.
        Builds a standalone ComfyUI workflow that sharpens textures and/or
        enhances face detail, returning a new take linked to the parent.
        """
        source = _required(payload, "source")
        take_id = str(payload.get("take_id") or payload.get("parent_take") or "")
        width = _int(payload, "width")
        height = _int(payload, "height")
        frames = int(payload.get("frames") or 124)
        seconds = float(payload.get("seconds") or 5.0)
        seed = int(payload.get("seed") or 42)
        prompt_text = str(payload.get("prompt_text") or "")

        try:
            spec = refine.RefineSpec.from_request(payload)
        except (ValueError, TypeError) as exc:
            raise ApiError(400, str(exc)) from exc

        # Check for project-level model overrides and resolve parent_take / character reference
        model_overrides = None
        project_name = payload.get("project")
        source_take = None
        parent_take = payload.get("parent_take")

        if project_name and self.projects.exists(str(project_name)):
            project = self.projects.load(str(project_name))
            model_overrides = (project.settings or {}).get("refine_models")

            source_take = self.gallery.find(str(project_name), take_id)
            # The take's own length beats a guess: the guided pass crops its
            # output to `seconds`, so a default would cut a longer take short.
            if source_take and not payload.get("seconds") and source_take.seconds:
                seconds = float(source_take.seconds)
            if source_take and not payload.get("frames") and source_take.frames:
                frames = int(source_take.frames)
            if source_take and not parent_take:
                parent_take = source_take.parent_take or None

            # Pick a reference face only when it cannot be the wrong person.
            # Face refine works on the largest face in frame, so with two
            # characters in the shot either picture could land on the other
            # one; and a shot with nobody cast has no one to borrow a face
            # from. In both cases refine without a reference.
            if spec.enhancement_mode == "h3" and spec.face_refine and spec.face_target != "small_faces" and not spec.reference_face:
                shot_id = (source_take.shot if source_take else None) or payload.get("shot") or payload.get("source_id")
                source_obj = project.source(shot_id) if shot_id else None
                characters: list[Member] = []
                for seg in source_obj.segments if source_obj else []:
                    for cid in seg.prompt.cast or []:
                        member = project.cast.get(cid)
                        if member and member.kind == "character" and member not in characters:
                            characters.append(member)
                if len(characters) == 1:
                    images = [r for r in characters[0].references if r.kind == "image"]
                    faces = [r for r in images if "face" in r.path.lower() or "face" in r.role.lower()]
                    chosen = (faces or images or [None])[0]
                    if chosen is not None:
                        spec.reference_face = chosen.path

        try:
            prepared = refine.prepare_refine(
                source_video=source,
                source_take_id=take_id,
                source_width=width,
                source_height=height,
                seconds=seconds,
                prompt_text=prompt_text,
                seed=seed,
                spec=spec,
                model_overrides=model_overrides,
            )
        except (ValueError, TypeError) as exc:
            raise ApiError(400, str(exc)) from exc

        return {
            **prepared.to_dict(),
            # The words the parent rendered from. Without them the refined take
            # records an empty prompt and cannot be re-shot.
            "prompt_text": prompt_text,
            "quality": "refined",
            "parent_take": parent_take if parent_take else None,
            "refined_from": take_id,
            "refinement": prepared.refinement,
            "reference_face": prepared.reference_face,
            "findings": [],
            "mode": "refine",
            "seconds": seconds,
            "frames": frames,
        }

    def prepare_extend(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Continue a take with H3 Motion Context.

        Unlike a re-shoot, the source is another clip, not a fresh generation:
        the client sends the take's ComfyUI-visible video reference (`source`,
        e.g. `"video/take.mp4 [output]"`) and a freeform continuation
        `prompt_text`. `context` (frames to pin) and `latent` (a saved AV latent
        for the cleaner audio path) are optional; a take rendered without a
        latent simply uses the video path.

        Like a re-shoot, the continuation renders through the shot's *own*
        graph — same weights, same LoRAs, same cast references — with Motion
        Context wired in. So `profile` and `prompt` matter here for the same
        reason they matter there: without them the continuation of a shot built
        on `<Subject 1>` renders t2v, and the model invents a stranger.
        """
        mode, profile, plan = self._extend_plan(payload)
        with _translated(), _unblocked():
            prepared = render.prepare_extend(profile, **plan)
        return self._with_video_loras(prepared, payload, mode)

    def prepare_extend_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """`count` continuations of one clip at consecutive seeds.

        The same payload as a single extend plus `count`. A continuation is
        guesswork past the pinned run, so it earns candidates the same way a
        fresh generation does; queueing them stays the client's job.
        """
        mode, profile, plan = self._extend_plan(payload)
        count = _batch_count(payload)
        with _translated(), _unblocked():
            prepared = render.prepare_extend_batch(profile, count=count, **plan)
        tier = self._tier(payload)
        return {"renders": [self._with_video_loras(entry, payload, mode) for entry in prepared],
                "mode": mode, "quality": tier}

    def _extend_plan(
        self, payload: dict[str, Any]
    ) -> tuple[str, RenderProfile, dict[str, Any]]:
        """Validate an extend payload into the arguments `render` wants.

        Shared by the single and batch paths, which differ only in how many
        seeds come out the far end — every check here applies to both, and a
        sweep that skipped one would fail on the GPU instead of at the door.
        """
        source = _required(payload, "source")
        prompt_text = payload.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ApiError(400, "An extend needs a continuation `prompt_text`.")

        latent = payload.get("latent") or None
        parent_take = str(payload.get("parent_take") or "") or None
        context_length = payload.get("context")
        if context_length is not None:
            if isinstance(context_length, bool) or not isinstance(context_length, int):
                raise ApiError(400, "`context` must be a whole number of frames.")
            if context_length < 1:
                raise ApiError(400, "`context` must be at least 1 frame.")
            # The node snaps off-grid requests DOWN and only logs it. Snap here
            # too, so the value we report back is the one actually pinned.
            context_length = extend.snap_context(context_length)

        frames, clip_seconds = _clip_length(payload)
        if context_length is not None and context_length >= frames:
            # The node raises on this, but only after the job reaches the GPU.
            raise ApiError(
                400,
                f"{context_length} frames of context will not fit in a {frames} frame "
                "clip — the carried-over run has to be shorter than what it starts.",
            )
        width, height = _int(payload, "width"), _int(payload, "height")
        if self._is_selflift(payload) or self._is_singularity_2pass(payload):
            # The continuation graph only knows the classic two-stage split;
            # rendering that under SelfLift or Singularity 2-pass would mislabel the take.
            raise ApiError(
                400,
                f"`{payload.get('quality')}` cannot continue a take yet. Use "
                "turbo_2pass or final_2pass for a two-pass continuation.",
            )
        megapixels, steps, turbo_strength = self._quality(payload)
        refine = self._refine(payload)
        if refine is not None:
            self._check_extend_two_pass(width, height, megapixels)

        mode, base = self._extend_base(payload)
        with _translated(), _unblocked():
            profile = extend.build_extend_profile(
                base=base, latent_backed=bool(latent)
            )

        return mode, profile, {
            "prompt_text": prompt_text,
            "source_video": source,
            "width": width,
            "height": height,
            "frames": frames,
            "seconds": clip_seconds,
            "seed": payload.get("seed"),
            "steps": steps,
            "turbo_strength": turbo_strength,
            "megapixels": megapixels,
            "refine": refine,
            "source_latent": latent,
            "context_length": context_length,
            "parent_take": parent_take,
        }

    def _extend_base(
        self, payload: dict[str, Any]
    ) -> tuple[str, RenderProfile | None]:
        """The (mode, profile) a continuation should be built on top of.

        The same choice `prepare_render` makes, off the same document, so a
        continuation is the shot's own render with Motion Context added rather
        than a second, plainer pipeline that happens to run beside it. Omitting
        `profile` falls back to a built-in local-H3 graph — enough for the CLI
        and the bare node, never enough for a shot with media, which is why
        that combination is refused rather than quietly rendered text-only.
        """
        base = None
        if payload.get("profile"):
            base = self._profile(payload)

        if not isinstance(payload.get("prompt"), dict):
            return "t2v", base

        with _translated(), _unblocked():
            prompt = cast_module.resolve(self._prompt(payload), self._cast(payload))
            mode = imagery.render_mode(prompt)
            if mode == "t2v":
                return mode, base
            if base is None:
                raise ApiError(
                    400,
                    f"This shot renders as {mode}, so continuing it needs the "
                    "render profile its keyframes or references hang off. Send "
                    "`profile`.",
                )
            return imagery.profile_for(prompt, base, self._reference_size(payload))

    @staticmethod
    def _check_extend_two_pass(
        width: int, height: int, megapixels: float | None
    ) -> None:
        """Refuse a two-pass continuation with nothing to gain from a second pass.

        A continuation finishes on the canvas of the clip it continues (see
        `render.prepare_extend`), so a two-pass tier only means anything when
        that canvas is meaningfully bigger than the tier's first pass. Extending
        a 608x352 draft at `final_2pass` would upscale it to 608x352 — the cost
        of two passes for none of the benefit — and, worse, would record itself
        as a two-pass take.
        """
        if megapixels is None:
            return
        if width * height < 2 * megapixels * 1e6:
            raise ApiError(
                400,
                f"A two-pass continuation finishes on the canvas it continues, "
                f"and {width}x{height} is not enough bigger than the tier's own "
                f"{megapixels} MP first pass to be worth a second one. Extend a "
                f"720p clip at this tier, or pick a single-pass tier here.",
            )

    # -- text-to-image (keyframe generation) ---------------------------------

    def prepare_t2i(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate a still image from a text prompt, for use as an i2v keyframe.

        Uses Krea 2 Turbo (distilled, 4 steps, cfg 1) against the Wan 2.1 VAE.
        Defaults to H3's native canvas so the image feeds into i2v at the
        right resolution without resize artifacts.

        Optional keys: ``width``, ``height`` (delivery size, default 1920×1088),
        ``seed``, ``steps`` (default 8), ``cfg``, ``style`` (a cinematic preset
        from :func:`t2i.style_names`), ``loras`` (a stack of
        ``{name, strength_model, strength_clip, enabled}`` entries patched onto
        the model for grade/lens/character control), and ``supersample`` (>1
        renders larger then lanczos-downscales for sharper background detail).
        """
        prompt_text = payload.get("prompt")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ApiError(400, "t2i needs a `prompt` string.")

        width = int(payload["width"]) if "width" in payload else None
        height = int(payload["height"]) if "height" in payload else None
        seed = payload.get("seed")
        steps = payload.get("steps")
        cfg = payload.get("cfg")
        style = payload.get("style")
        loras = payload.get("loras")
        supersample = payload.get("supersample")

        if style is not None and not isinstance(style, str):
            raise ApiError(400, "`style` must be a string preset name.")
        if loras is not None and not isinstance(loras, list):
            raise ApiError(400, "`loras` must be a list of LoRA objects.")

        kwargs: dict[str, Any] = {}
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if seed is not None:
            kwargs["seed"] = seed
        if steps is not None:
            kwargs["steps"] = int(steps)
        if cfg is not None:
            kwargs["cfg"] = float(cfg)
        if style:
            kwargs["style"] = style
        if loras:
            kwargs["loras"] = loras
        if supersample is not None:
            kwargs["supersample"] = float(supersample)

        try:
            return t2i.prepare_t2i(prompt_text.strip(), **kwargs)
        except ValueError as error:
            raise ApiError(400, str(error)) from error

    def t2i_styles(self) -> dict[str, Any]:
        """The cinematic style presets available for keyframe generation.

        Returns each preset's key and the prompt-suffix language it applies, so
        an editor or agent can offer them without hard-coding the list.
        """
        return {
            "styles": [
                {"name": name, "prompt": t2i.STYLE_PRESETS[name]}
                for name in t2i.style_names()
            ]
        }

    def prepare_t2i_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """`count` images from one prompt, at consecutive seeds.

        Seed hunting is the whole job of the keyframe generator — one prompt
        rarely lands the composition first try — so the sweep is a first-class
        call rather than the client looping. A `seed` in the payload becomes the
        base, which makes the whole sweep reproducible from that one number.
        """
        count = _batch_count(payload)
        seed = payload.get("seed")
        if seed is None:
            from .profiles import random_seed
            # Headroom for the run, so consecutive sweeps cannot collide.
            base = random_seed() - count
        else:
            base = int(seed)
        return {
            "renders": [
                self.prepare_t2i({**payload, "seed": base + offset})
                for offset in range(count)
            ]
        }

    # -- the still library ---------------------------------------------------

    def list_stills(self, project: str, source: str | None = None) -> dict[str, Any]:
        """Every generated image for a project, newest first."""
        with _translated():
            return {
                "stills": [
                    still.to_dict() for still in self.stills.list(project, source)
                ]
            }

    def record_stills(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record finished image generations, adopting each into the input dir.

        Takes raw ComfyUI history, like `record_takes`, and for the same
        reason: an editor reopened halfway through a sweep has history and
        nothing else. Adoption happens here rather than when the author picks
        the image, so a still is usable as a keyframe the moment it appears.
        """
        project = _required(payload, "project")
        found: list[Still] = [
            Still.from_dict(entry) for entry in payload.get("stills") or []
        ]
        for prompt_id, entry in (payload.get("history") or {}).items():
            if not isinstance(entry, dict):
                continue
            if project_of(entry) != project:
                continue
            found.extend(stills_from_history(str(prompt_id), entry))

        for still in found:
            if still.path:
                continue
            try:
                still.path = self._adopt_image(still.filename, still.subfolder)
            except ApiError:
                # No input dir, or the file has been cleaned up. The still is
                # still worth recording — its recipe is the part that cannot be
                # recovered — it just cannot be attached to a prompt.
                still.path = ""

        with _translated():
            added = self.stills.record(project, found)
            return {"added": [still.to_dict() for still in added], "count": len(added)}

    def annotate_still(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Flag a still, or re-file it against a source or a cast member."""
        project = _required(payload, "project")
        still_id = _required(payload, "still")
        fields: dict[str, Any] = {}
        for key in ("favorite", "discarded"):
            if key in payload:
                fields[key] = bool(payload[key])
        for key in ("source", "member"):
            if key in payload:
                fields[key] = str(payload[key] or "")
        with _translated():
            found = self.stills.annotate(project, still_id, **fields)
        if found is None:
            raise ApiError(404, f"No still {still_id!r} in {project!r}.")
        return {"still": found.to_dict()}

    def forget_still(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Drop a still from the library. The image file itself is left alone."""
        project = _required(payload, "project")
        still_id = _required(payload, "still")
        with _translated():
            return {"forgotten": self.stills.forget(project, still_id)}

    def adopt_output_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Copy a generated image from ComfyUI's output dir to the input dir.

        After a t2i render completes, the image lives in ``output/``. To use it
        as a first-frame keyframe for i2v, ``LoadImage`` needs it in ``input/``.
        This copies the file and returns the input-relative filename that goes
        on a segment's ``first_frame`` field.
        """
        filename = _required(payload, "filename")
        subfolder = str(payload.get("subfolder") or "")
        source = _output_image_path(filename, subfolder)
        return {
            "filename": self._adopt_image(filename, subfolder),
            "source": str(source),
            "bytes": source.stat().st_size,
        }

    def reveal_file(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Open the desktop file manager on one rendered file, with it selected.

        The way out of this tool. A take someone wants in Resolve or Premiere is
        already a file on disk; the only thing between the card and a drag into
        another app is knowing which file it is. Everything here happens on the
        machine ComfyUI runs on, which for this editor is the machine being sat
        at — and the response carries the path back regardless, so a caller
        reached from another machine can show the text instead of pretending a
        window opened somewhere useful.
        """
        filename = _required(payload, "filename")
        subfolder = str(payload.get("subfolder") or "")
        kind = str(payload.get("type") or "output")
        target = _media_path(filename, subfolder, kind)
        opened = _reveal(target)
        return {
            "path": str(target),
            "folder": str(target.parent),
            "opened": opened,
        }

    def _adopt_image(self, filename: str, subfolder: str = "") -> str:
        """Copy one output image into the input dir; returns its stored path.

        The name is made unique on the way in, so adopting the same generation
        twice never overwrites a file some other segment already points at.
        """
        source = _output_image_path(filename, subfolder)
        target, ref = self._input_target(filename)
        import shutil
        shutil.copy2(source, target)
        return ref

    def list_video_loras(self) -> dict[str, Any]:
        return video_loras.h3_catalog()

    def update_video_loras(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = _required(payload, "project")
        try:
            rows = video_loras.normalize(payload.get("loras"))
        except ValueError as error:
            raise ApiError(400, str(error)) from error
        with _translated():
            def update(project: Project) -> None:
                project.settings["video_loras"] = rows
            self.projects.update(name, update)
        return {"loras": rows}

    def _with_video_loras(self, prepared: Any, payload: dict[str, Any], mode: str) -> dict[str, Any]:
        settings = self._quality_settings(payload) or {}
        rows = payload.get("loras", settings.get("video_loras", []))
        shot_setting = None
        if "loras" not in payload and payload.get("project") and payload.get("source_id"):
            shot_setting = self._source_of(str(payload["project"]), str(payload["source_id"])).video_loras
        # An explicit empty list is a deliberate bypass, including on extends.
        if "loras" not in payload and payload.get("parent_take") and payload.get("project"):
            parent = self.gallery.find(str(payload["project"]), str(payload["parent_take"]))
            if parent is not None and parent.loras is not None:
                rows = parent.loras
        # An explicit shot choice takes precedence over continuity defaults.
        if "loras" not in payload and shot_setting:
            if shot_setting["mode"] == "off":
                rows = []
            elif shot_setting["mode"] == "custom":
                rows = shot_setting["loras"]
        try:
            graph, active = video_loras.apply(prepared.workflow, rows)
        except ValueError as error:
            raise ApiError(400, str(error)) from error
        return {**prepared.to_dict(), "workflow": graph, "mode": mode,
                "quality": self._tier(payload), "loras": active}

    def _profile(self, payload: dict[str, Any]) -> Any:
        name = _required(payload, "profile")
        with _translated():
            if not self.profiles.exists(name):
                raise ApiError(404, f"No render profile named {name!r}.")
            return self.profiles.load(name)

    def _quality(
        self, payload: dict[str, Any]
    ) -> tuple[float | None, int | None, float | None]:
        """(megapixels, steps, turbo_strength) for the preset, or all `None`.

        Opt-in: a request with no `quality` renders exactly as before, leaving
        the workflow's own size, steps and LoRA strength. A `project` name, when
        given, lets a project override the preset numbers.
        """
        name = payload.get("quality")
        if name is None:
            return None, None, None
        return quality.resolve(str(name), self._quality_settings(payload))

    def _reference_size(self, payload: dict[str, Any]) -> str:
        """How large identity references may arrive, from the shot's tier.

        Unlike the other quality knobs this one is baked into the ref2v graph
        rather than injected, so it has to be resolved before `profile_for`
        builds it.
        """
        return quality.reference_size(
            payload.get("quality"), self._quality_settings(payload)
        )

    def _tier(self, payload: dict[str, Any]) -> str | None:
        """The tier name this render was asked for, or None if none was.

        Reported back so the client can record it on the take. A take's tier
        used to be read back off its canvas and step count, which works for a
        fresh generation — every tier has its own signature there — and does
        not work for a continuation, which renders at the canvas of the clip it
        continues rather than its tier's. A `preview` extend of a 720p parent is
        1.0 MP at 8 steps, and that is `turbo_hd`, exactly. Names do not have
        that problem.

        Falls back the way `quality.resolve` does, so what comes back is the
        tier that really drove the render rather than the string that was sent.
        """
        name = payload.get("quality")
        if name is None:
            return None
        return str(name) if str(name) in quality.QUALITY_PRESETS else quality.DEFAULT_QUALITY

    def _refine(self, payload: dict[str, Any]) -> tuple[int, float] | None:
        """(pass1_steps, output megapixels) when the tier renders in two passes.

        `None` for every single-pass tier and for a request with no `quality` at
        all, so this rides alongside `_quality` without changing what a caller
        that never asked for two passes gets.
        """
        name = payload.get("quality")
        if name is None:
            return None
        return quality.refine(str(name), self._quality_settings(payload))

    def _quality_settings(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """The project's preset overrides, when the payload names a project."""
        project_name = payload.get("project")
        if not (isinstance(project_name, str) and project_name.strip()):
            return None
        with _translated():
            if not self.projects.exists(project_name):
                return None
            return self.projects.load(project_name).settings

    def _is_selflift(self, payload: dict[str, Any]) -> bool:
        name = payload.get("quality")
        return bool(name and quality.is_selflift(str(name)))

    def _selflift_settings(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        settings = self._quality_settings(payload) or {}
        return settings.get("selflift_settings")

    def _is_singularity_2pass(self, payload: dict[str, Any]) -> bool:
        name = payload.get("quality")
        return bool(name and quality.is_singularity_2pass(str(name)))

    def _singularity_settings(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        settings = self._quality_settings(payload) or {}
        return settings.get("singularity_settings")

    # -- export -----------------------------------------------------------

    def export(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve the picked takes and write the edit out.

        `dry_run` stops after resolving, which is what the editor calls to
        show what an export *would* contain before spending minutes on it.
        """
        name = _required(payload, "project")
        fmt = str(payload.get("format") or "ffmpeg")
        if fmt not in export.FORMATS:
            raise ApiError(
                400, f"Unknown export format {fmt!r}; expected one of "
                f"{', '.join(export.FORMATS)}."
            )

        with _translated():
            if not self.projects.exists(name):
                raise ApiError(404, f"No project named {name!r}.")
            project = self.projects.load(name)
            takes = self.gallery.list(name)

            output_dir = self._output_dir(payload)
            try:
                clips = export.resolve(project, takes, output_dir)
            except export.ExportError as error:
                raise ApiError(409, str(error)) from error
            # Never raises: a clip whose sound cannot be found exports silent
            # rather than blocking a picture that is finished.
            sounds = export.resolve_sound(
                project, takes, output_dir, self.projects.audio_dir(name),
            )

        mixable, unreadable = export.playable_sounds(sounds)

        result: dict[str, Any] = {
            "format": fmt,
            "ffmpeg_available": export.have_ffmpeg(),
            # A take records the length that was asked for. When the file
            # disagrees the export quietly comes out short, so say so first.
            "short_sources": export.short_sources(clips),
            # Clips whose sound file has no audio track, or has gone missing.
            # Dropped from the mix, because mapping one would fail the render
            # outright and take the picture with it.
            "unreadable_audio": unreadable,
            **export.summary(clips, sounds),
        }

        if fmt == "ffmpeg":
            destination = self._export_path(name, payload, ".mp4")
            command = export.ffmpeg_command(clips, destination, mixable)
            result["command"] = command
            result["destination"] = str(destination)
            if payload.get("dry_run"):
                return result
            missing = [clip.path for clip in clips if not clip.path.is_file()]
            if missing:
                raise ApiError(
                    409,
                    f"{len(missing)} take file(s) are not where the record says. "
                    f"First missing: {missing[0]}",
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            ok, message = export.run_ffmpeg(command)
            result["ok"] = ok
            result["message"] = message
            return result

        # The listings say what was *intended*, so they get every resolved
        # track — including one ffmpeg would have refused. A person reading an
        # EDL wants to know the sound was meant to come from there.
        text = (
            export.edl(clips, name, sounds)
            if fmt == "edl"
            else export.clip_list(clips, name, sounds)
        )
        result["text"] = text
        if payload.get("dry_run"):
            return result

        destination = self._export_path(name, payload, ".edl" if fmt == "edl" else ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        result["destination"] = str(destination)
        result["ok"] = True
        result["message"] = f"Wrote {destination.name}."
        return result

    def _output_dir(self, payload: dict[str, Any]) -> Path:
        """Where take files really live. Overridable, because tests have no ComfyUI."""
        override = payload.get("output_dir")
        if override:
            return Path(str(override))
        found = paths.comfy_output_dir()
        if found is None:
            raise ApiError(
                409,
                "Cannot find ComfyUI's output directory, so the take files "
                "cannot be located. Pass `output_dir` to say where they are.",
            )
        return found

    def _export_path(self, project: str, payload: dict[str, Any], suffix: str) -> Path:
        destination = payload.get("destination")
        if destination:
            dest_path = Path(str(destination))
            if ".." in dest_path.parts:
                raise ApiError(400, "Export destination must not contain '..' path traversal segments.")
            return dest_path
        stamp = utc_now_iso().replace(":", "").replace("-", "")
        return paths.exports_root() / project / f"{project}_{stamp}{suffix}"

    def _prompt(self, payload: dict[str, Any]) -> Any:
        return self._read_prompt(payload.get("prompt"))

    def _read_prompt(self, data: Any) -> VideoPrompt:
        if not isinstance(data, dict):
            raise ApiError(400, "Expected a `prompt` object.")
        try:
            return prompt_from_dict(data)
        except (TypeError, ValueError) as error:
            raise ApiError(400, f"Malformed prompt: {error}") from error


def _clamp(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ApiError(400, f"Expected a number, got {value!r}.") from error
    return round(min(max(number, low), high), 3)


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, f"`{key}` must be a number of seconds.")
    return float(value)


def _mute_span(start: Any, end: Any, available: float) -> tuple[float, float]:
    """A band's two ends, ordered and inside the lineage.

    Clamped to the whole lineage rather than to the clip's current trim: a mute
    is allowed to reach past an edit point, which is what lets a trim be undone
    without the silence having been quietly eaten in the meantime.
    """
    ceiling = max(0.0, available)
    low = _clamp(start, 0.0, ceiling)
    high = _clamp(end, 0.0, ceiling)
    if high < low:
        low, high = high, low
    if high - low <= 0:
        raise ApiError(400, "A mute needs two different ends to sit between.")
    return low, high


def _apply_sound(clip: Clip, data: Any, bin: edit.Bin) -> None:
    """Change what a clip is heard as, from the editor's vocabulary.

    The caller says which *take* it wants to hear; the path to it is worked out
    here from the take record. Letting a caller post a path would be a second
    opinion about which take follows which, and that is the disagreement the
    Source model exists to prevent.
    """
    if not isinstance(data, dict):
        raise ApiError(400, "`audio` must be an object.")

    sound = clip.audio
    if "muted" in data:
        sound.muted = bool(data["muted"])
    if "gain_db" in data:
        sound.gain_db = _clamp(data["gain_db"], MIN_GAIN_DB, MAX_GAIN_DB)
    if "offset" in data:
        sound.offset = _clamp(data["offset"], -60.0, 60.0)
    if "take" in data:
        take_id = data["take"]
        if not take_id:
            # Back to its own generation's sound — both halves cleared, so
            # nothing is left half-borrowed.
            sound.source_id = ""
            sound.path_override = None
            return
        node = bin.index.get(str(take_id))
        if node is None:
            raise ApiError(404, f"No take {take_id!r} to take sound from.")
        sound.source_id = node.take.shot or clip.source_id
        sound.path_override = edit.path_to(bin, str(take_id))


def _int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, f"{key!r} must be a number.")
    return int(value)


def _num(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, f"{key!r} must be a number.")
    return float(value)


def _output_image_path(filename: str, subfolder: str = "") -> Path:
    """A generated image's real file in ComfyUI's output dir.

    Raises rather than returning None: every caller needs the file to exist,
    and the two failures — no output directory, no such image — want different
    status codes.
    """
    output_dir = paths.comfy_output_dir()
    if output_dir is None:
        raise ApiError(503, "Cannot locate ComfyUI's output directory.")
    root = output_dir.resolve()
    source = (output_dir / subfolder / filename if subfolder else output_dir / filename).resolve()
    if root not in source.parents:
        raise ApiError(400, "Output image path must stay inside the output directory.")
    if not source.is_file():
        raise ApiError(404, f"Output image not found: {source}")
    return source


# The three places ComfyUI writes a file, by the `type` a history entry gives.
_MEDIA_DIRS = {
    "output": paths.comfy_output_dir,
    "input": paths.comfy_input_dir,
    "temp": paths.comfy_temp_dir,
}


def _media_path(filename: str, subfolder: str, kind: str, *, must_exist: bool = True) -> Path:
    """One ComfyUI file's real path, refused if it points outside its own dir.

    `filename` and `subfolder` come from a history entry, so they are normally
    tame — but they arrive over HTTP, and "show me this file" is a request to
    hand a path to the operating system. Resolving and then checking the result
    is still under the directory it claims to be in is what keeps `../` from
    turning a take card into a file browser for the whole disk.
    """
    if kind not in _MEDIA_DIRS:
        raise ApiError(400, f"Unknown file type {kind!r} — expected output, input or temp.")
    found = _MEDIA_DIRS[kind]()
    if found is None:
        raise ApiError(503, f"Cannot locate ComfyUI's {kind} directory.")
    root = found.resolve()
    target = (root / subfolder / filename if subfolder else root / filename).resolve()
    if root not in target.parents:
        raise ApiError(400, "That file is outside ComfyUI's media directories.")
    if must_exist and not target.is_file():
        raise ApiError(404, f"No such file: {target}")
    return target


def _reveal(target: Path) -> bool:
    """Show `target` in the desktop file manager. False when there is none.

    Windows' explorer wants the path as one `/select,<path>` token and exits
    non-zero even when it worked, so nothing here reads a return code — the
    file was found before we got this far, and a file manager that failed to
    draw is not something this process can fix. A headless box (a ComfyUI on a
    server) simply has no file manager, and says so rather than erroring: the
    path is in the response either way.
    """
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            # A string, not a list: the list form would quote the whole
            # argument and explorer would take it as a filename. Windows
            # filenames cannot contain a quote, so the quoting below is safe.
            subprocess.Popen(f'explorer /select,"{target}"')
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target.parent)])
    except OSError:
        return False
    return True


def _batch_count(payload: dict[str, Any]) -> int:
    """How many seeds a sweep covers, refused early if it is not a sane number.

    Shared by every batch endpoint so one cap governs them all: the ceiling is
    what keeps a single click from filling the GPU queue for the night.
    """
    count = payload.get("count", 2)
    if isinstance(count, bool) or not isinstance(count, int):
        raise ApiError(400, "`count` must be a whole number.")
    if not 1 <= count <= render.MAX_BATCH:
        raise ApiError(400, f"A batch runs 1-{render.MAX_BATCH} renders.")
    return count


def _clip_length(payload: dict[str, Any]) -> tuple[int, float]:
    """One clip's `frames` and `seconds`, from whichever the caller sent.

    A caller working from a plan knows seconds; one repeating a take knows
    frames. Either alone is enough — the missing half follows from the model's
    frame grid, so no client has to carry `align_frames` around. Sending both
    keeps them exactly as given, which is how a re-shoot reproduces a take.

    A budget in seconds reports back the *grid* length, not the round number
    asked for, the same way `VideoPrompt.rendered_seconds` does: 12s of intent
    is 294 frames, and calling that 12.0s would put the timeline out by a
    quarter second per generation.
    """
    def number(key: str) -> bool:
        value = payload.get(key)
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if not number("frames") and not number("seconds"):
        raise ApiError(400, "A clip needs `frames` or `seconds`.")

    if number("frames"):
        frames = _int(payload, "frames")
        seconds = _num(payload, "seconds") if number("seconds") else             h3_spec.seconds_for_frames(frames)
    else:
        frames = h3_spec.frames_for_seconds(_num(payload, "seconds"))
        seconds = h3_spec.seconds_for_frames(frames)
    return frames, seconds


def _required(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, f"Missing required field {key!r}.")
    return value


_COPY_SUFFIX = re.compile(r"^(?P<stem>.*?)\s*\((?P<number>\d+)\)$")


def _next_name(name: str) -> str:
    """"Rooftop" -> "Rooftop (2)" -> "Rooftop (3)". Copying twice reads right."""
    match = _COPY_SUFFIX.match(name)
    if match is None:
        return f"{name} (2)"
    return f"{match['stem']} ({int(match['number']) + 1})"


def _source(entry: Any) -> Source:
    """A source from either a source object or a v1 shot object.

    Accepting the old shape costs one type check and means an agent or a script
    written against v1 still creates something sensible instead of a blank.
    """
    if not isinstance(entry, dict):
        raise ApiError(400, "Each source must be an object.")
    if "segments" in entry:
        try:
            return Source.from_dict(entry)
        except ValueError as error:
            raise ApiError(400, str(error)) from error
    source, _ = migrate.split_shot(entry)
    return Source.from_dict(source)


def _segment_in(source: Source, segment_id: Any = None) -> Segment:
    """The named segment, or depth 0 when none was named.

    Depth 0 is the default because most sources are one generation long, and
    asking every caller to name a segment to edit the only one there is would
    be ceremony.
    """
    segment = (
        next((s for s in source.segments if s.id == segment_id), None)
        if segment_id else source.segments[0]
    )
    if segment is None:
        raise ApiError(404, f"No segment {segment_id!r} in {source.id!r}.")
    return segment


def _carry_revisions(stored: Source, incoming: Source, note: str = "") -> None:
    """Give an incoming source the stored history, filing whatever changed.

    Revisions are the server's to write, never a client's. A save replaces a
    source wholesale, so a tab that loaded before this existed — or an agent
    posting a source it built from a schema — would otherwise overwrite the
    history with nothing. Reading it off disk here means every route into a
    save leaves the same trail without having to know the trail exists.

    Segments are matched by id. One that arrives with an id nobody has seen was
    planned in this same save and has no past to keep.
    """
    kept = {segment.id: segment for segment in stored.segments}
    for segment in incoming.segments:
        was = kept.get(segment.id)
        if was is None:
            segment.revisions = []
            continue
        # Rewind to what disk holds, then let `revise` decide whether the
        # arriving prompt is a change worth filing.
        arriving, segment.prompt = segment.prompt, was.prompt
        segment.revisions = was.revisions
        segment.revise(arriving, note)


def _rename_mentions(project: Project, old: str, new: str) -> int:
    """Repoint every `@old` in the project at `@new`. Returns the fields moved."""
    return sum(
        mentions.rename(segment.prompt, old, new)
        for source in project.sources
        for segment in source.segments
    )


def _members_used(project: Project, member_id: str) -> list[dict[str, Any]]:
    """Every segment that names this member, as `{source, segment, speaking}`.

    Speaking membership is worth separating from silent: a deleted member who
    only decorated the opening costs a sentence, while one who carried a
    speaker key leaves dialogue attributed to nobody.
    """
    used: list[dict[str, Any]] = []
    for source in project.sources:
        for segment in source.segments:
            prompt = segment.prompt
            speaking = any(
                member_id in line.speakers
                for beat in prompt.storyboard.ordered_beats()
                for line in beat.lines
            )
            if not speaking and member_id not in prompt.cast:
                continue
            used.append(
                {
                    "source": source.id,
                    "name": source.name,
                    "segment": segment.id,
                    "speaking": speaking,
                }
            )
    return used


def _defaults(data: dict[str, Any]) -> Defaults:
    return Defaults.from_dict(data)


class _llm_errors:
    """Turn an `LlmError` into an `ApiError` carrying the status it chose.

    A model that refused, a key that is missing and a network that is down are
    different answers, and flattening them to 500 would leave the editor with
    nothing useful to say.
    """

    def __enter__(self) -> "_llm_errors":
        return self

    def __exit__(self, kind, error, traceback) -> bool:
        if isinstance(error, llm.LlmError):
            raise ApiError(error.status, str(error)) from error
        return False


class _unblocked:
    """Turn a blocked render into a 422 that carries the findings.

    422 rather than 400: the request was well formed, the prompt is not, and
    the findings are the answer — a render with overlapping timecodes costs a
    generation to discover.
    """

    def __enter__(self) -> "_unblocked":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if isinstance(exc, render.RenderBlocked):
            raise ApiError(
                422,
                "Prompt has errors that would waste a generation.",
                findings=[render.finding_dict(f) for f in exc.findings],
            ) from exc
        return False


class _translated:
    """Turn store-level errors into ApiErrors with the right status.

    A stale profile is 409 rather than 400: the request was fine, the workflow
    moved underneath it, and the fix is to re-capture.
    """

    def __enter__(self) -> "_translated":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is None or isinstance(exc, ApiError):
            return False
        if isinstance(exc, StaleProfileError):
            raise ApiError(409, str(exc)) from exc
        if isinstance(
            exc,
            (ProjectError, PresetError, ProfileError, extend.ExtendError,
             twostage.TwoStageError, selflift.SelfLiftError),
        ):
            raise ApiError(400, str(exc)) from exc
        if isinstance(exc, FileNotFoundError):
            raise ApiError(404, str(exc)) from exc
        return False
