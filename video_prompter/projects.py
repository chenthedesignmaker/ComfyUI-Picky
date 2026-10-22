"""Project storage: a bin of sources and a timeline of clips, saved atomically.

A project is two lists. `sources` is what was generated — one lineage each, with
the segments that planned it. `timeline` is how the edit uses them — clips
pointing into a source, in order. Splitting those apart is what `migrate` does to
a v1 project, and `from_dict` runs it on the way in, so a file written by an
older build loads as v2 without anyone converting anything by hand.

The shape is carried over from `comfy prompt node`, which earned it: writes go
through a temp file and `os.replace` so a crashed save cannot leave a truncated
`project.json`, every save snapshots the previous state into a tiered backup
directory, and saves that sharply shrink a project are flagged into a
longer-lived tier.

Projects also live on one of three shelves, which are real directories rather
than a flag inside the file: active projects sit at the root, archived ones
under `.archive`, trashed ones under `.trash`. Moving a directory is the whole
operation, so listing a shelf stays a directory scan that opens nothing, and
emptying the trash can only ever delete inside `.trash`. Every sidecar store
(`takes.json`, `stills.json`) resolves through `path_for`, so they follow a
project across shelves without knowing shelves exist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from . import migrate
from .cast import Cast
from .sequence import Clip, Timeline
from .sources import Segment, Source

PROJECT_JSON = "project.json"
# Sound brought in from outside, kept beside the project that uses it.
AUDIO_DIR = "audio"
BACKUPS_DIRNAME = ".backups"

# The shelves, as directories. Dot-prefixed so they cannot collide with a
# project name: `sanitize_project_name` strips the dot.
ARCHIVE_DIRNAME = ".archive"
TRASH_DIRNAME = ".trash"

ACTIVE, ARCHIVED, TRASHED = "active", "archived", "trashed"
SHELVES = (ACTIVE, ARCHIVED, TRASHED)
_SHELF_DIRS = {ACTIVE: "", ARCHIVED: ARCHIVE_DIRNAME, TRASHED: TRASH_DIRNAME}

# Sidecars a duplicate inherits. Takes and stills are pointers at files ComfyUI
# owns, so copying them costs kilobytes and hands the copy its own footage and
# image library rather than an empty gallery. `.backups` is deliberately not
# here: the original's undo history is not the copy's.
SIDECARS = ("takes.json", "stills.json", "image_prompts.json")

# Per-save ring. Churns fast under rapid saves, which is the point.
SAVE_BACKUP_LIMIT = 20
# Coarser tiers are immune to save bursts.
HOURLY_BACKUP_HOURS = 48
DAILY_BACKUP_DAYS = 30
REDFLAG_BACKUP_LIMIT = 40

# A save keeping less than this share of the previous content is suspicious.
SHRINK_RATIO = 0.7

SCHEMA_VERSION = migrate.SCHEMA_V2

_UNSAFE = re.compile(r"[^\w \-]", re.UNICODE)

T = TypeVar("T")


class ProjectError(ValueError):
    pass


def sanitize_project_name(name: str) -> str:
    cleaned = _UNSAFE.sub("", name or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise ProjectError(f"Not a usable project name: {name!r}")
    return cleaned


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_lineage_id() -> str:
    return f"lin_{uuid.uuid4().hex[:8]}"


@dataclass
class Defaults:
    """Blocks a new segment starts from.

    Not live inheritance — a segment copies these when it is created and is
    free to diverge afterwards. Live inheritance would have to be unpicked in
    `compose` and would fight the per-segment edits that are the whole point;
    copy-on-create plus an explicit push-to-all does the same job for the
    thing that actually matters, which is source nine not drifting from one.
    """

    style: str = ""
    camera: str = ""
    negatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "camera": self.camera,
            "negatives": list(self.negatives),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Defaults":
        return cls(
            style=data.get("style", ""),
            camera=data.get("camera", ""),
            negatives=list(data.get("negatives") or []),
        )

    def apply_to(self, segment: Segment) -> Segment:
        """Seed a segment's blocks, leaving anything already written alone."""
        if not segment.prompt.style.strip():
            segment.prompt.style = self.style
        if not segment.prompt.camera.strip():
            segment.prompt.camera = self.camera
        if not segment.prompt.negatives:
            segment.prompt.negatives = list(self.negatives)
        return segment


@dataclass
class Project:
    """A bin and an edit: `sources` is what was generated, `timeline` uses it."""

    name: str
    sources: list[Source] = field(default_factory=list)
    timeline: Timeline = field(default_factory=Timeline)
    defaults: Defaults = field(default_factory=Defaults)
    # Reusable identities segments point at by id. Additive: a project
    # written before the cast existed simply loads with an empty one.
    cast: Cast = field(default_factory=Cast)
    created: str = field(default_factory=utc_now_iso)
    modified: str = field(default_factory=utc_now_iso)
    render_profile: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    # Version family. Every project duplicated from this one carries the same
    # lineage id, so a family can be gathered and shown in the order it was
    # made. Stamped on first save; a project written before lineage existed
    # loads with an empty one and is a family of itself until it is saved.
    lineage: str = ""
    # The project this was duplicated from, by name, for display only. Names
    # can be reused or gone; `lineage` is what actually holds a family together.
    derived_from: str = ""

    @property
    def clips(self) -> list[Clip]:
        return self.timeline.clips

    def source(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def source_named(self, name: str) -> Source | None:
        return next((s for s in self.sources if s.name == name), None)

    def clip(self, clip_id: str) -> Clip | None:
        return self.timeline.get(clip_id)

    def source_of(self, clip: Clip) -> Source | None:
        return self.source(clip.source_id)

    def add_source(self, source: Source) -> Source:
        self.sources.append(source)
        return source

    def remove_source(self, source_id: str) -> bool:
        """Drop a source and everything in the edit that reads it.

        Its clips, their mutes, and any lane item lifted from its lineage.
        Leaving orphans behind would put holes in the edit that point at nothing,
        which is worse than the ripple. `Timeline.forget_source` does the work,
        so the ripple is written down once.
        """
        before = len(self.sources)
        self.sources = [s for s in self.sources if s.id != source_id]
        if len(self.sources) == before:
            return False
        self.timeline.forget_source(source_id)
        return True

    def to_dict(self, *, history: bool = True) -> dict[str, Any]:
        """The project as stored. `history=False` is the wire shape.

        See `Segment.to_dict`: the disk keeps every revision, API replies leave
        them out and serve them per segment on request.
        """
        return {
            "version": SCHEMA_VERSION,
            "name": self.name,
            "created": self.created,
            "modified": self.modified,
            "render_profile": self.render_profile,
            "lineage": self.lineage,
            "derived_from": self.derived_from,
            "settings": dict(self.settings),
            "defaults": self.defaults.to_dict(),
            "cast": self.cast.to_dict(),
            "sources": [s.to_dict(history=history) for s in self.sources],
            "timeline": self.timeline.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        """Read a project, migrating a v1 one on the way in.

        Migration happens here rather than at the store so that every route into
        a project — a load, a restored backup, a dict posted by an agent — lands
        on the same shape. `to_v2` returns already-v2 input untouched, so this
        costs a type check on the common path.
        """
        data = migrate.to_v2(data)
        return cls(
            name=data.get("name", ""),
            sources=[Source.from_dict(entry) for entry in data.get("sources") or []],
            timeline=Timeline.from_dict(data.get("timeline") or {}),
            defaults=Defaults.from_dict(data.get("defaults") or {}),
            cast=Cast.from_dict(data.get("cast") or []),
            created=data.get("created", utc_now_iso()),
            modified=data.get("modified", utc_now_iso()),
            render_profile=data.get("render_profile"),
            lineage=str(data.get("lineage") or ""),
            derived_from=str(data.get("derived_from") or ""),
            settings=dict(data.get("settings") or {}),
        )


@dataclass(frozen=True)
class Entry:
    """One project as a picker sees it: enough to sort, group and label a row.

    Deliberately not a `Project` — a list of thirty of those is thirty parsed
    edits nobody is going to look at.
    """

    name: str
    shelf: str
    created: str
    modified: str
    lineage: str
    derived_from: str
    sources: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shelf": self.shelf,
            "created": self.created,
            "modified": self.modified,
            "lineage": self.lineage,
            "derived_from": self.derived_from,
            "sources": self.sources,
        }


@dataclass(frozen=True)
class Backup:
    tier: str
    path: Path
    modified: str
    size: int


class ProjectStore:
    """Projects under `root`, one directory each."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        # Listing summaries, keyed by (shelf, name) and stamped with the mtime
        # they were read at. See `entries`.
        self._entries: dict[tuple[str, str], tuple[int, Entry]] = {}

    # -- locating ---------------------------------------------------------

    def shelf_root(self, shelf: str = ACTIVE) -> Path:
        """The directory a shelf's projects live in."""
        if shelf not in _SHELF_DIRS:
            raise ProjectError(f"Not a shelf: {shelf!r}")
        subdir = _SHELF_DIRS[shelf]
        return self.root / subdir if subdir else self.root

    def audio_dir(self, name: str) -> Path:
        """Where a project keeps sound imported from outside.

        Inside the project rather than referenced where it was dragged from: a
        project that stops playing because somebody tidied their Downloads folder
        is not a project. Not created here — the importer makes it when there is
        finally something to put in it.
        """
        return self.path_for(name) / AUDIO_DIR

    def path_for(self, name: str, shelf: str | None = None) -> Path:
        """Where a project is, or where it would go.

        Naming a shelf answers the second question. Without one this searches
        active, then archived, then trashed, so every caller that only knows a
        name — the gallery, the stills store, a load — keeps working when the
        project has been shelved underneath it. A name nothing matches resolves
        to the active shelf, which is where a new project belongs.
        """
        safe = sanitize_project_name(name)
        if shelf is not None:
            return self.shelf_root(shelf) / safe
        for candidate in SHELVES:
            path = self.shelf_root(candidate) / safe
            if (path / PROJECT_JSON).is_file():
                return path
        return self.root / safe

    def exists(self, name: str, shelf: str | None = None) -> bool:
        return (self.path_for(name, shelf) / PROJECT_JSON).is_file()

    def shelf_of(self, name: str) -> str | None:
        """Which shelf a project sits on, or None when there is no such project."""
        safe = sanitize_project_name(name)
        for candidate in SHELVES:
            if (self.shelf_root(candidate) / safe / PROJECT_JSON).is_file():
                return candidate
        return None

    def list(self, shelf: str = ACTIVE) -> list[str]:
        """Names on one shelf. Opens nothing — a shelf is a directory."""
        root = self.shelf_root(shelf)
        if not root.is_dir():
            return []
        names = [
            entry.name
            for entry in root.iterdir()
            if entry.is_dir()
            and entry.name not in (ARCHIVE_DIRNAME, TRASH_DIRNAME)
            and (entry / PROJECT_JSON).is_file()
        ]
        return sorted(names, key=str.casefold)

    def entries(self, shelves: Iterable[str] = SHELVES) -> list["Entry"]:
        """Every project on the named shelves, with the bits a picker shows.

        This is the one listing that opens files, because grouping a version
        family needs the lineage id inside each project. It is a read per
        project and it is memoised against the file's mtime, so the editor
        reopening the manager costs nothing until something is written.
        """
        found: list[Entry] = []
        for shelf in shelves:
            for name in self.list(shelf):
                found.append(self._entry(name, shelf))
        return found

    # -- locking ----------------------------------------------------------

    def lock_for(self, name: str) -> threading.Lock:
        key = sanitize_project_name(name)
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def locked(self, name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run `fn` with this project's lock held.

        Read-modify-write cycles must go through here, or concurrent saves
        silently drop each other's shots.
        """
        with self.lock_for(name):
            return fn(*args, **kwargs)

    # -- reading and writing ----------------------------------------------

    def create(self, name: str, sources: list[Source] | None = None) -> Project:
        """A new project on the active shelf, one clip per source given.

        The shelf is named rather than looked up. Without that, a name that is
        currently sitting in the trash would resolve there and the new project
        would be written straight over the trashed one — and a trash can that
        stops you reusing a name is not a trash can.
        """
        safe = sanitize_project_name(name)
        if self.exists(safe, ACTIVE):
            raise ProjectError(f"Project already exists: {safe}")
        project = Project(name=safe, sources=list(sources or []))
        for source in project.sources:
            project.timeline.add(Clip(source_id=source.id))
        self.save(project, shelf=ACTIVE)
        return project

    def load(self, name: str) -> Project:
        path = self.path_for(name) / PROJECT_JSON
        if not path.is_file():
            raise ProjectError(f"Project not found: {name}")
        with path.open("r", encoding="utf-8") as handle:
            return Project.from_dict(json.load(handle))

    def save(self, project: Project, shelf: str | None = None) -> Path:
        """Back up the previous state, then replace `project.json` atomically.

        Without a `shelf` the project is written wherever it already lives, so
        editing an archived project keeps it archived. `create` is the one
        caller that names a shelf, because a project that does not exist yet
        has nowhere to be found.
        """
        directory = self.path_for(project.name, shelf)
        directory.mkdir(parents=True, exist_ok=True)
        project.modified = utc_now_iso()
        # A project written before version families existed joins one — its
        # own — the first time it is saved, rather than in a migration pass
        # nobody asked for.
        if not project.lineage:
            project.lineage = new_lineage_id()

        self._backup(directory, red_flag=self._is_shrinking(directory, project))
        return write_json_atomic(directory / PROJECT_JSON, project.to_dict())

    def update(self, name: str, mutate: Callable[[Project], None]) -> Project:
        """Load, mutate and save under one lock. The safe way to edit."""

        def apply() -> Project:
            project = self.load(name)
            mutate(project)
            self.save(project)
            return project

        return self.locked(name, apply)

    def delete(self, name: str) -> bool:
        """Erase a project's directory outright, from whichever shelf it is on.

        The blunt instrument, kept for API callers that mean it. The editor
        goes through `shelve` and `purge` instead, so a click is always
        recoverable until the trash is emptied.
        """
        directory = self.path_for(name)
        if not directory.is_dir():
            return False
        shutil.rmtree(directory)
        self._forget(name)
        return True

    # -- shelves ----------------------------------------------------------

    def shelve(self, name: str, shelf: str) -> str:
        """Move a project to a shelf. Returns the name it ended up under.

        The name only changes when the destination shelf already holds one — a
        project trashed, remade and trashed again — in which case the arriving
        copy is suffixed rather than silently overwriting what is already
        there. `project.json` is rewritten to match, so the name inside the
        file never disagrees with the directory it is in.
        """
        if shelf not in _SHELF_DIRS:
            raise ProjectError(f"Not a shelf: {shelf!r}")
        was = self.shelf_of(name)
        if was is None:
            raise ProjectError(f"Project not found: {name}")

        source = self.path_for(name, was)
        if was == shelf:
            return source.name

        destination = self.shelf_root(shelf)
        destination.mkdir(parents=True, exist_ok=True)
        target = _free_path(destination, source.name)

        with self.lock_for(name):
            shutil.move(str(source), str(target))
        self._forget(name)

        if target.name != source.name:
            self._rename_in_place(target)
        return target.name

    def purge(self, name: str) -> bool:
        """Delete a trashed project for good. Refuses anything not in the trash.

        This is the only method that erases without a way back, so it is fenced
        twice: the project must be on the trash shelf, and the directory it
        resolves to must physically sit inside `.trash`. What it removes is the
        project's own JSON and its backups. Rendered videos live in ComfyUI's
        output directory and are only ever pointed at from `takes.json`, so
        nothing here can reach them.
        """
        if self.shelf_of(name) != TRASHED:
            return False
        directory = self.path_for(name, TRASHED)
        if not _is_within(directory, self.shelf_root(TRASHED)):
            raise ProjectError("Refusing to purge from outside the trash.")
        if not directory.is_dir():
            return False
        shutil.rmtree(directory)
        self._forget(name)
        return True

    def empty_trash(self) -> list[str]:
        """Purge every trashed project. Returns what went."""
        gone: list[str] = []
        for name in self.list(TRASHED):
            if self.purge(name):
                gone.append(name)
        return gone

    def duplicate(self, name: str, new_name: str | None = None) -> Project:
        """Copy a project into a new one on the active shelf, same family.

        The copy keeps the original's `lineage`, which is what makes the two of
        them a version chain rather than two unrelated projects, and records
        `derived_from` so a row can say where it came from. Takes and stills
        come along — they are pointers, so the copy opens with the same footage
        and image library instead of an empty gallery — while the original's
        backups stay behind.
        """
        source_dir = self.path_for(name)
        if not (source_dir / PROJECT_JSON).is_file():
            raise ProjectError(f"Project not found: {name}")

        original = self.load(name)
        if not original.lineage:
            original.lineage = new_lineage_id()
            self.save(original)

        wanted = sanitize_project_name(new_name) if new_name else None
        taken = [entry.name for entry in self.entries()]
        target = _free_path(self.root, wanted or _next_version_name(taken, source_dir.name))
        if wanted and target.name != wanted:
            raise ProjectError(f"Project {wanted!r} already exists.")

        copy = Project.from_dict(original.to_dict())
        copy.name = target.name
        copy.lineage = original.lineage
        copy.derived_from = source_dir.name
        copy.created = utc_now_iso()
        # A fresh copy has not been edited since it was made. Carrying the
        # original's `modified` across would date the copy before its own
        # creation, which reads as nonsense in a version chain.
        copy.modified = copy.created

        target.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target / PROJECT_JSON, copy.to_dict())
        for sidecar in SIDECARS:
            beside = source_dir / sidecar
            if beside.is_file():
                shutil.copy2(beside, target / sidecar)
        self._forget(target.name)
        return copy

    # -- listing internals -------------------------------------------------

    def _entry(self, name: str, shelf: str) -> Entry:
        path = self.shelf_root(shelf) / name / PROJECT_JSON
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            stamp = 0
        key = (shelf, name)
        cached = self._entries.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]

        data: dict[str, Any] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            # A project we cannot parse still deserves a row, or it becomes
            # invisible and unrecoverable.
            pass

        entry = Entry(
            name=name,
            shelf=shelf,
            created=str(data.get("created") or ""),
            modified=str(data.get("modified") or ""),
            lineage=str(data.get("lineage") or ""),
            derived_from=str(data.get("derived_from") or ""),
            sources=len(data.get("sources") or data.get("shots") or []),
        )
        self._entries[key] = (stamp, entry)
        return entry

    def _forget(self, name: str) -> None:
        safe = sanitize_project_name(name)
        for shelf in SHELVES:
            self._entries.pop((shelf, safe), None)

    def _rename_in_place(self, directory: Path) -> None:
        """Make `project.json`'s name agree with the directory it now sits in."""
        path = directory / PROJECT_JSON
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        data["name"] = directory.name
        write_json_atomic(path, data)

    # -- backups ----------------------------------------------------------

    def list_backups(self, name: str) -> list[Backup]:
        """Every snapshot for a project, newest first."""
        backups_dir = self.path_for(name) / BACKUPS_DIRNAME
        if not backups_dir.is_dir():
            return []

        found: list[Backup] = []
        for tier, subdir, pattern in _BACKUP_TIERS:
            directory = backups_dir / subdir if subdir else backups_dir
            if not directory.is_dir():
                continue
            for path in directory.glob(pattern):
                stat = path.stat()
                found.append(
                    Backup(
                        tier=tier,
                        path=path,
                        modified=datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).replace(microsecond=0).isoformat(),
                        size=stat.st_size,
                    )
                )
        return sorted(found, key=lambda backup: backup.modified, reverse=True)

    def restore(self, name: str, backup: Path) -> Project:
        """Restore a snapshot. The pre-restore state is itself backed up first."""
        backup = Path(backup)
        directory = self.path_for(name)
        if not _is_within(backup, directory / BACKUPS_DIRNAME):
            raise ProjectError("Refusing to restore from outside the project's backups.")
        if not backup.is_file():
            raise ProjectError(f"No such backup: {backup.name}")

        with backup.open("r", encoding="utf-8") as handle:
            restored = Project.from_dict(json.load(handle))

        restored.name = sanitize_project_name(name)
        self._backup(directory, red_flag=True)
        write_json_atomic(directory / PROJECT_JSON, restored.to_dict())
        return restored

    # -- internals --------------------------------------------------------

    def _is_shrinking(self, directory: Path, project: Project) -> bool:
        """True when this save drops a large share of the existing content.

        Not a veto — the save still happens — but it earns a snapshot in the
        tier that is not churned by ordinary edits.
        """
        path = directory / PROJECT_JSON
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                previous = Project.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError):
            return False

        old_sources, new_sources = len(previous.sources), len(project.sources)
        old_beats = _total_beats(previous)
        new_beats = _total_beats(project)

        return (old_sources >= 2 and new_sources < old_sources * SHRINK_RATIO) or (
            old_beats >= 5 and new_beats < old_beats * SHRINK_RATIO
        )

    def _backup(self, directory: Path, red_flag: bool = False) -> None:
        """Snapshot the current project.json. Never allowed to block a save."""
        source = directory / PROJECT_JSON
        if not source.is_file():
            return

        try:
            backups = directory / BACKUPS_DIRNAME
            backups.mkdir(parents=True, exist_ok=True)
            now = datetime.now()
            stamp = now.strftime("%Y%m%d_%H%M%S_%f")

            shutil.copy2(source, backups / f"project_{stamp}.json")
            _trim(sorted(backups.glob("project_*.json")), SAVE_BACKUP_LIMIT)

            _tier_snapshot(
                source,
                backups / "hourly",
                f"project_{now.strftime('%Y%m%d_%H')}.json",
                timedelta(hours=HOURLY_BACKUP_HOURS),
            )
            _tier_snapshot(
                source,
                backups / "daily",
                f"project_{now.strftime('%Y%m%d')}.json",
                timedelta(days=DAILY_BACKUP_DAYS),
            )

            if red_flag:
                shutil.copy2(source, backups / f"redflag_{stamp}.json")
                _trim(sorted(backups.glob("redflag_*.json")), REDFLAG_BACKUP_LIMIT)
        except OSError:
            # A failed backup must never cost the user their save.
            pass


_BACKUP_TIERS = (
    ("save", "", "project_*.json"),
    ("hourly", "hourly", "project_*.json"),
    ("daily", "daily", "project_*.json"),
    ("redflag", "", "redflag_*.json"),
)


def _total_beats(project: Project) -> int:
    return sum(
        len(segment.prompt.storyboard.beats)
        for source in project.sources
        for segment in source.segments
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    """Write via a temp file in the same directory, then rename over the target."""
    import time

    fd, temp_path = tempfile.mkstemp(prefix="project.", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        for attempt in range(5):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    return path


def _tier_snapshot(source: Path, directory: Path, filename: str, max_age: timedelta) -> None:
    """One snapshot per period, with age-based pruning."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.exists():
        return
    shutil.copy2(source, target)
    _prune_stamped(directory, "project_*.json", max_age)


def _prune_stamped(directory: Path, pattern: str, max_age: timedelta) -> None:
    """Delete snapshots whose filename stamp is older than `max_age`.

    Unparseable filenames are left alone — better a stale file than a deleted
    one we did not understand.
    """
    cutoff = datetime.now() - max_age
    for snapshot in directory.glob(pattern):
        stamp = snapshot.stem.split("_", 1)[-1]
        for fmt in ("%Y%m%d_%H", "%Y%m%d"):
            try:
                if datetime.strptime(stamp, fmt) < cutoff:
                    snapshot.unlink()
                break
            except ValueError:
                continue
            except OSError:
                break


def _trim(paths: list[Path], limit: int) -> None:
    for stale in paths[:-limit] if limit else paths:
        try:
            stale.unlink()
        except OSError:
            pass


# " v2", " V10", "_v3" at the end of a name. Matching what the author already
# types by hand is the point: a project called "PLANET OFFICE v2" duplicates to
# "PLANET OFFICE v3", not to "PLANET OFFICE v2 v2".
_VERSION_SUFFIX = re.compile(r"^(?P<stem>.+?)[ _]v(?P<number>\d+)$", re.IGNORECASE)


def _version_stem(name: str) -> tuple[str, int]:
    """Split a name into its family stem and version number. Unnumbered is v1."""
    match = _VERSION_SUFFIX.match(name)
    if match is None:
        return name, 1
    return match.group("stem"), int(match.group("number"))


def _next_version_name(taken: Iterable[str], base: str) -> str:
    """The next free `<stem> vN` for whatever family `base` belongs to."""
    stem, _ = _version_stem(base)
    highest = 1
    for name in taken:
        other, number = _version_stem(name)
        if other.casefold() == stem.casefold():
            highest = max(highest, number)
    return f"{stem} v{highest + 1}"


def _free_path(directory: Path, name: str) -> Path:
    """`directory / name`, suffixed until nothing is there to overwrite.

    The suffix is " - 2", not " (2)", because a directory name here is also a
    project name and has to survive `sanitize_project_name` unchanged — which
    strips brackets but keeps spaces and hyphens. A name that did not survive
    would be findable on disk and unloadable by name.
    """
    candidate = directory / name
    if not candidate.exists():
        return candidate
    for suffix in range(2, 1000):
        candidate = directory / f"{name} - {suffix}"
        if not candidate.exists():
            return candidate
    raise ProjectError(f"Cannot find a free name for {name!r}.")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True
