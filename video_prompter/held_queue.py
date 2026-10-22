"""Persistent render jobs that have not entered ComfyUI's queue yet."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import storage_root

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = (
    "Interrupted while being sent to ComfyUI. Check the ComfyUI queue before "
    "releasing it again, or it may render twice."
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_release_at(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("release_at must be an ISO date-time or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("release_at must be a valid ISO date-time") from error
    if parsed.tzinfo is None:
        raise ValueError("release_at must include a timezone")
    return parsed.astimezone(timezone.utc)


class HeldQueue:
    """Small atomic JSON store for frozen prompt payloads."""

    def __init__(self, path: Path | None = None):
        self.path = path or (storage_root() / "held_queue.json")
        self._lock = threading.RLock()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError:
            return []
        except json.JSONDecodeError:
            # Reading this as empty is fine; writing that back is not — the
            # next hold would replace every frozen job with one. Move the
            # damaged file aside first so its contents can still be recovered.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            aside = self.path.with_name(f"{self.path.stem}.corrupt-{stamp}{self.path.suffix}")
            try:
                os.replace(self.path, aside)
                logger.error("held queue file was unreadable; moved it to %s", aside)
            except OSError:
                logger.exception("held queue file is unreadable and could not be moved aside")
            return []
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        return [job for job in jobs if isinstance(job, dict)]

    def _write(self, jobs: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".held_queue-", suffix=".json", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "jobs": jobs}, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._read()

    def add(self, payload: dict[str, Any], summary: dict[str, Any], release_at: Any = None) -> dict[str, Any]:
        if not isinstance(payload.get("prompt"), dict):
            raise ValueError("Held job requires a ComfyUI prompt workflow")
        parsed_release = parse_release_at(release_at)
        job = {
            "id": str(uuid.uuid4()),
            "created_at": utc_now_iso(),
            "release_at": parsed_release.isoformat() if parsed_release else None,
            "status": "scheduled" if parsed_release else "held",
            "attempts": 0,
            "last_error": "",
            "payload": payload,
            "summary": summary if isinstance(summary, dict) else {},
        }
        with self._lock:
            jobs = self._read()
            jobs.append(job)
            self._write(jobs)
        return job

    def update_schedule(self, job_id: str, release_at: Any) -> dict[str, Any]:
        parsed_release = parse_release_at(release_at)
        with self._lock:
            jobs = self._read()
            for job in jobs:
                if job.get("id") == job_id:
                    if job.get("status") == "dispatching":
                        raise ValueError("Held render is already being dispatched.")
                    job["release_at"] = parsed_release.isoformat() if parsed_release else None
                    job["status"] = "scheduled" if parsed_release else "held"
                    job["last_error"] = ""
                    self._write(jobs)
                    return job
        raise KeyError(job_id)

    def update_schedule_all(self, release_at: Any) -> list[dict[str, Any]]:
        """Give every waiting job the same release time, in a single write.

        A shelf is filled one render at a time but emptied as one decision —
        "run the lot at 1am" — so setting that per job is 28 round trips for
        one intent. Jobs already on their way into ComfyUI are skipped: the
        dispatcher is mid-flight with them and a new time would not be read.
        """
        parsed_release = parse_release_at(release_at)
        stamp = parsed_release.isoformat() if parsed_release else None
        status = "scheduled" if parsed_release else "held"
        with self._lock:
            jobs = self._read()
            changed = []
            for job in jobs:
                if job.get("status") == "dispatching":
                    continue
                job["release_at"] = stamp
                job["status"] = status
                job["last_error"] = ""
                changed.append(job)
            if changed:
                self._write(jobs)
            return changed

    def remove(self, job_id: str) -> bool:
        with self._lock:
            jobs = self._read()
            kept = [job for job in jobs if job.get("id") != job_id]
            if len(kept) == len(jobs):
                return False
            self._write(kept)
            return True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return next((job for job in self._read() if job.get("id") == job_id), None)

    def due(self, now: datetime | None = None) -> list[dict[str, Any]]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        due_jobs = []
        with self._lock:
            for job in self._read():
                if job.get("status") != "scheduled":
                    continue
                release = parse_release_at(job.get("release_at"))
                if release and release <= current:
                    due_jobs.append(job)
        return due_jobs

    def mark_dispatching(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            jobs = self._read()
            for job in jobs:
                if job.get("id") == job_id:
                    if job.get("status") not in ("held", "scheduled", "error"):
                        raise ValueError("Held render is already being dispatched.")
                    job["status"] = "dispatching"
                    job["attempts"] = int(job.get("attempts", 0)) + 1
                    job["last_error"] = ""
                    self._write(jobs)
                    return job
        raise KeyError(job_id)

    def recover_interrupted(self) -> list[dict[str, Any]]:
        """Unstick jobs a previous process left mid-dispatch.

        Call once at startup, before anything can dispatch: a job still
        "dispatching" then belongs to a server that died during the POST, and
        nothing would ever move it on — it cannot be released, rescheduled or
        retried. It becomes an error rather than being resubmitted, because
        whether ComfyUI accepted it before the crash is unknown.
        """
        with self._lock:
            jobs = self._read()
            stuck = [job for job in jobs if job.get("status") == "dispatching"]
            for job in stuck:
                job["status"] = "error"
                job["last_error"] = INTERRUPTED_MESSAGE
            if stuck:
                self._write(jobs)
            return stuck

    def mark_error(self, job_id: str, message: str) -> dict[str, Any]:
        with self._lock:
            jobs = self._read()
            for job in jobs:
                if job.get("id") == job_id:
                    job["status"] = "error"
                    job["last_error"] = str(message)
                    self._write(jobs)
                    return job
        raise KeyError(job_id)
