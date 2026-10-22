"""Render sources on the local GPU, end to end: prepare, queue, wait, record, pick.

The video-prompt skill used to describe this loop in prose and leave the agent
to write it — so every session wrote it again, and the repo root filled up with
near-identical ``render_*.py`` files. This is that loop, once. It sends exactly
what the editor sends (``buildPromptPayload`` in ``js/standalone/main.js``), so
a take rendered from here is indistinguishable from one rendered by the button.

    # fresh render of one source at its segment's own tier
    python tools/render_sources.py --project "Night run" --source src_1a2b

    # every source in the bin, as quick previews
    python tools/render_sources.py --project "Night run" --all --quality preview

    # a four-seed sweep (takes are recorded, not picked — choosing is yours)
    python tools/render_sources.py --project "Night run" --source src_1a2b --count 4

    # continue a take; the text defaults to the next segment's composed prompt
    python tools/render_sources.py --project "Night run" --extend "PROMPT_ID:take.mp4"

    # re-render a picked draft from its frozen text at the keeper tier
    python tools/render_sources.py --project "Night run" --reshoot "PROMPT_ID:take.mp4" --quality final

Everything is prepared and queued first, then collected as it finishes, so a
batch keeps the GPU busy. Progress goes to stderr; one JSON summary goes to
stdout at the end. Exit status is 0 only when every render succeeded.

Renders take minutes — run it in the background and read the summary when it
exits, rather than blocking on it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8188"
CLIENT_ID = "agent-render"
POLL_SECONDS = 5.0
PROGRESS_EVERY = 30.0

# The editor's own names for each tier, which it writes into the queue
# metadata's `mode`. Kept identical so the queue reads the same either way.
QUALITY_LABELS = {
    "preview": "Quick", "turbo_hd": "Turbo HD", "turbo_2pass": "Turbo 2-pass",
    "selflift_hd": "SelfLift HD", "selflift_fhd": "SelfLift FHD",
    "selflift_final": "SelfLift Final", "final": "Fine", "final_2pass": "Fine 2-pass",
    "singularity_2pass": "Singularity 2-pass",
}


class ServerError(Exception):
    def __init__(self, status: int, message: str, body: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status
        self.body = body or {}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class Client:
    def __init__(self, url: str):
        self.root = url.rstrip("/")
        self.api = f"{self.root}/video_prompter"

    def _call(self, method: str, url: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = {"error": raw.decode("utf-8", "replace")}
            message = parsed.get("error")
            if isinstance(message, dict):
                message = message.get("message")
            raise ServerError(error.code, str(message or error.reason), parsed) from None

    def get(self, path: str, **query: str) -> Any:
        suffix = f"?{urllib.parse.urlencode(query)}" if query else ""
        return self._call("GET", f"{self.api}{path}{suffix}")

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self._call("POST", f"{self.api}{path}", body)

    def comfy_get(self, path: str) -> Any:
        return self._call("GET", f"{self.root}{path}")

    def comfy_post(self, path: str, body: dict[str, Any]) -> Any:
        return self._call("POST", f"{self.root}{path}", body)


@dataclass
class Job:
    """One prepared render on its way through ComfyUI."""

    source_id: str
    source_name: str
    label: str
    prepared: dict[str, Any]
    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    queued_at: float = 0.0
    status: str = "prepared"
    error: str = ""
    takes: list[str] = field(default_factory=list)
    picked: str = ""
    seconds_taken: float | None = None

    def payload(self, project: str) -> dict[str, Any]:
        """The `/prompt` body, field for field what the editor submits."""
        p = self.prepared
        meta = {
            "project": project,
            "shot": self.source_id,  # the key is `shot`; the value is a source id
            "shot_name": self.source_name,
            "mode": self.label,
            "frames": p.get("frames"),
            "seconds": p.get("seconds"),
            "seed": p.get("seed"),
            "prompt": p.get("prompt_text") or p.get("prompt") or "",
            "width": p.get("width"),
            "height": p.get("height"),
            "quality": p.get("quality") or None,
            "steps": p.get("steps"),
            "pass1_steps": p.get("pass1_steps") or None,
            "loras": p.get("loras"),
            "latent_path": p.get("latent_path"),
            "parent_take": p.get("parent_take") or None,
            "context_length": p.get("context_length") or None,
        }
        return {
            "prompt": p["workflow"],
            "client_id": CLIENT_ID,
            "prompt_id": self.prompt_id,
            "extra_data": {"video_prompter": {k: v for k, v in meta.items() if v is not None}},
        }

    def summary(self) -> dict[str, Any]:
        p = self.prepared
        return {
            "source": self.source_id,
            "source_name": self.source_name,
            "status": self.status,
            "error": self.error or None,
            "prompt_id": self.prompt_id,
            "quality": p.get("quality"),
            "mode": p.get("mode"),
            "seed": p.get("seed"),
            "width": p.get("width"),
            "height": p.get("height"),
            "seconds": p.get("seconds"),
            "takes": self.takes,
            "picked": self.picked or None,
            "render_seconds": self.seconds_taken,
        }


# -- choosing what to render ------------------------------------------------


def choose_profile(client: Client, project: dict[str, Any], wanted: str | None) -> str:
    """--profile, else the project's own, else the only one there is."""
    names = client.get("/profiles").get("profiles", [])
    if wanted:
        if wanted not in names:
            raise SystemExit(f"No render profile {wanted!r}. Have: {', '.join(names)}")
        return wanted
    if project.get("render_profile") in names:
        return project["render_profile"]
    if len(names) == 1:
        return names[0]
    raise SystemExit(
        "Several render profiles and the project names none, so say which with "
        f"--profile (or set the project's render_profile): {', '.join(names)}"
    )


def find_take(client: Client, project: str, take_id: str) -> dict[str, Any]:
    for take in client.get("/gallery", project=project).get("takes", []):
        if take.get("id") == take_id or f"{take.get('prompt_id')}:{take.get('filename')}" == take_id:
            return take
    raise SystemExit(f"No take {take_id!r} in {project!r}.")


def take_depth(client: Client, project: str, take: dict[str, Any]) -> int:
    """How many continuations sit above this take: 0 for a fresh render."""
    by_id = {
        f"{t.get('prompt_id')}:{t.get('filename')}": t
        for t in client.get("/gallery", project=project).get("takes", [])
    }
    depth, cursor, seen = 0, take, set()
    while cursor.get("parent_take") and cursor["parent_take"] in by_id:
        if cursor["parent_take"] in seen:
            break
        seen.add(cursor["parent_take"])
        cursor = by_id[cursor["parent_take"]]
        depth += 1
    return depth


def batch_or_single(client: Client, path: str, body: dict[str, Any], count: int) -> list[dict[str, Any]]:
    if count == 1:
        return [client.post(path, body)]
    # A fresh render's batch is `/render/batch`, not `/render/prepare/batch`.
    batch_path = "/render/batch" if path == "/render/prepare" else f"{path}/batch"
    batch = client.post(batch_path, {**body, "count": count})
    # A batch states the mode once for the run, not on each render.
    return [{"mode": batch.get("mode"), **render} for render in batch.get("renders", [])]


def prepare_fresh(client, args, project, profile) -> list[Job]:
    sources = project["sources"]
    if not args.all:
        wanted = set(args.source)
        sources = [s for s in sources if s["id"] in wanted or s.get("name") in wanted]
        missing = wanted - {s["id"] for s in sources} - {s.get("name") for s in sources}
        if missing:
            raise SystemExit(f"No source {', '.join(sorted(missing))} in {project['name']!r}.")

    jobs: list[Job] = []
    for source in sources:
        segment = source["segments"][0]
        tier = args.quality or segment.get("quality") or "preview"
        body = {
            "prompt": segment["prompt"], "profile": profile, "quality": tier,
            "project": project["name"],
            "source_id": source["id"],
        }
        if args.seed is not None:
            body["seed"] = args.seed
        try:
            renders = batch_or_single(client, "/render/prepare", body, args.count)
        except ServerError as error:
            jobs.append(refused(source, error))
            continue
        for prepared in renders:
            prepared.setdefault("quality", tier)
            jobs.append(Job(source["id"], source.get("name", ""), QUALITY_LABELS.get(tier, "Quick"), prepared))
    return jobs


def prepare_extend(client, args, project, profile) -> list[Job]:
    name = project["name"]
    take = find_take(client, name, args.extend)
    source_id = take.get("shot", "")
    source = next((s for s in project["sources"] if s["id"] == source_id), None)
    if source is None:
        raise SystemExit(f"Take {args.extend!r} belongs to no source in {name!r}.")

    depth = args.depth if args.depth is not None else take_depth(client, name, take) + 1
    unplanned = (f"{source.get('name') or source_id} has no segment {depth} authored to continue "
                 "into. Plan it first (POST /source/segment/add with its `prompt`), or pass --text.")
    if depth >= len(source["segments"]):
        if not args.text:
            raise SystemExit(unplanned)
        # The editor's own rule: an extend past the plan grows the plan first,
        # so intent and result stay the same length.
        client.post("/source/segment/add", {"project": name, "source": source_id,
                                            "quality": args.quality or None})
        source = client.get("/source", project=name, source=source_id)["source"]
    segment = source["segments"][min(depth, len(source["segments"]) - 1)]
    tier = args.quality or segment.get("quality") or "preview"

    text = args.text
    if not text:
        if not segment["prompt"].get("beats"):
            raise SystemExit(unplanned)  # a blank plan composes to nothing
        text = client.post("/preview", {"prompt": segment["prompt"], "project": name}).get("prompt", "")
    take_ref = f"{take['subfolder'] + '/' if take.get('subfolder') else ''}{take['filename']} [{take.get('type') or 'output'}]"
    body: dict[str, Any] = {
        "source": take_ref, "parent_take": args.extend, "prompt_text": text,
        "source_id": source_id,
        "project": name, "quality": tier, "profile": profile, "prompt": segment["prompt"],
        "width": take.get("width") or segment["prompt"].get("width") or 608,
        "height": take.get("height") or segment["prompt"].get("height") or 352,
    }
    if segment["prompt"].get("duration"):
        body["seconds"] = segment["prompt"]["duration"]
    else:
        body["frames"] = take.get("frames") or 124
        body["seconds"] = take.get("seconds") or 5.167
    if args.seed is not None:
        body["seed"] = args.seed
    if args.context:
        body["context"] = args.context
    if take.get("latent_path"):
        body["latent"] = take["latent_path"]

    try:
        renders = batch_or_single(client, "/render/extend", body, args.count)
    except ServerError as error:
        return [refused(source, error)]
    jobs = []
    for prepared in renders:
        prepared["parent_take"] = args.extend  # every candidate continues the same parent
        prepared.setdefault("quality", tier)
        jobs.append(Job(source_id, source.get("name", ""), "Extend", prepared))
    return jobs


def prepare_reshoot(client, args, project, profile) -> list[Job]:
    name = project["name"]
    take = find_take(client, name, args.reshoot)
    source = next((s for s in project["sources"] if s["id"] == take.get("shot")), None)
    if source is None:
        raise SystemExit(f"Take {args.reshoot!r} belongs to no source in {name!r}.")
    tier = args.quality or take.get("quality") or "final"
    body = {
        "prompt_text": take["prompt_text"], "width": take["width"], "height": take["height"],
        "frames": take["frames"], "seconds": take["seconds"],
        "seed": args.seed if args.seed is not None else take.get("seed"),
        "profile": profile, "quality": tier, "project": name,
        "source_id": source["id"],
        # Sent for its media only: a take freezes its words, never its keyframes
        # or references, so without this a ref2v take re-shoots text-only.
        "prompt": source["segments"][0]["prompt"],
    }
    try:
        prepared = client.post("/render/reshoot", body)
    except ServerError as error:
        return [refused(source, error)]
    prepared.setdefault("quality", tier)
    return [Job(source["id"], source.get("name", ""), QUALITY_LABELS.get(tier, "Reroll"), prepared)]


def refused(source: dict[str, Any], error: ServerError) -> Job:
    job = Job(source["id"], source.get("name", ""), "", {})
    job.status = "refused"
    if error.status == 422:
        errors = [f"{f.get('rule')}: {f.get('message')}" for f in error.body.get("findings", [])
                  if f.get("severity") == "error"]
        job.error = "lint errors — " + "; ".join(errors or [str(error)])
    elif error.status == 409:
        job.error = f"profile drifted from its workflow; re-capture it ({error})"
    else:
        job.error = f"{error.status}: {error}"
    log(f"  refused  {job.source_name or job.source_id}: {job.error}")
    return job


# -- running them -----------------------------------------------------------


def elapsed_from(entry: dict[str, Any]) -> float | None:
    """Generation time off ComfyUI's own execution timestamps."""
    stamps = {kind: data.get("timestamp") for kind, data in entry.get("status", {}).get("messages", [])
              if isinstance(data, dict)}
    start, end = stamps.get("execution_start"), stamps.get("execution_success")
    return round((end - start) / 1000, 1) if start and end else None


def progress_line(client: Client, jobs: dict[str, Job]) -> str | None:
    try:
        snapshot = client.get("/render/progress")
    except (ServerError, OSError):
        return None
    job = jobs.get(snapshot.get("prompt_id", ""))
    if job is None:
        return None
    running = [n for n in snapshot.get("nodes", {}).values() if n.get("state") == "running"]
    if not running:
        return None
    node = max(running, key=lambda n: n.get("max", 0))
    node_id = str(node.get("display_node_id") or node.get("node_id"))
    where = job.prepared.get("workflow", {}).get(node_id, {}).get("class_type") or node_id
    if node.get("max", 0) > 1:
        return f"  running  {job.source_name or job.source_id}: {where} {node['value']:.0f}/{node['max']:.0f}"
    return f"  running  {job.source_name or job.source_id}: {where}"


def run(client: Client, project: str, jobs: list[Job], pick: bool) -> None:
    for job in jobs:
        if job.status != "prepared":
            continue
        try:
            answer = client.comfy_post("/prompt", job.payload(project))
        except ServerError as error:
            job.status, job.error = "failed", f"ComfyUI refused the queue: {error}"
            log(f"  failed   {job.source_name}: {job.error}")
            continue
        job.prompt_id = answer.get("prompt_id") or job.prompt_id
        job.queued_at, job.status = time.time(), "queued"
        log(f"  queued   {job.source_name or job.source_id} [{job.prepared.get('mode')}, "
            f"{job.prepared.get('quality')}, seed {job.prepared.get('seed')}] {job.prompt_id}")

    waiting = {job.prompt_id: job for job in jobs if job.status == "queued"}
    # Pick only when each source rendered exactly one take. A sweep leaves the
    # choice to whoever looks at the candidates.
    per_source: dict[str, int] = {}
    for job in waiting.values():
        per_source[job.source_id] = per_source.get(job.source_id, 0) + 1

    last_progress = 0.0
    while waiting:
        time.sleep(POLL_SECONDS)
        for prompt_id, job in list(waiting.items()):
            try:
                entry = client.comfy_get(f"/history/{prompt_id}").get(prompt_id)
            except (ServerError, OSError):
                continue
            if not entry:
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                messages = [data.get("exception_message", "") for kind, data in status.get("messages", [])
                            if kind == "execution_error" and isinstance(data, dict)]
                job.status, job.error = "failed", "; ".join(m.strip() for m in messages if m) or "render error"
                log(f"  failed   {job.source_name}: {job.error}")
                del waiting[prompt_id]
                continue
            if not entry.get("outputs") and not status.get("completed"):
                continue
            job.seconds_taken = elapsed_from(entry) or round(time.time() - job.queued_at, 1)
            del waiting[prompt_id]
            collect(client, project, job, entry, pick and per_source[job.source_id] == 1)

        if waiting and time.time() - last_progress >= PROGRESS_EVERY:
            line = progress_line(client, waiting)
            if line:
                log(line)
            last_progress = time.time()


def collect(client: Client, project: str, job: Job, entry: dict[str, Any], pick: bool) -> None:
    try:
        recorded = client.post("/gallery/record", {
            "project": project,
            "history": {job.prompt_id: entry},
            "generation_times": {job.prompt_id: job.seconds_taken},
        })
    except ServerError as error:
        job.status, job.error = "failed", f"rendered but not recorded: {error}"
        log(f"  failed   {job.source_name}: {job.error}")
        return
    job.takes = [t["id"] for t in recorded.get("added", []) if "id" in t] or [
        f"{t.get('prompt_id')}:{t.get('filename')}" for t in recorded.get("added", [])]
    job.status = "done"
    if pick and len(job.takes) == 1:
        try:
            client.post("/gallery/pick", {"project": project, "source": job.source_id, "take": job.takes[0]})
            job.picked = job.takes[0]
        except ServerError as error:
            job.error = f"recorded, but not picked: {error}"
    log(f"  done     {job.source_name or job.source_id} in {job.seconds_taken}s -> "
        f"{', '.join(job.takes) or 'no take recorded'}{' (picked)' if job.picked else ''}")


# -- entry point ------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--project", required=True)
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--source", action="append", default=[], metavar="ID_OR_NAME",
                      help="a source to render fresh (repeatable)")
    what.add_argument("--all", action="store_true", help="render every source in the bin")
    what.add_argument("--extend", metavar="TAKE_ID", help="continue this take")
    what.add_argument("--reshoot", metavar="TAKE_ID", help="re-render this take's frozen text")
    p.add_argument("--quality", help="tier name; default is the segment's own tier")
    p.add_argument("--profile", help="render profile; default is the project's, or the only one")
    p.add_argument("--count", type=int, default=1, help="seeds per source, 1-8 (sweeps are not picked)")
    p.add_argument("--seed", type=int, help="the seed, or the base of a sweep")
    p.add_argument("--text", help="--extend only: continuation text (default: composed next segment)")
    p.add_argument("--depth", type=int, help="--extend only: which segment it fulfils")
    p.add_argument("--context", type=int, help="--extend only: frames of the parent to pin")
    p.add_argument("--no-pick", action="store_true", help="record takes without choosing them")
    p.add_argument("--dry-run", action="store_true", help="prepare and lint only; queue nothing")
    p.add_argument("--url", default=DEFAULT_URL)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    if not 1 <= args.count <= 8:
        raise SystemExit("--count must be 1-8.")

    client = Client(args.url)
    try:
        client.comfy_get("/queue")
    except (ServerError, OSError) as error:
        raise SystemExit(f"ComfyUI is not answering at {args.url}: {error}")

    project = client.get("/project/load", name=args.project)["project"]
    profile = choose_profile(client, project, args.profile)
    log(f"{project['name']}: profile {profile!r}")

    if args.extend:
        jobs = prepare_extend(client, args, project, profile)
    elif args.reshoot:
        jobs = prepare_reshoot(client, args, project, profile)
    else:
        jobs = prepare_fresh(client, args, project, profile)

    if args.dry_run:
        for job in jobs:
            if job.status == "prepared":
                p = job.prepared
                log(f"  ready    {job.source_name or job.source_id} [{p.get('mode')}, {p.get('quality')}, "
                    f"{p.get('width')}x{p.get('height')}, {p.get('seconds')}s, seed {p.get('seed')}]")
    else:
        run(client, project["name"], jobs, pick=not args.no_pick)

    print(json.dumps({"project": project["name"], "profile": profile,
                      "renders": [job.summary() for job in jobs]}, indent=1))
    ok = {"prepared"} if args.dry_run else {"done"}
    return 0 if jobs and all(job.status in ok for job in jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
