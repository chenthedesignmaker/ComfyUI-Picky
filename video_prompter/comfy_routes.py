"""aiohttp routes, registered on ComfyUI's own server.

A translation layer and nothing else: parse the request, call `Api`, turn an
`ApiError` into a status code. All the behaviour worth testing lives in
`video_prompter.api`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
from aiohttp import web
from server import PromptServer  # ComfyUI's own module

from .api import Api, ApiError
from .held_queue import HeldQueue

logger = logging.getLogger("ComfyUI.VideoPrompter")

ROUTE_PREFIX = "/video_prompter"
APP_DIR = Path(__file__).resolve().parent.parent / "js" / "standalone"

routes = PromptServer.instance.routes
_api = Api()
_held_queue = HeldQueue()
_held_dispatcher_task: asyncio.Task | None = None

# At import nothing in this process can be mid-dispatch, so any job still
# marked that way was orphaned by a previous server and would be stuck forever.
try:
    for _job in _held_queue.recover_interrupted():
        logger.warning("held render %s was interrupted mid-dispatch; marked as error", _job.get("id"))
except Exception:
    logger.exception("could not recover interrupted held renders")

# Ensure progress and execution events are broadcast to all connected web
# editor clients, even when the job was submitted by an agent or CLI client id.
_orig_send_sync = getattr(PromptServer.instance, "send_sync", None)
if _orig_send_sync is not None and not getattr(PromptServer.instance, "_vp_broadcast_patched", False):
    def _vp_broadcast_send_sync(event, data, sid=None):
        _orig_send_sync(event, data, sid=sid)
        if sid is not None and event in (
            "progress",
            "progress_state",
            "executing",
            "execution_start",
            "execution_cached",
            "execution_success",
            "execution_error",
            "execution_interrupted",
        ):
            try:
                _orig_send_sync(event, data, sid=None)
            except Exception:
                pass

    PromptServer.instance.send_sync = _vp_broadcast_send_sync
    PromptServer.instance._vp_broadcast_patched = True


def api() -> Api:
    return _api


def held_queue() -> HeldQueue:
    return _held_queue


async def _payload(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as error:
        raise ApiError(400, f"Expected a JSON body: {error}") from error
    if not isinstance(body, dict):
        raise ApiError(400, "Expected a JSON object.")
    return body


def _handle(fn: Callable[[web.Request], Awaitable[dict[str, Any]]]):
    """Wrap a handler so ApiError becomes a status and anything else a 500."""

    async def wrapped(request: web.Request) -> web.Response:
        try:
            return web.json_response(await fn(request))
        except ApiError as error:
            return web.json_response(error.to_dict(), status=error.status)
        except Exception as error:  # pragma: no cover - last-resort net
            logger.exception("video prompter route failed")
            return web.json_response({"error": str(error)}, status=500)

    wrapped.__name__ = fn.__name__
    return wrapped


def _query(request: web.Request, key: str) -> str:
    value = request.rel_url.query.get(key, "")
    if not value.strip():
        raise ApiError(400, f"Missing required query parameter {key!r}.")
    return value


def _notify(project: str, source: str | None = None, origin: str = "") -> None:
    """Tell connected editors a project changed, so an open one live-updates.

    Emitted over ComfyUI's own client socket, the same one the editor already
    listens on. `origin` carries the caller's client id so the browser that
    made the edit can ignore the echo of its own write; an agent omits it, so
    its writes always reach the browser. Never lets a failed notification break
    the write that succeeded.
    """
    send = getattr(PromptServer.instance, "send_sync", None)
    if send is None:  # an older ComfyUI, or a devserver without the shim
        return
    try:
        send(
            "video_prompter.changed",
            {"project": project, "source": source, "origin": origin},
        )
    except Exception:  # pragma: no cover - notification is best-effort
        logger.debug("video prompter change notification failed", exc_info=True)


def _notify_held_changed() -> None:
    send = getattr(PromptServer.instance, "send_sync", None)
    if send is None:
        return
    try:
        send("video_prompter.held_changed", {"count": len(held_queue().list())})
    except Exception:
        logger.debug("held queue notification failed", exc_info=True)


def _remember_dispatch_origin(request: web.Request) -> None:
    # A request may start the worker, but may never choose its destination.
    _ensure_held_dispatcher()


def _server_base_url() -> str:
    configured = os.environ.get("VIDEO_PROMPTER_COMFY_URL", "").strip().rstrip("/")
    if configured:
        return configured
    server = PromptServer.instance
    port = getattr(server, "port", None)
    if port is None:
        return ""
    address = str(getattr(server, "address", "127.0.0.1"))
    if address == "0.0.0.0":
        address = "127.0.0.1"
    elif address in ("::", "[::]"):
        address = "::1"
    if ":" in address and not address.startswith("["):
        address = f"[{address}]"
    return f"http://{address}:{port}"


async def _dispatch_held(job: dict[str, Any]) -> dict[str, Any]:
    job_id = str(job.get("id", ""))
    if not job_id:
        raise ValueError("Held job has no id")
    marked = held_queue().mark_dispatching(job_id)
    try:
        base_url = _server_base_url()
        if not base_url:
            raise RuntimeError("ComfyUI server address is not ready")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/prompt", json=marked["payload"]) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    detail = body.get("error", {}) if isinstance(body, dict) else {}
                    if isinstance(detail, dict):
                        detail = detail.get("message") or detail.get("details")
                    raise RuntimeError(str(detail or f"ComfyUI refused job ({response.status})"))
                if not isinstance(body, dict) or not body.get("prompt_id"):
                    raise RuntimeError("ComfyUI did not acknowledge a prompt_id")
        held_queue().remove(job_id)
        _notify_held_changed()
        return body
    except Exception as error:
        held_queue().mark_error(job_id, str(error))
        _notify_held_changed()
        raise


async def _held_dispatch_loop() -> None:
    while True:
        try:
            if _server_base_url():
                for job in held_queue().due():
                    try:
                        await _dispatch_held(job)
                    except Exception:
                        logger.exception("scheduled Video Prompter job could not be released")
        except Exception:
            logger.exception("held queue dispatcher failed")
        await asyncio.sleep(10)


def _ensure_held_dispatcher() -> None:
    global _held_dispatcher_task
    if _held_dispatcher_task is None or _held_dispatcher_task.done():
        try:
            _held_dispatcher_task = asyncio.get_running_loop().create_task(_held_dispatch_loop())
        except RuntimeError:
            pass


async def _start_held_dispatcher(app: web.Application) -> None:
    _ensure_held_dispatcher()


async def _stop_held_dispatcher(app: web.Application) -> None:
    global _held_dispatcher_task
    if _held_dispatcher_task is None:
        return
    _held_dispatcher_task.cancel()
    try:
        await _held_dispatcher_task
    except asyncio.CancelledError:
        pass
    _held_dispatcher_task = None


_server_app = getattr(PromptServer.instance, "app", None)
if _server_app is not None and not getattr(PromptServer.instance, "_vp_held_dispatcher_registered", False):
    _server_app.on_startup.append(_start_held_dispatcher)
    _server_app.on_cleanup.append(_stop_held_dispatcher)
    PromptServer.instance._vp_held_dispatcher_registered = True


def _id_of(payload: dict[str, Any], key: str) -> str | None:
    """An id from a payload, when that key holds a plain string."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


# -- the editor page ------------------------------------------------------


# The page holds the editor's whole stylesheet, so a cached copy pairs new
# JavaScript with old CSS and markup — the editor then looks broken in ways the
# source does not explain. It is served from disk beside the code it belongs
# to, so it is never cached.
_NO_CACHE = {"Cache-Control": "no-store"}


@routes.get(f"{ROUTE_PREFIX}/app")
@routes.get(f"{ROUTE_PREFIX}/app/")
@routes.get(f"{ROUTE_PREFIX}")
@routes.get(f"{ROUTE_PREFIX}/")
async def app_page(request: web.Request) -> web.Response:
    index = APP_DIR / "index.html"
    if not index.is_file():
        return web.Response(text="Editor page not installed.", status=404)
    return web.FileResponse(index, headers=_NO_CACHE)


@routes.get(ROUTE_PREFIX + "/app/{filename:.+}")
async def app_asset(request: web.Request) -> web.Response:
    """Serve the editor's own assets, refusing anything outside its directory."""
    target = (APP_DIR / request.match_info["filename"]).resolve()
    try:
        target.relative_to(APP_DIR.resolve())
    except ValueError:
        return web.Response(text="Not found.", status=404)
    if not target.is_file():
        return web.Response(text="Not found.", status=404)
    return web.FileResponse(target, headers=_NO_CACHE)


# -- contract -------------------------------------------------------------


@routes.get(f"{ROUTE_PREFIX}/schema")
@_handle
async def describe_schema(request: web.Request) -> dict[str, Any]:
    """The machine-readable contract an agent reads to drive the tool."""
    return api().describe_schema()


# -- projects -------------------------------------------------------------


@routes.get(f"{ROUTE_PREFIX}/projects")
@_handle
async def list_projects(request: web.Request) -> dict[str, Any]:
    return api().list_projects()


@routes.post(f"{ROUTE_PREFIX}/project/create")
@_handle
async def create_project(request: web.Request) -> dict[str, Any]:
    return api().create_project(await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/project/load")
@_handle
async def load_project(request: web.Request) -> dict[str, Any]:
    return api().load_project(_query(request, "name"))


@routes.post(f"{ROUTE_PREFIX}/project/update")
@_handle
async def update_project(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().update_project(payload)
    _notify(payload.get("name", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/project/delete")
@_handle
async def delete_project(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    return api().delete_project(payload.get("name", ""))


@routes.post(f"{ROUTE_PREFIX}/project/shelve")
@_handle
async def shelve_project(request: web.Request) -> dict[str, Any]:
    """Move a project to a shelf. Body {name, shelf: active|archived|trashed}."""
    return api().shelve_project(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/project/purge")
@_handle
async def purge_project(request: web.Request) -> dict[str, Any]:
    """Erase one trashed project for good. Body {name}."""
    return api().purge_project(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/project/empty-trash")
@_handle
async def empty_trash(request: web.Request) -> dict[str, Any]:
    return api().empty_trash()


@routes.post(f"{ROUTE_PREFIX}/project/duplicate")
@_handle
async def duplicate_project(request: web.Request) -> dict[str, Any]:
    """Copy a project as the next version of its family. Body {name, new_name?}."""
    return api().duplicate_project(await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/project/backups")
@_handle
async def project_backups(request: web.Request) -> dict[str, Any]:
    return api().project_backups(_query(request, "name"))


@routes.post(f"{ROUTE_PREFIX}/project/restore")
@_handle
async def restore_backup(request: web.Request) -> dict[str, Any]:
    return api().restore_backup(await _payload(request))


# -- sources (the bin) ----------------------------------------------------


@routes.get(f"{ROUTE_PREFIX}/source")
@_handle
async def get_source(request: web.Request) -> dict[str, Any]:
    """One source and its segments, for an agent editing a single thing."""
    return api().get_source(_query(request, "project"), _query(request, "source"))


@routes.post(f"{ROUTE_PREFIX}/source/add")
@_handle
async def add_source(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().add_source(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/source/update")
@_handle
async def update_source(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().update_source(payload)
    source = payload.get("source")
    source_id = source.get("id") if isinstance(source, dict) else None
    _notify(payload.get("project", ""), source_id, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/source/delete")
@_handle
async def delete_source(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().delete_source(payload)
    _notify(payload.get("project", ""), _id_of(payload, "source"), payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/source/duplicate")
@_handle
async def duplicate_source(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().duplicate_source(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/source/split")
@_handle
async def split_source(request: web.Request) -> dict[str, Any]:
    """Break some of a segment's shots out as a bin item of their own."""
    payload = await _payload(request)
    result = api().split_source(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/source/path")
@_handle
async def set_source_path(request: web.Request) -> dict[str, Any]:
    """Choose the lineage a source plays. Refused if the links do not hold up."""
    payload = await _payload(request)
    result = api().set_source_path(payload)
    _notify(payload.get("project", ""), _id_of(payload, "source"), payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/source/segment/add")
@_handle
async def add_segment(request: web.Request) -> dict[str, Any]:
    """Plan one more generation on the end of a source's chain."""
    payload = await _payload(request)
    result = api().add_segment(payload)
    _notify(payload.get("project", ""), _id_of(payload, "source"), payload.get("origin", ""))
    return result


@routes.get(f"{ROUTE_PREFIX}/source/revisions")
@_handle
async def segment_revisions(request: web.Request) -> dict[str, Any]:
    """One segment's past prompts. Off the ordinary project reply on purpose."""
    return api().segment_revisions(
        _query(request, "project"),
        _query(request, "source"),
        request.query.get("segment") or None,
    )


@routes.post(f"{ROUTE_PREFIX}/source/revisions/restore")
@_handle
async def restore_revision(request: web.Request) -> dict[str, Any]:
    """Put a past prompt back on one segment, filing the current one."""
    payload = await _payload(request)
    result = api().restore_revision(payload)
    _notify(payload.get("project", ""), _id_of(payload, "source"), payload.get("origin", ""))
    return result


# -- writing with a language model ----------------------------------------


@routes.get(f"{ROUTE_PREFIX}/llm/status")
@_handle
async def llm_status(request: web.Request) -> dict[str, Any]:
    """Whether a model is configured, and which. `?probe=1` also lists them."""
    probe = request.rel_url.query.get("probe", "") not in ("", "0", "false")
    return await asyncio.to_thread(api().llm_status, probe)


@routes.post(f"{ROUTE_PREFIX}/llm/draft")
@_handle
async def llm_draft(request: web.Request) -> dict[str, Any]:
    """Write or rewrite a prompt. Naming a project and source applies it.

    Off the event loop: this waits on Google, and blocking here would stall
    every other request the server is serving, renders included.
    """
    payload = await _payload(request)
    result = await asyncio.to_thread(api().llm_draft, payload)
    if payload.get("project") and payload.get("source"):
        _notify(payload["project"], _id_of(payload, "source"), payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/llm/draft-image")
@_handle
async def llm_draft_image(request: web.Request) -> dict[str, Any]:
    """Write or rewrite an image prompt. Naming a project files the old one.

    Off the event loop for the same reason as `llm_draft`. No `_notify`: this
    changes nothing in the project document, only the image prompt history,
    which no other client is showing.
    """
    return await asyncio.to_thread(api().llm_draft_image, await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/image_prompt/revisions")
@_handle
async def image_prompt_revisions(request: web.Request) -> dict[str, Any]:
    """Past image prompts for a project, newest first."""
    return api().image_prompt_revisions(_query(request, "project"))


@routes.post(f"{ROUTE_PREFIX}/image_prompt/revisions/file")
@_handle
async def file_image_prompt(request: web.Request) -> dict[str, Any]:
    """Keep an image prompt as a past version. Body {project, text, note?, target?}."""
    return api().file_image_prompt(await _payload(request))


# -- clips (the timeline) -------------------------------------------------


@routes.post(f"{ROUTE_PREFIX}/clip/add")
@_handle
async def add_clip(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().add_clip(payload)
    _notify(payload.get("project", ""), _id_of(payload, "source"), payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/clip/delete")
@_handle
async def delete_clip(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().delete_clip(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/clip/move")
@_handle
async def move_clip(request: web.Request) -> dict[str, Any]:
    """Reorder. Clip order is the edit, so this is the cut being changed."""
    payload = await _payload(request)
    result = api().move_clip(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/clip/edit")
@_handle
async def edit_clip(request: web.Request) -> dict[str, Any]:
    """Trim, speed and enablement, clamped to what the lineage can supply."""
    payload = await _payload(request)
    result = api().edit_clip(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/clip/split")
@_handle
async def split_clip(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().split_clip(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/clip/duplicate")
@_handle
async def duplicate_clip(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().duplicate_clip(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


# -- mutes (a stretch of one clip heard as silence) ------------------------


@routes.post(f"{ROUTE_PREFIX}/mute/add")
@_handle
async def add_mute(request: web.Request) -> dict[str, Any]:
    """Silence part of a clip. `start`/`end` are lineage seconds, so the band
    rides the footage through any later trim or move."""
    payload = await _payload(request)
    result = api().add_mute(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/mute/edit")
@_handle
async def edit_mute(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().edit_mute(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/mute/delete")
@_handle
async def delete_mute(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().delete_mute(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/defaults")
@_handle
async def set_defaults(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().set_defaults(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


# -- cast -----------------------------------------------------------------


@routes.get(f"{ROUTE_PREFIX}/cast")
@_handle
async def list_cast(request: web.Request) -> dict[str, Any]:
    return api().list_cast(_query(request, "project"))


@routes.get(f"{ROUTE_PREFIX}/cast/usage")
@_handle
async def member_usage(request: web.Request) -> dict[str, Any]:
    return api().member_usage(_query(request, "project"), _query(request, "member"))


@routes.post(f"{ROUTE_PREFIX}/cast/save")
@_handle
async def save_member(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().save_member(payload)
    # A member edit changes how every segment naming them composes, so the open
    # editor has to reload the project, not just the source that was touched.
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


@routes.post(f"{ROUTE_PREFIX}/cast/delete")
@_handle
async def delete_member(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().delete_member(payload)
    _notify(payload.get("project", ""), None, payload.get("origin", ""))
    return result


# -- presets --------------------------------------------------------------


@routes.get(f"{ROUTE_PREFIX}/presets")
@_handle
async def list_presets(request: web.Request) -> dict[str, Any]:
    return api().list_presets(request.rel_url.query.get("kind") or None)


@routes.get(f"{ROUTE_PREFIX}/preset")
@_handle
async def read_preset(request: web.Request) -> dict[str, Any]:
    return api().read_preset(_query(request, "kind"), _query(request, "name"))


@routes.post(f"{ROUTE_PREFIX}/preset/save")
@_handle
async def save_preset(request: web.Request) -> dict[str, Any]:
    return api().save_preset(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/preset/delete")
@_handle
async def delete_preset(request: web.Request) -> dict[str, Any]:
    return api().delete_preset(await _payload(request))


# -- render profiles ------------------------------------------------------


@routes.get(f"{ROUTE_PREFIX}/profiles")
@_handle
async def list_profiles(request: web.Request) -> dict[str, Any]:
    return api().list_profiles()


@routes.get(f"{ROUTE_PREFIX}/video-loras")
@_handle
async def list_video_loras(request: web.Request) -> dict[str, Any]:
    return await asyncio.to_thread(api().list_video_loras)


@routes.post(f"{ROUTE_PREFIX}/video-loras")
@_handle
async def update_video_loras(request: web.Request) -> dict[str, Any]:
    payload = await _payload(request)
    result = api().update_video_loras(payload)
    _notify(payload["project"], origin=str(payload.get("origin") or ""))
    return result


@routes.get(f"{ROUTE_PREFIX}/profile")
@_handle
async def load_profile(request: web.Request) -> dict[str, Any]:
    return api().load_profile(_query(request, "name"))


@routes.post(f"{ROUTE_PREFIX}/profile/save")
@_handle
async def save_profile(request: web.Request) -> dict[str, Any]:
    return api().save_profile(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/profile/delete")
@_handle
async def delete_profile(request: web.Request) -> dict[str, Any]:
    return api().delete_profile(await _payload(request))


# -- composing and rendering ----------------------------------------------


@routes.post(f"{ROUTE_PREFIX}/preview")
@_handle
async def preview(request: web.Request) -> dict[str, Any]:
    return api().preview(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/parse")
@_handle
async def parse_prompt(request: web.Request) -> dict[str, Any]:
    return api().parse_prompt(await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/reference_kinds")
@_handle
async def reference_kinds(request: web.Request) -> dict[str, Any]:
    return api().reference_kinds()


@routes.post(f"{ROUTE_PREFIX}/reference/upload")
@_handle
async def upload_reference(request: web.Request) -> dict[str, Any]:
    """Land a keyframe/reference file in ComfyUI's input dir.

    Two shapes: a multipart file part (the editor's picker) or a JSON `{url}`
    (an agent, or an online sample image) fetched server-side. Either returns
    the input-relative filename to put on a segment's keyframe or a reference.
    """
    if (request.content_type or "").startswith("multipart/"):
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                raise ApiError(400, "No file part in the upload.")
            if part.filename:
                data = await part.read(decode=False)
                return api().save_reference(part.filename, data)
    return await asyncio.to_thread(api().fetch_reference, await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/gallery")
@_handle
async def list_takes(request: web.Request) -> dict[str, Any]:
    return await asyncio.to_thread(
        api().list_takes,
        _query(request, "project"),
        request.rel_url.query.get("source") or request.rel_url.query.get("shot") or None,
    )


@routes.post(f"{ROUTE_PREFIX}/gallery/record")
@_handle
async def record_takes(request: web.Request) -> dict[str, Any]:
    """Takes the client parsed, or raw history entries for it to sort out."""
    return api().record_takes(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/gallery/context")
@_handle
async def take_context(request: web.Request) -> dict[str, Any]:
    """Copyable generation identity, file location and surrounding edit context."""
    return await asyncio.to_thread(api().take_context, await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/gallery/forget")
@_handle
async def forget_take(request: web.Request) -> dict[str, Any]:
    return api().forget_take(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/gallery/pick")
@_handle
async def pick_take(request: web.Request) -> dict[str, Any]:
    """Choose a take, as the whole path that ends at it."""
    return api().pick_take(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/gallery/annotate")
@_handle
async def annotate_take(request: web.Request) -> dict[str, Any]:
    return api().annotate_take(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/gallery/seen")
@_handle
async def mark_takes_seen(request: web.Request) -> dict[str, Any]:
    """Clear the new badge on a batch of takes in one write."""
    return api().mark_takes_seen(await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/lineage")
@_handle
async def take_lineage(request: web.Request) -> dict[str, Any]:
    """The continuation forest, built from what the renderer actually recorded."""
    return await asyncio.to_thread(
        api().take_lineage,
        _query(request, "project"),
        request.rel_url.query.get("source") or request.rel_url.query.get("shot") or None,
    )


@routes.post(f"{ROUTE_PREFIX}/lineage/resolve")
@_handle
async def resolve_lineage(request: web.Request) -> dict[str, Any]:
    """A path plus a range of lineage time, as the files that cover it."""
    return api().resolve_lineage(await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/sequence")
@_handle
async def sequence_report(request: web.Request) -> dict[str, Any]:
    """Timeline geometry and sequence-level findings, joined server side."""
    return await asyncio.to_thread(api().sequence_report, _query(request, "project"))


@routes.post(f"{ROUTE_PREFIX}/transform")
@_handle
async def transform(request: web.Request) -> dict[str, Any]:
    """Storyboard and canvas operations, so the editor holds none of those rules."""
    return api().transform(await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/render/progress")
@_handle
async def render_progress(request: web.Request) -> dict[str, Any]:
    """Return ComfyUI's live per-node progress for late-joining editors.

    WebSocket progress is transient.  This snapshot lets the standalone editor
    open halfway through an agent-submitted render and immediately identify the
    real executing node instead of guessing from elapsed time.
    """
    try:
        from comfy_execution.progress import get_progress_state
    except ImportError:  # Older ComfyUI releases only expose legacy events.
        return {"prompt_id": "", "nodes": {}}

    registry = get_progress_state()
    dynprompt = getattr(registry, "dynprompt", None)
    nodes: dict[str, dict[str, Any]] = {}
    for node_id, state in list(getattr(registry, "nodes", {}).items()):
        status = state.get("state", "pending")
        if hasattr(status, "value"):
            status = status.value
        node_id = str(node_id)

        def mapped(method: str) -> str:
            fn = getattr(dynprompt, method, None)
            if not callable(fn):
                return node_id
            try:
                return str(fn(node_id))
            except Exception:
                return node_id

        nodes[node_id] = {
            "value": state.get("value", 0),
            "max": state.get("max", 1),
            "state": str(status),
            "node_id": node_id,
            "display_node_id": mapped("get_display_node_id"),
            "parent_node_id": mapped("get_parent_node_id"),
            "real_node_id": mapped("get_real_node_id"),
        }
    return {"prompt_id": str(getattr(registry, "prompt_id", "")), "nodes": nodes}


@routes.get(f"{ROUTE_PREFIX}/render/held")
@_handle
async def list_held_renders(request: web.Request) -> dict[str, Any]:
    """Jobs stored outside ComfyUI until explicitly or automatically released."""
    _remember_dispatch_origin(request)
    return {"jobs": held_queue().list()}


@routes.post(f"{ROUTE_PREFIX}/render/hold")
@_handle
async def hold_render(request: web.Request) -> dict[str, Any]:
    _remember_dispatch_origin(request)
    body = await _payload(request)
    prompt_payload = body.get("payload")
    if not isinstance(prompt_payload, dict):
        raise ApiError(400, "Held render requires a payload object.")
    try:
        job = held_queue().add(
            prompt_payload,
            body.get("summary", {}),
            body.get("release_at"),
        )
    except ValueError as error:
        raise ApiError(400, str(error)) from error
    _notify_held_changed()
    return {"job": job}


@routes.post(f"{ROUTE_PREFIX}/render/held/schedule")
@_handle
async def schedule_held_render(request: web.Request) -> dict[str, Any]:
    _remember_dispatch_origin(request)
    body = await _payload(request)
    job_id = str(body.get("id", ""))
    try:
        job = held_queue().update_schedule(job_id, body.get("release_at"))
    except KeyError as error:
        raise ApiError(404, "Held render was not found.") from error
    except ValueError as error:
        raise ApiError(400, str(error)) from error
    _notify_held_changed()
    return {"job": job}


@routes.post(f"{ROUTE_PREFIX}/render/held/schedule_all")
@_handle
async def schedule_all_held_renders(request: web.Request) -> dict[str, Any]:
    """One run time for the whole shelf. `release_at: null` puts them back on hold."""
    _remember_dispatch_origin(request)
    body = await _payload(request)
    try:
        jobs = held_queue().update_schedule_all(body.get("release_at"))
    except ValueError as error:
        raise ApiError(400, str(error)) from error
    _notify_held_changed()
    return {"jobs": jobs, "count": len(jobs)}


@routes.post(f"{ROUTE_PREFIX}/render/held/release")
@_handle
async def release_held_render(request: web.Request) -> dict[str, Any]:
    _remember_dispatch_origin(request)
    body = await _payload(request)
    job = held_queue().get(str(body.get("id", "")))
    if job is None:
        raise ApiError(404, "Held render was not found.")
    try:
        dispatched = await _dispatch_held(job)
    except Exception as error:
        raise ApiError(409, f"Could not release held render: {error}") from error
    return {"released": True, "prompt_id": dispatched.get("prompt_id", "")}


@routes.post(f"{ROUTE_PREFIX}/render/held/release_all")
@_handle
async def release_all_held_renders(request: web.Request) -> dict[str, Any]:
    _remember_dispatch_origin(request)
    released: list[str] = []
    errors: list[dict[str, str]] = []
    for job in held_queue().list():
        try:
            result = await _dispatch_held(job)
            released.append(str(result.get("prompt_id", "")))
        except Exception as error:
            errors.append({"id": str(job.get("id", "")), "error": str(error)})
    return {"released": released, "errors": errors}


@routes.post(f"{ROUTE_PREFIX}/render/held/delete")
@_handle
async def delete_held_render(request: web.Request) -> dict[str, Any]:
    _remember_dispatch_origin(request)
    body = await _payload(request)
    removed = held_queue().remove(str(body.get("id", "")))
    if not removed:
        raise ApiError(404, "Held render was not found.")
    _notify_held_changed()
    return {"deleted": True}


@routes.post(f"{ROUTE_PREFIX}/render/prepare")
@_handle
async def prepare_render(request: web.Request) -> dict[str, Any]:
    """Returns a workflow ready for the client to POST to ComfyUI's `/prompt`.

    422 with findings means the prompt has errors worth fixing first; 409 means
    the profile's bindings no longer match its workflow.
    """
    return api().prepare_render(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/render/batch")
@_handle
async def prepare_batch(request: web.Request) -> dict[str, Any]:
    """N workflows at consecutive seeds, for producing candidates to compare."""
    return api().prepare_batch(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/render/reshoot")
@_handle
async def prepare_reshoot(request: web.Request) -> dict[str, Any]:
    """Re-render a take from its own frozen prompt, canvas and seed."""
    return api().prepare_reshoot(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/render/extend")
@_handle
async def prepare_extend(request: web.Request) -> dict[str, Any]:
    """Continue a take with H3 Motion Context: a new clip off an existing one."""
    return api().prepare_extend(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/render/extend/batch")
@_handle
async def prepare_extend_batch(request: web.Request) -> dict[str, Any]:
    """N continuations of one take at consecutive seeds, for comparing."""
    return api().prepare_extend_batch(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/render/refine")
@_handle
async def prepare_refine(request: web.Request) -> dict[str, Any]:
    """Refine an existing take with guided resample and/or face refine."""
    return api().prepare_refine(await _payload(request))


# -- text-to-image (keyframe generation) ----------------------------------


@routes.post(f"{ROUTE_PREFIX}/render/t2i")
@_handle
async def prepare_t2i(request: web.Request) -> dict[str, Any]:
    """Build a Krea 2 Turbo text-to-image workflow for generating a keyframe.

    Returns a workflow the editor POSTs to ``/prompt`` exactly like a video
    render. After the image lands in the output dir, ``/adopt`` copies it to
    the input dir so it can be used as a ``first_frame`` for i2v.
    """
    return api().prepare_t2i(await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/t2i/styles")
@_handle
async def t2i_styles(request: web.Request) -> dict[str, Any]:
    """The cinematic style presets available for keyframe generation."""
    return api().t2i_styles()


@routes.post(f"{ROUTE_PREFIX}/render/t2i/batch")
@_handle
async def prepare_t2i_batch(request: web.Request) -> dict[str, Any]:
    """`count` keyframe workflows at consecutive seeds, for hunting a composition."""
    return api().prepare_t2i_batch(await _payload(request))


@routes.get(f"{ROUTE_PREFIX}/stills")
@_handle
async def list_stills(request: web.Request) -> dict[str, Any]:
    """The project's generated images, newest first."""
    return api().list_stills(
        _query(request, "project"),
        request.rel_url.query.get("source") or None,
    )


@routes.post(f"{ROUTE_PREFIX}/stills/record")
@_handle
async def record_stills(request: web.Request) -> dict[str, Any]:
    """Finished image generations, from raw ComfyUI history. Adopts as it records."""
    return api().record_stills(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/stills/annotate")
@_handle
async def annotate_still(request: web.Request) -> dict[str, Any]:
    return api().annotate_still(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/stills/forget")
@_handle
async def forget_still(request: web.Request) -> dict[str, Any]:
    return api().forget_still(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/render/adopt")
@_handle
async def adopt_output_image(request: web.Request) -> dict[str, Any]:
    """Copy a generated image from output/ to input/ for use as a keyframe."""
    return api().adopt_output_image(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/reveal")
@_handle
async def reveal_file(request: web.Request) -> dict[str, Any]:
    """Open the file manager on a rendered file, so it can be dragged elsewhere."""
    return api().reveal_file(await _payload(request))


@routes.post(f"{ROUTE_PREFIX}/export")
@_handle
async def export(request: web.Request) -> dict[str, Any]:
    """Turn the picked takes into something outside this tool."""
    return await asyncio.to_thread(api().export, await _payload(request))
