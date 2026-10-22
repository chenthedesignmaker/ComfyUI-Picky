"""The route layer's live-sync notification, tested behind a fake ComfyUI.

`comfy_routes` does `from server import PromptServer` at import time, so a fake
`server` module is installed before importing it — the same trick the devserver
uses. Only these tests import the route module, so the fake stays contained.
"""

from __future__ import annotations

import importlib
import asyncio
import sys
import threading
import types

import pytest
from aiohttp import web


@pytest.fixture
def routes(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_PROMPTER_HOME", str(tmp_path))

    captured: list[tuple[str, dict]] = []
    instance = types.SimpleNamespace(routes=web.RouteTableDef())
    instance.send_sync = lambda event, data, sid=None: captured.append((event, data))

    fake = types.ModuleType("server")
    fake.PromptServer = types.SimpleNamespace(instance=instance)
    monkeypatch.setitem(sys.modules, "server", fake)

    sys.modules.pop("video_prompter.comfy_routes", None)
    module = importlib.import_module("video_prompter.comfy_routes")
    module._captured = captured
    try:
        yield module
    finally:
        sys.modules.pop("video_prompter.comfy_routes", None)


def test_notify_emits_a_change_event(routes):
    routes._notify("Proj", "src_1", "origin-x")
    assert routes._captured == [
        (
            "video_prompter.changed",
            {"project": "Proj", "source": "src_1", "origin": "origin-x"},
        )
    ]


def test_notify_defaults_source_and_origin(routes):
    routes._notify("Proj")
    assert routes._captured == [
        ("video_prompter.changed", {"project": "Proj", "source": None, "origin": ""})
    ]


def test_modern_progress_state_is_rebroadcast_to_editor_clients(routes):
    from server import PromptServer

    PromptServer.instance.send_sync(
        "progress_state",
        {"prompt_id": "run-1", "nodes": {"10": {"state": "running"}}},
        sid="render-owner",
    )
    # First delivery is to the render owner; the second is the broadcast that
    # lets the standalone editor track jobs launched by an agent or another UI.
    assert routes._captured == [
        ("progress_state", {"prompt_id": "run-1", "nodes": {"10": {"state": "running"}}}),
        ("progress_state", {"prompt_id": "run-1", "nodes": {"10": {"state": "running"}}}),
    ]


def test_notify_is_silent_without_send_sync(routes):
    # An older ComfyUI with no send_sync must not break the write that succeeded.
    from server import PromptServer

    del PromptServer.instance.send_sync
    routes._notify("Proj", None, "")  # no exception, nothing captured
    assert routes._captured == []


def test_an_id_is_only_read_from_a_string(routes):
    assert routes._id_of({"source": "src_1"}, "source") == "src_1"
    assert routes._id_of({"source": {"id": "src_1"}}, "source") is None
    assert routes._id_of({}, "source") is None


@pytest.mark.parametrize("handler,method,body", [
    ("take_context", "take_context", {"project": "Film", "take": "id:file.mp4"}),
    ("export", "export", {}),
    ("upload_reference", "fetch_reference", {"url": "https://example.invalid/image.png"}),
])
def test_blocking_work_does_not_block_route_loop(routes, monkeypatch, handler, method, body):
    released = threading.Event()
    def blocking(payload):
        assert released.wait(timeout=1), "The server loop could not run its timer"
        return {"ok": True}
    monkeypatch.setattr(routes.api(), method, blocking)
    async def run():
        async def json():
            return body
        request = types.SimpleNamespace(json=json, content_type="application/json")
        asyncio.get_running_loop().call_later(0.02, released.set)
        response = await getattr(routes, handler)(request)
        assert response.status == 200
    asyncio.run(run())


def test_request_host_cannot_change_dispatch_destination(routes, monkeypatch):
    monkeypatch.delenv("VIDEO_PROMPTER_COMFY_URL", raising=False)
    routes.PromptServer.instance.address = "0.0.0.0"
    routes.PromptServer.instance.port = 8188
    monkeypatch.setattr(routes, "_ensure_held_dispatcher", lambda: None)
    routes._remember_dispatch_origin(types.SimpleNamespace(scheme="https", host="attacker.invalid"))
    assert routes._server_base_url() == "http://127.0.0.1:8188"
    routes.PromptServer.instance.address = "::"
    assert routes._server_base_url() == "http://[::1]:8188"
    monkeypatch.setenv("VIDEO_PROMPTER_COMFY_URL", "https://configured.invalid/comfy/")
    assert routes._server_base_url() == "https://configured.invalid/comfy"


def test_concurrent_releases_submit_once_and_keep_failed_claim_outside_dispatch(routes, monkeypatch):
    routes.PromptServer.instance.address = "127.0.0.1"
    routes.PromptServer.instance.port = 8188
    monkeypatch.delenv("VIDEO_PROMPTER_COMFY_URL", raising=False)
    monkeypatch.setattr(routes, "_ensure_held_dispatcher", lambda: None)
    store = routes.held_queue()
    job = store.add({"prompt": {"1": {"inputs": {}}}}, {})

    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()
        submissions = []
        class Response:
            status = 200
            async def __aenter__(self):
                entered.set()
                await finish.wait()
                return self
            async def __aexit__(self, *args):
                pass
            async def json(self, **kwargs):
                return {"prompt_id": "mock-prompt"}
        class Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def post(self, url, **kwargs):
                submissions.append(url)
                return Response()
        monkeypatch.setattr(routes.aiohttp, "ClientSession", Session)
        async def json():
            return {"id": job["id"]}
        request = types.SimpleNamespace(json=json)
        first = asyncio.create_task(routes.release_held_render(request))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            second = await routes.release_held_render(request)
            assert second.status == 409
            assert store.get(job["id"])["status"] == "dispatching"
            bulk = await routes.release_all_held_renders(request)
            assert bulk.status == 200
            assert len(submissions) == 1
        finally:
            finish.set()
            result = await first
        assert result.status == 200
        assert store.list() == []
    asyncio.run(run())
