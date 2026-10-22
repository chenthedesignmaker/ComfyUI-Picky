"""Load the custom node the way ComfyUI does.

ComfyUI calls `spec_from_file_location(mangled_name, <dir>/__init__.py)` and
does not add the directory to `sys.path`, so this is the one test that catches
an entry point that imports fine under pytest but not under ComfyUI.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_comfy(monkeypatch, tmp_path):
    """Stand in for the ComfyUI modules the route layer imports."""
    routes_registered: list[tuple[str, str]] = []

    class FakeRoutes:
        def _record(self, method: str):
            def decorate(route: str):
                def wrap(fn):
                    routes_registered.append((method, route))
                    return fn

                return wrap

            return decorate

        get = property(lambda self: self._record("GET"))
        post = property(lambda self: self._record("POST"))
        delete = property(lambda self: self._record("DELETE"))

    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=FakeRoutes())
    )
    monkeypatch.setitem(sys.modules, "server", fake_server)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_user_directory = lambda: str(tmp_path / "user")
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    # Route registration is an import side effect, so a cached module would
    # make every test after the first see an empty list. Both the sys.modules
    # entry and the attribute on the parent package have to go: `from pkg
    # import sub` is satisfied by the attribute alone.
    import video_prompter

    monkeypatch.delitem(sys.modules, "video_prompter.comfy_routes", raising=False)
    monkeypatch.delattr(video_prompter, "comfy_routes", raising=False)

    return routes_registered


def load_entry_point(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class TestComfyLoad:
    def test_the_entry_point_loads_under_a_mangled_module_name(self, fake_comfy):
        # ComfyUI names the module after the path, not after the package.
        module = load_entry_point("custom_nodes_x_video_x_prompter")
        assert "VideoPromptNode" in module.NODE_CLASS_MAPPINGS

    def test_it_declares_a_display_name_and_web_directory(self, fake_comfy):
        module = load_entry_point("vp_entry_display")
        assert module.NODE_DISPLAY_NAME_MAPPINGS["VideoPromptNode"]
        assert module.WEB_DIRECTORY == "./js"

    def test_the_web_directory_exists(self, fake_comfy):
        module = load_entry_point("vp_entry_web")
        assert (REPO_ROOT / module.WEB_DIRECTORY).is_dir()

    def test_the_routes_register(self, fake_comfy):
        load_entry_point("vp_entry_routes")
        registered = {route for _, route in fake_comfy}
        assert "/video_prompter/projects" in registered
        assert "/video_prompter/render/prepare" in registered
        assert "/video_prompter/app" in registered

    def test_a_broken_route_layer_does_not_stop_the_node_loading(self, monkeypatch, tmp_path):
        # No `server` module at all: ComfyUI would still want the node itself.
        monkeypatch.setitem(sys.modules, "server", None)
        module = load_entry_point("vp_entry_no_server")
        assert "VideoPromptNode" in module.NODE_CLASS_MAPPINGS


class TestNodeContract:
    def test_input_types_are_shaped_the_way_comfy_expects(self, fake_comfy):
        node = load_entry_point("vp_entry_inputs").NODE_CLASS_MAPPINGS["VideoPromptNode"]
        spec = node.INPUT_TYPES()
        assert "project" in spec["required"]
        assert "shot" in spec["required"]

    def test_returns_line_up_with_their_names(self, fake_comfy):
        node = load_entry_point("vp_entry_returns").NODE_CLASS_MAPPINGS["VideoPromptNode"]
        assert len(node.RETURN_TYPES) == len(node.RETURN_NAMES)
        assert node.RETURN_NAMES == ("prompt", "length", "width", "height", "seconds")

    def test_an_empty_selection_still_produces_a_usable_tuple(self, fake_comfy):
        node = load_entry_point("vp_entry_empty").NODE_CLASS_MAPPINGS["VideoPromptNode"]()
        prompt, length, width, height, seconds = node.build("(none)", "")
        assert length == 124  # the 5s trained minimum, not the 1s allowed floor
        assert (width, height) == (1344, 768)
        assert seconds == pytest.approx(5.167, abs=0.001)

    def test_a_missing_project_is_reported_by_name(self, fake_comfy):
        node = load_entry_point("vp_entry_missing").NODE_CLASS_MAPPINGS["VideoPromptNode"]()
        with pytest.raises(ValueError, match="Ghost"):
            node.build("Ghost", "")

    def test_is_changed_moves_when_the_project_file_does(self, fake_comfy):
        node = load_entry_point("vp_entry_changed").NODE_CLASS_MAPPINGS["VideoPromptNode"]
        first = node.IS_CHANGED("(none)", "a")
        assert first != node.IS_CHANGED("(none)", "b")
