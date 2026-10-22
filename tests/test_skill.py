"""The video-prompt skill and the render script it hands every render to.

The skill ships twice — `.claude/skills` for Claude Code, `.agents/skills` for
other agents — and nothing but this test keeps the two copies the same.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COPIES = [REPO_ROOT / ".claude/skills/video-prompt", REPO_ROOT / ".agents/skills/video-prompt"]


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_both_skill_copies_are_identical():
    primary, mirror = (_files(root) for root in COPIES)
    assert sorted(primary) == sorted(mirror), "the two skill folders hold different files"
    drifted = [name for name in primary if primary[name] != mirror[name]]
    assert not drifted, f"skill copies differ in: {drifted} — edit one, copy it to the other"


def test_every_reference_is_linked_and_exists():
    root = COPIES[0]
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    named = set(re.findall(r"references/[\w.-]+\.md", skill))
    on_disk = {f"references/{p.name}" for p in (root / "references").glob("*.md")}
    assert named - on_disk == set(), "SKILL.md points at references that do not exist"
    assert on_disk - named == set(), "a reference file nothing in SKILL.md tells the agent to read"


def test_render_script_is_what_the_skill_runs():
    skill = (COPIES[0] / "SKILL.md").read_text(encoding="utf-8")
    assert "python tools/render_sources.py" in skill
    assert (REPO_ROOT / "tools/render_sources.py").is_file()


# -- tools/render_sources.py --------------------------------------------------


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "render_sources", REPO_ROOT / "tools/render_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while decorating
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


class FakeClient:
    def __init__(self, profiles=(), takes=()):
        self.profiles, self.takes = list(profiles), list(takes)

    def get(self, path, **query):
        if path == "/profiles":
            return {"profiles": self.profiles}
        if path == "/gallery":
            return {"takes": self.takes}
        raise AssertionError(path)


def test_payload_matches_what_the_editor_queues(script):
    job = script.Job("src_1", "Kettle", "Quick", {
        "workflow": {"1": {}}, "prompt": "text", "frames": 124, "seconds": 5.167,
        "seed": 7, "width": 576, "height": 352, "quality": "preview", "steps": 8,
        "pass1_steps": None, "latent_path": "h3_context/a.safetensors",
        "parent_take": None, "context_length": None,
    })
    payload = job.payload("Night run")
    meta = payload["extra_data"]["video_prompter"]
    assert payload["prompt"] == {"1": {}}
    assert payload["prompt_id"] == job.prompt_id
    # The key is `shot` and it holds a source id — every takes.json relies on it.
    assert meta["shot"] == "src_1" and meta["project"] == "Night run"
    assert meta["quality"] == "preview" and meta["mode"] == "Quick"
    # Absent means single-pass / not an extend; a null would read as a value.
    for key in ("pass1_steps", "parent_take", "context_length"):
        assert key not in meta


def test_profile_prefers_flag_then_project_then_only_one(script):
    three = FakeClient(profiles=["A", "B", "C"])
    assert script.choose_profile(three, {}, "B") == "B"
    assert script.choose_profile(three, {"render_profile": "C"}, None) == "C"
    assert script.choose_profile(FakeClient(profiles=["A"]), {}, None) == "A"
    with pytest.raises(SystemExit, match="say which"):
        script.choose_profile(three, {}, None)
    with pytest.raises(SystemExit, match="No render profile"):
        script.choose_profile(three, {}, "Z")


def test_take_depth_walks_parent_links(script):
    takes = [
        {"prompt_id": "a", "filename": "0.mp4"},
        {"prompt_id": "b", "filename": "1.mp4", "parent_take": "a:0.mp4"},
        {"prompt_id": "c", "filename": "2.mp4", "parent_take": "b:1.mp4"},
    ]
    client = FakeClient(takes=takes)
    assert [script.take_depth(client, "p", t) for t in takes] == [0, 1, 2]


def test_render_time_comes_from_comfyui_timestamps(script):
    entry = {"status": {"messages": [
        ["execution_start", {"timestamp": 1_000}],
        ["execution_success", {"timestamp": 47_300}],
    ]}}
    assert script.elapsed_from(entry) == 46.3
    assert script.elapsed_from({"status": {"messages": []}}) is None
