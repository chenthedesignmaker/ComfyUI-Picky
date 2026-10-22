"""Portable generation references must identify the take without changing the edit."""
from pathlib import Path

import pytest

from video_prompter.api import Api, ApiError
from video_prompter.gallery import Take
from video_prompter.projects import ProjectStore
from video_prompter.profiles import ProfileStore
from video_prompter.sequence import Clip
from video_prompter.sources import Source


@pytest.fixture
def setup(tmp_path, monkeypatch):
    output = tmp_path / "output"
    (output / "nested").mkdir(parents=True)
    monkeypatch.setenv("VIDEO_PROMPTER_OUTPUT_DIR", str(output))
    api = Api(projects=ProjectStore(tmp_path / "projects"),
              profiles=ProfileStore(tmp_path / "profiles"))
    source = Source(name="Door opens")
    other = Source(name="Reaction")
    project = api.projects.create("My film", [source, other])
    parent = Take(prompt_id="first", filename="first.mp4", shot=source.id, seconds=5)
    target = Take(prompt_id="exact", filename="target.mp4", subfolder="nested",
                  shot=source.id, seed=0, seconds=5, prompt_text="The original prompt",
                  parent_take=parent.id, latent_path="latents/target.latent")
    alternate = Take(prompt_id="alternate", filename="other.mp4", shot=source.id)
    api.gallery.record(project.name, [parent, target, alternate])
    source.path = [parent.id, target.id]
    project.clips[0].label = "Opening close-up"
    project.clips.append(Clip(source_id=source.id, label="Alternate use",
                              path_override=[alternate.id]))
    api.projects.save(project)
    return api, project, source, parent, target, output


def test_context_keeps_exact_identity_paths_prompt_and_surrounding_edit(setup):
    api, project, source, parent, target, output = setup
    directory = api.projects.path_for(project.name)
    before = {name: (directory / name).read_bytes() for name in ("project.json", "takes.json")}
    result = api.take_context({"project": project.name, "take": target.id,
                               "source": source.id, "clip": project.clips[0].id})
    assert result["take"]["id"] == target.id
    assert result["take"]["seed"] == 0
    assert result["take"]["prompt_text"] == "The original prompt"
    assert result["take"]["latent_path"] == target.latent_path
    assert result["take"]["file"] == {"path": str(output / "nested" / "target.mp4"), "exists": False}
    assert Path(result["project"]["project_file"]).is_absolute()
    assert Path(result["project"]["takes_file"]).is_file()
    assert result["shot"]["name"] == "Door opens"
    assert result["recorded_source"] == {"id": source.id, "name": source.name}
    assert result["lineage"]["ancestors"][0]["id"] == parent.id
    assert [c["name"] for c in result["timeline_order"]] == ["Opening close-up", "Reaction", "Alternate use"]
    first, alternate = result["related_clips"]
    assert first["selected_in_ui"] and first["target_in_chosen_path"]
    assert not alternate["selected_in_ui"] and not alternate["target_in_chosen_path"]
    assert {name: (directory / name).read_bytes() for name in before} == before


def test_orphan_take_and_unavailable_media_still_have_a_copyable_reference(setup, monkeypatch):
    api, project, source, parent, target, output = setup
    project.remove_source(source.id)
    api.projects.save(project)
    monkeypatch.setitem(__import__("video_prompter.api", fromlist=["_MEDIA_DIRS"])._MEDIA_DIRS,
                        "output", lambda: None)
    result = api.take_context({"project": project.name, "take": target.id})
    assert result["shot"] is None
    assert result["take"]["file"]["path"] is None
    assert "Cannot locate" in result["take"]["file"]["note"]
    assert result["related_clips"] == []


def test_unknown_take_is_rejected_instead_of_copying_another_generation(setup):
    api, project, *_ = setup
    with pytest.raises(ApiError) as error:
        api.take_context({"project": project.name, "take": "wrong:target.mp4"})
    assert error.value.status == 404


def test_context_does_not_resolve_paths_outside_media_directory(setup):
    api, project, source, *_ = setup
    target = Take(prompt_id="bad", filename="../secret.mp4", shot=source.id)
    api.gallery.record(project.name, [target])
    result = api.take_context({"project": project.name, "take": target.id})
    assert result["take"]["file"]["path"] is None
    assert "outside" in result["take"]["file"]["note"]


def test_context_reports_existing_file_without_opening_explorer(setup, monkeypatch):
    api, project, source, _, target, output = setup
    (output / "nested" / target.filename).touch()
    monkeypatch.setattr("video_prompter.api._reveal", lambda _: pytest.fail("Opened Explorer"))
    result = api.take_context({"project": project.name, "take": target.id})
    assert result["take"]["file"]["exists"] is True
