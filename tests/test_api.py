from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_prompter import PresetLibrary, ProfileStore, ProjectStore, VideoPrompt, bundled_root, twostage
from video_prompter.api import Api, ApiError
from video_prompter.profiles import MAX_SEED
from video_prompter.serialize import prompt_to_dict


@pytest.fixture
def api(tmp_path):
    return Api(
        projects=ProjectStore(tmp_path / "projects"),
        profiles=ProfileStore(tmp_path / "profiles"),
        presets=PresetLibrary(tmp_path / "presets", bundled_root()),
    )


@pytest.fixture
def real_workflow():
    path = Path(__file__).parent / "fixtures" / "h3_i2v_real.json"
    return json.loads(path.read_text(encoding="utf-8"))


def a_prompt(duration: int = 10) -> dict:
    prompt = VideoPrompt.blank(duration)
    prompt.style = "A cinematic look."
    prompt.audio = "rain"
    prompt.storyboard.beats = []
    from video_prompter import Beat

    prompt.storyboard.beats = [Beat(0, duration, "wide", "a door opens")]
    return prompt_to_dict(prompt)


def _one_of_class(workflow: dict, class_type: str) -> str:
    """The id of the single node of this class, asserting there is exactly one."""
    found = [nid for nid, node in workflow.items()
             if node.get("class_type") == class_type]
    assert len(found) == 1, f"expected one {class_type}, found {found}"
    return found[0]


def node_of(workflow: dict, class_type: str) -> dict:
    """The single node of this class. Graphs built by transform have generated
    ids, so tests address their nodes by what they are, not where they landed."""
    return workflow[_one_of_class(workflow, class_type)]


def _second_sampler(workflow: dict) -> str:
    """The refining pass: the sampler nothing else feeds its finished latent to."""
    samplers = [nid for nid, node in workflow.items()
                if node.get("class_type") == "SamplerCustomAdvanced"]
    assert len(samplers) == 2, samplers
    upstream = {nid for nid, node in workflow.items()
                for value in (node.get("inputs") or {}).values()
                if isinstance(value, list) and len(value) == 2 and value[1] == 1
                for nid in [str(value[0])]}
    return next(s for s in samplers if s not in upstream)


def status_of(fn, *args) -> int:
    with pytest.raises(ApiError) as raised:
        fn(*args)
    return raised.value.status


class TestProjects:
    def test_full_project_round_trip_saves_cast_and_linked_sources(self, api):
        doc = api.create_project({"name": "Cast round trip"})["project"]
        doc["cast"] = [{"id": "alice", "name": "Alice", "kind": "character"}]
        prompt = a_prompt()
        prompt["cast"] = ["alice"]
        doc["sources"] = [{"id": "src_alice", "name": "Alice enters",
                           "segments": [{"prompt": prompt}]}]
        saved = api.update_project({"name": doc["name"], "project": doc})["project"]
        loaded = api.load_project(doc["name"])["project"]
        assert loaded["cast"] == saved["cast"]
        assert loaded["cast"][0]["name"] == "Alice"
        assert loaded["sources"][0]["segments"][0]["prompt"]["cast"] == ["alice"]
        # Legacy clients do not know about cast; explicit [] still clears it.
        del saved["cast"]
        api.update_project({"name": doc["name"], "project": saved})
        assert api.load_project(doc["name"])["project"]["cast"] == loaded["cast"]
        saved["cast"] = []
        assert api.update_project({"name": doc["name"], "project": saved})["project"]["cast"] == []

    def test_a_new_store_lists_nothing(self, api):
        assert api.list_projects() == {"projects": [], "entries": []}

    def test_create_then_load(self, api):
        api.create_project({"name": "Rooftop"})
        assert api.load_project("Rooftop")["project"]["name"] == "Rooftop"

    def test_creating_twice_is_a_conflict(self, api):
        api.create_project({"name": "Rooftop"})
        assert status_of(api.create_project, {"name": "Rooftop"}) == 409

    def test_creating_without_a_name_is_a_bad_request(self, api):
        assert status_of(api.create_project, {}) == 400

    def test_loading_a_missing_project_is_a_not_found(self, api):
        assert status_of(api.load_project, "ghost") == 404

    def test_update_replaces_the_bin_and_the_edit(self, api):
        api.create_project({"name": "Edit"})
        payload = {
            "name": "Edit",
            "project": {
                "version": 2,
                "sources": [{"id": "src_1", "name": "One",
                             "segments": [{"prompt": a_prompt()}]}],
                "timeline": {"clips": [{"id": "clip_1", "source_id": "src_1"}]},
            },
        }
        result = api.update_project(payload)["project"]
        assert len(result["sources"]) == 1
        assert len(result["timeline"]["clips"]) == 1

    def test_a_v1_project_dict_is_migrated_on_the_way_in(self, api):
        # Every route into a project lands on the same shape, so a file or a
        # payload written by an older build loads without a conversion step.
        api.create_project({"name": "Edit"})
        payload = {
            "name": "Edit",
            "project": {"sequence": {"shots": [{"id": "sh_1", "name": "One",
                                                "prompt": a_prompt()}]}},
        }
        result = api.update_project(payload)["project"]
        assert result["version"] == 2
        assert result["sources"][0]["id"] == "sh_1"
        assert result["timeline"]["clips"][0]["source_id"] == "sh_1"

    def test_update_needs_a_project_object(self, api):
        api.create_project({"name": "Edit"})
        assert status_of(api.update_project, {"name": "Edit"}) == 400

    def test_update_of_a_missing_project_is_a_not_found(self, api):
        assert status_of(api.update_project, {"name": "ghost", "project": {}}) == 404

    def test_delete_reports_what_happened(self, api):
        api.create_project({"name": "Temp"})
        assert api.delete_project("Temp") == {"deleted": True}
        assert api.delete_project("Temp") == {"deleted": False}


class TestBackupsOverTheApi:
    def test_archiving_hides_a_project_from_the_active_list(self, api):
        api.create_project({"name": "Done"})
        api.create_project({"name": "Live"})
        api.shelve_project({"name": "Done", "shelf": "archived"})
        assert api.list_projects()["projects"] == ["Live"]

    def test_shelving_reports_the_new_listing(self, api):
        api.create_project({"name": "Done"})
        result = api.shelve_project({"name": "Done", "shelf": "trashed"})
        shelves = {e["name"]: e["shelf"] for e in result["entries"]}
        assert shelves == {"Done": "trashed"}

    def test_an_unknown_shelf_is_a_400(self, api):
        api.create_project({"name": "Real"})
        assert status_of(api.shelve_project, {"name": "Real", "shelf": "moon"}) == 400

    def test_shelving_a_missing_project_is_a_404(self, api):
        assert status_of(api.shelve_project, {"name": "Ghost", "shelf": "trashed"}) == 404

    def test_purging_something_not_in_the_trash_is_a_409(self, api):
        api.create_project({"name": "Live"})
        assert status_of(api.purge_project, {"name": "Live"}) == 409

    def test_purging_a_trashed_project_removes_it(self, api):
        api.create_project({"name": "Gone"})
        api.shelve_project({"name": "Gone", "shelf": "trashed"})
        assert api.purge_project({"name": "Gone"})["purged"] is True
        assert api.list_projects()["entries"] == []

    def test_emptying_the_trash_lists_what_went(self, api):
        api.create_project({"name": "A"})
        api.create_project({"name": "B"})
        api.shelve_project({"name": "A", "shelf": "trashed"})
        assert api.empty_trash()["purged"] == ["A"]
        assert api.list_projects()["projects"] == ["B"]

    def test_a_name_in_the_trash_can_be_reused(self, api):
        api.create_project({"name": "Reel"})
        api.shelve_project({"name": "Reel", "shelf": "trashed"})
        api.create_project({"name": "Reel"})
        assert api.list_projects()["projects"] == ["Reel"]

    def test_duplicating_names_the_next_version(self, api):
        api.create_project({"name": "Reel"})
        assert api.duplicate_project({"name": "Reel"})["project"]["name"] == "Reel v2"

    def test_a_duplicate_shares_the_family_and_says_its_parent(self, api):
        api.create_project({"name": "Reel"})
        copy = api.duplicate_project({"name": "Reel"})["project"]
        assert copy["derived_from"] == "Reel"
        assert copy["lineage"] == api.load_project("Reel")["project"]["lineage"]

    def test_duplicating_under_a_taken_name_is_refused(self, api):
        api.create_project({"name": "Reel"})
        api.create_project({"name": "Taken"})
        assert status_of(
            api.duplicate_project, {"name": "Reel", "new_name": "Taken"}
        ) == 400

    def test_duplicating_a_missing_project_is_a_404(self, api):
        assert status_of(api.duplicate_project, {"name": "Ghost"}) == 404

    def test_backups_appear_after_a_second_save(self, api):
        api.create_project({"name": "Kept"})
        api.update_project({"name": "Kept", "project": {"version": 2, "sources": []}})
        assert api.project_backups("Kept")["backups"]

    def test_a_backup_can_be_restored_by_filename(self, api):
        api.create_project(
            {"name": "Undo", "sources": [{"name": "One", "segments": [{"prompt": a_prompt()}]}]}
        )
        api.update_project({"name": "Undo", "project": {"version": 2, "sources": []}})

        snapshot = api.project_backups("Undo")["backups"][0]["file"]
        restored = api.restore_backup({"name": "Undo", "file": snapshot})
        assert len(restored["project"]["sources"]) == 1

    def test_restoring_an_unknown_file_is_a_not_found(self, api):
        api.create_project({"name": "Undo"})
        assert status_of(api.restore_backup, {"name": "Undo", "file": "nope.json"}) == 404


def sources_of(result: dict) -> list[dict]:
    return result["project"]["sources"]


def clips_of(result: dict) -> list[dict]:
    return result["project"]["timeline"]["clips"]


def prompt_of(source: dict, depth: int = 0) -> dict:
    return source["segments"][depth]["prompt"]


class TestSourcesOverTheApi:
    def test_adding_appends_to_the_bin(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "Rooftop"})
        result = api.add_source({"project": "Cut", "name": "Landing"})
        assert [s["name"] for s in sources_of(result)] == ["Rooftop", "Landing"]

    def test_adding_a_source_puts_a_clip_on_the_timeline(self, api):
        api.create_project({"name": "Cut"})
        result = api.add_source({"project": "Cut", "name": "Rooftop"})
        assert [c["source_id"] for c in clips_of(result)] == [sources_of(result)[0]["id"]]

    def test_a_bin_only_source_can_be_added_without_a_clip(self, api):
        api.create_project({"name": "Cut"})
        result = api.add_source({"project": "Cut", "name": "Spare", "clip": False})
        assert len(sources_of(result)) == 1
        assert clips_of(result) == []

    def test_a_new_source_is_named_for_its_position(self, api):
        api.create_project({"name": "Cut"})
        assert sources_of(api.add_source({"project": "Cut"}))[0]["name"] == "Source 1"

    def test_adding_at_a_position_inserts_the_clip_there(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        api.add_source({"project": "Cut", "name": "C"})
        result = api.add_source({"project": "Cut", "name": "B", "at": 1})
        by_id = {s["id"]: s["name"] for s in sources_of(result)}
        assert [by_id[c["source_id"]] for c in clips_of(result)] == ["A", "B", "C"]

    def test_a_new_source_inherits_the_project_defaults(self, api):
        api.create_project({"name": "Cut"})
        api.set_defaults(
            {"project": "Cut", "defaults": {"style": "Anamorphic night.", "camera": "Hard cuts."}}
        )
        source = sources_of(api.add_source({"project": "Cut", "name": "A"}))[0]
        assert prompt_of(source)["style"] == "Anamorphic night."
        assert prompt_of(source)["camera"] == "Hard cuts."

    def test_defaults_do_not_overwrite_what_a_segment_already_says(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        written = sources_of(api.load_project("Cut"))[0]
        prompt_of(written)["style"] = "Its own look."
        api.update_source({"project": "Cut", "source": written})

        api.set_defaults({"project": "Cut", "defaults": {"style": "Project look."}})
        assert prompt_of(sources_of(api.load_project("Cut"))[0])["style"] == "Its own look."

    def test_pushing_defaults_overwrites_every_segment(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        api.add_source({"project": "Cut", "name": "B"})
        result = api.set_defaults(
            {"project": "Cut", "defaults": {"style": "One look."}, "push": True}
        )
        assert {prompt_of(s)["style"] for s in sources_of(result)} == {"One look."}

    def test_deleting_removes_it(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        target = sources_of(api.load_project("Cut"))[0]["id"]
        assert sources_of(api.delete_source({"project": "Cut", "source": target})) == []

    def test_deleting_a_source_takes_its_clips_with_it(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        target = sources_of(api.load_project("Cut"))[0]["id"]
        assert clips_of(api.delete_source({"project": "Cut", "source": target})) == []

    def test_deleting_something_absent_is_a_not_found(self, api):
        api.create_project({"name": "Cut"})
        assert status_of(api.delete_source, {"project": "Cut", "source": "src_nope"}) == 404

    def test_updating_one_source_leaves_the_others_alone(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        api.add_source({"project": "Cut", "name": "B"})

        first = sources_of(api.load_project("Cut"))[0]
        first["name"] = "Renamed"
        result = api.update_source({"project": "Cut", "source": first})
        assert [s["name"] for s in sources_of(result)] == ["Renamed", "B"]

    def test_updating_an_unknown_source_is_a_not_found(self, api):
        api.create_project({"name": "Cut"})
        payload = {"project": "Cut", "source": {"id": "src_nope", "name": "Ghost",
                                                "segments": []}}
        assert status_of(api.update_source, payload) == 404

    def test_a_v1_shot_object_still_creates_something_sensible(self, api):
        # An agent or script written against the old shape should not get a blank.
        result = api.create_project(
            {"name": "Legacy", "shots": [{"name": "One", "prompt": a_prompt()}]}
        )
        assert sources_of(result)[0]["name"] == "One"
        assert prompt_of(sources_of(result)[0])["style"] == "A cinematic look."

    def test_renaming_does_not_orphan_a_take(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        source = sources_of(api.load_project("Cut"))[0]
        api.record_takes({
            "project": "Cut",
            "takes": [{"prompt_id": "r1", "filename": "a.mp4", "shot": source["id"]}],
        })

        source["name"] = "Renamed"
        api.update_source({"project": "Cut", "source": source})
        assert api.list_takes("Cut", source["id"])["takes"]

    def test_duplicating_copies_the_intent_and_not_the_takes(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        written = sources_of(api.load_project("Cut"))[0]
        prompt_of(written)["style"] = "A specific look."
        written["path"] = ["r1:a.mp4"]
        api.update_source({"project": "Cut", "source": written})

        copy = sources_of(api.duplicate_source({"project": "Cut", "source": written["id"]}))[1]
        assert copy["name"] == "A (2)"
        assert prompt_of(copy)["style"] == "A specific look."
        assert copy["path"] == []
        assert copy["id"] != written["id"]

    def test_editing_a_copy_does_not_touch_the_original(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        original = sources_of(api.load_project("Cut"))[0]["id"]
        api.duplicate_source({"project": "Cut", "source": original})

        copy = sources_of(api.load_project("Cut"))[1]
        prompt_of(copy)["style"] = "Changed."
        api.update_source({"project": "Cut", "source": copy})
        assert prompt_of(sources_of(api.load_project("Cut"))[0])["style"] != "Changed."


class TestSplittingShotsOverTheApi:
    """Breaking some of a segment's shots out as a bin item of their own."""

    def _three_shots(self, api) -> dict:
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "Hallway", "duration": 12})
        written = sources_of(api.load_project("Cut"))[0]
        prompt = prompt_of(written)
        prompt["duration"] = 12
        prompt["style"] = "Live-action, anamorphic."
        prompt["beats"] = [
            {"start": 0, "end": 4, "angle": "wide", "action": "the door opens"},
            {"start": 4, "end": 8, "angle": "medium", "action": "she crosses the room"},
            {"start": 8, "end": 12, "angle": "close", "action": "the clock reads midnight"},
        ]
        api.update_source({"project": "Cut", "source": written})
        return sources_of(api.load_project("Cut"))[0]

    def test_the_picked_shots_become_a_source_of_their_own(self, api):
        source = self._three_shots(api)
        result = api.split_source({"project": "Cut", "source": source["id"], "beats": [1, 2]})

        made = sources_of(result)[1]
        assert made["id"] == result["source"]
        assert made["name"] == "Hallway (shots 2-3)"
        assert [b["action"] for b in prompt_of(made)["beats"]] == [
            "she crosses the room", "the clock reads midnight",
        ]

    def test_the_timing_comes_with_them(self, api):
        source = self._three_shots(api)
        made = sources_of(
            api.split_source({"project": "Cut", "source": source["id"], "beats": [1, 2]})
        )[1]
        prompt = prompt_of(made)
        assert prompt["duration"] == 8
        assert [(b["start"], b["end"]) for b in prompt["beats"]] == [(0.0, 4.0), (4.0, 8.0)]

    def test_the_scene_around_them_comes_too(self, api):
        source = self._three_shots(api)
        made = sources_of(
            api.split_source({"project": "Cut", "source": source["id"], "beats": [1]})
        )[1]
        assert prompt_of(made)["style"] == "Live-action, anamorphic."

    def test_the_original_keeps_its_shots(self, api):
        # A copy, not a move: whether the original should lose them is a
        # decision worth making after seeing what the split generates.
        source = self._three_shots(api)
        result = api.split_source({"project": "Cut", "source": source["id"], "beats": [1, 2]})
        assert len(prompt_of(sources_of(result)[0])["beats"]) == 3

    def test_it_carries_no_takes(self, api):
        source = self._three_shots(api)
        api.record_takes({
            "project": "Cut",
            "takes": [{"prompt_id": "r1", "filename": "a.mp4", "shot": source["id"]}],
        })
        result = api.split_source({"project": "Cut", "source": source["id"], "beats": [1]})
        made = sources_of(result)[1]
        assert made["path"] == []
        assert api.list_takes("Cut", made["id"])["takes"] == []

    def test_the_clip_lands_beside_the_one_it_came_out_of(self, api):
        source = self._three_shots(api)
        api.add_source({"project": "Cut", "name": "Later"})
        result = api.split_source({"project": "Cut", "source": source["id"], "beats": [1]})

        made = result["source"]
        names = [clip["source_id"] for clip in clips_of(result)]
        assert names.index(made) == names.index(source["id"]) + 1

    def test_the_clip_can_be_left_off(self, api):
        source = self._three_shots(api)
        result = api.split_source(
            {"project": "Cut", "source": source["id"], "beats": [1], "clip": False}
        )
        assert [c["source_id"] for c in clips_of(result)] == [source["id"]]

    def test_a_shot_that_is_not_there_is_refused(self, api):
        source = self._three_shots(api)
        payload = {"project": "Cut", "source": source["id"], "beats": [1, 7]}
        assert status_of(api.split_source, payload) == 400

    def test_picking_nothing_is_refused(self, api):
        source = self._three_shots(api)
        payload = {"project": "Cut", "source": source["id"], "beats": []}
        assert status_of(api.split_source, payload) == 400

    def test_splitting_an_unknown_source_is_a_not_found(self, api):
        self._three_shots(api)
        payload = {"project": "Cut", "source": "src_nope", "beats": [0]}
        assert status_of(api.split_source, payload) == 404


class TestSegmentsOverTheApi:
    def _one(self, api) -> str:
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A", "duration": 8})
        return sources_of(api.load_project("Cut"))[0]["id"]

    def test_adding_a_segment_plans_one_more_generation(self, api):
        source_id = self._one(api)
        result = api.add_segment({"project": "Cut", "source": source_id})
        assert len(sources_of(result)[0]["segments"]) == 2

    def test_a_continuation_inherits_the_quality_before_it(self, api):
        # Or one chain comes out of the export at two grades.
        source_id = self._one(api)
        written = sources_of(api.load_project("Cut"))[0]
        written["segments"][0]["quality"] = "final"
        api.update_source({"project": "Cut", "source": written})

        result = api.add_segment({"project": "Cut", "source": source_id})
        assert sources_of(result)[0]["segments"][1]["quality"] == "final"

    def test_a_continuation_inherits_the_length_before_it(self, api):
        # The "Generate (s)" field has to start somewhere, and matching the
        # generation before it is what an extend used to do implicitly.
        source_id = self._one(api)
        result = api.add_segment({"project": "Cut", "source": source_id})
        assert prompt_of(sources_of(result)[0], 1)["duration"] == 8

    def test_a_continuation_can_be_given_its_own_length(self, api):
        source_id = self._one(api)
        result = api.add_segment(
            {"project": "Cut", "source": source_id, "duration": 12}
        )
        durations = [s["prompt"]["duration"] for s in sources_of(result)[0]["segments"]]
        assert durations == [8, 12]

    def test_a_segment_can_carry_its_own_prompt(self, api):
        source_id = self._one(api)
        result = api.add_segment(
            {"project": "Cut", "source": source_id, "prompt": a_prompt(8)}
        )
        assert prompt_of(sources_of(result)[0], 1)["style"] == "A cinematic look."

    def test_a_planned_segment_widens_the_clip_before_it_renders(self, api):
        source_id = self._one(api)
        before = api.sequence_report("Cut")["clips"][0]["timeline_seconds"]
        api.add_segment({"project": "Cut", "source": source_id, "duration": 8})
        after = api.sequence_report("Cut")["clips"][0]["timeline_seconds"]
        assert (before, after) == (8.0, 16.0)

    def test_adding_to_an_unknown_source_is_a_not_found(self, api):
        self._one(api)
        payload = {"project": "Cut", "source": "src_nope"}
        assert status_of(api.add_segment, payload) == 404



class TestMutesOverTheApi:
    """A stretch of one clip heard as silence, edited from the timeline.

    The routes speak lineage seconds and the report answers in both clocks, so
    most of what is checked here is that the two stay in step when the edit moves
    underneath a band.
    """

    def _one(self, api) -> tuple[str, str]:
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A", "duration": 8})
        project = api.load_project("Cut")
        return sources_of(project)[0]["id"], clips_of(project)[0]["id"]

    def _mutes(self, api, clip_id: str) -> list[dict]:
        report = api.sequence_report("Cut")
        entry = next(c for c in report["clips"] if c["id"] == clip_id)
        return entry["mutes"]

    def test_a_band_comes_back_in_both_clocks(self, api):
        _, clip_id = self._one(api)
        made = api.add_mute({"project": "Cut", "clip": clip_id, "start": 2.0, "end": 3.0})
        assert made["mute"].startswith("mute_")
        assert self._mutes(api, clip_id) == [
            {"id": made["mute"], "start": 2.0, "end": 3.0, "at": 2.0, "length": 1.0}
        ]

    def test_trimming_the_head_moves_where_it_draws_not_what_it_is(self, api):
        _, clip_id = self._one(api)
        api.add_mute({"project": "Cut", "clip": clip_id, "start": 2.0, "end": 3.0})
        api.edit_clip({"project": "Cut", "clip": clip_id, "source_in": 1.5})
        band = self._mutes(api, clip_id)[0]
        # The record is untouched; only its place on the ruler moved.
        assert (band["start"], band["end"]) == (2.0, 3.0)
        assert (band["at"], band["length"]) == (0.5, 1.0)

    def test_speed_rescales_what_it_draws(self, api):
        _, clip_id = self._one(api)
        api.add_mute({"project": "Cut", "clip": clip_id, "start": 2.0, "end": 4.0})
        api.edit_clip({"project": "Cut", "clip": clip_id, "speed": 2.0})
        band = self._mutes(api, clip_id)[0]
        assert (band["at"], band["length"]) == (1.0, 1.0)

    def test_a_band_the_trim_hides_stays_on_the_record_with_nothing_to_draw(self, api):
        _, clip_id = self._one(api)
        api.add_mute({"project": "Cut", "clip": clip_id, "start": 5.0, "end": 6.0})
        api.edit_clip({"project": "Cut", "clip": clip_id, "source_out": 3.0})
        assert self._mutes(api, clip_id)[0]["at"] is None

        api.edit_clip({"project": "Cut", "clip": clip_id, "source_out": None})
        assert self._mutes(api, clip_id)[0]["at"] == 5.0

    def test_a_band_is_clamped_to_the_lineage_not_to_the_trim(self, api):
        _, clip_id = self._one(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "source_out": 3.0})
        # Past the out point but inside the eight seconds the source plans for,
        # which is what makes trimming reversible.
        api.add_mute({"project": "Cut", "clip": clip_id, "start": 5.0, "end": 99.0})
        band = self._mutes(api, clip_id)[0]
        assert (band["start"], band["end"]) == (5.0, 8.0)

    def test_editing_moves_it_and_leaves_the_end_alone(self, api):
        _, clip_id = self._one(api)
        made = api.add_mute({"project": "Cut", "clip": clip_id, "start": 2.0, "end": 3.0})
        api.edit_mute({"project": "Cut", "mute": made["mute"], "start": 1.0})
        band = self._mutes(api, clip_id)[0]
        assert (band["start"], band["end"]) == (1.0, 3.0)

    def test_ends_given_backwards_are_the_same_band(self, api):
        _, clip_id = self._one(api)
        api.add_mute({"project": "Cut", "clip": clip_id, "start": 4.0, "end": 1.0})
        band = self._mutes(api, clip_id)[0]
        assert (band["start"], band["end"]) == (1.0, 4.0)

    def test_a_band_with_no_width_is_a_miss_not_a_mute(self, api):
        _, clip_id = self._one(api)
        with pytest.raises(ApiError) as caught:
            api.add_mute({"project": "Cut", "clip": clip_id, "start": 2.0, "end": 2.0})
        assert caught.value.status == 400

    def test_ends_have_to_be_numbers(self, api):
        _, clip_id = self._one(api)
        with pytest.raises(ApiError) as caught:
            api.add_mute({"project": "Cut", "clip": clip_id, "start": "two", "end": 3.0})
        assert caught.value.status == 400

    def test_deleting_takes_it_off_the_clip(self, api):
        _, clip_id = self._one(api)
        made = api.add_mute({"project": "Cut", "clip": clip_id, "start": 2.0, "end": 3.0})
        api.delete_mute({"project": "Cut", "mute": made["mute"]})
        assert self._mutes(api, clip_id) == []

    def test_a_clip_that_is_not_there_cannot_be_silenced(self, api):
        self._one(api)
        with pytest.raises(ApiError) as caught:
            api.add_mute({"project": "Cut", "clip": "clip_nope", "start": 1.0, "end": 2.0})
        assert caught.value.status == 404

    def test_editing_or_deleting_a_band_that_is_gone_says_so(self, api):
        self._one(api)
        for call in (api.edit_mute, api.delete_mute):
            with pytest.raises(ApiError) as caught:
                call({"project": "Cut", "mute": "mute_nope", "start": 1.0})
            assert caught.value.status == 404

    def test_removing_the_clip_removes_its_silences(self, api):
        _, clip_id = self._one(api)
        made = api.add_mute({"project": "Cut", "clip": clip_id, "start": 2.0, "end": 3.0})
        api.delete_clip({"project": "Cut", "clip": clip_id})
        stored = api.load_project("Cut")["project"]["timeline"]
        assert stored["mutes"] == []
        assert made["mute"]

    def test_bands_come_back_earliest_first_however_they_were_added(self, api):
        _, clip_id = self._one(api)
        api.add_mute({"project": "Cut", "clip": clip_id, "start": 5.0, "end": 6.0})
        api.add_mute({"project": "Cut", "clip": clip_id, "start": 1.0, "end": 2.0})
        assert [m["start"] for m in self._mutes(api, clip_id)] == [1.0, 5.0]


class TestClipsOverTheApi:
    def _one(self, api) -> tuple[str, str]:
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A", "duration": 8})
        project = api.load_project("Cut")
        return sources_of(project)[0]["id"], clips_of(project)[0]["id"]

    def test_a_second_clip_is_another_use_of_one_source(self, api):
        source_id, _ = self._one(api)
        result = api.add_clip({"project": "Cut", "source": source_id})
        assert [c["source_id"] for c in clips_of(result)] == [source_id, source_id]
        assert len(sources_of(result)) == 1

    def test_adding_a_clip_for_an_unknown_source_is_a_not_found(self, api):
        self._one(api)
        assert status_of(api.add_clip, {"project": "Cut", "source": "src_nope"}) == 404

    def test_deleting_a_clip_leaves_the_source_in_the_bin(self, api):
        _, clip_id = self._one(api)
        result = api.delete_clip({"project": "Cut", "clip": clip_id})
        assert clips_of(result) == []
        assert len(sources_of(result)) == 1

    def test_deleting_a_clip_that_is_not_there_is_a_not_found(self, api):
        self._one(api)
        assert status_of(api.delete_clip, {"project": "Cut", "clip": "clip_nope"}) == 404

    def test_moving_reorders_the_edit(self, api):
        api.create_project({"name": "Cut"})
        for name in ("A", "B", "C"):
            api.add_source({"project": "Cut", "name": name})
        project = api.load_project("Cut")
        names = {s["id"]: s["name"] for s in sources_of(project)}
        last = clips_of(project)[2]["id"]
        result = api.move_clip({"project": "Cut", "clip": last, "to": 0})
        assert [names[c["source_id"]] for c in clips_of(result)] == ["C", "A", "B"]

    def test_moving_an_unknown_clip_is_a_not_found(self, api):
        self._one(api)
        payload = {"project": "Cut", "clip": "clip_nope", "to": 0}
        assert status_of(api.move_clip, payload) == 404


class TestEditOps:
    def a_clip(self, api, duration=8) -> str:
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A", "duration": duration})
        return clips_of(api.load_project("Cut"))[0]["id"]

    def test_trimming_shortens_the_timeline(self, api):
        clip_id = self.a_clip(api)
        api.edit_clip(
            {"project": "Cut", "clip": clip_id, "source_in": 1.0, "source_out": 5.0}
        )
        assert api.sequence_report("Cut")["clips"][0]["timeline_seconds"] == 4.0

    def test_an_in_point_past_the_source_is_clamped(self, api):
        clip_id = self.a_clip(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "source_in": 99.0})
        # 8s is 192 frames, the one whole-second budget exactly on the grid.
        assert api.sequence_report("Cut")["clips"][0]["source_in"] == 8.0

    def test_an_inverted_trim_reopens_the_out_point(self, api):
        clip_id = self.a_clip(api)
        api.edit_clip(
            {"project": "Cut", "clip": clip_id, "source_in": 6.0, "source_out": 2.0}
        )
        assert api.sequence_report("Cut")["clips"][0]["timeline_seconds"] == 2.0

    def test_speed_is_clamped_to_something_sane(self, api):
        clip_id = self.a_clip(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "speed": 500})
        assert api.sequence_report("Cut")["clips"][0]["speed"] == 10.0

    def test_disabling_removes_it_from_the_running_time(self, api):
        clip_id = self.a_clip(api)
        api.add_source({"project": "Cut", "name": "B", "duration": 8})
        api.edit_clip({"project": "Cut", "clip": clip_id, "enabled": False})
        assert api.sequence_report("Cut")["duration"] == 8.0

    def test_a_label_overrides_the_source_name_on_one_clip(self, api):
        clip_id = self.a_clip(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "label": "The leap"})
        assert api.sequence_report("Cut")["clips"][0]["name"] == "The leap"

    def test_editing_an_unknown_clip_is_a_not_found(self, api):
        self.a_clip(api)
        payload = {"project": "Cut", "clip": "clip_nope", "speed": 2}
        assert status_of(api.edit_clip, payload) == 404

    def test_splitting_makes_two_clips_of_the_same_length(self, api):
        clip_id = self.a_clip(api)
        result = api.split_clip({"project": "Cut", "clip": clip_id, "at": 4.0})
        assert len(clips_of(result)) == 2
        lengths = [c["timeline_seconds"] for c in api.sequence_report("Cut")["clips"]]
        assert lengths == [4.0, 4.0]

    def test_a_split_keeps_the_total_length(self, api):
        clip_id = self.a_clip(api)
        before = api.sequence_report("Cut")["duration"]
        api.split_clip({"project": "Cut", "clip": clip_id, "at": 3.0})
        assert api.sequence_report("Cut")["duration"] == before

    def test_both_halves_are_windows_onto_one_source(self, api):
        # Splitting spends no render and copies no prompt: it is two clips.
        clip_id = self.a_clip(api)
        result = api.split_clip({"project": "Cut", "clip": clip_id, "at": 4.0})
        assert len(sources_of(result)) == 1
        assert len({c["source_id"] for c in clips_of(result)}) == 1

    def test_splitting_outside_the_clip_is_a_bad_request(self, api):
        clip_id = self.a_clip(api)
        payload = {"project": "Cut", "clip": clip_id, "at": 99.0}
        assert status_of(api.split_clip, payload) == 400

    def test_splitting_at_zero_is_a_bad_request(self, api):
        clip_id = self.a_clip(api)
        assert status_of(api.split_clip, {"project": "Cut", "clip": clip_id, "at": 0}) == 400

    def test_duplicating_a_clip_keeps_its_trim(self, api):
        clip_id = self.a_clip(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "source_in": 2.0})
        result = api.duplicate_clip({"project": "Cut", "clip": clip_id})
        assert [c["source_in"] for c in clips_of(result)] == [2.0, 2.0]
        assert clips_of(result)[0]["id"] != clips_of(result)[1]["id"]

    def test_duplicating_an_unknown_clip_is_a_not_found(self, api):
        self.a_clip(api)
        payload = {"project": "Cut", "clip": "clip_nope"}
        assert status_of(api.duplicate_clip, payload) == 404


def child_a_of(api) -> str:
    """The lower-rated of the two continuations in `a_chain`."""
    return "p1:childA.mp4"


class TestPathsOverTheApi:
    def a_project_with_a_take(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        source = sources_of(api.load_project("Cut"))[0]
        api.record_takes({
            "project": "Cut",
            "takes": [{"prompt_id": "r1", "filename": "a.mp4",
                       "shot": source["id"], "seconds": 15.083}],
        })
        return source["id"], api.list_takes("Cut")["takes"][0]["id"]

    def a_chain(self, api):
        """Two generations, forking at depth one, as the renderer recorded it."""
        api.create_project({"name": "Chain"})
        api.add_source({"project": "Chain", "name": "A"})
        source_id = sources_of(api.load_project("Chain"))[0]["id"]
        api.add_segment({"project": "Chain", "source": source_id})
        api.record_takes({
            "project": "Chain",
            "takes": [
                {"prompt_id": "p0", "filename": "root.mp4", "shot": source_id,
                 "seconds": 5.167},
                {"prompt_id": "p1", "filename": "childA.mp4", "shot": source_id,
                 "seconds": 4.25, "parent_take": "p0:root.mp4", "rating": 2},
                {"prompt_id": "p2", "filename": "childB.mp4", "shot": source_id,
                 "seconds": 4.25, "parent_take": "p0:root.mp4", "rating": 5},
            ],
        })
        return source_id, "p0:root.mp4", "p1:childA.mp4", "p2:childB.mp4"

    def test_picking_stores_the_whole_path(self, api):
        source_id, take_id = self.a_project_with_a_take(api)
        result = api.pick_take({"project": "Cut", "source": source_id, "take": take_id})
        assert sources_of(result)[0]["path"] == [take_id]

    def test_an_empty_take_clears_the_path(self, api):
        source_id, take_id = self.a_project_with_a_take(api)
        api.pick_take({"project": "Cut", "source": source_id, "take": take_id})
        result = api.pick_take({"project": "Cut", "source": source_id, "take": ""})
        assert sources_of(result)[0]["path"] == []

    def test_picking_a_take_that_does_not_exist_is_a_not_found(self, api):
        source_id, _ = self.a_project_with_a_take(api)
        payload = {"project": "Cut", "source": source_id, "take": "r9:gone.mp4"}
        assert status_of(api.pick_take, payload) == 404

    def test_picking_a_take_from_another_chain_is_refused(self, api):
        source_id, take_id = self.a_project_with_a_take(api)
        api.add_source({"project": "Cut", "name": "B"})
        other = sources_of(api.load_project("Cut"))[1]["id"]
        payload = {"project": "Cut", "source": other, "take": take_id}
        assert status_of(api.pick_take, payload) == 400

    def test_an_unrendered_source_holds_its_segments_budget(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        # A 5s budget is 124 frames, which really runs 5.167s.
        assert api.sequence_report("Cut")["clips"][0]["timeline_seconds"] == 5.167

    def test_a_take_is_used_by_default_before_anything_is_chosen(self, api):
        self.a_project_with_a_take(api)
        report = api.sequence_report("Cut")
        assert report["clips"][0]["timeline_seconds"] == 15.083
        # …and the report says the choice is a default, not the author's.
        assert report["sources"][0]["chosen"] is False

    def test_choosing_marks_the_path_as_the_authors(self, api):
        source_id, take_id = self.a_project_with_a_take(api)
        api.pick_take({"project": "Cut", "source": source_id, "take": take_id})
        assert api.sequence_report("Cut")["sources"][0]["chosen"] is True

    def test_picking_a_root_carries_the_path_on_down_the_chain(self, api):
        source_id, root, _, child_b = self.a_chain(api)
        result = api.pick_take({"project": "Chain", "source": source_id, "take": root})
        # Better-rated children sort first, so the default follows B.
        assert sources_of(result)[0]["path"] == [root, child_b]

    def test_picking_a_child_names_its_own_parent_as_the_step_before(self, api):
        # Parent and child cannot disagree: the parent's choice *is* path[n-1].
        source_id, root, child_a, _ = self.a_chain(api)
        result = api.pick_take({"project": "Chain", "source": source_id, "take": child_a})
        assert sources_of(result)[0]["path"] == [root, child_a]

    def test_switching_back_restores_the_remembered_branch(self, api):
        source_id, root, child_a, child_b = self.a_chain(api)
        api.pick_take({"project": "Chain", "source": source_id, "take": child_a})
        api.pick_take({"project": "Chain", "source": source_id, "take": child_b})
        result = api.pick_take({"project": "Chain", "source": source_id, "take": root})
        assert sources_of(result)[0]["path"] == [root, child_b]

    def test_a_path_can_be_set_outright(self, api):
        source_id, root, child_a, _ = self.a_chain(api)
        result = api.set_source_path(
            {"project": "Chain", "source": source_id, "path": [root, child_a]}
        )
        assert sources_of(result)[0]["path"] == [root, child_a]

    def test_a_path_whose_links_do_not_hold_up_is_refused(self, api):
        # The one check that replaces every heuristic, enforced before it stores.
        source_id, root, child_a, child_b = self.a_chain(api)
        with pytest.raises(ApiError) as raised:
            api.set_source_path(
                {"project": "Chain", "source": source_id, "path": [child_a, child_b]}
            )
        assert raised.value.status == 409
        assert raised.value.extra["problems"]

    def test_setting_a_path_on_an_unknown_source_is_a_not_found(self, api):
        self.a_chain(api)
        payload = {"project": "Chain", "source": "src_nope", "path": []}
        assert status_of(api.set_source_path, payload) == 404

    def test_a_clip_follows_its_source_across_a_branch_switch(self, api):
        source_id, root, child_a, child_b = self.a_chain(api)
        api.pick_take({"project": "Chain", "source": source_id, "take": child_a})
        api.pick_take({"project": "Chain", "source": source_id, "take": child_b})
        assert api.sequence_report("Chain")["clips"][0]["path"] == [root, child_b]

    def test_a_pinned_clip_stays_on_its_branch(self, api):
        source_id, root, child_a, child_b = self.a_chain(api)
        api.pick_take({"project": "Chain", "source": source_id, "take": child_a})
        clip_id = clips_of(api.load_project("Chain"))[0]["id"]
        api.edit_clip({"project": "Chain", "clip": clip_id, "pin": True})

        api.pick_take({"project": "Chain", "source": source_id, "take": child_b})
        report = api.sequence_report("Chain")
        assert report["clips"][0]["pinned"] is True
        assert report["clips"][0]["path"] == [root, child_a]

    def test_pinning_holds_the_branch_a_default_was_showing(self, api):
        # A source nobody has chosen for still plays a default path. Pinning
        # must freeze *that* — pinning to the empty stored path would look like
        # it worked and quietly leave the clip following.
        source_id, root, _, child_b = self.a_chain(api)
        clip_id = clips_of(api.load_project("Chain"))[0]["id"]
        api.edit_clip({"project": "Chain", "clip": clip_id, "pin": True})

        api.pick_take({"project": "Chain", "source": source_id, "take": child_a_of(api)})
        report = api.sequence_report("Chain")
        assert report["clips"][0]["pinned"] is True
        assert report["clips"][0]["path"] == [root, child_b]

    def test_pinning_a_clip_with_no_take_is_refused(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        clip_id = clips_of(api.load_project("Cut"))[0]["id"]
        payload = {"project": "Cut", "clip": clip_id, "pin": True}
        assert status_of(api.edit_clip, payload) == 409

    def test_unpinning_puts_a_clip_back_to_following(self, api):
        source_id, root, child_a, child_b = self.a_chain(api)
        api.pick_take({"project": "Chain", "source": source_id, "take": child_a})
        clip_id = clips_of(api.load_project("Chain"))[0]["id"]
        api.edit_clip({"project": "Chain", "clip": clip_id, "pin": True})
        api.edit_clip({"project": "Chain", "clip": clip_id, "pin": False})

        api.pick_take({"project": "Chain", "source": source_id, "take": child_b})
        assert api.sequence_report("Chain")["clips"][0]["path"] == [root, child_b]

    def test_rating_and_discarding_stick(self, api):
        _, take_id = self.a_project_with_a_take(api)
        api.annotate_take({"project": "Cut", "take": take_id, "rating": 4})
        result = api.annotate_take({"project": "Cut", "take": take_id, "discarded": True})
        assert (result["take"]["rating"], result["take"]["discarded"]) == (4, True)

    def test_annotating_an_unknown_take_is_a_not_found(self, api):
        self.a_project_with_a_take(api)
        payload = {"project": "Cut", "take": "r9:gone.mp4", "rating": 3}
        assert status_of(api.annotate_take, payload) == 404

    def test_a_take_arrives_unseen_and_can_be_marked(self, api):
        _, take_id = self.a_project_with_a_take(api)
        assert api.annotate_take({"project": "Cut", "take": take_id})["take"]["seen"] is False
        marked = api.mark_takes_seen({"project": "Cut", "takes": [take_id]})
        assert marked == {"changed": [take_id], "count": 1}
        assert api.annotate_take({"project": "Cut", "take": take_id})["take"]["seen"] is True

    def test_marking_the_same_take_twice_changes_nothing_the_second_time(self, api):
        _, take_id = self.a_project_with_a_take(api)
        api.mark_takes_seen({"project": "Cut", "takes": [take_id]})
        assert api.mark_takes_seen({"project": "Cut", "takes": [take_id]})["count"] == 0

    def test_marking_a_take_that_is_gone_is_ignored_not_refused(self, api):
        """The editor sweeps what it drew; a take forgotten since is not news."""
        self.a_project_with_a_take(api)
        assert api.mark_takes_seen({"project": "Cut", "takes": ["r9:gone.mp4"]})["count"] == 0

    def test_marking_seen_needs_a_list(self, api):
        self.a_project_with_a_take(api)
        assert status_of(api.mark_takes_seen, {"project": "Cut", "takes": "all"}) == 400


class TestFoldingAtOpen:
    """A chain made before sources existed is folded back together on open.

    See `fold.py`. The detail lives in `test_fold.py`; what matters here is that
    opening a project is what triggers it, and that the take record follows.
    """

    def a_legacy_chain(self, api):
        """Two bin rows the renderer recorded as one chain, as v1 left them."""
        api.create_project({"name": "Legacy"})
        api.add_source({"project": "Legacy", "name": "One"})
        api.add_source({"project": "Legacy", "name": "Two"})
        first, second = (s["id"] for s in sources_of(api.load_project("Legacy")))
        api.record_takes({
            "project": "Legacy",
            "takes": [
                {"prompt_id": "p0", "filename": "one.mp4", "shot": first,
                 "seconds": 5.167},
                {"prompt_id": "p1", "filename": "two.mp4", "shot": second,
                 "seconds": 4.25, "parent_take": "p0:one.mp4"},
            ],
        })
        # A v1 shot stood for exactly one take, so the migration leaves each row
        # holding a one-step path.
        api.set_source_path({"project": "Legacy", "source": first, "path": ["p0:one.mp4"]})
        api.set_source_path({"project": "Legacy", "source": second, "path": ["p1:two.mp4"]})
        return first, second

    def test_opening_folds_the_rows_into_one_source(self, api):
        first, _ = self.a_legacy_chain(api)
        opened = api.load_project("Legacy")
        assert [s["id"] for s in sources_of(opened)] == [first]
        assert len(sources_of(opened)[0]["segments"]) == 2

    def test_the_fold_is_reported_back_to_the_caller(self, api):
        self.a_legacy_chain(api)
        report = api.load_project("Legacy")["folded"]
        assert report["folds"][0]["absorbed_names"] == ["Two"]

    def test_the_two_clips_become_the_one_clip_they_always_were(self, api):
        self.a_legacy_chain(api)
        clips = clips_of(api.load_project("Legacy"))
        assert len(clips) == 1
        assert (clips[0]["source_in"], clips[0]["source_out"]) == (0.0, 9.417)

    def test_the_absorbed_takes_now_name_the_source_that_owns_them(self, api):
        first, _ = self.a_legacy_chain(api)
        api.load_project("Legacy")
        assert {t["shot"] for t in api.list_takes("Legacy")["takes"]} == {first}

    def test_opening_again_reports_nothing(self, api):
        self.a_legacy_chain(api)
        api.load_project("Legacy")
        assert "folded" not in api.load_project("Legacy")

    def test_a_project_with_nothing_to_fold_is_untouched(self, api):
        api.create_project({"name": "Clean"})
        api.add_source({"project": "Clean", "name": "One"})
        assert "folded" not in api.load_project("Clean")

    def test_a_clean_answer_is_dropped_when_the_files_change(self, api):
        # The editor reloads a project on every sequence refresh, so "nothing to
        # fold" is remembered — against the files it was read from, never
        # onwards.
        api.create_project({"name": "Legacy"})
        api.add_source({"project": "Legacy", "name": "One"})
        api.add_source({"project": "Legacy", "name": "Two"})
        assert "folded" not in api.load_project("Legacy")

        first, second = (s["id"] for s in sources_of(api.load_project("Legacy")))
        api.record_takes({
            "project": "Legacy",
            "takes": [
                {"prompt_id": "p0", "filename": "one.mp4", "shot": first, "seconds": 5.167},
                {"prompt_id": "p1", "filename": "two.mp4", "shot": second,
                 "seconds": 4.25, "parent_take": "p0:one.mp4"},
            ],
        })
        api.set_source_path({"project": "Legacy", "source": first, "path": ["p0:one.mp4"]})
        api.set_source_path({"project": "Legacy", "source": second, "path": ["p1:two.mp4"]})
        assert api.load_project("Legacy")["folded"]["folds"]


class TestBatchOverTheApi:
    def test_it_returns_one_render_per_count(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        payload = {"prompt": a_prompt(), "profile": "real", "count": 3}
        assert len(api.prepare_batch(payload)["renders"]) == 3

    def test_the_seeds_are_consecutive(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        payload = {"prompt": a_prompt(), "profile": "real", "count": 3, "seed": 100}
        seeds = [r["seed"] for r in api.prepare_batch(payload)["renders"]]
        assert seeds == [100, 101, 102]

    def test_an_absent_seed_still_varies_them(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        payload = {"prompt": a_prompt(), "profile": "real", "count": 4}
        seeds = [r["seed"] for r in api.prepare_batch(payload)["renders"]]
        assert len(set(seeds)) == 4

    def test_a_lone_render_records_the_seed_it_used(self, api, real_workflow):
        # inject would otherwise pick one and throw it away, leaving a render
        # that cannot be reproduced.
        api.save_profile({"name": "real", "api_format": real_workflow})
        prepared = api.prepare_render({"prompt": a_prompt(), "profile": "real"})
        assert isinstance(prepared["seed"], int)

    def test_the_recorded_seed_is_the_one_in_the_workflow(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        prepared = api.prepare_render({"prompt": a_prompt(), "profile": "real"})

        profile = api.profiles.load("real")
        written = {
            prepared["workflow"][b.node_id]["inputs"][b.input_name]
            for b in profile.targets("seed")
        }
        assert written == {prepared["seed"]}

    def test_seeds_survive_a_trip_through_javascript(self, api, real_workflow):
        # Seeds ride into the browser and back. JS numbers are float64, so a
        # seed past 2**53 would round and every render in the sweep would get
        # the same one — a batch of identical videos.
        api.save_profile({"name": "real", "api_format": real_workflow})
        payload = {"prompt": a_prompt(), "profile": "real", "count": 8}

        for _ in range(200):
            seeds = [r["seed"] for r in api.prepare_batch(payload)["renders"]]
            # float() is what a JS number does to them.
            assert len({float(seed) for seed in seeds}) == 8

    def test_a_pinned_base_still_leaves_room_for_the_whole_sweep(
        self, api, real_workflow
    ):
        # The editor lets a sweep start from a seed the author pinned. A base
        # near the ceiling would run the last seeds past 2**53, where they stop
        # being distinguishable once the browser has read them back.
        api.save_profile({"name": "real", "api_format": real_workflow})
        payload = {
            "prompt": a_prompt(), "profile": "real", "count": 5,
            "seed": MAX_SEED,
        }
        seeds = [r["seed"] for r in api.prepare_batch(payload)["renders"]]

        assert max(seeds) <= MAX_SEED
        assert len({float(seed) for seed in seeds}) == 5

    def test_a_pinned_base_is_otherwise_where_the_sweep_starts(
        self, api, real_workflow
    ):
        api.save_profile({"name": "real", "api_format": real_workflow})
        payload = {
            "prompt": a_prompt(), "profile": "real", "count": 3, "seed": 4242,
        }
        seeds = [r["seed"] for r in api.prepare_batch(payload)["renders"]]
        assert seeds == [4242, 4243, 4244]

    def test_a_batch_too_large_is_a_bad_request(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        payload = {"prompt": a_prompt(), "profile": "real", "count": 99}
        assert status_of(api.prepare_batch, payload) == 400

    def test_errors_block_the_whole_batch(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        broken = a_prompt(10)
        broken["beats"] = [
            {"start": 0, "end": 6, "angle": "a", "action": "x"},
            {"start": 4, "end": 10, "angle": "b", "action": "y"},
        ]
        payload = {"prompt": broken, "profile": "real", "count": 3}
        assert status_of(api.prepare_batch, payload) == 422


class TestExportOverTheApi:
    def a_picked_project(self, api, tmp_path, speed=1.0):
        output = tmp_path / "output"
        output.mkdir(exist_ok=True)
        (output / "a.mp4").write_bytes(b"not really a video")

        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A", "duration": 8})
        project = api.load_project("Cut")
        source_id = sources_of(project)[0]["id"]
        clip_id = clips_of(project)[0]["id"]
        api.record_takes({
            "project": "Cut",
            "takes": [{"prompt_id": "r1", "filename": "a.mp4",
                       "shot": source_id, "seconds": 8.0}],
        })
        take_id = api.list_takes("Cut")["takes"][0]["id"]
        api.pick_take({"project": "Cut", "source": source_id, "take": take_id})
        if speed != 1.0:
            api.edit_clip({"project": "Cut", "clip": clip_id, "speed": speed})
        return output

    def test_a_dry_run_describes_the_edit_without_running_it(self, api, tmp_path):
        output = self.a_picked_project(api, tmp_path)
        result = api.export({
            "project": "Cut", "format": "ffmpeg",
            "output_dir": str(output), "dry_run": True,
        })
        assert result["pieces"] == 1
        assert result["command"][0] == "ffmpeg"
        assert "ok" not in result

    def test_a_clip_with_nothing_to_play_is_a_conflict(self, api, tmp_path):
        output = self.a_picked_project(api, tmp_path)
        api.add_source({"project": "Cut", "name": "Unrendered"})
        payload = {"project": "Cut", "output_dir": str(output), "dry_run": True}
        assert status_of(api.export, payload) == 409

    def test_an_unknown_format_is_a_bad_request(self, api, tmp_path):
        output = self.a_picked_project(api, tmp_path)
        payload = {"project": "Cut", "format": "prores", "output_dir": str(output)}
        assert status_of(api.export, payload) == 400

    def test_a_missing_project_is_a_not_found(self, api):
        assert status_of(api.export, {"project": "ghost"}) == 404

    def test_an_edl_is_written(self, api, tmp_path):
        output = self.a_picked_project(api, tmp_path)
        destination = tmp_path / "cut.edl"
        result = api.export({
            "project": "Cut", "format": "edl",
            "output_dir": str(output), "destination": str(destination),
        })
        assert result["ok"] is True
        assert destination.read_text(encoding="utf-8").startswith("TITLE: Cut")

    def test_a_clip_list_is_written(self, api, tmp_path):
        output = self.a_picked_project(api, tmp_path)
        destination = tmp_path / "cut.txt"
        api.export({
            "project": "Cut", "format": "shotlist",
            "output_dir": str(output), "destination": str(destination),
        })
        assert "a.mp4" in destination.read_text(encoding="utf-8")

    def test_a_missing_take_file_is_a_conflict(self, api, tmp_path):
        output = self.a_picked_project(api, tmp_path)
        (output / "a.mp4").unlink()
        payload = {"project": "Cut", "format": "ffmpeg", "output_dir": str(output)}
        assert status_of(api.export, payload) == 409

    def test_speed_reaches_the_command(self, api, tmp_path):
        output = self.a_picked_project(api, tmp_path, speed=2.0)
        result = api.export({
            "project": "Cut", "format": "ffmpeg",
            "output_dir": str(output), "dry_run": True,
        })
        graph = result["command"][result["command"].index("-filter_complex") + 1]
        assert "setpts=PTS/2.0" in graph


class TestSequenceReport:
    def test_starts_run_end_to_end(self, api):
        api.create_project({"name": "Cut"})
        for name in ("A", "B", "C"):
            api.add_source({"project": "Cut", "name": name, "duration": 8})
        report = api.sequence_report("Cut")
        assert [clip["starts_at"] for clip in report["clips"]] == [0.0, 8.0, 16.0]
        assert report["duration"] == 24.0

    def test_it_carries_timeline_findings(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        rules = {finding["rule"] for finding in api.sequence_report("Cut")["findings"]}
        assert "source.no_takes" in rules

    def test_it_reports_the_bin_beside_the_edit(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        report = api.sequence_report("Cut")
        source = report["sources"][0]
        assert source["name"] == "A"
        assert source["used_by"] == [report["clips"][0]["id"]]

    def test_a_source_reports_what_is_on_offer_at_each_depth(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        source_id = sources_of(api.load_project("Cut"))[0]["id"]
        api.add_segment({"project": "Cut", "source": source_id})
        api.record_takes({
            "project": "Cut",
            "takes": [
                {"prompt_id": "p0", "filename": "root.mp4", "shot": source_id},
                {"prompt_id": "p1", "filename": "a.mp4", "shot": source_id,
                 "parent_take": "p0:root.mp4"},
                {"prompt_id": "p2", "filename": "b.mp4", "shot": source_id,
                 "parent_take": "p0:root.mp4"},
            ],
        })
        branches = api.sequence_report("Cut")["sources"][0]["branches"]
        assert [len(step["options"]) for step in branches] == [1, 2]

    def test_a_broken_path_is_reported_on_its_source(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        source = sources_of(api.load_project("Cut"))[0]
        source["path"] = ["r9:gone.mp4"]
        api.update_source({"project": "Cut", "source": source})
        assert api.sequence_report("Cut")["sources"][0]["problems"]

    def test_the_report_carries_the_edit_as_a_playlist(self, api):
        # Resolved server side so the monitor cuts at exactly the frame the
        # export does; one entry per playable piece.
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A", "duration": 8})
        source_id = sources_of(api.load_project("Cut"))[0]["id"]
        api.record_takes({
            "project": "Cut",
            "takes": [{"prompt_id": "r1", "filename": "a.mp4",
                       "shot": source_id, "seconds": 8.0}],
        })
        entry = api.sequence_report("Cut")["playlist"][0]
        assert entry["name"] == "A"
        assert (entry["start"], entry["length"]) == (0.0, 8.0)

    def test_an_unrendered_clip_contributes_nothing_to_the_playlist(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        assert api.sequence_report("Cut")["playlist"] == []

    def test_a_clip_across_a_join_plays_as_two_pieces(self, api):
        api.create_project({"name": "Chain"})
        api.add_source({"project": "Chain", "name": "A"})
        source_id = sources_of(api.load_project("Chain"))[0]["id"]
        api.add_segment({"project": "Chain", "source": source_id})
        api.record_takes({
            "project": "Chain",
            "takes": [
                {"prompt_id": "p0", "filename": "root.mp4", "shot": source_id,
                 "seconds": 5.167},
                {"prompt_id": "p1", "filename": "child.mp4", "shot": source_id,
                 "seconds": 4.25, "parent_take": "p0:root.mp4"},
            ],
        })
        playlist = api.sequence_report("Chain")["playlist"]
        assert [entry["depth"] for entry in playlist] == [0, 1]
        assert [entry["start"] for entry in playlist] == [0.0, 5.167]

    def test_a_missing_project_is_a_not_found(self, api):
        assert status_of(api.sequence_report, "ghost") == 404


class TestPresetsOverTheApi:
    def test_listing_covers_every_kind(self, api):
        listed = api.list_presets()["presets"]
        assert set(listed) == {"style", "camera", "audio", "negatives", "storyboard"}

    def test_the_bundled_library_is_visible(self, api):
        assert "Anamorphic night city" in api.list_presets("style")["presets"]["style"]

    def test_save_then_read(self, api):
        api.save_preset({"kind": "style", "name": "Mine", "text": "A look."})
        assert api.read_preset("style", "Mine")["text"] == "A look."

    def test_an_unknown_kind_is_a_bad_request(self, api):
        assert status_of(api.read_preset, "lighting", "x") == 400

    def test_a_missing_preset_is_a_bad_request(self, api):
        assert status_of(api.read_preset, "style", "nope") == 400

    def test_deleting_a_bundled_preset_does_not_touch_it(self, api):
        assert api.delete_preset({"kind": "style", "name": "Anamorphic night city"}) == {
            "deleted": False
        }
        assert "Anamorphic night city" in api.list_presets("style")["presets"]["style"]


class TestProfilesOverTheApi:
    def test_capture_discovers_and_saves(self, api, real_workflow):
        saved = api.save_profile({"name": "real", "api_format": real_workflow})
        assert "prompt" in saved["describes"]
        assert api.list_profiles() == {"profiles": ["real"]}

    def test_hand_pinned_bindings_are_honoured(self, api, real_workflow):
        saved = api.save_profile(
            {
                "name": "real",
                "api_format": real_workflow,
                "bindings": {"duration_s": [["105:111", "value"]]},
            }
        )
        assert saved["profile"]["bindings"]["duration_s"] == [["105:111", "value"]]

    def test_capture_without_a_workflow_is_a_bad_request(self, api):
        assert status_of(api.save_profile, {"name": "empty"}) == 400

    def test_a_workflow_with_no_prompt_input_is_a_bad_request(self, api):
        payload = {"name": "headless", "api_format": {"1": {"class_type": "KSampler", "inputs": {}}}}
        assert status_of(api.save_profile, payload) == 400

    def test_a_binding_to_a_missing_node_is_a_conflict(self, api, real_workflow):
        payload = {
            "name": "stale",
            "api_format": real_workflow,
            "bindings": {"duration_s": [["nonexistent", "value"]]},
        }
        assert status_of(api.save_profile, payload) == 409

    def test_loading_a_missing_profile_is_a_bad_request(self, api):
        assert status_of(api.load_profile, "nope") == 400


class TestReferenceVocabulary:
    def test_all_four_slots_are_offered(self, api):
        kinds = [entry["kind"] for entry in api.reference_kinds()["kinds"]]
        assert kinds == ["image", "video", "video_audio", "audio"]

    def test_the_limits_match_the_node(self, api):
        limits = {e["kind"]: e["limit"] for e in api.reference_kinds()["kinds"]}
        assert limits == {"image": 9, "video": 3, "video_audio": 3, "audio": 3}

    def test_every_slot_has_a_readable_label(self, api):
        assert all(entry["label"] for entry in api.reference_kinds()["kinds"])


class TestPreviewTags:
    """Tags come from the server so the editor never computes ordinals."""

    def with_refs(self, references) -> dict:
        prompt = a_prompt(10)
        prompt["references"] = references
        return prompt

    def test_images_are_numbered_as_pictures(self, api):
        result = api.preview(
            {"prompt": self.with_refs([{"kind": "image"}, {"kind": "image"}])}
        )
        assert [r["tag"] for r in result["references"]] == ["<Picture 1>", "<Picture 2>"]

    def test_soundtracks_take_audio_numbers_before_standalone_audio(self, api):
        result = api.preview(
            {"prompt": self.with_refs([{"kind": "audio"}, {"kind": "video_audio"}])}
        )
        # Declared audio-then-soundtrack, but the soundtrack is presented first.
        assert [r["tag"] for r in result["references"]] == ["<Audio 2>", "<Audio 1>"]

    def test_each_kind_counts_separately(self, api):
        result = api.preview(
            {"prompt": self.with_refs(
                [{"kind": "image"}, {"kind": "video"}, {"kind": "image"}])}
        )
        assert [r["tag"] for r in result["references"]] == [
            "<Picture 1>", "<Video 1>", "<Picture 2>"]

    def test_roles_come_back_with_the_tags(self, api):
        result = api.preview(
            {"prompt": self.with_refs([{"kind": "image", "role": "the mood"}])}
        )
        assert result["references"][0]["role"] == "the mood"

    def test_no_references_gives_an_empty_list(self, api):
        assert api.preview({"prompt": a_prompt()})["references"] == []


class TestAudioCuesOverTheApi:
    def test_an_instant_cue_composes_as_a_moment(self, api):
        prompt = a_prompt(10)
        prompt["cues"] = [{"start": 4, "end": None, "description": "an accent hit"}]
        assert "At 00:04.000, an accent hit." in api.preview({"prompt": prompt})["prompt"]

    def test_a_sustained_cue_composes_as_a_span(self, api):
        prompt = a_prompt(10)
        prompt["cues"] = [{"start": 6, "end": 10, "description": "the bass carries under"}]
        assert (
            "From 00:06.000 to 00:10.000, the bass carries under."
            in api.preview({"prompt": prompt})["prompt"]
        )

    def test_a_cue_past_the_budget_is_flagged(self, api):
        prompt = a_prompt(10)
        prompt["cues"] = [{"start": 14, "description": "too late"}]
        rules = {f["rule"] for f in api.preview({"prompt": prompt})["findings"]}
        assert "audio.cue_out_of_range" in rules

    def test_refitting_carries_cues_with_the_picture(self, api):
        prompt = a_prompt(10)
        prompt["cues"] = [{"start": 5, "description": "hit"}]
        result = api.transform({"prompt": prompt, "ops": [{"op": "refit", "duration": 15}]})
        assert result["prompt"]["cues"][0]["start"] == 7.5


class TestGalleryOverTheApi:
    def history(self, project="Rooftop", prompt_id="run-1", filename="clip.mp4"):
        return {
            prompt_id: {
                "prompt": [
                    1, prompt_id, {},
                    {"video_prompter": {"project": project, "shot": "One",
                                        "frames": 362, "seconds": 15.083}},
                    [],
                ],
                "outputs": {
                    "92": {"images": [{"filename": filename, "subfolder": "video",
                                       "type": "output"}]}
                },
            }
        }

    def test_a_new_project_has_an_empty_gallery(self, api):
        api.create_project({"name": "Rooftop"})
        assert api.list_takes("Rooftop") == {"takes": []}

    def test_history_can_be_handed_over_raw(self, api):
        api.create_project({"name": "Rooftop"})
        result = api.record_takes({"project": "Rooftop", "history": self.history()})
        assert result["count"] == 1
        assert api.list_takes("Rooftop")["takes"][0]["frames"] == 362

    def test_another_projects_runs_are_ignored(self, api):
        api.create_project({"name": "Rooftop"})
        history = {**self.history(), **self.history(project="Someone else", prompt_id="run-2")}
        assert api.record_takes({"project": "Rooftop", "history": history})["count"] == 1

    def test_replaying_the_same_history_adds_nothing(self, api):
        api.create_project({"name": "Rooftop"})
        api.record_takes({"project": "Rooftop", "history": self.history()})
        again = api.record_takes({"project": "Rooftop", "history": self.history()})
        assert again["count"] == 0

    def test_parsed_takes_are_accepted_too(self, api):
        api.create_project({"name": "Rooftop"})
        payload = {
            "project": "Rooftop",
            "takes": [{"prompt_id": "x", "filename": "a.mp4", "shot": "One"}],
        }
        assert api.record_takes(payload)["count"] == 1

    def test_a_take_carries_a_view_url(self, api):
        api.create_project({"name": "Rooftop"})
        api.record_takes({"project": "Rooftop", "history": self.history()})
        assert api.list_takes("Rooftop")["takes"][0]["url"].startswith("/view?")

    def test_takes_filter_by_source(self, api):
        api.create_project({"name": "Rooftop"})
        api.record_takes({"project": "Rooftop", "history": self.history()})
        assert api.list_takes("Rooftop", "One")["takes"]
        assert api.list_takes("Rooftop", "Other")["takes"] == []

    def test_forget_removes_it(self, api):
        api.create_project({"name": "Rooftop"})
        api.record_takes({"project": "Rooftop", "history": self.history()})
        assert api.forget_take({"project": "Rooftop", "prompt_id": "run-1"})["forgotten"] == 1
        assert api.list_takes("Rooftop")["takes"] == []

    def test_recording_without_a_project_is_a_bad_request(self, api):
        assert status_of(api.record_takes, {"history": self.history()}) == 400

    def test_generation_times_are_applied_when_recording_takes(self, api):
        api.create_project({"name": "Rooftop"})
        api.record_takes({
            "project": "Rooftop",
            "history": self.history(prompt_id="run-1"),
            "generation_times": {"run-1": 34.2},
        })
        takes = api.list_takes("Rooftop")["takes"]
        assert takes[0]["generation_time"] == 34.2

    def test_the_response_is_json_clean(self, api):
        api.create_project({"name": "Rooftop"})
        json.dumps(api.record_takes({"project": "Rooftop", "history": self.history()}))


class TestTransform:
    """The editor's timeline and canvas operations, so the JS holds no rules."""

    def two_shots(self) -> dict:
        prompt = a_prompt(10)
        prompt["beats"] = [
            {"start": 0, "end": 6, "angle": "a", "action": "x"},
            {"start": 4, "end": 10, "angle": "b", "action": "y"},
        ]
        return prompt

    def test_fix_closes_the_overlap(self, api):
        result = api.transform({"prompt": self.two_shots(), "ops": [{"op": "fix"}]})
        beats = result["prompt"]["beats"]
        assert beats[0]["end"] == beats[1]["start"]
        assert beats[-1]["end"] == 10

    def test_fix_clears_the_overlap_finding(self, api):
        result = api.transform({"prompt": self.two_shots(), "ops": [{"op": "fix"}]})
        rules = {f["rule"] for f in result["preview"]["findings"]}
        assert "storyboard.overlap" not in rules

    def test_refit_rescales_onto_a_new_budget(self, api):
        result = api.transform(
            {"prompt": self.two_shots(), "ops": [{"op": "refit", "duration": 15}]}
        )
        assert result["prompt"]["duration"] == 15
        assert result["preview"]["frames"] == 362

    def test_ratio_sets_the_canvas_by_the_nodes_rules(self, api):
        result = api.transform({"prompt": a_prompt(), "ops": [{"op": "ratio", "ratio": "21:9"}]})
        assert (result["prompt"]["width"], result["prompt"]["height"]) == (1536, 672)

    def test_ops_apply_in_order(self, api):
        result = api.transform(
            {"prompt": self.two_shots(), "ops": [{"op": "refit", "duration": 15}, {"op": "fix"}]}
        )
        assert result["prompt"]["beats"][-1]["end"] == 15

    def test_no_ops_is_a_pass_through(self, api):
        result = api.transform({"prompt": a_prompt(8), "ops": []})
        assert result["prompt"]["duration"] == 8

    def test_an_unknown_op_is_a_bad_request(self, api):
        assert status_of(api.transform, {"prompt": a_prompt(), "ops": [{"op": "explode"}]}) == 400

    def test_a_missing_ops_list_is_a_bad_request(self, api):
        assert status_of(api.transform, {"prompt": a_prompt()}) == 400

    def test_a_malformed_ratio_is_a_bad_request(self, api):
        payload = {"prompt": a_prompt(), "ops": [{"op": "ratio", "ratio": "wide"}]}
        assert status_of(api.transform, payload) == 400

    def test_a_refit_without_a_duration_is_a_bad_request(self, api):
        assert status_of(api.transform, {"prompt": a_prompt(), "ops": [{"op": "refit"}]}) == 400

    def test_the_response_is_json_clean(self, api):
        json.dumps(api.transform({"prompt": self.two_shots(), "ops": [{"op": "fix"}]}))


class TestPreviewAndRender:
    def test_preview_returns_text_and_frames(self, api):
        result = api.preview({"prompt": a_prompt(10)})
        assert result["frames"] == 243
        assert "A cinematic look" in result["prompt"]
        assert result["prompt"].startswith("integrated_multimodal_description:")

    def test_preview_needs_a_prompt_object(self, api):
        assert status_of(api.preview, {}) == 400

    def test_prepare_returns_a_submittable_workflow(self, api, real_workflow):
        api.save_profile(
            {
                "name": "real",
                "api_format": real_workflow,
                "bindings": {"duration_s": [["105:111", "value"]]},
            }
        )
        result = api.prepare_render({"prompt": a_prompt(12), "profile": "real", "seed": 5})

        assert result["workflow"]["105:15"]["inputs"]["noise_seed"] == 5
        assert result["workflow"]["105:111"]["inputs"]["value"] == 12.0
        assert result["frames"] == 294

    def test_prepare_without_a_profile_is_a_bad_request(self, api):
        assert status_of(api.prepare_render, {"prompt": a_prompt()}) == 400

    def test_prepare_with_an_unknown_profile_is_a_not_found(self, api):
        assert status_of(api.prepare_render, {"prompt": a_prompt(), "profile": "ghost"}) == 404

    def test_a_prompt_with_errors_is_unprocessable_and_says_why(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        broken = a_prompt(10)
        broken["beats"] = [
            {"start": 0, "end": 6, "angle": "a", "action": "x"},
            {"start": 4, "end": 10, "angle": "b", "action": "y"},
        ]

        with pytest.raises(ApiError) as raised:
            api.prepare_render({"prompt": broken, "profile": "real"})

        assert raised.value.status == 422
        assert any(f["rule"] == "storyboard.overlap" for f in raised.value.extra["findings"])

    def test_the_block_can_be_waived(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        broken = a_prompt(10)
        broken["beats"] = [
            {"start": 0, "end": 6, "angle": "a", "action": "x"},
            {"start": 4, "end": 10, "angle": "b", "action": "y"},
        ]
        assert api.prepare_render({"prompt": broken, "profile": "real", "strict": False})

    def test_every_response_is_json_clean(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        json.dumps(api.prepare_render({"prompt": a_prompt(8), "profile": "real", "seed": 1}))

    def test_a_quality_name_resolves_to_megapixels_and_steps(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "preview"}
        )
        assert result["workflow"]["105:9"]["inputs"]["steps"] == 8
        assert result["workflow"]["115"]["inputs"]["megapixels"] == 0.2

    def test_final_quality_drives_the_graph_harder(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "final"}
        )
        assert result["workflow"]["105:9"]["inputs"]["steps"] == 25
        assert result["workflow"]["115"]["inputs"]["megapixels"] == 1.0

    def test_the_two_pass_tier_splits_the_graph_in_two(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "final_2pass"}
        )
        graph = result["workflow"]
        # Pass 1 runs the obedient canvas; the schedule carries both passes.
        assert graph["115"]["inputs"]["megapixels"] == 0.2
        assert graph["105:9"]["inputs"]["steps"] == 20
        split = _one_of_class(graph, "SplitSigmas")
        assert graph[split]["inputs"]["step"] == 15
        assert _one_of_class(graph, "MinimaxH3LatentUpscaler3D")
        # Two samplers now, not one.
        assert len([n for n in graph.values()
                    if n.get("class_type") == "SamplerCustomAdvanced"]) == 2

    def test_a_two_pass_take_records_the_size_it_finished_at(self, api, real_workflow):
        # Not pass 1's canvas: the file on disk is the upscaled one, and the
        # gallery reads a take's tier back off what it recorded.
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "final_2pass"}
        )
        fine = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "final"}
        )
        assert (result["width"], result["height"]) == (fine["width"], fine["height"])
        upscale = _one_of_class(result["workflow"], "MinimaxH3LatentUpscaler3D")
        assert result["workflow"][upscale]["inputs"]["width"] == result["width"]
        assert result["workflow"][upscale]["inputs"]["height"] == result["height"]

    def test_the_turbo_two_pass_tier_cuts_its_eight_steps_at_six(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        graph = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "turbo_2pass"}
        )["workflow"]
        assert graph["115"]["inputs"]["megapixels"] == 0.2
        assert graph["105:9"]["inputs"]["steps"] == 8
        assert graph[_one_of_class(graph, "SplitSigmas")]["inputs"]["step"] == 6

    def test_the_turbo_tiers_finish_on_the_same_canvas(self, api, real_workflow):
        # Same steps, same finished size: the only difference between them is
        # where those steps were spent, which is what makes the A/B honest.
        api.save_profile({"name": "real", "api_format": real_workflow})
        flat = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "turbo_hd"}
        )
        split = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "turbo_2pass"}
        )
        assert (split["width"], split["height"]) == (flat["width"], flat["height"])
        assert split["steps"] == flat["steps"] == 8

    def test_a_two_pass_render_reports_where_it_cut(self, api, real_workflow):
        # `steps` and the finished canvas are identical for turbo_hd and
        # turbo_2pass, so this is the only thing that tells them apart later.
        api.save_profile({"name": "real", "api_format": real_workflow})
        split = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "turbo_2pass"}
        )
        flat = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "turbo_hd"}
        )
        assert split["pass1_steps"] == 6
        assert flat["pass1_steps"] is None

    def test_a_two_pass_reshoot_reports_where_it_cut(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_reshoot({
            "profile": "real",
            "prompt_text": "the take's frozen prompt",
            "width": 608, "height": 352, "frames": 192, "seconds": 8.0,
            "seed": 4242, "quality": "turbo_2pass",
        })
        assert result["pass1_steps"] == 6

    def test_a_moved_split_is_the_one_reported(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        api.create_project({"name": "P"})
        api.update_project({"name": "P", "project": {
            "name": "P",
            "settings": {"quality_presets": {"turbo_2pass": {"pass1_steps": 6}}},
        }})
        result = api.prepare_render({
            "prompt": a_prompt(8), "profile": "real",
            "quality": "turbo_2pass", "project": "P",
        })
        assert result["pass1_steps"] == 6
        graph = result["workflow"]
        assert graph[_one_of_class(graph, "SplitSigmas")]["inputs"]["step"] == 6

    def test_the_extend_latent_saved_is_the_finished_one(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        graph = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "final_2pass"}
        )["workflow"]
        save = _one_of_class(graph, "MiniMaxH3MotionContextSaveLatent")
        second = _second_sampler(graph)
        assert graph[save]["inputs"]["latent"] == [second, 0]

    def test_a_single_pass_tier_adds_no_second_pass(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        graph = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "quality": "final"}
        )["workflow"]
        assert not [n for n in graph.values() if n.get("class_type") == "SplitSigmas"]

    def test_a_two_pass_sweep_splits_every_render(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        renders = api.prepare_batch({
            "prompt": a_prompt(8), "profile": "real", "count": 3,
            "quality": "final_2pass",
        })["renders"]
        assert all(_one_of_class(r["workflow"], "SplitSigmas") for r in renders)

    def test_a_two_pass_reshoot_splits_too(self, api, real_workflow):
        # This is how a picked preview draft gets promoted without re-authoring.
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_reshoot({
            "profile": "real",
            "prompt_text": "the take's frozen prompt",
            "width": 608, "height": 352, "frames": 192, "seconds": 8.0,
            "seed": 4242, "quality": "final_2pass",
        })
        assert _one_of_class(result["workflow"], "SplitSigmas")
        assert result["width"] > 608

    def test_a_project_may_move_the_split(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        api.create_project({"name": "P"})
        api.update_project({"name": "P", "project": {
            "name": "P",
            "settings": {"quality_presets": {"final_2pass": {"pass1_steps": 12}}},
        }})
        graph = api.prepare_render({
            "prompt": a_prompt(8), "profile": "real",
            "quality": "final_2pass", "project": "P",
        })["workflow"]
        assert graph[_one_of_class(graph, "SplitSigmas")]["inputs"]["step"] == 12

    def test_no_quality_leaves_the_workflow_as_captured(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_render({"prompt": a_prompt(8), "profile": "real"})
        # Untouched: the graph keeps whatever it was captured with.
        assert result["workflow"]["105:9"]["inputs"]["steps"] == 8

    def test_a_seed_is_the_seed_that_renders(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_render(
            {"prompt": a_prompt(8), "profile": "real", "seed": 4242, "quality": "final"}
        )
        assert result["seed"] == 4242
        assert result["workflow"]["105:15"]["inputs"]["noise_seed"] == 4242

    def test_a_reshoot_injects_the_takes_frozen_text_and_seed(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_reshoot({
            "profile": "real",
            "prompt_text": "the take's frozen prompt",
            "width": 1344, "height": 768, "frames": 192, "seconds": 8.0,
            "seed": 4242, "quality": "final",
        })
        assert result["seed"] == 4242
        assert result["frames"] == 192
        assert result["workflow"]["105:104"]["inputs"]["prompt"] == "the take's frozen prompt"
        assert result["workflow"]["105:15"]["inputs"]["noise_seed"] == 4242
        assert result["workflow"]["105:9"]["inputs"]["steps"] == 25

    def test_a_reshoot_without_prompt_text_is_a_bad_request(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        payload = {"profile": "real", "width": 1344, "height": 768,
                   "frames": 192, "seconds": 8.0}
        assert status_of(api.prepare_reshoot, payload) == 400

    def test_an_extend_continues_a_take_from_its_own_video(self, api):
        # no saved profile needed: the extend graph is built server-side
        result = api.prepare_extend({
            "source": "video/take_00007_.mp4 [output]",
            "prompt_text": "she turns and walks away",
            "width": 608, "height": 352, "frames": 124, "seconds": 5.0, "seed": 5,
        })
        assert result["prompt"] == "she turns and walks away"
        wf = result["workflow"]
        assert node_of(wf, "LoadVideo")["inputs"]["file"] == (
            "video/take_00007_.mp4 [output]")
        assert node_of(wf, "MiniMaxH3ImageToVideo")["inputs"]["prompt"] == (
            "she turns and walks away")
        assert node_of(wf, "RandomNoise")["inputs"]["noise_seed"] == 5

    def test_an_extend_without_a_source_is_a_bad_request(self, api):
        payload = {"prompt_text": "x", "width": 608, "height": 352,
                   "frames": 124, "seconds": 5.0}
        assert status_of(api.prepare_extend, payload) == 400

    def test_an_extend_refuses_a_selflift_tier_rather_than_mislabel_it(self, api):
        # The continuation graph only runs the classic two-stage split; a take
        # rendered that way must not be filed under a SelfLift or Singularity name.
        for tier in ("selflift_hd", "selflift_fhd", "selflift_final", "singularity_2pass"):
            payload = {"source": "s.mp4", "prompt_text": "x", "width": 1344,
                       "height": 768, "frames": 124, "seconds": 5.0, "quality": tier}
            assert status_of(api.prepare_extend, payload) == 400

    def test_singularity_2pass_prepare_render(self, api, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        res = api.prepare_render({
            "prompt": a_prompt(8), "profile": "real",
            "quality": "singularity_2pass",
        })
        wf = res["workflow"]
        assert _one_of_class(wf, "MinimaxH3LatentUpscaler3D")
        assert wf["105:6"]["inputs"]["unet_name"] == twostage.DEFAULT_SINGULARITY_UNET
        assert res["width"] == 1920
        assert res["height"] == 1088
        assert res["steps"] == 24
        assert res["pass1_steps"] == 20
        assert res["quality"] == "singularity_2pass"

    def test_an_extend_without_prompt_text_is_a_bad_request(self, api):
        payload = {"source": "s.mp4", "width": 608, "height": 352,
                   "frames": 124, "seconds": 5.0}
        assert status_of(api.prepare_extend, payload) == 400

    def test_the_context_flag_reaches_both_windows(self, api):
        result = api.prepare_extend({
            "source": "s.mp4", "prompt_text": "x",
            "width": 608, "height": 352, "frames": 124, "seconds": 5.0, "context": 5,
        })
        mc = node_of(result["workflow"], "MiniMaxH3MotionContext")["inputs"]
        assert mc["context_length"] == 5
        assert mc["audio_context_length"] == 5

    def test_an_off_grid_context_is_snapped_down_and_reported_honestly(self, api):
        # the node would snap 30 -> 22 and only log it; the take must not claim 30
        result = api.prepare_extend({
            "source": "s.mp4", "prompt_text": "x",
            "width": 608, "height": 352, "frames": 124, "seconds": 5.0, "context": 30,
        })
        assert node_of(result["workflow"],
                       "MiniMaxH3MotionContext")["inputs"]["context_length"] == 22
        assert result["context_length"] == 22

    def test_a_context_longer_than_the_clip_is_refused_before_it_costs_a_render(self, api):
        # the node raises on this too, but only once the job reaches the GPU
        payload = {"source": "s.mp4", "prompt_text": "x", "width": 608,
                   "height": 352, "frames": 124, "seconds": 5.0, "context": 124}
        assert status_of(api.prepare_extend, payload) == 400

    def test_a_context_of_zero_is_a_bad_request(self, api):
        payload = {"source": "s.mp4", "prompt_text": "x", "width": 608,
                   "height": 352, "frames": 124, "seconds": 5.0, "context": 0}
        assert status_of(api.prepare_extend, payload) == 400

    def test_an_extend_that_chose_no_context_reports_the_default(self, api):
        result = api.prepare_extend({
            "source": "s.mp4", "prompt_text": "x",
            "width": 608, "height": 352, "frames": 124, "seconds": 5.0,
        })
        assert result["context_length"] == 22

    def test_an_extend_runs_the_length_it_was_asked_for(self, api):
        # A continuation is not obliged to match the clip it continues: a 12s
        # opener can be followed by an 8s beat, so a budget in seconds is
        # enough and the frame grid is applied here, not in the client.
        result = api.prepare_extend({
            "source": "s.mp4", "prompt_text": "x",
            "width": 608, "height": 352, "seconds": 8,
        })
        assert result["frames"] == 192
        assert result["seconds"] == 8.0
        assert result["workflow"]["i2v"]["inputs"]["length"] == 192

    def test_an_extend_reports_the_grid_length_not_the_round_one(self, api):
        # 12s is 288 frames, which is not on the 17k+5 grid; the node would snap
        # up to 294 either way, so the record says 12.25s rather than 12.0.
        result = api.prepare_extend({
            "source": "s.mp4", "prompt_text": "x",
            "width": 608, "height": 352, "seconds": 12,
        })
        assert (result["frames"], result["seconds"]) == (294, 12.25)

    def test_an_extend_with_neither_length_is_a_bad_request(self, api):
        payload = {"source": "s.mp4", "prompt_text": "x",
                   "width": 608, "height": 352}
        assert status_of(api.prepare_extend, payload) == 400

    def test_an_extend_preserves_parent_take(self, api):
        result = api.prepare_extend({
            "source": "s.mp4", "prompt_text": "x",
            "width": 608, "height": 352, "frames": 124, "seconds": 5.0,
            "parent_take": "parent_prompt_123:take1.mp4",
        })
        assert result["parent_take"] == "parent_prompt_123:take1.mp4"

    def an_extend(self, **extra):
        return {
            "source": "video/take_00007_.mp4 [output]",
            "prompt_text": "she turns and walks away",
            "width": 608, "height": 352, "frames": 124, "seconds": 5.0,
            **extra,
        }

    def test_the_tier_comes_back_by_name(self, api):
        # The client records it on the take, because a continuation's numbers
        # cannot be read backwards into a tier: it renders at its PARENT's
        # canvas, so Quick on a 720p parent is 1.0 MP at 8 steps, which is what
        # turbo_hd is. The badge used to say Turbo HD for exactly that reason.
        result = api.prepare_extend(self.an_extend(
            width=1344, height=768, quality="preview",
        ))
        assert result["quality"] == "preview"

    def test_a_sweep_reports_the_tier_on_every_candidate(self, api):
        result = api.prepare_extend_batch(self.an_extend(count=3, quality="preview"))
        assert {r["quality"] for r in result["renders"]} == {"preview"}

    def test_an_extend_that_named_no_tier_reports_none(self, api):
        # The CLI and the bare node render at the workflow's own numbers; a
        # name invented here would be a claim nobody made.
        assert api.prepare_extend(self.an_extend())["quality"] is None

    def test_a_stale_tier_name_reports_the_one_that_really_ran(self, api):
        # `quality.resolve` falls back to preview on an unknown name, so the
        # reported name has to fall back with it or the take records a tier the
        # render never used.
        result = api.prepare_extend(self.an_extend(quality="turbo_4pass_deluxe"))
        assert result["quality"] == "preview"

    def test_a_one_pass_tier_keeps_the_parents_canvas(self, api):
        # The pinned frames come from the parent, so a continuation cannot
        # deliver a resolution that clip never had. Only the steps vary.
        result = api.prepare_extend(self.an_extend(
            width=1344, height=768, quality="final",
        ))
        assert (result["width"], result["height"]) == (1344, 768)
        assert result["steps"] == 25

    def test_a_two_pass_tier_proves_it_small_and_finishes_at_the_parents_canvas(self, api):
        result = api.prepare_extend(self.an_extend(
            width=1344, height=768, frames=192, seconds=8.0, quality="final_2pass",
        ))
        assert result["pass1_steps"] == 15
        assert (result["width"], result["height"]) == (1344, 768)
        # Pass 1 runs at the tier's own small canvas...
        assert node_of(result["workflow"], "MiniMaxH3ImageToVideo")["inputs"]["width"] < 1344
        # ...and pass 2 gets a Motion Context of its own, because H3 packs
        # continuation guides into the grid of the latent being sampled.
        contexts = [n for n in result["workflow"].values()
                    if n["class_type"] == "MiniMaxH3MotionContext"]
        assert len(contexts) == 2

    def test_a_two_pass_extend_of_a_small_clip_is_refused(self, api):
        # It would upscale 608x352 to 608x352: two passes, no second canvas,
        # and a take that recorded itself as two-pass regardless.
        payload = self.an_extend(width=608, height=352, quality="final_2pass")
        assert status_of(api.prepare_extend, payload) == 400

    def test_an_extend_sweep_varies_only_the_seed(self, api):
        # The point of sweeping a continuation: same parent, same pinned run,
        # same words, so the takes differ only in what the model invented.
        result = api.prepare_extend_batch(self.an_extend(
            count=3, seed=77, context=8, parent_take="p:take1.mp4",
        ))
        renders = result["renders"]

        assert [r["seed"] for r in renders] == [77, 78, 79]
        assert {node_of(r["workflow"], "RandomNoise")["inputs"]["noise_seed"]
                for r in renders} == {77, 78, 79}
        assert {r["prompt"] for r in renders} == {"she turns and walks away"}
        assert {r["parent_take"] for r in renders} == {"p:take1.mp4"}
        # 8 is off the node's grid; the batch path snaps it down the same way
        # the single one does, and reports what was really pinned.
        assert {r["context_length"] for r in renders} == {5}
        assert {node_of(r["workflow"], "LoadVideo")["inputs"]["file"]
                for r in renders} == {"video/take_00007_.mp4 [output]"}

    def test_an_extend_sweep_rolls_a_base_when_none_is_pinned(self, api):
        renders = api.prepare_extend_batch(self.an_extend(count=4))["renders"]
        seeds = [r["seed"] for r in renders]

        assert seeds == [seeds[0] + i for i in range(4)]
        # float() is what a JS number does to them on the way back.
        assert len({float(seed) for seed in seeds}) == 4

    def test_an_extend_sweep_leaves_room_below_the_seed_ceiling(self, api):
        renders = api.prepare_extend_batch(
            self.an_extend(count=5, seed=MAX_SEED)
        )["renders"]
        seeds = [r["seed"] for r in renders]

        assert max(seeds) <= MAX_SEED
        assert len({float(seed) for seed in seeds}) == 5

    def test_an_extend_sweep_too_large_is_a_bad_request(self, api):
        payload = self.an_extend(count=99)
        assert status_of(api.prepare_extend_batch, payload) == 400

    def test_an_extend_sweep_is_checked_like_a_single_one(self, api):
        # Every door the single path guards, the batch path guards too — a
        # sweep that skipped one would fail on the GPU instead of at the door.
        no_source = {"prompt_text": "x", "width": 608, "height": 352,
                     "frames": 124, "seconds": 5.0, "count": 3}
        assert status_of(api.prepare_extend_batch, no_source) == 400

        no_text = {"source": "s.mp4", "width": 608, "height": 352,
                   "frames": 124, "seconds": 5.0, "count": 3}
        assert status_of(api.prepare_extend_batch, no_text) == 400

        # Context longer than the clip it starts.
        too_much = self.an_extend(count=3, context=200)
        assert status_of(api.prepare_extend_batch, too_much) == 400


class TestClipSound:
    """Muting a clip, and hearing a take whose picture was rejected."""

    def two_takes(self, api):
        """One source, two re-rolls of it, and the clip that uses them."""
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A"})
        source_id = sources_of(api.load_project("Cut"))[0]["id"]
        api.record_takes({
            "project": "Cut",
            "takes": [
                {"prompt_id": "p0", "filename": "seen.mp4", "shot": source_id,
                 "seconds": 5.167},
                {"prompt_id": "p1", "filename": "heard.mp4", "shot": source_id,
                 "seconds": 5.167},
            ],
        })
        api.pick_take({"project": "Cut", "source": source_id, "take": "p0:seen.mp4"})
        clip_id = clips_of(api.load_project("Cut"))[0]["id"]
        return source_id, clip_id, "p1:heard.mp4"

    def sound_of(self, api, clip_id):
        report = api.sequence_report("Cut")
        return next(c["audio"] for c in report["clips"] if c["id"] == clip_id)

    def test_a_clip_starts_out_playing_its_own_sound(self, api):
        _, clip_id, _ = self.two_takes(api)
        sound = self.sound_of(api, clip_id)
        assert (sound["muted"], sound["borrowed"]) == (False, False)

    def test_muting_a_clip_is_recorded_on_the_clip(self, api):
        _, clip_id, _ = self.two_takes(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "audio": {"muted": True}})
        assert self.sound_of(api, clip_id)["muted"] is True

    def test_naming_a_take_works_out_the_path_to_it(self, api):
        source_id, clip_id, heard = self.two_takes(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "audio": {"take": heard}})
        sound = self.sound_of(api, clip_id)
        assert sound["borrowed"] is True
        assert sound["take"] == heard
        assert sound["path"] == [heard]
        assert sound["source_id"] == source_id

    def test_borrowing_leaves_the_picture_alone(self, api):
        _, clip_id, heard = self.two_takes(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "audio": {"take": heard}})
        clip = next(c for c in api.sequence_report("Cut")["clips"] if c["id"] == clip_id)
        assert clip["path"] == ["p0:seen.mp4"]

    def test_clearing_the_take_gives_a_clip_its_own_sound_back(self, api):
        _, clip_id, heard = self.two_takes(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "audio": {"take": heard}})
        api.edit_clip({"project": "Cut", "clip": clip_id, "audio": {"take": None}})
        sound = self.sound_of(api, clip_id)
        assert (sound["borrowed"], sound["source_id"], sound["path"]) == (False, "", [])

    def test_borrowing_a_take_that_is_not_there_is_a_not_found(self, api):
        _, clip_id, _ = self.two_takes(api)
        payload = {"project": "Cut", "clip": clip_id, "audio": {"take": "p9:nope.mp4"}}
        assert status_of(api.edit_clip, payload) == 404

    def test_audio_that_is_not_an_object_is_a_bad_request(self, api):
        _, clip_id, _ = self.two_takes(api)
        payload = {"project": "Cut", "clip": clip_id, "audio": "mute"}
        assert status_of(api.edit_clip, payload) == 400

    def test_gain_is_clamped_to_what_is_useful(self, api):
        _, clip_id, _ = self.two_takes(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "audio": {"gain_db": 90}})
        assert self.sound_of(api, clip_id)["gain_db"] == 6.0

    def test_only_borrowed_sound_reaches_the_monitor_s_audio_list(self, api):
        _, clip_id, heard = self.two_takes(api)
        # Its own sound rides along with the picture the monitor already plays.
        assert api.sequence_report("Cut")["audio"] == []
        api.edit_clip({"project": "Cut", "clip": clip_id, "audio": {"take": heard}})
        played = api.sequence_report("Cut")["audio"]
        assert [entry["take_id"] for entry in played] == [heard]
        assert played[0]["start"] == 0.0

    def test_a_muted_clip_reaches_it_as_nothing_to_play(self, api):
        _, clip_id, _ = self.two_takes(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "audio": {"muted": True}})
        assert api.sequence_report("Cut")["audio"] == []

    def test_borrowed_sound_follows_the_clip_s_trim(self, api):
        _, clip_id, heard = self.two_takes(api)
        api.edit_clip({"project": "Cut", "clip": clip_id, "audio": {"take": heard}})
        api.edit_clip({"project": "Cut", "clip": clip_id,
                       "source_in": 1.0, "source_out": 3.0})
        entry = api.sequence_report("Cut")["audio"][0]
        assert (entry["from"], entry["to"]) == (1.0, 3.0)


class TestPromptHistory:
    """The server files a revision itself, so no caller can forget to."""

    def written(self, api, project: str = "Cut") -> dict:
        return sources_of(api.load_project(project))[0]

    def edit(self, api, style: str, note: str = "") -> dict:
        source = self.written(api)
        prompt_of(source)["style"] = style
        payload = {"project": "Cut", "source": source}
        if note:
            payload["note"] = note
        return api.update_source(payload)

    @pytest.fixture
    def cut(self, api):
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "Rooftop"})
        return api

    def test_a_save_that_changes_a_prompt_files_the_one_it_replaced(self, cut):
        self.edit(cut, "First look.")
        self.edit(cut, "Second look.")

        history = cut.segment_revisions("Cut", self.written(cut)["id"])
        assert [r["prompt"]["style"] for r in history["revisions"]] == [
            "First look.", ""
        ]

    def test_the_newest_revision_comes_first(self, cut):
        self.edit(cut, "First look.")
        self.edit(cut, "Second look.")
        history = cut.segment_revisions("Cut", self.written(cut)["id"])

        assert history["revisions"][0]["prompt"]["style"] == "First look."
        assert history["current"]["prompt"]["style"] == "Second look."

    def test_a_save_that_changes_nothing_files_nothing(self, cut):
        source = self.written(cut)
        cut.update_source({"project": "Cut", "source": source})

        assert cut.segment_revisions("Cut", source["id"])["revisions"] == []

    def test_the_note_says_where_the_save_came_from(self, cut):
        self.edit(cut, "Faster.", note="Gemini: made it faster")
        history = cut.segment_revisions("Cut", self.written(cut)["id"])

        assert history["revisions"][0]["note"] == "Gemini: made it faster"

    def test_a_client_that_posts_no_history_cannot_erase_it(self, cut):
        # A tab that loaded before this feature existed, or an agent that built
        # a source from the schema, sends segments with no `revisions` key. The
        # history is the server's, read off disk, so neither loses it.
        self.edit(cut, "First look.")
        bare = self.written(cut)
        bare["segments"][0].pop("revisions", None)
        bare["segments"][0]["prompt"]["style"] = "Third look."
        cut.update_source({"project": "Cut", "source": bare})

        styles = [
            r["prompt"]["style"]
            for r in cut.segment_revisions("Cut", bare["id"])["revisions"]
        ]
        assert styles == ["First look.", ""]

    def test_saving_the_whole_project_files_revisions_too(self, cut):
        project = cut.load_project("Cut")["project"]
        project["sources"][0]["segments"][0]["prompt"]["style"] = "Whole-bin save."
        cut.update_project({"name": "Cut", "project": project})

        history = cut.segment_revisions("Cut", self.written(cut)["id"])
        assert len(history["revisions"]) == 1

    def test_a_segment_planned_in_this_save_has_no_past(self, cut):
        cut.add_segment({"project": "Cut", "source": self.written(cut)["id"]})
        source = self.written(cut)

        assert source["segments"][1]["revision_count"] == 0

    def test_the_project_reply_counts_the_history_without_carrying_it(self, cut):
        self.edit(cut, "First look.")
        segment = self.written(cut)["segments"][0]

        assert "revisions" not in segment
        assert segment["revision_count"] == 1

    def test_each_revision_carries_the_text_the_editor_diffs(self, cut):
        self.edit(cut, "A cinematic look.")
        self.edit(cut, "Something else.")
        history = cut.segment_revisions("Cut", self.written(cut)["id"])

        assert "A cinematic look" in history["revisions"][0]["text"]

    def test_restoring_puts_a_past_prompt_back_on_that_segment_alone(self, cut):
        self.edit(cut, "First look.")
        self.edit(cut, "Second look.")
        source_id = self.written(cut)["id"]
        oldest = cut.segment_revisions("Cut", source_id)["revisions"][-1]

        cut.restore_revision(
            {"project": "Cut", "source": source_id, "revision": oldest["id"]}
        )
        assert prompt_of(self.written(cut))["style"] == ""

    def test_restoring_files_what_it_replaced(self, cut):
        self.edit(cut, "First look.")
        source_id = self.written(cut)["id"]
        revision = cut.segment_revisions("Cut", source_id)["revisions"][0]
        cut.restore_revision(
            {"project": "Cut", "source": source_id, "revision": revision["id"]}
        )

        latest = cut.segment_revisions("Cut", source_id)["revisions"][0]
        assert latest["prompt"]["style"] == "First look."
        assert latest["note"].startswith("restored ")

    def test_restoring_a_revision_that_is_not_there_is_a_not_found(self, cut):
        source_id = self.written(cut)["id"]
        assert status_of(
            cut.restore_revision,
            {"project": "Cut", "source": source_id, "revision": "rev_nope"},
        ) == 404

    def test_reading_the_history_of_a_missing_source_is_a_not_found(self, cut):
        assert status_of(cut.segment_revisions, "Cut", "src_nope") == 404

    def test_reading_the_history_of_a_missing_segment_is_a_not_found(self, cut):
        source_id = self.written(cut)["id"]
        assert status_of(cut.segment_revisions, "Cut", source_id, "seg_nope") == 404


class TestCast:
    """The project's reusable identities, over the API."""

    @staticmethod
    def _member(api, project="Cast", **over):
        member = {
            "name": "Mara",
            "kind": "character",
            "appearance": "a tall woman in a red coat",
            **over,
        }
        return api.save_member({"project": project, "member": member})["member"]

    @pytest.fixture
    def project(self, api):
        api.create_project({"name": "Cast"})
        return "Cast"

    def test_a_new_project_starts_with_an_empty_cast(self, api, project):
        assert api.list_cast(project)["cast"] == []

    def test_saving_a_member_mints_an_id(self, api, project):
        assert self._member(api)["id"].startswith("cast_")

    def test_a_saved_member_comes_back_in_the_list(self, api, project):
        saved = self._member(api)
        assert [m["id"] for m in api.list_cast(project)["cast"]] == [saved["id"]]

    def test_saving_the_same_id_again_overwrites_in_place(self, api, project):
        saved = self._member(api)
        self._member(api, id=saved["id"], appearance="a tall woman in a parka")
        cast = api.list_cast(project)["cast"]
        assert len(cast) == 1
        assert cast[0]["appearance"] == "a tall woman in a parka"

    def test_a_member_needs_a_name(self, api, project):
        assert status_of(api.save_member, {"project": project, "member": {}}) == 400

    def test_an_unknown_kind_is_rejected_before_it_is_stored(self, api, project):
        # from_dict is forgiving so a stored project always loads; the API is
        # not, so a typo in a write is answered rather than silently corrected.
        member = {"name": "Mara", "kind": "villain"}
        saved = api.save_member({"project": project, "member": member})["member"]
        assert saved["kind"] == "character"

    def test_deleting_a_member_removes_it(self, api, project):
        saved = self._member(api)
        api.delete_member({"project": project, "member": saved["id"]})
        assert api.list_cast(project)["cast"] == []

    def test_deleting_something_that_is_not_there_is_a_404(self, api, project):
        payload = {"project": project, "member": "cast_deadbeef"}
        assert status_of(api.delete_member, payload) == 404

    def test_the_cast_survives_a_reload(self, api, project):
        saved = self._member(api)
        assert api.load_project(project)["project"]["cast"][0]["id"] == saved["id"]

    def test_a_member_resolves_into_the_preview(self, api, project):
        saved = self._member(api)
        prompt = a_prompt()
        prompt["beats"][0]["lines"] = [
            {"speakers": [saved["id"]], "text": "We are late."}
        ]
        preview = api.preview({"project": project, "prompt": prompt})
        assert "tall woman in a red coat" in preview["prompt"]

    def test_without_the_project_name_nothing_resolves(self, api, project):
        saved = self._member(api)
        prompt = a_prompt()
        prompt["beats"][0]["lines"] = [
            {"speakers": [saved["id"]], "text": "We are late."}
        ]
        preview = api.preview({"prompt": prompt})
        assert "tall woman in a red coat" not in preview["prompt"]

    @staticmethod
    def _source_with(api, project, prompt) -> str:
        """A source in the bin whose first segment carries `prompt`."""
        created = api.add_source({"project": project})["project"]
        source = created["sources"][-1]
        source["segments"][0]["prompt"] = prompt
        api.update_source({"project": project, "source": source})
        return source["id"]

    def test_usage_names_the_segments_that_would_be_stranded(self, api, project):
        saved = self._member(api)
        prompt = a_prompt()
        prompt["beats"][0]["lines"] = [
            {"speakers": [saved["id"]], "text": "We are late."}
        ]
        source_id = self._source_with(api, project, prompt)
        usage = api.member_usage(project, saved["id"])["used_by"]
        assert [entry["source"] for entry in usage] == [source_id]
        assert usage[0]["speaking"] is True

    def test_a_silent_member_is_used_too(self, api, project):
        saved = self._member(api)
        prompt = a_prompt()
        prompt["cast"] = [saved["id"]]
        self._source_with(api, project, prompt)
        usage = api.member_usage(project, saved["id"])["used_by"]
        assert len(usage) == 1
        assert usage[0]["speaking"] is False

    def test_a_member_nobody_names_has_no_usage(self, api, project):
        saved = self._member(api)
        self._source_with(api, project, a_prompt())
        assert api.member_usage(project, saved["id"])["used_by"] == []

    def test_renaming_a_member_moves_their_mentions(self, api, project):
        """A mention is stored by name, so a rename has to reach the prose."""
        saved = self._member(api)
        prompt = a_prompt()
        prompt["beats"][0]["action"] = "@Mara opens the door."
        source_id = self._source_with(api, project, prompt)

        result = api.save_member(
            {"project": project, "member": {**saved, "name": "Marisol"}}
        )
        assert result["renamed_mentions"] == 1

        source = api.get_source(project, source_id)["source"]
        assert source["segments"][0]["prompt"]["beats"][0]["action"] == (
            "@Marisol opens the door."
        )

    def test_a_mention_composes_to_the_members_tag(self, api, project):
        saved = self._member(api, references=[{"kind": "image", "path": "mara.png"}])
        prompt = a_prompt()
        prompt["beats"][0]["action"] = "@Mara opens the door."
        preview = api.preview({"project": project, "prompt": prompt})
        assert "<Subject 1> opens the door." in preview["prompt"]
        assert [entry["label"] for entry in preview["mentions"]] == ["<Subject 1>"]
        assert preview["mentions"][0]["present"] is True

    def test_a_new_member_carries_no_mentions_to_rename(self, api, project):
        result = api.save_member({"project": project, "member": {"name": "Mara"}})
        assert result["renamed_mentions"] == 0

    def test_a_delete_reports_what_it_stranded(self, api, project):
        saved = self._member(api)
        prompt = a_prompt()
        prompt["cast"] = [saved["id"]]
        self._source_with(api, project, prompt)
        result = api.delete_member({"project": project, "member": saved["id"]})
        assert len(result["used_by"]) == 1

    def test_a_members_identity_files_reach_the_render_graph(self, api, project, real_workflow):
        """The shot attaches nothing; the member's face still has to be loaded.

        The mode used to be decided before the cast was folded in, so a shot
        like this built a plain t2v graph while its composed text went on to
        say `<Picture 1>` — the words arrived and the picture never did.
        """
        api.save_profile({"name": "real", "api_format": real_workflow})
        saved = self._member(
            api, references=[{"kind": "image", "path": "mara.png", "role": ""}]
        )
        prompt = a_prompt()
        prompt["cast"] = [saved["id"]]

        result = api.prepare_render(
            {"project": project, "prompt": prompt, "profile": "real"}
        )

        assert result["mode"] == "ref2v"
        loaded = [
            node["inputs"].get("image")
            for node in result["workflow"].values()
            if node.get("class_type") == "LoadImage"
        ]
        assert "mara.png" in loaded
        assert "<Picture 1>" in result["prompt"]

    def test_a_reshoot_keeps_the_shots_references(self, api, project, real_workflow):
        """Frozen words, live media.

        A take stores its composed text and nothing else, so a re-shoot that
        rendered through the bare profile had a prompt saying `<Picture 1>` and
        no picture wired anywhere — the model duly invented a stranger in a room
        nobody had shown it. The shot travels with the re-shoot for its media.
        """
        api.save_profile({"name": "real", "api_format": real_workflow})
        saved = self._member(
            api, references=[{"kind": "image", "path": "mara.png", "role": ""}]
        )
        prompt = a_prompt()
        prompt["cast"] = [saved["id"]]

        result = api.prepare_reshoot({
            "project": project,
            "profile": "real",
            "prompt": prompt,
            "prompt_text": "<Picture 1> the take's frozen prompt",
            "width": 1344, "height": 768, "frames": 192, "seconds": 8.0,
            "seed": 4242, "quality": "final_2pass",
        })

        assert result["mode"] == "ref2v"
        loaded = [
            node["inputs"].get("image")
            for node in result["workflow"].values()
            if node.get("class_type") == "LoadImage"
        ]
        assert "mara.png" in loaded
        # The frozen text renders verbatim; the shot came along for its media
        # only, and must not have been re-composed over the top of it.
        prompts = [
            node["inputs"].get("prompt")
            for node in result["workflow"].values()
            if node.get("class_type") == "MiniMaxH3ReferenceToVideo"
        ]
        assert prompts == ["<Picture 1> the take's frozen prompt"]
        # ...and it is still a two-pass render.
        assert result["pass1_steps"] == 15

    def test_an_extend_keeps_the_shots_references(self, api, project, real_workflow):
        """A continuation is the shot's own render with Motion Context added.

        It used to be its own fixed graph on its own hardcoded weights, with no
        reference node in it at all — so continuing a cast shot dropped the
        cast, and the model invented a new face for the second half of a chain.
        """
        api.save_profile({"name": "real", "api_format": real_workflow})
        saved = self._member(
            api, references=[{"kind": "image", "path": "mara.png", "role": ""}]
        )
        prompt = a_prompt()
        prompt["cast"] = [saved["id"]]

        result = api.prepare_extend({
            "project": project, "profile": "real", "prompt": prompt,
            "source": "video/take_00007_.mp4 [output]",
            "prompt_text": "<Picture 1> she turns and walks away",
            "width": 608, "height": 352, "frames": 124, "seconds": 5.0, "seed": 5,
        })

        assert result["mode"] == "ref2v"
        workflow = result["workflow"]
        loaded = [node["inputs"].get("image") for node in workflow.values()
                  if node.get("class_type") == "LoadImage"]
        assert "mara.png" in loaded
        # ...and it is still a continuation, hanging off the reference node.
        reference = _one_of_class(workflow, "MiniMaxH3ReferenceToVideo")
        assert node_of(workflow, "MiniMaxH3MotionContext")["inputs"]["conditioning"] \
            == [reference, 0]

    def test_an_extend_renders_on_the_profiles_own_weights(
        self, api, project, real_workflow
    ):
        """Not on a checkpoint of the extend graph's own choosing.

        A continuation on different weights than the clip it continues does not
        look like it, however well the motion carries across the join.
        """
        api.save_profile({"name": "real", "api_format": real_workflow})
        unet = [node["inputs"]["unet_name"] for node in real_workflow.values()
                if node.get("class_type") == "UNETLoader"]

        result = api.prepare_extend({
            "project": project, "profile": "real",
            "source": "s.mp4", "prompt_text": "x",
            "width": 608, "height": 352, "frames": 124, "seconds": 5.0,
        })

        assert [node["inputs"]["unet_name"] for node in result["workflow"].values()
                if node.get("class_type") == "UNETLoader"] == unet

    def test_an_extend_of_a_shot_with_media_needs_the_profile(self, api, project):
        """Better a refusal than a continuation that quietly dropped the face."""
        saved = self._member(
            api, references=[{"kind": "image", "path": "mara.png", "role": ""}]
        )
        prompt = a_prompt()
        prompt["cast"] = [saved["id"]]
        payload = {
            "project": project, "prompt": prompt,
            "source": "s.mp4", "prompt_text": "x",
            "width": 608, "height": 352, "frames": 124, "seconds": 5.0,
        }
        assert status_of(api.prepare_extend, payload) == 400

    def test_a_reshoot_without_a_shot_still_renders_text_only(self, api, real_workflow):
        """The CLI has no shot to hand; a re-shoot must not start needing one."""
        api.save_profile({"name": "real", "api_format": real_workflow})
        result = api.prepare_reshoot({
            "profile": "real",
            "prompt_text": "the take's frozen prompt",
            "width": 1344, "height": 768, "frames": 192, "seconds": 8.0,
        })
        assert result["mode"] == "t2v"

    def test_a_batch_picks_the_same_mode_as_a_single_render(self, api, project, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        saved = self._member(
            api, references=[{"kind": "image", "path": "mara.png", "role": ""}]
        )
        prompt = a_prompt()
        prompt["cast"] = [saved["id"]]

        result = api.prepare_batch(
            {"project": project, "prompt": prompt, "profile": "real", "count": 2}
        )

        assert result["mode"] == "ref2v"
        assert len(result["renders"]) == 2

    def test_a_shot_with_no_cast_still_renders_as_itself(self, api, project, real_workflow):
        api.save_profile({"name": "real", "api_format": real_workflow})
        self._member(api, references=[{"kind": "image", "path": "mara.png", "role": ""}])

        result = api.prepare_render(
            {"project": project, "prompt": a_prompt(), "profile": "real"}
        )

        assert result["mode"] == "t2v"

    def test_a_keyframe_and_a_members_files_together_say_where_they_came_from(
        self, api, project
    ):
        """The Media tab shows no references here, so the message has to."""
        saved = self._member(
            api, references=[{"kind": "image", "path": "mara.png", "role": ""}]
        )
        prompt = a_prompt()
        prompt["cast"] = [saved["id"]]
        prompt["first_frame"] = "opening.png"

        preview = api.preview({"project": project, "prompt": prompt})
        conflict = next(
            finding for finding in preview["findings"] if finding["rule"] == "mode.conflict"
        )
        assert "identity file" in conflict["message"]

    def test_a_stranded_prompt_keeps_the_id_and_starts_warning(self, api, project):
        saved = self._member(api)
        prompt = a_prompt()
        prompt["beats"][0]["lines"] = [
            {"speakers": [saved["id"]], "text": "We are late."}
        ]
        self._source_with(api, project, prompt)
        api.delete_member({"project": project, "member": saved["id"]})
        preview = api.preview({"project": project, "prompt": prompt})
        rules = {finding["rule"] for finding in preview["findings"]}
        assert "cast.unknown" in rules


class TestAuditFixes:
    def test_adopt_output_image_rejects_path_traversal(self, api, tmp_path, monkeypatch):
        from video_prompter import paths
        out_dir = tmp_path / "output"
        in_dir = tmp_path / "input"
        out_dir.mkdir()
        in_dir.mkdir()
        (tmp_path / "secret.png").write_bytes(b"secret")
        monkeypatch.setattr(paths, "comfy_output_dir", lambda: out_dir)
        monkeypatch.setattr(paths, "comfy_input_dir", lambda: in_dir)

        assert status_of(api.adopt_output_image, {"filename": "../secret.png"}) == 400
        assert status_of(api.adopt_output_image, {"filename": "secret.png", "subfolder": ".."}) == 400

    def test_export_rejects_traversal_in_destination(self, api, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        (output / "a.mp4").write_bytes(b"video")
        api.create_project({"name": "Cut"})
        api.add_source({"project": "Cut", "name": "A", "duration": 8})
        source_id = sources_of(api.load_project("Cut"))[0]["id"]
        api.record_takes({
            "project": "Cut",
            "takes": [{"prompt_id": "r1", "filename": "a.mp4", "shot": source_id, "seconds": 8.0}],
        })
        take_id = api.list_takes("Cut")["takes"][0]["id"]
        api.pick_take({"project": "Cut", "source": source_id, "take": take_id})

        payload = {
            "project": "Cut",
            "format": "edl",
            "output_dir": str(output),
            "destination": "../escape.edl",
        }
        assert status_of(api.export, payload) == 400

    def test_prepare_refine_translates_invalid_parameters_to_400(self, api):
        bad_mode = {
            "source": "a.mp4",
            "width": 1280,
            "height": 720,
            "enhancement_mode": "nonexistent_mode",
        }
        assert status_of(api.prepare_refine, bad_mode) == 400

        bad_dims = {
            "source": "a.mp4",
            "width": -64,
            "height": 720,
            "enhancement_mode": "redetail",
        }
        assert status_of(api.prepare_refine, bad_dims) == 400

    def test_parse_prompt_endpoint_recovers_structured_and_plain_prompts(self, api):
        composed = (
            "integrated_multimodal_description: Live-action drama. An office hallway. "
            "[Shot 1] Wide shot, a man walks down the corridor. "
            "[Shot 2] At 00:04.000, the camera cuts to Close-up, he stops at a door.\n\n"
            "overall_soundscape: Quiet fluorescent hum.\n\n"
            "non_diegetic_music: Low tense strings."
        )
        res = api.parse_prompt({"text": composed, "duration": 8})
        assert res["structured"] is True
        assert len(res["prompt"]["beats"]) == 2
        assert res["prompt"]["audio"] == "Quiet fluorescent hum."
        assert res["prompt"]["music"] == "Low tense strings."

        plain = api.parse_prompt({"text": "She turns toward the window.", "duration": 5})
        assert plain["structured"] is False
        assert plain["prompt"]["overview"] == "She turns toward the window."

    def test_corrupt_sidecars_are_quarantined_before_next_write(self, api):
        api.create_project({"name": "CorruptTest"})
        proj_dir = api.projects.path_for("CorruptTest")

        for filename in ("takes.json", "stills.json", "image_prompts.json"):
            target = proj_dir / filename
            target.write_text("{broken json", encoding="utf-8")

        assert api.list_takes("CorruptTest")["takes"] == []
        assert api.list_stills("CorruptTest")["stills"] == []
        assert api.image_prompt_revisions("CorruptTest")["revisions"] == []

        quarantined = [p.name for p in proj_dir.glob("*.corrupt-*")]
        assert len(quarantined) == 3

