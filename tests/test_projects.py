from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from video_prompter import (
    Beat, ProjectError, ProjectStore, Segment, Source, Storyboard, VideoPrompt,
)
from video_prompter.projects import (
    PROJECT_JSON,
    SAVE_BACKUP_LIMIT,
    sanitize_project_name,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "projects")


def source(name: str = "Opening", beats: int = 2) -> Source:
    storyboard = Storyboard(
        10, [Beat(i * 5, (i + 1) * 5, "wide", f"beat {i}") for i in range(beats)]
    )
    return Source(
        name=name,
        segments=[Segment(prompt=VideoPrompt(storyboard=storyboard, style="A look."))],
    )


class TestNames:
    def test_traversal_is_stripped(self):
        assert sanitize_project_name("../../etc") == "etc"

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ProjectError):
            sanitize_project_name("  ")


class TestLifecycle:
    def test_create_then_load(self, store):
        store.create("Rooftop", [source()])
        loaded = store.load("Rooftop")
        assert loaded.name == "Rooftop"
        assert [s.name for s in loaded.sources] == ["Opening"]

    def test_creating_puts_one_clip_on_the_timeline_per_source(self, store):
        store.create("Rooftop", [source("One"), source("Two")])
        loaded = store.load("Rooftop")
        assert [c.source_id for c in loaded.clips] == [s.id for s in loaded.sources]

    def test_sources_survive_the_round_trip(self, store):
        store.create("Rooftop", [source()])
        restored = store.load("Rooftop").sources[0].segments[0].prompt
        assert restored.style == "A look."
        assert len(restored.storyboard.beats) == 2

    def test_listing_finds_saved_projects_only(self, store, tmp_path):
        store.create("Alpha", [])
        store.create("beta", [])
        (store.root / "not-a-project").mkdir()
        assert store.list() == ["Alpha", "beta"]

    def test_exists_is_false_before_and_true_after(self, store):
        assert store.exists("Ghost") is False
        store.create("Ghost", [])
        assert store.exists("Ghost") is True

    def test_loading_a_missing_project_raises(self, store):
        with pytest.raises(ProjectError):
            store.load("Nope")

    def test_delete_reports_whether_anything_went(self, store):
        store.create("Temp", [])
        assert store.delete("Temp") is True
        assert store.delete("Temp") is False

    def test_modified_is_stamped_on_save(self, store):
        project = store.create("Stamped", [])
        project.modified = ""
        store.save(project)
        assert store.load("Stamped").modified

    def test_find_a_source_by_id(self, store):
        store.create("Multi", [source("One"), source("Two")])
        project = store.load("Multi")
        wanted = project.sources[1].id
        assert project.source(wanted).name == "Two"
        assert project.source("src_nope") is None

    def test_find_a_source_by_name(self, store):
        store.create("Multi", [source("One"), source("Two")])
        project = store.load("Multi")
        assert project.source_named("Two") is not None
        assert project.source_named("Missing") is None

    def test_ids_survive_the_round_trip(self, store):
        created = store.create("Stable", [source("One")])
        assert store.load("Stable").sources[0].id == created.sources[0].id

    def test_deleting_a_source_takes_its_clips_with_it(self, store):
        # An orphaned clip would put a hole in the edit that points at nothing.
        project = store.create("Ripple", [source("One"), source("Two")])
        assert project.remove_source(project.sources[0].id) is True
        assert len(project.sources) == 1
        assert [c.source_id for c in project.clips] == [project.sources[0].id]

    def test_deleting_a_source_that_is_not_there_says_so(self, store):
        project = store.create("Ripple", [source("One")])
        assert project.remove_source("src_nope") is False


class TestAtomicWrite:
    def test_no_temp_files_are_left_behind(self, store):
        store.create("Clean", [source()])
        leftovers = list(store.path_for("Clean").glob("project.*.json"))
        assert leftovers == []

    def test_the_file_on_disk_is_valid_json(self, store):
        store.create("Valid", [source()])
        with (store.path_for("Valid") / PROJECT_JSON).open(encoding="utf-8") as handle:
            assert json.load(handle)["name"] == "Valid"


class TestUpdate:
    def test_mutations_are_persisted(self, store):
        store.create("Edit", [source("One")])
        store.update("Edit", lambda p: p.sources.append(source("Two")))
        assert [s.name for s in store.load("Edit").sources] == ["One", "Two"]

    def test_concurrent_appends_do_not_lose_entries(self, store):
        store.create("Race", [])

        def add(index: int) -> None:
            store.update("Race", lambda p: p.sources.append(source(f"source {index}")))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(add, range(24)))

        names = {s.name for s in store.load("Race").sources}
        assert len(names) == 24

    def test_the_same_lock_is_handed_out_per_project(self, store):
        assert store.lock_for("Race") is store.lock_for("Race")
        assert store.lock_for("Race") is not store.lock_for("Other")


class TestBackups:
    def test_the_first_save_has_nothing_to_back_up(self, store):
        store.create("Fresh", [])
        assert store.list_backups("Fresh") == []

    def test_each_later_save_snapshots_the_previous_state(self, store):
        project = store.create("Kept", [source()])
        for _ in range(3):
            store.save(project)
        assert len([b for b in store.list_backups("Kept") if b.tier == "save"]) == 3

    def test_the_per_save_ring_is_capped(self, store):
        project = store.create("Ring", [source()])
        for _ in range(SAVE_BACKUP_LIMIT + 8):
            store.save(project)
        saves = [b for b in store.list_backups("Ring") if b.tier == "save"]
        assert len(saves) == SAVE_BACKUP_LIMIT

    def test_hourly_and_daily_hold_one_snapshot_each_per_period(self, store):
        project = store.create("Tiers", [source()])
        for _ in range(5):
            store.save(project)
        backups = store.list_backups("Tiers")
        assert len([b for b in backups if b.tier == "hourly"]) == 1
        assert len([b for b in backups if b.tier == "daily"]) == 1

    def test_a_save_that_drops_most_sources_is_red_flagged(self, store):
        project = store.create("Shrink", [source(f"c{i}") for i in range(6)])
        project.sources = project.sources[:1]
        store.save(project)
        assert any(b.tier == "redflag" for b in store.list_backups("Shrink"))

    def test_an_ordinary_edit_is_not_red_flagged(self, store):
        project = store.create("Steady", [source(f"c{i}") for i in range(6)])
        project.sources.append(source("one more"))
        store.save(project)
        assert not any(b.tier == "redflag" for b in store.list_backups("Steady"))

    def test_backups_are_listed_newest_first(self, store):
        project = store.create("Ordered", [source()])
        for _ in range(3):
            store.save(project)
        stamps = [b.modified for b in store.list_backups("Ordered")]
        assert stamps == sorted(stamps, reverse=True)


class TestRestore:
    def test_a_snapshot_brings_the_old_sources_back(self, store):
        project = store.create("Undo", [source("One"), source("Two")])
        store.save(project)  # snapshot with both sources

        project.sources = []
        project.timeline.clips = []
        store.save(project)
        assert store.load("Undo").sources == []

        snapshot = next(b for b in store.list_backups("Undo") if b.tier == "save")
        restored = store.restore("Undo", snapshot.path)
        assert len(restored.sources) == 2
        assert len(store.load("Undo").sources) == 2

    def test_restoring_snapshots_the_pre_restore_state_first(self, store):
        project = store.create("Safety", [source()])
        store.save(project)
        before = len(store.list_backups("Safety"))

        snapshot = next(b for b in store.list_backups("Safety") if b.tier == "save")
        store.restore("Safety", snapshot.path)
        assert len(store.list_backups("Safety")) > before

    def test_restoring_from_outside_the_backups_directory_is_refused(self, store, tmp_path):
        store.create("Guard", [source()])
        intruder = tmp_path / "elsewhere.json"
        intruder.write_text("{}", encoding="utf-8")
        with pytest.raises(ProjectError):
            store.restore("Guard", intruder)

    def test_restoring_a_missing_snapshot_raises(self, store):
        store.create("Guard", [source()])
        missing = store.path_for("Guard") / ".backups" / "project_nope.json"
        with pytest.raises(ProjectError):
            store.restore("Guard", missing)


class TestShelves:
    def test_a_new_project_is_active(self, store):
        store.create("Fresh", [])
        assert store.shelf_of("Fresh") == "active"

    def test_archiving_takes_it_off_the_active_list(self, store):
        store.create("Done", [source()])
        store.shelve("Done", "archived")
        assert store.list() == []
        assert store.list("archived") == ["Done"]

    def test_an_archived_project_still_loads_by_name(self, store):
        store.create("Done", [source()])
        store.shelve("Done", "archived")
        assert store.load("Done").sources[0].name == "Opening"

    def test_sidecars_follow_a_project_across_shelves(self, store):
        store.create("Moved", [source()])
        (store.path_for("Moved") / "takes.json").write_text("[]", encoding="utf-8")
        store.shelve("Moved", "trashed")
        assert (store.path_for("Moved") / "takes.json").is_file()

    def test_restoring_is_shelving_back_to_active(self, store):
        store.create("Oops", [source()])
        store.shelve("Oops", "trashed")
        store.shelve("Oops", "active")
        assert store.list() == ["Oops"]

    def test_a_name_already_on_the_shelf_is_suffixed_not_overwritten(self, store):
        store.create("Twice", [source("First")])
        store.shelve("Twice", "trashed")
        store.create("Twice", [source("Second")])
        landed = store.shelve("Twice", "trashed")
        assert landed == "Twice - 2"
        assert sorted(store.list("trashed")) == ["Twice", "Twice - 2"]

    def test_a_suffixed_name_survives_sanitising(self, store):
        store.create("Twice", [])
        store.shelve("Twice", "trashed")
        store.create("Twice", [])
        landed = store.shelve("Twice", "trashed")
        assert sanitize_project_name(landed) == landed

    def test_a_suffixed_move_rewrites_the_name_inside_the_file(self, store):
        store.create("Twice", [])
        store.shelve("Twice", "trashed")
        store.create("Twice", [])
        landed = store.shelve("Twice", "trashed")
        assert store.load(landed).name == landed

    def test_shelving_a_missing_project_raises(self, store):
        with pytest.raises(ProjectError):
            store.shelve("Ghost", "trashed")

    def test_an_unknown_shelf_is_refused(self, store):
        store.create("Real", [])
        with pytest.raises(ProjectError):
            store.shelve("Real", "somewhere")


class TestPurge:
    def test_purging_only_works_on_the_trash(self, store):
        store.create("Live", [source()])
        assert store.purge("Live") is False
        assert store.exists("Live")

    def test_purging_a_trashed_project_erases_it(self, store):
        store.create("Gone", [source()])
        store.shelve("Gone", "trashed")
        assert store.purge("Gone") is True
        assert store.shelf_of("Gone") is None

    def test_emptying_the_trash_reports_what_went(self, store):
        store.create("A", [])
        store.create("B", [])
        store.create("Kept", [])
        store.shelve("A", "trashed")
        store.shelve("B", "trashed")
        assert sorted(store.empty_trash()) == ["A", "B"]
        assert store.list() == ["Kept"]

    def test_emptying_an_empty_trash_is_fine(self, store):
        assert store.empty_trash() == []


class TestDuplicate:
    def test_the_copy_is_named_for_its_place_in_the_family(self, store):
        store.create("PLANET OFFICE", [source()])
        assert store.duplicate("PLANET OFFICE").name == "PLANET OFFICE v2"

    def test_duplicating_a_copy_keeps_counting(self, store):
        store.create("Reel", [source()])
        store.duplicate("Reel")
        assert store.duplicate("Reel v2").name == "Reel v3"

    def test_the_family_shares_one_lineage(self, store):
        store.create("Reel", [source()])
        first = store.duplicate("Reel")
        second = store.duplicate("Reel")
        assert first.lineage == second.lineage == store.load("Reel").lineage

    def test_the_copy_records_where_it_came_from_and_when(self, store):
        original = store.create("Reel", [source()])
        copy = store.duplicate("Reel")
        assert copy.derived_from == "Reel"
        assert copy.created >= original.created

    def test_the_content_comes_along(self, store):
        store.create("Reel", [source("Opening")])
        copy = store.load(store.duplicate("Reel").name)
        assert [s.name for s in copy.sources] == ["Opening"]
        assert len(copy.clips) == 1

    def test_takes_and_stills_come_along_but_backups_do_not(self, store):
        project = store.create("Reel", [source()])
        store.save(project)
        (store.path_for("Reel") / "takes.json").write_text("[]", encoding="utf-8")
        copy = store.duplicate("Reel")
        assert (store.path_for(copy.name) / "takes.json").is_file()
        assert store.list_backups(copy.name) == []

    def test_an_explicit_name_is_used(self, store):
        store.create("Reel", [source()])
        assert store.duplicate("Reel", "Director cut").name == "Director cut"

    def test_an_explicit_name_that_is_taken_is_refused(self, store):
        store.create("Reel", [source()])
        store.create("Taken", [])
        with pytest.raises(ProjectError):
            store.duplicate("Reel", "Taken")

    def test_duplicating_a_missing_project_raises(self, store):
        with pytest.raises(ProjectError):
            store.duplicate("Ghost")


class TestEntries:
    def test_every_shelf_is_reported(self, store):
        store.create("Live", [])
        store.create("Old", [])
        store.create("Bin", [])
        store.shelve("Old", "archived")
        store.shelve("Bin", "trashed")
        shelves = {e.name: e.shelf for e in store.entries()}
        assert shelves == {"Live": "active", "Old": "archived", "Bin": "trashed"}

    def test_an_entry_carries_what_a_row_shows(self, store):
        store.create("Reel", [source(), source("Two")])
        entry = next(e for e in store.entries() if e.name == "Reel")
        assert entry.sources == 2
        assert entry.lineage and entry.created and entry.modified

    def test_a_rewritten_project_is_not_served_from_the_cache(self, store):
        project = store.create("Reel", [source()])
        assert next(e for e in store.entries() if e.name == "Reel").sources == 1
        project.sources.append(source("Two"))
        store.save(project)
        assert next(e for e in store.entries() if e.name == "Reel").sources == 2

    def test_an_unparseable_project_still_gets_a_row(self, store):
        store.create("Broken", [])
        (store.path_for("Broken") / PROJECT_JSON).write_text("{oops", encoding="utf-8")
        assert [e.name for e in store.entries()] == ["Broken"]
