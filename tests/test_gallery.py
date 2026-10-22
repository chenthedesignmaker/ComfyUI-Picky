from __future__ import annotations

import json

import pytest

from video_prompter import ProjectStore
from video_prompter import gallery as gallery_module
from video_prompter.gallery import (
    EXTRA_DATA_KEY,
    Gallery,
    Take,
    project_of,
    takes_from_history,
)


@pytest.fixture
def gallery(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    store.create("Rooftop", [])
    return Gallery(store)


def history_entry(project="Rooftop", shot="Escape", filename="MiniMax_H3_00030_.mp4", **meta):
    """A ComfyUI history entry the way the server actually stores one."""
    return {
        "prompt": [
            1,
            "abc-123",
            {"92": {"class_type": "SaveVideo", "inputs": {}}},
            {
                EXTRA_DATA_KEY: {
                    "project": project,
                    "shot": shot,
                    "frames": 362,
                    "seconds": 15.083,
                    **meta,
                }
            },
            ["92"],
        ],
        "outputs": {
            # SaveVideo reports through PreviewVideo, so a video comes back
            # under "images" with animated set.
            "92": {
                "images": [{"filename": filename, "subfolder": "video", "type": "output"}],
                "animated": [True],
            }
        },
        "status": {"completed": True},
    }


class TestHistoryParsing:
    def test_face_refine_report_survives_history_and_storage(self):
        entry = history_entry()
        entry["outputs"]["face"] = {"face_refine_report": ["Processed 12 face tracks."]}
        take = takes_from_history("abc-123", entry)[0]
        assert take.face_refine_report == "Processed 12 face tracks."
        assert Take.from_dict(take.to_dict()).face_refine_report == take.face_refine_report

    @pytest.mark.parametrize("input_first", [True, False])
    def test_extend_load_video_preview_is_not_a_second_take(self, input_first):
        entry = history_entry(filename="extended.mp4", seed=123, parent_take="parent:parent.mp4")
        saved = entry["outputs"]
        preview = {"vp_extend_src": {"images": [
            {"filename": "parent.mp4", "subfolder": "video", "type": "input"},
        ], "animated": [True]}}
        entry["outputs"] = {**preview, **saved} if input_first else {**saved, **preview}
        takes = takes_from_history("extend", entry)
        assert len(takes) == 1
        assert (takes[0].filename, takes[0].seed, takes[0].parent_take) == (
            "extended.mp4", 123, "parent:parent.mp4",
        )

    def test_temp_render_and_multiple_real_outputs_are_preserved(self):
        entry = history_entry(filename="one.mp4", seed=123)
        entry["outputs"]["preview"] = {"videos": [{"filename": "two.mp4", "type": "temp"}]}
        assert [t.filename for t in takes_from_history("extend", entry)] == ["one.mp4", "two.mp4"]

    def test_a_video_under_the_images_key_is_found(self):
        takes = takes_from_history("abc-123", history_entry())
        assert len(takes) == 1
        assert takes[0].filename == "MiniMax_H3_00030_.mp4"
        assert takes[0].is_video

    def test_our_metadata_rides_through(self):
        take = takes_from_history("abc-123", history_entry())[0]
        assert (take.shot, take.frames, take.seconds) == ("Escape", 362, 15.083)

    def test_the_canvas_rides_through(self):
        # A take freezes its canvas so a fine re-shoot can scale the ratio up.
        take = takes_from_history("abc-123", history_entry(width=1344, height=768))[0]
        assert (take.width, take.height) == (1344, 768)

    def test_the_saved_latent_path_rides_through(self):
        # recorded so an Extend of this take can pin from its own AV latent
        take = takes_from_history(
            "abc-123", history_entry(latent_path="h3_context/take7_00001.safetensors")
        )[0]
        assert take.latent_path == "h3_context/take7_00001.safetensors"

    def test_a_take_without_a_saved_latent_has_none(self):
        assert takes_from_history("abc-123", history_entry())[0].latent_path == ""

    def test_steps_and_generation_time_ride_through(self):
        take = takes_from_history(
            "abc-123", history_entry(steps=30, generation_time=42.5)
        )[0]
        assert take.steps == 30
        assert take.generation_time == 42.5

    def test_steps_fallback_from_prompt_nodes_if_not_in_meta(self):
        entry = history_entry()
        # Add a sampler node with steps
        entry["prompt"][2]["50"] = {
            "class_type": "KSampler",
            "inputs": {"steps": 25},
        }
        take = takes_from_history("abc-123", entry)[0]
        assert take.steps == 25

    def test_the_tier_name_rides_through(self):
        # The numbers cannot always be read backwards into a tier: a `preview`
        # continuation renders at its parent's canvas, so a Quick extend of a
        # 720p clip is 1.0 MP at 8 steps — turbo_hd's signature exactly. The
        # name is what settles it.
        take = takes_from_history("abc-123", history_entry(quality="preview"))[0]
        assert take.quality == "preview"
        assert take.to_dict()["quality"] == "preview"

    def test_a_take_rendered_before_the_tier_was_recorded_has_none(self):
        # ...and the editor falls back to reading the numbers for those.
        assert takes_from_history("abc-123", history_entry())[0].quality == ""

    def test_the_split_of_a_two_pass_render_rides_through(self):
        take = takes_from_history("abc-123", history_entry(steps=8, pass1_steps=4))[0]
        assert (take.steps, take.pass1_steps) == (8, 4)

    def test_a_single_pass_render_records_no_split(self):
        # The absence is the signal: it is what says the whole schedule ran at
        # one canvas, which turbo_hd and turbo_2pass cannot otherwise be told
        # apart by.
        assert takes_from_history("abc-123", history_entry(steps=8))[0].pass1_steps is None

    def test_an_older_two_pass_take_is_read_off_its_split_sigmas(self):
        # Renders made before the metadata existed are still legible: the
        # transform leaves the node that cut the schedule in the graph.
        entry = history_entry(steps=20)
        entry["prompt"][2]["vp_split"] = {
            "class_type": "SplitSigmas",
            "inputs": {"sigmas": ["9", 0], "step": 15},
        }
        assert takes_from_history("abc-123", entry)[0].pass1_steps == 15

    def test_the_project_is_readable_from_the_queue_tuple(self):
        assert project_of(history_entry(project="Rooftop")) == "Rooftop"

    def test_someone_elses_run_has_no_project(self):
        assert project_of({"prompt": [1, "x", {}, {}, []], "outputs": {}}) == ""

    def test_a_still_saved_beside_the_video_does_not_bury_it(self):
        entry = history_entry()
        entry["outputs"]["93"] = {
            "images": [{"filename": "preview_00001_.png", "subfolder": "", "type": "output"}]
        }
        takes = takes_from_history("abc-123", entry)
        assert [take.filename for take in takes] == ["MiniMax_H3_00030_.mp4"]

    def test_a_run_with_only_stills_still_yields_them(self):
        entry = history_entry(filename="still_00001_.png")
        assert takes_from_history("abc-123", entry)[0].filename == "still_00001_.png"

    @pytest.mark.parametrize(
        "entry",
        [{}, {"outputs": None}, {"outputs": {"1": None}}, {"prompt": "nonsense", "outputs": {}}],
    )
    def test_malformed_entries_yield_nothing_rather_than_raising(self, entry):
        assert takes_from_history("x", entry) == []

    def test_an_output_without_a_filename_is_skipped(self):
        entry = history_entry()
        entry["outputs"]["92"]["images"] = [{"subfolder": "video", "type": "output"}]
        assert takes_from_history("abc-123", entry) == []

    def test_extra_data_passed_directly_also_works(self):
        entry = {
            "extra_data": {EXTRA_DATA_KEY: {"project": "Rooftop", "shot": "Escape"}},
            "outputs": {"92": {"images": [{"filename": "a.mp4"}]}},
        }
        assert takes_from_history("x", entry)[0].shot == "Escape"


class TestTake:
    def test_the_view_url_points_at_comfys_own_endpoint(self):
        take = Take(prompt_id="x", filename="a b.mp4", subfolder="video")
        url = take.view_url()
        assert url.startswith("/view?")
        assert "filename=a+b.mp4" in url
        assert "type=output" in url

    def test_video_extensions_are_recognised(self):
        assert Take(prompt_id="x", filename="a.mp4").is_video
        assert not Take(prompt_id="x", filename="a.png").is_video

    def test_it_round_trips(self):
        take = Take(
            prompt_id="x", filename="a.mp4", shot="One", frames=192, seconds=8.0,
            seed=4, width=1344, height=768,
        )
        assert Take.from_dict(take.to_dict()).to_dict() == take.to_dict()

    def test_a_fresh_take_has_not_been_seen(self):
        assert Take(prompt_id="x", filename="a.mp4").seen is False

    def test_a_take_described_without_the_flag_is_unseen(self):
        """What a renderer posts when it records a job it just finished."""
        assert Take.from_dict({"prompt_id": "x", "filename": "a.mp4"}).seen is False


class TestEmbeddedPromptRecovery:
    def test_the_h3_node_prompt_is_read_from_video_metadata(self, tmp_path, monkeypatch):
        video = tmp_path / "take.mp4"
        workflow = {
            "1": {"class_type": "Other", "inputs": {"prompt": "fallback"}},
            "5": {
                "class_type": "MiniMaxH3ImageToVideo",
                "inputs": {"prompt": "integrated_multimodal_description: the exact prompt"},
            },
        }

        def atom(kind, payload):
            return (len(payload) + 8).to_bytes(4, "big") + kind + payload

        workflow_text = json.dumps(workflow).encode()
        key = (8 + len(b"prompt")).to_bytes(4, "big") + b"mdta" + b"prompt"
        keys = atom(b"keys", b"\0\0\0\0" + (1).to_bytes(4, "big") + key)
        data = atom(b"data", b"\0\0\0\1\0\0\0\0" + workflow_text)
        ilst = atom(b"ilst", atom((1).to_bytes(4, "big"), data))
        meta = atom(b"meta", b"\0\0\0\0" + keys + ilst)
        video.write_bytes(atom(b"ftyp", b"isom") + atom(b"moov", atom(b"udta", meta)))

        assert gallery_module.embedded_workflow_prompt(video).endswith("the exact prompt")

    def test_a_short_label_is_replaced_for_the_editor(self, tmp_path, monkeypatch):
        store = ProjectStore(tmp_path / "projects")
        store.create("Rooftop", [])
        output = tmp_path / "output"
        (output / "video").mkdir(parents=True)
        (output / "video" / "take.mp4").write_bytes(b"video")
        gallery = Gallery(store, output_dir=output)
        gallery.record("Rooftop", [
            Take(prompt_id="x", filename="take.mp4", subfolder="video", prompt_text="Herd A")
        ])
        exact = "integrated_multimodal_description: the complete embedded prompt"
        monkeypatch.setattr(gallery_module, "embedded_workflow_prompt", lambda path: exact)

        take = gallery.list("Rooftop")[0]
        assert take.prompt_text == exact
        assert take.prompt_source == "video_metadata"

    def test_a_short_extension_label_is_recovered_from_metadata(self, tmp_path, monkeypatch):
        """Extends rendered natively can carry only a label; recovery applies."""
        store = ProjectStore(tmp_path / "projects")
        store.create("Rooftop", [])
        output = tmp_path / "output"
        (output / "video").mkdir(parents=True)
        (output / "video" / "take.mp4").write_bytes(b"video")
        gallery = Gallery(store, output_dir=output)
        gallery.record("Rooftop", [
            Take(
                prompt_id="x", filename="take.mp4", subfolder="video",
                prompt_text="Romance B", parent_take="parent:take.mp4",
            )
        ])
        exact = "integrated_multimodal_description: the real extend prompt"
        monkeypatch.setattr(gallery_module, "embedded_workflow_prompt", lambda path: exact)

        take = gallery.list("Rooftop")[0]
        assert take.prompt_text == exact
        assert take.prompt_source == "video_metadata"

    def test_a_structured_extension_prompt_is_not_replaced(self, tmp_path, monkeypatch):
        """Extends with a real structured prompt are left alone."""
        store = ProjectStore(tmp_path / "projects")
        store.create("Rooftop", [])
        output = tmp_path / "output"
        (output / "video").mkdir(parents=True)
        (output / "video" / "take.mp4").write_bytes(b"video")
        gallery = Gallery(store, output_dir=output)
        gallery.record("Rooftop", [
            Take(
                prompt_id="x", filename="take.mp4", subfolder="video",
                prompt_text="integrated_multimodal_description: already correct",
                parent_take="parent:take.mp4",
            )
        ])
        monkeypatch.setattr(
            gallery_module,
            "embedded_workflow_prompt",
            lambda path: pytest.fail("structured prompts should not be probed"),
        )

        take = gallery.list("Rooftop")[0]
        assert take.prompt_text == "integrated_multimodal_description: already correct"
        assert take.prompt_source == "recorded"


class TestGallery:
    def test_input_previews_are_not_recorded_as_new_takes(self, gallery):
        actual = Take(prompt_id="extend", filename="new.mp4", seed=42)
        preview = Take(prompt_id="extend", filename="parent.mp4", seed=42, type="input")
        assert gallery.record("Rooftop", [actual, preview]) == [actual]
        assert gallery.record("Rooftop", [actual, preview]) == []
        assert [t.filename for t in gallery.list("Rooftop")] == ["new.mp4"]

    def test_old_input_preview_records_are_hidden_without_rewriting_the_file(self, gallery):
        gallery._write("Rooftop", [
            Take(prompt_id="extend", filename="new.mp4", seed=42),
            Take(prompt_id="extend", filename="parent.mp4", seed=42, type="input"),
        ])
        before = gallery.path_for("Rooftop").read_bytes()
        assert [t.filename for t in gallery.list("Rooftop")] == ["new.mp4"]
        assert gallery.path_for("Rooftop").read_bytes() == before

    def test_a_new_project_has_no_takes(self, gallery):
        assert gallery.list("Rooftop") == []

    def test_recording_then_listing(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        assert len(gallery.list("Rooftop")) == 1

    def test_recording_the_same_run_twice_adds_nothing(self, gallery):
        takes = takes_from_history("abc-123", history_entry())
        assert len(gallery.record("Rooftop", takes)) == 1
        assert len(gallery.record("Rooftop", takes)) == 0
        assert len(gallery.list("Rooftop")) == 1

    def test_different_runs_of_the_same_shot_both_land(self, gallery):
        gallery.record("Rooftop", takes_from_history("run-1", history_entry(filename="a.mp4")))
        gallery.record("Rooftop", takes_from_history("run-2", history_entry(filename="b.mp4")))
        assert len(gallery.list("Rooftop")) == 2

    def test_listing_can_be_filtered_by_shot(self, gallery):
        gallery.record("Rooftop", takes_from_history("r1", history_entry(shot="Escape")))
        gallery.record("Rooftop", takes_from_history("r2", history_entry(shot="Landing")))
        assert len(gallery.list("Rooftop", shot="Escape")) == 1

    def test_newest_comes_first(self, gallery):
        gallery.record("Rooftop", [Take(prompt_id="a", filename="a.mp4", created="2026-01-01T00:00:00+00:00")])
        gallery.record("Rooftop", [Take(prompt_id="b", filename="b.mp4", created="2026-08-01T00:00:00+00:00")])
        assert [take.prompt_id for take in gallery.list("Rooftop")] == ["b", "a"]

    def test_a_recorded_take_lands_unseen_and_survives_a_reload(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        assert gallery.list("Rooftop")[0].seen is False

    def test_history_stored_before_the_flag_existed_reads_as_seen(self, gallery):
        """Otherwise opening an old project would badge every take it ever made."""
        gallery.path_for("Rooftop").parent.mkdir(parents=True, exist_ok=True)
        gallery.path_for("Rooftop").write_text(
            json.dumps({"takes": [{"prompt_id": "old", "filename": "a.mp4"}]}),
            encoding="utf-8",
        )
        assert gallery.list("Rooftop")[0].seen is True

    def test_marking_seen_sticks(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        take_id = gallery.list("Rooftop")[0].id
        assert gallery.mark_seen("Rooftop", [take_id]) == [take_id]
        assert gallery.list("Rooftop")[0].seen is True

    def test_marking_seen_twice_reports_nothing_the_second_time(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        take_id = gallery.list("Rooftop")[0].id
        gallery.mark_seen("Rooftop", [take_id])
        assert gallery.mark_seen("Rooftop", [take_id]) == []

    def test_marking_seen_ignores_ids_that_name_nothing(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        assert gallery.mark_seen("Rooftop", ["gone:x.mp4"]) == []

    def test_a_batch_is_marked_in_one_sweep(self, gallery):
        gallery.record("Rooftop", takes_from_history("r1", history_entry(filename="a.mp4")))
        gallery.record("Rooftop", takes_from_history("r2", history_entry(filename="b.mp4")))
        ids = [take.id for take in gallery.list("Rooftop")]
        assert sorted(gallery.mark_seen("Rooftop", ids)) == sorted(ids)
        assert all(take.seen for take in gallery.list("Rooftop"))

    def test_seen_can_be_put_back(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        take_id = gallery.list("Rooftop")[0].id
        gallery.mark_seen("Rooftop", [take_id])
        gallery.mark_seen("Rooftop", [take_id], seen=False)
        assert gallery.list("Rooftop")[0].seen is False

    def test_annotating_a_take_seen_sticks(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        take_id = gallery.list("Rooftop")[0].id
        assert gallery.annotate("Rooftop", take_id, seen=True).seen is True

    def test_rating_a_take_leaves_its_seen_flag_alone(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        take_id = gallery.list("Rooftop")[0].id
        assert gallery.annotate("Rooftop", take_id, rating=3).seen is False

    def test_forget_drops_the_record(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        assert gallery.forget("Rooftop", "abc-123") == 1
        assert gallery.list("Rooftop") == []

    def test_forgetting_something_absent_reports_zero(self, gallery):
        assert gallery.forget("Rooftop", "nope") == 0

    def test_takes_live_beside_the_project_not_inside_it(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        project_json = gallery.projects.path_for("Rooftop") / "project.json"
        assert "MiniMax_H3" not in project_json.read_text(encoding="utf-8")
        assert gallery.path_for("Rooftop").is_file()

    def test_a_corrupt_takes_file_does_not_break_the_project(self, gallery):
        gallery.path_for("Rooftop").write_text("{ not json", encoding="utf-8")
        assert gallery.list("Rooftop") == []

    def test_recording_nothing_is_a_no_op(self, gallery):
        assert gallery.record("Rooftop", []) == []
        assert not gallery.path_for("Rooftop").exists()

    def test_the_file_stays_valid_json(self, gallery):
        gallery.record("Rooftop", takes_from_history("abc-123", history_entry()))
        json.loads(gallery.path_for("Rooftop").read_text(encoding="utf-8"))

    def test_parent_take_is_parsed_and_serialized(self, gallery):
        entry = history_entry(parent_take="parent-prompt-1:take1.mp4")
        takes = takes_from_history("child-prompt", entry)
        assert len(takes) == 1
        assert takes[0].parent_take == "parent-prompt-1:take1.mp4"
        d = takes[0].to_dict()
        assert d["parent_take"] == "parent-prompt-1:take1.mp4"
        restored = Take.from_dict(d)
        assert restored.parent_take == "parent-prompt-1:take1.mp4"

    def test_historical_extend_workflow_recovers_parent_take(self, gallery):
        # Older extend without explicit parent_take in extra_data, but with LoadVideo node
        entry = history_entry(filename="child_take.mp4")
        # Add LoadVideo node in prompt workflow
        entry["prompt"][2]["src_load"] = {
            "class_type": "LoadVideo",
            "inputs": {"file": "video/parent_take.mp4 [output]"},
        }
        # Record parent take and child take
        parent = Take(prompt_id="parent_prompt", filename="parent_take.mp4", shot="ShotA")
        gallery.record("Rooftop", [parent])
        child_takes = takes_from_history("child_prompt", entry)
        gallery.record("Rooftop", child_takes)

        listed = gallery.list("Rooftop")
        child = next(t for t in listed if t.filename == "child_take.mp4")
        assert child.parent_take == "parent_prompt:parent_take.mp4"

    def test_a_parent_id_with_no_prompt_id_is_resolved_by_filename(self, gallery):
        # Some renders wrote down which file they were continuing but not the
        # job that made it. Left alone, the continuation reads as a root and the
        # whole chain is invisible.
        gallery.record("Rooftop", [
            Take(prompt_id="p0", filename="parent.mp4", shot="ShotA"),
            Take(prompt_id="p1", filename="child.mp4", shot="ShotA",
                 parent_take=":parent.mp4"),
        ])
        child = next(t for t in gallery.list("Rooftop") if t.filename == "child.mp4")
        assert child.parent_take == "p0:parent.mp4"

    def test_a_duplicated_filename_is_left_unresolved_rather_than_picked(self, gallery):
        # Two candidates and no way to tell them apart. A wrong parent is a
        # fabricated chain, which is worse than one that stays broken.
        gallery.record("Rooftop", [
            Take(prompt_id="p0", filename="parent.mp4", shot="ShotA"),
            Take(prompt_id="p1", filename="parent.mp4", shot="ShotB"),
            Take(prompt_id="p2", filename="child.mp4", shot="ShotA",
                 parent_take=":parent.mp4"),
        ])
        child = next(t for t in gallery.list("Rooftop") if t.filename == "child.mp4")
        assert child.parent_take == ":parent.mp4"

    def test_resolving_never_rewrites_the_file(self, gallery):
        # Every repair in `list` is in-memory, so the inspector can still show
        # what was actually recorded.
        gallery.record("Rooftop", [
            Take(prompt_id="p0", filename="parent.mp4", shot="ShotA"),
            Take(prompt_id="p1", filename="child.mp4", shot="ShotA",
                 parent_take=":parent.mp4"),
        ])
        gallery.list("Rooftop")
        stored = json.loads(gallery.path_for("Rooftop").read_text(encoding="utf-8"))
        parents = {t["filename"]: t["parent_take"] for t in stored["takes"]}
        assert parents["child.mp4"] == ":parent.mp4"
