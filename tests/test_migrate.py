from __future__ import annotations

import pytest

from video_prompter import migrate
from video_prompter.sequence import Timeline
from video_prompter.sources import Source


def v1(*shots: dict) -> dict:
    return {
        "version": 1,
        "name": "Cut",
        "sequence": {"shots": list(shots)},
        "defaults": {"style": "noir"},
    }


def shot(**overrides) -> dict:
    base = {
        "id": "sh_abc123",
        "name": "Rooftop",
        "prompt": {"overview": "a rooftop"},
        "pick": "",
        "source_in": 0.0,
        "source_out": None,
        "speed": 1.0,
        "enabled": True,
        "notes": "",
        "quality": "preview",
        "continuation_of": None,
        "branch_picks": {},
    }
    base.update(overrides)
    return base


class TestDetection:
    def test_a_v1_project_is_not_v2(self):
        assert not migrate.is_v2(v1(shot()))

    def test_a_migrated_project_is_v2(self):
        assert migrate.is_v2(migrate.to_v2(v1(shot())))

    def test_shape_counts_even_when_the_version_header_lies(self):
        # A file written by a newer build should be read as what it holds.
        assert migrate.is_v2({"version": 1, "sources": [], "timeline": {"clips": []}})

    def test_migrating_twice_changes_nothing(self):
        once = migrate.to_v2(v1(shot()))
        assert migrate.to_v2(once) == once


class TestSplit:
    def test_every_shot_becomes_one_source_and_one_clip(self):
        out = migrate.to_v2(v1(shot(id="a"), shot(id="b"), shot(id="c")))
        assert len(out["sources"]) == 3
        assert len(out["timeline"]["clips"]) == 3

    def test_the_shot_id_becomes_the_source_id(self):
        # Takes record the shot they rendered for; a fresh id would orphan them.
        out = migrate.to_v2(v1(shot(id="sh_keepme")))
        assert out["sources"][0]["id"] == "sh_keepme"
        assert out["timeline"]["clips"][0]["source_id"] == "sh_keepme"

    def test_generation_intent_goes_to_the_segment(self):
        out = migrate.to_v2(v1(shot(quality="final", notes="handheld", prompt={"overview": "x"})))
        segment = out["sources"][0]["segments"][0]
        assert segment["quality"] == "final"
        assert segment["notes"] == "handheld"
        assert segment["prompt"] == {"overview": "x"}

    def test_edit_intent_goes_to_the_clip(self):
        out = migrate.to_v2(v1(shot(source_in=1.5, source_out=4.0, speed=2.0, enabled=False)))
        clip = out["timeline"]["clips"][0]
        assert (clip["source_in"], clip["source_out"], clip["speed"]) == (1.5, 4.0, 2.0)
        assert clip["enabled"] is False

    def test_a_pick_becomes_a_one_step_path(self):
        out = migrate.to_v2(v1(shot(pick="p:take.mp4")))
        assert out["sources"][0]["path"] == ["p:take.mp4"]

    def test_an_unpicked_shot_has_an_empty_path(self):
        assert migrate.to_v2(v1(shot(pick="")))["sources"][0]["path"] == []

    def test_branch_picks_become_advisory_memory(self):
        out = migrate.to_v2(v1(shot(branch_picks={"p:a.mp4": "p:b.mp4"})))
        assert out["sources"][0]["remembered"] == {"p:a.mp4": "p:b.mp4"}

    def test_the_old_continuation_guess_is_kept_not_dropped(self):
        out = migrate.to_v2(v1(shot(continuation_of="sh_other")))
        assert out["sources"][0]["legacy_continuation_of"] == "sh_other"

    def test_clips_follow_their_source_rather_than_pinning(self):
        out = migrate.to_v2(v1(shot()))
        assert out["timeline"]["clips"][0]["path_override"] is None


class TestPreservation:
    def test_the_edit_is_unchanged_by_the_migration(self):
        # Chains spread over sibling shots stay separate, each trimmed as before,
        # so the timeline after the migration is the timeline before it.
        shots = [
            shot(id="a", source_in=0.0, source_out=3.0, speed=1.0),
            shot(id="b", source_in=1.0, source_out=None, speed=2.0),
            shot(id="c", enabled=False),
        ]
        out = migrate.to_v2(v1(*shots))
        clips = out["timeline"]["clips"]
        assert [c["source_id"] for c in clips] == ["a", "b", "c"]
        assert [(c["source_in"], c["source_out"], c["speed"]) for c in clips] == [
            (0.0, 3.0, 1.0), (1.0, None, 2.0), (0.0, None, 1.0),
        ]
        assert [c["enabled"] for c in clips] == [True, True, False]

    def test_shot_order_is_clip_order(self):
        out = migrate.to_v2(v1(shot(id="x"), shot(id="y")))
        assert [c["source_id"] for c in out["timeline"]["clips"]] == ["x", "y"]

    def test_everything_outside_the_sequence_survives(self):
        out = migrate.to_v2(v1(shot()))
        assert out["name"] == "Cut"
        assert out["defaults"] == {"style": "noir"}

    def test_the_old_sequence_key_is_gone(self):
        out = migrate.to_v2(v1(shot()))
        assert "sequence" not in out

    def test_the_pre_sequence_clips_bag_is_read_too(self):
        # Projects old enough to predate ordering kept an unordered `clips` list.
        legacy = {"version": 1, "name": "Old", "clips": [shot(id="q")]}
        out = migrate.to_v2(legacy)
        assert out["sources"][0]["id"] == "q"
        assert "clips" not in out or isinstance(out.get("timeline"), dict)

    def test_an_empty_project_migrates_to_an_empty_edit(self):
        out = migrate.to_v2({"version": 1, "name": "Blank"})
        assert out["sources"] == []
        assert out["timeline"] == {"clips": []}


class TestRoundTrip:
    def test_the_output_loads_into_the_real_types(self):
        out = migrate.to_v2(v1(shot(id="a", pick="p:t.mp4"), shot(id="b")))
        sources = [Source.from_dict(entry) for entry in out["sources"]]
        timeline = Timeline.from_dict(out["timeline"])

        assert [s.id for s in sources] == ["a", "b"]
        assert sources[0].path == ["p:t.mp4"]
        assert all(s.depth == 1 for s in sources)
        assert [c.source_id for c in timeline.clips] == ["a", "b"]

    def test_a_migrated_clip_measures_against_lineage_time(self):
        out = migrate.to_v2(v1(shot(source_in=1.0, source_out=4.0)))
        clip = Timeline.from_dict(out["timeline"]).clips[0]
        assert clip.timeline_seconds(15.5) == pytest.approx(3.0)
