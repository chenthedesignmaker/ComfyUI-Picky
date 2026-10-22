from __future__ import annotations

import json

from video_prompter import (
    AudioCue,
    Beat,
    Line,
    Reference,
    ReferenceSet,
    Storyboard,
    VideoPrompt,
    compose,
    prompt_from_dict,
    prompt_to_dict,
)


class TestRoundTrip:
    def test_a_full_prompt_survives(self, clean_prompt):
        clean_prompt.references = ReferenceSet([Reference("image", "a.png", "the mood")])
        clean_prompt.storyboard.cues = [AudioCue(4, 8, "the bed swells")]
        clean_prompt.screen_text = ["MIDNIGHT LINE"]
        clean_prompt.set_ratio("21:9")

        restored = prompt_from_dict(prompt_to_dict(clean_prompt))
        assert compose(restored) == compose(clean_prompt)
        assert (restored.width, restored.height) == (clean_prompt.width, clean_prompt.height)

    def test_the_seam_and_voiceover_fields_survive(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [
            Line(speakers=["m"], text="I still remember", voiceover=True,
                 possessive="his", continuity="into")
        ]
        clean_prompt.storyboard.beats[1].lines = [
            Line(speakers=["m"], text="every turn.", continuity="from", cutoff=True)
        ]
        restored = prompt_from_dict(prompt_to_dict(clean_prompt))
        assert compose(restored) == compose(clean_prompt)

    def test_the_placement_slot_survives(self, clean_prompt):
        clean_prompt.storyboard.beats[0].lines = [
            Line(speakers=["m"], text="I never meant to stay.", placement="open")
        ]
        restored = prompt_from_dict(prompt_to_dict(clean_prompt))
        assert restored.storyboard.beats[0].lines[0].placement == "open"
        assert compose(restored) == compose(clean_prompt)

    def test_an_unknown_placement_falls_back_to_the_default_slot(self):
        prompt = prompt_from_dict(
            {"beats": [{"start": 0, "end": 5,
                        "lines": [{"text": "Hi.", "placement": "middle"}]}]}
        )
        assert prompt.storyboard.beats[0].lines[0].placement == ""

    def test_an_unknown_continuity_marker_is_dropped(self):
        prompt = prompt_from_dict(
            {"beats": [{"start": 0, "end": 5,
                        "lines": [{"text": "Hi.", "continuity": "sideways"}]}]}
        )
        assert prompt.storyboard.beats[0].lines[0].continuity == ""

    def test_the_payload_is_json_clean(self, clean_prompt):
        assert json.loads(json.dumps(prompt_to_dict(clean_prompt)))

    def test_instant_and_sustained_cues_stay_distinct(self, clean_prompt):
        clean_prompt.storyboard.cues = [AudioCue(2, description="hit"), AudioCue(4, 6, "bed")]
        restored = prompt_from_dict(prompt_to_dict(clean_prompt))
        assert restored.storyboard.cues[0].is_instant
        assert restored.storyboard.cues[1].end == 6.0


class TestTolerance:
    def test_an_empty_payload_gives_a_usable_prompt(self):
        prompt = prompt_from_dict({})
        assert prompt.storyboard.beats == []
        assert (prompt.width, prompt.height) == (1344, 768)

    def test_unknown_keys_are_ignored(self, clean_prompt):
        payload = prompt_to_dict(clean_prompt)
        payload["invented_by_a_later_build"] = {"nested": True}
        assert prompt_from_dict(payload).style == clean_prompt.style

    def test_a_reference_of_unknown_kind_is_dropped_not_fatal(self):
        prompt = prompt_from_dict(
            {"references": [{"kind": "hologram", "path": "x"}, {"kind": "image", "path": "a.png"}]}
        )
        assert [item.kind for item in prompt.references.items] == ["image"]

    def test_missing_beat_fields_take_defaults(self):
        prompt = prompt_from_dict({"beats": [{"start": 0, "end": 3}]})
        assert prompt.storyboard.beats[0].angle == ""


class TestNoCutBeats:
    def test_the_cut_flag_survives_a_round_trip(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(
                duration=6,
                beats=[Beat(0, 3, action="a"), Beat(3, 6, action="b", cut=False)],
            )
        )
        back = prompt_from_dict(json.loads(json.dumps(prompt_to_dict(prompt))))
        assert [beat.cut for beat in back.storyboard.beats] == [True, False]

    def test_a_prompt_written_before_no_cut_beats_reads_as_all_cuts(self):
        # Absent means a cut: one shot per beat is what every older file meant.
        back = prompt_from_dict({"duration": 6, "beats": [{"start": 0, "end": 6}]})
        assert back.storyboard.beats[0].cut is True
