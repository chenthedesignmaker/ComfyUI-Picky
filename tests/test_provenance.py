from __future__ import annotations

from video_prompter import (
    AudioCue,
    Beat,
    Line,
    Reference,
    ReferenceSet,
    Storyboard,
    VideoPrompt,
    compose,
)
from video_prompter.provenance import spans


def _quoted(prompt: VideoPrompt) -> dict[str, str]:
    """Each span's field id mapped to the text it covers in the composed prompt."""
    text = compose(prompt)
    return {span["field"]: text[span["start"] : span["end"]] for span in spans(prompt)}


class TestSpansCoverTheAuthoredText:
    def test_every_block_field_is_located(self, clean_prompt):
        clean_prompt.overview = "A post-rain dusk metropolis."
        clean_prompt.screen_text = ["DAY ONE"]
        found = _quoted(clean_prompt)
        # The style opens [Shot 1] and the next clause continues it, so the
        # composed span stops short of the authored full stop.
        assert found["style"] == "Realistic live-action cinematic look, anamorphic lens, film grain"
        assert found["overview"] == "A post-rain dusk metropolis."
        # The soundscape is a paragraph of sentences now, so the bed is
        # re-cased where it lands.
        assert found["audio"] == "Rain, distant traffic, a low sustained string bed"
        assert found["music"] == "Sparse piano at a slow tempo under sustained low strings."
        assert found["screen_text.0"] == "DAY ONE"

    def test_beat_fields_are_located_per_beat(self, clean_prompt):
        found = _quoted(clean_prompt)
        assert found["beat.1.angle"] == "medium tracking"
        assert found["beat.1.action"] == "she crosses the empty plaza"

    def test_a_span_points_at_the_row_it_came_from_and_names_the_shot(self):
        # Beats are authored in list order but composed in time order.
        prompt = VideoPrompt(
            storyboard=Storyboard(6, [Beat(3, 6, "b", "second"), Beat(0, 3, "a", "first")])
        )
        by_field = {span["field"]: span for span in spans(prompt)}
        assert by_field["beat.0.action"]["label"] == "Shot 2 · Action"
        assert by_field["beat.1.action"]["label"] == "Shot 1 · Action"

    def test_camera_motion_keeps_its_prepended_subject_out_of_the_span(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(4, [Beat(0, 4, "wide", "they walk", camera="drifts left")])
        )
        assert "The camera drifts left." in compose(prompt)
        assert _quoted(prompt)["beat.0.camera"] == "drifts left"

    def test_camera_motion_that_names_the_camera_is_covered_whole(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(
                4, [Beat(0, 4, "wide", "they walk", camera="the camera holds still")]
            )
        )
        # Composed as its own sentence, so the author's lower-case subject is
        # re-cased; the span still covers the words they wrote.
        assert _quoted(prompt)["beat.0.camera"] == "The camera holds still"

    def test_dialogue_parts_are_located_separately(self):
        prompt = VideoPrompt(
            storyboard=Storyboard(
                4,
                [
                    Beat(
                        0,
                        4,
                        "close",
                        "she looks up",
                        lines=[
                            Line(
                                speakers=["w"],
                                text="It is starting.",
                                describe="the young woman",
                                delivery="softly",
                            )
                        ],
                    )
                ],
            )
        )
        found = _quoted(prompt)
        # `_capitalize` re-cases the first letter; the span still covers the word.
        assert found["line.0.0.describe"] == "The young woman"
        assert found["line.0.0.delivery"] == "softly"
        assert found["line.0.0.text"] == "It is starting."

    def test_a_cue_is_located_inside_the_soundscape(self, clean_prompt):
        clean_prompt.storyboard.cues = [AudioCue(4, None, "a distant siren")]
        assert _quoted(clean_prompt)["cue.0.description"] == "a distant siren"

    def test_reference_fields_are_located_in_the_six_section_document(self, clean_prompt):
        clean_prompt.references = ReferenceSet(
            [Reference("image", "lead.png", role="the lead, in a red coat", retention="strict")]
        )
        clean_prompt.ref_summary = "The target video follows <Picture 1> through the market."
        found = _quoted(clean_prompt)
        assert found["ref.0.role"] == "the lead, in a red coat"
        assert found["ref.0.retention"] == "strict"
        assert found["ref_summary"] == "The target video follows <Picture 1> through the market."


class TestSpansStayHonest:
    def test_spans_never_overlap_and_run_in_order(self, clean_prompt):
        clean_prompt.storyboard.cues = [AudioCue(2, 5, "thunder rolls")]
        found = spans(clean_prompt)
        assert found == sorted(found, key=lambda span: span["start"])
        assert all(
            later["start"] >= earlier["end"] for earlier, later in zip(found, found[1:])
        )

    def test_a_span_quotes_the_composed_text_it_covers(self, clean_prompt):
        text = compose(clean_prompt)
        for span in spans(clean_prompt):
            assert 0 <= span["start"] < span["end"] <= len(text)

    def test_an_empty_prompt_has_no_spans(self):
        assert spans(VideoPrompt(storyboard=Storyboard(10))) == []

    def test_an_empty_field_is_never_marked(self, clean_prompt):
        clean_prompt.music = ""
        assert "non_diegetic_music: N/A" in compose(clean_prompt)
        assert "music" not in {span["field"] for span in spans(clean_prompt)}

    def test_a_defaulted_transition_belongs_to_no_field(self, clean_prompt):
        # The beats carry no transition, so the default cut is compose's words.
        assert "the camera cuts to" in compose(clean_prompt)
        assert not [span for span in spans(clean_prompt) if span["kind"] == "transition"]
