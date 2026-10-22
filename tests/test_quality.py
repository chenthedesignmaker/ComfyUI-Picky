from __future__ import annotations

from video_prompter import quality


class TestResolve:
    def test_the_built_in_presets(self):
        assert quality.resolve("preview") == (0.2, 8, 1.0)
        assert quality.resolve("turbo_hd") == (1.0, 8, 1.0)
        assert quality.resolve("turbo_2pass") == (0.2, 8, 1.0)
        assert quality.resolve("final") == (1.0, 25, 0.0)

    def test_final_bypasses_the_turbo_lora(self):
        # The keeper renders on the full base model, so its turbo strength is 0.
        _, _, turbo = quality.resolve("final")
        assert turbo == 0.0

    def test_an_unknown_name_falls_back_to_the_default(self):
        assert quality.resolve("ludicrous") == quality.resolve(quality.DEFAULT_QUALITY)

    def test_a_missing_name_falls_back_to_the_default(self):
        assert quality.resolve(None) == quality.resolve(quality.DEFAULT_QUALITY)

    def test_a_project_may_override_the_numbers(self):
        settings = {"quality_presets": {"final": {"megapixels": 2.0, "steps": 30}}}
        assert quality.resolve("final", settings) == (2.0, 30, 0.0)

    def test_a_partial_override_keeps_the_other_default(self):
        settings = {"quality_presets": {"preview": {"steps": 4}}}
        assert quality.resolve("preview", settings) == (0.2, 4, 1.0)

    def test_a_project_may_override_the_turbo_strength_to_zero(self):
        # Zero is a valid turbo strength (a bypass), unlike megapixels/steps.
        settings = {"quality_presets": {"turbo_hd": {"turbo_strength": 0.0}}}
        assert quality.resolve("turbo_hd", settings) == (1.0, 8, 0.0)

    def test_a_nonsense_override_is_ignored(self):
        settings = {"quality_presets": {"preview": {"megapixels": -1, "steps": "lots"}}}
        assert quality.resolve("preview", settings) == (0.2, 8, 1.0)


class TestTurboTwoPass:
    """The turbo tier that splits its schedule instead of its canvas."""

    def test_it_costs_exactly_what_turbo_hd_costs(self):
        # The whole point of the tier is an A/B against turbo_hd, so the two
        # have to spend the same steps and finish on the same canvas. If this
        # ever drifts, a comparison between them stops meaning anything.
        _, turbo_steps, turbo_lora = quality.resolve("turbo_hd")
        _, split_steps, split_lora = quality.resolve("turbo_2pass")
        assert split_steps == turbo_steps
        assert split_lora == turbo_lora
        assert quality.refine("turbo_2pass")[1] == quality.resolve("turbo_hd")[0]

    def test_it_composes_at_the_obedient_canvas(self):
        assert quality.resolve("turbo_2pass")[0] == quality.resolve("preview")[0]

    def test_it_splits_its_eight_steps_six_and_two(self):
        pass1, _ = quality.refine("turbo_2pass")
        total = quality.resolve("turbo_2pass")[1]
        assert (pass1, total - pass1) == (6, 2)

    def test_it_hands_over_at_the_same_sigma_the_keeper_does(self):
        """Why the cut is at 6 rather than in the middle.

        The latent upscaler needs a nearly-clean latent, and the sigma reached
        is what decides that. At shift 12 with the `simple` scheduler the curve
        is back-loaded, so an 8-step run reaches `final_2pass`'s proven 0.800
        handover only at step 6. Cut at 4 and the handover is 0.923 — measured
        output is speckle. These are ComfyUI's own numbers, hardcoded because
        computing them needs torch and the whole model config.
        """
        eight = [1.0, 0.9882, 0.9730, 0.9524, 0.9231, 0.8780, 0.8000, 0.6316, 0.0]
        twenty_at_15 = 0.8000

        pass1, _ = quality.refine("turbo_2pass")
        assert eight[pass1] == twenty_at_15

        # And that really is where final_2pass hands over, so the two agree.
        keeper_split, _ = quality.refine("final_2pass")
        assert (keeper_split, quality.resolve("final_2pass")[1]) == (15, 20)

    def test_it_keeps_the_turbo_lora_on(self):
        # Unlike final_2pass, which bypasses it. This is a draft tier.
        assert quality.resolve("turbo_2pass")[2] == 1.0

    def test_it_pays_for_identity(self):
        # `match` would size references to pass 1's 0.2 MP, which would lose
        # the A/B against turbo_hd for a reason unrelated to the split.
        assert quality.reference_size("turbo_2pass") == "max"


class TestRefine:
    def test_the_single_pass_tiers_have_no_refine(self):
        assert quality.refine("preview") is None
        assert quality.refine("turbo_hd") is None
        assert quality.refine("final") is None

    def test_both_two_pass_tiers_have_one(self):
        assert quality.refine("turbo_2pass") is not None
        assert quality.refine("final_2pass") is not None

    def test_the_two_pass_tier_splits_its_own_step_count(self):
        # 20 steps cut at 15 leaves 5 for the finishing pass.
        assert quality.resolve("final_2pass") == (0.2, 20, 0.0)
        assert quality.refine("final_2pass") == (15, 1.0)

    def test_pass_one_renders_at_the_obedient_canvas(self):
        # The whole point: the composition is decided at preview's 0.2 MP, and
        # only the finish happens at 720p.
        megapixels, _, _ = quality.resolve("final_2pass")
        assert megapixels == quality.resolve("preview")[0]
        assert quality.refine("final_2pass")[1] == quality.resolve("final")[0]

    def test_it_runs_on_the_base_model(self):
        assert quality.resolve("final_2pass")[2] == 0.0

    def test_a_missing_name_has_no_refine(self):
        assert quality.refine(None) is None

    def test_a_project_may_move_the_split(self):
        settings = {"quality_presets": {"final_2pass": {"pass1_steps": 12}}}
        assert quality.refine("final_2pass", settings) == (12, 1.0)

    def test_a_project_may_change_the_finished_size(self):
        settings = {"quality_presets": {"final_2pass": {"refine_megapixels": 2.0}}}
        assert quality.refine("final_2pass", settings) == (15, 2.0)

    def test_a_split_past_the_schedule_is_clamped(self):
        # Otherwise pass 2 would be handed an empty sigma run and fail.
        settings = {"quality_presets": {"final_2pass": {"pass1_steps": 99}}}
        pass1, _ = quality.refine("final_2pass", settings)
        assert pass1 == quality.resolve("final_2pass", settings)[1] - 1

    def test_the_split_is_clamped_against_an_overridden_step_count(self):
        settings = {"quality_presets": {"final_2pass": {"steps": 10}}}
        assert quality.refine("final_2pass", settings) == (9, 1.0)


class TestReferenceSize:
    """How large identity references reach the ref2v node, per tier."""

    def test_the_cheap_tiers_scale_references_to_their_own_canvas(self):
        assert quality.reference_size("preview") == "match"
        assert quality.reference_size("turbo_hd") == "match"

    def test_the_keepers_pay_for_identity(self):
        assert quality.reference_size("final") == "max"
        assert quality.reference_size("final_2pass") == "max"

    def test_an_unknown_name_falls_back_to_the_default_tier(self):
        assert quality.reference_size("ludicrous") == quality.reference_size(
            quality.DEFAULT_QUALITY
        )

    def test_a_missing_name_falls_back_to_the_default_tier(self):
        assert quality.reference_size(None) == quality.reference_size(
            quality.DEFAULT_QUALITY
        )

    def test_a_project_may_buy_identity_on_the_cheap_tier(self):
        settings = {"quality_presets": {"preview": {"reference_size": "max"}}}
        assert quality.reference_size("preview", settings) == "max"

    def test_a_value_outside_the_vocabulary_is_ignored(self):
        settings = {"quality_presets": {"preview": {"reference_size": "enormous"}}}
        assert quality.reference_size("preview", settings) == "match"

    def test_overriding_the_numbers_leaves_the_size_alone(self):
        settings = {"quality_presets": {"final": {"megapixels": 2.0}}}
        assert quality.reference_size("final", settings) == "max"


class TestSelfLiftTiers:
    def test_is_selflift(self):
        assert quality.is_selflift("selflift_hd") is True
        assert quality.is_selflift("selflift_fhd") is True
        assert quality.is_selflift("selflift_final") is True
        assert quality.is_selflift("preview") is False
        assert quality.is_selflift("turbo_hd") is False
        assert quality.is_selflift("turbo_2pass") is False
        assert quality.is_selflift("final") is False
        assert quality.is_selflift("final_2pass") is False
        assert quality.is_selflift(None) is False

    def test_selflift_presets_resolve(self):
        assert quality.resolve("selflift_hd") == (0.2, 8, 1.0)
        assert quality.resolve("selflift_fhd") == (0.5, 8, 1.0)
        assert quality.resolve("selflift_final") == (0.5, 20, 0.0)

    def test_selflift_refine_presets(self):
        assert quality.refine("selflift_hd") == (6, 1.0)
        assert quality.refine("selflift_fhd") == (6, 2.1)
        assert quality.refine("selflift_final") == (15, 2.1)

    def test_selflift_reference_size_is_max(self):
        assert quality.reference_size("selflift_hd") == "max"
        assert quality.reference_size("selflift_fhd") == "max"
        assert quality.reference_size("selflift_final") == "max"


class TestSingularityTiers:
    def test_is_singularity_2pass(self):
        assert quality.is_singularity_2pass("singularity_2pass") is True
        assert quality.is_singularity_2pass("preview") is False
        assert quality.is_singularity_2pass("turbo_hd") is False
        assert quality.is_singularity_2pass("turbo_2pass") is False
        assert quality.is_singularity_2pass("selflift_hd") is False
        assert quality.is_singularity_2pass("final") is False
        assert quality.is_singularity_2pass("final_2pass") is False
        assert quality.is_singularity_2pass(None) is False

    def test_singularity_preset_resolves(self):
        # 0.5 MP pass 1, 24 total steps (20 pass 1 + 4 pass 2), turbo LoRA bypassed for pass 1 (0.0)
        assert quality.resolve("singularity_2pass") == (0.5, 24, 0.0)

    def test_singularity_refine_preset(self):
        # 20 pass 1 steps, 2.1 MP finished output (1080p equivalent)
        assert quality.refine("singularity_2pass") == (20, 2.1)

    def test_singularity_reference_size_is_max(self):
        # Keeper quality: full 2048px short edge reference tokens
        assert quality.reference_size("singularity_2pass") == "max"
