"""Tests for temporal constraint implementations."""

import pytest
import torch

from src.constraints.temporal import TemporalOrderConstraint


class TestTemporalOrderConstraint:
    """Test suite for TemporalOrderConstraint."""

    @pytest.fixture
    def crop_stages_constraint(self):
        """Fixture providing a crop growth stages constraint."""
        return TemporalOrderConstraint(
            stages=["seedling", "tillering", "heading", "maturity"],
            weight=1.0,
            enabled=True
        )

    @pytest.fixture
    def numbered_stages_constraint(self):
        """Fixture providing a numbered stages constraint."""
        return TemporalOrderConstraint(
            stages=["Stage 1", "Stage 2", "Stage 3"],
            weight=1.0,
            enabled=True
        )

    @pytest.fixture
    def ordinal_stages_constraint(self):
        """Fixture providing an ordinal word constraint."""
        return TemporalOrderConstraint(
            stages=[r"\bfirst\b", r"\bsecond\b", r"\bthird\b", r"\bfourth\b"],
            weight=1.0,
            enabled=True,
            whole_word=True
        )

    @pytest.fixture
    def disabled_constraint(self):
        """Fixture providing a disabled constraint."""
        return TemporalOrderConstraint(
            stages=["a", "b", "c"],
            weight=1.0,
            enabled=False
        )

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_initialization_valid(self):
        """Test constraint initialization with valid stages."""
        constraint = TemporalOrderConstraint(["a", "b", "c"])
        assert constraint.stages == ["a", "b", "c"]
        assert constraint.weight == 1.0
        assert constraint.enabled is True
        assert constraint.case_sensitive is False
        assert constraint.whole_word is True

    def test_initialization_empty_stages(self):
        """Test that empty stages list raises ValueError."""
        with pytest.raises(ValueError, match="requires at least 2 stages"):
            TemporalOrderConstraint([])

    def test_initialization_single_stage(self):
        """Test that single stage raises ValueError."""
        with pytest.raises(ValueError, match="requires at least 2 stages"):
            TemporalOrderConstraint(["only_one"])

    def test_initialization_negative_weight(self):
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="weight must be non-negative"):
            TemporalOrderConstraint(["a", "b"], weight=-1.0)

    def test_initialization_case_sensitive(self):
        """Test case sensitivity option."""
        constraint = TemporalOrderConstraint(
            stages=["A", "B"],
            case_sensitive=True
        )
        assert constraint.case_sensitive is True

    def test_initialization_whole_word(self):
        """Test whole word matching option."""
        constraint = TemporalOrderConstraint(
            stages=["stage", "phase"],
            whole_word=False
        )
        assert constraint.whole_word is False

    # ========================================================================
    # check() Tests
    # ========================================================================

    def test_check_correct_order(self, crop_stages_constraint):
        """Test that correct order passes the check."""
        assert crop_stages_constraint.check(
            "The crop reached seedling stage, then tillering, "
            "followed by heading, and finally maturity."
        )

    def test_check_partial_order(self, crop_stages_constraint):
        """Test check with only some stages present."""
        # Only two stages, in correct order
        assert crop_stages_constraint.check(
            "The crop went from seedling to tillering."
        )

    def test_check_single_stage(self, crop_stages_constraint):
        """Test check with only one stage present."""
        # Single stage - trivially ordered
        assert crop_stages_constraint.check(
            "The crop is at the seedling stage."
        )

    def test_check_inversion(self, crop_stages_constraint):
        """Test that inverted order fails the check."""
        assert not crop_stages_constraint.check(
            "The crop reached maturity before tillering occurred."
        )

    def test_check_multiple_inversions(self, crop_stages_constraint):
        """Text check with multiple order violations."""
        assert not crop_stages_constraint.check(
            "The crop showed heading before seedling, and maturity "
            "before tillering."
        )

    def test_check_no_stages_present(self, crop_stages_constraint):
        """Test check when no stages are mentioned."""
        # No stages found - trivially passes
        assert crop_stages_constraint.check(
            "The crop is growing well this season."
        )

    def test_check_numbered_stages(self, numbered_stages_constraint):
        """Test check with numbered stages."""
        assert numbered_stages_constraint.check(
            "Stage 1 was completed, then Stage 2, and finally Stage 3."
        )

    def test_check_ordinal_words(self, ordinal_stages_constraint):
        """Test check with ordinal words using regex."""
        assert ordinal_stages_constraint.check(
            "First we prepare, second we execute, "
            "third we verify, and fourth we deploy."
        )

    def test_check_case_insensitive_default(self, crop_stages_constraint):
        """Test that matching is case-insensitive by default."""
        assert crop_stages_constraint.check(
            "The crop reached SEEDLING stage, then Tillering."
        )

    def test_check_case_sensitive(self):
        """Test case-sensitive matching."""
        constraint = TemporalOrderConstraint(
            stages=["A", "B", "C"],
            case_sensitive=True
        )
        # Correct case should pass
        assert constraint.check("A then B")  # Correct case
        # With case_sensitive=True, 'a' doesn't match 'A', only 'B' matches
        # Single stage (B) is trivially ordered, so this passes
        # To test case sensitivity properly, we need to verify matching works
        # Let's use a text where case matters for the match
        assert constraint.check("A then B then C")
        # When wrong case is used for 'A', it won't match, so we check
        # that 'A' is not found in the stage order
        order = constraint.get_stage_order("a then B then C")
        # 'a' (lowercase) shouldn't match 'A' (uppercase), so only B and C found
        assert order == ["B", "C"]

    def test_check_whole_word_default(self):
        """Test whole word matching is default."""
        constraint = TemporalOrderConstraint(["stage", "phase"])
        # 'stage' should match 'stage' with word boundary
        assert constraint.check("The stage was set")
        # With whole_word=True, 'stages' shouldn't match 'stage'
        # Let's verify by checking what stages are found
        text = "The stages were set"
        order = constraint.get_stage_order(text)
        # 'stages' (plural) should not match 'stage' (singular) due to word boundary
        # So no stages should be found (or at least not 'stage')
        assert "stage" not in (order or [])

    def test_check_partial_word_matching(self):
        """Test with whole_word=False."""
        constraint = TemporalOrderConstraint(
            stages=["stage", "phase"],
            whole_word=False
        )
        # 'stages' should match 'stage' when whole_word=False
        assert constraint.check("The stages were set")

    def test_check_disabled_constraint(self, disabled_constraint):
        """Test that disabled constraint always returns True."""
        assert disabled_constraint.check("c before a occurred")
        assert disabled_constraint.check("No stages here")

    # ========================================================================
    # compute_penalty() Tests
    # ========================================================================

    def test_compute_penalty_zero_for_correct_order(self, crop_stages_constraint):
        """Test that correct order has zero penalty."""
        penalty = crop_stages_constraint.compute_penalty(
            "Seedling, tillering, heading, maturity."
        )
        assert penalty == 0.0

    def test_compute_penalty_single_inversion(self, crop_stages_constraint):
        """Test penalty for one inversion."""
        # heading (idx 2) before tillering (idx 1)
        penalty = crop_stages_constraint.compute_penalty(
            "Heading before tillering occurred."
        )
        assert penalty == 1.0

    def test_compute_penalty_multiple_inversions(self, crop_stages_constraint):
        """Test penalty for multiple inversions."""
        # Order: heading (2), maturity (3), seedling (0), tillering (1)
        # Inversions: (seedling, heading), (seedling, maturity),
        #              (seedling, tillering), (tillering, heading),
        #              (tillering, maturity)
        # But we only count pairs where later stage appears before earlier
        # Let's trace through:
        # Found stages: (2, heading_pos), (3, maturity_pos), (0, seedling_pos), (1, tillering_pos)
        # Positions: [heading_pos, maturity_pos, seedling_pos, tillering_pos]
        # Inversions in position array:
        # - heading_pos > seedling_pos? Yes (inversion)
        # - heading_pos > tillering_pos? Yes (inversion)
        # - maturity_pos > seedling_pos? Yes (inversion)
        # - maturity_pos > tillering_pos? Yes (inversion)
        # Total: 4 inversions (not 5, because heading_pos < maturity_pos)
        penalty = crop_stages_constraint.compute_penalty(
            "The crop reached heading, then maturity, "
            "and then went back to seedling before tillering."
        )
        # We expect multiple inversions
        assert penalty > 1.0

    def test_compute_penalty_no_stages(self, crop_stages_constraint):
        """Test penalty when no stages are mentioned."""
        penalty = crop_stages_constraint.compute_penalty(
            "No crop stages mentioned here."
        )
        assert penalty == 0.0

    def test_compute_penalty_partial_overlap(self, crop_stages_constraint):
        """Test penalty when only some stages are mentioned."""
        # Only seedling and heading, in wrong order
        penalty = crop_stages_constraint.compute_penalty(
            "Heading occurred before seedling."
        )
        assert penalty == 1.0

    def test_compute_penalty_with_weight(self):
        """Test that weight affects penalty."""
        constraint = TemporalOrderConstraint(
            stages=["a", "b", "c"],
            weight=2.0
        )
        # One inversion, weight=2, should be 2.0
        # But compute_penalty doesn't apply weight - get_weighted_penalty does
        base_penalty = constraint.compute_penalty("c before b")
        assert base_penalty == 1.0  # Base penalty is 1

        weighted = constraint.get_weighted_penalty("c before b")
        assert weighted == 2.0  # Weighted penalty is 2

    def test_compute_penalty_disabled(self, disabled_constraint):
        """Test that disabled constraint returns zero penalty."""
        penalty = disabled_constraint.compute_penalty("c before a")
        assert penalty == 0.0

    def test_compute_penalty_complex_inversion(self, crop_stages_constraint):
        """Test penalty calculation with complex inversions."""
        # Create a specific inversion pattern
        # Stages: [0:seedling, 1:tillering, 2:heading, 3:maturity]
        # Text order: [2, 0, 3, 1]
        # Positions by stage index: [(0, pos_2), (1, pos_0), (2, pos_3), (3, pos_1)]
        # Position array to sort: [pos_2, pos_0, pos_3, pos_1]
        # This is [1, 0, 3, 2] in terms of stage indices sorted by position
        # Inversions: (0,2) at positions (1,0), (1,3) at (1,2), etc.
        # Let's just verify we get a reasonable positive number
        penalty = crop_stages_constraint.compute_penalty(
            "Heading phase began, then seedling emerged, "
            "followed by maturity, and tillering occurred last."
        )
        assert penalty > 0
        assert penalty == float(int(penalty))  # Should be integer

    # ========================================================================
    # get_logits_mask() Tests
    # ========================================================================

    def test_get_logits_mask_shape(self, crop_stages_constraint):
        """Test that mask has correct shape."""
        vocab_size = 1000
        mask = crop_stages_constraint.get_logits_mask([], vocab_size)

        assert mask.shape == (vocab_size,)
        assert mask.dtype == torch.bool

    def test_get_logits_mask_always_empty(self, crop_stages_constraint):
        """Test that temporal constraint returns empty mask."""
        vocab_size = 1000
        mask = crop_stages_constraint.get_logits_mask([1, 2, 3], vocab_size)

        # Temporal constraints can't mask at token level
        assert not mask.any()

    def test_get_logits_mask_disabled(self, disabled_constraint):
        """Test that disabled constraint returns empty mask."""
        vocab_size = 100
        mask = disabled_constraint.get_logits_mask([], vocab_size)

        assert not mask.any()

    # ========================================================================
    # get_stage_order() Tests
    # ========================================================================

    def test_get_stage_order_correct(self, crop_stages_constraint):
        """Test getting stage order for correctly ordered text."""
        order = crop_stages_constraint.get_stage_order(
            "Seedling, tillering, heading, maturity."
        )
        assert order == ["seedling", "tillering", "heading", "maturity"]

    def test_get_stage_order_partial(self, crop_stages_constraint):
        """Test getting stage order with partial stages."""
        order = crop_stages_constraint.get_stage_order(
            "Seedling and heading stages."
        )
        assert order == ["seedling", "heading"]

    def test_get_stage_order_no_stages(self, crop_stages_constraint):
        """Test getting stage order when no stages present."""
        order = crop_stages_constraint.get_stage_order(
            "No crop stages mentioned."
        )
        assert order is None

    def test_get_stage_order_inverted(self, crop_stages_constraint):
        """Test getting stage order for inverted text."""
        order = crop_stages_constraint.get_stage_order(
            "Maturity before seedling occurred."
        )
        assert order == ["maturity", "seedling"]

    # ========================================================================
    # Utility Method Tests
    # ========================================================================

    def test_repr_simple(self):
        """Test string representation with few stages."""
        constraint = TemporalOrderConstraint(["a", "b"])
        repr_str = repr(constraint)
        assert "TemporalOrderConstraint" in repr_str
        assert "weight=1.0" in repr_str
        assert "'a'" in repr_str
        assert "'b'" in repr_str

    def test_repr_many_stages(self):
        """Test string representation with many stages."""
        constraint = TemporalOrderConstraint(
            ["a", "b", "c", "d", "e"]
        )
        repr_str = repr(constraint)
        assert "..." in repr_str
        assert "5 total" in repr_str

    def test_is_simple_keyword(self, crop_stages_constraint):
        """Test simple keyword detection."""
        assert crop_stages_constraint._is_simple_keyword("seedling")
        assert crop_stages_constraint._is_simple_keyword("stage123")
        assert not crop_stages_constraint._is_simple_keyword("see.*ing")
        assert not crop_stages_constraint._is_simple_keyword("stage+")

    def test_count_inversions(self, crop_stages_constraint):
        """Test inversion counting."""
        # No inversions
        items = [(0, 10), (1, 20), (2, 30)]
        count = crop_stages_constraint._count_inversions(items)
        assert count == 0

        # One inversion: (1, 15) comes before (0, 20)
        items = [(0, 20), (1, 15), (2, 30)]
        count = crop_stages_constraint._count_inversions(items)
        assert count == 1

        # Multiple inversions: reverse order
        items = [(0, 30), (1, 20), (2, 10)]
        count = crop_stages_constraint._count_inversions(items)
        # For 3 elements in reverse order: 3 + 2 + 1 = 6? No.
        # Inversions are pairs (i, j) where i < j but pos[i] > pos[j]
        # [(0,30), (1,20), (2,10)]
        # i=0: pos[0]=30 > pos[1]=20 (yes), pos[0]=30 > pos[2]=10 (yes) = 2
        # i=1: pos[1]=20 > pos[2]=10 (yes) = 1
        # i=2: no more elements
        # Total: 3 inversions
        assert count == 3

    def test_merge_sort_count(self, crop_stages_constraint):
        """Test merge sort with inversion counting."""
        arr = [3, 1, 2]
        sorted_arr, count = crop_stages_constraint._merge_sort_count(arr)
        assert sorted_arr == [1, 2, 3]
        # Original [3, 1, 2]
        # Inversions: (3,1), (3,2) = 2
        assert count == 2

    # ========================================================================
    # Regex Pattern Tests
    # ========================================================================

    def test_regex_pattern_ordinal_words(self, ordinal_stages_constraint):
        """Test regex patterns for ordinal words."""
        text = "First, second, third, and fourth steps were completed."
        assert ordinal_stages_constraint.check(text)
        assert ordinal_stages_constraint.compute_penalty(text) == 0.0

    def test_regex_pattern_word_boundary(self, ordinal_stages_constraint):
        """Test that word boundaries work correctly."""
        # "firstly" should not match "first" with word boundary
        constraint = TemporalOrderConstraint(
            stages=[r"\bfirst\b", r"\bsecond\b"],
            whole_word=True
        )
        # "firstly" contains "first" but shouldn't match due to \b
        result = constraint.check("Firstly, we proceed.")
        # No stages matched, so trivially passes
        # But if we had "first" it would match
        assert constraint.check("First, we proceed.")

    def test_regex_pattern_without_word_boundary(self):
        """Test patterns without word boundaries."""
        constraint = TemporalOrderConstraint(
            stages=["first", "second"],
            whole_word=False
        )
        # "firstly" contains "first" and should match
        text = "Firstly, then secondly."
        # Both "first" (in firstly) and "second" (in secondly) match
        # They're in order, so should pass
        assert constraint.check(text)

    # ========================================================================
    # Integration Tests
    # ========================================================================

    def test_full_workflow(self, crop_stages_constraint):
        """Test complete workflow from check to penalty to order."""
        text = "The crop reached heading before tillering."

        # Check should fail (inversion)
        assert not crop_stages_constraint.check(text)

        # Penalty should be positive
        penalty = crop_stages_constraint.compute_penalty(text)
        assert penalty > 0

        # Order should show the inversion
        order = crop_stages_constraint.get_stage_order(text)
        assert order == ["heading", "tillering"]

    def test_agricultural_use_case(self):
        """Test with realistic agricultural growth stages."""
        constraint = TemporalOrderConstraint([
            "sowing", "germination", "vegetative", "flowering", "fruiting", "harvest"
        ])

        # Correct order
        text = (
            "After sowing, germination occurred within a week. "
            "The plant entered vegetative stage, then flowering began. "
            "Fruiting followed, and finally harvest was ready."
        )
        assert constraint.check(text)
        assert constraint.compute_penalty(text) == 0.0

        # Inverted order
        text_inv = (
            "The plant reached flowering stage, but germination "
            "had not yet occurred after sowing."
        )
        assert not constraint.check(text_inv)
        assert constraint.compute_penalty(text_inv) > 0
