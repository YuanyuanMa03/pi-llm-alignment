"""Tests for factual constraint implementations."""

import pytest
import torch

from src.constraints.factual import (
    NumericalBoundsConstraint,
    KeywordPresenceConstraint
)


class TestNumericalBoundsConstraint:
    """Test suite for NumericalBoundsConstraint."""

    @pytest.fixture
    def simple_constraint(self):
        """Fixture providing a simple single-bound constraint."""
        return NumericalBoundsConstraint(
            bounds={"value": (0, 100)},
            weight=1.0,
            enabled=True
        )

    @pytest.fixture
    def multi_constraint(self):
        """Fixture providing a constraint with multiple bounds."""
        return NumericalBoundsConstraint(
            bounds={"height_cm": (0, 300), "weight_kg": (0, 200)},
            weight=1.0,
            enabled=True
        )

    @pytest.fixture
    def disabled_constraint(self):
        """Fixture providing a disabled constraint."""
        return NumericalBoundsConstraint(
            bounds={"value": (0, 100)},
            weight=1.0,
            enabled=False
        )

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_initialization_valid(self):
        """Test constraint initialization with valid bounds."""
        constraint = NumericalBoundsConstraint({"x": (0, 1)})
        assert constraint.bounds == {"x": (0, 1)}
        assert constraint.weight == 1.0
        assert constraint.enabled is True
        assert constraint.tokenizer is None

    def test_initialization_invalid_bounds(self):
        """Test that invalid bounds raise ValueError."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            NumericalBoundsConstraint({"x": (10, 5)})  # min > max

    def test_initialization_negative_weight(self):
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="weight must be non-negative"):
            NumericalBoundsConstraint({"x": (0, 1)}, weight=-1.0)

    # ========================================================================
    # check() Tests
    # ========================================================================

    def test_check_valid_input(self, simple_constraint):
        """Test that valid inputs pass the check."""
        assert simple_constraint.check("The value is 50")
        assert simple_constraint.check("Value: 25.5")
        assert simple_constraint.check("Values are 0 and 100")

    def test_check_invalid_above_max(self, simple_constraint):
        """Test that values above max fail the check."""
        assert not simple_constraint.check("The value is 150")
        assert not simple_constraint.check("Value: 100.1")

    def test_check_invalid_below_min(self, simple_constraint):
        """Test that values below min fail the check."""
        assert not simple_constraint.check("The value is -10")
        assert not simple_constraint.check("Value: -0.1")

    def test_check_multiple_bounds(self, multi_constraint):
        """Test checking with multiple bounds."""
        # Both within bounds
        assert multi_constraint.check("Height: 175 cm, Weight: 70 kg")

        # Height out of bounds
        assert not multi_constraint.check("Height: 350 cm, Weight: 70 kg")

        # Weight out of bounds
        assert not multi_constraint.check("Height: 175 cm, Weight: 250 kg")

    def test_check_negative_numbers(self, simple_constraint):
        """Test checking with negative numbers."""
        assert not simple_constraint.check("The value is -5")
        # Multiple negative numbers
        assert not simple_constraint.check("Values: -10, -20, -30")

    # ========================================================================
    # compute_penalty() Tests
    # ========================================================================

    def test_compute_penalty_zero_for_valid(self, simple_constraint):
        """Test that valid inputs have zero penalty."""
        assert simple_constraint.compute_penalty("The value is 50") == 0.0
        assert simple_constraint.compute_penalty("No numbers here") == 0.0

    def test_compute_penalty_positive_for_invalid(self, simple_constraint):
        """Test that invalid inputs have positive penalty."""
        penalty = simple_constraint.compute_penalty("The value is 150")
        assert penalty > 0
        # Penalty should be (150 - 100) / 100 = 0.5
        assert abs(penalty - 0.5) < 0.01

    def test_compute_penalty_disabled_constraint(self, disabled_constraint):
        """Test that disabled constraint returns zero penalty."""
        penalty = disabled_constraint.compute_penalty("The value is 999")
        assert penalty == 0.0

    def test_compute_penalty_multiple_violations(self, simple_constraint):
        """Test penalty with multiple violating numbers."""
        penalty = simple_constraint.compute_penalty("Values: 150, 200, -50")
        # Should sum penalties for all three violations
        assert penalty > 0

    # ========================================================================
    # get_logits_mask() Tests
    # ========================================================================

    def test_get_logits_mask_shape(self, simple_constraint):
        """Test that logit mask has correct shape."""
        vocab_size = 1000
        mask = simple_constraint.get_logits_mask([], vocab_size)

        assert mask.shape == (vocab_size,)
        assert mask.dtype == torch.bool

    def test_get_logits_mask_without_tokenizer(self, simple_constraint):
        """Test that mask is empty without tokenizer."""
        vocab_size = 100
        mask = simple_constraint.get_logits_mask([], vocab_size)

        # No tokenizer means no masking can occur
        assert not mask.any()

    def test_get_logits_mask_disabled_constraint(self, disabled_constraint):
        """Test that disabled constraint returns empty mask."""
        vocab_size = 100
        mask = disabled_constraint.get_logits_mask([], vocab_size)

        assert not mask.any()

    # ========================================================================
    # apply_mask_to_logits() Tests
    # ========================================================================

    def test_apply_mask_to_logits_shape(self, simple_constraint):
        """Test that apply_mask_to_logits preserves shape."""
        vocab_size = 100
        logits = torch.randn(1, vocab_size)

        result = simple_constraint.apply_mask_to_logits(logits, [])

        assert result.shape == logits.shape

    def test_apply_mask_to_logits_disabled(self, disabled_constraint):
        """Test that disabled constraint doesn't modify logits."""
        vocab_size = 100
        logits = torch.randn(1, vocab_size)

        result = disabled_constraint.apply_mask_to_logits(logits, [])

        # Should return unchanged logits
        assert torch.equal(result, logits)

    # ========================================================================
    # get_weighted_penalty() Tests
    # ========================================================================

    def test_get_weighted_penalty_default_weight(self, simple_constraint):
        """Test weighted penalty with default weight."""
        penalty = simple_constraint.get_weighted_penalty("The value is 150")
        expected = simple_constraint.compute_penalty("The value is 150")
        assert penalty == expected  # weight = 1.0

    def test_get_weighted_penalty_custom_weight(self):
        """Test weighted penalty with custom weight."""
        constraint = NumericalBoundsConstraint({"x": (0, 100)}, weight=2.0)
        penalty = constraint.get_weighted_penalty("The value is 150")
        expected = constraint.compute_penalty("The value is 150") * 2.0
        assert abs(penalty - expected) < 0.01

    def test_get_weighted_penalty_disabled(self, disabled_constraint):
        """Test that disabled constraint returns zero weighted penalty."""
        penalty = disabled_constraint.get_weighted_penalty("The value is 999")
        assert penalty == 0.0

    # ========================================================================
    # Utility Method Tests
    # ========================================================================

    def test_extract_numbers(self, simple_constraint):
        """Test number extraction from text."""
        text = "Values: 10, 20.5, -5, 100"
        numbers = simple_constraint._extract_numbers(text)

        assert 10 in numbers
        assert 20.5 in numbers
        assert -5 in numbers
        assert 100 in numbers

    def test_extract_partial_number(self, simple_constraint):
        """Test partial number extraction."""
        assert simple_constraint._extract_partial_number("Value: 15") == "15"
        assert simple_constraint._extract_partial_number("Value: 15.5") == "15.5"
        assert simple_constraint._extract_partial_number("Value: -10") == "-10"
        assert simple_constraint._extract_partial_number("No numbers") is None

    def test_could_continue_number(self, simple_constraint):
        """Test number continuation detection."""
        assert simple_constraint._could_continue_number("123")
        assert simple_constraint._could_continue_number(".5")
        assert simple_constraint._could_continue_number("-10")

        assert not simple_constraint._could_continue_number("")
        assert not simple_constraint._could_continue_number("abc")
        assert not simple_constraint._could_continue_number("12a3")

    def test_repr(self, simple_constraint):
        """Test string representation."""
        repr_str = repr(simple_constraint)
        assert "NumericalBoundsConstraint" in repr_str
        assert "weight=1.0" in repr_str
        assert "enabled=True" in repr_str
        assert "has_tokenizer=" in repr_str


class TestKeywordPresenceConstraint:
    """Test suite for KeywordPresenceConstraint."""

    @pytest.fixture
    def single_keyword_constraint(self):
        """Fixture providing a single-keyword constraint."""
        return KeywordPresenceConstraint(
            required_keywords=["temperature"],
            weight=1.0,
            enabled=True
        )

    @pytest.fixture
    def multi_keyword_constraint(self):
        """Fixture providing a multi-keyword constraint."""
        return KeywordPresenceConstraint(
            required_keywords=["temperature", "pressure", "volume"],
            weight=2.0,
            enabled=True
        )

    @pytest.fixture
    def disabled_constraint(self):
        """Fixture providing a disabled constraint."""
        return KeywordPresenceConstraint(
            required_keywords=["test"],
            weight=1.0,
            enabled=False
        )

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_initialization(self):
        """Test constraint initialization."""
        constraint = KeywordPresenceConstraint(["test", "example"])
        assert constraint.required_keywords == ["test", "example"]
        assert constraint.weight == 1.0
        assert constraint.enabled is True

    # ========================================================================
    # check() Tests
    # ========================================================================

    def test_check_all_present(self, single_keyword_constraint):
        """Test that check passes when all keywords present."""
        assert single_keyword_constraint.check("The temperature is 25 degrees")
        assert single_keyword_constraint.check("TEMPERATURE is high")

    def test_check_missing_keyword(self, single_keyword_constraint):
        """Test that check fails when keyword is missing."""
        assert not single_keyword_constraint.check("The pressure is high")
        assert not single_keyword_constraint.check("No relevant words")

    def test_check_multiple_keywords(self, multi_keyword_constraint):
        """Test checking with multiple required keywords."""
        # All present
        assert multi_keyword_constraint.check(
            "Temperature, pressure, and volume are related"
        )

        # Missing one
        assert not multi_keyword_constraint.check(
            "Temperature and pressure are measured"
        )

    # ========================================================================
    # compute_penalty() Tests
    # ========================================================================

    def test_compute_penalty_zero_when_present(self, single_keyword_constraint):
        """Test that penalty is zero when keywords are present."""
        penalty = single_keyword_constraint.compute_penalty("The temperature is 25")
        assert penalty == 0.0

    def test_compute_penalty_counts_missing(self, single_keyword_constraint):
        """Test that penalty counts missing keywords."""
        penalty = single_keyword_constraint.compute_penalty("The pressure is high")
        assert penalty == 1.0  # 1 missing keyword

    def test_compute_penalty_multiple_missing(self, multi_keyword_constraint):
        """Test penalty with multiple missing keywords."""
        # Missing "pressure" and "volume"
        penalty = multi_keyword_constraint.compute_penalty("The temperature is 25")
        assert penalty == 2.0  # 2 missing keywords (weight doesn't affect count)

    def test_compute_penalty_disabled(self, disabled_constraint):
        """Test that disabled constraint has zero penalty."""
        penalty = disabled_constraint.compute_penalty("No keywords")
        assert penalty == 0.0

    # ========================================================================
    # get_logits_mask() Tests
    # ========================================================================

    def test_get_logits_mask_always_empty(self, single_keyword_constraint):
        """Test that logit mask is always empty."""
        vocab_size = 1000
        mask = single_keyword_constraint.get_logits_mask([], vocab_size)

        assert not mask.any()  # All False
        assert mask.shape == (vocab_size,)

    def test_get_logits_mask_shape(self, single_keyword_constraint):
        """Test that mask has correct shape."""
        for vocab_size in [100, 1000, 50000]:
            mask = single_keyword_constraint.get_logits_mask([], vocab_size)
            assert mask.shape == (vocab_size,)

    # ========================================================================
    # get_weighted_penalty() Tests
    # ========================================================================

    def test_get_weighted_penalty(self, multi_keyword_constraint):
        """Test weighted penalty calculation."""
        # Missing 2 keywords, weight = 2.0
        penalty = multi_keyword_constraint.get_weighted_penalty("Just temperature here")
        # compute_penalty returns 2.0 (2 missing), weight multiplies to 4.0
        # But wait - compute_penalty returns the count, not weighted
        # Let's check the implementation...
        # Actually, looking at the code, compute_penalty returns float(missing)
        # and get_weighted_penalty multiplies by weight
        expected = 2.0 * 2.0  # 2 missing * weight 2
        assert penalty == expected

    # ========================================================================
    # Utility Tests
    # ========================================================================

    def test_repr(self, multi_keyword_constraint):
        """Test string representation."""
        repr_str = repr(multi_keyword_constraint)
        assert "KeywordPresenceConstraint" in repr_str
        assert "weight=2.0" in repr_str

    def test_case_insensitive(self, single_keyword_constraint):
        """Test that keyword matching is case-insensitive."""
        assert single_keyword_constraint.check("TEMPERATURE")
        assert single_keyword_constraint.check("Temperature")
        assert single_keyword_constraint.check("tEmPeRaTuRe")

    def test_substring_matching(self):
        """Test that keyword matching finds substrings."""
        # "temperature" contains "temp" but constraint looks for "temperature"
        constraint = KeywordPresenceConstraint(["temp"])
        assert constraint.check("The temperature is high")
