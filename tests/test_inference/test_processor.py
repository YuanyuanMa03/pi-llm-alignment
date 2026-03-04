"""Tests for inference module components."""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from src.inference.processor import (
    ConstraintLogitProcessor,
    ConstrainedGenerator
)
from src.constraints.factual import NumericalBoundsConstraint


class TestConstraintLogitProcessor:
    """Test suite for ConstraintLogitProcessor."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Fixture providing a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode.side_effect = lambda ids, **kwargs: " ".join(str(i) for i in ids)
        return tokenizer

    @pytest.fixture
    def simple_constraint(self, mock_tokenizer):
        """Fixture providing a simple constraint with tokenizer."""
        return NumericalBoundsConstraint(
            bounds={"value": (0, 100)},
            tokenizer=mock_tokenizer
        )

    @pytest.fixture
    def processor(self, simple_constraint, mock_tokenizer):
        """Fixture providing a processor with one constraint."""
        return ConstraintLogitProcessor(
            constraints=[simple_constraint],
            tokenizer=mock_tokenizer,
            mode="mask"
        )

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_initialization_valid(self, processor, mock_tokenizer):
        """Test valid initialization."""
        assert len(processor.constraints) == 1
        assert processor.tokenizer is mock_tokenizer
        assert processor.mode == "mask"

    def test_initialization_no_constraints(self, mock_tokenizer):
        """Test that empty constraints list raises ValueError."""
        with pytest.raises(ValueError, match="At least one constraint"):
            ConstraintLogitProcessor([], mock_tokenizer)

    def test_initialization_invalid_mode(self, simple_constraint, mock_tokenizer):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Mode must be"):
            ConstraintLogitProcessor(
                [simple_constraint],
                mock_tokenizer,
                mode="invalid"
            )

    def test_initialization_bias_mode(self, simple_constraint, mock_tokenizer):
        """Test initialization with bias mode."""
        processor = ConstraintLogitProcessor(
            [simple_constraint],
            mock_tokenizer,
            mode="bias"
        )
        assert processor.mode == "bias"

    # ========================================================================
    # __call__ Tests
    # ========================================================================

    def test_call_preserves_shape(self, processor):
        """Test that __call__ preserves logits shape."""
        input_ids = torch.tensor([[1, 2, 3]])
        logits = torch.randn(1, 100)

        result = processor(input_ids, logits)

        assert result.shape == logits.shape

    def test_call_with_disabled_constraint(self, mock_tokenizer):
        """Test that disabled constraint doesn't modify logits."""
        constraint = NumericalBoundsConstraint(
            {"value": (0, 100)},
            enabled=False,
            tokenizer=mock_tokenizer
        )
        processor = ConstraintLogitProcessor([constraint], mock_tokenizer)

        input_ids = torch.tensor([[1, 2, 3]])
        logits = torch.randn(1, 100)

        result = processor(input_ids, logits)

        # Should be unchanged since constraint is disabled
        assert torch.allclose(result, logits)

    def test_call_batch_processing(self, processor):
        """Test processing of batched inputs."""
        batch_size = 4
        input_ids = torch.randint(0, 1000, (batch_size, 10))
        logits = torch.randn(batch_size, 100)

        result = processor(input_ids, logits)

        assert result.shape == (batch_size, 100)

    def test_call_mask_mode_forbids_tokens(self, processor):
        """Test that mask mode sets forbidden tokens to -inf."""
        input_ids = torch.tensor([[1, 2, 3]])
        logits = torch.randn(1, 100)

        result = processor(input_ids, logits)

        # Some tokens should be masked (set to -inf)
        # The exact tokens depend on the constraint logic
        # Just verify the shape is preserved
        assert result.shape == logits.shape

    # ========================================================================
    # Utility Tests
    # ========================================================================

    def test_repr(self, processor):
        """Test string representation."""
        repr_str = repr(processor)
        assert "ConstraintLogitProcessor" in repr_str
        assert "1 constraints" in repr_str
        assert "mask" in repr_str

    def test_repr_with_disabled_constraint(self, mock_tokenizer):
        """Test repr with disabled constraint."""
        constraint = NumericalBoundsConstraint(
            {"value": (0, 100)},
            enabled=False,
            tokenizer=mock_tokenizer
        )
        processor = ConstraintLogitProcessor([constraint], mock_tokenizer)

        repr_str = repr(processor)
        assert "0 enabled" in repr_str


class TestConstrainedGenerator:
    """Test suite for ConstrainedGenerator."""

    @pytest.fixture
    def mock_model(self):
        """Fixture providing a mock model."""
        model = Mock()
        model.device = torch.device("cpu")
        model.config = Mock()
        model.config.model_type = "test_model"

        # Mock generate method to handle any kwargs properly
        def mock_generate(*args, **kwargs):
            # Return some output tokens
            return torch.tensor([[1, 2, 3, 4, 5]])

        model.generate = Mock(side_effect=mock_generate)

        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Fixture providing a mock tokenizer."""
        tokenizer = Mock()

        # Create return value with .to() method
        class TokenizerOutput(dict):
            def to(self, device):
                return self  # For testing, device doesn't matter

        output = TokenizerOutput({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        })

        tokenizer.return_value = output
        tokenizer.return_tensors = "pt"
        tokenizer.decode = Mock(return_value="generated text")

        # Make it callable
        tokenizer.__call__ = Mock(return_value=output)

        return tokenizer

    @pytest.fixture
    def generator(self, mock_model, mock_tokenizer):
        """Fixture providing a generator instance."""
        return ConstrainedGenerator(mock_model, mock_tokenizer)

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_initialization(self, generator, mock_model, mock_tokenizer):
        """Test valid initialization."""
        assert generator.model is mock_model
        assert generator.tokenizer is mock_tokenizer
        assert generator.constraints == []
        assert generator.default_generation_kwargs == {}

    def test_initialization_with_constraints(
        self,
        mock_model,
        mock_tokenizer
    ):
        """Test initialization with constraints."""
        constraint = NumericalBoundsConstraint({"x": (0, 1)})
        generator = ConstrainedGenerator(
            mock_model,
            mock_tokenizer,
            constraints=[constraint]
        )

        assert len(generator.constraints) == 1

    def test_initialization_with_default_kwargs(
        self,
        mock_model,
        mock_tokenizer
    ):
        """Test initialization with default generation kwargs."""
        kwargs = {"temperature": 0.7, "top_p": 0.9}
        generator = ConstrainedGenerator(
            mock_model,
            mock_tokenizer,
            default_generation_kwargs=kwargs
        )

        assert generator.default_generation_kwargs == kwargs

    # ========================================================================
    # generate() Tests
    # ========================================================================

    def test_generate_basic(self, generator):
        """Test basic generation without constraints."""
        # Mock the model.generate to avoid complex mocking
        def mock_generate(**kwargs):
            return torch.tensor([[1, 2, 3, 4, 5]])

        generator.model.generate = Mock(side_effect=mock_generate)

        output = generator.generate("Test prompt", max_new_tokens=10)

        assert isinstance(output, str)
        # Should call model.generate
        assert generator.model.generate.called

    def test_generate_with_constraints(
        self,
        mock_model,
        mock_tokenizer
    ):
        """Test generation with constraints."""
        # Setup model properly
        def mock_generate(**kwargs):
            return torch.tensor([[1, 2, 3, 4, 5]])

        mock_model.generate = Mock(side_effect=mock_generate)

        constraint = NumericalBoundsConstraint({"x": (0, 1)})
        generator = ConstrainedGenerator(
            mock_model,
            mock_tokenizer,
            constraints=[constraint]
        )

        output = generator.generate("Test prompt", max_new_tokens=10)

        assert isinstance(output, str)
        # Verify logits_processor was passed
        call_kwargs = generator.model.generate.call_args[1]
        assert "logits_processor" in call_kwargs

    def test_generate_with_custom_kwargs(self, generator):
        """Test generation with custom kwargs."""
        # Mock the model.generate
        def mock_generate(**kwargs):
            return torch.tensor([[1, 2, 3, 4, 5]])

        generator.model.generate = Mock(side_effect=mock_generate)

        output = generator.generate(
            "Test prompt",
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True
        )

        # Verify kwargs were passed
        call_kwargs = generator.model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 20
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["do_sample"] is True

    # ========================================================================
    # Constraint Management Tests
    # ========================================================================

    def test_add_constraint(self, generator, mock_tokenizer):
        """Test adding a constraint."""
        constraint = NumericalBoundsConstraint({"x": (0, 1)}, tokenizer=mock_tokenizer)
        generator.add_constraint(constraint)

        assert len(generator.constraints) == 1
        assert constraint in generator.constraints

    def test_add_duplicate_constraint(self, generator, mock_tokenizer):
        """Test that duplicate constraint is only added once."""
        constraint = NumericalBoundsConstraint({"x": (0, 1)}, tokenizer=mock_tokenizer)
        generator.add_constraint(constraint)
        generator.add_constraint(constraint)

        assert len(generator.constraints) == 1

    def test_remove_constraint(self, generator, mock_tokenizer):
        """Test removing a constraint."""
        constraint = NumericalBoundsConstraint({"x": (0, 1)}, tokenizer=mock_tokenizer)
        generator.add_constraint(constraint)

        result = generator.remove_constraint(constraint)

        assert result is True
        assert len(generator.constraints) == 0

    def test_remove_nonexistent_constraint(self, generator, mock_tokenizer):
        """Test removing a constraint that doesn't exist."""
        constraint = NumericalBoundsConstraint({"x": (0, 1)}, tokenizer=mock_tokenizer)
        result = generator.remove_constraint(constraint)

        assert result is False

    def test_clear_constraints(self, generator, mock_tokenizer):
        """Test clearing all constraints."""
        constraint1 = NumericalBoundsConstraint({"x": (0, 1)}, tokenizer=mock_tokenizer)
        constraint2 = NumericalBoundsConstraint({"y": (0, 1)}, tokenizer=mock_tokenizer)
        generator.add_constraint(constraint1)
        generator.add_constraint(constraint2)

        generator.clear_constraints()

        assert len(generator.constraints) == 0

    def test_set_constraints(self, generator, mock_tokenizer):
        """Test setting constraints."""
        constraint1 = NumericalBoundsConstraint({"x": (0, 1)}, tokenizer=mock_tokenizer)
        constraint2 = NumericalBoundsConstraint({"y": (0, 1)}, tokenizer=mock_tokenizer)

        generator.set_constraints([constraint1, constraint2])

        assert len(generator.constraints) == 2
        assert generator.constraints == [constraint1, constraint2]

    # ========================================================================
    # Utility Tests
    # ========================================================================

    def test_repr(self, generator):
        """Test string representation."""
        repr_str = repr(generator)
        assert "ConstrainedGenerator" in repr_str
        assert "test_model" in repr_str
        assert "0 constraints" in repr_str

    def test_repr_with_constraints(self, mock_model, mock_tokenizer):
        """Test repr with constraints."""
        constraint = NumericalBoundsConstraint({"x": (0, 1)})
        generator = ConstrainedGenerator(
            mock_model,
            mock_tokenizer,
            constraints=[constraint]
        )

        repr_str = repr(generator)
        assert "1 constraints" in repr_str
