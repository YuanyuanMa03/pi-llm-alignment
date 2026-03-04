"""Inference-time constraint integration for PI-LLM Alignment.

This module provides the ConstraintLogitProcessor class that integrates
our constraint system with HuggingFace's generation pipeline.
"""

from typing import Any

import torch

try:
    from transformers import LogitsProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    LogitsProcessor = object  # type: ignore

from ..constraints.base import LLMConstraint


class ConstraintLogitProcessor(LogitsProcessor):
    """Applies constraint masks to model logits during generation.

    This processor integrates with HuggingFace's generate() method
    to enforce constraints at inference time. It modifies the model's
    output logits by masking tokens that would violate constraints.

    Parameters
    ----------
    constraints : list[LLMConstraint]
        List of constraints to enforce during generation.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for decoding token IDs to text.
    mode : str, optional
        Either "mask" (default) to fully forbid violating tokens,
        or "bias" to softly discourage them.

    Attributes
    ----------
    constraints : list[LLMConstraint]
        The constraints to enforce.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for decoding token IDs.

    Examples
    --------
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from src.constraints import NumericalBoundsConstraint
    >>> from src.inference import ConstraintLogitProcessor
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Instruct")
    >>>
    >>> constraint = NumericalBoundsConstraint({"age": (0, 120)}, tokenizer=tokenizer)
    >>> processor = ConstraintLogitProcessor([constraint], tokenizer)
    >>>
    >>> output = model.generate(
    ...     "The person's age is ",
    ...     logits_processor=processor,
    ...     max_new_tokens=10
    ... )
    """

    def __init__(
        self,
        constraints: list[LLMConstraint],
        tokenizer: Any,
        mode: str = "mask"
    ) -> None:
        """Initialize the constraint logits processor.

        Args:
            constraints: List of constraints to enforce.
            tokenizer: Tokenizer for decoding token IDs.
            mode: Either "mask" or "bias".

        Raises
        ------
        ValueError
            If mode is not "mask" or "bias", or if no constraints provided.
        """
        if not constraints:
            raise ValueError("At least one constraint must be provided")

        if mode not in ("mask", "bias"):
            raise ValueError(f"Mode must be 'mask' or 'bias', got {mode!r}")

        self.constraints: list[LLMConstraint] = constraints
        self.tokenizer: Any = tokenizer
        self.mode: str = mode

    def __call__(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply constraint masks to logits.

        This method is called by HuggingFace's generate() at each step.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            logits: Model logits of shape (batch_size, vocab_size).

        Returns
        -------
        torch.FloatTensor
            Modified logits with constraint masks applied.

        Examples
        --------
        >>> processor = ConstraintLogitProcessor([constraint], tokenizer)
        >>> input_ids = torch.tensor([[1, 2, 3]])
        >>> logits = torch.randn(1, 100000)
        >>> masked_logits = processor(input_ids, logits)
        """
        # Handle batch dimension
        batch_size = input_ids.shape[0]
        vocab_size = logits.shape[-1]

        # Process each sample in the batch
        for i in range(batch_size):
            current_sequence = input_ids[i].tolist()

            # Get the device of the logits for this sample
            logits_device = logits[i].device

            # Apply each constraint's mask
            for constraint in self.constraints:
                if not constraint.enabled:
                    continue

                mask = constraint.get_logits_mask(current_sequence, vocab_size)

                # Move mask to the same device as logits
                mask = mask.to(logits_device)

                if self.mode == "mask":
                    # Set masked tokens to -inf (fully forbidden)
                    logits[i] = logits[i].masked_fill(mask, -float('inf'))
                else:  # mode == "bias"
                    # Add negative bias (soft discouragement)
                    bias = mask.float() * -10.0
                    logits[i] = logits[i] + bias

        return logits

    def __repr__(self) -> str:
        """Return string representation."""
        enabled_count = sum(1 for c in self.constraints if c.enabled)
        return (
            f"ConstraintLogitProcessor("
            f"{len(self.constraints)} constraints, "
            f"{enabled_count} enabled, mode='{self.mode}')"
        )


class ConstrainedGenerator:
    """High-level interface for constrained text generation.

    This class wraps a model and tokenizer to provide a simple
    interface for generating text with constraints.

    Parameters
    ----------
    model : PreTrainedModel
        The language model to use for generation.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for the model.
    constraints : list[LLMConstraint], optional
        Constraints to enforce during generation.

    Examples
    --------
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from src.constraints import NumericalBoundsConstraint
    >>> from src.inference import ConstrainedGenerator
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Instruct")
    >>>
    >>> constraint = NumericalBoundsConstraint({"value": (0, 100)})
    >>> generator = ConstrainedGenerator(model, tokenizer, [constraint])
    >>>
    >>> output = generator.generate("The value is ", max_new_tokens=20)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        constraints: list[LLMConstraint] | None = None,
        default_generation_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Initialize the constrained generator.

        Args:
            model: The language model.
            tokenizer: Tokenizer for the model.
            constraints: List of constraints to enforce.
            default_generation_kwargs: Default kwargs for generation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.constraints = constraints or []
        self.default_generation_kwargs = default_generation_kwargs or {}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> str:
        """Generate text with constraints enforced.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum number of tokens to generate.
            **generation_kwargs: Additional arguments for model.generate().

        Returns
        -------
        str
            Generated text (including prompt).
        """
        # Prepare inputs
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.model.device)

        # Create logits processor if constraints exist
        logits_processor = None
        if self.constraints:
            logits_processor = ConstraintLogitProcessor(
                self.constraints,
                self.tokenizer
            )

        # Merge with default kwargs
        kwargs = {
            "max_new_tokens": max_new_tokens,
            **self.default_generation_kwargs,
            **generation_kwargs
        }

        if logits_processor is not None:
            kwargs["logits_processor"] = [logits_processor]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def add_constraint(self, constraint: LLMConstraint) -> None:
        """Add a constraint to the generator.

        Args:
            constraint: Constraint to add.
        """
        if constraint not in self.constraints:
            self.constraints.append(constraint)

    def remove_constraint(self, constraint: LLMConstraint) -> bool:
        """Remove a constraint from the generator.

        Args:
            constraint: Constraint to remove.

        Returns
        -------
        bool
            True if constraint was removed, False if not found.
        """
        try:
            self.constraints.remove(constraint)
            return True
        except ValueError:
            return False

    def clear_constraints(self) -> None:
        """Remove all constraints."""
        self.constraints.clear()

    def set_constraints(self, constraints: list[LLMConstraint]) -> None:
        """Set the list of constraints.

        Args:
            constraints: New list of constraints.
        """
        self.constraints = list(constraints)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConstrainedGenerator("
            f"model={self.model.config.model_type}, "
            f"{len(self.constraints)} constraints)"
        )


__all__ = [
    "ConstraintLogitProcessor",
    "ConstrainedGenerator",
]
