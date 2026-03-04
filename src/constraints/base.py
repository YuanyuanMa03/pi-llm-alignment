"""Base constraint classes for PI-LLM Alignment.

This module provides the abstract base class and protocol definitions
for dual-mode constraints that support both evaluation and intervention.

The design follows a dual-mode approach:
1. Evaluation Mode: Post-generation scoring via check() and compute_penalty()
2. Intervention Mode: Inference-time control via get_logits_mask()
"""

from abc import ABC, abstractmethod
from typing import Any
import torch


class LLMConstraint(ABC):
    """Abstract base class for dual-mode LLM constraints.

    A constraint must support both:
    1. Evaluation mode: Post-generation penalty scoring for DPO/RLAIF
    2. Intervention mode: Inference-time logit manipulation

    This dual-mode design enables constraints to be used for
    both training (via penalty scores as rewards) and inference
    (via logit masks for guided generation).

    Attributes
    ----------
    weight : float
        Weight multiplier for penalty scoring (evaluation mode).
        Higher weights increase the penalty severity for violations.
    enabled : bool
        Whether the constraint is currently active. When False,
        all methods return neutral values (no violation, zero penalty,
        empty mask).

    Examples
    --------
    >>> from src.constraints.factual import NumericalBoundsConstraint
    >>> constraint = NumericalBoundsConstraint({"value": (0, 100)})
    >>> constraint.check("The value is 50")
    True
    >>> constraint.compute_penalty("The value is 150")
    0.5
    """

    def __init__(self, weight: float = 1.0, enabled: bool = True) -> None:
        """Initialize the constraint.

        Args:
            weight: Weight multiplier for penalty scoring.
                    Must be non-negative. Applied to the output
                    of compute_penalty() for final penalty value.
            enabled: Whether the constraint is active. When False,
                     all methods return neutral values regardless of input.

        Raises
        ------
        ValueError
            If weight is negative.
        """
        if weight < 0:
            raise ValueError(
                f"Constraint weight must be non-negative, got {weight!r}"
            )
        self.weight: float = weight
        self.enabled: bool = enabled

    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def check(self, text: str) -> bool:
        """Perform a hard constraint check on generated text.

        This method evaluates whether the given text satisfies the
        constraint according to a binary (True/False) criterion.

        Args:
            text: The generated text to evaluate.

        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise.

        Examples
        --------
        >>> constraint = NumericalBoundsConstraint({"age": (0, 120)})
        >>> constraint.check("The person is 25 years old")
        True
        >>> constraint.check("The person is 150 years old")
        False
        """
        ...

    @abstractmethod
    def compute_penalty(self, text: str) -> float:
        """Compute penalty for constraint violation (for DPO/RLAIF reward modeling).

        This method returns a continuous penalty score that indicates
        the severity of constraint violation. Used for:
        - DPO (Direct Preference Optimization) as reward signal
        - RLAIF (AI Feedback) as automated preference scorer
        - Training loss computation

        The penalty should follow this convention:
        - 0.0 if perfectly aligned (no violation)
        - Higher values for worse violations
        - Typically normalized to [0, inf) range

        Args:
            text: The generated text to evaluate.

        Returns
        -------
        float
            Non-negative penalty score. Returns 0.0 if the constraint
            is perfectly satisfied. Higher values indicate more severe
            violations. The value will be multiplied by self.weight
            for the final penalty when used in training.

        Examples
        --------
        >>> constraint = NumericalBoundsConstraint({"value": (0, 100)})
        >>> constraint.compute_penalty("The value is 50")  # Within bounds
        0.0
        >>> constraint.compute_penalty("The value is 150")  # 50% over max
        0.5
        """
        ...

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """Compute logits mask for inference-time intervention.

        This method returns a boolean mask indicating which tokens
        should be forbidden (masked) at the next generation step.
        Tokens marked True will have their logits set to -inf,
        effectively preventing their selection during sampling.

        This is the key method for constrained decoding: it allows
        the constraint to guide generation in real-time without
        requiring model retraining.

        Default implementation returns an all-False mask (no masking).
        Subclasses should override this to implement constraint-specific
        token filtering logic.

        Args:
            current_sequence: The token sequence generated so far,
                              represented as a list of token IDs.
            vocab_size: The size of the vocabulary. Determines the
                        shape of the output tensor.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (vocab_size,). Tokens marked True
            will be masked (logit set to -inf). Returns all-zeros
            (no masking) if constraint is disabled.

        Examples
        --------
        >>> constraint = NumericalBoundsConstraint({"value": (0, 100)})
        >>> sequence = [1, 2, 3]  # Token IDs
        >>> mask = constraint.get_logits_mask(sequence, vocab_size=1000)
        >>> mask.shape
        torch.Size([1000])
        >>> mask.dtype
        torch.bool
        """
        if not self.enabled:
            return torch.zeros(vocab_size, dtype=torch.bool)

        # Default stub implementation: no tokens masked
        # Subclasses should override this with actual constraint logic
        return torch.zeros(vocab_size, dtype=torch.bool)

    # ========================================================================
    # CONCRETE METHODS - Shared implementation
    # ========================================================================

    def get_weighted_penalty(self, text: str) -> float:
        """Get the weighted penalty for the given text.

        This is a convenience method that applies the constraint's
        weight to the computed penalty.

        Args:
            text: The generated text to evaluate.

        Returns
        -------
        float
            The weighted penalty (compute_penalty() * self.weight).
            Returns 0.0 if constraint is disabled.
        """
        if not self.enabled:
            return 0.0
        return self.weight * self.compute_penalty(text)

    def apply_mask_to_logits(
        self,
        logits: torch.Tensor,
        current_sequence: list[int]
    ) -> torch.Tensor:
        """Apply the constraint mask to model logits.

        This method modifies the logits tensor by setting masked
        token positions to negative infinity, preventing their
        selection during sampling.

        Args:
            logits: Model logits of shape (batch_size, vocab_size)
                    or (vocab_size,).
            current_sequence: The token sequence generated so far.

        Returns
        -------
        torch.Tensor
            Modified logits with the same shape as input. Tokens
            marked in the mask have their logits set to -inf.
            Returns unchanged logits if constraint is disabled.

        Examples
        --------
        >>> import torch
        >>> logits = torch.randn(1, 1000)
        >>> sequence = [1, 2, 3]
        >>> masked_logits = constraint.apply_mask_to_logits(logits, sequence)
        >>> masked_logits.shape
        torch.Size([1, 1000])
        """
        if not self.enabled:
            return logits

        vocab_size = logits.shape[-1]
        mask = self.get_logits_mask(current_sequence, vocab_size)

        # Apply mask: set masked positions to -inf
        return logits.masked_fill(mask, -float('inf'))

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def __repr__(self) -> str:
        """Return string representation of the constraint.

        Returns
        -------
        str
            String showing constraint class, weight, and enabled status.
        """
        class_name = self.__class__.__name__
        return f"{class_name}(weight={self.weight}, enabled={self.enabled})"

    def __str__(self) -> str:
        """Return user-friendly string representation.

        Returns
        -------
        str
            Human-readable description of the constraint.
        """
        status = "enabled" if self.enabled else "disabled"
        return f"{self.__class__.__name__} [{status}, weight={self.weight}]"


class ConstraintRegistry:
    """Registry for managing constraint classes.

    This registry allows dynamic constraint instantiation by name,
    which is useful for configuration-based experiment setup.

    Supports registration and creation of constraint classes with
    arbitrary initialization parameters.

    Attributes
    ----------
    _constraints : dict[str, type[LLMConstraint]]
        Mapping from constraint names to their classes.

    Examples
    --------
    >>> registry = ConstraintRegistry()
    >>> registry.register("numerical_bounds", NumericalBoundsConstraint)
    >>> constraint = registry.create(
    ...     "numerical_bounds",
    ...     bounds={"value": (0, 100)},
    ...     weight=1.0
    ... )
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._constraints: dict[str, type[LLMConstraint]] = {}

    def register(
        self,
        name: str,
        constraint_class: type[LLMConstraint]
    ) -> None:
        """Register a constraint class.

        Args:
            name: Name to register the constraint under. This name
                  will be used to retrieve/create instances.
            constraint_class: The constraint class to register.
                             Must be a subclass of LLMConstraint.

        Raises
        ------
        ValueError
            If a constraint with this name already exists.
        TypeError
            If constraint_class is not a subclass of LLMConstraint.
        """
        if name in self._constraints:
            raise ValueError(
                f"Constraint '{name}' already registered. "
                f"Use a different name or unregister first."
            )

        if not issubclass(constraint_class, LLMConstraint):
            raise TypeError(
                f"Constraint class must be a subclass of LLMConstraint, "
                f"got {constraint_class!r}"
            )

        self._constraints[name] = constraint_class

    def unregister(self, name: str) -> None:
        """Remove a constraint from the registry.

        Args:
            name: Name of the constraint to unregister.

        Raises
        ------
        KeyError
            If no constraint with this name is registered.
        """
        if name not in self._constraints:
            available = ", ".join(self._constraints.keys())
            raise KeyError(
                f"Cannot unregister '{name}'. "
                f"Available constraints: {available}"
            )
        del self._constraints[name]

    def create(self, name: str, **kwargs: Any) -> LLMConstraint:
        """Create a constraint instance by name.

        Args:
            name: Name of the registered constraint.
            **kwargs: Arguments to pass to the constraint constructor.

        Returns
        -------
        LLMConstraint
            Instantiated constraint object.

        Raises
        ------
        KeyError
            If no constraint with this name is registered.
        """
        if name not in self._constraints:
            available = ", ".join(self._constraints.keys())
            raise KeyError(
                f"Unknown constraint '{name}'. "
                f"Available: {available}"
            )
        return self._constraints[name](**kwargs)

    def list_available(self) -> list[str]:
        """List names of all registered constraints.

        Returns
        -------
        list[str]
            Names of registered constraints in alphabetical order.
        """
        return sorted(self._constraints.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a constraint name is registered.

        Args:
            name: Constraint name to check.

        Returns
        -------
        bool
            True if constraint is registered, False otherwise.
        """
        return name in self._constraints

    def __len__(self) -> int:
        """Return the number of registered constraints.

        Returns
        -------
        int
            Number of constraints in the registry.
        """
        return len(self._constraints)

    def __repr__(self) -> str:
        """Return string representation of the registry.

        Returns
        -------
        str
            String showing registered constraint count.
        """
        return f"ConstraintRegistry({len(self)} constraints)"


# ============================================================================
# GLOBAL REGISTRY
# ============================================================================

_global_registry: ConstraintRegistry | None = None


def get_global_registry() -> ConstraintRegistry:
    """Get the global constraint registry.

    Creates a new registry instance on first call and returns
    the same instance on subsequent calls (singleton pattern).

    Returns
    -------
    ConstraintRegistry
        The global registry instance.

    Examples
    --------
    >>> registry = get_global_registry()
    >>> from src.constraints.factual import NumericalBoundsConstraint
    >>> registry.register("numerical_bounds", NumericalBoundsConstraint)
    >>> constraint = registry.create("numerical_bounds", bounds={"x": (0, 1)})
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ConstraintRegistry()
    return _global_registry


def create_constraint(name: str, **kwargs: Any) -> LLMConstraint:
    """Create a constraint using the global registry.

    Convenience function that forwards to get_global_registry().create().

    Args:
        name: Name of the registered constraint.
        **kwargs: Arguments for the constraint constructor.

    Returns
    -------
    LLMConstraint
        Instantiated constraint object.

    Raises
    ------
    KeyError
        If no constraint with this name is registered.
    """
    return get_global_registry().create(name, **kwargs)


def list_constraints() -> list[str]:
    """List all registered constraint names.

    Convenience function that forwards to get_global_registry().list_available().

    Returns
    -------
    list[str]
        Names of registered constraints.
    """
    return get_global_registry().list_available()


__all__ = [
    "LLMConstraint",
    "ConstraintRegistry",
    "get_global_registry",
    "create_constraint",
    "list_constraints",
]
