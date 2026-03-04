"""Constraint registry for managing constraint classes.

This module provides a global registry for constraint classes,
enabling dynamic instantiation and configuration-based setup.
"""

from typing import Any

from .base import LLMConstraint, get_global_registry
from .factual import NumericalBoundsConstraint, KeywordPresenceConstraint


def register_default_constraints() -> None:
    """Register all default constraint classes.

    This function is called automatically on module import
    to populate the global registry with built-in constraints.
    """
    registry = get_global_registry()

    # Register factual constraints
    registry.register("numerical_bounds", NumericalBoundsConstraint)
    registry.register("keyword_presence", KeywordPresenceConstraint)


# Auto-register default constraints on import
register_default_constraints()


def create_constraint(name: str, **kwargs: Any) -> LLMConstraint:
    """Create a constraint instance by name.

    This is a convenience function that uses the global registry.

    Args:
        name: Name of the constraint class to instantiate.
        **kwargs: Arguments to pass to the constraint constructor.

    Returns
    -------
    LLMConstraint
        Instantiated constraint object.

    Raises
    ------
    KeyError
        If no constraint with this name is registered.

    Examples
    --------
    >>> from src.constraints.registry import create_constraint
    >>> constraint = create_constraint(
    ...     "numerical_bounds",
    ...     bounds={"value": (0, 100)},
    ...     weight=1.0
    ... )
    """
    return get_global_registry().create(name, **kwargs)


def list_available_constraints() -> list[str]:
    """List names of all available constraints.

    Returns
    -------
    list of str
        Names of registered constraints.
    """
    return get_global_registry().list_available_constraints()


__all__ = [
    "register_default_constraints",
    "create_constraint",
    "list_available_constraints",
]
