"""PI-LLM Alignment: Physics-Informed Constraints for LLM Hallucination Reduction.

This package implements dual-mode constraints that support both
post-generation evaluation and inference-time intervention.
"""

__version__ = "0.1.0-alpha"

from src.constraints.base import (
    LLMConstraint,
    ConstraintRegistry,
    get_global_registry,
)

from src.constraints.registry import (
    create_constraint,
    list_available_constraints,
)

__all__ = [
    "LLMConstraint",
    "ConstraintRegistry",
    "get_global_registry",
    "create_constraint",
    "list_available_constraints",
]
