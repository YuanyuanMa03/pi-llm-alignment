"""Evaluation modules for PI-LLM Alignment.

This package provides constraint-aware evaluation utilities including:
- ConstraintAwareEvaluator: Evaluate models with constraint metrics
"""

from .metrics import ConstraintAwareEvaluator

__all__ = ["ConstraintAwareEvaluator"]
