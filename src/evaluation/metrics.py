"""Constraint-aware evaluation metrics for PI-LLM Alignment.

This module provides evaluation utilities to measure constraint satisfaction
and generation quality for trained models.
"""

from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..constraints.base import LLMConstraint


class ConstraintAwareEvaluator:
    """Evaluate models with constraint-aware metrics.

    Computes metrics related to constraint satisfaction and
    generation quality for trained models.

    Parameters
    ----------
    model : PreTrainedModel
        Model to evaluate.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for the model.
    constraints : list[LLMConstraint]
        Constraints to evaluate against.

    Examples
    --------
    >>> from src.evaluation import ConstraintAwareEvaluator
    >>> from src.constraints import NumericalBoundsConstraint
    >>>
    >>> constraint = NumericalBoundsConstraint({"age": (0, 120)})
    >>> evaluator = ConstraintAwareEvaluator(model, tokenizer, [constraint])
    >>> metrics = evaluator.evaluate(
    ...     prompts=["The person's age is "],
    ...     generation_kwargs={"max_new_tokens": 50}
    ... )
    >>> print(metrics["violation_rate"])
    {'NumericalBoundsConstraint': 0.15}
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        constraints: list[LLMConstraint]
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: Model to evaluate.
            tokenizer: Model tokenizer.
            constraints: Constraints to evaluate against.
        """
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.constraints = constraints

    def generate_responses(
        self,
        prompts: list[str],
        generation_kwargs: dict[str, Any] | None = None
    ) -> list[str]:
        """Generate responses for evaluation prompts.

        Args:
            prompts: List of input prompts.
            generation_kwargs: Generation parameters.

        Returns
        -------
        list[str]
            Generated responses.
        """
        if generation_kwargs is None:
            generation_kwargs = {}

        generation_kwargs.setdefault("max_new_tokens", 50)
        generation_kwargs.setdefault("do_sample", True)
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        responses = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    return_attention_mask=True
                ).to(self.model.device)

                outputs = self.model.generate(**inputs, **generation_kwargs)

                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_text[len(prompt):].strip()
                responses.append(response)

        return responses

    def compute_violation_rate(
        self,
        texts: list[str]
    ) -> dict[str, float]:
        """Compute per-constraint violation rates.

        Args:
            texts: Generated texts to evaluate.

        Returns
        -------
        dict[str, float]
            Dictionary mapping constraint names to violation rates.
        """
        if not texts:
            return {}

        rates = {}
        for constraint in self.constraints:
            if not constraint.enabled:
                continue
            violations = sum(1 for text in texts if not constraint.check(text))
            rates[constraint.__class__.__name__] = violations / len(texts)
        return rates

    def compute_average_penalty(
        self,
        texts: list[str]
    ) -> dict[str, float]:
        """Compute average penalty per constraint.

        Args:
            texts: Generated texts to evaluate.

        Returns
        -------
        dict[str, float]
            Dictionary mapping constraint names to average penalties.
        """
        if not texts:
            return {}

        penalties = {}
        for constraint in self.constraints:
            if not constraint.enabled:
                continue
            total = sum(constraint.compute_penalty(text) for text in texts)
            penalties[constraint.__class__.__name__] = total / len(texts)
        return penalties

    def _compute_overall_violation_rate(
        self,
        texts: list[str]
    ) -> float:
        """Compute overall violation rate across all constraints.

        A text violates if ANY constraint fails.

        Args:
            texts: Generated texts to evaluate.

        Returns
        -------
        float
            Overall violation rate.
        """
        if not texts:
            return 0.0

        enabled_constraints = [c for c in self.constraints if c.enabled]
        if not enabled_constraints:
            return 0.0

        violations = 0
        for text in texts:
            if any(not c.check(text) for c in enabled_constraints):
                violations += 1

        return violations / len(texts)

    def _compute_length_stats(
        self,
        texts: list[str]
    ) -> dict[str, float]:
        """Compute response length statistics.

        Args:
            texts: Generated texts.

        Returns
        -------
        dict[str, float]
            Length statistics.
        """
        if not texts:
            return {}

        lengths = [len(t.split()) for t in texts]
        return {
            "mean": sum(lengths) / len(lengths) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0
        }

    def evaluate(
        self,
        prompts: list[str],
        generation_kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run complete evaluation.

        Args:
            prompts: Evaluation prompts.
            generation_kwargs: Generation parameters.

        Returns
        -------
        dict[str, Any]
            Comprehensive evaluation metrics including:
            - violation_rate: Per-constraint violation rates
            - average_penalty: Per-constraint average penalties
            - overall_violation_rate: Aggregate violation rate
            - response_length: Statistics on response length
            - num_responses: Number of responses evaluated
        """
        responses = self.generate_responses(prompts, generation_kwargs)

        return {
            "violation_rate": self.compute_violation_rate(responses),
            "average_penalty": self.compute_average_penalty(responses),
            "overall_violation_rate": self._compute_overall_violation_rate(responses),
            "response_length": self._compute_length_stats(responses),
            "num_responses": len(responses)
        }

    def compare_models(
        self,
        before_model: PreTrainedModel,
        after_model: PreTrainedModel,
        prompts: list[str],
        generation_kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compare metrics before and after training.

        Args:
            before_model: Model before training.
            after_model: Model after training.
            prompts: Evaluation prompts.
            generation_kwargs: Generation parameters.

        Returns
        -------
        dict[str, Any]
            Comparison metrics including improvement percentages.
        """
        # Evaluate before model
        evaluator_before = ConstraintAwareEvaluator(
            before_model, self.tokenizer, self.constraints
        )
        metrics_before = evaluator_before.evaluate(prompts, generation_kwargs)

        # Evaluate after model
        evaluator_after = ConstraintAwareEvaluator(
            after_model, self.tokenizer, self.constraints
        )
        metrics_after = evaluator_after.evaluate(prompts, generation_kwargs)

        # Compute improvements
        before_violation = metrics_before["overall_violation_rate"]
        after_violation = metrics_after["overall_violation_rate"]

        improvement = 0.0
        if before_violation > 0:
            improvement = (before_violation - after_violation) / before_violation

        return {
            "before": metrics_before,
            "after": metrics_after,
            "improvement": {
                "violation_rate_reduction": improvement,
                "absolute_reduction": before_violation - after_violation
            }
        }


__all__ = ["ConstraintAwareEvaluator"]
