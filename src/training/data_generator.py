"""Preference pair data generation for constraint-aware DPO.

This module provides utilities for generating DPO training data using
the existing constraint system to score and rank model responses.
"""

from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..constraints.base import LLMConstraint


class PreferencePairGenerator:
    """Generate (prompt, chosen, rejected) triples for DPO training.

    This class generates multiple diverse responses per prompt and uses
    the constraint system to score them, creating preference pairs where
    'chosen' has lower constraint penalty than 'rejected'.

    Parameters
    ----------
    model : PreTrainedModel
        Base language model for generation.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for the model.
    constraints : list[LLMConstraint]
        List of constraints to score responses with.

    Examples
    --------
    >>> from src.constraints import NumericalBoundsConstraint
    >>> from src.training import PreferencePairGenerator
    >>>
    >>> constraint = NumericalBoundsConstraint({"age": (0, 120)})
    >>> generator = PreferencePairGenerator(model, tokenizer, [constraint])
    >>>
    >>> pairs = generator.generate_pairs(
    ...     prompts=["The person's age is "],
    ...     n_samples=8,
    ...     generation_kwargs={"temperature": 0.8, "max_new_tokens": 50}
    ... )
    >>> print(pairs[0])
    {
        "prompt": "The person's age is ",
        "chosen": "25 years old",
        "rejected": "250 years old",
        "chosen_score": 0.0,
        "rejected_score": 1.08
    }
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        constraints: list[LLMConstraint]
    ) -> None:
        """Initialize the generator.

        Args:
            model: Base language model.
            tokenizer: Model tokenizer.
            constraints: Constraints for scoring.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.constraints = constraints
        self.model.eval()

    def generate_responses(
        self,
        prompt: str,
        n_samples: int,
        generation_kwargs: dict[str, Any] | None = None
    ) -> list[str]:
        """Generate diverse responses for a single prompt.

        Uses different sampling strategies to generate diverse responses
        for preference pair creation.

        Args:
            prompt: Input prompt.
            n_samples: Number of responses to generate.
            generation_kwargs: Generation parameters.

        Returns:
            List of generated responses.
        """
        if generation_kwargs is None:
            generation_kwargs = {}

        # Prepare inputs
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True
        )
        # Move to device if it's a tensor-like object
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)

        responses: list[str] = []

        # Generate multiple responses with varying parameters
        base_kwargs = {
            "max_new_tokens": generation_kwargs.get("max_new_tokens", 50),
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Generate with different sampling strategies
        strategies = [
            {"temperature": 0.7, "top_p": 0.9},
            {"temperature": 0.9, "top_p": 0.95},
            {"temperature": 1.0, "top_p": 0.8},
            {"temperature": 0.8, "top_k": 50},
        ]

        samples_per_strategy = max(1, n_samples // len(strategies))

        with torch.no_grad():
            for strategy in strategies:
                for _ in range(samples_per_strategy):
                    if len(responses) >= n_samples:
                        break

                    kwargs = {**base_kwargs, **strategy, **generation_kwargs}

                    outputs = self.model.generate(
                        **inputs,
                        **kwargs
                    )

                    # Decode and remove prompt
                    full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = full_text[len(prompt):].strip()
                    responses.append(response)

        # Fill remaining if needed
        while len(responses) < n_samples:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **base_kwargs)
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_text[len(prompt):].strip()
                responses.append(response)

        return responses[:n_samples]

    def score_response(self, response: str) -> float:
        """Compute total constraint penalty for a response.

        Args:
            response: Generated response to score.

        Returns:
            Total weighted penalty across all constraints.
        """
        total_penalty = 0.0
        for constraint in self.constraints:
            if constraint.enabled:
                total_penalty += constraint.get_weighted_penalty(response)
        return total_penalty

    def create_preference_pair(
        self,
        prompt: str,
        responses: list[str],
        scores: list[float],
        min_margin: float = 0.1
    ) -> dict[str, Any] | None:
        """Create a single preference pair from scored responses.

        Selects the lowest-scoring response as 'chosen' and finds a
        higher-scoring 'rejected' response with sufficient margin.

        Args:
            prompt: Original prompt.
            responses: List of generated responses.
            scores: Corresponding penalty scores.
            min_margin: Minimum score margin for meaningful learning.

        Returns:
            Dictionary with 'prompt', 'chosen', 'rejected', 'chosen_score',
            'rejected_score' keys, or None if no valid pair exists.
        """
        if len(responses) < 2:
            return None

        # Sort by score (ascending - lower is better)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])

        chosen_idx = sorted_indices[0]
        chosen_score = scores[chosen_idx]

        # Find rejected with sufficient margin
        for idx in sorted_indices[1:]:
            if scores[idx] - chosen_score >= min_margin:
                return {
                    "prompt": prompt,
                    "chosen": responses[chosen_idx],
                    "rejected": responses[idx],
                    "chosen_score": chosen_score,
                    "rejected_score": scores[idx],
                    "margin": scores[idx] - chosen_score
                }

        # If no sufficient margin, still return the best pair
        # but with a warning flag
        if len(sorted_indices) > 1:
            return {
                "prompt": prompt,
                "chosen": responses[chosen_idx],
                "rejected": responses[sorted_indices[1]],
                "chosen_score": chosen_score,
                "rejected_score": scores[sorted_indices[1]],
                "margin": scores[sorted_indices[1]] - chosen_score,
                "low_margin": True
            }

        return None

    def generate_pairs(
        self,
        prompts: list[str],
        n_samples: int = 8,
        generation_kwargs: dict[str, Any] | None = None,
        min_margin: float = 0.1
    ) -> list[dict[str, Any]]:
        """Generate preference pairs for multiple prompts.

        Args:
            prompts: List of input prompts.
            n_samples: Number of responses to generate per prompt.
            generation_kwargs: Generation parameters.
            min_margin: Minimum score margin for pairs.

        Returns:
            List of preference triple dictionaries.
        """
        pairs = []

        for prompt in prompts:
            # Generate responses
            responses = self.generate_responses(prompt, n_samples, generation_kwargs)

            # Score responses
            scores = [self.score_response(r) for r in responses]

            # Create preference pair
            pair = self.create_preference_pair(prompt, responses, scores, min_margin)
            if pair is not None:
                pairs.append(pair)

        return pairs

    def generate_dataset(
        self,
        prompts: list[str],
        n_samples_per_prompt: int = 8,
        generation_kwargs: dict[str, Any] | None = None,
        min_margin: float = 0.1
    ) -> list[dict[str, str]]:
        """Generate complete DPO dataset.

        Returns dataset in format expected by TRL's DPOTrainer.

        Args:
            prompts: List of input prompts.
            n_samples_per_prompt: Responses to generate per prompt.
            generation_kwargs: Generation parameters.
            min_margin: Minimum score margin for pairs.

        Returns:
            List of dictionaries with 'prompt', 'chosen', 'rejected' keys.
        """
        pairs = self.generate_pairs(
            prompts=prompts,
            n_samples=n_samples_per_prompt,
            generation_kwargs=generation_kwargs,
            min_margin=min_margin
        )

        # Convert to TRL format
        dataset = []
        for pair in pairs:
            dataset.append({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"]
            })

        return dataset


__all__ = ["PreferencePairGenerator"]
