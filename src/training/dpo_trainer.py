"""Constraint-aware DPO trainer for PI-LLM Alignment.

This module extends TRL's DPOTrainer to incorporate constraint penalties
into the preference learning objective.
"""

from typing import Any

import torch
from transformers import PreTrainedModel

try:
    from trl import DPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    DPOTrainer = object  # type: ignore

from ..constraints.base import LLMConstraint


class ConstraintAwareDPOTrainer(DPOTrainer):
    """DPO trainer with constraint-aware loss modification.

    Extends TRL's DPOTrainer to incorporate constraint penalties
    into the preference learning objective. The model learns to prefer
    responses with lower constraint violations.

    The modified loss is:
    L_total = L_DPO + λ_constraint * L_constraint

    where:
    - L_DPO: Standard DPO preference loss
    - L_constraint: Weighted penalty on chosen response
    - λ_constraint: Balance hyperparameter

    Parameters
    ----------
    model : PreTrainedModel
        Model to train (typically with LoRA adapters).
    ref_model : PreTrainedModel
        Reference model for stable training.
    constraints : list[LLMConstraint]
        Constraints to incorporate into training.
    constraint_weight : float, optional
        Weight for constraint penalty term (λ), by default 0.1.
    **kwargs : Any
        Additional arguments passed to DPOTrainer.

    Raises
    ------
    ImportError
        If TRL library is not available.

    Examples
    --------
    >>> from src.training import ConstraintAwareDPOTrainer
    >>> from src.constraints import NumericalBoundsConstraint
    >>>
    >>> constraints = [NumericalBoundsConstraint({"age": (0, 120)})]
    >>> trainer = ConstraintAwareDPOTrainer(
    ...     model=model,
    ...     ref_model=ref_model,
    ...     train_dataset=dataset,
    ...     constraints=constraints,
    ...     constraint_weight=0.1
    ... )
    >>> trainer.train()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        constraints: list[LLMConstraint],
        constraint_weight: float = 0.1,
        **kwargs: Any
    ) -> None:
        """Initialize the constraint-aware DPO trainer.

        Args:
            model: Model to train.
            ref_model: Reference model.
            constraints: Constraints for training.
            constraint_weight: Weight for constraint penalty.
            **kwargs: Additional DPOTrainer arguments.

        Raises
        ------
        ImportError
            If TRL library is not available.
        """
        if not TRL_AVAILABLE:
            raise ImportError(
                "TRL library required for DPO training. "
                "Install with: pip install trl"
            )

        # Store constraints before parent init
        self.constraints = constraints
        self.constraint_weight = constraint_weight

        # Initialize parent DPOTrainer
        super().__init__(model=model, ref_model=ref_model, **kwargs)

    def compute_constraint_penalty(
        self,
        chosen_texts: list[str]
    ) -> torch.Tensor:
        """Compute constraint penalty for chosen responses.

        Args:
            chosen_texts: List of chosen response texts.

        Returns
        -------
        torch.Tensor
            Tensor of penalty scores, shape (batch_size,).
        """
        penalties = []
        for text in chosen_texts:
            total_penalty = 0.0
            for constraint in self.constraints:
                if constraint.enabled:
                    total_penalty += constraint.get_weighted_penalty(text)
            penalties.append(total_penalty)
        return torch.tensor(
            penalties,
            device=self.model.device,
            dtype=torch.float32
        )

    def compute_loss(
        self,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        """Compute constraint-aware DPO loss.

        Overrides parent compute_loss to add constraint penalty term.

        Returns
        -------
        torch.Tensor
            Total loss: L_DPO + λ_constraint * L_constraint
        """
        # Get standard DPO loss from parent
        dpo_loss = super().compute_loss(*args, **kwargs)

        # Try to extract chosen texts from batch
        # TRL passes different arguments depending on version
        chosen_texts = self._extract_chosen_texts(*args, **kwargs)

        if chosen_texts:
            # Compute constraint penalty
            constraint_penalty = self.compute_constraint_penalty(chosen_texts)
            constraint_loss = constraint_penalty.mean()

            # Weighted sum
            total_loss = dpo_loss + self.constraint_weight * constraint_loss

            # Log for monitoring
            if hasattr(self, "state") and hasattr(self.state, "log_history"):
                self.state.log_history.append({
                    "dpo_loss": dpo_loss.item(),
                    "constraint_loss": constraint_loss.item(),
                    "total_loss": total_loss.item()
                })

            return total_loss

        return dpo_loss

    def _extract_chosen_texts(
        self,
        *args: Any,
        **kwargs: Any
    ) -> list[str] | None:
        """Extract chosen response texts from batch.

        Implementation depends on TRL's internal batch format.
        This is a best-effort extraction.

        Returns
        -------
        list[str] or None
            List of chosen texts if extractable, None otherwise.
        """
        # Try different extraction methods based on TRL version
        # Method 1: Check for 'labels' or 'chosen_input_ids' in kwargs
        if "labels" in kwargs:
            labels = kwargs["labels"]
            if hasattr(labels, "tolist"):
                # Decode token IDs to text
                texts = []
                for label_ids in labels.tolist():
                    # Remove padding and special tokens
                    filtered = [t for t in label_ids if t != -100]
                    text = self.tokenizer.decode(filtered, skip_special_tokens=True)
                    texts.append(text)
                return texts

        # Method 2: Check batch dictionary format
        if args and hasattr(args[0], "__contains__"):
            batch = args[0]
            if "chosen_input_ids" in batch:
                chosen_ids = batch["chosen_input_ids"]
                texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in chosen_ids
                ]
                return texts

        return None


__all__ = ["ConstraintAwareDPOTrainer"]
