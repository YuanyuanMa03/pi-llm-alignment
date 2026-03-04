"""Factual constraint implementations for PI-LLM Alignment.

This module provides constraints that enforce factual correctness,
such as numerical bounds and known entity properties.
"""

import re
from typing import Any
import torch

try:
    from transformers import PreTrainedTokenizerBase
except ImportError:
    PreTrainedTokenizerBase = Any  # type: ignore

from .base import LLMConstraint


class NumericalBoundsConstraint(LLMConstraint):
    """Constraint ensuring numerical values stay within specified bounds.

    This is a "physics-like" constraint where variables must satisfy
    explicit bounds (e.g., height must be positive and under 3 meters).

    The constraint extracts all numerical values from the text and
    checks if any fall outside the specified bounds.

    Parameters
    ----------
    bounds : dict[str, tuple[float, float]]
        Dictionary mapping variable names to (min, max) tuples.
        All numbers in the text are checked against all bounds.
    weight : float, optional
        Weight multiplier for penalty scoring, by default 1.0.
    enabled : bool, optional
        Whether the constraint is active, by default True.
    tokenizer : PreTrainedTokenizerBase, optional
        Tokenizer for decoding token IDs. Required for get_logits_mask()
        to work properly during inference.

    Attributes
    ----------
    bounds : dict[str, tuple[float, float]]
        The numerical bounds to enforce.
    tokenizer : PreTrainedTokenizerBase or None
        Tokenizer for decoding token IDs during inference.

    Examples
    --------
    >>> constraint = NumericalBoundsConstraint({"height_cm": (0, 300)})
    >>> constraint.check("Height: 175 cm")
    True
    >>> constraint.check("Height: 400 cm")
    False
    >>> constraint.compute_penalty("Height: 400 cm")
    0.333...  # (400 - 300) / 300
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        weight: float = 1.0,
        enabled: bool = True,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """Initialize the numerical bounds constraint.

        Args:
            bounds: Dictionary mapping variable names to (min, max) tuples.
            weight: Penalty weight multiplier.
            enabled: Whether constraint is active.
            tokenizer: Tokenizer for decoding token IDs during inference.

        Raises
        ------
        ValueError
            If any bound has min > max, or if weight is negative.
        """
        super().__init__(weight, enabled)

        # Validate bounds
        for name, (min_val, max_val) in bounds.items():
            if min_val > max_val:
                raise ValueError(
                    f"Invalid bounds for '{name}': "
                    f"min ({min_val}) > max ({max_val})"
                )

        self.bounds: dict[str, tuple[float, float]] = bounds
        self.tokenizer: PreTrainedTokenizerBase | None = tokenizer

    def check(self, text: str) -> bool:
        """Check if all numbers in text are within bounds.

        Args:
            text: Generated text to evaluate.

        Returns
        -------
        bool
            True if all numbers are within bounds, False otherwise.
        """
        numbers = self._extract_numbers(text)

        for num in numbers:
            for (min_val, max_val) in self.bounds.values():
                if not (min_val <= num <= max_val):
                    return False

        return True

    def compute_penalty(self, text: str) -> float:
        """Compute penalty based on violation severity.

        Penalty is computed as the normalized distance outside bounds.
        For example, if bounds are (0, 100) and value is 150:
        penalty = (150 - 100) / 100 = 0.5

        Args:
            text: Generated text to evaluate.

        Returns
        -------
        float
            Non-negative penalty (0 = no violation).
        """
        if not self.enabled:
            return 0.0

        numbers = self._extract_numbers(text)
        total_penalty = 0.0

        for num in numbers:
            for (min_val, max_val) in self.bounds.values():
                if num < min_val:
                    # Normalized penalty for being below min
                    total_penalty += (min_val - num) / max(abs(min_val), 1.0)
                elif num > max_val:
                    # Normalized penalty for being above max
                    total_penalty += (num - max_val) / max(abs(max_val), 1.0)

        return total_penalty

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """Compute mask for tokens that would create out-of-bounds numbers.

        This method checks each token to see if continuing the current
        sequence with that token would create a number outside bounds.

        Note: Requires tokenizer to be set. If no tokenizer is provided,
        returns an empty mask (no masking).

        Args:
            current_sequence: Token IDs generated so far.
            vocab_size: Size of the vocabulary.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (vocab_size,). True indicates
            the token should be masked.
        """
        if not self.enabled:
            return torch.zeros(vocab_size, dtype=torch.bool)

        # If no tokenizer, cannot perform token-level masking
        if self.tokenizer is None:
            return torch.zeros(vocab_size, dtype=torch.bool)

        # Decode current sequence to get the prefix text
        prefix = self.tokenizer.decode(current_sequence, skip_special_tokens=True)

        # Find partial number at end of prefix
        partial = self._extract_partial_number(prefix)

        if partial is None:
            # No number in progress, no tokens to mask
            return torch.zeros(vocab_size, dtype=torch.bool)

        # Create mask tensor
        mask = torch.zeros(vocab_size, dtype=torch.bool)

        # Check each token ID to see if it would continue to an out-of-bounds number
        for token_id in range(vocab_size):
            # Decode single token
            token = self.tokenizer.decode([token_id], skip_special_tokens=True)

            # Skip tokens that don't look like number continuations
            if not self._could_continue_number(token):
                continue

            # Check if continuing the partial number with this token violates bounds
            continued = partial + token
            try:
                value = float(continued)
                for (min_val, max_val) in self.bounds.values():
                    if value < min_val or value > max_val:
                        mask[token_id] = True
                        break
            except ValueError:
                # Not a valid number, don't mask
                pass

        return mask

    def _extract_numbers(self, text: str) -> list[float]:
        """Extract all numbers (including negative and decimal) from text.

        Args:
            text: Text to extract numbers from.

        Returns
        -------
        list of float
            All numbers found in the text.
        """
        # Match integers and decimals, including negative numbers
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        return [float(m) for m in matches]

    def _extract_partial_number(self, text: str) -> str | None:
        """Extract partial number at the end of text.

        Args:
            text: Text to check.

        Returns
        -------
        str or None
            The partial number string if found, None otherwise.
        """
        # Match number at end of string (including partial)
        match = re.search(r'(-?\d+\.?\d*)$', text)
        return match.group(1) if match else None

    def _could_continue_number(self, token: str) -> bool:
        """Check if a token could continue a partial number.

        Args:
            token: Token to check.

        Returns
        -------
        bool
            True if token looks like it could be part of a number.
        """
        # Token should be mostly digits/decimal point
        if not token:
            return False

        # Allow: digits, decimal point, minus sign (if at start)
        token_stripped = token.strip()
        if not token_stripped:
            return False

        # First character can be minus
        rest = token_stripped
        if rest[0] == '-':
            rest = rest[1:]

        # Rest should be digits or decimal point
        return all(c.isdigit() or c == '.' for c in rest)

    def __repr__(self) -> str:
        """Return string representation."""
        bounds_str = ", ".join(f"{k}={v}" for k, v in self.bounds.items())
        has_tokenizer = self.tokenizer is not None
        return (
            f"NumericalBoundsConstraint({bounds_str}, weight={self.weight}, "
            f"enabled={self.enabled}, has_tokenizer={has_tokenizer})"
        )


class KeywordPresenceConstraint(LLMConstraint):
    """Constraint requiring certain keywords to be present in output.

    This is useful for ensuring that generated text contains
    required factual elements or domain-specific terms.

    Parameters
    ----------
    required_keywords : list[str]
        List of keywords that must appear in the text.
    weight : float, optional
        Weight for penalty scoring, by default 1.0.
    enabled : bool, optional
        Whether the constraint is active, by default True.

    Examples
    --------
    >>> constraint = KeywordPresenceConstraint(["temperature", "pressure"])
    >>> constraint.check("The temperature is 25 degrees")
    False  # Missing "pressure"
    >>> constraint.compute_penalty("The temperature is 25 degrees")
    1.0  # 1 missing keyword
    """

    def __init__(
        self,
        required_keywords: list[str],
        weight: float = 1.0,
        enabled: bool = True
    ) -> None:
        """Initialize the keyword presence constraint.

        Args:
            required_keywords: Keywords that must be present.
            weight: Penalty weight multiplier.
            enabled: Whether constraint is active.
        """
        super().__init__(weight, enabled)
        self.required_keywords: list[str] = [
            kw.lower() for kw in required_keywords
        ]

    def check(self, text: str) -> bool:
        """Check if all required keywords are present.

        Args:
            text: Generated text to evaluate.

        Returns
        -------
        bool
            True if all keywords found, False otherwise.
        """
        text_lower = text.lower()
        return all(kw in text_lower for kw in self.required_keywords)

    def compute_penalty(self, text: str) -> float:
        """Compute penalty based on missing keywords.

        Args:
            text: Generated text to evaluate.

        Returns
        -------
        float
            Penalty equal to number of missing keywords.
        """
        if not self.enabled:
            return 0.0

        text_lower = text.lower()
        missing = sum(1 for kw in self.required_keywords if kw not in text_lower)
        return float(missing)

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """Compute mask for tokens.

        For keyword presence, we don't mask at the token level
        since we can't know if future tokens will contain keywords.
        This always returns an empty mask.

        Args:
            current_sequence: Token IDs generated so far (unused).
            vocab_size: Size of the vocabulary.

        Returns
        -------
        torch.Tensor
            All-false mask (no tokens masked).
        """
        # Cannot enforce keyword presence at token level
        return torch.zeros(vocab_size, dtype=torch.bool)

    def __repr__(self) -> str:
        """Return string representation."""
        kw_str = ", ".join(f"'{kw}'" for kw in self.required_keywords)
        return f"KeywordPresenceConstraint([{kw_str}], weight={self.weight}, enabled={self.enabled})"
