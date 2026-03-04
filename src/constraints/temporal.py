"""Temporal constraint implementations for PI-LLM Alignment.

This module provides constraints that enforce temporal and logical ordering,
such as event sequences, procedural steps, or growth stages.
"""

import re
from typing import Any
import torch

from .base import LLMConstraint


class TemporalOrderConstraint(LLMConstraint):
    """Constraint enforcing that temporal stages appear in sequential order.

    This constraint checks if sequences of events or procedural steps follow
    a logical, monotonic order. It is useful for ensuring that generated text
    respects the temporal progression of processes like crop growth,
    manufacturing workflows, or historical events.

    Parameters
    ----------
    stages : list[str]
        Ordered list of stage keywords or regex patterns. Each pattern
        will be searched for in the text, and they must appear in the
        same order as specified in this list.
        Examples:
            - ['seedling', 'tillering', 'heading', 'maturity']
            - ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
            - [r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfourth\b']
    weight : float, optional
        Weight multiplier for penalty scoring, by default 1.0.
    enabled : bool, optional
        Whether the constraint is active, by default True.
    case_sensitive : bool, optional
        Whether pattern matching is case-sensitive, by default False.
    whole_word : bool, optional
        Whether to match whole words only, by default True.
        When True, 'stage' will not match 'stages'.

    Attributes
    ----------
    stages : list[str]
        The ordered stage patterns to enforce.
    case_sensitive : bool
        Whether matching is case-sensitive.
    whole_word : bool
        Whether to match whole words only.

    Examples
    --------
    >>> constraint = TemporalOrderConstraint(
    ...     stages=['seedling', 'tillering', 'heading', 'maturity']
    ... )
    >>> constraint.check("The crop reached tillering after seedling stage.")
    True
    >>> constraint.check("The crop reached maturity before tillering.")
    False
    >>> constraint.compute_penalty("Heading before tillering occurred.")
    1.0  # 1 inversion: heading (index 2) before tillering (index 1)
    """

    def __init__(
        self,
        stages: list[str],
        weight: float = 1.0,
        enabled: bool = True,
        case_sensitive: bool = False,
        whole_word: bool = True,
    ) -> None:
        """Initialize the temporal order constraint.

        Args:
            stages: Ordered list of stage keywords or regex patterns.
            weight: Penalty weight multiplier.
            enabled: Whether constraint is active.
            case_sensitive: Whether pattern matching is case-sensitive.
            whole_word: Whether to match whole words only.

        Raises
        ------
        ValueError
            If stages is empty or contains fewer than 2 stages,
            or if weight is negative.
        """
        if len(stages) < 2:
            raise ValueError(
                f"TemporalOrderConstraint requires at least 2 stages, "
                f"got {len(stages)}"
            )

        super().__init__(weight, enabled)
        self.stages: list[str] = stages
        self.case_sensitive: bool = case_sensitive
        self.whole_word: bool = whole_word

    def check(self, text: str) -> bool:
        """Check if stages appear in correct sequential order.

        This method extracts the positions of all stage keywords/patterns
        in the text and verifies they appear in the same order as
        specified in the stages list.

        Args:
            text: Generated text to evaluate.

        Returns
        -------
        bool
            True if all found stages appear in order, False otherwise.
            Missing stages are ignored (only checks relative order
            of stages that are actually present).
        """
        if not self.enabled:
            return True

        # Find positions of each stage in text
        positions = self._find_stage_positions(text)

        # Filter out stages that weren't found (position = -1)
        found_stages = [
            (stage_idx, pos)
            for stage_idx, pos in enumerate(positions)
            if pos >= 0
        ]

        # Check if positions are in ascending order
        for i in range(len(found_stages) - 1):
            current_idx, current_pos = found_stages[i]
            next_idx, next_pos = found_stages[i + 1]

            # If a later stage appears before an earlier one, violation
            if current_idx < next_idx and current_pos > next_pos:
                return False

        return True

    def compute_penalty(self, text: str) -> float:
        """Compute penalty based on inversion count.

        The penalty is calculated as the number of inversions, where
        an inversion is a pair (i, j) with i < j but stage j appears
        before stage i in the text.

        For example, if stages are [A, B, C, D] and text mentions
        them in order [C, A, B, D], the inversions are:
        - (A, C): A (idx 0) appears after C (idx 2)
        - (B, C): B (idx 1) appears after C (idx 2)
        Total: 2 inversions

        Args:
            text: Generated text to evaluate.

        Returns
        -------
        float
            Number of inversions (non-negative). Returns 0.0 if all
            stages appear in correct order.
        """
        if not self.enabled:
            return 0.0

        # Find positions of each stage
        positions = self._find_stage_positions(text)

        # Get list of (stage_index, position) for found stages
        found_stages = [
            (stage_idx, pos)
            for stage_idx, pos in enumerate(positions)
            if pos >= 0
        ]

        # Count inversions using merge sort algorithm
        inversion_count = self._count_inversions(found_stages)

        return float(inversion_count)

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """Compute logits mask for temporal constraint.

        Note: Temporal constraints cannot be enforced at the token level
        during generation because we cannot predict which tokens will
        lead to out-of-order stages. This implementation returns an
        empty mask (no masking).

        Subclasses that want to implement token-level masking could
        analyze the current partial sequence and mask tokens that
        would immediately create an inversion.

        Args:
            current_sequence: Token IDs generated so far (unused).
            vocab_size: Size of the vocabulary.

        Returns
        -------
        torch.Tensor
            All-zeros mask (no tokens masked).
        """
        # Cannot enforce temporal ordering at token level
        # without knowing future context
        return torch.zeros(vocab_size, dtype=torch.bool)

    def _find_stage_positions(self, text: str) -> list[int]:
        """Find the character position of each stage in the text.

        For each stage pattern, finds the first occurrence in the text.
        Returns -1 for stages that are not found.

        Args:
            text: Text to search.

        Returns
        -------
        list[int]
            List of positions, one per stage. Position is the character
            index of the first match, or -1 if not found.
        """
        search_text = text if self.case_sensitive else text.lower()
        positions: list[int] = []

        for stage in self.stages:
            pattern = self._compile_pattern(stage)
            match = pattern.search(search_text)
            positions.append(match.start() if match else -1)

        return positions

    def _compile_pattern(self, stage: str) -> re.Pattern:
        """Compile a stage string into a regex pattern.

        Args:
            stage: Stage keyword or pattern.

        Returns
        -------
        re.Pattern
            Compiled regex pattern.
        """
        pattern = stage

        # Handle case sensitivity
        flags = 0 if self.case_sensitive else re.IGNORECASE

        # Add whole word boundary if requested
        if self.whole_word and self._is_simple_keyword(pattern):
            pattern = r'\b' + re.escape(pattern) + r'\b'
        elif not self._is_simple_keyword(pattern):
            # Already a regex pattern, use as-is
            pass
        elif self.whole_word:
            pattern = r'\b' + re.escape(pattern) + r'\b'
        else:
            pattern = re.escape(pattern)

        return re.compile(pattern, flags)

    def _is_simple_keyword(self, pattern: str) -> bool:
        """Check if a pattern is a simple keyword (not a regex).

        Args:
            pattern: Pattern string to check.

        Returns
        -------
        bool
            True if the pattern contains no special regex characters.
        """
        # Check for regex special characters
        special_chars = set(r'\.^$*+?{}[]|()')
        return not any(c in pattern for c in special_chars)

    def _count_inversions(self, items: list[tuple[int, int]]) -> int:
        """Count the number of inversions in the stage positions.

        An inversion is a pair (i, j) where i < j but position[i] > position[j].

        Uses merge sort for O(n log n) complexity.

        Args:
            items: List of (stage_index, position) tuples.

        Returns
        -------
        int
            Number of inversions found.
        """
        # Extract just the positions for inversion counting
        positions = [pos for _, pos in items]

        # Count inversions using merge sort
        _, count = self._merge_sort_count(positions)
        return count

    def _merge_sort_count(self, arr: list[int]) -> tuple[list[int], int]:
        """Sort array and count inversions using merge sort.

        Args:
            arr: List of positions to sort.

        Returns
        -------
        tuple[list[int], int]
            Sorted list and inversion count.
        """
        if len(arr) <= 1:
            return arr, 0

        mid = len(arr) // 2
        left, left_inv = self._merge_sort_count(arr[:mid])
        right, right_inv = self._merge_sort_count(arr[mid:])

        merged, merge_inv = self._merge_count(left, right)

        total_inv = left_inv + right_inv + merge_inv
        return merged, total_inv

    def _merge_count(
        self,
        left: list[int],
        right: list[int]
    ) -> tuple[list[int], int]:
        """Merge two sorted lists and count split inversions.

        Args:
            left: First sorted list.
            right: Second sorted list.

        Returns
        -------
        tuple[list[int], int]
            Merged list and split inversion count.
        """
        result: list[int] = []
        i = j = 0
        inversions = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                # All remaining elements in left are inversions with right[j]
                result.append(right[j])
                inversions += len(left) - i
                j += 1

        # Add remaining elements
        result.extend(left[i:])
        result.extend(right[j:])

        return result, inversions

    def get_stage_order(self, text: str) -> list[str] | None:
        """Get the actual order of stages found in the text.

        Useful for debugging and analysis of violations.

        Args:
            text: Text to analyze.

        Returns
        -------
        list[str] or None
            List of stage names in the order they appear in the text.
            Returns None if no stages are found.
        """
        positions = self._find_stage_positions(text)

        # Create list of (position, stage_name) for found stages
        found = [
            (pos, self.stages[i])
            for i, pos in enumerate(positions)
            if pos >= 0
        ]

        if not found:
            return None

        # Sort by position and extract stage names
        found.sort(key=lambda x: x[0])
        return [stage for _, stage in found]

    def __repr__(self) -> str:
        """Return string representation."""
        stages_repr = ", ".join(f"'{s}'" for s in self.stages[:3])
        if len(self.stages) > 3:
            stages_repr += f", ... ({len(self.stages)} total)"
        return (
            f"TemporalOrderConstraint(stages=[{stages_repr}], "
            f"weight={self.weight}, enabled={self.enabled})"
        )


__all__ = ["TemporalOrderConstraint"]
