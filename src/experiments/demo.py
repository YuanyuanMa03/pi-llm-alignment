#!/usr/bin/env python3
"""
PI-LLM Alignment - Quick Demo

This script demonstrates a simple constraint-based generation:
- Generate character descriptions
- Enforce numerical bounds (height, weight)
- Compare constrained vs unconstrained generation
"""

from typing import List
import random


class SimpleBoundsConstraint:
    """Simple numerical bounds constraint for demo purposes."""

    def __init__(self, bounds: dict):
        """
        Args:
            bounds: dict mapping variable names to (min, max) tuples
                    e.g., {"height_cm": (0, 300), "weight_kg": (0, 200)}
        """
        self.bounds = bounds
        self.violations = []

    def check(self, text: str) -> bool:
        """Check if text violates any bounds."""
        import re

        # Find all numbers in text
        numbers = re.findall(r'\d+\.?\d*', text)

        # Simple heuristic: check if any number violates any bound
        # In a real system, we'd use NLP to identify which numbers correspond to which variables
        for num_str in numbers:
            num = float(num_str)
            for var, (min_val, max_val) in self.bounds.items():
                if not (min_val <= num <= max_val):
                    self.violations.append({
                        'variable': var,
                        'value': num,
                        'bounds': (min_val, max_val),
                        'context': text
                    })
                    return False
        return True

    def penalty(self, text: str) -> float:
        """Calculate penalty score for violations."""
        if self.check(text):
            return 0.0
        return len(self.violations)

    def get_violations(self) -> List[dict]:
        """Return list of violations found."""
        return self.violations


class MockLLM:
    """Mock LLM for demonstration (replace with real model)."""

    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text (mock implementation).
        In real usage, this would call an actual LLM.
        """
        # Mock responses for demo
        responses = [
            "The person is 175 cm tall and weighs 70 kg.",
            "The person is 250 cm tall and weighs 80 kg.",  # Violates height
            "The person is 180 cm tall and weighs -5 kg.",  # Violates weight (negative)
            "The person is 165 cm tall and weighs 55 kg.",
            "The person is 400 cm tall and weighs 300 kg.",  # Violates both
        ]
        return random.choice(responses)


def run_demo():
    """Run the constraint demo."""
    print("=" * 60)
    print("PI-LLM Alignment: Simple Constraint Demo")
    print("=" * 60)
    print()

    # Define constraints for human physical characteristics
    constraints = {
        "height_cm": (0, 300),   # 0 to 300 cm
        "weight_kg": (0, 200),   # 0 to 200 kg
    }

    constraint_checker = SimpleBoundsConstraint(constraints)
    llm = MockLLM("demo-model")

    print("Constraints:")
    for var, (min_val, max_val) in constraints.items():
        print(f"  {var}: {min_val} <= value <= {max_val}")
    print()

    # Run generations
    num_generations = 10
    print(f"Running {num_generations} generations...\n")

    violations_count = 0
    valid_count = 0

    for i in range(num_generations):
        prompt = "Generate a description of a person with their height and weight."
        output = llm.generate(prompt)

        # Check constraints
        constraint_checker.violations = []
        is_valid = constraint_checker.check(output)

        status = "✓ VALID" if is_valid else "✗ VIOLATION"
        print(f"[{i+1}] {status}")
        print(f"    Output: {output}")

        if not is_valid:
            violations_count += 1
            print(f"    Violations:")
            for v in constraint_checker.get_violations():
                print(f"      - {v['variable']}: {v['value']} (bounds: {v['bounds']})")
        else:
            valid_count += 1
        print()

    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Total generations: {num_generations}")
    print(f"  Valid outputs: {valid_count} ({100*valid_count/num_generations:.1f}%)")
    print(f"  Violations: {violations_count} ({100*violations_count/num_generations:.1f}%)")
    print("=" * 60)
    print()
    print("This is a simplified demo. The full PI-Align framework will:")
    print("  - Use real LLMs (GPT-2, LLaMA, etc.)")
    print("  - Implement more sophisticated constraints")
    print("  - Support constraint-aware training")
    print("  - Integrate with standard benchmarks (TruthfulQA, etc.)")
    print()


if __name__ == "__main__":
    run_demo()
