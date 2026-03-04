#!/usr/bin/env python3
"""
Quick Demo: Constraint System Mock Test (No Model Download Required)

This script demonstrates the constraint system using a mock generator,
so you can test the functionality without downloading large models.
"""

import sys
sys.path.insert(0, "src")

from typing import Any
import torch

# Import constraints
from src.constraints.factual import NumericalBoundsConstraint, KeywordPresenceConstraint
from src.constraints.temporal import TemporalOrderConstraint


class MockGenerator:
    """Mock generator for testing without a real model."""

    def __init__(self):
        self.constraints = []

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate mock text based on prompt keywords."""
        # Simple mock responses based on prompt
        if "age" in prompt.lower():
            # Without constraint: might generate out-of-bounds
            if any(c.enabled for c in self.constraints):
                return f"{prompt}25 years old"  # In bounds
            else:
                return f"{prompt}150 years old"  # Out of bounds on purpose

        elif "stage" in prompt.lower() or "growth" in prompt.lower():
            if any(c.enabled for c in self.constraints):
                return f"{prompt}seedling stage, then tillering, and finally maturity"
            else:
                return f"{prompt}maturity, then heading, and seedling stage"  # Wrong order

        return f"{prompt}[mock response]"

    def set_constraints(self, constraints):
        self.constraints = constraints

    def clear_constraints(self):
        self.constraints = []


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_numerical_constraint():
    """Demonstrate numerical bounds constraint."""
    print_section("Demo 1: Numerical Bounds Constraint")

    constraint = NumericalBoundsConstraint(
        bounds={"age": (0, 100)}
    )

    test_cases = [
        ("The person's age is ", "25 years old", 0.0),
        ("The person's age is ", "150 years old", 0.5),
        ("The person's age is ", "-5 years old", 0.05),
    ]

    for test_item in test_cases:
        prompt, response, expected_penalty = test_item
        full_text = prompt + response
        passed = constraint.check(full_text)
        penalty = constraint.compute_penalty(full_text)

        status = "✅" if passed else "❌"
        print(f"\n{status} Test: {repr(full_text)}")
        print(f"   Check: {passed}")
        print(f"   Penalty: {penalty} (expected: ~{expected_penalty})")


def demo_temporal_constraint():
    """Demonstrate temporal order constraint."""
    print_section("Demo 2: Temporal Order Constraint")

    constraint = TemporalOrderConstraint([
        "seedling", "tillering", "heading", "maturity"
    ])

    test_cases = [
        ("Correct order", "The crop went from seedling to tillering, then heading, and reached maturity.", True),
        ("Inverted", "The crop reached maturity before tillering occurred.", False),
        ("Partial", "Seedling stage was followed by heading.", True),
    ]

    for name, text, should_pass in test_cases:
        passed = constraint.check(text)
        penalty = constraint.compute_penalty(text)
        order = constraint.get_stage_order(text)

        status = "✅" if passed == should_pass else "❌"
        print(f"\n{status} {name}: {text}")
        print(f"   Check: {passed}")
        print(f"   Penalty: {penalty}")
        print(f"   Order: {order}")


def demo_inversion_counting():
    """Demonstrate inversion counting algorithm."""
    print_section("Demo 3: Inversion Counting")

    constraint = TemporalOrderConstraint([
        "first", "second", "third", "fourth"
    ])

    # Create specific test case
    text = "We completed fourth step, then first step, followed by third, and second step last."

    passed = constraint.check(text)
    penalty = constraint.compute_penalty(text)
    order = constraint.get_stage_order(text)

    print(f"\n📝 Text: {text}")
    print(f"\n📋 Stage order found: {order}")
    print(f"   Expected order: ['first', 'second', 'third', 'fourth']")
    print(f"\n🔢 Inversions:")
    print(f"   (first, fourth)  - first (idx 0) appears after fourth (idx 3)")
    print(f"   (second, fourth) - second (idx 1) appears after fourth (idx 3)")
    print(f"   (second, third)  - second (idx 1) appears after third (idx 2)")
    print(f"   (third, fourth)  - third (idx 2) appears after fourth (idx 3)")
    print(f"   Total: {int(penalty)} inversions")


def demo_constraint_combination():
    """Demonstrate multiple constraints."""
    print_section("Demo 4: Multiple Constraints")

    constraints = [
        NumericalBoundsConstraint({"age": (0, 120), "height_cm": (0, 250)}),
        TemporalOrderConstraint(["morning", "afternoon", "evening", "night"])
    ]

    text = (
        "In the morning, the patient (age 25, height 170 cm) was admitted. "
        "By afternoon, diagnosis was complete. Evening treatment began. "
        "By night, discharge was planned."
    )

    print(f"\n📝 Text: {text}")
    print(f"\n🔒 Constraints:")

    all_passed = True
    for i, constraint in enumerate(constraints, 1):
        passed = constraint.check(text)
        penalty = constraint.compute_penalty(text)

        status = "✅" if passed else "❌"
        print(f"\n{status} Constraint {i}: {constraint.__class__.__name__}")
        print(f"   Passed: {passed}")
        print(f"   Penalty: {penalty}")

        if not passed:
            all_passed = False

    print(f"\n{'✅ All constraints satisfied!' if all_passed else '❌ Some constraints violated'}")


def demo_mock_generator():
    """Demonstrate mock generator with constraints."""
    print_section("Demo 5: Mock Generator")

    generator = MockGenerator()

    # Without constraint
    print("\n--- Without Constraint ---")
    generator.clear_constraints()
    output = generator.generate("The person's age is ")
    print(f"Prompt: 'The person's age is '")
    print(f"Output: {repr(output)}")

    # With constraint
    print("\n--- With NumericalBoundsConstraint ---")
    generator.set_constraints([
        NumericalBoundsConstraint({"age": (0, 100)})
    ])
    output_constrained = generator.generate("The person's age is ")
    print(f"Prompt: 'The person's age is '")
    print(f"Constraint: age ∈ [0, 100]")
    print(f"Output: {repr(output_constrained)}")

    # Check constraint
    constraint = generator.constraints[0]
    passed = constraint.check(output_constrained)
    print(f"Constraint satisfied: {passed}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  PI-LLM Alignment: Constraint System Mock Demo")
    print("  (No model download required)")
    print("=" * 60)

    print("\n💡 This demonstrates the constraint system without requiring")
    print("   you to download large models. Each demo shows how constraints")
    print("   check and score generated text.\n")

    try:
        demo_numerical_constraint()
        demo_temporal_constraint()
        demo_inversion_counting()
        demo_constraint_combination()
        demo_mock_generator()

    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted by user")

    finally:
        print("\n" + "=" * 60)
        print("  Mock Demo Complete!")
        print("=" * 60)
        print("\n🚀 To run the real demo with Qwen3.5-4B:")
        print("   python demo_qwen.py")
        print("   (Requires ~8GB RAM and ~30 minutes for first download)\n")


if __name__ == "__main__":
    main()
