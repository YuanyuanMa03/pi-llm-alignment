#!/usr/bin/env python3
"""
Quick Demo: PI-LLM Alignment with Qwen3.5-2B

This script demonstrates constraint-based text generation using
Qwen3.5-2B with numerical bounds constraints.

Uses local model at models/Qwen/Qwen3___5-2B by default.
"""

import sys
import torch
from typing import Any

# Check for transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not installed. Run: pip install transformers")
    sys.exit(1)

# Add src to path
sys.path.insert(0, "src")

from src.constraints.factual import NumericalBoundsConstraint
from src.constraints.temporal import TemporalOrderConstraint
from src.inference.processor import ConstrainedGenerator


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_model(model_name: str = "models/Qwen/Qwen3___5-2B"):
    """Load Qwen3.5-2B model from local path.

    Args:
        model_name: Local path or HuggingFace model identifier.
                   Default uses local model at models/Qwen/Qwen3___5-2B

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"\n📥 Loading model: {model_name}")

    # Check if using local path
    import os
    if os.path.exists(model_name):
        print(f"   Using local model (offline mode)")
        local_files_only = True
    else:
        print(f"   Downloading from HuggingFace...")
        local_files_only = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only
    )

    # Load model with automatic device selection
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only
    )

    print(f"✅ Model loaded on {model.device}")
    print(f"   Model type: {model.config.model_type}")
    print(f"   Vocab size: {model.config.vocab_size}")

    return model, tokenizer


def demo_numerical_constraint(generator: ConstrainedGenerator) -> None:
    """Demonstrate numerical bounds constraint.

    Args:
        generator: The constrained generator.
    """
    print_section("Demo 1: Numerical Bounds Constraint")

    # Test case: generate text about a person's age
    prompt = "The person's age is "

    # Add constraint: age must be between 0 and 120
    constraint = NumericalBoundsConstraint(
        bounds={"age": (0, 120)},
        tokenizer=generator.tokenizer
    )
    generator.set_constraints([constraint])

    print(f"\n📝 Prompt: {repr(prompt)}")
    print(f"🔒 Constraint: age ∈ [0, 120]")
    print(f"🎯 Generating...")

    try:
        output = generator.generate(prompt, max_new_tokens=30, temperature=0.8)
        print(f"\n✨ Generated (with constraint):")
        print(f"   {repr(output)}")

        # Check if constraint was satisfied
        if constraint.check(output):
            print(f"\n✅ Constraint SATISFIED: age within [0, 120]")
        else:
            penalty = constraint.compute_penalty(output)
            print(f"\n❌ Constraint VIOLATED (penalty: {penalty})")

    except Exception as e:
        print(f"\n⚠️  Generation failed: {e}")

    # Now try without constraint for comparison
    generator.clear_constraints()
    print(f"\n--- Without constraint ---")
    print(f"🎯 Generating...")

    try:
        output_unconstrained = generator.generate(prompt, max_new_tokens=30, temperature=0.8)
        print(f"\n✨ Generated (unconstrained):")
        print(f"   {repr(output_unconstrained)}")

    except Exception as e:
        print(f"\n⚠️  Generation failed: {e}")


def demo_temporal_constraint(generator: ConstrainedGenerator) -> None:
    """Demonstrate temporal order constraint.

    Args:
        generator: The constrained generator.
    """
    print_section("Demo 2: Temporal Order Constraint")

    # Test case: crop growth stages
    prompt = "The rice crop growth cycle: "

    # Add constraint: stages must appear in order
    constraint = TemporalOrderConstraint([
        "seedling", "tillering", "heading", "maturity"
    ])
    generator.set_constraints([constraint])

    print(f"\n📝 Prompt: {repr(prompt)}")
    print(f"🔒 Constraint: stages in order")
    print(f"   seedling → tillering → heading → maturity")
    print(f"🎯 Generating...")

    try:
        output = generator.generate(prompt, max_new_tokens=80, temperature=0.7)
        print(f"\n✨ Generated (with constraint):")
        print(f"   {repr(output)}")

        # Check if constraint was satisfied
        if constraint.check(output):
            print(f"\n✅ Temporal order MAINTAINED")
        else:
            penalty = constraint.compute_penalty(output)
            order = constraint.get_stage_order(output)
            print(f"\n❌ Temporal order VIOLATED (penalty: {penalty})")
            print(f"   Actual order: {order}")

    except Exception as e:
        print(f"\n⚠️  Generation failed: {e}")


def demo_multiple_constraints(generator: ConstrainedGenerator) -> None:
    """Demonstrate multiple constraints simultaneously.

    Args:
        generator: The constrained generator.
    """
    print_section("Demo 3: Multiple Constraints")

    prompt = "Patient information: "

    # Add multiple constraints
    constraints = [
        NumericalBoundsConstraint(
            bounds={"age": (0, 120), "weight": (0, 200), "temperature": (35, 42)},
            tokenizer=generator.tokenizer
        ),
        TemporalOrderConstraint([
            "admission", "diagnosis", "treatment", "discharge"
        ])
    ]
    generator.set_constraints(constraints)

    print(f"\n📝 Prompt: {repr(prompt)}")
    print(f"🔒 Constraints:")
    print(f"   • age ∈ [0, 120]")
    print(f"   • weight ∈ [0, 200]")
    print(f"   • temperature ∈ [35, 42]")
    print(f"   • temporal: admission → diagnosis → treatment → discharge")
    print(f"🎯 Generating...")

    try:
        output = generator.generate(prompt, max_new_tokens=100, temperature=0.6)
        print(f"\n✨ Generated (with all constraints):")
        print(f"   {repr(output)}")

        # Check each constraint
        for i, constraint in enumerate(constraints, 1):
            if constraint.check(output):
                print(f"\n✅ Constraint {i} SATISFIED: {constraint.__class__.__name__}")
            else:
                penalty = constraint.compute_penalty(output)
                print(f"\n❌ Constraint {i} VIOLATED: {constraint.__class__.__name__} (penalty: {penalty})")

    except Exception as e:
        print(f"\n⚠️  Generation failed: {e}")


def main():
    """Main demo function."""
    print("\n" + "=" * 60)
    print("  PI-LLM Alignment: Qwen3.5-2B Demo")
    print("  Physics-Informed Constraints for LLM Generation")
    print("=" * 60)

    # Check for MPS (Apple Silicon) or CUDA
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"\n🍎 Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"\n🚀 Using CUDA GPU acceleration")
    else:
        device = "cpu"
        print(f"\n⚠️  No GPU detected, using CPU (will be slow)")

    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print(f"\n💡 Make sure you have:")
        print(f"   1. Sufficient RAM (16GB recommended)")
        print(f"   2. Accepted the model license on HuggingFace")
        print(f"   3. Internet connection for first download")
        sys.exit(1)

    # Create constrained generator
    generator = ConstrainedGenerator(
        model=model,
        tokenizer=tokenizer,
        default_generation_kwargs={
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    )

    # Run demos
    print("\n" + "▶" * 60)
    print("  Starting Demos")
    print("▶" * 60)

    try:
        demo_numerical_constraint(generator)
        demo_temporal_constraint(generator)
        demo_multiple_constraints(generator)

    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted by user")

    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "=" * 60)
        print("  Demo Complete")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
