# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Project**: PI-LLM Alignment - Physics-Informed Constraints for LLM Hallucination Reduction

**Core Insight**: Hard constraint mechanisms from physics-informed ML transfer to language models through (1) logit-level intervention during decoding and (2) automated preference data generation for DPO.

**Current Phase**: Variant A (Logit-Constrained Decoding) - Primary focus on inference-time constraint enforcement via HuggingFace LogitsProcessor integration.

---

## Table of Contents

- [Research Approach: Three Variants](#research-approach-three-variants)
- [Tech Stack](#tech-stack)
- [Design Philosophy](#design-philosophy)
- [Architecture Overview](#architecture-overview)
- [Constraint Interface](#constraint-interface)
- [Common Commands](#common-commands)
- [Local Model Setup](#local-model-setup)
- [Testing Guidelines](#testing-guidelines)
- [Development Workflow](#development-workflow)

---

## Research Approach: Three Variants

The project has been redesigned to focus on engineering-feasible approaches:

| Variant | Focus | Status | Use Case |
|---------|-------|--------|----------|
| **A: Logit-Constrained Decoding** | Inference-time logit masking | ✅ **Primary** | Zero-overhead constraint enforcement via LogitsProcessor |
| **B: Constraint-Aware DPO** | Automated preference data generation | 🚧 Planned | Use constraint checkers to score responses for DPO pairs |
| **C: Domain-Specific** | Code/math generation with verifiable constraints | 📋 Fallback | Objective evaluation in constrained domains |

**Current implementation focuses on Variant A**: The `ConstraintLogitProcessor` class integrates with HuggingFace's generation pipeline to mask invalid tokens at each decoding step.

---

## Tech Stack

### Core Requirements

```yaml
python: ">=3.10"
type_checking: "strict"  # Use mypy with --strict flag
data_validation: "pydantic>=2.0"
```

### Key Dependencies

```python
# Core ML
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0

# Type Hints & Validation
typing>=3.10.0  # Use built-in typing module
pydantic>=2.0   # For data validation and settings

# HuggingFace Integration
# - LogitsProcessor for inference-time intervention
# - Trainer for DPO experiments

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Development
mypy>=1.5.0
ruff>=0.1.0  # Fast linter
black>=23.7.0
```

### Type Hinting Requirements

**Strict type hints are mandatory for all new code**:

```python
from typing import Protocol, TypeVar, Generic
from collections.abc import Callable, Sequence
from dataclasses import dataclass

T = TypeVar('T')

class ExampleProcessor(Generic[T]):
    def __init__(self, config: Config) -> None:
        self.config = config

    def process(self, inputs: Sequence[T]) -> dict[str, float]:
        """Process a sequence of inputs."""
        return {key: self._compute(value) for key, value in inputs}

    def _compute(self, value: T) -> float:
        """Helper method with explicit return type."""
        return float(value)
```

**Use Protocol for interface definitions**:

```python
from typing import Protocol

class ConstraintChecker(Protocol):
    """Protocol defining the constraint interface."""

    def check(self, text: str) -> bool: ...

    def compute_penalty(self, text: str) -> float: ...

    def get_logits_mask(self, current_sequence: list[int], vocab_size: int) -> torch.Tensor: ...
```

---

## Design Philosophy

### Dual-Mode Constraint Design

**All constraints MUST support two operational modes**:

| Mode | Purpose | Return Type | Use Case |
|------|---------|-------------|----------|
| **Evaluation** | Post-generation scoring | `float` penalty | Training loss, ranking, evaluation |
| **Intervention** | Inference-time control | `Tensor` logit mask | Generation guidance |

```python
from abc import ABC, abstractmethod
import torch

class LLMConstraint(ABC):
    """Base class for dual-mode constraints.

    All concrete constraints must implement both evaluation
    and intervention methods to support the full PI-Align framework.
    """

    @abstractmethod
    def check(self, text: str) -> bool:
        """Check if generated text satisfies the constraint."""
        ...

    @abstractmethod
    def compute_penalty(self, text: str) -> float:
        """Compute a penalty score for constraint violation.

        Returns 0.0 if perfectly aligned, higher values for worse violations.
        """
        ...

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """Compute logits mask for inference-time intervention.

        Returns boolean tensor where True = mask this token.
        """
        ...
```

    @abstractmethod
    def get_logit_bias(
        self,
        prefix: str,
        vocab_tokens: list[str],
        device: str | None = None
    ) -> Tensor:
        """Compute logit bias for inference-time guidance.

        Unlike masking, bias adds a continuous value to logits.

        Args:
            prefix: The prefix generated so far.
            vocab_tokens: List of all vocabulary tokens.
            device: Device for the output tensor.

        Returns:
            A float tensor of shape (vocab_size,) with bias values.
            Negative values discourage tokens, positive encourage.
        """
        ...
```

### Modularity for Ablation Studies

**Code structure enables easy ablation**:

```python
# All constraints are independently swappable
from src.constraints import TemporalConstraint, FactualConstraint, LogicalConstraint

# Ablation: test each constraint independently
ablation_configs = {
    "none": [],
    "temporal_only": [TemporalConstraint()],
    "factual_only": [FactualConstraint()],
    "logical_only": [LogicalConstraint()],
    "temporal_factual": [TemporalConstraint(), FactualConstraint()],
    "all": [TemporalConstraint(), FactualConstraint(), LogicalConstraint()],
}
```

### Configuration Management

**Use Pydantic for all configuration**:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class ConstraintConfig(BaseModel):
    """Configuration for a single constraint."""

    name: str
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    mode: Literal["mask", "bias", "both"] = "both"

    @field_validator("weight")
    def validate_weight(cls, v: float) -> float:
        """Ensure weight is non-negative."""
        if v < 0:
            raise ValueError("Constraint weight must be non-negative")
        return v


class ExperimentConfig(BaseModel):
    """Master configuration for experiments."""

    model_name: str
    constraints: list[ConstraintConfig]
    max_length: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    seed: int = Field(default=42)

    class Config:
        # Enable strict validation
        strict = True
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PI-Align Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Constraint Registry                    │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │
│  │  │  Temporal    │  │   Factual    │  │   Logical    │   │    │
│  │  │  Constraint  │  │  Constraint  │  │  Constraint  │   │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │    │
│  └─────────┼─────────────────┼─────────────────┼───────────┘    │
│            │                 │                 │                 │
│            └─────────────────┼─────────────────┘                 │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Constraint Logit Processor                 │    │
│  │                                                          │    │
│  │   logits = base_model(input_ids)                        │    │
│  │   for constraint in constraints:                        │    │
│  │       logits = constraint.apply(logits, prefix)         │    │
│  │   return logits                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Sampling                              │    │
│  │   (temperature, top-k, top-p, etc.)                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Output Token Selection                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
pi-llm-alignment/
├── src/
│   ├── constraints/              # Constraint implementations
│   │   ├── base.py               # LLMConstraint abstract class + ConstraintRegistry
│   │   ├── factual.py            # NumericalBoundsConstraint, KeywordPresenceConstraint
│   │   ├── temporal.py           # TemporalOrderConstraint
│   │   └── registry.py           # Default constraint registration
│   │
│   ├── inference/                # Inference-time components
│   │   └── processor.py          # ConstraintLogitProcessor, ConstrainedGenerator
│   │
│   └── experiments/
│       └── demo.py               # Simple constraint demos
│
├── tests/
│   ├── conftest.py              # Pytest fixtures
│   ├── test_constraints/        # Constraint tests
│   └── test_inference/          # Inference tests
│
├── models/                       # Local model storage
│   └── Qwen/                    # Qwen2.5-3B-Instruct (for offline dev)
│
├── data/                         # Data storage
│   └── constraints/             # Constraint definitions
│
├── results/                      # Experimental outputs
│   ├── logs/
│   └── figures/
│
├── paper/                        # Paper materials
│   ├── literature_review.md
│   └── figures/
│
├── docs/
│   └── revised_approach.md      # Three-variant approach documentation
│
├── demo_qwen.py                 # Main demo script (Qwen model)
├── demo_mock.py                 # Mock demo (no model required)
├── CLAUDE.md                     # This file
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Constraint Interface

### Abstract Base Class (Actual Implementation)

```python
# src/constraints/base.py

from abc import ABC, abstractmethod
import torch

class LLMConstraint(ABC):
    """Abstract base class for dual-mode LLM constraints.

    A constraint must support both:
    1. Evaluation mode: Post-generation penalty scoring via check() and compute_penalty()
    2. Intervention mode: Inference-time logit manipulation via get_logits_mask()

    Key design decision: get_logits_mask() operates on TOKEN IDs, not text.
    This is because the LogitsProcessor receives token IDs from the generation loop.
    Subclasses must decode token IDs to text to apply their constraint logic.
    """

    def __init__(self, weight: float = 1.0, enabled: bool = True) -> None:
        """Initialize the constraint.

        Args:
            weight: Weight multiplier for penalty scoring (evaluation mode).
            enabled: Whether the constraint is active.
        """
        self.weight: float = weight
        self.enabled: bool = enabled

    # ========== Abstract Methods (must implement) ==========

    @abstractmethod
    def check(self, text: str) -> bool:
        """Perform a hard constraint check on generated text.

        Args:
            text: The generated text to evaluate.

        Returns:
            True if constraint satisfied, False otherwise.
        """
        ...

    @abstractmethod
    def compute_penalty(self, text: str) -> float:
        """Compute penalty for constraint violation (for DPO/RLAIF).

        Args:
            text: The generated text to evaluate.

        Returns:
            Non-negative float penalty (0 = no violation).
        """
        ...

    # ========== Intervention Mode (override for logit control) ==========

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """Compute logits mask for inference-time intervention.

        Tokens marked True will have their logits set to -inf.

        IMPORTANT: This method receives token IDs, not text. Subclasses
        must decode token IDs to text using the tokenizer to apply constraints.

        Args:
            current_sequence: Token IDs generated so far (as list of ints).
            vocab_size: Size of the vocabulary.

        Returns:
            Boolean tensor of shape (vocab_size,). True = mask this token.

        Default implementation returns all-False (no masking).
        """
        if not self.enabled:
            return torch.zeros(vocab_size, dtype=torch.bool)
        return torch.zeros(vocab_size, dtype=torch.bool)
```

### Example: NumericalBoundsConstraint (Actual Implementation)

```python
# src/constraints/factual.py

class NumericalBoundsConstraint(LLMConstraint):
    """Constraint ensuring numerical values stay within bounds.

    This constraint demonstrates the key pattern: get_logits_mask()
    must decode partial text from current_sequence to determine which
    tokens to mask next.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        tokenizer: Any = None,  # Required for get_logits_mask()
        weight: float = 1.0,
        enabled: bool = True
    ) -> None:
        super().__init__(weight, enabled)
        self.bounds = bounds
        self.tokenizer = tokenizer

    def check(self, text: str) -> bool:
        """Check if all numbers in text are within bounds."""
        numbers = self._extract_numbers(text)
        for num in numbers:
            for (min_val, max_val) in self.bounds.values():
                if not (min_val <= num <= max_val):
                    return False
        return True

    def compute_penalty(self, text: str) -> float:
        """Compute penalty based on violation severity."""
        numbers = self._extract_numbers(text)
        total_penalty = 0.0
        for num in numbers:
            for (min_val, max_val) in self.bounds.values():
                if num < min_val:
                    total_penalty += (min_val - num) / min_val
                elif num > max_val:
                    total_penalty += (num - max_val) / max_val
        return self.weight * total_penalty

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """Mask tokens that would create out-of-bounds numbers.

        CRITICAL: Must decode current_sequence to text to check partial state.
        """
        if not self.enabled or self.tokenizer is None:
            return torch.zeros(vocab_size, dtype=torch.bool)

        # Decode current sequence to text
        prefix = self.tokenizer.decode(current_sequence, skip_special_tokens=True)
        partial = self._extract_partial_number(prefix)

        # If no partial number, no tokens to mask
        if partial is None:
            return torch.zeros(vocab_size, dtype=torch.bool)

        # Create mask by testing each vocab token
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in range(vocab_size):
            token = self.tokenizer.decode([token_id])
            continued = partial + token
            try:
                value = float(continued)
                for (min_val, max_val) in self.bounds.values():
                    if value < min_val or value > max_val:
                        mask[token_id] = True
            except ValueError:
                pass  # Not a number token, don't mask

        return mask
```

**Key Pattern**: `get_logits_mask()` requires a tokenizer to decode the current sequence. The `ConstraintLogitProcessor` handles this automatically—constraints receive the processor's tokenizer.

---

## Common Commands

### Environment Setup

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install dev dependencies (optional, for testing/linting)
pip install -e ".[dev]"

# Run type checking
mypy src/

# Run linting
ruff check src/
ruff format src/
```

### Running Demos

```bash
# Quick mock demo (no model download required)
python demo_mock.py

# Full demo with Qwen model (requires model download)
python demo_qwen.py

# Simple constraint demo
python src/experiments/demo.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_constraints/test_factual.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_constraints/test_factual.py::test_check_valid

# Run only unit tests (skip slow integration tests)
pytest -m "not slow"
```

### Code Quality

```bash
# Type checking with mypy (strict mode)
mypy src/ --strict

# Linting with ruff
ruff check src/

# Auto-fix linting issues
ruff check --fix src/

# Format code with ruff
ruff format src/

# Format check (CI mode)
ruff format --check src/
```

---

## Local Model Setup

For offline development, models can be stored locally in `models/Qwen/`:

```bash
# Download Qwen2.5-3B-Instruct locally
# (Requires ~6GB disk space)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"
local_path = "models/Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)
```

Then use local path in code:

```python
model = AutoModelForCausalLM.from_pretrained(
    "models/Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True  # Force local only
)
```

---

## Testing Guidelines

### Test Structure

```python
# tests/test_constraints/test_factual.py

import pytest
import torch
from src.constraints.factual import NumericalBoundsConstraint

class TestNumericalBoundsConstraint:
    """Test suite for NumericalBoundsConstraint."""

    @pytest.fixture
    def simple_constraint(self):
        """Fixture providing a test constraint."""
        return NumericalBoundsConstraint(
            bounds={"value": (0, 100)},
            weight=1.0,
            enabled=True
        )

    # ========== check() Tests ==========
    def test_check_valid_input(self, simple_constraint):
        """Test that valid inputs pass the check."""
        assert simple_constraint.check("The value is 50")

    def test_check_invalid_above_max(self, simple_constraint):
        """Test that values above max fail the check."""
        assert not simple_constraint.check("The value is 150")

    # ========== compute_penalty() Tests ==========
    def test_compute_penalty_zero_for_valid(self, simple_constraint):
        """Test that valid inputs have zero penalty."""
        assert simple_constraint.compute_penalty("The value is 50") == 0.0

    def test_compute_penalty_positive_for_invalid(self, simple_constraint):
        """Test that invalid inputs have positive penalty."""
        penalty = simple_constraint.compute_penalty("The value is 150")
        assert penalty > 0

    # ========== get_logits_mask() Tests ==========
    def test_get_logits_mask_shape(self, simple_constraint):
        """Test that logit mask has correct shape."""
        vocab_size = 1000
        mask = simple_constraint.get_logits_mask([], vocab_size)
        assert mask.shape == (vocab_size,)
        assert mask.dtype == torch.bool

    def test_get_logits_mask_without_tokenizer(self, simple_constraint):
        """Test that mask is empty without tokenizer."""
        vocab_size = 100
        mask = simple_constraint.get_logits_mask([], vocab_size)
        assert not mask.any()  # No tokenizer = no masking
```

### Coverage Requirements

- Aim for >80% code coverage
- All public methods must have tests
- Edge cases must be covered (empty inputs, extreme values, disabled constraints)
- Test both enabled and disabled constraint states

---

## Documentation Standards

### NumPy-Style Docstrings

All public methods should use NumPy-style docstrings:

```python
def get_logits_mask(
    self,
    current_sequence: list[int],
    vocab_size: int
) -> torch.Tensor:
    """Compute logits mask for inference-time intervention.

    This method returns a boolean mask indicating which tokens
    should be forbidden at the next generation step.

    Parameters
    ----------
    current_sequence : list of int
        The token IDs generated so far.
    vocab_size : int
        The size of the vocabulary.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (vocab_size,). Tokens marked True
        will be masked (logit set to -inf).

    Examples
    --------
    >>> constraint = NumericalBoundsConstraint({"value": (0, 100)})
    >>> sequence = [1, 2, 3]
    >>> mask = constraint.get_logits_mask(sequence, vocab_size=1000)
    >>> mask.shape
    torch.Size([1000])
    """
```

---

## Development Workflow

### Adding a New Constraint

1. **Create the constraint class**

```bash
touch src/constraints/my_constraint.py
```

2. **Implement all abstract methods**

```python
from .base import LLMConstraint
import torch

class MyConstraint(LLMConstraint):
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer  # Required for get_logits_mask()

    def check(self, text: str) -> bool:
        """Binary constraint check on generated text."""
        ...

    def compute_penalty(self, text: str) -> float:
        """Continuous penalty score for DPO/RLAIF."""
        ...

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """Return boolean mask of tokens to forbid at next step.

        CRITICAL: Receives token IDs, not text. Must decode using tokenizer.
        """
        # Decode current sequence to check partial state
        prefix = self.tokenizer.decode(current_sequence, skip_special_tokens=True)
        # ... determine which tokens to mask ...
        return mask
```

3. **Add to registry**

```python
# src/constraints/registry.py
from .my_constraint import MyConstraint

def register_default_constraints() -> None:
    registry = get_global_registry()
    registry.register("my_constraint", MyConstraint)
```

4. **Write tests**

```bash
touch tests/test_constraints/test_my_constraint.py
```

5. **Run tests**

```bash
pytest tests/test_constraints/test_my_constraint.py -v
```

6. **Type check**

```bash
mypy src/constraints/my_constraint.py
```

### Commit Conventions

```bash
# Feature
git commit -m "feat(constraints): add temporal consistency constraint"

# Bug fix
git commit -m "fix(inference): correct device handling in logit processor"

# Tests
git commit -m "test(constraints): add tests for numerical bounds"

# Docs
git commit -m "docs: update CLAUDE.md with dual-mode design"
```

---

## Project Status

**Current Phase**: Variant A (Logit-Constrained Decoding) - ✅ **Primary Implementation Complete**

- [x] Project structure
- [x] Base constraint interface (`LLMConstraint` with dual-mode support)
- [x] Constraint implementations (`NumericalBoundsConstraint`, `TemporalOrderConstraint`, `KeywordPresenceConstraint`)
- [x] LogitProcessor framework (`ConstraintLogitProcessor`, `ConstrainedGenerator`)
- [x] Demo scripts (`demo_qwen.py`, `demo_mock.py`)
- [x] Test suite (pytest with coverage)
- [ ] Evaluation framework (benchmarks, metrics)
- [ ] Variant B: DPO data generation
- [ ] Variant C: Domain-specific experiments

**Next Steps**:
1. Run evaluation on TruthfulQA or similar benchmark
2. Implement ablation study framework
3. Explore Variant B (DPO data generation)

---

## Target Publication

**Primary**: ICLR / NeurIPS Workshop on Reliable & Responsible LLMs
**Secondary**: ACL / EMNLP (if focusing on NLP aspects)

**Key Differentiators**:
1. **Three-variant approach**: Logit decoding, DPO data generation, domain-specific
2. **Zero-overhead inference**: Logit masking with minimal performance impact
3. **Physics-informed transfer**: Constraint mechanisms from scientific ML to language

---

**Last Updated**: 2026-03-04
**Version**: 0.1.0-alpha
