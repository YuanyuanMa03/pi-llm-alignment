# PI-LLM Alignment: Physics-Informed Constraints for LLM Hallucination Reduction

> **Research Question**: Can constraint mechanisms from physics-informed machine learning transfer to Large Language Model alignment?

---

## Overview

This project explores a novel approach to LLM alignment by adapting constraint enforcement mechanisms from Physics-Informed Neural Networks (PINNs). Our prior work on PI-KD (Physics-Informed Knowledge Distillation for agricultural models) achieved **zero physical constraint violations** through sophisticated loss design. We investigate whether similar constraint-based methods can reduce LLM hallucinations.

---

## Research Motivation

| Problem | Current Solutions | Our Approach |
|---------|------------------|--------------|
| **LLM Hallucination** | RLHF, DPO, Constitutional AI | **Hard constraint enforcement** |
| **Soft alignment only** | Reward-based optimization | **Hybrid: constraint + objective** |
| **Domain-agnostic** | General methods | **Transfer from physics-informed ML** |

### Key Insight

Physics-informed ML has developed sophisticated mechanisms to enforce constraints like:
- **Monotonicity** (quantities only increase/decrease as expected)
- **Non-negativity** (physical quantities cannot be negative)
- **Correlation** (variables maintain expected relationships)

These constraint types have direct analogues in language generation:
- **Temporal Consistency** (narrative events occur in logical order)
- **Factual Bounds** (no impossible values)
- **Multi-modal Coherence** (cross-modality relationships preserved)

---

## Research Questions

### RQ1 (Main)
Can physics-informed constraint mechanisms effectively reduce LLM hallucinations?

### RQ2 (Ablation)
Which constraint types transfer most effectively from physical to linguistic domains?

### RQ3 (Comparison)
How does constraint-based alignment compare to RLHF/DPO in terms of:
- Hallucination reduction
- Training efficiency
- Output quality

---

## Method: PI-Align

Our proposed framework consists of three components:

### 1. Constraint Formalization
```python
class LLMConstraint(ABC):
    """Base class for LLM output constraints"""
    @abstractmethod
    def check(self, text: str) -> bool: pass

    @abstractmethod
    def penalty(self, text: str) -> float: pass
```

### 2. Constraint-Aware Generation
- Generate with constraint checking
- Iterative refinement on violations
- Backtracking to valid states

### 3. Training with Constraint Loss
```python
L_total = L_language + λ * L_constraint
```

---

## Project Structure

```
pi-llm-alignment/
├── README.md                   # This file
├── CLAUDE.md                   # Project instructions for Claude Code
├── requirements.txt            # Dependencies
│
├── src/
│   ├── constraints/
│   │   ├── base.py            # Abstract constraint classes
│   │   ├── temporal.py        # Temporal consistency
│   │   ├── factual.py         # Factual bounds
│   │   └── logical.py         # Logical rules
│   │
│   ├── models/
│   │   ├── constraint_aware_llm.py  # Main model wrapper
│   │   └── training.py               # Training loop with constraints
│   │
│   ├── evaluation/
│   │   ├── metrics.py         # Constraint violation metrics
│   │   └── benchmarks.py      # Benchmark runners
│   │
│   └── experiments/
│       ├── demo.py            # Quick demo
│       ├── ablation.py        # Ablation studies
│       └── comparison.py      # Baseline comparisons
│
├── data/
│   └── constraints/            # Constraint definitions
│
├── results/
│   ├── logs/                  # Training logs
│   └── figures/               # Generated figures
│
└── paper/
    ├── figures/               # Paper figures
    └── draft.tex             # Paper draft
```

---

## Quick Start

### Installation
```bash
# Clone repository (if on GitHub)
git clone https://github.com/YuanyuanMa03/pi-llm-alignment.git
cd pi-llm-alignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo
```bash
# Simple constraint demo
python src/experiments/demo.py

# Full experiment
python src/experiments/comparison.py --benchmark truthfulqa
```

### Using Local Models

For offline development, you can use locally downloaded models:

```bash
# The demo uses local Qwen3.5-2B model by default
python demo_qwen.py
```

Model is stored at: `models/Qwen/Qwen3___5-2B/`

To use a different model:
```python
# Edit demo_qwen.py and change the model_name parameter
model, tokenizer = load_model("path/to/your/model")
# Or use HuggingFace: load_model("Qwen/Qwen2.5-3B-Instruct")
```

---

## Experimental Setup

### Baselines
| Method | Description |
|--------|-------------|
| Vanilla LLM | No alignment |
| RLHF | Reinforcement Learning from Human Feedback |
| DPO | Direct Preference Optimization |
| **PI-Align (Ours)** | Physics-Informed Constraint Alignment |

### Benchmarks
| Dataset | Focus |
|---------|-------|
| TruthfulQA | Factuality |
| HALU-EVAL | Hallucination detection |
| BIG-Bench Hard | Reasoning |
| Custom (Temporal) | Temporal consistency |

### Metrics
- **Constraint Violation Rate** (primary)
- Accuracy / F1 Score
- Generation Quality (perplexity, human eval)
- Training Efficiency (time, compute)

---

## Expected Contributions

1. **Novel Paradigm**: First systematic transfer of physics-informed constraints to LLM alignment
2. **Framework**: Reusable constraint formalization for language models
3. **Empirical Evidence**: Comprehensive comparison with existing alignment methods
4. **Analysis**: Which constraints transfer and why

---

## Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| **Phase 1** | Month 1 | Literature review + constraint formalization |
| **Phase 2** | Month 2 | PI-Align framework implementation |
| **Phase 3** | Month 3 | Benchmark experiments |
| **Phase 4** | Month 4 | Ablation studies + analysis |
| **Phase 5** | Month 5 | Workshop paper submission |
| **Phase 6** | Month 6 | Camera-ready + conference version |

---

## Related Work

- **Physics-Informed ML**: Raissi et al. (2019), PINNs
- **LLM Alignment**: Bai et al. (2022) RLHF, Touvron et al. (2023) LLaMA-2
- **Constrained Decoding**: Holtzman et al. (2020) Nucleus sampling
- **Our Prior Work**: PI-KD for agricultural CH4 prediction

---

## License

MIT License

---

## Contact

Yuanyuan Ma - [GitHub](https://github.com/YuanyuanMa03)

---

## Citation

```bibtex
@misc{pi-llm-alignment,
  title={From Physics to Language: Constraint-Based Alignment for LLMs},
  author={Ma, Yuanyuan},
  year={2024},
  note={Research in progress}
}
```
