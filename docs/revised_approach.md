# PI-LLM Alignment: Revised Approach

> **Pivot Summary**: Based on engineering feasibility analysis, we refocus from "constraint-in-training" to "constraint-in-inference" and "constraint-for-data-generation".

---

## The Problem with Original Approach

| Issue | Why It Fails | Solution |
|-------|--------------|----------|
| **Differentiability** | Discrete tokens can't propagate constraint gradients | Use constraints for decoding, not training |
| **Inference Cost** | Backtracking destroys KV cache, 10-100× slower | Logit-level masking (zero overhead) |
| **Constraint Generalization** | Language constraints are context-dependent | Start with hard domains (code/math) |

---

## New Approach: Three Variants

### Variant A: Logit-Constrained Decoding (Primary Focus)

**Idea**: Apply constraints at the logit level during generation, no training needed.

```python
# Before: Standard generation
output = model.generate(prompt)

# After: Constrained generation
processor = ConstraintLogitProcessor([TemporalConstraint(), FactualConstraint()])
output = model.generate(prompt, logits_processor=processor)
```

**Why This Works**:
- Compatible with HuggingFace `LogitsProcessor` interface
- Zero performance overhead (just logit addition)
- No retraining required
- Easy to ablate (add/remove constraints)

**Research Question**:
> Can logit-level constraint masking reduce factual hallucinations without retraining?

---

### Variant B: Constraint-Aware DPO (Secondary)

**Idea**: Use constraint checkers to automatically generate preference data for DPO.

```
1. Generate N responses from base model
2. Score each with constraint checker
3. Create (chosen, rejected) pairs
4. Train with standard DPO
```

**Why This Works**:
- Leverages existing DPO infrastructure
- Automates preference data generation
- No gradient through constraints needed
- Novel data generation paradigm

**Research Question**:
> Can constraint-based scoring replace human annotation for DPO data?

---

### Variant C: Domain-Specific Application (Fallback)

**Idea**: Apply to code generation where constraints are verifiable.

| Code Constraint | Verification Method |
|----------------|---------------------|
| Syntax validity | Parser |
| Type consistency | Type checker |
| API correctness | Documentation lookup |
| Test passing | Unit tests |

**Why This Works**:
- Objective evaluation (pass/fail)
- Clear baseline (CodeLlama, StarCoder)
- Industrial relevance
- Easier to publish

---

## Revised Timeline

| Month | Focus | Deliverable |
|-------|-------|-------------|
| **Month 1** | Variant A | Logit processor framework + math problem demo |
| **Month 2** | Variant B | DPO data generation + small-scale training |
| **Month 3-4** | Variant C | Code generation experiments |
| **Month 5** | Paper writing | Workshop submission |
| **Month 6** | Revision | Conference version |

---

## New Paper Narrative

### Old Narrative (Problematic)
> "We add physics-informed constraints to the LLM training loss..."

### New Narrative (Compelling)
> "We introduce a novel constrained decoding framework that operates at the logit level, eliminating hallucinations without retraining. Our approach bridges the gap between hard constraint enforcement in physics-informed ML and the soft nature of language generation through two mechanisms: (1) differentiable logit masking and (2) automated preference data generation for DPO."

---

## Implementation Priorities

### Priority 1: LogitProcessor Framework
```python
# src/inference/logit_processor.py
class ConstraintLogitProcessor(LogitsProcessor):
    def __init__(self, constraints):
        self.constraints = constraints

    def __call__(self, input_ids, logits):
        # Apply constraint masks
        return masked_logits
```

### Priority 2: Constraint Implementations
```python
# src/constraints/factual.py
class FactualBoundsConstraint:
    def get_forbidden_tokens(self, prefix):
        # Return token IDs that would violate factual bounds
        return forbidden_ids
```

### Priority 3: Evaluation Framework
```python
# src/evaluation/hallucination.py
def evaluate_hallucination_rate(model, dataset, processor=None):
    # Compare constrained vs unconstrained
    pass
```

---

## Expected Outcomes

### Variant A (Logit Decoding)
| Metric | Expected | Why |
|--------|----------|-----|
| Hallucination Rate | -15-30% | Mask prevents invalid tokens |
| Generation Speed | -5% | Logit ops overhead |
| Fluency | Same | Sampling unchanged |

### Variant B (DPO Data)
| Metric | Expected | Why |
|--------|----------|-----|
| Training Cost | Same | Standard DPO |
| Data Quality | TBD | Depends on constraint quality |
| Alignment | +10-20% | More focused preference data |

---

## Key Papers to Cite (Updated)

### Constrained Decoding
- Holtzman et al. (2020) - Nucleus sampling (original constrained decoding)
- Liu et al. (2022) - CFG for diffusion (inspiration for logit guidance)

### DPO / Alignment
- Rafailov et al. (2023) - DPO
- Bai et al. (2023) - RLAIF (AI feedback - our approach is similar)

### Logit-level Manipulation
- Anil et al. (2023) - Gemma logit control techniques
- - [Search for more]

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Logit masking doesn't work | Fall back to DPO data generation |
| Constraints too restrictive | Start with narrow domains |
| Can't beat baselines | Focus on analysis of why |
| No compute for training | Stick to Variant A (inference-only) |

---

## Success Criteria

### Minimum Viable Paper
- [ ] Logit processor framework implemented
- [ ] At least 2 constraint types working
- [ ] Demo on 1 benchmark (TruthfulQA or similar)
- [ ] Comparison with unconstrained baseline

### Strong Paper
- [ ] All three variants explored
- [ ] Multiple benchmarks
- [ ] Ablation study
- [ ] Human evaluation

### Ideal Paper
- [ ] All of above + SOTA comparison
- [ ] Theoretical analysis
- [ ] Open-source release

---

**Status**: 🟢 Pivot Complete - Ready to implement Variant A
