# Literature Review: Physics-Informed LLM Alignment

> **Goal**: Systematically survey existing work on LLM alignment and physics-informed ML to identify research gaps and position our contribution.

---

## Research Themes

1. **LLM Alignment Methods**
   - RLHF (Reinforcement Learning from Human Feedback)
   - DPO (Direct Preference Optimization)
   - Constitutional AI
   - Constrained Decoding

2. **Physics-Informed Machine Learning**
   - PINNs (Physics-Informed Neural Networks)
   - Constraint enforcement mechanisms
   - Knowledge distillation with constraints

3. **Our Position**
   - Constraint transfer from physical to linguistic domains
   - Hard vs soft alignment

---

## Theme 1: LLM Alignment

### 1.1 RLHF (Reinforcement Learning from Human Feedback)

| Paper | Venue | Year | Key Insights | Connection to Our Work |
|-------|-------|------|--------------|----------------------|
| Bai et al. "Training a Helpful and Harmless Assistant with RLHF" | arXiv | 2022 | | |
| Ouyang et al. "Training language models to follow instructions with human feedback" | NeurIPS | 2022 | | |
| Touvron et al. "LLaMA 2: Open Foundation and Fine-Tuned Chat Models" | arXiv | 2023 | | |

**Summary**: [Fill in after reading]

**Limitations**:
- [ ] What are the main limitations of RLHF?
- [ ] How does it handle constraint violations?

---

### 1.2 DPO (Direct Preference Optimization)

| Paper | Venue | Year | Key Insights | Connection |
|-------|-------|------|--------------|------------|
| Rafailov et al. "Direct Preference Optimization" | arXiv | 2023 | | |
|  |  |  |  |  |

**Summary**: [Fill in after reading]

---

### 1.3 Constitutional AI

| Paper | Venue | Year | Key Insights | Connection |
|-------|-------|------|--------------|------------|
| Bai et al. "Constitutional AI" | arXiv | 2023 | | |

**Summary**: [Fill in after reading]

---

### 1.4 Constrained Decoding

| Paper | Venue | Year | Key Insights | Connection |
|-------|-------|------|--------------|------------|
| Holtzman et al. "The Curious Case of Neural Text Degeneration" | ACL | 2020 | | |
|  |  |  |  |  |

**Summary**: [Fill in after reading]

---

## Theme 2: Physics-Informed ML

### 2.1 PINNs (Physics-Informed Neural Networks)

| Paper | Venue | Year | Key Insights | Connection to Our Work |
|-------|-------|------|--------------|----------------------|
| Raissi et al. "Physics-informed neural networks" | JCP | 2019 | | Base methodology |
| Karniadakis et al. "Physics-informed machine learning" | Nat Rev Phys | 2021 | | Survey |
|  |  |  |  |  |

**Summary**: [Fill in after reading]

**Constraint Mechanisms Identified**:
- [ ] Monotonicity constraints
- [ ] Boundary condition enforcement
- [ ] Physical law embedding
- [ ] Conservation laws

---

### 2.2 Constraint Enforcement in ML

| Paper | Venue | Year | Key Insights | Connection |
|-------|-------|------|--------------|------------|
|  |  |  |  |  |

**Summary**: [Fill in after reading]

---

### 2.3 Knowledge Distillation with Constraints

| Paper | Venue | Year | Key Insights | Connection |
|-------|-------|------|--------------|------------|
|  |  |  |  |  |

**Summary**: [Fill in after reading]

---

## Theme 3: Hallucination Detection & Mitigation

| Paper | Venue | Year | Key Insights | Connection |
|-------|-------|------|--------------|------------|
| Ji et al. "Survey on Hallucination in Large Language Models" | arXiv | 2023 | Survey | |
| Zhang et al. "Detecting Pre-training Data from Large Language Models" | ACL | 2023 | | |
|  |  |  |  |  |

**Summary**: [Fill in after reading]

---

## Theme 4: Cross-Domain Knowledge Transfer

| Paper | Venue | Year | Key Insights | Connection |
|-------|-------|------|--------------|------------|
|  |  |  |  |  |

**Summary**: [Fill in after reading]

---

## Research Gap Analysis

### What's Missing?

1. **Gap 1**: [Identify after literature review]
   - Existing work focuses on...
   - Our contribution: ...

2. **Gap 2**: [Identify after literature review]
   - Existing work focuses on...
   - Our contribution: ...

3. **Gap 3**: [Identify after literature review]
   - Existing work focuses on...
   - Our contribution: ...

---

## Related Work Categories

### Directly Relevant (Must Read)
- [ ] RLHF papers (Bai et al., 2022; Ouyang et al., 2022)
- [ ] PINN fundamentals (Raissi et al., 2019)
- [ ] Constrained decoding (Holtzman et al., 2020)

### Indirectly Relevant (Should Read)
- [ ] DPO and alternatives
- [ ] Hallucination surveys
- [ ] Constitutional AI

### Background (Nice to Have)
- [ ] Knowledge distillation literature
- [ ] Constraint optimization

---

## Reading Order

### Week 1: Foundations
1. [ ] Raissi et al. (2019) - PINNs basics
2. [ ] Bai et al. (2022) - RLHF
3. [ ] Holtzman et al. (2020) - Constrained decoding

### Week 2: LLM Alignment
1. [ ] Ouyang et al. (2022) - InstructGPT
2. [ ] Touvron et al. (2023) - LLaMA 2
3. [ ] Rafailov et al. (2023) - DPO

### Week 3: Hallucination & Evaluation
1. [ ] Ji et al. (2023) - Hallucination survey
2. [ ] TruthfulQA paper (Lin et al., 2022)
3. [ ] HALU-EVAL papers

---

## Notes Template

### Paper: [Title]

**Authors**: [Names]
**Venue**: [Conference/Journal] [Year]
**PDF**: [Link]

**Key Problem**: [What problem does it solve?]

**Method**: [Brief description of approach]

**Results**: [Main findings]

**Relevance to Our Work**: [Why is this important?]

**Limitations**: [What are the limitations?]

**Potential Borrowing**: [What can we use/adapt?]

---

## Citation Manager

Use BibTeX for managing citations:

```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational Physics},
  year={2019}
}

@article{bai2022training,
  title={Training a helpful and harmless assistant with reinforcement learning from human feedback},
  author={Bai, Yuntao and Jones, Andy and Ndousse, Kamal and Askell, Amanda and Chen, Anna and DasSarma, Stanislav and Drain, Dawn and Fort, Stanislav and Ganguli, Deepak and Henighan, Tom and Hume, Tom and Joseph, Nicholas and Kernion, Jackson and Khoury, Nova and Lovitt, Luke and Mann, Ben and Power, Alethea and others},
  journal={arXiv preprint arXiv:2204.05862},
  year={2022}
}
```

---

## Progress Tracking

- [ ] Theme 1: LLM Alignment (0/4 sections)
- [ ] Theme 2: Physics-Informed ML (0/3 sections)
- [ ] Theme 3: Hallucination (0/1 sections)
- [ ] Theme 4: Cross-Domain Transfer (0/1 sections)

**Overall Progress**: 0/9 sections complete

---

## Next Actions

1. [ ] Download PDFs for "Directly Relevant" papers
2. [ ] Set up citation manager (Zotero, BibTeX, etc.)
3. [ ] Start with Raissi et al. (2019) - PINNs
4. [ ] Take structured notes using template above

---

## Search Queries for Finding Related Work

```
# LLM Alignment
"reinforcement learning from human feedback" LLM
"direct preference optimization" language models
"constitutional AI" Anthropic
"constrained decoding" language generation

# Physics-Informed ML
"physics-informed neural networks" PINN
"constraint enforcement" neural networks
"knowledge distillation" physics

# Hallucination
"hallucination" large language models survey
"factuality" language models
"faithfulness" neural text generation
```

---

## Conferences/Journals to Monitor

**ML/AI**: NeurIPS, ICML, ICLR, AAAI
**NLP**: ACL, EMNLP, NAACL
**Applied**: AI in Agriculture, Computers & Electronics in Agriculture
**Journals**: JMLR, TMLR, Neural Computation

---

**Last Updated**: 2024-03-03
