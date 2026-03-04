# PI-LLM Alignment：面向 LLM 幻觉抑制的物理信息约束

> **研究问题**：物理信息机器学习中的约束机制能否迁移到大语言模型对齐？

---

## 概览

本项目探索一种新的 LLM 对齐思路：将物理信息神经网络（PINNs）的约束执行机制迁移到语言模型。我们此前在 PI-KD（面向农业模型的物理信息知识蒸馏）中通过精细的损失设计实现了**零物理约束违背**。本研究旨在验证类似的约束方法能否降低 LLM 幻觉。

---

## 研究动机

| 问题 | 现有方案 | 我们的方案 |
|---------|------------------|--------------|
| **LLM 幻觉** | RLHF、DPO、Constitutional AI | **硬约束执行** |
| **仅软对齐** | 基于奖励的优化 | **混合：约束 + 目标** |
| **领域无关** | 通用方法 | **源自物理信息 ML 的迁移** |

### 关键洞察

物理信息 ML 已发展出一系列强约束机制，例如：
- **单调性**（量仅按预期方向增/减）
- **非负性**（物理量不能为负）
- **相关性**（变量维持预期关系）

这些约束在语言生成中存在直接类比：
- **时间一致性**（叙事事件按逻辑顺序发生）
- **事实边界**（无不可能的数值）
- **多模态一致性**（跨模态关系保持一致）

---

## 研究问题

### RQ1（核心）
物理信息约束机制能否有效降低 LLM 幻觉？

### RQ2（消融）
哪些约束类型从物理到语言迁移效果最好？

### RQ3（对比）
基于约束的对齐与 RLHF/DPO 相比，在以下方面表现如何：
- 幻觉降低
- 训练效率
- 输出质量

---

## 方法：PI-Align

我们提出的框架包含三个组成部分：

### 1. 约束形式化
```python
class LLMConstraint(ABC):
    """Base class for LLM output constraints"""
    @abstractmethod
    def check(self, text: str) -> bool: pass

    @abstractmethod
    def penalty(self, text: str) -> float: pass
```

### 2. 约束感知生成
- 生成过程中进行约束检查
- 违规时迭代修复
- 回溯到合法状态

### 3. 约束损失训练
```python
L_total = L_language + λ * L_constraint
```

---

## 项目结构

```
pi-llm-alignment/
├── README.md                   # 原始英文文档
├── CLAUDE.md                   # Claude Code 项目说明
├── requirements.txt            # 依赖项
│
├── src/
│   ├── constraints/
│   │   ├── base.py            # 抽象约束类
│   │   ├── temporal.py        # 时间一致性
│   │   ├── factual.py         # 事实边界
│   │   └── logical.py         # 逻辑规则
│   │
│   ├── models/
│   │   ├── constraint_aware_llm.py  # 主模型封装
│   │   └── training.py               # 约束训练循环
│   │
│   ├── evaluation/
│   │   ├── metrics.py         # 约束违规指标
│   │   └── benchmarks.py      # 基准测试运行器
│   │
│   └── experiments/
│       ├── demo.py            # 快速演示
│       ├── ablation.py        # 消融实验
│       └── comparison.py      # 基线对比
│
├── data/
│   └── constraints/            # 约束定义
│
├── results/
│   ├── logs/                  # 训练日志
│   └── figures/               # 生成图表
│
└── paper/
    ├── figures/               # 论文图表
    └── draft.tex             # 论文草稿
```

---

## 快速开始

### 安装
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

### 运行演示
```bash
# Simple constraint demo
python src/experiments/demo.py

# Full experiment
python src/experiments/comparison.py --benchmark truthfulqa
```

### 使用本地模型

离线开发可使用本地下载模型：

```bash
# The demo uses local Qwen3.5-2B model by default
python demo_qwen.py
```

模型存放路径：`models/Qwen/Qwen3___5-2B/`

如需使用其他模型：
```python
# Edit demo_qwen.py and change the model_name parameter
model, tokenizer = load_model("path/to/your/model")
# Or use HuggingFace: load_model("Qwen/Qwen2.5-3B-Instruct")
```

---

## 实验设置

### 基线方法
| 方法 | 描述 |
|--------|-------------|
| Vanilla LLM | 无对齐 |
| RLHF | 基于人类反馈的强化学习 |
| DPO | 直接偏好优化 |
| **PI-Align（我们）** | 物理信息约束对齐 |

### 基准数据集
| 数据集 | 重点 |
|---------|-------|
| TruthfulQA | 事实性 |
| HALU-EVAL | 幻觉检测 |
| BIG-Bench Hard | 推理 |
| Custom (Temporal) | 时间一致性 |

### 评估指标
- **约束违规率**（主要指标）
- 准确率 / F1
- 生成质量（困惑度、人类评测）
- 训练效率（时间、算力）

---

## 预期贡献

1. **新范式**：首次系统性地将物理信息约束迁移到 LLM 对齐
2. **框架**：可复用的语言模型约束形式化
3. **实证证据**：与现有对齐方法的全面对比
4. **分析**：哪些约束能迁移以及原因

---

## 时间规划

| 阶段 | 时长 | 里程碑 |
|-------|----------|-----------|
| **阶段 1** | 月份 1 | 文献综述 + 约束形式化 |
| **阶段 2** | 月份 2 | PI-Align 框架实现 |
| **阶段 3** | 月份 3 | 基准实验 |
| **阶段 4** | 月份 4 | 消融实验 + 分析 |
| **阶段 5** | 月份 5 | 研讨会论文提交 |
| **阶段 6** | 月份 6 | 终稿 + 会议版本 |

---

## 相关工作

- **物理信息 ML**：Raissi et al. (2019), PINNs
- **LLM 对齐**：Bai et al. (2022) RLHF, Touvron et al. (2023) LLaMA-2
- **受约束解码**：Holtzman et al. (2020) Nucleus sampling
- **我们此前的工作**：PI-KD 用于农业 CH4 预测

---

## 许可

MIT License

---

## 联系方式

Yuanyuan Ma - [GitHub](https://github.com/YuanyuanMa03)

---

## 引用

```bibtex
@misc{pi-llm-alignment,
  title={From Physics to Language: Constraint-Based Alignment for LLMs},
  author={Ma, Yuanyuan},
  year={2024},
  note={Research in progress}
}
```
