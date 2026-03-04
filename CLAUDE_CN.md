# CLAUDE_CN.md - PI-LLM Alignment 项目规则

本文件为 Claude Code 提供 **PI-LLM Alignment** 项目的工作指南（中文版）。

---

## ⚠️ 重要规则

### 语言规则
**在与用户交流时，Claude Code 应尽量使用中文回答。**

- 所有回复、解释、建议都应使用中文
- 代码注释可以使用英文或中文
- 技术术语（如API名称、函数名）保留英文
- 错误信息可以保留英文原文，但应附上中文解释

---

## 项目概述

**研究问题**: 能否将物理信息机器学习（Physics-Informed ML）中的约束机制迁移到大语言模型对齐中，以减少LLM幻觉？

**核心洞察**: 物理信息ML中的硬约束机制可以通过两种方式迁移到语言模型：
1. **推理时干预** - 通过logit级别的token掩码
2. **训练时引导** - 为DPO自动生成偏好数据

**当前阶段**: Variant A（Logit约束解码）- 主要关注通过HuggingFace LogitsProcessor进行推理时约束执行。

---

## 研究方法：三个变体

| 变体 | 重点 | 状态 | 用途 |
|------|------|------|------|
| **A: Logit约束解码** | 推理时logit掩码 | ✅ **主要** | 通过LogitsProcessor实现零开销约束 |
| **B: 约束感知DPO** | 自动生成偏好数据 | 🚧 计划中 | 使用约束检查器为DPO生成评分 |
| **C: 领域特定** | 可验证约束的代码/数学生成 | 📋 备选 | 受限领域的客观评估 |

---

## 技术栈

### 核心依赖

```bash
# 机器学习核心
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0

# 测试
pytest>=7.4.0
pytest-cov>=4.1.0

# 开发工具
mypy>=1.5.0
ruff>=0.1.0
black>=23.7.0
```

### 类型提示要求

**所有新代码必须使用严格类型提示**：

```python
from typing import Protocol
from collections.abc import Sequence

class ConstraintChecker(Protocol):
    """约束接口协议。"""

    def check(self, text: str) -> bool: ...

    def compute_penalty(self, text: str) -> float: ...

    def get_logits_mask(self, current_sequence: list[int], vocab_size: int) -> torch.Tensor: ...
```

---

## 设计哲学

### 双模式约束设计

**所有约束必须支持两种操作模式**：

| 模式 | 目的 | 返回类型 | 用途 |
|------|------|----------|------|
| **评估模式** | 生成后评分 | `float` 惩罚值 | 训练损失、排序、评估 |
| **干预模式** | 推理时控制 | `Tensor` logit掩码 | 生成引导 |

```python
from abc import ABC, abstractmethod
import torch

class LLMConstraint(ABC):
    """双模式约束的抽象基类。"""

    @abstractmethod
    def check(self, text: str) -> bool:
        """检查生成的文本是否满足约束。"""
        ...

    @abstractmethod
    def compute_penalty(self, text: str) -> float:
        """计算约束违反的惩罚分数。"""
        ...

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """计算推理时干预的logit掩码。"""
        ...
```

---

## 项目结构

```
pi-llm-alignment/
├── src/
│   ├── constraints/              # 约束实现
│   │   ├── base.py               # LLMConstraint抽象类
│   │   ├── factual.py            # 数值边界、关键词约束
│   │   ├── temporal.py           # 时序约束
│   │   └── logical.py            # 逻辑规则约束
│   │
│   ├── inference/                # 推理时组件
│   │   └── processor.py          # ConstraintLogitProcessor
│   │
│   └── experiments/              # 实验脚本
│       └── demo.py               # 简单演示
│
├── tests/                        # 测试
│   ├── test_constraints/
│   └── test_inference/
│
├── models/                       # 本地模型存储
│   └── Qwen/
│       └── Qwen3___5-2B/         # Qwen3.5-2B本地模型
│
├── demo_qwen.py                  # 主演示脚本
├── demo_mock.py                  # 模拟演示（无需模型）
└── requirements.txt
```

---

## 常用命令

### 环境设置

```bash
# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 类型检查
mypy src/

# 代码检查
ruff check src/
ruff format src/
```

### 运行演示

```bash
# 快速模拟演示（无需下载模型）
python demo_mock.py

# 完整Qwen演示（需要本地模型）
python demo_qwen.py
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行带覆盖率的测试
pytest --cov=src --cov-report=html

# 运行特定测试
pytest tests/test_constraints/test_factual.py -v
```

---

## 本地模型设置

本地模型存储在 `models/Qwen/Qwen3___5-2B/`：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/Qwen/Qwen3___5-2B"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True  # 强制使用本地文件
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)
```

---

## 测试指南

### 测试结构

```python
# tests/test_constraints/test_factual.py

import pytest
import torch
from src.constraints.factual import NumericalBoundsConstraint

class TestNumericalBoundsConstraint:
    """NumericalBoundsConstraint测试套件。"""

    @pytest.fixture
    def simple_constraint(self):
        """提供测试约束的fixture。"""
        return NumericalBoundsConstraint(
            bounds={"value": (0, 100)},
            weight=1.0,
            enabled=True
        )

    def test_check_valid_input(self, simple_constraint):
        """测试有效输入通过检查。"""
        assert simple_constraint.check("The value is 50")

    def test_check_invalid_above_max(self, simple_constraint):
        """测试超过最大值失败检查。"""
        assert not simple_constraint.check("The value is 150")
```

---

## 添加新约束的工作流程

### 1. 创建约束类

```bash
touch src/constraints/my_constraint.py
```

### 2. 实现所有抽象方法

```python
from .base import LLMConstraint
import torch

class MyConstraint(LLMConstraint):
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer  # get_logits_mask()必需

    def check(self, text: str) -> bool:
        """对生成文本进行二进制约束检查。"""
        ...

    def compute_penalty(self, text: str) -> float:
        """用于DPO/RLAIF的连续惩罚分数。"""
        ...

    def get_logits_mask(
        self,
        current_sequence: list[int],
        vocab_size: int
    ) -> torch.Tensor:
        """返回下一步禁止的token的布尔掩码。

        关键：接收token ID而非文本，必须使用tokenizer解码。
        """
        # 解码当前序列以检查部分状态
        prefix = self.tokenizer.decode(current_sequence, skip_special_tokens=True)
        # ... 确定要掩码哪些token ...
        return mask
```

### 3. 编写测试

```bash
touch tests/test_constraints/test_my_constraint.py
```

### 4. 运行测试

```bash
pytest tests/test_constraints/test_my_constraint.py -v
mypy src/constraints/my_constraint.py
```

---

## 文档标准

### NumPy风格文档字符串

```python
def get_logits_mask(
    self,
    current_sequence: list[int],
    vocab_size: int
) -> torch.Tensor:
    """计算推理时干预的logit掩码。

    此方法返回一个布尔掩码，指示在下一个生成步骤
    应该禁止哪些token。

    Parameters
    ----------
    current_sequence : list of int
        到目前为止生成的token ID。
    vocab_size : int
        词汇表大小。

    Returns
    -------
    torch.Tensor
        形状为(vocab_size,)的布尔张量。标记为True的token
        将被掩码（logit设为-inf）。
    """
```

---

## Git提交约定

```bash
# 新功能
git commit -m "feat(constraints): 添加时序一致性约束"

# 修复bug
git commit -m "fix(inference): 修正logit处理器中的设备处理"

# 测试
git commit -m "test(constraints): 添加数值边界测试"

# 文档
git commit -m "docs: 更新CLAUDE_CN.md双模式设计"
```

---

## 项目状态

**当前阶段**: Variant A（Logit约束解码）- ✅ **主要实现已完成**

- [x] 项目结构
- [x] 基础约束接口（双模式支持）
- [x] 约束实现（数值边界、时序、关键词）
- [x] LogitProcessor框架
- [x] 演示脚本
- [x] 测试套件
- [ ] 评估框架（基准测试、指标）
- [ ] Variant B: DPO数据生成
- [ ] Variant C: 领域特定实验

**下一步**:
1. 在TruthfulQA等基准上运行评估
2. 实现消融研究框架
3. 探索Variant B（DPO数据生成）

---

## 目标发表会议

**主要目标**: ICLR / NeurIPS Workshop on Reliable & Responsible LLMs
**次要目标**: ACL / EMNLP（如果侧重NLP方面）

**核心差异点**:
1. **三变体方法**: Logit解码、DPO数据生成、领域特定
2. **零开销推理**: 最小性能影响的logit掩码
3. **物理信息迁移**: 从科学ML到语言的约束机制

---

## 关键设计决策

### 为什么get_logits_mask()接收token ID而非文本？

因为HuggingFace的LogitsProcessor在生成循环中传递的是token ID，而非解码后的文本。约束子类必须：
1. 使用tokenizer解码当前序列
2. 应用约束逻辑
3. 返回要掩码的token ID掩码

```python
def get_logits_mask(self, current_sequence: list[int], vocab_size: int) -> torch.Tensor:
    # 关键：必须先解码token ID为文本
    prefix = self.tokenizer.decode(current_sequence, skip_special_tokens=True)
    # 然后应用约束逻辑
    ...
```

### 为什么需要双模式设计？

1. **评估模式**（check/compute_penalty）: 用于生成后评分、训练损失、评估指标
2. **干预模式**（get_logits_mask）: 用于生成时引导，阻止违规token

这种设计使约束系统同时支持：
- Variant A（推理时干预）
- Variant B（DPO数据生成）

---

**最后更新**: 2026-03-04
**版本**: 0.1.0-alpha
