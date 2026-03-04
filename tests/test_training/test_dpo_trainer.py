"""Tests for ConstraintAwareDPOTrainer."""

import pytest
import sys
from unittest.mock import MagicMock
import torch

# Create a proper mock DPOTrainer base class
class MockDPOTrainer:
    """Mock DPOTrainer base class for testing."""

    def __init__(self, model=None, ref_model=None, tokenizer=None, **kwargs):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.state = MagicMock()
        self.state.log_history = []

    def compute_loss(self, *args, **kwargs):
        """Return mock DPO loss."""
        return torch.tensor(1.0, requires_grad=True)

# Mock TRL module
mock_trl = MagicMock()
mock_trl.DPOTrainer = MockDPOTrainer
sys.modules["trl"] = mock_trl

from src.training.dpo_trainer import ConstraintAwareDPOTrainer
from src.constraints.factual import NumericalBoundsConstraint


class MockModel:
    """Mock model."""
    def __init__(self):
        self.device = torch.device("cpu")


class TestConstraintAwareDPOTrainer:
    """ConstraintAwareDPOTrainer测试套件。"""

    @pytest.fixture
    def mock_model(self):
        """提供模拟模型。"""
        return MockModel()

    @pytest.fixture
    def mock_ref_model(self):
        """提供模拟参考模型。"""
        return MockModel()

    @pytest.fixture
    def simple_constraints(self):
        """提供简单约束。"""
        return [
            NumericalBoundsConstraint({"age": (0, 120)})
        ]

    @pytest.fixture
    def trainer(self, mock_model, mock_ref_model, simple_constraints):
        """提供训练器实例。"""
        return ConstraintAwareDPOTrainer(
            model=mock_model,
            ref_model=mock_ref_model,
            constraints=simple_constraints,
            constraint_weight=0.1
        )

    def test_initialization(self, mock_model, mock_ref_model, simple_constraints):
        """测试初始化。"""
        trainer = ConstraintAwareDPOTrainer(
            model=mock_model,
            ref_model=mock_ref_model,
            constraints=simple_constraints,
            constraint_weight=0.5
        )

        assert len(trainer.constraints) == 1
        assert trainer.constraint_weight == 0.5

    def test_compute_constraint_penalty(self, trainer):
        """测试约束惩罚计算。"""
        texts = [
            "The age is 25",
            "The age is 150",
            "The age is 30"
        ]

        penalties = trainer.compute_constraint_penalty(texts)

        assert len(penalties) == 3
        assert penalties[0] == 0.0  # 有效
        assert penalties[1] > 0     # 无效
        assert penalties[2] == 0.0  # 有效

    def test_compute_constraint_penalty_disabled_constraint(self, mock_model, mock_ref_model):
        """测试禁用约束不产生惩罚。"""
        constraint = NumericalBoundsConstraint({"age": (0, 120)}, enabled=False)
        trainer = ConstraintAwareDPOTrainer(
            model=mock_model,
            ref_model=mock_ref_model,
            constraints=[constraint],
            constraint_weight=0.1
        )

        penalties = trainer.compute_constraint_penalty(["The age is 150"])
        assert penalties[0] == 0.0

    def test_compute_constraint_penalty_multiple_constraints(self, mock_model, mock_ref_model):
        """测试多个约束的惩罚累加。"""
        constraints = [
            NumericalBoundsConstraint({"age": (0, 120)}, weight=1.0),
            NumericalBoundsConstraint({"age": (0, 120)}, weight=2.0)
        ]
        trainer = ConstraintAwareDPOTrainer(
            model=mock_model,
            ref_model=mock_ref_model,
            constraints=constraints,
            constraint_weight=0.1
        )

        penalties = trainer.compute_constraint_penalty(["The age is 150"])
        # 两个约束都应该惩罚，第二个权重更高
        assert penalties[0] > 0
