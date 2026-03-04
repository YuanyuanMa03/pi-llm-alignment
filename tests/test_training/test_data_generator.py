"""Tests for PreferencePairGenerator."""

import pytest

from src.training.data_generator import PreferencePairGenerator
from src.constraints.factual import NumericalBoundsConstraint
from src.constraints.temporal import TemporalOrderConstraint


class MockTokenizer:
    """模拟tokenizer。"""

    def __call__(self, text, return_tensors=None, return_attention_mask=None):
        # 简单的单词级别tokenization
        tokens = text.split()
        # 返回字典格式（模拟HuggingFace tokenizer输出）
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens)
        }

    def decode(self, tokens, skip_special_tokens=False):
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if isinstance(tokens, list):
            return " ".join(str(t) for t in tokens)
        return str(tokens)

    @property
    def pad_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 0

    @property
    def pad_token(self):
        return "<pad>"


class MockModel:
    """模拟模型。"""

    def __init__(self):
        self.device = "cpu"

    def eval(self):
        """模拟eval方法。"""
        return self

    def __call__(self, *args, **kwargs):
        return self

    def generate(self, inputs=None, **kwargs):
        # 返回模拟的token IDs
        return [[1, 2, 3, 4, 5]]


class TestPreferencePairGenerator:
    """PreferencePairGenerator测试套件。"""

    @pytest.fixture
    def mock_model(self):
        """提供模拟模型。"""
        return MockModel()

    @pytest.fixture
    def mock_tokenizer(self):
        """提供模拟tokenizer。"""
        return MockTokenizer()

    @pytest.fixture
    def simple_constraints(self):
        """提供简单约束。"""
        return [
            NumericalBoundsConstraint({"age": (0, 120)}),
            TemporalOrderConstraint(["first", "second", "third"])
        ]

    @pytest.fixture
    def generator(self, mock_model, mock_tokenizer, simple_constraints):
        """提供数据生成器实例。"""
        return PreferencePairGenerator(mock_model, mock_tokenizer, simple_constraints)

    def test_initialization(self, mock_model, mock_tokenizer, simple_constraints):
        """测试初始化。"""
        gen = PreferencePairGenerator(mock_model, mock_tokenizer, simple_constraints)
        assert gen.model is mock_model
        assert gen.tokenizer is mock_tokenizer
        assert len(gen.constraints) == 2

    def test_score_response_valid(self, generator):
        """测试评分有效回复。"""
        penalty = generator.score_response("The age is 25 years")
        assert penalty == 0.0

    def test_score_response_invalid(self, generator):
        """测试评分无效回复。"""
        penalty = generator.score_response("The age is 250 years")
        assert penalty > 0

    def test_score_response_temporal_violation(self, generator):
        """测试时序违规评分。"""
        penalty = generator.score_response("Second, then first")
        assert penalty > 0

    def test_create_preference_pair_valid(self, generator):
        """测试创建有效偏好对。"""
        responses = [
            "The age is 25 years",
            "The age is 150 years",
            "The age is 30 years"
        ]
        scores = [0.0, 0.25, 0.0]

        pair = generator.create_preference_pair("The age is ", responses, scores)

        assert pair is not None
        assert pair["prompt"] == "The age is "
        assert "chosen" in pair
        assert "rejected" in pair
        assert pair["chosen_score"] <= pair["rejected_score"]

    def test_create_preference_pair_insufficient_margin(self, generator):
        """测试边际不足时的处理。"""
        responses = [
            "The age is 25 years",
            "The age is 26 years"
        ]
        scores = [0.0, 0.05]  # 边际小于默认的0.1

        pair = generator.create_preference_pair(
            "The age is ", responses, scores, min_margin=0.1
        )

        # 仍然应该返回一个pair（带low_margin标记）
        assert pair is not None
        assert pair.get("low_margin", False)

    def test_create_preference_pair_single_response(self, generator):
        """测试只有单个响应时返回None。"""
        pair = generator.create_preference_pair(
            "Test ",
            ["Single response"],
            [0.0]
        )
        assert pair is None

    def test_generate_pairs(self, generator):
        """测试批量生成偏好对。"""
        prompts = ["The age is ", "The value is "]

        pairs = generator.generate_pairs(
            prompts=prompts,
            n_samples=2,
            generation_kwargs={"max_new_tokens": 10}
        )

        # 注意：由于使用mock模型，可能不会生成有效对
        # 这里主要测试不会崩溃
        assert isinstance(pairs, list)

    def test_generate_dataset_format(self, generator):
        """测试数据集格式符合TRL要求。"""
        prompts = ["The age is "]

        dataset = generator.generate_dataset(
            prompts=prompts,
            n_samples_per_prompt=2,
            generation_kwargs={"max_new_tokens": 10}
        )

        # 检查格式
        for item in dataset:
            assert "prompt" in item
            assert "chosen" in item
            assert "rejected" in item
            # 不应包含分数（TRL格式不需要）
            assert "chosen_score" not in item
            assert "rejected_score" not in item
