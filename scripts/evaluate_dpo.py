#!/usr/bin/env python3
"""约束感知模型评估脚本

评估微调前后模型的约束遵守情况。
"""

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加src到路径
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import ConstraintAwareEvaluator
from src.constraints import create_constraint


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="评估微调模型的约束遵守情况"
    )

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="微调后的模型路径"
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        required=True,
        help="基础模型路径（用于对比）"
    )

    # 数据参数
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="评估提示词JSON文件"
    )

    # 约束参数
    parser.add_argument(
        "--constraint_config",
        type=str,
        required=True,
        help="约束配置JSON文件"
    )

    # 输出参数
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="评估结果输出文件（JSON格式）"
    )

    # 生成参数
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度"
    )

    return parser.parse_args()


def load_constraints(config_path: str, tokenizer=None) -> list:
    """从配置文件加载约束。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    constraints = []
    for constraint_config in config.get("constraints", []):
        constraint_type = constraint_config["type"]
        params = constraint_config.get("params", {}).copy()

        if "tokenizer" not in params and constraint_type == "NumericalBoundsConstraint":
            params["tokenizer"] = tokenizer

        try:
            constraint = create_constraint(constraint_type, **params)
            constraints.append(constraint)
        except Exception as e:
            print(f"警告: 无法创建约束 {constraint_type}: {e}")

    return constraints


def main():
    """主评估函数。"""
    args = parse_args()

    # 设备检测
    if torch.backends.mps.is_available():
        device = "mps"
        print("🍎 使用 Apple Silicon GPU 加速 (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 使用 CUDA GPU 加速")
    else:
        device = "cpu"
        print("⚠️  使用 CPU")

    print("\n" + "=" * 60)
    print("  约束感知模型评估")
    print("=" * 60)

    # 加载tokenizer
    print(f"\n📥 加载 tokenizer: {args.baseline_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.baseline_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载约束
    print(f"\n🔒 加载约束配置: {args.constraint_config}")
    constraints = load_constraints(args.constraint_config, tokenizer)
    print(f"✅ 已加载 {len(constraints)} 个约束")

    # 加载评估提示词
    print(f"\n📝 加载评估提示词: {args.data_file}")
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data.get("prompts", [])]
    print(f"✅ 已加载 {len(prompts)} 个提示词")

    # 生成参数
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True
    }

    # 加载基础模型
    print(f"\n📥 加载基础模型: {args.baseline_model}")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.baseline_model,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ 基础模型已加载")

    # 加载微调模型
    print(f"\n📥 加载微调模型: {args.model_path}")
    try:
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ 微调模型已加载")
    except Exception as e:
        print(f"⚠️  无法加载微调模型: {e}")
        print("将只评估基础模型")
        finetuned_model = None

    # 评估基础模型
    print("\n" + "-" * 40)
    print("评估基础模型...")
    print("-" * 40)

    baseline_evaluator = ConstraintAwareEvaluator(
        baseline_model, tokenizer, constraints
    )
    baseline_metrics = baseline_evaluator.evaluate(prompts, generation_kwargs)

    print(f"\n📊 基础模型结果:")
    print(f"   整体违规率: {baseline_metrics['overall_violation_rate']:.2%}")
    print(f"   平均响应长度: {baseline_metrics['response_length']['mean']:.1f} 词")

    if baseline_metrics['violation_rate']:
        print(f"\n   各约束违规率:")
        for name, rate in baseline_metrics['violation_rate'].items():
            print(f"      {name}: {rate:.2%}")

    # 评估微调模型（如果可用）
    finetuned_metrics = None
    if finetuned_model is not None:
        print("\n" + "-" * 40)
        print("评估微调模型...")
        print("-" * 40)

        finetuned_evaluator = ConstraintAwareEvaluator(
            finetuned_model, tokenizer, constraints
        )
        finetuned_metrics = finetuned_evaluator.evaluate(prompts, generation_kwargs)

        print(f"\n📊 微调模型结果:")
        print(f"   整体违规率: {finetuned_metrics['overall_violation_rate']:.2%}")
        print(f"   平均响应长度: {finetuned_metrics['response_length']['mean']:.1f} 词")

        if finetuned_metrics['violation_rate']:
            print(f"\n   各约束违规率:")
            for name, rate in finetuned_metrics['violation_rate'].items():
                print(f"      {name}: {rate:.2%}")

    # 对比结果
    if finetuned_metrics is not None:
        print("\n" + "=" * 60)
        print("  对比结果")
        print("=" * 60)

        before_rate = baseline_metrics['overall_violation_rate']
        after_rate = finetuned_metrics['overall_violation_rate']

        print(f"\n整体违规率:")
        print(f"   基础模型: {before_rate:.2%}")
        print(f"   微调模型: {after_rate:.2%}")

        if before_rate > 0:
            improvement = (before_rate - after_rate) / before_rate * 100
            print(f"   改进: {improvement:+.1f}%")
        else:
            print(f"   变化: {(after_rate - before_rate):.2%}")

        # 约束级别对比
        if baseline_metrics['violation_rate'] and finetuned_metrics['violation_rate']:
            print(f"\n各约束违规率对比:")
            for name in baseline_metrics['violation_rate']:
                before = baseline_metrics['violation_rate'].get(name, 0)
                after = finetuned_metrics['violation_rate'].get(name, 0)
                print(f"   {name}:")
                print(f"      基础: {before:.2%} → 微调: {after:.2%}")
                if before > 0:
                    improvement = (before - after) / before * 100
                    print(f"      改进: {improvement:+.1f}%")

    # 保存结果
    results = {
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics
    }

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到: {args.output_file}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
