#!/usr/bin/env python3
"""约束感知DPO训练脚本

使用约束系统自动生成偏好数据并训练模型。
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

from src.training.data_generator import PreferencePairGenerator
from src.training.lora_config import LoRAConfig, QLoRAConfig, apply_lora_to_model
from src.constraints import create_constraint


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用约束感知DPO训练模型"
    )

    # 模型参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="models/Qwen/Qwen3___5-2B",
        help="模型路径或HuggingFace模型ID"
    )

    # 数据参数
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="包含提示词的JSON文件路径"
    )
    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=8,
        help="每个提示词生成的回复数量"
    )
    parser.add_argument(
        "--min_margin",
        type=float,
        default=0.1,
        help="chosen和rejected之间的最小分数差距"
    )

    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="模型检查点输出目录"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="训练批次大小"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="学习率"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="预热比例"
    )

    # LoRA参数
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="使用LoRA微调"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="使用QLoRA（4bit量化）"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA秩"
    )

    # 约束参数
    parser.add_argument(
        "--constraint_config",
        type=str,
        required=True,
        help="约束配置JSON文件路径"
    )
    parser.add_argument(
        "--constraint_weight",
        type=float,
        default=0.1,
        help="约束惩罚在损失中的权重"
    )

    # 生成参数
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度"
    )
    parser.add_argument(
        "--generation_temperature",
        type=float,
        default=0.8,
        help="回复生成的温度参数"
    )

    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用的设备（不指定则自动检测）"
    )

    return parser.parse_args()


def load_constraints(config_path: str, tokenizer=None) -> list:
    """从配置文件加载约束。

    Args:
        config_path: 约束配置JSON路径。
        tokenizer: 需要tokenizer的约束使用。

    Returns:
        约束实例列表。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    constraints = []
    for constraint_config in config.get("constraints", []):
        constraint_type = constraint_config["type"]
        params = constraint_config.get("params", {}).copy()

        # 添加tokenizer（如果约束需要）
        if "tokenizer" not in params and constraint_type in ["NumericalBoundsConstraint"]:
            params["tokenizer"] = tokenizer

        # 创建约束
        try:
            constraint = create_constraint(constraint_type, **params)
            constraints.append(constraint)
        except Exception as e:
            print(f"警告: 无法创建约束 {constraint_type}: {e}")

    return constraints


def main():
    """主训练函数。"""
    args = parse_args()

    # 设备设置
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("🍎 使用 Apple Silicon GPU 加速 (MPS)")
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 使用 CUDA GPU 加速")
        else:
            device = "cpu"
            print("⚠️  使用 CPU（将会很慢）")
    else:
        device = args.device
        print(f"使用设备: {device}")

    # 加载模型和tokenizer
    print(f"\n📥 加载模型: {args.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model_load_kwargs = {
        "torch_dtype": torch.float16 if device in ["cuda", "mps"] else torch.float32,
        "device_map": "auto",
        "trust_remote_code": True
    }

    # QLoRA加载参数
    if args.use_qlora:
        qlora_config = QLoRAConfig(r=args.lora_r)
        model_load_kwargs.update(qlora_config.get_load_kwargs())

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_load_kwargs
    )

    print(f"✅ 模型已加载")

    # 应用LoRA（如果需要）
    if args.use_qlora:
        print(f"🔧 使用 QLoRA (rank={args.lora_r}, 4bit量化)")
        lora_config = QLoRAConfig(r=args.lora_r)
        model = apply_lora_to_model(model, lora_config)
    elif args.use_lora:
        print(f"🔧 使用 LoRA (rank={args.lora_r})")
        lora_config = LoRAConfig.get_default_for_model(args.model_name_or_path)
        lora_config.r = args.lora_r
        model = apply_lora_to_model(model, lora_config)

    # 加载约束
    print(f"\n🔒 加载约束配置: {args.constraint_config}")
    constraints = load_constraints(args.constraint_config, tokenizer)
    print(f"✅ 已加载 {len(constraints)} 个约束")

    # 打印约束信息
    for i, constraint in enumerate(constraints, 1):
        print(f"   {i}. {constraint.__class__.__name__}")

    # 加载提示词
    print(f"\n📝 加载提示词: {args.data_file}")
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data.get("prompts", [])]
    print(f"✅ 已加载 {len(prompts)} 个提示词")

    # 生成偏好数据集
    print(f"\n🔄 生成偏好数据集...")
    print(f"   每个提示词生成 {args.num_samples_per_prompt} 个回复")

    generator = PreferencePairGenerator(model, tokenizer, constraints)

    train_dataset = generator.generate_dataset(
        prompts=prompts,
        n_samples_per_prompt=args.num_samples_per_prompt,
        generation_kwargs={
            "temperature": args.generation_temperature,
            "max_new_tokens": args.max_length,
        },
        min_margin=args.min_margin
    )

    print(f"✅ 已生成 {len(train_dataset)} 个偏好对")

    # 打印示例
    if train_dataset:
        print("\n📊 数据集示例:")
        example = train_dataset[0]
        print(f"   提示词: {repr(example['prompt'][:50])}")
        print(f"   选中: {repr(example['chosen'][:50])}")
        print(f"   拒绝: {repr(example['rejected'][:50])}")

    # 保存数据集
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "train_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)
    print(f"\n💾 数据集已保存到: {dataset_path}")

    # 创建参考模型
    print(f"\n🔄 创建参考模型...")

    ref_model_kwargs = {
        "torch_dtype": torch.float16 if device in ["cuda", "mps"] else torch.float32,
        "device_map": "auto",
        "trust_remote_code": True
    }

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **ref_model_kwargs
    )
    print(f"✅ 参考模型已创建")

    # 保存训练配置
    config = {
        "model_name_or_path": args.model_name_or_path,
        "num_samples_per_prompt": args.num_samples_per_prompt,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "constraint_weight": args.constraint_weight,
        "use_lora": args.use_lora,
        "use_qlora": args.use_qlora,
        "lora_r": args.lora_r,
        "dataset_size": len(train_dataset),
        "num_constraints": len(constraints)
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"💾 配置已保存到: {config_path}")

    print("\n" + "=" * 60)
    print("  准备工作完成！")
    print("=" * 60)
    print("\n📌 下一步:")
    print("   数据集已生成并保存")
    print("   如需完整DPO训练，请安装 trl 库:")
    print("   pip install trl")
    print(f"\n   数据集位置: {dataset_path}")
    print(f"   配置位置: {config_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
