"""Training modules for PI-LLM Alignment.

This package provides constraint-aware training utilities including:
- PreferencePairGenerator: Generate DPO training data from constraints
- ConstraintAwareDPOTrainer: DPO trainer with constraint-aware loss
- LoRAConfig: LoRA/QLoRA configuration utilities
"""

from .data_generator import PreferencePairGenerator
from .lora_config import LoRAConfig, QLoRAConfig, apply_lora_to_model

__all__ = [
    "PreferencePairGenerator",
    "LoRAConfig",
    "QLoRAConfig",
    "apply_lora_to_model",
]
