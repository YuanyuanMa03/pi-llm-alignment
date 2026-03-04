"""LoRA/QLoRA configuration utilities for efficient fine-tuning.

This module provides configuration classes and utilities for applying
LoRA (Low-Rank Adaptation) and QLoRA (4-bit quantized LoRA) to
language models for memory-efficient fine-tuning.
"""

from dataclasses import dataclass, field
from typing import Any

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import PreTrainedModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = object  # type: ignore
    PreTrainedModel = object  # type: ignore


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.

    Provides sensible defaults for different model sizes and
    convenient conversion to PEFT's LoraConfig format.

    Parameters
    ----------
    r : int
        LoRA rank (dimension of low-rank matrices).
        Higher values = more parameters but better capacity.
    lora_alpha : int
        LoRA scaling factor (typically 2*r).
    target_modules : list[str]
        Module names to apply LoRA to.
    lora_dropout : float
        Dropout probability for LoRA layers.
    bias : str
        Bias training strategy: "none", "all", or "lora_only".

    Examples
    --------
    >>> config = LoRAConfig.get_default_for_model("Qwen/Qwen2.5-7B")
    >>> peft_config = config.to_peft_config()
    >>> model = get_peft_model(model, peft_config)
    >>> model.print_trainable_parameters()
    """

    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self) -> "LoraConfig":
        """Convert to PEFT LoraConfig.

        Returns
        -------
        LoraConfig
            PEFT-compatible configuration.

        Raises
        ------
        ImportError
            If PEFT library is not available.
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library required for LoRA. "
                "Install with: pip install peft"
            )

        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type
        )

    @classmethod
    def get_default_for_model(cls, model_name: str) -> "LoRAConfig":
        """Get default LoRA config for specific model.

        Args:
            model_name: Model identifier (e.g., "Qwen/Qwen2.5-7B").

        Returns
        -------
        LoRAConfig
            Configuration optimized for the model size.
        """
        # Extract model size from name
        if "7B" in model_name or "6B" in model_name:
            return cls(
                r=32,
                lora_alpha=64,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            )
        elif "14B" in model_name or "13B" in model_name:
            return cls(
                r=64,
                lora_alpha=128,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            )
        else:  # Default for 2B-3B models
            return cls(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"]
            )


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (4-bit quantized LoRA).

    Extends LoRAConfig with quantization parameters for
    memory-efficient training on limited hardware.

    Parameters
    ----------
    load_in_4bit : bool
        Whether to load model in 4-bit precision.
    bnb_4bit_compute_dtype : str
        Compute dtype for 4-bit layers.
    bnb_4bit_quant_type : str
        Quantization type ("nf4" or "fp4").
    bnb_4bit_use_double_quant : bool
        Whether to use double quantization.
    """

    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def to_peft_config(self) -> "LoraConfig":
        """Convert to PEFT LoraConfig for QLoRA."""
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library required for LoRA. "
                "Install with: pip install peft"
            )

        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type
        )

    def get_load_kwargs(self) -> dict[str, Any]:
        """Get kwargs for loading model in 4-bit.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for AutoModelForCausalLM.from_pretrained.
        """
        try:
            import bitsandbytes as bnb  # noqa: F401
        except ImportError:
            raise ImportError(
                "bitsandbytes required for QLoRA. "
                "Install with: pip install bitsandbytes"
            )

        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_compute_dtype": getattr(torch, self.bnb_4bit_compute_dtype),
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
        }


def apply_lora_to_model(
    model: "PreTrainedModel",
    config: LoRAConfig
) -> "PreTrainedModel":
    """Apply LoRA adapters to a model.

    Args:
        model: Base model to apply LoRA to.
        config: LoRA configuration.

    Returns
    -------
    PreTrainedModel
        Model with LoRA adapters applied.

    Raises
    ------
    ImportError
        If PEFT library is not available.
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library required for LoRA. "
            "Install with: pip install peft"
        )

    peft_config = config.to_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


# Import torch for QLoRA dtype
import torch


__all__ = [
    "LoRAConfig",
    "QLoRAConfig",
    "apply_lora_to_model",
]
