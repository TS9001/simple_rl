"""HuggingFace model loading utilities."""

import torch
from typing import Dict, Any, Optional, Tuple


def get_attention_implementation(config: Dict[str, Any]) -> Optional[str]:
    """
    Determine the appropriate attention implementation based on device.

    Args:
        config: Configuration dictionary with model settings

    Returns:
        Attention implementation name or None
    """
    model_config = config.get("model", {})
    attn_impl = model_config.get("attn_implementation")

    if attn_impl is not None:
        return attn_impl

    # Auto-select based on device
    if torch.cuda.is_available():
        return "flash_attention_2"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "sdpa"
    else:
        return None


def create_model_loader_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create loader kwargs for HuggingFace model loading.

    Args:
        config: Configuration dictionary with model settings

    Returns:
        Dictionary of loader arguments
    """
    model_config = config.get("model", {})
    loader_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }

    # Determine model dtype
    dtype_cfg = model_config.get("torch_dtype") or model_config.get("model_type")
    if dtype_cfg is not None:
        alias_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        if isinstance(dtype_cfg, str):
            target_dtype = alias_map.get(dtype_cfg.lower())
            if target_dtype is not None:
                loader_kwargs["torch_dtype"] = target_dtype
    else:
        # Default to bf16 on CUDA (better stability than fp16)
        # Default to fp32 on MPS (better stability)
        if torch.cuda.is_available():
            loader_kwargs["torch_dtype"] = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            loader_kwargs["torch_dtype"] = torch.float32

    # CUDA-specific optimizations
    if torch.cuda.is_available():
        loader_kwargs["device_map"] = "auto"
        loader_kwargs.setdefault("low_cpu_mem_usage", True)
        loader_kwargs.setdefault("use_cache", False)

    # Attention implementation
    attn_impl = get_attention_implementation(config)
    if attn_impl:
        loader_kwargs["attn_implementation"] = attn_impl

    return loader_kwargs


def setup_tokenizer_and_model_config(model, tokenizer) -> None:
    """
    Set up tokenizer and model configuration for decoder-only models.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
    """
    # Set up tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Use left padding by default

    # Set up model config
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id


def load_huggingface_model_and_tokenizer(
    model_name: str, config: Dict[str, Any], **additional_kwargs
) -> Tuple[Any, Any]:
    """
    Load HuggingFace model and tokenizer with configuration.

    Args:
        model_name: Name or path of the model
        config: Configuration dictionary
        **additional_kwargs: Additional arguments for model loading

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loader_kwargs = create_model_loader_kwargs(config)
    loader_kwargs.update(additional_kwargs)

    # Try loading with attention implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **loader_kwargs)
    except TypeError:
        # Remove attention implementation if not supported
        loader_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **loader_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=False,
        add_bos_token=False,
    )

    # Set up tokenizer and model config
    setup_tokenizer_and_model_config(model, tokenizer)

    return model, tokenizer


class ModelLoader:
    """HuggingFace model and tokenizer loader with configuration."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_name = config.get("model", {}).get("model_name")

    def load(self) -> Tuple[Any, Any]:
        """
        Load model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model_name is None:
            raise ValueError("Model name not specified in configuration")

        self.model, self.tokenizer = load_huggingface_model_and_tokenizer(
            self.model_name, self.config
        )

        return self.model, self.tokenizer

    def get_model(self):
        """Get the loaded model."""
        return self.model

    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self.tokenizer
