import torch
from typing import Dict, Any, Optional, Tuple


def get_attention_implementation(config: Dict[str, Any]) -> Optional[str]:
    model_config = config.get("model", {})
    attn_impl = model_config.get("attn_implementation")

    if attn_impl is not None:
        return attn_impl

    if torch.cuda.is_available():
        return "flash_attention_2"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "sdpa"
    else:
        return None


def create_model_loader_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = config.get("model", {})
    loader_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }

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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id


def load_huggingface_model_and_tokenizer(
    model_name: str, config: Dict[str, Any], **additional_kwargs
) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loader_kwargs = create_model_loader_kwargs(config)
    loader_kwargs.update(additional_kwargs)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **loader_kwargs)
    except TypeError:
        loader_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **loader_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=False,
        add_bos_token=False,
    )

    setup_tokenizer_and_model_config(model, tokenizer)

    return model, tokenizer
