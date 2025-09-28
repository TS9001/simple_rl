"""Optimizer configuration utilities."""

import torch
from typing import Dict, Any, Optional, Union


def get_dtype_from_config(dtype_config: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    """
    Convert dtype configuration to torch.dtype.

    Args:
        dtype_config: String ("fp16", "bf16", "fp32") or torch.dtype object

    Returns:
        Corresponding torch.dtype or None
    """
    if dtype_config is None:
        return None

    alias_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    if isinstance(dtype_config, str):
        return alias_map.get(dtype_config.lower())
    elif isinstance(dtype_config, torch.dtype):
        return dtype_config
    else:
        raise ValueError(f"Unsupported dtype specification: {dtype_config!r}")


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: PyTorch model to optimize
        config: Configuration dictionary with optimizer settings

    Returns:
        Configured optimizer
    """
    optimizer_cfg = config.get("optimizer", {})
    optimizer_type = str(optimizer_cfg.get("type", "adam")).lower()
    lr = optimizer_cfg.get("lr", 1e-3)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    eps = optimizer_cfg.get("eps", 1e-8)

    optimizer_kwargs: Dict[str, Any] = {"lr": lr, "betas": betas, "eps": eps}

    if optimizer_type == "adamw":
        optimizer_kwargs["weight_decay"] = optimizer_cfg.get("weight_decay", 0.0)
        optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.Adam

    # Try fused optimizer if available and requested
    fused_requested = optimizer_cfg.get("fused", True)
    if torch.cuda.is_available() and fused_requested:
        optimizer_kwargs["fused"] = True

    try:
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    except TypeError:
        # Fused optimizer not available, remove fused parameter
        optimizer_kwargs.pop("fused", None)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    return optimizer


class OptimizerConfig:
    """Configuration manager for optimizers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimizer configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimizer = None

    def setup_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Set up optimizer for the given model.

        Args:
            model: PyTorch model to optimize

        Returns:
            Configured optimizer
        """
        self.optimizer = create_optimizer(model, self.config)
        return self.optimizer

    def get_optimizer(self) -> Optional[torch.optim.Optimizer]:
        """Get the configured optimizer."""
        return self.optimizer


def configure_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Configure optimizer for a model using configuration.

    Args:
        model: PyTorch model to optimize
        config: Configuration dictionary

    Returns:
        Configured optimizer
    """
    return create_optimizer(model, config)
