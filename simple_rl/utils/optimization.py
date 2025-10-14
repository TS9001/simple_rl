"""Optimizer configuration utilities."""

import torch
from typing import Dict, Any, Optional, Union, Tuple


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


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Create optimizer and optional learning rate scheduler from configuration.

    Args:
        model: PyTorch model to optimize
        config: Configuration dictionary with optimizer settings

    Returns:
        Tuple of (optimizer, optional_scheduler)

    Warmup configuration (in config["optimizer"]):
        - warmup_steps: Number of warmup steps (default: 0, no warmup)
        - warmup_start_lr: Starting LR for warmup (default: 1e-8)
        - warmup_type: "linear" or "constant" (default: "linear")
    """
    optimizer_cfg = config.get("optimizer", {})
    optimizer_type = str(optimizer_cfg.get("type", "adam")).lower()
    lr = optimizer_cfg.get("lr", 1e-3)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    eps = optimizer_cfg.get("eps", 1e-8)

    optimizer_kwargs: Dict[str, Any] = {"lr": lr, "betas": betas, "eps": eps}

    # Apply weight_decay for both Adam and AdamW (both support it)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    if weight_decay > 0:
        optimizer_kwargs["weight_decay"] = weight_decay

    if optimizer_type == "adamw":
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

    # Create warmup scheduler if requested
    warmup_steps = optimizer_cfg.get("warmup_steps", 0)
    scheduler = None

    if warmup_steps > 0:
        warmup_start_lr = optimizer_cfg.get("warmup_start_lr", 1e-8)
        warmup_type = optimizer_cfg.get("warmup_type", "linear")

        if warmup_type == "linear":
            # LinearLR: LR goes from start_factor * base_lr to base_lr over warmup_steps
            start_factor = warmup_start_lr / lr
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            print(f"✓ LR warmup enabled: {warmup_steps} steps, {warmup_start_lr:.2e} → {lr:.2e}")
        else:
            # Constant warmup: stay at warmup_start_lr for warmup_steps, then jump to lr
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=warmup_start_lr / lr,
                total_iters=warmup_steps
            )
            print(f"✓ LR warmup enabled (constant): {warmup_steps} steps at {warmup_start_lr:.2e}")

    return optimizer, scheduler


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


def configure_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Configure optimizer and optional scheduler for a model using configuration.

    Args:
        model: PyTorch model to optimize
        config: Configuration dictionary

    Returns:
        Tuple of (optimizer, optional_scheduler)
    """
    return create_optimizer(model, config)
