"""Optimizer configuration utilities."""

import torch
from typing import Dict, Any, Optional, Tuple


def configure_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
    logger=None
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    """Create optimizer and optional LR scheduler from config."""
    optimizer_cfg = config.get("optimizer", {})
    optimizer_type = str(optimizer_cfg.get("type", "adam")).lower()
    lr = optimizer_cfg.get("lr", 1e-3)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    eps = optimizer_cfg.get("eps", 1e-8)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)

    optimizer_kwargs = {
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay
    }

    optimizer_class = torch.optim.AdamW if optimizer_type == "adamw" else torch.optim.Adam

    if torch.cuda.is_available() and optimizer_cfg.get("fused", True):
        optimizer_kwargs["fused"] = True

    try:
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    warmup_steps = optimizer_cfg.get("warmup_steps", 0)
    scheduler = None

    if warmup_steps > 0:
        warmup_start_lr = optimizer_cfg.get("warmup_start_lr", 1e-8)
        warmup_type = optimizer_cfg.get("warmup_type", "linear")

        if warmup_type == "linear":
            start_factor = warmup_start_lr / lr
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            if logger:
                logger.info(f"✓ LR warmup: {warmup_steps} steps, {warmup_start_lr:.2e} → {lr:.2e}")
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=warmup_start_lr / lr,
                total_iters=warmup_steps
            )
            if logger:
                logger.info(f"✓ LR warmup (constant): {warmup_steps} steps at {warmup_start_lr:.2e}")

    return optimizer, scheduler
