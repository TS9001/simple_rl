import torch
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import asdict

from simple_rl.dto import CheckpointData

if TYPE_CHECKING:
    from simple_rl.algorithms.grpo import GRPO


def save_checkpoint(algorithm, path: str) -> None:
    """Save checkpoint for GRPO algorithm."""
    from dataclasses import asdict

    ref_policy_state = algorithm.ref_policy.state_dict() if algorithm.ref_policy is not None else None

    checkpoint = CheckpointData(
        policy_state_dict=algorithm.policy.state_dict(),
        ref_policy_state_dict=ref_policy_state,
        optimizer_state_dict=algorithm.optimizer.state_dict(),
        scheduler_state_dict=algorithm.lr_scheduler.state_dict() if algorithm.lr_scheduler is not None else None,
        scaler_state_dict=algorithm.grad_scaler.state_dict() if algorithm.grad_scaler is not None else None,
        config=algorithm.config,
        total_steps=algorithm.total_steps,
        episode=algorithm.episode,
        current_episode=algorithm.current_episode,
    )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(checkpoint), path)


def load_checkpoint(algorithm, path: str) -> None:
    """Load checkpoint for GRPO algorithm."""
    checkpoint = torch.load(path, map_location=algorithm.device)
    algorithm.policy.load_state_dict(checkpoint["policy_state_dict"])
    if algorithm.ref_policy is not None and checkpoint.get("ref_policy_state_dict") is not None:
        algorithm.ref_policy.load_state_dict(checkpoint["ref_policy_state_dict"])
    algorithm.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if algorithm.grad_scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        algorithm.grad_scaler.load_state_dict(checkpoint["scaler_state_dict"])

    checkpoint_lr = algorithm.optimizer.param_groups[0]['lr']
    config_lr = algorithm.config.get("optimizer", {}).get("lr")
    total_steps = checkpoint.get("total_steps", 0)
    initial_warmup_steps = algorithm.config.get("optimizer", {}).get("warmup_steps", 30)

    if config_lr is not None:
        lr_increase_ratio = config_lr / checkpoint_lr if checkpoint_lr > 0 else 1.0

        if config_lr > checkpoint_lr and lr_increase_ratio >= 1.5:
            resume_warmup_steps = algorithm.config.get("optimizer", {}).get("resume_warmup_steps", 15)
            from torch.optim.lr_scheduler import LambdaLR

            def warmup_lambda(step):
                if step < resume_warmup_steps:
                    alpha = step / resume_warmup_steps
                    target_lr = checkpoint_lr + alpha * (config_lr - checkpoint_lr)
                    return target_lr / config_lr
                else:
                    return 1.0

            for param_group in algorithm.optimizer.param_groups:
                param_group['lr'] = config_lr

            algorithm.lr_scheduler = LambdaLR(algorithm.optimizer, lr_lambda=warmup_lambda)
            algorithm.optimizer.param_groups[0]['lr'] = checkpoint_lr

        elif config_lr != checkpoint_lr:
            for param_group in algorithm.optimizer.param_groups:
                param_group['lr'] = config_lr
            algorithm.lr_scheduler = None

        else:
            if total_steps >= initial_warmup_steps:
                algorithm.lr_scheduler = None
            elif algorithm.lr_scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                algorithm.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    algorithm.total_steps = checkpoint.get("total_steps", 0)
    algorithm.episode = checkpoint.get("episode", 0)
    algorithm.current_episode = checkpoint.get("current_episode", checkpoint.get("episode", 0))
