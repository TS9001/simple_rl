import torch
from typing import Optional, Tuple, Dict


def compute_ppo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_epsilon: Optional[float] = None,
    clip_epsilon_low: Optional[float] = None,
    clip_epsilon_high: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if clip_epsilon is not None:
        clip_epsilon_low = clip_epsilon
        clip_epsilon_high = clip_epsilon
    elif clip_epsilon_low is None or clip_epsilon_high is None:
        raise ValueError(
            "Must provide either clip_epsilon OR both clip_epsilon_low and clip_epsilon_high"
        )

    new_log_probs = new_log_probs.float()
    old_log_probs = old_log_probs.float()
    completion_mask = completion_mask.float()

    new_log_probs_sum = (new_log_probs * completion_mask).sum(dim=-1)
    old_log_probs_sum = (old_log_probs * completion_mask).sum(dim=-1).detach()

    log_ratio = new_log_probs_sum - old_log_probs_sum
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages.detach()
    ratio_clipped = torch.clamp(
        ratio,
        1.0 - clip_epsilon_low,
        1.0 + clip_epsilon_high,
    )
    surr2 = ratio_clipped * advantages.detach()

    policy_loss = -torch.min(surr1, surr2).mean()

    metrics = {
        "ratio_mean": ratio.mean().item(),
        "ratio_min": ratio.min().item(),
        "ratio_max": ratio.max().item(),
        "ratio_clipped_frac": (
            (ratio < 1.0 - clip_epsilon_low) | (ratio > 1.0 + clip_epsilon_high)
        ).float().mean().item(),
        "log_ratio_mean": log_ratio.mean().item(),
        "log_ratio_min": log_ratio.min().item(),
        "log_ratio_max": log_ratio.max().item(),
    }

    return policy_loss, metrics


def compute_ppo_policy_loss_token_level(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_epsilon: Optional[float] = None,
    clip_epsilon_low: Optional[float] = None,
    clip_epsilon_high: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if clip_epsilon is not None:
        clip_epsilon_low = clip_epsilon
        clip_epsilon_high = clip_epsilon
    elif clip_epsilon_low is None or clip_epsilon_high is None:
        raise ValueError(
            "Must provide either clip_epsilon OR both clip_epsilon_low and clip_epsilon_high"
        )

    new_log_probs = new_log_probs.float()
    old_log_probs = old_log_probs.float()
    completion_mask = completion_mask.float()

    log_ratio_per_token = new_log_probs - old_log_probs.detach()
    ratio_per_token = torch.exp(log_ratio_per_token)

    ratio_per_token_clipped = torch.clamp(
        ratio_per_token,
        1.0 - clip_epsilon_low,
        1.0 + clip_epsilon_high
    )

    advantages_broadcast = advantages.detach().unsqueeze(1)
    valid_tokens_per_seq = completion_mask.sum(dim=-1).clamp(min=1.0)

    surr1 = (ratio_per_token * advantages_broadcast * completion_mask).sum(dim=-1) / valid_tokens_per_seq
    surr2 = (ratio_per_token_clipped * advantages_broadcast * completion_mask).sum(dim=-1) / valid_tokens_per_seq

    policy_loss = -torch.min(surr1, surr2).mean()

    valid_mask = completion_mask.bool()
    ratio_valid = ratio_per_token[valid_mask]
    log_ratio_valid = log_ratio_per_token[valid_mask]

    metrics = {
        "ratio_mean": ratio_valid.mean().item() if ratio_valid.numel() > 0 else 1.0,
        "ratio_min": ratio_valid.min().item() if ratio_valid.numel() > 0 else 1.0,
        "ratio_max": ratio_valid.max().item() if ratio_valid.numel() > 0 else 1.0,
        "ratio_clipped_frac": (
            (ratio_valid < 1.0 - clip_epsilon_low) | (ratio_valid > 1.0 + clip_epsilon_high)
        ).float().mean().item() if ratio_valid.numel() > 0 else 0.0,
        "log_ratio_mean": log_ratio_valid.mean().item() if log_ratio_valid.numel() > 0 else 0.0,
        "log_ratio_min": log_ratio_valid.min().item() if log_ratio_valid.numel() > 0 else 0.0,
        "log_ratio_max": log_ratio_valid.max().item() if log_ratio_valid.numel() > 0 else 0.0,
    }

    return policy_loss, metrics


def compute_vanilla_policy_gradient_loss(
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # DEPRECATED: Not currently used, kept for reference
    new_log_probs_sum = (new_log_probs * completion_mask).sum(dim=-1)
    policy_loss = -(new_log_probs_sum * advantages.detach()).mean()

    metrics = {
        "log_prob_mean": new_log_probs_sum.mean().item(),
        "log_prob_std": new_log_probs_sum.std().item(),
    }

    return policy_loss, metrics
