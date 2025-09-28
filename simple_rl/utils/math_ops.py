"""Common mathematical operations for reinforcement learning."""

import torch
from typing import Tuple


def normalize_rewards(rewards: torch.Tensor, group_size: int = 1, normalize_within_groups: bool = True) -> torch.Tensor:
    """
    Normalize rewards within groups or globally.

    Args:
        rewards: Reward tensor [batch_size]
        group_size: Size of each group
        normalize_within_groups: Whether to normalize within groups or globally

    Returns:
        Normalized rewards
    """
    if normalize_within_groups and group_size > 1:
        batch_size = rewards.shape[0]
        num_groups = batch_size // group_size

        # Reshape to groups
        grouped_rewards = rewards.view(num_groups, group_size)

        # Normalize within each group
        group_mean = grouped_rewards.mean(dim=1, keepdim=True)
        group_std = grouped_rewards.std(dim=1, keepdim=True)

        normalized_rewards = (grouped_rewards - group_mean) / (group_std + 1e-8)

        # Flatten back
        advantages = normalized_rewards.view(-1)
    else:
        # Global normalization
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    return advantages


def compute_advantages(
    rewards: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    group_size: int = 1,
    normalize_within_groups: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages with group normalization and KL penalty.

    Args:
        rewards: Reward values [batch_size]
        log_probs: Policy log probabilities [batch_size, seq_len]
        ref_log_probs: Reference log probabilities [batch_size, seq_len]
        completion_mask: Mask for completion tokens [batch_size, seq_len]
        group_size: Size of each group
        normalize_within_groups: Whether to normalize within groups

    Returns:
        Tuple of (advantages, kl_penalty)
    """
    # Normalize rewards
    advantages = normalize_rewards(rewards, group_size, normalize_within_groups)

    # Get sequence-level log probabilities by summing
    log_probs_sum = log_probs.sum(dim=-1)

    # TRL adds KL at the per-token level, then averages
    delta = log_probs - ref_log_probs
    per_token_kl = torch.exp(delta) - delta - 1.0  # Schulman approx
    kl_penalty = (per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)

    return advantages, kl_penalty


def compute_policy_gradient_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute policy gradient loss.

    Args:
        log_probs: Policy log probabilities [batch_size, seq_len]
        advantages: Advantage values [batch_size]
        completion_mask: Mask for completion tokens [batch_size, seq_len]

    Returns:
        Policy gradient loss
    """
    # Get sequence-level log probabilities by summing
    log_probs_sum = log_probs.sum(dim=-1)

    # Policy gradient loss (using summed log probs)
    pg_loss = -(log_probs_sum * advantages.detach()).mean()

    return pg_loss


def compute_total_loss(
    pg_loss: torch.Tensor,
    kl_penalty: torch.Tensor,
    kl_coef: float = 0.05,
) -> torch.Tensor:
    """
    Compute total loss combining policy gradient and KL penalty.

    Args:
        pg_loss: Policy gradient loss
        kl_penalty: KL divergence penalty
        kl_coef: KL coefficient

    Returns:
        Total loss
    """
    return pg_loss + (kl_coef * kl_penalty)
