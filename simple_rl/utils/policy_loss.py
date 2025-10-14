"""Policy loss calculation utilities for PPO-style RL algorithms.

Provides clipped surrogate objective for policy gradient optimization,
supporting both symmetric and asymmetric clipping.

Offers two normalization modes:
- SEQUENCE-LEVEL: Each sequence contributes equally (traditional PPO)
- TOKEN-LEVEL: Normalized by token count (reduces gradient magnitude on long sequences)

References:
- PPO paper: https://arxiv.org/abs/1707.06347
- GRPO paper: https://arxiv.org/abs/2402.03300
"""

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
    """
    Compute PPO-style clipped surrogate policy loss.

    This implements the clipped objective from the PPO paper:
        L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

    where r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t) is the probability ratio.

    Args:
        new_log_probs: Log probabilities from current policy [batch, seq_len]
        old_log_probs: Log probabilities from old policy [batch, seq_len]
        advantages: Advantage estimates [batch] (sequence-level)
        completion_mask: Binary mask for valid tokens [batch, seq_len]
        clip_epsilon: Symmetric clipping range (e.g., 0.2 for [0.8, 1.2])
        clip_epsilon_low: Lower clipping range (asymmetric mode)
        clip_epsilon_high: Upper clipping range (asymmetric mode)
        log_ratio_clamp_min: Minimum log ratio before exp (prevents underflow)
        log_ratio_clamp_max: Maximum log ratio before exp (prevents overflow)

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - ratio_mean: Mean probability ratio
            - ratio_min: Minimum probability ratio
            - ratio_max: Maximum probability ratio
            - ratio_clipped_frac: Fraction of ratios that were clipped
            - log_ratio_mean: Mean log ratio
            - log_ratio_min: Minimum log ratio
            - log_ratio_max: Maximum log ratio

    Example:
        >>> # Symmetric clipping
        >>> loss, metrics = compute_ppo_policy_loss(
        ...     new_log_probs, old_log_probs, advantages, mask,
        ...     clip_epsilon=0.2
        ... )
        >>>
        >>> # Asymmetric clipping (allow more increase than decrease)
        >>> loss, metrics = compute_ppo_policy_loss(
        ...     new_log_probs, old_log_probs, advantages, mask,
        ...     clip_epsilon_low=0.2, clip_epsilon_high=0.4
        ... )

    Note:
        - Log probabilities are summed over sequence length to get sequence-level
          probability, then ratio is computed at sequence level.
        - Uses proper PPO clipping in ratio-space (NOT log-space).
          With clip_epsilon=0.2: ratio ∈ [0.8, 1.2]
        - Advantages are detached to prevent backprop through advantage computation.
        - Old log probs are detached to prevent backprop through old policy.
    """
    # Validate clipping parameters
    if clip_epsilon is not None:
        clip_epsilon_low = clip_epsilon
        clip_epsilon_high = clip_epsilon
    elif clip_epsilon_low is None or clip_epsilon_high is None:
        raise ValueError(
            "Must provide either clip_epsilon OR both clip_epsilon_low and clip_epsilon_high"
        )

    # Sum log probs over sequence to get sequence-level log probability
    # This makes policy loss sequence-level (each sequence contributes equally)
    new_log_probs_sum = (new_log_probs * completion_mask).sum(dim=-1)  # [batch]
    old_log_probs_sum = (old_log_probs * completion_mask).sum(dim=-1).detach()  # [batch] - detach old policy

    # Compute log ratio: log(π_new / π_old)
    log_ratio = new_log_probs_sum - old_log_probs_sum  # [batch]

    # Compute probability ratio: π_new / π_old
    # NOTE: We compute ratio directly without pre-clamping log_ratio
    # PPO clipping happens in ratio-space, not log-space
    ratio = torch.exp(log_ratio)  # [batch]

    # PPO clipped surrogate objective (proper PPO clipping in ratio-space)
    # surr1: unclipped objective
    # surr2: clipped objective (ratio constrained to [1-ε_low, 1+ε_high])
    # With ε=0.2: ratio is clipped to [0.8, 1.2]
    surr1 = ratio * advantages.detach()  # [batch]
    ratio_clipped = torch.clamp(
        ratio,
        1.0 - clip_epsilon_low,   # Lower bound (e.g., 0.8 if clip_epsilon_low=0.2)
        1.0 + clip_epsilon_high,  # Upper bound (e.g., 1.2 if clip_epsilon_high=0.2)
    )
    surr2 = ratio_clipped * advantages.detach()  # [batch]

    # Take minimum to pessimistically bound the objective
    # Negative because we want to maximize, but optimizer minimizes
    policy_loss = -torch.min(surr1, surr2).mean()

    # Compute metrics for logging and debugging
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
    """
    Compute PPO-style clipped policy loss with TOKEN-LEVEL normalization.

    This variant normalizes by token count instead of treating each sequence equally.
    It typically reduces gradient magnitude by 10-100x on long sequences, improving
    stability when sequence lengths vary significantly.

    Key difference from sequence-level:
        - Sequence-level: Sum log probs → compute ratio → mean over sequences
        - Token-level: Compute ratio per token → mean over valid tokens

    Formula:
        For each sequence i:
            surr1_i = mean_over_valid_tokens(ratio_t * A_i)
            surr2_i = mean_over_valid_tokens(clip(ratio_t) * A_i)
        L = -mean_over_sequences(min(surr1_i, surr2_i))

    Args:
        new_log_probs: Log probabilities from current policy [batch, seq_len]
        old_log_probs: Log probabilities from old policy [batch, seq_len]
        advantages: Advantage estimates [batch] (sequence-level)
        completion_mask: Binary mask for valid tokens [batch, seq_len]
        clip_epsilon: Symmetric clipping range (e.g., 0.2 for [0.8, 1.2])
        clip_epsilon_low: Lower clipping range (asymmetric mode)
        clip_epsilon_high: Upper clipping range (asymmetric mode)

    Returns:
        Tuple of (loss, metrics_dict) - SAME interface as sequence-level variant

    Example:
        >>> # Token-level normalization (better for variable-length sequences)
        >>> loss, metrics = compute_ppo_policy_loss_token_level(
        ...     new_log_probs, old_log_probs, advantages, mask,
        ...     clip_epsilon=0.2
        ... )

    Benefits:
        - 10-100x smaller gradients on long sequences
        - More stable when sequence lengths vary (e.g., 50 tokens vs 500 tokens)
        - Each token contributes equally to the loss
        - Better gradient flow in early training when model generates long sequences
    """
    # Validate clipping parameters
    if clip_epsilon is not None:
        clip_epsilon_low = clip_epsilon
        clip_epsilon_high = clip_epsilon
    elif clip_epsilon_low is None or clip_epsilon_high is None:
        raise ValueError(
            "Must provide either clip_epsilon OR both clip_epsilon_low and clip_epsilon_high"
        )

    # Compute log ratio per token: log(π_new(t) / π_old(t))
    log_ratio_per_token = new_log_probs - old_log_probs.detach()  # [batch, seq_len] - detach old policy

    # Compute probability ratio per token (no log clamping - proper PPO)
    ratio_per_token = torch.exp(log_ratio_per_token)  # [batch, seq_len]

    # PPO clipping in ratio-space (proper PPO clipping)
    # With ε=0.2: ratio is clipped to [0.8, 1.2]
    ratio_per_token_clipped = torch.clamp(
        ratio_per_token,
        1.0 - clip_epsilon_low,
        1.0 + clip_epsilon_high
    )  # [batch, seq_len]

    # Broadcast advantages to match token dimension
    advantages_broadcast = advantages.detach().unsqueeze(1)  # [batch, 1]

    # Count valid tokens per sequence (for normalization)
    valid_tokens_per_seq = completion_mask.sum(dim=-1).clamp(min=1.0)  # [batch]

    # Compute surrogate objectives per sequence (normalized by token count)
    # surr1: (ratio_t * A * mask).sum() / num_valid_tokens
    surr1 = (ratio_per_token * advantages_broadcast * completion_mask).sum(dim=-1) / valid_tokens_per_seq
    surr2 = (ratio_per_token_clipped * advantages_broadcast * completion_mask).sum(dim=-1) / valid_tokens_per_seq

    # Take minimum and mean over batch
    policy_loss = -torch.min(surr1, surr2).mean()

    # Compute metrics (aggregate over all valid tokens for consistency)
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
    """
    Compute vanilla policy gradient loss (REINFORCE).

    L^PG(θ) = -E[log π_θ(a_t | s_t) * A_t]

    This is the simplest policy gradient without clipping. Less stable than PPO
    but can be useful for comparison or when you want unrestricted updates.

    Args:
        new_log_probs: Log probabilities from current policy [batch, seq_len]
        advantages: Advantage estimates [batch] (sequence-level)
        completion_mask: Binary mask for valid tokens [batch, seq_len]

    Returns:
        Tuple of (loss, metrics_dict)

    Example:
        >>> loss, metrics = compute_vanilla_policy_gradient_loss(
        ...     new_log_probs, advantages, mask
        ... )
    """
    # Sum log probs over sequence
    new_log_probs_sum = (new_log_probs * completion_mask).sum(dim=-1)  # [batch]

    # Vanilla policy gradient: -log π * A
    policy_loss = -(new_log_probs_sum * advantages.detach()).mean()

    metrics = {
        "log_prob_mean": new_log_probs_sum.mean().item(),
        "log_prob_std": new_log_probs_sum.std().item(),
    }

    return policy_loss, metrics
