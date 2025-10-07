"""KL divergence calculation utilities for RL policy optimization.

Provides multiple estimators for computing KL divergence between policies,
including Monte Carlo approximations and low-variance unbiased estimators.

References:
- John Schulman's blog: http://joschu.net/blog/kl-approx.html
- PPO paper: https://arxiv.org/abs/1707.06347
- GRPO paper: https://arxiv.org/abs/2402.03300
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_kl_divergence(
    ref_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    estimator: str = "k3",
    clamp_min: float = -5.0,
    clamp_max: float = 5.0,
) -> torch.Tensor:
    """
    Compute KL divergence between reference and new policy distributions.

    Args:
        ref_log_probs: Log probabilities from reference policy [batch, seq_len]
        new_log_probs: Log probabilities from new policy [batch, seq_len]
        mask: Optional mask for valid tokens [batch, seq_len]
        estimator: KL estimator type ("mc", "k3", "abs", or "mse")
        clamp_min: Minimum value for log ratio clamping (default: -5.0)
        clamp_max: Maximum value for log ratio clamping (default: 5.0)

    Returns:
        Scalar KL divergence value (mean over valid tokens)

    Estimator types:
        - "mc": Monte Carlo approximation (simple difference, can be negative)
        - "k3": Low-variance unbiased estimator (r - 1) - log(r)
        - "abs": Absolute difference (always positive, symmetric)
        - "mse": Mean squared error of log ratio (biased but low variance)
    """
    if estimator == "mc":
        return compute_kl_mc(ref_log_probs, new_log_probs, mask)
    elif estimator == "k3":
        return compute_kl_k3(ref_log_probs, new_log_probs, mask, clamp_min, clamp_max)
    elif estimator == "abs":
        return compute_kl_abs(ref_log_probs, new_log_probs, mask)
    elif estimator == "mse":
        return compute_kl_mse(ref_log_probs, new_log_probs, mask)
    else:
        raise ValueError(
            f"Unknown estimator: {estimator}. Choose from: 'mc', 'k3', 'abs', 'mse'"
        )


def compute_kl_mc(
    ref_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Monte Carlo approximation of KL divergence.

    KL(P_ref || P_new) ≈ E[log P_ref(x) - log P_new(x)] for x ~ P_ref

    This is the simplest and most commonly used approximation in RLHF.
    It's an unbiased estimator but can be negative (when new policy is more
    confident on sampled tokens than reference).

    Args:
        ref_log_probs: Log probabilities from reference policy [batch, seq_len]
        new_log_probs: Log probabilities from new policy [batch, seq_len]
        mask: Optional mask for valid tokens [batch, seq_len]

    Returns:
        Scalar KL divergence (mean over valid tokens)

    Example:
        >>> ref = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]])
        >>> new = torch.tensor([[-1.2, -1.8], [-0.6, -1.4]])
        >>> kl = compute_kl_mc(ref, new)
        >>> # kl ≈ mean([0.2, -0.2, 0.1, -0.1]) = 0.0
    """
    # Simple difference: log(P_ref) - log(P_new) = log(P_ref / P_new)
    kl_per_token = ref_log_probs - new_log_probs

    if mask is not None:
        # Average over valid tokens only
        return (kl_per_token * mask).sum() / (mask.sum() + 1e-8)
    else:
        return kl_per_token.mean()


def compute_kl_k3(
    ref_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_min: float = -5.0,
    clamp_max: float = 5.0,
) -> torch.Tensor:
    """
    Low-variance unbiased KL divergence estimator (k3).

    Formula: KL ≈ E[(r - 1) - log(r)] where r = P_ref(x) / P_new(x)

    This estimator has lower variance than the simple MC estimator while
    remaining unbiased. It's always non-negative (true KL property).

    The log ratio is clamped to prevent numerical instability when policies
    diverge significantly.

    Args:
        ref_log_probs: Log probabilities from reference policy [batch, seq_len]
        new_log_probs: Log probabilities from new policy [batch, seq_len]
        mask: Optional mask for valid tokens [batch, seq_len]
        clamp_min: Minimum value for log ratio (default: -5.0)
        clamp_max: Maximum value for log ratio (default: 5.0)

    Returns:
        Scalar KL divergence (mean over valid tokens)

    Example with clamp_max=5.0:
        - If log_ratio = 5.0: kl = exp(5) - 5 - 1 ≈ 148 - 5 - 1 = 142
        - If log_ratio = 3.0: kl = exp(3) - 3 - 1 ≈ 20 - 3 - 1 = 16
        - If log_ratio = 0.0: kl = exp(0) - 0 - 1 = 1 - 0 - 1 = 0 (no divergence)

    Reference:
        John Schulman's blog: http://joschu.net/blog/kl-approx.html
    """
    # Compute log ratio: log(P_ref / P_new) = log(P_ref) - log(P_new)
    log_ratio = ref_log_probs - new_log_probs

    # Clamp to prevent exp() overflow
    # With clamp_max=5: exp(5) ≈ 148 (manageable)
    # With clamp_max=20: exp(20) ≈ 485 million (explosion!)
    log_ratio = torch.clamp(log_ratio, min=clamp_min, max=clamp_max)

    # Compute ratio: r = P_ref / P_new = exp(log_ratio)
    ratio = torch.exp(log_ratio)

    # k3 estimator: (r - 1) - log(r)
    # This simplifies to: r - 1 - log_ratio
    kl_per_token = ratio - 1.0 - log_ratio

    if mask is not None:
        # Average over valid tokens only
        return (kl_per_token * mask).sum() / (mask.sum() + 1e-8)
    else:
        return kl_per_token.mean()


def compute_kl_abs(
    ref_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Absolute difference KL estimator.

    Formula: KL ≈ E[|log P_ref(x) - log P_new(x)|]

    This is not a true KL divergence but provides a symmetric distance
    metric between policies. Always non-negative.

    Args:
        ref_log_probs: Log probabilities from reference policy [batch, seq_len]
        new_log_probs: Log probabilities from new policy [batch, seq_len]
        mask: Optional mask for valid tokens [batch, seq_len]

    Returns:
        Scalar KL divergence (mean over valid tokens)
    """
    kl_per_token = torch.abs(ref_log_probs - new_log_probs)

    if mask is not None:
        return (kl_per_token * mask).sum() / (mask.sum() + 1e-8)
    else:
        return kl_per_token.mean()


def compute_kl_mse(
    ref_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean squared error KL estimator (k2).

    Formula: KL ≈ 0.5 * E[(log P_ref(x) - log P_new(x))²]

    This estimator has very low variance but is biased (not true KL).
    Always non-negative.

    Args:
        ref_log_probs: Log probabilities from reference policy [batch, seq_len]
        new_log_probs: Log probabilities from new policy [batch, seq_len]
        mask: Optional mask for valid tokens [batch, seq_len]

    Returns:
        Scalar KL divergence (mean over valid tokens)

    Reference:
        John Schulman's blog: http://joschu.net/blog/kl-approx.html
    """
    log_diff = ref_log_probs - new_log_probs
    kl_per_token = 0.5 * log_diff ** 2

    if mask is not None:
        return (kl_per_token * mask).sum() / (mask.sum() + 1e-8)
    else:
        return kl_per_token.mean()


def compute_kl_true(
    ref_logits: torch.Tensor,
    new_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    True KL divergence between full categorical distributions.

    KL(P_ref || P_new) = Σ P_ref(token) * log(P_ref(token) / P_new(token))

    This computes the exact KL divergence by summing over the entire vocabulary
    at each position. Much more expensive than MC approximations but mathematically
    correct. Always non-negative.

    Args:
        ref_logits: Logits from reference policy [batch, seq_len, vocab_size]
        new_logits: Logits from new policy [batch, seq_len, vocab_size]
        mask: Optional mask for valid positions [batch, seq_len]

    Returns:
        Scalar KL divergence (mean over valid positions)

    Note:
        This requires full vocabulary logits, not just log probs of sampled tokens.
        Use only when you have access to full logits (much more expensive).
    """
    # Compute probability distributions
    ref_probs = F.softmax(ref_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)

    # KL(P || Q) = Σ P(x) * (log P(x) - log Q(x))
    # Sum over vocabulary dimension
    kl_per_position = (ref_probs * (ref_log_probs - new_log_probs)).sum(dim=-1)

    if mask is not None:
        return (kl_per_position * mask).sum() / (mask.sum() + 1e-8)
    else:
        return kl_per_position.mean()
