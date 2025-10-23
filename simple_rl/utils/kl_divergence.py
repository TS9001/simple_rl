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
    reduction: str = "mean",
) -> torch.Tensor:
    if estimator == "mc":
        return compute_kl_mc(ref_log_probs, new_log_probs, mask, reduction)
    elif estimator == "k3":
        return compute_kl_k3(ref_log_probs, new_log_probs, mask, clamp_min, clamp_max, reduction)
    elif estimator == "abs":
        return compute_kl_abs(ref_log_probs, new_log_probs, mask, reduction)
    elif estimator == "mse":
        return compute_kl_mse(ref_log_probs, new_log_probs, mask, reduction)
    else:
        raise ValueError(
            f"Unknown estimator: {estimator}. Choose from: 'mc', 'k3', 'abs', 'mse'"
        )


def compute_kl_mc(
    ref_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    kl_per_token = ref_log_probs - new_log_probs

    if mask is not None:
        kl_sum = (kl_per_token * mask).sum()
        if reduction == "mean":
            return kl_sum / (mask.sum() + 1e-8)
        else:  # "sum"
            return kl_sum
    else:
        if reduction == "mean":
            return kl_per_token.mean()
        else:  # "sum"
            return kl_per_token.sum()


def compute_kl_k3(
    ref_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_min: float = -5.0,
    clamp_max: float = 5.0,
    reduction: str = "mean",
) -> torch.Tensor:
    ref_log_probs = ref_log_probs.float()
    new_log_probs = new_log_probs.float()

    log_ratio = ref_log_probs - new_log_probs
    log_ratio = torch.clamp(log_ratio, min=clamp_min, max=clamp_max)

    ratio = torch.exp(log_ratio)
    kl_per_token = ratio - 1.0 - log_ratio

    if mask is not None:
        kl_sum = (kl_per_token * mask).sum()
        if reduction == "mean":
            return kl_sum / (mask.sum() + 1e-8)
        else:  # "sum"
            return kl_sum
    else:
        if reduction == "mean":
            return kl_per_token.mean()
        else:  # "sum"
            return kl_per_token.sum()


def compute_kl_abs(
    ref_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    kl_per_token = torch.abs(ref_log_probs - new_log_probs)

    if mask is not None:
        kl_sum = (kl_per_token * mask).sum()
        if reduction == "mean":
            return kl_sum / (mask.sum() + 1e-8)
        else:  # "sum"
            return kl_sum
    else:
        if reduction == "mean":
            return kl_per_token.mean()
        else:  # "sum"
            return kl_per_token.sum()


def compute_kl_mse(
    ref_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    log_diff = ref_log_probs - new_log_probs
    kl_per_token = 0.5 * log_diff ** 2

    if mask is not None:
        kl_sum = (kl_per_token * mask).sum()
        if reduction == "mean":
            return kl_sum / (mask.sum() + 1e-8)
        else:  # "sum"
            return kl_sum
    else:
        if reduction == "mean":
            return kl_per_token.mean()
        else:  # "sum"
            return kl_per_token.sum()


def compute_kl_true(
    ref_logits: torch.Tensor,
    new_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ref_probs = F.softmax(ref_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)

    kl_per_position = (ref_probs * (ref_log_probs - new_log_probs)).sum(dim=-1)

    if mask is not None:
        return (kl_per_position * mask).sum() / (mask.sum() + 1e-8)
    else:
        return kl_per_position.mean()
