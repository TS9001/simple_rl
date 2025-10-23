from dataclasses import dataclass


@dataclass
class LossMetrics:
    """Comprehensive metrics from loss computation including policy, KL, and entropy terms."""

    policy_loss: float
    kl_divergence: float
    entropy: float
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    ratio_clipped_frac: float
    log_ratio_mean: float
    log_ratio_min: float
    log_ratio_max: float
    tokens_generated: float
    kl_term: float
    entropy_term: float
