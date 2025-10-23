from dataclasses import dataclass, field


@dataclass
class EpochMetrics:
    """Accumulator for metrics during training epoch updates."""

    policy_loss: float = 0.0
    kl_divergence: float = 0.0
    entropy: float = 0.0
    ratio_mean: float = 0.0
    ratio_clipped_frac: float = 0.0
    ratio_min: float = float('inf')
    ratio_max: float = float('-inf')
    log_ratio_mean: float = 0.0
    log_ratio_min: float = float('inf')
    log_ratio_max: float = float('-inf')
    grad_norm: float = 0.0
    grad_norm_post_clip: float = 0.0
    grad_clipped_frac: float = 0.0
    relative_param_update: float = 0.0

    def accumulate(self, loss_metrics: 'LossMetrics', grad_norm: float,
                   grad_norm_post_clip: float, grad_clipped_frac: float,
                   relative_param_update: float) -> None:
        """Accumulate metrics from a single update step."""
        self.policy_loss += loss_metrics.policy_loss
        self.kl_divergence += loss_metrics.kl_divergence
        self.entropy += loss_metrics.entropy
        self.ratio_mean += loss_metrics.ratio_mean
        self.ratio_clipped_frac += loss_metrics.ratio_clipped_frac
        self.ratio_min = min(self.ratio_min, loss_metrics.ratio_min)
        self.ratio_max = max(self.ratio_max, loss_metrics.ratio_max)
        self.log_ratio_mean += loss_metrics.log_ratio_mean
        self.log_ratio_min = min(self.log_ratio_min, loss_metrics.log_ratio_min)
        self.log_ratio_max = max(self.log_ratio_max, loss_metrics.log_ratio_max)
        self.grad_norm += grad_norm
        self.grad_norm_post_clip += grad_norm_post_clip
        self.grad_clipped_frac += grad_clipped_frac
        self.relative_param_update += relative_param_update

    def average(self, num_updates: int) -> None:
        """Average accumulated metrics over number of updates."""
        if num_updates > 0:
            self.policy_loss /= num_updates
            self.kl_divergence /= num_updates
            self.entropy /= num_updates
            self.ratio_mean /= num_updates
            self.ratio_clipped_frac /= num_updates
            self.log_ratio_mean /= num_updates
            self.grad_norm /= num_updates
            self.grad_norm_post_clip /= num_updates
            self.grad_clipped_frac /= num_updates
            self.relative_param_update /= num_updates

    def to_dict(self):
        """Convert to dictionary for compatibility."""
        return {
            "policy_loss": self.policy_loss,
            "kl_divergence": self.kl_divergence,
            "entropy": self.entropy,
            "ratio_mean": self.ratio_mean,
            "ratio_clipped_frac": self.ratio_clipped_frac,
            "ratio_min": self.ratio_min,
            "ratio_max": self.ratio_max,
            "log_ratio_mean": self.log_ratio_mean,
            "log_ratio_min": self.log_ratio_min,
            "log_ratio_max": self.log_ratio_max,
            "grad_norm": self.grad_norm,
            "grad_norm_post_clip": self.grad_norm_post_clip,
            "grad_clipped_frac": self.grad_clipped_frac,
            "relative_param_update": self.relative_param_update,
        }
