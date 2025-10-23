from dataclasses import dataclass, asdict
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from simple_rl.dto.advantages import AdvantageStats


@dataclass
class TrainingMetrics:
    """Metrics from a single training step."""

    total_loss: float
    pg_loss: float
    kl_divergence: float
    reward_mean: float
    reward_std: float
    format_reward_mean: float
    correctness_reward_mean: float

    ratio_mean: float
    ratio_min: float
    ratio_max: float
    ratio_clipped_frac: float
    log_ratio_mean: float
    log_ratio_min: float
    log_ratio_max: float

    grad_norm: float
    grad_norm_post_clip: float
    grad_clipped_frac: float
    relative_param_update: float

    tokens_generated: int

    advantages_mean: float
    advantages_std: float
    advantages_min_raw: float
    advantages_max_raw: float
    advantages_min: float
    advantages_max: float
    advantages_clamped_low_frac: float
    advantages_clamped_high_frac: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return asdict(self)


class MetricsBuilder:
    """Builder for constructing training metrics from epoch and trajectory data."""

    @staticmethod
    def build_training_metrics(
        epoch_metrics: Dict[str, float],
        reward_mean: float,
        reward_std: float,
        format_reward_mean: float,
        correctness_reward_mean: float,
        total_tokens: int,
        advantages_mean: float,
        advantages_std: float,
        advantage_stats: "AdvantageStats",
        kl_coef: float,
    ) -> TrainingMetrics:
        """Build training metrics from components."""
        return TrainingMetrics(
            total_loss=epoch_metrics["policy_loss"] + kl_coef * epoch_metrics["kl_divergence"],
            pg_loss=epoch_metrics["policy_loss"],
            kl_divergence=epoch_metrics["kl_divergence"],
            reward_mean=reward_mean,
            reward_std=reward_std,
            format_reward_mean=format_reward_mean,
            correctness_reward_mean=correctness_reward_mean,
            ratio_mean=epoch_metrics["ratio_mean"],
            ratio_min=epoch_metrics["ratio_min"],
            ratio_max=epoch_metrics["ratio_max"],
            ratio_clipped_frac=epoch_metrics["ratio_clipped_frac"],
            log_ratio_mean=epoch_metrics["log_ratio_mean"],
            log_ratio_min=epoch_metrics["log_ratio_min"],
            log_ratio_max=epoch_metrics["log_ratio_max"],
            grad_norm=epoch_metrics["grad_norm"],
            grad_norm_post_clip=epoch_metrics["grad_norm_post_clip"],
            grad_clipped_frac=epoch_metrics["grad_clipped_frac"],
            relative_param_update=epoch_metrics["relative_param_update"],
            tokens_generated=total_tokens,
            advantages_mean=advantages_mean,
            advantages_std=advantages_std,
            advantages_min_raw=advantage_stats.advantages_min_raw,
            advantages_max_raw=advantage_stats.advantages_max_raw,
            advantages_min=advantage_stats.advantages_min,
            advantages_max=advantage_stats.advantages_max,
            advantages_clamped_low_frac=advantage_stats.advantages_clamped_low_frac,
            advantages_clamped_high_frac=advantage_stats.advantages_clamped_high_frac,
        )
