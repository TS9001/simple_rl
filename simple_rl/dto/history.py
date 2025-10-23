from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class TrainingHistory:
    """Storage for training history across all episodes."""

    episode: List[int] = field(default_factory=list)
    total_loss: List[float] = field(default_factory=list)
    pg_loss: List[float] = field(default_factory=list)
    kl_divergence: List[float] = field(default_factory=list)
    reward_mean: List[float] = field(default_factory=list)
    reward_std: List[float] = field(default_factory=list)
    format_reward_mean: List[float] = field(default_factory=list)
    correctness_reward_mean: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    grad_norm_post_clip: List[float] = field(default_factory=list)
    grad_clipped_frac: List[float] = field(default_factory=list)
    relative_param_update: List[float] = field(default_factory=list)
    tokens_generated: List[int] = field(default_factory=list)
    episode_time: List[float] = field(default_factory=list)
    total_tokens: List[int] = field(default_factory=list)
    tokens_per_second: List[float] = field(default_factory=list)

    def append(self, episode: int, metrics: Dict[str, float],
               episode_time: float, episode_tokens: int,
               total_tokens: int, tokens_per_sec: float) -> None:
        """Append metrics from a single episode."""
        self.episode.append(episode)
        self.total_loss.append(metrics["total_loss"])
        self.pg_loss.append(metrics["pg_loss"])
        self.kl_divergence.append(metrics["kl_divergence"])
        self.reward_mean.append(metrics["reward_mean"])
        self.reward_std.append(metrics["reward_std"])
        self.format_reward_mean.append(metrics["format_reward_mean"])
        self.correctness_reward_mean.append(metrics["correctness_reward_mean"])
        self.grad_norm.append(metrics["grad_norm"])
        self.grad_norm_post_clip.append(metrics["grad_norm_post_clip"])
        self.grad_clipped_frac.append(metrics["grad_clipped_frac"])
        self.relative_param_update.append(metrics["relative_param_update"])
        self.tokens_generated.append(int(metrics["tokens_generated"]))
        self.episode_time.append(episode_time)
        self.total_tokens.append(total_tokens)
        self.tokens_per_second.append(tokens_per_sec)

    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary for compatibility."""
        return {
            "episode": self.episode,
            "total_loss": self.total_loss,
            "pg_loss": self.pg_loss,
            "kl_divergence": self.kl_divergence,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "format_reward_mean": self.format_reward_mean,
            "correctness_reward_mean": self.correctness_reward_mean,
            "grad_norm": self.grad_norm,
            "grad_norm_post_clip": self.grad_norm_post_clip,
            "grad_clipped_frac": self.grad_clipped_frac,
            "relative_param_update": self.relative_param_update,
            "tokens_generated": self.tokens_generated,
            "episode_time": self.episode_time,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass
class ValidationHistory:
    """Storage for validation evaluation metrics across episodes."""

    episode: List[int] = field(default_factory=list)
    exact_accuracy: List[float] = field(default_factory=list)
    numeric_accuracy: List[float] = field(default_factory=list)
    format_compliance: List[float] = field(default_factory=list)
    avg_format_score: List[float] = field(default_factory=list)
    avg_correctness_score: List[float] = field(default_factory=list)

    def append(self, episode: int, val_metrics: Dict[str, float]) -> None:
        """Append validation metrics from a single episode."""
        self.episode.append(episode)
        self.exact_accuracy.append(val_metrics.get("exact_accuracy", 0.0))
        self.numeric_accuracy.append(val_metrics.get("numeric_accuracy", 0.0))
        self.format_compliance.append(val_metrics.get("format_compliance", 0.0))
        self.avg_format_score.append(val_metrics.get("avg_format_score", 0.0))
        self.avg_correctness_score.append(val_metrics.get("avg_correctness_score", 0.0))

    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary for compatibility."""
        return {
            "episode": self.episode,
            "exact_accuracy": self.exact_accuracy,
            "numeric_accuracy": self.numeric_accuracy,
            "format_compliance": self.format_compliance,
            "avg_format_score": self.avg_format_score,
            "avg_correctness_score": self.avg_correctness_score,
        }
