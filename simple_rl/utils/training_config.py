"""Training configuration utilities."""

from typing import Dict, Any, Optional


class TrainingConfig:
    """Configuration manager for training parameters."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._parsed_config = self._parse_config()

    def _parse_config(self) -> Dict[str, Any]:
        """Parse and validate training configuration."""
        training_config = self.config.get("training", {})

        # Algorithm-specific parameters
        algo_config = self.config.get("algorithm", {})

        parsed = {
            # Training parameters
            "learning_rate": training_config.get("learning_rate", 1e-5),
            "batch_size": training_config.get("batch_size", 8),
            "minibatch_size": training_config.get("minibatch_size", None),
            "max_new_tokens": training_config.get("max_new_tokens", 128),
            "temperature": training_config.get("temperature", 0.9),
            "top_k": training_config.get("top_k", None),
            "top_p": training_config.get("top_p", 0.9),
            "gradient_clip": training_config.get("gradient_clip", 1.0),

            # Algorithm parameters
            "group_size": algo_config.get("group_size", 4),
            "kl_coef": algo_config.get("kl_coef", 0.05),
            "normalize_rewards": algo_config.get("normalize_rewards", True),
            "clip_epsilon": algo_config.get("clip_epsilon", 0.2),
            "clip_epsilon_low": algo_config.get("clip_epsilon_low", algo_config.get("clip_epsilon", 0.2)),
            "clip_epsilon_high": algo_config.get("clip_epsilon_high", algo_config.get("clip_epsilon", 0.2)),
            "store_completions": algo_config.get("store_completions", True),

            # Rollout generation batching (to avoid OOM during generation)
            "rollout_batch_size": training_config.get("rollout_batch_size", None),  # Number of prompts to generate at once

            # Update parameters
            "update_epochs": training_config.get("update_epochs", 1),  # GRPO uses single epoch (DeepSeekMath)
        }

        # Handle minibatch_size default
        if parsed["minibatch_size"] is None or parsed["minibatch_size"] == 0:
            parsed["minibatch_size"] = parsed["batch_size"]

        # Handle rollout_batch_size default
        if parsed["rollout_batch_size"] is None or parsed["rollout_batch_size"] == 0:
            # Default: generate 2 prompts at a time (conservative to avoid OOM)
            parsed["rollout_batch_size"] = min(2, parsed["batch_size"])

        return parsed

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._parsed_config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict-like access."""
        return self._parsed_config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._parsed_config

    def get_all(self) -> Dict[str, Any]:
        """Get all parsed configuration."""
        return self._parsed_config.copy()

    @property
    def learning_rate(self) -> float:
        """Get learning rate."""
        return self._parsed_config["learning_rate"]

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self._parsed_config["batch_size"]

    @property
    def minibatch_size(self) -> int:
        """Get minibatch size."""
        return self._parsed_config["minibatch_size"]

    @property
    def max_new_tokens(self) -> int:
        """Get maximum new tokens."""
        return self._parsed_config["max_new_tokens"]

    @property
    def temperature(self) -> float:
        """Get sampling temperature."""
        return self._parsed_config["temperature"]

    @property
    def top_k(self) -> Optional[int]:
        """Get top-k sampling parameter."""
        return self._parsed_config["top_k"]

    @property
    def top_p(self) -> float:
        """Get top-p sampling parameter."""
        return self._parsed_config["top_p"]

    @property
    def gradient_clip(self) -> float:
        """Get gradient clipping value."""
        return self._parsed_config["gradient_clip"]

    @property
    def group_size(self) -> int:
        """Get group size for GRPO."""
        return self._parsed_config["group_size"]

    @property
    def kl_coef(self) -> float:
        """Get KL divergence coefficient."""
        return self._parsed_config["kl_coef"]

    @property
    def normalize_rewards(self) -> bool:
        """Get reward normalization flag."""
        return self._parsed_config["normalize_rewards"]

    @property
    def clip_epsilon(self) -> float:
        """Get clipping epsilon."""
        return self._parsed_config["clip_epsilon"]

    @property
    def clip_epsilon_low(self) -> float:
        """Get lower clipping epsilon (for asymmetric clipping)."""
        return self._parsed_config["clip_epsilon_low"]

    @property
    def clip_epsilon_high(self) -> float:
        """Get upper clipping epsilon (for asymmetric clipping)."""
        return self._parsed_config["clip_epsilon_high"]

    @property
    def store_completions(self) -> bool:
        """Get completion storage flag."""
        return self._parsed_config["store_completions"]

    @property
    def update_epochs(self) -> int:
        """Get number of update epochs."""
        return self._parsed_config["update_epochs"]

    @property
    def rollout_batch_size(self) -> int:
        """Get rollout batch size for generation."""
        return self._parsed_config["rollout_batch_size"]


def create_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """
    Create training configuration from config dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(config)
