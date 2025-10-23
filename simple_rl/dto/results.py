from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TrainResult:
    """Final training summary with metrics and timing information."""

    training_metrics: Dict[str, List]
    validation_metrics: Dict[str, List]
    total_time: float
    final_reward: float


@dataclass
class EvaluationResult:
    """Evaluation statistics."""

    eval_reward_mean: float
    eval_reward_std: float
