"""
Reward functions package for reinforcement learning training.
"""

from .math_reward import (
    extract_single_number,
    extract_last_number,
    extract_all_numbers,
    compute_format_reward,
    compute_correctness_reward,
    compute_math_reward,
    compute_math_reward_batch,
    extract_answer_from_model_output,
    extract_answer_from_dataset,
    compute_math_rewards_batch,  # Backward compatibility alias
)

__all__ = [
    "extract_single_number",
    "extract_last_number",
    "extract_all_numbers",
    "compute_format_reward",
    "compute_correctness_reward",
    "compute_math_reward",
    "compute_math_reward_batch",
    "extract_answer_from_model_output",
    "extract_answer_from_dataset",
    "compute_math_rewards_batch",  # Backward compatibility alias
]

