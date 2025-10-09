"""
Simple reward functions for mathematical problem solving in reinforcement learning.

This module provides reward functions that evaluate model completions based on:
1. Correctness of the answer compared to ground truth
2. Proper formatting with required XML tags

Functions are designed to work with mathematical reasoning tasks where models
should provide answers in a structured format with <reasoning> and <answer> tags.
"""

import re
from typing import List, Optional, Union

import torch

# Precompile regex for performance
NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def extract_answer_from_model_output(text: str) -> Optional[str]:
    """
    Extract answer from model output enclosed in <answer> tags.

    Args:
        text: Model completion text

    Returns:
        Extracted answer string or None if not found/invalid
    """
    # Use regex to find all <answer>...</answer> blocks
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    if not matches:
        return None
    
    # Take the last answer in case there are multiple
    answer = matches[-1].strip()
    
    # Filter out placeholder text
    if answer == "..." or answer == "":
        return None
    
    return answer


def extract_answer_from_dataset(text: str) -> Optional[str]:
    """
    Extract answer from dataset text format (after #### delimiter).

    Args:
        text: Dataset text containing answer after ####

    Returns:
        Extracted answer string or None if not found
    """
    return text.split("####")[1].strip() if "####" in text else None


def extract_single_number(text: Optional[str]) -> Optional[float]:
    """
    Extract a single number from text if exactly one exists.

    Matches the original GRPO notebook implementation:
    - Returns the number ONLY if exactly one number is found
    - Returns None if zero numbers or multiple numbers found

    Args:
        text: Input text that may contain numbers

    Returns:
        Single number as float, or None if not exactly one number
    """
    if text is None:
        return None

    # Find all numbers
    numbers = NUMBER_PATTERN.findall(str(text))

    # Return the number ONLY if exactly one is found
    if len(numbers) == 1:
        try:
            return float(numbers[0])
        except ValueError:
            return None

    return None


def correctness_reward(completion: str, answer: Optional[str] = None) -> float:
    """
    Compute correctness reward based on answer accuracy.

    Matches the original GRPO notebook implementation:
    - 2.0: Exact string match
    - 1.5: Numerical equality (both single numbers, exact match)
    - 0.0: No match or missing answer

    Args:
        completion: Model completion text
        answer: Ground truth answer (optional)

    Returns:
        Correctness reward (0.0, 1.5, or 2.0)
    """
    if answer is None:
        return 0.0

    model_answer = extract_answer_from_model_output(completion)

    # Exact string match
    if model_answer == answer:
        return 2.0

    # Try numeric equivalence (exact equality, no tolerance)
    model_num = extract_single_number(str(model_answer))
    answer_num = extract_single_number(str(answer))

    if model_num is not None and answer_num is not None and model_num == answer_num:
        return 1.5

    return 0.0


def format_reward(completion: str) -> float:
    """
    Compute format reward based on presence of required XML tags.

    Matches the original GRPO notebook implementation:
    - 0.2 for each tag: <reasoning>, </reasoning>, <answer>, </answer>
    - Maximum score: 0.8 (all 4 tags present)

    Args:
        completion: Model completion text

    Returns:
        Format reward (0.0 to 0.8)
    """
    score = 0.0
    if "<reasoning>" in completion:
        score += 0.20
    if "</reasoning>" in completion:
        score += 0.20
    if "<answer>" in completion:
        score += 0.20
    if "</answer>" in completion:
        score += 0.20
    return score
    


def compute_math_reward(completion: str, answer: Optional[str] = None) -> float:
    """
    Compute total reward combining correctness and format rewards.

    Args:
        completion: Model completion text
        answer: Ground truth answer (optional)

    Returns:
        Total reward (correctness + format)
    """
    return correctness_reward(completion, answer) + format_reward(completion)


def compute_math_rewards_batch(
    completions: Union[List[str], torch.Tensor],
    answers: Optional[List[str]] = None,
    device: Union[str, torch.device, None] = None,
    return_breakdown: bool = False,
) -> Union[torch.Tensor, tuple]:
    """
    Compute rewards for a batch of completions.
    Returns tensor directly on specified device (cuda/mps/cpu).

    Args:
        completions: List of completion strings or tensor of token ids
        answers: List of ground truth answers (optional)
        device: Target device for output tensor
        return_breakdown: If True, returns (total, format, correctness) tuple

    Returns:
        If return_breakdown=False: Tensor of total rewards on specified device
        If return_breakdown=True: Tuple of (total_rewards, format_rewards, correctness_rewards)

    Raises:
        NotImplementedError: If completions are provided as token tensor
    """
    # If completions are token ids, decode them first
    if isinstance(completions, torch.Tensor):
        # Assume you have tokenizer available globally or pass it in
        # completions = tokenizer.batch_decode(completions, skip_special_tokens=True)
        raise NotImplementedError(
            "Pass decoded strings or implement tokenizer decoding"
        )

    batch_size = len(completions)

    # Vectorized processing
    if answers is None:
        answers = [None] * batch_size

    if return_breakdown:
        # Compute format and correctness separately
        format_rewards = list(map(format_reward, completions))
        correctness_rewards = list(map(correctness_reward, completions, answers))
        total_rewards = [f + c for f, c in zip(format_rewards, correctness_rewards)]

        return (
            torch.tensor(total_rewards, dtype=torch.float32, device=device),
            torch.tensor(format_rewards, dtype=torch.float32, device=device),
            torch.tensor(correctness_rewards, dtype=torch.float32, device=device),
        )
    else:
        rewards = list(map(compute_math_reward, completions, answers))
        # Convert to tensor on target device
        return torch.tensor(rewards, dtype=torch.float32, device=device)