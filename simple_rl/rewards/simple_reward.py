"""
Simple reward functions for mathematical problem solving in reinforcement learning.

This module provides reward functions that evaluate model completions based on:
1. Correctness of the answer compared to ground truth
2. Proper formatting with required XML tags

Functions are designed to work with mathematical reasoning tasks where models
should provide answers in a structured format with <reasoning> and <answer> tags.
"""

import re
import torch
from typing import List, Optional, Union


# Precompile regex for performance
NUMBER_PATTERN = re.compile(r'-?\d+\.?\d*')


def extract_answer_from_model_output(text: str) -> Optional[str]:
    """
    Extract answer from model output enclosed in <answer> tags.
    
    Args:
        text: Model completion text
        
    Returns:
        Extracted answer string or None if not found/invalid
    """
    text_parts = text.split("<answer>")
    if len(text_parts) < 2:
        return None
    end = text_parts[-1]
    if "</answer>" not in end:
        return None
    answer = end.split("</answer>")[0].strip()
    return None if answer == "..." else answer


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
    Extract the first number from text, handling currency symbols and commas.
    
    Args:
        text: Input text that may contain numbers
        
    Returns:
        First number found as float or None if no valid number
    """
    if text is None:
        return None
    numbers = NUMBER_PATTERN.findall(str(text).replace('$', '').replace(',', ''))
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


def correctness_reward(completion: str, answer: Optional[str] = None) -> float:
    """
    Compute correctness reward based on answer accuracy.
    
    Reward structure:
    - 2.0: Exact string match
    - 1.5: Numerical match (within 0.01 tolerance)
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
    if model_answer == answer:
        return 2.0
    
    model_num = extract_single_number(model_answer)
    answer_num = extract_single_number(answer)
    
    if model_num is not None and answer_num is not None:
        if abs(model_num - answer_num) < 0.01:
            return 1.5
    return 0.0


def format_reward(completion: str) -> float:
    """
    Compute format reward based on presence of required XML tags.
    
    Reward structure:
    - 0.25 for each tag: <reasoning>, </reasoning>, <answer>, </answer>
    - 0.5 bonus if all tags are present (total 1.5)
    
    Args:
        completion: Model completion text
        
    Returns:
        Format reward (0.0 to 1.5)
    """
    score = 0.0
    score += 0.25 if "<reasoning>" in completion else 0
    score += 0.25 if "</reasoning>" in completion else 0
    score += 0.25 if "<answer>" in completion else 0
    score += 0.25 if "</answer>" in completion else 0
    
    if score == 1.0:  # All tags present
        score += 0.5
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
    device: Union[str, torch.device, None] = None
) -> torch.Tensor:
    """
    Compute rewards for a batch of completions.
    Returns tensor directly on specified device (cuda/mps/cpu).
    
    Args:
        completions: List of completion strings or tensor of token ids
        answers: List of ground truth answers (optional)
        device: Target device for output tensor
    
    Returns:
        Tensor of rewards on specified device
        
    Raises:
        NotImplementedError: If completions are provided as token tensor
    """
    # If completions are token ids, decode them first
    if isinstance(completions, torch.Tensor):
        # Assume you have tokenizer available globally or pass it in
        # completions = tokenizer.batch_decode(completions, skip_special_tokens=True)
        raise NotImplementedError("Pass decoded strings or implement tokenizer decoding")
    
    batch_size = len(completions)
        
    # Vectorized processing
    if answers is None:
        answers = [None] * batch_size
    
    rewards = list(map(compute_math_reward, completions, answers))
    
    # Convert to tensor on target device
    return torch.tensor(rewards, dtype=torch.float32, device=device)
