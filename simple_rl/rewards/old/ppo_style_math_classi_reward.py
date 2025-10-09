"""Classic math reward with fallback numeric extraction.

This module provides a correctness + format reward commonly used in math RL:
- Prefer extracting the answer from <answer>...</answer> when present
- Otherwise fall back to numeric extraction from the whole output

Exports a batched API compatible with GRPO's expected signature.
"""

from typing import List, Optional, Union
import re
import torch

from .simple_reward import (
    extract_answer_from_model_output,
    extract_single_number,
    format_reward,
)


_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def extract_last_number(text: Optional[str]) -> Optional[float]:
    """Extract the last number in the text; None if not found."""
    if text is None:
        return None
    matches = _NUMBER_PATTERN.findall(str(text))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def correctness_reward_classic(completion: str, answer: Optional[str] = None) -> float:
    """
    Classic correctness reward with fallback extraction when tags are missing:
    - 2.0 if exact string match on extracted <answer>
    - 1.5 if single-number equality on extracted <answer>
    - else fallback: single/last-number equality on the whole completion → 1.5
    - 0.0 otherwise
    """
    if answer is None:
        return 0.0

    # Prefer structured extraction when available
    model_answer = extract_answer_from_model_output(completion)
    if model_answer is not None:
        if model_answer == answer:
            return 2.0
        mnum = extract_single_number(model_answer)
        anum = extract_single_number(answer)
        if mnum is not None and anum is not None and mnum == anum:
            return 1.5

    # Fallback to numeric extraction from the whole output
    mnum = extract_single_number(completion)
    if mnum is None:
        mnum = extract_last_number(completion)
    anum = extract_single_number(answer)
    if anum is None:
        anum = extract_last_number(answer)

    if mnum is not None and anum is not None and mnum == anum:
        return 1.5

    return 0.0


def compute_math_reward_classic(completion: str, answer: Optional[str] = None) -> float:
    return correctness_reward_classic(completion, answer) + format_reward(completion)


def compute_math_rewards_batch_classic(
    completions: Union[List[str], torch.Tensor],
    answers: Optional[List[str]] = None,
    device: Union[str, torch.device, None] = None,
    return_breakdown: bool = False,
):
    """
    Classic batch reward: prefers <answer> tags but falls back to numbers in output.
    Signature matches the existing batch reward used by GRPO.
    """
    if isinstance(completions, torch.Tensor):
        raise NotImplementedError(
            "Pass decoded strings or implement tokenizer decoding"
        )

    batch_size = len(completions)
    if answers is None:
        answers = [None] * batch_size

    if return_breakdown:
        format_rewards = list(map(format_reward, completions))
        correctness_rewards = list(map(correctness_reward_classic, completions, answers))
        total_rewards = [f + c for f, c in zip(format_rewards, correctness_rewards)]
        return (
            torch.tensor(total_rewards, dtype=torch.float32, device=device),
            torch.tensor(format_rewards, dtype=torch.float32, device=device),
            torch.tensor(correctness_rewards, dtype=torch.float32, device=device),
        )
    else:
        rewards = list(map(compute_math_reward_classic, completions, answers))
        return torch.tensor(rewards, dtype=torch.float32, device=device)


