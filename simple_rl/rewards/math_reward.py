"""Math reward function with anti-gaming protections."""

import re
from typing import List, Optional, Union
import torch


NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def extract_answer_from_model_output(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not matches:
        return None
    answer = matches[-1].strip()
    if answer in ("...", ""):
        return None
    return answer


def extract_single_number(text: Optional[str]) -> Optional[float]:
    """Extract single number if exactly one exists."""
    if text is None:
        return None
    numbers = NUMBER_PATTERN.findall(str(text))
    if len(numbers) == 1:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


def extract_last_number(text: Optional[str]) -> Optional[float]:
    """Extract last number in text."""
    if text is None:
        return None
    matches = NUMBER_PATTERN.findall(str(text))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def extract_answer_from_dataset(text: str) -> Optional[str]:
    """Extract answer from dataset text (after #### delimiter)."""
    return text.split("####")[1].strip() if "####" in text else None


def compute_format_reward(completion: str) -> float:
    """
    Format reward with anti-gaming protections.

    - 0.1 per tag (max 0.4 instead of 0.8)
    - Penalties for trivial answers
    - Penalties for repetitive tags
    - Penalties for missing numbers in answer

    Returns:
        Score between 0.0 and 0.4
    """
    score = 0.0

    if "<reasoning>" in completion:
        score += 0.1
    if "</reasoning>" in completion:
        score += 0.1
    if "<answer>" in completion:
        score += 0.1
    if "</answer>" in completion:
        score += 0.1

    answer_content = extract_answer_from_model_output(completion)
    if answer_content is not None:
        answer_clean = answer_content.strip()

        # Penalty: trivial answers
        if len(answer_clean) <= 2:
            score *= 0.1
        elif answer_clean in [".", "..", "...", "-", "?", "N/A"]:
            score *= 0.05

        # Penalty: no numbers in answer
        if not re.search(r'\d', answer_clean):
            score *= 0.3

    # Penalty: repetitive tags
    answer_close_count = completion.count("</answer>")
    if answer_close_count > 1:
        score *= (0.5 ** (answer_close_count - 1))

    reasoning_close_count = completion.count("</reasoning>")
    if reasoning_close_count > 1:
        score *= (0.5 ** (reasoning_close_count - 1))

    return score


def compute_correctness_reward(
    completion: str,
    answer: Optional[str] = None,
    partial_credit: bool = True
) -> float:
    """
    Correctness reward with optional partial credit and fallback.

    - 2.0: Exact match
    - 1.5: Numeric match
    - 0.5: Wrong but has numeric attempt (if partial_credit=True)
    - 0.2: Non-trivial text attempt (if partial_credit=True)
    - Fallback: Search whole completion if no <answer> tag
    - 0.0: No number found or wrong (if partial_credit=False)

    Args:
        completion: Model completion text
        answer: Ground truth answer
        partial_credit: If False, only give reward for correct answers (for evaluation)

    Returns:
        Score between 0.0 and 2.0
    """
    if answer is None:
        return 0.0

    model_answer = extract_answer_from_model_output(completion)

    # Try <answer> tag first
    if model_answer is not None:
        if model_answer.strip() == answer.strip():
            return 2.0

        mnum = extract_single_number(model_answer)
        anum = extract_single_number(answer)

        if mnum is not None and anum is not None:
            if mnum == anum:
                return 1.5
            return 0.5 if partial_credit else 0.0

        if len(model_answer.strip()) > 2:
            return 0.2 if partial_credit else 0.0

    # Fallback: search whole completion
    mnum = extract_single_number(completion)
    if mnum is None:
        mnum = extract_last_number(completion)

    anum = extract_single_number(answer)
    if anum is None:
        anum = extract_last_number(answer)

    if mnum is not None and anum is not None:
        if mnum == anum:
            return 1.5
        return 0.5 if partial_credit else 0.0

    return 0.0


def compute_math_reward(
    completion: str,
    answer: Optional[str] = None
) -> float:
    """
    Total reward: format + correctness.

    Returns:
        Score between 0.0 and 2.4
    """
    return compute_format_reward(completion) + compute_correctness_reward(completion, answer)


def compute_math_rewards_batch(
    completions: Union[List[str], torch.Tensor],
    answers: Optional[List[str]] = None,
    device: Union[str, torch.device, None] = None,
    return_breakdown: bool = False,
) -> Union[torch.Tensor, tuple]:
    """
    Batch computation of math rewards.

    Args:
        completions: List of completion strings
        answers: List of ground truth answers
        device: Target device for tensors
        return_breakdown: If True, returns (total, format, correctness)

    Returns:
        Tensor of rewards or tuple of tensors
    """
    if isinstance(completions, torch.Tensor):
        raise NotImplementedError("Pass decoded strings")

    batch_size = len(completions)
    if answers is None:
        answers = [None] * batch_size

    if return_breakdown:
        format_rewards = [compute_format_reward(c) for c in completions]
        correctness_rewards = [compute_correctness_reward(c, a) for c, a in zip(completions, answers)]
        total_rewards = [f + c for f, c in zip(format_rewards, correctness_rewards)]

        return (
            torch.tensor(total_rewards, dtype=torch.float32, device=device),
            torch.tensor(format_rewards, dtype=torch.float32, device=device),
            torch.tensor(correctness_rewards, dtype=torch.float32, device=device),
        )
    else:
        rewards = [compute_math_reward(c, a) for c, a in zip(completions, answers)]
        return torch.tensor(rewards, dtype=torch.float32, device=device)
