"""Math reward function with anti-gaming protections."""

import math
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

    - 0.1 per tag (max 0.4)
    - Bonus 0.1 for clean completion (no text after </answer>)
    - Penalties for trivial answers
    - Penalties for repetitive tags (both opening and closing)
    - Penalties for missing numbers in answer

    Repetitive tag penalty: score *= 0.5^(count-1) for each duplicate tag
    Example: 2 tags = 0.5x, 3 tags = 0.25x, 4 tags = 0.125x

    Returns:
        Score between 0.0 and 0.5
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

    # Bonus: Clean completion (no extra text after </answer>)
    if "</answer>" in completion:
        after_answer = completion.split("</answer>", 1)[-1].strip()
        if len(after_answer) == 0:
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

    # Penalty: repetitive tags (both opening and closing)
    answer_open_count = completion.count("<answer>")
    answer_close_count = completion.count("</answer>")
    reasoning_open_count = completion.count("<reasoning>")
    reasoning_close_count = completion.count("</reasoning>")

    # Penalize multiple answer tags
    if answer_open_count > 1:
        score *= (0.5 ** (answer_open_count - 1))
    if answer_close_count > 1:
        score *= (0.5 ** (answer_close_count - 1))

    # Penalize multiple reasoning tags
    if reasoning_open_count > 1:
        score *= (0.5 ** (reasoning_open_count - 1))
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
    - 0.05-0.5: Distance-based partial credit (closer = higher reward, if partial_credit=True)
    - 0.0: Non-numeric answer or no answer (strict)
    - Fallback: Search whole completion if no <answer> tag
    - 0.0: No number found or wrong (if partial_credit=False)

    Distance-based partial credit formula (strict exponential decay):
        reward = 0.5 * exp(-5 * relative_error)
        where relative_error = |model - correct| / (|correct| + 1)
        Examples:
          - Off by 1 when answer is 100: relative_error=0.01, reward≈0.48
          - Off by 10 when answer is 100: relative_error=0.1, reward≈0.30
          - Off by 20 when answer is 100: relative_error=0.2, reward≈0.18
          - Off by 50 when answer is 100: relative_error=0.5, reward≈0.04
          - Off by 100 when answer is 100: relative_error=1.0, reward≈0.003
          - Non-numeric text answer: 0.0 (no reward for non-numeric attempts)

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

            # Distance-based partial credit (strict exponential decay)
            if partial_credit:
                # Calculate relative error (add 1 to denominator to avoid division by zero)
                relative_error = abs(mnum - anum) / (abs(anum) + 1.0)
                # Exponential decay: reward = 0.5 * exp(-5 * relative_error)
                # This is much stricter - only close answers get significant reward
                # 10% error → 0.30, 20% error → 0.18, 50% error → 0.04
                distance_reward = 0.5 * math.exp(-5.0 * relative_error)
                return max(0.0, min(0.5, distance_reward))
            else:
                return 0.0

        # Non-numeric answer gets no reward (strict)
        if len(model_answer.strip()) > 2:
            return 0.0

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

        # Distance-based partial credit for fallback (same strict exponential decay)
        if partial_credit:
            relative_error = abs(mnum - anum) / (abs(anum) + 1.0)
            distance_reward = 0.5 * math.exp(-5.0 * relative_error)
            return max(0.0, min(0.5, distance_reward))
        else:
            return 0.0

    return 0.0


def compute_math_reward(
    completion: str,
    answer: Optional[str] = None
) -> float:
    """
    Total reward: format + correctness, with format conditional on correctness.

    Format reward is scaled based on correctness to prevent reward hacking:
    - correctness >= 1.0: Full format reward (100%)
    - correctness >= 0.5: Partial format reward (50%)
    - correctness < 0.5: Minimal format reward (10%)

    This prevents the model from gaming format rewards while ignoring correctness.

    Returns:
        Score between 0.0 and 2.5 (0.5 format + 2.0 correctness)
    """
    format_score = compute_format_reward(completion)
    correctness_score = compute_correctness_reward(completion, answer)

    # Scale format reward based on correctness
    # CRITICAL: Only reward format for CORRECT answers to prevent gaming
    # Correctness ranges: 2.0 = exact, 1.5 = numeric match, 0.5-1.0 = close, <0.5 = wrong
    if correctness_score >= 2.0:
        # Exact match: full format reward
        format_multiplier = 1.0
    elif correctness_score >= 1.5:
        # Numeric match: moderate format reward
        format_multiplier = 0.5
    elif correctness_score >= 0.5:
        # Close but NOT correct: small format reward
        format_multiplier = 0.3
    else:
        # Wrong answer: NO format reward
        format_multiplier = 0.0

    return format_score * format_multiplier + correctness_score


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

        # Apply conditional format scaling (same logic as compute_math_reward)
        total_rewards = []
        for f, c in zip(format_rewards, correctness_rewards):
            if c >= 2.0:
                format_multiplier = 1.0  # Exact match
            elif c >= 1.5:
                format_multiplier = 0.5  # Numeric match
            elif c >= 0.5:
                format_multiplier = 0.3  # Close but not correct
            else:
                format_multiplier = 0.0  # Wrong answer
            total_rewards.append(f * format_multiplier + c)

        return (
            torch.tensor(total_rewards, dtype=torch.float32, device=device),
            torch.tensor(format_rewards, dtype=torch.float32, device=device),
            torch.tensor(correctness_rewards, dtype=torch.float32, device=device),
        )
    else:
        rewards = [compute_math_reward(c, a) for c, a in zip(completions, answers)]
        return torch.tensor(rewards, dtype=torch.float32, device=device)
