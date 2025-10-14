#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
math_reward.py
--------------
Reward function for math QA with chain-of-thought (CoT) formatting.

Design goals
- Tolerant to different numeric formats (floats, scientific notation, simple fractions).
- Prefer *numeric* equality over string quirks (treat "4" == "4.0" == "4e0" == "8/2").
- Provide smooth partial credit for near-miss numbers.
- Reward helpful structure (<answer>...</answer>) with cheap binary bonus.
- Preserve within-group variance for GRPO's advantage normalization (CRITICAL).
- Use additive composition (NOT multiplicative) to avoid variance reduction.

Outputs (designed to preserve variance for RL training)
- correctness in [0.0, 1.0] (1.0 for exact, up to 0.15 for partial, 0.0 for wrong)
- format bonus: 0.1 per tag (answer + reasoning = 0.0, 0.1, or 0.2 total)
- final reward in [0.0, 1.35] (NOT normalized to [0,1] to preserve variance)

Key principle: Format is a CONSTANT SHIFT, not a multiplicative factor.
This preserves the variance of the main correctness signal, which is critical
for GRPO's group-based advantage normalization where relative within-group
signal matters most.

Public API
- compute_math_reward(completion: str, gold_answer: Optional[str]) -> float
- compute_math_reward_batch(completions: Sequence[str], gold_answers: Optional[Sequence[str]]) -> List[float]
- compute_correctness_reward(...) -> Tuple[float, Optional[float], Optional[float]]
- compute_format_reward(completion: str, *, max_bonus: float = 0.2) -> float
- extract_answer_from_model_output(completion: str) -> Optional[str]
- extract_answer_from_dataset(sample: Mapping[str, Any]) -> Optional[str]
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "compute_math_reward",
    "compute_math_reward_batch",
    "compute_math_rewards_batch",  # Backward compatibility alias
    "compute_correctness_reward",
    "compute_format_reward",
    "extract_answer_from_model_output",
    "extract_answer_from_dataset",
    "extract_all_numbers",
    "extract_single_number",
    "extract_last_number",
    "extract_tag_contents",
    "last_tag_content",
    "numeric_equal",
    "relative_error",
]

# ---------- Numeric parsing utilities ----------

# Integers, decimals, scientific notation, or simple fractions like 3/4
# IMPORTANT: Fractions must come FIRST in the alternation, otherwise "1/3" matches as "1" and "3"
_NUMBER_REGEX = re.compile(
    r"""
    (?P<num>
        [+-]?(
            (?:\d+/\d+)                          # 3/4 (must be first!)
            |
            (?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)  # 12, 12.3, 1e-3, 1.2e+5
        )
    )
    """,
    re.VERBOSE,
)


def _normalize_thousands_separators(s: str) -> str:
    """
    Remove thousands separators without harming decimals.
    Strategy:
      - If both '.' and ',' appear, assume ',' are thousands and remove them.
      - Else if only ',' appears and there are patterns like d{1,3}(,d{3})+, remove commas.
      - Else leave as-is.
    """
    if "." in s and "," in s:
        return s.replace(",", "")
    if "," in s and re.search(r"\d{1,3}(?:,\d{3})+(?!\d)", s):
        return s.replace(",", "")
    return s


def _to_float(tok: str) -> Optional[float]:
    """Parse a numeric token that may be a fraction or scientific notation."""
    tok = tok.strip()
    tok = _normalize_thousands_separators(tok)
    # Fraction (but not scientific notation with 'e/E')
    if "/" in tok and not any(c in tok for c in "eE"):
        n, d = tok.split("/", 1)
        try:
            nf, df = float(n), float(d)
            if df == 0.0:
                return None
            return nf / df
        except Exception:
            return None
    # Plain float or scientific notation
    try:
        return float(tok)
    except Exception:
        return None


def extract_all_numbers(text: Optional[str]) -> List[float]:
    if text is None:
        return []
    matches = [m.group("num") for m in _NUMBER_REGEX.finditer(str(text))]
    vals: List[float] = []
    for m in matches:
        v = _to_float(m)
        if v is not None:
            vals.append(v)
    return vals


def extract_single_number(text: Optional[str]) -> Optional[float]:
    """Return only if exactly one number appears; else None."""
    nums = extract_all_numbers(text)
    return nums[0] if len(nums) == 1 else None


def extract_last_number(text: Optional[str]) -> Optional[float]:
    """Return the last parsed number in the text (useful for CoT)."""
    nums = extract_all_numbers(text)
    return nums[-1] if nums else None


# ---------- Tag helpers ----------

def extract_tag_contents(text: str, tag: str) -> List[str]:
    """Return all non-overlapping contents of <tag>...</tag>, case-insensitive."""
    pattern = re.compile(rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>", re.IGNORECASE | re.DOTALL)
    return [m.group(1).strip() for m in pattern.finditer(text or "")]


def last_tag_content(text: str, tag: str) -> Optional[str]:
    items = extract_tag_contents(text, tag)
    return items[-1] if items else None


# ---------- Equality / distance ----------

def numeric_equal(a: float, b: float, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    """Robust numeric equality (works for small and large magnitudes)."""
    return abs(a - b) <= max(atol, rtol * max(1.0, abs(a), abs(b)))


def relative_error(pred: float, gold: float) -> float:
    denom = max(1.0, abs(gold))
    return abs(pred - gold) / denom


def _canonical_number_string(x: float) -> str:
    """
    Canonical string form for a numeric value:
    - If very close to an integer, return integer string.
    - Otherwise, 12 significant digits, trimmed.
    """
    if numeric_equal(x, round(x), rtol=0.0, atol=1e-9):
        return str(int(round(x)))
    s = f"{x:.12g}"
    return s


# ---------- Correctness reward ----------

def compute_correctness_reward(
    completion: str,
    gold_answer: Optional[str],
    partial_credit: bool = True,  # Backward compatibility parameter
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Compute correctness in [0.0, 1.0]. Returns (correctness, model_number, gold_number).

    Rules:
    - If a numeric gold answer cannot be parsed, return 0.0 (caller may handle non-numeric tasks differently).
    - Prefer the numeric value inside the last <answer>...</answer> if present.
    - Else fall back to the *last* number in the completion.
    - Exact numeric equality → 1.0.
    - Otherwise (if a numeric pred exists and partial_credit=True) partial credit: 0..0.15 via exp(-15*relative_error).
    - If partial_credit=False, wrong answers get 0.0 (for evaluation).

    Args:
        completion: Model output text
        gold_answer: Ground truth answer
        partial_credit: Whether to give partial credit for close answers (default: True)
    """
    # Parse gold
    gold_num: Optional[float] = None
    if gold_answer is not None:
        gold_num = extract_single_number(gold_answer)
        if gold_num is None:
            gold_num = extract_last_number(gold_answer)

    if gold_num is None:
        return 0.0, None, None

    # Model number: prefer last <answer> content else last number in whole completion
    answer_text = last_tag_content(completion or "", "answer")
    model_num: Optional[float] = None
    if answer_text is not None:
        model_num = extract_single_number(answer_text)
        if model_num is None:
            model_num = extract_last_number(answer_text)
    if model_num is None:
        model_num = extract_last_number(completion or "")

    if model_num is None:
        return 0.0, None, gold_num

    # Numeric equality
    if numeric_equal(model_num, gold_num):
        return 1.0, model_num, gold_num

    # Partial credit (only if enabled)
    if partial_credit:
        rerr = relative_error(model_num, gold_num)
        # Partial credit with steeper decay for better separation
        # Max 0.15, decay exp(-15*error) - much steeper than before
        # This creates bigger difference between close (2% → 0.11) and moderate (12% → 0.02)
        partial = max(0.0, min(0.15, 0.15 * math.exp(-15.0 * rerr)))
        # Floor: if partial credit is tiny (< 0.015), treat as 0.0
        # This ensures only truly close answers get partial credit
        if partial < 0.015:
            partial = 0.0
        return partial, model_num, gold_num
    else:
        # No partial credit for evaluation
        return 0.0, model_num, gold_num


# ---------- Format reward ----------

def compute_format_reward(completion: str, *, max_bonus: float = 0.2) -> float:
    """
    Binary/cheap format reward: 0.1 per tag (answer + reasoning).

    Returns a small constant bonus for each properly formatted tag:
    - +0.1 for <answer> tag with non-empty content
    - +0.1 for <reasoning> tag with non-empty content
    - Total: 0.0, 0.1, or 0.2

    This keeps format reward cheap but encourages structure without
    reducing the variance of the main correctness signal.

    This design preserves within-group variance for GRPO's advantage normalization
    by keeping format as a constant shift rather than multiplicative factor.

    Args:
        completion: Model output text
        max_bonus: Maximum format bonus (default: 0.2)

    Returns:
        0.0, 0.1, or 0.2 depending on which tags are present (clamped by max_bonus)
    """
    if not completion:
        return 0.0

    answer_chunks = extract_tag_contents(completion, "answer")
    reasoning_chunks = extract_tag_contents(completion, "reasoning")

    bonus = 0.0

    # +0.1 for answer tag with non-empty content
    if answer_chunks:
        last_ans = answer_chunks[-1].strip()
        if last_ans:
            bonus += 0.1

    # +0.1 for reasoning tag with non-empty content
    if reasoning_chunks and any(x.strip() for x in reasoning_chunks):
        bonus += 0.1

    # Clamp to max_bonus
    return min(bonus, max_bonus)


# ---------- Combined reward (single) ----------

def compute_math_reward(completion: str, gold_answer: Optional[str] = None) -> float:
    """
    Final reward with additive composition preserving within-group variance.

    Reward composition (additive, NOT multiplicative):
      - Correctness: main signal (0.0 or 1.0, or 0..0.15 for partial credit)
      - Format bonus: 0.1 per tag (answer + reasoning = 0.0, 0.1, or 0.2 total)

    This gives:
      - Correct with both tags: 1.0 + 0.2 = 1.2
      - Correct with answer tag: 1.0 + 0.1 = 1.1
      - Correct without format: 1.0 + 0.0 = 1.0
      - Wrong answer: 0.0 + 0.0 = 0.0
      - Partial credit with both tags: [0.0..0.15] + 0.2

    Key design principle:
      Format is a CONSTANT SHIFT (not multiplicative factor), preserving the
      variance of the main correctness signal. This is critical for GRPO's
      group-based advantage normalization, where relative within-group signal
      matters most.

    Example within-group rewards:
      Old (multiplicative): [0.85, 0.82, 0.80, 0.0] → variance reduced
      New (additive): [1.2, 1.1, 1.0, 0.0] → variance preserved

    Returns:
        Reward in [0.0, 1.35] range (not normalized to [0,1] to preserve variance)
        Typical range: [0.0, 1.2] for correct answers with 0-2 tags
    """
    completion = completion or ""

    format_bonus = compute_format_reward(completion)  # 0.0, 0.1, or 0.2
    correctness, _model_num, _gold_num = compute_correctness_reward(completion, gold_answer)

    if correctness <= 0.0:
        # Wrong answer or no answer → no reward (no format bonus for wrong answers)
        return 0.0

    # Additive composition: correctness + constant format bonus
    # This preserves variance for GRPO's group normalization
    reward = correctness + format_bonus

    # No upper clamp - allow rewards > 1.0 to preserve variance
    # Only clamp at 0.0 for safety
    return float(max(0.0, reward))


# ---------- Combined reward (batch) ----------

def compute_math_reward_batch(
    completions: Sequence[str],
    gold_answers: Optional[Sequence[Optional[str]]] = None,
) -> List[float]:
    """
    Vectorized wrapper over `compute_math_reward`.

    Args
    ----
    completions : sequence of model outputs (strings).
    gold_answers : sequence of gold answers aligned with `completions`.
                   If None or shorter than `completions`, missing entries are treated as None.

    Returns
    -------
    List[float] : rewards per example, each in [0.0, 1.35] range (NOT normalized to [0,1]).
                  0.0 for wrong answers, 1.0-1.2 for correct answers (depends on format).

    Notes
    -----
    - Gracefully handles length mismatches by using `None` for missing gold items.
    - Keeps identical logic with the single-example path to avoid divergence.
    - Uses additive composition (correctness + format_bonus) to preserve variance.
    """
    n = len(completions)
    rewards: List[float] = []
    for i in range(n):
        gold = None
        if gold_answers is not None and i < len(gold_answers):
            gold = gold_answers[i]
        rewards.append(compute_math_reward(completions[i], gold))
    return rewards


# ---------- Convenience extractors ----------

def extract_answer_from_model_output(completion: str) -> Optional[str]:
    """
    Extract a *string* answer from a model completion.

    Priority:
      1) Return the raw content of the last <answer>...</answer> tag if present (stripped).
      2) Otherwise return the *last* number found anywhere in the completion, formatted canonically.
      3) If nothing parseable is found, return None.

    This mirrors the reward's own extraction rules so logs and demos match training behavior.
    """
    if not completion:
        return None

    ans = last_tag_content(completion, "answer")
    if ans:
        ans = ans.strip()
        if ans:
            return ans

    num = extract_last_number(completion)
    if num is not None:
        return _canonical_number_string(num)

    return None


def extract_answer_from_dataset(sample: Mapping[str, Any]) -> Optional[str]:
    """
    Extract a gold answer string from a dataset sample.

    Tries common key names first (in priority order), then falls back to scanning
    other text fields for a final/last number.

    Returns:
      - A string (preferably as stored in the dataset), or a canonicalized numeric string,
      - None if nothing sensible is found.
    """
    if sample is None:
        return None

    # Priority keys commonly used across math datasets
    priority_keys = [
        "answer", "gold", "gold_answer", "final_answer", "label", "target",
        "expected", "ground_truth", "groundtruth", "solution", "output",
        "outputs", "reference", "references",
    ]

    # 1) Direct keys
    for k in priority_keys:
        if k in sample:
            val = sample[k]
            if isinstance(val, (int, float)):
                return _canonical_number_string(float(val))
            if isinstance(val, str) and val.strip():
                return val.strip()
            # Handle small containers
            if isinstance(val, (list, tuple)) and val:
                v = val[-1]
                if isinstance(v, (int, float)):
                    return _canonical_number_string(float(v))
                if isinstance(v, str) and v.strip():
                    return v.strip()
            if isinstance(val, Mapping):
                # common nested variants
                nested = val.get("text") or val.get("answer") or val.get("value")
                if isinstance(nested, (int, float)):
                    return _canonical_number_string(float(nested))
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()

    # 2) Try other plausible text fields and extract last number
    candidate_text_keys = [
        "explanation", "rationale", "reasoning", "solution_text",
        "completion", "prediction", "response",
        "question", "prompt", "instruction",
    ]
    blobs: List[str] = []
    for k in candidate_text_keys:
        v = sample.get(k)
        if isinstance(v, str):
            blobs.append(v)
        elif isinstance(v, (list, tuple)):
            blobs.extend([str(x) for x in v if isinstance(x, (str, int, float))])
        elif isinstance(v, Mapping):
            w = v.get("text") or v.get("value")
            if isinstance(w, (str, int, float)):
                blobs.append(str(w))

    joined = "\n".join(blobs).strip()
    if joined:
        num = extract_last_number(joined)
        if num is not None:
            return _canonical_number_string(num)

    return None


# ---------- Lightweight tests ----------

def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _run_self_tests():
    # Single tests - Updated for additive reward composition
    # With new formula: correctness + format_bonus (0.1 per tag: answer + reasoning)
    # - correct (1.0) + both tags (0.2) → 1.2
    # - correct (1.0) + answer tag (0.1) → 1.1
    # - correct (1.0) + no tag → 1.0
    # - wrong (0.0) + any tags → 0.0 (no format bonus for wrong answers)
    tests = [
        ("<answer>4</answer>", "4", 1.05, "exact integer with answer tag"),  # 1.1
        ("<answer>4.0</answer>", "4", 1.05, "exact float with answer tag"),  # 1.1
        ("<reasoning>stuff</reasoning><answer>4</answer>", "4", 1.15, "both tags"),  # 1.2
        ("We compute... final: <answer>3</answer>", "4", None, "wrong answer (partial credit possible but should be 0.0 due to floor)"),
        ("No tag but the last number is 4", "4", 0.95, "fallback last number, no format"),  # 1.0
        ("<answer>2e-3</answer>", "0.002", 1.05, "scientific notation with answer tag"),  # 1.1
        ("<answer>1/3</answer>", "0.3333333", 1.05, "fraction equals decimal with answer tag"),  # 1.1
        ("<reasoning>stuff</reasoning><answer>4</answer><answer>4</answer>", "4", 1.15, "multiple answers (gets both bonuses)"),  # 1.2
    ]

    for comp, gold, min_reward, name in tests:
        r = compute_math_reward(comp, gold)
        if min_reward is not None and r + 1e-6 < min_reward:
            raise AssertionError(f"Test failed: {name}, got {r:.4f}, expected at least {min_reward}")
    # Spot check partial: 3 vs 4 now gives 0.0 due to floor (25% error is too large)
    r_partial = compute_math_reward("<answer>3</answer>", "4")
    if not (0.0 <= r_partial <= 1.35):
        raise AssertionError(f"Partial credit out of range: {r_partial}")

    # Check that close answer (3.9 vs 4, 2.5% error) does get partial credit
    r_close = compute_math_reward("<answer>3.9</answer>", "4")
    if not (0.0 < r_close <= 1.35):
        raise AssertionError(f"Close answer should get partial credit: {r_close}")

    # No number anywhere - now gets NO reward (0.0 for wrong answers, no format bonus)
    r_zero = compute_math_reward("I refuse to answer.", "4")
    if r_zero != 0.0:  # Must be exactly 0.0 for wrong/missing answers
        raise AssertionError(f"No-number case should be 0.0, got {r_zero}")

    # Batch tests - Updated with additive rewards
    batch_comps = ["<answer>4</answer>", "<answer>3</answer>", "final 0.002"]
    batch_gold = ["4", "4", "0.002"]
    br = compute_math_reward_batch(batch_comps, batch_gold)
    # br[0]: correct (4 vs 4) + answer tag (0.1) → 1.1
    # br[1]: wrong (3 vs 4, 25% error exceeds floor) → 0.0
    # br[2]: correct (0.002 vs 0.002), no format tag → 1.0
    if not (len(br) == 3 and br[0] >= 1.05 and br[1] == 0.0 and br[2] >= 0.95):
        raise AssertionError(f"Batch path failed: {br}")

    # Extractor tests (model output)
    assert extract_answer_from_model_output("<answer> 004 </answer>") == "004"
    assert extract_answer_from_model_output("... therefore the answer is 42.") == "42"
    assert extract_answer_from_model_output("no numbers here") is None

    # Dataset extractor tests
    sample1 = {"answer": "4"}
    sample2 = {"gold_answer": 4.0}
    sample3 = {"solution": {"text": "Answer: 1/2"}}
    sample4 = {"completion": "We get 0.125 as the final value."}
    assert extract_answer_from_dataset(sample1) == "4"
    assert extract_answer_from_dataset(sample2) == "4"
    assert extract_answer_from_dataset(sample3) == "Answer: 1/2"
    assert extract_answer_from_dataset(sample4) == "0.125"


# Backward compatibility wrapper (old code uses plural "rewards" and different signature)
def compute_math_rewards_batch(
    completions: Sequence[str],
    answers: Optional[Sequence[Optional[str]]] = None,
    device: Optional[str] = None,
    return_breakdown: bool = False,
):
    """
    Backward-compatible wrapper for compute_math_reward_batch.

    Old signature used by GRPO:
        batch_reward_fn(completions, answers, device, return_breakdown=True)

    Returns:
        - If return_breakdown=True: (total_rewards, format_rewards, correctness_rewards) as tensors
        - If return_breakdown=False: total_rewards as tensor
    """
    import torch

    # Get rewards using the new implementation
    rewards_list = compute_math_reward_batch(completions, answers)

    if return_breakdown:
        # Need to also compute format and correctness separately
        format_rewards_list = []
        correctness_rewards_list = []

        for comp, ans in zip(completions, answers or [None] * len(completions)):
            format_score = compute_format_reward(comp)
            correctness, _, _ = compute_correctness_reward(comp, ans)
            format_rewards_list.append(format_score)
            correctness_rewards_list.append(correctness)

        # Convert to tensors
        total_rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        format_rewards = torch.tensor(format_rewards_list, dtype=torch.float32, device=device)
        correctness_rewards = torch.tensor(correctness_rewards_list, dtype=torch.float32, device=device)

        return total_rewards, format_rewards, correctness_rewards
    else:
        # Just return total rewards as tensor
        return torch.tensor(rewards_list, dtype=torch.float32, device=device)


if __name__ == "__main__":
    _run_self_tests()
    print("math_reward.py: self-tests passed.")
