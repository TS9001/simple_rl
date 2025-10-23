"""Reward function for math QA with CoT formatting."""

from __future__ import annotations

import math
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "compute_math_reward",
    "compute_math_reward_batch",
    "compute_math_rewards_batch",
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

_NUMBER_REGEX = re.compile(
    r"""
    (?P<num>
        [+-]?(
            (?:\d+/\d+)
            |
            (?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
        )
    )
    """,
    re.VERBOSE,
)


def _normalize_thousands_separators(s: str) -> str:
    """Remove thousands separators without harming decimals."""
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
    """Return only if exactly one number appears."""
    nums = extract_all_numbers(text)
    return nums[0] if len(nums) == 1 else None


def extract_last_number(text: Optional[str]) -> Optional[float]:
    """Return the last parsed number in the text."""
    nums = extract_all_numbers(text)
    return nums[-1] if nums else None


def extract_tag_contents(text: str, tag: str) -> List[str]:
    """Return all non-overlapping contents of <tag>...</tag>."""
    pattern = re.compile(rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>", re.IGNORECASE | re.DOTALL)
    return [m.group(1).strip() for m in pattern.finditer(text or "")]


def last_tag_content(text: str, tag: str) -> Optional[str]:
    items = extract_tag_contents(text, tag)
    return items[-1] if items else None


def numeric_equal(a: float, b: float, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    """Robust numeric equality."""
    return abs(a - b) <= max(atol, rtol * max(1.0, abs(a), abs(b)))


def relative_error(pred: float, gold: float) -> float:
    denom = max(1.0, abs(gold))
    return abs(pred - gold) / denom


def _canonical_number_string(x: float) -> str:
    """Canonical string form for a numeric value."""
    if numeric_equal(x, round(x), rtol=0.0, atol=1e-9):
        return str(int(round(x)))
    s = f"{x:.12g}"
    return s


def compute_correctness_reward(
    completion: str,
    gold_answer: Optional[str],
    partial_credit: bool = True,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Compute correctness in [0.0, 1.0]. Returns (correctness, model_number, gold_number)."""
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

    if partial_credit:
        rerr = relative_error(model_num, gold_num)
        partial = max(0.0, min(0.15, 0.15 * math.exp(-15.0 * rerr)))
        if partial < 0.015:
            partial = 0.0
        return partial, model_num, gold_num
    else:
        return 0.0, model_num, gold_num


def compute_format_reward(completion: str, *, max_bonus: float = 0.2) -> float:
    """Format reward: 0.1 per tag (answer + reasoning), total 0.0-0.2."""
    if not completion:
        return 0.0

    answer_chunks = extract_tag_contents(completion, "answer")
    reasoning_chunks = extract_tag_contents(completion, "reasoning")

    bonus = 0.0

    if answer_chunks:
        last_ans = answer_chunks[-1].strip()
        if last_ans:
            bonus += 0.1

    if reasoning_chunks and any(x.strip() for x in reasoning_chunks):
        bonus += 0.1

    return min(bonus, max_bonus)


def compute_math_reward(completion: str, gold_answer: Optional[str] = None) -> float:
    """Combined reward: correctness + format bonus (additive). Returns [0.0, 1.35]."""
    completion = completion or ""

    format_bonus = compute_format_reward(completion)
    correctness, _model_num, _gold_num = compute_correctness_reward(completion, gold_answer)

    if correctness <= 0.0:
        return 0.0

    reward = correctness + format_bonus
    return float(max(0.0, reward))


def compute_math_reward_batch(
    completions: Sequence[str],
    gold_answers: Optional[Sequence[Optional[str]]] = None,
) -> List[float]:
    """Batch wrapper over compute_math_reward."""
    n = len(completions)
    rewards: List[float] = []
    for i in range(n):
        gold = None
        if gold_answers is not None and i < len(gold_answers):
            gold = gold_answers[i]
        rewards.append(compute_math_reward(completions[i], gold))
    return rewards


def extract_answer_from_model_output(completion: str) -> Optional[str]:
    """Extract answer from model completion (prefers <answer> tag, falls back to last number)."""
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
    """Extract gold answer from dataset sample using common key names."""
    if sample is None:
        return None

    priority_keys = [
        "answer", "gold", "gold_answer", "final_answer", "label", "target",
        "expected", "ground_truth", "groundtruth", "solution", "output",
        "outputs", "reference", "references",
    ]
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
                nested = val.get("text") or val.get("answer") or val.get("value")
                if isinstance(nested, (int, float)):
                    return _canonical_number_string(float(nested))
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()

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

def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _run_self_tests():
    tests = [
        ("<answer>4</answer>", "4", 1.05, "exact integer"),
        ("<answer>4.0</answer>", "4", 1.05, "exact float"),
        ("<reasoning>stuff</reasoning><answer>4</answer>", "4", 1.15, "both tags"),
        ("We compute... final: <answer>3</answer>", "4", None, "wrong answer"),
        ("No tag but the last number is 4", "4", 0.95, "fallback"),
        ("<answer>2e-3</answer>", "0.002", 1.05, "scientific"),
        ("<answer>1/3</answer>", "0.3333333", 1.05, "fraction"),
        ("<reasoning>stuff</reasoning><answer>4</answer><answer>4</answer>", "4", 1.15, "multiple"),
    ]

    for comp, gold, min_reward, name in tests:
        r = compute_math_reward(comp, gold)
        if min_reward is not None and r + 1e-6 < min_reward:
            raise AssertionError(f"Test failed: {name}, got {r:.4f}, expected at least {min_reward}")

    r_partial = compute_math_reward("<answer>3</answer>", "4")
    if not (0.0 <= r_partial <= 1.35):
        raise AssertionError(f"Partial credit out of range: {r_partial}")

    r_close = compute_math_reward("<answer>3.9</answer>", "4")
    if not (0.0 < r_close <= 1.35):
        raise AssertionError(f"Close answer should get partial credit: {r_close}")

    r_zero = compute_math_reward("I refuse to answer.", "4")
    if r_zero != 0.0:
        raise AssertionError(f"No-number case should be 0.0, got {r_zero}")

    batch_comps = ["<answer>4</answer>", "<answer>3</answer>", "final 0.002"]
    batch_gold = ["4", "4", "0.002"]
    br = compute_math_reward_batch(batch_comps, batch_gold)
    if not (len(br) == 3 and br[0] >= 1.05 and br[1] == 0.0 and br[2] >= 0.95):
        raise AssertionError(f"Batch path failed: {br}")
    assert extract_answer_from_model_output("<answer> 004 </answer>") == "004"
    assert extract_answer_from_model_output("... therefore the answer is 42.") == "42"
    assert extract_answer_from_model_output("no numbers here") is None

    sample1 = {"answer": "4"}
    sample2 = {"gold_answer": 4.0}
    sample3 = {"solution": {"text": "Answer: 1/2"}}
    sample4 = {"completion": "We get 0.125 as the final value."}
    assert extract_answer_from_dataset(sample1) == "4"
    assert extract_answer_from_dataset(sample2) == "4"
    assert extract_answer_from_dataset(sample3) == "Answer: 1/2"
    assert extract_answer_from_dataset(sample4) == "0.125"



def compute_math_rewards_batch(
    completions: Sequence[str],
    answers: Optional[Sequence[Optional[str]]] = None,
    device: Optional[str] = None,
    return_breakdown: bool = False,
):
    """Backward-compatible wrapper returning tensors."""
    import torch

    rewards_list = compute_math_reward_batch(completions, answers)

    if return_breakdown:
        format_rewards_list = []
        correctness_rewards_list = []

        for comp, ans in zip(completions, answers or [None] * len(completions)):
            format_score = compute_format_reward(comp)
            correctness, _, _ = compute_correctness_reward(comp, ans)
            format_rewards_list.append(format_score)
            correctness_rewards_list.append(correctness)

        total_rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        format_rewards = torch.tensor(format_rewards_list, dtype=torch.float32, device=device)
        correctness_rewards = torch.tensor(correctness_rewards_list, dtype=torch.float32, device=device)

        return total_rewards, format_rewards, correctness_rewards
    else:
        return torch.tensor(rewards_list, dtype=torch.float32, device=device)


if __name__ == "__main__":
    _run_self_tests()
    print("math_reward.py: self-tests passed.")
