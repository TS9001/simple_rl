"""
Comprehensive tests for math reward functions.

Tests the reward computation for mathematical reasoning tasks with
CoT (Chain-of-Thought) formatting.
"""

import pytest
import torch
import numpy as np

from simple_rl.rewards.math_reward import (
    extract_single_number,
    extract_last_number,
    extract_all_numbers,
    compute_format_reward,
    compute_correctness_reward,
    compute_math_reward,
    compute_math_reward_batch,
    extract_answer_from_model_output,
    extract_answer_from_dataset,
    compute_math_rewards_batch,
)


class TestNumberExtraction:
    """Test number extraction functions."""

    def test_extract_single_number_simple(self):
        """Test extracting single numbers."""
        assert extract_single_number("42") == 42.0
        assert extract_single_number("3.14") == 3.14
        assert extract_single_number("-17") == -17.0
        assert extract_single_number("0.5") == 0.5

    def test_extract_single_number_with_text(self):
        """Test extracting numbers from text."""
        assert extract_single_number("The answer is 42") == 42.0
        assert extract_single_number("Price: $99.99") == 99.99

    def test_extract_single_number_no_number(self):
        """Test when no number is present."""
        assert extract_single_number("no numbers here") is None
        assert extract_single_number("") is None

    def test_extract_last_number(self):
        """Test extracting the last number from text."""
        assert extract_last_number("1, 2, 3, 4") == 4.0
        assert extract_last_number("First 10, then 20, final 30") == 30.0
        assert extract_last_number("Only one: 42") == 42.0
        assert extract_last_number("no numbers") is None

    def test_extract_all_numbers(self):
        """Test extracting all numbers from text."""
        assert extract_all_numbers("1 2 3") == [1.0, 2.0, 3.0]
        assert extract_all_numbers("a=10, b=20.5, c=-3") == [10.0, 20.5, -3.0]
        assert extract_all_numbers("no numbers") == []

    def test_number_formats(self):
        """Test different number formats."""
        # Integers
        assert extract_single_number("123") == 123.0
        # Floats
        assert extract_single_number("123.456") == 123.456
        # Negative numbers
        assert extract_single_number("-456") == -456.0
        # Scientific notation (if supported)
        result = extract_single_number("1.5e3")
        assert result == 1500.0 or result == 1.5  # Either interpretation is reasonable


class TestFormatReward:
    """Test format reward computation."""

    def test_perfect_format(self):
        """Test completion with perfect formatting."""
        completion = "<reasoning>Let me think...</reasoning>\n<answer>42</answer>"
        reward = compute_format_reward(completion)
        assert abs(reward - 0.45) < 0.01, f"Expected ~0.45, got {reward}"

    def test_reasoning_only(self):
        """Test completion with only reasoning tags."""
        completion = "<reasoning>Some thinking</reasoning>"
        reward = compute_format_reward(completion)
        assert reward == 0.15, f"Expected 0.15, got {reward}"

    def test_answer_only(self):
        """Test completion with only answer tags."""
        completion = "<answer>42</answer>"
        reward = compute_format_reward(completion)
        assert reward == 0.30, f"Expected 0.30, got {reward}"

    def test_no_tags(self):
        """Test completion without any tags."""
        completion = "Just a plain answer: 42"
        reward = compute_format_reward(completion)
        assert reward == 0.0, f"Expected 0.0, got {reward}"

    def test_malformed_tags(self):
        """Test completions with malformed tags."""
        # Missing closing tag
        completion = "<reasoning>Some thinking"
        reward = compute_format_reward(completion)
        assert reward == 0.0, f"Expected 0.0, got {reward}"

        # Wrong order (answer before reasoning)
        completion = "<answer>42</answer>\n<reasoning>thought</reasoning>"
        reward = compute_format_reward(completion)
        # Should still get credit for having both tags
        assert abs(reward - 0.45) < 0.01, f"Expected ~0.45, got {reward}"


class TestCorrectnessReward:
    """Test correctness reward computation."""

    def test_exact_match(self):
        """Test exact answer match."""
        completion = "<answer>42</answer>"
        gold_answer = "42"
        correctness, model_num, gold_num = compute_correctness_reward(completion, gold_answer)
        assert correctness == 2.0, f"Expected 2.0, got {correctness}"
        assert model_num == 42.0
        assert gold_num == 42.0

    def test_wrong_answer(self):
        """Test completely wrong answer."""
        completion = "<answer>100</answer>"
        gold_answer = "42"
        correctness, model_num, gold_num = compute_correctness_reward(completion, gold_answer)
        # Should get partial credit due to relative error
        assert 0.0 < correctness < 2.0, f"Expected partial credit, got {correctness}"

    def test_no_answer_in_completion(self):
        """Test when model doesn't generate answer."""
        completion = "I don't know the answer"
        gold_answer = "42"
        correctness, model_num, gold_num = compute_correctness_reward(completion, gold_answer)
        assert correctness == 0.0
        assert model_num is None

    def test_no_gold_answer(self):
        """Test when no gold answer is provided."""
        completion = "<answer>42</answer>"
        gold_answer = None

        # Test correctness reward separately
        # When gold_answer is None, compute_correctness_reward returns (0.0, None, None)
        correctness, model_num, gold_num = compute_correctness_reward(completion, gold_answer)
        assert correctness == 0.0
        assert model_num is None  # Returns None when no gold answer
        assert gold_num is None

        # Test full reward (should get small format reward with mult=0.2)
        total_reward = compute_math_reward(completion, gold_answer)
        # format_r = 0.30, mult = 0.2 (since correctness < 0.5), total = 0.0 + 0.30*0.2 = 0.06
        assert abs(total_reward - 0.06) < 0.01, f"Expected ~0.06, got {total_reward}"

    def test_partial_credit_disabled(self):
        """Test that partial credit can be disabled."""
        completion = "<answer>40</answer>"
        gold_answer = "42"

        # With partial credit (default)
        correctness_with, _, _ = compute_correctness_reward(completion, gold_answer, partial_credit=True)

        # Without partial credit
        correctness_without, _, _ = compute_correctness_reward(completion, gold_answer, partial_credit=False)

        # With partial credit should give some points
        assert correctness_with > 0.0
        # Without partial credit should give 0
        assert correctness_without == 0.0

    def test_partial_credit_close_answer(self):
        """Test partial credit for close answers."""
        completion = "<answer>42.5</answer>"
        gold_answer = "42"
        correctness, _, _ = compute_correctness_reward(completion, gold_answer, partial_credit=True)

        # Should get partial credit for being close
        assert 0.0 < correctness < 2.0
        # Should get decent credit (low relative error)
        assert correctness > 0.3

    def test_partial_credit_far_answer(self):
        """Test partial credit for far answers."""
        completion = "<answer>1000</answer>"
        gold_answer = "42"
        correctness, _, _ = compute_correctness_reward(completion, gold_answer, partial_credit=True)

        # Should get minimal partial credit
        assert 0.0 < correctness < 0.1


class TestMathReward:
    """Test combined math reward computation."""

    def test_perfect_response(self):
        """Test perfect completion with correct format and answer."""
        completion = "<reasoning>42 is the answer</reasoning>\n<answer>42</answer>"
        gold_answer = "42"
        reward = compute_math_reward(completion, gold_answer)

        # Should get max reward: 0.45 (format) * 2.0 (mult) + 2.0 (correctness) = 2.9
        assert reward == 2.9, f"Expected 2.9, got {reward}"

    def test_correct_answer_bad_format(self):
        """Test correct answer without proper formatting."""
        completion = "The answer is 42"
        gold_answer = "42"
        reward = compute_math_reward(completion, gold_answer)

        # Gets correctness (2.0) + format floor (0.30 * 2.0 = 0.6) = 2.6
        # The code applies format_score floor of 0.30 when correctness >= 1.5
        assert reward == 2.6, f"Expected 2.6, got {reward}"

    def test_wrong_answer_good_format(self):
        """Test wrong answer with proper formatting."""
        completion = "<reasoning>I think it's 100</reasoning>\n<answer>100</answer>"
        gold_answer = "42"
        reward = compute_math_reward(completion, gold_answer)

        # Should get format reward (0.45 * 0.2 = 0.09) + partial correctness
        # Partial correctness for 100 vs 42 should be low (< 0.1)
        assert reward > 0.09, f"Expected > 0.09, got {reward}"
        assert reward < 2.9, f"Expected < 2.9, got {reward}"

    def test_no_answer(self):
        """Test completion without answer."""
        completion = "I don't know"
        gold_answer = "42"
        reward = compute_math_reward(completion, gold_answer)

        # Should get 0 reward
        assert reward == 0.0, f"Expected 0.0, got {reward}"


class TestBatchProcessing:
    """Test batch processing of rewards."""

    def test_compute_math_reward_batch(self):
        """Test batch reward computation."""
        completions = [
            "<reasoning>test</reasoning>\n<answer>42</answer>",  # Perfect
            "The answer is 43",  # Close, no format
            "<reasoning>wrong</reasoning>\n<answer>0</answer>",  # Wrong but formatted
        ]
        answers = ["42", "42", "42"]

        rewards = compute_math_reward_batch(completions, answers)

        assert len(rewards) == 3
        # First should be highest (perfect format + correct)
        assert rewards[0] == 2.9  # 0.45*2.0 + 2.0 = 2.9
        # Second gets partial credit for close answer (43 vs 42)
        # partial = 0.5 * exp(-5 * 0.024) ≈ 0.444, no format, mult=0.2, total ≈ 0.44
        assert 0.4 < rewards[1] < 0.5  # Partial credit for close answer
        # Third should be low (wrong answer 0 vs 42, has format)
        assert 0.09 < rewards[2] < 1.0

    def test_batch_with_none_answers(self):
        """Test batch processing with None answers."""
        completions = [
            "<answer>42</answer>",
            "<answer>100</answer>",
        ]
        answers = [None, None]

        rewards = compute_math_reward_batch(completions, answers)

        # Should get small format rewards (0.30 * 0.2 = 0.06)
        assert all(abs(r - 0.06) < 0.01 for r in rewards)  # Only <answer> tag, mult=0.2

    def test_empty_batch(self):
        """Test empty batch."""
        rewards = compute_math_reward_batch([], [])
        assert rewards == []


class TestBackwardCompatibility:
    """Test backward-compatible wrapper functions."""

    def test_compute_math_rewards_batch_tensor_output(self):
        """Test that batch function returns tensors."""
        completions = [
            "<reasoning>test</reasoning>\n<answer>42</answer>",
            "<answer>43</answer>",
        ]
        answers = ["42", "42"]

        # Test without device specification
        rewards = compute_math_rewards_batch(completions, answers, return_breakdown=False)

        assert isinstance(rewards, torch.Tensor)
        assert len(rewards) == 2

    def test_compute_math_rewards_batch_with_device(self):
        """Test batch function with device specification."""
        completions = ["<answer>42</answer>"]
        answers = ["42"]

        device = "cpu"
        rewards = compute_math_rewards_batch(completions, answers, device=device, return_breakdown=False)

        assert rewards.device.type == "cpu"

    def test_compute_math_rewards_batch_breakdown(self):
        """Test batch function with breakdown."""
        completions = [
            "<reasoning>test</reasoning>\n<answer>42</answer>",
            "42",
        ]
        answers = ["42", "42"]

        total, format_r, correctness_r = compute_math_rewards_batch(
            completions, answers, return_breakdown=True
        )

        # Check all are tensors
        assert isinstance(total, torch.Tensor)
        assert isinstance(format_r, torch.Tensor)
        assert isinstance(correctness_r, torch.Tensor)

        # Check shapes match
        assert total.shape == format_r.shape == correctness_r.shape

        # Note: format_r returns BASE format scores (not weighted)
        # Total = correctness + format_r * multiplier(correctness)
        # First: total = 2.0 + 0.45*2.0 = 2.9, format_r_base = 0.45, correctness = 2.0
        # Second: total = 2.0 + 0.30*2.0 = 2.6, format_r_base = 0.0 (but floor applied), correctness = 2.0

        # First completion has formatting
        assert format_r[0] > 0
        assert abs(format_r[0].item() - 0.45) < 0.01  # Base format score
        assert abs(total[0].item() - 2.9) < 0.01  # Total with weighted format
        # Second has no formatting tags, but gets floor
        assert format_r[1] == 0  # No format tags
        assert abs(total[1].item() - 2.6) < 0.01  # Total with floor applied (0.30*2.0)

    def test_extract_answer_from_model_output(self):
        """Test backward-compatible answer extraction."""
        output = "<answer>42</answer>"
        answer = extract_answer_from_model_output(output)
        assert answer == "42"

    def test_extract_answer_from_dataset(self):
        """Test backward-compatible dataset answer extraction."""
        example = {"answer": "42"}
        answer = extract_answer_from_dataset(example)
        assert answer == "42"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_large_numbers(self):
        """Test with very large numbers."""
        completion = "<answer>1000000000</answer>"
        gold_answer = "999999999"
        reward = compute_math_reward(completion, gold_answer)

        # Should handle large numbers gracefully
        # Max reward is now 2.9 (0.45*2.0 + 2.0)
        assert 0.0 <= reward <= 2.9

    def test_negative_numbers(self):
        """Test with negative numbers."""
        completion = "<answer>-42</answer>"
        gold_answer = "-42"
        reward = compute_math_reward(completion, gold_answer)

        # 0.30 (format) * 2.0 (mult for correct) + 2.0 (correctness) = 2.6
        assert reward == 2.6

    def test_decimal_precision(self):
        """Test with decimal numbers requiring precision."""
        completion = "<answer>3.14159</answer>"
        gold_answer = "3.14159"
        reward = compute_math_reward(completion, gold_answer)

        assert reward >= 2.0  # Should match exactly

    def test_unicode_and_special_chars(self):
        """Test with unicode and special characters."""
        completion = "<reasoning>π ≈ 3.14</reasoning>\n<answer>3.14</answer>"
        gold_answer = "3.14"
        reward = compute_math_reward(completion, gold_answer)

        # Should handle unicode gracefully
        assert reward >= 2.0

    def test_empty_completion(self):
        """Test with empty completion."""
        completion = ""
        gold_answer = "42"
        reward = compute_math_reward(completion, gold_answer)

        assert reward == 0.0

    def test_very_long_completion(self):
        """Test with very long completion."""
        completion = "<reasoning>" + "word " * 1000 + "</reasoning>\n<answer>42</answer>"
        gold_answer = "42"
        reward = compute_math_reward(completion, gold_answer)

        # Should still compute correctly regardless of length
        # 0.45 (format) * 2.0 (mult for correct) + 2.0 (correctness) = 2.9
        assert reward == 2.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
