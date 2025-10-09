"""
Test anti-gaming reward function to verify it penalizes reward hacking attempts.
"""

import torch
from simple_rl.rewards import (
    compute_anti_gaming_rewards_batch,
    anti_gaming_format_reward,
    enhanced_correctness_reward,
)


def test_gaming_attempts():
    """Test that gaming attempts get low rewards."""
    print("\n" + "=" * 80)
    print("TEST 1: Gaming Attempts (Should Get LOW Rewards)")
    print("=" * 80)

    test_cases = [
        {
            "name": "Just period in answer",
            "completion": '<reasoning>Short.</reasoning><answer>.</answer>',
            "answer": "42",
            "expected_range": (0.0, 0.1),
        },
        {
            "name": "Empty answer",
            "completion": '<reasoning>Short.</reasoning><answer></answer>',
            "answer": "42",
            "expected_range": (0.0, 0.1),
        },
        {
            "name": "Repetitive closing tags",
            "completion": '<reasoning>x</reasoning><answer>.</answer></answer></answer>',
            "answer": "42",
            "expected_range": (0.0, 0.05),
        },
        {
            "name": "No numbers in answer",
            "completion": '<reasoning>Calculating</reasoning><answer>Hello</answer>',
            "answer": "42",
            "expected_range": (0.0, 0.15),
        },
    ]

    for case in test_cases:
        total, fmt, correct = compute_anti_gaming_rewards_batch(
            [case["completion"]], [case["answer"]], return_breakdown=True
        )
        total_val = total.item()
        fmt_val = fmt.item()
        correct_val = correct.item()

        expected_min, expected_max = case["expected_range"]
        status = "✓" if expected_min <= total_val <= expected_max else "✗"

        print(f"\n{status} {case['name']}:")
        print(f"  Completion: {case['completion'][:60]}...")
        print(f"  Total: {total_val:.3f} (expected: {expected_min:.2f}-{expected_max:.2f})")
        print(f"  Format: {fmt_val:.3f}, Correctness: {correct_val:.3f}")


def test_correct_answers():
    """Test that correct answers get high rewards."""
    print("\n" + "=" * 80)
    print("TEST 2: Correct Answers (Should Get HIGH Rewards)")
    print("=" * 80)

    test_cases = [
        {
            "name": "Exact match",
            "completion": '<reasoning>The answer is 42.</reasoning><answer>42</answer>',
            "answer": "42",
            "expected_range": (2.0, 2.5),
        },
        {
            "name": "Numeric match",
            "completion": '<reasoning>Calculating 2+2.</reasoning><answer>4</answer>',
            "answer": "4",
            "expected_range": (1.8, 2.2),
        },
        {
            "name": "Good reasoning",
            "completion": '<reasoning>Step 1: Add the numbers. Step 2: Get result.</reasoning><answer>100</answer>',
            "answer": "100",
            "expected_range": (2.0, 2.5),
        },
    ]

    for case in test_cases:
        total, fmt, correct = compute_anti_gaming_rewards_batch(
            [case["completion"]], [case["answer"]], return_breakdown=True
        )
        total_val = total.item()
        fmt_val = fmt.item()
        correct_val = correct.item()

        expected_min, expected_max = case["expected_range"]
        status = "✓" if expected_min <= total_val <= expected_max else "✗"

        print(f"\n{status} {case['name']}:")
        print(f"  Completion: {case['completion'][:60]}...")
        print(f"  Total: {total_val:.3f} (expected: {expected_min:.2f}-{expected_max:.2f})")
        print(f"  Format: {fmt_val:.3f}, Correctness: {correct_val:.3f}")


def test_partial_credit():
    """Test that wrong but numeric answers get partial credit."""
    print("\n" + "=" * 80)
    print("TEST 3: Partial Credit (Should Get MEDIUM Rewards)")
    print("=" * 80)

    test_cases = [
        {
            "name": "Wrong but numeric",
            "completion": '<reasoning>Let me calculate... I get 50.</reasoning><answer>50</answer>',
            "answer": "42",
            "expected_range": (0.8, 1.1),
        },
        {
            "name": "Wrong but trying",
            "completion": '<reasoning>Adding numbers gives 100.</reasoning><answer>100</answer>',
            "answer": "42",
            "expected_range": (0.8, 1.1),
        },
    ]

    for case in test_cases:
        total, fmt, correct = compute_anti_gaming_rewards_batch(
            [case["completion"]], [case["answer"]], return_breakdown=True
        )
        total_val = total.item()
        fmt_val = fmt.item()
        correct_val = correct.item()

        expected_min, expected_max = case["expected_range"]
        status = "✓" if expected_min <= total_val <= expected_max else "✗"

        print(f"\n{status} {case['name']}:")
        print(f"  Completion: {case['completion'][:60]}...")
        print(f"  Total: {total_val:.3f} (expected: {expected_min:.2f}-{expected_max:.2f})")
        print(f"  Format: {fmt_val:.3f}, Correctness: {correct_val:.3f}")


def test_format_penalties():
    """Test specific format penalties."""
    print("\n" + "=" * 80)
    print("TEST 4: Format Penalty Details")
    print("=" * 80)

    # Test 1: Trivial answer penalty
    completion1 = '<reasoning>x</reasoning><answer>.</answer>'
    fmt1 = anti_gaming_format_reward(completion1)
    print(f"\n1. Trivial answer ('.') penalty:")
    print(f"   Completion: {completion1}")
    print(f"   Format: {fmt1:.3f} (should be ~0.04, which is 0.4 * 0.1)")

    # Test 2: No numbers penalty
    completion2 = '<reasoning>x</reasoning><answer>Hello</answer>'
    fmt2 = anti_gaming_format_reward(completion2)
    print(f"\n2. No numbers in answer penalty:")
    print(f"   Completion: {completion2}")
    print(f"   Format: {fmt2:.3f} (should be ~0.12, which is 0.4 * 0.3)")

    # Test 3: Repetitive tags penalty
    completion3 = '<reasoning>x</reasoning><answer>y</answer></answer></answer>'
    fmt3 = anti_gaming_format_reward(completion3)
    print(f"\n3. Repetitive closing tags penalty:")
    print(f"   Completion: {completion3}")
    print(f"   Format: {fmt3:.3f} (should be ~0.1, which is 0.4 * 0.5^2)")

    # Test 4: Good format (baseline)
    completion4 = '<reasoning>Proper reasoning here</reasoning><answer>42</answer>'
    fmt4 = anti_gaming_format_reward(completion4)
    print(f"\n4. Good format (no penalties):")
    print(f"   Completion: {completion4}")
    print(f"   Format: {fmt4:.3f} (should be 0.4)")


def test_correctness_partial_credit():
    """Test correctness with partial credit details."""
    print("\n" + "=" * 80)
    print("TEST 5: Correctness Partial Credit Details")
    print("=" * 80)

    # Test 1: Exact match
    completion1 = '<reasoning>x</reasoning><answer>42</answer>'
    correct1 = enhanced_correctness_reward(completion1, "42")
    print(f"\n1. Exact match:")
    print(f"   Completion: {completion1}")
    print(f"   Correctness: {correct1:.3f} (should be 2.0)")

    # Test 2: Numeric match
    completion2 = '<reasoning>x</reasoning><answer>42.0</answer>'
    correct2 = enhanced_correctness_reward(completion2, "42")
    print(f"\n2. Numeric match:")
    print(f"   Completion: {completion2}")
    print(f"   Correctness: {correct2:.3f} (should be 1.5)")

    # Test 3: Wrong but numeric (partial credit)
    completion3 = '<reasoning>x</reasoning><answer>50</answer>'
    correct3 = enhanced_correctness_reward(completion3, "42")
    print(f"\n3. Wrong but numeric (partial credit):")
    print(f"   Completion: {completion3}")
    print(f"   Correctness: {correct3:.3f} (should be 0.5)")

    # Test 4: No number in answer
    completion4 = '<reasoning>x</reasoning><answer>Hello</answer>'
    correct4 = enhanced_correctness_reward(completion4, "42")
    print(f"\n4. No number (small format credit):")
    print(f"   Completion: {completion4}")
    print(f"   Correctness: {correct4:.3f} (should be 0.2)")

    # Test 5: Trivial answer
    completion5 = '<reasoning>x</reasoning><answer>.</answer>'
    correct5 = enhanced_correctness_reward(completion5, "42")
    print(f"\n5. Trivial answer:")
    print(f"   Completion: {completion5}")
    print(f"   Correctness: {correct5:.3f} (should be 0.0)")


def test_reward_comparison():
    """Compare old vs new reward for typical gaming scenarios."""
    print("\n" + "=" * 80)
    print("TEST 6: Old vs New Reward Comparison")
    print("=" * 80)

    from simple_rl.rewards import compute_math_rewards_batch_classic

    test_cases = [
        {
            "name": "Gaming: Just period",
            "completion": '<reasoning>x</reasoning><answer>.</answer>',
            "answer": "42",
        },
        {
            "name": "Correct answer",
            "completion": '<reasoning>The answer is 42</reasoning><answer>42</answer>',
            "answer": "42",
        },
        {
            "name": "Wrong but trying",
            "completion": '<reasoning>I calculate 50</reasoning><answer>50</answer>',
            "answer": "42",
        },
    ]

    for case in test_cases:
        # Old reward (classic)
        old_total, old_fmt, old_correct = compute_math_rewards_batch_classic(
            [case["completion"]], [case["answer"]], return_breakdown=True
        )

        # New reward (anti-gaming)
        new_total, new_fmt, new_correct = compute_anti_gaming_rewards_batch(
            [case["completion"]], [case["answer"]], return_breakdown=True
        )

        print(f"\n{case['name']}:")
        print(f"  OLD: Total={old_total.item():.3f}, Fmt={old_fmt.item():.3f}, Correct={old_correct.item():.3f}")
        print(f"  NEW: Total={new_total.item():.3f}, Fmt={new_fmt.item():.3f}, Correct={new_correct.item():.3f}")
        print(f"  Difference: {new_total.item() - old_total.item():+.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ANTI-GAMING REWARD FUNCTION TESTS")
    print("=" * 80)

    test_gaming_attempts()
    test_correct_answers()
    test_partial_credit()
    test_format_penalties()
    test_correctness_partial_credit()
    test_reward_comparison()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nIf all tests show ✓, the reward function is working correctly!")
    print("Key takeaways:")
    print("  - Gaming attempts get < 0.1 reward (down from 0.8)")
    print("  - Correct answers get > 2.0 reward")
    print("  - Wrong but numeric attempts get ~0.9 reward (partial credit)")
    print("  - Repetitive tags get exponentially penalized")
