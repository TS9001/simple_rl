"""
Exact replication of notebook training setup.

This test uses:
- Same model: Qwen/Qwen2.5-0.5B-Instruct
- Same data splits: GSM8K train[500:] for GRPO
- Same hyperparameters from the training script
- SFT checkpoint loading

This validates that your exact notebook/script setup works correctly.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_rl.algorithms import GRPO
from simple_rl.evaluation.gsm8k import (
    load_gsm8k_dataset,
    prepare_gsm8k_prompts,
)
from simple_rl.rewards import compute_math_rewards_batch


@pytest.mark.slow
class TestNotebookReplication:
    """Test that replicates the exact notebook setup."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @pytest.fixture
    def system_prompt(self):
        """System prompt from notebook."""
        return """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

    @pytest.fixture
    def notebook_config(self, device):
        """Exact config from the training script."""
        return {
            "algorithm": {
                "name": "grpo",
                "group_size": 8,
                "kl_coef": 0.1,
                "clip_epsilon": 0.2,
                "normalize_rewards": True,
                "store_completions": True,
            },
            "training": {
                "batch_size": 16,
                "rollout_batch_size": 4,
                "gradient_clip": 1.0,
                "max_new_tokens": 400,
                "temperature": 0.6,
                "num_episodes": 10,  # Reduced for testing (was 500)
                "minibatch_size": 32,
                "update_epochs": 1,
                "top_p": 0.9,
                "entropy_coef": 0.002,
                "policy_loss_type": "token",  # Options: "sequence" or "token"
                # Clipping parameters
                "kl_clamp_min": -2.0,
                "kl_clamp_max": 2.0,
                "kl_reduction": "mean",  # Options: "mean" (average per token) or "sum" (total)
                "policy_log_ratio_clamp_min": -2.0,
                "policy_log_ratio_clamp_max": 2.0,
                "advantage_clip_min": -3.0,
                "advantage_clip_max": 3.0,
                "stop_sequences": ["</answer>"],  # Multi-token stopping for early termination
            },
            "model": {
                "max_length": 1024,
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "model_type": "fp32",
                "device": str(device),
                "compile": {
                    "enabled": False,
                    "backend": "aot_eager"
                }
            },
            "device_optimizations": {
                "clear_cache_on_mps": True,
            },
            "logging": {
                "log_interval": 1,
                "save_interval": 1000,
                "show_trajectory_progress": False,
            },
            "validation": {
                "enabled": False,  # Disabled for faster testing
            },
            "wandb": {
                "enabled": False,
            },
            "optimizer": {
                "type": "adamw",
                "lr": 3e-6,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "fused": False
            },
            "optimization": {
                "mixed_precision": {
                    "enabled": False,
                    "dtype": "fp32",
                    "mode": "fp32"
                }
            },
            "timing": {
                "enabled": False,
            },
            "debug": {
                "enabled": False,
            }
        }

    def test_notebook_training_workflow(self, notebook_config, system_prompt, device):
        """
        Test the exact workflow from the notebook:
        1. Load GSM8K data with same splits
        2. Load Qwen model
        3. Train for 10 episodes (short version)
        4. Validate metrics
        """
        print("\n" + "="*70)
        print("NOTEBOOK REPLICATION TEST")
        print("="*70)
        print(f"Model: {notebook_config['model']['model_name']}")
        print(f"Device: {device}")
        print(f"Episodes: {notebook_config['training']['num_episodes']}")
        print("="*70 + "\n")

        # 1. Load GSM8K data with exact same splits as notebook
        print("Loading GSM8K data (same splits as notebook)...")
        dataset_grpo_train, _, dataset_test = load_gsm8k_dataset(
            train_split="train[500:520]",  # Just 20 samples for testing (was train[500:])
            val_split="test[200:1200]",
            test_split="test[:200]"
        )

        grpo_train_prompts, grpo_train_answers = prepare_gsm8k_prompts(
            dataset_grpo_train, system_prompt
        )

        print(f"✓ Loaded {len(grpo_train_prompts)} training examples")

        # 2. Load Qwen model
        print(f"\nLoading {notebook_config['model']['model_name']}...")
        model = AutoModelForCausalLM.from_pretrained(
            notebook_config['model']['model_name'],
            dtype=torch.float32,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            notebook_config['model']['model_name'],
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✓ Model and tokenizer loaded")

        # 3. Initialize GRPO
        print("\nInitializing GRPO...")
        grpo = GRPO(
            config=notebook_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        print("✓ GRPO initialized")

        # 4. Train
        print("\nStarting training (10 episodes)...")
        grpo_train_data = {
            "prompts": grpo_train_prompts,
            "answers": grpo_train_answers
        }

        results = grpo.train(
            train_data=grpo_train_data,
            val_data=None,
            num_episodes=10
        )

        metrics = results["training_metrics"]

        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)

        # 5. Validate metrics
        print("\nValidating metrics...")

        # Check all expected metrics are present
        expected_metrics = [
            "episode", "total_loss", "pg_loss", "kl_divergence",
            "reward_mean", "reward_std", "format_reward_mean",
            "correctness_reward_mean", "grad_norm"
        ]

        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert len(metrics[metric]) == 10, f"{metric} has wrong length"

        print("✓ All metrics present")

        # Check for NaN/Inf
        for key in ["total_loss", "kl_divergence", "reward_mean", "grad_norm"]:
            values = metrics[key]
            assert not any(np.isnan(v) for v in values), f"NaN in {key}"
            assert not any(np.isinf(v) for v in values), f"Inf in {key}"

        print("✓ No NaN or Inf values")

        # Check KL divergence is reasonable
        max_kl = max(metrics["kl_divergence"])
        mean_kl = np.mean(metrics["kl_divergence"])
        assert max_kl < 10.0, f"KL divergence exploded: {max_kl}"
        print(f"✓ KL divergence bounded (max: {max_kl:.3f}, mean: {mean_kl:.3f})")

        # Check gradient norms
        mean_grad = np.mean(metrics["grad_norm"])
        max_grad = np.max(metrics["grad_norm"])
        assert mean_grad > 0.001, f"Gradients vanishing: {mean_grad}"
        assert max_grad < 100.0, f"Gradients exploding: {max_grad}"
        print(f"✓ Gradient norms reasonable (mean: {mean_grad:.3f}, max: {max_grad:.3f})")

        # Check reward ranges
        mean_reward = np.mean(metrics["reward_mean"])
        assert 0.0 <= mean_reward <= 3.0, f"Reward out of expected range: {mean_reward}"

        mean_format = np.mean(metrics["format_reward_mean"])
        assert 0.0 <= mean_format <= 0.3, f"Format reward out of range: {mean_format}"

        mean_correctness = np.mean(metrics["correctness_reward_mean"])
        assert 0.0 <= mean_correctness <= 2.0, f"Correctness reward out of range: {mean_correctness}"

        print(f"✓ Reward ranges valid (mean: {mean_reward:.3f})")

        # Print summary
        print("\n" + "="*70)
        print("NOTEBOOK REPLICATION TEST RESULTS")
        print("="*70)
        print(f"Episodes completed:     {len(metrics['episode'])}")
        print(f"Final loss:             {metrics['total_loss'][-1]:.4f}")
        print(f"Final reward:           {metrics['reward_mean'][-1]:.3f}")
        print(f"Final format reward:    {metrics['format_reward_mean'][-1]:.3f}")
        print(f"Final correctness:      {metrics['correctness_reward_mean'][-1]:.3f}")
        print(f"Mean KL divergence:     {mean_kl:.4f}")
        print(f"Mean gradient norm:     {mean_grad:.3f}")
        print(f"Total time:             {results['total_time']:.2f}s")
        print(f"Total tokens:           {metrics['total_tokens'][-1]:,}")
        print("="*70)
        print("✓ NOTEBOOK SETUP VALIDATED - All checks passed!")
        print("="*70 + "\n")

    @pytest.mark.skipif(
        not Path("checkpoints/pipeline_stages/01_after_sft/sft_complete.pt").exists(),
        reason="SFT checkpoint not found - run SFT training first or skip this test"
    )
    def test_with_sft_checkpoint(self, notebook_config, system_prompt, device):
        """
        Test loading and using an SFT checkpoint (exact notebook workflow).

        This test will only run if the SFT checkpoint exists.
        """
        print("\n" + "="*70)
        print("NOTEBOOK WORKFLOW WITH SFT CHECKPOINT")
        print("="*70)

        sft_checkpoint_path = Path("checkpoints/pipeline_stages/01_after_sft/sft_complete.pt")

        print(f"Loading SFT checkpoint from: {sft_checkpoint_path}")

        # Load checkpoint
        sft_checkpoint = torch.load(sft_checkpoint_path, map_location=device)
        model_name = sft_checkpoint["config"]["model"]["model_name"]

        print(f"Model from checkpoint: {model_name}")

        # Load fresh model and apply SFT weights
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.load_state_dict(sft_checkpoint["model_state_dict"])
        print("✓ SFT weights loaded")

        # Load data
        dataset_grpo_train, _, _ = load_gsm8k_dataset(
            train_split="train[500:516]",  # Just 16 samples
            val_split="test[200:1200]",
            test_split="test[:200]"
        )

        grpo_train_prompts, grpo_train_answers = prepare_gsm8k_prompts(
            dataset_grpo_train, system_prompt
        )

        # Initialize GRPO with SFT model
        grpo = GRPO(
            config=notebook_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Short training
        notebook_config["training"]["num_episodes"] = 5
        grpo_train_data = {
            "prompts": grpo_train_prompts,
            "answers": grpo_train_answers
        }

        results = grpo.train(
            train_data=grpo_train_data,
            val_data=None,
            num_episodes=5
        )

        # Validate
        metrics = results["training_metrics"]

        # With SFT checkpoint, we expect better initial format rewards
        initial_format = metrics["format_reward_mean"][0]
        print(f"\nInitial format reward with SFT: {initial_format:.3f}")

        # SFT models should produce some format compliance
        if initial_format > 0.05:
            print("✓ SFT checkpoint shows format learning!")
        else:
            print("⚠ Low initial format reward - SFT may not have learned tags well")

        print("✓ SFT checkpoint test completed")


def test_notebook_reward_function():
    """
    Test the exact reward function used in the notebook.

    Validates:
    - Format reward computation (0.1 for <reasoning>, 0.2 for <answer>, 0.3 for both)
    - Correctness reward (2.0 for exact match, partial credit for close)
    - Total reward ranges (0.0 to 2.3)
    """
    print("\n" + "="*70)
    print("REWARD FUNCTION TEST (notebook setup)")
    print("="*70)

    from simple_rl.rewards import (
        compute_format_reward,
        compute_correctness_reward,
        compute_math_reward,
        compute_math_reward_batch,
        compute_math_rewards_batch,  # Backward compatible version
    )

    # Test cases matching notebook scenarios
    test_cases = [
        {
            "name": "Perfect response",
            "completion": "<reasoning>2+2=4 because 2 plus 2 equals 4</reasoning>\n<answer>4</answer>",
            "gold_answer": "4",
            "expected_total": 2.3,  # 0.3 format + 2.0 correctness
            "expected_format": 0.3,
            "expected_correctness": 2.0,
        },
        {
            "name": "Correct answer, no format",
            "completion": "The answer is 4",
            "gold_answer": "4",
            "expected_total": 2.0,  # 0.0 format + 2.0 correctness
            "expected_format": 0.0,
            "expected_correctness": 2.0,
        },
        {
            "name": "Wrong answer, good format",
            "completion": "<reasoning>I think 2+2=5</reasoning>\n<answer>5</answer>",
            "gold_answer": "4",
            "expected_format": 0.3,
            # Correctness will be partial credit (close to 0.3-0.4 due to relative error)
        },
        {
            "name": "Only reasoning tag",
            "completion": "<reasoning>Let me think about this</reasoning> The answer is 4",
            "gold_answer": "4",
            "expected_format": 0.1,
            "expected_correctness": 2.0,
            "expected_total": 2.1,
        },
        {
            "name": "Only answer tag",
            "completion": "<answer>4</answer>",
            "gold_answer": "4",
            "expected_format": 0.2,
            "expected_correctness": 2.0,
            "expected_total": 2.2,
        },
        {
            "name": "No tags, no answer",
            "completion": "I don't know",
            "gold_answer": "4",
            "expected_total": 0.0,
            "expected_format": 0.0,
            "expected_correctness": 0.0,
        },
    ]

    print("\nTesting individual reward components:")
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Completion: {test['completion'][:60]}...")

        # Test format reward
        format_r = compute_format_reward(test['completion'])
        print(f"   Format reward: {format_r:.3f}", end="")
        if "expected_format" in test:
            assert abs(format_r - test['expected_format']) < 0.01, \
                f"Expected {test['expected_format']}, got {format_r}"
            print(" ✓")
        else:
            print()

        # Test correctness reward
        correctness_r, model_num, gold_num = compute_correctness_reward(
            test['completion'], test['gold_answer']
        )
        print(f"   Correctness reward: {correctness_r:.3f}", end="")
        if "expected_correctness" in test:
            assert abs(correctness_r - test['expected_correctness']) < 0.01, \
                f"Expected {test['expected_correctness']}, got {correctness_r}"
            print(" ✓")
        else:
            print(f" (model: {model_num}, gold: {gold_num})")

        # Test total reward
        total_r = compute_math_reward(test['completion'], test['gold_answer'])
        print(f"   Total reward: {total_r:.3f}", end="")
        if "expected_total" in test:
            assert abs(total_r - test['expected_total']) < 0.01, \
                f"Expected {test['expected_total']}, got {total_r}"
            print(" ✓")
        else:
            print()

    print("\n" + "-"*70)
    print("Testing batch reward function:")

    # Test batch processing
    completions = [tc["completion"] for tc in test_cases]
    answers = [tc["gold_answer"] for tc in test_cases]

    # New API
    batch_rewards = compute_math_reward_batch(completions, answers)
    print(f"  compute_math_reward_batch: {len(batch_rewards)} rewards")
    assert len(batch_rewards) == len(completions)

    # Backward compatible API (used by GRPO)
    import torch
    device = "cpu"

    # Without breakdown
    batch_rewards_tensor = compute_math_rewards_batch(completions, answers, device=device)
    assert isinstance(batch_rewards_tensor, torch.Tensor)
    assert batch_rewards_tensor.device.type == device
    assert len(batch_rewards_tensor) == len(completions)
    print(f"  compute_math_rewards_batch (no breakdown): {batch_rewards_tensor.shape} ✓")

    # With breakdown
    total_r, format_r, correct_r = compute_math_rewards_batch(
        completions, answers, device=device, return_breakdown=True
    )
    assert isinstance(total_r, torch.Tensor)
    assert isinstance(format_r, torch.Tensor)
    assert isinstance(correct_r, torch.Tensor)
    assert torch.allclose(total_r, format_r + correct_r, atol=1e-5)
    print(f"  compute_math_rewards_batch (with breakdown): 3 tensors ✓")

    # Verify breakdown matches
    print(f"\n  Sample breakdown (test case 1 - perfect response):")
    print(f"    Total:       {total_r[0]:.3f}")
    print(f"    Format:      {format_r[0]:.3f}")
    print(f"    Correctness: {correct_r[0]:.3f}")
    print(f"    Sum:         {(format_r[0] + correct_r[0]):.3f}")
    assert abs(total_r[0].item() - 2.3) < 0.01, "Perfect response should get 2.3"
    assert abs(format_r[0].item() - 0.3) < 0.01, "Perfect format should get 0.3"
    assert abs(correct_r[0].item() - 2.0) < 0.01, "Correct answer should get 2.0"

    print("\n" + "="*70)
    print("REWARD FUNCTION TEST PASSED")
    print("="*70)
    print("✓ Format rewards: 0.1 (reasoning), 0.2 (answer), 0.3 (both)")
    print("✓ Correctness rewards: 2.0 (exact), partial credit (close)")
    print("✓ Total reward range: 0.0 - 2.3")
    print("✓ Batch processing works correctly")
    print("✓ Backward compatible API works")
    print("="*70 + "\n")


def test_notebook_data_loading():
    """
    Quick test to validate data loading matches notebook setup.
    """
    print("\n" + "="*70)
    print("DATA LOADING TEST (matching notebook)")
    print("="*70)

    system_prompt = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

    # Load with notebook splits
    print("Loading with notebook splits...")
    dataset_sft_train, dataset_val, dataset_test = load_gsm8k_dataset(
        train_split="train[:500]",
        val_split="test[200:1200]",
        test_split="test[:200]"
    )

    dataset_grpo_train, _, _ = load_gsm8k_dataset(
        train_split="train[500:]",
        val_split="test[200:1200]",
        test_split="test[:200]"
    )

    # Prepare prompts
    grpo_train_prompts, grpo_train_answers = prepare_gsm8k_prompts(
        dataset_grpo_train, system_prompt
    )

    print(f"\nDataset splits (matching notebook):")
    print(f"  SFT train:       {len(dataset_sft_train)} examples")
    print(f"  GRPO train:      {len(grpo_train_prompts)} examples")
    print(f"  Validation:      {len(dataset_val)} examples")
    print(f"  Test:            {len(dataset_test)} examples")

    # Validate no overlap
    sft_questions = {ex["question"] for ex in dataset_sft_train}
    grpo_questions = {ex["question"] for ex in dataset_grpo_train}
    overlap = sft_questions & grpo_questions

    assert len(overlap) == 0, f"Found {len(overlap)} overlapping questions between SFT and GRPO!"
    print("\n✓ No overlap between SFT and GRPO data splits")

    # Check prompt format
    sample_prompt = grpo_train_prompts[0]
    assert system_prompt.strip() in sample_prompt, "System prompt not in generated prompt"
    print("✓ Prompts include system prompt")

    # Check answers
    assert len(grpo_train_answers) == len(grpo_train_prompts), "Mismatched prompts/answers"
    assert all(isinstance(a, str) for a in grpo_train_answers), "Answers should be strings"
    print("✓ Answers properly formatted")

    print("\n" + "="*70)
    print("DATA LOADING TEST PASSED")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run notebook replication tests
    pytest.main([__file__, "-v", "-s"])
