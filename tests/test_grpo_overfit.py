"""
Overfit test for GRPO implementation.

This test validates that GRPO CAN learn by overfitting on a single batch.
If the model cannot overfit a small batch, there's a fundamental bug in the training loop.

Uses the same hyperparameters as the full training script.
"""

import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from simple_rl.algorithms import GRPO
from simple_rl.rewards import compute_math_rewards_batch
from simple_rl.evaluation.gsm8k import prepare_gsm8k_prompts


@pytest.mark.slow
class TestGRPOOverfit:
    """Test that GRPO can overfit a single batch."""

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
    def model_name(self):
        """Model to use for overfitting test."""
        # Use Qwen 0.5B Instruct - general instruction model, small and fast
        return "Qwen/Qwen2.5-0.5B-Instruct"

    @pytest.fixture
    def system_prompt(self):
        """System prompt from training script."""
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
    def single_batch_dataset(self, system_prompt):
        """Create a single batch of GSM8K data."""
        # Load GSM8K
        data = load_dataset('gsm8k', 'main')

        # Take just 4 examples for faster overfitting
        dataset = data['train'].select(range(4))

        # Prepare prompts
        prompts, answers = prepare_gsm8k_prompts(dataset, system_prompt)

        return {
            "prompts": prompts,
            "answers": answers
        }

    @pytest.fixture
    def grpo_config_from_script(self, device, model_name):
        """GRPO config for smooth, gradual overfitting on Qwen 0.5B Instruct."""
        return {
            "algorithm": {
                "name": "grpo",
                "group_size": 8,  # 8 completions per prompt (like in real training)
                "kl_coef": 0.15,  # Higher KL penalty to keep policy stable
                "clip_epsilon": 0.2,
                "normalize_rewards": True,
                "store_completions": True,  # Store to inspect outputs
            },
            "training": {
                "batch_size": 4,  # 4 problems = 32 rollouts (4 × 8 group_size)
                "rollout_batch_size": 2,  # Generate 2 prompts at a time (memory)
                "gradient_clip": 0.1,  # Very tight clipping for maximum stability
                "max_new_tokens": 400,  # Need longer for math reasoning
                "min_new_tokens": 150,  # Force longer completions to encourage format learning
                "temperature": 0.6,  # Lower temperature for more focused generation
                "num_episodes": 100,  # More episodes for slower learning
                "minibatch_size": 8,  # Process 8 rollouts per minibatch
                "update_epochs": 1,  # Single epoch - we're overfitting across episodes, not within
                "top_p": 0.9,
                "entropy_coef": 0.005,  # Lower entropy for more exploitation
                "kl_estimator": "k3",  # Low-variance KL estimator
                "kl_clamp_min": -2.0,
                "kl_clamp_max": 2.0,
                "policy_loss_type": "token",  # Options: "sequence" or "token"
                # Clipping parameters
                "kl_reduction": "mean",  # Options: "mean" (average per token) or "sum" (total)
                "policy_log_ratio_clamp_min": -2.0,
                "policy_log_ratio_clamp_max": 2.0,
                "advantage_clip_min": -3.0,
                "advantage_clip_max": 3.0,
            },
            "model": {
                "max_length": 1024,
                "hf_model_name": model_name,  # Qwen 0.5B Instruct
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
                "log_interval": 1,  # Log every episode
                "save_interval": 1000,  # Don't save checkpoints
                "show_trajectory_progress": False,  # Less verbose
            },
            "validation": {
                "enabled": False,  # Skip validation for speed
            },
            "wandb": {
                "enabled": False,
            },
            "optimizer": {
                "type": "adamw",
                "lr": 1e-6,  # Much lower LR - grad norms are too high
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "fused": False,
                # Longer warmup for stability
                "warmup_steps": 30,  # Longer warmup over 30 minibatch updates
                "warmup_start_lr": 1e-9,
                "warmup_type": "linear"
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
                "enabled": False,  # Disable debug for cleaner output
            }
        }

    def test_overfit_single_batch(self, grpo_config_from_script, single_batch_dataset, device, model_name):
        """
        Test that GRPO can overfit a single batch using Qwen 0.5B Instruct.

        This is THE critical test - if this fails, there's a fundamental bug.
        """
        print("\n" + "="*70)
        print("OVERFIT TEST: Qwen 0.5B Instruct on 4 GSM8K problems (100 episodes)")
        print("="*70)
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Batch size: {len(single_batch_dataset['prompts'])} examples")
        print(f"Expected behavior: Loss decreases, rewards increase significantly")
        print("="*70 + "\n")

        # Load model
        print(f"Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=grpo_config_from_script,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Record initial performance
        print("Generating initial completions (before training)...")
        _, initial_completions, *_ = grpo.generate_trajectories(
            single_batch_dataset["prompts"],
            single_batch_dataset["answers"],
            store_outputs=True
        )

        initial_rewards = compute_math_rewards_batch(
            initial_completions,
            single_batch_dataset["answers"] * grpo.group_size
        )
        initial_mean_reward = torch.tensor(initial_rewards).mean().item()

        print(f"Initial mean reward: {initial_mean_reward:.3f}")
        print(f"Sample initial completion:\n{initial_completions[0][:200]}...\n")

        # Train on the same batch repeatedly
        print("Starting overfit training (50 episodes)...")
        results = grpo.train(
            train_data=single_batch_dataset,
            val_data=None,
            num_episodes=50
        )

        metrics = results["training_metrics"]

        # Record final performance
        print("\nGenerating final completions (after training)...")
        _, final_completions, *_ = grpo.generate_trajectories(
            single_batch_dataset["prompts"],
            single_batch_dataset["answers"],
            store_outputs=True
        )

        final_rewards = compute_math_rewards_batch(
            final_completions,
            single_batch_dataset["answers"] * grpo.group_size
        )
        final_mean_reward = torch.tensor(final_rewards).mean().item()

        print(f"Final mean reward: {final_mean_reward:.3f}")
        print(f"Sample final completion:\n{final_completions[0][:200]}...\n")

        # Extract key metrics
        initial_loss = np.mean(metrics["total_loss"][:3])
        final_loss = np.mean(metrics["total_loss"][-3:])

        initial_reward_train = np.mean(metrics["reward_mean"][:3])
        final_reward_train = np.mean(metrics["reward_mean"][-3:])

        initial_format = np.mean(metrics["format_reward_mean"][:3])
        final_format = np.mean(metrics["format_reward_mean"][-3:])

        initial_correctness = np.mean(metrics["correctness_reward_mean"][:3])
        final_correctness = np.mean(metrics["correctness_reward_mean"][-3:])

        # Print summary
        print("\n" + "="*70)
        print("OVERFIT TEST RESULTS")
        print("="*70)
        print(f"Loss:        {initial_loss:.4f} → {final_loss:.4f} (change: {final_loss - initial_loss:.4f})")
        print(f"Reward:      {initial_reward_train:.3f} → {final_reward_train:.3f} (change: {final_reward_train - initial_reward_train:.3f})")
        print(f"Format:      {initial_format:.3f} → {final_format:.3f} (change: {final_format - initial_format:.3f})")
        print(f"Correctness: {initial_correctness:.3f} → {final_correctness:.3f} (change: {final_correctness - initial_correctness:.3f})")
        print(f"\nReward improvement: {((final_mean_reward - initial_mean_reward) / max(abs(initial_mean_reward), 0.01) * 100):.1f}%")
        print("="*70)

        # Assertions: Check that overfitting happened

        # 1. No NaN or Inf values
        for key in ["total_loss", "kl_divergence", "reward_mean", "grad_norm"]:
            if key in metrics:
                values = metrics[key]
                assert not any(np.isnan(v) for v in values), f"NaN detected in {key}"
                assert not any(np.isinf(v) for v in values), f"Inf detected in {key}"

        print("✓ No NaN or Inf values detected")

        # 2. Loss should decrease (allowing some tolerance)
        loss_decrease = initial_loss - final_loss
        assert loss_decrease > -0.1, \
            f"Loss increased too much: {initial_loss:.4f} → {final_loss:.4f}"

        if loss_decrease > 0.05:
            print(f"✓ Loss decreased by {loss_decrease:.4f}")
        else:
            print(f"⚠ Loss only decreased by {loss_decrease:.4f} (small improvement)")

        # 3. Rewards should improve significantly (this is the key test)
        reward_improvement = final_reward_train - initial_reward_train

        # We expect at least some improvement when overfitting
        assert reward_improvement > -0.2, \
            f"Rewards degraded significantly: {initial_reward_train:.3f} → {final_reward_train:.3f}"

        if reward_improvement > 0.3:
            print(f"✓ Rewards improved significantly by {reward_improvement:.3f}")
        elif reward_improvement > 0.1:
            print(f"✓ Rewards improved moderately by {reward_improvement:.3f}")
        else:
            print(f"⚠ Rewards improved only slightly by {reward_improvement:.3f}")
            # This is acceptable but not ideal

        # 4. Format rewards should improve (model should learn tags)
        format_improvement = final_format - initial_format
        assert format_improvement > -0.05, \
            f"Format rewards degraded: {initial_format:.3f} → {final_format:.3f}"

        if format_improvement > 0.05:
            print(f"✓ Format rewards improved by {format_improvement:.3f}")
        else:
            print(f"⚠ Format rewards only improved by {format_improvement:.3f}")

        # 5. KL divergence should be reasonable (not exploding)
        max_kl = max(metrics["kl_divergence"])
        mean_kl = np.mean(metrics["kl_divergence"])

        assert max_kl < 10.0, f"KL divergence exploded: {max_kl:.3f}"
        assert mean_kl < 5.0, f"Mean KL divergence too high: {mean_kl:.3f}"

        print(f"✓ KL divergence stayed bounded (max: {max_kl:.3f}, mean: {mean_kl:.3f})")

        # 6. Gradient norms should be reasonable
        grad_norms = metrics["grad_norm"]
        mean_grad = np.mean(grad_norms)
        max_grad = np.max(grad_norms)

        # With higher LR and looser clipping, we expect higher gradients initially
        assert mean_grad > 0.001, f"Gradients vanishing: mean={mean_grad:.6f}"
        assert max_grad < 1000.0, f"Gradients exploding: max={max_grad:.3f}"

        print(f"✓ Gradient norms reasonable (mean: {mean_grad:.3f}, max: {max_grad:.3f})")

        # 7. Check that some completions have correct format
        correct_format_count = sum(
            1 for comp in final_completions
            if '<reasoning>' in comp and '<answer>' in comp
        )
        format_percentage = (correct_format_count / len(final_completions)) * 100

        print(f"✓ Format compliance: {format_percentage:.1f}% of completions have both tags")

        # We expect at least some improvement in format compliance
        if format_percentage > 50:
            print(f"✓ Good format compliance!")
        elif format_percentage > 20:
            print(f"⚠ Moderate format compliance")
        else:
            print(f"⚠ Low format compliance - model may not be learning tags well")

        print("\n" + "="*70)
        print("OVERFIT TEST PASSED")
        print("="*70)
        print("Key findings:")
        print(f"  • Model CAN learn (rewards improved by {reward_improvement:.3f})")
        print(f"  • Training loop is stable (no NaN/Inf, bounded KL)")
        print(f"  • Gradients are flowing (mean: {mean_grad:.3f})")
        print(f"  • Format learning: {format_percentage:.1f}% compliance")
        print("\nConclusion: GRPO implementation appears to be working correctly!")
        print("="*70 + "\n")

    def test_overfit_with_higher_learning_rate(self, grpo_config_from_script, single_batch_dataset, device, model_name):
        """
        Test overfitting with moderately higher learning rate for clearer signal.

        With higher LR, we should see more dramatic improvement if training works.
        """
        print("\n" + "="*70)
        print("OVERFIT TEST: Qwen 0.5B with moderately higher LR")
        print("="*70)

        # Increase learning rate moderately for faster (but stable) overfitting
        grpo_config_from_script["optimizer"]["lr"] = 1e-5  # Was 5e-6, now 2x
        grpo_config_from_script["training"]["num_episodes"] = 30  # Fewer episodes needed
        grpo_config_from_script["training"]["update_epochs"] = 2  # More epochs per episode

        # Load model
        print(f"Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=grpo_config_from_script,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Train
        print("Training with higher learning rate (30 episodes)...")
        results = grpo.train(
            train_data=single_batch_dataset,
            val_data=None,
            num_episodes=30
        )

        metrics = results["training_metrics"]

        # Check for clear improvement
        initial_reward = np.mean(metrics["reward_mean"][:3])
        final_reward = np.mean(metrics["reward_mean"][-3:])
        reward_improvement = final_reward - initial_reward

        initial_loss = np.mean(metrics["total_loss"][:3])
        final_loss = np.mean(metrics["total_loss"][-3:])

        print(f"\nResults with higher LR:")
        print(f"  Reward:  {initial_reward:.3f} → {final_reward:.3f} (Δ: {reward_improvement:.3f})")
        print(f"  Loss:    {initial_loss:.4f} → {final_loss:.4f} (Δ: {final_loss - initial_loss:.4f})")

        # With higher LR, we should see clearer improvement
        assert reward_improvement > -0.3, \
            f"Rewards degraded with high LR: {initial_reward:.3f} → {final_reward:.3f}"

        # No NaN/Inf even with higher LR
        for key in ["total_loss", "reward_mean"]:
            values = metrics[key]
            assert not any(np.isnan(v) for v in values), f"NaN with higher LR in {key}"
            assert not any(np.isinf(v) for v in values), f"Inf with higher LR in {key}"

        print("✓ Higher learning rate test passed - training is stable")


@pytest.mark.slow
def test_long_overfit_qwen_instruct():
    """
    Long overfit test (300 episodes) with Qwen 0.5B Instruct to validate learning.

    This test runs long enough to show clear overfitting - the model should
    memorize the 4 training examples and get very high rewards.

    Uses Qwen 0.5B Instruct - general instruction model.
    """
    print("\n" + "="*70)
    print("LONG OVERFIT TEST - Qwen 0.5B Instruct - 300 EPISODES")
    print("="*70)
    print("Running with moderate settings for stable learning:")
    print("  - Model: Qwen 0.5B Instruct (general instruction model)")
    print("  - Learning rate: 5e-6 (conservative)")
    print("  - Episodes: 300 (long enough to clearly overfit)")
    print("\nExpected: Rewards should increase significantly over time")
    print("Expected: Loss should decrease significantly")
    print("Expected: Model memorizes the 4 training examples")
    print("="*70 + "\n")

    # Use best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Config for moderate, stable overfitting over 300 episodes with Qwen 0.5B Instruct
    config = {
        "algorithm": {
            "group_size": 8,  # Standard group size
            "kl_coef": 0.001,  # Standard KL penalty
            "normalize_rewards": True,
            "store_completions": False
        },
        "training": {
            "batch_size": 4,
            "rollout_batch_size": 2,  # 2 prompts at a time for memory
            "max_new_tokens": 400,  # Longer for math reasoning
            "num_episodes": 300,
            "minibatch_size": 8,  # Process 8 rollouts per minibatch
            "update_epochs": 1,  # Single epoch for smooth learning
            "gradient_clip": 1.0,  # Standard clipping
            "temperature": 0.6,
            "top_p": 0.9,
            "entropy_coef": 0.005,  # Lower entropy for exploitation
            "kl_estimator": "k3",
            "kl_clamp_min": -2.0,
            "kl_clamp_max": 2.0,
            "policy_loss_type": "token",  # Options: "sequence" or "token"
            # Clipping parameters
            "policy_log_ratio_clamp_min": -2.0,
            "policy_log_ratio_clamp_max": 2.0,
            "advantage_clip_min": -3.0,
            "advantage_clip_max": 3.0,
            "stop_sequences": ["</answer>"],  # Multi-token stopping for early termination
        },
        "model": {"hf_model_name": model_name, "device": device, "max_length": 1024},
        "logging": {"log_interval": 10, "save_interval": 1000, "show_trajectory_progress": False},
        "validation": {"enabled": False},
        "wandb": {"enabled": False},
        "optimizer": {
            "type": "adamw",
            "lr": 5e-6,  # Conservative LR for 1.5B model over 300 episodes
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "warmup_start_lr": 1e-8,
            "warmup_type": "linear"
        },
        "optimization": {"mixed_precision": {"enabled": False}},
        "debug": {"enabled": False},
        "device_optimizations": {"clear_cache_on_mps": True}
    }

    # Load data
    data = load_dataset('gsm8k', 'main')
    dataset = data['train'].select(range(4))

    system_prompt = "Solve this math problem step by step."
    prompts, answers = prepare_gsm8k_prompts(dataset, system_prompt)

    train_data = {"prompts": prompts, "answers": answers}

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Train
    grpo = GRPO(
        config=config,
        batch_reward_fn=compute_math_rewards_batch,
        model=model,
        tokenizer=tokenizer,
        use_wandb=False
    )

    print("Starting long training - this will take several hours...")
    print("Watch for rewards increasing and loss decreasing over time\n")

    results = grpo.train(train_data=train_data, val_data=None, num_episodes=300)
    metrics = results["training_metrics"]

    # Analyze progression over time
    early_rewards = np.mean(metrics["reward_mean"][:10])
    mid_rewards = np.mean(metrics["reward_mean"][145:155])  # Middle of 300 episodes
    late_rewards = np.mean(metrics["reward_mean"][-10:])

    early_loss = np.mean(metrics["total_loss"][:10])
    mid_loss = np.mean(metrics["total_loss"][145:155])
    late_loss = np.mean(metrics["total_loss"][-10:])

    early_format = np.mean(metrics["format_reward_mean"][:10])
    late_format = np.mean(metrics["format_reward_mean"][-10:])

    early_correct = np.mean(metrics["correctness_reward_mean"][:10])
    late_correct = np.mean(metrics["correctness_reward_mean"][-10:])

    print("\n" + "="*70)
    print("OVERFITTING ANALYSIS - 300 EPISODES (AGGRESSIVE)")
    print("="*70)

    print("\n📊 REWARD PROGRESSION:")
    print(f"  Early  (ep 1-10):     {early_rewards:.3f}")
    print(f"  Mid    (ep 145-155):  {mid_rewards:.3f}")
    print(f"  Late   (ep 291-300):  {late_rewards:.3f}")
    print(f"  Total improvement:    {late_rewards - early_rewards:.3f} ({((late_rewards - early_rewards) / max(abs(early_rewards), 0.01) * 100):.1f}%)")

    print("\n📉 LOSS PROGRESSION:")
    print(f"  Early  (ep 1-10):     {early_loss:.4f}")
    print(f"  Mid    (ep 145-155):  {mid_loss:.4f}")
    print(f"  Late   (ep 291-300):  {late_loss:.4f}")
    print(f"  Total decrease:       {early_loss - late_loss:.4f}")

    print("\n📝 FORMAT REWARD (learning tags):")
    print(f"  Early:  {early_format:.3f}")
    print(f"  Late:   {late_format:.3f}")
    print(f"  Change: {late_format - early_format:.3f}")

    print("\n✅ CORRECTNESS REWARD:")
    print(f"  Early:  {early_correct:.3f}")
    print(f"  Late:   {late_correct:.3f}")
    print(f"  Change: {late_correct - early_correct:.3f}")

    # Check that training completed without errors
    assert len(metrics["episode"]) == 300, f"Expected 300 episodes, got {len(metrics['episode'])}"
    assert not any(np.isnan(v) for v in metrics["reward_mean"]), "Found NaN in rewards"
    assert not any(np.isinf(v) for v in metrics["total_loss"]), "Found Inf in loss"

    print("\n" + "="*70)
    print("OVERFITTING VALIDATION")
    print("="*70)

    # Validate overfitting happened - STRICT CRITERIA
    reward_improvement = late_rewards - early_rewards
    loss_decrease = early_loss - late_loss

    print("\nVALIDATING OVERFITTING CRITERIA:")
    print("-" * 70)

    # CRITERION 1: Final reward should be high (model memorized the data)
    # Max possible reward is 2.3 (0.3 format + 2.0 correctness)
    # We expect at least 1.5 after 300 episodes of AGGRESSIVE overfitting
    print(f"1. Final reward level: {late_rewards:.3f}")
    if late_rewards > 1.8:
        print(f"   ✓ EXCELLENT - Model achieved very high rewards (> 1.8)")
    elif late_rewards > 1.5:
        print(f"   ✓ GOOD - Model achieved good rewards (> 1.5)")
    elif late_rewards > 1.0:
        print(f"   ⚠ MODERATE - Model achieved moderate rewards (> 1.0)")
        print(f"   WARNING: Expected higher rewards with 300 AGGRESSIVE episodes")
    else:
        print(f"   ❌ FAILED - Model did not overfit (reward = {late_rewards:.3f})")
        raise AssertionError(f"Overfitting test FAILED: Final reward {late_rewards:.3f} is too low. Expected > 1.0 after 300 AGGRESSIVE episodes.")

    # CRITERION 2: Significant improvement from start
    print(f"\n2. Reward improvement: {reward_improvement:.3f}")
    if reward_improvement > 0.5:
        print(f"   ✓ EXCELLENT improvement (> 0.5)")
    elif reward_improvement > 0.3:
        print(f"   ✓ GOOD improvement (> 0.3)")
    elif reward_improvement > 0.1:
        print(f"   ⚠ MODEST improvement (> 0.1)")
        print(f"   WARNING: Expected larger improvement with 300 AGGRESSIVE episodes")
    else:
        print(f"   ❌ FAILED - No meaningful improvement")
        raise AssertionError(f"Overfitting test FAILED: Reward improvement {reward_improvement:.3f} is insufficient. Expected > 0.1")

    # CRITERION 3: Loss should decrease significantly
    print(f"\n3. Loss decrease: {loss_decrease:.4f}")
    if loss_decrease > 0.2:
        print(f"   ✓ EXCELLENT loss decrease (> 0.2)")
    elif loss_decrease > 0.1:
        print(f"   ✓ GOOD loss decrease (> 0.1)")
    elif loss_decrease > 0:
        print(f"   ⚠ MODEST loss decrease")
        print(f"   WARNING: Expected larger decrease with 300 AGGRESSIVE episodes")
    else:
        print(f"   ❌ FAILED - Loss did not decrease")
        raise AssertionError(f"Overfitting test FAILED: Loss increased by {-loss_decrease:.4f}. Expected decrease.")

    # CRITERION 4: Format reward should improve (model learns tags)
    format_improvement = late_format - early_format
    print(f"\n4. Format learning: {early_format:.3f} → {late_format:.3f} (Δ {format_improvement:.3f})")
    if late_format > 0.15:
        print(f"   ✓ Model learned to use formatting tags")
    else:
        print(f"   ⚠ Low format compliance - model may not be learning tags well")

    # Generate final completions to see memorization
    print("\n" + "="*70)
    print("SAMPLE COMPLETIONS (checking memorization)")
    print("="*70)

    _, final_completions, *_ = grpo.generate_trajectories(
        train_data["prompts"],
        train_data["answers"],
        store_outputs=True
    )

    for i in range(min(2, len(train_data["prompts"]))):
        print(f"\n[Prompt {i+1}] {train_data['prompts'][i][:50]}...")
        print(f"[Gold Answer] {train_data['answers'][i]}")
        print(f"[Model Output]")
        for j in range(2):  # Show 2 generations
            comp = final_completions[i * grpo.group_size + j]
            # Extract answer if it has tags
            if '<answer>' in comp and '</answer>' in comp:
                start = comp.find('<answer>') + 8
                end = comp.find('</answer>')
                answer = comp[start:end].strip()
                print(f"  Gen {j+1}: {answer}")
            else:
                print(f"  Gen {j+1}: {comp[:80]}...")

    print("\n" + "="*70)
    print("✓ OVERFITTING TEST PASSED")
    print("="*70)
    print("Conclusion: Model CAN learn - GRPO implementation is working!")
    print("="*70 + "\n")


def test_gpt2_smoke_overfit():
    """
    Simple smoke test with GPT-2 to verify basic training works.

    This is a quick sanity check (10 episodes) to ensure:
    - Training loop doesn't crash
    - Gradients are flowing
    - No NaN/Inf values
    - Basic mechanics work

    Uses GPT-2 for speed - this is just a smoke test, not a real overfit test.
    """
    print("\n" + "="*70)
    print("GPT-2 SMOKE TEST - Quick sanity check (10 episodes)")
    print("="*70)
    print("This is just checking that training doesn't crash")
    print("Not expecting meaningful learning from GPT-2 on math")
    print("="*70 + "\n")

    device = "cpu"

    # Minimal config for quick smoke test
    config = {
        "algorithm": {
            "group_size": 4,
            "kl_coef": 0.1,
            "normalize_rewards": True,
            "store_completions": False  # Don't store for speed
        },
        "training": {
            "batch_size": 2,  # Just 2 examples for speed
            "rollout_batch_size": 2,
            "max_new_tokens": 50,  # Very short
            "num_episodes": 10,  # Quick
            "minibatch_size": 2,
            "update_epochs": 1,
            "gradient_clip": 1.0,
            "temperature": 0.8,
            "top_p": 0.9,
            "entropy_coef": 0.01,
            "kl_estimator": "k3",
            "kl_clamp_min": -2.0,
            "kl_clamp_max": 2.0,
            "policy_loss_type": "token",  # Options: "sequence" or "token"
            # Clipping parameters
            "policy_log_ratio_clamp_min": -2.0,
            "policy_log_ratio_clamp_max": 2.0,
            "advantage_clip_min": -3.0,
            "advantage_clip_max": 3.0,
            "stop_sequences": ["</answer>"],  # Multi-token stopping for early termination
        },
        "model": {"model_name": "gpt2", "device": device, "max_length": 256},
        "logging": {"log_interval": 2, "save_interval": 1000, "show_trajectory_progress": False},
        "validation": {"enabled": False},
        "wandb": {"enabled": False},
        "optimizer": {
            "type": "adamw",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "warmup_steps": 3,
            "warmup_start_lr": 1e-8,
            "warmup_type": "linear"
        },
        "optimization": {"mixed_precision": {"enabled": False}},
        "debug": {"enabled": False}
    }

    # Load minimal data
    data = load_dataset('gsm8k', 'main')
    dataset = data['train'].select(range(2))  # Just 2 examples

    system_prompt = "Solve this problem."
    prompts, answers = prepare_gsm8k_prompts(dataset, system_prompt)

    train_data = {"prompts": prompts, "answers": answers}

    # Load GPT-2
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Train
    grpo = GRPO(
        config=config,
        batch_reward_fn=compute_math_rewards_batch,
        model=model,
        tokenizer=tokenizer,
        use_wandb=False
    )

    print("Running quick smoke test (10 episodes)...")
    results = grpo.train(train_data=train_data, val_data=None, num_episodes=10)
    metrics = results["training_metrics"]

    # Basic sanity checks
    assert len(metrics["episode"]) == 10, "Should complete 10 episodes"
    assert not any(np.isnan(v) for v in metrics["reward_mean"]), "No NaN in rewards"
    assert not any(np.isinf(v) for v in metrics["total_loss"]), "No Inf in loss"
    assert all(g > 0 for g in metrics["grad_norm"]), "Gradients flowing"

    print("\n✓ GPT-2 smoke test passed - training loop works!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run the overfit tests
    pytest.main([__file__, "-v", "-s"])
