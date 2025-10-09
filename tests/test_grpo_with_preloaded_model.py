"""Test GRPO initialization with pre-loaded model and tokenizer."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_rl.algorithms.grpo import GRPO
from simple_rl.rewards.simple_reward import compute_math_rewards_batch


def test_grpo_with_config_loading():
    """Test GRPO with traditional config-based loading (backward compatibility)."""
    print("\n" + "="*60)
    print("Test 1: GRPO with config-based loading")
    print("="*60)

    config = {
        "algorithm": {
            "name": "grpo",
            "group_size": 4,
            "kl_coef": 0.04,
            "clip_epsilon": 0.2,
            "normalize_rewards": True,
            "store_completions": False,
        },
        "training": {
            "batch_size": 8,
            "rollout_batch_size": 2,
            "gradient_clip": 0.1,
            "max_new_tokens": 64,
            "temperature": 1.0,
            "num_episodes": 5,
            "minibatch_size": 32,
            "update_epochs": 1,
            "top_p": 1.0
        },
        "model": {
            "max_length": 512,
            "model_name": "Qwen/Qwen2.5-0.5B",
            "model_type": "fp32",
            "device": "cpu",
            "compile": {
                "enabled": False,
            }
        },
        "logging": {
            "log_interval": 1,
            "save_interval": 10,
        },
        "optimizer": {
            "type": "adam",
            "lr": 5e-6,
        },
    }

    grpo = GRPO(
        config=config,
        batch_reward_fn=compute_math_rewards_batch,
        use_wandb=False
    )

    print("✓ GRPO initialized successfully with config-based loading")
    print(f"  Model type: {type(grpo.policy.model)}")
    print(f"  Tokenizer type: {type(grpo.policy.tokenizer)}")

    return grpo


def test_grpo_with_preloaded_model():
    """Test GRPO with pre-loaded model and tokenizer."""
    print("\n" + "="*60)
    print("Test 2: GRPO with pre-loaded model and tokenizer")
    print("="*60)

    # Pre-load model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Pre-loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    print(f"✓ Model and tokenizer pre-loaded")
    print(f"  Model type: {type(model)}")
    print(f"  Tokenizer type: {type(tokenizer)}")

    # Create config (model_name can be omitted or just for reference)
    config = {
        "algorithm": {
            "name": "grpo",
            "group_size": 4,
            "kl_coef": 0.04,
            "clip_epsilon": 0.2,
            "normalize_rewards": True,
            "store_completions": False,
        },
        "training": {
            "batch_size": 8,
            "rollout_batch_size": 2,
            "gradient_clip": 0.1,
            "max_new_tokens": 64,
            "temperature": 1.0,
            "num_episodes": 5,
            "minibatch_size": 32,
            "update_epochs": 1,
            "top_p": 1.0
        },
        "model": {
            "max_length": 512,
            "model_name": model_name,  # For reference only
            "device": "cpu",
            "compile": {
                "enabled": False,
            }
        },
        "logging": {
            "log_interval": 1,
            "save_interval": 10,
        },
        "optimizer": {
            "type": "adam",
            "lr": 5e-6,
        },
    }

    # Initialize GRPO with pre-loaded model
    grpo = GRPO(
        config=config,
        batch_reward_fn=compute_math_rewards_batch,
        model=model,
        tokenizer=tokenizer,
        use_wandb=False
    )

    print("✓ GRPO initialized successfully with pre-loaded model")
    print(f"  Policy model type: {type(grpo.policy.model)}")
    print(f"  Policy tokenizer type: {type(grpo.policy.tokenizer)}")

    # Verify the models are the same instance (before compilation)
    assert grpo.policy.tokenizer is tokenizer, "Tokenizer should be the same instance"
    print("✓ Pre-loaded tokenizer is being used")

    return grpo


def test_sft_to_grpo_workflow():
    """Test loading SFT checkpoint into GRPO (simulated)."""
    print("\n" + "="*60)
    print("Test 3: SFT → GRPO workflow (simulated)")
    print("="*60)

    # Simulate SFT checkpoint structure
    model_name = "Qwen/Qwen2.5-0.5B"

    # Load fresh model
    print("Loading model from HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Simulate loading from SFT checkpoint
    # (In real use: model.load_state_dict(checkpoint["model_state_dict"]))
    print("✓ Model loaded (simulating SFT checkpoint)")

    # Create GRPO config
    config = {
        "algorithm": {
            "name": "grpo",
            "group_size": 4,
            "kl_coef": 0.04,
            "clip_epsilon": 0.2,
            "normalize_rewards": True,
        },
        "training": {
            "batch_size": 8,
            "rollout_batch_size": 2,
            "minibatch_size": 32,
            "max_new_tokens": 64,
        },
        "model": {
            "device": "cpu",
            "compile": {"enabled": False},
        },
        "optimizer": {
            "type": "adam",
            "lr": 5e-6,
        },
    }

    # Pass pre-loaded model to GRPO
    print("\nInitializing GRPO with SFT model...")
    grpo = GRPO(
        config=config,
        batch_reward_fn=compute_math_rewards_batch,
        model=model,
        tokenizer=tokenizer,
        use_wandb=False
    )

    print("✓ GRPO initialized with SFT-trained model")
    print("  This workflow allows seamless SFT → GRPO transition")

    return grpo


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GRPO Pre-loaded Model Tests")
    print("="*60)

    # Run tests
    try:
        # Test 1: Config-based loading (backward compatibility)
        grpo1 = test_grpo_with_config_loading()
        del grpo1  # Free memory

        # Test 2: Pre-loaded model
        grpo2 = test_grpo_with_preloaded_model()
        del grpo2  # Free memory

        # Test 3: SFT → GRPO workflow
        grpo3 = test_sft_to_grpo_workflow()
        del grpo3  # Free memory

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
