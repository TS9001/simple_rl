#!/usr/bin/env python3
"""
Full SFT + GRPO Training Pipeline
==================================

This script runs the complete training pipeline:
1. (Optional) Supervised Fine-Tuning (SFT) with CoT examples
2. Group Relative Policy Optimization (GRPO) using SFT checkpoint
3. Evaluation and visualization

Usage:
    python scripts/full_pipeline_sft_grpo_training.py
"""

import os
import sys
import hashlib
import tarfile
import shutil
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import requests
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_rl.algorithms import SFT, GRPO
from simple_rl.evaluation.gsm8k import (
    load_gsm8k_dataset,
    prepare_gsm8k_prompts,
    prepare_gsm8k_for_sft,
    evaluate_on_gsm8k,
    demonstrate_model_responses
)
from simple_rl.rewards import (
    extract_answer_from_model_output,
    compute_math_rewards_batch,
)
from simple_rl.utils.dataset_cache import (
    load_gsm8k_cache,
    save_gsm8k_cache,
    load_cot_cache,
    save_cot_cache,
)
from simple_rl.utils.notebook_logger import setup_notebook_logger

# ============================================================
# Configuration
# ============================================================

# Training flags
RUN_SFT = False  # Set to True to run SFT training, False to load from checkpoint

# Resume training configuration
CONTINUE_FROM = 0  # Set to episode number to resume from GRPO checkpoint, 0 = start from SFT/base model

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# System prompt for math formatting
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# SFT checkpoint path (used when RUN_SFT=False and CONTINUE_FROM=0)
SFT_CHECKPOINT_PATH = Path("checkpoints/pipeline_stages/01_after_sft") / "sft_complete.pt"

# Dataset cache directory (for deterministic loading)
DATASET_CACHE_DIR = "dataset_cache"


# ============================================================
# Helper Functions
# ============================================================

def download_and_extract_cot_archive(url, extract_path="cot_archive"):
    """Download and extract the CoT archive if not already done."""
    archive_path = os.path.join(extract_path, "cot.tar.gz")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)

    if not os.path.exists(archive_path):
        print("Downloading CoT archive from reference notebook...")
        r = requests.get(url, stream=True)
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✓ Downloaded to {archive_path}")

    # Extract the archive if not already extracted
    extract_dir = os.path.join(extract_path, "cot_files")
    if not os.path.exists(extract_dir):
        print("Extracting CoT archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"✓ Extracted to {extract_dir}")

    return extract_dir


def prepare_cot_dataset_for_sft(system_prompt, num_examples=500):
    """
    Prepare high-quality CoT examples from the reference notebook's dataset.
    Returns data in the same format as prepare_gsm8k_for_sft().
    """
    cot_url = "https://github.com/aburkov/theLMbook/releases/download/v1.0.0/cot.tar.gz"
    extract_dir = download_and_extract_cot_archive(cot_url)

    # Load GSM8K to get questions
    data = load_dataset('gsm8k', 'main')["train"]

    prompts = []
    completions = []

    for example in data:
        question = example["question"].strip()

        # Compute the filename based on SHA-256 hash of the question
        filename = hashlib.sha256(question.encode()).hexdigest() + ".txt"
        file_path = os.path.join(extract_dir, filename)

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                cot_output = f.read().strip()

            # Build prompt with system prompt
            prompt = f"{system_prompt.strip()}\n{question}"
            prompts.append(prompt)
            completions.append(cot_output)

        if len(prompts) >= num_examples:
            break

    print(f"\n✓ Loaded {len(prompts)} high-quality CoT examples")
    print(f"  These have cleaner reasoning than raw GSM8K")

    return {"prompts": prompts, "completions": completions}


def setup_device():
    """Setup and return the appropriate device."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device detection
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(42)
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def run_sft_training(sft_config, sft_train_data_cot, sft_val_data):
    """Run SFT training and return results."""
    print("\n" + "="*60)
    print("STARTING SFT TRAINING WITH HIGH-QUALITY CoT DATASET")
    print("="*60)

    # Initialize SFT
    print("Initializing SFT...")
    sft = SFT(config=sft_config, use_wandb=False)

    print(f"Using {len(sft_train_data_cot['prompts'])} high-quality CoT examples")
    print(f"Note: Checkpoints will be auto-saved every {sft_config['logging']['save_interval']} steps")
    print("="*60 + "\n")

    # Train the model
    sft_train_data_cot["debug_boundary"] = True
    sft_results = sft.train(
        train_data=sft_train_data_cot,
        val_data=sft_val_data,
        num_episodes=sft_config['training']['num_epochs']
    )

    print("\n" + "="*60)
    print("SFT TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {sft_results['total_time']:.2f} seconds ({sft_results['total_time']/60:.2f} minutes)")
    print(f"Final loss: {sft_results['final_loss']:.4f}")

    # Save SFT checkpoint
    sft_checkpoint_dir = Path("checkpoints/pipeline_stages/01_after_sft")
    sft_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sft_checkpoint_path = sft_checkpoint_dir / "sft_complete.pt"

    sft.save_checkpoint(str(sft_checkpoint_path))

    if os.path.exists(sft_checkpoint_path):
        file_size = os.path.getsize(sft_checkpoint_path) / (1024**3)
        print(f"✓ Saved SFT checkpoint: {sft_checkpoint_path}")
        print(f"  Checkpoint size: {file_size:.2f} GB")

    return sft, sft_results, sft_checkpoint_path


def visualize_grpo_results(grpo_training_metrics, grpo_validation_metrics, grpo):
    """Create visualization plots for GRPO training."""
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    fig.suptitle('GRPO Training and Validation Metrics', fontsize=16)

    # Total Loss
    axes[0, 0].plot(grpo_training_metrics["episode"], grpo_training_metrics["total_loss"], 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss over Training')
    axes[0, 0].grid(True, alpha=0.3)

    # Mean Reward
    axes[0, 1].plot(grpo_training_metrics["episode"], grpo_training_metrics["reward_mean"], 'purple', alpha=0.7, label='Mean')
    axes[0, 1].fill_between(
        grpo_training_metrics["episode"],
        np.array(grpo_training_metrics["reward_mean"]) - np.array(grpo_training_metrics["reward_std"]),
        np.array(grpo_training_metrics["reward_mean"]) + np.array(grpo_training_metrics["reward_std"]),
        alpha=0.3, color='purple'
    )
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Mean Reward ± Std')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL Divergence
    axes[1, 0].plot(grpo_training_metrics["episode"], grpo_training_metrics["kl_divergence"], 'r-', alpha=0.7)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence from Reference')
    axes[1, 0].grid(True, alpha=0.3)

    # Validation Accuracy
    if grpo_validation_metrics["episode"]:
        axes[1, 1].plot(grpo_validation_metrics["episode"], grpo_validation_metrics["exact_accuracy"],
                        'go-', label='Exact Match', markersize=8)
        axes[1, 1].plot(grpo_validation_metrics["episode"], grpo_validation_metrics["numeric_accuracy"],
                        'bs-', label='Numeric Match', markersize=8)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Format Compliance
    if grpo_validation_metrics["episode"]:
        axes[2, 0].plot(grpo_validation_metrics["episode"], grpo_validation_metrics["format_compliance"],
                        'mo-', label='Format Compliance %', markersize=8)
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Compliance (%)')
        axes[2, 0].set_title('Format Compliance')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

    # Episode Time
    axes[2, 1].plot(grpo_training_metrics["episode"], grpo_training_metrics["episode_time"],
                    'orange', alpha=0.7, linewidth=2)
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Time (seconds)')
    axes[2, 1].set_title('Episode Training Time')
    axes[2, 1].grid(True, alpha=0.3)

    # Cumulative Tokens
    axes[3, 0].plot(grpo_training_metrics["episode"],
                    [t/1000 for t in grpo_training_metrics["total_tokens"]],
                    'green', alpha=0.7, linewidth=2)
    axes[3, 0].set_xlabel('Episode')
    axes[3, 0].set_ylabel('Tokens (thousands)')
    axes[3, 0].set_title('Cumulative Tokens Processed')
    axes[3, 0].grid(True, alpha=0.3)

    # Processing Speed
    axes[3, 1].plot(grpo_training_metrics["episode"], grpo_training_metrics["tokens_per_second"],
                    'darkblue', alpha=0.7, linewidth=2)
    axes[3, 1].set_xlabel('Episode')
    axes[3, 1].set_ylabel('Tokens/Second')
    axes[3, 1].set_title('Token Processing Speed')
    axes[3, 1].grid(True, alpha=0.3)

    # Gradient Norm
    axes[4, 0].plot(grpo_training_metrics["episode"],
                    grpo_training_metrics["grad_norm"],
                    'purple', alpha=0.7, linewidth=2)
    axes[4, 0].axhline(y=grpo.gradient_clip, color='r', linestyle='--',
                       label=f'Clip={grpo.gradient_clip}', linewidth=1)
    axes[4, 0].set_xlabel('Episode')
    axes[4, 0].set_ylabel('Gradient Norm')
    axes[4, 0].set_title('Gradient Norm (Before Clipping)')
    axes[4, 0].legend()
    axes[4, 0].grid(True, alpha=0.3)

    # Gradient vs Loss scatter
    axes[4, 1].scatter(grpo_training_metrics["grad_norm"],
                       grpo_training_metrics["total_loss"],
                       alpha=0.6, s=30, c=grpo_training_metrics["episode"],
                       cmap='viridis')
    axes[4, 1].set_xlabel('Gradient Norm')
    axes[4, 1].set_ylabel('Total Loss')
    axes[4, 1].set_title('Gradient Norm vs Loss')
    axes[4, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "grpo_training_metrics.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_dir / 'grpo_training_metrics.png'}")

    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("GRPO TRAINING SUMMARY")
    print("="*60)
    print(f"Final Total Loss: {grpo_training_metrics['total_loss'][-1]:.4f}")
    print(f"Final KL Divergence: {grpo_training_metrics['kl_divergence'][-1]:.4f}")
    print(f"Final Mean Reward: {grpo_training_metrics['reward_mean'][-1]:.3f}")
    print(f"Average Reward (last 5 episodes): {np.mean(grpo_training_metrics['reward_mean'][-5:]):.3f}")
    print(f"Total Tokens Processed: {grpo_training_metrics['total_tokens'][-1]:,}")
    print(f"Average Processing Speed: {np.mean(grpo_training_metrics['tokens_per_second']):.1f} tokens/second")

    # Gradient statistics
    print("\n" + "="*60)
    print("GRADIENT NORM STATISTICS")
    print("="*60)
    print(f"Mean Gradient Norm: {np.mean(grpo_training_metrics['grad_norm']):.3f}")
    print(f"Max Gradient Norm: {np.max(grpo_training_metrics['grad_norm']):.3f}")
    print(f"Gradient Clip Threshold: {grpo.gradient_clip}")
    grad_norms = np.array(grpo_training_metrics['grad_norm'])
    clipped_fraction = (grad_norms > grpo.gradient_clip).sum() / len(grad_norms) * 100
    print(f"Gradients Clipped: {clipped_fraction:.1f}% of updates")


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    # --------------------------------------------------------
    # Initialize Logging (console + file)
    # --------------------------------------------------------
    logger, log_path = setup_notebook_logger("logs", "training")

    # Logger already outputs to both console and file automatically
    print("="*60)
    print("Full SFT + GRPO Training Pipeline")
    print("="*60)
    print(f"📋 Log file: {log_path}")
    print("="*60)

    # Setup
    device = setup_device()
    print(f"Model: {MODEL_NAME}")
    print(f"System Prompt: {SYSTEM_PROMPT.strip()}")

    # --------------------------------------------------------
    # 1. Load and Prepare Dataset (with caching for deterministic loading)
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("Loading and Preparing Dataset (with caching)")
    print("="*60)

    # Define splits (used for cache key generation)
    gsm8k_splits = {
        "sft_train": "train[:500]",
        "grpo_train": "train[500:]",
        "val": "test[200:1200]",
        "test": "test[:200]"
    }

    # Try to load from cache first
    print("\n🔍 Checking for cached GSM8K dataset...")
    cached_data = load_gsm8k_cache(
        cache_dir=DATASET_CACHE_DIR,
        splits=gsm8k_splits,
        system_prompt=SYSTEM_PROMPT
    )

    if cached_data is not None:
        # Load from cache (fast, deterministic)
        print("✓ Loading dataset from cache (deterministic)")
        (sft_train_data, sft_val_data, sft_test_data,
         grpo_train_prompts, grpo_train_answers,
         grpo_val_prompts, grpo_val_answers,
         grpo_test_prompts, grpo_test_answers) = cached_data
    else:
        # Load from HuggingFace and process (slow, first time only)
        print("⏳ Cache not found, loading from HuggingFace and processing...")

        # Load SFT data
        dataset_sft_train, dataset_val, dataset_test = load_gsm8k_dataset(
            train_split=gsm8k_splits["sft_train"],
            val_split=gsm8k_splits["val"],
            test_split=gsm8k_splits["test"]
        )

        # Load GRPO data (different split - NO OVERLAP)
        dataset_grpo_train, _, _ = load_gsm8k_dataset(
            train_split=gsm8k_splits["grpo_train"],
            val_split=gsm8k_splits["val"],
            test_split=gsm8k_splits["test"]
        )

        # Prepare SFT data
        sft_train_data = prepare_gsm8k_for_sft(dataset_sft_train, SYSTEM_PROMPT)
        sft_val_data = prepare_gsm8k_for_sft(dataset_val, SYSTEM_PROMPT)
        sft_test_data = prepare_gsm8k_for_sft(dataset_test, SYSTEM_PROMPT)

        # Prepare GRPO data
        grpo_train_prompts, grpo_train_answers = prepare_gsm8k_prompts(dataset_grpo_train, SYSTEM_PROMPT)
        grpo_val_prompts, grpo_val_answers = prepare_gsm8k_prompts(dataset_val, SYSTEM_PROMPT)
        grpo_test_prompts, grpo_test_answers = prepare_gsm8k_prompts(dataset_test, SYSTEM_PROMPT)

        # Save to cache for next time
        save_gsm8k_cache(
            cache_dir=DATASET_CACHE_DIR,
            splits=gsm8k_splits,
            system_prompt=SYSTEM_PROMPT,
            sft_train_data=sft_train_data,
            sft_val_data=sft_val_data,
            sft_test_data=sft_test_data,
            grpo_train_prompts=grpo_train_prompts,
            grpo_train_answers=grpo_train_answers,
            grpo_val_prompts=grpo_val_prompts,
            grpo_val_answers=grpo_val_answers,
            grpo_test_prompts=grpo_test_prompts,
            grpo_test_answers=grpo_test_answers
        )

    # Load high-quality CoT dataset (also with caching)
    print("\n🔍 Checking for cached CoT dataset...")
    cot_num_examples = 500
    sft_train_data_cot = load_cot_cache(
        cache_dir=DATASET_CACHE_DIR,
        system_prompt=SYSTEM_PROMPT,
        num_examples=cot_num_examples
    )

    if sft_train_data_cot is None:
        # Load from source and cache
        print("⏳ CoT cache not found, downloading and processing...")
        sft_train_data_cot = prepare_cot_dataset_for_sft(SYSTEM_PROMPT, num_examples=cot_num_examples)

        # Save to cache
        save_cot_cache(
            cache_dir=DATASET_CACHE_DIR,
            system_prompt=SYSTEM_PROMPT,
            num_examples=cot_num_examples,
            sft_train_data_cot=sft_train_data_cot
        )
    else:
        print(f"✓ Loaded {len(sft_train_data_cot['prompts'])} CoT examples from cache")

    print(f"\n{'='*60}")
    print("Dataset prepared for full pipeline (NO OVERLAP)")
    print(f"{'='*60}")
    print(f"SFT Training samples: {len(sft_train_data_cot['prompts'])} (high-quality CoT)")
    print(f"GRPO Training samples: {len(grpo_train_prompts)}")
    print(f"Validation samples: {len(sft_val_data['prompts'])}")
    print(f"Test samples: {len(sft_test_data['prompts'])}")

    # --------------------------------------------------------
    # 2. SFT Training (Optional)
    # --------------------------------------------------------
    if RUN_SFT:
        sft_config = {
            "model": {
                "model_name": MODEL_NAME,
                "max_length": 512,
                "device": str(device),
                "model_type": "fp32",
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-5,
                "num_epochs": 3,
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1.0,
                "warmup_steps": 20,
                "mask_prompt": True,
                "gradient_checkpointing": False,
                "weight_decay": 0.01,
                "label_smoothing": 0.00,
                "mixed_precision": {
                    "enabled": False,
                    "dtype": "fp32"
                }
            },
            "logging": {"log_interval": 10, "save_interval": 50},
            "validation": {"enabled": True, "interval": 20, "num_samples": 100},
            "wandb": {"enabled": False}
        }

        sft, sft_results, sft_checkpoint_path = run_sft_training(
            sft_config, sft_train_data_cot, sft_val_data
        )
    else:
        print(f"\nSkipping SFT training, loading from checkpoint: {SFT_CHECKPOINT_PATH}")
        sft_checkpoint_path = SFT_CHECKPOINT_PATH

    # --------------------------------------------------------
    # 3. GRPO Configuration - Updated with working settings from overfit test
    # --------------------------------------------------------
    grpo_config = {
        "algorithm": {
            "name": "grpo",
            "group_size": 8,  # Keep at 8 (working well in overfit test)
            "kl_coef": 0.01,  # Increased from 0.1 → 0.15 for more stability
            "clip_epsilon": 0.2,
            "normalize_rewards": True,
            "store_completions": False,
        },
        "training": {
            "batch_size": 16,  # Keep at 16 for full training (not 4 like overfit)
            "rollout_batch_size": 4,
            "gradient_clip": 0.1,  # TIGHTENED from 1.0 → 0.1 for maximum stability
            "max_new_tokens": 400,
            "min_new_tokens": 50,  # LOWERED from 150 → 50 to allow </answer> early stopping
            "temperature": 0.6,
            "num_episodes": 500,
            "minibatch_size": 32,  # Keep at 32 for full training
            "update_epochs": 1,
            "top_p": 0.9,
            "entropy_coef": 0.005,  # Increased from 0.002 → 0.005 for more exploration
            "policy_loss_type": "token",
            "resample_batch_per_episode": True,  # ← CRITICAL: Set to True to disable fixed batch!
            # Clipping parameters (all validated in overfit test)
            "kl_estimator": "k3",
            "kl_clamp_min": -2.0,
            "kl_clamp_max": 2.0,
            "kl_reduction": "mean",
            "policy_log_ratio_clamp_min": -2.0,
            "policy_log_ratio_clamp_max": 2.0,
            "advantage_clip_min": -3.0,
            "advantage_clip_max": 3.0,
            "stop_sequences": ["</answer>"],
        },
        "model": {
            "max_length": 1024,
            "model_name": MODEL_NAME,
            "model_type": "fp32",
            "device": str(device),
            "compile": {
                "enabled": False,
                "backend": "aot_eager"
            }
        },
        "device_optimizations": {
            "clear_cache_on_mps": True,  # ENABLED for M4 unified memory
        },
        "logging": {
            "log_interval": 1,
            "save_interval": 15,
            "show_trajectory_progress": True,
        },
        "validation": {
            "enabled": True,
            "interval": 20,
            "num_samples": 50,
            "num_demo_examples": 5,
        },
        "wandb": {
            "enabled": False,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-6,  # DECREASED from 3e-6 → 1e-6 for more stability
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "fused": False,
            # INCREASED warmup for stability
            "warmup_steps": 30,  # Increased from 10 → 30
            "warmup_start_lr": 1e-9,  # Decreased from 1e-8 → 1e-9
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
            "enabled": False,  # Disabled for production training
            "log_dir": "debug_logs",
            "loss": False,
            "advantages": False,
            "gradients": False,
            "generation": False,
            "alignment": False,
        }
    }

    print("\n" + "="*70)
    print("GRPO Configuration - UPDATED WITH OVERFIT TEST SETTINGS")
    print("="*70)
    print("\n📊 KEY CHANGES FROM OVERFIT TEST:")
    print("  ✅ KL coefficient:     0.10 → 0.15 (more stability)")
    print("  ✅ Gradient clip:      1.0 → 0.1 (MUCH tighter for stability)")
    print("  ✅ Min new tokens:     150 → 50 (allow </answer> early stopping)")
    print("  ✅ Entropy coef:       0.002 → 0.005 (more exploration)")
    print("  ✅ Learning rate:      3e-6 → 1e-6 (more conservative)")
    print("  ✅ Warmup steps:       10 → 30 (longer warmup)")
    print("  ✅ Warmup start LR:    1e-8 → 1e-9 (lower start)")
    print("  ✅ RESAMPLE BATCH:     ENABLED (normal training mode - not fixed batch!)")
    print("\n📊 KEPT FOR FULL TRAINING (not from overfit):")
    print("  • Batch size:          16 (not 4 - need more data coverage)")
    print("  • Minibatch size:      32 (not 8 - better for full training)")
    print("  • Num episodes:        500 (not 100 - full training run)")
    print("\n⚠️  STOPPING BEHAVIOR:")
    print("  • min_new_tokens=50: Forces at least 50 tokens before stopping")
    print("  • stop_sequences=['</answer>']: Stops when </answer> appears")
    print("  • Result: Model generates 50-400 tokens, stops at </answer> if present")
    print("\n🧹 MEMORY MANAGEMENT:")
    print("  • Cache clearing:      ENABLED on MPS (M4 unified memory)")
    print("  • Garbage collection:  Forced every episode")
    print("  • Timing data:         Auto-reset every 10 episodes")
    print("="*70)

    # --------------------------------------------------------
    # 4. Initialize GRPO with Checkpoint (SFT or Resume)
    # --------------------------------------------------------
    if CONTINUE_FROM > 0:
        # Resume from GRPO checkpoint
        print("\n" + "="*60)
        print(f"RESUMING FROM GRPO CHECKPOINT (Episode {CONTINUE_FROM})")
        print("="*60)

        grpo_checkpoint_path = Path(f"checkpoints/grpo_qwen_math/checkpoint_episode_{CONTINUE_FROM}.pt")
        print(f"Checkpoint path: {grpo_checkpoint_path}")

        # Load GRPO checkpoint
        grpo_checkpoint = torch.load(grpo_checkpoint_path, map_location=device)

        print("\n" + "="*60)
        print("CHECKPOINT CONTENTS:")
        print("="*60)
        print(f"Keys in checkpoint: {list(grpo_checkpoint.keys())}")
        print(f"Episode: {grpo_checkpoint.get('episode')}")
        print(f"Total steps: {grpo_checkpoint.get('total_steps')}")
        print(f"Current episode: {grpo_checkpoint.get('current_episode')}")

        # Show optimizer config from checkpoint
        opt_state = grpo_checkpoint["optimizer_state_dict"]
        print(f"\nOptimizer state keys: {list(opt_state.keys())}")
        if "param_groups" in opt_state:
            pg = opt_state["param_groups"][0]
            print(f"Optimizer param_group[0]:")
            for key in ['lr', 'betas', 'eps', 'weight_decay']:
                if key in pg:
                    print(f"  {key}: {pg[key]}")

        # Show scheduler state from checkpoint
        sched_state = grpo_checkpoint.get("scheduler_state_dict")
        print(f"\nScheduler state: {sched_state}")

        # Show optimizer config from checkpoint's config
        checkpoint_config = grpo_checkpoint.get("config", {})
        checkpoint_opt_config = checkpoint_config.get("optimizer", {})
        print(f"\nOptimizer config (from checkpoint):")
        print(f"  lr: {checkpoint_opt_config.get('lr')}")
        print(f"  warmup_steps: {checkpoint_opt_config.get('warmup_steps')}")
        print(f"  warmup_start_lr: {checkpoint_opt_config.get('warmup_start_lr')}")
        print("="*60)

        model_name = checkpoint_config["model"]["model_name"]
        print(f"\nModel: {model_name}")
        print(f"Resuming from episode: {grpo_checkpoint['episode']}")

        # Load tokenizer only (lightweight, no weights)
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")

        # Load model architecture without pretrained weights (much faster!)
        print("\nLoading model architecture from config...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to(dtype=torch.float32)
        print("✓ Model architecture loaded (no pretrained weights downloaded)")

        # Load trained weights from GRPO checkpoint
        # GRPO wraps model in LanguageModel class, so keys have "model." prefix that needs to be removed
        print("\nLoading trained weights from checkpoint...")
        policy_state = grpo_checkpoint["policy_state_dict"]

        # Strip "model." prefix from keys (e.g., "model.model.embed_tokens.weight" -> "model.embed_tokens.weight")
        model_state = {k.replace("model.", "", 1): v for k, v in policy_state.items() if k.startswith("model.")}

        model.load_state_dict(model_state)
        print("✓ Checkpoint weights loaded successfully")

        # Initialize GRPO
        print("\nInitializing GRPO and restoring optimizer/scheduler state...")
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Restore optimizer state (momentum, variance, AND learning rate from checkpoint)
        grpo.optimizer.load_state_dict(grpo_checkpoint["optimizer_state_dict"])
        checkpoint_lr = grpo.optimizer.param_groups[0]['lr']

        print(f"  ✓ Optimizer state restored")
        print(f"  ✓ Learning rate from checkpoint: {checkpoint_lr:.2e}")

        # CRITICAL: Disable the scheduler when resuming!
        # At episode 45, we're WAY past warmup (30 steps). We should continue with the checkpoint's LR.
        # The scheduler was created fresh in GRPO.__init__ and would restart warmup - we don't want that!
        total_steps = grpo_checkpoint.get("total_steps", 0)
        warmup_steps = grpo_config["optimizer"]["warmup_steps"]

        if total_steps >= warmup_steps:
            # Past warmup - disable scheduler entirely
            grpo.lr_scheduler = None
            print(f"  ✓ Scheduler disabled (past warmup: step {total_steps}/{warmup_steps})")
            print(f"  ✓ Will continue with fixed LR = {checkpoint_lr:.2e}")
        else:
            # Still in warmup phase - need to advance scheduler to correct position
            print(f"  ⚠️  Still in warmup phase (step {total_steps}/{warmup_steps})")
            print(f"  ⚠️  Advancing scheduler to step {total_steps}...")

            # Step the scheduler to catch up to where we were
            for _ in range(total_steps):
                grpo.lr_scheduler.step()

            final_lr = grpo.optimizer.param_groups[0]['lr']
            print(f"  ✓ Scheduler advanced, current LR: {final_lr:.2e}")

        # CRITICAL: Restore reference policy (the original SFT model, not a copy of trained policy!)
        if "ref_policy_state_dict" in grpo_checkpoint and grpo_checkpoint["ref_policy_state_dict"] is not None:
            grpo.ref_policy.load_state_dict(grpo_checkpoint["ref_policy_state_dict"])
            print(f"  ✓ Reference policy restored (KL constraint preserved)")
        else:
            print(f"  ⚠️  WARNING: No reference policy in checkpoint - KL divergence will be incorrect!")

        # Restore total_steps for proper tracking
        if "total_steps" in grpo_checkpoint:
            grpo.total_steps = grpo_checkpoint["total_steps"]
            print(f"  ✓ Training steps restored: {grpo.total_steps}")

        # Restore episode counter
        # Checkpoint contains the LAST completed episode, so resume at NEXT episode
        # E.g., checkpoint_episode_45.pt has episode=44, so we start at 45
        last_completed_episode = grpo_checkpoint["episode"]
        grpo.current_episode = last_completed_episode + 1

        print(f"  Last completed episode in checkpoint: {last_completed_episode}")
        print(f"  Will resume training from episode: {grpo.current_episode}")

        print("\n" + "="*60)
        print(f"✓ GRPO successfully resumed from episode {CONTINUE_FROM}")
        print("="*60)
    else:
        # Start from SFT checkpoint or base model
        print("\n" + "="*60)
        print("Loading SFT checkpoint and initializing GRPO")
        print("="*60)
        print(f"Checkpoint path: {sft_checkpoint_path}")

        # Load SFT checkpoint
        sft_checkpoint = torch.load(sft_checkpoint_path, map_location=device)
        model_name = sft_checkpoint["config"]["model"]["model_name"]
        print(f"Model: {model_name}")

        # Load fresh model and tokenizer
        print("\nLoading fresh model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load SFT weights
        print("Loading SFT weights into model...")
        model.load_state_dict(sft_checkpoint["model_state_dict"])
        print("✓ SFT weights loaded successfully")

        # Initialize GRPO
        print("\nInitializing GRPO with SFT-trained model...")
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        print("\n" + "="*60)
        print("✓ GRPO successfully initialized with SFT-trained model")
        print("="*60)

    # --------------------------------------------------------
    # 5. GRPO Training
    # --------------------------------------------------------
    grpo_train_data = {
        "prompts": grpo_train_prompts,
        "answers": grpo_train_answers
    }

    grpo_val_data = {
        "prompts": grpo_val_prompts,
        "answers": grpo_val_answers
    }

    print("\n" + "="*60)
    print("STARTING GRPO TRAINING")
    print("="*60)

    grpo_results = grpo.train(
        train_data=grpo_train_data,
        val_data=grpo_val_data,
        num_episodes=grpo_config['training']['num_episodes']
    )

    print("\n" + "="*60)
    print("GRPO TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {grpo_results['total_time']:.2f} seconds ({grpo_results['total_time']/60:.2f} minutes)")
    print(f"Total tokens processed: {grpo_results['training_metrics']['total_tokens'][-1]:,}")
    print(f"Final reward: {grpo_results['final_reward']:.3f}")

    # --------------------------------------------------------
    # 6. Evaluation
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("GRPO MODEL EVALUATION")
    print("="*60)

    grpo_metrics = evaluate_on_gsm8k(
        grpo,
        grpo_test_prompts,
        grpo_test_answers,
        len(grpo_test_prompts),
        model_name="GRPO Model (After RL Training)",
        save_results=True,
        results_file="results/grpo_eval_results.json",
        step=grpo_config["training"]["num_episodes"]
    )

    # Show examples
    demonstrate_model_responses(
        grpo,
        grpo_test_prompts,
        grpo_test_answers,
        5,
        title="GRPO MODEL EXAMPLES (After RL Training)"
    )

    # --------------------------------------------------------
    # 7. Visualization
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)

    visualize_grpo_results(
        grpo_results['training_metrics'],
        grpo_results['validation_metrics'],
        grpo
    )

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    # Close logging
    from simple_rl.utils.notebook_logger import close_logger
    close_logger(logger)
    print(f"✓ Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
