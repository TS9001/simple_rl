#!/usr/bin/env python3
"""
Full SFT + GRPO Training Pipeline
==================================

This script runs the complete training pipeline:
1. (Optional) Supervised Fine-Tuning (SFT) with CoT examples
2. Group Relative Policy Optimization (GRPO) using SFT checkpoint
3. Evaluation and visualization

Usage:
    # Local machine:
    python scripts/full_pipeline_sft_grpo_training.py

    # Remote server (keeps running after SSH disconnect):
    nohup python -u scripts/full_pipeline_sft_grpo_training.py > training.log 2>&1 &

    # Monitor progress from another terminal:
    tail -f training.log                    # View log output
    watch -n 5 cat logs/progress.json       # Watch JSON progress
    python scripts/monitor_training.py      # Use monitoring script

Remote Server Setup:
    1. Connect to server: ssh user@server
    2. Navigate to project: cd /path/to/simple_rl
    3. Start training: nohup python -u scripts/full_pipeline_sft_grpo_training.py > training.log 2>&1 &
    4. Get process ID: echo $!
    5. Disconnect safely: exit
    6. Reconnect and monitor: tail -f training.log
    7. Kill if needed: kill <PID>

Output Files:
    - training.log: Complete console output (all print statements)
    - logs/training_*.log: Structured logging output
    - logs/progress.json: Real-time progress tracking (updated every episode)
    - checkpoints/: Model checkpoints (saved every N episodes)
"""

import os
import sys
import hashlib
import tarfile
import shutil
import json
import time
from datetime import datetime
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
from simple_rl.utils.logging import setup_logging

# ============================================================
# Progress Tracking for Remote Monitoring
# ============================================================

class ProgressTracker:
    """Tracks training progress and writes to JSON file for remote monitoring."""

    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.log_dir / "progress.json"
        self.start_time = time.time()

        self.progress = {
            "status": "initializing",
            "phase": "setup",
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "elapsed_time_seconds": 0,
            "current_episode": 0,
            "total_episodes": 0,
            "progress_percent": 0.0,
            "latest_metrics": {},
            "checkpoints": [],
            "error": None
        }
        self._save()

    def update(self, **kwargs):
        """Update progress with new information."""
        self.progress.update(kwargs)
        self.progress["last_update"] = datetime.now().isoformat()
        self.progress["elapsed_time_seconds"] = int(time.time() - self.start_time)

        # Calculate progress percentage
        if self.progress["total_episodes"] > 0:
            self.progress["progress_percent"] = (
                self.progress["current_episode"] / self.progress["total_episodes"] * 100.0
            )

        self._save()

    def update_metrics(self, metrics):
        """Update latest metrics."""
        self.progress["latest_metrics"] = {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in metrics.items()
        }
        self.progress["last_update"] = datetime.now().isoformat()
        self._save()

    def add_checkpoint(self, checkpoint_path):
        """Record a new checkpoint."""
        self.progress["checkpoints"].append({
            "path": str(checkpoint_path),
            "episode": self.progress["current_episode"],
            "timestamp": datetime.now().isoformat()
        })
        self._save()

    def set_error(self, error_msg):
        """Record an error."""
        self.progress["status"] = "error"
        self.progress["error"] = str(error_msg)
        self.progress["last_update"] = datetime.now().isoformat()
        self._save()

    def complete(self):
        """Mark training as complete."""
        self.progress["status"] = "completed"
        self.progress["progress_percent"] = 100.0
        self.progress["last_update"] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """Save progress to JSON file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)


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

def download_and_extract_cot_archive(url, extract_path="cot_archive", logger=None):
    """Download and extract the CoT archive if not already done."""
    archive_path = os.path.join(extract_path, "cot.tar.gz")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)

    if not os.path.exists(archive_path):
        if logger:

            logger.info("Downloading CoT archive from reference notebook...")
        r = requests.get(url, stream=True)
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        if logger:

            logger.info(f"✓ Downloaded to {archive_path}")

    # Extract the archive if not already extracted
    extract_dir = os.path.join(extract_path, "cot_files")
    if not os.path.exists(extract_dir):
        if logger:

            logger.info("Extracting CoT archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        if logger:

            logger.info(f"✓ Extracted to {extract_dir}")

    return extract_dir


def prepare_cot_dataset_for_sft(system_prompt, num_examples=500, logger=None):
    """
    Prepare high-quality CoT examples from the reference notebook's dataset.
    Returns data in the same format as prepare_gsm8k_for_sft().
    """
    cot_url = "https://github.com/aburkov/theLMbook/releases/download/v1.0.0/cot.tar.gz"
    extract_dir = download_and_extract_cot_archive(cot_url, logger=logger)

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

    if logger:


        logger.info(f"\n✓ Loaded {len(prompts)} high-quality CoT examples")
    if logger:

        logger.info(f"  These have cleaner reasoning than raw GSM8K")

    return {"prompts": prompts, "completions": completions}


def setup_device(logger):
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

    if logger:


        logger.info(f"Using device: {device}")
    return device


def run_sft_training(sft_config, sft_train_data_cot, sft_val_data, logger):
    """Run SFT training and return results."""
    if logger:

        logger.info("\n" + "="*60)
    if logger:

        logger.info("STARTING SFT TRAINING WITH HIGH-QUALITY CoT DATASET")
    if logger:

        logger.info("="*60)

    # Initialize SFT
    if logger:

        logger.info("Initializing SFT...")
    sft = SFT(config=sft_config, use_wandb=False)

    if logger:


        logger.info(f"Using {len(sft_train_data_cot['prompts'])} high-quality CoT examples")
    if logger:

        logger.info(f"Note: Checkpoints will be auto-saved every {sft_config['logging']['save_interval']} steps")
    if logger:

        logger.info("="*60 + "\n")

    # Train the model
    sft_train_data_cot["debug_boundary"] = True
    sft_results = sft.train(
        train_data=sft_train_data_cot,
        val_data=sft_val_data,
        num_episodes=sft_config['training']['num_epochs']
    )

    if logger:


        logger.info("\n" + "="*60)
    if logger:

        logger.info("SFT TRAINING COMPLETE")
    if logger:

        logger.info("="*60)
    if logger:

        logger.info(f"Total time: {sft_results['total_time']:.2f} seconds ({sft_results['total_time']/60:.2f} minutes)")
    if logger:

        logger.info(f"Final loss: {sft_results['final_loss']:.4f}")

    # Save SFT checkpoint
    sft_checkpoint_dir = Path("checkpoints/pipeline_stages/01_after_sft")
    sft_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sft_checkpoint_path = sft_checkpoint_dir / "sft_complete.pt"

    sft.save_checkpoint(str(sft_checkpoint_path))

    if os.path.exists(sft_checkpoint_path):
        file_size = os.path.getsize(sft_checkpoint_path) / (1024**3)
        if logger:

            logger.info(f"✓ Saved SFT checkpoint: {sft_checkpoint_path}")
        if logger:

            logger.info(f"  Checkpoint size: {file_size:.2f} GB")

    return sft, sft_results, sft_checkpoint_path


def visualize_grpo_results(grpo_training_metrics, grpo_validation_metrics, grpo, logger):
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
    if logger:

        logger.info(f"✓ Saved plot to {output_dir / 'grpo_training_metrics.png'}")

    plt.show()

    # Print summary
    if logger:

        logger.info("\n" + "="*60)
    if logger:

        logger.info("GRPO TRAINING SUMMARY")
    if logger:

        logger.info("="*60)
    if logger:

        logger.info(f"Final Total Loss: {grpo_training_metrics['total_loss'][-1]:.4f}")
    if logger:

        logger.info(f"Final KL Divergence: {grpo_training_metrics['kl_divergence'][-1]:.4f}")
    if logger:

        logger.info(f"Final Mean Reward: {grpo_training_metrics['reward_mean'][-1]:.3f}")
    if logger:

        logger.info(f"Average Reward (last 5 episodes): {np.mean(grpo_training_metrics['reward_mean'][-5:]):.3f}")
    if logger:

        logger.info(f"Total Tokens Processed: {grpo_training_metrics['total_tokens'][-1]:,}")
    if logger:

        logger.info(f"Average Processing Speed: {np.mean(grpo_training_metrics['tokens_per_second']):.1f} tokens/second")

    # Gradient statistics
    if logger:

        logger.info("\n" + "="*60)
    if logger:

        logger.info("GRADIENT NORM STATISTICS")
    if logger:

        logger.info("="*60)
    if logger:

        logger.info(f"Mean Gradient Norm: {np.mean(grpo_training_metrics['grad_norm']):.3f}")
    if logger:

        logger.info(f"Max Gradient Norm: {np.max(grpo_training_metrics['grad_norm']):.3f}")
    if logger:

        logger.info(f"Gradient Clip Threshold: {grpo.gradient_clip}")
    grad_norms = np.array(grpo_training_metrics['grad_norm'])
    clipped_fraction = (grad_norms > grpo.gradient_clip).sum() / len(grad_norms) * 100
    if logger:

        logger.info(f"Gradients Clipped: {clipped_fraction:.1f}% of updates")


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    # --------------------------------------------------------
    # Initialize Logging (console + file)
    # --------------------------------------------------------
    from datetime import datetime
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"training_{timestamp}.log"

    logger = setup_logging(level="INFO", log_file=str(log_path))

    # Logger already outputs to both console and file automatically
    logger.info("="*60)
    logger.info("Full SFT + GRPO Training Pipeline")
    logger.info("="*60)
    logger.info(f"📋 Log file: {log_path}")
    logger.info("="*60)

    # Setup
    device = setup_device(logger)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"System Prompt: {SYSTEM_PROMPT.strip()}")

    # --------------------------------------------------------
    # 1. Load and Prepare Dataset (with caching for deterministic loading)
    # --------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("Loading and Preparing Dataset (with caching)")
    logger.info("="*60)

    # Define splits (used for cache key generation)
    gsm8k_splits = {
        "sft_train": "train[:500]",
        "grpo_train": "train[500:]",
        "val": "test[200:1200]",
        "test": "test[:200]"
    }

    # Try to load from cache first
    logger.info("\n🔍 Checking for cached GSM8K dataset...")
    cached_data = load_gsm8k_cache(
        cache_dir=DATASET_CACHE_DIR,
        splits=gsm8k_splits,
        system_prompt=SYSTEM_PROMPT
    )

    if cached_data is not None:
        # Load from cache (fast, deterministic)
        logger.info("✓ Loading dataset from cache (deterministic)")
        (sft_train_data, sft_val_data, sft_test_data,
         grpo_train_prompts, grpo_train_answers,
         grpo_val_prompts, grpo_val_answers,
         grpo_test_prompts, grpo_test_answers) = cached_data
    else:
        # Load from HuggingFace and process (slow, first time only)
        logger.info("⏳ Cache not found, loading from HuggingFace and processing...")

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
    logger.info("\n🔍 Checking for cached CoT dataset...")
    cot_num_examples = 500
    sft_train_data_cot = load_cot_cache(
        cache_dir=DATASET_CACHE_DIR,
        system_prompt=SYSTEM_PROMPT,
        num_examples=cot_num_examples
    )

    if sft_train_data_cot is None:
        # Load from source and cache
        logger.info("⏳ CoT cache not found, downloading and processing...")
        sft_train_data_cot = prepare_cot_dataset_for_sft(SYSTEM_PROMPT, num_examples=cot_num_examples, logger=logger)

        # Save to cache
        save_cot_cache(
            cache_dir=DATASET_CACHE_DIR,
            system_prompt=SYSTEM_PROMPT,
            num_examples=cot_num_examples,
            sft_train_data_cot=sft_train_data_cot
        )
    else:
        logger.info(f"✓ Loaded {len(sft_train_data_cot['prompts'])} CoT examples from cache")

    logger.info(f"\n{'='*60}")
    logger.info("Dataset prepared for full pipeline (NO OVERLAP)")
    logger.info(f"{'='*60}")
    logger.info(f"SFT Training samples: {len(sft_train_data_cot['prompts'])} (high-quality CoT)")
    logger.info(f"GRPO Training samples: {len(grpo_train_prompts)}")
    logger.info(f"Validation samples: {len(sft_val_data['prompts'])}")
    logger.info(f"Test samples: {len(sft_test_data['prompts'])}")

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
            sft_config, sft_train_data_cot, sft_val_data, logger
        )
    else:
        logger.info(f"\nSkipping SFT training, loading from checkpoint: {SFT_CHECKPOINT_PATH}")
        sft_checkpoint_path = SFT_CHECKPOINT_PATH

    # --------------------------------------------------------
    # 3. GRPO Configuration - Updated with working settings from overfit test
    # --------------------------------------------------------
    grpo_config = {
        "algorithm": {
            "name": "grpo",
            "group_size": 16,  # INCREASED from 8 → 16 for more advantage dynamic range
            "kl_coef": 0.08,  # INCREASED from 0.01 → 0.08 to prevent policy drift
            "clip_epsilon": 0.2,
            "normalize_rewards": True,
            "store_completions": False,
        },
        "training": {
            "batch_size": 32,  # Keep at 16 for full training (not 4 like overfit)
            "rollout_batch_size": 8,
            "gradient_clip": 0.1,  # TIGHTENED from 1.0 → 0.1 for maximum stability
            "max_new_tokens": 400,
            "min_new_tokens": 50,  # LOWERED from 150 → 50 to allow </answer> early stopping
            "temperature": 0.6,
            "num_episodes": 500,
            "minibatch_size": 64,  # Keep at 32 for full training
            "update_epochs": 1,
            "top_p": 0.9,
            "entropy_coef": 0.005,  # Increased from 0.002 → 0.005 for more exploration
            "policy_loss_type": "sequence",  # CHANGED from "token" → "sequence" for better gradients
            "resample_batch_per_episode": True,  # ← CRITICAL: Set to True to disable fixed batch!
            # Clipping parameters (all validated in overfit test)
            "kl_estimator": "k3",
            "kl_clamp_min": -2.0,
            "kl_clamp_max": 2.0,
            "kl_reduction": "mean",
            "policy_log_ratio_clamp_min": -2.0,
            "policy_log_ratio_clamp_max": 2.0,
            "advantage_clip_min": -2.0,  # TIGHTENED from -3.0 → -2.0 (effective constraint)
            "advantage_clip_max": 2.0,  # TIGHTENED from 3.0 → 2.0 (effective constraint)
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
            "lr": 5e-6,  # INCREASED from 1e-6 → 5e-6 (will use warmup when resuming)
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "fused": False,
            # Warmup parameters (for fresh start)
            "warmup_steps": 30,  # Increased from 10 → 30
            "warmup_start_lr": 1e-9,  # Decreased from 1e-8 → 1e-9
            "warmup_type": "linear",
            # Resume-specific warmup (when LR changes between checkpoint and config)
            "resume_warmup_steps": 15,  # Warmup steps when resuming with higher LR
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

    logger.info("\n" + "="*70)
    logger.info("GRPO Configuration - STABILITY FIXES APPLIED")
    logger.info("="*70)
    logger.info("\n🔧 STABILITY FIXES (to prevent training collapse):")
    logger.info("  ✅ KL coefficient:     0.01 → 0.08 (8x stronger - prevents policy drift)")
    logger.info("  ✅ Learning rate:      1e-6 → 5e-6 (5x higher, with warmup on resume)")
    logger.info("  ✅ Group size:         8 → 16 (more advantage dynamic range)")
    logger.info("  ✅ Advantage clamps:   ±3.0 → ±2.0 (tighter, effective constraint)")
    logger.info("  ✅ Policy loss:        token → sequence (better gradient magnitude)")
    logger.info("\n📊 MAINTAINED FROM PREVIOUS CONFIG:")
    logger.info("  • Gradient clip:       0.1 (tight for stability)")
    logger.info("  • Batch size:          16 (good data coverage)")
    logger.info("  • Minibatch size:      32 (efficient training)")
    logger.info("  • Num episodes:        500 (full training run)")
    logger.info("  • Min new tokens:      50 (allow </answer> early stopping)")
    logger.info("  • Entropy coef:        0.005 (exploration)")
    logger.info("\n🔄 RESUME BEHAVIOR:")
    logger.info("  • If resuming with higher LR: 15-step warmup from checkpoint LR → config LR")
    logger.info("  • If resuming with same/lower LR: No warmup, use config LR immediately")
    logger.info("  • Past warmup phase: Scheduler disabled, fixed LR used")
    logger.info("\n⚠️  STOPPING BEHAVIOR:")
    logger.info("  • min_new_tokens=50: Forces at least 50 tokens before stopping")
    logger.info("  • stop_sequences=['</answer>']: Stops when </answer> appears")
    logger.info("  • Result: Model generates 50-400 tokens, stops at </answer> if present")
    logger.info("\n🧹 MEMORY MANAGEMENT:")
    logger.info("  • Cache clearing:      ENABLED on MPS (M4 unified memory)")
    logger.info("  • Garbage collection:  Forced every episode")
    logger.info("  • Timing data:         Auto-reset every 10 episodes")
    logger.info("="*70)

    # --------------------------------------------------------
    # 4. Initialize GRPO with Checkpoint (SFT or Resume)
    # --------------------------------------------------------
    if CONTINUE_FROM > 0:
        # Resume from GRPO checkpoint
        logger.info("\n" + "="*60)
        logger.info(f"RESUMING FROM GRPO CHECKPOINT (Episode {CONTINUE_FROM})")
        logger.info("="*60)

        grpo_checkpoint_path = Path(f"checkpoints/grpo_qwen_math/checkpoint_episode_{CONTINUE_FROM}.pt")
        logger.info(f"Checkpoint path: {grpo_checkpoint_path}")

        # Load GRPO checkpoint
        grpo_checkpoint = torch.load(grpo_checkpoint_path, map_location=device)

        logger.info("\n" + "="*60)
        logger.info("CHECKPOINT CONTENTS:")
        logger.info("="*60)
        logger.info(f"Keys in checkpoint: {list(grpo_checkpoint.keys())}")
        logger.info(f"Episode: {grpo_checkpoint.get('episode')}")
        logger.info(f"Total steps: {grpo_checkpoint.get('total_steps')}")
        logger.info(f"Current episode: {grpo_checkpoint.get('current_episode')}")

        # Show optimizer config from checkpoint
        opt_state = grpo_checkpoint["optimizer_state_dict"]
        logger.info(f"\nOptimizer state keys: {list(opt_state.keys())}")
        if "param_groups" in opt_state:
            pg = opt_state["param_groups"][0]
            logger.info(f"Optimizer param_group[0]:")
            for key in ['lr', 'betas', 'eps', 'weight_decay']:
                if key in pg:
                    logger.info(f"  {key}: {pg[key]}")

        # Show scheduler state from checkpoint
        sched_state = grpo_checkpoint.get("scheduler_state_dict")
        logger.info(f"\nScheduler state: {sched_state}")

        # Show optimizer config from checkpoint's config
        checkpoint_config = grpo_checkpoint.get("config", {})
        checkpoint_opt_config = checkpoint_config.get("optimizer", {})
        logger.info(f"\nOptimizer config (from checkpoint):")
        logger.info(f"  lr: {checkpoint_opt_config.get('lr')}")
        logger.info(f"  warmup_steps: {checkpoint_opt_config.get('warmup_steps')}")
        logger.info(f"  warmup_start_lr: {checkpoint_opt_config.get('warmup_start_lr')}")
        logger.info("="*60)

        model_name = checkpoint_config["model"]["model_name"]
        logger.info(f"\nModel: {model_name}")
        logger.info(f"Resuming from episode: {grpo_checkpoint['episode']}")

        # Load tokenizer only (lightweight, no weights)
        logger.info("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Tokenizer loaded")

        # Load model architecture without pretrained weights (much faster!)
        logger.info("\nLoading model architecture from config...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to(dtype=torch.float32)
        logger.info("✓ Model architecture loaded (no pretrained weights downloaded)")

        # Load trained weights from GRPO checkpoint
        # GRPO wraps model in LanguageModel class, so keys have "model." prefix that needs to be removed
        logger.info("\nLoading trained weights from checkpoint...")
        policy_state = grpo_checkpoint["policy_state_dict"]

        # Strip "model." prefix from keys (e.g., "model.model.embed_tokens.weight" -> "model.embed_tokens.weight")
        model_state = {k.replace("model.", "", 1): v for k, v in policy_state.items() if k.startswith("model.")}

        model.load_state_dict(model_state)
        logger.info("✓ Checkpoint weights loaded successfully")

        # Initialize GRPO
        logger.info("\nInitializing GRPO and restoring optimizer/scheduler state...")
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Restore optimizer state (momentum, variance, etc.)
        grpo.optimizer.load_state_dict(grpo_checkpoint["optimizer_state_dict"])

        # Get LR values for smart warmup decision
        checkpoint_lr = grpo.optimizer.param_groups[0]['lr']
        config_lr = grpo_config["optimizer"]["lr"]
        total_steps = grpo_checkpoint.get("total_steps", 0)
        initial_warmup_steps = grpo_config["optimizer"]["warmup_steps"]

        logger.info(f"  ✓ Optimizer state restored")
        logger.info(f"  Checkpoint LR: {checkpoint_lr:.2e}, Config LR: {config_lr:.2e}")

        # Smart LR handling based on checkpoint vs config
        lr_increase_ratio = config_lr / checkpoint_lr if checkpoint_lr > 0 else 1.0

        if config_lr > checkpoint_lr and lr_increase_ratio >= 1.5:
            # Significant LR increase - use warmup to prevent instability
            resume_warmup_steps = grpo_config["optimizer"].get("resume_warmup_steps", 15)
            logger.info(f"  LR increasing {lr_increase_ratio:.1f}x - applying {resume_warmup_steps}-step warmup: {checkpoint_lr:.2e} → {config_lr:.2e}")

            # Create a simple linear warmup scheduler
            from torch.optim.lr_scheduler import LambdaLR

            def warmup_lambda(step):
                # Linear warmup from checkpoint_lr to config_lr
                if step < resume_warmup_steps:
                    alpha = step / resume_warmup_steps
                    target_lr = checkpoint_lr + alpha * (config_lr - checkpoint_lr)
                    return target_lr / config_lr  # LambdaLR multiplies by base_lr
                else:
                    return 1.0  # Use config_lr

            # Set base LR to config_lr
            for param_group in grpo.optimizer.param_groups:
                param_group['lr'] = config_lr

            # Create warmup scheduler
            grpo.lr_scheduler = LambdaLR(grpo.optimizer, lr_lambda=warmup_lambda)

            # Start at step 0 of warmup
            grpo.optimizer.param_groups[0]['lr'] = checkpoint_lr

        elif config_lr != checkpoint_lr:
            # Small change or decrease - apply immediately, disable scheduler
            for param_group in grpo.optimizer.param_groups:
                param_group['lr'] = config_lr
            grpo.lr_scheduler = None
            logger.info(f"  ✓ LR set to {config_lr:.2e} (scheduler disabled)")

        else:
            # Same LR - check if we need to continue warmup or disable scheduler
            if total_steps >= initial_warmup_steps:
                grpo.lr_scheduler = None
                logger.info(f"  ✓ Scheduler disabled (past warmup), LR = {config_lr:.2e}")
            else:
                logger.info(f"  Still in warmup phase, advancing scheduler to step {total_steps}...")
                for _ in range(total_steps):
                    grpo.lr_scheduler.step()
                final_lr = grpo.optimizer.param_groups[0]['lr']
                logger.info(f"  ✓ Scheduler advanced, current LR: {final_lr:.2e}")

        # CRITICAL: Restore reference policy (the original SFT model, not a copy of trained policy!)
        if "ref_policy_state_dict" in grpo_checkpoint and grpo_checkpoint["ref_policy_state_dict"] is not None:
            grpo.ref_policy.load_state_dict(grpo_checkpoint["ref_policy_state_dict"])
            logger.info(f"  ✓ Reference policy restored (KL constraint preserved)")
        else:
            logger.info(f"  ⚠️  WARNING: No reference policy in checkpoint - KL divergence will be incorrect!")

        # Restore total_steps for proper tracking
        if "total_steps" in grpo_checkpoint:
            grpo.total_steps = grpo_checkpoint["total_steps"]
            logger.info(f"  ✓ Training steps restored: {grpo.total_steps}")

        # Restore episode counter
        # Checkpoint contains the LAST completed episode, so resume at NEXT episode
        # E.g., checkpoint_episode_45.pt has episode=44, so we start at 45
        last_completed_episode = grpo_checkpoint["episode"]
        grpo.current_episode = last_completed_episode + 1

        logger.info(f"  Last completed episode in checkpoint: {last_completed_episode}")
        logger.info(f"  Will resume training from episode: {grpo.current_episode}")

        logger.info("\n" + "="*60)
        logger.info(f"✓ GRPO successfully resumed from episode {CONTINUE_FROM}")
        logger.info("="*60)
    else:
        # Start from SFT checkpoint or base model
        logger.info("\n" + "="*60)
        logger.info("Loading SFT checkpoint and initializing GRPO")
        logger.info("="*60)
        logger.info(f"Checkpoint path: {sft_checkpoint_path}")

        # Load SFT checkpoint
        sft_checkpoint = torch.load(sft_checkpoint_path, map_location=device)
        model_name = sft_checkpoint["config"]["model"]["model_name"]
        logger.info(f"Model: {model_name}")

        # Load fresh model and tokenizer
        logger.info("\nLoading fresh model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load SFT weights
        logger.info("Loading SFT weights into model...")
        model.load_state_dict(sft_checkpoint["model_state_dict"])
        logger.info("✓ SFT weights loaded successfully")

        # Initialize GRPO
        logger.info("\nInitializing GRPO with SFT-trained model...")
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        logger.info("\n" + "="*60)
        logger.info("✓ GRPO successfully initialized with SFT-trained model")
        logger.info("="*60)

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

    logger.info("\n" + "="*60)
    logger.info("STARTING GRPO TRAINING")
    logger.info("="*60)

    grpo_results = grpo.train(
        train_data=grpo_train_data,
        val_data=grpo_val_data,
        num_episodes=grpo_config['training']['num_episodes']
    )

    logger.info("\n" + "="*60)
    logger.info("GRPO TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total time: {grpo_results['total_time']:.2f} seconds ({grpo_results['total_time']/60:.2f} minutes)")
    logger.info(f"Total tokens processed: {grpo_results['training_metrics']['total_tokens'][-1]:,}")
    logger.info(f"Final reward: {grpo_results['final_reward']:.3f}")

    # --------------------------------------------------------
    # 6. Evaluation
    # --------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("GRPO MODEL EVALUATION")
    logger.info("="*60)

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
    logger.info("\n" + "="*60)
    logger.info("Creating Visualizations")
    logger.info("="*60)

    visualize_grpo_results(
        grpo_results['training_metrics'],
        grpo_results['validation_metrics'],
        grpo,
        logger
    )

    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)

    # Final message
    logger.info(f"✓ Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
