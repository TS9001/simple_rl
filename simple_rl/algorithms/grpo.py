"""
GRPO (Group Relative Policy Optimization) implementation.

GRPO normalizes rewards within groups/batches to reduce variance
in policy gradient estimation.
"""

import copy
import math
import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sympy import N
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from simple_rl.algorithms.base import BaseAlgorithm
from simple_rl.utils.huggingface_wrappers import LanguageModel
from simple_rl.utils.device import get_target_device, clear_device_cache
from simple_rl.utils.amp import create_amp_config
from simple_rl.utils.optimization import configure_optimizer
from simple_rl.utils.timing import TimingManager
from simple_rl.utils.training_config import create_training_config
from simple_rl.utils.math_ops import compute_advantages, compute_policy_gradient_loss, compute_total_loss, normalize_rewards
from simple_rl.utils.checkpointing import save_checkpoint, load_checkpoint
from simple_rl.utils.logging_utils import create_logger


class GRPO(BaseAlgorithm):
    """
    Group Relative Policy Optimization algorithm.

    Key features:
    - Generates multiple completions per prompt
    - Normalizes rewards within groups
    - Uses KL divergence penalty for stability
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        batch_reward_fn: Optional[Callable] | None = None,
        use_wandb: bool = False,
    ):
        """
        Initialize GRPO algorithm.

        Args:
            config: Configuration dictionary
            batch_reward_fn: Function to compute rewards (prompt, completion, answer) -> float
            use_wandb: Whether to use Weights & Biases logging
        """
        # Store config and setup device
        self.config = config or {}
        self.use_wandb = use_wandb

        # Determine target device
        device_config = self.config.get("device", None)
        self.device = get_target_device(device_config)

        # Initialize or create model

        if not config:
            raise ValueError("Either model or config must be provided")

        self.policy = LanguageModel(config)

        # Move policy to device
        self.policy = self.policy.to(self.device)

        # Create reference model (frozen copy for KL divergence)
        self.ref_policy = copy.deepcopy(self.policy)
        self.ref_policy = self.ref_policy.to(self.device)

        # Freeze ALL parameters
        for name, param in self.ref_policy.named_parameters():
            param.requires_grad = False

        # Put in eval mode (disables dropout, batchnorm updates, etc.)
        self.ref_policy.eval()

        # Verify freezing worked
        total_params = sum(1 for _ in self.ref_policy.parameters())
        frozen_params = sum(
            1 for p in self.ref_policy.parameters() if not p.requires_grad
        )
        assert (
            frozen_params == total_params
        ), f"Not all parameters frozen: {frozen_params}/{total_params}"
        print(f"✓ Reference model frozen: {frozen_params} parameters")

        # Use training config utility
        self.training_config = create_training_config(self.config)

        # GRPO-specific parameters (from training config)
        self.group_size = self.training_config.group_size
        self.kl_coef = self.training_config.kl_coef
        self.normalize_rewards = self.training_config.normalize_rewards
        self.clip_epsilon = self.training_config.clip_epsilon
        self.store_completions = self.training_config.store_completions

        # Training parameters (from training config)
        self.learning_rate = self.training_config.learning_rate
        self.batch_size = self.training_config.batch_size
        self.minibatch_size = self.training_config.minibatch_size
        self.max_new_tokens = self.training_config.max_new_tokens
        self.temperature = self.training_config.temperature
        self.top_k = self.training_config.top_k
        self.top_p = self.training_config.top_p
        self.gradient_clip = self.training_config.gradient_clip

        # Set up AMP configuration
        self._amp_config = create_amp_config(self.config)

        # Set up training components using utilities
        self._configure_training_components()

        # Reward function
        self.batch_reward_fn = batch_reward_fn
        self.old_log_probs = None

        # Initialize logger
        self.logger = create_logger(self.config)
        if self.use_wandb:
            self.logger.init_wandb()

        # Training statistics
        self.total_steps = 0
        self.episode = 0

        # Timing infrastructure
        self.timing_manager = TimingManager()

    def _configure_training_components(self) -> None:
        """Initialize optimizer, AMP scaler, and autocast context."""
        # Configure optimizer using utility
        self.optimizer = configure_optimizer(self.policy, self.config)

    def _start_timer(self, operation_name: str):
        """Start timing an operation."""
        self.timing_manager.start_timer(operation_name)

    def _end_timer(self, operation_name: str) -> float:
        """End timing an operation and return elapsed time."""
        return self.timing_manager.end_timer(operation_name)

    def _print_timing_summary(self, title: str = "GRPO Operation Timings"):
        """Print a technical timing summary with detailed statistics."""
        self.timing_manager.print_timing_summary(title)

    def reset_timings(self):
        """Reset all timing data."""
        self.timing_manager.reset_timings()

    def _create_completion_mask(self, completion_ids):
        """
        Creates a mask for completion tokens that excludes tokens after the EOS token.

        Args:
            completion_ids: Token IDs of the generated completions [batch_size, seq_len]

        Returns:
            Binary mask with 1s for valid tokens and 0s after the EOS token
        """
        eos_token_id = self.policy.tokenizer.eos_token_id
        is_eos = completion_ids == eos_token_id

        # Find the index of the first EOS token in each sequence
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )

        # Update indices where EOS tokens exist
        mask_exists = is_eos.any(dim=1)
        eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]

        # Create sequence indices
        sequence_indices = torch.arange(
            is_eos.size(1), device=completion_ids.device
        ).expand(is_eos.size(0), -1)

        # Create mask: 1 for positions <= first EOS, 0 for positions after EOS
        return (sequence_indices <= eos_idx.unsqueeze(1)).float()

    def _generate_grouped_completions(
        self, prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Generate interleaved completions for a batch of prompts."""

        self._start_timer("batch_tokenization")
        tokenized = self.policy.tokenize(prompts, padding_side="left")
        batch_prompt_ids = tokenized["input_ids"].to(self.device)
        batch_prompt_mask = tokenized["attention_mask"].to(self.device)
        self._end_timer("batch_tokenization")

        self._start_timer("prompt_replication")
        num_prompts = len(prompts)
        total_sequences = num_prompts * self.group_size

        replicated_prompt_ids = batch_prompt_ids.repeat_interleave(
            self.group_size, dim=0
        )
        replicated_prompt_mask = batch_prompt_mask.repeat_interleave(
            self.group_size, dim=0
        )
        self._end_timer("prompt_replication")

        self._start_timer("batch_text_generation")
        prev = self.policy.training
        self.policy.eval()
        with torch.no_grad():
            generated_ids, generated_mask = self.policy.generate(
                replicated_prompt_ids,
                attention_mask=replicated_prompt_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_return_sequences=1,
                min_new_tokens=1,
            )
        self.policy.train(prev)
        self._end_timer("batch_text_generation")

        self._start_timer("completion_extraction")
        prompt_lengths = batch_prompt_mask.sum(dim=-1)
        prompt_start_positions = torch.argmax(batch_prompt_mask, dim=-1)
        prompt_end_positions = prompt_start_positions + prompt_lengths
        replicated_prompt_end_positions = prompt_end_positions.repeat_interleave(
            self.group_size, dim=0
        )

        all_completion_ids = []
        for seq_idx in range(total_sequences):
            prompt_end = int(replicated_prompt_end_positions[seq_idx])
            seq_completion_ids = generated_ids[seq_idx, prompt_end:]
            all_completion_ids.append(seq_completion_ids)

        max_completion_length = (
            max(comp_ids.size(0) for comp_ids in all_completion_ids)
            if all_completion_ids
            else 0
        )
        completion_ids = torch.stack(
            [
                torch.nn.functional.pad(
                    comp_ids,
                    (0, max_completion_length - comp_ids.size(0)),
                    value=self.policy.tokenizer.pad_token_id,
                )
                for comp_ids in all_completion_ids
            ]
        )
        completion_texts = self.policy.decode(completion_ids)
        self._end_timer("completion_extraction")

        return {
            "generated_ids": generated_ids,
            "generated_mask": generated_mask,
            "completion_ids": completion_ids,
            "completion_texts": completion_texts,
            "all_completion_ids": all_completion_ids,
            "prompt_end_positions": replicated_prompt_end_positions,
            "total_sequences": total_sequences,
        }

    def generate_trajectories(
        self,
        prompts: List[str],
        answers: Optional[List[str]] = None,
        use_formatting: bool = True,
        store_outputs: Optional[bool] = None,
    ) -> Tuple[
        Optional[List[str]],
        Optional[List[str]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generate trajectories for a batch of prompts.

        Args:
            prompts: List of prompt strings
            answers: Optional list of correct answers for reward computation
            use_formatting: Whether to apply prompt formatting
            store_outputs: Override flag controlling whether decoded prompts/completions are returned

        Returns:
            Tuple of (prompts, completions, rewards, log_probs, ref_log_probs, completion_mask)
        """
        store = self.store_completions if store_outputs is None else store_outputs
        all_prompts = [] if store else None
        all_completions = [] if store else None
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []
        all_completion_mask = []

        generation = self._generate_grouped_completions(prompts)
        generated_ids = generation["generated_ids"]
        generated_mask = generation["generated_mask"]
        completion_ids = generation["completion_ids"]
        completion_texts = generation["completion_texts"]
        all_completion_ids = generation["all_completion_ids"]
        replicated_prompt_end_positions = generation["prompt_end_positions"]
        total_sequences = generation["total_sequences"]

        # Compute log probabilities for policy (batched)
        self._start_timer("batch_policy_log_probs")
        prev_mode = self.policy.training
        self.policy.eval()
        with torch.enable_grad():
            prev_cache = getattr(self.policy.model.config, "use_cache", None)
            if prev_cache is not None:
                self.policy.model.config.use_cache = False
            try:
                policy_log_probs = self.policy.compute_log_probs(
                    generated_ids,
                    attention_mask=generated_mask,
                    target_mask=None,
                )
            finally:
                if prev_cache is not None:
                    self.policy.model.config.use_cache = prev_cache
        if prev_mode:
            self.policy.train()
        self._end_timer("batch_policy_log_probs")

        # Compute log probabilities for reference model (batched)
        self._start_timer("batch_ref_log_probs")
        with torch.no_grad():
            prev_cache = getattr(self.ref_policy.model.config, "use_cache", None)
            if prev_cache is not None:
                self.ref_policy.model.config.use_cache = False
            try:
                ref_log_probs = self.ref_policy.compute_log_probs(
                    generated_ids, attention_mask=generated_mask, target_mask=None
                )
            finally:
                if prev_cache is not None:
                    self.ref_policy.model.config.use_cache = prev_cache
        self._end_timer("batch_ref_log_probs")

        # Create proper completion masks and extract completion log probs
        self._start_timer("mask_and_extract_completions")

        # Create EOS-aware completion mask
        completion_mask = self._create_completion_mask(completion_ids)

        # Extract completion log probs handling variable prompt lengths
        for seq_idx in range(total_sequences):
            # Get the actual completion length for this sequence
            seq_completion_ids = all_completion_ids[seq_idx]
            actual_completion_length = seq_completion_ids.size(0)

            prompt_end = int(replicated_prompt_end_positions[seq_idx])

            # Extract log probs for this sequence's completion (matching actual length)
            completion_start = max(prompt_end - 1, 0)  # Account for shift by 1
            completion_end = completion_start + actual_completion_length

            seq_policy_log_probs = policy_log_probs[
                seq_idx, completion_start:completion_end
            ]
            seq_ref_log_probs = ref_log_probs[seq_idx, completion_start:completion_end]

            # Get completion mask for the actual completion length
            seq_completion_mask = completion_mask[seq_idx, :actual_completion_length]

            # Ensure all tensors have the same length
            min_length = min(
                len(seq_policy_log_probs),
                len(seq_ref_log_probs),
                len(seq_completion_mask)
            )

            seq_policy_log_probs = seq_policy_log_probs[:min_length]
            seq_ref_log_probs = seq_ref_log_probs[:min_length]
            seq_completion_mask = seq_completion_mask[:min_length]

            # Apply masking
            seq_policy_log_probs = seq_policy_log_probs * seq_completion_mask
            seq_ref_log_probs = seq_ref_log_probs * seq_completion_mask

            all_log_probs.append(seq_policy_log_probs)
            all_ref_log_probs.append(seq_ref_log_probs)
            all_completion_mask.append(seq_completion_mask)
        self._end_timer("mask_and_extract_completions")

        # Batch compute rewards by group
        self._start_timer("batch_reward_computation")
        for prompt_idx, (prompt, answer) in enumerate(zip(prompts, answers)):
            # Get completions for this prompt (group_size consecutive sequences)
            start_idx = prompt_idx * self.group_size
            end_idx = start_idx + self.group_size
            group_completions = completion_texts[start_idx:end_idx]

            # Compute rewards for this group
            group_rewards = self.batch_reward_fn(
                group_completions, [answer] * self.group_size, self.device
            )
            all_rewards.append(group_rewards)

            # Store prompts and completions if needed
            if store:
                all_prompts.extend([prompt] * self.group_size)
                all_completions.extend(group_completions)
        self._end_timer("batch_reward_computation")

        # Stack into tensors
        self._start_timer("tensor_stacking")
        all_log_probs = pad_sequence(all_log_probs, batch_first=True, padding_value=0.0)
        all_ref_log_probs = pad_sequence(
            all_ref_log_probs, batch_first=True, padding_value=0.0
        )
        all_completion_mask = pad_sequence(
            all_completion_mask, batch_first=True, padding_value=0.0
        )
        all_rewards = torch.cat(all_rewards, dim=0)
        self._end_timer("tensor_stacking")

        if store:
            return (
                all_prompts,
                all_completions,
                all_rewards,
                all_log_probs,
                all_ref_log_probs,
                all_completion_mask,
            )
        return (
            None,
            None,
            all_rewards,
            all_log_probs,
            all_ref_log_probs,
            all_completion_mask,
        )

    def compute_advantages(
        self,
        adjusted_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantages with group normalization and KL penalty.

        Args:
            rewards: Reward values [batch_size]
            log_probs: Policy log probabilities [batch_size, seq_len]
            ref_log_probs: Reference log probabilities [batch_size, seq_len]

        Returns:
            Advantages [batch_size]
        """
        self._start_timer("advantage_computation")
        # Use math operations utility
        advantages = normalize_rewards(
            adjusted_rewards,
            self.group_size,
            self.normalize_rewards
        )
        self._end_timer("advantage_computation")
        return advantages

    def compute_loss(self, log_probs, advantages, ref_log_probs, completion_mask):

        self._start_timer("loss_computation")
        # Use math operations utilities
        advantages, kl_penalty = compute_advantages(
            advantages, log_probs, ref_log_probs, completion_mask,
            self.group_size, self.normalize_rewards
        )

        pg_loss = compute_policy_gradient_loss(log_probs, advantages, completion_mask)
        loss = compute_total_loss(pg_loss, kl_penalty, self.kl_coef)

        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_divergence": kl_penalty.item(),  # Per-token average for monitoring
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "tokens_generated": completion_mask.sum().item(),
        }

        self._end_timer("loss_computation")
        return loss, metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single training step with gradient accumulation.

        Args:
            batch: Dictionary with 'prompts' and optionally 'answers' keys

        Returns:
            Dictionary of training metrics
        """
        self.policy.train()
        prompts = batch["prompts"]
        answers = batch["answers"]  # Optional answers for reward computation

        # Calculate number of accumulation steps (ceil to cover partial minibatches)
        num_accumulation_steps = max(1, math.ceil(len(prompts) / self.minibatch_size))

        # Zero gradients at the start
        self.optimizer.zero_grad()

        # Accumulate metrics across minibatches
        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl = 0.0
        total_reward_mean = 0.0
        total_reward_std = 0.0
        total_tokens = 0

        # Total sequences across full batch (each prompt yields group_size sequences)
        total_sequences = len(prompts) * self.group_size

        # Process minibatches
        autocast_ctx = self._amp_config.autocast

        for i in range(0, len(prompts), self.minibatch_size):
            end_idx = min(i + self.minibatch_size, len(prompts))
            mb_prompts = prompts[i:end_idx]
            mb_answers = answers[i:end_idx] if answers else None

            # Generate trajectories for this minibatch
            self._start_timer("trajectory_generation")
            (
                prompts_out,
                completions_out,
                rewards,
                log_probs,
                ref_log_probs,
                completion_mask,
            ) = self.generate_trajectories(
                mb_prompts, answers=mb_answers, store_outputs=self.store_completions
            )
            self._end_timer("trajectory_generation")

            # Compute advantages
            advantages = self.compute_advantages(rewards)

            # Compute loss under autocast context
            with autocast_ctx:
                loss, mb_metrics = self.compute_loss(
                    log_probs, advantages, ref_log_probs, completion_mask
                )

            # Weight loss by fraction of sequences in this minibatch for proper averaging
            mb_sequences = len(mb_prompts) * self.group_size
            weight = mb_sequences / max(1, total_sequences)
            scaled_loss = loss * weight

            # Backward pass (accumulate gradients)
            self._start_timer("backward_pass")
            if self._amp_config.enabled and self._amp_config.grad_scaler is not None:
                self._amp_config.grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            self._end_timer("backward_pass")

            # Accumulate metrics (use weighted values to match actual training)
            total_loss += scaled_loss.item()
            total_pg_loss += mb_metrics["pg_loss"] * weight
            total_kl += mb_metrics["kl_divergence"] * weight
            total_reward_mean += rewards.mean().item() * weight
            total_reward_std += rewards.std().item() * weight
            total_tokens += mb_metrics.get("tokens_generated", 0)

            # Clean up intermediate tensors to prevent memory buildup
            self._start_timer("memory_cleanup")
            if self.store_completions:
                del prompts_out, completions_out
            del (
                rewards,
                log_probs,
                ref_log_probs,
                completion_mask,
                advantages,
                loss,
                scaled_loss,
            )
            clear_device_cache(self.device)
            self._end_timer("memory_cleanup")

        # Gradient clipping on accumulated gradients
        self._start_timer("gradient_clipping")
        if self._amp_config.enabled and self._amp_config.grad_scaler is not None:
            self._amp_config.grad_scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=self.gradient_clip
        )
        self._end_timer("gradient_clipping")

        # Update parameters
        self._start_timer("parameter_update")
        if self._amp_config.enabled and self._amp_config.grad_scaler is not None:
            self._amp_config.grad_scaler.step(self.optimizer)
            self._amp_config.grad_scaler.update()
        else:
            self.optimizer.step()
        self._end_timer("parameter_update")

        # Update statistics
        self.total_steps += 1

        # Aggregate metrics (we used weighted accumulation, so values are already averaged)
        metrics = {
            "total_loss": total_loss,
            "pg_loss": total_pg_loss,
            "kl_divergence": total_kl,
            "reward_mean": total_reward_mean,
            "reward_std": total_reward_std,
            "tokens_generated": total_tokens,  # Total tokens, not averaged
        }

        # Log to logger
        self.logger.log_metrics(metrics, self.total_steps)

        return metrics

    def train(self, num_episodes: int) -> Dict[str, float]:
        """
        Train for a specified number of episodes.

        Args:
            num_episodes: Number of training episodes

        Returns:
            Dictionary of final metrics
        """
        print(f"Starting GRPO training for {num_episodes} episodes...")

        final_metrics = {}

        for episode in range(num_episodes):
            self.episode = episode

            # Generate random prompts for demo (replace with actual data)
            prompts = [
                f"Question {i}: What is {i}+{i}?" for i in range(self.batch_size)
            ]

            # Training with gradient accumulation
            training_cfg = self.config.get("training", {})
            update_epochs = training_cfg.get("update_epochs", 1)

            total_loss = 0.0
            total_pg_loss = 0.0
            total_kl = 0.0
            total_reward_mean = 0.0
            total_reward_std = 0.0
            total_tokens = 0.0

            for _ in range(update_epochs):
                # Shuffle prompts each epoch
                perm = torch.randperm(len(prompts)).tolist()
                shuffled_prompts = [prompts[idx] for idx in perm]

                # Process entire batch with gradient accumulation
                batch = {"prompts": shuffled_prompts}
                metrics = self.train_step(batch)

                total_loss += metrics["total_loss"]
                total_pg_loss += metrics["pg_loss"]
                total_kl += metrics["kl_divergence"]
                total_reward_mean += metrics["reward_mean"]
                total_reward_std += metrics.get("reward_std", 0.0)
                total_tokens += metrics.get("tokens_generated", 0.0)

            # Average metrics over epochs
            metrics = {
                "total_loss": total_loss / update_epochs,
                "pg_loss": total_pg_loss / update_epochs,
                "kl_divergence": total_kl / update_epochs,
                "reward_mean": total_reward_mean / update_epochs,
                "reward_std": total_reward_std / update_epochs,
                "tokens_generated": total_tokens / update_epochs,
            }

            # Print progress and timing summary
            if episode % max(1, num_episodes // 10) == 0:
                self.logger.print_progress(episode, num_episodes, metrics)

                # Print timing summary every few episodes
                if episode > 0:
                    self._print_timing_summary(f"Episode {episode} Timing Summary")

            final_metrics = metrics

            # Save checkpoint periodically
            if episode % max(1, num_episodes // 5) == 0:
                checkpoint_path = f"checkpoints/grpo_episode_{episode}.pt"
                save_checkpoint(
                    checkpoint_path,
                    self.policy.state_dict(),
                    self.ref_policy.state_dict(),
                    self.optimizer.state_dict(),
                    self.config,
                    self.total_steps,
                    episode,
                )

        # Print final comprehensive timing summary
        print(f"\n[TRAINING] Completed {num_episodes} episodes")
        self._print_timing_summary("FINAL TRAINING PERFORMANCE ANALYSIS")

        return final_metrics

    def evaluate(self, num_episodes: int = 1) -> Dict[str, float]:
        """
        Evaluate the policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()

        total_rewards = []

        with torch.no_grad():
            for _ in range(num_episodes):
                # Generate test prompts (replace with actual eval data)
                prompts = [f"Test {i}: Calculate {i}*2" for i in range(4)]

                # Generate without training
                _, _, rewards, _, _, _ = self.generate_trajectories(prompts)
                total_rewards.extend(rewards.cpu().numpy())

        self.policy.train()

        return {
            "eval_reward_mean": np.mean(total_rewards),
            "eval_reward_std": np.std(total_rewards),
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        save_checkpoint(
            path,
            self.policy.state_dict(),
            self.ref_policy.state_dict(),
            self.optimizer.state_dict(),
            self.config,
            self.total_steps,
            self.episode,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = load_checkpoint(path, self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.ref_policy.load_state_dict(checkpoint["ref_policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.episode = checkpoint.get("episode", 0)
