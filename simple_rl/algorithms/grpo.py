"""
GRPO (Group Relative Policy Optimization) implementation.

GRPO normalizes rewards within groups/batches to reduce variance
in policy gradient estimation.
"""

import copy
import math
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.utils.rnn import pad_sequence

from simple_rl.algorithms.base import BaseAlgorithm
from simple_rl.utils.huggingface_wrappers import LanguageModel


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
        # Prefer CUDA, then MPS, else CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Initialize or create model

        if not config:
            raise ValueError("Either model or config must be provided")

        self.policy = LanguageModel(config)

        # Move policy to device
        self.policy = self.policy.to(self.device)

        # Create reference model (frozen copy for KL divergence)
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

        # GRPO-specific parameters
        algo_config = self.config.get("algorithm", {})
        self.group_size = algo_config.get("group_size", 4)
        self.kl_coef = algo_config.get("kl_coef", 0.05)
        self.normalize_rewards = algo_config.get("normalize_rewards", True)
        self.clip_epsilon = algo_config.get("clip_epsilon", 0.2)
        self.store_completions = algo_config.get("store_completions", True)

        # Training parameters
        training_config = self.config.get("training", {})
        self.learning_rate = training_config.get("learning_rate", 1e-5)
        self.batch_size = training_config.get("batch_size", 8)
        self.minibatch_size = training_config.get("minibatch_size", self.batch_size)
        if self.minibatch_size in (None, 0):
            self.minibatch_size = self.batch_size
        self.max_new_tokens = training_config.get("max_new_tokens", 128)
        self.temperature = training_config.get("temperature", 0.9)
        self.top_k = training_config.get("top_k", None)
        self.top_p = training_config.get("top_p", 0.9)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)

        # Generation prompt configuration
        generation_config = self.config.get("generation", {})
        self.generation_prompt_template = generation_config.get("prompt_template", None)
        self.system_prompt = generation_config.get("system_prompt", None)
        self.response_prefix = generation_config.get("response_prefix", None)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )

        # Reward function
        self.batch_reward_fn = batch_reward_fn
        self.old_log_probs = None

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=self.config.get("project_name", "grpo"),
                config=self.config,
                name=self.config.get("run_name", None),
            )

        # Training statistics
        self.total_steps = 0
        self.episode = 0

        # Timing infrastructure
        self.timings = defaultdict(list)
        self.current_timings = {}
        self.timing_enabled = True

    def set_generation_prompt(
        self,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        response_prefix: Optional[str] = None,
    ):
        """
        Update generation prompt configuration.

        Args:
            system_prompt: System prompt to prepend
            prompt_template: Template with {prompt} placeholder
            response_prefix: Prefix to append after prompt
        """
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if prompt_template is not None:
            self.generation_prompt_template = prompt_template
        if response_prefix is not None:
            self.response_prefix = response_prefix

    def _start_timer(self, operation_name: str):
        """Start timing an operation."""
        if self.timing_enabled:
            self.current_timings[operation_name] = time.perf_counter()

    def _end_timer(self, operation_name: str) -> float:
        """End timing an operation and return elapsed time."""
        if not self.timing_enabled or operation_name not in self.current_timings:
            return 0.0

        elapsed = time.perf_counter() - self.current_timings[operation_name]
        self.timings[operation_name].append(elapsed)
        del self.current_timings[operation_name]
        return elapsed

    def _print_timing_summary(self, title: str = "GRPO Operation Timings"):
        """Print a technical timing summary with detailed statistics."""
        if not self.timings:
            return

        print(f"\n[TIMING] {title}")
        print("-" * 120)

        # Calculate statistics for each operation
        timing_stats = OrderedDict()
        total_time = 0
        total_calls = 0

        for op_name, times in self.timings.items():
            if times:
                mean_time = np.mean(times)
                total_time += np.sum(times)
                total_calls += len(times)
                timing_stats[op_name] = {
                    "mean": mean_time,
                    "median": np.median(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "std": np.std(times),
                    "var": np.var(times),
                    "count": len(times),
                    "total": np.sum(times),
                    "p95": np.percentile(times, 95),
                    "p99": np.percentile(times, 99),
                }

        # Sort by total time (descending)
        timing_stats = OrderedDict(
            sorted(timing_stats.items(), key=lambda x: x[1]["total"], reverse=True)
        )

        # Print header
        print(
            f"{'Operation':<25} {'Total(s)':<9} {'Mean(s)':<9} {'Median(s)':<10} {'Min(s)':<8} {'Max(s)':<8} "
            f"{'StdDev(s)':<10} {'Var(s)':<9} {'P95(s)':<8} {'P99(s)':<8} {'Count':<6} {'%Total':<7}"
        )
        print("-" * 120)

        # Print each operation with detailed statistics
        for op_name, stats in timing_stats.items():
            percentage = (stats["total"] / total_time * 100) if total_time > 0 else 0

            print(
                f"{op_name:<25} {stats['total']:<9.6f} {stats['mean']:<9.6f} {stats['median']:<10.6f} "
                f"{stats['min']:<8.6f} {stats['max']:<8.6f} {stats['std']:<10.6f} {stats['var']:<9.6f} "
                f"{stats['p95']:<8.6f} {stats['p99']:<8.6f} {stats['count']:<6} {percentage:<7.2f}"
            )

        print("-" * 120)
        print(f"SUMMARY: Total execution time: {total_time:.6f}s | Total operations: {total_calls} | Operations tracked: {len(timing_stats)}")

        # Additional technical metrics
        if len(timing_stats) > 0:
            times_per_op = [stats["mean"] for stats in timing_stats.values()]
            print(f"STATS: Mean operation time: {np.mean(times_per_op):.6f}s | "
                  f"Operation time stddev: {np.std(times_per_op):.6f}s | "
                  f"Slowest operation: {max(timing_stats.keys(), key=lambda x: timing_stats[x]['total'])}")
        print()

    def reset_timings(self):
        """Reset all timing data."""
        self.timings.clear()
        self.current_timings.clear()

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

        # Process each prompt to generate completions and collect log probs
        for prompt, answer in zip(prompts, answers):
            self._start_timer("tokenization")
            tokenized = self.policy.tokenize([prompt])
            prompt_ids = tokenized["input_ids"].to(self.device)
            prompt_mask = tokenized["attention_mask"].to(self.device)
            self._end_timer("tokenization")

            # Generate completions
            self._start_timer("text_generation")
            prev = self.policy.training
            self.policy.eval()
            with torch.no_grad():
                generated_ids, generated_mask = self.policy.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    num_return_sequences=self.group_size,
                    min_new_tokens=1,  # also avoids empty completion windows
                )
            self.policy.train(prev)
            self._end_timer("text_generation")

            # Get prompt length to extract completions
            prompt_length = prompt_ids.shape[1]
            generated_length = generated_ids.shape[1] - prompt_length
            # Decode completions
            self._start_timer("completion_decoding")
            completion_ids = generated_ids[:, prompt_length:]
            completions = self.policy.decode(completion_ids)
            self._end_timer("completion_decoding")

            # Compute log probabilities for policy (only for completion tokens)
            self._start_timer("policy_log_probs")
            prev_mode = self.policy.training
            self.policy.eval()
            with torch.enable_grad():  # Ensure gradients are enabled for policy
                prev_cache = getattr(self.policy.model.config, "use_cache", None)
                if prev_cache is not None:
                    self.policy.model.config.use_cache = False
                try:
                    policy_log_probs = self.policy.compute_log_probs(
                        generated_ids,
                        attention_mask=generated_mask,
                        target_mask=None,  # Let the model handle masking internally
                    )
                finally:
                    if prev_cache is not None:
                        self.policy.model.config.use_cache = prev_cache
                # Extract only completion log probs (account for shift by 1)
                policy_log_probs = policy_log_probs[
                    :, prompt_length - 1 : prompt_length - 1 + generated_length
                ]
            if prev_mode:
                self.policy.train()
            self._end_timer("policy_log_probs")

            # Compute log probabilities for reference model
            self._start_timer("ref_log_probs")
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
                # Extract only completion log probs (account for shift by 1)
                ref_log_probs = ref_log_probs[
                    :, prompt_length - 1 : prompt_length - 1 + generated_length
                ]
            self._end_timer("ref_log_probs")
            generated_mask = generated_mask[:, 1:]
            # Create mask for completion tokens only
            completion_mask = generated_mask[
                :, prompt_length - 1 : prompt_length - 1 + generated_length
            ]

            # Store results (except rewards - we'll compute those in batch)
            if store:
                all_prompts.extend([prompt] * self.group_size)
                all_completions.extend(completions)

            # Apply masking to log probs before storing
            policy_log_probs = policy_log_probs * completion_mask
            ref_log_probs = ref_log_probs * completion_mask

            assert policy_log_probs.shape == ref_log_probs.shape
            assert completion_mask.shape == policy_log_probs.shape
            assert policy_log_probs[0].shape[0] == (completion_ids[0].shape[0])

            # Compute rewards for this group of completions
            self._start_timer("reward_computation")
            group_rewards = self.batch_reward_fn(
                completions, [answer] * self.group_size, self.device
            )
            all_rewards.append(group_rewards)  # Append the tensor for this group
            self._end_timer("reward_computation")

            # Store each sample's log probs separately
            for i in range(self.group_size):
                all_log_probs.append(policy_log_probs[i])
                all_ref_log_probs.append(ref_log_probs[i])
                all_completion_mask.append(completion_mask[i])

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
        batch_size = adjusted_rewards.shape[0]

        self._start_timer("advantage_computation")
        # Normalize rewards within groups if configured
        if self.normalize_rewards and self.group_size > 1:
            # Reshape to groups
            num_groups = batch_size // self.group_size
            grouped_rewards = adjusted_rewards.view(num_groups, self.group_size)

            # Normalize within each group
            group_mean = grouped_rewards.mean(dim=1, keepdim=True)
            group_std = grouped_rewards.std(dim=1, keepdim=True)
            normalized_rewards = (grouped_rewards - group_mean) / (group_std + 1e-8)

            # Flatten back
            advantages = normalized_rewards.view(-1)
        else:
            # Global normalization
            advantages = (adjusted_rewards - adjusted_rewards.mean()) / (
                adjusted_rewards.std() + 1e-8
            )

        self._end_timer("advantage_computation")
        return advantages

    def compute_loss(self, log_probs, advantages, ref_log_probs, completion_mask):

        self._start_timer("loss_computation")
        # Get sequence-level log probabilities by summing
        log_probs_sum = log_probs.sum(dim=-1)

        # Policy gradient loss (using summed log probs)
        pg_loss = -(log_probs_sum * advantages.detach()).mean()

        # TRL adds KL at the per-token level, then averages
        delta = log_probs - ref_log_probs
        per_token_kl = torch.exp(delta) - delta - 1.0  # Schulman approx
        kl_penalty = (per_token_kl * completion_mask).sum() / (
            completion_mask.sum() + 1e-8
        )

        # Total loss
        loss = pg_loss + (self.kl_coef * kl_penalty)

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

            # Compute loss
            loss, mb_metrics = self.compute_loss(
                log_probs, advantages, ref_log_probs, completion_mask
            )

            # Weight loss by fraction of sequences in this minibatch for proper averaging
            mb_sequences = len(mb_prompts) * self.group_size
            weight = mb_sequences / max(1, total_sequences)
            scaled_loss = loss * weight

            # Backward pass (accumulate gradients)
            self._start_timer("backward_pass")
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
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._end_timer("memory_cleanup")

        # Gradient clipping on accumulated gradients
        self._start_timer("gradient_clipping")
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=self.gradient_clip
        )
        self._end_timer("gradient_clipping")

        # Update parameters
        self._start_timer("parameter_update")
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

        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(metrics, step=self.total_steps)

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
                print(
                    f"Episode {episode}/{num_episodes} - "
                    f"Loss: {metrics['total_loss']:.4f}, "
                    f"Reward: {metrics['reward_mean']:.4f}, "
                    f"KL: {metrics['kl_divergence']:.4f}"
                )

                # Print timing summary every few episodes
                if episode > 0:
                    self._print_timing_summary(f"Episode {episode} Timing Summary")

            final_metrics = metrics

            # Save checkpoint periodically
            if episode % max(1, num_episodes // 5) == 0:
                checkpoint_path = f"checkpoints/grpo_episode_{episode}.pt"
                self.save_checkpoint(checkpoint_path)

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
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "ref_policy_state_dict": self.ref_policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "total_steps": self.total_steps,
            "episode": self.episode,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.ref_policy.load_state_dict(checkpoint["ref_policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.episode = checkpoint.get("episode", 0)
