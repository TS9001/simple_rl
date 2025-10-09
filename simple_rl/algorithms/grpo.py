"""
GRPO (Group Relative Policy Optimization) - Proper Implementation.

Based on DeepSeekMath paper (arxiv.org/abs/2402.03300).

Key features:
- PPO-style clipped surrogate objective (NOT REINFORCE)
- Multiple epochs of updates on collected trajectories
- Group-based advantage normalization (no value network)
- KL divergence penalty with reference model
- Proper minibatch updates with log prob recomputation
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from simple_rl.algorithms.base import BaseAlgorithm
from simple_rl.utils.huggingface_wrappers import LanguageModel
from simple_rl.utils.device import get_target_device, clear_device_cache
from simple_rl.utils.amp import create_amp_config
from simple_rl.utils.optimization import configure_optimizer
from simple_rl.utils.timing import TimingManager
from simple_rl.utils.training_config import create_training_config
from simple_rl.utils.checkpointing import save_checkpoint, load_checkpoint
from simple_rl.utils.logging_utils import create_logger
from simple_rl.utils.kl_divergence import compute_kl_divergence
from simple_rl.utils.debug_logger import GRPODebugLogger


class GRPO(BaseAlgorithm):
    """
    Proper Group Relative Policy Optimization (GRPO) algorithm.

    Implements PPO-style clipped objective with group-based advantages.
    No value network - baseline comes from group mean rewards.

    Reference: DeepSeekMath paper (https://arxiv.org/abs/2402.03300)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        batch_reward_fn: Optional[Callable] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        use_wandb: bool = False,
    ):
        """
        Initialize GRPO algorithm.

        Args:
            config: Configuration dictionary
            batch_reward_fn: Batch reward function
            model: Optional pre-loaded HuggingFace model
            tokenizer: Optional pre-loaded HuggingFace tokenizer
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config or {}
        self.use_wandb = use_wandb

        # Device setup
        device_config = self.config.get("device", None)
        self.device = get_target_device(device_config)

        # Initialize model
        if not config:
            raise ValueError("Config must be provided")

        self.policy = LanguageModel(config, model=model, tokenizer=tokenizer)
        self.policy = self.policy.to(self.device)

        # Print model information
        model_config = config.get("model", {})
        if model is not None:
            model_name = "pre-loaded model"
        else:
            model_name = model_config.get("hf_model_name") or model_config.get("model_name", "unknown")
        total_params = sum(p.numel() for p in self.policy.parameters())
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"✓ GRPO initialized with model: {model_name}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: {self.device}")

        # Create frozen reference model for KL penalty
        self.ref_policy = copy.deepcopy(self.policy)
        self.ref_policy = self.ref_policy.to(self.device)

        # Freeze reference model
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        self.ref_policy.eval()

        # Verify freezing
        frozen_params = sum(1 for p in self.ref_policy.parameters() if not p.requires_grad)
        total_params = sum(1 for _ in self.ref_policy.parameters())
        assert frozen_params == total_params, "Not all reference parameters frozen"
        print(f"✓ Reference model frozen: {frozen_params} parameters")

        # Print KL divergence configuration (will be set after training_config)
        self._print_kl_config = True

        # Training config
        self.training_config = create_training_config(self.config)

        # GRPO parameters
        self.group_size = self.training_config.group_size
        self.kl_coef = self.training_config.kl_coef
        self.normalize_rewards = self.training_config.normalize_rewards

        # Clipping parameters - support both symmetric and asymmetric clipping
        self.clip_epsilon = self.training_config.clip_epsilon
        self.clip_epsilon_low = self.training_config.clip_epsilon_low
        self.clip_epsilon_high = self.training_config.clip_epsilon_high

        self.store_completions = self.training_config.store_completions
        self.update_epochs = self.training_config.update_epochs

        # KL divergence estimator configuration
        # Options: "mc" (Monte Carlo), "k3" (low-variance unbiased), "abs", "mse"
        self.kl_estimator = self.config.get("training", {}).get("kl_estimator", "k3")
        self.kl_clamp_min = self.config.get("training", {}).get("kl_clamp_min", -5.0)
        self.kl_clamp_max = self.config.get("training", {}).get("kl_clamp_max", 5.0)

        # Print KL configuration
        if self._print_kl_config:
            print(f"✓ KL divergence estimator: {self.kl_estimator}")
            if self.kl_estimator == "k3":
                print(f"  Clamp range: [{self.kl_clamp_min}, {self.kl_clamp_max}]")
            del self._print_kl_config

        # Training parameters
        self.learning_rate = self.training_config.learning_rate
        self.batch_size = self.training_config.batch_size
        self.minibatch_size = self.training_config.minibatch_size
        self.rollout_batch_size = self.training_config.rollout_batch_size
        self.max_new_tokens = self.training_config.max_new_tokens
        self.temperature = self.training_config.temperature
        self.top_k = self.training_config.top_k
        self.top_p = self.training_config.top_p
        self.gradient_clip = self.training_config.gradient_clip

        # AMP setup
        self._amp_config = create_amp_config(self.config)
        try:
            print(f"AMP: enabled={self._amp_config.enabled}, device={self._amp_config.device}")
        except Exception:
            pass

        # Optimizer
        self.optimizer = configure_optimizer(self.policy, self.config)

        # Reward function
        self.batch_reward_fn = batch_reward_fn

        # Logger
        self.logger = create_logger(self.config)
        if self.use_wandb:
            self.logger.init_wandb()

        # Debug logger
        debug_config = self.config.get("debug", {})
        debug_enabled = debug_config.get("enabled", False)
        debug_log_dir = debug_config.get("log_dir", "debug_logs")
        self.debug_logger = GRPODebugLogger(log_dir=debug_log_dir, enabled=debug_enabled)

        # Statistics
        self.total_steps = 0
        self.episode = 0

        # Timing
        self.timing_manager = TimingManager()
        timing_config = self.config.get("timing", {})
        self.print_timing = timing_config.get("enabled", False)  # Disabled by default

        # Storage for trajectory data (needed for multiple epochs)
        self.stored_generated_ids = None
        self.stored_attention_mask = None
        self.stored_prompt_end_positions = None

        # Early stopping tokens for generation
        self._setup_stopping_tokens()

    def _setup_stopping_tokens(self):
        """Set up additional stopping tokens for early termination."""
        # Try to encode </answer> tag as stopping token
        self.answer_end_token_id = None
        try:
            # Encode the closing answer tag
            encoded = self.policy.tokenizer.encode("</answer>", add_special_tokens=False)
            if encoded:
                # Use the last token (most specific)
                self.answer_end_token_id = encoded[-1]
        except Exception:
            # If encoding fails, just use default EOS
            pass

    def reset_timings(self):
        """Reset timing data."""
        self.timing_manager.reset_timings()

    def _print_timing_summary(self, title: str):
        """Print timing summary."""
        self.timing_manager.print_timing_summary(title)

    def _create_completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        """
        Create mask for completion tokens, excluding tokens after EOS or padding.

        Args:
            completion_ids: Token IDs [batch_size, seq_len]

        Returns:
            Binary mask [batch_size, seq_len]
        """
        eos_token_id = self.policy.tokenizer.eos_token_id
        pad_token_id = self.policy.tokenizer.pad_token_id

        # Check for both EOS and PAD tokens
        is_eos = completion_ids == eos_token_id
        is_pad = completion_ids == pad_token_id

        # A position is masked out if it's EOS, PAD, or after the first EOS/PAD
        is_stop = is_eos | is_pad

        # Find first stop position (EOS or PAD)
        stop_idx = torch.full(
            (is_stop.size(0),),
            is_stop.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )

        mask_exists = is_stop.any(dim=1)
        stop_idx[mask_exists] = is_stop.int().argmax(dim=1)[mask_exists]

        # Create mask: 1 for valid tokens (before stop), 0 for stop and after
        sequence_indices = torch.arange(
            is_stop.size(1), device=completion_ids.device
        ).expand(is_stop.size(0), -1)

        return (sequence_indices < stop_idx.unsqueeze(1)).float()

    def _generate_grouped_completions(
        self, prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Generate group_size completions for each prompt."""

        self.timing_manager.start_timer("batch_tokenization")
        tokenized = self.policy.tokenize(prompts, padding_side="left")
        batch_prompt_ids = tokenized["input_ids"].to(self.device)
        batch_prompt_mask = tokenized["attention_mask"].to(self.device)
        self.timing_manager.end_timer("batch_tokenization")

        self.timing_manager.start_timer("prompt_replication")
        num_prompts = len(prompts)
        total_sequences = num_prompts * self.group_size

        # Replicate for group sampling
        replicated_prompt_ids = batch_prompt_ids.repeat_interleave(
            self.group_size, dim=0
        )
        replicated_prompt_mask = batch_prompt_mask.repeat_interleave(
            self.group_size, dim=0
        )
        self.timing_manager.end_timer("prompt_replication")

        self.timing_manager.start_timer("batch_text_generation")
        prev_mode = self.policy.training
        self.policy.eval()

        # Build EOS token list for early stopping
        eos_token_id = self.policy.tokenizer.eos_token_id
        if self.answer_end_token_id is not None:
            # Stop at either EOS or </answer> tag
            eos_token_id = [eos_token_id, self.answer_end_token_id]

        debug_generation = self.config.get("debug", {}).get("generation", False)
        if debug_generation:
            print(f"\n🔍 GENERATION DEBUG:")
            print(f"  AMP enabled: {self._amp_config.enabled}")
            print(f"  AMP device: {self._amp_config.device}")
            print(f"  Temperature: {self.temperature}")
            print(f"  Max new tokens: {self.max_new_tokens}")
            print(f"  EOS token ID: {eos_token_id}")
            print(f"  Device: {self.device}")

        with torch.no_grad():
            with self._amp_config.autocast():
                if debug_generation:
                    if hasattr(self.policy.model, 'dtype'):
                        print(f"  Model dtype: {self.policy.model.dtype}")

                    test_tensor = torch.randn(1, 1, device=self.device)
                    test_result = test_tensor @ test_tensor.T
                    print(f"  Computation dtype (matmul): {test_result.dtype}")

                generated_ids, generated_mask = self.policy.generate(
                replicated_prompt_ids,
                attention_mask=replicated_prompt_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_return_sequences=1,
                min_new_tokens=1,
                eos_token_id=eos_token_id,
                )

        if debug_generation:
            total_tokens_generated = generated_mask.sum().item()
            avg_tokens_per_seq = total_tokens_generated / total_sequences
            print(f"  Total tokens generated: {total_tokens_generated}")
            print(f"  Avg tokens/sequence: {avg_tokens_per_seq:.1f}")

        self.policy.train(prev_mode)
        self.timing_manager.end_timer("batch_text_generation")

        self.timing_manager.start_timer("completion_extraction")
        # Extract completion IDs
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

        max_completion_length = max(
            comp_ids.size(0) for comp_ids in all_completion_ids
        ) if all_completion_ids else 0

        completion_ids = torch.stack([
            F.pad(
                comp_ids,
                (0, max_completion_length - comp_ids.size(0)),
                value=self.policy.tokenizer.pad_token_id,
            )
            for comp_ids in all_completion_ids
        ])

        if debug_generation:
            completion_lengths = [comp_ids.size(0) for comp_ids in all_completion_ids]
            completion_lengths_tensor = torch.tensor(completion_lengths, dtype=torch.float32)
            print(f"  Completion lengths - min: {min(completion_lengths)}, max: {max(completion_lengths)}, "
                  f"mean: {completion_lengths_tensor.mean().item():.1f}, std: {completion_lengths_tensor.std().item():.1f}")

            eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
            num_with_eos = sum(1 for comp_ids in all_completion_ids if any(tok.item() in eos_ids for tok in comp_ids))
            print(f"  Sequences with EOS: {num_with_eos}/{len(all_completion_ids)}")

        completion_texts = self.policy.decode(completion_ids)
        self.timing_manager.end_timer("completion_extraction")

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
        Generate trajectories in batches to avoid OOM.

        Processes prompts in batches of size rollout_batch_size to avoid
        memory issues during generation.

        Returns:
            (prompts, completions, rewards, old_log_probs, ref_log_probs, completion_mask)
        """
        store = self.store_completions if store_outputs is None else store_outputs

        # Shuffle prompts to distribute hard/easy problems across batches
        # This prevents one batch from consistently being slower
        num_prompts = len(prompts)
        shuffle_indices = torch.randperm(num_prompts).tolist()
        prompts = [prompts[i] for i in shuffle_indices]
        if answers is not None:
            answers = [answers[i] for i in shuffle_indices]

        # Storage for aggregated results across batches
        all_prompts = [] if store else None
        all_completions = [] if store else None
        all_rewards = []
        all_format_rewards = []
        all_correctness_rewards = []
        all_log_probs = []
        all_ref_log_probs = []
        all_completion_mask = []

        all_generated_ids = []
        all_generated_mask = []
        all_prompt_end_positions = []

        # Process prompts in batches
        for batch_start in range(0, num_prompts, self.rollout_batch_size):
            batch_idx = batch_start // self.rollout_batch_size + 1
            self.timing_manager.start_timer(f"batch_{batch_idx}")

            batch_end = min(batch_start + self.rollout_batch_size, num_prompts)
            batch_prompts = prompts[batch_start:batch_end]
            batch_answers = answers[batch_start:batch_end] if answers else None

            # Generate for this batch
            self.timing_manager.start_timer(f"batch_{batch_idx}_generation")
            generation = self._generate_grouped_completions(batch_prompts)
            self.timing_manager.end_timer(f"batch_{batch_idx}_generation")
            generated_ids = generation["generated_ids"]
            generated_mask = generation["generated_mask"]
            completion_ids = generation["completion_ids"]
            completion_texts = generation["completion_texts"]
            batch_completion_ids = generation["all_completion_ids"]
            replicated_prompt_end_positions = generation["prompt_end_positions"]
            total_sequences = generation["total_sequences"]

            # Store for later recomputation
            all_generated_ids.append(generated_ids.detach())
            all_generated_mask.append(generated_mask.detach())
            all_prompt_end_positions.append(replicated_prompt_end_positions.detach())

            # Compute log probs for this batch
            self.timing_manager.start_timer(f"batch_{batch_idx}_policy_log_probs")
            prev_mode = self.policy.training
            self.policy.eval()

            # Calculate max completion length to use logits_to_keep
            max_completion_len = max(c.size(0) for c in batch_completion_ids)

            with torch.no_grad():  # No gradients needed during trajectory generation
                # Keep KV cache enabled for faster inference
                # Only compute log probs for completion tokens (model still sees full context)
                with self._amp_config.autocast():
                    policy_log_probs = self.policy.compute_log_probs(
                        generated_ids,
                        attention_mask=generated_mask,
                        logits_to_keep=max_completion_len,
                    )
            if prev_mode:
                self.policy.train()
            self.timing_manager.end_timer(f"batch_{batch_idx}_policy_log_probs")

            # Compute reference log probs for this batch
            self.timing_manager.start_timer(f"batch_{batch_idx}_ref_log_probs")
            self.ref_policy.eval()  # Ensure ref policy is in eval mode
            with torch.no_grad():
                # Keep KV cache enabled for faster inference
                # Only compute log probs for completion tokens (model still sees full context)
                with self._amp_config.autocast():
                    ref_log_probs = self.ref_policy.compute_log_probs(
                        generated_ids,
                        attention_mask=generated_mask,
                        logits_to_keep=max_completion_len,
                    )
            self.timing_manager.end_timer(f"batch_{batch_idx}_ref_log_probs")

            # Extract completion log probs for this batch
            self.timing_manager.start_timer(f"batch_{batch_idx}_extract_completions")
            completion_mask = self._create_completion_mask(completion_ids)

            # policy_log_probs and ref_log_probs are [batch, max_completion_len]
            # With left-padding: logits_to_keep gives [completion_tokens ... padding_tokens]
            # The actual completion tokens are at the BEGINNING of the tensor
            for seq_idx in range(total_sequences):
                seq_completion_ids = batch_completion_ids[seq_idx]
                actual_completion_length = seq_completion_ids.size(0)

                # Extract the FIRST actual_completion_length log probs
                # (completions are at the start due to left-padding)
                seq_policy_log_probs = policy_log_probs[seq_idx, :actual_completion_length]
                seq_ref_log_probs = ref_log_probs[seq_idx, :actual_completion_length]
                seq_completion_mask = completion_mask[seq_idx, :actual_completion_length]

                # Apply mask to zero out tokens after EOS
                seq_policy_log_probs = seq_policy_log_probs * seq_completion_mask
                seq_ref_log_probs = seq_ref_log_probs * seq_completion_mask

                all_log_probs.append(seq_policy_log_probs)
                all_ref_log_probs.append(seq_ref_log_probs)
                all_completion_mask.append(seq_completion_mask)

            self.timing_manager.end_timer(f"batch_{batch_idx}_extract_completions")

            # Compute rewards for this batch
            self.timing_manager.start_timer(f"batch_{batch_idx}_rewards")
            for prompt_idx, (prompt, answer) in enumerate(zip(batch_prompts, batch_answers)):
                start_idx = prompt_idx * self.group_size
                end_idx = start_idx + self.group_size
                group_completions = completion_texts[start_idx:end_idx]

                # Get rewards with breakdown (total, format, correctness) in one call
                group_rewards, group_format_rewards, group_correctness_rewards = self.batch_reward_fn(
                    group_completions, [answer] * self.group_size, self.device, return_breakdown=True
                )
                all_rewards.append(group_rewards)
                all_format_rewards.append(group_format_rewards)
                all_correctness_rewards.append(group_correctness_rewards)

                if store:
                    all_prompts.extend([prompt] * self.group_size)
                    all_completions.extend(group_completions)
            self.timing_manager.end_timer(f"batch_{batch_idx}_rewards")

            # Cleanup batch data
            del generated_ids, generated_mask, completion_ids, policy_log_probs, ref_log_probs
            clear_device_cache(self.device)

            self.timing_manager.end_timer(f"batch_{batch_idx}")

        # Concatenate all batches (pad to same length first)
        # Find max sequence length across all batches
        max_seq_len = max(tensor.size(1) for tensor in all_generated_ids)

        # Pad all tensors to same length - PAD ON THE LEFT to match original padding
        # CRITICAL: Adjust prompt_end_positions when adding left padding!
        padded_generated_ids = []
        padded_generated_mask = []
        adjusted_prompt_end_positions = []
        batch_lengths = [tensor.size(1) for tensor in all_generated_ids]
        padding_adjustments = []

        for gen_ids, gen_mask, prompt_ends in zip(
            all_generated_ids, all_generated_mask, all_prompt_end_positions
        ):
            pad_len = 0
            if gen_ids.size(1) < max_seq_len:
                # Pad on the LEFT (prepend padding) to match tokenizer's left-padding
                pad_len = max_seq_len - gen_ids.size(1)
                gen_ids = F.pad(gen_ids, (pad_len, 0), value=self.policy.tokenizer.pad_token_id)
                gen_mask = F.pad(gen_mask, (pad_len, 0), value=0)

            # Adjust prompt_end_positions by the amount of left-padding added
            # This ensures positions remain valid after padding shifts everything right
            adjusted_prompt_ends = prompt_ends + pad_len

            padded_generated_ids.append(gen_ids)
            padded_generated_mask.append(gen_mask)
            adjusted_prompt_end_positions.append(adjusted_prompt_ends)
            padding_adjustments.append(pad_len)

        self.stored_generated_ids = torch.cat(padded_generated_ids, dim=0)
        self.stored_attention_mask = torch.cat(padded_generated_mask, dim=0)
        self.stored_prompt_end_positions = torch.cat(adjusted_prompt_end_positions, dim=0)

        # Validate that prompt_end_positions are within bounds after padding adjustment
        assert (self.stored_prompt_end_positions <= self.stored_generated_ids.size(1)).all(), \
            f"Invalid prompt_end_positions after padding: max={self.stored_prompt_end_positions.max()}, seq_len={self.stored_generated_ids.size(1)}"

        # Stack into tensors (OLD log probs from generation - won't change)
        # Note: Already detached from torch.no_grad() context, no need for explicit .detach()
        self.timing_manager.start_timer("tensor_stacking")
        all_log_probs = pad_sequence(all_log_probs, batch_first=True, padding_value=0.0)
        all_ref_log_probs = pad_sequence(all_ref_log_probs, batch_first=True, padding_value=0.0)
        all_completion_mask = pad_sequence(all_completion_mask, batch_first=True, padding_value=0.0)
        all_rewards = torch.cat(all_rewards, dim=0)
        all_format_rewards = torch.cat(all_format_rewards, dim=0)
        all_correctness_rewards = torch.cat(all_correctness_rewards, dim=0)

        # Store format and correctness rewards for metrics tracking
        self.stored_format_rewards = all_format_rewards
        self.stored_correctness_rewards = all_correctness_rewards

        self.timing_manager.end_timer("tensor_stacking")

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
        rewards: torch.Tensor,
        group_size: int = 1,
        normalize_within_groups: bool = True
    ) -> torch.Tensor:
        """
        Compute group-normalized advantages (GRPO's key innovation).

        Args:
            rewards: Reward tensor [batch_size]
            group_size: Number of samples per prompt
            normalize_within_groups: Whether to normalize per group

        Returns:
            Advantages tensor [batch_size]
        """
        if normalize_within_groups and group_size > 1:
            batch_size = rewards.shape[0]
            num_groups = batch_size // group_size

            # Reshape to groups
            grouped_rewards = rewards.view(num_groups, group_size)

            # Normalize within each group (baseline = group mean)
            group_mean = grouped_rewards.mean(dim=1, keepdim=True)
            group_std = grouped_rewards.std(dim=1, keepdim=True)

            normalized_rewards = (grouped_rewards - group_mean) / (group_std + 1e-4)

            # Flatten back
            advantages = normalized_rewards.view(-1)
        else:
            # Global normalization
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        return advantages

    def compute_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss with PPO-style clipping.

        Loss = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * KL

        Args:
            new_log_probs: Current policy log probs [batch, seq_len]
            old_log_probs: Old policy log probs (from generation) [batch, seq_len]
            advantages: Group-normalized advantages [batch]
            ref_log_probs: Reference model log probs [batch, seq_len]
            completion_mask: Valid token mask [batch, seq_len]

        Returns:
            (loss, metrics)
        """
        self.timing_manager.start_timer("loss_computation")

        # Sum log probs over sequence (per-sequence log prob)
        new_log_probs_sum = (new_log_probs * completion_mask).sum(dim=-1)
        old_log_probs_sum = (old_log_probs * completion_mask).sum(dim=-1)

        # Compute ratio: π_new / π_old
        ratio = torch.exp(new_log_probs_sum - old_log_probs_sum)

        # PPO clipped surrogate objective (with asymmetric clipping support)
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon_low,   # Lower bound (e.g., 0.8 if clip_epsilon_low=0.2)
            1.0 + self.clip_epsilon_high   # Upper bound (e.g., 1.4 if clip_epsilon_high=0.4)
        ) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence penalty (per-token)
        # KL(π_ref || π_new) penalizes when new policy diverges from reference
        # This prevents the policy from becoming overconfident
        # NOTE: Detach ref_log_probs - no backprop through reference model!
        ref_log_probs_detached = ref_log_probs.detach()

        # Compute KL divergence using configured estimator
        # Options: "mc" (Monte Carlo), "k3" (low-variance unbiased), "abs", "mse"
        kl_penalty = compute_kl_divergence(
            ref_log_probs_detached,
            new_log_probs,
            mask=completion_mask,
            estimator=self.kl_estimator,
            clamp_min=self.kl_clamp_min,
            clamp_max=self.kl_clamp_max,
        )

        # Total GRPO loss
        loss = policy_loss + self.kl_coef * kl_penalty

        # Metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_penalty.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_min": ratio.min().item(),
            "ratio_max": ratio.max().item(),
            "ratio_clipped_frac": (
                (ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)
            ).float().mean().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "tokens_generated": completion_mask.sum().item(),
        }

        self.timing_manager.end_timer("loss_computation")
        return loss, metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform proper GRPO training step (single epoch as per DeepSeekMath).

        Training procedure:
        1. Collect all rollouts/trajectories (store in RAM)
        2. Compute group-normalized advantages
        3. Single epoch: shuffle, iterate minibatches, update policy
        4. Minibatch = subset of rollouts per optimization step

        Args:
            batch: Dictionary with 'prompts' and 'answers'

        Returns:
            Dictionary of training metrics
        """
        self.policy.train()
        prompts = batch["prompts"]
        answers = batch["answers"]

        self.timing_manager.start_timer("trajectory_generation")
        (
            _,
            _,
            rewards,
            old_log_probs,
            ref_log_probs,
            completion_mask,
        ) = self.generate_trajectories(
            prompts, answers=answers, store_outputs=False
        )
        self.timing_manager.end_timer("trajectory_generation")

        self.timing_manager.start_timer("advantage_computation")
        advantages = self.compute_advantages(
            rewards,
            group_size=self.group_size,
            normalize_within_groups=self.normalize_rewards
        )

        self.timing_manager.end_timer("advantage_computation")

        self.timing_manager.start_timer("optimization")

        total_sequences = rewards.shape[0]

        # Aggregate metrics across all minibatch updates
        epoch_metrics = {
            "policy_loss": 0.0,
            "kl_divergence": 0.0,
            "ratio_mean": 0.0,
            "ratio_clipped_frac": 0.0,
            "ratio_min": 0.0,
            "ratio_max": 0.0,
        }
        num_updates = 0

        # Shuffle rollouts once (single epoch)
        indices = torch.randperm(total_sequences, device=self.device)

        # Iterate through minibatches
        for mb_idx, mb_start in enumerate(range(0, total_sequences, self.minibatch_size), 1):
            self.timing_manager.start_timer(f"minibatch_{mb_idx}")

            mb_end = min(mb_start + self.minibatch_size, total_sequences)
            mb_indices = indices[mb_start:mb_end]

            # Extract minibatch data
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_ref_log_probs = ref_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_completion_mask = completion_mask[mb_indices]

            # Recompute log probs with CURRENT policy
            self.timing_manager.start_timer(f"minibatch_{mb_idx}_recompute")
            mb_generated_ids = self.stored_generated_ids[mb_indices]
            mb_attention_mask = self.stored_attention_mask[mb_indices]

            # Calculate max completion length for this minibatch
            mb_max_completion_len = mb_completion_mask.size(1)

            # Use eval mode to disable dropout (for deterministic log probs)
            # but keep gradients enabled for backprop
            self.policy.eval()
            prev_cache = getattr(self.policy.model.config, "use_cache", None)
            if prev_cache is not None:
                self.policy.model.config.use_cache = False
            try:
                # Gradients are enabled by default (not in no_grad context)
                # Use logits_to_keep to get LAST mb_max_completion_len tokens
                with self._amp_config.autocast():
                    mb_full_log_probs = self.policy.compute_log_probs(
                        mb_generated_ids,
                        attention_mask=mb_attention_mask,
                        logits_to_keep=mb_max_completion_len,
                    )
            finally:
                if prev_cache is not None:
                    self.policy.model.config.use_cache = prev_cache

            # Extract completion log probs for each sequence
            # mb_full_log_probs contains log probs for LAST mb_max_completion_len positions
            # We need to find where each completion starts within this window
            mb_new_log_probs_list = []
            mb_prompt_end_positions = self.stored_prompt_end_positions[mb_indices]

            for i in range(len(mb_indices)):
                # Get sequence length and prompt end position
                seq_len = mb_generated_ids.size(1)
                prompt_end = mb_prompt_end_positions[i].item()

                # Completion starts at prompt_end position in the full sequence
                # logits_to_keep gives us log probs for positions [seq_len - mb_max_completion_len : seq_len]
                # So in mb_full_log_probs, position 0 corresponds to sequence position (seq_len - mb_max_completion_len)
                logits_start_pos = seq_len - mb_max_completion_len

                # Where does the completion start within mb_full_log_probs?
                completion_start_in_logits = prompt_end - logits_start_pos

                # Get actual completion length from mask
                completion_len = mb_completion_mask[i].sum().int().item()

                # Extract completion log probs from correct position
                seq_new_log_probs = mb_full_log_probs[i, completion_start_in_logits:completion_start_in_logits + completion_len]

                # Pad to match completion_mask size
                if seq_new_log_probs.size(0) < mb_completion_mask.size(1):
                    seq_new_log_probs = F.pad(
                        seq_new_log_probs,
                        (0, mb_completion_mask.size(1) - seq_new_log_probs.size(0)),
                        value=0.0
                    )
                else:
                    seq_new_log_probs = seq_new_log_probs[:mb_completion_mask.size(1)]

                mb_new_log_probs_list.append(seq_new_log_probs)

            mb_new_log_probs = torch.stack(mb_new_log_probs_list)
            mb_new_log_probs = mb_new_log_probs * mb_completion_mask

            self.timing_manager.end_timer(f"minibatch_{mb_idx}_recompute")

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            self.timing_manager.start_timer(f"minibatch_{mb_idx}_loss")
            with self._amp_config.autocast():
                loss, mb_metrics = self.compute_loss(
                    mb_new_log_probs,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_ref_log_probs,
                    mb_completion_mask,
                )
            self.timing_manager.end_timer(f"minibatch_{mb_idx}_loss")

            # Backward pass
            self.timing_manager.start_timer(f"minibatch_{mb_idx}_backward")
            if self._amp_config.enabled and self._amp_config.grad_scaler is not None:
                self._amp_config.grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            self.timing_manager.end_timer(f"minibatch_{mb_idx}_backward")

            # Gradient clipping
            self.timing_manager.start_timer(f"minibatch_{mb_idx}_clip")
            if self._amp_config.enabled and self._amp_config.grad_scaler is not None:
                self._amp_config.grad_scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.gradient_clip
            )
            self.timing_manager.end_timer(f"minibatch_{mb_idx}_clip")

            # Parameter update
            self.timing_manager.start_timer(f"minibatch_{mb_idx}_update")
            if self._amp_config.enabled and self._amp_config.grad_scaler is not None:
                self._amp_config.grad_scaler.step(self.optimizer)
                self._amp_config.grad_scaler.update()
            else:
                self.optimizer.step()
            self.timing_manager.end_timer(f"minibatch_{mb_idx}_update")

            # Accumulate metrics
            for key in epoch_metrics:
                if key in mb_metrics:
                    epoch_metrics[key] += mb_metrics[key]
            num_updates += 1

            # Cleanup
            self.timing_manager.start_timer(f"minibatch_{mb_idx}_cleanup")
            del mb_new_log_probs, mb_full_log_probs, loss
            clear_device_cache(self.device)
            self.timing_manager.end_timer(f"minibatch_{mb_idx}_cleanup")

            self.timing_manager.end_timer(f"minibatch_{mb_idx}")

        self.timing_manager.end_timer("optimization")

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, num_updates)

        # Final metrics
        metrics = {
            "total_loss": epoch_metrics["policy_loss"] + self.kl_coef * epoch_metrics["kl_divergence"],
            "pg_loss": epoch_metrics["policy_loss"],
            "kl_divergence": epoch_metrics["kl_divergence"],
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "format_reward_mean": self.stored_format_rewards.mean().item(),
            "correctness_reward_mean": self.stored_correctness_rewards.mean().item(),
            "ratio_mean": epoch_metrics["ratio_mean"],
            "ratio_min": epoch_metrics["ratio_min"],
            "ratio_max": epoch_metrics["ratio_max"],
            "ratio_clipped_frac": epoch_metrics["ratio_clipped_frac"],
            "tokens_generated": completion_mask.sum().item(),
        }

        # Update statistics
        self.total_steps += 1

        # Log metrics
        self.logger.log_metrics(metrics, self.total_steps)

        # Cleanup
        del old_log_probs, ref_log_probs, advantages, completion_mask, rewards
        clear_device_cache(self.device)

        return metrics

    def train(
        self,
        train_data: Dict[str, List[str]],
        val_data: Optional[Dict[str, List[str]]] = None,
        num_episodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model using GRPO.

        Args:
            train_data: Dict with 'prompts' and 'answers' lists
            val_data: Optional validation data with same format
            num_episodes: Number of episodes to train (overrides config)

        Returns:
            Dictionary of training and validation metrics
        """
        import time
        from pathlib import Path

        # Get training parameters from config
        if num_episodes is None:
            num_episodes = self.config['training']['num_episodes']

        batch_size = self.config['training']['batch_size']
        update_epochs = self.config['training'].get('update_epochs', 1)
        log_interval = self.config['logging']['log_interval']
        save_interval = self.config['logging']['save_interval']

        # Validation parameters
        validation_enabled = self.config.get('validation', {}).get('enabled', False)
        validation_interval = self.config.get('validation', {}).get('interval', 10)
        validation_num_samples = self.config.get('validation', {}).get('num_samples', 20)
        validation_num_demo_examples = self.config.get('validation', {}).get('num_demo_examples', 5)

        # Get generation parameters from config
        max_new_tokens = self.config['training']['max_new_tokens']
        temperature = self.config['training']['temperature']
        top_p = self.config['training']['top_p']

        train_prompts = train_data["prompts"]
        train_answers = train_data["answers"]

        print(f"Starting proper GRPO training for {num_episodes} episodes...")
        # Show asymmetric clipping if different, otherwise show symmetric
        if self.clip_epsilon_low == self.clip_epsilon_high:
            print(f"  - Using PPO clipped objective (clip_epsilon={self.clip_epsilon})")
        else:
            print(f"  - Using PPO clipped objective (asymmetric: low={self.clip_epsilon_low}, high={self.clip_epsilon_high})")
            print(f"    Ratio clipped to [{1-self.clip_epsilon_low:.2f}, {1+self.clip_epsilon_high:.2f}]")
        print(f"  - Training samples: {len(train_prompts)}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Update epochs: {update_epochs}")
        print(f"  - Group size: {self.group_size}")
        print(f"  - Rollout batch size: {self.rollout_batch_size} prompts (OOM prevention)")
        print(f"  - Minibatch size: {self.minibatch_size} rollouts per update")
        if validation_enabled:
            print(f"  - Validation enabled: every {validation_interval} episodes")
            print(f"    - Evaluating {validation_num_samples} validation samples")
            print(f"    - Showing {validation_num_demo_examples} demo examples")
        print("=" * 50)

        # Training metrics storage
        training_metrics = {
            "episode": [],
            "total_loss": [],
            "pg_loss": [],
            "kl_divergence": [],
            "reward_mean": [],
            "reward_std": [],
            "format_reward_mean": [],
            "correctness_reward_mean": [],
            "tokens_generated": [],
            "episode_time": [],
            "total_tokens": [],
            "tokens_per_second": []
        }

        # Validation metrics storage
        validation_metrics = {
            "episode": [],
            "exact_accuracy": [],
            "numeric_accuracy": [],
            "format_compliance": [],
            "avg_format_score": [],
            "avg_correctness_score": []
        }

        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints/grpo_qwen_math")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        cumulative_tokens = 0
        training_start_time = time.time()

        for episode in range(num_episodes):
            self.episode = episode
            episode_start_time = time.time()

            # Aggregate metrics across all epochs in this episode
            sum_total_loss = 0.0
            sum_pg_loss = 0.0
            sum_kl = 0.0
            sum_reward_mean = 0.0
            sum_reward_std = 0.0
            sum_format_reward_mean = 0.0
            sum_correctness_reward_mean = 0.0
            sum_tokens_generated = 0
            episode_tokens = 0

            for _ in range(update_epochs):
                # Sample a fresh batch of problems per epoch
                batch_indices = np.random.choice(len(train_prompts), batch_size, replace=True)

                # Get prompts and answers for this batch
                batch_prompts = [train_prompts[i] for i in batch_indices]
                batch_answers = [train_answers[i] for i in batch_indices]

                # Pass entire batch to train_step
                batch_data = {
                    "prompts": batch_prompts,
                    "answers": batch_answers
                }
                metrics = self.train_step(batch_data)

                # Accumulate metrics
                sum_total_loss += metrics["total_loss"]
                sum_pg_loss += metrics["pg_loss"]
                sum_kl += metrics["kl_divergence"]
                sum_reward_mean += metrics["reward_mean"]
                sum_reward_std += metrics.get("reward_std", 0.0)
                sum_format_reward_mean += metrics.get("format_reward_mean", 0.0)
                sum_correctness_reward_mean += metrics.get("correctness_reward_mean", 0.0)

                # Track tokens
                tokens_in_batch = metrics.get("tokens_generated", 0)
                sum_tokens_generated += tokens_in_batch
                episode_tokens += tokens_in_batch

            # Calculate episode time
            episode_time = time.time() - episode_start_time
            cumulative_tokens += episode_tokens

            # Calculate tokens per second for this episode
            tokens_per_sec = episode_tokens / episode_time if episode_time > 0 else 0

            # Episode-level averaged metrics
            avg_metrics = {
                "total_loss": sum_total_loss / update_epochs,
                "pg_loss": sum_pg_loss / update_epochs,
                "kl_divergence": sum_kl / update_epochs,
                "reward_mean": sum_reward_mean / update_epochs,
                "reward_std": sum_reward_std / update_epochs,
                "format_reward_mean": sum_format_reward_mean / update_epochs,
                "correctness_reward_mean": sum_correctness_reward_mean / update_epochs,
                "tokens_generated": sum_tokens_generated / update_epochs,
            }

            # Store metrics
            training_metrics["episode"].append(episode)
            training_metrics["total_loss"].append(avg_metrics["total_loss"])
            training_metrics["pg_loss"].append(avg_metrics["pg_loss"])
            training_metrics["kl_divergence"].append(avg_metrics["kl_divergence"])
            training_metrics["reward_mean"].append(avg_metrics["reward_mean"])
            training_metrics["reward_std"].append(avg_metrics.get("reward_std", 0.0))
            training_metrics["format_reward_mean"].append(avg_metrics["format_reward_mean"])
            training_metrics["correctness_reward_mean"].append(avg_metrics["correctness_reward_mean"])
            training_metrics["tokens_generated"].append(avg_metrics.get("tokens_generated", 0))
            training_metrics["episode_time"].append(episode_time)
            training_metrics["total_tokens"].append(cumulative_tokens)
            training_metrics["tokens_per_second"].append(tokens_per_sec)

            # Logging
            if episode % log_interval == 0:
                print(f"Episode {int(episode):3d} | "
                      f"Loss: {avg_metrics['total_loss']:7.4f} | "
                      f"PG Loss: {avg_metrics['pg_loss']:7.4f} | "
                      f"KL: {avg_metrics['kl_divergence']:7.4f} | "
                      f"Reward: {avg_metrics['reward_mean']:6.3f} ± {avg_metrics.get('reward_std', 0.0):5.3f} | "
                      f"Fmt: {avg_metrics['format_reward_mean']:5.3f} | "
                      f"Correct: {avg_metrics['correctness_reward_mean']:5.3f} | "
                      f"Tokens: {int(episode_tokens):5d} | "
                      f"Time: {episode_time:5.2f}s | "
                      f"Speed: {tokens_per_sec:6.1f} tok/s")

            # Run validation if enabled and at the right interval
            if validation_enabled and val_data and (episode + 1) % validation_interval == 0:
                print(f"\n{'='*60}")
                print(f"VALIDATION AT EPISODE {episode + 1}")
                print(f"{'='*60}")

                # Import evaluation function
                from simple_rl.evaluation.gsm8k import evaluate_on_gsm8k, demonstrate_model_responses

                # Run evaluation on validation set
                val_metrics = evaluate_on_gsm8k(
                    self,
                    val_data["prompts"],
                    val_data["answers"],
                    validation_num_samples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    model_name=f"Episode {episode + 1}",
                    save_results=True,
                    results_file="results/grpo_eval_results.json",
                    step=episode + 1
                )

                # Store validation metrics
                validation_metrics["episode"].append(episode + 1)
                validation_metrics["exact_accuracy"].append(val_metrics['exact_accuracy'])
                validation_metrics["numeric_accuracy"].append(val_metrics['numeric_accuracy'])
                validation_metrics["format_compliance"].append(val_metrics['format_compliance'])
                validation_metrics["avg_format_score"].append(val_metrics['avg_format_score'])
                validation_metrics["avg_correctness_score"].append(val_metrics['avg_correctness_score'])

                # Demonstrate model responses
                demonstrate_model_responses(
                    self,
                    val_data["prompts"][:validation_num_demo_examples],
                    val_data["answers"][:validation_num_demo_examples],
                    validation_num_demo_examples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    title=f"EPISODE {episode + 1} RESPONSE EXAMPLES"
                )

                print(f"{'='*60}\n")

            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode+1}.pt"
                self.save_checkpoint(str(checkpoint_path))
                print(f"  → Saved checkpoint to {checkpoint_path}")

            # Print timing summary periodically
            if self.print_timing and episode > 0 and episode % 10 == 0:
                self.timing_manager.print_timing_summary(f"Episode {episode} Summary")

        # Calculate total training time
        total_training_time = time.time() - training_start_time
        print("=" * 50)
        print("Training complete!")
        print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        print(f"Total tokens processed: {cumulative_tokens:,}")
        print(f"Average speed: {cumulative_tokens/total_training_time:.1f} tokens/second")

        # Save final model
        final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
        self.save_checkpoint(str(final_checkpoint_path))
        print(f"\n✓ Final model saved to: {final_checkpoint_path}")

        if self.print_timing:
            self.timing_manager.print_timing_summary("FINAL TRAINING PERFORMANCE")

        return {
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "total_time": total_training_time,
            "final_reward": training_metrics["reward_mean"][-1] if training_metrics["reward_mean"] else 0.0
        }

    def evaluate(self, num_episodes: int = 1) -> Dict[str, float]:
        """Evaluate the policy."""
        self.policy.eval()
        total_rewards = []

        with torch.no_grad():
            for _ in range(num_episodes):
                prompts = [f"Test {i}: Calculate {i}*2" for i in range(4)]
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