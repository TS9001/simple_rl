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

import contextlib
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import StoppingCriteria, StoppingCriteriaList

from simple_rl.algorithms.base import BaseAlgorithm
from simple_rl.utils.huggingface_wrappers import LanguageModel
from simple_rl.utils.device import get_target_device, clear_device_cache
# from simple_rl.utils.amp import create_amp_config  # COMMENTED OUT - mixed precision disabled
from simple_rl.utils.optimization import configure_optimizer
from simple_rl.utils.timing import TimingManager
from simple_rl.utils.training_config import create_training_config
from simple_rl.utils.checkpointing import save_checkpoint, load_checkpoint
from simple_rl.utils.logging_utils import create_logger
from simple_rl.utils.kl_divergence import compute_kl_divergence
from simple_rl.utils.policy_loss import (
    compute_ppo_policy_loss,
    compute_ppo_policy_loss_token_level,
)
from simple_rl.utils.debug_logger import GRPODebugLogger


class MultiTokenStoppingCriteria(StoppingCriteria):
    """
    Stopping criterion that checks for complete multi-token sequences.

    This is more robust than checking single token IDs, especially for
    sequences like </answer> that may be tokenized into multiple tokens.
    """

    def __init__(self, stop_sequences: List[str], tokenizer, prompt_length: int):
        """
        Args:
            stop_sequences: List of string sequences to stop on (e.g., ["</answer>"])
            tokenizer: HuggingFace tokenizer
            prompt_length: Length of the prompt (to only check generated tokens)
        """
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Check if any sequence contains a stop sequence in the generated portion."""
        # Decode only the generated portion (after prompt)
        for sequence_ids in input_ids:
            generated_ids = sequence_ids[self.prompt_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Check if any stop sequence appears in the generated text
            for stop_seq in self.stop_sequences:
                if stop_seq in generated_text:
                    return True

        return False


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

        # Disable ALL dropout for deterministic log probs (critical for PPO/GRPO)
        # This prevents train/eval mode from affecting log probability computation
        dropout_disabled = []

        # Step 1: Update config values
        if hasattr(self.policy.model, 'config'):
            model_cfg = self.policy.model.config

            # Disable attention dropout
            if hasattr(model_cfg, 'attention_dropout') and model_cfg.attention_dropout > 0:
                dropout_disabled.append(f"config.attention_dropout: {model_cfg.attention_dropout} → 0")
                model_cfg.attention_dropout = 0.0

            # Disable hidden/residual dropout (different models use different names)
            for attr in ['hidden_dropout', 'hidden_dropout_prob', 'resid_pdrop', 'dropout']:
                if hasattr(model_cfg, attr):
                    old_val = getattr(model_cfg, attr)
                    if old_val > 0:
                        dropout_disabled.append(f"config.{attr}: {old_val} → 0")
                        setattr(model_cfg, attr, 0.0)

        # Step 2: Directly disable all Dropout modules (more reliable than config)
        # This ensures dropout is disabled even if config changes don't propagate
        dropout_modules_disabled = 0
        for _, module in self.policy.model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                if module.p > 0:
                    dropout_modules_disabled += 1
                    module.p = 0.0

        if dropout_modules_disabled > 0:
            dropout_disabled.append(f"{dropout_modules_disabled} Dropout modules: p → 0.0")

        if dropout_disabled:
            print(f"✓ Disabled dropout for deterministic log probs:")
            for change in dropout_disabled:
                print(f"    {change}")

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

        # Training config (need this first to check kl_coef)
        self.training_config = create_training_config(self.config)

        # GRPO parameters
        self.group_size = self.training_config.group_size
        self.kl_coef = self.training_config.kl_coef
        self.normalize_rewards = self.training_config.normalize_rewards

        # Create frozen reference model for KL penalty (ONLY if kl_coef > 0)
        if self.kl_coef > 0:
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
        else:
            self.ref_policy = None
            print(f"✓ KL penalty disabled (kl_coef=0), skipping reference model")

        # Print KL divergence configuration
        self._print_kl_config = True

        # Clipping parameters - support both symmetric and asymmetric clipping
        self.clip_epsilon = self.training_config.clip_epsilon

        if self.clip_epsilon is None:
            self.clip_epsilon_low = self.training_config.clip_epsilon_low
            self.clip_epsilon_high = self.training_config.clip_epsilon_high
        else:
            self.clip_epsilon_low = self.training_config.clip_epsilon
            self.clip_epsilon_high = self.training_config.clip_epsilon

        self.store_completions = self.training_config.store_completions
        self.update_epochs = self.training_config.update_epochs

        # KL divergence estimator configuration
        # Options: "mc" (Monte Carlo), "k3" (low-variance unbiased), "abs", "mse"
        self.kl_estimator = self.config.get("training", {}).get("kl_estimator", "k3")
        # Reduced clamp range to prevent KL explosion: clamp_max=2.0 limits max KL to ~5.4 per token
        # (vs clamp_max=5.0 which allowed KL up to ~142 per token, causing gradient explosion!)
        self.kl_clamp_min = self.config.get("training", {}).get("kl_clamp_min", -2.0)
        self.kl_clamp_max = self.config.get("training", {}).get("kl_clamp_max", 2.0)
        # KL reduction: "mean" (average per token) or "sum" (total across tokens)
        self.kl_reduction = self.config.get("training", {}).get("kl_reduction", "mean")

        # Policy loss configuration
        # Options: "sequence" (traditional PPO, sum over tokens), "token" (normalize by token count)
        self.policy_loss_type = self.config.get("training", {}).get("policy_loss_type", "token")

        # Select policy loss function based on configuration
        if self.policy_loss_type == "token":
            self._compute_policy_loss_fn = compute_ppo_policy_loss_token_level
        elif self.policy_loss_type == "sequence":
            self._compute_policy_loss_fn = compute_ppo_policy_loss
        else:
            raise ValueError(
                f"Unknown policy_loss_type: {self.policy_loss_type}. "
                f"Choose from: 'sequence', 'token'"
            )

        # Clipping parameters for stability
        # Note: Proper PPO clipping is done in ratio-space (not log-space) inside policy loss functions

        # Advantage clipping (prevents extreme policy updates)
        self.advantage_clip_min = self.config.get("training", {}).get("advantage_clip_min", -3.0)
        self.advantage_clip_max = self.config.get("training", {}).get("advantage_clip_max", 3.0)

        # Print configuration
        if self._print_kl_config:
            print(f"✓ KL divergence estimator: {self.kl_estimator}")
            if self.kl_estimator == "k3":
                print(f"  KL clamp range: [{self.kl_clamp_min}, {self.kl_clamp_max}]")
            print(f"✓ Policy loss type: {self.policy_loss_type}")
            if self.policy_loss_type == "token":
                print(f"  Token-level normalization (10-100x lower gradients on long sequences)")
            else:
                print(f"  Sequence-level normalization (traditional PPO)")
            print(f"✓ Clipping parameters:")
            print(f"  PPO ratio clipping: [{1.0-self.clip_epsilon_low}, {1.0+self.clip_epsilon_high}]")
            print(f"  Advantages: [{self.advantage_clip_min}, {self.advantage_clip_max}]")
            del self._print_kl_config

        # Training parameters
        self.learning_rate = self.training_config.learning_rate
        self.batch_size = self.training_config.batch_size
        self.minibatch_size = self.training_config.minibatch_size
        self.rollout_batch_size = self.training_config.rollout_batch_size
        self.max_new_tokens = self.training_config.max_new_tokens
        self.min_new_tokens = self.config.get("training", {}).get("min_new_tokens", 1)
        self.temperature = self.training_config.temperature
        self.top_k = self.training_config.top_k
        self.top_p = self.training_config.top_p
        self.gradient_clip = self.training_config.gradient_clip

        # AMP setup - COMMENTED OUT - mixed precision disabled
        # self._amp_config = create_amp_config(self.config)
        # try:
        #     print(f"AMP: enabled={self._amp_config.enabled}, device={self._amp_config.device}")
        # except Exception:
        #     pass

        # Optimizer and optional LR scheduler (for warmup)
        self.optimizer, self.lr_scheduler = configure_optimizer(self.policy, self.config)

        # Reward function
        self.batch_reward_fn = batch_reward_fn

        # Logger
        self.logger = create_logger(self.config)
        if self.use_wandb:
            self.logger.init_wandb()

        # Debug logger - initialize with all debug flags from config
        debug_config = self.config.get("debug", {})
        debug_enabled = debug_config.get("enabled", False)
        debug_log_dir = debug_config.get("log_dir", "debug_logs")
        self.debug_logger = GRPODebugLogger(
            log_dir=debug_log_dir,
            enabled=debug_enabled,
            debug_generation=debug_config.get("generation", False),
            debug_alignment=debug_config.get("alignment", False),
            debug_advantages=debug_config.get("advantages", False),
            debug_loss=debug_config.get("loss", False),
            debug_gradients=debug_config.get("gradients", False),
        )

        # Statistics
        self.total_steps = 0
        self.episode = 0
        self.current_episode = 0  # Track current episode for reproducible seeding

        # Timing
        self.timing_manager = TimingManager()
        timing_config = self.config.get("timing", {})
        self.print_timing = timing_config.get("enabled", False)  # Disabled by default

        # Device-specific optimizations
        device_opts = self.config.get("device_optimizations", {})
        self.clear_cache_on_mps = device_opts.get("clear_cache_on_mps", False)  # Default: False (don't clear on MPS)

        # Progress logging
        logging_cfg = self.config.get("logging", {})
        self.log_trajectory_progress = logging_cfg.get("show_trajectory_progress", False)  # Default: False

        # Early stopping tokens for generation
        self._setup_stopping_tokens()

    def _setup_stopping_tokens(self):
        """Set up additional stopping tokens for early termination."""
        # Multi-token stopping sequences (more robust than single-token stopping)
        self.stop_sequences = self.config.get("training", {}).get("stop_sequences", ["</answer>"])
        self.use_multi_token_stopping = len(self.stop_sequences) > 0

        # Legacy single-token stopping (kept for backward compatibility)
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

    def _set_episode_seed(self, episode: int, base_seed: int = 42):
        """
        Set deterministic random seeds for an episode.

        This ensures that episode N always uses the same random seed,
        making training fully reproducible even when resuming from checkpoints.

        Args:
            episode: Current episode number
            base_seed: Base random seed (default: 42)
        """
        # Compute episode-specific seed
        episode_seed = base_seed + episode

        # Set Python random seed
        import random
        random.seed(episode_seed)

        # Set NumPy random seed
        np.random.seed(episode_seed)

        # Set PyTorch random seeds
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(episode_seed)
            torch.cuda.manual_seed_all(episode_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(episode_seed)

    def _create_completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        """
        Create mask for completion tokens, excluding padding and tokens after EOS.

        Masks out:
        - PAD tokens (anywhere in sequence)
        - EOS token and everything after it (including custom stop tokens)

        Args:
            completion_ids: Token IDs [batch_size, seq_len]

        Returns:
            Binary mask [batch_size, seq_len] where 1 = valid token, 0 = masked
        """
        pad_token_id = self.policy.tokenizer.pad_token_id
        eos_token_id = self.policy.tokenizer.eos_token_id

        # Mask out padding tokens (but don't stop at them)
        not_pad = (completion_ids != pad_token_id).float()

        # Find EOS and mask everything after it (including EOS itself)
        is_eos = (completion_ids == eos_token_id).int()

        # ALSO check for custom stop token (e.g., </answer>)
        if self.answer_end_token_id is not None:
            is_answer_end = (completion_ids == self.answer_end_token_id).int()
            is_eos = is_eos | is_answer_end  # Stop at either EOS or custom stop token

        eos_cumsum = is_eos.cumsum(dim=1)
        not_after_eos = (eos_cumsum == 0).float()

        return not_pad * not_after_eos


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

        # Build EOS token list for early stopping
        eos_token_id = self.policy.tokenizer.eos_token_id
        if self.answer_end_token_id is not None:
            # Stop at either EOS or </answer> tag
            eos_token_id = [eos_token_id, self.answer_end_token_id]

        # Build multi-token stopping criteria if configured
        stopping_criteria = None
        if self.use_multi_token_stopping:
            prompt_length = replicated_prompt_ids.shape[1]
            stopping_criteria = StoppingCriteriaList([
                MultiTokenStoppingCriteria(
                    stop_sequences=self.stop_sequences,
                    tokenizer=self.policy.tokenizer,
                    prompt_length=prompt_length
                )
            ])

        # Generation with no_grad (dropout is disabled, so train/eval doesn't matter)
        with torch.no_grad():
            # with self._amp_config.autocast():  # COMMENTED OUT - mixed precision disabled
            with contextlib.nullcontext():
                generated_ids, generated_mask = self.policy.generate(
                replicated_prompt_ids,
                attention_mask=replicated_prompt_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_return_sequences=1,
                min_new_tokens=self.min_new_tokens,
                eos_token_id=eos_token_id,
                stopping_criteria=stopping_criteria,
                )

        self.timing_manager.end_timer("batch_text_generation")

        self.timing_manager.start_timer("completion_extraction")

        # With left-padding, each sequence has padding at the start, then prompt tokens, then completion
        # batch_prompt_mask has 1s for real tokens and 0s for padding
        device = generated_ids.device
        total_sequences, seq_len = generated_ids.shape

        # Same for all in batch (they're padded to same width)
        prompt_start_index = batch_prompt_mask.shape[1] - batch_prompt_mask.sum(-1)

        # Replicate prompt_start_index for all generated sequences (group_size per prompt)
        prompt_start_index_per_seq = prompt_start_index.repeat_interleave(self.group_size).to(device)  # Shape: (total_sequences,)

        # Step 1: Convert left-padded generated_ids and generated_mask to RIGHT-padded
        # For each sequence, extract content from prompt_start to end, then left-align
        max_content_length = seq_len - prompt_start_index_per_seq.min().item()

        # Create index tensors for vectorized gathering
        batch_indices = torch.arange(total_sequences, device=device).unsqueeze(1)  # Shape: (total_sequences, 1)
        seq_positions = torch.arange(max_content_length, device=device).unsqueeze(0)  # Shape: (1, max_content_length)

        # Calculate source positions in generated_ids for each sequence
        source_positions = prompt_start_index_per_seq.unsqueeze(1) + seq_positions  # Shape: (total_sequences, max_content_length)

        # Clamp to valid range to avoid out-of-bounds
        source_positions = source_positions.clamp(0, seq_len - 1)

        # Create mask for valid positions (not beyond sequence end)
        valid_mask = seq_positions < (seq_len - prompt_start_index_per_seq.unsqueeze(1))

        # Gather from generated_ids and generated_mask using advanced indexing
        right_padded_ids = generated_ids[batch_indices, source_positions]
        right_padded_mask = generated_mask[batch_indices, source_positions]

        # Apply padding where invalid
        right_padded_ids = torch.where(valid_mask, right_padded_ids,
                                       torch.tensor(self.policy.tokenizer.pad_token_id, dtype=generated_ids.dtype, device=device))
        right_padded_mask = torch.where(valid_mask, right_padded_mask,
                                        torch.zeros(1, dtype=generated_mask.dtype, device=device))

        # Step 2: Extract ONLY completions (skip prompt tokens) from right-padded tensor
        # Compute the actual prompt length (without padding) for each sequence
        batch_prompt_lengths = batch_prompt_mask.sum(dim=1)  # Shape: (batch_size,)

        # Replicate prompt lengths for all generated sequences (group_size per prompt)
        prompt_lengths_per_seq = batch_prompt_lengths.repeat_interleave(self.group_size).to(device)

        # Calculate max completion length
        max_completion_length = max_content_length - prompt_lengths_per_seq.max().item()

        # Extract ONLY completions (skip prompt tokens) from right-padded tensor (VECTORIZED)
        # Create index tensors for vectorized gathering
        batch_indices = torch.arange(total_sequences, device=device).unsqueeze(1)  # Shape: (total_sequences, 1)
        completion_positions = torch.arange(max_completion_length, device=device).unsqueeze(0)  # Shape: (1, max_completion_length)

        # Calculate source positions in right_padded_ids (skip prompt_lengths_per_seq tokens)
        source_positions = prompt_lengths_per_seq.unsqueeze(1) + completion_positions  # Shape: (total_sequences, max_completion_length)

        # Clamp to valid range
        source_positions = source_positions.clamp(0, max_content_length - 1)

        # Create mask for valid positions (within bounds)
        valid_mask = source_positions < max_content_length

        # Gather completions using advanced indexing
        completion_ids = right_padded_ids[batch_indices, source_positions]
        completion_mask = right_padded_mask[batch_indices, source_positions]

        # Apply padding where invalid
        completion_ids = torch.where(valid_mask, completion_ids,
                                     torch.tensor(self.policy.tokenizer.pad_token_id, dtype=generated_ids.dtype, device=device))
        completion_mask = torch.where(valid_mask, completion_mask,
                                      torch.zeros(1, dtype=generated_mask.dtype, device=device))

        completion_texts = self.policy.decode(completion_ids)
        self.timing_manager.end_timer("completion_extraction")

        # ASSERTION 1: Verify text reconstruction (prompt + completion = full generation)
        # Episode 0, Batch 1: Check first few sequences
        if self.episode == 0:
            for seq_idx in range(min(3, total_sequences)):
                # Decode full generated sequence
                # IMPORTANT: Use skip_special_tokens=True to remove padding tokens
                full_generated_text = self.policy.tokenizer.decode(
                    generated_ids[seq_idx], skip_special_tokens=True
                )

                # Get the prompt for this sequence (accounting for group replication)
                prompt_idx = seq_idx // self.group_size
                prompt_text = prompts[prompt_idx]

                # Reconstruct: prompt + completion
                completion_text = completion_texts[seq_idx]

                # Decode completion_ids to verify they match completion_texts
                # IMPORTANT: Use skip_special_tokens=True to match how policy.decode() works
                decoded_completion = self.policy.tokenizer.decode(
                    completion_ids[seq_idx], skip_special_tokens=True
                )

                # Check that decoded completion matches stored completion
                # Note: There may be subtle whitespace differences, so we check if they're close enough
                if decoded_completion != completion_text:
                    # Show the actual difference
                    print(f"\n⚠️  Seq {seq_idx}: Decoded completion differs from stored (checking if significant)...")
                    print(f"  Decoded length: {len(decoded_completion)}, Stored length: {len(completion_text)}")
                    print(f"  Decoded repr: {repr(decoded_completion[:100])}")
                    print(f"  Stored repr:  {repr(completion_text[:100])}")

                    # Check if difference is just whitespace
                    decoded_stripped = decoded_completion.strip()
                    stored_stripped = completion_text.strip()

                    if decoded_stripped == stored_stripped:
                        print(f"  ✓ Only whitespace difference (safe to ignore)")
                    else:
                        # Check character-by-character where they differ
                        min_len = min(len(decoded_completion), len(completion_text))
                        for i in range(min_len):
                            if decoded_completion[i] != completion_text[i]:
                                print(f"  First difference at position {i}:")
                                print(f"    Decoded: {repr(decoded_completion[max(0,i-10):i+10])}")
                                print(f"    Stored:  {repr(completion_text[max(0,i-10):i+10])}")
                                break

                        # For now, we'll just warn but not fail - the tokens are what matter
                        print(f"  ⚠️  Text representations differ, but tokens are the same")
                        print(f"  (This is expected due to tokenizer decode variations)")
                else:
                    if seq_idx == 0:
                        print(f"✓ Decoded completion matches stored completion")

                # CRITICAL: Verify that prompt + completion = full generation
                # Use TEXT level (with skip_special_tokens) to automatically handle padding

                # Decode full generated text (skip_special_tokens removes padding)
                # Note: full_generated_text already decoded above

                # Check if full generation starts with prompt
                if not full_generated_text.startswith(prompt_text):
                    print(f"\n❌ Seq {seq_idx}: Full generation doesn't start with prompt!")
                    print(f"  Prompt length: {len(prompt_text)}")
                    print(f"  Full generation length: {len(full_generated_text)}")
                    print(f"  Prompt (first 100 chars): {repr(prompt_text[:100])}")
                    print(f"  Generation (first 100 chars): {repr(full_generated_text[:100])}")

                    # Find where they differ
                    min_len = min(len(prompt_text), len(full_generated_text))
                    for i in range(min_len):
                        if prompt_text[i] != full_generated_text[i]:
                            print(f"  First diff at char {i}:")
                            print(f"    Prompt: {repr(prompt_text[max(0,i-20):i+20])}")
                            print(f"    Generated: {repr(full_generated_text[max(0,i-20):i+20])}")
                            break

                assert full_generated_text.startswith(prompt_text), (
                    f"Seq {seq_idx}: Generated text doesn't start with prompt!\n"
                    f"  This means generation or extraction is broken."
                )

                # Check if prompt + completion reconstructs the full generation
                reconstructed = prompt_text + completion_text
                if reconstructed != full_generated_text:
                    print(f"\n⚠️  Seq {seq_idx}: Prompt + completion doesn't exactly match full generation")
                    print(f"  Prompt length: {len(prompt_text)}")
                    print(f"  Completion length: {len(completion_text)}")
                    print(f"  Reconstructed length: {len(reconstructed)}")
                    print(f"  Full generation length: {len(full_generated_text)}")

                    # This is often OK due to tokenization artifacts, but warn
                    # Check if they're close
                    if abs(len(reconstructed) - len(full_generated_text)) > 5:
                        print(f"  ⚠️  Length difference > 5 chars - might be a problem!")
                    else:
                        print(f"  ✓ Length difference small - likely tokenization artifact")

                if seq_idx == 0:
                    print(f"\n✓ [Episode {self.episode}] Text reconstruction verified for first sequence")
                    print(f"  Prompt length: {len(prompt_text)} chars")
                    print(f"  Completion length: {len(completion_text)} chars")
                    print(f"  Full generation length: {len(full_generated_text)} chars")
                    print(f"  Completion IDs shape: {completion_ids.shape}")

        return {
            "generated_ids": right_padded_ids,
            "generated_mask": right_padded_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "completion_texts": completion_texts,
            "prompt_end_positions": prompt_lengths_per_seq,  # Per-sequence real prompt lengths (for any future use)
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
        torch.Tensor,
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
            (prompts, completions, rewards, old_log_probs, ref_log_probs, completion_mask,
             format_rewards, correctness_rewards, generated_ids, attention_mask, prompt_end_positions)
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

        # Calculate total number of batches for progress logging
        num_batches = (num_prompts + self.rollout_batch_size - 1) // self.rollout_batch_size
        
        # Process prompts in batches
        for batch_start in range(0, num_prompts, self.rollout_batch_size):
            batch_idx = batch_start // self.rollout_batch_size + 1
            
            # Progress indicator (compact, non-intrusive)
            if self.log_trajectory_progress:
                print(f"  [Trajectory {batch_idx}/{num_batches}]", end=" ", flush=True)
            
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
            completion_mask = generation["completion_mask"]
            completion_texts = generation["completion_texts"]
            replicated_prompt_end_positions = generation["prompt_end_positions"]
            total_sequences = generation["total_sequences"]

            # Store for later recomputation
            all_generated_ids.append(generated_ids.detach())
            all_generated_mask.append(generated_mask.detach())
            all_prompt_end_positions.append(replicated_prompt_end_positions.detach())

            # Compute log probs for this batch (dropout is disabled, so train/eval doesn't matter)
            self.timing_manager.start_timer(f"batch_{batch_idx}_policy_log_probs")

            # Calculate max completion length for output shape
            max_completion_len = completion_ids.size(1)

            with torch.no_grad():  # No gradients needed during trajectory generation
                # Compute log probs for full sequence (model needs prompt as context)
                # with self._amp_config.autocast():  # COMMENTED OUT - mixed precision disabled
                with contextlib.nullcontext():
                    full_log_probs = self.policy.compute_log_probs(
                        generated_ids,
                        attention_mask=generated_mask,
                    )

                    # Extract ONLY completion log probs (vectorized)
                    # Note: log_probs are shifted (predicting next token), so:
                    # - full_log_probs[:, i] predicts token at position i+1 in generated_ids
                    # - Completion starts at prompt_lengths_per_seq in generated_ids
                    # - So completion log probs start at (prompt_lengths_per_seq - 1) in full_log_probs

                    device = full_log_probs.device
                    batch_indices = torch.arange(total_sequences, device=device).unsqueeze(1)  # [total_sequences, 1]
                    completion_positions = torch.arange(max_completion_len, device=device).unsqueeze(0)  # [1, max_completion_len]

                    # Source positions in full_log_probs for completion tokens
                    # Subtract 1 because log probs are shifted
                    source_positions = (replicated_prompt_end_positions - 1).unsqueeze(1) + completion_positions  # [total_sequences, max_completion_len]

                    # Clamp to valid range
                    source_positions = source_positions.clamp(0, full_log_probs.size(1) - 1)

                    # Create mask for valid positions
                    valid_mask = source_positions < full_log_probs.size(1)

                    # Gather completion log probs (right-padded)
                    policy_log_probs = full_log_probs[batch_indices, source_positions]

                    # Set padding positions to 0.0
                    policy_log_probs = torch.where(valid_mask, policy_log_probs,
                                                   torch.tensor(0.0, device=device, dtype=policy_log_probs.dtype))

            self.timing_manager.end_timer(f"batch_{batch_idx}_policy_log_probs")

            # DEBUG: Check policy_log_probs right after computation
            if self.episode == 0 and batch_idx == 1:
                print(f"\n[DEBUG] RIGHT AFTER compute_log_probs:")
                print(f"  generated_ids shape: {generated_ids.shape}")
                print(f"  generated_mask shape: {generated_mask.shape}")
                print(f"  policy_log_probs shape: {policy_log_probs.shape}")
                print(f"  logits_to_keep (max_completion_len): {max_completion_len}")
                print(f"  completion_ids shape: {completion_ids.shape}")

                # Check last few sequences
                for seq_idx in range(max(0, total_sequences - 5), total_sequences):
                    gen_valid = (generated_mask[seq_idx] > 0).sum().item()
                    logprob_nonzero = (policy_log_probs[seq_idx] != 0.0).sum().item()
                    print(f"  Seq {seq_idx}: generated_mask valid={gen_valid}, policy_log_probs non-zero={logprob_nonzero}/{policy_log_probs.shape[1]}")

            # Compute reference log probs for this batch (ONLY if KL penalty enabled)
            if self.kl_coef > 0 and self.ref_policy is not None:
                self.timing_manager.start_timer(f"batch_{batch_idx}_ref_log_probs")
                with torch.no_grad():
                    # Compute log probs for full sequence (model needs prompt as context)
                    # with self._amp_config.autocast():  # COMMENTED OUT - mixed precision disabled
                    with contextlib.nullcontext():
                        full_ref_log_probs = self.ref_policy.compute_log_probs(
                            generated_ids,
                            attention_mask=generated_mask,
                        )

                        # Extract ONLY completion log probs (vectorized, same as policy)
                        # Reuse the same indexing as policy_log_probs
                        ref_log_probs = full_ref_log_probs[batch_indices, source_positions]

                        # Set padding positions to 0.0
                        ref_log_probs = torch.where(valid_mask, ref_log_probs,
                                                    torch.tensor(0.0, device=device, dtype=ref_log_probs.dtype))

                self.timing_manager.end_timer(f"batch_{batch_idx}_ref_log_probs")
            else:
                # KL penalty disabled - create dummy ref_log_probs (will be ignored in loss)
                ref_log_probs = torch.zeros_like(policy_log_probs)

            # Extract completion log probs for this batch
            self.timing_manager.start_timer(f"batch_{batch_idx}_extract_completions")
            # completion_mask is already created in _generate_grouped_completions and returned

            # ASSERTION 2: Verify mask correctness (stops at EOS, PAD, custom stop tokens)
            # Episode 0, Batch 1: Check that masks correctly identify end tokens
            if self.episode == 0 and batch_idx == 1:
                pad_token_id = self.policy.tokenizer.pad_token_id
                eos_token_id = self.policy.tokenizer.eos_token_id

                print(f"\n✓ [Episode {self.episode}, Batch {batch_idx}] Verifying completion masks...")

                for seq_idx in range(min(3, total_sequences)):
                    mask_seq = completion_mask[seq_idx]
                    ids_seq = completion_ids[seq_idx]

                    # Count valid (non-masked) tokens
                    valid_count = (mask_seq > 0).sum().item()

                    # Find where mask transitions from 1 to 0 (if it does)
                    mask_diffs = torch.diff(mask_seq, prepend=torch.tensor([1.0], device=mask_seq.device))
                    transition_idx = torch.where(mask_diffs < 0)[0]

                    if len(transition_idx) > 0:
                        # Mask transitions to 0 at some point
                        stop_idx = transition_idx[0].item()
                        stop_token_id = ids_seq[stop_idx].item()

                        # Check that the stop token is one of: PAD, EOS, or custom stop token
                        valid_stop_tokens = [pad_token_id, eos_token_id]
                        if self.answer_end_token_id is not None:
                            valid_stop_tokens.append(self.answer_end_token_id)

                        is_valid_stop = stop_token_id in valid_stop_tokens

                        if seq_idx < 3:  # Print for first 3 sequences
                            token_name = "PAD" if stop_token_id == pad_token_id else (
                                "EOS" if stop_token_id == eos_token_id else (
                                    "CUSTOM_STOP" if stop_token_id == self.answer_end_token_id else "UNKNOWN"
                                )
                            )
                            print(f"  Seq {seq_idx}: valid_tokens={valid_count}, "
                                  f"stops_at={stop_idx}, stop_token={stop_token_id} ({token_name})")

                        # Assert that mask stops at a valid token
                        assert is_valid_stop, (
                            f"Seq {seq_idx}: Mask stops at invalid token {stop_token_id}!\n"
                            f"  Valid stop tokens: {valid_stop_tokens}\n"
                            f"  Token sequence around stop: {ids_seq[max(0, stop_idx-5):stop_idx+5].tolist()}"
                        )

                        # Assert that all tokens AFTER stop are masked (should be 0)
                        tokens_after_stop = mask_seq[stop_idx+1:]
                        if len(tokens_after_stop) > 0:
                            assert torch.all(tokens_after_stop == 0), (
                                f"Seq {seq_idx}: Found non-zero mask values after stop token!\n"
                                f"  Stop index: {stop_idx}\n"
                                f"  Mask after stop: {tokens_after_stop.tolist()}"
                            )
                    else:
                        # Mask doesn't transition (all 1s or all 0s)
                        if valid_count == 0:
                            # All masked - this can happen if sequence is all PAD
                            print(f"  Seq {seq_idx}: ALL MASKED (valid_tokens=0) - likely all PAD tokens")
                            # Check that ids are indeed all PAD
                            all_pad = torch.all(ids_seq == pad_token_id).item()
                            if not all_pad:
                                print(f"    ⚠️  WARNING: Not all PAD! First 20 tokens: {ids_seq[:20].tolist()}")
                        else:
                            # No stop found - sequence ran to max length
                            print(f"  Seq {seq_idx}: NO STOP (valid_tokens={valid_count}) - ran to max_new_tokens")

                print(f"✓ Mask correctness verified for batch {batch_idx}")

            # policy_log_probs contains log probs for the LAST max_completion_len positions in generated_ids
            # completion_ids are extracted starting from prompt_end_position for each sequence
            # Since all sequences in this batch have the same prompt_end_position (no left-padding within batch),
            # the completion log probs are already aligned - just take the first max_completion_len
            batch_policy_log_probs = policy_log_probs * completion_mask
            batch_ref_log_probs = ref_log_probs * completion_mask

            # Save detailed generation debug log to file (AFTER batch_policy_log_probs is created)
            if self.episode == 0 and batch_idx == 1:
                import os
                debug_log_dir = "debug_logs"
                os.makedirs(debug_log_dir, exist_ok=True)
                debug_log_file = os.path.join(debug_log_dir, f"trajectory_generations_ep0_batch{batch_idx}.txt")

                with open(debug_log_file, "w") as f:
                    f.write("="*80 + "\n")
                    f.write(f"TRAJECTORY GENERATION DEBUG LOG - Episode 0, Batch {batch_idx}\n")
                    f.write("="*80 + "\n\n")

                    for seq_idx in range(total_sequences):
                        f.write(f"\n{'='*80}\n")
                        f.write(f"SEQUENCE {seq_idx}\n")
                        f.write(f"{'='*80}\n\n")

                        # Get prompt for this sequence
                        prompt_idx = seq_idx // self.group_size
                        prompt_text = batch_prompts[prompt_idx]

                        # Get completion
                        completion_text = completion_texts[seq_idx]

                        # Get full generated text
                        full_text = self.policy.tokenizer.decode(
                            generated_ids[seq_idx], skip_special_tokens=True
                        )

                        # Get IDs and masks
                        comp_ids = completion_ids[seq_idx]
                        mask_seq = completion_mask[seq_idx]
                        policy_lp = batch_policy_log_probs[seq_idx]
                        ref_lp = batch_ref_log_probs[seq_idx]

                        # Statistics
                        valid_tokens = (mask_seq > 0).sum().item()
                        policy_nonzero = (policy_lp != 0.0).sum().item()
                        ref_nonzero = (ref_lp != 0.0).sum().item()

                        f.write(f"Prompt (group {prompt_idx}):\n")
                        f.write(f"{prompt_text[:200]}...\n\n")

                        f.write(f"Completion:\n")
                        f.write(f"{completion_text[:200]}...\n\n")

                        f.write(f"Full Generated Text:\n")
                        f.write(f"{full_text[:400]}...\n\n")

                        f.write(f"Statistics:\n")
                        f.write(f"  Completion length: {comp_ids.shape[0]}\n")
                        f.write(f"  Valid tokens (mask=1): {valid_tokens}\n")
                        f.write(f"  Policy logprobs non-zero: {policy_nonzero}\n")
                        f.write(f"  Ref logprobs non-zero: {ref_nonzero}\n\n")

                        f.write(f"Completion Token IDs (first 30):\n")
                        f.write(f"{comp_ids[:30].tolist()}\n\n")

                        f.write(f"Completion Mask (first 30):\n")
                        f.write(f"{mask_seq[:30].tolist()}\n\n")

                        f.write(f"Policy Logprobs (first 20 non-zero):\n")
                        policy_nonzero_vals = policy_lp[policy_lp != 0.0][:20]
                        f.write(f"{policy_nonzero_vals.tolist()}\n\n")

                        f.write(f"Ref Logprobs (first 20 non-zero):\n")
                        ref_nonzero_vals = ref_lp[ref_lp != 0.0][:20]
                        f.write(f"{ref_nonzero_vals.tolist()}\n\n")

                        # Check if old = ref
                        if policy_nonzero > 0 and ref_nonzero > 0:
                            diff = (policy_lp - ref_lp).abs().max().item()
                            f.write(f"Max diff (policy vs ref): {diff:.2e}\n")
                            if diff > 1e-6:
                                f.write(f"  ⚠️  WARNING: policy and ref differ by {diff:.2e}!\n")

                print(f"✓ Saved generation debug log to: {debug_log_file}")

            # Append per-sequence tensors (needed for variable-length padding later)
            # ALSO store the actual number of non-zero log probs for each sequence
            # This is needed during recomputation to request the correct number of tokens
            for seq_idx in range(total_sequences):
                # DEBUG: Check stored values (Episode 0 only)
                if self.episode == 0:
                    mask_valid = (completion_mask[seq_idx] > 0).sum().item()
                    policy_nonzero = (batch_policy_log_probs[seq_idx] != 0.0).sum().item()
                    ref_nonzero = (batch_ref_log_probs[seq_idx] != 0.0).sum().item()

                    # Calculate global trajectory index
                    global_traj_idx = len(all_log_probs)  # Current length = index of next item

                    if mask_valid == 0 or policy_nonzero == 0 or ref_nonzero == 0:
                        print(f"  ⚠️  [Episode 0, Batch {batch_idx}] Trajectory {global_traj_idx} (Seq {seq_idx} in batch) PROBLEM:")
                        print(f"       completion_mask valid={mask_valid}")
                        print(f"       batch_policy_log_probs non-zero={policy_nonzero}")
                        print(f"       batch_ref_log_probs non-zero={ref_nonzero}")
                        print(f"       completion_ids first 30: {completion_ids[seq_idx][:30].tolist()}")
                        print(f"       PAD={self.policy.tokenizer.pad_token_id}, EOS={self.policy.tokenizer.eos_token_id}")

                all_log_probs.append(batch_policy_log_probs[seq_idx, :])
                all_ref_log_probs.append(batch_ref_log_probs[seq_idx, :])
                all_completion_mask.append(completion_mask[seq_idx, :])

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
            if self.device.type == "cuda" or (self.device.type == "mps" and self.clear_cache_on_mps):
                clear_device_cache(self.device)

            self.timing_manager.end_timer(f"batch_{batch_idx}")
            
            # Show completion for this batch
            if self.log_trajectory_progress:
                print("✓", end="", flush=True)

        # Newline after all batches complete
        if self.log_trajectory_progress:
            print()  # Move to next line after progress indicators

        # Don't concatenate - return as LISTS of variable-length tensors
        # Padding will be done per-minibatch to save memory
        # Convert all_generated_ids and all_generated_mask from list of batches to list of sequences
        generated_ids = []
        attention_mask = []
        for batch_ids, batch_mask in zip(all_generated_ids, all_generated_mask):
            # Split batch into individual sequences
            for seq_idx in range(batch_ids.size(0)):
                generated_ids.append(batch_ids[seq_idx])
                attention_mask.append(batch_mask[seq_idx])

        # Flatten prompt_end_positions from list of batches to single tensor
        prompt_end_positions = torch.cat(all_prompt_end_positions, dim=0)

        # Concatenate rewards (these are scalars per sequence, no padding needed)
        self.timing_manager.start_timer("tensor_stacking")
        all_rewards = torch.cat(all_rewards, dim=0)
        all_format_rewards = torch.cat(all_format_rewards, dim=0)
        all_correctness_rewards = torch.cat(all_correctness_rewards, dim=0)
        self.timing_manager.end_timer("tensor_stacking")

        # Keep log probs and completion masks as LISTS of variable-length tensors
        # Padding will be done per-minibatch
        # Note: all_log_probs, all_ref_log_probs, all_completion_mask are already lists

        if store:
            return (
                all_prompts,
                all_completions,
                all_rewards,
                all_log_probs,
                all_ref_log_probs,
                all_completion_mask,
                all_format_rewards,
                all_correctness_rewards,
                generated_ids,
                attention_mask,
                prompt_end_positions,
            )
        return (
            None,
            None,
            all_rewards,
            all_log_probs,
            all_ref_log_probs,
            all_completion_mask,
            all_format_rewards,
            all_correctness_rewards,
            generated_ids,
            attention_mask,
            prompt_end_positions,
        )

    def _validate_log_probs_episode_zero(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> None:
        """
        Validate log probs in first minibatch to ensure dropout is disabled.

        This check verifies:
        - old vs ref: should be exactly 0 (same model, same forward pass, no gradients)
        - old vs new: can differ slightly due to gradient tracking overhead (< 5e-4)
        """
        # Compute per-sequence differences (absolute max per sequence)
        old_new_diff_per_seq = (old_log_probs - new_log_probs).abs().max(dim=1).values
        old_ref_diff_per_seq = (old_log_probs - ref_log_probs).abs().max(dim=1).values

        # Compute overall max
        old_new_diff = old_new_diff_per_seq.max().item()
        old_ref_diff = old_ref_diff_per_seq.max().item()

        # Print per-sequence diagnostics
        print(f"\n{'='*80}")
        print(f"[ASSERTION 3: Log Prob Validation - Episode 0, MB1]")
        print(f"{'='*80}")
        print(f"\n✓ Checking old vs ref (MUST be ~0, same model before training):")
        for i, diff in enumerate(old_ref_diff_per_seq):
            status = "✓" if diff < 1e-6 else "❌"
            print(f"  {status} Seq {i}: {diff.item():.2e}")
        print(f"\n✓ Checking old vs new (can differ slightly due to gradient tracking):")
        for i, diff in enumerate(old_new_diff_per_seq):
            status = "✓" if diff < 5e-4 else "❌"
            print(f"  {status} Seq {i}: {diff.item():.2e}")

        print(f"\nSummary:")
        print(f"  Overall max old vs ref: {old_ref_diff:.2e} (must be < 1e-6)")
        print(f"  Overall max old vs new: {old_new_diff:.2e} (must be < 5e-4)")

        # CRITICAL: old vs ref MUST be near-zero (they're the same model!)
        if old_ref_diff >= 1e-6:
            print(f"\n❌ CRITICAL ERROR: old vs ref differ by {old_ref_diff:.2e}!")
            print(f"   This means:")
            print(f"   1. Dropout is still active (model is non-deterministic), OR")
            print(f"   2. ref_policy and policy are different models, OR")
            print(f"   3. Data corruption occurred during storage")

            # Detailed analysis
            ref_problematic = (old_ref_diff_per_seq >= 1e-6).nonzero(as_tuple=True)[0]
            print(f"\n   Problematic sequences (old≠ref): {ref_problematic.tolist()}")

            for seq_idx in ref_problematic[:3]:
                seq_idx_int = seq_idx.item()
                old_seq = old_log_probs[seq_idx_int]
                ref_seq = ref_log_probs[seq_idx_int]

                old_nonzero = (old_seq != 0.0).sum().item()
                ref_nonzero = (ref_seq != 0.0).sum().item()

                print(f"\n   Seq {seq_idx_int} detail:")
                print(f"     Old non-zero: {old_nonzero}, Ref non-zero: {ref_nonzero}")
                print(f"     First 10 old: {old_seq[:10].tolist()}")
                print(f"     First 10 ref: {ref_seq[:10].tolist()}")
                print(f"     Diff:         {(old_seq[:10] - ref_seq[:10]).abs().tolist()}")

        # Identify problematic sequences (diff > 1e-3)
        problematic_seqs = (old_new_diff_per_seq > 1e-3).nonzero(as_tuple=True)[0]
        if len(problematic_seqs) > 0:
            print(f"\n  ⚠️  Found {len(problematic_seqs)} sequences with old≠new (diff > 1e-3): {problematic_seqs.tolist()}")
            print(f"  Analyzing these sequences in detail...")

            for seq_idx in problematic_seqs[:3]:  # Show first 3 problematic sequences
                seq_idx_int = seq_idx.item()
                print(f"\n  Sequence {seq_idx_int}:")

                # Check for non-zero values in old and new
                old_seq = old_log_probs[seq_idx_int]
                new_seq = new_log_probs[seq_idx_int]

                # Count non-zero positions (actual data, not padding)
                old_nonzero = (old_seq != 0.0).sum().item()
                new_nonzero = (new_seq != 0.0).sum().item()

                print(f"    Old log probs non-zero positions: {old_nonzero}")
                print(f"    New log probs non-zero positions: {new_nonzero}")

                # Check if lengths match
                if old_nonzero != new_nonzero:
                    print(f"    ❌ LENGTH MISMATCH! Old has {old_nonzero} tokens, new has {new_nonzero} tokens")
                else:
                    print(f"    ✓ Lengths match, but values differ significantly")

                # Show first few values
                print(f"    First 10 old values: {old_seq[:10].tolist()}")
                print(f"    First 10 new values: {new_seq[:10].tolist()}")
                print(f"    Difference: {(old_seq[:10] - new_seq[:10]).abs().tolist()}")

        print(f"\n{'='*80}")

        # ASSERTION: old = ref (they MUST be the same model before training!)
        assert old_ref_diff < 1e-6, (
            f"❌ ASSERTION FAILED: Old and ref log probs differ by {old_ref_diff:.2e}\n"
            f"   Expected: < 1e-6 (near-zero difference)\n"
            f"   This indicates dropout is still active or model corruption!\n"
            f"   See detailed analysis above."
        )

        # ASSERTION: old ≈ new (can differ slightly due to gradient tracking)
        assert old_new_diff < 5e-4, (
            f"❌ ASSERTION FAILED: Old and new log probs differ by {old_new_diff:.2e}\n"
            f"   Expected: < 5e-4 (small difference from gradient tracking)\n"
            f"   This indicates dropout is still active or major randomness!\n"
            f"   See detailed analysis above."
        )

        print(f"✓ All assertions passed! Logprobs are consistent.")
        print(f"{'='*80}\n")

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
            # Reshape to groups
            batch_size = rewards.shape[0]
            num_groups = batch_size // group_size
            grouped_rewards = rewards.view(num_groups, group_size)

            # Normalize within each group (baseline = group mean)
            group_mean = grouped_rewards.mean(dim=1, keepdim=True)
            group_std = grouped_rewards.std(dim=1, keepdim=True)

            # Check for very small std values
            # small_std_count = int((group_std < 1e-3).sum().item())

            # Use larger epsilon to prevent explosion when std is small
            normalized_rewards = (grouped_rewards - group_mean) / (group_std + 0.1)

            # Flatten back
            advantages = normalized_rewards.view(-1)
        else:
            # Global normalization
            advantages = (rewards - rewards.mean()) / (rewards.std() + 0.1)

        # Clip advantages to prevent extreme policy updates
        # advantages_before_clip = advantages.clone()
        advantages = torch.clamp(advantages, min=self.advantage_clip_min, max=self.advantage_clip_max)

        # clipped_count = int(((advantages_before_clip < self.advantage_clip_min) | (advantages_before_clip > self.advantage_clip_max)).sum().item())
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
        Compute GRPO loss with PPO-style clipping and entropy bonus.

        Loss = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * KL - α * H

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

        # Compute PPO-style clipped policy loss using configured function
        # (either sequence-level or token-level normalization)
        policy_loss, policy_metrics = self._compute_policy_loss_fn(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            completion_mask=completion_mask,
            clip_epsilon_low=self.clip_epsilon_low,
            clip_epsilon_high=self.clip_epsilon_high,
        )

        # KL divergence penalty (per-token) - ONLY if enabled
        # KL(π_ref || π_new) penalizes when new policy diverges from reference
        # This prevents the policy from becoming overconfident
        if self.kl_coef > 0:
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
                reduction=self.kl_reduction,
            )

        else:
            # KL penalty disabled - return zero
            kl_penalty = torch.tensor(0.0, device=new_log_probs.device)

        # Entropy bonus for exploration (per-token average to prevent gradient explosion)
        new_probs = torch.exp(new_log_probs)
        # Average over tokens (not sum!) to keep entropy and gradients scaled properly
        num_tokens = completion_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # Avoid division by zero
        entropy_per_seq = -(new_probs * new_log_probs * completion_mask).sum(dim=-1) / num_tokens.squeeze(-1)
        entropy = entropy_per_seq.mean()

        # Get entropy coefficient from config (default: 0.01)
        entropy_coef = self.config.get("training", {}).get("entropy_coef", 0.01)

        # Total GRPO loss with entropy bonus
        loss = policy_loss + self.kl_coef * kl_penalty - entropy_coef * entropy

        # Combine metrics from policy loss computation with additional metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_penalty.item(),
            "entropy": entropy.item(),
            # Policy loss metrics (from compute_ppo_policy_loss)
            "ratio_mean": policy_metrics["ratio_mean"],
            "ratio_min": policy_metrics["ratio_min"],
            "ratio_max": policy_metrics["ratio_max"],
            "ratio_clipped_frac": policy_metrics["ratio_clipped_frac"],
            "log_ratio_mean": policy_metrics["log_ratio_mean"],
            "log_ratio_min": policy_metrics["log_ratio_min"],
            "log_ratio_max": policy_metrics["log_ratio_max"],
            # Additional metrics
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "tokens_generated": completion_mask.sum().item(),
            # Loss component breakdown for debugging
            "kl_term": (self.kl_coef * kl_penalty).item(),
            "entropy_term": (entropy_coef * entropy).item(),
        }

        self.timing_manager.end_timer("loss_computation")
        return loss, metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform proper GRPO training step with multiple epochs over same trajectories.

        Training procedure:
        1. Collect all rollouts/trajectories ONCE (expensive - store in RAM)
        2. Compute group-normalized advantages
        3. Multiple epochs over SAME data (3-5 epochs):
           - Shuffle data each epoch
           - Iterate minibatches
           - Update policy per minibatch
        4. This squeezes more learning from expensive trajectory generation

        Args:
            batch: Dictionary with 'prompts' and 'answers'

        Returns:
            Dictionary of training metrics averaged across all epochs
        """
        self.policy.train()
        prompts = batch["prompts"]
        answers = batch["answers"]

        # 1. GENERATE TRAJECTORIES ONCE (expensive!)
        self.timing_manager.start_timer("trajectory_generation")
        (
            _,
            _,
            rewards,
            old_log_probs,
            ref_log_probs,
            completion_mask,
            format_rewards,
            correctness_rewards,
            generated_ids,
            attention_mask,
            prompt_end_positions,
        ) = self.generate_trajectories(
            prompts, answers=answers, store_outputs=False
        )
        self.timing_manager.end_timer("trajectory_generation")

        # 2. COMPUTE ADVANTAGES ONCE
        self.timing_manager.start_timer("advantage_computation")
        advantages = self.compute_advantages(
            rewards,
            group_size=self.group_size,
            normalize_within_groups=self.normalize_rewards
        )
        self.timing_manager.end_timer("advantage_computation")

        # Compute metrics NOW and save scalars (before optimization loop)
        # This allows us to free reward tensors early
        total_sequences = rewards.shape[0]
        reward_mean_scalar = rewards.mean().item()
        reward_std_scalar = rewards.std().item()
        format_reward_mean_scalar = format_rewards.mean().item()
        correctness_reward_mean_scalar = correctness_rewards.mean().item()
        # completion_mask is now a list of tensors - sum across all
        total_tokens_scalar = sum(mask.sum().item() for mask in completion_mask)

        # Free reward tensors before optimization loop (only need advantages now)
        del rewards, format_rewards, correctness_rewards

        self.timing_manager.start_timer("optimization")

        # Aggregate metrics across ALL epochs and minibatches
        epoch_metrics = {
            "policy_loss": 0.0,
            "kl_divergence": 0.0,
            "entropy": 0.0,
            "ratio_mean": 0.0,
            "ratio_clipped_frac": 0.0,
            "ratio_min": 0.0,
            "ratio_max": 0.0,
            "grad_norm": 0.0,  # Track gradient norm (before clipping)
            "grad_norm_post_clip": 0.0,  # Track gradient norm after clipping
            "grad_clipped_frac": 0.0,  # Track percentage of updates where gradients were clipped
            "relative_param_update": 0.0,  # Track relative parameter updates
        }
        num_updates = 0

        # Calculate total minibatches for progress logging
        num_minibatches = (total_sequences + self.minibatch_size - 1) // self.minibatch_size

        prev_cache = getattr(self.policy.model.config, "use_cache", None)
        if prev_cache is not None:
            self.policy.model.config.use_cache = False

        try:
            # 3. MULTIPLE EPOCHS OVER SAME TRAJECTORIES (3-5 typically)
            for epoch in range(self.update_epochs):
                # Shuffle rollouts for this epoch (different order each time)
                indices = torch.randperm(total_sequences, device=self.device)

                # Iterate through minibatches
                for mb_idx, mb_start in enumerate(range(0, total_sequences, self.minibatch_size), 1):
                    # Progress indicator for optimization
                    if self.log_trajectory_progress:
                        print(f"  [Epoch {epoch+1}/{self.update_epochs} | Minibatch {mb_idx}/{num_minibatches}]", end=" ", flush=True)

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}")

                    mb_end = min(mb_start + self.minibatch_size, total_sequences)
                    mb_indices = indices[mb_start:mb_end]

                    # Extract minibatch data - advantages are scalars, no padding needed
                    mb_advantages = advantages[mb_indices]

                    # Extract selected sequences from lists (variable-length)
                    selected_gen_ids = [generated_ids[idx] for idx in mb_indices]
                    selected_attn_mask = [attention_mask[idx] for idx in mb_indices]
                    selected_old_log_probs = [old_log_probs[idx] for idx in mb_indices]
                    selected_ref_log_probs = [ref_log_probs[idx] for idx in mb_indices]
                    selected_completion_mask = [completion_mask[idx] for idx in mb_indices]
                    selected_prompt_end_positions = [prompt_end_positions[idx] for idx in mb_indices]

                    # Recompute log probs with CURRENT policy
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_recompute")

                    # Pad generated_ids and attention_mask first
                    mb_generated_ids = pad_sequence(
                        selected_gen_ids,
                        batch_first=True,
                        padding_value=self.policy.tokenizer.pad_token_id
                    )
                    mb_attention_mask = pad_sequence(
                        selected_attn_mask,
                        batch_first=True,
                        padding_value=0
                    )

                    # Recompute log probs per sequence with sequence-specific logits_to_keep
                    # Stay in train() mode - dropout is disabled at initialization, so no randomness
                    new_log_probs_list = []

                    # DEBUG: Track recomputation for episode 0, minibatch 1
                    if self.episode == 0 and epoch == 0 and mb_idx == 1:
                        print(f"\n[DEBUG - Episode 0, MB1] RECOMPUTATION:")
                        print(f"  Number of sequences in minibatch: {len(selected_gen_ids)}")
                        print(f"  Minibatch indices (from full trajectory): {mb_indices.tolist()[:10]}...")

                    for seq_idx, (seq_gen_ids, seq_attn_mask, seq_old_log_probs, seq_completion_mask, seq_prompt_end_position) in enumerate(zip(
                        selected_gen_ids, selected_attn_mask, selected_old_log_probs, selected_completion_mask, selected_prompt_end_positions
                    )):
                        # Use the SAME length as stored during generation
                        # The stored tensor length equals completion length
                        max_completion_len = seq_old_log_probs.size(0)

                        # Check if completion_mask is all-zero (invalid sequence)
                        mask_valid = int(seq_completion_mask.sum().item())

                        # DEBUG: Show details for first 5 sequences AND any sequences with all-zero masks
                        if self.episode == 0 and epoch == 0 and mb_idx == 1:
                            old_nonzero = (seq_old_log_probs != 0.0).sum().item()
                            attn_valid = (seq_attn_mask > 0).sum().item()

                            # Show first 5 OR if mask is all-zero (problematic)
                            if seq_idx < 5 or mask_valid == 0:
                                print(f"  Seq {seq_idx}: stored_old_log_probs length={max_completion_len}, "
                                      f"non-zero={old_nonzero}, "
                                      f"completion_mask valid={mask_valid}, "
                                      f"attention_mask valid={attn_valid}, "
                                      f"prompt_end_position={seq_prompt_end_position.item()}")

                                # If mask is all-zero, show WHY
                                if mask_valid == 0:
                                    print(f"    ⚠️  Seq {seq_idx} has ALL-ZERO completion_mask!")
                                    print(f"       This means all completion tokens were masked out during generation")
                                    print(f"       (likely all PAD or all after EOS)")
                                    print(f"       old_log_probs should be all-zero: {old_nonzero == 0}")
                                    print(f"       SKIPPING recomputation for this sequence (will use zeros)")
                                    # Show trajectory index for cross-reference
                                    traj_idx = mb_indices[seq_idx].item()
                                    print(f"       Trajectory index: {traj_idx}")

                        # CRITICAL FIX: Skip recomputation for sequences with all-zero masks
                        # These sequences have no valid tokens, so recomputing log probs is meaningless
                        # and creates mismatches. Just use zeros (matches old_log_probs which are all-zero).
                        if mask_valid == 0:
                            # Create all-zero tensor matching the stored length
                            seq_new_log_probs = torch.zeros(max_completion_len, device=seq_old_log_probs.device, dtype=seq_old_log_probs.dtype)
                        else:
                            # Compute log probs for full sequence (model needs prompt as context)
                            full_log_probs = self.policy.compute_log_probs(
                                seq_gen_ids.unsqueeze(0),  # Add batch dim
                                attention_mask=seq_attn_mask.unsqueeze(0),  # Add batch dim
                            )  # Shape: [1, seq_len - 1]

                            # Extract ONLY completion log probs (same logic as trajectory generation)
                            # Log probs are shifted, so completion starts at (prompt_end_position - 1)
                            device = full_log_probs.device
                            completion_positions = torch.arange(max_completion_len, device=device)

                            # Source positions in full_log_probs
                            source_positions = (seq_prompt_end_position.item() - 1) + completion_positions

                            # Clamp to valid range
                            source_positions = source_positions.clamp(0, full_log_probs.size(1) - 1)

                            # Create mask for valid positions
                            valid_mask = source_positions < full_log_probs.size(1)

                            # Gather completion log probs
                            seq_new_log_probs = full_log_probs[0, source_positions]

                            # Set padding positions to 0.0
                            seq_new_log_probs = torch.where(valid_mask, seq_new_log_probs,
                                                           torch.tensor(0.0, device=device, dtype=seq_new_log_probs.dtype))

                        # Store (either computed or zeros)
                        new_log_probs_list.append(seq_new_log_probs)

                    # Pad old, ref, and mask tensors first to get target length
                    mb_old_log_probs = pad_sequence(
                        selected_old_log_probs,
                        batch_first=True,
                        padding_value=0.0
                    )
                    mb_ref_log_probs = pad_sequence(
                        selected_ref_log_probs,
                        batch_first=True,
                        padding_value=0.0
                    )
                    mb_completion_mask = pad_sequence(
                        selected_completion_mask,
                        batch_first=True,
                        padding_value=0.0
                    )

                    # Get the target length (from old_log_probs after padding)
                    target_len = mb_old_log_probs.size(1)

                    # Pad or trim new_log_probs to match target length
                    # Since we requested variable lengths, we need to ensure they match
                    padded_new_log_probs_list = []
                    for seq_new in new_log_probs_list:
                        current_len = seq_new.size(0)
                        if current_len < target_len:
                            # Pad with zeros
                            padding = torch.zeros(target_len - current_len, device=seq_new.device, dtype=seq_new.dtype)
                            padded_seq = torch.cat([seq_new, padding], dim=0)
                        elif current_len > target_len:
                            # Trim (shouldn't happen, but handle it)
                            padded_seq = seq_new[:target_len]
                        else:
                            padded_seq = seq_new
                        padded_new_log_probs_list.append(padded_seq)

                    # Stack into batch tensor
                    mb_new_log_probs = torch.stack(padded_new_log_probs_list, dim=0)

                    # CRITICAL: Delete tensor lists to free references
                    del new_log_probs_list, padded_new_log_probs_list

                    # Apply the mask to new_log_probs (use the original completion_mask from generation)
                    mb_new_log_probs = mb_new_log_probs * mb_completion_mask
                    # NOTE: old_log_probs and ref_log_probs are ALREADY masked during generation!
                    # Do NOT mask them again here, or we'll be masking zeros which changes nothing
                    # but the stored values are already correct.

                    # DEBUG: Show non-zero counts AFTER masking for first 5 sequences
                    if self.episode == 0 and epoch == 0 and mb_idx == 1:
                        print(f"\n[DEBUG - Episode 0, MB1] AFTER MASKING:")
                        for seq_idx in range(min(5, mb_new_log_probs.size(0))):
                            old_nonzero_post = (mb_old_log_probs[seq_idx] != 0.0).sum().item()
                            new_nonzero_post = (mb_new_log_probs[seq_idx] != 0.0).sum().item()
                            ref_nonzero_post = (mb_ref_log_probs[seq_idx] != 0.0).sum().item()
                            mask_valid = int(mb_completion_mask[seq_idx].sum().item())
                            print(f"  Seq {seq_idx}: old non-zero={old_nonzero_post}, new non-zero={new_nonzero_post}, "
                                  f"ref non-zero={ref_nonzero_post}, mask valid={mask_valid}")

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_recompute")

                    # Validate log probs in first minibatch to ensure dropout is disabled
                    if self.episode == 0 and epoch == 0 and mb_idx == 1:
                        # Save detailed log probs to file for visual inspection
                        import os
                        log_dir = "debug_logs"
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "logprobs_comparison_ep0_mb1.txt")

                        with open(log_file, "w") as f:
                            f.write("="*80 + "\n")
                            f.write("LOG PROBS COMPARISON - Episode 0, Minibatch 1\n")
                            f.write("="*80 + "\n\n")

                            for seq_idx in range(mb_old_log_probs.size(0)):
                                f.write(f"\n{'='*80}\n")
                                f.write(f"SEQUENCE {seq_idx}\n")
                                f.write(f"{'='*80}\n")

                                old_seq = mb_old_log_probs[seq_idx]
                                new_seq = mb_new_log_probs[seq_idx]
                                ref_seq = mb_ref_log_probs[seq_idx]
                                mask_seq = mb_completion_mask[seq_idx]

                                # Find non-zero (valid) positions
                                old_nonzero = (old_seq != 0.0).sum().item()
                                new_nonzero = (new_seq != 0.0).sum().item()
                                ref_nonzero = (ref_seq != 0.0).sum().item()
                                mask_ones = (mask_seq > 0).sum().item()

                                f.write(f"\nSequence Statistics:\n")
                                f.write(f"  Old non-zero tokens: {old_nonzero}\n")
                                f.write(f"  New non-zero tokens: {new_nonzero}\n")
                                f.write(f"  Ref non-zero tokens: {ref_nonzero}\n")
                                f.write(f"  Mask valid tokens:   {mask_ones}\n")

                                if old_nonzero != new_nonzero:
                                    f.write(f"  ❌ LENGTH MISMATCH! Old={old_nonzero}, New={new_nonzero}\n")

                                # Print token-by-token comparison
                                f.write(f"\nToken-by-Token Comparison:\n")
                                f.write(f"{'Pos':<6} {'Mask':<6} {'Old':<15} {'New':<15} {'Ref':<15} {'Diff(O-N)':<15} {'Diff(O-R)':<15}\n")
                                f.write("-"*97 + "\n")

                                max_len = max(old_seq.size(0), new_seq.size(0), ref_seq.size(0))
                                for pos in range(max_len):
                                    old_val = old_seq[pos].item() if pos < old_seq.size(0) else float('nan')
                                    new_val = new_seq[pos].item() if pos < new_seq.size(0) else float('nan')
                                    ref_val = ref_seq[pos].item() if pos < ref_seq.size(0) else float('nan')
                                    mask_val = mask_seq[pos].item() if pos < mask_seq.size(0) else 0.0

                                    old_new_diff = abs(old_val - new_val) if not (torch.isnan(torch.tensor(old_val)) or torch.isnan(torch.tensor(new_val))) else float('nan')
                                    old_ref_diff = abs(old_val - ref_val) if not (torch.isnan(torch.tensor(old_val)) or torch.isnan(torch.tensor(ref_val))) else float('nan')

                                    # Highlight if difference is large
                                    marker = " ⚠️ " if old_new_diff > 1e-3 else ""

                                    f.write(f"{pos:<6} {mask_val:<6.0f} {old_val:<15.6f} {new_val:<15.6f} {ref_val:<15.6f} {old_new_diff:<15.6e} {old_ref_diff:<15.6e} {marker}\n")

                                # Compute max differences
                                old_new_diff = (old_seq - new_seq).abs().max().item() if old_seq.size(0) == new_seq.size(0) else float('inf')
                                old_ref_diff = (old_seq - ref_seq).abs().max().item() if old_seq.size(0) == ref_seq.size(0) else float('inf')

                                f.write(f"\nMax Differences:\n")
                                f.write(f"  Old vs New: {old_new_diff:.6e}\n")
                                f.write(f"  Old vs Ref: {old_ref_diff:.6e}\n")

                        print(f"\n✓ Saved detailed log probs comparison to: {log_file}")

                        self._validate_log_probs_episode_zero(
                            mb_old_log_probs, mb_new_log_probs, mb_ref_log_probs
                        )

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Compute loss
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_loss")
                    # with self._amp_config.autocast():  # COMMENTED OUT - mixed precision disabled
                    with contextlib.nullcontext():
                        loss, mb_metrics = self.compute_loss(
                            mb_new_log_probs,
                            mb_old_log_probs,
                            mb_advantages,
                            mb_ref_log_probs,
                            mb_completion_mask,
                        )
                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_loss")

                    # Backward pass
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_backward")
                    # if self._amp_config.enabled and self._amp_config.grad_scaler is not None:  # COMMENTED OUT - mixed precision disabled
                    #     self._amp_config.grad_scaler.scale(loss).backward()
                    # else:
                    loss.backward()
                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_backward")

                    # Gradient clipping
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_clip")
                    # if self._amp_config.enabled and self._amp_config.grad_scaler is not None:  # COMMENTED OUT - mixed precision disabled
                    #     self._amp_config.grad_scaler.unscale_(self.optimizer)

                    # NOTE: clip_grad_norm_ returns the TOTAL norm BEFORE clipping
                    # The actual gradients ARE clipped to max_norm, but the return value shows the original norm
                    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), max_norm=self.gradient_clip
                    )

                    # Compute post-clip gradient norm (actual norm after clipping)
                    # Also track whether clipping occurred
                    with torch.no_grad():
                        grad_sq_sum = 0.0
                        for p in self.policy.parameters():
                            if p.grad is not None:
                                g = p.grad.detach().float()
                                grad_sq_sum += (g.norm(2)**2).item()
                        grad_norm_post_clip = (grad_sq_sum**0.5)

                        # Calculate if gradients were clipped
                        # Clipping occurs when norm_before > gradient_clip
                        grad_was_clipped = 1.0 if grad_norm_before_clip > self.gradient_clip else 0.0

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_clip")

                    # Capture parameter norm before update (for relative update calculation)
                    with torch.no_grad():
                        theta_before = torch.nn.utils.parameters_to_vector(self.policy.parameters()).float()
                        theta_norm_before = theta_before.norm().item()

                    # Parameter update
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_update")
                    # if self._amp_config.enabled and self._amp_config.grad_scaler is not None:  # COMMENTED OUT - mixed precision disabled
                    #     self._amp_config.grad_scaler.step(self.optimizer)
                    #     self._amp_config.grad_scaler.update()
                    # else:
                    self.optimizer.step()

                    # Compute relative parameter update (how much parameters changed)
                    with torch.no_grad():
                        theta_after = torch.nn.utils.parameters_to_vector(self.policy.parameters()).float()
                        param_update_norm = (theta_after - theta_before).norm().item()
                        # Relative update: ||theta_new - theta_old|| / ||theta_old||
                        relative_param_update = param_update_norm / max(theta_norm_before, 1e-12)

                        # CRITICAL: Delete large parameter vectors to prevent memory leak
                        del theta_before, theta_after

                    # Step LR scheduler if warmup is enabled
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_update")

                    # Accumulate metrics
                    for key in epoch_metrics:
                        if key in mb_metrics:
                            epoch_metrics[key] += mb_metrics[key]

                    # Accumulate gradient metrics
                    epoch_metrics["grad_norm"] += grad_norm_before_clip.item()
                    epoch_metrics["grad_norm_post_clip"] += grad_norm_post_clip
                    epoch_metrics["grad_clipped_frac"] += grad_was_clipped
                    epoch_metrics["relative_param_update"] += relative_param_update
                    num_updates += 1

                    # Debug logging for Episode 0 to track KL explosion
                    if self.episode == 0 and epoch == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"\n    [Episode 0 Debug | Minibatch {mb_idx}] LR={current_lr:.2e}")
                        print(f"      Loss Components: policy={mb_metrics['policy_loss']:.4f}, "
                              f"kl_term={mb_metrics['kl_term']:.4f}, "
                              f"entropy_term={mb_metrics['entropy_term']:.4f}")
                        print(f"      Log Ratio: min={mb_metrics['log_ratio_min']:.4f}, "
                              f"mean={mb_metrics['log_ratio_mean']:.4f}, "
                              f"max={mb_metrics['log_ratio_max']:.4f}")
                        print(f"      KL Divergence: {mb_metrics['kl_divergence']:.4f}")
                        print(f"      Grad Norm (before clip): {grad_norm_before_clip.item():.2f}")

                        # Detailed log prob verification for MB1
                        if mb_idx == 1:
                            print(f"\n      [Log Prob Verification - MB1]")

                            # Check log prob ranges
                            print(f"        new_log_probs: min={mb_new_log_probs.min().item():.4f}, "
                                  f"mean={mb_new_log_probs[mb_completion_mask > 0].mean().item():.4f}, "
                                  f"max={mb_new_log_probs.max().item():.4f}")
                            print(f"        old_log_probs: min={mb_old_log_probs.min().item():.4f}, "
                                  f"mean={mb_old_log_probs[mb_completion_mask > 0].mean().item():.4f}, "
                                  f"max={mb_old_log_probs.max().item():.4f}")

                            # CRITICAL: Check mask is working correctly
                            print(f"\n      [Mask Verification - MB1]")
                            print(f"        Completion mask shape: {mb_completion_mask.shape}")
                            print(f"        Total valid tokens: {mb_completion_mask.sum().item():.0f}")
                            print(f"        Sequence lengths: {mb_completion_mask.sum(dim=1).tolist()}")

                            # Check if masked positions have zero log probs (they should!)
                            masked_positions = mb_completion_mask == 0
                            if masked_positions.any():
                                masked_new = mb_new_log_probs[masked_positions]
                                masked_old = mb_old_log_probs[masked_positions]
                                print(f"        Masked positions exist: {masked_positions.sum().item():.0f} positions")
                                print(f"        new_log_probs at masked: min={masked_new.min().item():.4f}, "
                                      f"max={masked_new.max().item():.4f}, "
                                      f"nonzero={(masked_new != 0).sum().item()}")
                                print(f"        old_log_probs at masked: min={masked_old.min().item():.4f}, "
                                      f"max={masked_old.max().item():.4f}, "
                                      f"nonzero={(masked_old != 0).sum().item()}")

                                # WARNING: If masked positions have non-zero values, we're computing gradients through padding!
                                if (masked_new != 0).any() or (masked_old != 0).any():
                                    print(f"        ⚠️  WARNING: Masked positions have NON-ZERO log probs!")
                                    print(f"        This means gradients are flowing through padded tokens!")

                            # Check for any NaN or Inf
                            print(f"\n      [Numerical Stability - MB1]")
                            print(f"        new_log_probs has NaN: {torch.isnan(mb_new_log_probs).any().item()}")
                            print(f"        new_log_probs has Inf: {torch.isinf(mb_new_log_probs).any().item()}")
                            print(f"        Advantages: min={mb_advantages.min().item():.4f}, "
                                  f"max={mb_advantages.max().item():.4f}, "
                                  f"mean={mb_advantages.mean().item():.4f}")

                    # Cleanup - Delete ALL minibatch tensors to free memory
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_cleanup")
                    del (
                        mb_new_log_probs, loss,
                        mb_old_log_probs, mb_ref_log_probs, mb_advantages, mb_completion_mask,
                        mb_generated_ids, mb_attention_mask,
                        # Delete temporary selection lists
                        selected_gen_ids, selected_attn_mask, selected_old_log_probs,
                        selected_ref_log_probs, selected_completion_mask,
                        # Delete scalars that might hold tensor references
                        param_update_norm, relative_param_update, grad_norm_before_clip, grad_norm_post_clip
                    )
                    if self.device.type == "cuda" or (self.device.type == "mps" and self.clear_cache_on_mps):
                        clear_device_cache(self.device)
                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_cleanup")

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}")

                    # Show completion for this minibatch
                    if self.log_trajectory_progress:
                        print("✓", end="", flush=True)

                # Newline after all minibatches in this epoch complete
                if self.log_trajectory_progress:
                    print()  # Move to next line after epoch minibatches
        
        finally:
            # Restore cache config
            if prev_cache is not None:
                self.policy.model.config.use_cache = prev_cache

        self.timing_manager.end_timer("optimization")

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, num_updates)

        # Final metrics (using pre-computed scalars)
        metrics = {
            "total_loss": epoch_metrics["policy_loss"] + self.kl_coef * epoch_metrics["kl_divergence"],
            "pg_loss": epoch_metrics["policy_loss"],
            "kl_divergence": epoch_metrics["kl_divergence"],
            "reward_mean": reward_mean_scalar,
            "reward_std": reward_std_scalar,
            "format_reward_mean": format_reward_mean_scalar,
            "correctness_reward_mean": correctness_reward_mean_scalar,
            "ratio_mean": epoch_metrics["ratio_mean"],
            "ratio_min": epoch_metrics["ratio_min"],
            "ratio_max": epoch_metrics["ratio_max"],
            "ratio_clipped_frac": epoch_metrics["ratio_clipped_frac"],
            "grad_norm": epoch_metrics["grad_norm"],
            "grad_norm_post_clip": epoch_metrics["grad_norm_post_clip"],
            "grad_clipped_frac": epoch_metrics["grad_clipped_frac"],
            "relative_param_update": epoch_metrics["relative_param_update"],
            "tokens_generated": total_tokens_scalar,
        }

        # Update statistics
        self.total_steps += 1

        # Log metrics
        self.logger.log_metrics(metrics, self.total_steps)

        # Cleanup after episode
        del old_log_probs, ref_log_probs, advantages, completion_mask
        del generated_ids, attention_mask, prompt_end_positions  # Lists get garbage collected

        if self.device.type == "cuda" or (self.device.type == "mps" and self.clear_cache_on_mps):
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

        # Get generation parameters from config (with defaults)
        training_config = self.config.get('training', {})
        max_new_tokens = training_config.get('max_new_tokens', 128)
        temperature = training_config.get('temperature', 0.9)
        top_p = training_config.get('top_p', 0.9)

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
            "grad_norm": [],
            "grad_norm_post_clip": [],
            "grad_clipped_frac": [],
            "relative_param_update": [],
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

        # Determine starting episode (support resuming from checkpoint)
        start_episode = self.current_episode
        if start_episode > 0:
            print(f"\n⏭️  RESUMING FROM EPISODE {start_episode}")
            print(f"   Will train episodes {start_episode} to {num_episodes-1}")
            print(f"   Scheduler state restored (warmup may already be complete)")

        # Check if we should resample batch each episode (default: False for overfitting tests)
        # Set to True for real training to sample different batches each episode
        resample_batch = self.config.get('training', {}).get('resample_batch_per_episode', False)

        # Sample initial batch (used for all episodes if resample_batch=False)
        if not resample_batch:
            fixed_batch_indices = np.arange(min(batch_size, len(train_prompts)))
            print(f"  - Using FIXED batch (same {len(fixed_batch_indices)} samples every episode for overfitting test)")
        else:
            print(f"  - Resampling batch every episode (normal training mode)")

        for episode in range(start_episode, num_episodes):
            self.episode = episode
            self.current_episode = episode  # Update current episode counter

            # Set deterministic seed for this episode
            # This ensures episode N always has the same randomness,
            # making training reproducible even when resuming from checkpoints
            self._set_episode_seed(episode)

            episode_start_time = time.time()

            # Sample a batch of problems for this episode
            if resample_batch:
                # Normal training: sample different batch each episode
                batch_indices = np.random.choice(len(train_prompts), batch_size, replace=True)
            else:
                # Overfitting test: use same batch every episode
                batch_indices = fixed_batch_indices

            batch_prompts = [train_prompts[i] for i in batch_indices]
            batch_answers = [train_answers[i] for i in batch_indices]

            # Pass batch to train_step (which handles multiple epochs internally)
            batch_data = {
                "prompts": batch_prompts,
                "answers": batch_answers
            }
            metrics = self.train_step(batch_data)

            # Calculate episode time
            episode_time = time.time() - episode_start_time
            episode_tokens = metrics.get("tokens_generated", 0)
            cumulative_tokens += episode_tokens

            # Calculate tokens per second for this episode
            tokens_per_sec = episode_tokens / episode_time if episode_time > 0 else 0

            # Store metrics
            training_metrics["episode"].append(episode)
            training_metrics["total_loss"].append(metrics["total_loss"])
            training_metrics["pg_loss"].append(metrics["pg_loss"])
            training_metrics["kl_divergence"].append(metrics["kl_divergence"])
            training_metrics["reward_mean"].append(metrics["reward_mean"])
            training_metrics["reward_std"].append(metrics.get("reward_std", 0.0))
            training_metrics["format_reward_mean"].append(metrics["format_reward_mean"])
            training_metrics["correctness_reward_mean"].append(metrics["correctness_reward_mean"])
            training_metrics["grad_norm"].append(metrics.get("grad_norm", 0.0))
            training_metrics["grad_norm_post_clip"].append(metrics.get("grad_norm_post_clip", 0.0))
            training_metrics["grad_clipped_frac"].append(metrics.get("grad_clipped_frac", 0.0))
            training_metrics["relative_param_update"].append(metrics.get("relative_param_update", 0.0))
            training_metrics["tokens_generated"].append(metrics.get("tokens_generated", 0))
            training_metrics["episode_time"].append(episode_time)
            training_metrics["total_tokens"].append(cumulative_tokens)
            training_metrics["tokens_per_second"].append(tokens_per_sec)

            # Logging
            if episode % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                grad_norm_pre = metrics.get('grad_norm', 0.0)
                grad_norm_post = metrics.get('grad_norm_post_clip', 0.0)
                grad_clipped_pct = metrics.get('grad_clipped_frac', 0.0) * 100.0
                rel_param_update = metrics.get('relative_param_update', 0.0)

                print(f"Episode {int(episode):3d} | "
                      f"Loss: {metrics['total_loss']:7.4f} | "
                      f"PG Loss: {metrics['pg_loss']:7.4f} | "
                      f"KL: {metrics['kl_divergence']:7.4f} | "
                      f"Reward: {metrics['reward_mean']:6.3f} ± {metrics.get('reward_std', 0.0):5.3f} | "
                      f"Fmt: {metrics['format_reward_mean']:5.3f} | "
                      f"Correct: {metrics['correctness_reward_mean']:5.3f} | "
                      f"GradNorm: {grad_norm_pre:6.3f}→{grad_norm_post:6.3f} ({grad_clipped_pct:.0f}% clip) | "
                      f"ParamΔ: {rel_param_update*100:.4f}% | "
                      f"LR: {current_lr:.2e} | "
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

            # Print timing summary periodically and RESET to prevent memory leak
            if self.print_timing and episode > 0 and episode % 10 == 0:
                self.timing_manager.print_timing_summary(f"Episode {episode} Summary")
                # CRITICAL: Reset timing data to prevent memory accumulation
                self.timing_manager.reset_timings()

            # Even if timing printing is disabled, reset every 10 episodes to prevent leak
            if not self.print_timing and episode > 0 and episode % 10 == 0:
                self.timing_manager.reset_timings()

            # CRITICAL: Force cleanup at end of each episode to prevent accumulation
            # On M4 (unified memory), this is especially important
            if self.device.type in ["cuda", "mps"]:
                clear_device_cache(self.device)
            # Force Python garbage collection every episode (helps with M4 unified memory)
            import gc
            gc.collect()

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
                _, _, rewards, _, _, _, _, _, _, _, _ = self.generate_trajectories(prompts)
                total_rewards.extend(rewards.cpu().numpy())

        self.policy.train()

        return {
            "eval_reward_mean": np.mean(total_rewards),
            "eval_reward_std": np.std(total_rewards),
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        # Handle case where ref_policy may be None (when kl_coef=0)
        ref_policy_state = self.ref_policy.state_dict() if self.ref_policy is not None else None

        # Create checkpoint with scheduler state
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "ref_policy_state_dict": ref_policy_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "config": self.config,
            "total_steps": self.total_steps,
            "episode": self.episode,
            "current_episode": self.current_episode,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = load_checkpoint(path, self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        # Only load ref_policy if it exists and was saved (may be None when kl_coef=0)
        if self.ref_policy is not None and checkpoint.get("ref_policy_state_dict") is not None:
            self.ref_policy.load_state_dict(checkpoint["ref_policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler state if it exists
        if self.lr_scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.total_steps = checkpoint.get("total_steps", 0)
        self.episode = checkpoint.get("episode", 0)
        self.current_episode = checkpoint.get("current_episode", checkpoint.get("episode", 0))