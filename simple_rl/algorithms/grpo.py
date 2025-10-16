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
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import StoppingCriteria, StoppingCriteriaList

from simple_rl.algorithms.base import BaseAlgorithm
from simple_rl.utils.huggingface_wrappers import LanguageModel
from simple_rl.utils.device import get_target_device, clear_device_cache
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
        if hasattr(self.policy.model, 'config'):
            model_cfg = self.policy.model.config

            # Disable attention dropout
            if hasattr(model_cfg, 'attention_dropout') and model_cfg.attention_dropout > 0:
                model_cfg.attention_dropout = 0.0

            # Disable hidden/residual dropout (different models use different names)
            for attr in ['hidden_dropout', 'hidden_dropout_prob', 'resid_pdrop', 'dropout']:
                if hasattr(model_cfg, attr):
                    old_val = getattr(model_cfg, attr)
                    if old_val > 0:
                        setattr(model_cfg, attr, 0.0)

        # Directly disable all Dropout modules (more reliable than config)
        # This ensures dropout is disabled even if config changes don't propagate
        for _, module in self.policy.model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                if module.p > 0:
                    module.p = 0.0

        # Training config (need this first to check kl_coef)
        self.training_config = create_training_config(self.config)

        # GRPO parameters
        self.group_size = self.training_config.group_size
        self.kl_coef = self.training_config.kl_coef
        self.normalize_rewards = self.training_config.normalize_rewards

        # Create reference model for KL penalty (ONLY if kl_coef > 0)
        # Keep requires_grad=True (same as policy) for identical kernel selection
        # We simply won't call optimizer.step() on it
        if self.kl_coef > 0:
            self.ref_policy = copy.deepcopy(self.policy)
            self.ref_policy = self.ref_policy.to(self.device)
            self.ref_policy.eval()  # Keep in eval mode
            # Note: parameters still have requires_grad=True (same as policy)
            # This ensures old and ref use identical computation paths
        else:
            self.ref_policy = None

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
        
        # Logprobs batch size for chunked processing (memory optimization)
        # If None, processes entire batch at once
        self.logprobs_batch_size = self.config.get("training", {}).get("logprobs_batch_size", None)
        
        # CPU offloading for generated sequences (memory optimization)
        # If True, moves generated sequences to CPU after generation and brings chunks back for logprobs
        self.offload_generated_to_cpu = self.config.get("training", {}).get("offload_generated_to_cpu", True)

        # Logger (initialize before optimizer so it can log warmup info)
        self.logger = create_logger(self.config)
        if self.use_wandb:
            self.logger.init_wandb()

        # Optimizer and optional LR scheduler (for warmup)
        self.optimizer, self.lr_scheduler = configure_optimizer(self.policy, self.config, logger=self.logger)

        # Reward function
        self.batch_reward_fn = batch_reward_fn

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

        # Mixed precision training setup (CUDA only)
        # Note: BF16 doesn't need gradient scaling (same exponent range as FP32)
        # FP16 would need GradScaler to prevent gradient underflow
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp else None

        # Gradient scaler (only needed for FP16, not BF16)
        # BF16 has same exponent range as FP32, so no underflow issues
        model_dtype = next(self.policy.parameters()).dtype
        self.use_grad_scaler = (model_dtype == torch.float16) and self.use_amp
        if self.use_grad_scaler:
            self.grad_scaler = torch.cuda.amp.GradScaler()
        else:
            self.grad_scaler = None

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
        is_eos = (completion_ids == eos_token_id).int()

        if self.answer_end_token_id is not None:
            is_answer_end = (completion_ids == self.answer_end_token_id).int()
            is_eos = is_eos | is_answer_end

        eos_cumsum = is_eos.cumsum(dim=1)
        not_after_eos = (eos_cumsum == 0).float()

        return not_pad * not_after_eos

    def _compute_sequence_log_probs_unified(
        self,
        model: Any,
        generated_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_end_position: torch.Tensor,
        completion_mask: torch.Tensor,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Unified method to compute log probabilities for ONE sequence.

        Used by old/new/ref to ensure IDENTICAL:
        - Input processing
        - Forward pass configuration
        - Output extraction
        - Padding/masking logic
        - Shape and dtype

        Args:
            model: LanguageModel instance (policy or ref_policy)
            generated_ids: [seq_len] token IDs
            attention_mask: [seq_len] attention mask
            prompt_end_position: scalar tensor (where completion starts)
            completion_mask: [max_completion_len] binary mask
            requires_grad: whether to enable gradients

        Returns:
            [max_completion_len] tensor of log probs (zeros where masked)
        """
        max_len = completion_mask.size(0)
        device = generated_ids.device

        # Fast path: empty completion
        if int(completion_mask.sum().item()) == 0:
            return torch.zeros(max_len, device=device, dtype=torch.float32)

        # Compute full sequence log probs
        # Note: compute_log_probs returns [1, seq_len-1] because it does next-token prediction
        full_log_probs = model.compute_log_probs(
            generated_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )  # [1, S-1] in fp32

        # Extract completion-aligned log probs
        # prompt_end_position points to first completion token in generated_ids
        # full_log_probs[i] predicts generated_ids[i+1], so we need to offset by -1
        completion_positions = torch.arange(max_len, device=device)
        source_positions = (prompt_end_position.item() - 1) + completion_positions

        # Clamp and track validity
        source_positions_clamped = source_positions.clamp(0, full_log_probs.size(1) - 1)
        valid_mask = source_positions < full_log_probs.size(1)

        # Gather log probs
        seq_log_probs = full_log_probs[0, source_positions_clamped]

        # Zero out invalid positions (beyond sequence length)
        seq_log_probs = torch.where(
            valid_mask,
            seq_log_probs,
            torch.tensor(0.0, device=device, dtype=seq_log_probs.dtype)
        )

        # Apply completion mask (zeros out padding and post-EOS tokens)
        seq_log_probs = seq_log_probs * completion_mask.to(seq_log_probs.dtype)

        return seq_log_probs

    def _compute_batch_log_probs_vectorized(
        self,
        model: Any,
        generated_ids: List[torch.Tensor],
        attention_mask: List[torch.Tensor],
        prompt_end_positions: torch.Tensor,
        completion_mask: List[torch.Tensor],
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Fully vectorized batch computation of log probabilities with chunking support.

        Processes sequences in chunks to reduce memory usage.
        Uses self.logprobs_batch_size to control chunk size.

        Args:
            model: LanguageModel instance (policy or ref_policy)
            generated_ids: List of [seq_len] token ID tensors
            attention_mask: List of [seq_len] attention mask tensors
            prompt_end_positions: [batch_size] tensor of prompt end positions
            completion_mask: List of [completion_len] mask tensors
            requires_grad: Whether to compute gradients (True for new, False for old/ref)

        Returns:
            [batch_size, max_completion_len] tensor of log probs (right-padded, no prompts)
        """
        batch_size = len(generated_ids)
        # Use model's device, not tensor device (tensors may be on CPU due to offloading)
        device = next(model.model.parameters()).device

        # Find max completion length for padding
        max_completion_len = max(mask.size(0) for mask in completion_mask)

        # Determine chunk size
        chunk_size = self.logprobs_batch_size if self.logprobs_batch_size is not None else batch_size

        # Process in chunks if chunk_size < batch_size
        if chunk_size < batch_size:
            all_completion_log_probs = []
            
            for chunk_start in range(0, batch_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, batch_size)
                
                # Get chunk
                chunk_gen_ids = generated_ids[chunk_start:chunk_end]
                chunk_attn_mask = attention_mask[chunk_start:chunk_end]
                chunk_prompt_end_pos = prompt_end_positions[chunk_start:chunk_end]
                chunk_completion_mask = completion_mask[chunk_start:chunk_end]
                
                # Process chunk
                chunk_log_probs = self._compute_chunk_log_probs(
                    model=model,
                    generated_ids=chunk_gen_ids,
                    attention_mask=chunk_attn_mask,
                    prompt_end_positions=chunk_prompt_end_pos,
                    completion_mask=chunk_completion_mask,
                    max_completion_len=max_completion_len,
                    requires_grad=requires_grad,
                    device=device,
                )
                
                all_completion_log_probs.append(chunk_log_probs)
            
            # Concatenate all chunks
            return torch.cat(all_completion_log_probs, dim=0)
        else:
            # Process entire batch at once
            return self._compute_chunk_log_probs(
                model=model,
                generated_ids=generated_ids,
                attention_mask=attention_mask,
                prompt_end_positions=prompt_end_positions,
                completion_mask=completion_mask,
                max_completion_len=max_completion_len,
                requires_grad=requires_grad,
                device=device,
            )

    def _compute_chunk_log_probs(
        self,
        model: Any,
        generated_ids: List[torch.Tensor],
        attention_mask: List[torch.Tensor],
        prompt_end_positions: torch.Tensor,
        completion_mask: List[torch.Tensor],
        max_completion_len: int,
        requires_grad: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute log probs for a single chunk of sequences.
        
        This is the core vectorized computation used by _compute_batch_log_probs_vectorized.
        """
        from torch.nn.utils.rnn import pad_sequence
        
        chunk_size = len(generated_ids)

        # Move tensors to GPU if they're on CPU (due to offloading)
        # This is necessary for torch.compile compatibility
        if generated_ids[0].device != device:
            generated_ids = [g.to(device) for g in generated_ids]
            attention_mask = [a.to(device) for a in attention_mask]
            prompt_end_positions = prompt_end_positions.to(device)
            completion_mask = [c.to(device) for c in completion_mask]

        # Pad all sequences to same length for batch processing
        padded_ids = pad_sequence(generated_ids, batch_first=True, padding_value=self.policy.tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # Pad completion masks to max_completion_len (vectorized, no loop)
        padded_completion_mask = pad_sequence(completion_mask, batch_first=True, padding_value=0)
        
        # Ensure completion mask has correct length
        if padded_completion_mask.size(1) < max_completion_len:
            padding = torch.zeros(
                chunk_size, 
                max_completion_len - padded_completion_mask.size(1),
                device=device,
                dtype=padded_completion_mask.dtype
            )
            padded_completion_mask = torch.cat([padded_completion_mask, padding], dim=1)
        elif padded_completion_mask.size(1) > max_completion_len:
            padded_completion_mask = padded_completion_mask[:, :max_completion_len]

        # Compute log probs for chunk (vectorized)
        # This returns [chunk_size, seq_len-1] in fp32
        with torch.set_grad_enabled(requires_grad):
            chunk_log_probs = model.compute_log_probs(
                padded_ids,
                attention_mask=padded_attention_mask,
            )  # [C, S-1] in fp32

        # Extract completion portions (vectorized)
        # Create batch indices and position indices for gathering
        batch_indices = torch.arange(chunk_size, device=device).unsqueeze(1)  # [C, 1]
        completion_positions = torch.arange(max_completion_len, device=device).unsqueeze(0)  # [1, L]

        # Compute source positions for each sequence
        # prompt_end_positions[i] points to first completion token in generated_ids[i]
        # chunk_log_probs[i, j] predicts token at position j+1, so offset by -1
        source_positions = (prompt_end_positions.unsqueeze(1) - 1) + completion_positions  # [C, L]

        # Clamp positions to valid range
        max_valid_position = chunk_log_probs.size(1) - 1
        source_positions_clamped = source_positions.clamp(0, max_valid_position)

        # Track valid positions (before sequence end)
        valid_mask = source_positions < chunk_log_probs.size(1)  # [C, L]

        # Gather log probs using advanced indexing (fully vectorized)
        completion_log_probs = chunk_log_probs[batch_indices, source_positions_clamped]  # [C, L]

        # Zero out invalid positions (beyond sequence length)
        completion_log_probs = torch.where(
            valid_mask,
            completion_log_probs,
            torch.zeros(1, device=device, dtype=completion_log_probs.dtype)
        )

        # Apply completion mask (zeros out padding and post-EOS tokens)
        completion_log_probs = completion_log_probs * padded_completion_mask

        return completion_log_probs


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

        replicated_prompt_ids = batch_prompt_ids.repeat_interleave(
            self.group_size, dim=0
        )
        replicated_prompt_mask = batch_prompt_mask.repeat_interleave(
            self.group_size, dim=0
        )
        self.timing_manager.end_timer("prompt_replication")

        self.timing_manager.start_timer("batch_text_generation")

        eos_token_id = self.policy.tokenizer.eos_token_id
        if self.answer_end_token_id is not None:
            eos_token_id = [eos_token_id, self.answer_end_token_id]

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

        with torch.no_grad():
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

        device = generated_ids.device
        total_sequences, seq_len = generated_ids.shape
        prompt_start_index = batch_prompt_mask.shape[1] - batch_prompt_mask.sum(-1)
        prompt_start_index_per_seq = prompt_start_index.repeat_interleave(self.group_size).to(device)
        max_content_length = seq_len - prompt_start_index_per_seq.min().item()

        batch_indices = torch.arange(total_sequences, device=device).unsqueeze(1)
        seq_positions = torch.arange(max_content_length, device=device).unsqueeze(0)
        source_positions = prompt_start_index_per_seq.unsqueeze(1) + seq_positions
        source_positions = source_positions.clamp(0, seq_len - 1)
        valid_mask = seq_positions < (seq_len - prompt_start_index_per_seq.unsqueeze(1))

        right_padded_ids = generated_ids[batch_indices, source_positions]
        right_padded_mask = generated_mask[batch_indices, source_positions]
        right_padded_ids = torch.where(valid_mask, right_padded_ids,
                                       torch.tensor(self.policy.tokenizer.pad_token_id, dtype=generated_ids.dtype, device=device))
        right_padded_mask = torch.where(valid_mask, right_padded_mask,
                                        torch.zeros(1, dtype=generated_mask.dtype, device=device))

        batch_prompt_lengths = batch_prompt_mask.sum(dim=1)
        prompt_lengths_per_seq = batch_prompt_lengths.repeat_interleave(self.group_size).to(device)
        max_completion_length = max_content_length - prompt_lengths_per_seq.max().item()

        batch_indices = torch.arange(total_sequences, device=device).unsqueeze(1)
        completion_positions = torch.arange(max_completion_length, device=device).unsqueeze(0)
        source_positions = prompt_lengths_per_seq.unsqueeze(1) + completion_positions
        source_positions = source_positions.clamp(0, max_content_length - 1)
        valid_mask = source_positions < max_content_length

        completion_ids = right_padded_ids[batch_indices, source_positions]
        completion_mask = right_padded_mask[batch_indices, source_positions]
        completion_ids = torch.where(valid_mask, completion_ids,
                                     torch.tensor(self.policy.tokenizer.pad_token_id, dtype=generated_ids.dtype, device=device))
        completion_mask = torch.where(valid_mask, completion_mask,
                                      torch.zeros(1, dtype=generated_mask.dtype, device=device))

        completion_texts = self.policy.decode(completion_ids)
        self.timing_manager.end_timer("completion_extraction")

        if self.episode == 0:
            for seq_idx in range(min(3, total_sequences)):
                full_generated_text = self.policy.tokenizer.decode(
                    generated_ids[seq_idx], skip_special_tokens=True
                )

                prompt_idx = seq_idx // self.group_size
                prompt_text = prompts[prompt_idx]
                assert full_generated_text.startswith(prompt_text), (
                    f"Seq {seq_idx}: Generated text doesn't start with prompt"
                )

        return {
            "generated_ids": right_padded_ids,
            "generated_mask": right_padded_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "completion_texts": completion_texts,
            "prompt_end_positions": prompt_lengths_per_seq,
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
            (prompts, completions, rewards, completion_mask,
             format_rewards, correctness_rewards, generated_ids, attention_mask, prompt_end_positions)
        """
        store = self.store_completions if store_outputs is None else store_outputs

        num_prompts = len(prompts)
        shuffle_indices = torch.randperm(num_prompts).tolist()
        prompts = [prompts[i] for i in shuffle_indices]
        if answers is not None:
            answers = [answers[i] for i in shuffle_indices]

        all_prompts = [] if store else None
        all_completions = [] if store else None
        all_rewards = []
        all_format_rewards = []
        all_correctness_rewards = []
        all_completion_mask = []

        all_generated_ids = []
        all_generated_mask = []
        all_prompt_end_positions = []

        num_batches = (num_prompts + self.rollout_batch_size - 1) // self.rollout_batch_size

        for batch_start in range(0, num_prompts, self.rollout_batch_size):
            batch_idx = batch_start // self.rollout_batch_size + 1

            if self.log_trajectory_progress:
                self.logger.info(f"  [Trajectory {batch_idx}/{num_batches}]")
            
            self.timing_manager.start_timer(f"batch_{batch_idx}")

            batch_end = min(batch_start + self.rollout_batch_size, num_prompts)
            batch_prompts = prompts[batch_start:batch_end]
            batch_answers = answers[batch_start:batch_end] if answers else None

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

            # Offload to CPU to save GPU memory (if enabled)
            if self.offload_generated_to_cpu and self.device.type == "cuda":
                all_generated_ids.append(generated_ids.detach().cpu())
                all_generated_mask.append(generated_mask.detach().cpu())
                all_prompt_end_positions.append(replicated_prompt_end_positions.detach().cpu())
            else:
                all_generated_ids.append(generated_ids.detach())
                all_generated_mask.append(generated_mask.detach())
                all_prompt_end_positions.append(replicated_prompt_end_positions.detach())

            # Store completion masks for later use (also offload if enabled)
            for seq_idx in range(total_sequences):
                if self.offload_generated_to_cpu and self.device.type == "cuda":
                    all_completion_mask.append(completion_mask[seq_idx, :].detach().cpu())
                else:
                    all_completion_mask.append(completion_mask[seq_idx, :].detach())

            self.timing_manager.start_timer(f"batch_{batch_idx}_rewards")
            for prompt_idx, (prompt, answer) in enumerate(zip(batch_prompts, batch_answers)):
                start_idx = prompt_idx * self.group_size
                end_idx = start_idx + self.group_size
                group_completions = completion_texts[start_idx:end_idx]

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

            del generated_ids, generated_mask, completion_ids
            if self.device.type == "cuda" or (self.device.type == "mps" and self.clear_cache_on_mps):
                clear_device_cache(self.device)

            self.timing_manager.end_timer(f"batch_{batch_idx}")

        generated_ids = []
        attention_mask = []
        for batch_ids, batch_mask in zip(all_generated_ids, all_generated_mask):
            for seq_idx in range(batch_ids.size(0)):
                generated_ids.append(batch_ids[seq_idx].clone())
                attention_mask.append(batch_mask[seq_idx].clone())

        prompt_end_positions = torch.cat(all_prompt_end_positions, dim=0)

        self.timing_manager.start_timer("tensor_stacking")
        all_rewards = torch.cat(all_rewards, dim=0)
        all_format_rewards = torch.cat(all_format_rewards, dim=0)
        all_correctness_rewards = torch.cat(all_correctness_rewards, dim=0)
        self.timing_manager.end_timer("tensor_stacking")

        if store:
            return (
                all_prompts,
                all_completions,
                all_rewards,
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
            all_completion_mask,
            all_format_rewards,
            all_correctness_rewards,
            generated_ids,
            attention_mask,
            prompt_end_positions,
        )

    def compute_logprobs(
        self,
        logprob_type: str,  # "old", "ref", or "new"
        generated_ids: List[torch.Tensor],
        attention_mask: List[torch.Tensor],
        prompt_end_positions: torch.Tensor,
        completion_mask: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Unified method to compute ANY type of logprobs (old, ref, or new).

        This single method ensures all logprob types use IDENTICAL computation.

        Args:
            logprob_type: Type of logprobs to compute ("old", "ref", or "new")
            generated_ids: List of [seq_len] token ID tensors
            attention_mask: List of [seq_len] attention mask tensors
            prompt_end_positions: [batch_size] tensor of prompt end positions
            completion_mask: List of [completion_len] mask tensors

        Returns:
            [batch_size, max_completion_len] tensor of log probs
        """

        # Start timing
        self.timing_manager.start_timer(f"compute_{logprob_type}_logprobs")

        # Select model based on type
        if logprob_type == "ref":
            if self.kl_coef > 0 and self.ref_policy is not None:
                model = self.ref_policy
            else:
                # If no KL penalty or ref model, return zeros
                batch_size = len(completion_mask)
                max_completion_len = max(mask.size(0) for mask in completion_mask)
                self.timing_manager.end_timer(f"compute_{logprob_type}_logprobs")
                return torch.zeros(batch_size, max_completion_len, device=self.device)
        else:
            # Both "old" and "new" use the policy model
            model = self.policy

        # Save model state
        prev_mode = model.training
        prev_cache = getattr(model.model.config, "use_cache", None)

        # Force eval() for ALL logprob types to ensure identical forward path
        # This prevents training/inference kernel divergence (e.g., Flash Attention behavior)
        model.eval()
        
        # Always disable cache for consistency
        if prev_cache is not None:
            model.model.config.use_cache = False

        try:
            # CRITICAL: Compute ALL with torch.enable_grad() for consistent kernels
            # Both policy and ref_policy have requires_grad=True (same state)
            # This ensures old/new/ref use identical computation paths
            with torch.enable_grad():
                # NO autocast - logprobs must be computed in FP32 for numerical stability
                log_probs = self._compute_batch_log_probs_vectorized(
                    model=model,
                    generated_ids=generated_ids,
                    attention_mask=attention_mask,
                    prompt_end_positions=prompt_end_positions,
                    completion_mask=completion_mask,
                    requires_grad=True,  # Always True for consistent kernels
                )
        finally:
            # Restore model state
            if prev_cache is not None:
                model.model.config.use_cache = prev_cache
            if prev_mode:
                model.train()

        self.timing_manager.end_timer(f"compute_{logprob_type}_logprobs")

        # For old/ref: detach to remove gradients
        # For new: keep gradients for backward pass
        if logprob_type in ["old", "ref"]:
            return log_probs.detach()
        else:
            return log_probs  # Keep gradients for new

    def _validate_logprobs_episode_zero(
        self,
        mb_old_log_probs: torch.Tensor,
        mb_new_log_probs: torch.Tensor,
        mb_ref_log_probs: torch.Tensor,
        selected_gen_ids: List[torch.Tensor] = None,
        selected_prompt_end_positions: torch.Tensor = None,
    ) -> None:
        """
        Validate that old/new/ref logprobs are consistent (episode 0 check).
        
        This ensures the unified logprob computation produces identical results
        and that gradient flow is set up correctly.
        """
        from logprobs_debugger import validate_logprobs, save_logprobs_debug
        
        print("\n" + "="*80)
        print("LOGPROBS VALIDATION - EPISODE 0, MINIBATCH 1")
        print("="*80)
        
        # Validate
        model_dtype = next(self.policy.parameters()).dtype
        results = validate_logprobs(
            mb_old_log_probs,
            mb_new_log_probs.detach(),  # Detach for comparison
            mb_ref_log_probs,
            model_dtype,
            save_on_fail=True,
            generated_ids=selected_gen_ids,
            prompt_end_positions=selected_prompt_end_positions,
            tokenizer=self.policy.tokenizer,
            output_dir=".",
        )
        
        # Report results
        print(f"Old-Ref diff: {results['old_ref_diff']:.6e} (threshold: {results['ref_threshold']:.0e})")
        print(f"Old-New diff: {results['old_new_diff']:.6e} (threshold: {results['new_threshold']:.0e})")
        print()
        
        if results['old_ref_pass']:
            print(f"✅ PASSED: Old-Ref consistency check")
        else:
            print(f"❌ FAILED: Old-Ref diff {results['old_ref_diff']:.2e} >= threshold {results['ref_threshold']:.0e}")
            raise AssertionError(
                f"Old and ref log probs differ by {results['old_ref_diff']:.2e} "
                f"(expected < {results['ref_threshold']:.0e})"
            )
        
        if results['old_new_pass']:
            print(f"✅ PASSED: Old-New consistency check")
        else:
            print(f"❌ FAILED: Old-New diff {results['old_new_diff']:.2e} >= threshold {results['new_threshold']:.0e}")
            raise AssertionError(
                f"Old and new log probs differ by {results['old_new_diff']:.2e} "
                f"(expected < {results['new_threshold']:.0e})"
            )
        
        print("="*80)
        print("✅ ALL VALIDATIONS PASSED")
        print("="*80)
        print()

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int = 1,
        normalize_within_groups: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute group-normalized advantages (GRPO's key innovation)."""
        if normalize_within_groups and group_size > 1:
            batch_size = rewards.shape[0]
            num_groups = batch_size // group_size
            grouped_rewards = rewards.view(num_groups, group_size)
            group_mean = grouped_rewards.mean(dim=1, keepdim=True)
            group_std = grouped_rewards.std(dim=1, keepdim=True)
            normalized_rewards = (grouped_rewards - group_mean) / (group_std + 0.1)
            advantages_raw = normalized_rewards.view(-1)
        else:
            advantages_raw = (rewards - rewards.mean()) / (rewards.std() + 0.1)

        # Track statistics before clamping
        adv_min_raw = advantages_raw.min().item()
        adv_max_raw = advantages_raw.max().item()

        # Clamp advantages
        advantages = torch.clamp(advantages_raw, min=self.advantage_clip_min, max=self.advantage_clip_max)

        # Compute clipping statistics
        clamped_low = (advantages_raw < self.advantage_clip_min).sum().item()
        clamped_high = (advantages_raw > self.advantage_clip_max).sum().item()
        total = advantages_raw.numel()
        clamped_low_frac = clamped_low / max(total, 1)
        clamped_high_frac = clamped_high / max(total, 1)

        stats = {
            "advantages_min_raw": adv_min_raw,
            "advantages_max_raw": adv_max_raw,
            "advantages_min": advantages.min().item(),
            "advantages_max": advantages.max().item(),
            "advantages_clamped_low_frac": clamped_low_frac,
            "advantages_clamped_high_frac": clamped_high_frac,
        }

        return advantages, stats

    def compute_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss with PPO-style clipping and entropy bonus."""
        self.timing_manager.start_timer("loss_computation")

        policy_loss, policy_metrics = self._compute_policy_loss_fn(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            completion_mask=completion_mask,
            clip_epsilon_low=self.clip_epsilon_low,
            clip_epsilon_high=self.clip_epsilon_high,
        )

        if self.kl_coef > 0:
            ref_log_probs_detached = ref_log_probs.detach()

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
            kl_penalty = torch.tensor(0.0, device=new_log_probs.device)

        new_probs = torch.exp(new_log_probs)
        num_tokens = completion_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        entropy_per_seq = -(new_probs * new_log_probs * completion_mask).sum(dim=-1) / num_tokens.squeeze(-1)
        entropy = entropy_per_seq.mean()
        entropy_coef = self.config.get("training", {}).get("entropy_coef", 0.01)
        loss = policy_loss + self.kl_coef * kl_penalty - entropy_coef * entropy

        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_penalty.item(),
            "entropy": entropy.item(),
            "ratio_mean": policy_metrics["ratio_mean"],
            "ratio_min": policy_metrics["ratio_min"],
            "ratio_max": policy_metrics["ratio_max"],
            "ratio_clipped_frac": policy_metrics["ratio_clipped_frac"],
            "log_ratio_mean": policy_metrics["log_ratio_mean"],
            "log_ratio_min": policy_metrics["log_ratio_min"],
            "log_ratio_max": policy_metrics["log_ratio_max"],
            "tokens_generated": completion_mask.sum().item(),
            "kl_term": (self.kl_coef * kl_penalty).item(),
            "entropy_term": (entropy_coef * entropy).item(),
        }

        self.timing_manager.end_timer("loss_computation")
        return loss, metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform GRPO training step with multiple epochs over same trajectories."""
        self.policy.train()
        prompts = batch["prompts"]
        answers = batch["answers"]

        self.timing_manager.start_timer("trajectory_generation")
        (
            _,
            _,
            rewards,
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

        # Compute frozen old/ref logprobs once, right before optimization
        # Use unified method to ensure IDENTICAL computation for all types
        old_log_probs = self.compute_logprobs(
            "old", generated_ids, attention_mask, prompt_end_positions, completion_mask
        )
        ref_log_probs = self.compute_logprobs(
            "ref", generated_ids, attention_mask, prompt_end_positions, completion_mask
        )

        self.timing_manager.start_timer("advantage_computation")
        advantages, advantage_stats = self.compute_advantages(
            rewards,
            group_size=self.group_size,
            normalize_within_groups=self.normalize_rewards
        )
        self.timing_manager.end_timer("advantage_computation")

        total_sequences = rewards.shape[0]
        reward_mean_scalar = rewards.mean().item()
        reward_std_scalar = rewards.std().item()
        format_reward_mean_scalar = format_rewards.mean().item()
        correctness_reward_mean_scalar = correctness_rewards.mean().item()
        total_tokens_scalar = sum(mask.sum().item() for mask in completion_mask)
        advantages_mean_scalar = advantages.mean().item()
        advantages_std_scalar = advantages.std().item()
        del rewards, format_rewards, correctness_rewards

        self.timing_manager.start_timer("optimization")

        epoch_metrics = {
            "policy_loss": 0.0,
            "kl_divergence": 0.0,
            "entropy": 0.0,
            "ratio_mean": 0.0,
            "ratio_clipped_frac": 0.0,
            "ratio_min": 0.0,
            "ratio_max": 0.0,
            "log_ratio_mean": 0.0,
            "log_ratio_min": 0.0,
            "log_ratio_max": 0.0,
            "grad_norm": 0.0,
            "grad_norm_post_clip": 0.0,
            "grad_clipped_frac": 0.0,
            "relative_param_update": 0.0,
        }
        num_updates = 0
        num_minibatches = (total_sequences + self.minibatch_size - 1) // self.minibatch_size

        prev_cache = getattr(self.policy.model.config, "use_cache", None)
        if prev_cache is not None:
            self.policy.model.config.use_cache = False

        
        try:
            for epoch in range(self.update_epochs):
                indices = torch.randperm(total_sequences, device=self.device)

                for mb_idx, mb_start in enumerate(range(0, total_sequences, self.minibatch_size), 1):
                    if self.log_trajectory_progress:
                        self.logger.info(f"  [Epoch {epoch+1}/{self.update_epochs} | Minibatch {mb_idx}/{num_minibatches}]")

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}")

                    mb_end = min(mb_start + self.minibatch_size, total_sequences)
                    mb_indices = indices[mb_start:mb_end]
                    mb_advantages = advantages[mb_indices]

                    # Select minibatch data
                    selected_gen_ids = [generated_ids[idx] for idx in mb_indices]
                    selected_attn_mask = [attention_mask[idx] for idx in mb_indices]
                    selected_completion_mask = [completion_mask[idx] for idx in mb_indices]
                    selected_prompt_end_positions = prompt_end_positions[mb_indices]

                    # Extract minibatch slices from batch tensors
                    mb_old_log_probs = old_log_probs[mb_indices]
                    mb_ref_log_probs = ref_log_probs[mb_indices]

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_recompute")

                    # NO autocast - logprobs must be computed in FP32 for numerical stability
                    # Compute new log probs using unified method (same as old/ref)
                    mb_new_log_probs = self.compute_logprobs(
                        "new",  # Enable gradients for policy update
                        selected_gen_ids,
                        selected_attn_mask,
                        selected_prompt_end_positions,
                        selected_completion_mask,
                    )

                    # Create padded completion mask to match log probs shape (vectorized, no loop)
                    mb_completion_mask = pad_sequence(
                        selected_completion_mask,
                        batch_first=True,
                        padding_value=0
                    )

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_recompute")

                    if self.episode == 0 and epoch == 0 and mb_idx == 1:
                        # Validate logprobs consistency (saves debug files on failure)
                        self._validate_logprobs_episode_zero(
                            mb_old_log_probs,
                            mb_new_log_probs,
                            mb_ref_log_probs,
                            selected_gen_ids,
                            selected_prompt_end_positions,
                        )

                    self.optimizer.zero_grad()

                    # Loss computation (NO autocast - already in FP32)
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_loss")
                    loss, mb_metrics = self.compute_loss(
                        mb_new_log_probs,
                        mb_old_log_probs,
                        mb_advantages,
                        mb_ref_log_probs,
                        mb_completion_mask,
                    )
                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_loss")

                    # Backward pass with optional gradient scaling (FP16 only, not BF16)
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_backward")
                    if self.use_grad_scaler:
                        # FP16: Scale loss to prevent gradient underflow
                        self.grad_scaler.scale(loss).backward()
                    else:
                        # BF16/FP32: No scaling needed
                        loss.backward()
                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_backward")

                    # Gradient clipping (with scaler support for FP16)
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_clip")
                    if self.use_grad_scaler:
                        # FP16: Unscale gradients before clipping
                        self.grad_scaler.unscale_(self.optimizer)

                    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), max_norm=self.gradient_clip
                    )

                    with torch.no_grad():
                        grad_sq_sum = 0.0
                        for p in self.policy.parameters():
                            if p.grad is not None:
                                g = p.grad.detach().float()
                                grad_sq_sum += (g.norm(2)**2).item()
                        grad_norm_post_clip = (grad_sq_sum**0.5)
                        grad_was_clipped = 1.0 if grad_norm_before_clip > self.gradient_clip else 0.0

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_clip")

                    with torch.no_grad():
                        theta_before = torch.nn.utils.parameters_to_vector(self.policy.parameters()).float()
                        theta_norm_before = theta_before.norm().item()

                    # Optimizer step (with scaler support for FP16)
                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_update")
                    if self.use_grad_scaler:
                        # FP16: Scaled optimizer step + scaler update
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        # BF16/FP32: Normal optimizer step
                        self.optimizer.step()

                    with torch.no_grad():
                        theta_after = torch.nn.utils.parameters_to_vector(self.policy.parameters()).float()
                        param_update_norm = (theta_after - theta_before).norm().item()
                        relative_param_update = param_update_norm / max(theta_norm_before, 1e-12)
                        del theta_before, theta_after

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_update")

                    for key in epoch_metrics:
                        if key in mb_metrics:
                            epoch_metrics[key] += mb_metrics[key]

                    epoch_metrics["grad_norm"] += grad_norm_before_clip.item()
                    epoch_metrics["grad_norm_post_clip"] += grad_norm_post_clip
                    epoch_metrics["grad_clipped_frac"] += grad_was_clipped
                    epoch_metrics["relative_param_update"] += relative_param_update
                    num_updates += 1

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_cleanup")
                    del (
                        mb_new_log_probs, loss, mb_old_log_probs, mb_ref_log_probs, mb_advantages,
                        mb_completion_mask,
                        selected_gen_ids, selected_attn_mask, selected_completion_mask,
                        selected_prompt_end_positions,
                        param_update_norm, relative_param_update, grad_norm_before_clip, grad_norm_post_clip
                    )
                    if self.device.type == "cuda" or (self.device.type == "mps" and self.clear_cache_on_mps):
                        clear_device_cache(self.device)
                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_cleanup")

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}")

        finally:
            if prev_cache is not None:
                self.policy.model.config.use_cache = prev_cache

        self.timing_manager.end_timer("optimization")

        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, num_updates)

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
            "log_ratio_mean": epoch_metrics["log_ratio_mean"],
            "log_ratio_min": epoch_metrics["log_ratio_min"],
            "log_ratio_max": epoch_metrics["log_ratio_max"],
            "grad_norm": epoch_metrics["grad_norm"],
            "grad_norm_post_clip": epoch_metrics["grad_norm_post_clip"],
            "grad_clipped_frac": epoch_metrics["grad_clipped_frac"],
            "relative_param_update": epoch_metrics["relative_param_update"],
            "tokens_generated": total_tokens_scalar,
            # Advantage statistics (from full batch)
            "advantages_mean": advantages_mean_scalar,
            "advantages_std": advantages_std_scalar,
            "advantages_min_raw": advantage_stats["advantages_min_raw"],
            "advantages_max_raw": advantage_stats["advantages_max_raw"],
            "advantages_min": advantage_stats["advantages_min"],
            "advantages_max": advantage_stats["advantages_max"],
            "advantages_clamped_low_frac": advantage_stats["advantages_clamped_low_frac"],
            "advantages_clamped_high_frac": advantage_stats["advantages_clamped_high_frac"],
        }

        self.total_steps += 1
        self.logger.log_metrics(metrics, self.total_steps)

        del old_log_probs, ref_log_probs, advantages, completion_mask
        del generated_ids, attention_mask, prompt_end_positions

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

        self.logger.info(f"Starting proper GRPO training for {num_episodes} episodes...")
        # Show asymmetric clipping if different, otherwise show symmetric
        if self.clip_epsilon_low == self.clip_epsilon_high:
            self.logger.info(f"  - Using PPO clipped objective (clip_epsilon={self.clip_epsilon})")
        else:
            self.logger.info(f"  - Using PPO clipped objective (asymmetric: low={self.clip_epsilon_low}, high={self.clip_epsilon_high})")
            self.logger.info(f"    Ratio clipped to [{1-self.clip_epsilon_low:.2f}, {1+self.clip_epsilon_high:.2f}]")
        self.logger.info(f"  - Training samples: {len(train_prompts)}")
        self.logger.info(f"  - Batch size: {batch_size}")
        self.logger.info(f"  - Update epochs: {update_epochs}")
        self.logger.info(f"  - Group size: {self.group_size}")
        self.logger.info(f"  - Rollout batch size: {self.rollout_batch_size} prompts (OOM prevention)")
        self.logger.info(f"  - Minibatch size: {self.minibatch_size} rollouts per update")
        if validation_enabled:
            self.logger.info(f"  - Validation enabled: every {validation_interval} episodes")
            self.logger.info(f"    - Evaluating {validation_num_samples} validation samples")
            self.logger.info(f"    - Showing {validation_num_demo_examples} demo examples")
        self.logger.info("=" * 50)

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
            self.logger.info(f"\n⏭️  RESUMING FROM EPISODE {start_episode}")
            self.logger.info(f"   Will train episodes {start_episode} to {num_episodes-1}")
            self.logger.info(f"   Scheduler state restored (warmup may already be complete)")

        # Check if we should resample batch each episode (default: False for overfitting tests)
        # Set to True for real training to sample different batches each episode
        resample_batch = self.config.get('training', {}).get('resample_batch_per_episode', False)

        # Sample initial batch (used for all episodes if resample_batch=False)
        if not resample_batch:
            fixed_batch_indices = np.arange(min(batch_size, len(train_prompts)))
            self.logger.info(f"  - Using FIXED batch (same {len(fixed_batch_indices)} samples every episode for overfitting test)")
        else:
            self.logger.info(f"  - Resampling batch every episode (normal training mode)")

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

                # PPO ratio metrics
                ratio_mean = metrics.get('ratio_mean', 1.0)
                ratio_min = metrics.get('ratio_min', 1.0)
                ratio_max = metrics.get('ratio_max', 1.0)
                ratio_clipped_pct = metrics.get('ratio_clipped_frac', 0.0) * 100.0

                # Log ratio metrics (for KL/debugging)
                log_ratio_mean = metrics.get('log_ratio_mean', 0.0)
                log_ratio_min = metrics.get('log_ratio_min', 0.0)
                log_ratio_max = metrics.get('log_ratio_max', 0.0)

                # Advantage metrics (shows if clamping is active)
                adv_mean = metrics.get('advantages_mean', 0.0)
                adv_std = metrics.get('advantages_std', 1.0)
                adv_min_raw = metrics.get('advantages_min_raw', 0.0)
                adv_max_raw = metrics.get('advantages_max_raw', 0.0)
                adv_min = metrics.get('advantages_min', 0.0)
                adv_max = metrics.get('advantages_max', 0.0)
                adv_clamped_low_pct = metrics.get('advantages_clamped_low_frac', 0.0) * 100.0
                adv_clamped_high_pct = metrics.get('advantages_clamped_high_frac', 0.0) * 100.0

                self.logger.info(f"Episode {int(episode):3d}")
                self.logger.info(f"  Loss: Total={metrics['total_loss']:7.4f} | PG={metrics['pg_loss']:7.4f} | KL={metrics['kl_divergence']:7.4f}")
                self.logger.info(f"  Reward: {metrics['reward_mean']:6.3f} ± {metrics.get('reward_std', 0.0):5.3f} | Fmt={metrics['format_reward_mean']:5.3f} | Correct={metrics['correctness_reward_mean']:5.3f}")
                self.logger.info(f"  PPO Ratio: μ={ratio_mean:5.3f} [{ratio_min:5.3f}, {ratio_max:5.3f}] | Clipped={ratio_clipped_pct:5.1f}% | ε=[{1-self.clip_epsilon_low:.2f}, {1+self.clip_epsilon_high:.2f}]")
                self.logger.info(f"  Log Ratio: μ={log_ratio_mean:6.3f} [{log_ratio_min:6.3f}, {log_ratio_max:6.3f}] | Clamp=[{self.kl_clamp_min:.1f}, {self.kl_clamp_max:.1f}]")
                self.logger.info(f"  Advantages: μ={adv_mean:6.3f} σ={adv_std:5.3f} | Raw=[{adv_min_raw:6.3f}, {adv_max_raw:6.3f}] | Clamped=[{adv_min:6.3f}, {adv_max:6.3f}]")
                self.logger.info(f"    Clamp: [{self.advantage_clip_min:.1f}, {self.advantage_clip_max:.1f}] | Low={adv_clamped_low_pct:5.1f}% High={adv_clamped_high_pct:5.1f}%")
                self.logger.info(f"  Gradients: {grad_norm_pre:6.3f}→{grad_norm_post:6.3f} | Clipped={grad_clipped_pct:5.1f}% | Max={self.gradient_clip:.1f}")
                self.logger.info(f"  Updates: ParamΔ={rel_param_update*100:.4f}% | LR={current_lr:.2e}")
                self.logger.info(f"  Throughput: Tokens={int(episode_tokens):5d} | Time={episode_time:5.2f}s | Speed={tokens_per_sec:6.1f} tok/s")
                self.logger.info("")

            # Run validation if enabled and at the right interval
            if validation_enabled and val_data and (episode + 1) % validation_interval == 0:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"VALIDATION AT EPISODE {episode + 1}")
                self.logger.info(f"{'='*60}")

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
                    step=episode + 1,
                    logger=self.logger
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
                    title=f"EPISODE {episode + 1} RESPONSE EXAMPLES",
                    logger=self.logger
                )

                self.logger.info(f"{'='*60}\n")

            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode+1}.pt"
                self.save_checkpoint(str(checkpoint_path))
                self.logger.info(f"  → Saved checkpoint to {checkpoint_path}")

            # Print timing summary periodically and RESET to prevent memory leak
            if self.print_timing and episode > 0 and episode % 10 == 0:
                self.timing_manager.print_timing_summary(f"Episode {episode} Summary")
                # CRITICAL: Reset timing data to prevent memory accumulation
                self.timing_manager.reset_timings()

            # Even if timing printing is disabled, reset every 10 episodes to prevent leak
            if not self.print_timing and episode > 0 and episode % 10 == 0:
                self.timing_manager.reset_timings()

            # Force cleanup at end of each episode to prevent accumulation
            if self.device.type in ["cuda", "mps"]:
                clear_device_cache(self.device)
            gc.collect()

        # Calculate total training time
        total_training_time = time.time() - training_start_time
        self.logger.info("=" * 50)
        self.logger.info("Training complete!")
        self.logger.info(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        self.logger.info(f"Total tokens processed: {cumulative_tokens:,}")
        self.logger.info(f"Average speed: {cumulative_tokens/total_training_time:.1f} tokens/second")

        # Save final model
        final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
        self.save_checkpoint(str(final_checkpoint_path))
        self.logger.info(f"\n✓ Final model saved to: {final_checkpoint_path}")

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
                _, _, rewards, _, _, _, _, _, _ = self.generate_trajectories(prompts)
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

        # Create checkpoint with scheduler and scaler state
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "ref_policy_state_dict": ref_policy_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "scaler_state_dict": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "config": self.config,
            "total_steps": self.total_steps,
            "episode": self.episode,
            "current_episode": self.current_episode,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint with smart LR warmup."""
        checkpoint = load_checkpoint(path, self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        # Only load ref_policy if it exists and was saved (may be None when kl_coef=0)
        if self.ref_policy is not None and checkpoint.get("ref_policy_state_dict") is not None:
            self.ref_policy.load_state_dict(checkpoint["ref_policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load gradient scaler state if it exists (for FP16 training)
        if self.grad_scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            self.grad_scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Get LR values for smart warmup decision
        checkpoint_lr = self.optimizer.param_groups[0]['lr']
        config_lr = self.config.get("optimizer", {}).get("lr")
        total_steps = checkpoint.get("total_steps", 0)
        initial_warmup_steps = self.config.get("optimizer", {}).get("warmup_steps", 30)

        if config_lr is not None:
            # Smart LR handling based on checkpoint vs config
            lr_increase_ratio = config_lr / checkpoint_lr if checkpoint_lr > 0 else 1.0

            if config_lr > checkpoint_lr and lr_increase_ratio >= 1.5:
                # Significant LR increase - use warmup to prevent instability
                resume_warmup_steps = self.config.get("optimizer", {}).get("resume_warmup_steps", 15)

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
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = config_lr

                # Create warmup scheduler
                self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)

                # Start at step 0 of warmup
                self.optimizer.param_groups[0]['lr'] = checkpoint_lr

            elif config_lr != checkpoint_lr:
                # Small change or decrease - apply immediately, disable scheduler
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = config_lr
                self.lr_scheduler = None

            else:
                # Same LR - check if we need to continue warmup or disable scheduler
                if total_steps >= initial_warmup_steps:
                    # Past initial warmup - disable scheduler
                    self.lr_scheduler = None
                elif self.lr_scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                    # Still in initial warmup phase - restore scheduler
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.total_steps = checkpoint.get("total_steps", 0)
        self.episode = checkpoint.get("episode", 0)
        self.current_episode = checkpoint.get("current_episode", checkpoint.get("episode", 0))