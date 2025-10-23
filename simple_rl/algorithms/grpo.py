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
from transformers import StoppingCriteriaList

from simple_rl.algorithms.base import BaseAlgorithm
from simple_rl.utils.huggingface_wrappers import LanguageModel
from simple_rl.utils.optimization import configure_optimizer
from simple_rl.utils.timing import TimingManager
from simple_rl.utils.checkpointing import save_checkpoint, load_checkpoint
from simple_rl.utils.logging_utils import create_logger
from simple_rl.utils.validators import validate_logprobs_episode_zero
from simple_rl.utils.kl_divergence import compute_kl_divergence
from simple_rl.utils.policy_loss import (
    compute_ppo_policy_loss,
    compute_ppo_policy_loss_token_level,
)
from simple_rl.utils.stopping_criteria import MultiTokenStoppingCriteria
from simple_rl.dto import (
    TrajectoryBatch,
    MetricsBuilder,
    GeneratedCompletionsResult,
    AdvantageStats,
    LossMetrics,
    EpochMetrics,
    TrainingHistory,
    ValidationHistory,
    BatchData,
    TrainResult,
    EvaluationResult,
    CheckpointData,
)


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
        self.config = config or {}
        self.use_wandb = use_wandb

        if not config:
            raise ValueError("Config must be provided")

        device_config = self.config.get("device", None)
        if device_config:
            self.device = torch.device(device_config)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.policy = LanguageModel(config, model=model, tokenizer=tokenizer)
        self.policy = self.policy.to(self.device)
        self._disable_dropout()

        algo_config = self.config.get("algorithm", {})
        training_config = self.config.get("training", {})

        self.group_size = algo_config.get("group_size", 4)
        self.kl_coef = algo_config.get("kl_coef", 0.05)
        self.normalize_rewards = algo_config.get("normalize_rewards", True)

        if self.kl_coef > 0:
            self.ref_policy = copy.deepcopy(self.policy)
            self.ref_policy = self.ref_policy.to(self.device)
            self.ref_policy.eval()
        else:
            self.ref_policy = None

        self.clip_epsilon = algo_config.get("clip_epsilon", 0.2)
        if self.clip_epsilon is None:
            self.clip_epsilon_low = algo_config.get("clip_epsilon_low", 0.2)
            self.clip_epsilon_high = algo_config.get("clip_epsilon_high", 0.2)
        else:
            self.clip_epsilon_low = self.clip_epsilon
            self.clip_epsilon_high = self.clip_epsilon

        self.store_completions = algo_config.get("store_completions", True)
        self.update_epochs = training_config.get("update_epochs", 1)

        self.kl_estimator = training_config.get("kl_estimator", "k3")
        self.kl_clamp_min = training_config.get("kl_clamp_min", -2.0)
        self.kl_clamp_max = training_config.get("kl_clamp_max", 2.0)
        self.kl_reduction = training_config.get("kl_reduction", "mean")

        self.policy_loss_type = training_config.get("policy_loss_type", "token")
        if self.policy_loss_type == "token":
            self._compute_policy_loss_fn = compute_ppo_policy_loss_token_level
        elif self.policy_loss_type == "sequence":
            self._compute_policy_loss_fn = compute_ppo_policy_loss
        else:
            raise ValueError(
                f"Unknown policy_loss_type: {self.policy_loss_type}. "
                f"Choose from: 'sequence', 'token'"
            )

        self.advantage_clip_min = training_config.get("advantage_clip_min", -3.0)
        self.advantage_clip_max = training_config.get("advantage_clip_max", 3.0)

        self.batch_size = training_config.get("batch_size", 8)
        self.minibatch_size = training_config.get("minibatch_size", None) or self.batch_size
        self.rollout_batch_size = training_config.get("rollout_batch_size", None) or min(2, self.batch_size)
        self.max_new_tokens = training_config.get("max_new_tokens", 128)
        self.min_new_tokens = training_config.get("min_new_tokens", 1)
        self.temperature = training_config.get("temperature", 0.9)
        self.top_k = training_config.get("top_k", None)
        self.top_p = training_config.get("top_p", 0.9)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)

        self.logprobs_batch_size = training_config.get("logprobs_batch_size", None)
        self.offload_generated_to_cpu = training_config.get("offload_generated_to_cpu", True)

        self.logger = create_logger(self.config)
        if self.use_wandb:
            self.logger.init_wandb()

        self.optimizer, self.lr_scheduler = configure_optimizer(self.policy, self.config, logger=self.logger)
        self.batch_reward_fn = batch_reward_fn

        self.total_steps = 0
        self.episode = 0
        self.current_episode = 0

        self.timing_manager = TimingManager()
        self.print_timing = self.config.get("timing", {}).get("enabled", False)

        self.clear_cache_on_mps = self.config.get("device_optimizations", {}).get("clear_cache_on_mps", False)
        self.log_trajectory_progress = self.config.get("logging", {}).get("show_trajectory_progress", False)

        self.use_amp = self.device.type == "cuda"

        model_dtype = next(self.policy.parameters()).dtype
        self.use_grad_scaler = (model_dtype == torch.float16) and self.use_amp
        if self.use_grad_scaler:
            self.grad_scaler = torch.cuda.amp.GradScaler()
        else:
            self.grad_scaler = None

        self._setup_stopping_tokens()

    def _disable_dropout(self):
        """Disable all dropout for deterministic log probs."""
        if hasattr(self.policy.model, 'config'):
            model_cfg = self.policy.model.config
            if hasattr(model_cfg, 'attention_dropout') and model_cfg.attention_dropout > 0:
                model_cfg.attention_dropout = 0.0

            for attr in ['hidden_dropout', 'hidden_dropout_prob', 'resid_pdrop', 'dropout']:
                if hasattr(model_cfg, attr):
                    old_val = getattr(model_cfg, attr)
                    if old_val > 0:
                        setattr(model_cfg, attr, 0.0)

        for _, module in self.policy.model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                if module.p > 0:
                    module.p = 0.0

    def _setup_stopping_tokens(self):
        training_config = self.config.get("training", {})
        raw_stop = training_config.get("stop_sequences", ["</answer>"])

        if raw_stop is None:
            self.stop_sequences = []
            self.use_multi_token_stopping = False
        else:
            if isinstance(raw_stop, str):
                norm = [raw_stop]
            else:
                try:
                    norm = list(raw_stop)
                except Exception:
                    norm = []
            self.stop_sequences = [s for s in norm if isinstance(s, str) and s]
            self.use_multi_token_stopping = len(self.stop_sequences) > 0

        self.answer_end_token_id = None
        if self.use_multi_token_stopping and any(s == "</answer>" for s in self.stop_sequences):
            try:
                encoded = self.policy.tokenizer.encode("</answer>", add_special_tokens=False)
                if encoded:
                    self.answer_end_token_id = encoded[-1]
            except Exception:
                pass

    def reset_timings(self):
        self.timing_manager.reset_timings()

    def _set_episode_seed(self, episode: int, base_seed: int = 42):
        """Set deterministic random seeds for reproducible training."""
        import random
        episode_seed = base_seed + episode
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(episode_seed)
            torch.cuda.manual_seed_all(episode_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(episode_seed)

    def _compute_batch_log_probs_vectorized(
        self,
        model: Any,
        generated_ids: List[torch.Tensor],
        attention_mask: List[torch.Tensor],
        prompt_end_positions: torch.Tensor,
        completion_mask: List[torch.Tensor],
        requires_grad: bool = False,
        keep_gradients: bool = False,
    ) -> torch.Tensor:
        """Vectorized batch log probability computation with memory-efficient chunking."""
        batch_size = len(generated_ids)
        device = next(model.model.parameters()).device
        max_completion_len = max(mask.size(0) for mask in completion_mask)
        chunk_size = self.logprobs_batch_size if self.logprobs_batch_size is not None else batch_size

        if chunk_size < batch_size:
            all_completion_log_probs = []

            for chunk_start in range(0, batch_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, batch_size)

                chunk_gen_ids = generated_ids[chunk_start:chunk_end]
                chunk_attn_mask = attention_mask[chunk_start:chunk_end]
                chunk_prompt_end_pos = prompt_end_positions[chunk_start:chunk_end]
                chunk_completion_mask = completion_mask[chunk_start:chunk_end]

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

                if not keep_gradients:
                    chunk_log_probs = chunk_log_probs.detach()

                if self.device.type == "cuda" and self.offload_generated_to_cpu:
                    chunk_log_probs = chunk_log_probs.cpu()
                    if not keep_gradients:
                        torch.cuda.empty_cache()

                all_completion_log_probs.append(chunk_log_probs)

            return torch.cat(all_completion_log_probs, dim=0)
        else:
            result = self._compute_chunk_log_probs(
                model=model,
                generated_ids=generated_ids,
                attention_mask=attention_mask,
                prompt_end_positions=prompt_end_positions,
                completion_mask=completion_mask,
                max_completion_len=max_completion_len,
                requires_grad=requires_grad,
                device=device,
            )
            if not keep_gradients:
                result = result.detach()
            return result

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
        """Compute log probs for a single chunk of sequences."""
        chunk_size = len(generated_ids)

        if generated_ids[0].device != device:
            generated_ids = [g.to(device) for g in generated_ids]
            attention_mask = [a.to(device) for a in attention_mask]
            prompt_end_positions = prompt_end_positions.to(device)
            completion_mask = [c.to(device) for c in completion_mask]

        padded_ids = pad_sequence(generated_ids, batch_first=True, padding_value=self.policy.tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_completion_mask = pad_sequence(completion_mask, batch_first=True, padding_value=0)

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

        with torch.set_grad_enabled(requires_grad):
            chunk_log_probs = model.compute_log_probs(
                padded_ids,
                attention_mask=padded_attention_mask,
            )

        batch_indices = torch.arange(chunk_size, device=device).unsqueeze(1)
        completion_positions = torch.arange(max_completion_len, device=device).unsqueeze(0)
        source_positions = (prompt_end_positions.unsqueeze(1) - 1) + completion_positions

        max_valid_position = chunk_log_probs.size(1) - 1
        source_positions_clamped = source_positions.clamp(0, max_valid_position)
        valid_mask = source_positions < chunk_log_probs.size(1)

        completion_log_probs = chunk_log_probs[batch_indices, source_positions_clamped]

        completion_log_probs = torch.where(
            valid_mask,
            completion_log_probs,
            torch.zeros(1, device=device, dtype=completion_log_probs.dtype)
        )

        completion_log_probs = completion_log_probs * padded_completion_mask

        return completion_log_probs


    def _generate_grouped_completions(
        self, prompts: List[str]
    ) -> GeneratedCompletionsResult:
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

        del generated_ids, generated_mask
        del replicated_prompt_ids, replicated_prompt_mask
        del batch_prompt_ids, batch_prompt_mask
        del batch_indices, seq_positions, source_positions, valid_mask, completion_positions
        del prompt_start_index, prompt_start_index_per_seq
        del batch_prompt_lengths

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return GeneratedCompletionsResult(
            generated_ids=right_padded_ids,
            generated_mask=right_padded_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_texts=completion_texts,
            prompt_end_positions=prompt_lengths_per_seq,
            total_sequences=total_sequences,
        )


    def generate_trajectories(
        self,
        prompts: List[str],
        answers: Optional[List[str]] = None,
        store_outputs: Optional[bool] = None,
    ) -> TrajectoryBatch:
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
            generated_ids = generation.generated_ids
            generated_mask = generation.generated_mask
            completion_ids = generation.completion_ids
            completion_mask = generation.completion_mask
            completion_texts = generation.completion_texts
            replicated_prompt_end_positions = generation.prompt_end_positions
            total_sequences = generation.total_sequences

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
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps" and self.clear_cache_on_mps:
                torch.mps.empty_cache()

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

        return TrajectoryBatch(
            prompts=all_prompts if store else None,
            completions=all_completions if store else None,
            rewards=all_rewards,
            completion_mask=all_completion_mask,
            format_rewards=all_format_rewards,
            correctness_rewards=all_correctness_rewards,
            generated_ids=generated_ids,
            attention_mask=attention_mask,
            prompt_end_positions=prompt_end_positions,
        )

    def compute_logprobs(
        self,
        logprob_type: str,
        generated_ids: List[torch.Tensor],
        attention_mask: List[torch.Tensor],
        prompt_end_positions: torch.Tensor,
        completion_mask: List[torch.Tensor],
    ) -> torch.Tensor:
        """Unified method to compute logprobs (old/ref/new) with identical computation."""
        self.timing_manager.start_timer(f"compute_{logprob_type}_logprobs")

        if logprob_type == "ref":
            if self.kl_coef > 0 and self.ref_policy is not None:
                model = self.ref_policy
            else:
                batch_size = len(completion_mask)
                max_completion_len = max(mask.size(0) for mask in completion_mask)
                self.timing_manager.end_timer(f"compute_{logprob_type}_logprobs")
                return torch.zeros(batch_size, max_completion_len, device=self.device)
        else:
            model = self.policy

        prev_mode = model.training
        prev_cache = getattr(model.model.config, "use_cache", None)

        model.eval()
        if prev_cache is not None:
            model.model.config.use_cache = False

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        try:
            with torch.enable_grad():
                log_probs = self._compute_batch_log_probs_vectorized(
                    model=model,
                    generated_ids=generated_ids,
                    attention_mask=attention_mask,
                    prompt_end_positions=prompt_end_positions,
                    completion_mask=completion_mask,
                    requires_grad=True,
                    keep_gradients=(logprob_type == "new"),
                )
        finally:
            if prev_cache is not None:
                model.model.config.use_cache = prev_cache
            if prev_mode:
                model.train()

        self.timing_manager.end_timer(f"compute_{logprob_type}_logprobs")

        if logprob_type in ["old", "ref"]:
            return log_probs.detach()
        else:
            return log_probs

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int = 1,
        normalize_within_groups: bool = True
    ) -> Tuple[torch.Tensor, AdvantageStats]:
        """Compute group-normalized advantages."""
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

        adv_min_raw = advantages_raw.min().item()
        adv_max_raw = advantages_raw.max().item()

        advantages = torch.clamp(advantages_raw, min=self.advantage_clip_min, max=self.advantage_clip_max)

        clamped_low = (advantages_raw < self.advantage_clip_min).sum().item()
        clamped_high = (advantages_raw > self.advantage_clip_max).sum().item()
        total = advantages_raw.numel()
        clamped_low_frac = clamped_low / max(total, 1)
        clamped_high_frac = clamped_high / max(total, 1)

        stats = AdvantageStats(
            advantages_min_raw=adv_min_raw,
            advantages_max_raw=adv_max_raw,
            advantages_min=advantages.min().item(),
            advantages_max=advantages.max().item(),
            advantages_clamped_low_frac=clamped_low_frac,
            advantages_clamped_high_frac=clamped_high_frac,
        )

        return advantages, stats

    def compute_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, LossMetrics]:
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

        metrics = LossMetrics(
            policy_loss=policy_loss.item(),
            kl_divergence=kl_penalty.item(),
            entropy=entropy.item(),
            ratio_mean=policy_metrics["ratio_mean"],
            ratio_min=policy_metrics["ratio_min"],
            ratio_max=policy_metrics["ratio_max"],
            ratio_clipped_frac=policy_metrics["ratio_clipped_frac"],
            log_ratio_mean=policy_metrics["log_ratio_mean"],
            log_ratio_min=policy_metrics["log_ratio_min"],
            log_ratio_max=policy_metrics["log_ratio_max"],
            tokens_generated=completion_mask.sum().item(),
            kl_term=(self.kl_coef * kl_penalty).item(),
            entropy_term=(entropy_coef * entropy).item(),
        )

        self.timing_manager.end_timer("loss_computation")
        return loss, metrics

    def train_step(self, batch: BatchData) -> Dict[str, float]:
        """Perform GRPO training step with multiple epochs over same trajectories."""
        self.policy.train()
        prompts = batch.prompts
        answers = batch.answers

        self.timing_manager.start_timer("trajectory_generation")
        trajectories = self.generate_trajectories(
            prompts, answers=answers, store_outputs=False
        )
        self.timing_manager.end_timer("trajectory_generation")

        rewards = trajectories.rewards
        completion_mask = trajectories.completion_mask
        format_rewards = trajectories.format_rewards
        correctness_rewards = trajectories.correctness_rewards
        generated_ids = trajectories.generated_ids
        attention_mask = trajectories.attention_mask
        prompt_end_positions = trajectories.prompt_end_positions

        old_log_probs = self.compute_logprobs(
            "old", generated_ids, attention_mask, prompt_end_positions, completion_mask
        )
        ref_log_probs = self.compute_logprobs(
            "ref", generated_ids, attention_mask, prompt_end_positions, completion_mask
        )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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

        epoch_metrics = EpochMetrics()
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
                    mb_indices_cpu = mb_indices.cpu()

                    mb_advantages = advantages[mb_indices]
                    if mb_advantages.device != self.device:
                        mb_advantages = mb_advantages.to(self.device)

                    selected_gen_ids = [generated_ids[idx] for idx in mb_indices_cpu]
                    selected_attn_mask = [attention_mask[idx] for idx in mb_indices_cpu]
                    selected_completion_mask = [completion_mask[idx] for idx in mb_indices_cpu]
                    selected_prompt_end_positions = prompt_end_positions[mb_indices_cpu]

                    indices_for_logprobs = mb_indices_cpu if old_log_probs.device.type == 'cpu' else mb_indices
                    mb_old_log_probs = old_log_probs[indices_for_logprobs]
                    mb_ref_log_probs = ref_log_probs[indices_for_logprobs]

                    if mb_old_log_probs.device != self.device:
                        mb_old_log_probs = mb_old_log_probs.to(self.device)
                    if mb_ref_log_probs.device != self.device:
                        mb_ref_log_probs = mb_ref_log_probs.to(self.device)

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_recompute")

                    mb_new_log_probs = self.compute_logprobs(
                        "new",
                        selected_gen_ids,
                        selected_attn_mask,
                        selected_prompt_end_positions,
                        selected_completion_mask,
                    )

                    if mb_new_log_probs.device != self.device:
                        mb_new_log_probs = mb_new_log_probs.to(self.device)
                    mb_completion_mask = pad_sequence(
                        selected_completion_mask,
                        batch_first=True,
                        padding_value=0
                    )

                    if mb_completion_mask.device != self.device:
                        mb_completion_mask = mb_completion_mask.to(self.device)

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_recompute")

                    if self.episode == 0 and epoch == 0 and mb_idx == 1:
                        val_gen_ids = [g.to(self.device) if g.device != self.device else g for g in selected_gen_ids]
                        val_prompt_end = selected_prompt_end_positions.to(self.device) if selected_prompt_end_positions.device != self.device else selected_prompt_end_positions
                        model_dtype = next(self.policy.parameters()).dtype

                        validate_logprobs_episode_zero(
                            mb_old_log_probs,
                            mb_new_log_probs,
                            mb_ref_log_probs,
                            model_dtype,
                            self.policy.tokenizer,
                            val_gen_ids,
                            val_prompt_end,
                        )

                    self.optimizer.zero_grad()

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_loss")
                    loss, mb_metrics = self.compute_loss(
                        mb_new_log_probs,
                        mb_old_log_probs,
                        mb_advantages,
                        mb_ref_log_probs,
                        mb_completion_mask,
                    )
                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_loss")

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_backward")
                    if self.use_grad_scaler:
                        self.grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_backward")

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_clip")
                    if self.use_grad_scaler:
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

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_update")
                    if self.use_grad_scaler:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        self.optimizer.step()

                    with torch.no_grad():
                        theta_after = torch.nn.utils.parameters_to_vector(self.policy.parameters()).float()
                        param_update_norm = (theta_after - theta_before).norm().item()
                        relative_param_update = param_update_norm / max(theta_norm_before, 1e-12)
                        del theta_before, theta_after

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_update")

                    epoch_metrics.accumulate(
                        mb_metrics,
                        grad_norm_before_clip.item(),
                        grad_norm_post_clip,
                        grad_was_clipped,
                        relative_param_update
                    )
                    num_updates += 1

                    self.timing_manager.start_timer(f"epoch_{epoch}_minibatch_{mb_idx}_cleanup")
                    del (
                        mb_new_log_probs, loss, mb_old_log_probs, mb_ref_log_probs, mb_advantages,
                        mb_completion_mask,
                        selected_gen_ids, selected_attn_mask, selected_completion_mask,
                        selected_prompt_end_positions,
                        param_update_norm, relative_param_update, grad_norm_before_clip, grad_norm_post_clip
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device.type == "mps" and self.clear_cache_on_mps:
                        torch.mps.empty_cache()
                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}_cleanup")

                    self.timing_manager.end_timer(f"epoch_{epoch}_minibatch_{mb_idx}")

        finally:
            if prev_cache is not None:
                self.policy.model.config.use_cache = prev_cache

        self.timing_manager.end_timer("optimization")

        epoch_metrics.average(num_updates)

        metrics = MetricsBuilder.build_training_metrics(
            epoch_metrics=epoch_metrics.to_dict(),
            reward_mean=reward_mean_scalar,
            reward_std=reward_std_scalar,
            format_reward_mean=format_reward_mean_scalar,
            correctness_reward_mean=correctness_reward_mean_scalar,
            total_tokens=total_tokens_scalar,
            advantages_mean=advantages_mean_scalar,
            advantages_std=advantages_std_scalar,
            advantage_stats=advantage_stats,
            kl_coef=self.kl_coef,
        )

        self.total_steps += 1
        self.logger.log_metrics(metrics.to_dict(), self.total_steps)

        del old_log_probs, ref_log_probs, advantages, completion_mask
        del generated_ids, attention_mask, prompt_end_positions

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and self.clear_cache_on_mps:
            torch.mps.empty_cache()

        return metrics.to_dict()

    def train(
        self,
        train_data: Dict[str, List[str]],
        val_data: Optional[Dict[str, List[str]]] = None,
        num_episodes: Optional[int] = None
    ) -> TrainResult:
        import time
        from pathlib import Path

        if num_episodes is None:
            num_episodes = self.config['training']['num_episodes']

        batch_size = self.config['training']['batch_size']
        update_epochs = self.config['training'].get('update_epochs', 1)
        log_interval = self.config['logging']['log_interval']
        save_interval = self.config['logging']['save_interval']

        validation_enabled = self.config.get('validation', {}).get('enabled', False)
        validation_interval = self.config.get('validation', {}).get('interval', 10)
        validation_num_samples = self.config.get('validation', {}).get('num_samples', 20)
        validation_num_demo_examples = self.config.get('validation', {}).get('num_demo_examples', 5)

        training_config = self.config.get('training', {})
        max_new_tokens = training_config.get('max_new_tokens', 128)
        temperature = training_config.get('temperature', 0.9)
        top_p = training_config.get('top_p', 0.9)

        train_prompts = train_data["prompts"]
        train_answers = train_data["answers"]

        self.logger.info(f"Starting proper GRPO training for {num_episodes} episodes...")
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
        training_metrics = TrainingHistory()
        validation_metrics = ValidationHistory()

        checkpoint_dir = Path("checkpoints/grpo_qwen_math")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        cumulative_tokens = 0
        training_start_time = time.time()

        start_episode = self.current_episode
        if start_episode > 0:
            self.logger.info(f"\n⏭️  RESUMING FROM EPISODE {start_episode}")
            self.logger.info(f"   Will train episodes {start_episode} to {num_episodes-1}")
            self.logger.info(f"   Scheduler state restored (warmup may already be complete)")

        resample_batch = self.config.get('training', {}).get('resample_batch_per_episode', False)

        if not resample_batch:
            fixed_batch_indices = np.arange(min(batch_size, len(train_prompts)))
            self.logger.info(f"  - Using FIXED batch (same {len(fixed_batch_indices)} samples every episode for overfitting test)")
        else:
            self.logger.info(f"  - Resampling batch every episode (normal training mode)")

        for episode in range(start_episode, num_episodes):
            self.episode = episode
            self.current_episode = episode

            self._set_episode_seed(episode)
            episode_start_time = time.time()

            if resample_batch:
                batch_indices = np.random.choice(len(train_prompts), batch_size, replace=True)
            else:
                batch_indices = fixed_batch_indices

            batch_prompts = [train_prompts[i] for i in batch_indices]
            batch_answers = [train_answers[i] for i in batch_indices]

            batch_data = BatchData(
                prompts=batch_prompts,
                answers=batch_answers
            )
            metrics = self.train_step(batch_data)

            episode_time = time.time() - episode_start_time
            episode_tokens = metrics.get("tokens_generated", 0)
            cumulative_tokens += episode_tokens
            tokens_per_sec = episode_tokens / episode_time if episode_time > 0 else 0
            training_metrics.append(episode, metrics, episode_time, episode_tokens, cumulative_tokens, tokens_per_sec)

            if episode % log_interval == 0:
                self._log_training_progress(episode, metrics, episode_tokens, episode_time, tokens_per_sec)

            if validation_enabled and val_data and (episode + 1) % validation_interval == 0:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"VALIDATION AT EPISODE {episode + 1}")
                self.logger.info(f"{'='*60}")

                from simple_rl.evaluation.gsm8k import evaluate_on_gsm8k, demonstrate_model_responses

                val_metrics = evaluate_on_gsm8k(
                    self,
                    val_data["prompts"],
                    val_data["answers"],
                    validation_num_samples,
                    max_new_tokens=min(max_new_tokens, 256),
                    temperature=1.0,
                    top_p=1.0,
                    model_name=f"Episode {episode + 1}",
                    save_results=True,
                    results_file="results/grpo_eval_results.json",
                    step=episode + 1,
                    logger=self.logger,
                    batch_size=8,
                    use_stopping_criteria=False,
                    sample=False,
                )

                validation_metrics.append(episode + 1, val_metrics)

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

            if (episode + 1) % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode+1}.pt"
                save_checkpoint(self, str(checkpoint_path))
                self.logger.info(f"  → Saved checkpoint to {checkpoint_path}")

            if self.print_timing and episode > 0 and episode % 10 == 0:
                self.timing_manager.print_timing_summary(f"Episode {episode} Summary")
                self.timing_manager.reset_timings()

            if not self.print_timing and episode > 0 and episode % 10 == 0:
                self.timing_manager.reset_timings()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

        total_training_time = time.time() - training_start_time
        self.logger.info("=" * 50)
        self.logger.info("Training complete!")
        self.logger.info(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        self.logger.info(f"Total tokens processed: {cumulative_tokens:,}")
        self.logger.info(f"Average speed: {cumulative_tokens/total_training_time:.1f} tokens/second")

        final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
        save_checkpoint(self, str(final_checkpoint_path))
        self.logger.info(f"\n✓ Final model saved to: {final_checkpoint_path}")

        if self.print_timing:
            self.timing_manager.print_timing_summary("FINAL TRAINING PERFORMANCE")

        return TrainResult(
            training_metrics=training_metrics.to_dict(),
            validation_metrics=validation_metrics.to_dict(),
            total_time=total_training_time,
            final_reward=training_metrics.reward_mean[-1] if training_metrics.reward_mean else 0.0
        )

    def _log_training_progress(
        self,
        episode: int,
        metrics: Dict[str, float],
        episode_tokens: int,
        episode_time: float,
        tokens_per_sec: float,
    ) -> None:
        """Log training progress for current episode."""
        current_lr = self.optimizer.param_groups[0]['lr']

        m = metrics
        ratio_clipped_pct = m['ratio_clipped_frac'] * 100.0
        grad_clipped_pct = m['grad_clipped_frac'] * 100.0
        adv_clamped_low_pct = m['advantages_clamped_low_frac'] * 100.0
        adv_clamped_high_pct = m['advantages_clamped_high_frac'] * 100.0

        self.logger.info(f"Episode {int(episode):3d}")
        self.logger.info(f"  Loss: Total={m['total_loss']:7.4f} | PG={m['pg_loss']:7.4f} | KL={m['kl_divergence']:7.4f}")
        self.logger.info(f"  Reward: {m['reward_mean']:6.3f} ± {m['reward_std']:5.3f} | Fmt={m['format_reward_mean']:5.3f} | Correct={m['correctness_reward_mean']:5.3f}")
        self.logger.info(f"  PPO Ratio: μ={m['ratio_mean']:5.3f} [{m['ratio_min']:5.3f}, {m['ratio_max']:5.3f}] | Clipped={ratio_clipped_pct:5.1f}% | ε=[{1-self.clip_epsilon_low:.2f}, {1+self.clip_epsilon_high:.2f}]")
        self.logger.info(f"  Log Ratio: μ={m['log_ratio_mean']:6.3f} [{m['log_ratio_min']:6.3f}, {m['log_ratio_max']:6.3f}] | Clamp=[{self.kl_clamp_min:.1f}, {self.kl_clamp_max:.1f}]")
        self.logger.info(f"  Advantages: μ={m['advantages_mean']:6.3f} σ={m['advantages_std']:5.3f} | Raw=[{m['advantages_min_raw']:6.3f}, {m['advantages_max_raw']:6.3f}] | Clamped=[{m['advantages_min']:6.3f}, {m['advantages_max']:6.3f}]")
        self.logger.info(f"    Clamp: [{self.advantage_clip_min:.1f}, {self.advantage_clip_max:.1f}] | Low={adv_clamped_low_pct:5.1f}% High={adv_clamped_high_pct:5.1f}%")
        self.logger.info(f"  Gradients: {m['grad_norm']:6.3f}→{m['grad_norm_post_clip']:6.3f} | Clipped={grad_clipped_pct:5.1f}% | Max={self.gradient_clip:.1f}")
        self.logger.info(f"  Updates: ParamΔ={m['relative_param_update']*100:.4f}% | LR={current_lr:.2e}")
        self.logger.info(f"  Throughput: Tokens={int(episode_tokens):5d} | Time={episode_time:5.2f}s | Speed={tokens_per_sec:6.1f} tok/s")
        self.logger.info("")

    def evaluate(self, num_episodes: int = 1) -> EvaluationResult:
        """Evaluate the policy."""
        self.policy.eval()
        total_rewards = []

        with torch.no_grad():
            for _ in range(num_episodes):
                prompts = [f"Test {i}: Calculate {i}*2" for i in range(4)]
                trajectories = self.generate_trajectories(prompts)
                total_rewards.extend(trajectories.rewards.cpu().numpy())

        self.policy.train()

        return EvaluationResult(
            eval_reward_mean=np.mean(total_rewards),
            eval_reward_std=np.std(total_rewards),
        )

    def load_checkpoint(self, path: str):
        """Load checkpoint from path."""
        load_checkpoint(self, path)