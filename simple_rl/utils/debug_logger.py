"""Debug logging utility for GRPO training diagnostics.

Provides detailed logging of tensor shapes, masks, log probabilities, and other
debug information to help diagnose padding, masking, and KL divergence issues.
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json


class GRPODebugLogger:
    """Debug logger for GRPO training diagnostics."""

    def __init__(self, log_dir: str = "debug_logs", enabled: bool = True):
        """
        Initialize debug logger.

        Args:
            log_dir: Directory to store debug logs
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        if not self.enabled:
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"grpo_debug_{timestamp}.log"

        # Initialize log file
        with open(self.log_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("GRPO DEBUG LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        print(f"✓ Debug logging enabled: {self.log_file}")

    def _write(self, text: str):
        """Write text to log file."""
        if not self.enabled:
            return

        with open(self.log_file, "a") as f:
            f.write(text)

    def _format_tensor_info(self, name: str, tensor: torch.Tensor, show_values: bool = False) -> str:
        """Format tensor information for logging."""
        if tensor is None:
            return f"{name}: None\n"

        lines = []
        lines.append(f"{name}:")
        lines.append(f"  Shape: {list(tensor.shape)}")
        lines.append(f"  Device: {tensor.device}")
        lines.append(f"  Dtype: {tensor.dtype}")

        if tensor.numel() > 0:
            lines.append(f"  Min: {tensor.min().item():.6f}")
            lines.append(f"  Max: {tensor.max().item():.6f}")
            # Only compute mean for float tensors
            if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
                lines.append(f"  Mean: {tensor.mean().item():.6f}")

            if show_values:
                # Show first few values for inspection
                if tensor.dim() == 1:
                    vals = tensor[:min(10, len(tensor))].cpu().numpy()
                    lines.append(f"  Values (first {len(vals)}): {self._format_array(vals)}")
                elif tensor.dim() == 2:
                    # Show first sequence
                    vals = tensor[0, :min(20, tensor.size(1))].cpu().numpy()
                    lines.append(f"  Seq[0] (first {len(vals)}): {self._format_array(vals)}")

        return "\n".join(lines) + "\n"

    def _format_array(self, arr: np.ndarray) -> str:
        """Format numpy array for compact display."""
        if len(arr) == 0:
            return "[]"
        return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"

    def log_episode_start(self, episode: int):
        """Log start of new episode."""
        if not self.enabled:
            return

        separator = "\n" + "=" * 80 + "\n"
        self._write(separator)
        self._write(f"EPISODE {episode}\n")
        self._write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._write(separator + "\n")

    def log_generation_batch(
        self,
        batch_idx: int,
        prompts: List[str],
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_texts: List[str],
        prompt_end_positions: torch.Tensor,
    ):
        """Log details of trajectory generation batch."""
        if not self.enabled:
            return

        self._write(f"\n{'─' * 80}\n")
        self._write(f"GENERATION BATCH {batch_idx}\n")
        self._write(f"{'─' * 80}\n\n")

        # Prompt info
        self._write(f"Number of prompts: {len(prompts)}\n")
        self._write(f"Example prompt: {prompts[0][:100]}...\n\n")

        # Tensor shapes
        self._write(self._format_tensor_info("generated_ids", generated_ids))
        self._write(self._format_tensor_info("generated_mask", generated_mask))
        self._write(self._format_tensor_info("completion_ids", completion_ids))
        self._write(self._format_tensor_info("prompt_end_positions", prompt_end_positions, show_values=True))

        # Example completions
        self._write(f"\nExample completions (first 3):\n")
        for i in range(min(3, len(completion_texts))):
            self._write(f"  [{i}] {completion_texts[i][:150]}\n")

        # Detailed inspection of first sequence
        self._write(f"\nDETAILED INSPECTION - Sequence 0:\n")
        seq_gen_ids = generated_ids[0].cpu().numpy()
        seq_mask = generated_mask[0].cpu().numpy()
        prompt_end = prompt_end_positions[0].item() if prompt_end_positions.dim() > 0 else prompt_end_positions.item()

        self._write(f"  Prompt end position: {prompt_end}\n")
        self._write(f"  Generated sequence length: {len(seq_gen_ids)}\n")
        self._write(f"  Attention mask sum: {seq_mask.sum()}\n")
        self._write(f"  Prompt region mask: {seq_mask[:int(prompt_end)]}\n")
        self._write(f"  Completion region mask: {seq_mask[int(prompt_end):]}\n")

        # Show actual token IDs
        self._write(f"  Generated token IDs (first 30): {seq_gen_ids[:30]}\n")
        self._write(f"  Completion token IDs: {completion_ids[0].cpu().numpy()}\n\n")

    def log_logprobs_computation(
        self,
        batch_idx: int,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        completion_mask: torch.Tensor,
        seq_idx: int = 0,
        batch_completion_ids: list = None,
        max_completion_len: int = None,
    ):
        """Log log probability computation details."""
        if not self.enabled:
            return

        self._write(f"\nLOG PROBS COMPUTATION - Batch {batch_idx}\n")
        self._write(f"{'─' * 60}\n\n")

        if max_completion_len is not None:
            self._write(f"max_completion_len: {max_completion_len}\n")
        if batch_completion_ids is not None:
            self._write(f"Number of sequences: {len(batch_completion_ids)}\n")
            self._write(f"Completion lengths: {[c.size(0) for c in batch_completion_ids[:5]]}...\n\n")

        self._write(self._format_tensor_info("policy_log_probs", policy_log_probs))
        self._write(self._format_tensor_info("ref_log_probs", ref_log_probs))
        self._write(self._format_tensor_info("completion_mask", completion_mask))

        # Detailed inspection of one sequence
        if seq_idx < policy_log_probs.size(0):
            self._write(f"\nSequence {seq_idx} details:\n")

            if batch_completion_ids is not None and seq_idx < len(batch_completion_ids):
                actual_len = batch_completion_ids[seq_idx].size(0)
                self._write(f"  Actual completion length: {actual_len}\n")

            pol_lp = policy_log_probs[seq_idx].cpu().numpy()
            ref_lp = ref_log_probs[seq_idx].cpu().numpy()
            mask = completion_mask[seq_idx].cpu().numpy()

            self._write(f"  Policy log probs: {self._format_array(pol_lp[:20])}\n")
            self._write(f"  Ref log probs:    {self._format_array(ref_lp[:20])}\n")
            self._write(f"  Mask:             {mask[:20]}\n")
            self._write(f"  Num valid tokens: {mask.sum()}\n")

            # Show masked values
            masked_pol = pol_lp * mask
            masked_ref = ref_lp * mask
            self._write(f"  Masked policy sum: {masked_pol.sum():.6f}\n")
            self._write(f"  Masked ref sum:    {masked_ref.sum():.6f}\n")
            self._write(f"  Log prob diff:     {(masked_ref - masked_pol).sum():.6f}\n\n")

    def log_batch_concatenation(
        self,
        num_batches: int,
        batch_lengths: List[int],
        max_seq_len: int,
        stored_generated_ids: torch.Tensor,
        stored_attention_mask: torch.Tensor,
        stored_prompt_end_positions: torch.Tensor,
        padding_adjustments: List[int],
    ):
        """Log batch concatenation and padding details."""
        if not self.enabled:
            return

        self._write(f"\n{'═' * 80}\n")
        self._write(f"BATCH CONCATENATION\n")
        self._write(f"{'═' * 80}\n\n")

        self._write(f"Number of batches: {num_batches}\n")
        self._write(f"Batch lengths: {batch_lengths}\n")
        self._write(f"Max sequence length: {max_seq_len}\n")
        self._write(f"Padding adjustments: {padding_adjustments}\n\n")

        self._write(self._format_tensor_info("stored_generated_ids", stored_generated_ids))
        self._write(self._format_tensor_info("stored_attention_mask", stored_attention_mask))
        self._write(self._format_tensor_info("stored_prompt_end_positions", stored_prompt_end_positions, show_values=True))

        # Verify padding
        self._write(f"\nPADDING VERIFICATION:\n")
        for i in range(min(5, stored_generated_ids.size(0))):
            mask_sum = stored_attention_mask[i].sum().item()
            prompt_end = stored_prompt_end_positions[i].item()
            seq_len = stored_generated_ids.size(1)

            self._write(f"  Seq {i}: prompt_end={prompt_end}, mask_sum={mask_sum}, seq_len={seq_len}")
            if prompt_end > seq_len:
                self._write(" ⚠️ WARNING: prompt_end > seq_len!\n")
            else:
                self._write(" ✓\n")

        self._write("\n")

    def log_training_step_start(self, episode: int, num_prompts: int, num_sequences: int):
        """Log start of training step."""
        if not self.enabled:
            return

        self._write(f"\n{'═' * 80}\n")
        self._write(f"TRAINING STEP - Episode {episode}\n")
        self._write(f"{'═' * 80}\n\n")

        self._write(f"Number of prompts: {num_prompts}\n")
        self._write(f"Total sequences: {num_sequences}\n\n")

    def log_advantages_computation(
        self,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        group_size: int,
    ):
        """Log advantage computation details."""
        if not self.enabled:
            return

        self._write(f"\nADVANTAGES COMPUTATION\n")
        self._write(f"{'─' * 60}\n\n")

        self._write(self._format_tensor_info("rewards", rewards, show_values=True))
        self._write(self._format_tensor_info("advantages", advantages, show_values=True))

        self._write(f"\nGroup size: {group_size}\n")
        num_groups = rewards.size(0) // group_size

        # Show group statistics
        self._write(f"Number of groups: {num_groups}\n")
        for i in range(min(3, num_groups)):
            start = i * group_size
            end = start + group_size
            group_rewards = rewards[start:end].cpu().numpy()
            group_advantages = advantages[start:end].cpu().numpy()

            self._write(f"\nGroup {i}:\n")
            self._write(f"  Rewards: {self._format_array(group_rewards)}\n")
            self._write(f"  Advantages: {self._format_array(group_advantages)}\n")

        self._write("\n")

    def log_minibatch_update(
        self,
        mb_idx: int,
        mb_start: int,
        mb_end: int,
        mb_generated_ids: torch.Tensor,
        mb_attention_mask: torch.Tensor,
        mb_old_log_probs: torch.Tensor,
        mb_ref_log_probs: torch.Tensor,
        mb_new_log_probs: torch.Tensor,
        mb_completion_mask: torch.Tensor,
        mb_advantages: torch.Tensor,
    ):
        """Log minibatch update details."""
        if not self.enabled:
            return

        self._write(f"\n{'─' * 80}\n")
        self._write(f"MINIBATCH {mb_idx} (indices {mb_start}-{mb_end})\n")
        self._write(f"{'─' * 80}\n\n")

        self._write(self._format_tensor_info("mb_generated_ids", mb_generated_ids))
        self._write(self._format_tensor_info("mb_attention_mask", mb_attention_mask))
        self._write(self._format_tensor_info("mb_old_log_probs", mb_old_log_probs))
        self._write(self._format_tensor_info("mb_ref_log_probs", mb_ref_log_probs))
        self._write(self._format_tensor_info("mb_new_log_probs", mb_new_log_probs))
        self._write(self._format_tensor_info("mb_completion_mask", mb_completion_mask))
        self._write(self._format_tensor_info("mb_advantages", mb_advantages, show_values=True))

        # Detailed comparison for first sequence
        self._write(f"\nSEQUENCE 0 DETAILED COMPARISON:\n")
        if mb_old_log_probs.size(0) > 0:
            old_lp = mb_old_log_probs[0].detach().cpu().numpy()
            ref_lp = mb_ref_log_probs[0].detach().cpu().numpy()
            new_lp = mb_new_log_probs[0].detach().cpu().numpy()
            mask = mb_completion_mask[0].detach().cpu().numpy()

            self._write(f"  Old log probs: {self._format_array(old_lp[:15])}\n")
            self._write(f"  Ref log probs: {self._format_array(ref_lp[:15])}\n")
            self._write(f"  New log probs: {self._format_array(new_lp[:15])}\n")
            self._write(f"  Mask:          {mask[:15]}\n")

            # Compute differences
            old_new_diff = new_lp - old_lp
            ref_new_diff = ref_lp - new_lp

            self._write(f"  New - Old diff: {self._format_array(old_new_diff[:15])}\n")
            self._write(f"  Ref - New diff: {self._format_array(ref_new_diff[:15])}\n")

            # Masked sums
            self._write(f"  Old sum (masked): {(old_lp * mask).sum():.6f}\n")
            self._write(f"  Ref sum (masked): {(ref_lp * mask).sum():.6f}\n")
            self._write(f"  New sum (masked): {(new_lp * mask).sum():.6f}\n")

        self._write("\n")

    def log_kl_computation(
        self,
        kl_penalty: float,
        estimator: str,
        clamp_min: float,
        clamp_max: float,
        ref_log_probs: torch.Tensor = None,
        new_log_probs: torch.Tensor = None,
        completion_mask: torch.Tensor = None,
    ):
        """Log KL divergence computation details."""
        if not self.enabled:
            return

        self._write(f"\nKL DIVERGENCE COMPUTATION\n")
        self._write(f"{'─' * 60}\n\n")

        self._write(f"Estimator: {estimator}\n")
        if estimator == "k3":
            self._write(f"Clamp range: [{clamp_min}, {clamp_max}]\n")
        self._write(f"KL penalty value: {kl_penalty:.6f}\n")

        # If KL is suspiciously high, dump full tensors for analysis
        if kl_penalty > 5.0:
            self._write(f"\n⚠️  HIGH KL DIVERGENCE DETECTED: {kl_penalty:.6f}\n")
            self._write(f"{'=' * 60}\n")

            if ref_log_probs is not None and new_log_probs is not None and completion_mask is not None:
                self._write(f"\nDUMPING FULL TENSORS FOR FIRST SEQUENCE:\n\n")

                # Dump first sequence fully
                ref_lp = ref_log_probs[0].detach().cpu().numpy()
                new_lp = new_log_probs[0].detach().cpu().numpy()
                mask = completion_mask[0].detach().cpu().numpy()

                self._write(f"Reference log probs (full, length={len(ref_lp)}):\n")
                self._write(f"{ref_lp}\n\n")

                self._write(f"New log probs (full, length={len(new_lp)}):\n")
                self._write(f"{new_lp}\n\n")

                self._write(f"Completion mask (full, length={len(mask)}):\n")
                self._write(f"{mask}\n\n")

                # Compute per-token KL
                log_ratio = ref_lp - new_lp
                self._write(f"Log ratio (ref - new):\n")
                self._write(f"{log_ratio}\n\n")

                # Show only valid tokens
                valid_indices = mask > 0
                if valid_indices.sum() > 0:
                    self._write(f"\nVALID TOKENS ONLY (mask=1):\n")
                    self._write(f"Number of valid tokens: {valid_indices.sum()}\n")
                    self._write(f"Ref log probs: {ref_lp[valid_indices]}\n")
                    self._write(f"New log probs: {new_lp[valid_indices]}\n")
                    self._write(f"Log ratio: {log_ratio[valid_indices]}\n")

                    # Compute per-token KL with k3 formula
                    clamped_ratio = np.clip(log_ratio[valid_indices], clamp_min, clamp_max)
                    ratio = np.exp(clamped_ratio)
                    kl_per_token = ratio - 1.0 - clamped_ratio
                    self._write(f"KL per token: {kl_per_token}\n")
                    self._write(f"KL sum: {kl_per_token.sum():.6f}\n")
                    self._write(f"KL mean: {kl_per_token.mean():.6f}\n")

                self._write(f"\n{'=' * 60}\n")

        self._write("\n")

    def log_loss_computation(
        self,
        loss: float,
        policy_loss: float,
        kl_divergence: float,
        metrics: Dict[str, Any],
    ):
        """Log loss computation and metrics."""
        if not self.enabled:
            return

        self._write(f"\nLOSS COMPUTATION\n")
        self._write(f"{'─' * 60}\n\n")

        self._write(f"Total loss: {loss:.6f}\n")
        self._write(f"Policy loss: {policy_loss:.6f}\n")
        self._write(f"KL divergence: {kl_divergence:.6f}\n\n")

        self._write("Metrics:\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._write(f"  {key}: {value:.6f}\n")
            else:
                self._write(f"  {key}: {value}\n")

        self._write("\n")

    def log_episode_summary(self, episode: int, metrics: Dict[str, Any]):
        """Log episode summary."""
        if not self.enabled:
            return

        self._write(f"\n{'═' * 80}\n")
        self._write(f"EPISODE {episode} SUMMARY\n")
        self._write(f"{'═' * 80}\n\n")

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._write(f"{key}: {value:.6f}\n")
            else:
                self._write(f"{key}: {value}\n")

        self._write(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._write("\n")

    def flush(self):
        """Ensure all logs are written to file."""
        if not self.enabled:
            return
        # File is opened/closed for each write, so nothing to flush
        pass
