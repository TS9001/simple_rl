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

    def __init__(
        self,
        log_dir: str = "debug_logs",
        enabled: bool = True,
        debug_generation: bool = False,
        debug_alignment: bool = False,
        debug_advantages: bool = False,
        debug_loss: bool = False,
        debug_gradients: bool = False,
    ):
        """
        Initialize debug logger.

        Args:
            log_dir: Directory to store debug logs
            enabled: Master switch - whether ANY logging is enabled
            debug_generation: Log generation details (token counts, EOS, etc.)
            debug_alignment: Log alignment between OLD and NEW log probs
            debug_advantages: Log advantage computation details
            debug_loss: Log loss computation details
            debug_gradients: Log gradient norms and clipping
        """
        self.enabled = enabled
        self.debug_generation = debug_generation and enabled
        self.debug_alignment = debug_alignment and enabled
        self.debug_advantages = debug_advantages and enabled
        self.debug_loss = debug_loss and enabled
        self.debug_gradients = debug_gradients and enabled

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
            f.write(f"Debug flags:\n")
            f.write(f"  generation: {self.debug_generation}\n")
            f.write(f"  alignment: {self.debug_alignment}\n")
            f.write(f"  advantages: {self.debug_advantages}\n")
            f.write(f"  loss: {self.debug_loss}\n")
            f.write(f"  gradients: {self.debug_gradients}\n\n")

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

    def log_generation_debug(
        self,
        temperature: float,
        max_new_tokens: int,
        eos_token_id: Any,
        device: torch.device,
        model_dtype: Any = None,
        total_sequences: int = 0,
        total_tokens_generated: int = 0,
        avg_tokens_per_seq: float = 0.0,
        completion_lengths: Optional[torch.Tensor] = None,
        has_eos_count: int = 0,
    ):
        """Log generation configuration and statistics."""
        if not self.debug_generation:
            return

        print(f"\n🔍 GENERATION DEBUG:")
        print(f"  Temperature: {temperature}")
        print(f"  Max new tokens: {max_new_tokens}")
        print(f"  EOS token ID: {eos_token_id}")
        print(f"  Device: {device}")
        if model_dtype is not None:
            print(f"  Model dtype: {model_dtype}")

        if total_sequences > 0:
            print(f"  Total tokens generated: {total_tokens_generated}")
            print(f"  Avg tokens/sequence: {avg_tokens_per_seq:.1f}")

        if completion_lengths is not None:
            print(f"  Completion lengths - min: {completion_lengths.min().item()}, "
                  f"max: {completion_lengths.max().item()}, "
                  f"mean: {completion_lengths.float().mean().item():.1f}, "
                  f"std: {completion_lengths.float().std().item():.1f}")

        if total_sequences > 0:
            print(f"  Sequences with EOS: {has_eos_count}/{total_sequences}")

    def log_alignment_generation(
        self,
        batch_idx: int,
        total_sequences: int,
        seq_len: int,
        max_completion_len: int,
        logits_start_pos: int,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        generated_ids: torch.Tensor,
        replicated_prompt_end_positions: torch.Tensor,
        actual_completion_lengths: torch.Tensor,
        completion_start_in_logits: torch.Tensor,
    ):
        """Log alignment verification during generation (writes to file)."""
        if not self.debug_alignment or batch_idx != 1:
            return

        # Print to console
        print(f"\n🔍 Alignment debug flag: {self.debug_alignment}")

        # Write detailed logs to file
        filepath = self.log_dir / "generation_extraction.txt"
        with open(filepath, "w") as f:
            f.write(f"=== GENERATION TRAJECTORY EXTRACTION (Batch {batch_idx}) ===\n")
            f.write(f"total_sequences: {total_sequences}\n")
            f.write(f"seq_len (generated_ids): {seq_len}\n")
            f.write(f"max_completion_len: {max_completion_len}\n")
            f.write(f"logits_start_pos: {logits_start_pos}\n")
            f.write(f"policy_log_probs.shape: {policy_log_probs.shape}\n")
            f.write(f"ref_log_probs.shape: {ref_log_probs.shape}\n")
            f.write(f"completion_ids.shape: {completion_ids.shape}\n")
            f.write(f"completion_mask.shape: {completion_mask.shape}\n\n")

            # Log first 3 sequences in detail
            for seq_idx in range(min(3, total_sequences)):
                f.write(f"\n--- Sequence {seq_idx} ---\n")
                f.write(f"  prompt_end_position: {replicated_prompt_end_positions[seq_idx].item()}\n")
                f.write(f"  actual_completion_length: {actual_completion_lengths[seq_idx].item()}\n")
                f.write(f"  completion_start_in_logits: {completion_start_in_logits[seq_idx].item()}\n")

                # Show FULL generated sequence
                gen_seq = generated_ids[seq_idx]
                prompt_end = int(replicated_prompt_end_positions[seq_idx].item())
                f.write(f"  generated_ids[:20]: {gen_seq[:20].tolist()}\n")
                f.write(f"  generated_ids[{prompt_end}:{prompt_end+20}] (completion start): {gen_seq[prompt_end:prompt_end+20].tolist()}\n")
                f.write(f"  generated_ids[-20:]: {gen_seq[-20:].tolist()}\n")

                # Show completion tokens
                comp_ids = completion_ids[seq_idx, :min(20, int(actual_completion_lengths[seq_idx]))]
                f.write(f"  completion_ids[:20]: {comp_ids.tolist()}\n")

                # Verify extraction is correct
                f.write(f"  VERIFICATION: generated_ids[{prompt_end}] == completion_ids[0]? ")
                f.write(f"{gen_seq[prompt_end].item()} == {completion_ids[seq_idx, 0].item()} -> ")
                f.write(f"{'✓ MATCH' if gen_seq[prompt_end].item() == completion_ids[seq_idx, 0].item() else '✗ MISMATCH'}\n")

                # Show gather indices
                gather_start = completion_start_in_logits[seq_idx].item()
                f.write(f"  gather_indices[0:10]: {list(range(int(gather_start), int(gather_start + 10)))}\n")

            # Log extracted log probs
            f.write(f"\n=== EXTRACTED LOG PROBS ===\n")
            for seq_idx in range(min(3, total_sequences)):
                actual_len = int(actual_completion_lengths[seq_idx])
                seq_policy_lp = policy_log_probs[seq_idx, :actual_len]
                seq_mask = completion_mask[seq_idx, :actual_len]
                f.write(f"\n--- Sequence {seq_idx} ---\n")
                f.write(f"  policy_log_probs[:10]: {seq_policy_lp[:10].tolist()}\n")
                f.write(f"  completion_mask[:10]: {seq_mask[:10].tolist()}\n")
                f.write(f"  sum(policy_log_probs * mask): {(seq_policy_lp * seq_mask).sum().item()}\n")

    def log_alignment_minibatch(
        self,
        mb_idx: int,
        batch_size: int,
        seq_len: int,
        mb_max_completion_len: int,
        logits_start_pos: int,
        mb_full_log_probs: torch.Tensor,
        mb_completion_mask: torch.Tensor,
        mb_indices: torch.Tensor,
        mb_old_log_probs: torch.Tensor,
        mb_new_log_probs: torch.Tensor,
        mb_prompt_end_positions: torch.Tensor,
        valid_indices_mask: torch.Tensor,
    ):
        """Log alignment verification during minibatch recompute (writes to file)."""
        if not self.debug_alignment or mb_idx > 2:
            return

        filepath = self.log_dir / f"minibatch_{mb_idx}_extraction.txt"
        with open(filepath, "w") as f:
            f.write(f"=== MINIBATCH {mb_idx} RECOMPUTE EXTRACTION ===\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"seq_len (mb_generated_ids): {seq_len}\n")
            f.write(f"mb_max_completion_len: {mb_max_completion_len}\n")
            f.write(f"logits_start_pos: {logits_start_pos}\n")
            f.write(f"mb_full_log_probs.shape: {mb_full_log_probs.shape}\n")
            f.write(f"mb_completion_mask.shape: {mb_completion_mask.shape}\n")
            f.write(f"mb_indices: {mb_indices.tolist()[:5]}... (first 5)\n\n")

            # Log first 3 sequences in detail
            for seq_idx in range(min(3, batch_size)):
                orig_idx = mb_indices[seq_idx].item()
                f.write(f"\n--- Minibatch seq {seq_idx} (original idx={orig_idx}) ---\n")
                f.write(f"  prompt_end_position: {mb_prompt_end_positions[seq_idx].item()}\n")
                f.write(f"  completion_start_in_logits: {(mb_prompt_end_positions[seq_idx] - logits_start_pos).item()}\n")

                # Show old log probs for comparison
                old_lp_sum = (mb_old_log_probs[seq_idx] * mb_completion_mask[seq_idx]).sum().item()
                f.write(f"  OLD log_probs sum: {old_lp_sum}\n")
                f.write(f"  OLD log_probs[:10]: {mb_old_log_probs[seq_idx, :10].tolist()}\n")

            # Log extracted NEW log probs
            f.write(f"\n=== EXTRACTED NEW LOG PROBS ===\n")
            for seq_idx in range(min(3, batch_size)):
                orig_idx = mb_indices[seq_idx].item()
                new_lp_sum = (mb_new_log_probs[seq_idx] * mb_completion_mask[seq_idx]).sum().item()
                f.write(f"\n--- Minibatch seq {seq_idx} (original idx={orig_idx}) ---\n")
                f.write(f"  NEW log_probs[:10]: {mb_new_log_probs[seq_idx, :10].tolist()}\n")
                f.write(f"  NEW log_probs sum: {new_lp_sum}\n")
                f.write(f"  completion_mask[:10]: {mb_completion_mask[seq_idx, :10].tolist()}\n")
                f.write(f"  valid_indices_mask[:10]: {valid_indices_mask[seq_idx, :10].tolist()}\n")

    def log_advantages_debug(
        self,
        rewards: torch.Tensor,
        group_size: int,
        normalize_within_groups: bool,
        num_groups: int,
        group_mean: Optional[torch.Tensor] = None,
        group_std: Optional[torch.Tensor] = None,
        normalized_rewards: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        small_std_count: int = 0,
        clipped_count: int = 0,
    ):
        """Log advantage computation debug information."""
        if not self.debug_advantages:
            return

        print(f"\n🔍 ADVANTAGE COMPUTATION DEBUG:")
        print(f"  rewards - shape: {rewards.shape}, min: {rewards.min().item():.4f}, "
              f"max: {rewards.max().item():.4f}, mean: {rewards.mean().item():.4f}, "
              f"std: {rewards.std().item():.4f}")
        print(f"  group_size: {group_size}, normalize_within_groups: {normalize_within_groups}")

        if normalize_within_groups and group_size > 1:
            print(f"  num_groups: {num_groups}")
            if group_mean is not None:
                print(f"  group_mean - min: {group_mean.min().item():.4f}, "
                      f"max: {group_mean.max().item():.4f}, mean: {group_mean.mean().item():.4f}")
            if group_std is not None:
                print(f"  group_std - min: {group_std.min().item():.4f}, "
                      f"max: {group_std.max().item():.4f}, mean: {group_std.mean().item():.4f}")
            if small_std_count > 0:
                print(f"  WARNING: {small_std_count} groups have std < 1e-3")

            if normalized_rewards is not None:
                print(f"  normalized_rewards (before clipping) - min: {normalized_rewards.min().item():.4f}, "
                      f"max: {normalized_rewards.max().item():.4f}")

        if advantages is not None:
            print(f"  advantages (after clipping) - min: {advantages.min().item():.4f}, "
                  f"max: {advantages.max().item():.4f}, mean: {advantages.mean().item():.4f}")
            if clipped_count > 0:
                print(f"  WARNING: {clipped_count} advantages were clipped")

    def log_loss_debug(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None,
        new_log_probs_sum: Optional[torch.Tensor] = None,
        old_log_probs_sum: Optional[torch.Tensor] = None,
        ratio: Optional[torch.Tensor] = None,
        surr1: Optional[torch.Tensor] = None,
        surr2: Optional[torch.Tensor] = None,
        policy_loss: Optional[float] = None,
        kl_penalty: Optional[float] = None,
        entropy: Optional[float] = None,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.0,
        loss: Optional[float] = None,
    ):
        """Log loss computation debug information."""
        if not self.debug_loss:
            return

        print(f"\n🔍 LOSS COMPUTATION DEBUG:")
        print(f"  new_log_probs - shape: {new_log_probs.shape}, min: {new_log_probs.min().item():.4f}, "
              f"max: {new_log_probs.max().item():.4f}, mean: {new_log_probs.mean().item():.4f}")
        print(f"  old_log_probs - shape: {old_log_probs.shape}, min: {old_log_probs.min().item():.4f}, "
              f"max: {old_log_probs.max().item():.4f}, mean: {old_log_probs.mean().item():.4f}")
        print(f"  advantages - shape: {advantages.shape}, min: {advantages.min().item():.4f}, "
              f"max: {advantages.max().item():.4f}, mean: {advantages.mean().item():.4f}")
        print(f"  completion_mask - shape: {completion_mask.shape}, sum: {completion_mask.sum().item()}")

        if new_log_probs_sum is not None and old_log_probs_sum is not None:
            print(f"  new_log_probs_sum - min: {new_log_probs_sum.min().item():.4f}, "
                  f"max: {new_log_probs_sum.max().item():.4f}, mean: {new_log_probs_sum.mean().item():.4f}")
            print(f"  old_log_probs_sum - min: {old_log_probs_sum.min().item():.4f}, "
                  f"max: {old_log_probs_sum.max().item():.4f}, mean: {old_log_probs_sum.mean().item():.4f}")

        if ratio is not None:
            log_ratio = new_log_probs_sum - old_log_probs_sum if new_log_probs_sum is not None else None
            if log_ratio is not None:
                print(f"  log_ratio (new - old) - min: {log_ratio.min().item():.4f}, "
                      f"max: {log_ratio.max().item():.4f}")
            print(f"  ratio - min: {ratio.min().item():.4f}, max: {ratio.max().item():.4f}, "
                  f"mean: {ratio.mean().item():.4f}")
            print(f"  ratio > 10: {(ratio > 10).sum().item()} sequences")
            print(f"  ratio < 0.1: {(ratio < 0.1).sum().item()} sequences")

        if surr1 is not None and surr2 is not None:
            print(f"  surr1 - min: {surr1.min().item():.4f}, max: {surr1.max().item():.4f}, "
                  f"mean: {surr1.mean().item():.4f}")
            print(f"  surr2 - min: {surr2.min().item():.4f}, max: {surr2.max().item():.4f}, "
                  f"mean: {surr2.mean().item():.4f}")

        if policy_loss is not None:
            print(f"  policy_loss (before KL/entropy): {policy_loss:.4f}")

        if ref_log_probs is not None:
            print(f"  ref_log_probs - min: {ref_log_probs.min().item():.4f}, "
                  f"max: {ref_log_probs.max().item():.4f}, mean: {ref_log_probs.mean().item():.4f}")

        if kl_penalty is not None:
            print(f"  kl_penalty (raw): {kl_penalty:.4f}")
            print(f"  kl_penalty * kl_coef: {(kl_coef * kl_penalty):.4f}")

        if entropy is not None:
            print(f"  entropy (raw): {entropy:.4f}")
            print(f"  entropy * entropy_coef: {(entropy_coef * entropy):.4f}")

        if loss is not None:
            print(f"  TOTAL LOSS: {loss:.4f}")
            if policy_loss is not None and kl_penalty is not None and entropy is not None:
                print(f"    = policy_loss: {policy_loss:.4f}")
                print(f"    + kl_term: {(kl_coef * kl_penalty):.4f}")
                print(f"    - entropy_term: {(entropy_coef * entropy):.4f}")

    def log_gradients_debug(
        self,
        mb_idx: int,
        policy_parameters: Any,
        gradient_clip: float,
        norm_before: Optional[float] = None,
        norm_after: Optional[float] = None,
    ):
        """Log gradient norms and clipping debug information."""
        if not self.debug_gradients:
            return

        print(f"\n🔍 GRADIENT DEBUG (minibatch {mb_idx}):")

        if norm_before is not None:
            print(f"  Gradient norm before clipping: {norm_before:.4f}")

        if norm_after is not None:
            print(f"  Gradient norm after clipping: {norm_after:.4f}")
            print(f"  Max allowed gradient norm: {gradient_clip}")

            if norm_before is not None and norm_before > gradient_clip:
                print(f"  ✓ Gradients were clipped: {norm_before:.4f} → {norm_after:.4f}")
            else:
                print(f"  ✓ No clipping needed (norm was {norm_before:.4f})" if norm_before else "  ✓ No clipping needed")

    def flush(self):
        """Ensure all logs are written to file."""
        if not self.enabled:
            return
        # File is opened/closed for each write, so nothing to flush
        pass
