#!/usr/bin/env python
"""
Log Probability Debugger

Validates and debugs log probability computations for RL algorithms.
Saves detailed logs for analysis when issues are detected.
"""

import torch
import numpy as np
from pathlib import Path


def save_logprobs_debug(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    generated_ids=None,
    prompt_end_positions=None,
    tokenizer=None,
    output_dir: str = ".",
):
    """
    Save detailed log probability comparison to files for debugging.
    
    Args:
        old_log_probs: [batch, seq_len] old policy logprobs
        new_log_probs: [batch, seq_len] new policy logprobs  
        ref_log_probs: [batch, seq_len] reference policy logprobs
        generated_ids: Optional list of token ID tensors for context
        prompt_end_positions: Optional tensor of prompt end positions
        tokenizer: Optional tokenizer for decoding tokens
        output_dir: Directory to save debug files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Ensure all in FP32 for comparison
    old_log_probs = old_log_probs.float()
    new_log_probs = new_log_probs.float()
    ref_log_probs = ref_log_probs.float()
    
    old_new_diff_per_seq = (old_log_probs - new_log_probs).abs().max(dim=1).values
    old_ref_diff_per_seq = (old_log_probs - ref_log_probs).abs().max(dim=1).values
    old_new_diff = old_new_diff_per_seq.max().item()
    old_ref_diff = old_ref_diff_per_seq.max().item()
    
    # Save main debug file
    with open(output_dir / "LOGPROBS_DEBUG.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("LOG PROBABILITY VALIDATION\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Batch shape: {old_log_probs.shape}\n")
        f.write(f"Total sequences: {old_log_probs.shape[0]}\n")
        f.write(f"Max sequence length: {old_log_probs.shape[1]}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Old-Ref max diff:  {old_ref_diff:.6e}\n")
        f.write(f"Old-New max diff:  {old_new_diff:.6e}\n")
        f.write(f"Old-Ref mean diff: {(old_log_probs - ref_log_probs).abs().mean().item():.6e}\n")
        f.write(f"Old-New mean diff: {(old_log_probs - new_log_probs).abs().mean().item():.6e}\n\n")
        
        f.write("="*80 + "\n")
        f.write("COMPLETE LOG PROBABILITIES FOR ALL SEQUENCES\n")
        f.write("="*80 + "\n\n")
        
        # Write complete log probs for every sequence
        for seq_idx in range(old_log_probs.shape[0]):
            f.write(f"\n{'='*80}\n")
            f.write(f"SEQUENCE {seq_idx + 1} / {old_log_probs.shape[0]}\n")
            f.write(f"{'='*80}\n\n")
            
            old_seq = old_log_probs[seq_idx].detach().cpu().numpy()
            new_seq = new_log_probs[seq_idx].detach().cpu().numpy()
            ref_seq = ref_log_probs[seq_idx].detach().cpu().numpy()
            
            # Get the generated IDs for this sequence if available
            gen_ids = None
            if generated_ids is not None and seq_idx < len(generated_ids):
                gen_ids = generated_ids[seq_idx]
                prompt_end = prompt_end_positions[seq_idx].item() if prompt_end_positions is not None else 0
                
                if tokenizer is not None:
                    # Decode full sequence
                    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                    f.write(f"Full generated text:\n{full_text}\n\n")
                    
                    # Get prompt/completion split
                    if prompt_end > 0:
                        prompt_ids = gen_ids[:prompt_end]
                        completion_ids = gen_ids[prompt_end:]
                        
                        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
                        
                        f.write(f"Prompt (first {prompt_end} tokens):\n{prompt_text}\n\n")
                        f.write(f"Completion (tokens {prompt_end} onwards):\n{completion_text}\n\n")
            
            # Find non-zero positions
            nonzero_mask = (old_seq != 0.0) | (new_seq != 0.0) | (ref_seq != 0.0)
            nonzero_indices = nonzero_mask.nonzero()[0] if nonzero_mask.any() else []
            
            f.write(f"Valid tokens: {len(nonzero_indices)} / {old_log_probs.shape[1]}\n")
            f.write(f"Max Old-New diff: {old_new_diff_per_seq[seq_idx].item():.6e}\n")
            f.write(f"Max Old-Ref diff: {old_ref_diff_per_seq[seq_idx].item():.6e}\n\n")
            
            # Write header
            f.write(f"{'Token':<8} {'Input Token':<20} {'Old LogProb':<15} {'New LogProb':<15} {'Ref LogProb':<15} {'Old-New Diff':<15} {'Old-Ref Diff':<15}\n")
            f.write("-"*120 + "\n")
            
            # Write all tokens
            for token_idx in range(old_log_probs.shape[1]):
                old_val = old_seq[token_idx]
                new_val = new_seq[token_idx]
                ref_val = ref_seq[token_idx]
                old_new_diff = abs(old_val - new_val)
                old_ref_diff = abs(old_val - ref_val)
                
                # Get token text
                input_token_str = "[N/A]"
                if gen_ids is not None and tokenizer is not None and prompt_end_positions is not None:
                    prompt_end = prompt_end_positions[seq_idx].item()
                    actual_token_pos = prompt_end + token_idx + 1
                    if actual_token_pos < len(gen_ids):
                        token_id = gen_ids[actual_token_pos].item()
                        input_token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                        input_token_str = repr(input_token_str)[1:-1][:20]
                
                is_padding = (old_val == 0.0 and new_val == 0.0 and ref_val == 0.0)
                marker = " [PAD]" if is_padding else ""
                
                f.write(f"{token_idx:<8} {input_token_str:<20} {old_val:>14.6f} {new_val:>14.6f} {ref_val:>14.6f} {old_new_diff:>14.6e} {old_ref_diff:>14.6e}{marker}\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF LOG PROBABILITY DUMP\n")
        f.write("="*80 + "\n")
    
    # Save inputs file if we have the data
    if generated_ids is not None and tokenizer is not None:
        with open(output_dir / "INPUTS_DEBUG.txt", "w") as f:
            f.write("="*80 + "\n")
            f.write("INPUT SEQUENCES FOR OLD, NEW, AND REF MODELS\n")
            f.write("="*80 + "\n\n")
            f.write("Note: All three models use the SAME input sequences.\n\n")
            
            for seq_idx in range(min(len(generated_ids), old_log_probs.shape[0])):
                f.write(f"\n{'='*60}\n")
                f.write(f"SEQUENCE {seq_idx + 1}\n")
                f.write(f"{'='*60}\n\n")
                
                gen_ids = generated_ids[seq_idx]
                prompt_end = prompt_end_positions[seq_idx].item() if prompt_end_positions is not None else 0
                
                # Full sequence
                full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                f.write("FULL SEQUENCE:\n")
                f.write(f"{full_text}\n\n")
                
                # Prompt/completion split
                if prompt_end > 0:
                    prompt_ids = gen_ids[:prompt_end]
                    completion_ids = gen_ids[prompt_end:]
                    
                    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
                    
                    f.write(f"PROMPT (tokens 0-{prompt_end-1}):\n{prompt_text}\n\n")
                    f.write(f"COMPLETION (tokens {prompt_end}-{len(gen_ids)-1}):\n{completion_text}\n\n")
                    
                    # Token-by-token breakdown
                    f.write("TOKEN-BY-TOKEN BREAKDOWN:\n")
                    f.write("-"*60 + "\n")
                    f.write(f"{'Position':<10} {'Token ID':<10} {'Token Text':<30}\n")
                    f.write("-"*60 + "\n")
                    
                    for i, token_id in enumerate(completion_ids):
                        token_text = tokenizer.decode([token_id.item()], skip_special_tokens=False)
                        token_text_repr = repr(token_text)[1:-1]
                        f.write(f"{i:<10} {token_id.item():<10} {token_text_repr:<30}\n")
                
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF INPUTS DUMP\n")
            f.write("="*80 + "\n")
    
    print(f"\n✓ Debug files saved to {output_dir}/")
    print(f"  - LOGPROBS_DEBUG.txt: Full log probabilities with diffs")
    if generated_ids is not None and tokenizer is not None:
        print(f"  - INPUTS_DEBUG.txt: Input sequences and token breakdown")


def validate_logprobs(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    model_dtype: torch.dtype,
    save_on_fail: bool = True,
    generated_ids=None,
    prompt_end_positions=None,
    tokenizer=None,
    output_dir: str = ".",
) -> dict:
    """
    Validate that old/new/ref log probabilities are consistent.
    
    Args:
        old_log_probs: Old policy logprobs
        new_log_probs: New policy logprobs
        ref_log_probs: Reference policy logprobs
        model_dtype: Model dtype for threshold selection
        save_on_fail: Whether to save debug files on validation failure
        generated_ids: Optional, for debug output
        prompt_end_positions: Optional, for debug output
        tokenizer: Optional, for debug output
        output_dir: Directory for debug files
        
    Returns:
        dict with validation results and metrics
    """
    # Ensure FP32 for comparison
    old_log_probs = old_log_probs.float()
    new_log_probs = new_log_probs.float()
    ref_log_probs = ref_log_probs.float()
    
    # Compute differences
    old_new_diff = (old_log_probs - new_log_probs).abs().max().item()
    old_ref_diff = (old_log_probs - ref_log_probs).abs().max().item()
    
    # Dtype-aware thresholds
    if model_dtype == torch.bfloat16:
        ref_threshold = 1e-4
        new_threshold = 1e-4
    elif model_dtype == torch.float16:
        ref_threshold = 1e-5
        new_threshold = 1e-5
    else:
        ref_threshold = 1e-6
        new_threshold = 1e-6
    
    # Validate
    old_ref_pass = old_ref_diff < ref_threshold
    old_new_pass = old_new_diff < new_threshold
    
    results = {
        "old_ref_diff": old_ref_diff,
        "old_new_diff": old_new_diff,
        "ref_threshold": ref_threshold,
        "new_threshold": new_threshold,
        "old_ref_pass": old_ref_pass,
        "old_new_pass": old_new_pass,
        "all_pass": old_ref_pass and old_new_pass,
    }
    
    # Save debug files if validation failed
    if not results["all_pass"] and save_on_fail:
        save_logprobs_debug(
            old_log_probs, new_log_probs, ref_log_probs,
            generated_ids, prompt_end_positions, tokenizer, output_dir
        )
    
    return results

