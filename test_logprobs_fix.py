"""Quick test to verify log probs extraction fix."""
import torch

def test_logprobs_extraction():
    """Test that log probs extraction works correctly."""

    # Simulate the scenario
    batch_size = 4
    total_sequences = 12  # 4 prompts * 3 group_size
    full_seq_len = 50

    # Different prompt lengths for each original prompt
    prompt_lengths = torch.tensor([10, 15, 12, 18])

    # Replicate for group_size=3
    prompt_lengths_per_seq = prompt_lengths.repeat_interleave(3)

    # Max completion length
    max_completion_len = full_seq_len - prompt_lengths.min().item()

    print(f"Total sequences: {total_sequences}")
    print(f"Prompt lengths per seq: {prompt_lengths_per_seq}")
    print(f"Max completion length: {max_completion_len}")

    # Simulate full log probs (shifted by 1, so seq_len - 1)
    full_log_probs = torch.randn(total_sequences, full_seq_len - 1)

    # Extract completion log probs (vectorized)
    device = full_log_probs.device
    batch_indices = torch.arange(total_sequences, device=device).unsqueeze(1)
    completion_positions = torch.arange(max_completion_len, device=device).unsqueeze(0)

    # Source positions in full_log_probs for completion tokens
    # Subtract 1 because log probs are shifted
    source_positions = (prompt_lengths_per_seq - 1).unsqueeze(1) + completion_positions

    # Clamp to valid range
    source_positions = source_positions.clamp(0, full_log_probs.size(1) - 1)

    # Create mask for valid positions
    valid_mask = source_positions < full_log_probs.size(1)

    # Gather completion log probs (right-padded)
    completion_log_probs = full_log_probs[batch_indices, source_positions]

    # Set padding positions to 0.0
    completion_log_probs = torch.where(valid_mask, completion_log_probs,
                                      torch.tensor(0.0, device=device, dtype=completion_log_probs.dtype))

    print(f"\nCompletion log probs shape: {completion_log_probs.shape}")
    print(f"Expected shape: ({total_sequences}, {max_completion_len})")

    # Verify shape
    assert completion_log_probs.shape == (total_sequences, max_completion_len), \
        f"Shape mismatch! Got {completion_log_probs.shape}, expected ({total_sequences}, {max_completion_len})"

    # Verify that sequences with shorter prompts have fewer padding
    for seq_idx in range(total_sequences):
        prompt_len = prompt_lengths_per_seq[seq_idx].item()
        actual_completion_len = full_seq_len - prompt_len

        # Count non-zero log probs
        non_zero_count = (completion_log_probs[seq_idx] != 0.0).sum().item()

        print(f"Seq {seq_idx}: prompt_len={prompt_len}, actual_completion_len={actual_completion_len}, non_zero_log_probs={non_zero_count}")

        # Check that non-zero count matches actual completion length (or is close due to padding)
        assert non_zero_count <= actual_completion_len, \
            f"Seq {seq_idx}: Too many non-zero log probs! Got {non_zero_count}, max should be {actual_completion_len}"

    print("\n✓ All checks passed!")
    return True

if __name__ == "__main__":
    test_logprobs_extraction()
