import torch
from typing import List, Optional


def validate_logprobs_episode_zero(
    mb_old_log_probs: torch.Tensor,
    mb_new_log_probs: torch.Tensor,
    mb_ref_log_probs: torch.Tensor,
    model_dtype: torch.dtype,
    tokenizer,
    selected_gen_ids: Optional[List[torch.Tensor]] = None,
    selected_prompt_end_positions: Optional[torch.Tensor] = None,
) -> None:
    """Validate old/new/ref logprobs consistency at episode 0."""
    from logprobs_debugger import validate_logprobs

    print("\n" + "="*80)
    print("LOGPROBS VALIDATION - EPISODE 0, MINIBATCH 1")
    print("="*80)

    # Validate
    results = validate_logprobs(
        mb_old_log_probs,
        mb_new_log_probs.detach(),  # Detach for comparison
        mb_ref_log_probs,
        model_dtype,
        save_on_fail=True,
        generated_ids=selected_gen_ids,
        prompt_end_positions=selected_prompt_end_positions,
        tokenizer=tokenizer,
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
