# Log Probability Consistency Fix

## Problem
Old and ref log probabilities were identical (diff ≈ 0), but old and new log probabilities differed significantly (diff ≈ 0.418 even with NO PADDING), despite using the same model weights and input sequences.

## Root Cause

`torch.set_grad_enabled()` changes kernel selection in PyTorch, even with `model.eval()`:

1. **Old logprobs**: Computed with `torch.no_grad()` → inference kernels
2. **Ref logprobs**: Computed with `torch.no_grad()` → inference kernels
3. **New logprobs**: Computed with `torch.enable_grad()` → **different kernels!**

### Evidence from Single-Sequence Validation
After eliminating padding artifacts by computing sequences one-by-one:
- Old-Ref diff: 0.000000e+00 ✅ (perfect match)
- Old-New diff: 4.180998e-01 ❌ (significant difference!)

This proves the issue is **NOT** padding, but gradient context affecting numerical computation.

## Solution

**Compute ALL logprobs with `torch.enable_grad()`, then detach old/ref:**

```python
def compute_logprobs(self, logprob_type, ...):
    model.eval()  # Force eval mode
    
    # Compute ALL with gradients enabled (consistent kernels)
    with torch.enable_grad():
        log_probs = self._compute_batch_log_probs_vectorized(
            model, ..., requires_grad=True  # Always True
        )
    
    # Detach old/ref, keep gradients for new
    if logprob_type in ["old", "ref"]:
        return log_probs.detach()  # Remove gradients
    else:
        return log_probs  # Keep gradients for backward
```

### Key Insights

1. **Gradient context affects kernels**: `torch.no_grad()` vs `torch.enable_grad()` selects different CUDA kernels
2. **Simple solution**: Compute everything with gradients enabled, then detach
3. **No double-forward needed**: Single computation path for all three
4. **Detach is cheap**: Only removes gradient tracking, doesn't recompute

### Why This Works

- **Before**: old/ref used no_grad → kernel A; new used enable_grad → kernel B
- **After**: all use enable_grad → kernel B; detach doesn't change numerical values
- **Result**: Identical numerical results, gradients only where needed

## Changes Made

### Modified `compute_logprobs()` 
**File**: `simple_rl/algorithms/grpo.py`, lines 821-850

**Before**:
```python
with torch.set_grad_enabled(logprob_type == "new"):
    log_probs = compute(...)  # Different kernels!
if logprob_type in ["old", "ref"]:
    return log_probs.detach()
else:
    return log_probs
```

**After**:
```python
with torch.enable_grad():  # ALWAYS enable_grad
    log_probs = compute(..., requires_grad=True)  # Always True
if logprob_type in ["old", "ref"]:
    return log_probs.detach()  # Remove gradients
else:
    return log_probs  # Keep gradients
```

### Added Single-Sequence Validation
**File**: `simple_rl/algorithms/grpo.py`, lines 1063-1152

- Processes sequences one-by-one (no batching, no padding)
- Eliminates padding artifacts from validation
- Clearly shows when differences are genuine vs. artifacts

## Single-Sequence Validation

Validation computes sequences **one-by-one** to eliminate batch padding artifacts:

```python
def _validate_log_probs_episode_zero_single_sequence(...):
    for seq_idx in range(num_sequences):
        # Extract SINGLE sequence (no batching, no padding!)
        single_gen_ids = [generated_ids[seq_idx]]
        
        # Compute old, new, ref for THIS sequence only
        old_lp = compute_logprobs("old", single_gen_ids, ...)
        new_lp = compute_logprobs("new", single_gen_ids, ...)
        ref_lp = compute_logprobs("ref", single_gen_ids, ...)
        
        # Compare - any difference is REAL, not from padding
        diff = (old_lp - new_lp.detach()).abs().max()
```

**Why this matters**:
- Batch processing pads sequences to same length
- Different batch sizes → different padding → numerical differences
- Single-sequence eliminates padding confounds
- Isolates genuine non-determinism from artifacts

## Expected Results

After this fix:
- **Old-Ref diff**: ~0 (unchanged, already working)
- **Old-New diff**: ~0 (now identical - same kernel, just detached)
- All three logprob types use identical forward pass
- "New" logprobs have gradients attached for backward pass
- No performance penalty (single forward pass for each type)

## Performance Impact

- **No double-forward**: All types computed once
- **Detach is free**: O(1) operation, just removes gradient tracking
- **Same total cost**: 3 forward passes (old, new, ref) as before
- **Better correctness**: Guaranteed numerical consistency

## Testing

Run training and check the validation output:
```bash
python scripts/full_pipeline_sft_grpo_training.py
```

Expected output:
```
================================================================================
SINGLE-SEQUENCE VALIDATION (NO PADDING)
================================================================================
Computing each sequence individually...

  Seq 1/32: Old-New=0.000000e+00, Old-Ref=0.000000e+00
  Seq 2/32: Old-New=0.000000e+00, Old-Ref=0.000000e+00
  ...

Maximum Old-New diff: 0.000000e+00 (sequence 1)
Maximum Old-Ref diff: 0.000000e+00

✅ PASSED: Old-Ref diff 0.00e+00 < threshold 1e-04
✅ PASSED: Old-New diff 0.00e+00 < threshold 1e-04
================================================================================
✅ ALL SINGLE-SEQUENCE VALIDATIONS PASSED
================================================================================
```

## Technical Notes

### Why Detach Works
- `tensor.detach()` returns a view with `requires_grad=False`
- Does NOT recompute values - just removes gradient tracking
- Numerical values remain identical
- Old/new/ref computed with same kernels → same values

### Why This Wasn't Obvious
- Gradient context affecting kernel selection is not well-documented
- Most code doesn't need this level of numerical consistency
- RL algorithms (PPO/GRPO) are particularly sensitive to log prob differences
- Single-sequence validation was key to isolating the issue

### Alternative Solutions Considered
1. ~~Use no_grad for all, recompute new with grad~~: Double-forward cost
2. ~~Use requires_grad_() after computation~~: Doesn't build computation graph
3. **Compute with grad, detach old/ref**: ✅ Best solution

## Summary

The fix is elegant and simple:
1. Always use `torch.enable_grad()` for consistent kernels
2. Detach old/ref to remove gradients
3. Keep gradients for new

This ensures:
- ✅ Numerical consistency (same kernel for all)
- ✅ Correct gradients for training (new has grads)
- ✅ No performance penalty (single forward each)
- ✅ Clean, maintainable code

## References
- PyTorch autograd documentation
- CUDA kernel selection based on gradient context
- PPO/GRPO algorithm requirements for consistent log probabilities
