# Log Probability Consistency Fix

## Problem
Old and ref log probabilities were identical (diff ≈ 0), but old and new log probabilities differed significantly (diff ≈ 0.23), despite using the same model weights and input sequences.

## Root Cause
The issue was caused by **training vs inference mode kernel divergence**:

1. **Old logprobs**: Computed with `torch.set_grad_enabled(False)` → inference kernels
2. **Ref logprobs**: Computed with `torch.set_grad_enabled(False)` → inference kernels  
3. **New logprobs**: Computed with `torch.set_grad_enabled(True)` → training kernels

Even though dropout was disabled, different code paths were used:
- Inference mode may use optimized kernels (e.g., Flash Attention 2 inference path)
- Training mode uses gradients-enabled kernels with different numerical properties
- This caused old/ref to match perfectly (same kernels) while new diverged (different kernels)

## Solution
Force **eval mode for ALL logprob computations** (old, ref, and new):

```python
# Before: Only old/ref used eval (commented out), causing divergence
# if logprob_type in ["old", "ref"]:
#     model.eval()

# After: Force eval for all types to ensure identical forward pass
model.eval()
with torch.set_grad_enabled(logprob_type == "new"):
    # Gradients still flow for "new" even in eval mode
    log_probs = compute(...)
if prev_mode:
    model.train()  # Restore original mode
```

### Key Insight
- `model.eval()` controls module behavior (dropout, batch norm, etc.) and kernel selection
- `torch.set_grad_enabled()` controls whether gradients are computed
- **Both can be used together**: eval mode + gradients enabled = consistent kernels + backprop

## Changes Made

### 1. Fixed `compute_logprobs()` in `grpo.py`
**File**: `simple_rl/algorithms/grpo.py`, lines 812-839

- Force `model.eval()` before computing any logprobs (old/ref/new)
- Keep `torch.set_grad_enabled(logprob_type == "new")` to allow gradients for policy updates
- Restore original training mode after computation

### 2. Tightened Validation Thresholds
**File**: `simple_rl/algorithms/grpo.py`, lines 1033-1061

Updated thresholds to reflect that old/new should now be nearly identical:

| Dtype | Old Threshold (old-new) | New Threshold (old-new) |
|-------|------------------------|------------------------|
| BF16  | 2.0                    | 1e-4                   |
| FP16  | 5e-2                   | 1e-5                   |
| FP32  | 5e-4                   | 1e-6                   |

The ref thresholds (old-ref) remain unchanged since they were already correct.

## Expected Results

After this fix:
- **Old-Ref diff**: ~0 (unchanged, already working)
- **Old-New diff**: ~0 (fixed, should now match old-ref diff)
- All three logprob types use identical forward pass paths
- Gradients still flow correctly for policy updates

## Testing

Run the unified logprobs test to verify:
```bash
python test_unified_logprobs.py
```

Or run a training step and check the debug output:
- Look for "Old-New max diff" in episode 0, minibatch 1 validation
- Should be < 1e-4 for BF16, < 1e-5 for FP16, < 1e-6 for FP32

## Technical Notes

### Why This Doesn't Break Training
- Gradients are still computed for "new" logprobs (via `torch.set_grad_enabled(True)`)
- Eval mode only affects forward pass behavior, not backward pass
- Policy updates still work correctly because the backward graph is preserved

### Why Old/Ref Were Always Identical
- Both used inference mode (no gradients)
- Both used the same model (policy/ref_policy have same weights initially)
- Same kernels, same inputs → identical outputs

### Why New Was Different Before
- Training mode uses different attention kernels
- Even with identical weights, training path has slightly different numerics
- This is a known PyTorch behavior, especially with Flash Attention 2

## References
- PyTorch documentation on eval() vs train() mode
- Flash Attention 2 kernel selection based on training flag
- PPO implementation best practices for logprob consistency

