# Log Probability Consistency Fix

## Problem
Old and new log probabilities differed significantly (diff ≈ 0.418), despite using the same model weights and input sequences.

## Root Cause

`torch.set_grad_enabled()` changes kernel selection in PyTorch:

1. **Old/Ref logprobs**: Computed with `torch.no_grad()` → inference kernels
2. **New logprobs**: Computed with `torch.enable_grad()` → training kernels → **different results!**

Even with `model.eval()` and dropout disabled, the gradient context affects which CUDA kernels are selected.

## Solution

**Compute ALL logprobs with `torch.enable_grad()`, then detach old/ref:**

```python
def compute_logprobs(self, logprob_type, ...):
    model.eval()  # Force eval mode (no dropout)
    
    # Compute ALL with gradients enabled (consistent kernels)
    with torch.enable_grad():
        log_probs = compute(..., requires_grad=True)
    
    # Detach old/ref, keep gradients for new
    if logprob_type in ["old", "ref"]:
        return log_probs.detach()  # Remove gradients
    else:
        return log_probs  # Keep gradients for backward
```

### Key Insights

1. **Gradient context affects kernels**: `torch.no_grad()` vs `torch.enable_grad()` → different CUDA kernels
2. **Simple solution**: Compute everything with gradients enabled, then detach
3. **Detach is free**: O(1) operation, only removes gradient tracking
4. **Ref policy not frozen**: Keep `requires_grad=True` (same as policy), just don't train it

### Why Ref Policy Shouldn't Be Frozen

**Before** (broken):
```python
ref_policy = deepcopy(policy)
for param in ref_policy.parameters():
    param.requires_grad = False  # ❌ Changes kernel selection!
```

**After** (correct):
```python
ref_policy = deepcopy(policy)
# Keep requires_grad=True (same as policy)
# Just don't call optimizer.step() on it
```

**Reason**: `requires_grad=False` can change kernel selection even with `torch.enable_grad()` context. Keeping it True ensures policy and ref_policy use identical computation paths.

## Changes Made

### 1. Fixed `compute_logprobs()` 
**File**: `simple_rl/algorithms/grpo.py`, lines 815-843

- Always use `torch.enable_grad()` for ALL logprob types
- Set `requires_grad=True` for consistent kernels
- Detach old/ref to remove gradients

### 2. Removed ref_policy Freezing
**File**: `simple_rl/algorithms/grpo.py`, lines 145-155

- Removed `param.requires_grad = False` from ref_policy initialization
- Keep `requires_grad=True` (same as policy)
- Ref policy won't be trained (not included in optimizer)

### 3. Added Single-Sequence Validation
**File**: `simple_rl/algorithms/grpo.py`, lines 1063-1168

- Processes sequences one-by-one (no padding artifacts)
- Validates old vs new AND old vs ref
- Clear error messages for debugging

## Expected Results

After this fix:
- ✅ **Old-New diff**: 0.000000e+00 (same model, detached vs not)
- ✅ **Old-Ref diff**: 0.000000e+00 (both deepcopies with same state)

## Testing

Run training:
```bash
python scripts/full_pipeline_sft_grpo_training.py
```

Expected output:
```
================================================================================
SINGLE-SEQUENCE VALIDATION (NO PADDING)
================================================================================
  Seq 1/32: Old-New=0.000000e+00, Old-Ref=0.000000e+00
  Seq 2/32: Old-New=0.000000e+00, Old-Ref=0.000000e+00
  ...

Maximum Old-New diff: 0.000000e+00
Maximum Old-Ref diff: 0.000000e+00

✅ PASSED: Old-Ref diff 0.00e+00 < threshold 1e-04
✅ PASSED: Old-New diff 0.00e+00 < threshold 1e-04
================================================================================
✅ ALL SINGLE-SEQUENCE VALIDATIONS PASSED
================================================================================
```

## Summary

The fix is simple and elegant:
1. Keep ref_policy parameters with `requires_grad=True` (same as policy)
2. Always compute logprobs with `torch.enable_grad()`
3. Detach old/ref to remove gradients

This ensures:
- ✅ Numerical consistency (same kernels for all)
- ✅ Correct gradients (new has grads attached)
- ✅ No performance penalty (single forward each)
- ✅ Clean implementation

## References
- PyTorch autograd and kernel selection
- PPO/GRPO requirements for consistent log probabilities
