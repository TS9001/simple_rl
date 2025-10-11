# GRPO Debug Guide

This guide explains how to enable debugging features in the GRPO algorithm to investigate training issues like loss spikes.

## Quick Start - Investigating Loss Spikes

If you see a spike like `Episode 4 | Loss: 35957.7164`, add this to your config:

```python
config = {
    # ... your existing config ...
    "debug": {
        "enabled": True,
        "log_dir": "debug_logs",
        "loss": True,        # ⭐ Shows detailed loss breakdown
        "advantages": True,   # ⭐ Shows advantage computation details
        "gradients": True,    # ⭐ Shows gradient norms
    }
}
```

## All Available Debug Flags

### 1. `"loss": True` - Loss Computation Debug
**When to use**: Investigating loss spikes, unstable training, or understanding what contributes to high loss.

**What it shows**:
- **Input tensors**: min/max/mean of new_log_probs, old_log_probs, advantages, completion_mask
- **Log probability sums**: Statistics of summed log probs per sequence
- **Policy ratio (π_new / π_old)**:
  - Min/max/mean values
  - Count of extreme ratios (>10 or <0.1)
  - Log ratio before exp
- **Surrogate objectives**: surr1 and surr2 statistics
- **Policy loss**: Value before KL/entropy terms
- **KL divergence**:
  - Reference log probs statistics
  - Raw KL penalty
  - Weighted KL penalty (kl_coef * KL)
- **Entropy**:
  - Raw entropy value
  - Weighted entropy term
- **Total loss breakdown**:
  ```
  TOTAL LOSS: X.XXXX
    = policy_loss: X.XXXX
    + kl_term: X.XXXX
    - entropy_term: X.XXXX
  ```

**Example output**:
```
🔍 LOSS COMPUTATION DEBUG:
  new_log_probs - shape: torch.Size([64, 128]), min: -5.2341, max: -0.0231, mean: -1.2456
  old_log_probs - shape: torch.Size([64, 128]), min: -5.1234, max: -0.0198, mean: -1.2123
  advantages - shape: torch.Size([64]), min: -2.1234, max: 1.9876, mean: 0.0012
  ...
  ratio - min: 0.8234, max: 1.2345, mean: 1.0012
  ratio > 10: 0 sequences
  ratio < 0.1: 0 sequences
  ...
  TOTAL LOSS: 35957.7164
    = policy_loss: 35950.2341
    + kl_term: 7.4823
    - entropy_term: 0.0000
```

**What to look for**:
- **Extreme ratios**: ratio > 10 or < 0.1 indicates policy changed drastically
- **Large policy_loss**: The main contributor to the spike
- **Large kl_term**: Policy diverging too much from reference
- **Vanishing entropy**: entropy ≈ 0 means policy is too confident

---

### 2. `"advantages": True` - Advantage Computation Debug
**When to use**: Understanding reward normalization issues, checking if advantage clipping is active.

**What it shows**:
- **Raw rewards**: min/max/mean/std before normalization
- **Group statistics** (if group normalization):
  - Number of groups
  - Group means and stds
  - Warning if any group has std < 1e-3 (potential numerical instability)
- **Normalized rewards**: min/max before clipping
- **Final advantages**: After clipping to [-10, 10]
- **Clipping warning**: How many advantages were clipped

**Example output**:
```
🔍 ADVANTAGE COMPUTATION DEBUG:
  rewards - shape: torch.Size([64]), min: 0.0000, max: 1.0000, mean: 0.0234, std: 0.0523
  group_size: 8, normalize_within_groups: True
  num_groups: 8
  group_mean - min: 0.0000, max: 0.0625, mean: 0.0234
  group_std - min: 0.0001, max: 0.1234, mean: 0.0523
  WARNING: 2 groups have std < 1e-3
  normalized_rewards (before clipping) - min: -12.3456, max: 8.9012
  advantages (after clipping) - min: -10.0000, max: 8.9012, mean: -0.0123
  WARNING: 3 advantages were clipped
```

**What to look for**:
- **Very small group std**: Groups with std < 1e-3 can cause numerical instability
- **Clipped advantages**: If many advantages are clipped, the normalization might be too aggressive
- **Extreme normalized rewards**: Values > 100 before clipping indicate potential issues

---

### 3. `"gradients": True` - Gradient Debug
**When to use**: Investigating gradient explosion, vanishing gradients, or training instability.

**What it shows**:
- **Gradient norm before clipping**: L2 norm of all gradients
- **Gradient norm after clipping**: After applying max_norm constraint
- **Warning**: If gradients exceeded the clip threshold

**Example output**:
```
🔍 GRADIENT DEBUG (minibatch 1):
  Gradient norm before clipping: 145.2341
  Gradient norm after clipping: 1.0000
  WARNING: Gradients were clipped (norm was 145.2341, max allowed: 1.0)
```

**What to look for**:
- **Gradient norm > 10x clip threshold**: Indicates training instability
- **Consistent clipping**: If gradients are clipped every minibatch, might need to:
  - Increase gradient_clip value
  - Lower learning rate
  - Investigate why gradients are so large

---

### 4. `"generation": True` - Text Generation Debug
**When to use**: Debugging generation issues, understanding token generation behavior.

**What it shows**:
- **Generation parameters**: temperature, max_new_tokens, EOS token IDs
- **Model dtype**: What precision the model is using
- **Computation dtype**: Actual computation precision
- **Token statistics**: Total tokens generated, average per sequence
- **Completion lengths**: min/max/mean/std of completion lengths
- **EOS presence**: How many sequences ended with EOS token

**Example output**:
```
🔍 GENERATION DEBUG:
  Temperature: 0.7
  Max new tokens: 512
  EOS token ID: [151643, 151645]
  Device: mps
  Model dtype: torch.float32
  Computation dtype (matmul): torch.float32
  Total tokens generated: 44685
  Avg tokens/sequence: 69.8
  Completion lengths - min: 23, max: 512, mean: 69.8, std: 102.3
  Sequences with EOS: 432/640
```

**What to look for**:
- **Low EOS presence**: Many sequences hitting max_new_tokens limit
- **High variance in lengths**: std close to mean indicates inconsistent generations
- **Dtype mismatches**: Mixed precision issues

---

### 5. `"enabled": True` - GRPODebugLogger
**When to use**: Always enable when using any debug flags.

**What it does**:
- Creates a debug log directory (default: `debug_logs/`)
- Logs episode-level statistics
- Required infrastructure for other debug features

---

## Complete Configuration Example

```python
config = {
    "model": {
        "hf_model_name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    },
    "training": {
        "num_episodes": 100,
        "batch_size": 8,
        "group_size": 8,
        "minibatch_size": 64,
        "rollout_batch_size": 4,
        "learning_rate": 1e-6,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "kl_coef": 0.1,
        "clip_epsilon": 0.2,
        "gradient_clip": 1.0,
        "update_epochs": 1,
    },
    "debug": {
        # Enable debugging infrastructure
        "enabled": True,
        "log_dir": "debug_logs",

        # Debug flags (enable as needed)
        "loss": True,         # ⭐ For loss spike investigation
        "advantages": True,    # ⭐ For reward/advantage issues
        "gradients": True,     # ⭐ For gradient explosion
        "generation": False,   # For generation issues (verbose)
    },
    "logging": {
        "log_interval": 1,
        "save_interval": 10,
        "show_trajectory_progress": False,
    },
}
```

---

## Debugging Workflow for Loss Spikes

### Step 1: Enable Core Debug Flags
```python
"debug": {
    "enabled": True,
    "loss": True,
    "advantages": True,
    "gradients": True,
}
```

### Step 2: Run Training and Monitor Output
Watch for the episode with the spike. The debug output will show detailed information for that episode.

### Step 3: Analyze the Output

**If you see**:
```
TOTAL LOSS: 35957.7164
  = policy_loss: 35950.2341  ← ⚠️ This is the problem!
  + kl_term: 7.4823
  - entropy_term: 0.0000
```
→ **Problem**: Policy loss is the main contributor

**Then check**:
```
ratio - min: 0.0001, max: 856.3421  ← ⚠️ Extreme ratio!
ratio > 10: 23 sequences  ← ⚠️ Many extreme ratios!
```
→ **Root cause**: Policy changed too much between old and new

**And investigate**:
```
🔍 ADVANTAGE COMPUTATION DEBUG:
  WARNING: 5 groups have std < 1e-3  ← ⚠️ Numerical instability!
  WARNING: 15 advantages were clipped  ← ⚠️ Extreme normalization!
```
→ **Contributing factor**: Advantage normalization issues

**Finally check gradients**:
```
🔍 GRADIENT DEBUG:
  Gradient norm before clipping: 1245.2341  ← ⚠️ Very large!
  WARNING: Gradients were clipped
```
→ **Result**: Gradients are exploding due to the large loss

### Step 4: Apply Fixes

Based on the analysis above, you might:

1. **Extreme ratios detected**:
   - Decrease learning rate (e.g., 1e-6 → 5e-7)
   - Tighten clip_epsilon (e.g., 0.2 → 0.1)
   - Increase advantage epsilon in compute_advantages (already set to 1e-2)

2. **Small group std detected**:
   - Increase group_size (more diverse samples per group)
   - Check reward function for returning identical rewards
   - Increase advantage normalization epsilon (already at 1e-2, could go to 1e-1)

3. **Gradient explosion**:
   - Lower learning rate
   - Decrease gradient_clip threshold (1.0 → 0.5)
   - Add warmup to learning rate

4. **Vanishing entropy**:
   - Increase temperature (0.7 → 0.9)
   - Increase entropy_coef (default 0.01 → 0.05)

---

## Performance Impact

Debug flags have minimal performance impact:
- `"loss": True` - Adds ~0.1% overhead (only during loss computation)
- `"advantages": True` - Adds ~0.05% overhead (only during advantage computation)
- `"gradients": True` - Adds ~0.2% overhead (only during gradient clipping)
- `"generation": True` - Adds ~1% overhead (during text generation)

You can safely leave them enabled during training for continuous monitoring.

---

## Disabling Debug Output

To disable debug output, set all flags to `False`:

```python
"debug": {
    "enabled": False,  # Disables GRPODebugLogger
    "loss": False,
    "advantages": False,
    "gradients": False,
    "generation": False,
}
```

Or simply omit the debug section entirely from your config.

---

## Tips

1. **Start minimal**: Enable only `loss` debug first, then add others as needed
2. **Watch for patterns**: Debug output at each minibatch helps identify when spikes occur
3. **Compare episodes**: Compare debug output from stable vs. spike episodes
4. **Save output**: Redirect stdout to a file to analyze later: `python train.py > debug_output.txt 2>&1`
5. **Use with W&B**: Debug output complements W&B metrics for deeper investigation
