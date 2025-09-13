# GRPO (Group Relative Policy Optimization) Implementation Plan

## 1. What We're Implementing
A standard, pure GRPO algorithm implementation that performs relative reward computation within groups/batches for policy gradient optimization. GRPO normalizes rewards relative to other samples in the same batch to reduce variance and improve training stability.

## 2. Why We're Implementing This
- GRPO is effective for RLHF (Reinforcement Learning from Human Feedback) tasks
- Reduces variance in policy gradient estimates through relative reward computation
- More stable training compared to vanilla policy gradient methods
- Useful for language model fine-tuning and text generation tasks

## 3. Problem It Solves
- High variance in standard policy gradient methods
- Instability when training on sparse or noisy reward signals
- Need for a simple but effective RL algorithm for text generation tasks
- Baseline-free variance reduction through group normalization

## 4. Affected Code Areas

### New Files to Create:
- File: `simple_rl/algorithms/grpo.py`
  - Main GRPO algorithm implementation
  - Inherits from BaseAlgorithm
  
- File: `simple_rl/models/policy_model.py`
  - Policy network for text generation
  - HuggingFace transformer integration
  
- File: `tests/test_grpo.py`
  - Unit tests for GRPO algorithm
  - Test reward normalization
  - Test policy updates

### Files to Modify:
- File: `simple_rl/algorithms/__init__.py`
  - Register GRPO algorithm
  
- File: `configs/grpo_config.yaml`
  - GRPO-specific configuration
  
- File: `simple_rl/utils/buffers.py` (if needed)
  - Add GRPO-specific buffer for trajectory storage

## 5. Implementation Details

### Phase 1: Core GRPO Algorithm
**File: `simple_rl/algorithms/grpo.py`**
- Class `GRPO(BaseAlgorithm)`:
  - `__init__()`: Initialize policy network, optimizer
  - `compute_relative_rewards()`: Normalize rewards within batch
  - `compute_policy_loss()`: GRPO loss with KL penalty
  - `train_step()`: Single training iteration
  - `generate_trajectories()`: Sample actions from policy
  - `train()`: Main training loop
  - `evaluate()`: Evaluation method

### Phase 2: Policy Model
**File: `simple_rl/models/policy_model.py`**
- Class `LanguageModel(BaseModel)`:
  - Support for HuggingFace transformers
  - `forward()`: Generate logits for next token
  - `generate()`: Text generation method
  - `compute_log_probs()`: Log probability computation

### Phase 3: Configuration
**File: `configs/grpo_config.yaml`**
```yaml
algorithm:
  name: grpo
  group_size: 8  # Number of samples for relative reward
  kl_coef: 0.1   # KL divergence coefficient
  clip_range: 0.2  # Optional PPO-style clipping
  
model:
  type: policy
  hf_model_name: "gpt2"  # Or any HF model
  
training:
  batch_size: 32
  learning_rate: 1e-5
  num_epochs: 100
```

### Phase 4: Testing
**File: `tests/test_grpo.py`**
- Test relative reward computation
- Test policy loss calculation
- Test KL divergence computation
- Test end-to-end training step

## 6. Testing Strategy

### Unit Tests:
1. **Test Relative Rewards**:
   - Input: Raw rewards [1, 2, 3, 4]
   - Expected: Normalized rewards with zero mean
   
2. **Test Policy Loss**:
   - Verify loss computation with mock data
   - Check gradient flow
   
3. **Test KL Divergence**:
   - Ensure KL penalty is computed correctly
   - Test with identical and different distributions

### Integration Tests:
1. **Test Training Loop**:
   - Run mini training on toy data
   - Verify loss decreases
   
2. **Test with HuggingFace Model**:
   - Load small model (distilgpt2)
   - Verify generation works

### Manual Testing:
1. Train on simple text completion task
2. Monitor loss curves with W&B
3. Evaluate generated text quality

## 7. Implementation Phases

### Phase 1: Basic Structure (Day 1)
- Create GRPO class skeleton
- Implement relative reward computation
- Basic configuration setup

### Phase 2: Core Algorithm (Day 2)
- Implement policy loss
- Add KL divergence penalty
- Complete training loop

### Phase 3: Model Integration (Day 3)
- Integrate HuggingFace models
- Add generation capabilities
- Text-specific utilities

### Phase 4: Testing & Refinement (Day 4)
- Write comprehensive tests
- Debug and optimize
- Documentation

## 8. Success Criteria
- [ ] GRPO algorithm runs without errors
- [ ] Loss decreases during training
- [ ] Can generate coherent text with HF models
- [ ] All tests pass
- [ ] W&B logging works correctly
- [ ] Configuration is flexible and clear

## 9. Potential Challenges
- Memory management with large language models
- Efficient batch processing for text data
- Balancing KL penalty vs reward optimization
- Handling variable-length sequences

## 10. Future Extensions (Not in initial implementation)
- Advanced reward models
- Multi-GPU training support
- Adaptive KL coefficient
- Different normalization strategies