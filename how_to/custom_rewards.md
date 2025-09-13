# How to Create Custom Reward Functions

## Overview

The reward function is crucial for GRPO training. It determines what behaviors the model learns to produce.

## Basic Reward Function

```python
def simple_reward(prompt: str, completion: str) -> float:
    """
    Basic reward function.
    
    Args:
        prompt: The input prompt
        completion: Model's generated text
        
    Returns:
        Reward score (typically -1 to 1)
    """
    reward = 0.0
    
    # Reward for appropriate length
    length = len(completion.split())
    if 10 <= length <= 100:
        reward += 0.5
    
    return reward
```

## Using Custom Rewards with GRPO

```python
from simple_rl.algorithms.grpo import GRPO

# Define your reward function
def my_reward(prompt, completion):
    # Your logic here
    return reward_score

# Initialize GRPO with custom reward
grpo = GRPO(
    config=config,
    reward_fn=my_reward
)
```

## Example Reward Functions

### 1. Math Problem Rewards

```python
import re

def math_reward(prompt: str, completion: str) -> float:
    """Reward correct mathematical answers."""
    reward = 0.0
    
    # Extract answer from completion
    answer_match = re.search(r"answer is:?\s*([-]?\d+\.?\d*)", completion.lower())
    if not answer_match:
        return -1.0  # No answer found
    
    model_answer = float(answer_match.group(1))
    
    # Get correct answer (you need to provide this)
    correct_answer = get_correct_answer(prompt)
    
    # Check correctness
    if abs(model_answer - correct_answer) < 0.01:
        reward = 2.0  # Correct
    elif abs(model_answer - correct_answer) < correct_answer * 0.1:
        reward = 0.5  # Close
    else:
        reward = -0.5  # Wrong
    
    # Bonus for showing work
    if "step" in completion.lower():
        reward += 0.3
    
    return reward
```

### 2. Code Quality Rewards

```python
import ast

def code_reward(prompt: str, completion: str) -> float:
    """Reward syntactically correct, well-structured code."""
    reward = 0.0
    
    # Extract code block
    code = extract_code_block(completion)
    if not code:
        return -1.0
    
    # Check syntax
    try:
        ast.parse(code)
        reward += 1.0  # Valid syntax
    except SyntaxError:
        return -1.0  # Invalid syntax
    
    # Check for good practices
    if "def " in code:  # Has functions
        reward += 0.3
    if '"""' in code or "'''" in code:  # Has docstrings
        reward += 0.3
    if "import" in code:  # Uses libraries appropriately
        reward += 0.2
    
    # Penalize bad practices
    if "exec(" in code or "eval(" in code:
        reward -= 0.5
    
    return reward
```

### 3. Safety and Helpfulness Rewards

```python
def safety_reward(prompt: str, completion: str) -> float:
    """Reward helpful, harmless responses."""
    reward = 0.0
    
    # Check for harmful content
    harmful_keywords = ["dangerous", "illegal", "harmful"]
    for keyword in harmful_keywords:
        if keyword in completion.lower():
            return -2.0  # Strong penalty
    
    # Check for helpfulness indicators
    helpful_phrases = [
        "here's how", "you can", "try this",
        "solution is", "answer is"
    ]
    for phrase in helpful_phrases:
        if phrase in completion.lower():
            reward += 0.3
            break
    
    # Check for appropriate refusals
    if "cannot" in completion or "unable" in completion:
        if any(harm in prompt.lower() for harm in harmful_keywords):
            reward += 1.0  # Good refusal
        else:
            reward -= 0.5  # Unnecessary refusal
    
    return reward
```

### 4. Factual Accuracy Rewards

```python
def factual_reward(prompt: str, completion: str) -> float:
    """Reward factually accurate information."""
    reward = 0.0
    
    # Use a fact-checking model or database
    facts = extract_factual_claims(completion)
    
    for fact in facts:
        if verify_fact(fact):  # Your verification logic
            reward += 0.5
        else:
            reward -= 1.0  # Penalize false information
    
    # Reward citations or sources
    if "according to" in completion.lower() or "source:" in completion.lower():
        reward += 0.3
    
    return reward
```

## Combining Multiple Reward Signals

```python
def combined_reward(prompt: str, completion: str) -> float:
    """Combine multiple reward signals."""
    
    # Individual rewards
    length_reward = compute_length_reward(completion)
    quality_reward = compute_quality_reward(completion)
    safety_reward = compute_safety_reward(completion)
    
    # Weighted combination
    weights = {
        "length": 0.2,
        "quality": 0.5,
        "safety": 0.3
    }
    
    total_reward = (
        weights["length"] * length_reward +
        weights["quality"] * quality_reward +
        weights["safety"] * safety_reward
    )
    
    return total_reward
```

## Using External Reward Models

```python
from transformers import AutoModelForSequenceClassification

class RewardModel:
    def __init__(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def __call__(self, prompt: str, completion: str) -> float:
        # Tokenize input
        inputs = self.tokenizer(
            prompt + completion,
            return_tensors="pt",
            truncation=True
        )
        
        # Get reward score
        with torch.no_grad():
            outputs = self.model(**inputs)
            reward = outputs.logits.squeeze().item()
        
        return reward

# Use with GRPO
reward_model = RewardModel("reward-model-name")
grpo = GRPO(config=config, reward_fn=reward_model)
```

## Best Practices

### 1. Normalize Rewards
Keep rewards in a consistent range (e.g., -1 to 1):
```python
def normalize_reward(raw_reward, min_val=-10, max_val=10):
    return 2 * (raw_reward - min_val) / (max_val - min_val) - 1
```

### 2. Avoid Reward Hacking
Be specific about what you want:
```python
# Bad: Easy to game
if len(completion) > 50:
    reward += 1

# Good: More specific
if 50 <= len(completion.split()) <= 200:
    reward += 0.5
```

### 3. Test Your Rewards
```python
test_cases = [
    ("prompt1", "good completion", expected_reward),
    ("prompt2", "bad completion", expected_reward),
]

for prompt, completion, expected in test_cases:
    reward = my_reward_fn(prompt, completion)
    print(f"Reward: {reward:.2f} (expected: {expected:.2f})")
```

### 4. Log Reward Statistics
Monitor reward distribution during training:
```python
import numpy as np

rewards = []
for batch in training_data:
    batch_rewards = [reward_fn(p, c) for p, c in batch]
    rewards.extend(batch_rewards)

print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Std reward: {np.std(rewards):.2f}")
print(f"Min/Max: {np.min(rewards):.2f}/{np.max(rewards):.2f}")
```

## Common Pitfalls

1. **Too Sparse**: Rewards only for perfect outputs
2. **Too Dense**: Rewards for trivial features
3. **Unbounded**: Rewards can grow infinitely
4. **Inconsistent**: Same output gets different rewards
5. **Gameable**: Easy to exploit patterns

## Debugging Rewards

```python
def debug_reward(prompt: str, completion: str) -> float:
    """Reward function with debugging output."""
    reward = 0.0
    components = {}
    
    # Length component
    length = len(completion.split())
    if 10 <= length <= 100:
        components["length"] = 0.5
        reward += 0.5
    
    # Quality component
    if "step" in completion:
        components["has_steps"] = 0.3
        reward += 0.3
    
    # Print breakdown
    print(f"Reward breakdown: {components}")
    print(f"Total reward: {reward:.2f}")
    
    return reward
```