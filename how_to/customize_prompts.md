# How to Customize Prompts in GRPO

## Overview

GRPO supports dynamic prompt customization to control how the model receives and processes inputs.

## Basic Prompt Configuration

Set prompts in the config:

```python
config = {
    "generation": {
        "system_prompt": "You are a helpful assistant.",
        "prompt_template": "User: {prompt}\nAssistant:",
        "response_prefix": " ",
    }
}
```

## Dynamic Prompt Switching

Change prompts during runtime:

```python
# Initialize GRPO
grpo = GRPO(config=config)

# Switch to different format
grpo.set_generation_prompt(
    system_prompt="You are an expert mathematician.",
    prompt_template="Problem: {prompt}\nSolution:",
    response_prefix=" Let me solve this:\n"
)
```

## Common Prompt Formats

### 1. Instruction Format (Alpaca-style)
```python
grpo.set_generation_prompt(
    system_prompt="Below is an instruction that describes a task.",
    prompt_template="### Instruction:\n{prompt}\n\n### Response:",
    response_prefix=" "
)
```

### 2. Chat Format (ChatML)
```python
grpo.set_generation_prompt(
    system_prompt="<|im_start|>system\nYou are helpful.<|im_end|>",
    prompt_template="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant",
    response_prefix="\n"
)
```

### 3. Chain-of-Thought
```python
grpo.set_generation_prompt(
    system_prompt="Think step by step.",
    prompt_template="{prompt}\n\nLet's think step by step:",
    response_prefix=" "
)
```

### 4. Math Problem Solving
```python
grpo.set_generation_prompt(
    system_prompt="Solve the problem showing all work.",
    prompt_template="Problem: {prompt}\n\nStep-by-step solution:",
    response_prefix="\n1. "
)
```

## Testing Prompt Formats

```python
# Test how prompts are formatted
test_prompt = "What is 2 + 2?"
formatted = grpo.format_prompt(test_prompt)
print(f"Formatted prompt:\n{formatted}")
```

## Disabling Formatting

Generate without prompt formatting:

```python
trajectories = grpo.generate_trajectories(
    prompts=["raw prompt"],
    num_samples_per_prompt=4,
    use_formatting=False  # Disable formatting
)
```

## Prompt Format Guidelines

### DO:
- Keep system prompts concise
- Use clear delimiters between sections
- Match the format your model was trained on
- Test different formats for your task

### DON'T:
- Make prompts too long (wastes tokens)
- Use ambiguous formatting
- Mix different format styles

## Examples for Different Tasks

### Code Generation
```python
grpo.set_generation_prompt(
    system_prompt="You are a Python expert.",
    prompt_template="# Task: {prompt}\n# Code:",
    response_prefix="\n```python\n"
)
```

### Translation
```python
grpo.set_generation_prompt(
    system_prompt="Translate to French.",
    prompt_template="English: {prompt}\nFrench:",
    response_prefix=" "
)
```

### Summarization
```python
grpo.set_generation_prompt(
    system_prompt="Provide concise summaries.",
    prompt_template="Text: {prompt}\n\nSummary:",
    response_prefix=" "
)
```

## Switching Formats During Training

```python
# Start with one format
grpo.set_generation_prompt(...)
metrics1 = grpo.train_step(batch1)

# Switch to another format
grpo.set_generation_prompt(...)
metrics2 = grpo.train_step(batch2)

# Reset to original
grpo.set_generation_prompt(reset=True)
```

## Best Practices

1. **Match Training Data**: Use formats similar to model's pretraining
2. **Be Consistent**: Don't change formats too frequently
3. **Test Empirically**: Different formats work better for different tasks
4. **Document Your Choice**: Keep track of which format works best