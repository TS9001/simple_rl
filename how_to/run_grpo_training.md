# How to Run GRPO Training

## Quick Start

1. **Launch Jupyter:**
   ```bash
   ./run_jupyter.sh
   ```

2. **Open the notebook:**
   - Select `grpo_qwen_training.ipynb`

3. **Run all cells:**
   - Click "Cell" → "Run All"
   - Or press `Shift+Enter` on each cell

## Training from Command Line

```python
from simple_rl.algorithms.grpo import GRPO

# Basic configuration
config = {
    "algorithm": {
        "group_size": 4,
        "kl_coef": 0.05,
        "normalize_rewards": True,
    },
    "training": {
        "batch_size": 8,
        "learning_rate": 1e-5,
        "num_episodes": 100,
    },
    "generation": {
        "system_prompt": "You are a helpful assistant.",
        "prompt_template": "User: {prompt}\nAssistant:",
    }
}

# Initialize GRPO
grpo = GRPO(config=config)

# Train
grpo.train(num_episodes=100)
```

## Configuration Options

### Algorithm Parameters

- `group_size`: Number of completions per prompt (default: 4)
- `kl_coef`: KL divergence penalty coefficient (default: 0.05)
- `clip_range`: PPO-style clipping (default: 0.2)
- `normalize_rewards`: Normalize rewards within groups (default: True)
- `update_epochs`: Number of PPO update epochs (default: 2)

### Training Parameters

- `batch_size`: Total batch size (must be divisible by group_size)
- `learning_rate`: Learning rate for optimizer
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Generation temperature
- `gradient_clip`: Gradient clipping value

### Generation Parameters

- `system_prompt`: System message for the model
- `prompt_template`: Template with `{prompt}` placeholder
- `response_prefix`: Text to add before model response

## Monitoring Training

Training metrics are logged every N episodes:
- Total loss
- Policy gradient loss
- KL divergence
- Mean reward

Checkpoints are saved to `checkpoints/` directory.

## Using Different Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a different model
model = AutoModelForCausalLM.from_pretrained("model_name")
tokenizer = AutoTokenizer.from_pretrained("model_name")

# Use with GRPO
grpo = GRPO(
    model=model,
    tokenizer=tokenizer,
    config=config
)
```

## GPU vs CPU Training

The setup automatically detects GPU availability. To force CPU usage:

```python
import torch
device = torch.device("cpu")
model = model.to(device)
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `max_new_tokens`
- Use smaller model
- Use gradient accumulation

### Slow Training
- Check GPU is being used
- Reduce `group_size`
- Use mixed precision training

### Poor Performance
- Adjust `kl_coef` (lower = more exploration)
- Improve reward function
- Try different prompt formats
- Increase training episodes