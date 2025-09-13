# Simple RL

Educational reinforcement learning implementations with GRPO focus.

## Setup

```bash
./scripts/setup.sh  # Installs uv, creates venv, installs deps
```

## Usage

### Run GRPO Training
```bash
./scripts/run_jupyter.sh  # Opens notebook for training
```

### Run Tests
```bash
./scripts/run_tests.sh grpo  # Test GRPO implementation
```

## GRPO Quick Start

```python
from simple_rl.algorithms.grpo import GRPO

config = {
    "algorithm": {
        "group_size": 4,  # Completions per prompt
        "kl_coef": 0.05,  # KL penalty
    },
    "training": {
        "batch_size": 8,
        "learning_rate": 1e-5,
        "num_episodes": 100,
    }
}

def reward_fn(prompt: str, completion: str) -> float:
    # Your reward logic
    return 1.0 if correct else -0.5

grpo = GRPO(config=config, reward_fn=reward_fn)

# Train
for episode in range(100):
    batch = {"prompts": your_prompts}
    grpo.train_step(batch)
```

## Project Structure

```
simple_rl/
├── simple_rl/
│   ├── algorithms/       # RL algorithms (GRPO, etc.)
│   ├── models/          # Model definitions
│   ├── utils/           # Utilities
│   └── config/          # Config files
├── notebooks/           # Training notebooks
├── tests/              # Test suite
├── scripts/            # Setup & run scripts
└── how_to/             # Quick guides
```

## Notebooks

- `notebooks/grpo_qwen_training.ipynb` - Full GRPO training with Qwen 0.5B on GSM8K

## Documentation

See `how_to/` folder for:
- Dataset loading patterns
- Reward function examples
- Prompt customization

## Development

```bash
# Format & lint
black . && isort . && flake8 .

# Run tests
pytest
```

# Disclaimer
- AI tools like claude code or GPT5 were used to generate parts of the code, It makes development faster
  documentation easier and basic building blocks can be implemented really quickly.
- 