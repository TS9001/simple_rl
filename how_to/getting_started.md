# Getting Started with Simple RL

## Prerequisites

- Python 3.8 or higher
- Unix-like environment (Linux, macOS, WSL on Windows)
- 4GB+ RAM (8GB+ recommended for training)

## Quick Setup

1. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
   This will:
   - Install `uv` package manager if not present
   - Create a virtual environment
   - Install all dependencies
   - Configure Jupyter notebooks

2. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Launch Jupyter notebooks:**
   ```bash
   ./run_jupyter.sh
   ```

## Project Structure

```
simple_rl/
├── setup.sh           # Main setup script (uses uv)
├── run_jupyter.sh     # Launch Jupyter notebooks
├── run_tests.sh       # Run test suite
├── .venv/            # Virtual environment (created by setup)
├── simple_rl/        # Main package source
│   └── algorithms/
│       └── grpo.py   # GRPO implementation
├── notebooks/        # Jupyter notebooks
│   └── grpo_qwen_training.ipynb
├── tests/           # Test suite
├── how_to/          # Documentation
└── checkpoints/     # Model checkpoints
```

## Available Scripts

- `./setup.sh` - Initial setup with uv
- `./run_jupyter.sh` - Launch Jupyter
- `./run_tests.sh [type]` - Run tests
  - `grpo` - GRPO tests only
  - `quick` - Fast tests only
  - `coverage` - With coverage report
  - `all` - All tests (default)

## Environment Management with uv

We use `uv` for fast, reliable dependency management:

```bash
# Install a package
uv pip install package_name

# List installed packages
uv pip list

# Upgrade a package
uv pip install --upgrade package_name
```

## Next Steps

1. [Run GRPO Training](run_grpo_training.md)
2. [Customize Prompts](customize_prompts.md)
3. [Create Custom Rewards](custom_rewards.md)
4. [Train on Your Dataset](custom_datasets.md)