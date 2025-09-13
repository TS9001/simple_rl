# 🚀 Quick Start Guide - GRPO Training with Jupyter

This guide will get you up and running with GRPO (Group Relative Policy Optimization) training in under 5 minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- (Optional) CUDA-capable GPU for faster training

## 🎯 One-Line Setup

### Option 1: Bash Script (Linux/Mac)
```bash
./setup_and_run.sh
```

### Option 2: Python Script (All Platforms)
```bash
python quick_start.py
```

Both scripts will:
1. ✅ Create a virtual environment
2. ✅ Install all dependencies
3. ✅ Set up Jupyter notebooks
4. ✅ Launch Jupyter in your browser

## 📦 Manual Installation

If you prefer to install manually:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements-grpo.txt

# Install simple_rl package
pip install -e .

# Launch Jupyter
jupyter notebook
```

## 📓 Available Notebooks

### 1. **grpo_qwen_training.ipynb**
Train Qwen 0.5B model for mathematical reasoning using GRPO.

**Features:**
- 🧮 GSM8K math dataset
- 🎯 Custom prompt formatting
- 📊 Math-specific reward function
- 🔄 Dynamic prompt customization
- 📈 Real-time training visualization

**What You'll Learn:**
- How to format prompts for different tasks
- Implementing custom reward functions
- Training with GRPO algorithm
- Evaluating model performance

## 🎮 Quick Usage

### Start Training (Notebook)
1. Run the setup script
2. Select `grpo_qwen_training.ipynb`
3. Run all cells with `Shift+Enter`
4. Watch the model learn math!

### Start Training (Python)
```python
from simple_rl.algorithms.grpo import GRPO

# Configure GRPO
config = {
    "algorithm": {
        "group_size": 4,
        "kl_coef": 0.05,
    },
    "generation": {
        "system_prompt": "You are a helpful assistant.",
        "prompt_template": "User: {prompt}\nAssistant:",
    }
}

# Initialize and train
grpo = GRPO(model=None, config=config)
grpo.train(num_episodes=100)
```

## 🔧 Customization

### Change Prompt Format
```python
# Switch to different prompt styles dynamically
grpo.set_generation_prompt(
    system_prompt="You are a math tutor.",
    prompt_template="Problem: {prompt}\nSolution:",
    response_prefix=" Let me solve this: "
)
```

### Use Different Datasets
```python
# Load any HuggingFace dataset
from datasets import load_dataset
dataset = load_dataset("your_dataset", split="train")
```

### Custom Reward Functions
```python
def my_reward_function(prompt, completion):
    # Your reward logic here
    return reward_score

grpo = GRPO(reward_fn=my_reward_function)
```

## 🏃 Quick Commands

### Train with Default Settings
```bash
python -c "from simple_rl.algorithms.grpo import GRPO; GRPO().train(100)"
```

### Launch Jupyter
```bash
source venv/bin/activate && jupyter notebook
```

### Run Tests
```bash
pytest tests/test_grpo*.py -v
```

## 📊 Monitor Training

Training metrics are automatically logged:
- **Loss curves**: Policy gradient loss, KL divergence
- **Rewards**: Mean reward per episode
- **Checkpoints**: Saved every N episodes

## 🆘 Troubleshooting

### Issue: "No module named torch"
**Solution:** Run `pip install torch` or use the setup script

### Issue: "CUDA out of memory"
**Solution:** Reduce batch_size in config or use CPU

### Issue: "Jupyter kernel not found"
**Solution:** Run `python -m ipykernel install --user --name=simple_rl`

## 📚 Learn More

- [Full Documentation](README.md)
- [GRPO Paper](https://arxiv.org/...)
- [Example Notebooks](notebooks/)

## 💡 Tips

1. **Start Small**: Use small models (0.5B-1B params) for testing
2. **Monitor KL**: Keep KL divergence < 10 for stable training
3. **Reward Design**: Spend time on good reward functions
4. **Prompt Engineering**: Test different prompt formats
5. **Save Often**: Enable checkpointing for long runs

## 🎉 Ready to Train!

Run the quick start script and begin training your first GRPO model:

```bash
python quick_start.py
```

Happy training! 🚀