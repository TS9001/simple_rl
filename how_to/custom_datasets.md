# How to Train on Custom Datasets

## Overview

GRPO can train on any dataset that provides prompts. This guide shows how to use different data sources.

## Using HuggingFace Datasets

### Load from Hub

```python
from datasets import load_dataset

# Load a dataset
dataset = load_dataset("dataset_name", split="train")

# Extract prompts
prompts = [item["text"] for item in dataset]

# Use in training
for episode in range(num_episodes):
    batch_prompts = random.sample(prompts, batch_size)
    grpo.train_step({"prompts": batch_prompts})
```

### Popular Datasets

```python
# Math problems
dataset = load_dataset("gsm8k", "main", split="train")
prompts = [item["question"] for item in dataset]

# Instructions
dataset = load_dataset("tatsu-lab/alpaca", split="train")
prompts = [item["instruction"] for item in dataset]

# Conversations
dataset = load_dataset("ShareGPT", split="train")
prompts = [item["conversations"][0]["value"] for item in dataset]
```

## Using Local Files

### CSV Files

```python
import pandas as pd

# Load CSV
df = pd.read_csv("data/prompts.csv")
prompts = df["prompt"].tolist()

# With additional metadata
answers = df["answer"].tolist() if "answer" in df else None
```

### JSON/JSONL Files

```python
import json

# JSON file
with open("data/prompts.json", "r") as f:
    data = json.load(f)
    prompts = [item["prompt"] for item in data]

# JSONL file (one JSON per line)
prompts = []
with open("data/prompts.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        prompts.append(item["prompt"])
```

### Text Files

```python
# One prompt per line
with open("data/prompts.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

# With delimiters
with open("data/prompts.txt", "r") as f:
    content = f.read()
    prompts = content.split("\n---\n")  # Split by delimiter
```

## Creating a Custom Dataset Class

```python
class CustomDataset:
    def __init__(self, data_path):
        self.prompts = self.load_data(data_path)
        self.index = 0
    
    def load_data(self, path):
        # Your loading logic
        prompts = []
        # ... load from path
        return prompts
    
    def get_batch(self, batch_size):
        """Get a batch of prompts."""
        if self.index + batch_size > len(self.prompts):
            # Shuffle and reset
            random.shuffle(self.prompts)
            self.index = 0
        
        batch = self.prompts[self.index:self.index + batch_size]
        self.index += batch_size
        return batch
    
    def __len__(self):
        return len(self.prompts)

# Use with GRPO
dataset = CustomDataset("data/my_data.json")

for episode in range(num_episodes):
    batch_prompts = dataset.get_batch(batch_size)
    grpo.train_step({"prompts": batch_prompts})
```

## Dataset with Answers (for Reward Calculation)

```python
class DatasetWithAnswers:
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
    
    def load_data(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "prompt": item["question"],
                    "answer": item["answer"]
                })
        return data
    
    def get_batch(self, batch_size):
        batch = random.sample(self.data, batch_size)
        prompts = [item["prompt"] for item in batch]
        answers = [item["answer"] for item in batch]
        return prompts, answers

# Custom reward using answers
def reward_with_answers(prompt, completion, correct_answer):
    # Check if completion matches correct_answer
    if correct_answer in completion:
        return 1.0
    return -0.5

# Training loop
dataset = DatasetWithAnswers("data/qa_pairs.jsonl")

for episode in range(num_episodes):
    prompts, answers = dataset.get_batch(batch_size)
    
    # Store answers for reward calculation
    CORRECT_ANSWERS = dict(zip(prompts, answers))
    
    grpo.train_step({"prompts": prompts})
```

## Streaming Large Datasets

```python
from datasets import load_dataset

# Stream dataset without loading all into memory
dataset = load_dataset(
    "large_dataset",
    split="train",
    streaming=True
)

# Create iterator
data_iter = iter(dataset)

def get_batch(iterator, batch_size):
    batch = []
    for _ in range(batch_size):
        try:
            item = next(iterator)
            batch.append(item["text"])
        except StopIteration:
            # Reset iterator
            iterator = iter(dataset)
            item = next(iterator)
            batch.append(item["text"])
    return batch

# Training
for episode in range(num_episodes):
    batch_prompts = get_batch(data_iter, batch_size)
    grpo.train_step({"prompts": batch_prompts})
```

## Data Preprocessing

### Cleaning Prompts

```python
def clean_prompt(text):
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters if needed
    text = text.replace("\n", " ")
    
    # Truncate if too long
    max_length = 500
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text

prompts = [clean_prompt(p) for p in raw_prompts]
```

### Filtering Prompts

```python
def filter_prompts(prompts):
    filtered = []
    for prompt in prompts:
        # Skip too short
        if len(prompt.split()) < 3:
            continue
        
        # Skip too long
        if len(prompt.split()) > 200:
            continue
        
        # Skip inappropriate content
        if any(word in prompt.lower() for word in blacklist):
            continue
        
        filtered.append(prompt)
    
    return filtered

prompts = filter_prompts(raw_prompts)
```

## Dataset Augmentation

```python
def augment_prompts(prompts):
    augmented = []
    
    for prompt in prompts:
        # Original
        augmented.append(prompt)
        
        # Add instruction variations
        augmented.append(f"Please {prompt}")
        augmented.append(f"Can you {prompt}?")
        
        # Add context
        augmented.append(f"Context: General knowledge\n{prompt}")
    
    return augmented

prompts = augment_prompts(original_prompts)
```

## Curriculum Learning

```python
class CurriculumDataset:
    def __init__(self, easy_data, medium_data, hard_data):
        self.datasets = {
            "easy": easy_data,
            "medium": medium_data,
            "hard": hard_data
        }
        self.difficulty = "easy"
    
    def set_difficulty(self, level):
        self.difficulty = level
    
    def get_batch(self, batch_size):
        return random.sample(self.datasets[self.difficulty], batch_size)

# Training with curriculum
curriculum = CurriculumDataset(easy, medium, hard)

for episode in range(num_episodes):
    # Progress through curriculum
    if episode < 100:
        curriculum.set_difficulty("easy")
    elif episode < 200:
        curriculum.set_difficulty("medium")
    else:
        curriculum.set_difficulty("hard")
    
    batch = curriculum.get_batch(batch_size)
    grpo.train_step({"prompts": batch})
```

## Dataset Statistics

```python
def analyze_dataset(prompts):
    """Print dataset statistics."""
    lengths = [len(p.split()) for p in prompts]
    
    print(f"Total prompts: {len(prompts)}")
    print(f"Average length: {np.mean(lengths):.1f} words")
    print(f"Min/Max length: {min(lengths)}/{max(lengths)} words")
    print(f"Unique prompts: {len(set(prompts))}")
    
    # Length distribution
    plt.hist(lengths, bins=50)
    plt.xlabel("Prompt Length (words)")
    plt.ylabel("Count")
    plt.title("Prompt Length Distribution")
    plt.show()

analyze_dataset(prompts)
```

## Best Practices

1. **Shuffle Data**: Randomize order to avoid patterns
2. **Balance Dataset**: Ensure diverse prompt types
3. **Validate Data**: Check for errors/duplicates
4. **Cache Preprocessed Data**: Save time on repeated runs
5. **Monitor Coverage**: Track which prompts are used

## Example: Complete Custom Dataset Setup

```python
# Complete example with math dataset
from datasets import load_dataset
import random

# Load and prepare dataset
print("Loading dataset...")
dataset = load_dataset("gsm8k", "main", split="train[:1000]")

# Extract prompts and answers
math_problems = []
for item in dataset:
    math_problems.append({
        "prompt": item["question"],
        "answer": item["answer"].split("####")[-1].strip()
    })

print(f"Loaded {len(math_problems)} problems")

# Create reward function using answers
def math_reward(prompt, completion):
    # Find the correct answer
    for problem in math_problems:
        if problem["prompt"] == prompt:
            correct = problem["answer"]
            if correct in completion:
                return 2.0  # Correct
            return -0.5  # Wrong
    return 0.0  # Unknown prompt

# Initialize GRPO
grpo = GRPO(
    config=config,
    reward_fn=math_reward
)

# Training loop
for episode in range(100):
    # Sample batch
    batch = random.sample(math_problems, 4)
    batch_prompts = [p["prompt"] for p in batch]
    
    # Train
    metrics = grpo.train_step({"prompts": batch_prompts})
    
    if episode % 10 == 0:
        print(f"Episode {episode}: Loss={metrics['total_loss']:.4f}")
```