import re
import torch
import numpy as np
from typing import List, Optional, Tuple

# Precompile all patterns for efficiency
NUMBER_PATTERN = re.compile(r'-?\d+\.?\d*')
REASONING_PATTERN = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
MATH_OPERATIONS = re.compile(r'[\+\-\*\/\=]|\d+\.?\d*|sum|total|each|per|cost|earn|spend')
STEP_INDICATORS = re.compile(r'step|first|then|next|finally|so|therefore|thus')
GARBAGE_PATTERNS = re.compile(r'<\|im_start\|>|<\|im_end\|>|\.\.\.|Respond in the following format')

def extract_answer_from_model_output(text):
    match = ANSWER_PATTERN.search(text)
    if match:
        answer = match.group(1).strip()
        if answer and answer != "..." and len(answer) > 0:
            return answer
    return None

def extract_answer_from_dataset(text):
    return text.split("####")[1].strip() if "####" in text else None

def extract_single_number(text):
    if text is None:
        return None
    numbers = NUMBER_PATTERN.findall(str(text).replace('$', '').replace(',', ''))
    if numbers:
        try:
            # Take the last number (usually the answer)
            return float(numbers[-1])
        except ValueError:
            return None
    return None

def format_reward(completion: str) -> float:
    """
    Comprehensive analysis of completion quality. Returns a float score.
    """
    quality_score = 0.0
    penalties = 0.0
    
    # 1. Check for garbage/broken generation
    garbage_count = len(GARBAGE_PATTERNS.findall(completion))
    if garbage_count > 0:
        penalties += garbage_count * 2.0
    
    # 2. Check for repeated instructions
    if "Respond in the following format" in completion:
        repeat_count = completion.count("Respond in the following format")
        penalties += repeat_count * 3.0
    
    # 3. Extract reasoning and answer
    reasoning_match = REASONING_PATTERN.search(completion)
    answer_match = ANSWER_PATTERN.search(completion)
    
    if not reasoning_match or not answer_match:
        penalties += 5.0  # Heavy penalty for missing structure
        return -penalties
    
    reasoning = reasoning_match.group(1).strip()
    answer = answer_match.group(1).strip()
    
    # 4. Check for placeholder content
    if reasoning in ["", "...", "....", "."] or answer in ["", "...", "....", "."]:
        penalties += 10.0  # Very heavy penalty
        return -penalties
    
    # 5. Quality checks on reasoning
    reasoning_length = len(reasoning)
    if reasoning_length < 30:
        penalties += 2.0
    elif reasoning_length > 30 and reasoning_length < 100:
        quality_score += 0.5
    elif reasoning_length >= 100 and reasoning_length < 500:
        quality_score += 1.0  # Good length
    elif reasoning_length >= 500:
        quality_score += 0.5  # Might be too verbose
    
    # 6. Check for mathematical content in reasoning
    math_ops = len(MATH_OPERATIONS.findall(reasoning.lower()))
    numbers_in_reasoning = len(NUMBER_PATTERN.findall(reasoning))
    
    if numbers_in_reasoning == 0:
        penalties += 3.0  # No numbers in math problem reasoning!
    else:
        quality_score += min(numbers_in_reasoning * 0.1, 1.0)
    
    if math_ops < 2:
        penalties += 1.0  # Too few math operations
    else:
        quality_score += min(math_ops * 0.1, 1.0)
    
    # 7. Check for step-by-step reasoning
    step_words = len(STEP_INDICATORS.findall(reasoning.lower()))
    if step_words > 0:
        quality_score += min(step_words * 0.2, 1.0)
    
    # 8. Check answer quality
    if len(answer) == 0:
        penalties += 5.0
    elif not NUMBER_PATTERN.search(answer):
        penalties += 2.0  # Math problem should have number in answer
    
    # 9. Diversity bonus (avoid mode collapse)
    unique_chars = len(set(completion))
    if unique_chars < 20:
        penalties += 2.0  # Too repetitive
    
    return quality_score - penalties

def correctness_reward(completion: str, answer: Optional[str] = None) -> float:
    """Enhanced correctness checking with quality penalties."""
    if answer is None:
        return -2.0
    
    model_answer = extract_answer_from_model_output(completion)
    
    if model_answer is None:
        return -3.0  # Heavy penalty for no extractable answer
    
    # Exact match - best case
    if model_answer.strip() == answer.strip():
        return 5.0  # Increased reward for exact match
    
    # Numeric comparison
    model_num = extract_single_number(model_answer)
    answer_num = extract_single_number(answer)
    
    if model_num is not None and answer_num is not None:
        diff = abs(model_num - answer_num)
        if diff < 0.01:
            return 4.0  # Very close
        elif diff < 1.0:
            return 1.0  # Somewhat close
        elif diff < 10.0:
            return 0.0  # In ballpark
        else:
            return -1.0  # Wrong
    
    return -1.5  # Couldn't parse numbers

def compute_math_reward_effective(
    completion: str, 
    answer: Optional[str] = None,
) -> float:
    """
    Highly effective reward that prevents gaming.
    """
    # Quality analysis
    quality_score = format_reward(completion)
    
    # Correctness check
    correct_score = correctness_reward(completion, answer)
    
    # Total reward
    total = quality_score + correct_score
    
    # Extreme penalty for completely broken outputs
    if quality_score < -5:
        total = -10.0  # Maximum penalty
    
    # Bonus for high quality AND correct
    if quality_score > 2 and correct_score > 3:
        total += 2.0  # Synergy bonus

    return total

def compute_math_rewards_batch(
    completions: List[str],
    answers: Optional[List[str]] = None,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Batch processing with comprehensive rewards.
    """
    import random
    
    if answers is None:
        answers = [None] * len(completions)
    
    # Process answers if needed
    processed_answers = []
    for a in answers:
        if a and "####" in a:
            processed_answers.append(extract_answer_from_dataset(a))
        else:
            processed_answers.append(a)
    
    rewards = []
    quality_scores = []
    correct_scores = []
    
    for i, (comp, ans) in enumerate(zip(completions, processed_answers)):
        # Get detailed reward
        rewards.append(compute_math_reward_effective(comp, ans))
    
    # Add small exploration noise
    rewards = np.array(rewards, dtype=np.float32)
    noise = np.random.normal(0, 0.01, size=rewards.shape)
    rewards = rewards + noise
    
    # Print batch statistics
    # print(f"\nBatch Reward Stats:")
    # print(f"  Mean: {rewards.mean():.3f} ± {rewards.std():.3f}")
    # print(f"  Min: {rewards.min():.3f}, Max: {rewards.max():.3f}")
    # print(f"  Negative rewards: {(rewards < 0).sum()}/{len(rewards)}")
    # print(f"  High rewards (>3): {(rewards > 3).sum()}/{len(rewards)}")
    
    # Check for mode collapse
    unique_completions = len(set(completions))
    if unique_completions < len(completions) * 0.5:
        # print(f"  WARNING: Possible mode collapse! Only {unique_completions}/{len(completions)} unique completions")
        # Apply diversity penalty
        rewards = rewards - 2.0
    
    return torch.tensor(rewards, dtype=torch.float32, device=device)

# Optional: Shaped reward version for curriculum learning
def compute_math_reward_curriculum(
    completion: str,
    answer: Optional[str] = None,
    epoch: int = 0,
    max_epochs: int = 100
) -> float:
    """
    Curriculum learning version that gradually increases difficulty.
    """
    base_reward = compute_math_reward_effective(completion, answer)
    
    # Early epochs: be more forgiving
    # Late epochs: be more strict
    difficulty_scale = epoch / max_epochs
    
    if epoch < 10:
        # Early: reward any structure
        if "<reasoning>" in completion and "<answer>" in completion:
            base_reward += 1.0
    elif epoch < 30:
        # Mid: require some math content
        if NUMBER_PATTERN.search(completion):
            base_reward += 0.5
    else:
        # Late: full strictness, penalize wrong answers more
        if base_reward < 0:
            base_reward *= (1 + difficulty_scale)
    
    return base_reward