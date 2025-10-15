"""
Dataset caching utilities for deterministic loading.

This module provides functions to:
1. Save processed datasets to disk
2. Load cached datasets from disk
3. Ensure deterministic dataset ordering
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def _compute_cache_key(dataset_name: str, splits: Dict[str, str], system_prompt: str) -> str:
    """
    Compute a unique cache key for a dataset configuration.

    Args:
        dataset_name: Name of the dataset (e.g., "gsm8k")
        splits: Dictionary of split names and their specifications
        system_prompt: System prompt used for formatting

    Returns:
        Unique cache key (hash)
    """
    # Create a deterministic string representation
    key_data = {
        "dataset_name": dataset_name,
        "splits": splits,
        "system_prompt": system_prompt
    }
    key_str = json.dumps(key_data, sort_keys=True)
    # Use first 16 chars of SHA256 hash
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def save_dataset_cache(
    cache_dir: str,
    dataset_name: str,
    splits: Dict[str, str],
    system_prompt: str,
    data: Dict[str, Any],
    logger=None
) -> Path:
    """
    Save processed dataset to disk.

    Args:
        cache_dir: Base directory for caching
        dataset_name: Name of the dataset
        splits: Dictionary of split names and specifications
        system_prompt: System prompt used
        data: Dictionary containing processed data to save
        logger: Optional logger for info messages

    Returns:
        Path to saved cache file
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    cache_key = _compute_cache_key(dataset_name, splits, system_prompt)
    cache_file = cache_path / f"{dataset_name}_{cache_key}.json"

    # Save metadata + data
    cache_data = {
        "metadata": {
            "dataset_name": dataset_name,
            "splits": splits,
            "system_prompt": system_prompt,
            "cache_key": cache_key
        },
        "data": data
    }

    if logger:
        logger.info(f"\n💾 Saving dataset cache to: {cache_file}")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    # Log file size
    size_mb = cache_file.stat().st_size / (1024 * 1024)
    if logger:
        logger.info(f"   Cache file size: {size_mb:.2f} MB")
        logger.info(f"   ✓ Dataset cached successfully")

    return cache_file


def load_dataset_cache(
    cache_dir: str,
    dataset_name: str,
    splits: Dict[str, str],
    system_prompt: str,
    logger=None
) -> Optional[Dict[str, Any]]:
    """
    Load processed dataset from cache if available.

    Args:
        cache_dir: Base directory for caching
        dataset_name: Name of the dataset
        splits: Dictionary of split names and specifications
        system_prompt: System prompt used
        logger: Optional logger for info messages

    Returns:
        Cached data dictionary if found, None otherwise
    """
    cache_path = Path(cache_dir)

    # Check if cache exists
    cache_key = _compute_cache_key(dataset_name, splits, system_prompt)
    cache_file = cache_path / f"{dataset_name}_{cache_key}.json"

    if not cache_file.exists():
        if logger:
            logger.info(f"\n📂 No cache found for {dataset_name} (key: {cache_key})")
        return None

    if logger:
        logger.info(f"\n📂 Loading dataset from cache: {cache_file}")

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # Verify metadata matches
        metadata = cache_data.get("metadata", {})
        if (metadata.get("dataset_name") == dataset_name and
            metadata.get("splits") == splits and
            metadata.get("system_prompt") == system_prompt):

            size_mb = cache_file.stat().st_size / (1024 * 1024)
            if logger:
                logger.info(f"   Cache file size: {size_mb:.2f} MB")
                logger.info(f"   ✓ Cache loaded successfully")

            return cache_data.get("data")
        else:
            if logger:
                logger.info(f"   ⚠️  Cache metadata mismatch, will reload from source")
            return None

    except Exception as e:
        if logger:
            logger.info(f"   ⚠️  Error loading cache: {e}")
        return None


def save_gsm8k_cache(
    cache_dir: str,
    splits: Dict[str, str],
    system_prompt: str,
    sft_train_data: Dict[str, List[str]],
    sft_val_data: Dict[str, List[str]],
    sft_test_data: Dict[str, List[str]],
    grpo_train_prompts: List[str],
    grpo_train_answers: List[str],
    grpo_val_prompts: List[str],
    grpo_val_answers: List[str],
    grpo_test_prompts: List[str],
    grpo_test_answers: List[str]
) -> Path:
    """
    Save complete GSM8K dataset cache (SFT + GRPO splits).

    Args:
        cache_dir: Cache directory
        splits: Dictionary of split specifications
        system_prompt: System prompt used
        sft_train_data: SFT training data
        sft_val_data: SFT validation data
        sft_test_data: SFT test data
        grpo_train_prompts: GRPO training prompts
        grpo_train_answers: GRPO training answers
        grpo_val_prompts: GRPO validation prompts
        grpo_val_answers: GRPO validation answers
        grpo_test_prompts: GRPO test prompts
        grpo_test_answers: GRPO test answers

    Returns:
        Path to cache file
    """
    data = {
        "sft_train": sft_train_data,
        "sft_val": sft_val_data,
        "sft_test": sft_test_data,
        "grpo_train": {
            "prompts": grpo_train_prompts,
            "answers": grpo_train_answers
        },
        "grpo_val": {
            "prompts": grpo_val_prompts,
            "answers": grpo_val_answers
        },
        "grpo_test": {
            "prompts": grpo_test_prompts,
            "answers": grpo_test_answers
        }
    }

    return save_dataset_cache(
        cache_dir=cache_dir,
        dataset_name="gsm8k_full_pipeline",
        splits=splits,
        system_prompt=system_prompt,
        data=data
    )


def load_gsm8k_cache(
    cache_dir: str,
    splits: Dict[str, str],
    system_prompt: str
) -> Optional[Tuple[
    Dict[str, List[str]],  # sft_train_data
    Dict[str, List[str]],  # sft_val_data
    Dict[str, List[str]],  # sft_test_data
    List[str],  # grpo_train_prompts
    List[str],  # grpo_train_answers
    List[str],  # grpo_val_prompts
    List[str],  # grpo_val_answers
    List[str],  # grpo_test_prompts
    List[str],  # grpo_test_answers
]]:
    """
    Load complete GSM8K dataset cache (SFT + GRPO splits).

    Args:
        cache_dir: Cache directory
        splits: Dictionary of split specifications
        system_prompt: System prompt used

    Returns:
        Tuple of (sft_train, sft_val, sft_test, grpo_train_prompts, grpo_train_answers,
                  grpo_val_prompts, grpo_val_answers, grpo_test_prompts, grpo_test_answers)
        or None if cache not found
    """
    data = load_dataset_cache(
        cache_dir=cache_dir,
        dataset_name="gsm8k_full_pipeline",
        splits=splits,
        system_prompt=system_prompt
    )

    if data is None:
        return None

    return (
        data["sft_train"],
        data["sft_val"],
        data["sft_test"],
        data["grpo_train"]["prompts"],
        data["grpo_train"]["answers"],
        data["grpo_val"]["prompts"],
        data["grpo_val"]["answers"],
        data["grpo_test"]["prompts"],
        data["grpo_test"]["answers"],
    )


def save_cot_cache(
    cache_dir: str,
    system_prompt: str,
    num_examples: int,
    sft_train_data_cot: Dict[str, List[str]]
) -> Path:
    """
    Save CoT dataset cache.

    Args:
        cache_dir: Cache directory
        system_prompt: System prompt used
        num_examples: Number of examples
        sft_train_data_cot: CoT training data

    Returns:
        Path to cache file
    """
    splits = {"num_examples": str(num_examples)}

    return save_dataset_cache(
        cache_dir=cache_dir,
        dataset_name="gsm8k_cot",
        splits=splits,
        system_prompt=system_prompt,
        data=sft_train_data_cot
    )


def load_cot_cache(
    cache_dir: str,
    system_prompt: str,
    num_examples: int
) -> Optional[Dict[str, List[str]]]:
    """
    Load CoT dataset cache.

    Args:
        cache_dir: Cache directory
        system_prompt: System prompt used
        num_examples: Number of examples

    Returns:
        CoT training data or None if not cached
    """
    splits = {"num_examples": str(num_examples)}

    return load_dataset_cache(
        cache_dir=cache_dir,
        dataset_name="gsm8k_cot",
        splits=splits,
        system_prompt=system_prompt
    )
