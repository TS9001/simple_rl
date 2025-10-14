#!/usr/bin/env python3
"""
Quick test to verify dataset caching works.

Run this twice to see the difference:
  First run:  Loads from source, saves to cache
  Second run: Loads from cache (fast)
"""

import time
from simple_rl.utils.dataset_cache import save_dataset_cache, load_dataset_cache

# Test configuration
CACHE_DIR = "dataset_cache"
DATASET_NAME = "test_dataset"
SPLITS = {"train": "100", "test": "20"}
SYSTEM_PROMPT = "Test prompt"

print("=" * 60)
print("Dataset Caching Test")
print("=" * 60)

# Try loading from cache
print("\n🔍 Checking for cached data...")
start_time = time.time()

cached_data = load_dataset_cache(
    cache_dir=CACHE_DIR,
    dataset_name=DATASET_NAME,
    splits=SPLITS,
    system_prompt=SYSTEM_PROMPT
)

if cached_data is not None:
    elapsed = time.time() - start_time
    print(f"✓ Loaded from cache in {elapsed:.3f} seconds")
    print(f"  Data keys: {list(cached_data.keys())}")
    print(f"  Sample data: {list(cached_data.values())[0][:50]}...")
else:
    print("⏳ No cache found, creating test data...")

    # Simulate data processing
    test_data = {
        "prompts": [f"Question {i}" for i in range(100)],
        "answers": [f"Answer {i}" for i in range(100)],
        "metadata": {
            "version": "1.0",
            "created_at": "2025-10-12"
        }
    }

    # Save to cache
    cache_file = save_dataset_cache(
        cache_dir=CACHE_DIR,
        dataset_name=DATASET_NAME,
        splits=SPLITS,
        system_prompt=SYSTEM_PROMPT,
        data=test_data
    )

    elapsed = time.time() - start_time
    print(f"✓ Created and cached data in {elapsed:.3f} seconds")
    print(f"  Cache file: {cache_file}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("\nRun this script again to see cache loading in action:")
print("  $ python test_dataset_cache.py")
print("\nTo clear cache:")
print("  $ rm -rf dataset_cache/test_dataset_*")
