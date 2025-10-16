#!/usr/bin/env python3
"""
Test script to verify that INPUTS are correctly stored alongside LOGPROBS.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_debug_files():
    """Check if the debug files are created with INPUTS."""

    print("=" * 60)
    print("Testing INPUTS alongside LOGPROBS implementation")
    print("=" * 60)
    print()

    # Check if LOGPROBS_FULL_DEBUG.txt exists
    logprobs_file = "LOGPROBS_FULL_DEBUG.txt"
    inputs_file = "INPUTS_DEBUG.txt"

    files_to_check = [logprobs_file, inputs_file]

    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file} exists")

            # Show first few lines
            with open(file, 'r') as f:
                lines = f.readlines()
                print(f"  File has {len(lines)} lines")

                # Check for key indicators
                if file == logprobs_file:
                    # Check if INPUTS column is present
                    has_inputs = any("Input Token" in line for line in lines)
                    if has_inputs:
                        print(f"  ✓ Contains 'Input Token' column")
                    else:
                        print(f"  ✗ Missing 'Input Token' column")

                elif file == inputs_file:
                    # Check for key sections
                    has_full = any("FULL SEQUENCE:" in line for line in lines)
                    has_prompt = any("PROMPT" in line for line in lines)
                    has_completion = any("COMPLETION" in line for line in lines)
                    has_breakdown = any("TOKEN-BY-TOKEN BREAKDOWN" in line for line in lines)

                    if has_full:
                        print(f"  ✓ Contains FULL SEQUENCE section")
                    if has_prompt:
                        print(f"  ✓ Contains PROMPT section")
                    if has_completion:
                        print(f"  ✓ Contains COMPLETION section")
                    if has_breakdown:
                        print(f"  ✓ Contains TOKEN-BY-TOKEN BREAKDOWN")

        else:
            print(f"✗ {file} does not exist")
            print(f"  Note: This file is created during the first training episode")

    print()
    print("=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print()

    if all(os.path.exists(f) for f in files_to_check):
        print("✓ All debug files exist")
        print("✓ INPUTS implementation appears to be working")
        print()
        print("To fully test the implementation, run the training script:")
        print("  python scripts/full_pipeline_sft_grpo_training.py")
        print()
        print("The debug files will be created in the first episode and will contain:")
        print("  - LOGPROBS_FULL_DEBUG.txt: Log probs with input tokens for each position")
        print("  - INPUTS_DEBUG.txt: Detailed view of input sequences")
    else:
        print("⚠ Debug files not found. This is normal if training hasn't started yet.")
        print()
        print("To test the implementation, run:")
        print("  python scripts/full_pipeline_sft_grpo_training.py")
        print()
        print("The files will be created automatically during the first training episode.")

    print("=" * 60)

if __name__ == "__main__":
    check_debug_files()