#!/usr/bin/env python3
"""Script to reorganize GRPO tests into proper structure."""

import re
from pathlib import Path

# Read the extensive test file
extensive_file = Path("tests/test_grpo_extensive.py")
with open(extensive_file, 'r') as f:
    content = f.read()

# Extract test classes and their methods
class_pattern = r'class (Test\w+)[^:]*:.*?(?=^class |\Z)'
classes = re.findall(class_pattern, content, re.DOTALL | re.MULTILINE)

# Parse each class and its tests
test_structure = {}
for match in re.finditer(class_pattern, content, re.DOTALL):
    class_name = match.group(1)
    class_content = match.group(0)
    
    # Find all test methods in this class
    test_methods = re.findall(r'    def (test_\w+)\(self', class_content)
    test_structure[class_name] = test_methods

# Print the structure
print("Test Classes and Methods in test_grpo_extensive.py:")
print("=" * 60)
total_tests = 0
for class_name, methods in test_structure.items():
    print(f"\n{class_name}: {len(methods)} tests")
    for method in methods:
        print(f"  - {method}")
    total_tests += len(methods)

print(f"\nTotal tests in extensive file: {total_tests}")

# Categorize tests for new structure
print("\n" + "=" * 60)
print("Proposed reorganization:")
print("=" * 60)

masking_tests = [
    "test_no_double_masking_in_compute_log_probs",
    "test_masking_with_variable_length_sequences", 
    "test_policy_loss_receives_premasked_logprobs",
    "test_kl_divergence_with_premasked_inputs",
    "test_masking_consistency_across_updates",
    "test_masking_correctness",  # from test_grpo.py
    "test_memory_efficient_masking"
]

edge_case_tests = [
    "test_empty_completion",
    "test_single_token_completion",
    "test_all_padding_sequences",
    "test_mismatched_group_size",
    "test_extreme_reward_values",
    "test_zero_kl_coefficient",
    "test_extreme_kl_coefficient",
    "test_negative_rewards",
    "test_zero_rewards",
    "test_mismatched_batch_group_sizes",
    "test_temperature_effects",
    "test_no_clipping",
    "test_batch_divisibility"  # from test_grpo.py
]

training_tests = [
    "test_full_training_step",
    "test_gradient_clipping", 
    "test_minibatch_processing",
    "test_reference_model_frozen",
    "test_train_method_basic",
    "test_train_method_with_episodes",
    "test_train_with_custom_reward_function",
    "test_train_with_custom_dataset",
    "test_train_with_varying_prompts",
    "test_policy_loss",  # from test_grpo.py
    "test_train_step",  # if exists
    "test_update_epochs"  # if exists
]

performance_tests = [
    "test_large_batch_processing",
    "test_caching_reference_outputs",
    "test_performance_with_long_sequences",
    "test_batch_size_scaling",
    "test_no_gradient_accumulation_in_eval"
]

stability_tests = [
    "test_log_prob_stability",
    "test_kl_divergence_stability",
    "test_reward_normalization_stability",
    "test_gradient_stability",
    "test_loss_computation_stability"
]

basic_tests = [
    "test_initialization",
    "test_relative_rewards",
    "test_kl_divergence",
    "test_trajectory_generation_shapes"
]

rewards_tests = [
    "test_relative_rewards",
    "test_compute_relative_rewards",
    "test_reward_normalization_stability",
    "test_negative_rewards",
    "test_zero_rewards",
    "test_extreme_reward_values"
]

print("\ntests/algorithms/grpo/test_basic.py:")
for test in basic_tests:
    print(f"  - {test}")

print("\ntests/algorithms/grpo/test_masking.py:")
for test in masking_tests:
    print(f"  - {test}")

print("\ntests/algorithms/grpo/test_rewards.py:")  
for test in rewards_tests:
    print(f"  - {test}")

print("\ntests/algorithms/grpo/test_training.py:")
for test in training_tests:
    print(f"  - {test}")

print("\ntests/algorithms/grpo/test_edge_cases.py:")
for test in edge_case_tests:
    print(f"  - {test}")

print("\ntests/algorithms/grpo/test_performance.py:")
for test in performance_tests:
    print(f"  - {test}")

print("\ntests/algorithms/grpo/test_stability.py:")
for test in stability_tests:
    print(f"  - {test}")

print("\ntests/algorithms/grpo/test_prompts.py: (keep as is)")
print("  - All 10 prompt customization tests")