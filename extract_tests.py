#!/usr/bin/env python3
"""Extract all test methods from test files."""

import re

files_to_check = [
    "tests/test_grpo.py",
    "tests/test_grpo_extensive.py", 
    "tests/test_grpo_prompts.py"
]

for filename in files_to_check:
    print(f"\n{'='*60}")
    print(f"Tests in {filename}:")
    print('='*60)
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Find all test methods
        test_methods = re.findall(r'def (test_\w+)\(', content)
        
        # Find all test classes
        test_classes = re.findall(r'^class (Test\w+)', content, re.MULTILINE)
        
        print(f"Classes: {', '.join(test_classes)}")
        print(f"\nTest methods ({len(test_methods)}):")
        for i, method in enumerate(test_methods, 1):
            print(f"  {i:2}. {method}")
            
    except FileNotFoundError:
        print(f"File not found: {filename}")