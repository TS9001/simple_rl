#!/usr/bin/env python3
"""Test that generation stops at </answer> tag"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    trust_remote_code=True
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.eval()

# Test prompt
prompt = """Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
What is 2 + 2?"""

print("\n" + "="*60)
print("TEST 1: Without stop_strings (should generate past </answer>)")
print("="*60)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"\nGenerated text:\n{response}")
print(f"\nContains </answer>: {('</answer>' in response)}")
if '</answer>' in response:
    after_answer = response.split('</answer>')[1]
    print(f"Text after </answer>: '{after_answer[:100]}...'")
    print(f"Length after </answer>: {len(after_answer)} characters")

print("\n" + "="*60)
print("TEST 2: With stop_strings=['</answer>'] (should stop AT </answer>)")
print("="*60)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        stop_strings=["</answer>"],
        tokenizer=tokenizer,
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"\nGenerated text:\n{response}")
print(f"\nContains </answer>: {('</answer>' in response)}")
if '</answer>' in response:
    after_answer = response.split('</answer>')[1]
    print(f"Text after </answer>: '{after_answer}'")
    print(f"Length after </answer>: {len(after_answer)} characters")
else:
    print("✓ CORRECT: Generation stopped before </answer>")

print("\n" + "="*60)
print("TEST 3: Check if </answer> is even generated")
print("="*60)
print(f"Response ends with: '...{response[-50:]}'")
