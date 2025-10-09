"""
Generate high-quality Chain-of-Thought dataset from GSM8K.

This script takes the first 500 GSM8K examples and creates a cleaner,
more educational CoT dataset by:
1. Removing <<>> calculation annotations
2. Improving formatting and readability
3. Making reasoning more step-by-step
"""

import json
import re
from pathlib import Path
from datasets import load_dataset


def clean_reasoning(raw_reasoning: str) -> str:
    """
    Clean up GSM8K reasoning by removing annotations and improving readability.

    GSM8K format:
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May."

    Cleaned format:
        "Natalia sold 48/2 = 24 clips in May."
    """
    # Remove <<calculation=result>> annotations
    cleaned = re.sub(r'<<[^>]+>>', '', raw_reasoning)

    # Clean up any double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Split into sentences for better formatting
    sentences = [s.strip() for s in cleaned.split('.') if s.strip()]

    # Rejoin with proper spacing
    cleaned = '\n'.join(sentences) + '.'

    return cleaned.strip()


def format_cot_example(question: str, answer_text: str) -> dict:
    """
    Format a GSM8K example into CoT format with <reasoning> and <answer> tags.

    Args:
        question: The math problem
        answer_text: Raw answer from GSM8K (format: "reasoning\\n####\\nfinal_answer")

    Returns:
        Dict with 'question', 'reasoning', and 'answer' fields
    """
    # Split by #### delimiter
    if "####" in answer_text:
        reasoning_part, final_answer = answer_text.split("####")
        reasoning_part = reasoning_part.strip()
        final_answer = final_answer.strip()
    else:
        # Fallback if no #### found
        reasoning_part = ""
        final_answer = answer_text.strip()

    # Clean up the reasoning
    cleaned_reasoning = clean_reasoning(reasoning_part) if reasoning_part else "Let me solve this step by step."

    # Format as CoT
    formatted_completion = f"""<reasoning>
{cleaned_reasoning}
</reasoning>
<answer>
{final_answer}
</answer>"""

    return {
        "question": question.strip(),
        "reasoning": cleaned_reasoning,
        "answer": final_answer,
        "formatted_completion": formatted_completion
    }


def generate_cot_dataset(num_examples: int = 500, output_path: str = "cot_dataset_500.json"):
    """
    Generate a CoT dataset from the first N GSM8K examples.

    Args:
        num_examples: Number of examples to generate
        output_path: Path to save the dataset
    """
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")

    print(f"Generating CoT dataset for {num_examples} examples...")
    cot_examples = []

    for i, example in enumerate(dataset):
        if i >= num_examples:
            break

        question = example["question"]
        answer = example["answer"]

        formatted = format_cot_example(question, answer)
        cot_examples.append(formatted)

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_examples} examples...")

    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cot_examples, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(cot_examples)} CoT examples to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Show a sample
    print("\n" + "="*60)
    print("SAMPLE EXAMPLE:")
    print("="*60)
    sample = cot_examples[0]
    print(f"Question: {sample['question'][:100]}...")
    print(f"\n{sample['formatted_completion']}")
    print("="*60)

    return cot_examples


if __name__ == "__main__":
    # Generate 500 examples
    cot_data = generate_cot_dataset(
        num_examples=500,
        output_path="data/cot_dataset_500.json"
    )

    print(f"\n✓ Dataset generation complete!")
    print(f"  Total examples: {len(cot_data)}")
    print(f"  Use this dataset for SFT training")
