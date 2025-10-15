"""
GSM8K-specific evaluation utilities.

This module provides evaluation functions for the GSM8K mathematical reasoning dataset.
Works with GRPO, SFT, and raw model instances.

Functions:
- load_gsm8k_dataset: Load GSM8K dataset splits
- prepare_gsm8k_prompts: Format GSM8K prompts with system prompt (for GRPO)
- prepare_gsm8k_for_sft: Format GSM8K data for supervised fine-tuning
- evaluate_on_gsm8k: Evaluate model on GSM8K
- demonstrate_model_responses: Show example model responses
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

from simple_rl.rewards import (
    extract_answer_from_model_output,
    extract_single_number,
    compute_format_reward,
    compute_correctness_reward,
)


def load_gsm8k_dataset(
    train_split: str = "train[:2000]",
    val_split: str = "test[200:1200]",
    test_split: str = "test[:200]",
    logger=None
) -> Tuple[Any, Any, Any]:
    """
    Load GSM8K math dataset splits.

    Args:
        train_split: Training split specification
        val_split: Validation split specification
        test_split: Test split specification
        logger: Optional logger for info messages

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if logger:
        logger.info("Loading GSM8K math dataset...")

    dataset_train = load_dataset("gsm8k", "main", split=train_split)
    dataset_val = load_dataset("gsm8k", "main", split=val_split)
    dataset_test = load_dataset("gsm8k", "main", split=test_split)

    if logger:
        logger.info(f"  Training samples: {len(dataset_train)}")
        logger.info(f"  Validation samples: {len(dataset_val)}")
        logger.info(f"  Test samples: {len(dataset_test)}")

    return dataset_train, dataset_val, dataset_test


def prepare_gsm8k_prompts(
    dataset: Any,
    system_prompt: str
) -> Tuple[List[str], List[str]]:
    """
    Prepare GSM8K dataset prompts and answers.

    Args:
        dataset: HuggingFace dataset with 'question' and 'answer' fields
        system_prompt: System prompt to prepend to each question

    Returns:
        Tuple of (prompts, answers) where:
        - prompts: List of formatted prompts with system prompt
        - answers: List of final numerical answers
    """
    math_prompts = []
    math_answers = []

    for item in dataset:
        # GSM8K format: question and answer with step-by-step solution
        question = item['question']
        answer = item['answer']

        # Create prompt with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.strip()}
        ]

        # Format as simple concatenation (not using chat template)
        prompt = "\n".join([msg["content"].strip() for msg in messages])
        math_prompts.append(prompt)

        # Extract the final numerical answer (after ####)
        final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
        math_answers.append(final_answer)

    return math_prompts, math_answers


def prepare_gsm8k_for_sft(
    dataset: Any,
    system_prompt: str
) -> Dict[str, List[str]]:
    """
    Prepare GSM8K dataset for supervised fine-tuning (SFT).

    This function formats GSM8K data with prompts and full completions
    including reasoning steps and final answers in the required format.

    Args:
        dataset: HuggingFace dataset with 'question' and 'answer' fields
        system_prompt: System prompt to prepend to each question

    Returns:
        Dictionary with 'prompts' and 'completions' lists where:
        - prompts: List of formatted prompts with system prompt
        - completions: List of formatted completions with reasoning and answer
    """
    prompts = []
    completions = []

    for item in dataset:
        question = item['question']
        answer = item['answer']

        # Create prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.strip()}
        ]
        prompt = "\n".join([msg["content"].strip() for msg in messages])
        prompts.append(prompt)

        # Extract reasoning and final answer
        if "####" in answer:
            reasoning_part = answer.split("####")[0].strip()
            final_answer = answer.split("####")[1].strip()
        else:
            reasoning_part = ""
            final_answer = answer.strip()

        # Format completion
        completion = f"<reasoning>\n{reasoning_part}\n</reasoning>\n<answer>\n{final_answer}\n</answer>"
        completions.append(completion)

    return {"prompts": prompts, "completions": completions}


def evaluate_on_gsm8k(
    model_or_algo: Any,
    test_prompts: List[str],
    test_answers: List[str],
    num_samples: int,
    max_new_tokens: int = 400,  # Increased to 400 to handle longer CoT completions (avg 288 tokens)
    temperature: float = 1.0,
    top_p: float = 1.0,
    model_name: str = "Model",
    sample: bool = True,
    save_results: bool = False,
    results_file: str = "eval_results.json",
    step: int = 0,
    logger=None,
) -> Dict[str, Any]:
    """
    Evaluate a model on GSM8K test set.

    This function works with:
    - GRPO instances (uses grpo.policy.generate)
    - SFT instances (uses sft.generate)
    - Raw models (uses model.generate directly)

    Args:
        model_or_algo: GRPO, SFT instance, or raw model
        test_prompts: List of test prompts
        test_answers: List of correct answers
        num_samples: Number of samples to evaluate
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        model_name: Name for display purposes
        sample: Whether to sample or use greedy decoding
        save_results: If True, save detailed results to JSON file
        results_file: Base path for results file (will be modified with step/date)
        step: Step/episode number for tracking training progress
        logger: Optional logger for info messages

    Returns:
        Dictionary with evaluation metrics:
        - exact_accuracy: Exact string match percentage
        - numeric_accuracy: Numeric equivalence percentage
        - format_compliance: Percentage with proper format
        - avg_format_score: Average format reward score
        - avg_correctness_score: Average correctness score
        - num_evaluated: Number of examples evaluated
        - num_correct: Total number of correct answers
        - results: List of individual results
    """
    # Extract tokenizer from model/algo object
    if hasattr(model_or_algo, 'policy'):
        # GRPO instance
        tokenizer = model_or_algo.policy.tokenizer
    elif hasattr(model_or_algo, 'tokenizer'):
        # SFT instance or model with tokenizer attribute
        tokenizer = model_or_algo.tokenizer
    else:
        raise ValueError("Could not find tokenizer in model_or_algo. "
                        "Ensure your model has either policy.tokenizer or tokenizer attribute.")

    # Detect model type and setup generation function
    if hasattr(model_or_algo, 'policy'):
        # GRPO instance
        device = model_or_algo.device

        # Build stopping criteria from GRPO config (if available)
        stopping_criteria = None
        if hasattr(model_or_algo, 'stop_sequences') and hasattr(model_or_algo, 'use_multi_token_stopping'):
            if model_or_algo.use_multi_token_stopping:
                from transformers import StoppingCriteriaList
                # Import stopping criteria class from GRPO
                from simple_rl.algorithms.grpo import MultiTokenStoppingCriteria

                stopping_criteria = StoppingCriteriaList([
                    MultiTokenStoppingCriteria(
                        stop_sequences=model_or_algo.stop_sequences,
                        tokenizer=tokenizer,
                        prompt_length=0  # Will be updated per generation
                    )
                ])

        def generate_fn(prompt_text):
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Update stopping criteria prompt length for this specific generation
            if stopping_criteria is not None:
                stopping_criteria[0].prompt_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                generated_ids, _ = model_or_algo.policy.generate(
                    prompt_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=sample,
                    top_p=top_p,
                    stopping_criteria=stopping_criteria,  # ✓ Now includes stopping!
                )

            # Extract completion
            prompt_len = inputs["input_ids"].shape[1]
            completion_ids = generated_ids[:, prompt_len:]
            completion = model_or_algo.policy.decode(completion_ids)[0].strip()

            if not completion:
                full_text = model_or_algo.policy.decode(generated_ids)[0]
                completion = full_text[len(prompt_text):].strip()

            return completion

    elif hasattr(model_or_algo, 'generate') and hasattr(model_or_algo, 'model'):
        # SFT instance
        device = model_or_algo.device

        def generate_fn(prompt_text):
            completions = model_or_algo.generate(
                prompts=[prompt_text],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return completions[0]

    else:
        # Raw model
        device = next(model_or_algo.parameters()).device if hasattr(model_or_algo, 'parameters') else 'cuda'

        def generate_fn(prompt_text):
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model_or_algo.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Extract completion
            prompt_len = inputs["input_ids"].shape[1]
            completion_ids = generated_ids[:, prompt_len:]
            completion = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()

            return completion

    # Sample subset if needed
    if num_samples < len(test_prompts):
        indices = np.random.choice(len(test_prompts), num_samples, replace=False)
        eval_prompts = [test_prompts[i] for i in indices]
        eval_answers = [test_answers[i] for i in indices]
    else:
        eval_prompts = test_prompts[:num_samples]
        eval_answers = test_answers[:num_samples]

    # Metrics tracking
    correct_exact = 0
    correct_numeric = 0
    format_compliant = 0
    total_format_score = 0.0
    total_correctness_score = 0.0

    results = []

    if logger:
        logger.info(f"\nEvaluating {model_name} on {len(eval_prompts)} GSM8K problems...")
        logger.info("=" * 60)

    # Check if we can use batched generation (much faster!)
    if hasattr(model_or_algo, 'generate') and hasattr(model_or_algo, 'model'):
        # SFT instance - use batched generation for speed
        if logger:
            logger.info("Using batched generation for speed...")
        completions = model_or_algo.generate(
            prompts=eval_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=8  # Process 8 prompts at a time
        )
    else:
        # GRPO or raw model - generate one by one (slower)
        if logger:
            logger.info("Generating responses one by one...")
        completions = []
        for prompt in tqdm(eval_prompts, desc="Generating"):
            completion = generate_fn(prompt)
            completions.append(completion)

    # Now evaluate all completions
    if logger:
        logger.info("Evaluating responses...")
    for prompt, correct_answer, completion in tqdm(zip(eval_prompts, eval_answers, completions),
                                                     total=len(eval_prompts), desc="Evaluating"):
        # Extract model's answer
        model_answer = extract_answer_from_model_output(completion)

        # Check correctness
        is_exact = False
        is_numeric = False

        if model_answer:
            # Exact match
            if model_answer == correct_answer:
                correct_exact += 1
                is_exact = True
                is_numeric = True
            else:
                # Numeric equivalence (exact equality, matching original notebook)
                model_num = extract_single_number(model_answer)
                correct_num = extract_single_number(correct_answer)
                if model_num is not None and correct_num is not None and model_num == correct_num:
                    correct_numeric += 1
                    is_numeric = True

        # Check format compliance
        fmt_score = compute_format_reward(completion)
        total_format_score += fmt_score

        # Full format compliance (all 4 tags present)
        if all(tag in completion for tag in ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]):
            format_compliant += 1

        # Calculate correctness reward (no partial credit for evaluation)
        # compute_correctness_reward returns (correctness, model_num, gold_num)
        correct_score, _, _ = compute_correctness_reward(completion, correct_answer, partial_credit=False)
        total_correctness_score += correct_score

        # Store result
        results.append({
            'prompt': prompt,
            'completion': completion,
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'is_exact': is_exact,
            'is_numeric': is_numeric,
            'format_score': fmt_score,
            'correctness_score': correct_score
        })

    # Calculate metrics
    n = len(eval_prompts)
    total_correct = correct_exact + correct_numeric

    metrics = {
        'exact_accuracy': correct_exact / n * 100,
        'numeric_accuracy': correct_numeric / n * 100,
        'format_compliance': format_compliant / n * 100,
        'avg_format_score': total_format_score / n,
        'avg_correctness_score': total_correctness_score / n,
        'num_evaluated': n,
        'num_correct': total_correct,
        'results': results
    }

    # Print summary
    if logger:
        logger.info(f"\n{model_name} Evaluation Results:")
        logger.info(f"  Total Correct: {total_correct}/{n} ({total_correct/n*100:.1f}%)")
        logger.info(f"  Exact Match Accuracy: {metrics['exact_accuracy']:.1f}%")
        logger.info(f"  Numeric Accuracy: {metrics['numeric_accuracy']:.1f}%")
        logger.info(f"  Format Compliance: {metrics['format_compliance']:.1f}%")
        logger.info(f"  Avg Format Score: {metrics['avg_format_score']:.3f}")
        logger.info(f"  Avg Correctness Score: {metrics['avg_correctness_score']:.3f}")

    # Save results to file if requested
    if save_results:
        import json
        from pathlib import Path
        from datetime import datetime

        # Create filename with model type, step, and timestamp
        # Extract model type from results_file (e.g., "results/sft_eval.json" -> "SFT")
        base_path = Path(results_file)
        if 'sft' in str(results_file).lower():
            model_type = 'SFT'
        elif 'grpo' in str(results_file).lower():
            model_type = 'GRPO'
        else:
            model_type = 'MODEL'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_type}_step_{step}_{timestamp}.json"
        results_path = base_path.parent / filename

        output_data = {
            'model_name': model_name,
            'step': step,
            'timestamp': timestamp,
            'metrics': {
                'total_correct': total_correct,
                'num_evaluated': n,
                'accuracy': total_correct / n * 100,
                'exact_accuracy': metrics['exact_accuracy'],
                'numeric_accuracy': metrics['numeric_accuracy'],
                'format_compliance': metrics['format_compliance'],
                'avg_format_score': metrics['avg_format_score'],
                'avg_correctness_score': metrics['avg_correctness_score'],
            },
            'detailed_results': [
                {
                    'prompt': r['prompt'],
                    'completion': r['completion'],
                    'model_answer': r['model_answer'],
                    'correct_answer': r['correct_answer'],
                    'is_correct': r['is_exact'] or r['is_numeric'],
                    'is_exact': r['is_exact'],
                    'is_numeric': r['is_numeric'],
                    'format_score': r['format_score'],
                    'correctness_score': r['correctness_score'],
                }
                for r in results
            ]
        }

        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        if logger:
            logger.info(f"\n✓ Detailed results saved to: {results_path}")

    return metrics


def demonstrate_model_responses(
    model_or_algo: Any,
    test_prompts: List[str],
    test_answers: List[str],
    num_examples: int,
    max_new_tokens: int = 400,  # Increased to 400 to handle longer CoT completions (avg 288 tokens)
    temperature: float = 1.0,
    top_p: float = 1.0,
    title: str = "MODEL RESPONSE EXAMPLES",
    logger=None
) -> List[Dict[str, Any]]:
    """
    Demonstrate actual model responses on a few examples.

    Args:
        model_or_algo: GRPO, SFT instance, or raw model
        test_prompts: List of test prompts
        test_answers: List of correct answers
        num_examples: Number of examples to show
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        title: Title for the demonstration section
        logger: Optional logger for info messages

    Returns:
        List of demonstration results with prompt, answers, and correctness info
    """
    # Extract tokenizer from model/algo object
    if hasattr(model_or_algo, 'policy'):
        # GRPO instance
        tokenizer = model_or_algo.policy.tokenizer
    elif hasattr(model_or_algo, 'tokenizer'):
        # SFT instance or model with tokenizer attribute
        tokenizer = model_or_algo.tokenizer
    else:
        raise ValueError("Could not find tokenizer in model_or_algo. "
                        "Ensure your model has either policy.tokenizer or tokenizer attribute.")

    # Detect model type and setup generation function
    if hasattr(model_or_algo, 'policy'):
        # GRPO instance
        device = model_or_algo.device

        # Build stopping criteria from GRPO config (if available)
        stopping_criteria = None
        if hasattr(model_or_algo, 'stop_sequences') and hasattr(model_or_algo, 'use_multi_token_stopping'):
            if model_or_algo.use_multi_token_stopping:
                from transformers import StoppingCriteriaList
                # Import stopping criteria class from GRPO
                from simple_rl.algorithms.grpo import MultiTokenStoppingCriteria

                stopping_criteria = StoppingCriteriaList([
                    MultiTokenStoppingCriteria(
                        stop_sequences=model_or_algo.stop_sequences,
                        tokenizer=tokenizer,
                        prompt_length=0  # Will be updated per generation
                    )
                ])

        def generate_fn(prompt_text):
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Update stopping criteria prompt length for this specific generation
            if stopping_criteria is not None:
                stopping_criteria[0].prompt_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                generated_ids, _ = model_or_algo.policy.generate(
                    prompt_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    stopping_criteria=stopping_criteria,  # ✓ Now includes stopping!
                )

            # Extract completion
            prompt_len = inputs["input_ids"].shape[1]
            completion_ids = generated_ids[:, prompt_len:]
            completion = model_or_algo.policy.decode(completion_ids)[0].strip()

            if not completion:
                full_text = model_or_algo.policy.decode(generated_ids)[0]
                completion = full_text[len(prompt_text):].strip()

            return completion

    elif hasattr(model_or_algo, 'generate') and hasattr(model_or_algo, 'model'):
        # SFT instance
        device = model_or_algo.device

        def generate_fn(prompt_text):
            completions = model_or_algo.generate(
                prompts=[prompt_text],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return completions[0]

    else:
        # Raw model
        device = next(model_or_algo.parameters()).device if hasattr(model_or_algo, 'parameters') else 'cuda'

        def generate_fn(prompt_text):
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model_or_algo.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Extract completion
            prompt_len = inputs["input_ids"].shape[1]
            completion_ids = generated_ids[:, prompt_len:]
            completion = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()

            return completion

    if logger:
        logger.info("\n" + "=" * 60)
        logger.info(title)
        logger.info("=" * 60)

    # Select random examples
    indices = np.random.choice(len(test_prompts), min(num_examples, len(test_prompts)), replace=False)
    demo_prompts = [test_prompts[i] for i in indices]
    demo_answers = [test_answers[i] for i in indices]

    results = []

    for i, (prompt, correct_answer) in enumerate(zip(demo_prompts, demo_answers), 1):
        if logger:
            logger.info(f"\nExample {i}:")
            logger.info(f"Problem: {prompt[:150]}...")
            logger.info(f"Correct Answer: {correct_answer}")
            logger.info("-" * 40)

        # Generate completion
        completion = generate_fn(prompt)

        # Show more of the response (800 chars) to see if <answer> tags exist
        if logger:
            logger.info(f"Model Response:\n{completion[:800]}{'...' if len(completion) > 800 else ''}")

        # Extract and check answer
        model_answer = extract_answer_from_model_output(completion)
        if logger:
            logger.info(f"\nExtracted Answer: {model_answer if model_answer else 'None (format issue)'}")

        # Check correctness
        is_correct = False
        if model_answer:
            if model_answer == correct_answer:
                if logger:
                    logger.info("Status: ✓ CORRECT (exact match)")
                is_correct = True
            else:
                model_num = extract_single_number(model_answer)
                correct_num = extract_single_number(correct_answer)
                if model_num and correct_num and abs(model_num - correct_num) < 0.01:
                    if logger:
                        logger.info("Status: ✓ CORRECT (numeric)")
                    is_correct = True
                else:
                    if logger:
                        logger.info("Status: ✗ INCORRECT")
        else:
            if logger:
                logger.info("Status: ✗ NO ANSWER")

        # Check format
        has_format = all(tag in completion for tag in ["<reasoning>", "</reasoning>", "<answer>", "</answer>"])
        if logger:
            logger.info(f"Format Compliance: {'Yes' if has_format else 'No'}")

        results.append({
            'prompt': prompt[:100],
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'has_format': has_format
        })

    # Summary
    correct_count = sum(r['is_correct'] for r in results)
    format_count = sum(r['has_format'] for r in results)
    if logger:
        logger.info(f"\n{'-'*60}")
        logger.info(f"Summary: {correct_count}/{len(results)} correct, {format_count}/{len(results)} with proper format")

    return results
