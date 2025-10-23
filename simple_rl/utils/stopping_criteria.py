from typing import List
import torch
from transformers import StoppingCriteria


class MultiTokenStoppingCriteria(StoppingCriteria):
    """
    Stopping criterion that checks for complete multi-token sequences.

    This is more robust than checking single token IDs, especially for
    sequences like </answer> that may be tokenized into multiple tokens.
    """

    def __init__(self, stop_sequences: List[str], tokenizer, prompt_length: int):
        """
        Args:
            stop_sequences: List of string sequences to stop on (e.g., ["</answer>"])
            tokenizer: HuggingFace tokenizer
            prompt_length: Length of the prompt (to only check generated tokens)
        """
        self.stop_sequences = [s for s in (stop_sequences or []) if isinstance(s, str) and s]
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Check if any sequence contains a stop sequence in the generated portion."""
        for sequence_ids in input_ids:
            generated_ids = sequence_ids[self.prompt_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            for stop_seq in self.stop_sequences:
                if stop_seq in generated_text:
                    return True

        return False
