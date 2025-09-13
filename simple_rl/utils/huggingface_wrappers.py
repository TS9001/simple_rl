"""
HuggingFace model wrappers and utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


class LanguageModel(nn.Module):
    """
    Language model wrapper for HuggingFace causal language models.
    
    Handles text generation, log probability computation, and tokenization.
    Can be used by any algorithm that needs language generation capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize language model with HuggingFace model.
        
        Args:
            config: Configuration dictionary with model settings
        """
        super().__init__()
        
        model_config = config.get("model", {})
        self.model_name = model_config.get("hf_model_name", "gpt2")
        self.max_length = model_config.get("max_length", 512)
        
        # Load HuggingFace model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Get model config
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the language model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs.logits
    
    def generate(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate completions for given prompts.
        
        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            attention_mask: Attention mask for prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Tuple of (generated_ids, attention_mask)
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
                **kwargs
            )
            
        generated_ids = outputs.sequences
        
        # Create attention mask for generated sequence
        generated_attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        
        return generated_ids, generated_attention_mask
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities for a sequence.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            target_mask: Mask for target tokens to compute log probs for
            
        Returns:
            Log probabilities [batch_size, seq_len-1]
        """
        # Get logits from model
        logits = self.forward(input_ids, attention_mask=attention_mask)
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        log_probs = torch.gather(
            log_probs_all,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply masks
        combined_mask = None
        
        if attention_mask is not None:
            shift_attention = attention_mask[:, 1:].contiguous()
            combined_mask = shift_attention
        
        if target_mask is not None:
            shift_mask = target_mask[:, 1:].contiguous()
            
            if combined_mask is not None:
                combined_mask = combined_mask * shift_mask
            else:
                combined_mask = shift_mask
        
        if combined_mask is not None:
            log_probs = log_probs * combined_mask
            
        return log_probs
    
    def tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text strings.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            truncation: Whether to truncate
            padding: Whether to pad
            return_tensors: Return type ("pt" for PyTorch tensors)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_length
            
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors
        )
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def get_prompt_length(self, prompt_ids: torch.Tensor) -> int:
        """
        Get the length of prompt in tokens.
        
        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            
        Returns:
            Length of prompt
        """
        return prompt_ids.shape[1]