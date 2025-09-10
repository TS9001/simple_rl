"""
GRPO (Group Relative Policy Optimization) implementation.

GRPO normalizes rewards within groups/batches to reduce variance
in policy gradient estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import copy

from simple_rl.algorithms.base import BaseAlgorithm
from simple_rl.models.policy_model import PolicyModel


class GRPO(BaseAlgorithm):
    """
    Group Relative Policy Optimization algorithm.
    
    Key features:
    - Generates multiple completions per prompt
    - Normalizes rewards within groups
    - Uses KL divergence penalty for stability
    """
    
    def __init__(
        self,
        model: Optional[nn.Module],
        config: Dict[str, Any],
        reference_model: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        use_wandb: bool = True
    ):
        """
        Initialize GRPO algorithm.
        
        Args:
            model: Policy model for text generation
            config: Configuration dictionary
            reference_model: Reference model for KL computation (if None, uses initial policy)
            reward_model: Model to compute rewards (if None, uses heuristic)
            use_wandb: Whether to use Weights & Biases logging
        """
        # Initialize model if not provided
        if model is None:
            model = PolicyModel(config)
        
        super().__init__(model, config, use_wandb)
        
        # Ensure model is PolicyModel for text generation
        if not isinstance(self.model, PolicyModel):
            # Wrap the model if it's not already a PolicyModel
            self.model = PolicyModel(config)
        
        # GRPO-specific parameters
        algo_config = config.get("algorithm", {})
        self.group_size = algo_config.get("group_size", 8)
        self.kl_coef = algo_config.get("kl_coef", 0.1)
        self.clip_range = algo_config.get("clip_range", None)  # Optional PPO-style clipping
        self.normalize_rewards = algo_config.get("normalize_rewards", True)
        
        # Reference model for KL divergence
        if reference_model is None:
            # Use a frozen copy of the initial model
            self.reference_model = self._create_reference_model()
        else:
            self.reference_model = reference_model
        
        # Ensure reference model doesn't update
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()
        
        # Reward model
        self.reward_model = reward_model
        
        # Training config
        training_config = config.get("training", {})
        self.batch_size = training_config.get("batch_size", 32)
        self.max_new_tokens = training_config.get("max_new_tokens", 128)
        self.temperature = training_config.get("temperature", 1.0)
        self.top_k = training_config.get("top_k", None)
        self.top_p = training_config.get("top_p", None)
        
    def _create_reference_model(self) -> nn.Module:
        """Create a frozen copy of the policy model to use as reference."""
        reference_model = copy.deepcopy(self.model)
        reference_model.to(self.device)
        for param in reference_model.parameters():
            param.requires_grad = False
        return reference_model
    
    def compute_relative_rewards(
        self,
        rewards: torch.Tensor,
        group_size: int
    ) -> torch.Tensor:
        """
        Normalize rewards within groups to compute relative rewards.
        
        Args:
            rewards: Tensor of shape (batch_size,) containing raw rewards
            group_size: Number of samples per group
            
        Returns:
            Normalized rewards with zero mean and unit variance per group
        """
        batch_size = rewards.shape[0]
        num_groups = batch_size // group_size
        
        if batch_size % group_size != 0:
            raise ValueError(
                f"Batch size {batch_size} must be divisible by group size {group_size}"
            )
        
        # Reshape to separate groups
        grouped_rewards = rewards.view(num_groups, group_size)
        
        # Normalize within each group
        if self.normalize_rewards:
            # Compute mean and std per group
            group_mean = grouped_rewards.mean(dim=1, keepdim=True)
            group_std = grouped_rewards.std(dim=1, keepdim=True)
            
            # Add small epsilon to prevent division by zero
            eps = 1e-8
            normalized = (grouped_rewards - group_mean) / (group_std + eps)
        else:
            # Just center without scaling
            group_mean = grouped_rewards.mean(dim=1, keepdim=True)
            normalized = grouped_rewards - group_mean
        
        # Flatten back to original shape
        return normalized.view(batch_size)
    
    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards for generated completions.
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            
        Returns:
            Tensor of rewards
        """
        if self.reward_model is not None:
            # Use reward model
            # This would need proper implementation based on reward model type
            raise NotImplementedError("Reward model integration not yet implemented")
        else:
            # Placeholder: Use completion length as a simple heuristic
            # In practice, this would be replaced with actual reward computation
            rewards = []
            for completion in completions:
                # Simple heuristic: reward based on length within reasonable bounds
                length = len(completion.split())
                if length < 5:
                    reward = -1.0  # Too short
                elif length > 50:
                    reward = -0.5  # Too long
                else:
                    reward = 1.0  # Good length
                rewards.append(reward)
            
            return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO policy gradient loss with KL penalty.
        
        Args:
            log_probs: Log probabilities from current policy [batch_size, seq_len]
            ref_log_probs: Log probabilities from reference policy [batch_size, seq_len]
            rewards: Normalized rewards [batch_size]
            action_mask: Mask for valid actions [batch_size, seq_len]
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        if action_mask is not None:
            # Mask out padding tokens
            log_probs = log_probs * action_mask
            ref_log_probs = ref_log_probs * action_mask
            
            # Sum over sequence length, mean over batch
            log_probs_sum = log_probs.sum(dim=1)
            ref_log_probs_sum = ref_log_probs.sum(dim=1)
        else:
            log_probs_sum = log_probs.sum(dim=1)
            ref_log_probs_sum = ref_log_probs.sum(dim=1)
        
        # Policy gradient loss: -log_prob * reward
        pg_loss = -(log_probs_sum * rewards).mean()
        
        # KL divergence penalty
        kl_div = self.compute_kl_divergence(log_probs_sum, ref_log_probs_sum)
        
        # Optional PPO-style clipping
        if self.clip_range is not None:
            ratio = torch.exp(log_probs_sum - ref_log_probs_sum.detach())
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            clipped_loss = -torch.min(
                ratio * rewards,
                clipped_ratio * rewards
            ).mean()
            pg_loss = clipped_loss
        
        # Total loss
        total_loss = pg_loss + self.kl_coef * kl_div
        
        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_div": kl_div.item(),
            "total_loss": total_loss.item(),
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item()
        }
        
        return total_loss, metrics
    
    def generate_trajectories(
        self,
        prompts: List[str],
        num_samples_per_prompt: int
    ) -> Dict[str, Any]:
        """
        Generate multiple trajectories for each prompt.
        
        Args:
            prompts: List of prompt strings
            num_samples_per_prompt: Number of completions per prompt
            
        Returns:
            Dictionary containing trajectories and metadata
        """
        all_generated_ids = []
        all_attention_masks = []
        all_prompt_lengths = []
        all_generated_texts = []
        
        self.model.eval()
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                prompt_encoding = self.model.tokenize(
                    [prompt],
                    truncation=True,
                    padding=False,
                    return_tensors="pt"
                )
                prompt_ids = prompt_encoding["input_ids"].to(self.device)
                prompt_attention_mask = prompt_encoding["attention_mask"].to(self.device)
                prompt_length = prompt_ids.shape[1]
                
                # Generate multiple completions for this prompt
                for _ in range(num_samples_per_prompt):
                    # Generate completion
                    generated_ids, attention_mask = self.model.generate(
                        prompt_ids=prompt_ids,
                        attention_mask=prompt_attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        top_k=self.top_k,
                        top_p=self.top_p
                    )
                    
                    # Decode generated text
                    generated_text = self.model.decode(generated_ids)[0]
                    
                    all_generated_ids.append(generated_ids)
                    all_attention_masks.append(attention_mask)
                    all_prompt_lengths.append(prompt_length)
                    all_generated_texts.append(generated_text)
        
        # Pad all sequences to same length before stacking
        max_length = max(ids.shape[1] for ids in all_generated_ids)
        pad_token_id = self.model.tokenizer.pad_token_id
        
        padded_ids = []
        padded_masks = []
        
        for ids, mask in zip(all_generated_ids, all_attention_masks):
            current_length = ids.shape[1]
            if current_length < max_length:
                # Pad on the right
                padding_length = max_length - current_length
                ids = F.pad(ids, (0, padding_length), value=pad_token_id)
                mask = F.pad(mask, (0, padding_length), value=0)
            padded_ids.append(ids)
            padded_masks.append(mask)
        
        # Stack all tensors
        generated_ids = torch.cat(padded_ids, dim=0)
        attention_masks = torch.cat(padded_masks, dim=0)
        
        return {
            "generated_ids": generated_ids,
            "attention_masks": attention_masks,
            "prompt_lengths": all_prompt_lengths,
            "texts": all_generated_texts,
            "prompts": prompts
        }
    
    def compute_log_probs_for_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: List[int],
        model: nn.Module
    ) -> torch.Tensor:
        """
        Compute log probabilities for generated sequences.
        
        Args:
            input_ids: Generated token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            prompt_lengths: List of prompt lengths for each sequence
            model: Model to use for computation
            
        Returns:
            Log probabilities for completion tokens [batch_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create mask for completion tokens only
        # Important: The mask should mark which tokens are part of the completion
        # We want to compute loss only on tokens AFTER the prompt
        completion_masks = []
        for i, prompt_len in enumerate(prompt_lengths):
            mask = torch.zeros(seq_len, dtype=torch.float32)
            # For next-token prediction, position i predicts token i+1
            # So we want to include positions from prompt_len-1 onwards in the mask
            # This way, position prompt_len-1 predicts the first completion token
            if prompt_len < seq_len:
                mask[prompt_len:] = 1.0  # Mark positions that predict completion tokens
            completion_masks.append(mask)
        
        completion_mask = torch.stack(completion_masks).to(self.device)
        
        # Compute log probabilities using the model
        log_probs = model.compute_log_probs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mask=completion_mask
        )
        
        # Sum log probs over sequence length (only for completion tokens)
        # log_probs shape is [batch_size, seq_len-1] after compute_log_probs
        # because it's for next-token prediction
        log_probs_sum = log_probs.sum(dim=1)
        
        return log_probs_sum
    
    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single GRPO training step.
        
        Args:
            batch_data: Dictionary containing batch data
            
        Returns:
            Dictionary of training metrics
        """
        prompts = batch_data["prompts"]
        
        # Generate trajectories
        trajectories = self.generate_trajectories(
            prompts, 
            num_samples_per_prompt=self.group_size
        )
        
        # Compute rewards
        rewards = self.compute_rewards(prompts, trajectories["texts"])
        
        # Normalize rewards within groups
        normalized_rewards = self.compute_relative_rewards(rewards, self.group_size)
        
        # Compute log probabilities with current policy (with gradients)
        self.model.train()
        log_probs = self.compute_log_probs_for_sequences(
            trajectories["generated_ids"],
            trajectories["attention_masks"],
            trajectories["prompt_lengths"],
            model=self.model
        )
        
        # Compute log probabilities with reference policy (no gradients)
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs_for_sequences(
                trajectories["generated_ids"],
                trajectories["attention_masks"],
                trajectories["prompt_lengths"],
                model=self.reference_model
            )
        
        # Reshape for policy loss (which expects [batch_size] shaped tensors)
        # log_probs and ref_log_probs are already summed over sequence
        
        # Compute loss
        loss, metrics = self.compute_policy_loss(
            log_probs.unsqueeze(1),  # Add seq dimension for compatibility
            ref_log_probs.unsqueeze(1),
            normalized_rewards
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.get("training", {}).get("gradient_clip"):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["gradient_clip"]
            )
        
        self.optimizer.step()
        
        self.total_steps += 1
        
        return metrics
    
    def train(self, num_episodes: int) -> Dict[str, float]:
        """
        Train GRPO for specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            
        Returns:
            Dictionary of final metrics
        """
        from simple_rl.utils.data import DatasetLoader
        
        # Load dataset if configured
        data_config = self.config.get("data", {})
        if data_config.get("dataset_name") or data_config.get("dataset_path"):
            loader = DatasetLoader(self.config)
            dataset = loader.load_dataset()
            # For now, just use a simple prompt extraction
            # This will be improved when we integrate properly
            prompts = ["Sample prompt " + str(i) for i in range(100)]
        else:
            # Use dummy prompts for testing
            prompts = [
                "The capital of France is",
                "Machine learning is",
                "The best way to",
                "In the future, we will",
                "Science has shown that"
            ] * 20  # Repeat to have more data
        
        final_metrics = {}
        log_interval = self.config.get("logging", {}).get("log_interval", 10)
        
        for episode in range(num_episodes):
            self.current_episode = episode
            
            # Sample batch of prompts
            batch_size = self.batch_size // self.group_size
            batch_indices = np.random.choice(len(prompts), batch_size, replace=True)
            batch_prompts = [prompts[i] for i in batch_indices]
            
            # Training step
            batch_data = {"prompts": batch_prompts}
            metrics = self.train_step(batch_data)
            
            # Update final metrics
            final_metrics = metrics
            
            # Logging
            if episode % log_interval == 0:
                print(f"Episode {episode}: Loss={metrics['total_loss']:.4f}, "
                      f"KL={metrics['kl_div']:.4f}, "
                      f"Mean Reward={metrics['mean_reward']:.4f}")
                self.log_metrics(metrics, step=episode)
            
            # Save checkpoint
            save_interval = self.config.get("logging", {}).get("save_interval", 100)
            if episode > 0 and episode % save_interval == 0:
                checkpoint_path = f"checkpoints/grpo_episode_{episode}.pt"
                self.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        return final_metrics
    
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate the policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Placeholder implementation
        # Will be implemented in later phases
        return {"eval_placeholder": 0.0}
    
    def generate_completion(
        self,
        prompt: str,
        max_length: Optional[int] = None
    ) -> Tuple[str, torch.Tensor]:
        """
        Generate a completion for a given prompt.
        
        Args:
            prompt: Input prompt string
            max_length: Maximum length of generation
            
        Returns:
            Tuple of (completion_text, log_probabilities)
        """
        # Placeholder - will be implemented when we integrate HF models
        # For now, return dummy values
        completion = "placeholder completion"
        log_probs = torch.zeros(1, device=self.device)
        return completion, log_probs
    
    def compute_kl_divergence(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference distributions.
        
        Args:
            log_probs: Log probabilities from current policy
            ref_log_probs: Log probabilities from reference policy
            
        Returns:
            KL divergence value
        """
        # KL(P||Q) = sum(P * (log(P) - log(Q)))
        # Since we have log probs: exp(log_p) * (log_p - log_q)
        probs = torch.exp(log_probs)
        kl = probs * (log_probs - ref_log_probs)
        return kl.mean()