"""
GRPO (Group Relative Policy Optimization) implementation.

GRPO normalizes rewards within groups/batches to reduce variance
in policy gradient estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
import copy
from pathlib import Path
import wandb

from simple_rl.algorithms.base import BaseAlgorithm
from simple_rl.utils.huggingface_wrappers import LanguageModel


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
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        reward_fn: Optional[Callable[[str, str], float]] = None,
        use_wandb: bool = False
    ):
        """
        Initialize GRPO algorithm.
        
        Args:
            model: Language model for text generation (if None, creates from config)
            config: Configuration dictionary
            reward_fn: Function to compute rewards (prompt, completion) -> float
            use_wandb: Whether to use Weights & Biases logging
        """
        # Store config and setup device
        self.config = config or {}
        self.use_wandb = use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize or create model
        if model is None:
            if not config:
                raise ValueError("Either model or config must be provided")
            self.policy = LanguageModel(config)
        elif isinstance(model, LanguageModel):
            self.policy = model
        else:
            # Assume it's a HuggingFace model that needs wrapping
            if not config:
                config = {"model": {"hf_model_name": "gpt2"}}
            self.policy = LanguageModel(config)
        
        # Move policy to device
        self.policy = self.policy.to(self.device)
        
        # Create reference model (frozen copy for KL divergence)
        self.ref_policy = copy.deepcopy(self.policy)
        self.ref_policy.to(self.device)
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        self.ref_policy.eval()
        
        # GRPO-specific parameters
        algo_config = self.config.get("algorithm", {})
        self.group_size = algo_config.get("group_size", 4)
        self.kl_coef = algo_config.get("kl_coef", 0.05)
        self.normalize_rewards = algo_config.get("normalize_rewards", True)
        
        # Training parameters
        training_config = self.config.get("training", {})
        self.learning_rate = training_config.get("learning_rate", 1e-5)
        self.batch_size = training_config.get("batch_size", 8)
        self.max_new_tokens = training_config.get("max_new_tokens", 128)
        self.temperature = training_config.get("temperature", 0.9)
        
        # Generation prompt configuration
        generation_config = self.config.get("generation", {})
        self.generation_prompt_template = generation_config.get("prompt_template", None)
        self.system_prompt = generation_config.get("system_prompt", None)
        self.response_prefix = generation_config.get("response_prefix", None)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate
        )
        
        # Reward function
        self.reward_fn = reward_fn or self._default_reward_fn
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=self.config.get("project_name", "grpo"),
                config=self.config,
                name=self.config.get("run_name", None)
            )
        
        # Training statistics
        self.total_steps = 0
        self.episode = 0
    
    def _default_reward_fn(self, prompt: str, completion: str) -> float:
        """Default reward function based on completion length."""
        # Simple heuristic: longer completions get higher rewards
        # This should be replaced with actual reward logic
        return min(len(completion.split()) / 50.0, 1.0)
    
    def format_prompt(self, prompt: str, use_formatting: bool = True) -> str:
        """
        Format a prompt with system prompt and template if configured.
        
        Args:
            prompt: Raw prompt text
            use_formatting: Whether to apply formatting
            
        Returns:
            Formatted prompt string
        """
        if not use_formatting:
            return prompt
            
        formatted = prompt
        if self.generation_prompt_template:
            formatted = self.generation_prompt_template.replace("{prompt}", formatted)
        if self.system_prompt:
            formatted = f"{self.system_prompt}\n\n{formatted}"
        if self.response_prefix:
            formatted = f"{formatted}{self.response_prefix}"
        return formatted
    
    def set_generation_prompt(
        self,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        response_prefix: Optional[str] = None
    ):
        """
        Update generation prompt configuration.
        
        Args:
            system_prompt: System prompt to prepend
            prompt_template: Template with {prompt} placeholder
            response_prefix: Prefix to append after prompt
        """
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if prompt_template is not None:
            self.generation_prompt_template = prompt_template
        if response_prefix is not None:
            self.response_prefix = response_prefix
    
    def generate_trajectories(
        self,
        prompts: List[str],
        use_formatting: bool = True
    ) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate trajectories for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            use_formatting: Whether to apply prompt formatting
            
        Returns:
            Tuple of (prompts, completions, rewards, log_probs, ref_log_probs)
        """
        all_prompts = []
        all_completions = []
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []
        
        # Process each prompt
        for prompt in prompts:
            # Format prompt if configured
            formatted_prompt = self.format_prompt(prompt, use_formatting)
            
            # Generate multiple completions per prompt
            prompt_batch = [formatted_prompt] * self.group_size
            
            # Tokenize prompts
            tokenized = self.policy.tokenize(prompt_batch)
            prompt_ids = tokenized["input_ids"].to(self.device)
            prompt_mask = tokenized["attention_mask"].to(self.device)
            
            # Generate completions
            with torch.no_grad():
                generated_ids, generated_mask = self.policy.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True
                )
            
            # Get prompt length to extract completions
            prompt_length = prompt_ids.shape[1]
            
            # Decode completions
            completion_ids = generated_ids[:, prompt_length:]
            completions = self.policy.decode(completion_ids)
            
            # Create mask for completion tokens
            target_mask = torch.zeros_like(generated_ids)
            target_mask[:, prompt_length:] = 1
            
            # Compute log probabilities for policy
            policy_log_probs = self.policy.compute_log_probs(
                generated_ids,
                attention_mask=generated_mask,
                target_mask=target_mask
            )
            
            # Compute log probabilities for reference model
            with torch.no_grad():
                ref_log_probs = self.ref_policy.compute_log_probs(
                    generated_ids,
                    attention_mask=generated_mask,
                    target_mask=target_mask
                )
            
            # Compute rewards for each completion
            rewards = []
            for completion in completions:
                reward = self.reward_fn(prompt, completion)
                rewards.append(reward)
            
            # Store results
            all_prompts.extend([prompt] * self.group_size)
            all_completions.extend(completions)
            all_rewards.extend(rewards)
            all_log_probs.append(policy_log_probs)
            all_ref_log_probs.append(ref_log_probs)
        
        # Stack tensors
        all_log_probs = torch.cat(all_log_probs, dim=0)
        all_ref_log_probs = torch.cat(all_ref_log_probs, dim=0)
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)
        
        return all_prompts, all_completions, all_rewards, all_log_probs, all_ref_log_probs
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages with group normalization and KL penalty.
        
        Args:
            rewards: Reward values [batch_size]
            log_probs: Policy log probabilities [batch_size, seq_len]
            ref_log_probs: Reference log probabilities [batch_size, seq_len]
            
        Returns:
            Advantages [batch_size]
        """
        batch_size = rewards.shape[0]
        
        # Sum log probs across sequence dimension
        log_probs_sum = log_probs.sum(dim=-1)
        ref_log_probs_sum = ref_log_probs.sum(dim=-1)
        
        # Compute KL divergence
        kl_div = log_probs_sum - ref_log_probs_sum
        
        # Apply KL penalty to rewards
        adjusted_rewards = rewards - self.kl_coef * kl_div
        
        # Normalize rewards within groups if configured
        if self.normalize_rewards and self.group_size > 1:
            # Reshape to groups
            num_groups = batch_size // self.group_size
            grouped_rewards = adjusted_rewards.view(num_groups, self.group_size)
            
            # Normalize within each group
            group_mean = grouped_rewards.mean(dim=1, keepdim=True)
            group_std = grouped_rewards.std(dim=1, keepdim=True)
            normalized_rewards = (grouped_rewards - group_mean) / (group_std + 1e-8)
            
            # Flatten back
            advantages = normalized_rewards.view(-1)
        else:
            # Global normalization
            advantages = (adjusted_rewards - adjusted_rewards.mean()) / (adjusted_rewards.std() + 1e-8)
        
        return advantages
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss.
        
        Args:
            log_probs: Policy log probabilities [batch_size, seq_len]
            advantages: Advantages [batch_size]
            ref_log_probs: Reference log probabilities [batch_size, seq_len]
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Sum log probs across sequence
        log_probs_sum = log_probs.sum(dim=-1)
        ref_log_probs_sum = ref_log_probs.sum(dim=-1)
        
        # Policy gradient loss
        pg_loss = -(log_probs_sum * advantages).mean()
        
        # KL divergence for monitoring
        kl_div = (log_probs_sum - ref_log_probs_sum).mean()
        
        # Total loss
        loss = pg_loss
        
        # Metrics for logging
        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_divergence": kl_div.item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
        }
        
        return loss, metrics
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary with 'prompts' key containing list of prompts
            
        Returns:
            Dictionary of training metrics
        """
        prompts = batch["prompts"]
        
        # Generate trajectories
        _, completions, rewards, log_probs, ref_log_probs = self.generate_trajectories(prompts)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, log_probs, ref_log_probs)
        
        # Compute loss
        loss, metrics = self.compute_loss(log_probs, advantages, ref_log_probs)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update statistics
        self.total_steps += 1
        
        # Add rewards to metrics
        metrics["reward_mean"] = rewards.mean().item()
        metrics["reward_std"] = rewards.std().item()
        metrics["total_loss"] = loss.item()
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(metrics, step=self.total_steps)
        
        return metrics
    
    def train(self, num_episodes: int) -> Dict[str, float]:
        """
        Train for a specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            
        Returns:
            Dictionary of final metrics
        """
        print(f"Starting GRPO training for {num_episodes} episodes...")
        
        final_metrics = {}
        
        for episode in range(num_episodes):
            self.episode = episode
            
            # Generate random prompts for demo (replace with actual data)
            prompts = [f"Question {i}: What is {i}+{i}?" for i in range(self.batch_size)]
            
            # Train step
            batch = {"prompts": prompts}
            metrics = self.train_step(batch)
            
            # Print progress
            if episode % max(1, num_episodes // 10) == 0:
                print(f"Episode {episode}/{num_episodes} - "
                      f"Loss: {metrics['total_loss']:.4f}, "
                      f"Reward: {metrics['reward_mean']:.4f}, "
                      f"KL: {metrics['kl_divergence']:.4f}")
            
            final_metrics = metrics
            
            # Save checkpoint periodically
            if episode % max(1, num_episodes // 5) == 0:
                checkpoint_path = f"checkpoints/grpo_episode_{episode}.pt"
                self.save_checkpoint(checkpoint_path)
        
        return final_metrics
    
    def evaluate(self, num_episodes: int = 1) -> Dict[str, float]:
        """
        Evaluate the policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()
        
        total_rewards = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                # Generate test prompts (replace with actual eval data)
                prompts = [f"Test {i}: Calculate {i}*2" for i in range(4)]
                
                # Generate without training
                _, completions, rewards, _, _ = self.generate_trajectories(prompts)
                total_rewards.extend(rewards.cpu().numpy())
        
        self.policy.train()
        
        return {
            "eval_reward_mean": np.mean(total_rewards),
            "eval_reward_std": np.std(total_rewards)
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'ref_policy_state_dict': self.ref_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'episode': self.episode,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ref_policy.load_state_dict(checkpoint['ref_policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode = checkpoint.get('episode', 0)