"""
GRPO (Group Relative Policy Optimization) implementation.

GRPO normalizes rewards within groups/batches to reduce variance
in policy gradient estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
import math
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
        config: Optional[Dict[str, Any]] = None,
        batch_reward_fn: Optional[Callable] | None = None,
        use_wandb: bool = False
    ):
        """
        Initialize GRPO algorithm.
        
        Args:
            config: Configuration dictionary
            batch_reward_fn: Function to compute rewards (prompt, completion, answer) -> float
            use_wandb: Whether to use Weights & Biases logging
        """
        # Store config and setup device
        self.config = config or {}
        self.use_wandb = use_wandb
        # Prefer CUDA, then MPS, else CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Initialize or create model
        
        if not config:
            raise ValueError("Either model or config must be provided")

        self.policy = LanguageModel(config)
        
        # Move policy to device
        self.policy = self.policy.to(self.device)
        
        # Create reference model (frozen copy for KL divergence)
        # Create reference model (frozen copy for KL divergence)
        self.ref_policy = copy.deepcopy(self.policy)
        self.ref_policy = self.ref_policy.to(self.device)

        # Freeze ALL parameters
        for name, param in self.ref_policy.named_parameters():
            param.requires_grad = False
            
        # Put in eval mode (disables dropout, batchnorm updates, etc.)
        self.ref_policy.eval()

        # Verify freezing worked
        total_params = sum(1 for _ in self.ref_policy.parameters())
        frozen_params = sum(1 for p in self.ref_policy.parameters() if not p.requires_grad)
        assert frozen_params == total_params, f"Not all parameters frozen: {frozen_params}/{total_params}"
        print(f"✓ Reference model frozen: {frozen_params} parameters")

        # GRPO-specific parameters
        algo_config = self.config.get("algorithm", {})
        self.group_size = algo_config.get("group_size", 4)
        self.kl_coef = algo_config.get("kl_coef", 0.05)
        self.normalize_rewards = algo_config.get("normalize_rewards", True)
        self.clip_epsilon = algo_config.get("clip_epsilon", 0.2)
        self.store_completions = algo_config.get("store_completions", True)

        # Training parameters
        training_config = self.config.get("training", {})
        self.learning_rate = training_config.get("learning_rate", 1e-5)
        self.batch_size = training_config.get("batch_size", 8)
        self.minibatch_size = training_config.get("minibatch_size", self.batch_size)
        if self.minibatch_size in (None, 0):
            self.minibatch_size = self.batch_size
        self.max_new_tokens = training_config.get("max_new_tokens", 128)
        self.temperature = training_config.get("temperature", 0.9)
        self.top_k = training_config.get("top_k", None)
        self.top_p = training_config.get("top_p", 0.9)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)
        
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
        self.batch_reward_fn = batch_reward_fn
        self.old_log_probs = None
        
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
        answers: Optional[List[str]] = None,
        use_formatting: bool = True,
        store_outputs: Optional[bool] = None,
    ) -> Tuple[Optional[List[str]], Optional[List[str]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate trajectories for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            answers: Optional list of correct answers for reward computation
            use_formatting: Whether to apply prompt formatting
            store_outputs: Override flag controlling whether decoded prompts/completions are returned
            
        Returns:
            Tuple of (prompts, completions, rewards, log_probs, ref_log_probs, completion_mask)
        """
        store = self.store_completions if store_outputs is None else store_outputs
        all_prompts = [] if store else None
        all_completions = [] if store else None
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []
        all_completion_mask = []
        
        # Process each prompt to generate completions and collect log probs
        for prompt, answer in zip(prompts, answers):
            tokenized = self.policy.tokenize([prompt])
            prompt_ids = tokenized["input_ids"].to(self.device)
            prompt_mask = tokenized["attention_mask"].to(self.device)
            
            # Generate completions
            prev = self.policy.training
            self.policy.eval()
            with torch.no_grad():
                generated_ids, generated_mask = self.policy.generate(
                    prompt_ids, attention_mask=prompt_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k, top_p=self.top_p,
                    num_return_sequences=self.group_size,
                    min_new_tokens=1,  # also avoids empty completion windows
                )
            self.policy.train(prev)
            
            # Get prompt length to extract completions
            prompt_length = prompt_ids.shape[1]
            generated_length = generated_ids.shape[1] - prompt_length
            # Decode completions
            completion_ids = generated_ids[:, prompt_length:]
            completions = self.policy.decode(completion_ids)

            # Compute log probabilities for policy (only for completion tokens)
            prev_mode = self.policy.training
            self.policy.eval()
            with torch.enable_grad():  # Ensure gradients are enabled for policy
                prev_cache = getattr(self.policy.model.config, "use_cache", None)
                if prev_cache is not None:
                    self.policy.model.config.use_cache = False
                try:
                    policy_log_probs = self.policy.compute_log_probs(
                        generated_ids,
                        attention_mask=generated_mask,
                        target_mask=None  # Let the model handle masking internally
                    )
                finally:
                    if prev_cache is not None:
                        self.policy.model.config.use_cache = prev_cache
                # Extract only completion log probs (account for shift by 1)
                policy_log_probs = policy_log_probs[:, prompt_length -1 : prompt_length -1 + generated_length]
            if prev_mode:
                self.policy.train()

            # Compute log probabilities for reference model
            with torch.no_grad():
                prev_cache = getattr(self.ref_policy.model.config, "use_cache", None)
                if prev_cache is not None:
                    self.ref_policy.model.config.use_cache = False
                try:
                    ref_log_probs = self.ref_policy.compute_log_probs(
                        generated_ids,
                        attention_mask=generated_mask,
                        target_mask=None
                    )
                finally:
                    if prev_cache is not None:
                        self.ref_policy.model.config.use_cache = prev_cache
                # Extract only completion log probs (account for shift by 1)
                ref_log_probs = ref_log_probs[:, prompt_length -1 : prompt_length -1 + generated_length]
            generated_mask = generated_mask[:, 1:]
            # Create mask for completion tokens only
            completion_mask = generated_mask[:, prompt_length -1 : prompt_length -1 + generated_length]
            
            # Store results (except rewards - we'll compute those in batch)
            if store:
                all_prompts.extend([prompt] * self.group_size)
                all_completions.extend(completions)
            
            # Apply masking to log probs before storing
            policy_log_probs = policy_log_probs * completion_mask
            ref_log_probs = ref_log_probs * completion_mask

            assert policy_log_probs.shape == ref_log_probs.shape
            assert completion_mask.shape == policy_log_probs.shape
            assert policy_log_probs[0].shape[0] == (completion_ids[0].shape[0]) 

            # Compute rewards for this group of completions
            group_rewards = self.batch_reward_fn(completions, [answer] * self.group_size, self.device)
            all_rewards.append(group_rewards)  # Append the tensor for this group
            
            # Store each sample's log probs separately
            for i in range(self.group_size):
                all_log_probs.append(policy_log_probs[i])
                all_ref_log_probs.append(ref_log_probs[i])
                all_completion_mask.append(completion_mask[i])
        
        # Stack into tensors
        all_log_probs = pad_sequence(all_log_probs, batch_first=True, padding_value=0.0)
        all_ref_log_probs = pad_sequence(all_ref_log_probs, batch_first=True, padding_value=0.0)
        all_completion_mask = pad_sequence(all_completion_mask, batch_first=True, padding_value=0.0)
        all_rewards = torch.cat(all_rewards, dim=0)

        if store:
            return all_prompts, all_completions, all_rewards, all_log_probs, all_ref_log_probs, all_completion_mask
        return None, None, all_rewards, all_log_probs, all_ref_log_probs, all_completion_mask
        
    def compute_advantages(
        self,
        adjusted_rewards: torch.Tensor,
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
        batch_size = adjusted_rewards.shape[0]
        
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
    
    def compute_loss(self, log_probs, advantages, ref_log_probs, completion_mask):

        # Get sequence-level log probabilities by summing
        log_probs_sum = log_probs.sum(dim=-1)
        
        # Policy gradient loss (using summed log probs)
        pg_loss = -(log_probs_sum * advantages.detach()).mean()
        
        # TRL adds KL at the per-token level, then averages
        delta =  log_probs - ref_log_probs
        per_token_kl = torch.exp(delta) - delta - 1.0  # Schulman approx
        kl_penalty = (per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)

        # Total loss
        loss = pg_loss + (self.kl_coef * kl_penalty)
        
        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_divergence": kl_penalty.item(),  # Per-token average for monitoring
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "tokens_generated": completion_mask.sum().item(),
        }
        
        return loss, metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single training step with gradient accumulation.

        Args:
            batch: Dictionary with 'prompts' and optionally 'answers' keys

        Returns:
            Dictionary of training metrics
        """
        self.policy.train()
        prompts = batch["prompts"]
        answers = batch["answers"]  # Optional answers for reward computation

        # Calculate number of accumulation steps (ceil to cover partial minibatches)
        num_accumulation_steps = max(1, math.ceil(len(prompts) / self.minibatch_size))

        # Zero gradients at the start
        self.optimizer.zero_grad()

        # Accumulate metrics across minibatches
        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl = 0.0
        total_reward_mean = 0.0
        total_reward_std = 0.0
        total_tokens = 0

        # Total sequences across full batch (each prompt yields group_size sequences)
        total_sequences = len(prompts) * self.group_size

        # Process minibatches
        for i in range(0, len(prompts), self.minibatch_size):
            end_idx = min(i + self.minibatch_size, len(prompts))
            mb_prompts = prompts[i:end_idx]
            mb_answers = answers[i:end_idx] if answers else None

            # Generate trajectories for this minibatch
            prompts_out, completions_out, rewards, log_probs, ref_log_probs, completion_mask = self.generate_trajectories(
                mb_prompts, answers=mb_answers, store_outputs=self.store_completions
            )

            # Compute advantages
            advantages = self.compute_advantages(rewards)

            # Compute loss
            loss, mb_metrics = self.compute_loss(log_probs, advantages, ref_log_probs, completion_mask)

            # Weight loss by fraction of sequences in this minibatch for proper averaging
            mb_sequences = len(mb_prompts) * self.group_size
            weight = mb_sequences / max(1, total_sequences)
            scaled_loss = loss * weight

            # Backward pass (accumulate gradients)
            scaled_loss.backward()

            # Accumulate metrics (use weighted values to match actual training)
            total_loss += scaled_loss.item()
            total_pg_loss += mb_metrics["pg_loss"] * weight
            total_kl += mb_metrics["kl_divergence"] * weight
            total_reward_mean += rewards.mean().item() * weight
            total_reward_std += rewards.std().item() * weight
            total_tokens += mb_metrics.get("tokens_generated", 0)

            # Clean up intermediate tensors to prevent memory buildup
            if self.store_completions:
                del prompts_out, completions_out
            del rewards, log_probs, ref_log_probs, completion_mask, advantages, loss, scaled_loss
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Gradient clipping on accumulated gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.gradient_clip)

        # Update parameters
        self.optimizer.step()

        # Update statistics
        self.total_steps += 1

        # Aggregate metrics (we used weighted accumulation, so values are already averaged)
        metrics = {
            "total_loss": total_loss,
            "pg_loss": total_pg_loss,
            "kl_divergence": total_kl,
            "reward_mean": total_reward_mean,
            "reward_std": total_reward_std,
            "tokens_generated": total_tokens,  # Total tokens, not averaged
        }

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

            # Training with gradient accumulation
            training_cfg = self.config.get("training", {})
            update_epochs = training_cfg.get("update_epochs", 1)

            total_loss = 0.0
            total_pg_loss = 0.0
            total_kl = 0.0
            total_reward_mean = 0.0
            total_reward_std = 0.0
            total_tokens = 0.0

            for _ in range(update_epochs):
                # Shuffle prompts each epoch
                perm = torch.randperm(len(prompts)).tolist()
                shuffled_prompts = [prompts[idx] for idx in perm]

                # Process entire batch with gradient accumulation
                batch = {"prompts": shuffled_prompts}
                metrics = self.train_step(batch)

                total_loss += metrics["total_loss"]
                total_pg_loss += metrics["pg_loss"]
                total_kl += metrics["kl_divergence"]
                total_reward_mean += metrics["reward_mean"]
                total_reward_std += metrics.get("reward_std", 0.0)
                total_tokens += metrics.get("tokens_generated", 0.0)

            # Average metrics over epochs
            metrics = {
                "total_loss": total_loss / update_epochs,
                "pg_loss": total_pg_loss / update_epochs,
                "kl_divergence": total_kl / update_epochs,
                "reward_mean": total_reward_mean / update_epochs,
                "reward_std": total_reward_std / update_epochs,
                "tokens_generated": total_tokens / update_epochs,
            }
            
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
                _, completions, rewards, _, _, _ = self.generate_trajectories(prompts)
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