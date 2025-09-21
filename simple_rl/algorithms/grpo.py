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
        reward_fn: Optional[Callable[[str, str, Optional[str]], float]] = None,
        use_wandb: bool = False
    ):
        """
        Initialize GRPO algorithm.
        
        Args:
            model: Language model for text generation (if None, creates from config)
            config: Configuration dictionary
            reward_fn: Function to compute rewards (prompt, completion, answer) -> float
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
        if model is None:
            if not config:
                raise ValueError("Either model or config must be provided")
            self.policy = LanguageModel(config)
        elif isinstance(model, LanguageModel):
            self.policy = model
        else:
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

        # Training parameters
        training_config = self.config.get("training", {})
        self.learning_rate = training_config.get("learning_rate", 1e-5)
        self.batch_size = training_config.get("batch_size", 8)
        self.max_new_tokens = training_config.get("max_new_tokens", 128)
        self.temperature = training_config.get("temperature", 0.9)
        self.top_k = training_config.get("top_k", None)
        self.top_p = training_config.get("top_p", 0.9)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        
        # Prompt formatting configuration
        formatting_config = self.config.get("formatting", {})
        self.system_prompt = formatting_config.get("system_prompt", None)
        self.task_formatter = formatting_config.get("task_formatter", None)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate
        )
        
        # Reward function
        self.reward_fn = reward_fn or self._default_reward_fn
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

        # Initialize gradients to zero for accumulation
        self.optimizer.zero_grad()

    def _default_reward_fn(self, prompt: str, completion: str, answer: Optional[str] = None) -> float:
        """Default reward function based on completion length."""
        # Simple heuristic: longer completions get higher rewards
        # This should be replaced with actual reward logic
        return min(len(completion.split()) / 50.0, 1.0)
    
    def format_prompt(self, prompt: str, use_formatting: bool = True) -> str:
        """
        Apply task-specific formatting, then delegate to model's format_prompt.

        Args:
            prompt: Raw prompt text
            use_formatting: Whether to apply formatting

        Returns:
            Formatted prompt string
        """
        if not use_formatting:
            return prompt

        # Step 1: Apply task-specific formatting (if configured)
        formatted = prompt
        if self.task_formatter is not None:
            if callable(self.task_formatter):
                # If task_formatter is a function
                formatted = self.task_formatter(prompt)
            elif isinstance(self.task_formatter, str):
                # If task_formatter is a template string
                formatted = self.task_formatter.replace("{prompt}", prompt)

        # Step 2: Apply model-specific formatting (chat templates, etc.)
        # Delegate to the policy model's format_prompt method
        if hasattr(self.policy, 'format_prompt'):
            formatted = self.policy.format_prompt(
                formatted,
                system_prompt=self.system_prompt
            )

        return formatted
    
    def set_system_prompt(self, system_prompt: str):
        """
        Set the system prompt for chat models.

        Args:
            system_prompt: System prompt text
        """
        self.system_prompt = system_prompt

    def set_task_formatter(self, formatter):
        """
        Set the task-specific formatter.

        Args:
            formatter: Either a function (prompt -> formatted_prompt) or a template string with {prompt}
        """
        self.task_formatter = formatter
    
    def generate_trajectories(
        self,
        prompts: List[str],
        answers: Optional[List[str]] = None,
        use_formatting: bool = True
    ) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate trajectories for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            answers: Optional list of correct answers for reward computation
            use_formatting: Whether to apply prompt formatting
            
        Returns:
            Tuple of (prompts, completions, rewards, log_probs, ref_log_probs, completion_mask)
        """
        all_prompts = []
        all_completions = []
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []
        all_completion_mask = []
        # Process each prompt
        for i, prompt in enumerate(prompts):
            # Get answer if provided
            answer = answers[i] if answers else None
            # Format prompt if configured
            formatted_prompt = self.format_prompt(prompt, use_formatting)
            
            # Tokenize single prompt; we'll sample multiple completions via num_return_sequences
            tokenized = self.policy.tokenize([formatted_prompt])
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
                policy_log_probs = self.policy.compute_log_probs(
                    generated_ids,
                    attention_mask=generated_mask,
                    target_mask=None  # Let the model handle masking internally
                )
                # Extract only completion log probs (account for shift by 1)
                policy_log_probs = policy_log_probs[:, prompt_length -1 : prompt_length -1 + generated_length]
            if prev_mode:
                self.policy.train()

            # Compute log probabilities for reference model
            with torch.no_grad():
                ref_log_probs = self.ref_policy.compute_log_probs(
                    generated_ids,
                    attention_mask=generated_mask,
                    target_mask=None
                )
                # Extract only completion log probs (account for shift by 1)
                ref_log_probs = ref_log_probs[:, prompt_length -1 : prompt_length -1 + generated_length]
            generated_mask = generated_mask[:, 1:]
            # Create mask for completion tokens only
            completion_mask = generated_mask[:, prompt_length -1 : prompt_length -1 + generated_length]
            
            # Compute rewards for each completion
            rewards = []
            for completion in completions:
                reward = self.reward_fn(prompt, completion, answer)
                rewards.append(reward)
            
            # Store results
            all_prompts.extend([prompt] * self.group_size)
            all_completions.extend(completions)
            all_rewards.extend(rewards)
            
            # Apply masking to log probs before storing
            policy_log_probs = policy_log_probs * completion_mask
            ref_log_probs = ref_log_probs * completion_mask

            assert policy_log_probs.shape == ref_log_probs.shape
            assert completion_mask.shape == policy_log_probs.shape
            assert policy_log_probs[0].shape[0] == (completion_ids[0].shape[0]) 

            # Store each sample's log probs separately
            for i in range(self.group_size):
                all_log_probs.append(policy_log_probs[i])
                all_ref_log_probs.append(ref_log_probs[i])
                all_completion_mask.append(completion_mask[i])
        # Stack into tensors
        all_log_probs = pad_sequence(all_log_probs, batch_first=True, padding_value=0.0)
        all_ref_log_probs = pad_sequence(all_ref_log_probs, batch_first=True, padding_value=0.0)
        all_completion_mask = pad_sequence(all_completion_mask, batch_first=True, padding_value=0.0)
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)

        return all_prompts, all_completions, all_rewards, all_log_probs, all_ref_log_probs, all_completion_mask
        
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

    def train_step(self, batch: Dict[str, Any], accumulation_step: int = 0) -> Dict[str, float]:
        """
        Perform a single training step with gradient accumulation support.

        Args:
            batch: Dictionary with 'prompts' and optionally 'answers' keys
            accumulation_step: Current step in gradient accumulation (0-indexed)

        Returns:
            Dictionary of training metrics
        """
        self.policy.train()
        prompts = batch["prompts"]
        answers = batch.get("answers", None)  # Optional answers for reward computation

        # Generate trajectories
        _, _, rewards, log_probs, ref_log_probs, completion_mask = self.generate_trajectories(
            prompts, answers=answers
        )

        # Compute advantages
        advantages = self.compute_advantages(rewards)

        # Compute loss
        loss, metrics = self.compute_loss(log_probs, advantages, ref_log_probs, completion_mask)

        # Scale loss by accumulation steps to maintain effective learning rate
        loss = loss / self.gradient_accumulation_steps

        # Backward pass (accumulate gradients)
        loss.backward()

        # Only update weights after accumulating gradients
        if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.gradient_clip)

            # Optimizer step
            self.optimizer.step()

            # Clear gradients for next accumulation
            self.optimizer.zero_grad()

            # Update statistics only on actual updates
            self.total_steps += 1

        # Update old log probs
        self.old_log_probs = log_probs.detach()

        # Add rewards to metrics (scale back the loss for reporting)
        metrics["reward_mean"] = rewards.mean().item()
        metrics["reward_std"] = rewards.std().item()
        metrics["total_loss"] = (loss * self.gradient_accumulation_steps).item()

        # Log to wandb if enabled and on actual update step
        if self.use_wandb and (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
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

            # Minibatch training within an episode
            training_cfg = self.config.get("training", {})
            minibatch_size = training_cfg.get("minibatch_size", self.batch_size)
            update_epochs = training_cfg.get("update_epochs", 1)

            total_loss = 0.0
            total_pg_loss = 0.0
            total_kl = 0.0
            total_reward_mean = 0.0
            total_reward_std = 0.0
            num_updates = 0

            for _ in range(update_epochs):
                # Shuffle prompts each epoch
                perm = torch.randperm(len(prompts)).tolist()
                shuffled_prompts = [prompts[idx] for idx in perm]

                for start in range(0, len(shuffled_prompts), minibatch_size):
                    end = min(start + minibatch_size, len(shuffled_prompts))
                    mb_prompts = shuffled_prompts[start:end]
                    batch = {"prompts": mb_prompts}
                    metrics = self.train_step(batch)

                    total_loss += metrics["total_loss"]
                    total_pg_loss += metrics["pg_loss"]
                    total_kl += metrics["kl_divergence"]
                    total_reward_mean += metrics["reward_mean"]
                    total_reward_std += metrics.get("reward_std", 0.0)
                    num_updates += 1

            # Average metrics over minibatches/epochs
            if num_updates > 0:
                metrics = {
                    "total_loss": total_loss / num_updates,
                    "pg_loss": total_pg_loss / num_updates,
                    "kl_divergence": total_kl / num_updates,
                    "reward_mean": total_reward_mean / num_updates,
                    "reward_std": total_reward_std / num_updates,
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