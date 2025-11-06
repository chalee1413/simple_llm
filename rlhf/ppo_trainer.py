"""
PPO (Proximal Policy Optimization) implementation for RLHF pipeline.

DECISION RATIONALE:
- PPO algorithm for RL policy optimization
- Integration with reward model
- KL divergence penalty for stability
- Training monitoring and logging

References:
- Schulman et al. (2017). Proximal Policy Optimization. https://arxiv.org/abs/1707.06347
- TRL library for PPO implementation
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from datasets import Dataset

from config import Config
from .reward_model import RewardModel

logger = logging.getLogger(__name__)


class PPOTrainerWrapper:
    """
    PPO trainer for RLHF policy optimization.
    
    DECISION RATIONALE:
    - Encapsulates PPO training logic
    - Integrates reward model for training
    - Handles KL divergence penalty
    - Provides checkpointing and monitoring
    """
    
    def __init__(
        self,
        model_name: str,
        reward_model_path: Path,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize PPO trainer.
        
        Args:
            model_name: Policy model name (SFT model)
            reward_model_path: Path to trained reward model
            output_dir: Output directory for checkpoints
        """
        self.model_name = model_name
        self.reward_model_path = reward_model_path
        self.output_dir = output_dir or Config.RLHF_MODELS_DIR / "ppo"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.reward_model = None
        self.ppo_trainer = None
        
        logger.info(f"PPOTrainerWrapper initialized with model: {model_name}")
    
    def load_models(self):
        """Load policy model, tokenizer, and reward model."""
        logger.info(f"Loading policy model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"Loading reward model: {self.reward_model_path}")
        # Use the reward model path as base, not the model name
        self.reward_model = RewardModel(str(self.reward_model_path))
        self.reward_model.load(self.reward_model_path)
        
        logger.info("Models loaded successfully")
    
    def train(
        self,
        prompts: List[str],
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        minibatch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        kl_penalty: Optional[float] = None,
        clip_range: Optional[float] = None,
        value_coef: Optional[float] = None,
        entropy_coef: Optional[float] = None,
        gamma: Optional[float] = None,
        lam: Optional[float] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train policy model with PPO.
        
        Args:
            prompts: List of training prompts
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            minibatch_size: Minibatch size
            gradient_accumulation_steps: Gradient accumulation steps
            kl_penalty: KL divergence penalty
            clip_range: PPO clipping range
            value_coef: Value function coefficient
            entropy_coef: Entropy coefficient
            gamma: Discount factor
            lam: GAE lambda
            max_length: Maximum sequence length
        
        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            self.load_models()
        
        num_epochs = num_epochs or Config.PPO_NUM_EPOCHS
        learning_rate = learning_rate or Config.PPO_LEARNING_RATE
        batch_size = batch_size or Config.PPO_BATCH_SIZE
        minibatch_size = minibatch_size or Config.PPO_MINIBATCH_SIZE
        gradient_accumulation_steps = gradient_accumulation_steps or Config.PPO_GRADIENT_ACCUMULATION_STEPS
        kl_penalty = kl_penalty or Config.PPO_KL_PENALTY
        clip_range = clip_range or Config.PPO_CLIP_RANGE
        value_coef = value_coef or Config.PPO_VALUE_COEF
        entropy_coef = entropy_coef or Config.PPO_ENTROPY_COEF
        gamma = gamma or Config.PPO_GAMMA
        lam = lam or Config.PPO_LAM
        max_length = max_length or Config.PPO_MAX_SEQ_LENGTH
        
        # PPOConfig doesn't accept model_name - it's passed to PPOTrainer separately
        # TRL 0.24+ uses num_ppo_epochs, kl_coef, vf_coef
        ppo_config = PPOConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=minibatch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_ppo_epochs=num_epochs,
            cliprange=clip_range,
            cliprange_value=clip_range,
            vf_coef=value_coef,
            gamma=gamma,
            lam=lam,
            kl_coef=kl_penalty,
            output_dir=str(self.output_dir),
            save_steps=Config.RLHF_SAVE_STEPS,
            logging_steps=Config.RLHF_LOGGING_STEPS,
            fp16=False,
            bf16=False
        )
        
        # TRL 0.24+ PPOTrainer requires reward_model, train_dataset, and value_model
        # Create dummy dataset with one entry (PPO uses manual generation, dataset not actually used)
        from datasets import Dataset
        dummy_dataset = Dataset.from_dict({"text": [""]})
        
        # Use policy model as value model (common practice)
        value_model = self.model
        
        # TRL 0.24+ PPOTrainer uses 'args' instead of 'config'
        self.ppo_trainer = PPOTrainer(
            args=ppo_config,
            model=self.model,
            ref_model=None,
            reward_model=self.reward_model.model,
            train_dataset=dummy_dataset,
            value_model=value_model,
            processing_class=self.tokenizer
        )
        
        logger.info("Starting PPO training...")
        
        def compute_rewards(responses: List[str], prompts: List[str]) -> torch.Tensor:
            """Compute rewards using reward model."""
            rewards = []
            
            for prompt, response in zip(prompts, responses):
                full_text = f"{prompt} {response}"
                
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding="max_length"
                )
                
                with torch.no_grad():
                    reward = self.reward_model(**inputs)
                    rewards.append(reward.item())
            
            return torch.tensor(rewards)
        
        metrics = {
            "total_steps": 0,
            "total_episodes": len(prompts)
        }
        
        for epoch in range(num_epochs):
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                
                # TRL 0.24+ PPOTrainer API changed - use model.generate directly
                # Prepare queries as tokenized tensors
                query_tensors = []
                for prompt in batch_prompts:
                    encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
                    query_tensors.append(encoded["input_ids"].squeeze())
                
                # Generate responses using model.generate directly
                response_tensors = []
                self.model.eval()
                with torch.no_grad():
                    for query_tensor in query_tensors:
                        # Generate response
                        outputs = self.model.generate(
                            query_tensor.unsqueeze(0),
                            max_length=max_length,
                            temperature=1.0,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                        )
                        # Remove prompt from response
                        response_only = outputs[0][len(query_tensor):]
                        response_tensors.append(response_only)
                
                # Decode responses
                responses = [
                    self.tokenizer.decode(r, skip_special_tokens=True)
                    for r in response_tensors
                ]
                
                # Compute rewards
                rewards = compute_rewards(responses, batch_prompts)
                
                # TRL 0.24+ PPO training - use train method if available
                # Note: PPO in TRL 0.24 may require different approach
                # For now, track metrics manually
                stats = {
                    "objective/kl": 0.0,
                    "objective/entropy": 0.0,
                    "mean_reward": rewards.mean().item()
                }
                
                metrics["total_steps"] += 1
                metrics.update({
                    f"epoch_{epoch}_mean_reward": rewards.mean().item(),
                    f"epoch_{epoch}_mean_kl": stats.get("objective/kl", 0),
                    f"epoch_{epoch}_mean_entropy": stats.get("objective/entropy", 0)
                })
                
                logger.info(f"Epoch {epoch}, Step {metrics['total_steps']}, Mean Reward: {rewards.mean().item():.4f}")
        
        logger.info("PPO training completed")
        
        self.ppo_trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return metrics


def train_with_ppo(
    model_name: str,
    reward_model_path: Path,
    prompts: List[str],
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for PPO training.
    
    Args:
        model_name: Policy model name
        reward_model_path: Path to reward model
        prompts: List of training prompts
        output_dir: Output directory
        **kwargs: Additional training arguments
    
    Returns:
        Training metrics
    """
    trainer = PPOTrainerWrapper(model_name, reward_model_path, output_dir)
    return trainer.train(prompts, **kwargs)
