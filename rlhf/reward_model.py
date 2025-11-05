"""
Reward model implementation for RLHF pipeline.

DECISION RATIONALE:
- Reward model architecture based on policy model
- Training on human preference data
- Reward prediction for PPO training
- Integration with HuggingFace Transformers

References:
- Reward model training methodologies (2024-2025)
- Pairwise comparison training for reward models
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

from config import Config

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Reward model for predicting human preferences.
    
    DECISION RATIONALE:
    - Based on policy model architecture with regression head
    - Predicts reward scores for model outputs
    - Trained on pairwise preference comparisons
    """
    
    def __init__(self, model_name: str):
        """
        Initialize reward model.
        
        Args:
            model_name: Base model name (same as policy model)
        """
        super().__init__()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,  # Single output for reward score
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"RewardModel initialized with base model: {model_name}")
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass to predict reward.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Reward scores
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)
    
    def save(self, path: Path):
        """Save reward model."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        logger.info(f"Reward model saved to: {path}")
    
    def load(self, path: Path):
        """Load reward model from checkpoint."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        logger.info(f"Reward model loaded from: {path}")


def prepare_reward_dataset(
    preferences: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Prepare dataset for reward model training.
    
    Args:
        preferences: List of preference dictionaries
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        HuggingFace Dataset formatted for reward model training
    """
    def format_example(pref: Dict[str, Any]) -> Dict[str, str]:
        """Format preference example."""
        prompt = pref["prompt"]
        chosen = pref["chosen"]
        rejected = pref["rejected"]
        
        chosen_text = f"{prompt} {chosen}"
        rejected_text = f"{prompt} {rejected}"
        
        return {
            "chosen": chosen_text,
            "rejected": rejected_text
        }
    
    formatted_data = [format_example(pref) for pref in preferences]
    
    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        """Tokenize examples for reward model."""
        chosen_tokenized = tokenizer(
            examples["chosen"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        rejected_tokenized = tokenizer(
            examples["rejected"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_tokenized["input_ids"],
            "chosen_attention_mask": chosen_tokenized["attention_mask"],
            "rejected_input_ids": rejected_tokenized["input_ids"],
            "rejected_attention_mask": rejected_tokenized["attention_mask"]
        }
    
    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Prepared reward dataset with {len(tokenized_dataset)} examples")
    return tokenized_dataset


class RewardTrainer:
    """
    Trainer for reward model.
    
    DECISION RATIONALE:
    - Encapsulates reward model training logic
    - Handles pairwise comparison training
    - Provides checkpointing and evaluation
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize reward trainer.
        
        Args:
            model_name: Base model name
            output_dir: Output directory for checkpoints
        """
        self.model_name = model_name
        self.output_dir = output_dir or Config.RLHF_MODELS_DIR / "reward_model"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.reward_model = None
        self.trainer = None
        
        logger.info(f"RewardTrainer initialized with model: {model_name}")
    
    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading reward model base: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.reward_model = RewardModel(self.model_name)
        
        logger.info("Reward model loaded successfully")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        warmup_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train reward model on preference data.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
        
        Returns:
            Training metrics dictionary
        """
        if self.reward_model is None:
            self.load_model()
        
        num_epochs = num_epochs or Config.REWARD_MODEL_NUM_EPOCHS
        learning_rate = learning_rate or Config.REWARD_MODEL_LEARNING_RATE
        batch_size = batch_size or Config.REWARD_MODEL_BATCH_SIZE
        gradient_accumulation_steps = gradient_accumulation_steps or Config.REWARD_MODEL_GRADIENT_ACCUMULATION_STEPS
        warmup_steps = warmup_steps or Config.REWARD_MODEL_WARMUP_STEPS
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=Config.RLHF_LOGGING_STEPS,
            save_steps=Config.RLHF_SAVE_STEPS,
            eval_steps=Config.RLHF_EVAL_STEPS if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        def reward_loss_function(model_outputs, labels):
            """Compute reward loss for pairwise comparisons."""
            chosen_rewards = model_outputs["chosen_logits"]
            rejected_rewards = model_outputs["rejected_logits"]
            
            loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
            return loss
        
        self.trainer = Trainer(
            model=self.reward_model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        logger.info("Starting reward model training...")
        train_result = self.trainer.train()
        
        logger.info("Reward model training completed")
        
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "epoch": train_result.metrics.get("epoch", num_epochs)
        }
        
        if eval_dataset:
            eval_result = self.trainer.evaluate()
            metrics.update({
                "eval_loss": eval_result.get("eval_loss", 0),
                "eval_runtime": eval_result.get("eval_runtime", 0)
            })
        
        return metrics


def train_reward_model(
    model_name: str,
    preferences: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for reward model training.
    
    Args:
        model_name: Base model name
        preferences: List of preference dictionaries
        output_dir: Output directory
        **kwargs: Additional training arguments
    
    Returns:
        Training metrics
    """
    trainer = RewardTrainer(model_name, output_dir)
    trainer.load_model()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = prepare_reward_dataset(preferences, tokenizer)
    
    return trainer.train(train_dataset, **kwargs)
