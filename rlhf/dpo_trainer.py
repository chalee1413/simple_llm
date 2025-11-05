"""
DPO (Direct Preference Optimization) implementation for RLHF pipeline.

DECISION RATIONALE:
- DPO algorithm as alternative to PPO
- Direct optimization on preference data
- No separate reward model required
- Simpler training pipeline

References:
- Rafailov et al. (2024). Direct Preference Optimization. https://arxiv.org/abs/2305.18290
- TRL library for DPO implementation
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from trl import DPOTrainer
from datasets import Dataset

from config import Config

logger = logging.getLogger(__name__)


def prepare_dpo_dataset(
    preferences: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Prepare dataset for DPO training.
    
    Args:
        preferences: List of preference dictionaries
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        HuggingFace Dataset formatted for DPO training
    """
    def format_example(pref: Dict[str, Any]) -> Dict[str, str]:
        """Format preference example for DPO."""
        prompt = pref["prompt"]
        chosen = pref["chosen"]
        rejected = pref["rejected"]
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
    
    formatted_data = [format_example(pref) for pref in preferences]
    
    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        """Tokenize examples for DPO."""
        prompts = examples["prompt"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        
        prompt_tokenized = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        chosen_tokenized = tokenizer(
            [f"{p} {c}" for p, c in zip(prompts, chosen)],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        rejected_tokenized = tokenizer(
            [f"{p} {r}" for p, r in zip(prompts, rejected)],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            "prompt_input_ids": prompt_tokenized["input_ids"],
            "prompt_attention_mask": prompt_tokenized["attention_mask"],
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
    
    logger.info(f"Prepared DPO dataset with {len(tokenized_dataset)} examples")
    return tokenized_dataset


class DPOTrainerWrapper:
    """
    DPO trainer for RLHF policy optimization.
    
    DECISION RATIONALE:
    - Encapsulates DPO training logic
    - Direct optimization on preference data
    - No separate reward model required
    - Simpler than PPO pipeline
    """
    
    def __init__(
        self,
        model_name: str,
        ref_model_name: Optional[str] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize DPO trainer.
        
        Args:
            model_name: Policy model name (SFT model)
            ref_model_name: Optional reference model (defaults to model_name)
            output_dir: Output directory for checkpoints
        """
        self.model_name = model_name
        self.ref_model_name = ref_model_name or model_name
        self.output_dir = output_dir or Config.RLHF_MODELS_DIR / "dpo"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.dpo_trainer = None
        
        logger.info(f"DPOTrainerWrapper initialized with model: {model_name}")
    
    def load_models(self):
        """Load policy model, reference model, and tokenizer."""
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
        
        if self.ref_model_name != self.model_name:
            logger.info(f"Loading reference model: {self.ref_model_name}")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.ref_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
        else:
            logger.info("Using policy model as reference model")
            self.ref_model = self.model
        
        logger.info("Models loaded successfully")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        beta: Optional[float] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train policy model with DPO.
        
        Args:
            train_dataset: Training dataset (preferences)
            eval_dataset: Optional evaluation dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            beta: DPO temperature parameter
            max_length: Maximum sequence length
        
        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            self.load_models()
        
        num_epochs = num_epochs or Config.DPO_NUM_EPOCHS
        learning_rate = learning_rate or Config.DPO_LEARNING_RATE
        batch_size = batch_size or Config.DPO_BATCH_SIZE
        gradient_accumulation_steps = gradient_accumulation_steps or Config.DPO_GRADIENT_ACCUMULATION_STEPS
        beta = beta or Config.DPO_BETA
        max_length = max_length or Config.DPO_MAX_SEQ_LENGTH
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
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
        
        self.dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            beta=beta,
            max_length=max_length,
            max_prompt_length=max_length // 2
        )
        
        logger.info("Starting DPO training...")
        train_result = self.dpo_trainer.train()
        
        logger.info("DPO training completed")
        
        self.dpo_trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "epoch": train_result.metrics.get("epoch", num_epochs)
        }
        
        if eval_dataset:
            eval_result = self.dpo_trainer.evaluate()
            metrics.update({
                "eval_loss": eval_result.get("eval_loss", 0),
                "eval_runtime": eval_result.get("eval_runtime", 0)
            })
        
        return metrics
    
    def save_model(self, path: Optional[Path] = None):
        """Save fine-tuned model."""
        if self.dpo_trainer:
            self.dpo_trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
        elif self.model:
            save_path = path or self.output_dir
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to: {path or self.output_dir}")


def train_with_dpo(
    model_name: str,
    preferences: List[Dict[str, Any]],
    ref_model_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for DPO training.
    
    Args:
        model_name: Policy model name
        preferences: List of preference dictionaries
        ref_model_name: Optional reference model
        output_dir: Output directory
        **kwargs: Additional training arguments
    
    Returns:
        Training metrics
    """
    trainer = DPOTrainerWrapper(model_name, ref_model_name, output_dir)
    trainer.load_models()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = prepare_dpo_dataset(preferences, tokenizer)
    
    return trainer.train(train_dataset, **kwargs)
