"""
Supervised Fine-Tuning (SFT) implementation for RLHF pipeline.

DECISION RATIONALE:
- Initial model fine-tuning on instruction-following data
- Foundation for RLHF training
- Integration with HuggingFace Transformers
- Checkpointing and evaluation integration

References:
- Instruction fine-tuning methodologies (2024-2025)
- HuggingFace Transformers training
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

from config import Config

logger = logging.getLogger(__name__)


class SupervisedFineTuner:
    """
    Supervised fine-tuning for instruction-following models.
    
    DECISION RATIONALE:
    - Encapsulates SFT training logic
    - Supports checkpointing and evaluation
    - Integrates with HuggingFace Transformers
    - Provides training progress tracking
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: Optional[Path] = None,
        max_length: int = 512
    ):
        """
        Initialize SFT trainer.
        
        Args:
            model_name: HuggingFace model name
            output_dir: Output directory for checkpoints
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.output_dir = output_dir or Config.RLHF_MODELS_DIR / "sft"
        self.max_length = max_length
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        logger.info(f"SupervisedFineTuner initialized with model: {model_name}")
    
    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,  # Use float32 for small laptop compatibility
            device_map=None,  # Use CPU for small laptop compatibility
            low_cpu_mem_usage=True
        )
        
        logger.info("Model loaded successfully")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        weight_decay: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Train model on instruction dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
            weight_decay: Weight decay
        
        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            self.load_model()
        
        num_epochs = num_epochs or Config.SFT_NUM_EPOCHS
        learning_rate = learning_rate or Config.SFT_LEARNING_RATE
        batch_size = batch_size or 1  # Small batch size for small laptop (override Config.SFT_BATCH_SIZE)
        gradient_accumulation_steps = gradient_accumulation_steps or 8  # Increase accumulation for small laptop
        warmup_steps = warmup_steps or Config.SFT_WARMUP_STEPS
        weight_decay = weight_decay or Config.SFT_WEIGHT_DECAY
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=Config.RLHF_LOGGING_STEPS,
            save_steps=Config.RLHF_SAVE_STEPS,
            eval_steps=Config.RLHF_EVAL_STEPS if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=False,  # Disable fp16 for small laptop compatibility
            report_to="none"
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info("Starting SFT training...")
        train_result = self.trainer.train()
        
        logger.info("SFT training completed")
        
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
    
    def save_model(self, path: Optional[Path] = None):
        """Save fine-tuned model."""
        if self.trainer:
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
        elif self.model:
            save_path = path or self.output_dir
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to: {path or self.output_dir}")
    
    def load_model_from_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")


def train_sft_model(
    model_name: str,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for SFT training.
    
    Args:
        model_name: HuggingFace model name
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        output_dir: Output directory
        **kwargs: Additional training arguments
    
    Returns:
        Training metrics
    """
    trainer = SupervisedFineTuner(model_name, output_dir)
    return trainer.train(train_dataset, eval_dataset, **kwargs)
