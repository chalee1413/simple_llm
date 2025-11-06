"""
KTO (Kahneman-Tversky Optimization) implementation for RLHF pipeline.

DECISION RATIONALE:
- KTO algorithm as alternative to DPO/PPO
- Works with binary feedback (good/bad) instead of preference pairs
- Simpler data collection (no need for pairwise comparisons)
- Based on prospect theory (Kahneman-Tversky)

References:
- Ethayarajh et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization.
  https://arxiv.org/abs/2402.01306
- TRL library for KTO implementation (if available)
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
from datasets import Dataset

try:
    from trl import KTOTrainer, KTOConfig
    KTO_AVAILABLE = True
except ImportError:
    KTO_AVAILABLE = False
    KTOConfig = None
    logger = logging.getLogger(__name__)

from config import Config

logger = logging.getLogger(__name__)


def prepare_kto_dataset(
    feedback_data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Prepare dataset for KTO training.
    
    KTO works with binary feedback (good/bad) instead of preference pairs.
    Format: {"prompt": str, "response": str, "label": "good" or "bad"}
    
    TRL KTOTrainer expects text columns (prompt, response, label) and handles
    tokenization internally via its data collator.
    
    Args:
        feedback_data: List of feedback dictionaries with prompt, response, label
        tokenizer: HuggingFace tokenizer (not used for tokenization, just for validation)
        max_length: Maximum sequence length (used by KTOTrainer)
    
    Returns:
        HuggingFace Dataset with text columns for KTO training
    """
    def format_example(fb: Dict[str, Any]) -> Dict[str, Any]:
        """Format feedback example for KTO."""
        prompt = fb["prompt"]
        response = fb["response"]
        label = fb.get("label", "good")  # Default to "good"
        
        # TRL KTOTrainer expects 'completion' instead of 'response'
        # Label should be boolean: True for "good", False for "bad"
        label_bool = label.lower() in ("good", "true", "1", "yes")
        
        return {
            "prompt": prompt,
            "completion": response,
            "label": label_bool
        }
    
    formatted_data = [format_example(fb) for fb in feedback_data]
    dataset = Dataset.from_list(formatted_data)
    
    logger.info(f"Prepared KTO dataset with {len(dataset)} examples")
    return dataset


class KTOTrainerWrapper:
    """
    KTO trainer for RLHF policy optimization.
    
    DECISION RATIONALE:
    - Encapsulates KTO training logic
    - Works with binary feedback (simpler than preference pairs)
    - Based on prospect theory
    """
    
    def __init__(
        self,
        model_name: str,
        ref_model_name: Optional[str] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize KTO trainer.
        
        Args:
            model_name: Policy model name (SFT model)
            ref_model_name: Optional reference model (defaults to model_name)
            output_dir: Output directory for checkpoints
        """
        self.model_name = model_name
        self.ref_model_name = ref_model_name or model_name
        self.output_dir = output_dir or Config.RLHF_MODELS_DIR / "kto"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.kto_trainer = None
        
        logger.info(f"KTOTrainerWrapper initialized with model: {model_name}")
    
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
        Train policy model with KTO.
        
        Args:
            train_dataset: Training dataset (binary feedback)
            eval_dataset: Optional evaluation dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            beta: KTO temperature parameter
            max_length: Maximum sequence length
        
        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            self.load_models()
        
        num_epochs = num_epochs or Config.KTO_NUM_EPOCHS
        learning_rate = learning_rate or Config.KTO_LEARNING_RATE
        batch_size = batch_size or Config.KTO_BATCH_SIZE
        gradient_accumulation_steps = gradient_accumulation_steps or Config.KTO_GRADIENT_ACCUMULATION_STEPS
        beta = beta or Config.KTO_BETA
        max_length = max_length or Config.KTO_MAX_SEQ_LENGTH
        
        if not KTO_AVAILABLE or KTOConfig is None:
            raise ImportError(
                "KTOTrainer not available in TRL. "
                "Please install TRL >= 0.9.0 to use KTO training: pip install trl>=0.9.0"
            )
        
        # TRL 0.24+ requires KTOConfig instead of TrainingArguments
        kto_config = KTOConfig(
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
            fp16=False,
            bf16=False,
            report_to="none",
            beta=beta,
            max_length=max_length,
            max_prompt_length=max_length // 2
        )
        
        # TRL 0.24+ requires ref_model to be None or a copy, not the same object
        ref_model_for_kto = None if self.model_name == self.ref_model_name else self.ref_model
        
        self.kto_trainer = KTOTrainer(
            model=self.model,
            ref_model=ref_model_for_kto,
            args=kto_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer
        )
        
        logger.info("Starting KTO training...")
        train_result = self.kto_trainer.train()
        
        logger.info("KTO training completed")
        
        self.kto_trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "epoch": train_result.metrics.get("epoch", num_epochs)
        }
        
        if eval_dataset:
            eval_result = self.kto_trainer.evaluate()
            metrics.update({
                "eval_loss": eval_result.get("eval_loss", 0),
                "eval_runtime": eval_result.get("eval_runtime", 0)
            })
        
        return metrics
    
    def save_model(self, path: Optional[Path] = None):
        """Save fine-tuned model."""
        if self.kto_trainer:
            self.kto_trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
        elif self.model:
            save_path = path or self.output_dir
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to: {path or self.output_dir}")


def train_with_kto(
    model_name: str,
    feedback_data: List[Dict[str, Any]],
    ref_model_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for KTO training.
    
    Args:
        model_name: Policy model name
        feedback_data: List of feedback dictionaries with prompt, response, label
        ref_model_name: Optional reference model
        output_dir: Output directory
        **kwargs: Additional training arguments
    
    Returns:
        Training metrics
    
    Note:
        KTO works with binary feedback (good/bad) instead of preference pairs.
        Format: {"prompt": str, "response": str, "label": "good" or "bad"}
    """
    trainer = KTOTrainerWrapper(model_name, ref_model_name, output_dir)
    trainer.load_models()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = prepare_kto_dataset(feedback_data, tokenizer)
    
    return trainer.train(train_dataset, **kwargs)

