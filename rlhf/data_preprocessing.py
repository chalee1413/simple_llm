"""
Data preprocessing for RLHF pipeline.

DECISION RATIONALE:
- Instruction-following dataset preparation for SFT
- Data formatting and validation
- Dataset loading and preprocessing utilities
- Integration with HuggingFace datasets library

References:
- Instruction fine-tuning datasets (2024-2025 best practices)
- HuggingFace datasets library
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def format_instruction_prompt(instruction: str, context: Optional[str] = None, input_text: Optional[str] = None) -> str:
    """
    Format instruction prompt for fine-tuning.
    
    Args:
        instruction: Instruction text
        context: Optional context
        input_text: Optional input text
    
    Returns:
        Formatted prompt string
    """
    if context:
        prompt = f"Context: {context}\n\nInstruction: {instruction}\n\n"
    else:
        prompt = f"Instruction: {instruction}\n\n"
    
    if input_text:
        prompt += f"Input: {input_text}\n\n"
    
    prompt += "Response:"
    
    return prompt


def prepare_sft_dataset(
    data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Prepare dataset for supervised fine-tuning.
    
    Args:
        data: List of dictionaries with 'instruction', 'response', optional 'context', 'input'
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        HuggingFace Dataset formatted for training
    """
    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        """Format single example."""
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        context = example.get("context")
        input_text = example.get("input")
        
        prompt = format_instruction_prompt(instruction, context, input_text)
        full_text = prompt + " " + response
        
        return {
            "text": full_text,
            "prompt": prompt,
            "response": response
        }
    
    formatted_data = [format_example(ex) for ex in data]
    
    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """Tokenize examples."""
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Prepared SFT dataset with {len(tokenized_dataset)} examples")
    return tokenized_dataset


def load_instruction_dataset(
    file_path: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    split: str = "train"
) -> List[Dict[str, Any]]:
    """
    Load instruction dataset from file or HuggingFace hub.
    
    Args:
        file_path: Path to JSON file with instruction data
        dataset_name: HuggingFace dataset name
        split: Dataset split to load
    
    Returns:
        List of instruction examples
    """
    if file_path:
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "examples" in data:
            return data["examples"]
        else:
            raise ValueError(f"Invalid dataset format in {file_path}")
    
    elif dataset_name:
        try:
            dataset = load_dataset(dataset_name, split=split)
            return [example for example in dataset]
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    else:
        raise ValueError("Either file_path or dataset_name must be provided")


def validate_instruction_data(data: List[Dict[str, Any]]) -> bool:
    """
    Validate instruction dataset format.
    
    Args:
        data: List of instruction examples
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list")
    
    if len(data) == 0:
        raise ValueError("Dataset is empty")
    
    required_fields = ["instruction", "response"]
    
    for i, example in enumerate(data):
        if not isinstance(example, dict):
            raise ValueError(f"Example {i} is not a dictionary")
        
        for field in required_fields:
            if field not in example:
                raise ValueError(f"Example {i} missing required field: {field}")
            
            if not isinstance(example[field], str) or len(example[field].strip()) == 0:
                raise ValueError(f"Example {i} has invalid {field}")
    
    logger.info(f"Validated {len(data)} instruction examples")
    return True
