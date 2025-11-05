"""
Complete RLHF workflow example.

DECISION RATIONALE:
- Demonstrates full RLHF pipeline
- Shows SFT, reward model, and PPO/DPO stages
- Includes evaluation and baseline comparison
- Provides practical usage examples
- Designed for small laptop compatibility (lightweight models)

CONFIGURATION:
Edit the configuration section below to change:
- Which example to run (sft, preferences, reward, ppo, dpo, or full)
- Model name (default: gpt2 for small laptop compatibility)

ALTERNATIVE MODES:
- Individual examples: Set EXAMPLE_MODE to 'sft', 'preferences', 'reward', 'ppo', 'dpo'
- Full pipeline: Set EXAMPLE_MODE to 'full' (runs all stages sequentially)
- Custom model: Change MODEL_NAME to any HuggingFace model (larger models require more memory)
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import Config
from rlhf import (
    prepare_sft_dataset,
    SupervisedFineTuner,
    PreferenceCollector,
    train_reward_model,
    train_with_ppo,
    train_with_dpo
)
from transformers import AutoTokenizer
from llm_evaluation_demo import LLMAsJudge
from utils.baseline_tracking import BaselineTracker
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
# Edit these options to customize the example

# Example mode: 'sft', 'preferences', 'reward', 'ppo', 'dpo', or 'full'
# - 'sft': Supervised fine-tuning example
# - 'preferences': Preference collection example
# - 'reward': Reward model training example
# - 'ppo': PPO training example
# - 'dpo': DPO training example
# - 'full': Complete RLHF pipeline example
EXAMPLE_MODE = 'full'  # Change to run different examples

# Model configuration
MODEL_NAME = Config.HF_LLM_MODEL  # Default: gpt2 (small laptop compatible)
# Alternative models (require more memory):
# MODEL_NAME = 'microsoft/DialoGPT-small'  # ~120M parameters
# MODEL_NAME = 'distilgpt2'  # ~82M parameters
# MODEL_NAME = 'gpt2-medium'  # ~350M parameters (requires 4GB+ RAM)


def example_sft_training():
    """
    Example: Supervised Fine-Tuning.
    """
    print("=" * 80)
    print("EXAMPLE: SUPERVISED FINE-TUNING")
    print("=" * 80)
    
    model_name = Config.HF_LLM_MODEL
    
    instruction_data = [
        {
            "instruction": "Explain what machine learning is",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make predictions or decisions based on input data."
        },
        {
            "instruction": "What is the difference between supervised and unsupervised learning?",
            "response": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. Supervised learning requires examples with correct answers, whereas unsupervised learning discovers hidden structures."
        },
        {
            "instruction": "Describe neural networks",
            "response": "Neural networks are computing systems inspired by biological neurons. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections and activation functions."
        }
    ]
    
    print(f"\nPreparing SFT dataset with {len(instruction_data)} examples...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = prepare_sft_dataset(instruction_data, tokenizer)
    
    print("Initializing SFT trainer...")
    trainer = SupervisedFineTuner(model_name, Config.RLHF_MODELS_DIR / "sft")
    
    print("Starting SFT training...")
    metrics = trainer.train(train_dataset, num_epochs=1)  # Single epoch for demo
    
    print(f"\n[OK] SFT training completed")
    print(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Model saved to: {Config.RLHF_MODELS_DIR / 'sft'}")


def example_preference_collection():
    """
    Example: Preference Collection.
    """
    print("=" * 80)
    print("EXAMPLE: PREFERENCE COLLECTION")
    print("=" * 80)
    
    collector = PreferenceCollector()
    
    preferences = [
        {
            "prompt": "Explain what machine learning is",
            "chosen": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "rejected": "Machine learning is programming."
        },
        {
            "prompt": "What is the difference between supervised and unsupervised learning?",
            "chosen": "Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.",
            "rejected": "They are the same thing."
        }
    ]
    
    print(f"\nCollecting {len(preferences)} preferences...")
    for pref in preferences:
        collector.add_preference(
            prompt=pref["prompt"],
            response_chosen=pref["chosen"],
            response_rejected=pref["rejected"]
        )
    
    stats = collector.get_statistics()
    print(f"\nPreference statistics:")
    print(f"  Total preferences: {stats['total_preferences']}")
    print(f"  Avg prompt length: {stats['avg_prompt_length']:.1f}")
    print(f"  Avg chosen length: {stats['avg_chosen_length']:.1f}")
    print(f"  Avg rejected length: {stats['avg_rejected_length']:.1f}")
    
    file_path = collector.save_preferences()
    print(f"\n[OK] Preferences saved to: {file_path}")


def example_reward_model_training():
    """
    Example: Reward Model Training.
    """
    print("=" * 80)
    print("EXAMPLE: REWARD MODEL TRAINING")
    print("=" * 80)
    
    sft_model_path = Config.RLHF_MODELS_DIR / "sft"
    
    if not sft_model_path.exists():
        print(f"\n[FAIL] SFT model not found at: {sft_model_path}")
        print("Please run SFT training first (example_sft_training)")
        return
    
    preferences = [
        {
            "prompt": "Explain what machine learning is",
            "chosen": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "rejected": "Machine learning is programming."
        },
        {
            "prompt": "What is the difference between supervised and unsupervised learning?",
            "chosen": "Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.",
            "rejected": "They are the same thing."
        }
    ]
    
    print(f"\nTraining reward model on {len(preferences)} preferences...")
    print(f"SFT model path: {sft_model_path}")
    
    metrics = train_reward_model(
        model_name=str(sft_model_path),
        preferences=preferences,
        output_dir=Config.RLHF_MODELS_DIR / "reward_model",
        num_epochs=1  # Single epoch for demo
    )
    
    print(f"\n[OK] Reward model training completed")
    print(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Model saved to: {Config.RLHF_MODELS_DIR / 'reward_model'}")


def example_ppo_training():
    """
    Example: PPO Training.
    """
    print("=" * 80)
    print("EXAMPLE: PPO TRAINING")
    print("=" * 80)
    
    sft_model_path = Config.RLHF_MODELS_DIR / "sft"
    reward_model_path = Config.RLHF_MODELS_DIR / "reward_model"
    
    if not sft_model_path.exists():
        print(f"\n[FAIL] SFT model not found at: {sft_model_path}")
        return
    
    if not reward_model_path.exists():
        print(f"\n[FAIL] Reward model not found at: {reward_model_path}")
        return
    
    prompts = [
        "Explain what machine learning is",
        "What is the difference between supervised and unsupervised learning?",
        "Describe neural networks"
    ]
    
    print(f"\nTraining PPO on {len(prompts)} prompts...")
    print(f"SFT model path: {sft_model_path}")
    print(f"Reward model path: {reward_model_path}")
    
    metrics = train_with_ppo(
        model_name=str(sft_model_path),
        reward_model_path=reward_model_path,
        prompts=prompts,
        output_dir=Config.RLHF_MODELS_DIR / "ppo",
        num_epochs=1  # Single epoch for demo
    )
    
    print(f"\n[OK] PPO training completed")
    print(f"Total episodes: {metrics.get('total_episodes', 'N/A')}")
    print(f"Model saved to: {Config.RLHF_MODELS_DIR / 'ppo'}")


def example_dpo_training():
    """
    Example: DPO Training.
    """
    print("=" * 80)
    print("EXAMPLE: DPO TRAINING")
    print("=" * 80)
    
    sft_model_path = Config.RLHF_MODELS_DIR / "sft"
    
    if not sft_model_path.exists():
        print(f"\n[FAIL] SFT model not found at: {sft_model_path}")
        return
    
    preferences = [
        {
            "prompt": "Explain what machine learning is",
            "chosen": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "rejected": "Machine learning is programming."
        },
        {
            "prompt": "What is the difference between supervised and unsupervised learning?",
            "chosen": "Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.",
            "rejected": "They are the same thing."
        }
    ]
    
    print(f"\nTraining DPO on {len(preferences)} preferences...")
    print(f"SFT model path: {sft_model_path}")
    
    metrics = train_with_dpo(
        model_name=str(sft_model_path),
        preferences=preferences,
        output_dir=Config.RLHF_MODELS_DIR / "dpo",
        num_epochs=1  # Single epoch for demo
    )
    
    print(f"\n[OK] DPO training completed")
    print(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Model saved to: {Config.RLHF_MODELS_DIR / 'dpo'}")


def example_full_pipeline():
    """
    Example: Full RLHF Pipeline.
    """
    print("=" * 80)
    print("EXAMPLE: FULL RLHF PIPELINE")
    print("=" * 80)
    
    model_name = Config.HF_LLM_MODEL
    
    instruction_data = [
        {
            "instruction": "Explain what machine learning is",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
        }
    ]
    
    preferences = [
        {
            "prompt": "Explain what machine learning is",
            "chosen": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "rejected": "Machine learning is programming."
        }
    ]
    
    print("\nStep 1: Supervised Fine-Tuning")
    print("-" * 80)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = prepare_sft_dataset(instruction_data, tokenizer)
    trainer = SupervisedFineTuner(model_name, Config.RLHF_MODELS_DIR / "sft")
    sft_metrics = trainer.train(train_dataset, num_epochs=1)
    print(f"[OK] SFT completed: loss={sft_metrics.get('train_loss', 'N/A')}")
    
    print("\nStep 2: Reward Model Training")
    print("-" * 80)
    reward_metrics = train_reward_model(
        model_name=str(Config.RLHF_MODELS_DIR / "sft"),
        preferences=preferences,
        output_dir=Config.RLHF_MODELS_DIR / "reward_model",
        num_epochs=1
    )
    print(f"[OK] Reward model completed: loss={reward_metrics.get('train_loss', 'N/A')}")
    
    print("\nStep 3: PPO Training")
    print("-" * 80)
    prompts = [p["prompt"] for p in preferences]
    ppo_metrics = train_with_ppo(
        model_name=str(Config.RLHF_MODELS_DIR / "sft"),
        reward_model_path=Config.RLHF_MODELS_DIR / "reward_model",
        prompts=prompts,
        output_dir=Config.RLHF_MODELS_DIR / "ppo",
        num_epochs=1
    )
    print(f"[OK] PPO completed: episodes={ppo_metrics.get('total_episodes', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("[OK] FULL PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Final model: {Config.RLHF_MODELS_DIR / 'ppo'}")


if __name__ == "__main__":
    """
    Execute example based on configuration.
    
    Edit EXAMPLE_MODE in the configuration section to change which example runs.
    """
    if EXAMPLE_MODE == "sft":
        example_sft_training()
    elif EXAMPLE_MODE == "preferences":
        example_preference_collection()
    elif EXAMPLE_MODE == "reward":
        example_reward_model_training()
    elif EXAMPLE_MODE == "ppo":
        example_ppo_training()
    elif EXAMPLE_MODE == "dpo":
        example_dpo_training()
    elif EXAMPLE_MODE == "full":
        example_full_pipeline()
    else:
        print("Available examples:")
        print("  Set EXAMPLE_MODE to 'sft'          : Supervised Fine-Tuning")
        print("  Set EXAMPLE_MODE to 'preferences'  : Preference Collection")
        print("  Set EXAMPLE_MODE to 'reward'       : Reward Model Training")
        print("  Set EXAMPLE_MODE to 'ppo'          : PPO Training")
        print("  Set EXAMPLE_MODE to 'dpo'          : DPO Training")
        print("  Set EXAMPLE_MODE to 'full'         : Full RLHF Pipeline")
        print(f"\nCurrent EXAMPLE_MODE: {EXAMPLE_MODE}")
        print("Edit EXAMPLE_MODE in the configuration section to change")
