#!/usr/bin/env python3
"""
End-to-end RLHF pipeline script.

DECISION RATIONALE:
- Orchestrates complete RLHF pipeline (SFT -> Reward Model -> PPO/DPO)
- Integrates with evaluation framework
- Provides comprehensive reporting
- Demonstrates full RLHF workflow
- Designed to run on small laptops (lightweight models, memory efficient)

CONFIGURATION:
Edit the configuration section below to change:
- Model name (default: gpt2 for small laptop compatibility)
- Pipeline stage (sft, reward, ppo, dpo, or full)
- Data paths
- Algorithm (ppo or dpo)

ALTERNATIVE MODES:
- Stage-by-stage: Set PIPELINE_STAGE to 'sft', 'reward', 'ppo', or 'dpo'
- Full pipeline: Set PIPELINE_STAGE to 'full' (runs all stages sequentially)
- Custom model: Change MODEL_NAME to any HuggingFace model (larger models require more memory)
- Algorithm selection: Set RLHF_ALGORITHM to 'ppo' (requires reward model) or 'dpo' (direct optimization)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import Config
from rlhf import (
    prepare_sft_dataset,
    load_instruction_dataset,
    validate_instruction_data,
    SupervisedFineTuner,
    PreferenceCollector,
    load_preference_data,
    RewardModel,
    train_reward_model,
    PPOTrainerWrapper,
    train_with_ppo,
    DPOTrainerWrapper,
    train_with_dpo
)
from transformers import AutoTokenizer
from llm_evaluation_demo import LLMAsJudge
from utils.baseline_tracking import BaselineTracker

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Edit these options to customize the pipeline

# Pipeline stage: 'sft', 'reward', 'ppo', 'dpo', or 'full'
# - 'sft': Run supervised fine-tuning only
# - 'reward': Train reward model (requires SFT model)
# - 'ppo': Run PPO training (requires SFT and reward models)
# - 'dpo': Run DPO training (requires SFT model and preferences)
# - 'full': Run complete pipeline (SFT -> Reward/DPO -> PPO/DPO)
PIPELINE_STAGE = 'sft'  # Change to 'sft', 'reward', 'ppo', 'dpo', or 'full'
# For testing on small laptop, use 1 epoch and small batch size
# Set num_epochs=1 in stage_sft() call for quick testing

# Model configuration
MODEL_NAME = Config.HF_LLM_MODEL  # Default: gpt2 (small laptop compatible)
# Alternative models (require more memory):
# MODEL_NAME = 'microsoft/DialoGPT-small'  # ~120M parameters
# MODEL_NAME = 'distilgpt2'  # ~82M parameters
# MODEL_NAME = 'gpt2-medium'  # ~350M parameters (requires 4GB+ RAM)

# RLHF algorithm: 'ppo' or 'dpo'
# - 'ppo': Requires reward model training (more complex, better for complex tasks)
# - 'dpo': Direct optimization on preferences (simpler, faster, no reward model needed)
RLHF_ALGORITHM = 'ppo'  # Change to 'dpo' for direct preference optimization

# Data paths
DATA_PATH = Config.RLHF_DATA_DIR / "instructions.json"  # Instruction dataset for SFT
PREFERENCES_PATH = Config.RLHF_PREFERENCES_DIR / "preferences.json"  # Preference data
PROMPTS_PATH = Config.RLHF_DATA_DIR / "prompts.json"  # Test prompts for evaluation

# Output directory
OUTPUT_DIR = Config.RLHF_OUTPUT_DIR  # Pipeline output directory

# Model paths (for stage-by-stage execution)
SFT_MODEL_PATH = Config.RLHF_MODELS_DIR / "sft"  # Path to SFT model
REWARD_MODEL_PATH = Config.RLHF_MODELS_DIR / "reward_model"  # Path to reward model

# Baseline tracking
SAVE_BASELINE = None  # Set to string name to save baseline (e.g., 'v1.0')
COMPARE_BASELINE = None  # Set to baseline name to compare (e.g., 'v1.0')

# ============================================================================


def stage_sft(
    model_name: str,
    data_path: Path,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Stage 1: Supervised Fine-Tuning.
    
    Args:
        model_name: Base model name
        data_path: Path to instruction dataset
        output_dir: Output directory
        **kwargs: Additional training arguments
    
    Returns:
        Training metrics and model path
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: SUPERVISED FINE-TUNING")
    logger.info("=" * 80)
    
    output_dir = output_dir or Config.RLHF_MODELS_DIR / "sft"
    
    logger.info(f"Loading instruction dataset from: {data_path}")
    instruction_data = load_instruction_dataset(file_path=data_path)
    validate_instruction_data(instruction_data)
    
    logger.info(f"Preparing SFT dataset with {len(instruction_data)} examples")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = prepare_sft_dataset(instruction_data, tokenizer, Config.SFT_MAX_SEQ_LENGTH)
    
    logger.info("Initializing SFT trainer")
    trainer = SupervisedFineTuner(model_name, output_dir)
    
    logger.info("Starting SFT training...")
    # For small laptop testing: use 1 epoch, small batch size
    metrics = trainer.train(train_dataset, num_epochs=1, batch_size=1, gradient_accumulation_steps=8, **kwargs)
    
    logger.info("Saving SFT model...")
    trainer.save_model()
    
    logger.info(f"[OK] SFT training completed. Model saved to: {output_dir}")
    logger.info(f"Training metrics: {metrics}")
    
    return {
        "stage": "sft",
        "model_path": str(output_dir),
        "metrics": metrics
    }


def stage_reward_model(
    sft_model_path: Path,
    preferences_path: Path,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Stage 2: Reward Model Training.
    
    Args:
        sft_model_path: Path to SFT model
        preferences_path: Path to preference data
        output_dir: Output directory
        **kwargs: Additional training arguments
    
    Returns:
        Training metrics and model path
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: REWARD MODEL TRAINING")
    logger.info("=" * 80)
    
    output_dir = output_dir or Config.RLHF_MODELS_DIR / "reward_model"
    
    logger.info(f"Loading preferences from: {preferences_path}")
    preferences = load_preference_data(preferences_path)
    
    if len(preferences) < Config.RLHF_MIN_PREFERENCES:
        logger.warning(f"Only {len(preferences)} preferences found. Minimum recommended: {Config.RLHF_MIN_PREFERENCES}")
    
    logger.info(f"Training reward model on {len(preferences)} preferences")
    logger.info(f"SFT model path: {sft_model_path}")
    
    metrics = train_reward_model(
        model_name=str(sft_model_path),
        preferences=preferences,
        output_dir=output_dir,
        **kwargs
    )
    
    logger.info(f"[OK] Reward model training completed. Model saved to: {output_dir}")
    logger.info(f"Training metrics: {metrics}")
    
    return {
        "stage": "reward_model",
        "model_path": str(output_dir),
        "metrics": metrics
    }


def stage_ppo(
    sft_model_path: Path,
    reward_model_path: Path,
    prompts: List[str],
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Stage 3: PPO Training.
    
    Args:
        sft_model_path: Path to SFT model
        reward_model_path: Path to reward model
        prompts: List of training prompts
        output_dir: Output directory
        **kwargs: Additional training arguments
    
    Returns:
        Training metrics and model path
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: PPO TRAINING")
    logger.info("=" * 80)
    
    output_dir = output_dir or Config.RLHF_MODELS_DIR / "ppo"
    
    logger.info(f"Training PPO on {len(prompts)} prompts")
    logger.info(f"SFT model path: {sft_model_path}")
    logger.info(f"Reward model path: {reward_model_path}")
    
    metrics = train_with_ppo(
        model_name=str(sft_model_path),
        reward_model_path=reward_model_path,
        prompts=prompts,
        output_dir=output_dir,
        **kwargs
    )
    
    logger.info(f"[OK] PPO training completed. Model saved to: {output_dir}")
    logger.info(f"Training metrics: {metrics}")
    
    return {
        "stage": "ppo",
        "model_path": str(output_dir),
        "metrics": metrics
    }


def stage_dpo(
    sft_model_path: Path,
    preferences_path: Path,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Stage 4: DPO Training.
    
    Args:
        sft_model_path: Path to SFT model
        preferences_path: Path to preference data
        output_dir: Output directory
        **kwargs: Additional training arguments
    
    Returns:
        Training metrics and model path
    """
    logger.info("=" * 80)
    logger.info("STAGE 4: DPO TRAINING")
    logger.info("=" * 80)
    
    output_dir = output_dir or Config.RLHF_MODELS_DIR / "dpo"
    
    logger.info(f"Loading preferences from: {preferences_path}")
    preferences = load_preference_data(preferences_path)
    
    if len(preferences) < Config.RLHF_MIN_PREFERENCES:
        logger.warning(f"Only {len(preferences)} preferences found. Minimum recommended: {Config.RLHF_MIN_PREFERENCES}")
    
    logger.info(f"Training DPO on {len(preferences)} preferences")
    logger.info(f"SFT model path: {sft_model_path}")
    
    metrics = train_with_dpo(
        model_name=str(sft_model_path),
        preferences=preferences,
        output_dir=output_dir,
        **kwargs
    )
    
    logger.info(f"[OK] DPO training completed. Model saved to: {output_dir}")
    logger.info(f"Training metrics: {metrics}")
    
    return {
        "stage": "dpo",
        "model_path": str(output_dir),
        "metrics": metrics
    }


def evaluate_model(
    model_path: Path,
    test_prompts: List[str],
    llm_as_judge: Optional[LLMAsJudge] = None
) -> Dict[str, Any]:
    """
    Evaluate trained model.
    
    Args:
        model_path: Path to trained model
        test_prompts: List of test prompts
        llm_as_judge: Optional LLM-as-Judge evaluator
    
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating model...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    results = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=Config.SFT_MAX_SEQ_LENGTH,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        result = {
            "prompt": prompt,
            "response": response
        }
        
        if llm_as_judge:
            judgment = llm_as_judge.judge(prompt, response)
            result["judgment"] = judgment
        
        results.append(result)
    
    return {
        "total_prompts": len(test_prompts),
        "results": results
    }


def run_full_pipeline(
    model_name: str,
    data_path: Path,
    preferences_path: Path,
    test_prompts: Optional[List[str]] = None,
    algorithm: str = "ppo",
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run complete RLHF pipeline.
    
    Args:
        model_name: Base model name
        data_path: Path to instruction dataset
        preferences_path: Path to preference data
        test_prompts: Optional test prompts for evaluation
        algorithm: RLHF algorithm (ppo or dpo)
        output_dir: Output directory
    
    Returns:
        Complete pipeline results
    """
    logger.info("=" * 80)
    logger.info("FULL RLHF PIPELINE")
    logger.info("=" * 80)
    
    pipeline_results = {
        "started_at": datetime.now().isoformat(),
        "model_name": model_name,
        "algorithm": algorithm,
        "stages": {}
    }
    
    output_dir = output_dir or Config.RLHF_OUTPUT_DIR
    
    logger.info("Stage 1: Supervised Fine-Tuning")
    sft_results = stage_sft(model_name, data_path, output_dir / "sft")
    pipeline_results["stages"]["sft"] = sft_results
    
    sft_model_path = Path(sft_results["model_path"])
    
    if algorithm == "ppo":
        logger.info("Stage 2: Reward Model Training")
        reward_results = stage_reward_model(
            sft_model_path,
            preferences_path,
            output_dir / "reward_model"
        )
        pipeline_results["stages"]["reward_model"] = reward_results
        
        reward_model_path = Path(reward_results["model_path"])
        
        logger.info("Stage 3: PPO Training")
        prompts = [p["prompt"] for p in load_preference_data(preferences_path)[:Config.RLHF_PREFERENCE_BATCH_SIZE]]
        ppo_results = stage_ppo(
            sft_model_path,
            reward_model_path,
            prompts,
            output_dir / "ppo"
        )
        pipeline_results["stages"]["ppo"] = ppo_results
        final_model_path = Path(ppo_results["model_path"])
    
    else:
        logger.info("Stage 2: DPO Training")
        dpo_results = stage_dpo(
            sft_model_path,
            preferences_path,
            output_dir / "dpo"
        )
        pipeline_results["stages"]["dpo"] = dpo_results
        final_model_path = Path(dpo_results["model_path"])
    
    if test_prompts:
        logger.info("Evaluating final model...")
        llm_as_judge = LLMAsJudge()
        evaluation_results = evaluate_model(final_model_path, test_prompts, llm_as_judge)
        pipeline_results["evaluation"] = evaluation_results
    
    pipeline_results["completed_at"] = datetime.now().isoformat()
    pipeline_results["final_model_path"] = str(final_model_path)
    
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Final model: {final_model_path}")
    
    return pipeline_results


def main():
    """
    Main entry point.
    
    Executes the RLHF pipeline based on configuration options.
    Edit the configuration section at the top of this file to customize.
    """
    Config.create_directories()
    
    logger.info("=" * 80)
    logger.info("RLHF PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Stage: {PIPELINE_STAGE}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Algorithm: {RLHF_ALGORITHM}")
    logger.info("=" * 80)
    
    if PIPELINE_STAGE == 'full':
        if not DATA_PATH.exists():
            logger.error(f"Instruction dataset not found: {DATA_PATH}")
            logger.error("Please create the instruction dataset file or update DATA_PATH")
            return
        
        if not PREFERENCES_PATH.exists():
            logger.error(f"Preference data not found: {PREFERENCES_PATH}")
            logger.error("Please create the preference data file or update PREFERENCES_PATH")
            return
        
        test_prompts = None
        if PROMPTS_PATH.exists():
            with open(PROMPTS_PATH, 'r') as f:
                test_prompts = json.load(f)
        
        results = run_full_pipeline(
            model_name=MODEL_NAME,
            data_path=DATA_PATH,
            preferences_path=PREFERENCES_PATH,
            test_prompts=test_prompts,
            algorithm=RLHF_ALGORITHM,
            output_dir=OUTPUT_DIR
        )
        
        output_file = OUTPUT_DIR / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"[OK] Pipeline results saved to: {output_file}")
        
        if SAVE_BASELINE:
            baseline_tracker = BaselineTracker()
            baseline_tracker.save_baseline(SAVE_BASELINE, results, {"stage": "full_pipeline"})
            logger.info(f"[OK] Baseline saved: {SAVE_BASELINE}")
        
        if COMPARE_BASELINE:
            baseline_tracker = BaselineTracker()
            improvements = baseline_tracker.calculate_improvements(results, COMPARE_BASELINE)
            report = baseline_tracker.generate_improvement_report(improvements)
            logger.info("\n" + report)
    
    elif PIPELINE_STAGE == 'sft':
        if not DATA_PATH.exists():
            logger.error(f"Instruction dataset not found: {DATA_PATH}")
            logger.error("Please create the instruction dataset file or update DATA_PATH")
            return
        
        results = stage_sft(
            model_name=MODEL_NAME,
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR / "sft"
        )
        
        logger.info(f"[OK] SFT results: {results}")
    
    elif PIPELINE_STAGE == 'reward':
        if not SFT_MODEL_PATH.exists():
            logger.error(f"SFT model not found: {SFT_MODEL_PATH}")
            logger.error("Please run SFT stage first or update SFT_MODEL_PATH")
            return
        
        if not PREFERENCES_PATH.exists():
            logger.error(f"Preference data not found: {PREFERENCES_PATH}")
            logger.error("Please create the preference data file or update PREFERENCES_PATH")
            return
        
        results = stage_reward_model(
            sft_model_path=SFT_MODEL_PATH,
            preferences_path=PREFERENCES_PATH,
            output_dir=OUTPUT_DIR / "reward_model"
        )
        
        logger.info(f"[OK] Reward model results: {results}")
    
    elif PIPELINE_STAGE == 'ppo':
        if not SFT_MODEL_PATH.exists():
            logger.error(f"SFT model not found: {SFT_MODEL_PATH}")
            logger.error("Please run SFT stage first or update SFT_MODEL_PATH")
            return
        
        if not REWARD_MODEL_PATH.exists():
            logger.error(f"Reward model not found: {REWARD_MODEL_PATH}")
            logger.error("Please run reward model stage first or update REWARD_MODEL_PATH")
            return
        
        if not PROMPTS_PATH.exists():
            logger.error(f"Prompts file not found: {PROMPTS_PATH}")
            logger.error("Please create the prompts file or update PROMPTS_PATH")
            return
        
        with open(PROMPTS_PATH, 'r') as f:
            prompts = json.load(f)
        
        results = stage_ppo(
            sft_model_path=SFT_MODEL_PATH,
            reward_model_path=REWARD_MODEL_PATH,
            prompts=prompts,
            output_dir=OUTPUT_DIR / "ppo"
        )
        
        logger.info(f"[OK] PPO results: {results}")
    
    elif PIPELINE_STAGE == 'dpo':
        if not SFT_MODEL_PATH.exists():
            logger.error(f"SFT model not found: {SFT_MODEL_PATH}")
            logger.error("Please run SFT stage first or update SFT_MODEL_PATH")
            return
        
        if not PREFERENCES_PATH.exists():
            logger.error(f"Preference data not found: {PREFERENCES_PATH}")
            logger.error("Please create the preference data file or update PREFERENCES_PATH")
            return
        
        results = stage_dpo(
            sft_model_path=SFT_MODEL_PATH,
            preferences_path=PREFERENCES_PATH,
            output_dir=OUTPUT_DIR / "dpo"
        )
        
        logger.info(f"[OK] DPO results: {results}")
    
    else:
        logger.error(f"Invalid pipeline stage: {PIPELINE_STAGE}")
        logger.error("Valid stages: 'sft', 'reward', 'ppo', 'dpo', or 'full'")
        logger.error("Edit PIPELINE_STAGE in the configuration section to change")


if __name__ == "__main__":
    main()
