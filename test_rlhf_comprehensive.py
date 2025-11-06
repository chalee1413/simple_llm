#!/usr/bin/env python3
"""
Comprehensive RLHF testing script.

Tests all RLHF components and evaluates effectiveness of results.

DECISION RATIONALE:
- Tests SFT, reward model, PPO, DPO, and KTO
- Evaluates training effectiveness
- Generates comprehensive reports
- Validates all components work correctly

References:
- RLHF pipeline components
- Evaluation metrics from rlhf.evaluation_metrics
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Project imports
from config import Config
from rlhf import (
    prepare_sft_dataset,
    load_instruction_dataset,
    validate_instruction_data,
    SupervisedFineTuner,
    load_preference_data,
    train_reward_model,
    train_with_ppo,
    train_with_dpo,
    train_with_kto,
    evaluate_reward_model,
    RewardModelEvaluator,
    TrainingMetricsTracker
)
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLHFTestSuite:
    """
    Comprehensive RLHF testing suite.
    
    Tests all components and evaluates effectiveness.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize test suite.
        
        Args:
            output_dir: Output directory for test results
        """
        self.output_dir = output_dir or Config.RLHF_OUTPUT_DIR / "test_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "started_at": datetime.now().isoformat(),
            "tests": {}
        }
        
        logger.info(f"RLHF Test Suite initialized: {self.output_dir}")
    
    def test_sft(self) -> Dict[str, Any]:
        """
        Test Supervised Fine-Tuning.
        
        Returns:
            Test results dictionary
        """
        logger.info("=" * 80)
        logger.info("TEST: SUPERVISED FINE-TUNING")
        logger.info("=" * 80)
        
        try:
            model_name = Config.HF_LLM_MODEL
            data_path = Config.RLHF_DATA_DIR / "instructions.json"
            
            if not data_path.exists():
                raise FileNotFoundError(f"Instruction data not found: {data_path}")
            
            logger.info(f"Loading instruction data from: {data_path}")
            instruction_data = load_instruction_dataset(data_path)
            validate_instruction_data(instruction_data)
            
            logger.info(f"Preparing SFT dataset with {len(instruction_data)} examples")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            train_dataset = prepare_sft_dataset(instruction_data, tokenizer, Config.SFT_MAX_SEQ_LENGTH)
            
            output_dir = Config.RLHF_MODELS_DIR / "sft"
            trainer = SupervisedFineTuner(model_name, output_dir)
            
            logger.info("Starting SFT training...")
            metrics = trainer.train(
                train_dataset,
                num_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=8
            )
            
            trainer.save_model()
            
            result = {
                "status": "success",
                "model_path": str(output_dir),
                "metrics": metrics,
                "num_examples": len(instruction_data)
            }
            
            logger.info(f"[OK] SFT test completed: {metrics.get('train_loss', 'N/A')}")
            return result
        
        except Exception as e:
            logger.error(f"[FAIL] SFT test failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_reward_model(self, sft_model_path: Path) -> Dict[str, Any]:
        """
        Test Reward Model Training.
        
        Args:
            sft_model_path: Path to SFT model
        
        Returns:
            Test results dictionary
        """
        logger.info("=" * 80)
        logger.info("TEST: REWARD MODEL TRAINING")
        logger.info("=" * 80)
        
        try:
            preferences_path = Config.RLHF_PREFERENCES_DIR / "preferences.json"
            
            if not preferences_path.exists():
                raise FileNotFoundError(f"Preferences not found: {preferences_path}")
            
            if not sft_model_path.exists():
                raise FileNotFoundError(f"SFT model not found: {sft_model_path}")
            
            logger.info(f"Loading preferences from: {preferences_path}")
            preferences = load_preference_data(preferences_path)
            
            logger.info(f"Training reward model on {len(preferences)} preferences")
            output_dir = Config.RLHF_MODELS_DIR / "reward_model"
            
            metrics = train_reward_model(
                model_name=str(sft_model_path),
                preferences=preferences,
                output_dir=output_dir,
                num_epochs=1
            )
            
            logger.info("Evaluating reward model...")
            from rlhf.reward_model import RewardModel
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
            reward_model = RewardModel(str(sft_model_path))
            reward_model.load(output_dir)
            
            eval_metrics = evaluate_reward_model(reward_model, tokenizer, preferences)
            
            result = {
                "status": "success",
                "model_path": str(output_dir),
                "training_metrics": metrics,
                "evaluation_metrics": eval_metrics,
                "num_preferences": len(preferences)
            }
            
            logger.info(f"[OK] Reward model test completed")
            logger.info(f"  Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
            logger.info(f"  Ranking accuracy: {eval_metrics.get('ranking_accuracy', 0):.4f}")
            
            return result
        
        except Exception as e:
            logger.error(f"[FAIL] Reward model test failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_ppo(self, sft_model_path: Path, reward_model_path: Path) -> Dict[str, Any]:
        """
        Test PPO Training.
        
        Args:
            sft_model_path: Path to SFT model
            reward_model_path: Path to reward model
        
        Returns:
            Test results dictionary
        """
        logger.info("=" * 80)
        logger.info("TEST: PPO TRAINING")
        logger.info("=" * 80)
        
        try:
            prompts_path = Config.RLHF_DATA_DIR / "prompts.json"
            
            if not prompts_path.exists():
                raise FileNotFoundError(f"Prompts not found: {prompts_path}")
            
            with open(prompts_path, 'r') as f:
                prompts = json.load(f)
            
            logger.info(f"Training PPO on {len(prompts)} prompts")
            output_dir = Config.RLHF_MODELS_DIR / "ppo"
            
            metrics = train_with_ppo(
                model_name=str(sft_model_path),
                reward_model_path=reward_model_path,
                prompts=prompts,
                output_dir=output_dir,
                num_epochs=1
            )
            
            result = {
                "status": "success",
                "model_path": str(output_dir),
                "metrics": metrics,
                "num_prompts": len(prompts)
            }
            
            logger.info(f"[OK] PPO test completed: {metrics.get('total_episodes', 'N/A')} episodes")
            return result
        
        except Exception as e:
            logger.error(f"[FAIL] PPO test failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_dpo(self, sft_model_path: Path) -> Dict[str, Any]:
        """
        Test DPO Training.
        
        Args:
            sft_model_path: Path to SFT model
        
        Returns:
            Test results dictionary
        """
        logger.info("=" * 80)
        logger.info("TEST: DPO TRAINING")
        logger.info("=" * 80)
        
        try:
            preferences_path = Config.RLHF_PREFERENCES_DIR / "preferences.json"
            
            if not preferences_path.exists():
                raise FileNotFoundError(f"Preferences not found: {preferences_path}")
            
            if not sft_model_path.exists():
                raise FileNotFoundError(f"SFT model not found: {sft_model_path}")
            
            logger.info(f"Loading preferences from: {preferences_path}")
            preferences = load_preference_data(preferences_path)
            
            logger.info(f"Training DPO on {len(preferences)} preferences")
            output_dir = Config.RLHF_MODELS_DIR / "dpo"
            
            metrics = train_with_dpo(
                model_name=str(sft_model_path),
                preferences=preferences,
                output_dir=output_dir,
                num_epochs=1
            )
            
            result = {
                "status": "success",
                "model_path": str(output_dir),
                "metrics": metrics,
                "num_preferences": len(preferences)
            }
            
            logger.info(f"[OK] DPO test completed: {metrics.get('train_loss', 'N/A')}")
            return result
        
        except Exception as e:
            logger.error(f"[FAIL] DPO test failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_kto(self, sft_model_path: Path) -> Dict[str, Any]:
        """
        Test KTO Training.
        
        Args:
            sft_model_path: Path to SFT model
        
        Returns:
            Test results dictionary
        """
        logger.info("=" * 80)
        logger.info("TEST: KTO TRAINING")
        logger.info("=" * 80)
        
        try:
            feedback_path = Config.RLHF_DATA_DIR / "kto_feedback.json"
            
            if not feedback_path.exists():
                raise FileNotFoundError(f"KTO feedback not found: {feedback_path}")
            
            if not sft_model_path.exists():
                raise FileNotFoundError(f"SFT model not found: {sft_model_path}")
            
            with open(feedback_path, 'r') as f:
                feedback_data = json.load(f)
            
            logger.info(f"Training KTO on {len(feedback_data)} feedback examples")
            output_dir = Config.RLHF_MODELS_DIR / "kto"
            
            metrics = train_with_kto(
                model_name=str(sft_model_path),
                feedback_data=feedback_data,
                output_dir=output_dir,
                num_epochs=1
            )
            
            result = {
                "status": "success",
                "model_path": str(output_dir),
                "metrics": metrics,
                "num_feedback": len(feedback_data)
            }
            
            logger.info(f"[OK] KTO test completed: {metrics.get('train_loss', 'N/A')}")
            return result
        
        except Exception as e:
            logger.error(f"[FAIL] KTO test failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def evaluate_model_quality(self, model_path: Path, test_prompts: List[str]) -> Dict[str, Any]:
        """
        Evaluate trained model quality.
        
        Args:
            model_path: Path to trained model
            test_prompts: List of test prompts
        
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model quality...")
        
        try:
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
                results.append({
                    "prompt": prompt,
                    "response": response
                })
            
            return {
                "status": "success",
                "results": results,
                "num_prompts": len(test_prompts)
            }
        
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all RLHF tests.
        
        Returns:
            Complete test results
        """
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE RLHF TEST SUITE")
        logger.info("=" * 80)
        
        Config.create_directories()
        
        sft_result = self.test_sft()
        self.results["tests"]["sft"] = sft_result
        
        if sft_result.get("status") == "success":
            sft_model_path = Path(sft_result["model_path"])
            
            reward_result = self.test_reward_model(sft_model_path)
            self.results["tests"]["reward_model"] = reward_result
            
            if reward_result.get("status") == "success":
                reward_model_path = Path(reward_result["model_path"])
                
                ppo_result = self.test_ppo(sft_model_path, reward_model_path)
                self.results["tests"]["ppo"] = ppo_result
            
            dpo_result = self.test_dpo(sft_model_path)
            self.results["tests"]["dpo"] = dpo_result
            
            kto_result = self.test_kto(sft_model_path)
            self.results["tests"]["kto"] = kto_result
        
        self.results["completed_at"] = datetime.now().isoformat()
        
        output_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results saved to: {output_file}")
        
        return self.results
    
    def generate_report(self) -> str:
        """
        Generate test report.
        
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("RLHF COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        for test_name, result in self.results.get("tests", {}).items():
            report.append(f"Test: {test_name.upper()}")
            report.append("-" * 80)
            
            if result.get("status") == "success":
                report.append("Status: SUCCESS")
                if "metrics" in result:
                    report.append(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
                if "evaluation_metrics" in result:
                    eval_metrics = result["evaluation_metrics"]
                    report.append(f"Evaluation Metrics:")
                    report.append(f"  Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
                    report.append(f"  Ranking Accuracy: {eval_metrics.get('ranking_accuracy', 0):.4f}")
                    report.append(f"  Mean Reward Difference: {eval_metrics.get('mean_reward_diff', 0):.4f}")
            else:
                report.append(f"Status: FAILED")
                report.append(f"Error: {result.get('error', 'Unknown error')}")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    suite = RLHFTestSuite()
    results = suite.run_all_tests()
    
    report = suite.generate_report()
    print(report)
    
    output_file = suite.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")
    
    success_count = sum(1 for r in results.get("tests", {}).values() if r.get("status") == "success")
    total_count = len(results.get("tests", {}))
    
    print(f"\nSummary: {success_count}/{total_count} tests passed")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())

