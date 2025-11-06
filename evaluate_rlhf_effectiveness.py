#!/usr/bin/env python3
"""
RLHF Effectiveness Evaluation Script.

Analyzes training results and evaluates effectiveness of RLHF components.

DECISION RATIONALE:
- Evaluates model quality improvements
- Analyzes training metrics
- Compares different algorithms
- Generates effectiveness reports

References:
- Evaluation metrics from rlhf.evaluation_metrics
- Statistical testing utilities
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import statistics

from config import Config
from rlhf import evaluate_reward_model, RewardModelEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLHFEffectivenessEvaluator:
    """
    Evaluates effectiveness of RLHF training.
    
    Analyzes training results and model quality.
    """
    
    def __init__(self, results_dir: Path = None):
        """
        Initialize evaluator.
        
        Args:
            results_dir: Directory containing test results
        """
        self.results_dir = results_dir or Config.RLHF_OUTPUT_DIR / "test_results"
        logger.info(f"RLHF Effectiveness Evaluator initialized: {self.results_dir}")
    
    def evaluate_sft_effectiveness(self, model_path: Path, test_prompts: List[str]) -> Dict[str, Any]:
        """
        Evaluate SFT model effectiveness.
        
        Args:
            model_path: Path to SFT model
            test_prompts: List of test prompts
        
        Returns:
            Effectiveness metrics
        """
        logger.info("Evaluating SFT model effectiveness...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            responses = []
            response_lengths = []
            
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(
                    **inputs,
                    max_length=Config.SFT_MAX_SEQ_LENGTH,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
                response_lengths.append(len(response))
            
            avg_length = statistics.mean(response_lengths) if response_lengths else 0
            std_length = statistics.stdev(response_lengths) if len(response_lengths) > 1 else 0
            
            metrics = {
                "status": "success",
                "num_responses": len(responses),
                "avg_response_length": avg_length,
                "std_response_length": std_length,
                "responses": responses
            }
            
            logger.info(f"SFT effectiveness: {len(responses)} responses, avg length: {avg_length:.1f}")
            return metrics
        
        except Exception as e:
            logger.error(f"SFT evaluation failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def evaluate_reward_model_effectiveness(self, reward_model_path: Path, sft_model_path: Path, preferences: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate reward model effectiveness.
        
        Args:
            reward_model_path: Path to reward model
            sft_model_path: Path to SFT model
            preferences: List of preference dictionaries
        
        Returns:
            Effectiveness metrics
        """
        logger.info("Evaluating reward model effectiveness...")
        
        try:
            from rlhf.reward_model import RewardModel
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
            reward_model = RewardModel(str(sft_model_path))
            reward_model.load(reward_model_path)
            
            eval_metrics = evaluate_reward_model(reward_model, tokenizer, preferences)
            
            effectiveness = {
                "status": "success",
                "accuracy": eval_metrics.get("accuracy", 0),
                "ranking_accuracy": eval_metrics.get("ranking_accuracy", 0),
                "mean_reward_diff": eval_metrics.get("mean_reward_diff", 0),
                "correlation": eval_metrics.get("correlation", 0),
                "metrics": eval_metrics
            }
            
            logger.info(f"Reward model effectiveness:")
            logger.info(f"  Accuracy: {effectiveness['accuracy']:.4f}")
            logger.info(f"  Ranking Accuracy: {effectiveness['ranking_accuracy']:.4f}")
            
            return effectiveness
        
        except Exception as e:
            logger.error(f"Reward model evaluation failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def compare_algorithms(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare effectiveness of different algorithms.
        
        Args:
            test_results: Test results dictionary
        
        Returns:
            Comparison metrics
        """
        logger.info("Comparing algorithm effectiveness...")
        
        comparison = {
            "sft": {},
            "ppo": {},
            "dpo": {},
            "kto": {},
            "reward_model": {}
        }
        
        if "sft" in test_results.get("tests", {}):
            sft_result = test_results["tests"]["sft"]
            if sft_result.get("status") == "success":
                comparison["sft"] = {
                    "status": "success",
                    "loss": sft_result.get("metrics", {}).get("train_loss", None),
                    "num_examples": sft_result.get("num_examples", 0)
                }
        
        if "ppo" in test_results.get("tests", {}):
            ppo_result = test_results["tests"]["ppo"]
            if ppo_result.get("status") == "success":
                comparison["ppo"] = {
                    "status": "success",
                    "episodes": ppo_result.get("metrics", {}).get("total_episodes", None),
                    "num_prompts": ppo_result.get("num_prompts", 0)
                }
        
        if "dpo" in test_results.get("tests", {}):
            dpo_result = test_results["tests"]["dpo"]
            if dpo_result.get("status") == "success":
                comparison["dpo"] = {
                    "status": "success",
                    "loss": dpo_result.get("metrics", {}).get("train_loss", None),
                    "num_preferences": dpo_result.get("num_preferences", 0)
                }
        
        if "kto" in test_results.get("tests", {}):
            kto_result = test_results["tests"]["kto"]
            if kto_result.get("status") == "success":
                comparison["kto"] = {
                    "status": "success",
                    "loss": kto_result.get("metrics", {}).get("train_loss", None),
                    "num_feedback": kto_result.get("num_feedback", 0)
                }
        
        if "reward_model" in test_results.get("tests", {}):
            reward_result = test_results["tests"]["reward_model"]
            if reward_result.get("status") == "success":
                comparison["reward_model"] = {
                    "status": "success",
                    "accuracy": reward_result.get("evaluation_metrics", {}).get("accuracy", None),
                    "ranking_accuracy": reward_result.get("evaluation_metrics", {}).get("ranking_accuracy", None)
                }
        
        return comparison
    
    def generate_effectiveness_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate effectiveness report.
        
        Args:
            test_results: Test results dictionary
        
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("RLHF EFFECTIVENESS EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        comparison = self.compare_algorithms(test_results)
        
        report.append("Algorithm Effectiveness Summary:")
        report.append("-" * 80)
        
        for algo_name, algo_data in comparison.items():
            if algo_data.get("status") == "success":
                report.append(f"{algo_name.upper()}:")
                for key, value in algo_data.items():
                    if key != "status":
                        report.append(f"  {key}: {value}")
                report.append("")
        
        report.append("Training Metrics:")
        report.append("-" * 80)
        
        for test_name, result in test_results.get("tests", {}).items():
            if result.get("status") == "success":
                report.append(f"{test_name.upper()}:")
                if "metrics" in result:
                    for key, value in result["metrics"].items():
                        report.append(f"  {key}: {value}")
                if "evaluation_metrics" in result:
                    report.append("  Evaluation Metrics:")
                    for key, value in result["evaluation_metrics"].items():
                        if isinstance(value, (int, float)):
                            report.append(f"    {key}: {value:.4f}")
                report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main entry point."""
    import sys
    
    results_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not results_file:
        results_dir = Config.RLHF_OUTPUT_DIR / "test_results"
        result_files = list(results_dir.glob("test_results_*.json"))
        if not result_files:
            print("No test results found. Run test_rlhf_comprehensive.py first.")
            return 1
        results_file = sorted(result_files)[-1]
    
    evaluator = RLHFEffectivenessEvaluator()
    
    with open(results_file, 'r') as f:
        test_results = json.load(f)
    
    report = evaluator.generate_effectiveness_report(test_results)
    print(report)
    
    output_file = Path(results_file).parent / f"effectiveness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

