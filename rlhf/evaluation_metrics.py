"""
RLHF evaluation metrics for reward models and training.

DECISION RATIONALE:
- Reward model evaluation metrics (accuracy, correlation, ranking metrics)
- Training metrics tracking and visualization
- Model performance assessment
- Statistical validation of improvements

References:
- Reward model evaluation metrics (2024-2025 best practices)
- Statistical significance testing for RLHF evaluation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score

from config import Config

logger = logging.getLogger(__name__)


class RewardModelEvaluator:
    """
    Evaluator for reward models.
    
    DECISION RATIONALE:
    - Evaluates reward model quality with multiple metrics
    - Provides accuracy, correlation, and ranking metrics
    - Supports statistical validation
    """
    
    def __init__(self, reward_model, tokenizer):
        """
        Initialize reward model evaluator.
        
        Args:
            reward_model: Trained reward model
            tokenizer: Tokenizer for reward model
        """
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        logger.info("RewardModelEvaluator initialized")
    
    def evaluate(
        self,
        preferences: List[Dict[str, Any]],
        max_length: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate reward model on preference data.
        
        Args:
            preferences: List of preference dictionaries with prompt, chosen, rejected
            max_length: Maximum sequence length
        
        Returns:
            Dictionary of evaluation metrics
        """
        max_length = max_length or Config.PPO_MAX_SEQ_LENGTH
        
        chosen_rewards = []
        rejected_rewards = []
        correct_predictions = 0
        
        logger.info(f"Evaluating reward model on {len(preferences)} preferences")
        
        for pref in preferences:
            prompt = pref["prompt"]
            chosen = pref["chosen"]
            rejected = pref["rejected"]
            
            chosen_text = f"{prompt} {chosen}"
            rejected_text = f"{prompt} {rejected}"
            
            chosen_inputs = self.tokenizer(
                chosen_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            
            rejected_inputs = self.tokenizer(
                rejected_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            
            import torch
            self.reward_model.model.eval()
            with torch.no_grad():
                chosen_reward = self.reward_model(**chosen_inputs).item()
                rejected_reward = self.reward_model(**rejected_inputs).item()
            
            chosen_rewards.append(chosen_reward)
            rejected_rewards.append(rejected_reward)
            
            if chosen_reward > rejected_reward:
                correct_predictions += 1
        
        chosen_rewards = np.array(chosen_rewards)
        rejected_rewards = np.array(rejected_rewards)
        
        accuracy = correct_predictions / len(preferences)
        
        reward_diff = chosen_rewards - rejected_rewards
        mean_reward_diff = np.mean(reward_diff)
        std_reward_diff = np.std(reward_diff)
        
        correlation = pearsonr(chosen_rewards, rejected_rewards)[0]
        
        ranking_accuracy = self._calculate_ranking_accuracy(chosen_rewards, rejected_rewards)
        
        metrics = {
            "accuracy": float(accuracy),
            "mean_reward_diff": float(mean_reward_diff),
            "std_reward_diff": float(std_reward_diff),
            "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "ranking_accuracy": float(ranking_accuracy),
            "mean_chosen_reward": float(np.mean(chosen_rewards)),
            "mean_rejected_reward": float(np.mean(rejected_rewards)),
            "total_preferences": len(preferences)
        }
        
        logger.info(f"Reward model evaluation completed:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Mean reward difference: {mean_reward_diff:.4f}")
        logger.info(f"  Ranking accuracy: {ranking_accuracy:.4f}")
        
        return metrics
    
    def _calculate_ranking_accuracy(
        self,
        chosen_rewards: np.ndarray,
        rejected_rewards: np.ndarray
    ) -> float:
        """
        Calculate ranking accuracy (percentage of correct rankings).
        
        Args:
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses
        
        Returns:
            Ranking accuracy (0-1)
        """
        correct = np.sum(chosen_rewards > rejected_rewards)
        total = len(chosen_rewards)
        return correct / total if total > 0 else 0.0


class TrainingMetricsTracker:
    """
    Tracks training metrics during RLHF training.
    
    DECISION RATIONALE:
    - Tracks metrics across training steps
    - Provides metrics aggregation and summary
    - Supports visualization preparation
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history: Dict[str, List[float]] = {}
        self.step_history: List[int] = []
        logger.info("TrainingMetricsTracker initialized")
    
    def log_step(
        self,
        step: int,
        metrics: Dict[str, float]
    ):
        """
        Log metrics for a training step.
        
        Args:
            step: Training step number
            metrics: Dictionary of metric name -> value
        """
        self.step_history.append(step)
        
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary of metric summaries
        """
        summary = {}
        
        for metric_name, values in self.metrics_history.items():
            if values:
                summary[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "final": float(values[-1])
                }
        
        return summary
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Get latest metric values.
        
        Returns:
            Dictionary of latest metric values
        """
        latest = {}
        for metric_name, values in self.metrics_history.items():
            if values:
                latest[metric_name] = float(values[-1])
        return latest
    
    def reset(self):
        """Reset metrics history."""
        self.metrics_history.clear()
        self.step_history.clear()
        logger.info("Metrics tracker reset")
    
    def save_to_dict(self) -> Dict[str, Any]:
        """
        Save metrics to dictionary for serialization.
        
        Returns:
            Dictionary representation of metrics
        """
        return {
            "step_history": self.step_history,
            "metrics_history": self.metrics_history,
            "summary": self.get_summary()
        }


def evaluate_reward_model(
    reward_model,
    tokenizer,
    preferences: List[Dict[str, Any]],
    max_length: Optional[int] = None
) -> Dict[str, float]:
    """
    Convenience function for reward model evaluation.
    
    Args:
        reward_model: Trained reward model
        tokenizer: Tokenizer for reward model
        preferences: List of preference dictionaries
        max_length: Maximum sequence length
    
    Returns:
        Evaluation metrics dictionary
    """
    evaluator = RewardModelEvaluator(reward_model, tokenizer)
    return evaluator.evaluate(preferences, max_length)

