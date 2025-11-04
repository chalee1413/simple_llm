"""
Baseline tracking and improvement metrics for LLM evaluation.

DECISION RATIONALE:
- Baseline tracking enables comparison of improvements over time
- Improvement metrics quantify changes between versions
- Version comparison supports iterative development
- Statistical significance testing for improvement validation

References:
- Statistical significance testing for LLM evaluation (2024-2025)
- Baseline comparison methodologies (2024-2025 best practices)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from .statistical_testing import calculate_statistical_significance

logger = logging.getLogger(__name__)


class BaselineTracker:
    """
    Baseline tracking and improvement metrics calculator.
    
    DECISION RATIONALE:
    - Saves baseline results for future comparison
    - Calculates improvement metrics (current vs baseline)
    - Supports version-based comparison
    - Statistical validation of improvements
    """
    
    def __init__(self, baseline_dir: Path = Path("baselines")):
        """
        Initialize baseline tracker.
        
        Args:
            baseline_dir: Directory to store baseline results
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Baseline tracker initialized: {baseline_dir}")
    
    def save_baseline(
        self,
        baseline_name: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save baseline results for future comparison.
        
        Args:
            baseline_name: Name identifier for baseline
            results: Evaluation results to save as baseline
            metadata: Optional metadata (version, date, model, etc.)
        
        Returns:
            Path to saved baseline file
        """
        baseline_data = {
            "baseline_name": baseline_name,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "metadata": metadata or {}
        }
        
        baseline_file = self.baseline_dir / f"{baseline_name}.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)
        
        logger.info(f"Baseline saved: {baseline_file}")
        return baseline_file
    
    def load_baseline(self, baseline_name: str) -> Dict[str, Any]:
        """
        Load baseline results.
        
        Args:
            baseline_name: Name of baseline to load
        
        Returns:
            Baseline data dictionary
        
        Raises:
            FileNotFoundError: If baseline doesn't exist
        """
        baseline_file = self.baseline_dir / f"{baseline_name}.json"
        
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline not found: {baseline_file}")
        
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        logger.info(f"Baseline loaded: {baseline_file}")
        return baseline_data
    
    def list_baselines(self) -> List[str]:
        """
        List all available baselines.
        
        Returns:
            List of baseline names
        """
        baselines = [f.stem for f in self.baseline_dir.glob("*.json")]
        return sorted(baselines)
    
    def calculate_improvements(
        self,
        current_results: Dict[str, Any],
        baseline_name: str
    ) -> Dict[str, Any]:
        """
        Calculate improvement metrics compared to baseline.
        
        Args:
            current_results: Current evaluation results
            baseline_name: Name of baseline to compare against
        
        Returns:
            Dictionary with improvement metrics and statistical tests
        """
        baseline_data = self.load_baseline(baseline_name)
        baseline_results = baseline_data["results"]
        
        improvements = {
            "baseline_name": baseline_name,
            "baseline_timestamp": baseline_data["timestamp"],
            "comparison_timestamp": datetime.now().isoformat(),
            "improvements": {}
        }
        
        # Compare metrics
        for metric_name in current_results:
            if metric_name not in baseline_results:
                continue
            
            current_values = self._extract_metric_values(current_results[metric_name])
            baseline_values = self._extract_metric_values(baseline_results[metric_name])
            
            if not current_values or not baseline_values:
                continue
            
            # Calculate improvement metrics
            current_mean = np.mean(current_values)
            baseline_mean = np.mean(baseline_values)
            
            improvement_absolute = current_mean - baseline_mean
            improvement_relative = (improvement_absolute / baseline_mean * 100) if baseline_mean != 0 else 0
            
            # Statistical significance testing
            try:
                stats = calculate_statistical_significance(
                    baseline_values,
                    current_values,
                    confidence_level=0.95
                )
                is_significant = stats["paired_t_test"]["is_significant"]
                p_value = stats["paired_t_test"]["pvalue"]
            except Exception as e:
                logger.warning(f"Statistical test failed for {metric_name}: {e}")
                is_significant = False
                p_value = None
            
            improvements["improvements"][metric_name] = {
                "baseline_mean": float(baseline_mean),
                "current_mean": float(current_mean),
                "improvement_absolute": float(improvement_absolute),
                "improvement_relative": float(improvement_relative),
                "is_improvement": improvement_absolute > 0,
                "is_significant": is_significant,
                "p_value": p_value,
                "statistical_test": stats if 'stats' in locals() else None
            }
        
        return improvements
    
    def _extract_metric_values(self, metric_data: Any) -> List[float]:
        """
        Extract numeric values from metric data.
        
        Args:
            metric_data: Metric data (can be list, dict, or single value)
        
        Returns:
            List of numeric values
        """
        values = []
        
        if isinstance(metric_data, list):
            for item in metric_data:
                if isinstance(item, dict):
                    # Extract scores from dictionaries
                    if "score" in item:
                        values.append(float(item["score"]))
                    elif "mean_score" in item:
                        values.append(float(item["mean_score"]))
                    elif "toxicity_score" in item:
                        values.append(float(item["toxicity_score"]))
                elif isinstance(item, (int, float)):
                    values.append(float(item))
        elif isinstance(metric_data, dict):
            # Extract mean scores or similar
            if "mean_score" in metric_data:
                values.append(float(metric_data["mean_score"]))
            elif "mean_score1" in metric_data:
                values.append(float(metric_data["mean_score1"]))
            elif "mean_score2" in metric_data:
                values.append(float(metric_data["mean_score2"]))
        elif isinstance(metric_data, (int, float)):
            values.append(float(metric_data))
        
        return values
    
    def compare_versions(
        self,
        version1_name: str,
        version2_name: str
    ) -> Dict[str, Any]:
        """
        Compare two baseline versions.
        
        Args:
            version1_name: Name of first version baseline
            version2_name: Name of second version baseline
        
        Returns:
            Comparison results with improvements
        """
        version1_data = self.load_baseline(version1_name)
        version2_data = self.load_baseline(version2_name)
        
        improvements = self.calculate_improvements(
            version2_data["results"],
            version1_name
        )
        
        improvements["version1"] = version1_name
        improvements["version2"] = version2_name
        improvements["version1_timestamp"] = version1_data["timestamp"]
        improvements["version2_timestamp"] = version2_data["timestamp"]
        
        return improvements
    
    def generate_improvement_report(
        self,
        improvements: Dict[str, Any],
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable improvement report.
        
        Args:
            improvements: Improvement metrics dictionary
            output_file: Optional file to save report
        
        Returns:
            Report text
        """
        report_lines = [
            "=" * 80,
            "IMPROVEMENT REPORT",
            "=" * 80,
            f"\nBaseline: {improvements['baseline_name']}",
            f"Baseline Timestamp: {improvements.get('baseline_timestamp', 'N/A')}",
            f"Comparison Timestamp: {improvements.get('comparison_timestamp', 'N/A')}",
            "\n" + "-" * 80,
            "METRIC IMPROVEMENTS",
            "-" * 80,
        ]
        
        for metric_name, metric_data in improvements["improvements"].items():
            report_lines.append(f"\n{metric_name}:")
            report_lines.append(f"  Baseline Mean: {metric_data['baseline_mean']:.4f}")
            report_lines.append(f"  Current Mean:  {metric_data['current_mean']:.4f}")
            report_lines.append(f"  Improvement:  {metric_data['improvement_absolute']:+.4f} ({metric_data['improvement_relative']:+.2f}%)")
            report_lines.append(f"  Significant:   {metric_data['is_significant']}")
            if metric_data['p_value']:
                report_lines.append(f"  P-value:       {metric_data['p_value']:.4f}")
            report_lines.append(f"  Status:        {'[OK] IMPROVED' if metric_data['is_improvement'] and metric_data['is_significant'] else '[FAIL] NO SIGNIFICANT CHANGE'}")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Improvement report saved: {output_file}")
        
        return report_text

