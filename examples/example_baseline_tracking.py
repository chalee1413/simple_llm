"""
Example: Baseline Tracking for Production Monitoring

This example demonstrates how to use baseline tracking for continuous monitoring and improvement.

DECISION RATIONALE:
- Production monitoring workflow
- Baseline comparison over time
- Improvement tracking and reporting
- Real-world use case demonstration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.baseline_tracking import BaselineTracker
import json

def baseline_tracking_example():
    """
    Demonstrate baseline tracking workflow.
    """
    print("=" * 80)
    print("Baseline Tracking Example")
    print("=" * 80)
    
    # Initialize baseline tracker
    baseline_tracker = BaselineTracker()
    
    # Simulate initial production deployment (v1.0)
    print("\nStep 1: Establishing baseline v1.0...")
    v1_results = {
        "toxicity": [
            {"toxicity_score": 0.15, "is_toxic": False},
            {"toxicity_score": 0.20, "is_toxic": False},
            {"toxicity_score": 0.18, "is_toxic": False}
        ],
        "code_quality": [
            {"overall_quality_score": 0.75},
            {"overall_quality_score": 0.80},
            {"overall_quality_score": 0.70}
        ],
        "llm_judge": {
            "mean_score1": 0.65,
            "mean_score2": 0.70,
            "scores1": [0.60, 0.65, 0.70],
            "scores2": [0.65, 0.70, 0.75]
        }
    }
    
    baseline_tracker.save_baseline(
        "production_v1.0",
        v1_results,
        {
            "version": "v1.0",
            "deployment_date": "2025-11-01",
            "description": "Initial production deployment"
        }
    )
    print("Baseline v1.0 saved")
    
    # Simulate improved version (v2.0)
    print("\nStep 2: Evaluating improved version v2.0...")
    v2_results = {
        "toxicity": [
            {"toxicity_score": 0.10, "is_toxic": False},
            {"toxicity_score": 0.12, "is_toxic": False},
            {"toxicity_score": 0.11, "is_toxic": False}
        ],
        "code_quality": [
            {"overall_quality_score": 0.85},
            {"overall_quality_score": 0.90},
            {"overall_quality_score": 0.80}
        ],
        "llm_judge": {
            "mean_score1": 0.75,
            "mean_score2": 0.85,
            "scores1": [0.70, 0.75, 0.80],
            "scores2": [0.80, 0.85, 0.90]
        }
    }
    
    baseline_tracker.save_baseline(
        "production_v2.0",
        v2_results,
        {
            "version": "v2.0",
            "deployment_date": "2025-11-04",
            "description": "Improved version with optimizations"
        }
    )
    print("Baseline v2.0 saved")
    
    # List all baselines
    print("\nStep 3: Listing all baselines...")
    baselines = baseline_tracker.list_baselines()
    print(f"Available baselines: {', '.join(baselines)}")
    
    # Compare versions
    print("\nStep 4: Comparing v1.0 vs v2.0...")
    improvements = baseline_tracker.compare_versions("production_v1.0", "production_v2.0")
    
    # Generate report
    report = baseline_tracker.generate_improvement_report(improvements)
    print("\n" + "=" * 80)
    print("IMPROVEMENT REPORT")
    print("=" * 80)
    print(report)
    
    # Save report
    report_file = project_root / "output" / "baseline_tracking_example.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_tracker.generate_improvement_report(improvements, report_file)
    print(f"\nReport saved to: {report_file}")
    
    # Continuous monitoring example
    print("\n" + "=" * 80)
    print("CONTINUOUS MONITORING EXAMPLE")
    print("=" * 80)
    
    # Simulate current evaluation
    current_results = {
        "toxicity": [
            {"toxicity_score": 0.08, "is_toxic": False},
            {"toxicity_score": 0.09, "is_toxic": False},
            {"toxicity_score": 0.10, "is_toxic": False}
        ],
        "code_quality": [
            {"overall_quality_score": 0.88},
            {"overall_quality_score": 0.92},
            {"overall_quality_score": 0.85}
        ],
        "llm_judge": {
            "mean_score1": 0.78,
            "mean_score2": 0.88,
            "scores1": [0.75, 0.78, 0.81],
            "scores2": [0.85, 0.88, 0.91]
        }
    }
    
    # Compare against baseline
    print("\nComparing current results against baseline v1.0...")
    improvements_vs_baseline = baseline_tracker.calculate_improvements(
        current_results,
        "production_v1.0"
    )
    
    report_vs_baseline = baseline_tracker.generate_improvement_report(improvements_vs_baseline)
    print("\n" + report_vs_baseline)
    
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print("1. Establish baseline: Save initial results as baseline")
    print("2. Make improvements: Develop and test improvements")
    print("3. Save new baseline: Save improved results as new baseline")
    print("4. Compare versions: Use compare_versions() to compare baselines")
    print("5. Continuous monitoring: Compare current results against baseline")
    print("6. Generate reports: Use generate_improvement_report() for stakeholders")


if __name__ == "__main__":
    baseline_tracking_example()
