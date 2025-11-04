"""
Example: Model Comparison with Statistical Validation

This example demonstrates how to compare two model versions with statistical significance testing.

DECISION RATIONALE:
- LLM-as-Judge evaluation with statistical testing
- Baseline tracking for version comparison
- Statistical significance validation
- Real-world use case demonstration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_evaluation_demo import LLMAsJudge
from utils.baseline_tracking import BaselineTracker
import json

def compare_model_versions():
    """
    Compare two model versions with statistical validation.
    """
    print("=" * 80)
    print("Model Comparison Example")
    print("=" * 80)
    
    # Initialize LLM-as-Judge
    print("\nInitializing LLM-as-Judge...")
    judge = LLMAsJudge(llm_provider="huggingface")
    
    # Example outputs from v1.0 and v2.0
    v1_outputs = [
        "Machine learning is a subset of AI.",
        "Neural networks process data through layers.",
        "RAG combines retrieval with generation.",
        "Embeddings represent text as vectors.",
        "Vector search finds similar items."
    ]
    
    v2_outputs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        "Neural networks are computational models inspired by biological neurons that process information through interconnected layers of nodes.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval systems with language models to improve answer quality.",
        "Embeddings are dense vector representations of text that capture semantic meaning for similarity search.",
        "Vector search uses similarity metrics to find semantically similar items in high-dimensional spaces."
    ]
    
    # Evaluation criteria
    criteria = "Clarity, accuracy, and completeness"
    
    print(f"\nEvaluating {len(v1_outputs)} output pairs...")
    print(f"Criteria: {criteria}")
    
    # Evaluate with statistical significance
    results = judge.evaluate_with_statistics(
        outputs1=v1_outputs,
        outputs2=v2_outputs,
        criteria=criteria
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Version 1.0 Mean Score: {results['mean_score1']:.3f}")
    print(f"Version 2.0 Mean Score: {results['mean_score2']:.3f}")
    print(f"Improvement: {results['mean_score2'] - results['mean_score1']:.3f} ({(results['mean_score2'] - results['mean_score1']) / results['mean_score1'] * 100:.1f}%)")
    
    # Statistical significance
    stats = results['statistical_test']
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 80)
    
    if stats.get('is_significant'):
        print(f"P-value: {stats['paired_t_test']['pvalue']:.4f}")
        print(f"Is Significant: True (p < 0.05)")
        print(f"Effect Size (Cohen's d): {stats['effect_size']:.3f}")
        
        if stats['effect_size'] > 0.5:
            print("Effect Size: Large (practically meaningful)")
        elif stats['effect_size'] > 0.2:
            print("Effect Size: Medium (moderately meaningful)")
        else:
            print("Effect Size: Small (may not be meaningful)")
        
        print(f"Confidence Interval: [{stats['bootstrap_ci'][0]:.3f}, {stats['bootstrap_ci'][1]:.3f}]")
    else:
        print(f"P-value: {stats['paired_t_test']['pvalue']:.4f}")
        print("Is Significant: False (p >= 0.05)")
        print("Improvement may be due to chance")
    
    # Save baselines
    baseline_tracker = BaselineTracker()
    
    print("\nSaving baselines...")
    baseline_tracker.save_baseline(
        "v1.0",
        {"llm_judge": results},
        {"version": "v1.0", "description": "Initial model version"}
    )
    
    baseline_tracker.save_baseline(
        "v2.0",
        {"llm_judge": results},
        {"version": "v2.0", "description": "Improved model version"}
    )
    
    # Compare versions
    print("\nComparing versions...")
    improvements = baseline_tracker.compare_versions("v1.0", "v2.0")
    report = baseline_tracker.generate_improvement_report(improvements)
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT REPORT")
    print("=" * 80)
    print(report)
    
    # Save report
    report_file = project_root / "output" / "model_comparison_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_tracker.generate_improvement_report(improvements, report_file)
    
    print(f"\nReport saved to: {report_file}")
    
    # Decision
    print("\n" + "=" * 80)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 80)
    
    if stats.get('is_significant') and stats['effect_size'] > 0.2:
        print("RECOMMENDATION: Deploy v2.0")
        print("Reason: Statistically significant improvement with meaningful effect size")
    elif stats.get('is_significant'):
        print("RECOMMENDATION: Consider deploying v2.0")
        print("Reason: Statistically significant improvement, but small effect size")
    else:
        print("RECOMMENDATION: Do not deploy v2.0")
        print("Reason: Improvement is not statistically significant (may be due to chance)")


if __name__ == "__main__":
    compare_model_versions()
