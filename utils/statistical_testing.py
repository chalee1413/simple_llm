"""
Statistical testing utilities for LLM evaluation.

DECISION RATIONALE:
- Implements paired t-test for model comparison (SoTA approach for dependent samples)
- Bootstrap confidence intervals for robust statistical inference (2024-2025 best practice)
- Provides statistical significance testing for LLM-as-Judge evaluations
- Handles small sample sizes and non-normal distributions appropriately

References:
- Paired t-test methodology for dependent samples comparison
  https://en.wikipedia.org/wiki/Student%27s_t-test#Paired_samples
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. Annals of Statistics, 7(1), 1-26.
  Modern applications: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
- Statistical significance testing for LLM evaluation (2024-2025 research)
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel


def paired_t_test(
    sample1: List[float],
    sample2: List[float],
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Perform paired t-test for comparing two dependent samples.
    
    DECISION RATIONALE:
    - Paired t-test is appropriate when comparing same items across two conditions
    - Used for model comparison (same evaluation set, different models)
    - More powerful than independent t-test for dependent samples
    - Standard approach in LLM evaluation research (2024-2025)
    
    Formula: t = (mean(diff)) / (std(diff) / sqrt(n))
    where diff = sample1 - sample2
    
    Args:
        sample1: First sample of paired observations
        sample2: Second sample of paired observations (must match sample1 length)
        confidence_level: Confidence level for test (default: 0.95)
    
    Returns:
        Dict containing:
        - statistic: t-statistic value
        - pvalue: p-value of the test
        - is_significant: Whether result is statistically significant
        - mean_difference: Mean difference between samples
        - confidence_interval: Confidence interval for mean difference
        - effect_size: Cohen's d effect size
    
    Raises:
        ValueError: If samples have different lengths or are empty
    """
    if len(sample1) != len(sample2):
        raise ValueError("Samples must have the same length for paired t-test")
    
    if len(sample1) < 2:
        raise ValueError("Samples must have at least 2 observations")
    
    # Convert to numpy arrays
    sample1_arr = np.array(sample1)
    sample2_arr = np.array(sample2)
    
    # Calculate differences
    differences = sample1_arr - sample2_arr
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Sample standard deviation
    
    # Perform paired t-test
    t_statistic, p_value = ttest_rel(sample1_arr, sample2_arr)
    
    # Calculate confidence interval for mean difference
    n = len(differences)
    degrees_freedom = n - 1
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, degrees_freedom)
    margin_error = t_critical * (std_diff / np.sqrt(n))
    confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(sample1_arr, ddof=1) + np.var(sample2_arr, ddof=1)) / 2)
    if pooled_std > 0:
        cohens_d = mean_diff / pooled_std
    else:
        cohens_d = 0.0
    
    # Determine significance
    is_significant = p_value < (1 - confidence_level)
    
    return {
        "statistic": float(t_statistic),
        "pvalue": float(p_value),
        "is_significant": is_significant,
        "mean_difference": float(mean_diff),
        "confidence_interval": confidence_interval,
        "effect_size": float(cohens_d),
        "degrees_freedom": int(degrees_freedom),
        "sample_size": n
    }


def bootstrap_confidence_interval(
    data: List[float],
    statistic_func: callable = np.mean,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[float, Tuple[float, float], List[float]]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    DECISION RATIONALE:
    - Bootstrap provides robust confidence intervals without distribution assumptions
    - Standard approach for LLM evaluation metrics (2024-2025 research)
    - Handles small sample sizes and non-normal distributions
    - More robust than parametric methods for complex statistics
    
    Methodology:
    1. Resample data with replacement n_iterations times
    2. Calculate statistic for each resample
    3. Use percentile method to get confidence interval
    
    Args:
        data: Sample data
        statistic_func: Function to calculate statistic (default: mean)
        n_iterations: Number of bootstrap iterations (default: 1000)
        confidence_level: Confidence level (default: 0.95)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple containing:
        - statistic: Original statistic value
        - confidence_interval: (lower, upper) confidence interval
        - bootstrap_samples: List of bootstrap statistic values
    
    Raises:
        ValueError: If data is empty or n_iterations < 1
    """
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate original statistic
    original_statistic = statistic_func(data)
    
    # Bootstrap resampling
    n = len(data)
    bootstrap_samples = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        # Calculate statistic for resample
        bootstrap_statistic = statistic_func(bootstrap_sample)
        bootstrap_samples.append(bootstrap_statistic)
    
    # Calculate confidence interval using percentile method
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    confidence_interval = (
        np.percentile(bootstrap_samples, lower_percentile),
        np.percentile(bootstrap_samples, upper_percentile)
    )
    
    return (
        float(original_statistic),
        confidence_interval,
        bootstrap_samples
    )


def calculate_statistical_significance(
    metric_scores1: List[float],
    metric_scores2: List[float],
    confidence_level: float = 0.95,
    use_bootstrap: bool = True,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Calculate statistical significance between two sets of metric scores.
    
    DECISION RATIONALE:
    - Combines paired t-test and bootstrap for robust statistical inference
    - Provides both parametric and non-parametric results
    - Standard approach for LLM evaluation comparison (2024-2025)
    - Handles various metric distributions appropriately
    
    Args:
        metric_scores1: First set of metric scores
        metric_scores2: Second set of metric scores
        confidence_level: Confidence level for tests (default: 0.95)
        use_bootstrap: Whether to use bootstrap confidence intervals (default: True)
        n_bootstrap: Number of bootstrap iterations (default: 1000)
    
    Returns:
        Dict containing:
        - paired_t_test: Results from paired t-test
        - bootstrap_ci: Bootstrap confidence interval (if use_bootstrap=True)
        - mean_difference: Mean difference between samples
        - interpretation: Human-readable interpretation
    
    Raises:
        ValueError: If samples have different lengths or are empty
    """
    if len(metric_scores1) != len(metric_scores2):
        raise ValueError("Metric score lists must have the same length")
    
    if len(metric_scores1) < 2:
        raise ValueError("Must have at least 2 metric scores for comparison")
    
    # Perform paired t-test
    t_test_results = paired_t_test(metric_scores1, metric_scores2, confidence_level)
    
    results = {
        "paired_t_test": t_test_results,
        "mean_difference": t_test_results["mean_difference"]
    }
    
    # Calculate bootstrap confidence interval if requested
    if use_bootstrap:
        differences = np.array(metric_scores1) - np.array(metric_scores2)
        bootstrap_stat, bootstrap_ci, _ = bootstrap_confidence_interval(
            differences.tolist(),
            statistic_func=np.mean,
            n_iterations=n_bootstrap,
            confidence_level=confidence_level
        )
        results["bootstrap_ci"] = bootstrap_ci
        results["bootstrap_mean"] = bootstrap_stat
    
    # Generate interpretation
    mean_diff = t_test_results["mean_difference"]
    p_value = t_test_results["pvalue"]
    is_sig = t_test_results["is_significant"]
    
    if is_sig:
        direction = "higher" if mean_diff > 0 else "lower"
        interpretation = (
            f"Statistically significant difference (p={p_value:.4f}). "
            f"Sample 1 has {direction} scores (mean difference: {mean_diff:.4f})"
        )
    else:
        interpretation = (
            f"No statistically significant difference (p={p_value:.4f}). "
            f"Mean difference: {mean_diff:.4f}"
        )
    
    results["interpretation"] = interpretation
    
    return results

