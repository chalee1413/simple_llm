# Step-by-Step Tutorial Guide

This tutorial provides step-by-step instructions for using the LLM Evaluation Framework. Follow these steps to evaluate your LLM systems.

## Tutorial 1: Your First Evaluation in 5 Minutes

### Step 1: Setup

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Create Test Data

Create `example_inputs/test_samples.json`:

```json
{
  "texts": [
    "This is a test text for toxicity detection.",
    "Another sample text for evaluation."
  ],
  "code": [
    "def hello():\n    print('Hello, World!')",
    "def complex_function(x, y, z):\n    if x > 0:\n        if y > 0:\n            return x + y\n    return 0"
  ],
  "prompts": [
    "What is machine learning?",
    "Explain neural networks."
  ]
}
```

### Step 3: Run Evaluation

```bash
# Basic evaluation
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --output-dir output/
```

### Step 4: Review Results

Check `output/evaluation_results_*.json` for results:

```json
{
  "toxicity": [
    {
      "text": "This is a test text...",
      "toxicity_score": 0.0,
      "is_toxic": false
    }
  ],
  "code_quality": [
    {
      "mccabe_complexity": 1,
      "cognitive_complexity": 1.5,
      "overall_quality_score": 0.95
    }
  ],
  "adversarial": {
    "total_adversarial_tests": 6,
    "detected_injections": 2,
    "robustness_rate": 0.67
  }
}
```

### Step 5: Save Baseline

```bash
# Save baseline for future comparison
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --save-baseline initial_run
```

First evaluation completed.

---

## Tutorial 2: Understanding Results

### Understanding Toxicity Scores

- **Score 0.0-0.3**: Non-toxic (safe for production)
- **Score 0.3-0.7**: Potentially toxic (review needed)
- **Score 0.7-1.0**: Toxic (filter or block)

### Understanding Code Quality Scores

- **McCabe Complexity < 10**: Good complexity
- **McCabe Complexity 10-15**: Moderate complexity (consider refactoring)
- **McCabe Complexity > 15**: High complexity (requires refactoring)

- **Cognitive Complexity < 15**: Good maintainability
- **Cognitive Complexity 15-25**: Moderate maintainability
- **Cognitive Complexity > 25**: Poor maintainability

### Understanding LLM-as-Judge Scores

- **Score 0.0-0.5**: Low quality
- **Score 0.5-0.7**: Moderate quality
- **Score 0.7-0.9**: Good quality
- **Score 0.9-1.0**: High quality

### Understanding Statistical Significance

- **P-value < 0.05**: Statistically significant (improvement is valid)
- **P-value >= 0.05**: Not statistically significant (improvement may be due to chance)

- **Cohen's d < 0.2**: Small effect
- **Cohen's d 0.2-0.5**: Medium effect
- **Cohen's d > 0.5**: Large effect

---

## Tutorial 3: Comparing Baselines

### Step 1: Run Initial Evaluation

```bash
# Run evaluation and save baseline v1.0
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --save-baseline v1.0
```

### Step 2: Make Improvements

After improving your model or system, run evaluation again:

```bash
# Run evaluation and save baseline v2.0
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --save-baseline v2.0
```

### Step 3: Compare Versions

```bash
# Compare v1.0 vs v2.0
python llm_evaluation_demo.py --compare-versions v1.0 v2.0
```

### Step 4: Review Improvement Report

The comparison will show:
- Mean scores for each version
- Absolute and relative improvements
- Statistical significance (p-value)
- Effect size (Cohen's d)
- Whether improvement is statistically significant

### Output Format

The comparison generates a report showing:

- Mean scores for each version
- Absolute and relative improvements
- Statistical significance (p-value)
- Whether improvement is statistically significant

Run the comparison to see actual results with your data.

---

## Tutorial 4: Running Benchmarks

### Step 1: Run Vector Search Benchmark

```bash
# Run benchmark with visualizations
python vector_search_comparison.py \
    --sizes 1000 5000 10000 \
    --queries 50 \
    --k 10 \
    --visualize
```

### Step 2: Review Benchmark Results

Check `output/vector_search_benchmark_*.csv` for detailed results:

```csv
method,build_time,query_time_mean,queries_per_second,index_size_mb,filter_support,persistence
FAISS,0.0031,0.20,5031.3,7.32,False,False
Qdrant,2.1230,4.85,206.4,19.73,True,True
Chroma,2.4151,1.52,659.5,19.88,True,True
```

### Step 3: Generate Visualizations

```bash
# Generate charts from CSV
python visualize_benchmarks.py \
    --input output/vector_search_benchmark_*.csv \
    --format png
```

### Step 4: Review Charts

Check `output/benchmark_charts/` for:
- `query_performance.png`: Query latency vs dataset size
- `queries_per_second.png`: Throughput comparison
- `scalability_analysis.png`: Performance degradation analysis
- `feature_comparison.png`: Feature matrix
- `summary_dashboard.png`: Combined metrics

### Step 5: Make Decision

Based on results:
- Need maximum performance? Choose FAISS
- Need metadata filtering? Choose Qdrant or Chroma
- Need hybrid queries? Choose PostgreSQL + pgvector
- Need easy integration? Choose Chroma

---

## Tutorial 5: Understanding Statistical Significance

### Step 1: Run Evaluation with LLM-as-Judge

```bash
# Run LLM-as-Judge evaluation
python llm_evaluation_demo.py \
    --input-file example_inputs/test_samples.json \
    --evaluation-type llm-judge \
    --llm-provider huggingface
```

### Step 2: Review Statistical Test Results

Check the output for statistical significance:

The output shows mean scores, statistical test results, and effect size. Run the evaluation to see actual results with your data.

### Step 3: Interpret Results

- **P-value < 0.05**: Improvement is statistically significant (not due to chance)
- **P-value >= 0.05**: Improvement may be due to chance (not statistically significant)

- **Cohen's d > 0.5**: Large effect (practically meaningful)
- **Cohen's d 0.2-0.5**: Medium effect (moderately meaningful)
- **Cohen's d < 0.2**: Small effect (may not be meaningful)

### Step 4: Make Decision

Based on statistical significance:
- **Significant (p < 0.05) + Large effect (d > 0.5)**: Deploy improvement
- **Significant (p < 0.05) + Small effect (d < 0.2)**: Consider deployment
- **Not significant (p >= 0.05)**: Do not deploy (may be due to chance)

---

## Tutorial 6: Running Benchmarks

### Step 1: Prepare Test Data

Create test data file or use default samples.

### Step 2: Run Evaluation

```bash
# Run all evaluation types
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --evaluation-type all

# Run specific evaluation type
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --evaluation-type toxicity
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --evaluation-type code-quality
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --evaluation-type adversarial
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --evaluation-type llm-judge
```

### Step 3: Review Results

Check `output/evaluation_results_*.json` for detailed results.

### Step 4: Save Baseline

```bash
# Save baseline for future comparison
python llm_evaluation_demo.py \
    --input-file example_inputs/test_samples.json \
    --save-baseline production_v1.0
```

---

## Tutorial 7: Common Pitfalls and Solutions

### Pitfall 1: Memory Errors

**Problem**: Process crashes with exit code 138 (OOM killer)

**Solution**:
- Run evaluations separately: `--evaluation-type toxicity,code-quality,adversarial` (skip LLM-as-Judge)
- Use API-based providers: `--llm-provider openai` or `bedrock`
- Ensure >4GB available RAM

### Pitfall 2: Import Errors

**Problem**: Module not found errors

**Solution**:
- Install dependencies: `pip install -r requirements.txt`
- Check Python path: Ensure project root is in `sys.path`
- Activate virtual environment: `source venv/bin/activate`

### Pitfall 3: Statistical Significance Not Met

**Problem**: Improvements not statistically significant

**Solution**:
- Increase sample size (more test cases)
- Use appropriate statistical test (paired t-test for dependent samples)
- Check effect size (may be small but significant)

### Pitfall 4: Baseline Not Found

**Problem**: Baseline file not found error

**Solution**:
- Check baseline directory: `baselines/`
- List available baselines: `python llm_evaluation_demo.py --baseline-dir baselines`
- Ensure baseline name matches saved file

### Pitfall 5: Vector Search Benchmark Fails

**Problem**: Vector database connection fails

**Solution**:
- Check database availability (Qdrant, Chroma, PostgreSQL)
- Verify Docker is running (for PostgreSQL)
- Check connection settings (host, port, credentials)
- Review error logs for specific issues

---

## Tutorial 8: Production Deployment Checklist

### Pre-Deployment

- [ ] Run comprehensive evaluation on production samples
- [ ] Save baseline: `--save-baseline production_v1.0`
- [ ] Verify all metrics meet thresholds:
  - Toxicity < 0.3
  - Code Quality > 0.7
  - LLM-as-Judge > 0.7
  - Adversarial Robustness > 80%
- [ ] Review statistical significance of improvements
- [ ] Document baseline metrics

### Post-Deployment

- [ ] Run evaluation regularly (weekly/monthly)
- [ ] Compare against baseline: `--compare-baseline production_v1.0`
- [ ] Track improvements over time
- [ ] Monitor for degradation (toxicity increase, quality decrease)
- [ ] Generate reports for stakeholders

### Continuous Improvement

- [ ] Make improvements to model/system
- [ ] Save new baseline: `--save-baseline production_v2.0`
- [ ] Compare versions: `--compare-versions production_v1.0 production_v2.0`
- [ ] Validate statistical significance
- [ ] Deploy if improvements are significant and meaningful

---

## Next Steps

After completing these tutorials:

1. Review [EXAMPLES.md](EXAMPLES.md) for real-world use cases
2. Explore `examples/` directory for complete scripts
3. Read [METHODOLOGY.md](METHODOLOGY.md) for algorithm details
4. Check [INFRASTRUCTURE.md](INFRASTRUCTURE.md) for architecture decisions
5. See [CHALLENGES.md](CHALLENGES.md) for troubleshooting

For questions or issues, review the documentation or check the code comments for detailed explanations.

