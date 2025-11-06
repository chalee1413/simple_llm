# Real-World Use Cases and Examples

This document provides real-world use cases and examples for using the LLM Evaluation Framework.

## Use Case 1: Evaluating a RAG System for Production

### Scenario

You've built a RAG system for document Q&A and need to evaluate it before production deployment. You want to measure:
- Answer quality (faithfulness to context)
- Retrieval quality (context relevance)
- Overall system performance

### Solution

```python
from utils.evaluation_metrics import RAGEvaluator
from llm_evaluation_demo import ToxicityDetector, CodeQualityEvaluator
import json

# Initialize evaluators
rag_evaluator = RAGEvaluator()
toxicity_detector = ToxicityDetector()

# Example RAG outputs
test_cases = [
    {
        "question": "What is machine learning?",
        "context": ["Machine learning is a subset of artificial intelligence.", "It enables computers to learn from data."],
        "answer": "Machine learning is a subset of AI that enables computers to learn from data."
    },
    {
        "question": "How does neural networks work?",
        "context": ["Neural networks are inspired by biological neurons.", "They process information through interconnected nodes."],
        "answer": "Neural networks process information through interconnected nodes."
    }
]

# Evaluate each test case
results = []
for case in test_cases:
    faithfulness = rag_evaluator.calculate_faithfulness(
        answer=case["answer"],
        context=case["context"],
        question=case["question"]
    )
    
    answer_relevancy = rag_evaluator.calculate_answer_relevancy(
        answer=case["answer"],
        question=case["question"]
    )
    
    context_precision = rag_evaluator.calculate_context_precision(
        context=case["context"],
        question=case["question"]
    )
    
    context_recall = rag_evaluator.calculate_context_recall(
        context=case["context"],
        question=case["question"]
    )
    
    # Check for toxicity
    toxicity = toxicity_detector.detect_toxicity(case["answer"])
    
    results.append({
        "question": case["question"],
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "toxicity_score": toxicity["toxicity_score"],
        "is_toxic": toxicity["is_toxic"]
    })

# Save results
with open("output/rag_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Calculate averages
avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
avg_precision = sum(r["context_precision"] for r in results) / len(results)
avg_recall = sum(r["context_recall"] for r in results) / len(results)

print(f"Average Faithfulness: {avg_faithfulness:.3f}")
print(f"Average Answer Relevancy: {avg_relevancy:.3f}")
print(f"Average Context Precision: {avg_precision:.3f}")
print(f"Average Context Recall: {avg_recall:.3f}")
```

### Output Format

The script generates a JSON file with evaluation results for each test case:

```json
[
  {
    "question": "What is machine learning?",
    "faithfulness": <calculated_score>,
    "answer_relevancy": <calculated_score>,
    "context_precision": <calculated_score>,
    "context_recall": <calculated_score>,
    "toxicity_score": <calculated_score>,
    "is_toxic": <boolean>
  }
]
```

Run the script to see actual results.

### Decision Criteria

- Faithfulness > 0.8: Answers are grounded in context
- Answer Relevancy > 0.8: Answers are relevant to questions
- Context Precision > 0.7: Retrieved contexts are relevant
- Context Recall > 0.7: Retrieved contexts are complete
- Toxicity Score < 0.3: Answers are safe for production

---

## Use Case 2: Comparing Model Versions with Statistical Validation

### Scenario

You've improved your LLM model and want to validate that the new version (v2.0) is better than the previous version (v1.0) using statistical significance testing.

### Solution

```python
from llm_evaluation_demo import LLMAsJudge
from utils.baseline_tracking import BaselineTracker
from pathlib import Path

# Initialize LLM-as-Judge
judge = LLMAsJudge(llm_provider="huggingface")

# Example outputs from v1.0 and v2.0
v1_outputs = [
    "Machine learning is a subset of AI.",
    "Neural networks process data through layers."
]

v2_outputs = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
    "Neural networks are computational models inspired by biological neurons that process information through interconnected layers of nodes."
]

# Evaluation criteria
criteria = "Clarity, accuracy, and completeness"

# Evaluate with statistical significance
results = judge.evaluate_with_statistics(
    outputs1=v1_outputs,
    outputs2=v2_outputs,
    criteria=criteria
)

# Print results
print(f"Version 1.0 Mean Score: {results['mean_score1']:.3f}")
print(f"Version 2.0 Mean Score: {results['mean_score2']:.3f}")

# Statistical significance
stats = results['statistical_test']
if stats['is_significant']:
    print(f"Statistically significant improvement (p-value: {stats['paired_t_test']['pvalue']:.4f})")
    print(f"Effect size (Cohen's d): {stats['effect_size']:.3f}")
else:
    print("No statistically significant difference")

# Save baseline for future comparison
baseline_tracker = BaselineTracker()
baseline_tracker.save_baseline("v1.0", {"llm_judge": results}, {"version": "v1.0"})
baseline_tracker.save_baseline("v2.0", {"llm_judge": results}, {"version": "v2.0"})

# Compare versions
improvements = baseline_tracker.compare_versions("v1.0", "v2.0")
report = baseline_tracker.generate_improvement_report(improvements)
print("\n" + report)
```

### Output Format

The script generates a comparison report showing:

- Mean scores for each version
- Absolute and relative improvements
- Statistical significance (p-value)
- Effect size (Cohen's d)
- Whether improvement is statistically significant

Run the script to see actual results with your data.

### Decision Criteria

- Statistical significance (p-value < 0.05): Improvement is statistically valid
- Effect size (Cohen's d > 0.5): Improvement is practically meaningful
- Mean score increase > 10%: Improvement is substantial

---

## Use Case 3: Choosing Between Vector Search Solutions

### Scenario

You need to choose a vector search solution for your RAG system. You want to compare:
- FAISS (library)
- Qdrant (vector database)
- Chroma (embedded database)
- PostgreSQL + pgvector (RDBMS with vector extension)

### Solution

```bash
# Run comprehensive benchmark
python vector_search_comparison.py \
    --sizes 1000 5000 10000 \
    --queries 50 \
    --k 10 \
    --visualize

# Generate visualizations
python visualize_benchmarks.py \
    --input output/vector_search_benchmark_*.csv \
    --format png
```

### Analysis

Review the generated charts:
- `query_performance.png`: Query latency vs dataset size
- `queries_per_second.png`: Throughput comparison
- `scalability_analysis.png`: Performance degradation with scale
- `feature_comparison.png`: Feature matrix (filtering, persistence, etc.)
- `summary_dashboard.png`: Combined metrics

### Decision Framework

**Choose FAISS if:**
- Maximum query performance is critical
- Dataset fits in memory
- No need for metadata filtering
- Simple persistence is sufficient

**Choose Qdrant if:**
- Need metadata filtering and persistence
- Production deployment with scaling
- High ingestion rates required
- Can accept slower query performance for features

**Choose Chroma if:**
- Easy Python integration is priority
- Embedded database (no separate server)
- Small to medium datasets
- Rapid prototyping

**Choose PostgreSQL + pgvector if:**
- Already using PostgreSQL
- Need hybrid queries (SQL + vector search)
- ACID transactions required
- Complex filtering with SQL WHERE clauses

### Results

Benchmark results vary by dataset size and hardware. Run the benchmark to see actual performance metrics:

- Query performance (latency, throughput)
- Ingestion rate (vectors per second)
- Feature support (metadata filtering, persistence, ACID transactions)
- Scalability analysis (performance vs dataset size)

---

## Use Case 4: Production Deployment with Baseline Tracking

### Scenario

You're deploying a production LLM system and need to:
- Establish baseline performance
- Track improvements over time
- Validate changes with statistical significance
- Generate reports for stakeholders

### Solution

```python
from llm_evaluation_demo import main as run_evaluation
from utils.baseline_tracking import BaselineTracker
from pathlib import Path
import json

# Step 1: Establish baseline (v1.0)
print("Establishing baseline v1.0...")
# Run evaluation and save baseline
# python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --save-baseline v1.0

# Step 2: After improvements, evaluate again
print("Evaluating improved version v2.0...")
# Run evaluation
# python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --save-baseline v2.0

# Step 3: Compare versions
baseline_tracker = BaselineTracker()

# Compare current vs baseline
improvements = baseline_tracker.compare_versions("v1.0", "v2.0")
report = baseline_tracker.generate_improvement_report(improvements)

# Save report
report_file = Path("output/production_improvement_report.txt")
baseline_tracker.generate_improvement_report(improvements, report_file)

print(f"Improvement report saved: {report_file}")
print("\n" + report)

# Step 4: Continuous monitoring
# Run evaluation regularly and compare against baseline
# python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --compare-baseline v1.0
```

### Expected Workflow

1. **Initial Deployment (v1.0)**:
   - Run evaluation on production samples
   - Save baseline: `--save-baseline v1.0`
   - Document metrics: toxicity, code quality, LLM-as-Judge scores

2. **After Improvements (v2.0)**:
   - Run evaluation again
   - Save new baseline: `--save-baseline v2.0`
   - Compare versions: `--compare-versions v1.0 v2.0`

3. **Continuous Monitoring**:
   - Run evaluation regularly
   - Compare against baseline: `--compare-baseline v1.0`
   - Track improvements over time

### Key Metrics to Track

- **Toxicity Scores**: Should remain low (< 0.3)
- **Code Quality**: Should improve or maintain
- **LLM-as-Judge Scores**: Should improve with statistical significance
- **Statistical Significance**: Ensure improvements are statistically valid

---

## Use Case 5: Code Quality Assessment for Production Codebase

### Scenario

You need to assess code quality across your codebase to identify:
- Complex functions (high McCabe complexity)
- Cognitive complexity hotspots
- Areas needing refactoring

### Solution

```python
from llm_evaluation_demo import CodeQualityEvaluator
from pathlib import Path
import json

# Initialize code quality evaluator
code_evaluator = CodeQualityEvaluator()

# Example production code files
code_files = [
    ("utils/statistical_testing.py", """
def calculate_statistical_significance(sample1, sample2, confidence_level=0.95):
    # Implementation
    t_stat, p_value = paired_t_test(sample1, sample2)
    ci = bootstrap_confidence_interval(sample1, sample2)
    return {
        'is_significant': p_value < (1 - confidence_level),
        'p_value': p_value,
        'confidence_interval': ci
    }
    """),
    ("utils/baseline_tracking.py", """
def calculate_improvements(current_results, baseline_name):
    baseline_data = load_baseline(baseline_name)
    improvements = {}
    for metric in current_results:
        improvements[metric] = compare_metrics(current_results[metric], baseline_data[metric])
    return improvements
    """)
]

# Evaluate each file
results = []
for filename, code in code_files:
    quality = code_evaluator.evaluate_code_quality(code)
    results.append({
        "filename": filename,
        "mccabe_complexity": quality["mccabe_complexity"],
        "cognitive_complexity": quality["cognitive_complexity"],
        "overall_quality_score": quality["overall_quality_score"],
        "recommendation": quality["recommendation"]
    })

# Save results
with open("output/code_quality_assessment.json", "w") as f:
    json.dump(results, f, indent=2)

# Identify complex functions
complex_functions = [r for r in results if r["mccabe_complexity"] > 10]
if complex_functions:
    print("Complex functions identified:")
    for func in complex_functions:
        print(f"  - {func['filename']}: McCabe={func['mccabe_complexity']}, Cognitive={func['cognitive_complexity']}")
```

### Output Format

The script generates a JSON file with code quality metrics for each file:

```json
[
  {
    "filename": "utils/statistical_testing.py",
    "mccabe_complexity": <calculated_value>,
    "cognitive_complexity": <calculated_value>,
    "overall_quality_score": <calculated_value>,
    "recommendation": "<assessment_text>"
  }
]
```

Run the script to see actual results for your codebase.

### Decision Criteria

- McCabe Complexity < 10: Acceptable
- McCabe Complexity 10-15: Consider refactoring
- McCabe Complexity > 15: Requires refactoring
- Cognitive Complexity < 15: Acceptable
- Cognitive Complexity > 15: Consider simplification

---

## Use Case 6: Adversarial Testing for Security Validation

### Scenario

You need to validate your LLM system against prompt injection attacks and adversarial inputs before production deployment.

### Solution

```python
from llm_evaluation_demo import AdversarialTester

# Initialize adversarial tester
adversarial_tester = AdversarialTester()

# Your model function
def your_model(prompt: str) -> str:
    # Your model implementation
    return f"Response to: {prompt}"

# Test prompts
test_prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "How does RAG work?"
]

# Run adversarial testing
robustness_results = adversarial_tester.test_robustness(
    model=your_model,
    test_prompts=test_prompts,
    n_adversarial=5
)

# Analyze results
print(f"Total adversarial tests: {robustness_results['total_adversarial_tests']}")
print(f"Detected injections: {robustness_results['detected_injections']}")
print(f"Robustness rate: {robustness_results['robustness_rate']:.2%}")

# Identify vulnerable prompts
vulnerable = [r for r in robustness_results['results'] if r['is_vulnerable']]
if vulnerable:
    print("\nVulnerable prompts identified:")
    for v in vulnerable:
        print(f"  - {v['prompt']}: {v['injection_type']}")
```

### Output Format

The script prints a summary showing:

- Total adversarial tests performed
- Number of detected injections
- Robustness rate (percentage)
- List of vulnerable prompts with injection types

Run the script to see actual results with your model.

### Decision Criteria

- Robustness Rate > 90%: Acceptable for production
- Robustness Rate 80-90%: Additional hardening recommended
- Robustness Rate < 80%: Requires security improvements

---

## Summary

These use cases demonstrate practical applications of the LLM Evaluation Framework:

1. **RAG System Evaluation**: Comprehensive assessment before production
2. **Model Comparison**: Statistical validation of improvements
3. **Vector Search Selection**: Informed decision-making with benchmarks
4. **Production Monitoring**: Baseline tracking and continuous improvement
5. **Code Quality Assessment**: Maintainability and refactoring guidance
6. **Security Validation**: Adversarial testing for production readiness

Each use case includes:
- Real-world scenario
- Complete code solution
- Expected outputs
- Decision criteria
- Recommended practices

For more examples, see `examples/` directory for complete scripts and notebooks.

