# Methodology Documentation

## Algorithm Selection Rationale

Methodology and rationale for algorithm selection in the LLM Evaluation Framework.

## RAGAs Metrics

### Faithfulness

**Algorithm**: Semantic similarity between answer and context chunks.

**Formula**:
- Encode answer and context chunks using sentence transformers
- Calculate cosine similarity between answer and each context chunk
- Faithfulness = max(similarity) across all context chunks

**References**: 
- RAGAs: Retrieval-Augmented Generation Assessment (2024)
  Es, S., S Parthasarathy, S., Talukdar, P., et al.
  https://arxiv.org/abs/2312.10997

**Rationale**: Semantic similarity captures answer-context alignment. More accurate than keyword matching, captures semantic relationships, standard in production RAG systems.

**Alternatives**: Keyword overlap (too simplistic), BLEU score (not suitable), Custom NLI models (more complex, similar accuracy)

### Answer Relevancy

**Algorithm**: Semantic similarity between answer and question.

**Formula**:
- Encode answer and question using sentence transformers
- Calculate cosine similarity
- Answer relevancy = similarity score (normalized to 0-1)

**References**: RAGAs: Retrieval-Augmented Generation Assessment (2024). Es et al. https://arxiv.org/abs/2312.10997

**Rationale**: Captures semantic relevance beyond keyword matching, validated in research, standard in production.

### Context Precision

**Algorithm**: Proportion of relevant context chunks.

**Formula**:
- Calculate similarity between reference (question or ground truth) and each context chunk
- Count chunks above relevance threshold (default: 0.5)
- Context precision = relevant_chunks / total_chunks

**References**: RAGAs: Retrieval-Augmented Generation Assessment (2024). Es et al. https://arxiv.org/abs/2312.10997

**Rationale**: Measures retrieval quality, threshold-based for interpretability, standard in RAG evaluation.

### Context Recall

**Algorithm**: Semantic similarity between combined context and ground truth.

**Formula**:
- Combine context chunks into single text
- Calculate similarity between combined context and ground truth
- Context recall = similarity score (normalized to 0-1)

**References**: RAGAs: Retrieval-Augmented Generation Assessment (2024). Es et al. https://arxiv.org/abs/2312.10997

**Rationale**: Measures retrieval completeness, requires ground truth for accuracy, standard in RAG evaluation.

## Statistical Testing

### Paired T-Test

**Algorithm**: Paired t-test for dependent samples comparison.

**Formula**: t = (mean(diff)) / (std(diff) / sqrt(n)) where diff = sample1 - sample2

**References**:
- Paired t-test methodology: https://en.wikipedia.org/wiki/Student%27s_t-test#Paired_samples
- Statistical significance testing for LLM evaluation (2024-2025)

**Rationale**: Appropriate for dependent samples (same evaluation set), standard statistical test, provides p-values and confidence intervals.

**Alternatives**: Independent t-test (less powerful), Mann-Whitney U test (non-parametric), Wilcoxon signed-rank test (non-parametric)

### Bootstrap Confidence Intervals

**Algorithm**: Bootstrap resampling for confidence intervals.

**Methodology**:
1. Resample data with replacement n_iterations times
2. Calculate statistic for each resample
3. Use percentile method to get confidence interval

**References**:
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. Annals of Statistics, 7(1), 1-26.
- Modern applications: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

**Rationale**: No distribution assumptions, robust for small samples, standard in modern evaluation frameworks.

**Alternatives**: Parametric confidence intervals (require distribution assumptions), Jackknife (less robust), Bayesian methods (more complex)

## Knowledge Distillation

### Teacher-Student Architecture

**Algorithm**: Transfer knowledge from large teacher model to small student model.

**Methodology**:
- Train student model to match teacher predictions
- Use soft labels from teacher model
- Combine with hard labels for accuracy
- Statistical testing for performance comparison

**References**: 
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network.
  https://arxiv.org/abs/1503.02531

**Rationale**: Standard approach for model compression, proven effective, enables deployment of smaller models, cost-effective.

**Alternatives**: Model pruning (different approach), Quantization (reduces precision), Architecture search (more complex)

## LLM-as-Judge

### Evaluation Framework

**Algorithm**: Use LLM to evaluate outputs of other models.

**Methodology**:
- Construct evaluation prompt with criteria
- Use LLM to judge outputs
- Extract scores and reasoning
- Statistical testing for reliability

**References**: 
- Zheng, L., Chiang, W., Sheng, Y., et al. (2024). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.
  https://arxiv.org/abs/2306.05685

**Rationale**: Cost-effective alternative to human evaluation, scalable, standard in current research, provides reasoning.

**Alternatives**: Human evaluation (more accurate, expensive), Traditional metrics (less semantic understanding), Custom evaluation models (more complex)

## Baseline Tracking and Improvement Metrics

### Baseline Comparison

**Algorithm**: Compare current evaluation results against saved baseline.

**Methodology**:
- Save evaluation results as baseline with metadata
- Calculate improvement metrics (absolute and relative)
- Statistical significance testing for improvements
- Generate improvement reports

**Rationale**: Enables tracking improvements over time, validates changes with statistical rigor, supports iterative development.

**References**: Baseline comparison methodologies (2024-2025 best practices), Statistical significance testing for LLM evaluation.

### Improvement Metrics

**Algorithm**: Calculate improvement compared to baseline.

**Formula**:
- Improvement (absolute) = current_mean - baseline_mean
- Improvement (relative) = (improvement_absolute / baseline_mean) * 100
- Statistical significance: paired t-test between baseline and current results

**Rationale**: Quantifies improvements, validates changes statistically, supports evidence-based decision making.

**Alternatives**: Simple comparison (no statistical validation), Regression testing (different approach), A/B testing (more complex)

## Implementation Details

### Formulas

**Faithfulness**: similarity = cosine_similarity(encode(answer), encode(context_chunk)); faithfulness = max(similarities)

**Answer Relevancy**: relevancy = cosine_similarity(encode(answer), encode(question))

**Context Precision**: similarities = cosine_similarity(encode(reference), encode(context_chunks)); precision = count(similarities >= threshold) / len(context_chunks)

**Context Recall**: recall = cosine_similarity(encode(combined_context), encode(ground_truth))

**Paired T-Test**: t = mean(diff) / (std(diff) / sqrt(n)); p-value from t-distribution

**Bootstrap CI**: resamples = [resample(data) for _ in range(n_iterations)]; ci = [percentile(resamples, alpha/2), percentile(resamples, 1-alpha/2)]

### Parameters

- Relevance Threshold: 0.5 (for context precision)
- Confidence Level: 0.95 (for statistical tests)
- Bootstrap Iterations: 1000
- Toxicity Threshold: 0.7 (configurable)

### Validation

**Verification**: Unit tests for all metric calculations, validation against known examples, comparison with reference implementations, statistical validation.

**Testing Strategy**: Unit tests for individual metrics, integration tests for pipelines, validation against research benchmarks, manual inspection of edge cases.
