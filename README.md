# LLM Evaluation Framework

> Production-ready evaluation framework for Large Language Models following 2025 SoTA best practices.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## Overview

Complete evaluation pipeline for LLMs covering RAG systems, model compression, safety evaluation, and quality assessment. Implements state-of-the-art methodologies from recent research (2024-2025) with production-ready code, statistical rigor, and comprehensive benchmarking.

### Key Features

- **RAG Evaluation**: RAGAs framework (faithfulness, answer relevancy, context precision/recall)
- **Safety Assessment**: Toxicity detection, adversarial testing, prompt injection detection
- **Quality Metrics**: Code quality assessment (McCabe, cognitive complexity)
- **LLM-as-Judge**: Statistical significance testing with paired comparisons
- **Baseline Tracking**: Version comparison and improvement metrics over time
- **Vector Search Benchmarking**: Comprehensive comparison (FAISS, Qdrant, Chroma, PostgreSQL+pgvector)
- **Statistical Rigor**: Bootstrap confidence intervals, paired t-tests, effect size calculation

### Problem Statement

Evaluating LLMs requires multiple dimensions: semantic quality, safety, code quality, and statistical validation. Existing frameworks are either too narrow (single metric) or too complex (enterprise-only). This framework provides a complete, production-ready solution following 2025 SoTA practices.

### Solution

A comprehensive evaluation framework that:
- Implements multiple SoTA methodologies (RAGAs, LLM-as-Judge, Knowledge Distillation)
- Provides statistical validation for all comparisons
- Supports baseline tracking for iterative improvement
- Benchmarks vector search solutions for informed decisions
- Works with open-source models (no API keys required)

## Architecture

The framework consists of multiple evaluation components working together:

- **RAG Evaluation**: RAGAs framework metrics (faithfulness, answer relevancy, context precision/recall)
- **Safety Evaluation**: Toxicity detection and adversarial testing
- **Quality Evaluation**: Code quality assessment (McCabe, cognitive complexity)
- **LLM-as-Judge**: Statistical significance testing with paired comparisons
- **Statistical Testing**: Bootstrap confidence intervals, paired t-tests
- **Baseline Tracking**: Version comparison and improvement metrics
- **Vector Search Benchmarking**: Comprehensive comparison across libraries, databases, and RDBMS

See [INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) for detailed architecture documentation.

## Portfolio Highlights

### SoTA Methodologies Implemented

- **RAGAs Framework** (2024): Retrieval-Augmented Generation Assessment with faithfulness, answer relevancy, context precision, and context recall metrics
- **LLM-as-Judge** (Zheng et al., 2024): Statistical significance testing for LLM output evaluation
- **Knowledge Distillation** (Hinton et al., 2015): Teacher-student model comparison with inference time benchmarking
- **Statistical Testing**: Bootstrap confidence intervals (Efron, 1979), paired t-tests, effect size calculation

### Performance Benchmarks

Vector search comparison across multiple dimensions:
- Query performance (latency, throughput)
- Ingestion rate (vectors per second)
- Scalability (performance vs dataset size)
- Feature comparison (metadata filtering, persistence, ACID transactions)

See [VECTOR_SEARCH_COMPARISON.md](docs/VECTOR_SEARCH_COMPARISON.md) for detailed results.

### Key Results

Benchmark results vary by dataset size and hardware. Run the benchmark script to see actual performance metrics for your environment:

```bash
python vector_search_comparison.py --sizes 1000 5000 10000 --queries 50 --k 10 --visualize
```

The benchmark measures query performance, ingestion rate, scalability, and feature support across all solutions.

## Components

### small_llm_demo.ipynb

RAG and model management notebook demonstrating:
- RAG Pipeline: FAISS-based semantic search with entity resolution
- Knowledge Distillation: Teacher-student model comparison with statistical testing
- Model Hub Management: HuggingFace model discovery and multi-model comparison

### llm_evaluation_demo.py

Comprehensive evaluation script providing:
- Toxicity Detection: Context-aware analysis with semantic similarity
- Code Quality Assessment: McCabe and cognitive complexity metrics
- Adversarial Testing: Prompt injection detection and security validation
- LLM-as-Judge: Statistical significance testing with paired comparisons
- Baseline Tracking: Save and compare evaluation results over time
- Improvement Metrics: Quantify improvements with statistical validation

### vector_search_comparison.py

Comprehensive vector search benchmarking:
- Performance comparison: Query speed, ingestion rate, scalability
- Feature comparison: Metadata filtering, persistence, ACID transactions
- Multiple solutions: FAISS, NumPy, Scikit-learn, Qdrant, Chroma, PostgreSQL+pgvector
- Visualization: Generate charts and dashboards (PNG/PDF)

### visualize_benchmarks.py

Visualization script for benchmark results:
- Performance charts: Query latency, throughput, scalability
- Feature comparison: Feature matrix visualization
- Summary dashboard: Combined metrics visualization

## Quick Start

### One-Command Demo (Google Colab)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `small_llm_demo.ipynb` to Google Colab
2. Run first cell (auto-installs dependencies)
3. Execute notebook cells

### Local Setup (3 Steps)

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run evaluation
python llm_evaluation_demo.py --input-file data/test_samples.json
```

### Output Format

The script generates a JSON file with evaluation results and prints a summary:

```
EVALUATION SUMMARY
================================================================================
Results saved to: output/evaluation_results_<timestamp>.json
Toxicity tests: <number>
Code quality tests: <number>
Adversarial tests: <number>
LLM-as-Judge: Mean score 1=<value>, Mean score 2=<value>
```

Run the script to see actual results.

## Installation

```bash
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Default configuration uses open-source models from HuggingFace. No API keys required.

```bash
# Default: Open-source models (free Colab compatible)
USE_OPENSOURCE_MODELS=true
HF_LLM_MODEL=gpt2

# Optional customization
EVAL_BATCH_SIZE=8
STATISTICAL_CONFIDENCE_LEVEL=0.95
```

### Optional: API-Based Models

For OpenAI or AWS Bedrock, copy `env.example` to `.env` and configure:

```bash
cp env.example .env
# Edit .env with your API keys
```

## Usage

### Notebook

```bash
jupyter notebook small_llm_demo.ipynb
```

### Evaluation Script

```bash
# Basic evaluation
python llm_evaluation_demo.py --input-file data/test_samples.json --output-dir output/

# Save baseline for future comparison
python llm_evaluation_demo.py --input-file data/test_samples.json --save-baseline v1.0

# Compare against baseline
python llm_evaluation_demo.py --input-file data/test_samples.json --compare-baseline v1.0

# Compare two baseline versions
python llm_evaluation_demo.py --compare-versions v1.0 v2.0
```

### Vector Search Comparison

```bash
# Compare FAISS vs Qdrant vs Chroma (with visualizations)
python vector_search_comparison.py --sizes 1000 5000 10000 --queries 50 --k 10 --visualize

# Custom dimensions
python vector_search_comparison.py --dimension 768 --sizes 1000 10000

# Generate visualizations from existing CSV
python visualize_benchmarks.py --input output/vector_search_benchmark_*.csv
```

### Benchmark Visualization

```bash
# Generate charts from benchmark CSV
python visualize_benchmarks.py --input output/vector_search_benchmark_*.csv --format png

# Generate both PNG and PDF
python visualize_benchmarks.py --input output/vector_search_benchmark_*.csv --format both
```

## Documentation

- [INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) - Architecture decisions and technology stack
- [METHODOLOGY.md](docs/METHODOLOGY.md) - Algorithm selection and research citations
- [DATA_STRATEGY.md](docs/DATA_STRATEGY.md) - Data source selection and model criteria
- [CHALLENGES.md](docs/CHALLENGES.md) - Technical challenges and solutions
- [VECTOR_SEARCH_COMPARISON.md](docs/VECTOR_SEARCH_COMPARISON.md) - FAISS vs Qdrant vs Chroma vs PostgreSQL+pgvector benchmarks
- [EXAMPLES.md](docs/EXAMPLES.md) - Real-world use cases and examples
- [TUTORIAL.md](docs/TUTORIAL.md) - Step-by-step tutorial guide

## Examples

See `examples/` directory for complete scripts and notebooks:

- `example_rag_evaluation.py` - RAG system evaluation example
- `example_model_comparison.py` - Model comparison with statistical validation
- `example_baseline_tracking.py` - Baseline tracking for production monitoring
- `example_vector_search.py` - Vector search benchmark example

Run examples:

```bash
# RAG evaluation
python examples/example_rag_evaluation.py

# Model comparison
python examples/example_model_comparison.py

# Baseline tracking
python examples/example_baseline_tracking.py

# Vector search benchmark
python examples/example_vector_search.py
```

## SoTA Practices

Implements 2025 SoTA approaches:

- **RAGAs Framework**: RAG evaluation metrics (faithfulness, answer relevancy, context precision/recall)
- **Statistical Rigor**: Bootstrap confidence intervals and paired t-tests
- **LLM-as-Judge**: Evaluation methodology with statistical significance testing
- **Baseline Tracking**: Comparison and improvement metrics over time
- **Vector Search Benchmarking**: Comprehensive comparison across libraries, databases, and RDBMS
- **Enterprise Integration**: AWS Bedrock and OpenAI API with error handling

## Project Structure

```
llm/
├── config.py                 # Configuration management
├── requirements.txt          # Dependencies
├── env.example               # Environment variables template
├── small_llm_demo.ipynb      # RAG and model management
├── llm_evaluation_demo.py   # Evaluation script
├── vector_search_comparison.py # Vector search benchmarks
├── visualize_benchmarks.py  # Visualization script
├── docker-compose.yml       # PostgreSQL + pgvector setup
├── utils/
│   ├── evaluation_metrics.py    # RAGAs metrics
│   ├── statistical_testing.py   # Statistical utilities
│   └── baseline_tracking.py     # Baseline comparison and improvement tracking
├── examples/                # Example scripts and notebooks
├── baselines/                # Saved baseline results
├── output/                   # Evaluation results and charts
└── docs/                     # Documentation
```

## Requirements

- Python 3.11+
- Virtual environment (recommended)
- Memory: 4GB+ available RAM (8GB+ recommended for LLM-as-Judge)
- Optional: API keys for OpenAI or AWS Bedrock
- Optional: Docker (for PostgreSQL + pgvector benchmarking)

## Known Issues and Limitations

### Memory Requirements

- LLM-as-Judge evaluation requires significant memory (4GB+ available RAM)
- Process may crash on systems with <4GB available RAM
- For low-memory systems: Run evaluations separately (e.g., `--evaluation-type toxicity`, `--evaluation-type code-quality`)
- For production: Use API-based providers (`--llm-provider openai` or `bedrock`)

### Test Results

Based on actual testing:

- **Toxicity Detection**: Working (0.0 scores, non-toxic classification)
- **Code Quality Assessment**: Working (McCabe: 1.0-4.0, Cognitive: 1.5-4.5)
- **Adversarial Testing**: Working (pattern-based detection)
- **LLM-as-Judge**: Requires model loading (can crash on low-memory systems)

See [CHALLENGES.md](docs/CHALLENGES.md) for detailed information on memory issues and solutions.

## Google Colab Setup

See [COLAB_SETUP.md](COLAB_SETUP.md) for detailed instructions.

**Quick Start**:

1. Upload `small_llm_demo.ipynb` to Google Colab
2. Run first cell (auto-installs dependencies)
3. Enable GPU (optional): Runtime > Change runtime type > GPU

### Models Used

All models configured for free Colab tier:

- Text Generation: `gpt2`
- Embeddings: `all-MiniLM-L6-v2`
- Knowledge Distillation: `gpt2` (teacher), `distilgpt2` (student)

## License

MIT License - see LICENSE file for details.

## References

- RAGAs: Retrieval-Augmented Generation Assessment (2024). Es et al. https://arxiv.org/abs/2312.10997
- LLM-as-Judge: Zheng et al. (2024). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. https://arxiv.org/abs/2306.05685
- Knowledge Distillation: Hinton et al. (2015). Distilling the Knowledge in a Neural Network. https://arxiv.org/abs/1503.02531
- Bootstrap Methods: Efron, B. (1979). Bootstrap methods: Another look at the jackknife. Annals of Statistics, 7(1), 1-26.
