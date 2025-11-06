# LLM Evaluation Framework

> Evaluation framework for Large Language Models following 2025 SoTA best practices.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## TL;DR

LLM evaluation framework with 2025 SoTA methodologies. Evaluates RAG systems (RAGAs metrics), model safety (toxicity detection), code quality (McCabe complexity), and includes statistical validation (bootstrap confidence intervals, paired t-tests). Also includes complete RLHF pipeline (SFT, reward model, PPO/DPO/KTO) to train AI models to follow human preferences. Works with open-source models (no API keys required), runs on small laptops, and provides baseline tracking for iterative improvement.

**Quick Start**: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python llm_evaluation_demo.py --input-file example_inputs/test_samples.json`

## Overview

Complete evaluation pipeline for LLMs covering RAG systems, model compression, safety evaluation, and quality assessment. Implements state-of-the-art methodologies from recent research (2024-2025) with statistical rigor and comprehensive benchmarking.

### Key Features

- **RAG Evaluation**: RAGAs framework (faithfulness, answer relevancy, context precision/recall)
- **Safety Assessment**: Toxicity detection, adversarial testing, prompt injection detection
- **Quality Metrics**: Code quality assessment (McCabe, cognitive complexity)
- **LLM-as-Judge**: Statistical significance testing with paired comparisons
- **RLHF Pipeline**: Complete reinforcement learning from human feedback (SFT, reward model, PPO/DPO/KTO) - TL;DR: Trains AI to follow human preferences using reinforcement learning
- **Baseline Tracking**: Version comparison and improvement metrics over time
- **Vector Search Benchmarking**: Performance comparison across FAISS, Qdrant, Chroma, PostgreSQL+pgvector, NumPy, and Scikit-learn. See [VECTOR_SEARCH_COMPARISON.md](docs/VECTOR_SEARCH_COMPARISON.md) for detailed analysis.
- **Statistical Rigor**: Bootstrap confidence intervals, paired t-tests, effect size calculation

### Problem Statement

Evaluating LLMs requires multiple dimensions: semantic quality, safety, code quality, and statistical validation. Existing frameworks are either too narrow (single metric) or too complex (enterprise-only). This framework provides a complete solution following 2025 SoTA practices.

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
- **RLHF Pipeline**: Complete RLHF pipeline (SFT, reward model, PPO/DPO/KTO) with small laptop compatibility
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

Vector search comparison across multiple dimensions: query performance, ingestion rate, scalability, resource usage, and feature support. Experimental results averaged over 3 iterations for statistical significance.

| Method | Scalability (1K→10K) | Best For |
|--------|---------------------|----------|
| PostgreSQL+pgvector | 1.09x | Large datasets, production |
| Chroma | 1.24x | Large datasets (>10K) |
| Scikit-learn | 6.70x | Medium datasets (5K-10K) |
| Qdrant | 6.47x | Small-medium datasets (<10K) |
| NumPy | 26.39x | Small datasets (<5K) |

See [VECTOR_SEARCH_COMPARISON.md](docs/VECTOR_SEARCH_COMPARISON.md) for detailed analysis, including algorithm complexity analysis, verified citations, architecture comparisons, and performance trade-offs.

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

Vector search benchmarking across multiple solutions: FAISS, Qdrant, Chroma, PostgreSQL+pgvector, NumPy, and Scikit-learn. Measures query performance, ingestion rate, scalability, resource usage, and feature support. Generates performance visualizations and detailed analysis reports.

See [VECTOR_SEARCH_COMPARISON.md](docs/VECTOR_SEARCH_COMPARISON.md) for algorithm complexity analysis, verified citations, and performance trade-offs.

### rlhf_pipeline.py

Complete RLHF pipeline script:
- Supervised Fine-Tuning (SFT): Instruction-following model training
- Reward Model Training: Human preference learning for PPO
- PPO Training: Policy optimization with reward model
- DPO Training: Direct preference optimization (alternative to PPO)
- KTO Training: Kahneman-Tversky Optimization with binary feedback (alternative to DPO)
- Small Laptop Compatible: Lightweight models (gpt2, distilgpt2)
- Configuration-based: Edit configuration section to customize

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
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run evaluation (create your own test_samples.json or use example_inputs)
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json
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
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
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
# Basic evaluation (create your own test_samples.json or use example_inputs)
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --output-dir output/

# Save baseline for future comparison
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --save-baseline v1.0

# Compare against baseline
python llm_evaluation_demo.py --input-file example_inputs/test_samples.json --compare-baseline v1.0

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

### RLHF Pipeline

```bash
# Full pipeline (edit configuration in rlhf_pipeline.py first)
python rlhf_pipeline.py

# Run examples
python examples/example_rlhf.py
```

Configuration: Edit the configuration section in `rlhf_pipeline.py`:
- `PIPELINE_STAGE`: 'sft', 'reward', 'ppo', 'dpo', 'kto', or 'full'
- `MODEL_NAME`: Model name (default: gpt2 for small laptop compatibility)
- `RLHF_ALGORITHM`: 'ppo', 'dpo', or 'kto'

See [RLHF.md](docs/RLHF.md) for complete documentation.

## Documentation

- [INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) - Architecture decisions and technology stack
- [METHODOLOGY.md](docs/METHODOLOGY.md) - Algorithm selection and research citations
- [DATA_STRATEGY.md](docs/DATA_STRATEGY.md) - Data source selection and model criteria
- [CHALLENGES.md](docs/CHALLENGES.md) - Technical challenges and solutions
- [RLHF.md](docs/RLHF.md) - RLHF pipeline documentation (SFT, reward model, PPO/DPO/KTO)
- [VECTOR_SEARCH_COMPARISON.md](docs/VECTOR_SEARCH_COMPARISON.md) - Vector search performance comparison across FAISS, Qdrant, Chroma, PostgreSQL+pgvector, NumPy, and Scikit-learn. Includes experimental results, scalability analysis, algorithm complexity analysis, verified citations, and performance trade-offs.
- [EXAMPLES.md](docs/EXAMPLES.md) - Real-world use cases and examples
- [TUTORIAL.md](docs/TUTORIAL.md) - Step-by-step tutorial guide

## Examples

See `examples/` directory for complete scripts and notebooks:

- `example_rag_evaluation.py` - RAG system evaluation example
- `example_model_comparison.py` - Model comparison with statistical validation
- `example_baseline_tracking.py` - Baseline tracking for production monitoring
- `example_vector_search.py` - Vector search benchmark example
- `example_rlhf.py` - Complete RLHF pipeline example

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
- **RLHF Pipeline**: Complete RLHF pipeline (SFT, reward model, PPO/DPO/KTO) with small laptop compatibility
- **Baseline Tracking**: Comparison and improvement metrics over time
- **Vector Search Benchmarking**: Comprehensive comparison across libraries, databases, and RDBMS
- **Enterprise Integration**: AWS Bedrock and OpenAI API with error handling

## Project Structure

```
simple_llm/
├── config.py                      # Configuration management
├── requirements.txt               # Dependencies
├── env.example                    # Environment variables template
├── small_llm_demo.ipynb           # RAG and model management
├── llm_evaluation_demo.py        # Evaluation script
├── vector_search_comparison.py    # Vector search benchmarks
├── visualize_benchmarks.py        # Visualization script
├── rlhf_pipeline.py              # RLHF pipeline script
├── test_rlhf_comprehensive.py    # RLHF comprehensive test suite
├── evaluate_rlhf_effectiveness.py # RLHF effectiveness evaluation
├── docker-compose.yml             # PostgreSQL + pgvector setup
├── rlhf/                          # RLHF module
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── supervised_finetuning.py
│   ├── preference_collection.py
│   ├── reward_model.py
│   ├── ppo_trainer.py
│   ├── dpo_trainer.py
│   ├── kto_trainer.py
│   └── evaluation_metrics.py
├── utils/
│   ├── __init__.py
│   ├── evaluation_metrics.py    # RAGAs metrics
│   ├── statistical_testing.py    # Statistical utilities
│   └── baseline_tracking.py       # Baseline comparison and improvement tracking
├── example_inputs/                # Example input files (tracked in git)
│   └── rlhf/
│       ├── instructions.json
│       ├── prompts.json
│       ├── kto_feedback.json
│       └── preferences/
│           └── preferences.json
├── examples/                      # Example scripts
│   ├── example_rlhf.py
│   ├── example_rag_evaluation.py
│   ├── example_model_comparison.py
│   ├── example_baseline_tracking.py
│   └── example_vector_search.py
├── data/                          # Runtime data (ignored by git)
├── baselines/                     # Saved baseline results
├── output/                        # Evaluation results and charts
└── docs/                          # Documentation
```

## Requirements

- Python 3.11+
- Virtual environment (recommended)
- Memory: 4GB+ available RAM (8GB+ recommended for LLM-as-Judge, RLHF)
- Optional: API keys for OpenAI or AWS Bedrock
- Optional: Docker (for PostgreSQL + pgvector benchmarking)

### RLHF Requirements

- Small Laptop Compatible: Default models (gpt2, distilgpt2) work on 4GB+ RAM
- Configuration-based: Edit configuration section in `rlhf_pipeline.py` to customize
- No API keys required: Uses open-source models from HuggingFace

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
- **LLM-as-Judge**: Requires model loading 

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
- PPO: Schulman et al. (2017). Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347
- DPO: Rafailov et al. (2024). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. https://arxiv.org/abs/2305.18290
- KTO: Ethayarajh et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization. https://arxiv.org/abs/2402.01306
- Bootstrap Methods: Efron, B. (1979). Bootstrap methods: Another look at the jackknife. Annals of Statistics, 7(1), 1-26.
