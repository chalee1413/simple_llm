# RLHF (Reinforcement Learning from Human Feedback)

## Overview

RLHF aligns language models with human preferences through reinforcement learning. The framework implements a complete RLHF pipeline supporting both PPO (Proximal Policy Optimization) and DPO (Direct Preference Optimization) algorithms.

## Pipeline Stages

### Stage 1: Supervised Fine-Tuning (SFT)

Initial model fine-tuning on instruction-following data.

**Purpose**: Foundation for RLHF training.

**Process**:
1. Load instruction dataset
2. Format data for training
3. Fine-tune base model
4. Save SFT model checkpoint

**Output**: Instruction-following model ready for RLHF.

### Stage 2: Reward Model Training (PPO Path)

Train reward model on human preference data.

**Purpose**: Predict reward scores for model outputs.

**Process**:
1. Load preference data (chosen vs rejected responses)
2. Format for pairwise comparison training
3. Train reward model on preferences
4. Save reward model checkpoint

**Output**: Reward model for PPO training.

### Stage 3: PPO Training

Optimize policy using reward model.

**Purpose**: Improve model alignment with human preferences.

**Process**:
1. Load SFT model and reward model
2. Generate responses for prompts
3. Compute rewards using reward model
4. Optimize policy with PPO algorithm
5. Apply KL divergence penalty for stability

**Output**: RLHF-aligned model.

### Alternative: DPO Training

Direct optimization on preference data.

**Purpose**: Simpler alternative to PPO without separate reward model.

**Process**:
1. Load SFT model and preference data
2. Format preferences for DPO training
3. Optimize policy directly on preferences
4. No separate reward model required

**Output**: RLHF-aligned model.

## Algorithms

### PPO (Proximal Policy Optimization)

**Reference**: Schulman et al. (2017). Proximal Policy Optimization. https://arxiv.org/abs/1707.06347

**Advantages**:
- Handles complex reward functions
- Stable training with KL penalty
- Good for production systems

**Requirements**:
- SFT model
- Reward model
- Training prompts

**Use When**:
- Complex reward functions needed
- Production deployment
- Maximum control over training

### DPO (Direct Preference Optimization)

**Reference**: Rafailov et al. (2024). Direct Preference Optimization. https://arxiv.org/abs/2305.18290

**Advantages**:
- Simpler pipeline (no reward model)
- Faster training
- Direct optimization on preferences

**Requirements**:
- SFT model
- Preference data

**Use When**:
- Simpler workflow preferred
- Faster iteration needed
- Preference data available

## Usage

### Full Pipeline

Run complete RLHF pipeline:

```python
# Edit configuration in rlhf_pipeline.py
PIPELINE_STAGE = 'full'
MODEL_NAME = 'gpt2'  # Small laptop compatible
RLHF_ALGORITHM = 'ppo'  # or 'dpo'

# Run pipeline
python rlhf_pipeline.py
```

### Stage-by-Stage

Run individual stages:

```python
# Stage 1: SFT
PIPELINE_STAGE = 'sft'
python rlhf_pipeline.py

# Stage 2: Reward Model (PPO path)
PIPELINE_STAGE = 'reward'
python rlhf_pipeline.py

# Stage 3: PPO
PIPELINE_STAGE = 'ppo'
python rlhf_pipeline.py

# Alternative: DPO (skips reward model)
PIPELINE_STAGE = 'dpo'
python rlhf_pipeline.py
```

### Examples

Run example workflows:

```python
# Edit configuration in examples/example_rlhf.py
EXAMPLE_MODE = 'full'  # or 'sft', 'preferences', 'reward', 'ppo', 'dpo'

# Run example
python examples/example_rlhf.py
```

## Configuration

### Model Selection

**Small Laptop Compatible** (default):
- `gpt2`: ~124M parameters, ~500MB RAM
- `distilgpt2`: ~82M parameters, ~330MB RAM
- `microsoft/DialoGPT-small`: ~120M parameters, ~480MB RAM

**Medium Systems** (4GB+ RAM):
- `gpt2-medium`: ~350M parameters, ~1.4GB RAM
- `gpt2-large`: ~774M parameters, ~3GB RAM

**Large Systems** (8GB+ RAM):
- `gpt2-xl`: ~1.5B parameters, ~6GB RAM

### Algorithm Selection

**PPO**:
- Requires reward model training
- More complex pipeline
- Better for complex tasks
- Set `RLHF_ALGORITHM = 'ppo'`

**DPO**:
- No reward model needed
- Simpler pipeline
- Faster training
- Set `RLHF_ALGORITHM = 'dpo'`

## Data Requirements

### Instruction Dataset

Format: JSON list of instruction-response pairs

```json
[
  {
    "instruction": "Explain what machine learning is",
    "response": "Machine learning is a subset of AI..."
  }
]
```

**Location**: `data/rlhf/instructions.json`

### Preference Data

Format: JSON list of preference comparisons

```json
[
  {
    "prompt": "Explain what machine learning is",
    "chosen": "Machine learning is a subset of AI...",
    "rejected": "Machine learning is programming."
  }
]
```

**Location**: `data/rlhf/preferences/preferences.json`

**Minimum**: 100 preferences recommended (configurable via `RLHF_MIN_PREFERENCES`)

### Training Prompts

Format: JSON list of prompt strings

```json
[
  "Explain what machine learning is",
  "What is the difference between supervised and unsupervised learning?"
]
```

**Location**: `data/rlhf/prompts.json`

## Integration

### Evaluation Framework

RLHF pipeline integrates with existing evaluation framework:

- **LLM-as-Judge**: Evaluate model outputs
- **Baseline Tracking**: Compare model versions
- **Statistical Testing**: Validate improvements

### Baseline Comparison

Save and compare RLHF training results:

```python
# In rlhf_pipeline.py configuration
SAVE_BASELINE = 'v1.0'  # Save current results
COMPARE_BASELINE = 'v1.0'  # Compare with baseline
```

## Performance

### Small Laptop Compatibility

**Requirements**:
- 4GB+ RAM (8GB+ recommended)
- CPU-only execution
- Lightweight models (gpt2, distilgpt2)

**Expected Performance**:
- SFT: ~5-10 minutes per epoch (100 examples)
- Reward Model: ~3-5 minutes per epoch (100 preferences)
- PPO: ~10-15 minutes per epoch (10 prompts)
- DPO: ~5-10 minutes per epoch (100 preferences)

### Memory Optimization

Framework includes:
- Lazy model loading
- Memory-efficient training
- Gradient accumulation
- Checkpointing

## Troubleshooting

### Out of Memory

**Solutions**:
- Use smaller model (gpt2, distilgpt2)
- Reduce batch size
- Increase gradient accumulation steps
- Use CPU-only execution

### Slow Training

**Solutions**:
- Use smaller model
- Reduce dataset size for testing
- Use DPO instead of PPO (faster)
- Reduce number of epochs

### Model Not Improving

**Solutions**:
- Increase preference data quality
- Check data formatting
- Adjust learning rates
- Increase training epochs

## References

- Schulman et al. (2017). Proximal Policy Optimization. https://arxiv.org/abs/1707.06347
- Rafailov et al. (2024). Direct Preference Optimization. https://arxiv.org/abs/2305.18290
- TRL Library: https://github.com/huggingface/trl
- HuggingFace Transformers: https://huggingface.co/docs/transformers
