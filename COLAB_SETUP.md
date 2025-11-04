# Google Colab Setup Guide

## Quick Start

### Option 1: Upload Notebook

1. Open https://colab.research.google.com/
2. File > Upload > Select `small_llm_demo.ipynb`
3. Run first cell (installs dependencies)
4. Enable GPU (optional): Runtime > Change runtime type > GPU

### Option 2: Clone from GitHub

```python
!git clone <repo-url>
%cd llm
!pip install -r requirements.txt
```

## Models Used

All models configured for free Colab tier:

- Text Generation: `gpt2`
- Embeddings: `all-MiniLM-L6-v2`
- Knowledge Distillation: `gpt2` (teacher), `distilgpt2` (student)

## GPU Usage

### Enable GPU

1. Runtime > Change runtime type
2. Hardware Accelerator: GPU
3. Save

### GPU Types

- Free Tier: T4 GPU (16GB VRAM) - sufficient for configured models

## Troubleshooting

### Out of Memory

- Models configured for free tier
- Restart runtime if needed: Runtime > Restart runtime

### Slow Performance

- Enable GPU: Runtime > Change runtime type > GPU

### Model Download Issues

Set HuggingFace token:

```python
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token"
```

### FAISS Installation

Handled automatically in setup cell. No action needed.

## Testing

### Quick Test

1. Upload notebook to Colab
2. Run setup cell
3. Run RAG pipeline cell
4. Verify output

### Full Test

1. Run all cells in order
2. Check for errors
3. Verify GPU (if enabled): Check CUDA message
4. Monitor memory: Runtime > Manage sessions

## Best Practices

1. Models configured for free Colab
2. Monitor memory usage: Runtime > Manage sessions
3. Save progress periodically
4. Enable GPU for faster inference (optional)
5. Restart runtime if issues occur

## Notes

- Free tier compatible
- Models cached in Colab for faster reruns
- Save outputs to Google Drive if needed
- Notebooks can be shared
