# LLM From Scratch

Building core LLM components from scratch to understand the infrastructure behind large language models.

## Topics

- **Tokenization** — BPE and subword tokenizers
- **Embeddings** — Token and positional embeddings
- **Attention** — Scaled dot-product and multi-head attention
- **Transformer blocks** — Layer norm, feed-forward networks, residual connections
- **Decoder** — Autoregressive text generation with causal masking
- **Training loop** — Loss functions, optimizers, and gradient flow

## Structure

```
src/
  tokenizer.py      # Byte-pair encoding tokenizer
  embeddings.py      # Token + positional embeddings
  attention.py       # Scaled dot-product & multi-head attention
  transformer.py     # Transformer block (decoder)
  model.py           # Full GPT-style language model
  train.py           # Training loop
  generate.py        # Text generation / inference
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Goals

This is a learning project. The priority is clarity over performance — every component
is written to be readable and well-understood, not to compete with production frameworks.
