# LLM From Scratch

Building core LLM components from scratch to understand the infrastructure behind large language models.

**[Start here: End-to-end overview](docs/00-overview.md)**

## Topics

- **[Tokenization](docs/01-tokenization.md)** — BPE and subword tokenizers
- **[Embeddings](docs/02-embeddings.md)** — Token and positional embeddings
- **[Attention](docs/03-attention.md)** — Scaled dot-product and multi-head attention
- **[Transformer blocks](docs/04-transformer-block.md)** — Layer norm, feed-forward networks, residual connections
- **[Decoder](docs/05-decoder-and-generation.md)** — Autoregressive text generation with causal masking
- **[Training loop](docs/06-training.md)** — Loss functions, optimizers, and gradient flow

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

## Demos

Each component has a runnable demo:

```bash
python demo_tokenizer.py          # Train BPE and inspect merges
python demo_compression_curve.py  # Vocab size vs compression
python demo_embeddings.py         # Learned vs sinusoidal comparison
python demo_attention.py          # Attention weight visualization
python demo_transformer.py        # Layer-by-layer vector evolution
python demo_generation.py         # Full pipeline (untrained)
python demo_train.py              # Train on WikiMed and watch it learn
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
