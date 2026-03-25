# How a Language Model Works: End to End

This document ties together all six components into one complete picture. Each
section links to its detailed doc.

## The pipeline

A language model does one thing: given a sequence of tokens, predict the next one.
Everything else — conversations, code generation, translation — emerges from doing
this well enough at scale.

```
"The cat sat on the" → model → "mat" (with 73% probability)
```

Here is every step that happens between input and output:

```
    Raw text: "The cat sat on the"
         │
         ▼
┌─────────────────────┐
│  1. TOKENIZATION    │  Split text into token IDs
│     (01-tokenization)│  "The cat" → [84, 104, 101, 32, 99, 97, 116]
└────────┬────────────┘
         │  sequence of integers
         ▼
┌─────────────────────┐
│  2. EMBEDDINGS      │  Look up each ID in a table → vector
│     (02-embeddings) │  Add positional information
│                     │  Token 84 → [0.12, -0.83, 0.45, ...]
│                     │  + pos 0  → [0.33,  0.10, -0.62, ...]
└────────┬────────────┘
         │  sequence of d_model-dimensional vectors
         ▼
┌─────────────────────┐
│  3. TRANSFORMER     │  Repeat N times:
│     BLOCKS          │
│  ┌────────────────┐ │    a. Attention: tokens look at each other
│  │  Attention     │ │       Q·K scores → softmax → weighted sum of V
│  │  (03-attention)│ │       (causal mask: no looking ahead)
│  └───────┬────────┘ │
│  ┌───────▼────────┐ │    b. Feed-forward: each token processes
│  │  Feed-Forward  │ │       independently (expand → GELU → compress)
│  │  (04-transform)│ │
│  └───────┬────────┘ │    Both wrapped in residual connections
│          │          │    and layer norm
│  (04-transformer-   │
│      block)         │
└────────┬────────────┘
         │  sequence of context-aware vectors
         ▼
┌─────────────────────┐
│  5. OUTPUT HEAD     │  Project each vector to vocab_size scores
│  (05-decoder-and-   │  vector → [score_for_"a", score_for_"b", ...]
│      generation)    │  Softmax → probabilities
└────────┬────────────┘
         │  probability distribution over vocabulary
         ▼
┌─────────────────────┐
│  6. SAMPLING        │  Pick a token from the distribution
│  (05-decoder-and-   │  Temperature and top-k control randomness
│      generation)    │  Append token, repeat from step 2
└────────┬────────────┘
         │
         ▼
    Output: "mat"
```

## What each component contributes

**Tokenization** converts raw text into integers the model can process. BPE learns
common patterns from data — frequent subwords become single tokens, compressing
the sequence. Shorter sequences mean faster attention (which is quadratic in length)
and longer effective context.
→ [Detailed doc](01-tokenization.md)

**Embeddings** convert token IDs into vectors. The token embedding captures identity
("what is this token?") and the positional embedding captures order ("where is it in
the sequence?"). These are added together and form the model's initial representation
of each token. Both are learned during training.
→ [Detailed doc](02-embeddings.md)

**Attention** is the communication mechanism. Each token produces a Query ("what am I
looking for?"), Key ("what do I contain?"), and Value ("what do I share?"). The dot
product Q·K determines relevance, softmax normalizes it to weights, and the weighted
sum of V blends information across tokens. Multi-head attention runs this multiple
times in parallel, each head capturing a different type of relationship.
→ [Detailed doc](03-attention.md)

**Transformer blocks** combine attention (inter-token communication) with
feed-forward networks (per-token computation). Layer normalization keeps values
stable. Residual connections preserve information and enable gradient flow through
deep stacks. The same block architecture repeats N times, each with its own learned
weights, progressively refining the representation.
→ [Detailed doc](04-transformer-block.md)

**The decoder** maps refined vectors back to vocabulary space. A linear projection
produces a score for every token in the vocabulary. Weight tying shares this matrix
with the input embedding — the same mapping between tokens and vectors used in both
directions. Autoregressive generation feeds each predicted token back as input,
building the output one token at a time.
→ [Detailed doc](05-decoder-and-generation.md)

**Training** is how random weights become a language model. The model predicts the
next token at every position, cross-entropy loss measures how wrong it is,
backpropagation computes gradients for every weight, and the optimizer nudges them
to reduce the loss. After enough iterations across enough text, the model has
learned the statistical structure of language.
→ [Detailed doc](06-training.md)

## What's learned vs what's designed

Almost nothing in the model is hand-coded knowledge about language:

**Designed (by us):**
- The architecture: attention + feed-forward in a loop
- Hyperparameters: d_model, n_heads, n_layers, vocab_size
- The training objective: predict the next token

**Learned (from data):**
- What each token's embedding means
- What Q, K, V projections look for
- Which tokens attend to which (syntactic, semantic, positional patterns)
- What the feed-forward layers compute
- How to map vectors back to token predictions

The only inductive bias is the structure: "tokens should be able to look at other
tokens (attention) and then process that information (feed-forward), repeatedly."
Everything else emerges from training on enough text.

## Scale is what matters

Our tiny model (120K parameters, 2 layers, 200KB of training data) learned that
medical text has spaces, newlines, and fragments like "ine" and "amine." It went
from `"ttttttttt"` to structured gibberish in 20 seconds.

The same architecture at scale:

| | Our model | GPT-2 | GPT-3 | GPT-4 (est.) |
|---|---|---|---|---|
| Parameters | 120K | 1.5B | 175B | ~1.8T |
| Layers | 2 | 48 | 96 | ~120 |
| d_model | 64 | 1,600 | 12,288 | ~13,000 |
| Training data | 200KB | 40GB | 570GB | ~13TB |
| Training time | 20 seconds | days | weeks | months |
| Output | gibberish | coherent text | essays, code | reasoning |

The architecture is the same. The code is the same. Scale — more parameters, more
data, more compute — is what bridges the gap between `"amine azine"` and a coherent
conversation.

## The files

```
src/
  tokenizer.py       BPE tokenizer
  embeddings.py      Token + positional embeddings (learned and sinusoidal)
  attention.py       Multi-head attention
  transformer.py     Decoder block (attention + feed-forward + residuals)
  model.py           Full GPT model
  train.py           Training loop
  generate.py        Autoregressive generation

docs/
  00-overview.md     This file
  01-tokenization.md BPE tokenization
  02-embeddings.md   Token and positional embeddings
  03-attention.md    Scaled dot-product and multi-head attention
  04-transformer-block.md  Layer norm, feed-forward, residual connections
  05-decoder-and-generation.md  Output head, sampling, autoregressive loop
  06-training.md     Loss, backpropagation, optimization

demos/
  demo_tokenizer.py          Train BPE and inspect merges
  demo_compression_curve.py  Vocab size vs compression
  demo_embeddings.py         Learned vs sinusoidal comparison
  demo_attention.py          Attention weight visualization
  demo_transformer.py        Layer-by-layer vector evolution
  demo_generation.py         Full pipeline (untrained)
  demo_train.py              Train on WikiMed and watch it learn
```
