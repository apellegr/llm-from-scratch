# Embeddings: From Token IDs to Vectors

After tokenization, we have a sequence of integers like `[65, 272, 32, 296]`. The
model needs to do math on these — dot products, matrix multiplications — so each
integer must become a **vector of numbers**. That's what embeddings do.

Two separate embeddings get **added together** to produce the model's input:

```python
x = token_emb(tokens) + pos_emb(tokens)
```

## Token Embeddings

A lookup table. Every token ID in the vocabulary maps to a vector of size `d_model`:

```
vocab_size = 300, d_model = 32

Token 65  ("A")        → [1.009, -1.015, -3.159, ...]   (32 numbers)
Token 272 ("ttention") → [-1.073, -2.495, 2.067, ...]
Token 32  (" ")        → [-6.928, 5.447, -8.930, ...]
```

Mechanically, `nn.Embedding(vocab_size, d_model)` is just a matrix of shape
`(vocab_size, d_model)`. Token ID 272 fetches row 272. The values are initialized
randomly and **learned during training** — the model adjusts them so that tokens
used in similar contexts end up with similar vectors.

The embedding table is one of the largest components of the model. With
`vocab_size=50,257` and `d_model=768` (GPT-2), that's ~38.6M parameters in the
embedding alone.

### The sqrt(d_model) scaling

Token embeddings get multiplied by `sqrt(d_model)` before use. This comes from the
original Transformer paper. Embedding values are initialized with roughly unit
variance, but attention later divides by `sqrt(d_k)`. Without this scaling, the
embedding signal would be drowned out by the positional signal. The multiplication
keeps them in balance.

## Positional Embeddings

Token embeddings alone have no concept of order. The sequences "cat sat" and "sat
cat" would produce the same set of vectors, just swapped. The model needs to know
**where** each token is in the sequence.

Positional embeddings are indexed by **position** (0, 1, 2, ...), not by what token
is there:

```
Position 0 → [0.33,  0.10, -0.62, ...]
Position 1 → [0.77, -0.55,  0.28, ...]
Position 2 → [0.14,  0.88, -0.31, ...]
```

The same token at different positions gets different combined vectors:

```
"dog" at position 3:  token_emb("dog") + pos_emb(3)
"dog" at position 1:  token_emb("dog") + pos_emb(1)
```

Same token vector, different position vector. The sum is different, so the model
can tell them apart.

### How positions are generated

The code doesn't look at tokens at all:

```python
seq_len = x.shape[1]
positions = torch.arange(seq_len, device=x.device)  # [0, 1, 2, ..., seq_len-1]
return self.embedding(positions)
```

A 5-token sentence gets positions 0-4. A 20-token sentence gets 0-19. It's the
same table, just indexed differently.

### What about variable-length sentences?

During training, text is chunked into **fixed-length blocks** of `max_seq_len`
tokens, regardless of sentence boundaries:

```
max_seq_len = 512

Block 1: [token_0, token_1, ..., token_511]    → positions [0..511]
Block 2: [token_512, token_513, ..., token_1023] → positions [0..511]
```

Every training example is the same length. A block might contain 3 short sentences
or half of a long paragraph. Position 0 just means "first token in this block."

At inference time, shorter inputs just use positions 0 through N-1. Those positions
were heavily trained (every block uses them). The hard limit is `max_seq_len` — you
cannot exceed it because the model has no embedding for positions beyond it.

## Two Approaches: Learned vs Sinusoidal

### Learned (GPT-style)

Another `nn.Embedding` table, shape `(max_seq_len, d_model)`. Initialized randomly,
trained alongside the rest of the model.

Before training, there's no structure — cosine similarities between positions are
near zero and random:

```
pos  0 vs pos  1:  -0.069  (essentially random)
pos  0 vs pos  5:   0.141
pos  0 vs pos 50:   0.089
```

After training, the model discovers positional relationships from data.

### Sinusoidal (original Transformer)

A fixed mathematical encoding using sin and cos at different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each dimension oscillates at a different frequency. Low dimensions change rapidly
(capturing fine position differences), high dimensions change slowly (capturing
broad position). This gives two key properties:

**Nearby positions are similar, far positions are different:**
```
pos 0 vs pos   1:  0.957  (very similar)
pos 0 vs pos  10:  0.629  (moderately similar)
pos 0 vs pos  50:  0.369  (quite different)
```

**Same distance = same relationship, regardless of absolute position:**
```
pos   0 vs pos   1:  0.957
pos  10 vs pos  11:  0.957
pos  50 vs pos  51:  0.957
pos 100 vs pos 101:  0.957
```

The model can learn "the token 3 positions ago" as a general concept, no matter
where in the sequence it is.

### Comparison

```
Property                    Sinusoidal            Learned
─────────────────────────── ───────────────────── ──────────────────
Trainable parameters        0                     max_seq_len × d_model
Works beyond max_seq_len    In theory, yes        No
Structure at init           Yes (smooth)          None (random)
Can adapt to data           No (fixed)            Yes
Used by                     Original Transformer  GPT-2, GPT-3
```

In practice, learned embeddings slightly outperform sinusoidal for fixed-length
contexts. Modern models like LLaMA use **RoPE** (Rotary Position Embeddings), which
encodes *relative* distance between tokens by rotating vectors — combining the
benefits of both approaches.

## d_model: the embedding dimension

`d_model` is a hyperparameter you choose before training. It's the width of every
vector flowing through the entire model — embeddings, attention, feed-forward
layers, everything.

| Model      | d_model | Total Parameters |
|------------|---------|------------------|
| GPT-2 Small| 768     | 117M             |
| GPT-2 Large| 1,280   | 774M             |
| GPT-3      | 12,288  | 175B             |
| LLaMA 7B   | 4,096   | 7B               |

More dimensions = richer representations but more parameters and compute. The only
hard constraint is that `d_model` must be divisible by `n_heads` (number of attention
heads), since each head works on a `d_model / n_heads` slice.

## Running the demo

Compare learned vs sinusoidal embeddings, see similarity patterns, and watch how
token + position combine:

```bash
python demo_embeddings.py
```

## Files

- `src/embeddings.py` — TokenEmbedding, PositionalEmbedding, SinusoidalPositionalEncoding
- `demo_embeddings.py` — Side-by-side comparison demo
