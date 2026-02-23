# Transformer Block: Attention + Feed-Forward + Residuals

The transformer block is the repeating unit of the model. Stack enough of them and
you get GPT. Each block takes a sequence of token vectors in and produces a sequence
of refined token vectors out — same shape, richer content.

## The big picture

Before the block, tokens are isolated vectors. After the block, they carry context:

```
Isolated tokens → [Block] → Context-aware tokens → [Block] → Deeper understanding → ...
```

Each block does two things, in order:

1. **Attention** — tokens talk to each other, exchanging information
2. **Feed-forward** — each token independently processes what it learned

Then the output feeds into the next block for another round. GPT-2 Small does this
12 times. GPT-3 does it 96 times.

## Inside one block

```python
# Pre-norm architecture (GPT-2 style)
x = x + self.dropout(self.attention(self.ln1(x), mask))   # communicate
x = x + self.feed_forward(self.ln2(x))                     # compute
```

Those two lines contain four concepts: layer norm, attention, feed-forward, and
residual connections.

### Layer Norm

```python
self.ln1 = nn.LayerNorm(d_model)
self.ln2 = nn.LayerNorm(d_model)
```

For each token's vector, layer norm normalizes the values to mean 0 and variance 1,
then applies a learned scale and shift. This prevents values from drifting as they
pass through layers — without it, vectors can grow huge or shrink to nothing,
destabilizing training.

This is **pre-norm**: normalize before the operation. The original Transformer paper
used post-norm (normalize after). Pre-norm trains more stably because attention and
feed-forward always receive well-behaved inputs.

### Multi-Head Attention

Covered in detail in [03-attention.md](03-attention.md). The key point here: this is
where tokens communicate. Each token looks at the tokens before it (causal mask),
determines relevance via Q·K scores, and pulls information via weighted sum of V.

After attention, each token's vector has been updated with information from the
tokens it attended to.

### Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),     # expand: 256 → 1024
            nn.GELU(),                     # nonlinearity
            nn.Linear(d_ff, d_model),     # compress: 1024 → 256
            nn.Dropout(dropout),
        )
```

After attention mixes information between tokens, the feed-forward network processes
each token **independently** — no cross-talk between positions.

The pattern is expand → activate → compress:

- **Expand** (`d_model → d_ff`): project each token into a larger space. `d_ff` is
  typically 4× `d_model`. This gives the network more room to work.
- **GELU activation**: the nonlinearity. Without it, the two linear layers would
  collapse into a single matrix multiplication. GELU lets the model represent complex
  functions.
- **Compress** (`d_ff → d_model`): project back to the original dimension so the
  output can flow into the next block.

Attention is about **relationships between tokens**. Feed-forward is about
**processing what each token now knows** — compressing and refining the blended
information.

### Residual Connections

```python
x = x + self.attention(self.ln1(x), mask)    # not x = attention(x)
x = x + self.feed_forward(self.ln2(x))       # not x = feed_forward(x)
```

The `+ x` is the residual connection. The output is the original input **plus** the
result of the operation. This serves two purposes:

**Information preservation.** Each layer adds refinement rather than replacing the
representation. The original signal always passes through — it can be reinforced or
modified, but never lost entirely.

**Gradient flow.** In a 96-layer model, gradients during training must travel backward
through every layer. Without residuals, the signal degrades at each step (vanishing
gradients). The `+ x` creates a highway: gradients can flow directly through the
addition, making deep stacking trainable.

## The full block diagram

```
Input x (token vectors)
    │
    ├──────────────────────┐
    │                      │
    ▼                      │
  LayerNorm                │  ← normalize
    │                      │
    ▼                      │
  Multi-Head Attention     │  ← tokens talk to each other
    │                      │
    ▼                      │
  Dropout                  │
    │                      │
    ▼                      │
  + ◄──────────────────────┘  ← residual: add original back
    │
    ├──────────────────────┐
    │                      │
    ▼                      │
  LayerNorm                │  ← normalize again
    │                      │
    ▼                      │
  Feed-Forward             │  ← each token "thinks" independently
    │                      │
    ▼                      │
  + ◄──────────────────────┘  ← residual again
    │
    ▼
Output (same shape — ready for the next block)
```

## Stacking blocks

The model creates `n_layers` blocks, each with the same architecture but **its own
separate learned weights**:

```python
self.layers = nn.ModuleList(
    [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
)
```

In the forward pass, `x` flows through each block sequentially:

```python
for layer in self.layers:
    x = layer(x, mask)
```

Each block's W_q, W_k, W_v, feed-forward weights, and layer norms are independently
learned. Early blocks might specialize in syntax, later blocks in semantics, the
deepest blocks in abstract reasoning.

| Model       | Layers | d_model | n_heads | d_ff   |
|-------------|--------|---------|---------|--------|
| GPT-2 Small | 12     | 768     | 12      | 3,072  |
| GPT-2 Large | 36     | 1,280   | 20      | 5,120  |
| GPT-3       | 96     | 12,288  | 96      | 49,152 |
| LLaMA 7B    | 32     | 4,096   | 32      | 11,008 |

More layers = more rounds of communication and computation = deeper understanding,
but more parameters and slower inference.

## Fixed sequence, evolving content

The sequence length **never changes** through the blocks. No tokens are created,
removed, or rearranged. What changes is what each token's vector contains.

After enough layers, the vector at position 3 doesn't really represent the original
token `"on"` anymore — it represents "the concept encoded at position 3, given
everything the model has processed." The tokens are fixed buckets. The information
inside evolves with every layer.

This has real consequences:

- **No scratch space.** The model must pack all its understanding into `seq_len`
  vectors. This is why chain-of-thought prompting helps — generating reasoning
  tokens gives the model extra slots to store intermediate work.
- **Equal compute per position.** Every token goes through the same layers, same
  operations, same cost. There's no way to spend more compute on a hard part of the
  sentence.
- **Information propagates indirectly.** After one block, `"sat"` knows about
  `"cat"`. After several blocks, `"sat"` knows that `"cat"` was `"tired"` — because
  an earlier block passed adjective information into `"cat"`, which then flowed to
  `"sat"` in a later block.

## Q, K, V are recomputed each block

An important detail: Q, K, and V are not carried between blocks. Each block computes
its own Q, K, V from scratch using its own learned projection weights. What flows
from block to block is just `x` — the token vectors.

```
Block 1: takes x₀, computes its own Q₁, K₁, V₁, outputs x₁
Block 2: takes x₁, computes its own Q₂, K₂, V₂, outputs x₂
Block 3: takes x₂, computes its own Q₃, K₃, V₃, outputs x₃
```

## Running the demo

See how token vectors evolve through stacked transformer blocks:

```bash
python demo_transformer.py
```

## Files

- `src/transformer.py` — FeedForward and DecoderBlock
- `src/attention.py` — MultiHeadAttention (used inside DecoderBlock)
- `demo_transformer.py` — Layer-by-layer evolution demo
