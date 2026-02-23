# Attention: How Tokens Look at Each Other

After embeddings, each token is a vector — but the vectors are independent. Token 3
has no idea what token 0 says. Attention is the mechanism that lets every token
**gather information from every other token** in the sequence.

## Q, K, V — the communication channel

For each token, attention produces three vectors through three separate learned
linear projections:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What can I be found by?"
- **Value (V):** "What information do I give when found?"

```python
Q = W_q(x)   # x @ W_q matrix + bias
K = W_k(x)   # x @ W_k matrix + bias
V = W_v(x)   # x @ W_v matrix + bias
```

Each `W_q`, `W_k`, `W_v` is an `nn.Linear(d_model, d_model)` — a learned weight
matrix initialized randomly and adjusted during training via backpropagation. The
same token embedding goes through three different matrices to produce three different
vectors. The roles (what to look for, what to advertise, what to share) are not
designed — they **emerge** from training.

### Concrete example

For `"The cat sat"` with `d_k=4`:

```
        Query (what I want)      Key (what I have)       Value (what I give)
"The"   [0.1, -0.3, 0.5, 0.2]   [0.4, 0.7, -0.1, 0.3]  [0.2, 0.5, 0.1, 0.8]
"cat"   [0.8, 0.2, -0.1, 0.6]   [0.9, -0.2, 0.4, 0.1]  [0.7, -0.1, 0.3, 0.4]
"sat"   [0.6, 0.5, 0.3, -0.4]   [0.1, 0.3, 0.8, -0.2]  [0.3, 0.6, -0.2, 0.5]
```

## Computing attention scores

The dot product between a Query and a Key measures how much one token wants to
attend to another. For every pair of tokens, we compute:

```
score(sat, cat) = Q_sat · K_cat
               = 0.6×0.9 + 0.5×(-0.2) + 0.3×0.4 + (-0.4)×0.1
               = 0.52
```

Doing this for all pairs produces a `(seq_len, seq_len)` matrix:

```
              K_the   K_cat   K_sat
Q_the       [  0.01,   0.15,  0.38 ]
Q_cat       [  0.36,   0.82,  0.00 ]
Q_sat       [  0.53,   0.52,  0.22 ]
```

Entry `[i, j]` means "how much token `i` wants to look at token `j`."

### Scaling by sqrt(d_k)

```python
scores = Q @ K.T / sqrt(d_k)
```

Dot products between longer vectors produce larger numbers (more terms being summed).
Large values push softmax into saturation — one score dominates, gradients vanish,
learning stops. Dividing by `sqrt(d_k)` keeps scores in a range where softmax
produces smooth, useful distributions.

## The causal mask

For a decoder (autoregressive model like GPT), each token can only see tokens before
it — not the future. Without this, the model would simply copy the answer instead of
learning to predict it.

The mask sets future positions to `-inf` before softmax:

```
              K_the   K_cat   K_sat
Q_the       [  0.01,  -inf,   -inf  ]   ← "The" sees only itself
Q_cat       [  0.36,   0.82,  -inf  ]   ← "cat" sees "The" and itself
Q_sat       [  0.53,   0.52,   0.22 ]   ← "sat" sees everything before it
```

Since `e^(-inf) = 0`, those positions get exactly zero weight after softmax.

## Softmax → attention weights

Softmax converts each row into a probability distribution (sums to 1):

```
              K_the   K_cat   K_sat
Q_the       [  1.00,   0.00,   0.00 ]
Q_cat       [  0.39,   0.61,   0.00 ]
Q_sat       [  0.38,   0.37,   0.25 ]
```

Each row says: for this token, what fraction of attention goes to each other token.

## Weighted sum of values

Each token's output is a blend of Value vectors, weighted by the attention scores:

```
V_the = [0.2, 0.5, 0.1, 0.8]
V_cat = [0.7, -0.1, 0.3, 0.4]
V_sat = [0.3, 0.6, -0.2, 0.5]

output_sat = 0.38 × V_the + 0.37 × V_cat + 0.25 × V_sat
           = [0.41, 0.30, 0.10, 0.58]
```

The output for `"sat"` is no longer just about `"sat"` — it now carries information
from `"The"` and `"cat"`. After attention, each token's vector reflects **the whole
sequence** it was allowed to see.

## Multi-head attention

One set of Q, K, V captures one type of relationship. But tokens relate in multiple
ways simultaneously — syntactically, semantically, positionally.

Multi-head attention runs the process multiple times in parallel, each time on a
different **subspace** of the vector:

```
d_model = 256, n_heads = 4 → d_k = 64 per head

Head 1 (dims 0-63):   might learn syntactic relationships
Head 2 (dims 64-127):  might learn positional proximity
Head 3 (dims 128-191): might learn semantic similarity
Head 4 (dims 192-255): might learn something else
```

Each head has its own learned Q, K, V projections, so each asks different questions
and produces a different `(seq_len, seq_len)` attention pattern. For the same
token pair:

```
Head 1: score(sat, cat) = 0.82   ← strong syntactic match
Head 2: score(sat, cat) = 0.15   ← weak positional signal
Head 3: score(sat, cat) = 0.71   ← strong semantic match
Head 4: score(sat, cat) = 0.03   ← irrelevant to this head
```

A single head would collapse all of this into one score where the signals could
cancel each other out. Multiple heads keep them separate.

### Implementation: reshape, don't duplicate

We don't create 4 separate weight matrices. One big `W_q` projects to the full
`d_model`, then we reshape:

```python
Q = self.W_q(x)                   # (batch, seq_len, 256)
Q = Q.view(batch, seq, 4, 64)     # split into 4 heads of 64
Q = Q.transpose(1, 2)             # (batch, 4, seq_len, 64)
```

Each head operates on its 64-dim slice independently.

### Concatenate and project

After attention, each head's output (shape `seq_len, d_k`) gets concatenated back:

```
Head 1 out: [0.41, 0.30, ...]   ← 64 dims
Head 2 out: [0.22, -0.15, ...]  ← 64 dims
Head 3 out: [0.55, 0.08, ...]   ← 64 dims
Head 4 out: [0.13, 0.67, ...]   ← 64 dims

Concatenated: [0.41, 0.30, ..., 0.22, -0.15, ..., 0.55, 0.08, ..., 0.13, 0.67, ...]
              ← back to 256 dims
```

A final `W_o` projection mixes the heads' outputs. This is essential — without it,
heads would be completely independent. `W_o` lets the model combine what different
heads found into a single representation.

### Why smaller heads still work

Each head only sees `d_k` dimensions, not the full `d_model`. This works because:

1. **Projections rearrange information.** `W_q` and `W_k` are applied before the
   split — they can route any combination of original dimensions into any head's
   slice.
2. **Multiple heads can collaborate.** If a concept needs more capacity, several
   heads can each capture different aspects of it. `W_o` combines them.
3. **The constraint helps.** Restricting each head forces specialization, creating
   diversity of attention patterns across heads.

## The full picture

```
Token embeddings (what + where)
         ↓
    ┌────┴─────────────┐
    │  Multi-Head       │
    │  Attention        │
    │                   │
    │  x → Q, K, V     │  three learned projections
    │  scores = Q·K/√d  │  how much to attend
    │  mask futures     │  causal: no looking ahead
    │  softmax          │  normalize to weights
    │  output = w × V   │  weighted blend of values
    │  concat heads     │  reassemble
    │  W_o projection   │  mix head outputs
    └────┬──────────────┘
         ↓
    Context-aware token vectors
```

Before attention: each token knows only itself.
After attention: each token carries information from every token it attended to.

## Running the demo

Visualize attention weights and see which tokens attend to which:

```bash
python demo_attention.py
```

## Files

- `src/attention.py` — MultiHeadAttention implementation
- `demo_attention.py` — Attention weight visualization
