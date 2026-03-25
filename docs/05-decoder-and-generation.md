# Decoder and Generation: From Vectors to Text

Everything built so far — embeddings, attention, transformer blocks — produces
refined token vectors. But the model's job is to **predict the next token**. This
is where vectors become predictions, and predictions become text.

## The output head

After all transformer blocks, each position holds a `d_model`-dimensional vector
packed with context. To predict the next token, we project that vector into
vocabulary space:

```python
self.ln_f = nn.LayerNorm(d_model)
self.head = nn.Linear(d_model, vocab_size, bias=False)
```

In the forward pass:

```python
x = self.ln_f(x)          # final layer norm
logits = self.head(x)     # (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
```

For each position, this produces one score (**logit**) per token in the vocabulary.
With `vocab_size=300`, that's 300 numbers — one for each possible next token. High
logit = the model thinks that token is likely.

This is a single `nn.Linear` — a matrix multiplication. Each of the 300 rows in
the weight matrix corresponds to a token. The dot product between the token vector
and a row measures "how much does the current context point toward this token?"

## Weight tying

```python
self.head.weight = self.token_emb.embedding.weight
```

The output head and the input embedding table **share the same weight matrix**.
They're doing inverse operations:

- Embedding: token ID → vector (look up a row)
- Output head: vector → token scores (dot product with each row)

Using the same weights means the model learns a single consistent mapping between
tokens and vector space. This saves parameters and improves performance — a token's
embedding vector is also its "target" in the output space.

## Logits to probabilities

The raw logits are unconstrained numbers (positive, negative, any magnitude).
Softmax converts them to a probability distribution:

```python
probs = F.softmax(logits, dim=-1)    # all positive, sums to 1.0
```

Example for a tiny vocabulary:

```
logits: [2.1,  0.5, -1.3,  3.8,  0.1]
         "The" "cat" "dog" "sat" "on"

probs:  [0.12, 0.02, 0.00, 0.83, 0.02]
         "The" "cat" "dog" "sat" "on"
```

The model is 83% confident the next token is "sat".

## Autoregressive generation

Generation works as a loop. Each step predicts one token, appends it, and runs the
model again on the longer sequence:

```
Start:  ["The", "cat"]
Step 1: model(["The", "cat"])               → predict "sat"
Step 2: model(["The", "cat", "sat"])        → predict "on"
Step 3: model(["The", "cat", "sat", "on"])  → predict "the"
...
```

This is called **autoregressive** because each output feeds back as input. Every
prediction depends on all previous predictions.

### Why only the last position?

The model produces logits for every position, but position `i` predicts what comes
at position `i+1`. During generation, only the last position is predicting a token
we haven't seen yet — all earlier positions predict tokens we already have.

```python
logits = model(idx_cond)           # (batch, seq_len, vocab_size)
logits = logits[:, -1, :]          # take only the last position
```

### Temperature

Temperature controls how random the output is by scaling logits before softmax:

```python
logits = logits / temperature
```

```
temperature = 0.1  →  sharp: top token dominates, nearly deterministic
temperature = 1.0  →  normal: distribution as trained
temperature = 2.0  →  flat: unlikely tokens get a real chance
```

Low temperature is good for factual tasks (pick the most likely answer). High
temperature is good for creative tasks (explore surprising combinations).

Example with logits `[2.1, 3.8]`:

```
temp=1.0:  softmax([2.1, 3.8])  →  [0.15, 0.85]   normal
temp=0.5:  softmax([4.2, 7.6])  →  [0.03, 0.97]   very confident
temp=2.0:  softmax([1.05, 1.9]) →  [0.30, 0.70]   more uncertain
```

### Top-k filtering

Only keep the `k` highest-scoring tokens, zero out everything else:

```python
v, _ = torch.topk(logits, k)
logits[logits < v[:, [-1]]] = float("-inf")   # everything below top-k → 0 after softmax
```

This prevents the model from picking very unlikely tokens. With `top_k=50`, only
the 50 most probable tokens can be chosen, no matter how large the vocabulary.

### Sampling

```python
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

We **sample** from the distribution rather than always picking the highest. If "sat"
has 40% probability and "slept" has 30%, sometimes you'll get "slept." This creates
variety in generation. (Taking the argmax — always the top token — is called
**greedy decoding** and produces repetitive, dull text.)

### Sequence length limit

```python
idx_cond = idx[:, -model.max_seq_len:]
```

If the generated sequence exceeds `max_seq_len`, only the last `max_seq_len` tokens
are kept as context. Earlier tokens fall out of the window — the model "forgets"
them. This is the context length limit in practice.

## Training vs generation

An important distinction:

**Training** processes all positions in parallel. The causal mask prevents cheating,
and the loss is computed for every position simultaneously. A single forward pass
handles the entire sequence.

**Generation** is sequential. Each token requires a full forward pass through every
layer, and token 5 can't be predicted until token 4 exists. For a 100-token
response, that's 100 full forward passes. This is why generation is slow compared
to training.

## The full pipeline

```
Input: ["The", "cat"]  (token IDs)
          ↓
    Token embedding     (IDs → vectors)
    + Position embedding (add position info)
          ↓
    Transformer block 1  (attend + think)
          ↓
    Transformer block 2  (attend + think)
          ↓
         ...
          ↓
    Transformer block N  (attend + think)
          ↓
    Final layer norm
          ↓
    Output head          (vectors → vocab_size logits)
          ↓
    Softmax + sampling   (logits → probabilities → pick a token)
          ↓
    Output: "sat"        (append, repeat)
```

## Running the demo

See the full pipeline in action — from token IDs through logits to sampled text:

```bash
python demo_generation.py
```

## Files

- `src/model.py` — GPT model (ties everything together)
- `src/generate.py` — Autoregressive generation loop
- `demo_generation.py` — End-to-end generation demo
