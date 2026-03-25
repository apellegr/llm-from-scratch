# Training: Teaching the Model to Predict

Before training, every weight in the model is random. The embeddings are random, the
Q/K/V projections are random, the feed-forward layers are random. The model outputs
`"The patienttttttttttt..."` — complete garbage.

Training is the process that turns structured noise into a language model. The idea
is simple: show the model text, ask it to predict the next token at every position,
measure how wrong it is, and nudge the weights to be less wrong. Repeat millions of
times.

## The input/target split

```python
inputs  = batch[:, :-1]   # all tokens except last
targets = batch[:, 1:]    # all tokens except first
```

For the sequence `[The, cat, sat, on, the, mat]`:

```
inputs:  [The, cat, sat, on,  the]
targets: [cat, sat, on,  the, mat]
```

Position 0 sees `"The"` and should predict `"cat"`. Position 1 sees `"The cat"` and
should predict `"sat"`. The causal mask ensures each position only sees tokens before
it. All positions are trained simultaneously in one forward pass — this is why
training is parallel while generation is sequential.

## The loss function: cross-entropy

The model outputs logits of shape `(batch, seq_len, vocab_size)`. Cross-entropy
measures how far the model's predicted probability distribution is from the truth:

```python
loss = cross_entropy(
    logits.reshape(-1, vocab_size),    # (batch*seq_len, vocab_size)
    targets.reshape(-1),               # (batch*seq_len,)
)
```

If the target is `"cat"` and the model assigned 90% probability to `"cat"`, the loss
is low (~0.1). If it assigned 1% to `"cat"` and 90% to `"dog"`, the loss is high
(~4.6). The loss is averaged across all positions and all sequences in the batch.

Cross-entropy has a useful interpretation: it measures how many bits the model needs
to encode the actual token given its predictions. A perfect model would achieve the
entropy of the language itself (~1-2 bits per character for English). Our untrained
model starts near `log2(vocab_size)` ≈ 8 bits (random guessing among 256 bytes).

## Backpropagation

```python
optimizer.zero_grad()   # clear old gradients
loss.backward()         # compute gradients for every weight
```

`loss.backward()` traces backward through every operation — from the loss, through
the output head, through every transformer block, through attention and feed-forward,
back to the embeddings. For each of the model's thousands of weights, it computes a
**gradient**: "if this weight were slightly larger, the loss would change by this
much."

This is the chain rule from calculus, applied automatically by PyTorch. The entire
computation graph is recorded during the forward pass, then unwound during
`backward()`.

## Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Sometimes gradients explode — they become enormous, and the weight update would
destabilize the model. Clipping rescales all gradients so their combined norm doesn't
exceed 1.0. It's a safety net that keeps training stable, especially in the early
stages when the model is still chaotic.

## The optimizer step

```python
optimizer.step()
```

This adjusts every weight by a small amount in the direction that reduces the loss.
We use **AdamW**, which:

- Tracks a moving average of each gradient (momentum — keeps moving in a consistent
  direction even if individual gradients are noisy)
- Tracks a moving average of each gradient's magnitude (adaptive learning rate —
  weights with large gradients get smaller updates)
- Applies weight decay (slightly shrinks all weights each step, preventing any single
  weight from growing too large — a form of regularization)

The **learning rate** (e.g., `1e-3`) controls the step size. Too large and the model
overshoots and diverges. Too small and it learns too slowly. Finding the right
learning rate is one of the most important hyperparameter choices.

## The training data: chunked text

Text is broken into fixed-length chunks, regardless of sentence boundaries:

```python
class TextDataset(Dataset):
    def __init__(self, text: str, seq_len: int):
        data = list(text.encode("utf-8"))
        n = len(data) // (seq_len + 1) * (seq_len + 1)
        self.chunks = torch.tensor(data[:n]).view(-1, seq_len + 1)
```

A 200KB text with `seq_len=64` produces ~3,000 chunks. Each chunk is 65 bytes long
(64 input + 1 target). The DataLoader shuffles these and serves them in batches.

## What the model learns over time

We trained a tiny model (120K parameters, 2 layers) on 200KB of medical Wikipedia
for 5 epochs (~20 seconds on CPU). Here's what happened:

**Before training:**
```
"The patienttttttttttttttttttttttttttt..."
```
Random weights produce a single-token loop.

**Epoch 1 (loss=25.0):**
```
"The patienttttttistttts \n i  a eF m \n  a ((( ((aaaaaaa..."
```
Still mostly stuck, but spaces and newlines are appearing.

**Epoch 2 (loss=4.3):**
```
"The patientsere \n TEy a f \n \n \n 1T r33ggCC1s e 1 5 iile itre 00..."
```
The model has learned that text has structure — words separated by spaces, newlines
between sections.

**Epoch 4 (loss=3.4):**
```
"The patient d Ese 5Cio onlone sol-20 \n (ae \n ecamoderocs pinase opin..."
```
Medical-sounding fragments emerge: "amine", "ine", "azine" — pieces of drug names
and chemical terms from the training data.

**After training:**
```
"Blood pressure oned Dolantozg ne 51 CCe \n \n Pue inrorafoB2micorose
oxininerggalin. \n totarone ine cinshrenazMezi P"
```
Still gibberish, but *structured* gibberish. The model has learned character-level
statistics of medical text: common letter sequences, word shapes, spacing patterns,
punctuation placement.

With more data (gigabytes, not kilobytes), more parameters (billions, not thousands),
and more training (weeks, not seconds), these fragments become real words, then
sentences, then coherent paragraphs. The mechanism is identical — the same code,
just scaled up.

## The loss curve

```
Epoch 1: loss=25.088   (random guessing)
Epoch 2: loss=4.349    (dramatic improvement — learning basic structure)
Epoch 3: loss=3.705    (diminishing returns begin)
Epoch 4: loss=3.439
Epoch 5: loss=3.261    (still improving but slowly)
```

The loss drops fastest in the first few epochs as the model learns the obvious
patterns (spaces exist, vowels follow consonants, etc.). Later epochs refine
subtler patterns. Real training runs show the same curve shape — just stretched
over millions of steps.

## Temperature at inference

After training, temperature controls generation randomness:

```
temp=0.3: "The diagnosis \n ane \n \n ane ane \n \n \n ..."
           → Repetitive but "safe" — stuck in high-probability loops

temp=0.8: "The diagnosis )ine ale Caas ar f , k ..."
           → More variety, better exploration

temp=1.5: "The diagnosisp)iy kale Caas ertg , kmmi ..."
           → Chaotic — too much randomness
```

## Training vs inference: why the asymmetry?

**Training** is parallel: all positions are computed at once, the loss for every
position is summed, and one backward pass updates all weights. A single forward pass
processes the entire sequence.

**Inference** (generation) is sequential: each token must be generated before the
next can be predicted. For a 100-token response, that's 100 forward passes.

This asymmetry is fundamental to autoregressive models. Training is cheap per token
(parallel). Inference is expensive per token (sequential). This is why running a
model is the main cost, not training it.

## Running the demo

Train the model on WikiMed and watch it learn:

```bash
python demo_train.py
```

Extract training data from WikiMed ZIM (requires libzim):

```bash
python scripts/extract_training_data.py 2    # extract 2 MB
```

## Files

- `src/train.py` — Training loop function
- `src/model.py` — GPT model (forward pass produces logits)
- `demo_train.py` — End-to-end training demo
- `scripts/extract_training_data.py` — WikiMed ZIM text extraction
- `data/wikimed_train.txt` — Extracted training corpus
