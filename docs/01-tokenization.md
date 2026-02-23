# Tokenization: Byte-Pair Encoding (BPE)

Tokenization is the first step in any LLM pipeline. Raw text goes in, a sequence
of integer token IDs comes out. The model never sees characters — it sees tokens.

## Why not just use characters?

A sentence like "The attention mechanism" is 23 characters. If each character is a
token, the model's sequence is 23 steps long. Attention is O(n²) in sequence length,
so longer sequences are expensive.

BPE compresses text by learning common patterns. With a trained tokenizer, that same
sentence might be 5-8 tokens instead of 23 — shorter sequences, faster training,
longer effective context windows.

## How BPE works

### Step 0: Start with bytes

The base vocabulary is the 256 possible byte values (0x00–0xFF). Every possible
text can be represented as a sequence of bytes, so there are no "unknown" tokens.

```
"hello" → [104, 101, 108, 108, 111]
             h    e    l    l    o
```

### Step 1: Count adjacent pairs

Scan the token list and count every pair of neighbors:

```
"aabaa" → [97, 97, 98, 97, 97]

Pairs:  (97, 97) → 2    "aa"
        (97, 98) → 1    "ab"
        (98, 97) → 1    "ba"
```

### Step 2: Merge the most frequent pair

The most frequent pair becomes a new token. Replace every occurrence:

```
Before: [97, 97, 98, 97, 97]     5 tokens
After:  [256,    98, 256    ]     3 tokens
         "aa"    "b"  "aa"
```

Token 256 is now in the vocabulary: `{256: "aa"}`.

### Step 3: Repeat

Go back to step 1 with the shorter sequence. Each iteration creates one new token
from the current most frequent pair. Merges compose — later merges can combine
tokens that were themselves products of earlier merges:

```
Merge  3: t + i      → "ti"         (2 bytes)
Merge  5: o + n      → "on"         (2 bytes)
Merge  9: ti + on    → "tion"       (2 merged tokens → 4 bytes)
Merge 17: tten + tion → "ttention"  (keeps compounding)
Merge 23: " a" + ttention → " attention"
```

The algorithm discovered "attention" as a single token purely from frequency
statistics — no linguistic knowledge required.

## Vocab size is a tradeoff

`vocab_size = 256 (bytes) + num_merges`

More merges means more tokens to learn, which means better compression but a larger
embedding table. Here's what we measured on an 8KB Wikipedia article about attention
mechanisms:

```
Vocab Size  Merges  Tokens   Ratio
       257       1    7734   1.02x
       260       4    7333   1.08x
       270      14    6403   1.23x
       300      44    5242   1.51x
       400     144    3964   1.99x
       750     494    2465   3.20x
      1000     744    1965   4.02x
      2000    1744     927   8.51x
```

### What gets learned at each scale

**4 merges** — Only the most frequent byte pairs: `" a"`, `"e "`, `"ti"`, `"en"`.
Nearly character-level tokenization. The sentence "The attention mechanism" is still
71 tokens.

**14 merges** — Common bigrams and the first trigrams: `"th"`, `"on"`, `"in"`, and
the first composed token `"tion"` (from `"ti"` + `"on"`). Down to 62 tokens.

**44 merges** — Frequent subwords and short words: `"the "`, `" attention"`, `"is"`,
`"for"`, `"of"`. The model starts recognizing English structure. 48 tokens.

**144 merges** — Most common English subwords covered. Pairs like `"co"`, `"mp"`,
`"igh"` start appearing. Roughly 2x compression. 35 tokens.

**494 merges** — Full words emerge: `"natural language processing"` is a single
token. Near word-level chunking. 3.2x compression. 19 tokens.

**1244 merges** — Overfitting. The corpus is only 8KB, so the algorithm starts
memorizing entire paragraphs as single tokens. These would never match new text.
No improvement over 494 merges on our test sentence.

### The overfitting lesson

With too many merges for a small corpus, BPE memorizes instead of generalizes. The
last merges at vocab_size=1500 were literally the opening paragraph being extended
one character at a time. This is why real models (GPT-2, LLaMA) train BPE on
hundreds of gigabytes — the frequency statistics need to be stable and representative.

### What real models use

| Model   | Vocab Size | Notes                                    |
|---------|------------|------------------------------------------|
| GPT-2   | 50,257     | 256 bytes + 50,000 merges + 1 special    |
| GPT-4   | ~100,000   | Larger for multilingual coverage          |
| LLaMA   | 32,000     | Smaller to keep embedding table compact   |
| LLaMA 3 | 128,256    | Much larger for multilingual + code       |

The tradeoff:
- **Bigger vocab** → better compression (shorter sequences, faster inference) but
  larger embedding matrix and rare tokens get undertrained
- **Smaller vocab** → every token well-trained, smaller model, but longer sequences
  (more expensive attention)

## Encoding and decoding

**Encoding** new text replays the learned merges in order on raw bytes:

```python
token_ids = list(text.encode("utf-8"))      # start with bytes
for pair, new_id in self.merges.items():    # replay each merge
    token_ids = self._apply_merge(token_ids, pair, new_id)
```

**Decoding** is simpler — just look up each token ID in the vocab and concatenate
the bytes:

```python
raw_bytes = b"".join(self.vocab[t] for t in token_ids)
return raw_bytes.decode("utf-8", errors="replace")
```

Round-trip is lossless: `decode(encode(text)) == text` always holds.

## Running the demos

Train BPE on the Wikipedia corpus and inspect merges, encoding, and compression:

```bash
python demo_tokenizer.py
```

See how compression scales with vocab size and what tokens emerge at each tier:

```bash
python demo_compression_curve.py
```

## Files

- `src/tokenizer.py` — BPE tokenizer implementation
- `demo_tokenizer.py` — Train and inspect a tokenizer
- `demo_compression_curve.py` — Compression vs vocab size analysis
- `data/wiki_attention.txt` — Wikipedia corpus used for training
