"""Demo: how vocab size affects compression and what tokens get learned."""

import os

from src.tokenizer import BPETokenizer


def main():
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "wiki_attention.txt")
    with open(corpus_path) as f:
        text = f.read()

    raw_bytes = len(text.encode("utf-8"))

    # --- Compression curve ---
    vocab_sizes = [257, 258, 260, 265, 270, 280, 300, 350, 400, 500, 750, 1000, 1500, 2000]
    print("Compression vs vocab size")
    print(f"Corpus: {raw_bytes} bytes\n")
    print(f"{'Vocab Size':>10}  {'Merges':>6}  {'Tokens':>6}  {'Ratio':>6}  {'Bytes/Token':>11}")
    print("-" * 55)

    tokenizers = {}
    for vs in vocab_sizes:
        tok = BPETokenizer()
        tok.train(text, vs)
        tokenizers[vs] = tok
        n_tokens = len(tok.encode(text))
        ratio = raw_bytes / n_tokens
        print(f"{vs:>10}  {vs - 256:>6}  {n_tokens:>6}  {ratio:>5.2f}x  {ratio:>11.2f}")

    # --- What gets included at each tier ---
    print("\n\n" + "=" * 70)
    print("WHAT GETS LEARNED AT EACH VOCAB SIZE")
    print("=" * 70)

    tiers = [
        (260, "4 merges — only the most frequent byte pairs"),
        (270, "14 merges — common bigrams and a few trigrams"),
        (300, "44 merges — frequent subwords and short words"),
        (400, "144 merges — most common English subwords"),
        (750, "494 merges — full words and common phrases"),
        (1500, "1244 merges — long words and multi-word chunks"),
    ]

    for vs, description in tiers:
        tok = BPETokenizer()
        tok.train(text, vs)

        print(f"\n--- vocab_size={vs} ({description}) ---")
        print(f"New tokens learned ({vs - 256}):\n")

        # Show all merges for small vocabs, sample for large ones
        merges = list(tok.merges.items())
        if len(merges) <= 20:
            indices = list(range(len(merges)))
            label = "All merges"
        else:
            mid = len(merges) // 2
            indices = sorted(set(
                list(range(5))
                + list(range(mid - 2, mid + 3))
                + list(range(len(merges) - 5, len(merges)))
            ))
            label = "Sample merges (first / middle / last)"

        print(f"  {label}:")
        prev_idx = -1
        for idx in indices:
            if prev_idx >= 0 and idx > prev_idx + 1:
                print(f"    {'...':>4}")
            prev_idx = idx

            (a, b), new_id = merges[idx]
            try:
                display = tok.vocab[new_id].decode("utf-8")
            except UnicodeDecodeError:
                display = repr(tok.vocab[new_id])
            # Truncate absurdly long tokens (overfitting to small corpus)
            if len(display) > 40:
                display = display[:37] + "..."
            print(f"    {idx + 1:>4}. {display!r}")

        # Show how a sample sentence tokenizes at this vocab size
        sample = "The attention mechanism computes weighted representations of input sequences."
        ids = tok.encode(sample)
        pieces = []
        for tid in ids:
            try:
                pieces.append(tok.vocab[tid].decode("utf-8"))
            except UnicodeDecodeError:
                pieces.append(repr(tok.vocab[tid]))

        print(f"\n  Tokenization of: {sample!r}")
        print(f"  {len(ids)} tokens: {pieces}")


if __name__ == "__main__":
    main()
