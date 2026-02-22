"""Demo: train BPE on a Wikipedia article and inspect the results."""

import os

from src.tokenizer import BPETokenizer


def main():
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "wiki_attention.txt")
    with open(corpus_path) as f:
        text = f.read()

    print(f"Corpus: Wikipedia 'Attention (machine learning)'")
    print(f"Raw text: {len(text)} characters, {len(text.encode('utf-8'))} bytes")
    print()

    # --- Train ---
    vocab_size = 300  # 256 base bytes + 44 merges
    tok = BPETokenizer()
    tok.train(text, vocab_size)

    # --- Show the learned merges ---
    print(f"Learned {len(tok.merges)} merges:\n")
    print(f"{'#':>3}  {'Pair':>20}  {'New ID':>6}  Merged Token")
    print("-" * 60)
    for i, ((a, b), new_id) in enumerate(tok.merges.items()):
        pair_str = f"{tok.vocab[a]!r} + {tok.vocab[b]!r}"
        merged_str = tok.vocab[new_id]
        # Show printable representation
        try:
            display = merged_str.decode("utf-8")
        except UnicodeDecodeError:
            display = repr(merged_str)
        print(f"{i+1:>3}  {pair_str:>20}  {new_id:>6}  {display!r}")

    # --- Encode a sample sentence ---
    sample = "Attention is all you need"
    token_ids = tok.encode(sample)
    print(f"\n\nEncoding: {sample!r}")
    print(f"Token IDs ({len(token_ids)} tokens): {token_ids}")
    print(f"Token breakdown:")
    for tid in token_ids:
        try:
            display = tok.vocab[tid].decode("utf-8")
        except UnicodeDecodeError:
            display = repr(tok.vocab[tid])
        print(f"  {tid:>5} → {display!r}")

    # --- Compression stats ---
    raw_bytes = len(sample.encode("utf-8"))
    print(f"\nRaw bytes: {raw_bytes}  →  BPE tokens: {len(token_ids)}")
    print(f"Compression ratio: {raw_bytes / len(token_ids):.2f}x")

    # --- Full corpus compression ---
    all_ids = tok.encode(text)
    raw_total = len(text.encode("utf-8"))
    print(f"\nFull corpus: {raw_total} bytes → {len(all_ids)} tokens")
    print(f"Corpus compression: {raw_total / len(all_ids):.2f}x")

    # --- Round-trip test ---
    decoded = tok.decode(all_ids)
    print(f"\nRound-trip decode matches original: {decoded == text}")


if __name__ == "__main__":
    main()
