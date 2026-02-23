"""Demo: compare learned vs sinusoidal positional embeddings."""

import torch

from src.embeddings import (
    TokenEmbedding,
    PositionalEmbedding,
    SinusoidalPositionalEncoding,
)


def show_matrix(name: str, tensor: torch.Tensor, rows: int = 8, cols: int = 8):
    """Print a slice of a 2D tensor as a formatted table."""
    t = tensor[:rows, :cols]
    print(f"\n{name} (showing [{rows}, {cols}] of {list(tensor.shape)}):")
    # Header
    print(f"{'pos':>5}", end="")
    for c in range(cols):
        print(f"  dim_{c:>2}", end="")
    print()
    print("-" * (7 + cols * 9))
    for r in range(t.shape[0]):
        print(f"{r:>5}", end="")
        for c in range(t.shape[1]):
            print(f"  {t[r, c]:>7.3f}", end="")
        print()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    torch.manual_seed(42)

    max_seq_len = 128
    d_model = 32
    vocab_size = 300

    # =========================================================
    # 1. Token Embeddings
    # =========================================================
    print("=" * 65)
    print("TOKEN EMBEDDINGS")
    print("=" * 65)

    tok_emb = TokenEmbedding(vocab_size, d_model)
    tokens = torch.tensor([[65, 272, 32, 296, 256]])  # "A", "ttention", " ", "is", " a"
    token_labels = ["A", "ttention", " ", "is", " a"]

    vecs = tok_emb(tokens).squeeze(0).detach()
    show_matrix("Token embedding vectors", vecs, rows=5, cols=8)

    print(f"\nEach token is now a vector of {d_model} numbers.")
    print(f"Embedding table shape: ({vocab_size}, {d_model}) = {vocab_size * d_model:,} parameters")

    # =========================================================
    # 2. Sinusoidal Positional Encoding
    # =========================================================
    print(f"\n\n{'=' * 65}")
    print("SINUSOIDAL POSITIONAL ENCODING (fixed, no training)")
    print("=" * 65)

    sin_pe = SinusoidalPositionalEncoding(max_seq_len, d_model)
    dummy = torch.zeros(1, max_seq_len, dtype=torch.long)
    sin_vecs = sin_pe(dummy).squeeze(0)

    show_matrix("Sinusoidal encoding", sin_vecs, rows=8, cols=8)

    print("\nKey property: nearby positions have similar encodings.")
    print("Cosine similarity between positions:")
    pairs = [(0, 1), (0, 2), (0, 5), (0, 10), (0, 50), (0, 100)]
    for a, b in pairs:
        sim = cosine_sim(sin_vecs[a], sin_vecs[b])
        bar = "#" * int(max(0, sim) * 30)
        print(f"  pos {a:>3} vs pos {b:>3}: {sim:>6.3f}  {bar}")

    print("\nKey property: the same relative distance has a consistent relationship.")
    print("Cosine similarity between pairs that are 1 apart:")
    for start in [0, 10, 20, 50, 100]:
        sim = cosine_sim(sin_vecs[start], sin_vecs[start + 1])
        print(f"  pos {start:>3} vs pos {start + 1:>3}: {sim:>6.3f}")

    # =========================================================
    # 3. Learned Positional Embedding (untrained)
    # =========================================================
    print(f"\n\n{'=' * 65}")
    print("LEARNED POSITIONAL EMBEDDING (random at init, trained later)")
    print("=" * 65)

    learned_pe = PositionalEmbedding(max_seq_len, d_model)
    learned_vecs = learned_pe(dummy).squeeze(0).detach()

    show_matrix("Learned embedding (untrained)", learned_vecs, rows=8, cols=8)

    print("\nBefore training, positions are random — no structure yet:")
    for a, b in pairs:
        sim = cosine_sim(learned_vecs[a], learned_vecs[b])
        bar = "#" * int(max(0, sim) * 30)
        print(f"  pos {a:>3} vs pos {b:>3}: {sim:>6.3f}  {bar}")

    # =========================================================
    # 4. How they combine
    # =========================================================
    print(f"\n\n{'=' * 65}")
    print("COMBINED: token_emb + pos_emb")
    print("=" * 65)

    seq = torch.tensor([[65, 272, 32, 296, 256]])

    tok_vecs = tok_emb(seq).squeeze(0).detach()
    pos_sin = sin_pe(seq).squeeze(0).detach()
    combined = tok_vecs + pos_sin

    print("\nSame token ('A', id=65) at different positions would get:")
    a_tok = tok_emb(torch.tensor([[65]])).squeeze().detach()
    for pos in [0, 1, 5, 50]:
        result = a_tok + sin_vecs[pos]
        print(f"  pos {pos:>2}: [{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}, {result[3]:.3f}, ...]")

    print(f"\n'A' at pos 0 vs 'A' at pos 1:  similarity = {cosine_sim(a_tok + sin_vecs[0], a_tok + sin_vecs[1]):.3f}")
    print(f"'A' at pos 0 vs 'A' at pos 50: similarity = {cosine_sim(a_tok + sin_vecs[0], a_tok + sin_vecs[50]):.3f}")

    # =========================================================
    # 5. Comparison summary
    # =========================================================
    print(f"\n\n{'=' * 65}")
    print("COMPARISON")
    print("=" * 65)
    print(f"""
  {'Property':<35} {'Sinusoidal':<18} {'Learned'}
  {'-'*35} {'-'*18} {'-'*18}
  {'Trainable parameters':<35} {'0':<18} {max_seq_len * d_model:,}
  {'Works beyond max_seq_len':<35} {'In theory, yes':<18} {'No'}
  {'Structure at init':<35} {'Yes (smooth)':<18} {'None (random)'}
  {'Can adapt to data':<35} {'No (fixed)':<18} {'Yes'}
  {'Used by':<35} {'Original Transformer':<18} {'GPT-2, GPT-3'}
""")


if __name__ == "__main__":
    main()
