"""Demo: visualize attention weights and see how tokens attend to each other."""

import torch
import torch.nn.functional as F

from src.attention import MultiHeadAttention


def show_attention_weights(weights: torch.Tensor, tokens: list[str], head: int):
    """Display an attention weight matrix for one head."""
    w = weights[0, head].detach()  # (seq_len, seq_len)
    seq_len = len(tokens)

    # Header
    print(f"\n  Head {head} attention weights:")
    print(f"  {'':>12}", end="")
    for t in tokens:
        print(f"  {t:>8}", end="")
    print(f"  {'(max)':>8}")
    print(f"  {'':>12}", end="")
    for _ in tokens:
        print(f"  {'--------':>8}", end="")
    print(f"  {'--------':>8}")

    # Each row = what this token attends to
    for i in range(seq_len):
        print(f"  {tokens[i]:>10} →", end="")
        max_j = w[i, :i+1].argmax().item() if i >= 0 else 0
        for j in range(seq_len):
            val = w[i, j].item()
            if val < 0.001:
                print(f"  {'---':>8}", end="")
            else:
                marker = " *" if j == max_j and val > 0.01 else "  "
                print(f"{marker}{val:>6.3f}", end="")
        # Show which token gets most attention
        print(f"  → {tokens[max_j]}")


def main():
    torch.manual_seed(42)

    d_model = 32
    n_heads = 4
    d_k = d_model // n_heads  # 8

    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)

    print("=" * 70)
    print("ATTENTION DEMO")
    print("=" * 70)
    print(f"\nTokens: {tokens}")
    print(f"d_model={d_model}, n_heads={n_heads}, d_k={d_k}")

    # Create fake embeddings (random, since we haven't trained)
    x = torch.randn(1, seq_len, d_model)

    # =========================================================
    # 1. Show Q, K, V projections
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 1: Q, K, V PROJECTIONS")
    print("=" * 70)

    attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)

    with torch.no_grad():
        Q = attn.W_q(x)
        K = attn.W_k(x)
        V = attn.W_v(x)

    print(f"\nInput x shape:  {list(x.shape)}   (batch, seq_len, d_model)")
    print(f"Q shape:        {list(Q.shape)}   (same — each token gets a query vector)")
    print(f"K shape:        {list(K.shape)}")
    print(f"V shape:        {list(V.shape)}")

    print(f"\nSame token, three different projections (first 8 dims):")
    print(f"  x[0,'The']:  {x[0, 0, :8].tolist()}")
    print(f"  Q[0,'The']:  {[round(v, 3) for v in Q[0, 0, :8].tolist()]}")
    print(f"  K[0,'The']:  {[round(v, 3) for v in K[0, 0, :8].tolist()]}")
    print(f"  V[0,'The']:  {[round(v, 3) for v in V[0, 0, :8].tolist()]}")

    # =========================================================
    # 2. Show raw attention scores before and after mask
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 2: ATTENTION SCORES (Q · K^T / sqrt(d_k))")
    print("=" * 70)

    with torch.no_grad():
        # Reshape into heads
        Q_heads = Q.view(1, seq_len, n_heads, d_k).transpose(1, 2)
        K_heads = K.view(1, seq_len, n_heads, d_k).transpose(1, 2)

        # Raw scores for head 0
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (d_k ** 0.5)

    s = scores[0, 0].detach()
    print(f"\nRaw scores (head 0) — before mask and softmax:")
    print(f"  {'':>10}", end="")
    for t in tokens:
        print(f"  {t:>6}", end="")
    print()
    for i in range(seq_len):
        print(f"  {tokens[i]:>10}", end="")
        for j in range(seq_len):
            print(f"  {s[i, j].item():>6.2f}", end="")
        print()

    # Apply causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    masked_scores = scores.clone()
    masked_scores = masked_scores.masked_fill(causal_mask == 0, float("-inf"))

    ms = masked_scores[0, 0].detach()
    print(f"\nAfter causal mask (future = -inf):")
    print(f"  {'':>10}", end="")
    for t in tokens:
        print(f"  {t:>6}", end="")
    print()
    for i in range(seq_len):
        print(f"  {tokens[i]:>10}", end="")
        for j in range(seq_len):
            val = ms[i, j].item()
            if val == float("-inf"):
                print(f"  {'-inf':>6}", end="")
            else:
                print(f"  {val:>6.2f}", end="")
        print()

    # =========================================================
    # 3. Softmax → attention weights per head
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 3: SOFTMAX → ATTENTION WEIGHTS (per head)")
    print("=" * 70)
    print("Each row sums to 1.0. '*' marks where each token attends most.")
    print("'---' = masked (cannot see future tokens)")

    with torch.no_grad():
        attn_weights = F.softmax(masked_scores, dim=-1)

    for h in range(n_heads):
        show_attention_weights(attn_weights, tokens, h)

    # =========================================================
    # 4. Show how output blends values
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 4: WEIGHTED SUM OF VALUES")
    print("=" * 70)

    with torch.no_grad():
        V_heads = V.view(1, seq_len, n_heads, d_k).transpose(1, 2)
        attn_output = torch.matmul(attn_weights, V_heads)

    print(f"\nHead 0: how 'sat' (position 2) blends value vectors:")
    w = attn_weights[0, 0, 2].detach()
    for i in range(3):  # sat can see positions 0, 1, 2
        v = V_heads[0, 0, i, :4].detach()
        print(f"  {w[i].item():.3f} × V_{tokens[i]:>3} {[round(x, 3) for x in v.tolist()]}")

    result = attn_output[0, 0, 2, :4].detach()
    print(f"  = output    {[round(x, 3) for x in result.tolist()]}")
    print(f"\n  'sat' now carries a blend of information from 'The', 'cat', and itself.")

    # =========================================================
    # 5. Full forward pass
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 5: FULL MULTI-HEAD ATTENTION (all steps combined)")
    print("=" * 70)

    with torch.no_grad():
        output = attn(x, mask=causal_mask)

    print(f"\nInput shape:  {list(x.shape)}")
    print(f"Output shape: {list(output.shape)}  (same shape — attention doesn't change dimensions)")
    print(f"\nBut the content is different — each token now has context:")
    for i in range(seq_len):
        in_vec = x[0, i, :4].tolist()
        out_vec = output[0, i, :4].tolist()
        print(f"  {tokens[i]:>5}  in: [{in_vec[0]:>6.3f}, {in_vec[1]:>6.3f}, ...]"
              f"  →  out: [{out_vec[0]:>6.3f}, {out_vec[1]:>6.3f}, ...]")


if __name__ == "__main__":
    main()
