"""Demo: watch token vectors evolve through stacked transformer blocks."""

import torch
import torch.nn.functional as F

from src.transformer import DecoderBlock


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=0).item()


def main():
    torch.manual_seed(42)

    d_model = 32
    n_heads = 4
    d_ff = 128
    n_layers = 6

    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)

    # Causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    # Fake embeddings (random, untrained)
    x = torch.randn(1, seq_len, d_model)

    # Create stacked blocks (each with its own weights)
    blocks = [DecoderBlock(d_model, n_heads, d_ff, dropout=0.0) for _ in range(n_layers)]

    # =========================================================
    # 1. Track how vectors change layer by layer
    # =========================================================
    print("=" * 70)
    print("HOW TOKEN VECTORS EVOLVE THROUGH LAYERS")
    print("=" * 70)
    print(f"\n{n_layers} blocks, d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"Tokens: {tokens}\n")

    snapshots = [x.detach().clone()]

    current = x
    with torch.no_grad():
        for block in blocks:
            current = block(current, mask)
            snapshots.append(current.detach().clone())

    # Show how much each token's vector changes per layer
    print(f"Cosine similarity between a token's vector at layer N vs layer 0:")
    print(f"(1.0 = unchanged, 0.0 = completely different)\n")
    print(f"  {'Layer':>5}", end="")
    for t in tokens:
        print(f"  {t:>8}", end="")
    print()
    print(f"  {'-----':>5}", end="")
    for _ in tokens:
        print(f"  {'--------':>8}", end="")
    print()

    for layer_idx in range(1, n_layers + 1):
        print(f"  {layer_idx:>5}", end="")
        for tok_idx in range(seq_len):
            sim = cosine_sim(snapshots[0][0, tok_idx], snapshots[layer_idx][0, tok_idx])
            print(f"  {sim:>8.3f}", end="")
        print()

    # =========================================================
    # 2. Track how tokens become more similar to each other
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("HOW TOKENS RELATE TO EACH OTHER (cosine similarity)")
    print("=" * 70)

    pairs = [
        ("The", "the", 0, 4),
        ("cat", "mat", 1, 5),
        ("cat", "sat", 1, 2),
        ("The", "mat", 0, 5),
    ]

    print(f"\n  {'Pair':<15}", end="")
    for layer_idx in range(n_layers + 1):
        label = "embed" if layer_idx == 0 else f"L{layer_idx}"
        print(f"  {label:>6}", end="")
    print()
    print(f"  {'-'*15}", end="")
    for _ in range(n_layers + 1):
        print(f"  {'------':>6}", end="")
    print()

    for label_a, label_b, idx_a, idx_b in pairs:
        print(f"  {label_a+' vs '+label_b:<15}", end="")
        for layer_idx in range(n_layers + 1):
            sim = cosine_sim(
                snapshots[layer_idx][0, idx_a],
                snapshots[layer_idx][0, idx_b],
            )
            print(f"  {sim:>6.3f}", end="")
        print()

    # =========================================================
    # 3. Residual connection: how much does each sub-layer change?
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("RESIDUAL CONNECTIONS: contribution of each sub-layer")
    print("=" * 70)
    print(f"\nFor token 'sat' (position 2), showing the magnitude of")
    print(f"the residual (what attention/ff added) vs the input at each layer:\n")

    current = x.clone()
    sat_idx = 2

    print(f"  {'Layer':>5}  {'Input norm':>10}  {'Attn added':>10}  {'FF added':>10}  {'Output norm':>11}")
    print(f"  {'-----':>5}  {'----------':>10}  {'----------':>10}  {'----------':>10}  {'-----------':>11}")

    with torch.no_grad():
        for layer_idx, block in enumerate(blocks):
            input_norm = current[0, sat_idx].norm().item()

            # Manually run the two sub-layers to see each contribution
            normed = block.ln1(current)
            attn_out = block.attention(normed, mask)
            attn_residual = attn_out[0, sat_idx].norm().item()

            after_attn = current + block.dropout(attn_out)

            normed2 = block.ln2(after_attn)
            ff_out = block.feed_forward(normed2)
            ff_residual = ff_out[0, sat_idx].norm().item()

            current = after_attn + ff_out
            output_norm = current[0, sat_idx].norm().item()

            print(f"  {layer_idx+1:>5}  {input_norm:>10.3f}  {attn_residual:>10.3f}  {ff_residual:>10.3f}  {output_norm:>11.3f}")

    print(f"\n  The residual additions are typically smaller than the input —")
    print(f"  each layer makes incremental refinements, not wholesale replacements.")

    # =========================================================
    # 4. Information propagation across layers
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("INFORMATION PROPAGATION ACROSS LAYERS")
    print("=" * 70)
    print(f"\nDoes 'mat' (position 5) gain information about 'cat' (position 1)?")
    print(f"We measure cosine similarity between 'mat' and the ORIGINAL 'cat' vector.\n")

    print(f"  {'Layer':>5}  {'sim(mat, cat₀)':>14}  {'Interpretation'}")
    print(f"  {'-----':>5}  {'--------------':>14}  {'-' * 30}")
    cat_original = snapshots[0][0, 1]
    for layer_idx in range(n_layers + 1):
        mat_vec = snapshots[layer_idx][0, 5]
        sim = cosine_sim(mat_vec, cat_original)
        label = "embed" if layer_idx == 0 else f"L{layer_idx}"
        if layer_idx == 0:
            interp = "(no information exchange yet)"
        elif layer_idx <= 2:
            interp = "(direct attention can reach 'cat')"
        else:
            interp = "(multi-hop: info flows indirectly)"
        print(f"  {label:>5}  {sim:>14.3f}  {interp}")


if __name__ == "__main__":
    main()
