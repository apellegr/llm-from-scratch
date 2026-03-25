"""Demo: full pipeline from token IDs to generated text."""

import torch
import torch.nn.functional as F

from src.model import GPT
from src.generate import generate


def main():
    torch.manual_seed(42)

    vocab_size = 256  # byte-level for simplicity
    d_model = 64
    n_heads = 4
    n_layers = 4
    d_ff = 256
    max_seq_len = 128

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )

    print("=" * 70)
    print("FULL PIPELINE DEMO (untrained model)")
    print("=" * 70)
    print(f"\nModel: {n_layers} layers, d_model={d_model}, {n_heads} heads")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Vocab: {vocab_size} (raw bytes)")

    # =========================================================
    # 1. Forward pass: token IDs → logits
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 1: FORWARD PASS (input → logits)")
    print("=" * 70)

    prompt = "The cat"
    input_ids = torch.tensor([list(prompt.encode("utf-8"))])  # byte-level
    tokens = [chr(b) for b in input_ids[0].tolist()]

    print(f"\nPrompt: {prompt!r}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"Tokens: {tokens}")

    with torch.no_grad():
        logits = model(input_ids)

    print(f"\nLogits shape: {list(logits.shape)}")
    print(f"  = (batch={logits.shape[0]}, seq_len={logits.shape[1]}, vocab_size={logits.shape[2]})")
    print(f"\nEach position produces {vocab_size} scores — one per possible next byte.")

    # =========================================================
    # 2. Logits → probabilities for last position
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 2: LOGITS → PROBABILITIES (last position)")
    print("=" * 70)

    last_logits = logits[0, -1, :]  # last position
    probs = F.softmax(last_logits, dim=-1)

    print(f"\nPredicting what comes after: {prompt!r}")
    print(f"Last position logits (first 10): {[f'{v:.2f}' for v in last_logits[:10].tolist()]}")
    print(f"Sum of probabilities: {probs.sum().item():.4f}")

    # Top predictions
    top_k_vals, top_k_ids = torch.topk(probs, 10)
    print(f"\nTop 10 predicted next bytes:")
    for i in range(10):
        byte_val = top_k_ids[i].item()
        prob = top_k_vals[i].item()
        char = chr(byte_val) if 32 <= byte_val < 127 else f"\\x{byte_val:02x}"
        bar = "#" * int(prob * 100)
        print(f"  {byte_val:>3} ({char:>4}): {prob:.3f}  {bar}")

    print(f"\n  (Untrained model — predictions are near-uniform random.)")
    print(f"  After training, the model would strongly predict space or")
    print(f"  a continuation like 's', 'a', etc.")

    # =========================================================
    # 3. Temperature comparison
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 3: TEMPERATURE EFFECT")
    print("=" * 70)

    for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
        scaled = last_logits / temp
        p = F.softmax(scaled, dim=-1)
        top1 = p.max().item()
        entropy = -(p * (p + 1e-10).log()).sum().item()
        print(f"  temp={temp:<4}  top-1 prob={top1:.3f}  entropy={entropy:.2f}  "
              f"({'deterministic' if top1 > 0.5 else 'spread out' if entropy > 4 else 'moderate'})")

    # =========================================================
    # 4. Generation loop step by step
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 4: GENERATION LOOP (step by step)")
    print("=" * 70)

    print(f"\nStarting with: {prompt!r}")
    print(f"Generating 20 bytes, temperature=1.0\n")

    torch.manual_seed(123)
    idx = torch.tensor([list(prompt.encode("utf-8"))])

    model.eval()
    with torch.no_grad():
        for step in range(20):
            # Forward pass
            step_logits = model(idx[:, -max_seq_len:])
            next_logits = step_logits[:, -1, :]

            # Sample
            step_probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(step_probs, num_samples=1)
            byte_val = next_id.item()

            # Display
            char = chr(byte_val) if 32 <= byte_val < 127 else f"\\x{byte_val:02x}"
            prob = step_probs[0, byte_val].item()
            seq_so_far = idx[0].tolist() + [byte_val]
            text_so_far = bytes(seq_so_far).decode("utf-8", errors="replace")
            print(f"  Step {step+1:>2}: picked byte {byte_val:>3} ({char:>4}) "
                  f"prob={prob:.3f}  text so far: {text_so_far!r}")

            idx = torch.cat([idx, next_id], dim=1)

    # =========================================================
    # 5. Comparing temperature on full generation
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("STEP 5: FULL GENERATION AT DIFFERENT TEMPERATURES")
    print("=" * 70)

    for temp in [0.1, 0.5, 1.0, 2.0]:
        torch.manual_seed(42)
        start = torch.tensor([list(prompt.encode("utf-8"))])
        with torch.no_grad():
            out = generate(model, start, max_new_tokens=40, temperature=temp)
        text = bytes(out[0].tolist()).decode("utf-8", errors="replace")
        printable = repr(text)
        if len(printable) > 65:
            printable = printable[:62] + "...'"
        print(f"\n  temp={temp}: {printable}")

    print(f"\n  (All gibberish because the model is untrained. After training on")
    print(f"  real text, lower temperature produces coherent, predictable text")
    print(f"  while higher temperature produces more creative/random output.)")

    # =========================================================
    # 6. Weight tying
    # =========================================================
    print(f"\n\n{'=' * 70}")
    print("WEIGHT TYING")
    print("=" * 70)

    print(f"\n  Embedding weight shape:   {list(model.token_emb.embedding.weight.shape)}")
    print(f"  Output head weight shape: {list(model.head.weight.shape)}")
    print(f"  Same object in memory:    {model.head.weight is model.token_emb.embedding.weight}")
    print(f"\n  The embedding table and output head share the same matrix.")
    print(f"  Token ID → vector (embedding) and vector → token scores (head)")
    print(f"  are inverse operations, so sharing weights makes them consistent.")

    saved = vocab_size * d_model
    total = model.count_parameters()
    print(f"\n  Parameters saved by tying: {saved:,} ({saved/total*100:.1f}% of model)")


if __name__ == "__main__":
    main()
