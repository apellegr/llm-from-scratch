"""Demo: train our GPT model on WikiMed text and watch it learn."""

import os
import time

import torch
from torch.utils.data import Dataset, DataLoader

from src.model import GPT
from src.generate import generate


class TextDataset(Dataset):
    """Chunk raw text into fixed-length sequences of byte token IDs."""

    def __init__(self, text: str, seq_len: int):
        data = list(text.encode("utf-8"))
        # Truncate to a multiple of (seq_len + 1) so all chunks are the same size
        # +1 because we need one extra token for the target at each position
        n = len(data) // (seq_len + 1) * (seq_len + 1)
        data = data[:n]
        self.chunks = torch.tensor(data, dtype=torch.long).view(-1, seq_len + 1)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def sample_from_model(model, prompt: str, max_tokens: int = 100, temperature: float = 0.8):
    """Generate text from a prompt."""
    model.eval()
    idx = torch.tensor([list(prompt.encode("utf-8"))], device=next(model.parameters()).device)
    with torch.no_grad():
        out = generate(model, idx, max_new_tokens=max_tokens, temperature=temperature, top_k=50)
    return bytes(out[0].tolist()).decode("utf-8", errors="replace")


def main():
    # --- Config (sized for CPU training in ~5-10 minutes) ---
    seq_len = 64
    batch_size = 32
    n_epochs = 5
    lr = 1e-3
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256
    vocab_size = 256  # byte-level
    max_data_chars = 200_000  # use ~200KB of text to keep training fast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data ---
    data_path = os.path.join(os.path.dirname(__file__), "data", "wikimed_train.txt")
    with open(data_path) as f:
        text = f.read(max_data_chars)

    print("=" * 70)
    print("TRAINING OUR GPT MODEL")
    print("=" * 70)
    print(f"\nData: {len(text):,} characters ({len(text)/1024/1024:.1f} MB)")
    print(f"Device: {device}")

    dataset = TextDataset(text, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Sequences: {len(dataset):,} chunks of {seq_len} bytes")
    print(f"Batches per epoch: {len(dataloader)}")

    # --- Create model ---
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_len,
        dropout=0.1,
    ).to(device)

    print(f"\nModel: {n_layers} layers, d_model={d_model}, {n_heads} heads")
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- Sample before training ---
    print(f"\n{'=' * 70}")
    print("BEFORE TRAINING")
    print("=" * 70)

    prompts = ["The patient", "Blood pressure", "Treatment of"]
    for p in prompts:
        out = sample_from_model(model, p, max_tokens=60, temperature=1.0)
        print(f"  {p!r} → {out!r}")

    # --- Train ---
    print(f"\n{'=' * 70}")
    print("TRAINING")
    print("=" * 70)
    print()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in dataloader:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time

        # Sample every 2 epochs
        if epoch % 2 == 0 or epoch == 1:
            sample = sample_from_model(model, "The patient", max_tokens=80, temperature=0.8)
            print(f"  Epoch {epoch:>2}/{n_epochs}  loss={avg_loss:.3f}  "
                  f"time={elapsed:.1f}s  sample: {sample!r}")
        else:
            print(f"  Epoch {epoch:>2}/{n_epochs}  loss={avg_loss:.3f}  time={elapsed:.1f}s")

    # --- Sample after training ---
    print(f"\n{'=' * 70}")
    print("AFTER TRAINING")
    print("=" * 70)
    print()

    for p in prompts:
        out = sample_from_model(model, p, max_tokens=100, temperature=0.8)
        print(f"  Prompt: {p!r}")
        print(f"  Output: {out!r}")
        print()

    # --- Temperature comparison ---
    print(f"{'=' * 70}")
    print("TEMPERATURE COMPARISON")
    print("=" * 70)
    print()

    prompt = "The diagnosis"
    for temp in [0.3, 0.8, 1.5]:
        torch.manual_seed(42)
        out = sample_from_model(model, prompt, max_tokens=80, temperature=temp)
        print(f"  temp={temp}: {out!r}")
        print()


if __name__ == "__main__":
    main()
