"""Text generation / inference utilities."""

import torch
import torch.nn.functional as F

from .model import GPT


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Autoregressive generation.

    idx: (batch, seq_len) — starting token indices
    Returns: (batch, seq_len + max_new_tokens)
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Crop to max sequence length the model supports
        idx_cond = idx[:, -model.max_seq_len :]

        # Forward pass
        logits = model(idx_cond)

        # Take logits for the last position and apply temperature
        logits = logits[:, -1, :] / temperature

        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        idx = torch.cat([idx, idx_next], dim=1)

    return idx
