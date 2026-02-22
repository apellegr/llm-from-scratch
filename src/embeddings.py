"""Token and positional embeddings."""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale embeddings by sqrt(d_model) as in "Attention Is All You Need"
        return self.embedding(x) * (self.d_model ** 0.5)


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings (GPT-style)."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.embedding(positions)
