"""Token and positional embeddings."""

import math

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


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (original Transformer)."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Build the full encoding table once, shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len).unsqueeze(1).float()  # (max_seq_len, 1)

        # Frequency term: 10000^(2i / d_model) for each dimension pair
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

        # Register as a buffer (not a parameter — no gradients)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return self.pe[:, :seq_len, :]
