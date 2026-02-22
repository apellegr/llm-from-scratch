"""Full GPT-style decoder-only language model."""

import torch
import torch.nn as nn

from .embeddings import TokenEmbedding, PositionalEmbedding
from .transformer import DecoderBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding weights with output head
        self.head.weight = self.token_emb.embedding.weight

    def forward(
        self, idx: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        idx: (batch, seq_len) token indices
        Returns: (batch, seq_len, vocab_size) logits
        """
        x = self.token_emb(idx) + self.pos_emb(idx)
        x = self.dropout(x)

        # Build causal mask if none provided
        if mask is None:
            seq_len = idx.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device))
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
