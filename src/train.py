"""Training loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        inputs = batch[:, :-1].to(device)   # all tokens except last
        targets = batch[:, 1:].to(device)   # all tokens except first

        logits = model(inputs)

        # Cross-entropy loss: reshape to (batch * seq_len, vocab_size) vs (batch * seq_len,)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
