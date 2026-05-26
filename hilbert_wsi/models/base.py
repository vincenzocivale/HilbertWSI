from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn


class SequenceBackbone(nn.Module):
    """Sequence encoder over ordered tile embeddings.

    Inputs:
        seq: ``(B, N, D_in)`` tile embeddings, already reordered.
        mask: optional ``(B, N)`` bool tensor, True for valid positions.

    Output:
        ``(B, D_out)`` slide-level embedding.
    """

    embedding_dim: int

    @abstractmethod
    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor: ...


def masked_mean(seq: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return seq.mean(dim=1)
    m = mask.unsqueeze(-1).to(seq.dtype)
    return (seq * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class GatedAttentionPool(nn.Module):
    """Gated attention pooling (ABMIL-style): learned weighted sum over N tokens.

    Masks padded positions before softmax so they don't contribute to the
    denominator (masked_fill(-inf), not multiply-by-zero).
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        scores = self.proj(x)                              # (B, N, 1)
        if mask is not None:
            scores = scores.masked_fill(
                ~mask.unsqueeze(-1).bool(), float("-inf")
            )
        weights = torch.softmax(scores, dim=1)             # (B, N, 1)
        return (weights * x).sum(dim=1)                    # (B, D)
