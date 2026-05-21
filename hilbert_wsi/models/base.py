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
