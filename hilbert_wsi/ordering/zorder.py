from __future__ import annotations

import torch
from torch import Tensor

from hilbert_wsi.ordering.base import OrderingScheme, quantize


class ZOrderOrdering(OrderingScheme):
    """Order patches along the Morton (Z-order) curve."""

    name = "zorder"

    def __call__(self, coords: Tensor) -> Tensor:
        if coords.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=coords.device)
        cells, k = quantize(coords)
        idx = _morton(cells[:, 0], cells[:, 1], k)
        return torch.argsort(idx, stable=True)


def _morton(x: Tensor, y: Tensor, k: int) -> Tensor:
    out = torch.zeros_like(x, dtype=torch.long)
    for i in range(k):
        out = out | ((x >> i) & 1) << (2 * i)
        out = out | ((y >> i) & 1) << (2 * i + 1)
    return out
