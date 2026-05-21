from __future__ import annotations

import torch
from torch import Tensor

from hilbert_wsi.ordering.base import OrderingScheme, quantize


class PeanoOrdering(OrderingScheme):
    """Order patches along the Peano space-filling curve (base-3)."""

    name = "peano"

    def __call__(self, coords: Tensor) -> Tensor:
        if coords.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=coords.device)
        cells, k = quantize(coords)
        side = 1 << k
        m = 1
        while m < side:
            m *= 3
        idx = _peano_index(cells[:, 0], cells[:, 1], m)
        return torch.argsort(idx, stable=True)


def _peano_index(x: Tensor, y: Tensor, m: int) -> Tensor:
    """Map (x, y) on an ``m`` x ``m`` grid (m=3**p) to Peano curve index."""
    out = torch.zeros_like(x, dtype=torch.long)
    fx = x.clone().long()
    fy = y.clone().long()
    block = m
    rev_x = torch.zeros_like(fx, dtype=torch.bool)
    rev_y = torch.zeros_like(fy, dtype=torch.bool)
    while block > 1:
        block //= 3
        col = (fx // block) % 3
        row = (fy // block) % 3
        c = torch.where(rev_x, 2 - col, col)
        r = torch.where(rev_y, 2 - row, row)
        cell = torch.where(c % 2 == 0, c * 3 + r, c * 3 + (2 - r))
        out = out * 9 + cell
        rev_y = rev_y ^ (c % 2 == 1)
        rev_x = rev_x
    return out
