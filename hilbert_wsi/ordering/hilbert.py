from __future__ import annotations

import torch
from torch import Tensor

from hilbert_wsi.ordering.base import OrderingScheme, quantize


class HilbertOrdering(OrderingScheme):
    """Order patches along a Hilbert space-filling curve."""

    name = "hilbert"

    def __call__(self, coords: Tensor) -> Tensor:
        if coords.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=coords.device)
        cells, k = quantize(coords)
        idx = _hilbert_xy2d(cells[:, 0], cells[:, 1], k)
        return torch.argsort(idx, stable=True)


def _hilbert_xy2d(x: Tensor, y: Tensor, k: int) -> Tensor:
    """Vectorised Hilbert curve index for a ``2**k`` x ``2**k`` grid.

    Reference: Wikipedia "Hilbert curve" iterative bit interleave algorithm.
    """
    x = x.clone().long()
    y = y.clone().long()
    d = torch.zeros_like(x)
    n = 1 << k
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).long()
        ry = ((y & s) > 0).long()
        d = d + s * s * ((3 * rx) ^ ry)
        # Rotate quadrant
        flip = ry == 0
        if flip.any():
            mask = flip & (rx == 1)
            if mask.any():
                x = torch.where(mask, s - 1 - x, x)
                y = torch.where(mask, s - 1 - y, y)
            swap_x = torch.where(flip, y, x)
            swap_y = torch.where(flip, x, y)
            x, y = swap_x, swap_y
        s >>= 1
    return d
