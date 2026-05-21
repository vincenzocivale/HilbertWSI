from __future__ import annotations

import torch
from torch import Tensor

from hilbert_wsi.ordering.base import OrderingScheme, quantize
from hilbert_wsi.ordering.hilbert import _hilbert_xy2d


class MooreOrdering(OrderingScheme):
    """Order patches along a Moore curve (closed Hilbert variant).

    Built as four rotated Hilbert quadrants on a ``2**k`` grid so that the
    traversal forms a closed loop.
    """

    name = "moore"

    def __call__(self, coords: Tensor) -> Tensor:
        if coords.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=coords.device)
        cells, k = quantize(coords)
        k = max(1, k)
        half = 1 << (k - 1)
        x = cells[:, 0]
        y = cells[:, 1]
        quad = (x >= half).long() * 2 + (y >= half).long()
        lx = x % half
        ly = y % half
        sub_k = max(0, k - 1)
        base = _hilbert_xy2d(lx, ly, sub_k) if sub_k > 0 else torch.zeros_like(x)
        per_quad = (1 << (2 * sub_k)) if sub_k > 0 else 1
        # Quadrant order: 0 (top-left), 1 (top-right), 2 (bottom-right), 3 (bottom-left)
        moore_quad = torch.empty_like(quad)
        moore_quad[quad == 0] = 0
        moore_quad[quad == 1] = 3
        moore_quad[quad == 2] = 1
        moore_quad[quad == 3] = 2
        idx = moore_quad * per_quad + base
        return torch.argsort(idx, stable=True)
