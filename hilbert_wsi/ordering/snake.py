from __future__ import annotations

import torch
from torch import Tensor

from hilbert_wsi.ordering.base import OrderingScheme, quantize


class SnakeOrdering(OrderingScheme):
    """Boustrophedon (snake / raster-zigzag) ordering. Strong baseline."""

    name = "snake"

    def __call__(self, coords: Tensor) -> Tensor:
        if coords.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=coords.device)
        cells, _ = quantize(coords)
        x = cells[:, 0]
        y = cells[:, 1]
        width = int(x.amax().item()) + 1
        # Flip x within odd rows to form the snake
        flipped = torch.where(y % 2 == 0, x, (width - 1) - x)
        idx = y.long() * width + flipped.long()
        return torch.argsort(idx, stable=True)
