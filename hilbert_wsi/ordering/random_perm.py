from __future__ import annotations

import torch
from torch import Tensor

from hilbert_wsi.ordering.base import OrderingScheme


class RandomOrdering(OrderingScheme):
    """Random permutation. Negative-control baseline (spatial info destroyed)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def __call__(self, coords: Tensor) -> Tensor:
        n = coords.shape[0]
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=coords.device)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        return torch.randperm(n, generator=g).to(coords.device)
