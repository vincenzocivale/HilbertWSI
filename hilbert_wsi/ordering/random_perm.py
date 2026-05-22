from __future__ import annotations

import hashlib

import torch
from torch import Tensor

from hilbert_wsi.ordering.base import OrderingScheme


def _slide_seed(coords: Tensor, base_seed: int) -> int:
    """Deterministic 31-bit seed derived from coords + ``base_seed``.

    Stable across runs (blake2b, not Python's randomised ``hash``) so the same
    slide always gets the same permutation given a fixed ``base_seed``.
    """
    h = hashlib.blake2b(
        coords.detach().cpu().contiguous().numpy().tobytes(),
        digest_size=8,
    )
    h.update(int(base_seed & 0xFFFFFFFF).to_bytes(4, "little"))
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


class RandomOrdering(OrderingScheme):
    """Random permutation. Negative-control baseline (spatial info destroyed).

    The permutation is **per-slide**: derived from a deterministic hash of the
    slide's coords combined with ``base_seed``. Two slides therefore receive
    different permutations even when they contain the same number of tiles.
    Use ``base_seed`` to sweep multiple random orderings for multi-seed CIs.
    """

    name = "random"

    def __init__(self, base_seed: int = 0, seed: int | None = None):
        # ``seed`` kept as alias for backward-compat with existing configs.
        self.base_seed = int(seed if seed is not None else base_seed)

    def __call__(self, coords: Tensor) -> Tensor:
        n = coords.shape[0]
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=coords.device)
        g = torch.Generator(device="cpu")
        g.manual_seed(_slide_seed(coords, self.base_seed))
        return torch.randperm(n, generator=g).to(coords.device)
