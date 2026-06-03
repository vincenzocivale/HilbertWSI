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


def subsample_indices(coords: Tensor, max_seq_len: int, base_seed: int = 0) -> Tensor:
    """Deterministic per-slide **uniform-random** subsample of tile indices.

    Returns sorted indices to keep, of length ``min(N, max_seq_len)``. The
    sample is drawn uniformly over all N tiles, so it is spatially
    representative — unlike a first-``max_seq_len`` truncation in H5 arrival
    order, which keeps a contiguous spatial slab (e.g. the left/top of the
    slide) and discards a whole region.

    The seed derives from ``coords`` (via :func:`_slide_seed`), so two encoders
    given the same slide + ``max_seq_len`` keep the **same** subset (fairness),
    and the choice is stable across runs.
    """
    n = coords.shape[0]
    if max_seq_len is None or n <= max_seq_len:
        return torch.arange(n, device=coords.device)
    g = torch.Generator(device="cpu")
    g.manual_seed(_slide_seed(coords, base_seed))
    keep = torch.randperm(n, generator=g)[:max_seq_len]
    return keep.sort().values.to(coords.device)
