from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class OrderingScheme(ABC):
    """Map 2D patch coordinates to a 1D traversal permutation.

    Implementations must be stateless, deterministic for a given seed, and
    return a permutation of ``range(N)`` as a ``torch.long`` tensor.
    """

    name: str = ""

    @abstractmethod
    def __call__(self, coords: Tensor) -> Tensor:
        """Args:
            coords: ``(N, 2)`` integer tensor of patch top-left coordinates,
                expressed at level-0 (i.e. raw pixel space).

        Returns:
            ``(N,)`` long tensor ``perm`` such that ``coords[perm]`` is the
            traversal order.
        """


def quantize(coords: Tensor) -> tuple[Tensor, int]:
    """Quantize coordinates to a square integer grid of side ``2**k``.

    Returns the grid indices and ``k``.
    """
    if coords.numel() == 0:
        return coords.long(), 0
    coords = coords.long()
    cmin = coords.amin(dim=0, keepdim=True)
    shifted = coords - cmin
    step = _infer_step(shifted)
    if step <= 0:
        step = 1
    cells = shifted // step
    extent = int(cells.amax().item()) + 1
    k = max(1, (extent - 1).bit_length())
    return cells, k


def _infer_step(shifted: Tensor) -> int:
    """Infer patch stride as the smallest positive coord delta along either axis."""
    flat = shifted.flatten()
    pos = flat[flat > 0]
    if pos.numel() == 0:
        return 1
    return int(pos.min().item())
