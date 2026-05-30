"""Positional encoding utilities for tile-coordinate-based 2D PE.

Used by TileCoordEncoder for the fair 1D-ordering vs 2D-PE comparison.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TwoDSinCosPositionalEncoding(nn.Module):
    """2D sinusoidal positional encoding from tile (x, y) coordinates.

    Coordinates are normalized per-slide to [0, 1], then encoded with
    sin/cos at multiple frequencies and concatenated. Parameter-free.

    Encoding layout (embed_dim = 4k):
        [sin(f_0*x), ..., sin(f_{k-1}*x),   <- first quarter
         cos(f_0*x), ..., cos(f_{k-1}*x),   <- second quarter
         sin(f_0*y), ..., sin(f_{k-1}*y),   <- third quarter
         cos(f_0*y), ..., cos(f_{k-1}*y)]   <- fourth quarter

    Frequencies: f_i = 1 / 10000^(2i / (embed_dim/2)), matching the
    standard 1D sin/cos PE formula applied separately to x and y.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4 for 2D sin/cos PE, got {embed_dim}")
        self.embed_dim = embed_dim

    def forward(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords: (B, N, 2) tile coordinates in any unit (pixels, microns, …).
                    Normalized per-slide internally.
        Returns:
            pe: (B, N, embed_dim) positional encoding, float32 regardless of coords dtype.
        """
        x = coords[..., 0].float()   # (B, N)
        y = coords[..., 1].float()

        # Normalize each axis per slide to [0, 1].
        x = _normalize_per_slide(x)
        y = _normalize_per_slide(y)

        quarter = self.embed_dim // 4
        half = self.embed_dim // 2

        i = torch.arange(quarter, device=coords.device, dtype=torch.float32)
        freq = 1.0 / (10000.0 ** (2.0 * i / half))   # (quarter,)

        x_enc = x.unsqueeze(-1) * freq   # (B, N, quarter)
        y_enc = y.unsqueeze(-1) * freq

        pe = torch.cat(
            [x_enc.sin(), x_enc.cos(), y_enc.sin(), y_enc.cos()],
            dim=-1,
        )                                # (B, N, embed_dim) — float32 always
        return pe


def _normalize_per_slide(t: Tensor) -> Tensor:
    """Normalize each row (slide) of a (B, N) tensor to [0, 1]."""
    lo = t.min(dim=1, keepdim=True).values
    hi = t.max(dim=1, keepdim=True).values
    return (t - lo) / (hi - lo + 1e-8)
