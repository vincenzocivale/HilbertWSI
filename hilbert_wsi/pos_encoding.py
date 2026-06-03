"""Positional encoding utilities for tile-coordinate-based 2D PE.

Used by TileCoordEncoder for the fair 1D-ordering vs 2D-PE comparison.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TwoDSinCosPositionalEncoding(nn.Module):
    """2D sinusoidal positional encoding from tile (x, y) coordinates.

    Coordinates are mapped to **integer grid indices** per slide (origin shifted
    to 0, divided by the inferred patch stride), then encoded with sin/cos at
    multiple frequencies and concatenated. Parameter-free.

    Why grid indices and not [0, 1] normalisation: with coords squashed to
    [0, 1] the highest frequency (f_0 = 1.0) sweeps < 1 radian across the whole
    slide, so every tile receives a near-identical PE (neighbour/random cosine
    both ≈ 0.998) and the encoding carries almost no positional signal. Mapping
    to integer grid indices 0,1,2,…,~150 makes the highest frequency resolve a
    single tile (adjacent tiles differ by ~1 rad) while the lowest frequency
    spans the slide — the standard NLP sin/cos behaviour applied per axis.

    The mapping is invariant to global translation (origin shift) and uniform
    scaling of the coordinates (the inferred stride absorbs the scale), so the
    encoding does not depend on whether coords are in level-0 px, microns, etc.

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
                    Mapped to integer grid indices per slide internally.
        Returns:
            pe: (B, N, embed_dim) positional encoding, float32 regardless of coords dtype.
        """
        grid = _grid_indices(coords)              # (B, N, 2) float32 grid indices
        x = grid[..., 0]                          # (B, N)
        y = grid[..., 1]

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


def _grid_indices(coords: Tensor) -> Tensor:
    """Map (B, N, 2) coords to per-slide integer grid indices (returned as float32).

    Per slide: shift origin to the minimum, divide (floor) by the smallest
    positive coordinate delta along either axis (the inferred patch stride).
    Mirrors ``hilbert_wsi.ordering.base.quantize`` but batched and unit-agnostic.
    Computed in integer arithmetic so a uniform scale of the input (e.g. ×256)
    yields identical indices (exact scale invariance, no float rounding drift).
    """
    if coords.is_floating_point():
        c = coords.round().long()
    else:
        c = coords.long()
    cmin = c.amin(dim=1, keepdim=True)                       # (B, 1, 2)
    shifted = c - cmin                                       # (B, N, 2) >= 0

    flat = shifted.reshape(shifted.shape[0], -1)             # (B, 2N)
    big = torch.iinfo(torch.long).max
    flat_pos = flat.masked_fill(flat <= 0, big)              # ignore the zero origin
    step = flat_pos.min(dim=1).values.clamp_min(1)           # (B,) inferred stride
    grid = torch.div(shifted, step.view(-1, 1, 1), rounding_mode="floor")
    return grid.to(torch.float32)
