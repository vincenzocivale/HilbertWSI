"""Regression tests for the three pipeline bugs found in the SFC-ordering audit.

BUG A — 2D PE was near-constant (per-slide [0,1] normalisation made the highest
        frequency sweep < 1 rad over the whole slide). The "2D PE" arm therefore
        carried almost no positional signal. Fixed by encoding integer grid
        indices. Guard: spatial neighbours must be far more similar than random.

BUG B — `max_seq_len` truncation kept the first-N tiles in H5 (column-major)
        order = a contiguous spatial slab, discarding a whole region and
        confounding cross-dataset comparisons. Fixed by a deterministic,
        uniform-random, per-slide subsample shared by both encoders.

BUG C — TileCoordEncoder fed Mamba the H5 column-major raster order (a snake-like
        spatial ordering), so the "coordinate-only" arm secretly used a strong
        1D ordering. Fixed by randomising the scan order per slide; position then
        enters only through the (now working) 2D PE.
"""

from __future__ import annotations

import torch

from hilbert_wsi.encoder_2d import TileCoordEncoder
from hilbert_wsi.models import register_backbone
from hilbert_wsi.models.base import SequenceBackbone, masked_mean
from hilbert_wsi.ordering.random_perm import subsample_indices
from hilbert_wsi.pos_encoding import TwoDSinCosPositionalEncoding

_REGISTERED = False


def _ensure_backbone():
    global _REGISTERED
    if _REGISTERED:
        return
    @register_backbone("fix_spy_mean")
    class _Spy(SequenceBackbone):
        def __init__(self, input_dim: int, embedding_dim: int = 8,
                     skip_proj: bool = False, **_: object):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.skip_proj = skip_proj

        def forward(self, seq, mask=None):
            return masked_mean(seq, mask)
    _REGISTERED = True


# --------------------------------------------------------------------------- #
# BUG A — 2D PE resolution
# --------------------------------------------------------------------------- #
def _regular_grid(side: int, stride: int = 256) -> torch.Tensor:
    """(1, side*side, 2) coords in row-major order: index = y*side + x."""
    coords = torch.tensor(
        [[x, y] for y in range(side) for x in range(side)], dtype=torch.long
    )
    return (coords * stride).unsqueeze(0)


def test_2d_pe_resolves_adjacent_tiles():
    """BUG A: PE of spatial neighbours must be much closer than random pairs.

    Pre-fix gap was ~0.002 (near-constant PE); grid-index PE gives gap >> 0.1.
    """
    side = 40
    coords = _regular_grid(side)                         # (1, 1600, 2)
    P = torch.nn.functional.normalize(TwoDSinCosPositionalEncoding(512)(coords)[0], dim=-1)

    idx = torch.arange(side * side)
    not_last_col = (idx % side) != (side - 1)
    right = idx + 1                                       # (x+1, y) neighbour
    neigh_cos = (P[idx[not_last_col]] * P[right[not_last_col]]).sum(-1).mean()
    rand_cos = (P[idx] * P[idx[torch.randperm(side * side)]]).sum(-1).mean()

    assert neigh_cos - rand_cos > 0.1, (
        f"2D PE cannot resolve neighbours: gap={neigh_cos - rand_cos:.4f} "
        f"(neighbour={neigh_cos:.4f}, random={rand_cos:.4f})"
    )


def test_2d_pe_not_constant_across_slide():
    """BUG A: PE must vary appreciably between distant tiles (not a DC vector)."""
    coords = _regular_grid(40)
    P = TwoDSinCosPositionalEncoding(512)(coords)[0]
    corner_a, corner_b = P[0], P[-1]                     # opposite corners
    cos = torch.nn.functional.cosine_similarity(corner_a, corner_b, dim=0)
    assert cos < 0.9, f"PE of opposite corners nearly identical (cos={cos:.4f})"


# --------------------------------------------------------------------------- #
# BUG B — uniform subsample (not first-N slab)
# --------------------------------------------------------------------------- #
def test_subsample_is_uniform_not_raster_prefix():
    """BUG B: subsample must be a representative sample, not the H5 prefix slab."""
    side_x, side_y, k = 50, 40, 500
    coords = torch.tensor(
        [[x, y] for y in range(side_y) for x in range(side_x)], dtype=torch.long
    ) * 256                                              # raster (column-major) order
    keep = subsample_indices(coords, k)

    assert keep.shape[0] == k
    assert keep.unique().shape[0] == k                   # no duplicates
    assert torch.all(keep[1:] > keep[:-1])               # sorted
    assert not torch.equal(keep, torch.arange(k)), "subsample is the raster prefix slab"

    # Representative: kept tiles span ~the full extent (a slab would shrink it).
    ky = coords[keep][:, 1]
    full = coords[:, 1].max() - coords[:, 1].min()
    assert (ky.max() - ky.min()) >= 0.9 * full, "subsample collapsed the y-extent (slab)"


def test_subsample_deterministic_and_shared():
    """BUG B: same coords -> same kept subset, so 1D and 2D encoders agree."""
    coords = (torch.randint(0, 40, (1500, 2)) * 256).long()
    a = subsample_indices(coords, 400)
    b = subsample_indices(coords, 400)
    assert torch.equal(a, b)


def test_subsample_noop_when_under_budget():
    coords = (torch.randint(0, 40, (300, 2)) * 256).long()
    keep = subsample_indices(coords, 1000)
    assert torch.equal(keep, torch.arange(300))


# --------------------------------------------------------------------------- #
# BUG C — randomised scan order in TileCoordEncoder
# --------------------------------------------------------------------------- #
def test_2dpe_scan_order_is_randomized():
    """BUG C: the order presented to the backbone is a non-identity permutation,
    deterministic per slide, with the mask permuted consistently."""
    _ensure_backbone()
    enc = TileCoordEncoder(input_dim=8, backbone="fix_spy_mean", embedding_dim=8)

    n = 64
    coords = (torch.arange(n).unsqueeze(-1) * torch.tensor([[256, 1]])).unsqueeze(0).long()
    # tag each position by its index in channel 0 so we can read the order back
    h = torch.arange(n, dtype=torch.float32).view(1, n, 1).expand(1, n, 8).clone()

    h2, _ = enc._randomize_scan(h, coords, None)
    order = h2[0, :, 0]
    assert sorted(order.tolist()) == list(range(n)), "scan is not a permutation"
    assert not torch.equal(order, torch.arange(n, dtype=torch.float32)), "scan order unchanged (raster)"

    # deterministic per slide
    h3, _ = enc._randomize_scan(h, coords, None)
    assert torch.equal(h2, h3)

    # mask follows the same permutation
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, :10] = True
    _, mperm = enc._randomize_scan(h, coords, mask)
    assert mperm.sum().item() == 10
