"""Regression tests for TwoDSinCosPositionalEncoding.

Covers:
- §8.1 fix: PE must stay float32 even when coords are int64 (H5 default).
- PE values must be non-trivial (not destroyed by int cast).
- Encoding layout sanity.
"""

from __future__ import annotations

import pytest
import torch

from hilbert_wsi.pos_encoding import TwoDSinCosPositionalEncoding


def test_pe_float32_with_int64_coords():
    """§8.1: coords int64 (H5 pixel coords) must not cast PE to int64."""
    pe = TwoDSinCosPositionalEncoding(embed_dim=32)
    coords = torch.randint(0, 10000, (2, 64, 2), dtype=torch.int64)
    out = pe(coords)
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"


def test_pe_nontrivial_with_int64_coords():
    """§8.1: PE values must not be destroyed by int cast (all-zeros / all-±1)."""
    pe = TwoDSinCosPositionalEncoding(embed_dim=64)
    coords = torch.randint(0, 10000, (2, 64, 2), dtype=torch.int64)
    out = pe(coords)
    # With int cast bug, sin/cos → 0, so abs().max() ≤ 1 but mean ≈ 0.
    # Without bug, low-frequency bands produce values across [-1, 1].
    assert out.abs().max().item() > 0.5, "PE max looks zero-quantized"
    # Unique values count: int cast produces very few (0, ±1); float has many more.
    unique_values = out.flatten().unique().shape[0]
    assert unique_values > 100, f"Too few unique PE values ({unique_values}): likely integer-quantized"


def test_pe_float32_with_float_coords():
    """PE stays float32 when coords are already float."""
    pe = TwoDSinCosPositionalEncoding(embed_dim=32)
    coords = torch.randint(0, 1000, (3, 50, 2)).float()
    out = pe(coords)
    assert out.dtype == torch.float32


def test_pe_output_shape():
    embed_dim = 64
    pe = TwoDSinCosPositionalEncoding(embed_dim=embed_dim)
    coords = torch.randint(0, 1000, (4, 100, 2), dtype=torch.int64)
    out = pe(coords)
    assert out.shape == (4, 100, embed_dim)


def test_pe_finite():
    pe = TwoDSinCosPositionalEncoding(embed_dim=32)
    coords = torch.randint(0, 10000, (2, 50, 2), dtype=torch.int64)
    out = pe(coords)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("embed_dim", [32, 64, 128, 512])
def test_pe_embed_dim_divisible_by_4(embed_dim: int):
    pe = TwoDSinCosPositionalEncoding(embed_dim=embed_dim)
    coords = torch.randint(0, 1000, (1, 10, 2), dtype=torch.int64)
    out = pe(coords)
    assert out.shape[-1] == embed_dim


def test_pe_embed_dim_not_divisible_raises():
    with pytest.raises(ValueError, match="divisible by 4"):
        TwoDSinCosPositionalEncoding(embed_dim=33)
