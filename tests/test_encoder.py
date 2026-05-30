"""Smoke test for :class:`HilbertSequenceEncoder` with a fake backbone.

The Mamba backbone needs CUDA + ``mamba-ssm``; we substitute a tiny PyTorch
backbone for the smoke test so the test is hardware-agnostic. The Mamba path
is covered by a separate test that's skipped unless ``mamba_ssm`` is importable.
"""

from __future__ import annotations

import importlib

import pytest
import torch
from torch import Tensor, nn

from hilbert_wsi.encoder import HilbertSequenceEncoder
from hilbert_wsi.encoder_2d import TileCoordEncoder
from hilbert_wsi.models import register_backbone
from hilbert_wsi.models.base import SequenceBackbone, masked_mean
from hilbert_wsi.pos_encoding import TwoDSinCosPositionalEncoding


@register_backbone("dummy_mean")
class _DummyMeanBackbone(SequenceBackbone):
    def __init__(self, input_dim: int, embedding_dim: int = 32, skip_proj: bool = False, **_: object):
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.skip_proj = skip_proj

    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        x = seq if self.skip_proj else self.proj(seq)
        return masked_mean(x, mask)


def _dummy_sample(batch: int = 2, n: int = 64, d: int = 32) -> dict:
    coords = (torch.randint(0, 16, (batch, n, 2)) * 256).long()
    return {
        "features": torch.randn(batch, n, d),
        "coords": coords,
        "attributes": {"patch_size_level0": 256},
    }


def test_encoder_forward_shape_and_finite():
    enc = HilbertSequenceEncoder(
        input_dim=32,
        ordering="hilbert",
        backbone="dummy_mean",
        embedding_dim=16,
    )
    out = enc(_dummy_sample(d=32), device="cpu")
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("ordering", ["hilbert", "zorder", "snake", "moore", "peano", "random"])
def test_encoder_runs_with_all_simple_orderings(ordering: str):
    enc = HilbertSequenceEncoder(
        input_dim=32, ordering=ordering, backbone="dummy_mean", embedding_dim=8,
    )
    out = enc(_dummy_sample(), device="cpu")
    assert out.shape == (2, 8)


def test_encoder_respects_mask():
    enc = HilbertSequenceEncoder(
        input_dim=32, ordering="hilbert", backbone="dummy_mean", embedding_dim=8,
    )
    sample = _dummy_sample()
    sample["mask"] = torch.ones(2, sample["features"].shape[1], dtype=torch.bool)
    sample["mask"][1, 40:] = False
    out = enc(sample, device="cpu")
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_2d_pe_shape_and_finite():
    pe = TwoDSinCosPositionalEncoding(embed_dim=32)
    coords = torch.randint(0, 1000, (3, 64, 2)).float()
    out = pe(coords)
    assert out.shape == (3, 64, 32)
    assert torch.isfinite(out).all()


def test_2d_pe_normalized_coords_same_as_unnormalized():
    """PE is normalized per-slide, so scaling coords by a constant shouldn't change output."""
    pe = TwoDSinCosPositionalEncoding(embed_dim=32)
    coords = torch.randint(1, 100, (2, 50, 2)).float()
    out1 = pe(coords)
    out2 = pe(coords * 256)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_tile_coord_encoder_forward():
    enc = TileCoordEncoder(
        input_dim=32,
        backbone="dummy_mean",
        embedding_dim=16,
    )
    sample = _dummy_sample(d=32)
    out = enc(sample, device="cpu")
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_tile_coord_encoder_patho_bench_4d_input():
    """TileCoordEncoder handles (B, S, N, D) input from Patho-Bench collate_fn."""
    enc = TileCoordEncoder(input_dim=32, backbone="dummy_mean", embedding_dim=16)
    sample = {
        "features": torch.randn(2, 1, 64, 32),
        "coords": torch.randint(0, 1000, (2, 1, 64, 2)).float(),
    }
    out = enc(sample, device="cpu")
    assert out.shape == (2, 16)


def test_1d_and_2d_encoders_param_overhead_bounded():
    """2D PE encoder wasted overhead = backbone.proj_in(emb→emb), bounded by embedding_dim^2 + embedding_dim."""
    embedding_dim = 512
    enc1d = HilbertSequenceEncoder(32, "hilbert", "dummy_mean", embedding_dim=embedding_dim)
    enc2d = TileCoordEncoder(32, "dummy_mean", embedding_dim=embedding_dim)

    def p(m): return sum(x.numel() for x in m.parameters())
    overhead = p(enc2d) - p(enc1d)
    # Wasted backbone.proj_in = Linear(embedding_dim, embedding_dim)
    expected_overhead = embedding_dim * embedding_dim + embedding_dim
    assert overhead == expected_overhead, f"overhead {overhead} != expected {expected_overhead}"


@pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None,
    reason="mamba_ssm not installed",
)
def test_encoder_with_mamba_backbone():
    enc = HilbertSequenceEncoder(
        input_dim=32,
        ordering="hilbert",
        backbone="mamba",
        embedding_dim=16,
        backbone_kwargs={"depth": 1, "d_state": 8},
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(device)
    out = enc(_dummy_sample(d=32), device=device)
    assert out.shape == (2, 16)
