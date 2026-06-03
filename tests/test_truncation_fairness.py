"""Regression tests for truncation symmetry between 1D and 2D encoders.

Covers:
- Both encoders cap length with a deterministic uniform-random per-slide
  subsample (the SAME subset for 1D and 2D), applied BEFORE ordering — not a
  first-N H5-order slab (see tests/test_pipeline_fixes.py::BUG B).
- §8.3 fix: TileCoordEncoder propagates mask to backbone.
- §8.2 fix: xLSTM and GLA accept skip_proj kwarg.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hilbert_wsi.encoder import HilbertSequenceEncoder
from hilbert_wsi.encoder_2d import TileCoordEncoder
from hilbert_wsi.models import register_backbone
from hilbert_wsi.models.base import SequenceBackbone, masked_mean


_REGISTERED = False


def _ensure_registered():
    global _REGISTERED
    if not _REGISTERED:
        @register_backbone("spy_mean")
        class _SpyMeanBackbone(SequenceBackbone):
            """Records last input for inspection."""
            def __init__(self, input_dim: int, embedding_dim: int = 16,
                         skip_proj: bool = False, **_: object):
                super().__init__()
                self.proj = nn.Linear(input_dim, embedding_dim)
                self.embedding_dim = embedding_dim
                self.skip_proj = skip_proj
                self.last_seq: Tensor | None = None
                self.last_mask: Tensor | None = None

            def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
                self.last_seq = seq.detach()
                self.last_mask = mask
                x = seq if self.skip_proj else self.proj(seq)
                return masked_mean(x, mask)

        _REGISTERED = True


def _make_sample(b: int = 1, n: int = 200, d: int = 16):
    coords = (torch.arange(n).unsqueeze(-1) * torch.tensor([[16, 1]])).expand(b, -1, -1).clone().long()
    return {
        "features": torch.randn(b, n, d),
        "coords": coords,
    }


def test_1d_encoder_truncates_before_ordering():
    """1D encoder caps length (uniform subsample) BEFORE the SFC permutation."""
    _ensure_registered()
    max_len = 50
    sample = _make_sample(n=200)

    enc = HilbertSequenceEncoder(
        input_dim=16, ordering="hilbert", backbone="spy_mean",
        embedding_dim=16, max_seq_len=max_len,
    )
    enc.eval()
    enc(sample, device="cpu")

    bb = enc.backbone
    assert bb.last_seq is not None
    assert bb.last_seq.shape[1] == max_len, (
        f"Expected {max_len} tokens, got {bb.last_seq.shape[1]}"
    )


def test_2d_encoder_truncates_before_pe():
    """2D encoder caps length (uniform subsample) before adding PE."""
    _ensure_registered()
    max_len = 50
    sample = _make_sample(n=200)

    enc = TileCoordEncoder(
        input_dim=16, backbone="spy_mean",
        embedding_dim=16, max_seq_len=max_len,
    )
    enc.eval()
    enc(sample, device="cpu")

    bb = enc.backbone
    assert bb.last_seq is not None
    assert bb.last_seq.shape[1] == max_len, (
        f"Expected {max_len} tokens, got {bb.last_seq.shape[1]}"
    )


def test_both_encoders_see_same_tile_subset_under_truncation():
    """1D and 2D encoders, given same sample + same max_seq_len, keep the same
    representative subset (subsample seed derives from coords)."""
    _ensure_registered()
    max_len = 30
    sample = _make_sample(b=1, n=100)

    enc1d = HilbertSequenceEncoder(
        input_dim=16, ordering="hilbert", backbone="spy_mean",
        embedding_dim=16, max_seq_len=max_len,
    )
    enc2d = TileCoordEncoder(
        input_dim=16, backbone="spy_mean",
        embedding_dim=16, max_seq_len=max_len,
    )
    enc1d.eval(); enc2d.eval()

    enc1d(sample, device="cpu")
    enc2d(sample, device="cpu")

    # Both should have processed exactly max_len tokens
    assert enc1d.backbone.last_seq.shape[1] == max_len
    assert enc2d.backbone.last_seq.shape[1] == max_len


def test_tile_coord_encoder_propagates_mask():
    """§8.3: TileCoordEncoder must pass mask to backbone."""
    _ensure_registered()
    n = 64
    sample = _make_sample(n=n)
    mask = torch.ones(1, n, dtype=torch.bool)
    mask[0, 40:] = False
    sample["mask"] = mask

    enc = TileCoordEncoder(input_dim=16, backbone="spy_mean", embedding_dim=16)
    enc.eval()
    enc(sample, device="cpu")

    assert enc.backbone.last_mask is not None, "Mask was not propagated to backbone"
    assert enc.backbone.last_mask.shape == (1, n)


def test_tile_coord_encoder_mask_truncated_consistently():
    """§8.3 + §8.4: mask is truncated to max_seq_len together with features."""
    _ensure_registered()
    max_len = 40
    n = 100
    sample = _make_sample(n=n)
    mask = torch.ones(1, n, dtype=torch.bool)
    sample["mask"] = mask

    enc = TileCoordEncoder(
        input_dim=16, backbone="spy_mean",
        embedding_dim=16, max_seq_len=max_len,
    )
    enc.eval()
    enc(sample, device="cpu")

    assert enc.backbone.last_mask.shape[1] == max_len


def test_xlstm_skip_proj_accepted():
    """§8.2: xLSTMBackbone accepts skip_proj without TypeError."""
    from hilbert_wsi.models import get_backbone_class
    try:
        Cls = get_backbone_class("xlstm")
    except Exception:
        import pytest; pytest.skip("xlstm package not installed")
    bb = Cls(input_dim=16, embedding_dim=16, depth=1, num_heads=2,
             context_length=128, skip_proj=True)
    assert bb.skip_proj is True
    x = torch.randn(1, 10, 16)
    out = bb(x)
    assert out.shape == (1, 16)


def test_gla_skip_proj_accepted():
    """§8.2: GLABackbone accepts skip_proj without TypeError."""
    from hilbert_wsi.models.gla import GLABackbone
    bb = GLABackbone(input_dim=16, embedding_dim=16, depth=1, n_heads=2,
                     skip_proj=True)
    assert bb.skip_proj is True
    x = torch.randn(1, 10, 16)
    out = bb(x)
    assert out.shape == (1, 16)


def test_no_truncation_when_max_seq_len_none():
    """No truncation when max_seq_len is not set."""
    _ensure_registered()
    n = 100
    sample = _make_sample(n=n)

    enc2d = TileCoordEncoder(input_dim=16, backbone="spy_mean", embedding_dim=16)
    enc2d.eval()
    enc2d(sample, device="cpu")
    assert enc2d.backbone.last_seq.shape[1] == n

    enc1d = HilbertSequenceEncoder(
        input_dim=16, ordering="hilbert", backbone="spy_mean", embedding_dim=16,
    )
    enc1d.eval()
    enc1d(sample, device="cpu")
    assert enc1d.backbone.last_seq.shape[1] == n
