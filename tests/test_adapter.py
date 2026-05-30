"""Tests for adapters/patho_bench.py dispatch logic.

Covers §4.1:
- _parse_name correctness for all backbone/ordering combos
- _build() / _build_2dpe() construct the right encoder type
- register_hilbert_encoders() does not break non-HilbertWSI model names
"""

from __future__ import annotations

import pytest
import torch

from adapters.patho_bench import _parse_name
from hilbert_wsi.encoder import HilbertSequenceEncoder
from hilbert_wsi.encoder_2d import TileCoordEncoder


# ---------------------------------------------------------------------------
# _parse_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("hilbertwsi_snake_transformer", ("snake", "transformer")),
    ("hilbertwsi_hilbert_mamba", ("hilbert", "mamba")),
    ("hilbertwsi_zorder_mamba", ("zorder", "mamba")),
    ("hilbertwsi_moore_transformer", ("moore", "transformer")),
    ("hilbertwsi_random_xlstm", ("random", "xlstm")),
    ("hilbertwsi_hilbert_gla", ("hilbert", "gla")),
    ("hilbertwsi_peano_transformer", ("peano", "transformer")),
    ("hilbertwsi_2dpe_mamba", ("2dpe", "mamba")),
    ("hilbertwsi_2dpe_transformer", ("2dpe", "transformer")),
    ("hilbertwsi_2dpe_xlstm", ("2dpe", "xlstm")),
    ("hilbertwsi_2dpe_gla", ("2dpe", "gla")),
])
def test_parse_name_valid(name, expected):
    assert _parse_name(name) == expected


@pytest.mark.parametrize("name", [
    "abmil_baseline",
    "transmil_baseline",
    "hilbertwsi_",
    "hilbertwsi_snake",
    "hilbertwsi_badordering_mamba",
    "hilbertwsi_snake_badbackbone",
    "",
    "hilbert_mamba",
])
def test_parse_name_invalid(name):
    assert _parse_name(name) is None


# ---------------------------------------------------------------------------
# _build() — 1D encoder construction
# ---------------------------------------------------------------------------

def _base_kwargs(input_dim: int = 32, embedding_dim: int = 16) -> dict:
    return {"input_dim": input_dim, "embedding_dim": embedding_dim}


def test_build_1d_returns_hilbert_sequence_encoder():
    from adapters.patho_bench import _build
    enc = _build("hilbertwsi_snake_transformer", **_base_kwargs())
    assert isinstance(enc, HilbertSequenceEncoder)
    assert enc.ordering_name == "snake"
    assert enc.backbone_name == "transformer"


def test_build_1d_mamba():
    from adapters.patho_bench import _build
    enc = _build("hilbertwsi_hilbert_mamba", input_dim=32, embedding_dim=16,
                 backbone_kwargs={"depth": 1, "d_state": 8})
    assert isinstance(enc, HilbertSequenceEncoder)


def test_build_1d_max_seq_len():
    from adapters.patho_bench import _build
    enc = _build("hilbertwsi_hilbert_transformer", input_dim=32, embedding_dim=16,
                 max_seq_len=512)
    assert enc.max_seq_len == 512


def test_build_1d_unknown_kwarg_raises():
    from adapters.patho_bench import _build
    with pytest.raises(ValueError, match="Unexpected kwargs"):
        _build("hilbertwsi_snake_transformer", input_dim=32, embedding_dim=16,
               totally_unknown_kwarg=99)


def test_build_1d_missing_input_dim_raises():
    from adapters.patho_bench import _build
    with pytest.raises(ValueError, match="input_dim"):
        _build("hilbertwsi_snake_transformer", embedding_dim=16)


# ---------------------------------------------------------------------------
# _build_2dpe() — 2D PE encoder construction
# ---------------------------------------------------------------------------

def test_build_2dpe_returns_wrapper():
    from adapters.patho_bench import _build_2dpe, _TileCoordEncoderWrapper
    wrapped = _build_2dpe("hilbertwsi_2dpe_transformer", **_base_kwargs())
    assert isinstance(wrapped, _TileCoordEncoderWrapper)
    assert isinstance(wrapped.inner, TileCoordEncoder)


def test_build_2dpe_mamba():
    from adapters.patho_bench import _build_2dpe, _TileCoordEncoderWrapper
    wrapped = _build_2dpe("hilbertwsi_2dpe_mamba", input_dim=32, embedding_dim=16,
                          backbone_kwargs={"depth": 1, "d_state": 8})
    assert isinstance(wrapped, _TileCoordEncoderWrapper)


def test_build_2dpe_unknown_kwarg_raises():
    from adapters.patho_bench import _build_2dpe
    with pytest.raises(ValueError, match="Unexpected kwargs"):
        _build_2dpe("hilbertwsi_2dpe_transformer", input_dim=32, embedding_dim=16,
                    unknown_key=True)


def test_build_2dpe_missing_input_dim_raises():
    from adapters.patho_bench import _build_2dpe
    with pytest.raises(ValueError, match="input_dim"):
        _build_2dpe("hilbertwsi_2dpe_transformer", embedding_dim=16)


# ---------------------------------------------------------------------------
# End-to-end forward through wrapper
# ---------------------------------------------------------------------------

def _make_sample(b: int = 1, n: int = 32, d: int = 32) -> dict:
    return {
        "features": torch.randn(b, n, d),
        "coords": torch.randint(0, 1000, (b, n, 2), dtype=torch.int64),
    }


def test_1d_encoder_forward_via_build():
    from adapters.patho_bench import _build
    enc = _build("hilbertwsi_snake_transformer", input_dim=32, embedding_dim=16,
                 backbone_kwargs={"depth": 1, "n_heads": 2})
    enc.eval()
    out = enc(_make_sample(), device="cpu")
    assert out.shape == (1, 16)
    assert torch.isfinite(out).all()


def test_2dpe_encoder_forward_via_build():
    from adapters.patho_bench import _build_2dpe
    wrapped = _build_2dpe("hilbertwsi_2dpe_transformer", input_dim=32, embedding_dim=16,
                           backbone_kwargs={"depth": 1, "n_heads": 2})
    wrapped.eval()
    out = wrapped(_make_sample(), device="cpu")
    assert out.shape == (1, 16)
    assert torch.isfinite(out).all()
