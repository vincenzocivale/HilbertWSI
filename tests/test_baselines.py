"""Smoke tests for baseline MIL wrappers (§4.2).

Each baseline must:
- Construct without error given (input_dim, embedding_dim)
- Accept (B, 1, N, D) 4D input from Patho-Bench collate_fn
- Return (B, embedding_dim) finite output
- Expose .embedding_dim attribute

ABMIL is skipped here (requires trident package).
"""

from __future__ import annotations

import pytest
import torch


INPUT_DIM = 64
EMBED_DIM = 32
B, N = 2, 128

HAS_MAMBA = pytest.importorskip.__module__ and True  # evaluated lazily per test

# ---------------------------------------------------------------------------
# Helper: fake Patho-Bench batch (B, 1, N, D) and (B, 1, N, 2) coords
# ---------------------------------------------------------------------------

def _batch(input_dim: int = INPUT_DIM) -> dict:
    return {
        "features": torch.randn(B, 1, N, input_dim),
        "coords": torch.randint(0, 1000, (B, 1, N, 2)).long(),
    }


def _mamba_available() -> bool:
    import importlib
    return importlib.util.find_spec("mamba_ssm") is not None


# ---------------------------------------------------------------------------
# TransMIL
# ---------------------------------------------------------------------------

def test_transmil_forward_shape():
    from adapters.patho_bench import _build_transmil_baseline
    enc = _build_transmil_baseline(
        input_dim=INPUT_DIM,
        embedding_dim=EMBED_DIM,
    )
    enc.eval()
    out = enc.forward(_batch(), device="cpu")
    assert out.shape == (B, EMBED_DIM), f"Expected ({B}, {EMBED_DIM}), got {out.shape}"
    assert torch.isfinite(out).all()


def test_transmil_embedding_dim_attr():
    from adapters.patho_bench import _build_transmil_baseline
    enc = _build_transmil_baseline(input_dim=INPUT_DIM, embedding_dim=EMBED_DIM)
    assert enc.embedding_dim == EMBED_DIM


# ---------------------------------------------------------------------------
# MambaMIL (requires mamba_ssm)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _mamba_available(), reason="mamba_ssm not installed")
def test_mambamil_forward_shape():
    from adapters.patho_bench import _build_mambamil_baseline
    enc = _build_mambamil_baseline(input_dim=INPUT_DIM, embedding_dim=EMBED_DIM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(device)
    enc.eval()
    b = {k: v.to(device) for k, v in _batch(INPUT_DIM).items()}
    out = enc.forward(b, device=device)
    assert out.shape == (B, EMBED_DIM)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# CLAM-SB (uses d_model not embedding_dim)
# ---------------------------------------------------------------------------

def test_clam_sb_forward_shape():
    from adapters.patho_bench import _build_clam_sb_baseline
    enc = _build_clam_sb_baseline(input_dim=INPUT_DIM, d_model=EMBED_DIM)
    enc.eval()
    out = enc.forward(_batch(), device="cpu")
    assert out.shape == (B, EMBED_DIM)
    assert torch.isfinite(out).all()


def test_clam_sb_embedding_dim_attr():
    from adapters.patho_bench import _build_clam_sb_baseline
    enc = _build_clam_sb_baseline(input_dim=INPUT_DIM, d_model=EMBED_DIM)
    assert enc.embedding_dim == EMBED_DIM


# ---------------------------------------------------------------------------
# CLAM-MB (uses d_model not embedding_dim)
# ---------------------------------------------------------------------------

def test_clam_mb_forward_shape():
    from adapters.patho_bench import _build_clam_mb_baseline
    enc = _build_clam_mb_baseline(input_dim=INPUT_DIM, d_model=EMBED_DIM)
    enc.eval()
    out = enc.forward(_batch(), device="cpu")
    assert out.shape == (B, EMBED_DIM)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# DSMIL (uses d_model not embedding_dim)
# ---------------------------------------------------------------------------

def test_dsmil_forward_shape():
    from adapters.patho_bench import _build_dsmil_baseline
    enc = _build_dsmil_baseline(input_dim=INPUT_DIM, d_model=EMBED_DIM)
    enc.eval()
    out = enc.forward(_batch(), device="cpu")
    assert out.shape == (B, EMBED_DIM)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 2DMamba (requires mamba_ssm + vendor)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _mamba_available(), reason="mamba_ssm not installed")
def test_twodmamba_forward_shape():
    from adapters.patho_bench import _build_twodmamba_baseline
    enc = _build_twodmamba_baseline(
        input_dim=INPUT_DIM,
        embedding_dim=EMBED_DIM,
        mamba_2d_patch_size=16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(device)
    enc.eval()
    b = {k: v.to(device) for k, v in _batch(INPUT_DIM).items()}
    out = enc.forward(b, device=device)
    assert out.shape == (B, EMBED_DIM)
    assert torch.isfinite(out).all()
