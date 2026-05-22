"""MambaMIL baseline for the killer experiment.

Reimplementation of the MambaMIL architecture (Yang et al., MICCAI 2024) that
is faithful to the original block layout (LayerNorm + Mamba + residual, gated
attention pooling) but uses ``mamba_ssm.Mamba2`` for kernel parity with the
``BiMambaBlock`` already used by :mod:`hilbert_wsi.models.mamba`.

The encoder returns a *slide embedding* rather than classifier logits so it
plugs into the Patho-Bench / Trident slide-encoder contract via the wrapper
in :mod:`adapters.patho_bench`.

The pristine upstream code is preserved in
``hilbert_wsi/models/vendor/MambaMIL.py`` for provenance.

Role in the killer experiment
-----------------------------
``MambaMIL`` is the "Mamba-without-ordering" control: same selective-scan
kernel as ``HilbertWSI`` Mamba, but tiles are consumed in their original
(unordered) sequence. Comparing it against ``hilbertwsi_hilbert_mamba``
isolates the contribution of the Hilbert ordering for SSMs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hilbert_wsi.models.base import SequenceBackbone


def _largest_divisor(n: int, target: int) -> int:
    """Largest divisor of ``n`` that is <= ``target`` (matches mamba.py helper)."""
    for d in range(target, 0, -1):
        if n % d == 0:
            return d
    return 1


def _import_mamba2():
    try:
        from mamba_ssm import Mamba2  # type: ignore
        return Mamba2
    except ImportError as e:
        raise ImportError(
            "mamba-ssm is required for MambaMIL. "
            "Install with: pip install hilbert-wsi[mamba]"
        ) from e


class _MambaBlock(nn.Module):
    """LayerNorm + Mamba2 + residual, matching the upstream MambaMIL block."""

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ) -> None:
        super().__init__()
        Mamba2 = _import_mamba2()
        d_ssm = dim * expand
        safe_headdim = headdim if d_ssm % headdim == 0 else _largest_divisor(d_ssm, headdim)
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=safe_headdim,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mamba(self.norm(x))


class _GatedAttentionPool(nn.Module):
    """Attention pooling head from the MambaMIL paper.

    ``A = softmax(Linear(tanh(Linear(h))))`` over the tile axis, then
    aggregate ``bmm(A, h)`` -> ``(B, 1, dim)``.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        # h: (B, N, D)
        a = self.proj(h)                          # (B, N, 1)
        a = a.transpose(1, 2)                     # (B, 1, N)
        a = F.softmax(a, dim=-1)
        pooled = torch.bmm(a, h).squeeze(1)       # (B, D)
        return pooled


class MambaMILBackbone(SequenceBackbone):
    """MambaMIL paper architecture with Mamba2 kernel.

    Args mirror ``MambaBackbone`` for fair comparison. The slide embedding is
    produced by gated attention pooling over the Mamba output sequence.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        depth: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        act: str = "gelu",
        attn_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        proj: list[nn.Module] = [nn.Linear(input_dim, embedding_dim)]
        if act.lower() == "relu":
            proj.append(nn.ReLU())
        elif act.lower() == "gelu":
            proj.append(nn.GELU())
        if dropout > 0:
            proj.append(nn.Dropout(dropout))
        self.proj_in = nn.Sequential(*proj)

        self.blocks = nn.ModuleList(
            [
                _MambaBlock(embedding_dim, d_state=d_state, d_conv=d_conv,
                            expand=expand, headdim=headdim)
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(embedding_dim)
        self.pool = _GatedAttentionPool(embedding_dim, hidden=attn_hidden)

    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        # seq: (B, N, input_dim). ``mask`` is accepted for API parity with
        # other backbones but currently unused (Patho-Bench runs batch_size=1
        # so there is no padding).
        del mask
        h = self.proj_in(seq)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_out(h)
        return self.pool(h)
