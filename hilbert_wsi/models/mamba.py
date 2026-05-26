from __future__ import annotations

import torch
from torch import Tensor, nn

from hilbert_wsi.models.base import GatedAttentionPool, SequenceBackbone, masked_mean


def _largest_divisor(n: int, target: int) -> int:
    """Largest divisor of n that is <= target."""
    for d in range(target, 0, -1):
        if n % d == 0:
            return d
    return 1


def _import_mamba():
    try:
        from mamba_ssm import Mamba2  # type: ignore
        return Mamba2
    except ImportError as e:
        raise ImportError(
            "mamba-ssm is required for the Mamba backbone. "
            "Install with: pip install hilbert-wsi[mamba]"
        ) from e


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block: forward + reverse pass summed, then LN + MLP."""

    def __init__(self, dim: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 64):
        super().__init__()
        Mamba2 = _import_mamba()
        # d_ssm = dim * expand must be divisible by headdim
        d_ssm = dim * expand
        safe_headdim = headdim if d_ssm % headdim == 0 else _largest_divisor(d_ssm, headdim)
        self.fwd = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=safe_headdim)
        self.bwd = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=safe_headdim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        f = self.fwd(x)
        b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        x = x + self.norm(f + b)
        x = x + self.norm2(self.mlp(x))
        return x


class MambaBackbone(SequenceBackbone):
    """Stack of bidirectional Mamba blocks over the ordered tile sequence."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        depth: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        pooling: str = "mean",
        dropout: float = 0.0,
        skip_proj: bool = False,
    ):
        super().__init__()
        if pooling not in ("mean", "cls", "last", "attn"):
            raise ValueError(f"Unknown pooling '{pooling}'")
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.skip_proj = skip_proj
        self.proj_in = nn.Linear(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                BiMambaBlock(embedding_dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(embedding_dim)
        if pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)
        if pooling == "attn":
            self.attn_pool = GatedAttentionPool(embedding_dim)

    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        x = seq if self.skip_proj else self.dropout(self.proj_in(seq))
        if self.pooling == "cls":
            cls = self.cls.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            if mask is not None:
                cls_mask = torch.ones(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        if self.pooling == "mean":
            return masked_mean(x, mask)
        if self.pooling == "attn":
            return self.attn_pool(x, mask)
        if self.pooling == "cls":
            return x[:, 0]
        # "last": last valid position per sequence
        if mask is None:
            return x[:, -1]
        lengths = mask.long().sum(dim=1) - 1
        return x[torch.arange(x.shape[0], device=x.device), lengths]
