"""TransMIL baseline for the killer experiment.

Reimplementation of TransMIL (Shao et al., NeurIPS 2021) that uses the pip
``nystrom-attention`` package instead of the inlined Nystromformer in the
upstream code, fixes the hardcoded ``.cuda()`` device dispatch, and returns
a slide embedding instead of classifier logits so the network plugs into the
Patho-Bench / Trident slide-encoder contract via the wrapper in
:mod:`adapters.patho_bench`.

The pristine upstream copy is preserved in
``hilbert_wsi/models/vendor/TransMIL.py`` for provenance.

Role in the killer experiment
-----------------------------
``TransMIL`` is the "Transformer-without-ordering" control: same Nystromformer
attention as a generic Transformer MIL, no Hilbert / Snake reordering.
Comparing it against ``hilbertwsi_<order>_transformer`` isolates the
contribution of the space-filling-curve ordering for Transformers.
"""

from __future__ import annotations

from math import ceil, sqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _import_nystrom():
    try:
        from nystrom_attention import NystromAttention  # type: ignore
        return NystromAttention
    except ImportError as e:
        raise ImportError(
            "nystrom-attention is required for TransMIL. "
            "Install with: pip install nystrom-attention"
        ) from e


class _TransLayer(nn.Module):
    """LayerNorm + NystromAttention with residual (matches upstream)."""

    def __init__(self, dim: int = 512, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        NystromAttention = _import_nystrom()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // heads,
            heads=heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.attn(self.norm(x))


class _PPEG(nn.Module):
    """Pyramid Position Encoding Generator (depthwise conv on token grid)."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.proj7 = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj5 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj3 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        # x: (B, 1+H*W, C); first token is cls.
        cls_token, feat = x[:, :1], x[:, 1:]
        B, _, C = x.shape
        feat = feat.transpose(1, 2).view(B, C, h, w)
        feat = self.proj7(feat) + feat + self.proj5(feat) + self.proj3(feat)
        feat = feat.flatten(2).transpose(1, 2)
        return torch.cat([cls_token, feat], dim=1)


class TransMILBackbone(nn.Module):
    """TransMIL encoder returning a slide embedding (B, embedding_dim).

    No classifier head; that is provided by the Patho-Bench experiment factory.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        depth: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        act: str = "gelu",
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

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.normal_(self.cls_token, std=1e-6)

        self.layers = nn.ModuleList(
            [_TransLayer(dim=embedding_dim, heads=n_heads, dropout=dropout)
             for _ in range(depth)]
        )
        # PPEG is inserted between the first and second layer (matches paper).
        self.ppeg = _PPEG(dim=embedding_dim)
        self.norm_out = nn.LayerNorm(embedding_dim)

    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        # seq: (B, N, input_dim). ``mask`` accepted for API parity, unused
        # (Patho-Bench runs batch_size=1 → no padding).
        del mask
        h = self.proj_in(seq)                                  # (B, N, D)
        B, N, _ = h.shape

        # Pad sequence to a perfect square so PPEG can reshape it to (H, W).
        side = int(ceil(sqrt(N)))
        pad = side * side - N
        if pad > 0:
            h = torch.cat([h, h[:, :pad, :]], dim=1)            # repeat first tokens

        cls = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat([cls, h], dim=1)                          # (B, 1+H*W, D)

        if len(self.layers) >= 1:
            h = self.layers[0](h)
        h = self.ppeg(h, side, side)
        for layer in self.layers[1:]:
            h = layer(h)

        h = self.norm_out(h)
        return h[:, 0]                                          # (B, D) — cls token
