"""Gated Linear Attention (GLA) backbone for ordered tile sequences.

Pure-PyTorch chunk-parallel implementation. Represents the linear-attention
family in the SOTA comparison table.

Design:
- Scalar per-head decay gates (sigmoid) — simpler than head-dim gates but
  captures the key GLA mechanism
- Chunk-parallel: O(C²·d) within each chunk + O(N/C · H · d²) across chunks
- No external dependency: works without flash-linear-attention

Reference:
    Yang et al. 2024 "Gated Linear Attention Transformers with
    Hardware-Efficient Training" (simplified scalar-gate variant).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hilbert_wsi.models.base import SequenceBackbone, masked_mean

_CHUNK = 64  # within-chunk sequence length for chunk-parallel computation


class _GLALayer(nn.Module):
    """Single GLA layer: gated linear recurrence with chunk-parallel forward."""

    def __init__(self, dim: int, n_heads: int = 8, chunk_size: int = _CHUNK) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        # Scalar gate per head: controls how much of the state carries over
        self.g_proj = nn.Linear(dim, n_heads, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        H, d, C = self.n_heads, self.head_dim, self.chunk_size

        q = self.q_proj(x).view(B, N, H, d)          # (B, N, H, d)
        k = self.k_proj(x).view(B, N, H, d) / (d ** 0.5)
        v = self.v_proj(x).view(B, N, H, d)
        g = torch.sigmoid(self.g_proj(x))             # (B, N, H) gate in [0, 1]

        # Pad to chunk boundary
        pad = (-N) % C
        if pad:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            g = F.pad(g, (0, 0, 0, pad), value=1.0)

        Np = q.shape[1]
        n_chunks = Np // C

        # Reshape to chunks: (B, n_chunks, C, H, d)
        q_c = q.view(B, n_chunks, C, H, d)
        k_c = k.view(B, n_chunks, C, H, d)
        v_c = v.view(B, n_chunks, C, H, d)
        g_c = g.view(B, n_chunks, C, H)

        # Chunk-parallel processing
        outputs = []
        # S: accumulated KV state (B, H, d, d)
        S = q.new_zeros(B, H, d, d)

        for c in range(n_chunks):
            qt = q_c[:, c]   # (B, C, H, d)
            kt = k_c[:, c]
            vt = v_c[:, c]
            gt = g_c[:, c]   # (B, C, H)

            # Decay gate through whole chunk (product of per-step gates)
            g_chunk = gt.prod(dim=1)  # (B, H)

            # Cross-chunk: carry S into every position of this chunk.
            # Approximate: each position t uses the same chunk-level gate.
            # (More precise: cumulative product up to t within chunk — acceptable approx.)
            cross = torch.einsum("bhde,bchd->bche", S, qt)  # (B, C, H, d)

            # Intra-chunk: causal linear attention (exact, O(C²·d))
            # kv_t = k_t ⊗ v_t cumulative sum → output_t = cumkv @ q_t
            kv = torch.einsum("bchd,bche->bchde", kt, vt)   # (B, C, H, d, d)
            kv_cs = torch.cumsum(kv, dim=1)                 # (B, C, H, d, d)
            intra = torch.einsum("bchde,bchd->bche", kv_cs, qt)  # (B, C, H, d)

            out_c = cross + intra                           # (B, C, H, d)
            outputs.append(out_c)

            # Update state for next chunk
            S = g_chunk.view(B, H, 1, 1) * S + kv[:, -1]

        out = torch.cat(outputs, dim=1)[:, :N]             # (B, N, H, d)
        out = out.reshape(B, N, D)
        return self.norm(self.o_proj(out))


class _GLABlock(nn.Module):
    """GLA layer + pre-norm residual + SwiGLU FFN."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _GLALayer(dim, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        x = x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h)))
        return x


class GLABackbone(SequenceBackbone):
    """Stack of GLA blocks over ordered tile embeddings.

    Linear complexity O(N) in sequence length (chunk-parallel).
    Represents the gated-linear-attention family in the SOTA comparison:
    Yang et al. 2024 GLA / Sun et al. 2023 RetNet / Qin et al. 2023 HGRN.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        depth: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 2.67,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "cls", "last"):
            raise ValueError(f"Unknown pooling '{pooling}'")
        self.embedding_dim = embedding_dim
        self.pooling = pooling

        self.proj_in = nn.Linear(input_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _GLABlock(embedding_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(embedding_dim)

        if pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)

        nn.init.trunc_normal_(self.proj_in.weight, std=0.02)
        nn.init.zeros_(self.proj_in.bias)

    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.drop(self.proj_in(seq))

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
        if self.pooling == "cls":
            return x[:, 0]
        if mask is None:
            return x[:, -1]
        lengths = mask.long().sum(dim=1) - 1
        return x[torch.arange(x.shape[0], device=x.device), lengths]
