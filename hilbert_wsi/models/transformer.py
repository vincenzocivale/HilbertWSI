"""SOTA Transformer backbone for ordered tile sequences.

Block design:
    - Pre-norm RMSNorm
    - Multi-head self-attention with RoPE applied to Q,K
    - Attention computed via ``torch.nn.functional.scaled_dot_product_attention``
      (PyTorch 2.x autoselects FlashAttention 2 / mem-efficient kernel)
    - SwiGLU FFN (LLaMA-style)

Pooling: ``cls`` token (default) or masked ``mean``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hilbert_wsi.models.base import SequenceBackbone, masked_mean


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def _build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    """Return cos, sin tables of shape (seq_len, head_dim)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)                # (seq_len, half)
    cos = freqs.cos().repeat_interleave(2, dim=-1)    # (seq_len, head_dim)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos.to(dtype), sin.to(dtype)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to ``x`` of shape (B, H, N, D_head). cos/sin: (N, D_head)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    cos = cos.unsqueeze(0).unsqueeze(0)               # (1,1,N,D_head)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotated * sin


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"embedding_dim {dim} must be divisible by n_heads {n_heads}.")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm_attn = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = dropout
        self.norm_ffn = RMSNorm(dim)
        hidden = int(round(dim * mlp_ratio))
        # Round to multiple of 8 for tensor-core friendliness
        hidden = (hidden + 7) // 8 * 8
        self.ffn = _SwiGLU(dim, hidden, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        sdpa_mask: Tensor | None,
    ) -> Tensor:
        B, N, D = x.shape
        h = self.norm_attn(x)
        qkv = self.qkv(h).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                     # each (B, N, H, D_head)
        q = q.transpose(1, 2)                           # (B, H, N, D_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.proj(out)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class TransformerBackbone(SequenceBackbone):
    """SOTA Transformer with RoPE + SDPA FlashAttention + SwiGLU."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        depth: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 2.67,
        rope_base: float = 10000.0,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"Unknown pooling '{pooling}'. Choose 'cls' or 'mean'.")
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.rope_base = rope_base
        self.head_dim = embedding_dim // n_heads

        self.proj_in = nn.Linear(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_Block(embedding_dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm_out = RMSNorm(embedding_dim)

        if pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)

        nn.init.trunc_normal_(self.proj_in.weight, std=0.02)
        nn.init.zeros_(self.proj_in.bias)

    def _rope(self, n: int, device, dtype):
        return _build_rope_cache(n, self.head_dim, self.rope_base, device, dtype)

    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        B, N, _ = seq.shape
        x = self.dropout(self.proj_in(seq))

        if self.pooling == "cls":
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)               # (B, N+1, D)
            if mask is not None:
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

        seq_len = x.shape[1]
        cos, sin = self._rope(seq_len, x.device, x.dtype)

        # SDPA attn_mask: additive, float. True positions = keep, False = mask out.
        sdpa_mask: Tensor | None = None
        if mask is not None:
            # (B, 1, 1, N): broadcast over heads + query positions
            sdpa_mask = mask[:, None, None, :].to(dtype=x.dtype)
            sdpa_mask = (1.0 - sdpa_mask) * torch.finfo(x.dtype).min

        for blk in self.blocks:
            x = blk(x, cos, sin, sdpa_mask)
        x = self.norm_out(x)

        if self.pooling == "cls":
            return x[:, 0]
        return masked_mean(x, mask)
