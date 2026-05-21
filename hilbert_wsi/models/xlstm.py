"""xLSTM (mLSTM) backbone for ordered tile sequences.

Wraps the official ``xlstm`` package (mLSTMBlock) into a SequenceBackbone
compatible with HilbertSequenceEncoder.

Reference: Beck et al. 2024 "xLSTM: Extended Long Short-Term Memory"
Package: xlstm==2.0.5  (pip install xlstm)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hilbert_wsi.models.base import SequenceBackbone, masked_mean


def _import_xlstm():
    try:
        from xlstm.blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
        from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
        return mLSTMBlock, mLSTMBlockConfig, mLSTMLayerConfig
    except ImportError as e:
        raise ImportError(
            "xlstm is required for the xLSTM backbone. "
            "Install with: pip install xlstm"
        ) from e


class xLSTMBackbone(SequenceBackbone):
    """Stack of mLSTMBlocks (xLSTM) over ordered tile embeddings.

    Uses the official xlstm package's mLSTMBlock, which provides:
    - Matrix-memory LSTM with exponential gating
    - Parallel and sequential inference modes
    - Comparable compute to Mamba at same hidden dim
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        context_length: int = 16384,
        proj_factor: float = 2.0,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in ("mean", "cls", "last"):
            raise ValueError(f"Unknown pooling '{pooling}'")
        mLSTMBlock, mLSTMBlockConfig, mLSTMLayerConfig = _import_xlstm()

        self.embedding_dim = embedding_dim
        self.pooling = pooling

        self.proj_in = nn.Linear(input_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            mLSTMBlock(
                mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        context_length=context_length,
                        proj_factor=proj_factor,
                        dropout=dropout,
                        _num_blocks=depth,
                    ),
                    _num_blocks=depth,
                    _block_idx=i,
                )
            )
            for i in range(depth)
        ])
        self.norm_out = nn.LayerNorm(embedding_dim)

        if pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)

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
