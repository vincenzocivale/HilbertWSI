"""Trident-compatible slide encoder that uses 2D positional encoding from tile coordinates.

Fair comparison partner for HilbertSequenceEncoder (1D ordering):

    HilbertSequenceEncoder  — reorders tiles via SFC → 1D RoPE over ordered sequence
    TileCoordEncoder        — keeps original tile order → 2D sin/cos PE from (x, y) coords

Both use the same backbone architecture (Mamba or Transformer), same depth, same
embedding_dim. The only difference is HOW positional information enters the model:

    1D path:  ordering → sequence position → RoPE in attention (Transformer)
                                           → scan order (Mamba)
    2D path:  tile (x, y) coords → sin/cos PE → added to projected features
              Transformer: pe_type="none" (no RoPE, position is in input)
              Mamba:        2D PE in input, but scan order is arbitrary (random)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from hilbert_wsi.models import get_backbone_class
from hilbert_wsi.pos_encoding import TwoDSinCosPositionalEncoding


class TileCoordEncoder(nn.Module):
    """Slide encoder with 2D sin/cos PE from tile coordinates.

    Implements the Trident slide-encoder contract (same as HilbertSequenceEncoder):

    Args of ``forward``:
        sample: dict with keys
            - ``features``: ``(B, N, D_in)`` patch embeddings.
            - ``coords``: ``(B, N, 2)`` patch top-left coordinates (level-0 px).
        device: target device.

    Returns:
        ``(B, embedding_dim)`` slide embeddings.

    Fairness guarantee:
        - Same backbone class and kwargs as the paired HilbertSequenceEncoder run.
        - Backbone is created with skip_proj=True; proj_in + dropout are owned by
          this class (same parameter count as backbone's proj_in).
        - 2D PE is parameter-free (sin/cos), adding zero extra parameters.
        - For Transformer: pe_type="none" disables RoPE so position comes only from
          the 2D PE injected into input features.
        - For Mamba: no explicit PE exists; 2D PE is in features and scan is over
          the dataset-default tile order (typically raster-scan from H5 extraction).
    """

    def __init__(
        self,
        input_dim: int,
        backbone: str = "transformer",
        embedding_dim: int = 512,
        backbone_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        bb_kw = dict(backbone_kwargs or {})

        # Extract dropout from backbone_kwargs if present (for proj_in consistency).
        dropout = bb_kw.pop("dropout", dropout)

        # For Transformer: disable RoPE since position comes from 2D PE in features.
        if backbone == "transformer":
            bb_kw["pe_type"] = "none"

        # Backbone handles its own blocks; proj_in is owned here.
        bb_kw["skip_proj"] = True

        # Backbone is created with input_dim=embedding_dim so its proj_in is
        # Linear(embedding_dim, embedding_dim). Since skip_proj=True, that layer
        # is never called — but keeping it small (0.26M vs 0.79M) minimises the
        # parameter count gap between 1D and 2D encoders (~0.26M difference, documented).
        BackboneClass = get_backbone_class(backbone)
        self.backbone = BackboneClass(
            input_dim=embedding_dim,
            embedding_dim=embedding_dim,
            **bb_kw,
        )
        self.proj_in = nn.Linear(input_dim, embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.pos_enc = TwoDSinCosPositionalEncoding(embedding_dim)
        self.embedding_dim = embedding_dim
        self.precision = torch.float32

        nn.init.trunc_normal_(self.proj_in.weight, std=0.02)
        nn.init.zeros_(self.proj_in.bias)

    def _encode(self, features: Tensor, coords: Tensor) -> Tensor:
        """Core encode: (B, N, D_in) + (B, N, 2) → (B, embedding_dim)."""
        h = self.dropout_layer(self.proj_in(features))   # (B, N, embedding_dim)
        pe = self.pos_enc(coords.to(features.device))    # (B, N, embedding_dim)
        h = h + pe
        return self.backbone(h)                           # backbone skips its own proj_in

    def forward(self, sample: dict[str, Any], device: torch.device | str = "cpu") -> Tensor:
        features = sample["features"].to(device)
        coords = sample["coords"].to(device)

        if features.dim() == 4:
            B0, B1, N, D = features.shape
            features = features.reshape(B0 * B1, N, D)
            coords = coords.reshape(B0 * B1, N, 2)

        if features.dim() != 3 or coords.dim() != 3:
            raise ValueError(
                f"Expected 3D features (B,N,D) and coords (B,N,2) after reshape; "
                f"got {tuple(features.shape)} and {tuple(coords.shape)}."
            )
        return self._encode(features, coords)
