"""Trident-compatible slide encoder that orders tiles via a space-filling curve."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from hilbert_wsi.models import get_backbone
from hilbert_wsi.ordering import get_ordering


class HilbertSequenceEncoder(nn.Module):
    """Compose `ordering -> sequence backbone` as a slide-level encoder.

    Implements the Trident slide-encoder contract used by Patho-Bench:

    Args of ``forward``:
        sample: dict with keys
            - ``features``: ``(B, N, D_in)`` patch embeddings.
            - ``coords``: ``(B, N, 2)`` patch top-left coordinates (level-0 px).
            - ``attributes``: optional dict; ``patch_size_level0`` is read if
              present to normalise coordinates.
        device: target device.

    Returns:
        ``(B, embedding_dim)`` slide embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        ordering: str = "hilbert",
        backbone: str = "mamba",
        embedding_dim: int = 512,
        ordering_kwargs: dict[str, Any] | None = None,
        backbone_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ordering_name = ordering
        self.backbone_name = backbone
        self.ordering = get_ordering(ordering, **(ordering_kwargs or {}))
        self.backbone = get_backbone(
            backbone,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            **(backbone_kwargs or {}),
        )
        self.embedding_dim = embedding_dim
        self.precision = torch.float32

    def _compute_perms(self, coords: Tensor, features: Tensor) -> list[Tensor]:
        perms = []
        for b in range(coords.shape[0]):
            if self.ordering_name == "similarity":
                p = self.ordering(coords[b], features=features[b])
            else:
                p = self.ordering(coords[b])
            perms.append(p)
        return perms

    def forward(self, sample: dict[str, Any], device: torch.device | str = "cpu") -> Tensor:
        features = sample["features"].to(device)
        coords = sample["coords"].to(device)
        # Patho-Bench / TRIDENT passes (patient_batch, slide_batch, N, D).
        # Flatten leading dims to (B, N, D) where B = patient_batch * slide_batch.
        if features.dim() == 4:
            B0, B1, N, D = features.shape
            features = features.reshape(B0 * B1, N, D)
            coords = coords.reshape(B0 * B1, N, 2)
        if features.dim() != 3 or coords.dim() != 3:
            raise ValueError(
                f"Expected 3D features (B,N,D) and coords (B,N,2) after reshape; "
                f"got {tuple(features.shape)} and {tuple(coords.shape)}."
            )
        perms = self._compute_perms(coords, features)
        idx = torch.stack([p.to(features.device) for p in perms], dim=0)
        seq = torch.gather(features, 1, idx.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        mask = sample.get("mask")
        if mask is not None:
            mask = mask.to(device)
            mask = torch.gather(mask, 1, idx)
        return self.backbone(seq, mask=mask)
