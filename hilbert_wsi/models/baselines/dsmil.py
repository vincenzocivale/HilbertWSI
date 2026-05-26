"""DSMIL (Dual-Stream MIL) baseline.

Li et al., "Dual-stream multiple instance learning network for whole slide
image classification with self-supervised contrastive learning", CVPR 2021.

Key idea: the critical (max-scoring) instance acts as the bag-level query;
attention weights are inner products between all instances and that query.
This replaces learned query vectors (ABMIL/CLAM) with a data-driven one.

Slide encoder only — no classifier head, no instance-level stream classifier
loss. The Patho-Bench ExperimentFactory provides the downstream classifier.

Registered with Patho-Bench via adapters.patho_bench as:
  ``dsmil_baseline``
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DSMILBackbone(nn.Module):
    """DSMIL slide encoder.

    Forward pass:
      1. Project instances: H = proj(X)   (B, N, d_model)
      2. Score instances:   s = score(H)   (B, N)
      3. Select critical:   f_max = H[argmax(s)]   (B, d_model)
      4. Bag attention:     A = softmax(H @ b_proj(f_max) / sqrt(d))   (B, N, 1)
      5. Aggregate:         z = (A * H).sum(1)   (B, d_model)

    Returns (B, d_model). ~1.1M params at default settings.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        dropout: float = 0.0,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.embedding_dim = d_model
        self._scale = d_model ** -0.5

        act_fn = nn.ReLU() if act.lower() == "relu" else nn.GELU()
        self.proj_in = nn.Sequential(nn.Linear(input_dim, d_model), act_fn)
        self.i_score = nn.Linear(d_model, 1)   # instance scorer
        self.b_proj = nn.Linear(d_model, d_model)  # bag-query projection
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        del mask
        h = self.proj_in(seq)                                      # (B, N, d_model)

        # Critical instance: max by instance score
        scores = self.i_score(h).squeeze(-1)                       # (B, N)
        idx = scores.argmax(dim=1)                                  # (B,)
        f_max = h[torch.arange(h.size(0), device=h.device), idx]  # (B, d_model)

        # Bag-level attention with critical instance as query
        q = self.b_proj(f_max).unsqueeze(2)                        # (B, d_model, 1)
        attn = F.softmax(torch.bmm(h, q) * self._scale, dim=1)    # (B, N, 1)

        z = (attn * h).sum(dim=1)                                  # (B, d_model)
        return self.dropout(z)
