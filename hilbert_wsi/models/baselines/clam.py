"""CLAM-SB and CLAM-MB baselines.

Lu et al., "Data-Efficient and Weakly Supervised Computational Pathology on
Whole-Slide Images", Nature Biomedical Engineering 2021.

Implements the gated-attention slide encoder only; the classifier head and
the instance-level clustering auxiliary loss are handled externally (by
Patho-Bench ExperimentFactory). The slide embedding is the attention-pooled
bag representation fed into the downstream classifier.

Registered with Patho-Bench via adapters.patho_bench as:
  ``clam_sb_baseline``  — CLAM-SB (single attention branch)
  ``clam_mb_baseline``  — CLAM-MB (n_classes attention branches, mean-pooled)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _GatedAttention(nn.Module):
    """Gated attention (Eq. 1 in CLAM paper).

    A_k = softmax(w^T [tanh(V h_k) ⊙ sigmoid(U h_k)])
    """

    def __init__(self, d_model: int, attention_dim: int, n_branches: int) -> None:
        super().__init__()
        self.V = nn.Sequential(nn.Linear(d_model, attention_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(d_model, attention_dim), nn.Sigmoid())
        self.w = nn.Linear(attention_dim, n_branches)

    def forward(self, h: Tensor) -> Tensor:
        # h: (B, N, d_model) → (B, n_branches, N)
        a = self.w(self.V(h) * self.U(h))   # (B, N, n_branches)
        return F.softmax(a.transpose(1, 2), dim=-1)  # (B, n_branches, N)


class CLAMSBBackbone(nn.Module):
    """CLAM-SB slide encoder — single gated-attention branch.

    Returns (B, d_model) slide embedding. ~1.1M params at default settings.
    Compare against ~13-22M for Transformer/Mamba; document capacity gap.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.25,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.embedding_dim = d_model
        act_fn = nn.ReLU() if act.lower() == "relu" else nn.GELU()
        self.proj_in = nn.Sequential(
            nn.Linear(input_dim, d_model),
            act_fn,
            nn.Dropout(dropout),
        )
        self.attention = _GatedAttention(d_model, attention_dim, n_branches=1)

    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        del mask
        h = self.proj_in(seq)                         # (B, N, d_model)
        a = self.attention(h)                          # (B, 1, N)
        return torch.bmm(a, h).squeeze(1)             # (B, d_model)


class CLAMMBBackbone(nn.Module):
    """CLAM-MB slide encoder — n_classes gated-attention branches.

    Each branch produces a class-specific bag representation pooled by its own
    attention weights. Branches are averaged before returning, giving a single
    (B, d_model) embedding compatible with the Patho-Bench encoder contract.

    A full CLAM-MB training uses per-branch classifiers and per-branch instance-
    level clustering loss; those are not implemented here (Patho-Bench handles
    the classifier, clustering loss is omitted).

    n_classes must match the downstream task (set in YAML config).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        attention_dim: int = 256,
        n_classes: int = 3,
        dropout: float = 0.25,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.embedding_dim = d_model
        self.n_classes = n_classes
        act_fn = nn.ReLU() if act.lower() == "relu" else nn.GELU()
        self.proj_in = nn.Sequential(
            nn.Linear(input_dim, d_model),
            act_fn,
            nn.Dropout(dropout),
        )
        self.attention = _GatedAttention(d_model, attention_dim, n_branches=n_classes)
        self.branch_projs = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_classes)]
        )

    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        del mask
        h = self.proj_in(seq)                              # (B, N, d_model)
        a = self.attention(h)                              # (B, n_classes, N)
        branches = []
        for k in range(self.n_classes):
            z_k = torch.bmm(a[:, k : k + 1, :], h).squeeze(1)  # (B, d_model)
            branches.append(self.branch_projs[k](z_k))
        return torch.stack(branches, dim=1).mean(dim=1)    # (B, d_model)
