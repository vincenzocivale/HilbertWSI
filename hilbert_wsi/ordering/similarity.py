from __future__ import annotations

import torch
from torch import Tensor

from hilbert_wsi.ordering.base import OrderingScheme


class SimilarityOrdering(OrderingScheme):
    """Greedy nearest-neighbour ordering on feature similarity (MambaMIL-style).

    Requires features to be passed alongside coords. Cosine similarity unless
    ``metric='euclidean'``.
    """

    name = "similarity"

    def __init__(self, metric: str = "cosine"):
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unknown metric: {metric}")
        self.metric = metric

    def __call__(self, coords: Tensor, features: Tensor | None = None) -> Tensor:
        if features is None:
            raise ValueError("SimilarityOrdering requires `features` argument.")
        n = features.shape[0]
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=coords.device)
        if self.metric == "cosine":
            f = torch.nn.functional.normalize(features.float(), dim=-1)
            sim = f @ f.T
            dist = 1.0 - sim
        else:
            dist = torch.cdist(features.float(), features.float())
        dist.fill_diagonal_(float("inf"))
        visited = torch.zeros(n, dtype=torch.bool, device=features.device)
        order = torch.empty(n, dtype=torch.long, device=features.device)
        # Start from highest-norm point (deterministic, content-driven)
        start = int(torch.argmax(features.float().norm(dim=-1)).item())
        order[0] = start
        visited[start] = True
        for i in range(1, n):
            row = dist[order[i - 1]].clone()
            row[visited] = float("inf")
            nxt = int(torch.argmin(row).item())
            order[i] = nxt
            visited[nxt] = True
        return order
