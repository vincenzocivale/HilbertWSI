"""Space-filling curve & baseline orderings for WSI tile sequences."""

from __future__ import annotations

from hilbert_wsi.ordering.base import OrderingScheme
from hilbert_wsi.ordering.hilbert import HilbertOrdering
from hilbert_wsi.ordering.moore import MooreOrdering
from hilbert_wsi.ordering.peano import PeanoOrdering
from hilbert_wsi.ordering.random_perm import RandomOrdering
from hilbert_wsi.ordering.similarity import SimilarityOrdering
from hilbert_wsi.ordering.snake import SnakeOrdering
from hilbert_wsi.ordering.zorder import ZOrderOrdering

_REGISTRY: dict[str, type[OrderingScheme]] = {
    "hilbert": HilbertOrdering,
    "zorder": ZOrderOrdering,
    "peano": PeanoOrdering,
    "moore": MooreOrdering,
    "snake": SnakeOrdering,
    "random": RandomOrdering,
    "similarity": SimilarityOrdering,
}


def get_ordering(name: str, **kwargs) -> OrderingScheme:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown ordering '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_orderings() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "OrderingScheme",
    "get_ordering",
    "available_orderings",
    "HilbertOrdering",
    "ZOrderOrdering",
    "PeanoOrdering",
    "MooreOrdering",
    "SnakeOrdering",
    "RandomOrdering",
    "SimilarityOrdering",
]
