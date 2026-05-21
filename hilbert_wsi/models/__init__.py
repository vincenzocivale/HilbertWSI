"""Sequence backbones over ordered tile embeddings."""

from __future__ import annotations

from typing import Callable

from hilbert_wsi.models.base import SequenceBackbone

_REGISTRY: dict[str, Callable[..., SequenceBackbone]] = {}


def register_backbone(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def get_backbone(name: str, **kwargs) -> SequenceBackbone:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backbone '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_backbones() -> list[str]:
    return sorted(_REGISTRY)


# Register built-ins. Heavy imports are lazy inside each class.
from hilbert_wsi.models.mamba import MambaBackbone          # noqa: E402
from hilbert_wsi.models.transformer import TransformerBackbone  # noqa: E402
from hilbert_wsi.models.xlstm import xLSTMBackbone          # noqa: E402
from hilbert_wsi.models.gla import GLABackbone              # noqa: E402

_REGISTRY["mamba"] = MambaBackbone
_REGISTRY["transformer"] = TransformerBackbone
_REGISTRY["xlstm"] = xLSTMBackbone
_REGISTRY["gla"] = GLABackbone

__all__ = [
    "SequenceBackbone", "get_backbone", "register_backbone", "available_backbones",
    "MambaBackbone", "TransformerBackbone", "xLSTMBackbone", "GLABackbone",
]
