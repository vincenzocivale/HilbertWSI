"""2DMambaMIL baseline for the killer experiment (Phase 3).

Wraps the vendored 2DMambaMIL (Zhang et al., CVPR 2025, arXiv 2412.00678) so
it satisfies the Patho-Bench / Trident slide-encoder contract: returns a
slide embedding `(B, embedding_dim)` rather than classifier logits.

Role
----
2DMambaMIL is the **direct paradigm rival** to HilbertWSI: it argues a
2D-selective SSM applied to a dense tile grid removes the need to flatten
tiles into a 1D order at all. Comparing it head-to-head against
``hilbertwsi_hilbert_mamba`` / ``hilbertwsi_snake_transformer`` decides
whether 1D-ordering is a competitive paradigm or has already been bypassed.

Implementation notes
--------------------
* Pure-PyTorch parallel scan (``pscan=True``, ``use_cuda=False``); no CUDA
  kernel build needed. The CUDA scan in ``vendor/twodmamba/pscan_2d.py`` is
  available for future speed-ups but kept disabled here for portability.
* The upstream model takes an argparse-style ``args`` namespace; this wrapper
  hides that behind explicit ``__init__`` kwargs.
* The upstream classifier head is replaced with ``nn.Identity`` so the
  forward returns the pooled slide embedding (Patho-Bench attaches its own
  classifier downstream).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch import Tensor

from hilbert_wsi.models.baselines.vendor.twodmamba.MambaMIL_2D import MambaMIL_2D


def _build_args(
    *,
    input_dim: int,
    embedding_dim: int,
    depth: int,
    d_state: int,
    inner_layernorms: bool,
    dropout: float,
    pos_emb_type: str | None,
    pos_emb_dropout: float,
    mamba_2d_max_w: int,
    mamba_2d_max_h: int,
    mamba_2d_patch_size: int,
    mamba_2d_pad_token: str,
    n_classes: int,
    survival: bool,
    pscan: bool,
    cuda_pscan: bool,
    model_type: str,
    patch_encoder_batch_size: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        in_dim=input_dim,
        mambamil_dim=embedding_dim,
        mambamil_layer=depth,
        mambamil_state_dim=d_state,
        mambamil_inner_layernorms=inner_layernorms,
        drop_out=dropout,
        pos_emb_type=pos_emb_type,
        pos_emb_dropout=pos_emb_dropout,
        mamba_2d_max_w=mamba_2d_max_w,
        mamba_2d_max_h=mamba_2d_max_h,
        mamba_2d_patch_size=mamba_2d_patch_size,
        mamba_2d_pad_token=mamba_2d_pad_token,
        n_classes=n_classes,
        survival=survival,
        pscan=pscan,
        cuda_pscan=cuda_pscan,
        model_type=model_type,
        patch_encoder_batch_size=patch_encoder_batch_size,
    )


class TwoDMambaMILBackbone(nn.Module):
    """Slide-encoder wrapper around the vendored 2DMambaMIL.

    Returns ``(B, embedding_dim)`` slide embeddings. ``B`` is the batch axis
    over slides — Patho-Bench currently runs with ``B=1``, so the wrapper
    treats each slide independently and loops if a batch arrives.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 512,
        depth: int = 4,
        d_state: int = 64,
        inner_layernorms: bool = False,
        dropout: float = 0.1,
        pos_emb_type: str | None = "linear",
        pos_emb_dropout: float = 0.0,
        mamba_2d_max_w: int = 100000,
        mamba_2d_max_h: int = 100000,
        mamba_2d_patch_size: int = 256,
        mamba_2d_pad_token: str = "trainable",
        cuda_pscan: bool = False,
        patch_encoder_batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        args = _build_args(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            depth=depth,
            d_state=d_state,
            inner_layernorms=inner_layernorms,
            dropout=dropout,
            pos_emb_type=pos_emb_type,
            pos_emb_dropout=pos_emb_dropout,
            mamba_2d_max_w=mamba_2d_max_w,
            mamba_2d_max_h=mamba_2d_max_h,
            mamba_2d_patch_size=mamba_2d_patch_size,
            mamba_2d_pad_token=mamba_2d_pad_token,
            n_classes=embedding_dim,          # placeholder; classifier replaced below
            survival=False,
            pscan=True,
            cuda_pscan=cuda_pscan,
            model_type="2DMambaMIL",
            patch_encoder_batch_size=patch_encoder_batch_size,
        )
        self.inner = MambaMIL_2D(args)
        # Strip classifier so forward returns the pooled slide embedding.
        self.inner.classifier = nn.Identity()

    def forward(self, feats: Tensor, coords: Tensor) -> Tensor:
        """Single-slide forward.

        Args:
            feats:  ``(N, D)`` or ``(1, N, D)`` patch features.
            coords: ``(N, 2)`` patch top-left coordinates.

        Returns:
            ``(1, embedding_dim)`` slide embedding.
        """
        if feats.dim() == 3 and feats.shape[0] == 1:
            feats = feats.squeeze(0)
        if coords.dim() == 3 and coords.shape[0] == 1:
            coords = coords.squeeze(0)
        # Dynamic grid sizing: the upstream 2D scan allocates
        # ``(H, W, d_inner, d_state)`` activations per layer, so a globally
        # large ``mamba_2d_max_w/h`` blows GPU memory on small slides. Override
        # per-slide using the actual coord extent — the grid shrinks to the
        # tissue bounding box. Each ResidualBlock carries its own ``.config``
        # reference, so we patch all of them.
        max_w = int(coords[:, 0].max().item()) + 1
        max_h = int(coords[:, 1].max().item()) + 1
        for block in self.inner.layers.layers:
            block.config.mamba_2d_max_w = max_w
            block.config.mamba_2d_max_h = max_h
        # MambaMIL_2D returns (logits, Y_prob, Y_hat, results_dict, None);
        # classifier was swapped for Identity -> `logits` IS the embedding.
        embedding, *_ = self.inner(feats, coords.float())
        return embedding
