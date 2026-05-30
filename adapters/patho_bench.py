"""Register HilbertSequenceEncoder with Patho-Bench / Trident.

Patho-Bench builds slide encoders via ``trident.slide_encoder_models.load.encoder_factory``
(see ``Patho-Bench/patho_bench/Pooler.py:30`` and ``ExperimentFactory.py:321``). We wrap
that factory so model names ``hilbert_<ordering>_<backbone>`` resolve to a
:class:`hilbert_wsi.HilbertSequenceEncoder`.

Call :func:`register_hilbert_encoders` once at program start (e.g. inside
``scripts/run_benchmark.py``) before invoking Patho-Bench.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from hilbert_wsi.encoder import HilbertSequenceEncoder
from hilbert_wsi.encoder_2d import TileCoordEncoder
from hilbert_wsi.models import available_backbones
from hilbert_wsi.ordering import available_orderings

PREFIX = "hilbertwsi"


def inject_val_split(
    split_csv: str | Path,
    label_col: str,
    val_frac: float = 0.2,
    seed: int = 0,
    sample_col: str | None = None,
) -> bool:
    """Add a stratified 'val' subset to every ``fold_*`` column of a Patho-Bench split.

    Re-marks ``val_frac`` of rows currently labelled 'train' as 'val', stratified
    by ``label_col``. When ``sample_col`` is given (e.g. 'case_id'), all rows
    belonging to the same sample are marked consistently so multi-slide patients
    don't get inconsistent fold labels. Writes the CSV in place. Idempotent:
    returns False without modifying the file when any fold already contains 'val'.

    Used by ``scripts/run_benchmark.py`` for finetune runs so that
    ``save_which_checkpoints='best-val-loss'`` has an actual validation set.
    """
    import csv
    from collections import defaultdict
    import random

    path = Path(split_csv)
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    fold_cols = [c for c in fieldnames if c.startswith("fold_")]
    if not fold_cols:
        raise ValueError(f"No fold_* columns found in {path}")
    if label_col not in fieldnames:
        raise ValueError(
            f"Label column '{label_col}' not found in {path} (columns: {fieldnames})"
        )

    if any(r[fc] == "val" for r in rows for fc in fold_cols):
        return False  # already injected

    rng = random.Random(seed)
    for fc in fold_cols:
        if sample_col and sample_col in fieldnames:
            # Group by patient/sample — mark all slides of a patient together.
            # Build (sample_id -> [row_indices]) for train rows only.
            sample_label: dict[str, str] = {}
            sample_idxs: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(rows):
                if r[fc] == "train":
                    sid = r[sample_col]
                    sample_idxs[sid].append(i)
                    sample_label[sid] = r[label_col]
            # Stratify at sample level.
            samples_by_label: dict[str, list[str]] = defaultdict(list)
            for sid, lbl in sample_label.items():
                samples_by_label[lbl].append(sid)
            val_samples: set[str] = set()
            for lbl, sids in samples_by_label.items():
                rng.shuffle(sids)
                n_val = max(1, int(round(len(sids) * val_frac)))
                val_samples.update(sids[:n_val])
            for i, r in enumerate(rows):
                if r[fc] == "train" and r[sample_col] in val_samples:
                    rows[i][fc] = "val"
        else:
            # Fall back to row-level stratification (single-slide-per-patient case).
            train_idx_by_label: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(rows):
                if r[fc] == "train":
                    train_idx_by_label[r[label_col]].append(i)
            for label, idxs in train_idx_by_label.items():
                rng.shuffle(idxs)
                n_val = max(1, int(round(len(idxs) * val_frac)))
                for i in idxs[:n_val]:
                    rows[i][fc] = "val"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delim)
        writer.writeheader()
        writer.writerows(rows)
    return True


class ABMILWrapper(nn.Module):
    """Wrap trident ABMIL to handle (B, 1, N, D) features from Patho-Bench Pooler.

    Trident ABMIL expects (B, N, D) but Patho-Bench collate_fn adds a leading
    slide-batch dimension, producing (B, 1, N, D). This wrapper squeezes that
    extra dim before delegating to the inner ABMIL encoder.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        from trident.slide_encoder_models.load import encoder_factory
        # Remove pretrained/freeze from kwargs — handled outside
        pretrained = kwargs.pop("pretrained", False)
        freeze = kwargs.pop("freeze", True)
        # Trident ABMILSlideEncoder doesn't accept input_dim — remove it
        kwargs.pop("input_dim", None)
        self._inner = encoder_factory("abmil", pretrained=pretrained, freeze=freeze, **kwargs)
        self.embedding_dim: int = self._inner.embedding_dim
        self.precision: torch.dtype = self._inner.precision

    def forward(self, batch: dict[str, Any], device: str = "cuda") -> torch.Tensor:
        feats = batch["features"]
        # Squeeze extra leading dims: (B, 1, N, D) -> (B, N, D)
        if feats.dim() == 4:
            B, S, N, D = feats.shape
            feats = feats.reshape(B * S, N, D)
        return self._inner.forward({**batch, "features": feats}, device)


def _build_abmil_baseline(freeze: bool = True, **kwargs: Any) -> ABMILWrapper:
    kwargs.pop("pretrained", None)  # ABMIL has no pretrained weights
    return ABMILWrapper(pretrained=False, freeze=freeze, **kwargs)


class _SlideEncoderWrapper(nn.Module):
    """Adapt a sequence backbone (forward(seq) -> (B, D)) to the Patho-Bench
    slide-encoder contract: ``forward(batch, device) -> (B, embedding_dim)``.

    Handles the (B, 1, N, D) shape Patho-Bench's collate_fn produces and the
    device move, but ignores ``coords`` (no spatial ordering) — that is the
    whole point of TransMIL / MambaMIL as "no-ordering" controls.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.embedding_dim: int = inner.embedding_dim
        self.precision: torch.dtype = torch.float32

    def forward(self, batch: dict[str, Any], device: str | torch.device = "cuda") -> torch.Tensor:
        feats = batch["features"].to(device)
        if feats.dim() == 4:                            # (B, S, N, D) -> (B*S, N, D)
            B, S, N, D = feats.shape
            feats = feats.reshape(B * S, N, D)
        return self.inner(feats)


def _build_transmil_baseline(**kwargs: Any) -> _SlideEncoderWrapper:
    """Build TransMIL baseline. Kwargs follow ``configs/baselines/transmil_base.yaml``."""
    from hilbert_wsi.models.baselines.transmil import TransMILBackbone
    # Patho-Bench passes ``pretrained`` / ``freeze`` for every encoder — drop them.
    kwargs.pop("pretrained", None)
    kwargs.pop("freeze", None)
    input_dim = kwargs.pop("input_dim", None)
    if input_dim is None:
        raise ValueError("transmil_baseline needs `input_dim` (patch feature dim).")
    # `backbone_kwargs` envelope (if someone wraps under HilbertWSI-style YAML).
    bb_kwargs = kwargs.pop("backbone_kwargs", None) or {}
    merged = {**kwargs, **bb_kwargs}
    inner = TransMILBackbone(input_dim=input_dim, **merged)
    return _SlideEncoderWrapper(inner)


def _build_mambamil_baseline(**kwargs: Any) -> _SlideEncoderWrapper:
    """Build MambaMIL baseline. Kwargs follow ``configs/baselines/mambamil_base.yaml``."""
    from hilbert_wsi.models.baselines.mambamil import MambaMILBackbone
    kwargs.pop("pretrained", None)
    kwargs.pop("freeze", None)
    input_dim = kwargs.pop("input_dim", None)
    if input_dim is None:
        raise ValueError("mambamil_baseline needs `input_dim` (patch feature dim).")
    bb_kwargs = kwargs.pop("backbone_kwargs", None) or {}
    merged = {**kwargs, **bb_kwargs}
    inner = MambaMILBackbone(input_dim=input_dim, **merged)
    return _SlideEncoderWrapper(inner)


class TwoDMambaWrapper(nn.Module):
    """Slide-encoder wrapper for 2DMambaMIL.

    Unlike ``_SlideEncoderWrapper``, this passes ``coords`` through to the
    inner network — the whole point of 2DMambaMIL is that the 2D-selective SSM
    consumes coords as positional context, not just features.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.embedding_dim: int = inner.embedding_dim
        self.precision: torch.dtype = torch.float32

    def forward(self, batch: dict[str, Any], device: str | torch.device = "cuda") -> torch.Tensor:
        feats = batch["features"].to(device)
        coords = batch["coords"].to(device)
        # Patho-Bench collate: (B, S, N, D) and (B, S, N, 2). Flatten to (B*S, N, D).
        if feats.dim() == 4:
            B, S, N, D = feats.shape
            feats = feats.reshape(B * S, N, D)
            coords = coords.reshape(B * S, N, 2)
        # 2DMambaMIL forward is single-slide; loop over batch if needed.
        outs = [self.inner(feats[i], coords[i]) for i in range(feats.shape[0])]
        return torch.cat(outs, dim=0)


def _build_twodmamba_baseline(**kwargs: Any) -> TwoDMambaWrapper:
    """Build 2DMambaMIL baseline. Kwargs follow ``configs/baselines/twodmamba_base.yaml``."""
    from hilbert_wsi.models.baselines.twodmamba import TwoDMambaMILBackbone
    kwargs.pop("pretrained", None)
    kwargs.pop("freeze", None)
    input_dim = kwargs.pop("input_dim", None)
    if input_dim is None:
        raise ValueError("twodmamba_baseline needs `input_dim` (patch feature dim).")
    bb_kwargs = kwargs.pop("backbone_kwargs", None) or {}
    merged = {**kwargs, **bb_kwargs}
    inner = TwoDMambaMILBackbone(input_dim=input_dim, **merged)
    return TwoDMambaWrapper(inner)


def _build_clam_sb_baseline(**kwargs: Any) -> _SlideEncoderWrapper:
    """Build CLAM-SB baseline. Kwargs follow ``configs/baselines/clam_sb_base.yaml``."""
    from hilbert_wsi.models.baselines.clam import CLAMSBBackbone
    kwargs.pop("pretrained", None)
    kwargs.pop("freeze", None)
    input_dim = kwargs.pop("input_dim", None)
    if input_dim is None:
        raise ValueError("clam_sb_baseline needs `input_dim` (patch feature dim).")
    bb_kwargs = kwargs.pop("backbone_kwargs", None) or {}
    merged = {**kwargs, **bb_kwargs}
    inner = CLAMSBBackbone(input_dim=input_dim, **merged)
    return _SlideEncoderWrapper(inner)


def _build_clam_mb_baseline(**kwargs: Any) -> _SlideEncoderWrapper:
    """Build CLAM-MB baseline. Kwargs follow ``configs/baselines/clam_mb_base.yaml``."""
    from hilbert_wsi.models.baselines.clam import CLAMMBBackbone
    kwargs.pop("pretrained", None)
    kwargs.pop("freeze", None)
    input_dim = kwargs.pop("input_dim", None)
    if input_dim is None:
        raise ValueError("clam_mb_baseline needs `input_dim` (patch feature dim).")
    bb_kwargs = kwargs.pop("backbone_kwargs", None) or {}
    merged = {**kwargs, **bb_kwargs}
    inner = CLAMMBBackbone(input_dim=input_dim, **merged)
    return _SlideEncoderWrapper(inner)


def _build_dsmil_baseline(**kwargs: Any) -> _SlideEncoderWrapper:
    """Build DSMIL baseline. Kwargs follow ``configs/baselines/dsmil_base.yaml``."""
    from hilbert_wsi.models.baselines.dsmil import DSMILBackbone
    kwargs.pop("pretrained", None)
    kwargs.pop("freeze", None)
    input_dim = kwargs.pop("input_dim", None)
    if input_dim is None:
        raise ValueError("dsmil_baseline needs `input_dim` (patch feature dim).")
    bb_kwargs = kwargs.pop("backbone_kwargs", None) or {}
    merged = {**kwargs, **bb_kwargs}
    inner = DSMILBackbone(input_dim=input_dim, **merged)
    return _SlideEncoderWrapper(inner)


def _parse_name(name: str) -> tuple[str, str] | None:
    """Return (ordering, backbone) if ``name`` matches ``hilbertwsi_<ord>_<bb>``.

    Returns ("2dpe", backbone) for ``hilbertwsi_2dpe_<bb>`` (2D PE encoder).
    """
    if not name.startswith(f"{PREFIX}_"):
        return None
    parts = name[len(PREFIX) + 1 :].split("_")
    if len(parts) < 2:
        return None
    backbone = parts[-1]
    ordering = "_".join(parts[:-1])
    if backbone not in available_backbones():
        return None
    if ordering == "2dpe":
        return "2dpe", backbone
    if ordering not in available_orderings():
        return None
    return ordering, backbone


def _build_2dpe(name: str, **kwargs: Any) -> "_TileCoordEncoderWrapper":
    """Build a TileCoordEncoder (2D PE variant) from a hilbertwsi_2dpe_<bb> name."""
    parts = name[len(PREFIX) + 1 :].split("_")  # ["2dpe", backbone]
    backbone = parts[-1]
    input_dim = kwargs.pop("input_dim", None)
    if input_dim is None:
        raise ValueError(
            "hilbertwsi_2dpe encoders need `input_dim`. "
            "Pass it via the Patho-Bench `model_kwargs` config."
        )
    embedding_dim = kwargs.pop("embedding_dim", 512)
    backbone_kwargs = kwargs.pop("backbone_kwargs", None)
    max_seq_len = kwargs.pop("max_seq_len", None)
    kwargs.pop("freeze", None)
    kwargs.pop("pretrained", None)
    kwargs.pop("ordering_kwargs", None)
    if kwargs:
        raise ValueError(f"Unexpected kwargs for '{name}': {sorted(kwargs)}")
    encoder = TileCoordEncoder(
        input_dim=input_dim,
        backbone=backbone,
        embedding_dim=embedding_dim,
        backbone_kwargs=backbone_kwargs,
        max_seq_len=max_seq_len,
    )
    return _TileCoordEncoderWrapper(encoder)


class _TileCoordEncoderWrapper(nn.Module):
    """Adapt TileCoordEncoder to the Patho-Bench slide-encoder contract.

    TileCoordEncoder.forward expects (sample, device) which is already the
    Trident contract — this wrapper is a thin identity that exposes the
    standard embedding_dim / precision attributes.
    """

    def __init__(self, inner: TileCoordEncoder) -> None:
        super().__init__()
        self.inner = inner
        self.embedding_dim: int = inner.embedding_dim
        self.precision: torch.dtype = torch.float32

    def forward(self, batch: dict, device: str | torch.device = "cuda") -> torch.Tensor:
        return self.inner(batch, device=device)


def _build(name: str, **kwargs: Any) -> HilbertSequenceEncoder:
    parsed = _parse_name(name)
    if parsed is None:
        raise KeyError(
            f"'{name}' is not a HilbertWSI model name. "
            f"Use 'hilbertwsi_<ordering>_<backbone>', e.g. 'hilbertwsi_hilbert_mamba'."
        )
    ordering, backbone = parsed
    input_dim = kwargs.pop("input_dim", None)
    if input_dim is None:
        raise ValueError(
            "HilbertWSI encoders need `input_dim` (patch feature dim). "
            "Pass it via the Patho-Bench `model_kwargs` config."
        )
    embedding_dim = kwargs.pop("embedding_dim", 512)
    ordering_kwargs = kwargs.pop("ordering_kwargs", None)
    backbone_kwargs = kwargs.pop("backbone_kwargs", None)
    max_seq_len = kwargs.pop("max_seq_len", None)
    # Patho-Bench passes `freeze` / `pretrained`; we ignore those (no pretraining yet).
    kwargs.pop("freeze", None)
    kwargs.pop("pretrained", None)
    if kwargs:
        raise ValueError(f"Unexpected kwargs for '{name}': {sorted(kwargs)}")
    return HilbertSequenceEncoder(
        input_dim=input_dim,
        ordering=ordering,
        backbone=backbone,
        embedding_dim=embedding_dim,
        ordering_kwargs=ordering_kwargs,
        backbone_kwargs=backbone_kwargs,
        max_seq_len=max_seq_len,
    )


_PATCHED = False


def register_hilbert_encoders() -> None:
    """Monkey-patch ``trident.slide_encoder_models.load.encoder_factory``.

    Idempotent. Falls back gracefully when trident is not installed (useful for
    unit tests of the rest of the package).
    """
    global _PATCHED
    if _PATCHED:
        return
    try:
        from trident.slide_encoder_models import load as _load
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "trident must be installed to register HilbertWSI with Patho-Bench. "
            "Install with: pip install hilbert-wsi[bench]"
        ) from e

    original = _load.encoder_factory

    def factory(model_name: str, *args: Any, **kwargs: Any):
        parsed = _parse_name(model_name)
        if parsed is not None:
            ordering, backbone = parsed
            if ordering == "2dpe":
                return _build_2dpe(model_name, **kwargs)
            return _build(model_name, **kwargs)
        if model_name == "abmil_baseline":
            return _build_abmil_baseline(**kwargs)
        if model_name == "transmil_baseline":
            return _build_transmil_baseline(**kwargs)
        if model_name == "mambamil_baseline":
            return _build_mambamil_baseline(**kwargs)
        if model_name == "twodmamba_baseline":
            return _build_twodmamba_baseline(**kwargs)
        if model_name == "clam_sb_baseline":
            return _build_clam_sb_baseline(**kwargs)
        if model_name == "clam_mb_baseline":
            return _build_clam_mb_baseline(**kwargs)
        if model_name == "dsmil_baseline":
            return _build_dsmil_baseline(**kwargs)
        return original(model_name, *args, **kwargs)

    _load.encoder_factory = factory

    # Patho-Bench imports the symbol directly; patch those module references too.
    try:
        from patho_bench import Pooler as _Pooler  # type: ignore
        from patho_bench import ExperimentFactory as _ExpFactory  # type: ignore

        _Pooler.encoder_factory = factory
        _ExpFactory.encoder_factory = factory
    except ImportError:
        pass

    _PATCHED = True
