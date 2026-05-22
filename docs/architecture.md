# Architecture

## Data flow

```
              ┌──────────────────────┐
   H5 file ──►│ PatchEmbeddingsDataset│── features (N,D) ──┐
              │      (Patho-Bench)   │── coords   (N,2) ──┤
              └──────────────────────┘                     │
                                                           ▼
                                               ┌──────────────────────────┐
                                               │ HilbertSequenceEncoder   │
                                               │  ordering(coords) → perm │
                                               │  gather(features, perm)  │
                                               │  backbone(seq, mask)     │
                                               └─────────────┬────────────┘
                                                             │
                                                             ▼
                                              (B, embedding_dim) slide vec
                                                             │
                                                             ▼
                                       ┌─────────────────────────────────┐
                                       │ Patho-Bench experiment          │
                                       │ (finetune / linprobe / coxnet)  │
                                       └─────────────────────────────────┘
```

## Module contracts

### `OrderingScheme` (`hilbert_wsi/ordering/base.py`)

```python
class OrderingScheme(ABC):
    name: str
    def __call__(self, coords: Tensor) -> Tensor:
        """coords: (N, 2) int level-0 pixel coords.
        Returns: (N,) long permutation index."""
```

Properties:
- **Deterministic** for given coords (random uses `blake2b(coords + base_seed)` for per-slide seeding)
- **Stateless**: no learned parameters, no side effects
- **Empty-safe**: zero patches → empty permutation

### `SequenceBackbone` (`hilbert_wsi/models/base.py`)

```python
class SequenceBackbone(nn.Module):
    embedding_dim: int
    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        """seq: (B, N, D_in).  mask: (B, N) bool, True=valid.
        Returns: (B, embedding_dim)."""
```

### `HilbertSequenceEncoder` (`hilbert_wsi/encoder.py`)

Trident-compatible slide encoder. Exposes `.embedding_dim` and `.precision` (required by Patho-Bench `Pooler`).

`adapters/patho_bench.register_hilbert_encoders()` patches `trident.slide_encoder_models.load.encoder_factory`. Model names matching `hilbertwsi_<ordering>_<backbone>` route to `HilbertSequenceEncoder`; all others fall through.

## Input format from Patho-Bench / UNI2-h H5 files

UNI2-h H5 files store:
- `features: (1, N, 1536)` — leading dim = 1 (slide batch from TRIDENT)
- `coords: (1, N, 2)` — same leading dim

After `PatchEmbeddingsDataset.__getitem__` + `collate_fn([sample])`, the encoder receives `features` of shape `(1, 1, N, D)`:
- Dim 0: patient batch = 1
- Dim 1: slide dim from H5 leading = 1
- Dim 2: N patches
- Dim 3: D feature dim (1536 for UNI2-h)

`HilbertSequenceEncoder.forward()` detects 4D input and reshapes to `(B, N, D)` by merging dims 0 and 1.

## Ordering schemes

| Name | Algorithm | Locality |
|---|---|---|
| `hilbert` | Hilbert curve via z-order quantisation | very high |
| `zorder` | Z-order / Morton code | high |
| `peano` | Peano S-curve (canonical, column-major) | high |
| `moore` | Moore curve | high |
| `snake` | Row-major snake scan | medium |
| `similarity` | Feature-similarity nearest-neighbour chain | feature-space |
| `random` | Per-slide deterministic permutation (blake2b seed) | none (control) |

`base.quantize(coords)` shifts coordinates to non-negative integers, divides by the inferred patch stride, and returns grid cells + bit-depth `k` such that `2**k >= grid_side`.

## Sequence backbones

| Name | Params | Key detail |
|---|---|---|
| `mamba` | ~22M (depth=8) | mamba-ssm 2.3.2 Mamba2 blocks + mean pool |
| `transformer` | ~13M (depth=12) | standard TransformerEncoder + CLS token |
| `xlstm` | — | wraps `xlstm` pip package (mLSTMBlock) |
| `gla` | — | chunk-parallel scalar-gate GLA, pure-PyTorch |

## Baseline models (no-ordering controls)

These slide encoders appear directly in the Patho-Bench adapter and do **not** use ordering. Their purpose is to isolate whether gains come from the architecture or from spatial ordering.

| Adapter name | Class | Architecture | Params |
|---|---|---|---|
| `abmil_baseline` | `ABMILWrapper` | ABMIL (attention pooling) | ~1.5M |
| `transmil_baseline` | `_SlideEncoderWrapper` | TransMIL (Nyströmformer) | ~13M |
| `mambamil_baseline` | `_SlideEncoderWrapper` | MambaMIL (Mamba2 + attention pool) | ~21M |
| `twodmamba_baseline` | `TwoDMambaWrapper` | 2DMamba (2D-selective SSM, CVPR 2025) | ~14M |

### 2DMamba baseline

`twodmamba_baseline` wraps the vendored 2DMambaMIL (Zhang et al., CVPR 2025, arXiv 2412.00678). It places tile features on a 2D coordinate grid and applies a 2D-selective SSM scan — the direct paradigm rival to HilbertWSI (argues that 2D-native SSM removes the need for 1D ordering).

Experimental result: AUC 0.527 ± 0.015 on CPTAC-UCEC, indistinguishable from Random+Transformer and significantly below Hilbert+Mamba (CI disjoint). Confirms 1D ordering is not bypassed by 2D-native SSMs.

The wrapper (`hilbert_wsi/models/twodmamba.py`) handles:
- `batch.coords` passthrough (2DMamba consumes raw coordinates)
- Per-slide dynamic grid sizing: overrides `mamba_2d_max_w/h` at forward time to actual coord extent, preventing OOM on 12GB GPU
- Classifier head replaced with `nn.Identity` to return the pooled slide embedding

Vendored source: `hilbert_wsi/models/vendor/twodmamba/` with upstream patches documented in `NOTICE.md`.

## Vendor pattern

Upstream models that require non-trivial adaptation are vendored under `hilbert_wsi/models/vendor/<name>/`. Each vendor directory contains:
- The upstream source files (patched imports, minor bug fixes)
- `NOTICE.md`: original license, citation, and description of every patch applied

Current vendored models:
- `vendor/twodmamba/` — 2DMambaMIL (Zhang et al., CVPR 2025). Pure-Python pscan; no CUDA kernel build needed for the baseline.
- `vendor/MambaMIL.py` — MambaMIL upstream gated-attention pool
- `vendor/TransMIL.py` — TransMIL upstream Nyströmformer

## Adding a new ordering

1. Create `hilbert_wsi/ordering/<name>.py`
2. Subclass `OrderingScheme`, implement `__call__`
3. Register in `hilbert_wsi/ordering/__init__.py::_REGISTRY`
4. Add a parametrised test in `tests/test_orderings.py` if locality is testable

## Adding a new backbone

1. Create `hilbert_wsi/models/<name>.py`
2. Subclass `SequenceBackbone`, decorate with `@register_backbone("name")`
3. Import in `hilbert_wsi/models/__init__.py`

## Adding a new baseline (no ordering)

1. Implement the model (or vendor it under `hilbert_wsi/models/vendor/`)
2. Write a wrapper in `adapters/patho_bench.py` that satisfies the Trident encoder contract:
   ```python
   def forward(self, batch, device) -> Tensor:  # returns (B, embedding_dim)
   ```
3. Register in `_build_encoder()` dispatch table in `adapters/patho_bench.py`
