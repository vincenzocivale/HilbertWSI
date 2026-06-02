# Architecture

_Last updated: 2026-05-31 (post-fix `91bc137`)._

## Data flow

```
              ┌─────────────────────────┐
   H5 file ──►│ PatchEmbeddingsDataset  │── features (B,1,N,1536) ──┐
              │      (Patho-Bench)      │── coords   (B,1,N,2)     ─┤
              └─────────────────────────┘── mask?    (B,1,N)       ─┤
                                                                     │
                            ┌────────────────────────────────────────┴───┐
                            │                                            │
                ┌───────────▼──────────────┐         ┌───────────────────▼──────────────┐
                │ HilbertSequenceEncoder   │         │      TileCoordEncoder            │
                │   ordering(coords) → perm│         │   proj_in(features) + 2D PE(c)   │
                │   gather(features, perm) │         │   (RoPE disabled for Transformer)│
                │   backbone(seq, mask)    │         │   backbone(seq, mask)            │
                └───────────┬──────────────┘         └───────────────────┬──────────────┘
                            │                                            │
                            └──────────────────────┬─────────────────────┘
                                                   ▼
                                       (B, embedding_dim) slide vec
                                                   │
                                                   ▼
                                  ┌─────────────────────────────────┐
                                  │ Patho-Bench experiment          │
                                  │ (finetune / linprobe / coxnet)  │
                                  └─────────────────────────────────┘
```

Two encoders share the same backbone classes. Differ only in **how positional
information enters the model**:

| Encoder | Spatial signal | RoPE (Transformer) | Mamba scan order |
|---|---|---|---|
| `HilbertSequenceEncoder` (1D) | tile order = SFC | on sequence index | along SFC |
| `TileCoordEncoder` (2D PE) | sin/cos PE from `(x,y)` added to features | disabled (`pe_type="none"`) | along H5 raster order |

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
- **Deterministic** for given coords (random uses `blake2b(coords + base_seed)` for per-slide seeding).
- **Stateless**: no learned parameters, no side effects.
- **Empty-safe**: zero patches → empty permutation.

`base.quantize(coords)` shifts coordinates to non-negative integers, divides by the inferred patch stride, and returns grid cells + bit-depth `k` such that `2**k >= grid_side`.

### `SequenceBackbone` (`hilbert_wsi/models/base.py`)

```python
class SequenceBackbone(nn.Module):
    embedding_dim: int
    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        """seq: (B, N, D_in).  mask: (B, N) bool, True=valid.
        Returns: (B, embedding_dim)."""
```

All backbones accept `skip_proj: bool = False`. When `skip_proj=True`, the
backbone bypasses its own `proj_in` (`TileCoordEncoder` owns the projection
so the input dim must already be `embedding_dim`).

### `HilbertSequenceEncoder` (`hilbert_wsi/encoder.py`)

```python
HilbertSequenceEncoder(
    input_dim: int,                                  # 1536 for UNI2-h
    ordering: str = "hilbert",                       # any name in the ordering registry
    backbone: str = "mamba",                         # any name in the backbone registry
    embedding_dim: int = 512,
    ordering_kwargs: dict | None = None,
    backbone_kwargs: dict | None = None,
    max_seq_len: int | None = None,                  # FIFO truncation pre-permutation
)
```

Pipeline (`forward`):
1. Reshape Patho-Bench input `(B, 1, N, D)` → `(B, N, D)`.
2. If `max_seq_len` set and `N > max_seq_len`: truncate `features[:, :max_seq_len]` and `coords[:, :max_seq_len]` **in H5 arrival order** (so 1D and 2D encoders see the same tile subset — fairness contract for C1).
3. Compute per-batch permutation via `ordering(coords[b])`.
4. `torch.gather(features, perm)` → ordered sequence; mask gathered the same way.
5. `backbone(seq, mask)` → `(B, embedding_dim)`.

Trident-compatible. Exposes `.embedding_dim` and `.precision = torch.float32`.

### `TileCoordEncoder` (`hilbert_wsi/encoder_2d.py`)

```python
TileCoordEncoder(
    input_dim: int,
    backbone: str = "transformer",
    embedding_dim: int = 512,
    backbone_kwargs: dict | None = None,
    dropout: float = 0.1,
    max_seq_len: int | None = None,
)
```

Pipeline (`forward`):
1. Reshape + truncate (identical rule to `HilbertSequenceEncoder`).
2. `proj_in(features)` + `TwoDSinCosPositionalEncoding(coords)` (added).
3. Backbone is built with `skip_proj=True`; for Transformer `pe_type="none"` is forced (no RoPE) so the only positional signal is the 2D PE.

Parameter overhead vs. `HilbertSequenceEncoder`: +0.26M (the backbone's
unused `proj_in(embedding_dim → embedding_dim)`), documented and stable
across runs.

### `TwoDSinCosPositionalEncoding` (`hilbert_wsi/pos_encoding.py`)

```python
TwoDSinCosPositionalEncoding(embed_dim)
# forward(coords: (B, N, 2)) → (B, N, embed_dim), always float32
```

- Per-slide min/max normalization of `x`, `y` to `[0, 1]`.
- Standard sin/cos PE applied to each axis, concatenated:
  `[sin(f·x), cos(f·x), sin(f·y), cos(f·y)]`, `embed_dim` divisible by 4.
- **Always returns float32** regardless of `coords` dtype (post-`91bc137` fix
  — earlier code cast back to `coords.dtype`, silently zeroing PE when
  `coords` was int64).

### Patho-Bench adapter (`adapters/patho_bench.py`)

`register_hilbert_encoders()` monkey-patches `trident.slide_encoder_models.load.encoder_factory`
so the following names are routed to the relevant builder:

| Model name pattern | Builder | Encoder class |
|---|---|---|
| `hilbertwsi_<ord>_<bb>` | `_build` | `HilbertSequenceEncoder` |
| `hilbertwsi_2dpe_<bb>` | `_build_2dpe` | `TileCoordEncoder` |
| `abmil_baseline` | `_build_abmil_baseline` | `ABMILWrapper` (Trident ABMIL) |
| `transmil_baseline` | `_build_transmil_baseline` | `TransMILBackbone` |
| `mambamil_baseline` | `_build_mambamil_baseline` | `MambaMILBackbone` |
| `clam_sb_baseline` | `_build_clam_sb_baseline` | `CLAMSBBackbone` |
| `clam_mb_baseline` | `_build_clam_mb_baseline` | `CLAMMBBackbone` |
| `dsmil_baseline` | `_build_dsmil_baseline` | `DSMILBackbone` |
| `twodmamba_baseline` | `_build_twodmamba_baseline` | `TwoDMambaMILBackbone` |
| anything else | original `encoder_factory` | (Trident's own models) |

Unknown kwargs to `_build` / `_build_2dpe` now raise `ValueError`
(post-`91bc137`); baseline builders still silently merge — be careful when
adding kwargs.

## Input format from Patho-Bench / UNI2-h H5 files

UNI2-h H5 files store:
- `features: (1, N, 1536)` — leading dim = 1 (slide batch from TRIDENT).
- `coords: (1, N, 2)` — same leading dim.

After `PatchEmbeddingsDataset.__getitem__` + `collate_fn([sample])`, the encoder receives `features` of shape `(B, 1, N, D)`:
- Dim 0: patient batch.
- Dim 1: slide dim from H5 leading.
- Dim 2: N patches.
- Dim 3: D feature dim (1536 for UNI2-h).

Both encoders detect 4D input and reshape to `(B, N, D)` by merging dims 0 and 1.

## Ordering schemes

| Name | Algorithm | Locality | Notes |
|---|---|---|---|
| `hilbert` | Hilbert curve, iterative bit interleave | very high | exact on `2**k × 2**k` grid |
| `zorder` | Z-order / Morton code | high | exact on `2**k × 2**k` grid |
| `peano` | Peano S-curve (column-major) | high | exact only on `3**p × 3**p` grids; current `quantize → m=3^ceil(log_3(side))` may collide for non-3^p grids — Peano-only ablation row, not a fairness concern for the main C1 claim |
| `moore` | Moore curve | high | closed Hilbert variant |
| `snake` | Row-major snake / boustrophedon | medium | strong simple baseline |
| `similarity` | Greedy nearest-neighbour in feature space; start = argmax norm | feature-space | **content-dependent ordering** — start point depends on the slide's features, so this row is not a pure "spatial ordering" comparator |
| `random` | Per-slide deterministic permutation (`blake2b(coords + base_seed)`) | none | negative control |

## Sequence backbones

| Name | Class | Pool default | Params (depth=4) | Key detail |
|---|---|---|---|---|
| `mamba` | `MambaBackbone` | `mean` | ~22M | Bidirectional Mamba2 blocks (forward + reverse + LN + MLP) |
| `transformer` | `TransformerBackbone` | `mean` | ~13M | RoPE on Q/K + SDPA FlashAttention + SwiGLU + RMSNorm |
| `xlstm` | wraps `xlstm` pip package | mean | — | mLSTMBlock |
| `gla` | pure-PyTorch | mean | — | chunk-parallel scalar-gate GLA |

All backbones accept `pooling ∈ {"mean", "cls"}`; `mamba` also accepts `"last"` and `"attn"` (GatedAttentionPool).

## Baseline models (no-ordering controls)

These slide encoders appear directly in the Patho-Bench adapter and do **not** use ordering. They are the no-spatial-structure controls for the C1 comparison.

| Adapter name | Class | Pool | Params | Notes |
|---|---|---|---|---|
| `abmil_baseline` | Trident `ABMILSlideEncoder` (via `ABMILWrapper`) | gated attention | ~1.5M | Reference attention-pooling baseline |
| `transmil_baseline` | `TransMILBackbone` | CLS after Nyströmformer | ~13M (depth=12) | PPEG depthwise conv injects a 2D positional bias from the H5 raster order |
| `mambamil_baseline` | `MambaMILBackbone` | gated attention | ~21M (depth=12) | Uses `Mamba2`, not the original SRMamba |
| `clam_sb_baseline` | `CLAMSBBackbone` | gated attention | ~1.1M | Single-branch CLAM; SmoothTop-K instance loss omitted |
| `clam_mb_baseline` | `CLAMMBBackbone` | gated attention × `n_classes` | ~1.8M | `n_classes` hardcoded in `configs/baselines/clam_mb_base.yaml` (default 3) — override per task |
| `dsmil_baseline` | `DSMILBackbone` | critical-instance attention | ~1.1M | Instance-level stream loss omitted |
| `twodmamba_baseline` | `TwoDMambaMILBackbone` (via `TwoDMambaWrapper`) | wrapped | ~14M (depth=8) | 2D-selective SSM; per-slide dynamic grid sizing in wrapper |

`MambaMIL` and `TransMIL` are capacity-matched to HilbertWSI's Mamba/Transformer via `depth=12` (paper default is 2). Documented choice; results tables note "HilbertWSI Mamba (depth=4, BiMamba) vs MambaMIL (depth=12, uni-Mamba)".

## Pooling heads — asymmetry to watch

The C1 claim is "1D ordering helps **independently of the backbone**". To
isolate the ordering signal from the pooling-head signal, pool choices must
match across the compared models. Current state:

| Group | Default pool |
|---|---|
| HilbertWSI 1D (`hilbertwsi_<ord>_<bb>`) | **mean** (`*_base.yaml`) |
| HilbertWSI 2D PE (`hilbertwsi_2dpe_<bb>`) | **mean** (`*_2dpe_base.yaml`) |
| All MIL baselines | **attention / CLS** |

→ Comparisons _HilbertWSI 1D vs. HilbertWSI random_ and _HilbertWSI 1D vs.
HilbertWSI 2D PE_ are fair (same pool both sides). Comparisons _HilbertWSI
1D vs. MIL baselines_ confound ordering with pool-head choice. See
[`verification_protocol.md`](verification_protocol.md) for the
attention-pool rerun that disambiguates this.

`configs/backbones/mamba_attnpool_base.yaml` exists for the Mamba arm.
Equivalent `_attnpool` configs for Transformer / xLSTM / GLA are missing
and would need to be created to extend the disambiguation across backbones.

## Vendor pattern

Upstream models that require non-trivial adaptation are vendored under `hilbert_wsi/models/vendor/<name>/`. Each vendor directory contains:
- The upstream source files (patched imports, minor bug fixes).
- `NOTICE.md`: original license, citation, and description of every patch applied.

Current vendored models:
- `vendor/twodmamba/` — 2DMambaMIL (Zhang et al., CVPR 2025). Pure-Python pscan; no CUDA kernel build needed for the baseline.
- `vendor/MambaMIL.py` — MambaMIL upstream gated-attention pool.
- `vendor/TransMIL.py` — TransMIL upstream Nyströmformer.

## Extending the codebase

### Adding a new ordering

1. Create `hilbert_wsi/ordering/<name>.py`.
2. Subclass `OrderingScheme`, implement `__call__`.
3. Register in `hilbert_wsi/ordering/__init__.py::_REGISTRY`.
4. Add a parametrised test in `tests/test_orderings.py` covering at least one non-`2**k`/`3**p` grid if the scheme requires recursion.

### Adding a new backbone

1. Create `hilbert_wsi/models/<name>.py`.
2. Subclass `SequenceBackbone`, accept `skip_proj: bool = False` for `TileCoordEncoder` compatibility.
3. Decorate with `@register_backbone("name")`.
4. Import in `hilbert_wsi/models/__init__.py`.
5. Create both `configs/backbones/<name>_base.yaml` and `<name>_2dpe_base.yaml` so the 1D-vs-2D comparison stays arch-agnostic.

### Adding a new baseline (no ordering)

1. Implement the model (or vendor it under `hilbert_wsi/models/vendor/`).
2. Write a builder in `adapters/patho_bench.py` that returns an object satisfying the Trident encoder contract:
   ```python
   def forward(self, batch, device) -> Tensor:  # returns (B, embedding_dim)
   ```
3. Register in the dispatch table inside `register_hilbert_encoders()`.
4. Add a smoke test in `tests/test_baselines.py`.
