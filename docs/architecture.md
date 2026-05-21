# Architecture

## Data flow

```
              ┌──────────────────────┐
   H5 file ──►│ PatchEmbeddingsDataset│── features (N,D) ──┐
              │      (Patho-Bench)    │── coords   (N,2) ──┤
              └──────────────────────┘                     │
                                                            ▼
                                                ┌──────────────────────────┐
                                                │ HilbertSequenceEncoder   │
                                                │  • ordering(coords) ─► perm
                                                │  • gather(features, perm)
                                                │  • backbone(seq, mask)
                                                └─────────────┬────────────┘
                                                              │
                                                              ▼
                                               (B, embedding_dim) slide vec
                                                              │
                                                              ▼
                                        ┌─────────────────────────────────┐
                                        │ Patho-Bench experiment          │
                                        │ (linprobe / finetune / cox / …) │
                                        └─────────────────────────────────┘
```

## Module contract

### `OrderingScheme` (`hilbert_wsi/ordering/base.py`)

```python
class OrderingScheme(ABC):
    name: str
    def __call__(self, coords: Tensor) -> Tensor:
        """coords: (N, 2) int level-0 pixel coords.
        Returns:  (N,) long permutation."""
```

Properties:
- **Deterministic** for given coords (except `random`, which seeds itself).
- **Stateless**: no learned parameters, no side effects.
- **Empty-safe**: zero patches → empty permutation.

### `SequenceBackbone` (`hilbert_wsi/models/base.py`)

```python
class SequenceBackbone(nn.Module):
    embedding_dim: int
    def forward(self, seq: Tensor, mask: Tensor | None = None) -> Tensor:
        """seq: (B, N, D_in).  mask: (B, N) bool, True=valid.
        Returns:  (B, embedding_dim)."""
```

### `HilbertSequenceEncoder` (`hilbert_wsi/encoder.py`)

Trident-compatible slide encoder. Patho-Bench's `Pooler` and
`ExperimentFactory.encoder_factory` call this through `adapters/patho_bench.py`.
Required attributes: `embedding_dim`, `precision`.

## Input format from Patho-Bench / UNI2-h H5 files

UNI2-h H5 files store:
- `features: (1, N, 1536)` — leading dim = 1 (slide batch from TRIDENT)
- `coords: (1, N, 2)` — same leading dim

After `PatchEmbeddingsDataset.__getitem__` + `collate_fn([sample])`, the encoder
receives `cleaned_sample['features']` of shape `(1, 1, N, D)`:
- Dim 0: patient batch = 1 (Pooler always processes 1 patient at a time)
- Dim 1: slide dim from H5 leading = 1
- Dim 2: N patches
- Dim 3: D feature dim (1536 for UNI2-h)

`HilbertSequenceEncoder.forward()` detects 4D input and reshapes to `(B, N, D)`
by merging dims 0 and 1. For the common single-slide case B=1×1=1.

## Coordinate handling

`base.quantize(coords)` shifts coordinates to non-negative integers, divides by
the inferred patch stride (smallest positive Δ), and returns grid cells together
with the bit-depth `k` such that `2**k >= grid_side`. Each ordering builds its
1-D index from those cells.

## Adding a new ordering

1. Create `hilbert_wsi/ordering/<name>.py`.
2. Subclass `OrderingScheme`, implement `__call__`.
3. Register in `hilbert_wsi/ordering/__init__.py::_REGISTRY`.
4. Add a parametrised test entry in `tests/test_orderings.py` if locality matters.

## Adding a new backbone

1. Create `hilbert_wsi/models/<name>.py`.
2. Subclass `SequenceBackbone`, decorate with `@register_backbone("name")`.
3. Import the module in `hilbert_wsi/models/__init__.py`.

## Patho-Bench integration

`adapters/patho_bench.register_hilbert_encoders()` wraps
`trident.slide_encoder_models.load.encoder_factory`. Model names that match
`hilbertwsi_<ordering>_<backbone>` are routed to `HilbertSequenceEncoder`; all
other names fall through to the original factory unchanged. The wrapper is
idempotent and a no-op when invoked twice.

`model_kwargs` in the Patho-Bench config maps directly to
`HilbertSequenceEncoder.__init__`:

```yaml
input_dim: 1536
embedding_dim: 512
backbone_kwargs: { depth: 4, d_state: 64, pooling: mean }
ordering_kwargs: {}
```
