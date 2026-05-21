# HilbertWSI

> Order WSI tile embeddings along a **space-filling curve** (Hilbert by default), then process the sequence with **NLP-style architectures** (Mamba in v1).

Aggregation in MIL for whole-slide images is dominated by attention pooling (ABMIL) and graph networks. HilbertWSI explores an alternative: cast a slide as a **1-D sequence of patch embeddings**, where the order is induced by a 2-D space-filling curve so spatial locality is preserved. Off-the-shelf sequence models (Mamba, Transformer, …) become drop-in slide encoders.

```
H5 patch features                           Patho-Bench
+ patch coords    ──►  Ordering scheme  ──► Sequence    ──► task / eval
(N,D), (N,2)           (Hilbert, Z, …)      backbone        (linprobe, …)
                                            (Mamba, …)
```

## Status

v0.1 scaffold. Mamba backbone implemented; ordering registry has Hilbert, Z-order, Peano, Moore, snake, random, similarity. End-to-end Patho-Bench integration via `adapters/patho_bench.py`.

## Install

### Option A — conda (recommended for reproducibility)

```bash
git clone <repo>
cd HilbertWSI
conda env create -f environment.yml
conda activate hilbert-wsi
pip install -e .

# Optional GPU-only — on a CUDA box:
pip install "mamba-ssm>=2.2" "causal-conv1d>=1.4"

# Optional bench deps — install when you actually run Patho-Bench:
pip install "git+https://github.com/mahmoodlab/trident.git"
pip install "git+https://github.com/mahmoodlab/Patho-Bench.git"
```

### Option B — pip extras

```bash
pip install -e ".[mamba,bench,dev]"
```

`bench` pulls [TRIDENT](https://github.com/mahmoodlab/TRIDENT) and [Patho-Bench](https://github.com/mahmoodlab/Patho-Bench) directly from GitHub. The Mamba backbone needs CUDA; CPU-only smoke tests work without `mamba-ssm`.

## Quick start

1. **Extract patch features** with TRIDENT (UNI2-h, CONCH, Virchow, GigaPath, ...). Skip if you already have H5 files with `features` + `coords`.
2. **Run a Patho-Bench task** with a HilbertWSI encoder:

   ```bash
   python -m scripts.run_benchmark \
       --task tcga_brca/BRCA_Subtype \
       --experiment_type linprobe \
       --model_name hilbertwsi_hilbert_mamba \
       --patch_features_dir /path/to/uni2h_features \
       --model_kwargs_yaml configs/backbones/mamba_base.yaml \
       --saveto runs/brca_hilbert_mamba
   ```

3. **Ordering ablation** (same backbone, sweep ordering schemes):

   ```bash
   python -m scripts.ablation_orderings \
       --task tcga_brca/BRCA_Subtype \
       --backbone mamba \
       --patch_features_dir /path/to/uni2h_features \
       --model_kwargs_yaml configs/backbones/mamba_base.yaml \
       --saveto runs/ablation_brca
   ```

Model name format: `hilbertwsi_<ordering>_<backbone>`, e.g. `hilbertwsi_zorder_mamba`, `hilbertwsi_snake_mamba`.

## Repo layout

```
hilbert_wsi/
├── ordering/       # OrderingScheme ABC + 7 schemes (Hilbert, Z-order, Peano, Moore, snake, random, similarity)
├── models/         # SequenceBackbone ABC + Mamba backbone, registry
├── encoder.py      # HilbertSequenceEncoder — Trident-compatible slide encoder
└── multiscale.py   # phase-2 placeholder
adapters/
└── patho_bench.py  # register_hilbert_encoders() patches Trident encoder_factory
configs/            # ordering / backbone / experiment YAMLs
scripts/            # run_benchmark.py, ablation_orderings.py
tests/              # ordering invariants + encoder smoke test
docs/               # architecture.md, benchmarks.md
```

## Extending

- **New ordering**: subclass `OrderingScheme`, register in `hilbert_wsi/ordering/__init__.py`. Single method: `(coords: (N,2)) -> perm: (N,)`.
- **New backbone**: subclass `SequenceBackbone`, decorate with `@register_backbone("name")`. Single method: `(seq: (B,N,D), mask) -> (B,D_out)`.

See [docs/architecture.md](docs/architecture.md).

## Tests

```bash
pytest tests/
```

The Mamba backbone test is skipped automatically if `mamba-ssm` is missing.

## Differentiation vs prior art

| Prior work | Limitation we address |
|---|---|
| MambaBack (arXiv 2604.15729) | Only PANDA, only Mamba — we sweep multiple orderings and (planned) multiple sequence backbones over the full Patho-Bench task suite |
| MambaMIL (MICCAI 2024) | Re-orders by feature similarity, not space-filling — we include similarity as a baseline |
| Hilbert curves for WSI retrieval (arXiv 2005.06469, 2020) | No sequence model |

## Citations

If you build on this, please cite:

- Mamba2: Dao & Gu, 2024.
- Patho-Bench: Mahmood Lab, 2025. https://arxiv.org/abs/2502.06750
- TRIDENT: Mahmood Lab, 2025. https://github.com/mahmoodlab/TRIDENT
- MambaBack (prior art): arXiv 2604.15729.

## License

MIT — see [LICENSE](LICENSE).
