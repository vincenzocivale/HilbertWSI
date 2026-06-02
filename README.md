# HilbertWSI

> Order WSI tile embeddings along a **space-filling curve**, then process the sequence with **NLP-style sequence models** (Mamba, Transformer, xLSTM, GLA).

Aggregation in MIL for whole-slide images is dominated by attention pooling (ABMIL) and graph networks. HilbertWSI casts a slide as a **1-D sequence of patch embeddings** where order is induced by a 2-D space-filling curve, so spatial locality is preserved. Off-the-shelf sequence models become drop-in slide encoders.

```
H5 patch features                          Patho-Bench
+ patch coords   ──► Ordering scheme ──► Sequence    ──► task / eval
(N,D), (N,2)         (Hilbert, Z, …)     backbone        (finetune, linprobe, …)
                                         (Mamba, Transformer, …)
```

## Key findings (CPTAC-UCEC Immune_class, May 2026)

**Ordering matters: +0.11 AUC over no-ordering controls (CI disjoint).**

| Model | macro-OVR-AUC |
|---|---|
| Snake + Transformer ⭐ | **0.629 ± 0.012** |
| Hilbert + Mamba | 0.596 ± 0.011 |
| ABMIL (no ordering) | 0.594 ± 0.013 |
| 2DMamba (2D-native SSM) ⚠ | 0.527 ± 0.015 |
| TransMIL (no ordering) ⚠ | 0.542 ± 0.013 |
| Random + Transformer ⚠ | 0.519 ± 0.016 |
| MambaMIL (no ordering) ⚠ | 0.484 ± 0.014 |

⚠ = no-ordering control. Full table in [docs/results_summary.md](docs/results_summary.md).

**Decision tree outcome:**
- Snake+Transformer >> Random+Transformer (+0.11 AUC, **DISJOINT CI**) → ordering contributes independently of architecture
- Hilbert+Mamba >> MambaMIL (+0.11 AUC, **DISJOINT CI**) → ordering is *essential* for Mamba; without it Mamba collapses to near-random
- 2DMamba (CVPR 2025 rival) ≈ Random+Transformer (CI overlap) → 2D-native SSM does not bypass 1D ordering
- All no-ordering controls ≤ ABMIL → over-parameterised models without spatial structure fail to outperform simple pooling

> ⚠ **Caveat (2026-05-31)**: i confronti HilbertWSI (pool=mean) vs MIL
> baselines (pool=attn) confondono ordering ed pooling head. Per la
> verifica controllata vedi [**docs/verification_protocol.md**](docs/verification_protocol.md)
> — definisce gli esperimenti A1–A4 che isolano l'effetto causale del
> 1D SFC ordering rispetto al pool e all'architettura.

## Status

| Component | Status |
|---|---|
| Ordering registry (7 schemes) | ✅ implemented + tested |
| Mamba backbone (mamba-ssm 2.3.2) | ✅ implemented + tested |
| Transformer backbone | ✅ implemented + tested |
| xLSTM backbone | ✅ implemented (requires `pip install xlstm`) |
| GLA backbone | ✅ implemented |
| TransMIL baseline (no ordering) | ✅ implemented |
| MambaMIL baseline (no ordering) | ✅ implemented |
| 2DMamba baseline (CVPR 2025) | ✅ vendored + wrapper |
| Patho-Bench integration | ✅ full finetune / linprobe / coxnet |
| CPTAC-UCEC killer experiment | ✅ complete (16 models, 50 folds) |
| PANDA-ISUP ablation | ✅ complete (task saturated) |

## Install

```bash
# Recommended: use the pre-configured conda env
conda activate hilbert-wsi-clean  # Python 3.11, torch 2.5.1+cu124, mamba-ssm 2.3.2

# Or create from scratch
conda env create -f environment.yml
conda activate hilbert-wsi-clean
pip install -e ".[bench,dev]"

# Patho-Bench + TRIDENT (required for benchmarks)
pip install "git+https://github.com/mahmoodlab/trident.git"
pip install "git+https://github.com/mahmoodlab/Patho-Bench.git"
```

Note: `torch` is not in `pyproject.toml` — install manually with the CUDA wheel matching your system before `pip install -e .`.

## Quick start

```bash
# Single Patho-Bench task (finetune)
python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --experiment_type finetune \
    --model_name hilbertwsi_hilbert_mamba \
    --patch_features_dir /path/to/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/ucec_hilbert_mamba \
    --model_kwargs_yaml configs/backbones/mamba_base.yaml \
    --num_epochs 20 --balanced --gpu 0

# Sweep all orderings (same backbone)
python -m scripts.run_sweep \
    --source cptac_ucec --task Immune_class \
    --backbone transformer \
    --patch_features_dir /path/to/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/ucec_sweep_transformer
```

Model name format: `hilbertwsi_<ordering>_<backbone>`, e.g. `hilbertwsi_snake_transformer`.
Baselines: `abmil_baseline`, `transmil_baseline`, `mambamil_baseline`, `twodmamba_baseline`.

## Repo layout

```
hilbert_wsi/
├── ordering/           # OrderingScheme ABC + 7 schemes
│   ├── hilbert.py      # Hilbert curve (z-order quantisation)
│   ├── zorder.py       # Z-order / Morton code
│   ├── peano.py        # Peano S-curve (canonical, verified)
│   ├── moore.py        # Moore curve
│   ├── snake.py        # Row-major snake scan
│   ├── random_perm.py  # Per-slide deterministic random (blake2b seed)
│   └── similarity.py   # Feature-similarity nearest-neighbour chain
├── models/
│   ├── mamba.py        # MambaBackbone (mamba-ssm 2.3.2, 22M @ depth=8)
│   ├── transformer.py  # TransformerBackbone (13M @ depth=12)
│   ├── xlstm.py        # xLSTMBackbone (requires xlstm package)
│   ├── gla.py          # GLABackbone (pure-PyTorch chunk-parallel GLA)
│   ├── mambamil.py     # MambaMIL baseline (no ordering control)
│   ├── transmil.py     # TransMIL baseline (no ordering control)
│   ├── twodmamba.py    # 2DMamba wrapper (paradigm rival from CVPR 2025)
│   └── vendor/
│       ├── MambaMIL.py        # Vendored MambaMIL (Mahmood Lab)
│       ├── TransMIL.py        # Vendored TransMIL (Shao et al.)
│       └── twodmamba/         # Vendored 2DMambaMIL (Zhang et al., CVPR 2025)
├── encoder.py          # HilbertSequenceEncoder — Trident-compatible
└── multiscale.py       # Phase 3 placeholder
adapters/
└── patho_bench.py      # register_hilbert_encoders() — call before Patho-Bench
configs/
├── orderings/          # YAML per ordering scheme
├── backbones/          # mamba_base, transformer_base, transmil_base, …
└── experiments/        # experiment YAMLs
scripts/
├── run_benchmark.py    # CLI: single Patho-Bench experiment
├── run_sweep.py        # sweep all orderings × one backbone
├── ablation_orderings.py
├── run_killer_experiment.sh   # Phase 2 killer (TransMIL, MambaMIL, Random)
└── run_twodmamba_killer.sh    # Phase 3 killer (2DMamba)
tests/
├── test_orderings.py   # 30+ tests incl. Peano regression + random per-slide
└── test_encoder.py     # smoke test + Mamba CUDA forward
docs/
├── architecture.md           # codebase reference (modules, contracts, baselines)
├── benchmarks.md             # run commands + dataset listing
├── results_summary.md        # SSOT for numerical results (UCEC, BRCA, PANDA)
├── verification_protocol.md  # A1–A4 protocol to verify SFC advantage causally
├── roadmap_c1.md             # C1 experiments (ordering vs no-ordering, same arch)
├── roadmap_c2.md             # C2 experiments (HilbertWSI vs MIL baselines)
└── archive/
    └── repo_corrections_2026-05-30.md  # historical audit (resolved by 91bc137)
```

## Extending

**New ordering**: subclass `OrderingScheme` in `hilbert_wsi/ordering/`, register in `hilbert_wsi/ordering/__init__.py::_REGISTRY`. Single method: `(coords: (N,2)) -> perm: (N,)`.

**New backbone**: subclass `SequenceBackbone`, decorate with `@register_backbone("name")`, import in `hilbert_wsi/models/__init__.py`.

See [docs/architecture.md](docs/architecture.md).

## Tests

```bash
conda activate hilbert-wsi-clean
pytest tests/ -v  # 30+ tests, includes Mamba CUDA forward
```

## Differentiation vs prior art

| Prior work | Limitation addressed |
|---|---|
| MambaBack (arXiv 2604.15729) | Only PANDA, only Mamba — we sweep 7 orderings × 4 backbones on non-saturated tasks |
| MambaMIL (MICCAI 2024) | Re-orders by feature similarity, not space-filling; no Transformer comparison |
| 2DMamba (arXiv 2412.00678, CVPR 2025) | 2D-native SSM; our killer exp shows it falls in the no-ordering performance cluster |
| Hilbert curves for WSI retrieval (arXiv 2005.06469) | No sequence model; retrieval only |

## Citations

- Mamba2: Dao & Gu, 2024.
- Patho-Bench: Mahmood Lab, 2025. arXiv 2502.06750.
- TRIDENT: Mahmood Lab, 2025.
- 2DMamba: Zhang et al., CVPR 2025. arXiv 2412.00678.
- MambaMIL: Yang et al., MICCAI 2024.
- TransMIL: Shao et al., NeurIPS 2021.

## License

MIT — see [LICENSE](LICENSE).
