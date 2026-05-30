# Benchmarks

## Overview

All experiments use:
- **Features**: UNI2-h (1536-dim), extracted with TRIDENT
- **Framework**: Patho-Bench finetune (50-fold cross-validation, balanced classes, 20 epochs, AdamW lr=1e-4, cosine scheduler)
- **Metric**: macro-OVR-AUC (primary), ± = 1 SE across folds
- **CI test**: 2·SE overlap (≈95%)

---

## Results

> **Single source of truth for all numerical results**: [`docs/results_summary.md`](results_summary.md)

This file contains run commands, infrastructure, and model reference. Numerical tables live in `results_summary.md` to avoid divergence.

**CI test methodology**: 2·SE overlap test. Disjoint CIs ≈ 95% CI non-overlap → effect considered significant.

---

## Reproducing

### Prerequisites

```bash
conda activate hilbert-wsi-clean
# Features: download from MahmoodLab/UNI2-h-features on HuggingFace (gated)
# Splits: auto-downloaded from HuggingFace by SplitFactory
```

### Single experiment

```bash
python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --experiment_type finetune \
    --model_name hilbertwsi_snake_transformer \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/ucec_snake_transformer \
    --model_kwargs_yaml configs/backbones/transformer_base.yaml \
    --num_epochs 20 --balanced --gpu 0
```

### Full killer sweep (all no-ordering controls)

```bash
bash scripts/run_killer_experiment.sh    # TransMIL, MambaMIL, Random+Transformer
bash scripts/run_twodmamba_killer.sh     # 2DMamba
```

### Ordering ablation (one backbone, all orderings)

```bash
python -m scripts.run_sweep \
    --source cptac_ucec --task Immune_class \
    --backbone transformer \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/ucec_sweep_transformer
```

### Available model names

**HilbertWSI encoders** (format: `hilbertwsi_<ordering>_<backbone>`):
- Orderings: `hilbert`, `zorder`, `peano`, `moore`, `snake`, `random`, `similarity`
- Backbones: `mamba`, `transformer`, `xlstm`, `gla`

**Baselines** (no ordering):
- `abmil_baseline` — ABMIL (attention pooling)
- `transmil_baseline` — TransMIL (Nyströmformer MIL)
- `mambamil_baseline` — MambaMIL (Mamba MIL, no ordering)
- `twodmamba_baseline` — 2DMamba (2D-selective SSM, CVPR 2025)
- `mean-uni_v2` — mean pooling + linear probe

---

## Datasets available

| Dataset | Task | Slides | Features | Splits |
|---|---|---|---|---|
| CPTAC-UCEC | Immune_class | 95 | UNI2-h 1536-dim | ✅ splits/ |
| PANDA | isup_grade | ~11k | UNI2-h 1536-dim | ✅ splits/ |
| CPTAC-BRCA | Immune_class | — | UNI2-h 1536-dim | ✅ splits/ |
| CPTAC-BRCA | PIK3CA_mutation | — | UNI2-h 1536-dim | ✅ splits/ |
| CPTAC-CCRCC | OS (survival) | — | UNI2-h 1536-dim | ✅ splits/ |
| CPTAC-CCRCC | BAP1/Immune/PBRM1/VHL | — | UNI2-h 1536-dim | ✅ splits/ |
| EBRAINS | 30-class brain tumour | — | to extract | pending |
