# Benchmarks

## Overview

All experiments use:
- **Features**: UNI2-h (1536-dim), extracted with TRIDENT
- **Framework**: Patho-Bench finetune (50-fold cross-validation, balanced classes, 20 epochs, AdamW lr=1e-4, cosine scheduler)
- **Metric**: macro-OVR-AUC (primary), ± = 1 SE across folds
- **CI test**: 2·SE overlap (≈95%)

---

## CPTAC-UCEC — Immune_class (95 slides, 5 classes)

Full killer experiment. Task is **not saturated** — ideal for discriminating methods.

### Complete results table

| Model | acc | macro-OVR-AUC | macro-F1 | w-κ |
|---|---|---|---|---|
| Snake + Transformer ⭐ | **0.452 ± 0.013** | **0.629 ± 0.012** | **0.437 ± 0.013** | 0.355 ± 0.025 |
| Zorder + Transformer | 0.448 ± 0.017 | 0.622 ± 0.013 | 0.436 ± 0.016 | 0.337 ± 0.026 |
| Hilbert + Transformer | 0.444 ± 0.013 | 0.614 ± 0.010 | 0.425 ± 0.013 | 0.308 ± 0.026 |
| Peano + Transformer | 0.445 ± 0.015 | 0.613 ± 0.012 | 0.430 ± 0.015 | 0.317 ± 0.029 |
| Moore + Transformer | 0.435 ± 0.014 | 0.608 ± 0.011 | 0.422 ± 0.014 | 0.325 ± 0.025 |
| Mean-pool (linprobe) | 0.403 ± 0.012 | 0.601 ± 0.010 | 0.387 ± 0.013 | 0.200 ± 0.031 |
| Hilbert + Mamba | 0.405 ± 0.013 | 0.596 ± 0.011 | 0.387 ± 0.014 | 0.215 ± 0.030 |
| ABMIL | 0.387 ± 0.014 | 0.594 ± 0.013 | 0.373 ± 0.015 | 0.199 ± 0.031 |
| Peano + Mamba | 0.401 ± 0.014 | 0.594 ± 0.012 | 0.387 ± 0.014 | — |
| Similarity + Mamba | 0.408 ± 0.014 | 0.592 ± 0.011 | 0.391 ± 0.014 | — |
| Zorder + Mamba | 0.398 ± 0.012 | 0.588 ± 0.011 | 0.384 ± 0.013 | — |
| Snake + Mamba | 0.400 ± 0.014 | 0.582 ± 0.011 | 0.385 ± 0.014 | — |
| Moore + Mamba | 0.375 ± 0.013 | 0.573 ± 0.011 | 0.358 ± 0.013 | — |
| **TransMIL** (no ordering) ⚠ | 0.338 ± 0.013 | 0.542 ± 0.013 | 0.252 ± 0.013 | 0.024 ± 0.026 |
| **2DMamba** (2D-native SSM) ⚠† | 0.368 ± 0.014 | 0.527 ± 0.015 | 0.302 ± 0.017 | 0.042 ± 0.029 |
| **Random + Transformer** ⚠ | 0.344 ± 0.012 | 0.519 ± 0.016 | 0.269 ± 0.014 | 0.010 ± 0.028 |
| **MambaMIL** (no ordering) ⚠ | 0.319 ± 0.012 | 0.484 ± 0.014 | 0.258 ± 0.014 | -0.007 ± 0.025 |

⭐ best  |  ⚠ no-ordering controls  |  † patch_size=1024 (pure-Python pscan, 14.4M params; some tile collision loss)

### Decision tree (2·SE CI overlap test)

| Comparison | Δ AUC | CI overlap? | Interpretation |
|---|---|---|---|
| Snake+Transformer vs Random+Transformer | **+0.110** | **DISJOINT** | Ordering matters for Transformer: same arch, +0.11 AUC with spatial structure |
| Hilbert+Mamba vs MambaMIL | **+0.112** | **DISJOINT** | Ordering is **essential** for Mamba: without it Mamba collapses to ~random |
| Snake+Transformer vs TransMIL | **+0.087** | **DISJOINT** | Transformer arch alone doesn't explain the gain: ordering contributes independently |
| 2DMamba vs Hilbert+Mamba | **−0.069** | **DISJOINT** | 2D-native SSM (CVPR 2025) does NOT bypass 1D ordering; falls in no-ordering cluster |
| 2DMamba vs ABMIL | **−0.067** | **DISJOINT** | Classic ABMIL outperforms 2DMamba without explicit ordering |
| 2DMamba vs Random+Transformer | +0.008 | overlap | 2DMamba ≈ random ordering → 2D structure doesn't substitute explicit spatial ordering |
| Snake+Transformer vs ABMIL | +0.034 | overlap | Ordering+Transformer vs ABMIL: borderline (partial CI overlap) |
| TransMIL vs ABMIL | −0.052 | DISJOINT | TransMIL **worse than ABMIL** on UCEC (overfit 13M params, PPEG mismatched) |

### Conclusions

1. **Thesis holds in strong form**: 1D spatial ordering contributes ~+0.11 AUC independently for both Transformer and Mamba. CI disjoint in all critical comparisons.

2. **Mamba requires ordering more acutely than Transformer**: Mamba jumps from AUC 0.484 (no ordering) to 0.596 (Hilbert). Transformer jumps from 0.519 (random) to 0.629 (Snake). SSMs on WSI data *need* imposed spatial structure or they collapse to near-random.

3. **2D-native paradigm (2DMamba) does not bypass 1D ordering**: AUC 0.527 ± 0.015, CI disjoint vs Hilbert+Mamba and ABMIL. The no-ordering cluster groups as: MambaMIL (0.484) < Random+Transformer (0.519) ≈ 2DMamba (0.527) < TransMIL (0.542) — all significantly below ABMIL (0.594).

4. **No-ordering models underperform ABMIL**: Over-parameterised SSMs and Transformers (13–22M params) without spatial structure fail to outperform simple attention pooling (1.5M). Likely causes: overfit on small 50-fold UCEC cohort, architectures designed for N>>1000 patches, attention/PPEG heads inappropriate for diffuse immune signal.

5. **Ordering differences are secondary to ordering vs no-ordering**: Snake slightly edges Hilbert for Transformer (Δ≤0.015, CI overlap). The collective ordering vs random gap is large and significant; the ordering choice matters less.

---

## PANDA-ISUP grading (32GB, UNI2-h)

Task is **saturated** with UNI2-h — no meaningful differences between methods.

| Model | acc | macro-OVR-AUC | weighted-κ |
|---|---|---|---|
| ABMIL (no ordering) | 0.812 | 0.963 | 0.944 |
| Hilbert + Mamba | 0.802 | 0.962 | 0.943 |
| Z-order + Mamba | 0.793 | 0.962 | 0.949 |
| Mean-pool + linprobe | 0.753 | 0.943 | 0.910 |

All CI completely overlapping. PANDA-ISUP grading is at ceiling with UNI2-h features regardless of aggregation method.

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
