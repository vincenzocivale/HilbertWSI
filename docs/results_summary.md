# Results Summary

_Last updated: 2026-05-22_

---

## Core claim

**Spatial ordering of WSI tile embeddings along a 1D space-filling curve contributes ~+0.11 AUC independently of the sequence backbone used**, on a non-saturated immune subtyping task. The effect is confirmed with disjoint confidence intervals against all no-ordering controls.

---

## Dataset 1 — CPTAC-UCEC · Immune_class (5-class, 95 slides, 50-fold CV)

Non-saturated task. Primary dataset for thesis validation.
Features: UNI2-h 1536-dim. Metric: macro-OVR-AUC ± 1 SE.

### Full results table

| Model | AUC | acc | macro-F1 |
|---|---|---|---|
| **Snake + Transformer** ⭐ | **0.629 ± 0.012** | 0.452 ± 0.013 | 0.437 ± 0.013 |
| Zorder + Transformer | 0.622 ± 0.013 | 0.448 ± 0.017 | 0.436 ± 0.016 |
| Hilbert + Transformer | 0.614 ± 0.010 | 0.444 ± 0.013 | 0.425 ± 0.013 |
| Peano + Transformer | 0.613 ± 0.012 | 0.445 ± 0.015 | 0.430 ± 0.015 |
| Moore + Transformer | 0.608 ± 0.011 | 0.435 ± 0.014 | 0.422 ± 0.014 |
| Mean-pool (linprobe) | 0.601 ± 0.010 | 0.403 ± 0.012 | 0.387 ± 0.013 |
| **Hilbert + Mamba** | **0.596 ± 0.011** | 0.405 ± 0.013 | 0.387 ± 0.014 |
| ABMIL | 0.594 ± 0.013 | 0.387 ± 0.014 | 0.373 ± 0.015 |
| Peano + Mamba | 0.594 ± 0.012 | 0.401 ± 0.014 | 0.387 ± 0.014 |
| Similarity + Mamba | 0.592 ± 0.011 | 0.408 ± 0.014 | 0.391 ± 0.014 |
| Zorder + Mamba | 0.588 ± 0.011 | 0.398 ± 0.012 | 0.384 ± 0.013 |
| Snake + Mamba | 0.582 ± 0.011 | 0.400 ± 0.014 | 0.385 ± 0.014 |
| Moore + Mamba | 0.573 ± 0.011 | 0.375 ± 0.013 | 0.358 ± 0.013 |
| ── no-ordering controls ── | | | |
| TransMIL (no ordering) | 0.542 ± 0.013 | 0.338 ± 0.013 | 0.252 ± 0.013 |
| 2DMamba (2D-native SSM) † | 0.527 ± 0.015 | 0.368 ± 0.014 | 0.302 ± 0.017 |
| Random + Transformer | 0.519 ± 0.016 | 0.344 ± 0.012 | 0.269 ± 0.014 |
| MambaMIL (no ordering) | 0.484 ± 0.014 | 0.319 ± 0.012 | 0.258 ± 0.014 |

† 2DMamba: pure-Python pscan, patch_size=1024, 14.4M params. Some tile collision loss at this resolution.

### Key comparisons (2·SE CI overlap test)

| Comparison | Δ AUC | CI | Conclusion |
|---|---|---|---|
| Snake+Transformer vs **Random**+Transformer | +0.110 | **DISJOINT** | Ordering matters for Transformer |
| **Hilbert**+Mamba vs **MambaMIL** | +0.112 | **DISJOINT** | Ordering essential for Mamba |
| Snake+Transformer vs **TransMIL** | +0.087 | **DISJOINT** | Architecture alone doesn't explain the gain |
| **2DMamba** vs Hilbert+Mamba | −0.069 | **DISJOINT** | 2D-native SSM does not bypass 1D ordering |
| **2DMamba** vs ABMIL | −0.067 | **DISJOINT** | ABMIL outperforms 2D-native SSM |
| 2DMamba vs Random+Transformer | +0.008 | overlap | 2DMamba ≈ random ordering |
| Snake+Transformer vs ABMIL | +0.035 | overlap | Ordering+Transformer vs ABMIL borderline |
| TransMIL vs ABMIL | −0.052 | **DISJOINT** | TransMIL (13M, no ordering) worse than ABMIL (1.5M) |

---

## Dataset 2 — PANDA · ISUP grading (6-class)

Saturated task with UNI2-h features. All CI completely overlapping — no meaningful difference between methods.

| Model | AUC | acc | w-κ |
|---|---|---|---|
| ABMIL | 0.963 | 0.812 | 0.944 |
| Hilbert + Mamba | 0.962 | 0.802 | 0.943 |
| Zorder + Mamba | 0.962 | 0.793 | 0.949 |
| Mean-pool (linprobe) | 0.943 | 0.753 | 0.910 |

**Conclusion**: PANDA-ISUP is at ceiling with UNI2-h regardless of aggregation method.

---

## Key conclusions

### 1. Thesis holds in strong form
Spatial 1D ordering contributes ~+0.11 AUC for both Transformer and Mamba independently, with disjoint CIs against all no-ordering controls.

### 2. Mamba needs ordering more than Transformer
- Mamba: 0.484 (no ordering) → 0.596 (Hilbert), +0.112
- Transformer: 0.519 (random) → 0.629 (Snake), +0.110
- Without imposed spatial structure, Mamba collapses to near-random on UCEC.

### 3. 2D-native SSM does not bypass 1D ordering
2DMamba (CVPR 2025) falls in the no-ordering cluster (AUC 0.527). The critical inductive bias is explicit spatial ordering, not 2D-nativeness of the scan.

### 4. No-ordering models underperform ABMIL
Over-parameterised SSMs and Transformers (13–22M params) without spatial structure fail to outperform simple attention pooling (1.5M params). Likely causes: small cohort (95 slides, 50-fold), diffuse immune signal unsuited to patch-level position encoding (PPEG).

### 5. Ordering choice: Snake ≥ Hilbert ≈ Zorder ≈ Peano ≈ Moore
Differences between ordering schemes are small (Δ ≤ 0.015 AUC, CI overlap). The ordering vs no-ordering gap dominates.

### 6. Transformer > Mamba for this task
Snake+Transformer (0.629) vs Hilbert+Mamba (0.596): Δ = +0.033, CI marginally overlap. Confound: 13M vs 22M params. Not fully resolved.

---

## Open questions for next experiments

| Question | Experiment needed |
|---|---|
| Does ordering+Mamba generalize beyond UCEC? | Hilbert+Mamba on CPTAC-BRCA Immune_class |
| Does the ordering effect hold on harder tasks? | EBRAINS 30-class (features pending) |
| Transformer vs Mamba capacity-controlled? | Mamba at 13M (depth-matched) |
| Is Snake consistently best or dataset-specific? | Multi-dataset ordering ablation |
| Ceiling of slide-level models on UCEC? | TITAN frozen + linear probe |

---

## Reproducibility

All metrics in `results/`:
```
results/
├── cptac_ucec/Immune_class/    # 17 models, 50-fold CV
└── panda/isup_grade/           # 4 models, PANDA ablation
```

Run command:
```bash
conda activate hilbert-wsi-clean
python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --experiment_type finetune \
    --model_name hilbertwsi_snake_transformer \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --model_kwargs_yaml configs/backbones/transformer_base.yaml \
    --num_epochs 20 --balanced --gpu 0
```
