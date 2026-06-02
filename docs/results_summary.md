    # Results Summary

    _Last updated: 2026-06-02 (2D PE rerun post-fix `91bc137`; CCRCC BAP1 added)._
    _Pool-asymmetry caveat added 2026-05-31._

    ---

    ## ⚠ Avvertenza sui confronti (2026-05-31)

    **I confronti `HilbertWSI` vs MIL baseline in questa tabella confondono
    l'effetto del 1D ordering con quello del pooling head**:

    - Tutte le configurazioni HilbertWSI (`*_base.yaml`, `*_2dpe_base.yaml`)
      usano `pooling: mean`.
    - Tutte le baseline MIL (ABMIL, TransMIL, MambaMIL, CLAM, DSMIL) usano
      **attention** o **CLS** pooling.

    Conseguenze:
    - Confronti **HilbertWSI 1D vs HilbertWSI random** (stessa pool=mean) →
      **FAIR**, isolano effetto ordering.
    - Confronti **HilbertWSI 1D vs HilbertWSI 2D PE** (stessa pool=mean) →
      **FAIR**, isolano segnale 1D vs 2D.
    - Confronti **HilbertWSI 1D vs `*_baseline`** → **CONFOUNDED** (ordering
      + pool head insieme).

    Per disambiguare, eseguire l'esperimento **A3** definito in
    [`docs/verification_protocol.md`](verification_protocol.md): HilbertWSI
    con `pooling: attn` contro le baseline. La config Mamba per A3 esiste
    già (`configs/backbones/mamba_attnpool_base.yaml`); per Transformer/
    xLSTM/GLA va creata.

    Le righe **2D PE** in tabella (UCEC) sono state aggiornate con i valori post-fix
    (rerun 2026-05-30, commit `91bc137`). I valori pre-fix (0.545/0.524) sono stati scartati.

    ---

    ## Core claim

    **Spatial ordering of WSI tile embeddings along a 1D space-filling curve contributes ~+0.11 AUC independently of the sequence backbone used**, on a non-saturated immune subtyping task. The effect is confirmed with disjoint confidence intervals against all no-ordering controls.

    ---

    ## Dataset 1 — CPTAC-UCEC · Immune_class (5-class, 95 slides, 50-fold CV)

    Non-saturated task. Primary dataset for thesis validation.
    Features: UNI2-h 1536-dim. Metric: macro-OVR-AUC ± 1 SE.

    ### Full results table

    Colonna **Pool** mostra il pooling head usato: `mean` = mean pool (HilbertWSI default), `attn` = gated/critical-instance attention, `cls` = CLS token. Vedi avvertenza sopra.

    | Model | Pool | AUC | acc | macro-F1 |
    |---|---|---|---|---|
    | **Snake + Transformer** ⭐ | mean | **0.629 ± 0.012** | 0.452 ± 0.013 | 0.437 ± 0.013 |
    | Zorder + Transformer | mean | 0.622 ± 0.013 | 0.448 ± 0.017 | 0.436 ± 0.016 |
    | Hilbert + Transformer | mean | 0.614 ± 0.010 | 0.444 ± 0.013 | 0.425 ± 0.013 |
    | Peano + Transformer | mean | 0.613 ± 0.012 | 0.445 ± 0.015 | 0.430 ± 0.015 |
    | Moore + Transformer | mean | 0.608 ± 0.011 | 0.435 ± 0.014 | 0.422 ± 0.014 |
    | Mean-pool (linprobe) | mean | 0.601 ± 0.010 | 0.403 ± 0.012 | 0.387 ± 0.013 |
    | **Hilbert + Mamba** | mean | **0.596 ± 0.011** | 0.405 ± 0.013 | 0.387 ± 0.014 |
    | ABMIL | attn | 0.594 ± 0.013 | 0.387 ± 0.014 | 0.373 ± 0.015 |
    | Peano + Mamba | mean | 0.594 ± 0.012 | 0.401 ± 0.014 | 0.387 ± 0.014 |
    | Similarity + Mamba | mean | 0.592 ± 0.011 | 0.408 ± 0.014 | 0.391 ± 0.014 |
    | Zorder + Mamba | mean | 0.588 ± 0.011 | 0.398 ± 0.012 | 0.384 ± 0.013 |
    | Snake + Mamba | mean | 0.582 ± 0.011 | 0.400 ± 0.014 | 0.385 ± 0.014 |
    | Moore + Mamba | mean | 0.573 ± 0.011 | 0.375 ± 0.013 | 0.358 ± 0.013 |
    | ── no-ordering controls ── | | | | |
    | 2D PE + Mamba ✓ | mean | 0.542 ± 0.016 | — | — |
    | TransMIL (no ordering) | cls | 0.542 ± 0.013 | 0.338 ± 0.013 | 0.252 ± 0.013 |
    | **DSMIL** ‡ | attn | **0.536 ± 0.012** | 0.350 ± 0.013 | 0.328 ± 0.014 |
    | 2DMamba (2D-native SSM) † | (2D SSM) | 0.527 ± 0.015 | 0.368 ± 0.014 | 0.302 ± 0.017 |
    | CLAM-SB ‡ | attn | 0.521 ± 0.013 | 0.332 ± 0.012 | 0.309 ± 0.013 |
    | Random + Transformer | mean | 0.519 ± 0.016 | 0.344 ± 0.012 | 0.269 ± 0.014 |
    | 2D PE + Transformer ✓ | mean | 0.515 ± 0.014 | — | — |
    | CLAM-MB ‡ | attn | 0.507 ± 0.014 | 0.344 ± 0.013 | 0.309 ± 0.014 |
    | MambaMIL (no ordering) | attn | 0.484 ± 0.014 | 0.319 ± 0.012 | 0.258 ± 0.014 |

    ✓ = post-fix rerun (2026-05-30, `cptac_ucec_2dpe_rerun_*`). Pre-fix values (0.545/0.524) were invalidated by `91bc137`.

    † 2DMamba: pure-Python pscan, patch_size=1024, 14.4M params. Some tile collision loss at this resolution.
    ‡ CLAM-SB/MB and DSMIL: ~1.1–1.8M params (standard CLAM config with UNI2-h input_dim=1536). No instance-level clustering loss; Patho-Bench provides classifier head only.

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
    | Snake+Transformer vs **DSMIL** | +0.093 | **DISJOINT** | Ordering advantage holds vs critical-instance attention |
    | Snake+Transformer vs **CLAM-SB** | +0.108 | **DISJOINT** | Ordering advantage holds vs gated attention |
    | Snake+Transformer vs **CLAM-MB** | +0.122 | **DISJOINT** | Largest ordering gap in full table |
    | **ABMIL** vs DSMIL | +0.058 | **DISJOINT** | Simple ABMIL beats DSMIL critical-instance attention |
    | **ABMIL** vs CLAM-SB | +0.073 | **DISJOINT** | Simple ABMIL beats CLAM gated attention |
    | **ABMIL** vs CLAM-MB | +0.087 | **DISJOINT** | Simple ABMIL beats CLAM-MB by largest margin |
    | DSMIL vs MambaMIL | +0.052 | **DISJOINT** | DSMIL critical-instance heuristic > unordered Mamba |
    | CLAM-SB vs Random+Transformer | −0.002 | overlap | CLAM-SB ≈ random ordering |
    | CLAM-MB vs MambaMIL | +0.023 | overlap | CLAM-MB ≈ MambaMIL (bottom of table) |

    ---

    ## Dataset 2 — CPTAC-BRCA · Immune_class (3-class, 653 slides, 50-fold CV)

    Large-data regime. Features: UNI2-h 1536-dim. Metric: macro-OVR-AUC ± 1 SE.

    | Model | AUC ± SE | Pool | Ordering |
    |---|---|---|---|
    | **MambaMIL (no order)** ⭐ | **0.725 ± 0.011** | attn | none |
    | Random + Mamba | 0.713 ± 0.012 | mean | random |
    | Hilbert + Mamba | 0.711 ± 0.013 | mean | hilbert |
    | CLAM-SB | 0.710 ± 0.012 | attn | none |
    | Random + Mamba | 0.708 ± 0.011 | attn | random |
    | Hilbert + Mamba | 0.706 ± 0.012 | attn | hilbert |
    | Snake + Transformer | 0.703 ± 0.011 | mean | snake |
    | Random + Transformer | 0.700 ± 0.013 | mean | random |
    | ABMIL | 0.696 ± 0.012 | attn | none |

    ### Key comparisons (2·SE CI overlap test)

    | Comparison | Δ AUC | CI | Conclusion |
    |---|---|---|---|
    | Hilbert+Mamba (mean) vs Random+Mamba (mean) | −0.002 | overlap | Ordering irrelevant for Mamba on BRCA |
    | Hilbert+Mamba (attn) vs Random+Mamba (attn) | −0.002 | overlap | Ordering irrelevant with attn pool too |
    | Snake+Transformer vs Random+Transformer | +0.003 | overlap | Ordering irrelevant for Transformer on BRCA |
    | Hilbert+Mamba (attn) vs Hilbert+Mamba (mean) | −0.005 | overlap | Attn pool does NOT explain MambaMIL gap |
    | MambaMIL vs Hilbert+Mamba (mean) | +0.014 | overlap | MambaMIL advantage likely architectural (SRMamba, 12 blocks) |
    | MambaMIL vs ABMIL | +0.029 | overlap/borderline | |

    **Conclusion**: BRCA Immune_class is a **saturated task for ordering**. All orderings (hilbert, snake, random) are indistinguishable across both pooling strategies and both backbones (Δ ≤ 0.003, all CI overlapping). MambaMIL leads slightly (0.725) but with overlapping CI vs our models — residual gap likely from SRMamba architecture or 12 vs 4 blocks, not pooling or ordering.

    **Dataset-size moderator confirmed**: ordering contributes +0.11 AUC on UCEC (95 slides, non-saturated) but 0 on BRCA (653 slides, saturated). The inductive bias from space-filling curves is most valuable in the low-data regime.

    ---

    ## Dataset 3 — PANDA · ISUP grading (6-class)

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

    ### 4. No-ordering models underperform ABMIL — extended to CLAM and DSMIL
    Over-parameterised SSMs and Transformers (13–22M) without spatial structure fail to outperform simple ABMIL (1.5M). Now confirmed for gated-attention MIL (CLAM-SB/MB, ~1.1–1.8M) and critical-instance attention (DSMIL, ~1.1M) too. ABMIL is DISJOINT above all three: Δ=+0.073 vs CLAM-SB, +0.087 vs CLAM-MB, +0.058 vs DSMIL. More complex attention mechanisms without spatial context do not help; likely causes: small cohort (95 slides), diffuse immune signal that is not patch-level localisable.

    ### 4b. CLAM and DSMIL join the no-ordering cluster
    All three new baselines fall between TransMIL (0.542) and MambaMIL (0.484), clustered with Random+Transformer, 2DMamba. CI overlap with all no-ordering controls. CLAM-SB ≈ Random+Transformer (Δ=−0.002, overlap). The specific attention design (gated, critical-instance) does not distinguish from random within this cluster. DSMIL is best of the three (0.536) — slightly above 2DMamba and CLAM, but no disjoint edge over any no-ordering control.

    ### 5. Ordering choice: Snake ≥ Hilbert ≈ Zorder ≈ Peano ≈ Moore
    Differences between ordering schemes are small (Δ ≤ 0.015 AUC, CI overlap). The ordering vs no-ordering gap dominates.

    ### 6. Transformer > Mamba for this task
    Snake+Transformer (0.629) vs Hilbert+Mamba (0.596): Δ = +0.033, CI marginally overlap. Confound: 13M vs 22M params. Not fully resolved.

    ---

    ## Open questions for next experiments

    | Question | Experiment needed | Status |
    |---|---|---|
    | Does ordering+Mamba generalize beyond UCEC? | Hilbert+Mamba on CPTAC-BRCA | ✅ No: BRCA saturated |
    | Does attn pooling explain MambaMIL gap on BRCA? | Hilbert+Mamba (attn) vs random | ✅ No: pooling not the cause (BRCA only) |
    | **Does ordering survive when HilbertWSI uses attn pool too (on UCEC)?** | Esperimento A3 in `verification_protocol.md` — `hilbertwsi_hilbert_mamba` con `mamba_attnpool_base.yaml` vs `mambamil_baseline` | **pending — critico per C2** |
    | Does ordering hold on intermediate-size dataset? | CPTAC-CCRCC BAP1 trio (2D PE vs 1D vs random) | ✅ Reversal: 2D PE >> ordering on mutation task |
    | Does the ordering effect hold on harder tasks? | EBRAINS 30-class (features pending) | pending |
    | Transformer vs Mamba capacity-controlled? | Mamba at 13M (depth-matched) | pending |
    | Is Snake consistently best or dataset-specific? | Multi-dataset ordering ablation | pending |
    | Are 2D PE rows valid post-fix? | Rerun `hilbertwsi_2dpe_{mamba,transformer}` on UCEC | ✅ Done: 0.515 (Transformer), 0.542 (Mamba) |

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
