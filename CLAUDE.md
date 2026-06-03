# HilbertWSI — CLAUDE.md

## Progetto

Repo di ricerca: riordinare gli embedding di tile WSI lungo una **curva di Hilbert** (o altra space-filling curve) e processare la sequenza con architetture NLP-style (Mamba/SSM in v1).

Differenziazione vs **MambaBack** (arXiv 2604.15729):
1. Confronto sistematico ordering schemes (Hilbert, Z-order, Peano, Moore, snake, random, similarity)
2. Multi-architettura su stesso ordering (v1 = Mamba; phase 4 = Transformer/xLSTM)
3. Multi-scala / tissue-aware Hilbert (phase 3)
4. Eval su tutto Patho-Bench (vs solo PANDA di MambaBack)

## Ambiente

```bash
conda activate hilbert-wsi-clean   # Python 3.11, torch 2.5.1+cu124, mamba-ssm 2.3.2
```

**CUDA fix**: l'env ha un activation hook in
`etc/conda/activate.d/cuda_ld_library_path.sh`
che prioritizza le lib pip-bundled CUDA 12.4 rispetto al sistema (CUDA 12.1/12.2 in ldconfig).
Non serve nessuna variabile manuale.

**mamba_ssm patch**: `triton.set_allocator` è stato patchato con `hasattr` guard in:
- `mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py:438`
- `mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py:1788`
- `mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py:231`

## Struttura repo

```
hilbert_wsi/
├── ordering/           # 7 scheme + registry (get_ordering(name))
├── models/
│   ├── mamba.py        # BiMamba backbone (skip_proj, pooling: mean/attn/cls)
│   ├── transformer.py  # RoPE+SDPA+SwiGLU backbone (pe_type: rope|none, skip_proj)
│   ├── xlstm.py        # mLSTM backbone
│   ├── gla.py          # GLA backbone
│   └── baselines/      # no-ordering + classical MIL (messo da parte)
│       ├── transmil.py, mambamil.py, clam.py, dsmil.py, twodmamba.py
│       └── vendor/     # upstream code pristino per provenance
├── encoder.py          # HilbertSequenceEncoder — 1D ordering → backbone
├── encoder_2d.py       # TileCoordEncoder — 2D sin/cos PE da coords → stesso backbone
├── pos_encoding.py     # TwoDSinCosPositionalEncoding (parameter-free)
└── multiscale.py       # placeholder phase 3
adapters/
└── patho_bench.py      # register_hilbert_encoders() — chiama prima di Patho-Bench
configs/
├── orderings/          # un yaml per scheme
├── backbones/          # configs core: mamba_base, transformer_base, mamba_2dpe, transformer_2dpe
├── baselines/          # configs messi da parte: abmil, clam, dsmil, transmil, mambamil, twodmamba
└── experiments/        # yaml esperimenti Patho-Bench
scripts/
├── run_benchmark.py      # CLI: scarica splits HF + lancia Patho-Bench
└── ablation_orderings.py # sweep tutti gli ordering, stesso backbone
tests/              # 38 test, tutti passano (incluso Mamba CUDA forward)
```

### Encoder pair per confronto 1D vs 2D (fairness)

| Encoder | Modello Patho-Bench | Positional info | Arch |
|---|---|---|---|
| `HilbertSequenceEncoder` | `hilbertwsi_<ord>_<bb>` | 1D ordering → RoPE (Transformer) / scan order (Mamba) | uguale |
| `TileCoordEncoder` | `hilbertwsi_2dpe_<bb>` | 2D sin/cos PE da (x,y) coords, no ordering | uguale |
| — | `hilbertwsi_random_<bb>` | 1D random order, no spatial info (controllo) | uguale |

Parametri: 2D encoder ha +0.263M overhead fisso (backbone.proj_in(512→512) non usato, documentato).

## Dati — /data/hilbert-wsi/

```
/data/hilbert-wsi/
├── features/
│   ├── CPTAC_UCEC/        # 95 slides, UNI2-h 1536-dim, H5 estratto
│   │   └── *.h5           # shape keys: features(1,N,1536), coords(1,N,2)
│   ├── PANDA/             # 32GB, estratto
│   ├── CPTAC_BRCA/        # presente ✅
│   ├── CPTAC_CCRCC/       # presente ✅
│   └── ...
├── splits/                # symlink -> /home/oem/HilbertWSI/splits
├── runs/                  # symlink -> /home/oem/HilbertWSI/runs
└── checkpoints/           # symlink -> /home/oem/HilbertWSI/checkpoints
```

Symlink nella root del progetto: `features/`, `splits/`, `runs/`, `checkpoints/`.

### H5 format da Patho-Bench / UNI2-h

Il file H5 ha `features: (1, N, D)` e `coords: (1, N, 2)` con leading dim=1 (slide batch).
Dopo `collate_fn` di Patho-Bench il sample arriva all'encoder come `(1, 1, N, D)`.
`HilbertSequenceEncoder.forward()` lo gestisce con reshape a `(B, N, D)`.

## Task splits scaricati

- `cptac_ucec/Immune_class` → `splits/cptac_ucec/Immune_class/` ✅
- `panda/isup_grade` → `splits/panda/isup_grade/` ✅
- `cptac_brca/Immune_class` → `splits/cptac_brca/Immune_class/` ✅
- `cptac_brca/PIK3CA_mutation` → `splits/cptac_brca/PIK3CA_mutation/` ✅
- `cptac_ccrcc/OS` → `splits/cptac_ccrcc/OS/` ✅
- `cptac_ccrcc/{BAP1,Immune_class,PBRM1,VHL}_mutation` → `splits/cptac_ccrcc/*/` ✅
- Da scaricare: `ebrains/*` (30-class, priorità alta)

Scarica splits con:
```python
from patho_bench.SplitFactory import SplitFactory
split_path, config_path = SplitFactory.from_hf(
    saveto='splits', source='cptac_ucec', task='Immune_class'
)
```

## Risultati — Contributo 1: Ordering vs No-ordering

> ### ⚠ Pipeline audit 2026-06-02 — alcuni risultati INVALIDATI (re-run in corso)
> Audit della pipeline ordering ha trovato 3 difetti (curva Hilbert verificata **corretta**):
> - **BUG A — 2D PE quasi-costante** (`pos_encoding.py`): norm per-slide [0,1] + freq max 1.0 →
>   ogni tile riceve PE quasi identico (cosine vicini/random ≥0.998, gap 0.0017). L'arm "2D PE" era
>   di fatto **senza posizione**. → **tutti i numeri "2D PE" qui sotto sono invalidi.** Fix: indici-griglia (gap 0.286).
> - **BUG B — truncation slab** (`encoder*.py`): `max_seq_len=2048` teneva i primi-N tile in ordine H5
>   (column-major) = uno **slab spaziale** (scarta ~44% tessuto), applicato **solo a CCRCC, non UCEC**.
>   → **il "reversal" UCEC→CCRCC e i numeri CCRCC sono confondati.** Fix: subsample uniforme; re-run con `bag_size`.
> - **BUG C — 2D-PE+Mamba non era no-ordering** (`encoder_2d.py`): scansionava in ordine raster H5 ≈ snake.
>   Fix: scan order randomizzato (posizione solo dal PE).
> Fix applicati + 100 test passano. Re-run corretti (8 sweep): UCEC 2dpe full-seq, CCRCC ×6 `bag_size=2048`.
> Dettagli: vedi `/home/oem/.claude/plans/cosmic-purring-tide.md`. Numeri non marcati = ancora validi
> (UCEC ordering-vs-random full-seq: l' unico confronto pulito → ordering aiuta su dati piccoli).

**Domanda**: stessa architettura, ordering SFC vs random/no-ordering → effetto causale?
**Metrica primaria**: macro-OVR-AUC ± SE (50-fold Patho-Bench, CI = 2·SE ≈ 95%).
**Legenda**: ⭐ best, ⚠ no-ordering control, 🔲 da eseguire

### CPTAC-UCEC Immune_class — 95 slides, task non saturo ✅

**Ablation ordering (stessa architettura)**

| Ordering | Backbone | AUC ± SE | acc ± SE | F1 ± SE |
|---|---|---|---|---|
| Snake ⭐ | Transformer (13M) | **0.629 ± 0.012** | 0.452 ± 0.013 | 0.437 ± 0.013 |
| Zorder | Transformer | 0.622 ± 0.013 | 0.448 ± 0.017 | 0.436 ± 0.016 |
| Hilbert | Transformer | 0.614 ± 0.010 | 0.444 ± 0.013 | 0.425 ± 0.013 |
| Peano | Transformer | 0.613 ± 0.012 | 0.445 ± 0.015 | 0.430 ± 0.015 |
| Moore | Transformer | 0.608 ± 0.011 | 0.435 ± 0.014 | 0.422 ± 0.014 |
| **Random ⚠** | Transformer | **0.519 ± 0.016** | 0.344 ± 0.012 | 0.269 ± 0.014 |
| Hilbert ⭐ | Mamba (22M) | **0.596 ± 0.011** | 0.405 ± 0.013 | 0.387 ± 0.014 |
| Peano | Mamba | 0.594 ± 0.012 | 0.401 ± 0.014 | 0.387 ± 0.014 |
| Similarity | Mamba | 0.592 ± 0.011 | 0.408 ± 0.014 | 0.391 ± 0.014 |
| Zorder | Mamba | 0.588 ± 0.011 | 0.398 ± 0.012 | 0.384 ± 0.013 |
| Snake | Mamba | 0.582 ± 0.011 | 0.400 ± 0.014 | 0.385 ± 0.014 |
| Moore | Mamba | 0.573 ± 0.011 | 0.375 ± 0.013 | 0.358 ± 0.013 |

**Confronti causali (CI disjoint = effetto significativo)**

| Confronto | Δ AUC | CI | Interpretazione |
|---|---|---|---|
| Snake+Transformer vs Random+Transformer | **+0.110** | **DISJOINT** | Ordering causa +0.11 AUC su Transformer |
| Hilbert+Mamba vs MambaMIL-noorder† | **+0.112** | **DISJOINT** | Ordering causa +0.11 AUC su Mamba |
| Snake+Transformer vs TransMIL-noorder† | **+0.087** | **DISJOINT** | Arch Transformer sola non spiega il gap |
| Hilbert+Mamba vs 2DMamba† | **+0.069** | **DISJOINT** | 2D-nativeness non bypassa ordering 1D |
| 2DMamba vs Random+Transformer | +0.008 | overlap | 2DMamba ≈ random ordering |

† Cross-arch comparison (architettura diversa): MambaMIL=Mamba2+attnpool (0.484), TransMIL=Nystromformer+PPEG (0.542), 2DMamba=2D-SSM (0.527, patch_size=1024).

**No-ordering controls per backbone architettura (fair 2D vs 1D — stessa arch) ✅**

**No-ordering controls per backbone architettura (fair 2D vs 1D — stessa arch) ✅ post-fix**

Rerun eseguito il 2026-05-30 dopo fix `pos_encoding.py` (commit `91bc137`). I valori pre-fix (0.545/0.524) sono stati scartati.

| Modello | Arch | AUC ± SE | CI [95%] | Run |
|---|---|---|---|---|
| Random+Transformer ⚠ | Transformer | 0.519 ± 0.016 | [0.487, 0.551] | `cptac_ucec_Immune_class_random_transformer` |
| **2D PE+Transformer** ⚠ | Transformer | **0.515 ± 0.014** | [0.487, 0.543] | `cptac_ucec_2dpe_rerun_transformer` |
| **2D PE+Mamba** ⚠ | Mamba | **0.542 ± 0.016** | [0.510, 0.574] | `cptac_ucec_2dpe_rerun_mamba` |

**Confronti causali completi — 1D ordering vs 2D PE vs random (UCEC)**

| Confronto | Δ AUC | CI | Interpretazione |
|---|---|---|---|
| Snake+Transformer vs Random+Transformer | **+0.110** | **DISJOINT** | Ordering causa +0.11 AUC su Transformer |
| Snake+Transformer vs 2D PE+Transformer | **+0.114** | **DISJOINT** | Ordering >> coordinate spaziali 2D |
| **2D PE+Transformer vs Random+Transformer** | −0.004 | **OVERLAP** | **2D PE = random: coords senza ordering inutili** |
| Hilbert+Mamba vs 2D PE+Mamba | **+0.054** | **borderline DISJOINT** | Ordering >> coordinate spaziali 2D (Mamba) |
| **2D PE+Mamba vs MambaMIL-noorder†** | +0.058 | **DISJOINT** | 2D PE+Mamba > no-order: separa dal cluster |
| Hilbert+Mamba vs MambaMIL-noorder† | **+0.112** | **DISJOINT** | Ordering causa +0.11 AUC su Mamba |

† Cross-arch: MambaMIL=Mamba2+attnpool (0.484), TransMIL=Nystromformer+PPEG (0.542), 2DMamba=2D-SSM (0.527).

**Cluster no-ordering (OVERLAP tra loro):** TransMIL (0.542), 2DMamba (0.527), 2D PE+Transformer (0.515), Random+Transformer (0.519), MambaMIL (0.484). 2D PE+Mamba (0.542) esce dal cluster vs MambaMIL (DISJOINT), borderline Hilbert+Mamba.

**Conclusioni UCEC (1D SFC, confermate):**
- Ordering SFC → +0.110 AUC vs random (Transformer), +0.112 vs no-order (Mamba): entrambi **CI DISJOINT** → effetto causale robusto
- Mamba senza ordering collassa ad AUC ≈ 0.48 (< ABMIL 0.594): gli SSM richiedono struttura spaziale esplicita
- Differenze tra ordering SFC: deboli (Δ ≤ 0.021 tra Transformer orderings), non significative singolarmente

**Conclusioni UCEC (2D PE, post-fix rerun):**
- 2D PE+Transformer (0.515) = random (0.519): Δ=−0.004, OVERLAP → coords senza ordering non informano Transformer
- 2D PE+Mamba (0.542) > MambaMIL no-order (0.484): DISJOINT → 2D PE fornisce segnale spaziale a Mamba
- Hilbert+Mamba (0.596) vs 2D PE+Mamba (0.542): Δ=+0.054, CI borderline DISJOINT → ordering ancora superiore

### CPTAC-BRCA Immune_class — 653 slides, task saturo ✅

| Ordering | Backbone | AUC ± SE |
|---|---|---|
| Hilbert | Mamba (22M) | 0.711 ± 0.013 |
| Snake | Transformer (13M) | 0.703 ± 0.011 |
| **Random ⚠** | Mamba | **0.713 ± 0.012** |
| **Random ⚠** | Transformer | **0.700 ± 0.013** |

**Conclusione**: Δ ≤ 0.003, CI sovrapposte → **BRCA saturo**, ordering irrilevante (>600 slides).

### PANDA ISUP grading — >400 slides, task saturo ✅

| Modello | AUC | weighted-κ |
|---|---|---|
| ABMIL | 0.963 | 0.944 |
| Hilbert+Mamba | 0.962 | 0.943 |
| Zorder+Mamba | 0.962 | 0.949 |

**Conclusione**: CI completamente sovrapposti → **PANDA saturo** con UNI2-h.

### CPTAC-CCRCC BAP1_mutation — ~300 slides, intermedio ✅

Dataset chiave per la curva dataset-size moderator (UCEC=95 slides → CCRCC=~300 → BRCA=653).

**Trio confronto: 2D PE vs 1D ordering vs random (runs post-fix, 2026-05-30)**

| Modello | Backbone | AUC ± SE | CI [95%] |
|---|---|---|---|
| **2D PE ⭐** | Transformer | **0.787 ± 0.017** | [0.753, 0.821] |
| **2D PE ⭐** | Mamba | **0.762 ± 0.023** | [0.716, 0.808] |
| Hilbert | Transformer | 0.720 ± 0.026 | [0.668, 0.772] |
| **Random ⚠** | Transformer | 0.696 ± 0.025 | [0.646, 0.746] |
| Hilbert | Mamba | 0.674 ± 0.028 | [0.618, 0.730] |
| **Random ⚠** | Mamba | 0.652 ± 0.024 | [0.604, 0.700] |

**Confronti causali CCRCC (CI disjoint = effetto significativo)**

| Confronto | Δ AUC | CI | Interpretazione |
|---|---|---|---|
| 2D PE+Transformer vs Random+Transformer | **+0.091** | **DISJOINT** | 2D PE causa +0.09 AUC su Transformer |
| 2D PE+Mamba vs Random+Mamba | **+0.110** | **DISJOINT** | 2D PE causa +0.11 AUC su Mamba |
| Hilbert+Transformer vs Random+Transformer | +0.024 | OVERLAP | Ordering ≈ random su CCRCC |
| Hilbert+Mamba vs Random+Mamba | +0.022 | OVERLAP | Ordering ≈ random su CCRCC |
| 2D PE+Transformer vs Hilbert+Transformer | +0.067 | OVERLAP | 2D PE non sig. > ordering |

**Conclusioni CCRCC (inversione rispetto a UCEC):**
- 2D PE >> random (DISJOINT): le coordinate spaziali assolute informano la predizione di mutazione
- 1D ordering ≈ random (OVERLAP): la struttura sequenziale non aggiunge valore su questo task
- **Reversal UCEC→CCRCC**: su UCEC (immune class, 95 slides) ordering >> 2D PE; su CCRCC (mutation, ~300 slides) 2D PE >> ordering
- Ipotesi: le mutazioni BAP1 sono localizzate spazialmente → 2D PE ne cattura la posizione; l'immune class richiede co-occurrence spaziale locale → ordering la codifica meglio

---

## Risultati — Contributo 2: HilbertWSI vs Baseline classiche MIL

**Domanda**: il modello 1D ordinato supera le SOTA MIL classiche su stesso feature extractor?
**Avvertenza**: baseline classiche 1-2M params vs HilbertWSI 13-22M — documentare in tabella.

### CPTAC-UCEC Immune_class — 95 slides ✅

| Modello | Tipo | AUC ± SE | Params |
|---|---|---|---|
| Snake + Transformer ⭐ | HilbertWSI (1D) | **0.629 ± 0.012** | 13M |
| Mean-pool linprobe | baseline | 0.601 ± 0.010 | ~0 |
| Hilbert + Mamba | HilbertWSI (1D) | 0.596 ± 0.011 | 22M |
| ABMIL | classico MIL | 0.594 ± 0.013 | 1.5M |
| CLAM-SB | classico MIL | 0.710 ± 0.012† | 1.1M |

† CLAM su BRCA, non UCEC — da eseguire su UCEC.

**Conclusione**: Snake+Transformer vs ABMIL Δ=+0.035, CI parzialmente sovrapposte → **borderline** su UCEC (95 slides). Non conclusivo da solo; servono CCRCC e altri dataset.

### CPTAC-BRCA Immune_class — 653 slides ✅

| Modello | Tipo | AUC ± SE | Params |
|---|---|---|---|
| MambaMIL (no order)† | baseline no-order | **0.725 ± 0.011** | 20.7M |
| Random+Mamba | HilbertWSI | 0.713 ± 0.012 | 22M |
| Hilbert+Mamba | HilbertWSI (1D) | 0.711 ± 0.013 | 22M |
| CLAM-SB | classico MIL | 0.710 ± 0.012 | 1.1M |
| Snake+Transformer | HilbertWSI (1D) | 0.703 ± 0.011 | 13M |
| ABMIL | classico MIL | 0.696 ± 0.012 | 1.5M |

† MambaMIL usa SRMamba (custom fork), non Mamba2 — architettura diversa da HilbertWSI Mamba.

**Conclusione**: su BRCA saturo, nessuna arch si distingue. CLAM-SB (1.1M) competitive con modelli 15-20x più grandi. Confronto non discriminante su dataset saturi.

**Note pipeline:**
- Splits ufficiali Patho-Bench (HuggingFace) con train/val/test pre-definiti: 50-fold, UCEC 60/15/19 per fold
- Class balancing: `ExperimentFactory.finetune(balanced=True)` → `CrossEntropyLoss(weight=compute_class_weight(...))`
- Random ordering fixed: `blake2b(coords) ^ base_seed` per-slide (non seed=0 globale)
- Peano ordering: canonica S-curve column-major (verificata test regressione 3×3, 9×9)

## Prossimi esperimenti

**Orderings nel registry**: hilbert, zorder, snake, moore, peano, random, similarity
**Backbones nel registry**: mamba, transformer, xlstm, gla

### Priorità 1 — Confronto 2D PE vs 1D ordering (stessa arch, fairness) ✅ COMPLETATO

Rerun post-fix eseguito 2026-05-30. Risultati: 2D PE+Transformer=0.515±0.014, 2D PE+Mamba=0.542±0.016.
Conclusione: 2D PE+Transformer ≈ random (OVERLAP); 2D PE+Mamba borderline < Hilbert+Mamba. Vedi sezione risultati UCEC.
Config 2D PE disponibili: `mamba_2dpe_base.yaml`, `transformer_2dpe_base.yaml`, `xlstm_2dpe_base.yaml`, `gla_2dpe_base.yaml`.

### Priorità 2 — CPTAC-CCRCC (dataset intermedio, ~300 slides) ✅ COMPLETATO

BAP1_mutation completato (2026-05-30). Risultato principale: 2D PE >> ordering (inversione rispetto UCEC). Vedi sezione risultati CCRCC.

### Priorità 3 — EBRAINS 30-class (da scaricare features prima) 🔲

30-class classification — da scaricare features (alto priority per generalizzazione).

### Priorità 4 — xLSTM e GLA su UCEC (contributo 2, SOTA NLP) 🔲

```bash
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --experiment_type finetune \
    --model_name hilbertwsi_snake_xlstm \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/cptac_ucec_snake_xlstm \
    --model_kwargs_yaml configs/backbones/xlstm_base.yaml \
    --num_epochs 20
```

## Nome modello (adapter)

| Pattern | Encoder | Config |
|---|---|---|
| `hilbertwsi_<ordering>_<backbone>` | `HilbertSequenceEncoder` (1D) | `configs/backbones/<backbone>_base.yaml` |
| `hilbertwsi_2dpe_<backbone>` | `TileCoordEncoder` (2D PE) | `configs/backbones/<backbone>_2dpe_base.yaml` |

Esempio: `hilbertwsi_hilbert_mamba`, `hilbertwsi_snake_transformer`, `hilbertwsi_2dpe_transformer`.
Registrato chiamando `adapters.patho_bench.register_hilbert_encoders()`.

Baseline (messe da parte, configs in `configs/baselines/`):
`abmil_baseline`, `transmil_baseline`, `mambamil_baseline`, `clam_sb_baseline`, `clam_mb_baseline`, `dsmil_baseline`, `twodmamba_baseline`.

## Note importanti

- **torch non è in pyproject.toml** — va installato manualmente con wheel CUDA matching prima di `pip install -e .`
- **HilbertSequenceEncoder** e **TileCoordEncoder** espongono `.embedding_dim` e `.precision` (richiesti da Trident/Patho-Bench Pooler)
- **input_dim=1536** per UNI2-h; modificare se si usa diverso extractor
- Patho-Bench `ExperimentFactory.linprobe()` accetta `patch_embeddings_dirs` (list) + `model_name` per pooling on-the-fly, oppure `pooled_embeddings_dir` per embedding pre-poolati
- **TileCoordEncoder** ha +0.263M overhead vs HilbertSequenceEncoder (backbone.proj_in 512→512 non usato, skip_proj=True); documentare in paper
- **MambaMIL baseline** usa Mamba2, non SRMamba (paper originale usa fork custom) — denominare "MambaMIL-style" in tabelle paper
- **CLAM baseline** manca SmoothTop-K instance clustering loss — denominare "CLAM-style attention" in tabelle

## Test

```bash
pytest tests/ -v   # 94 passed
```

## Roadmap

- **Phase 1** ✅: PANDA-ISUP ablation ordering (runs_v2/). Task saturo, no differenza tra orderings.
- **Phase 2** ✅: UCEC + BRCA ablation completa (ordering vs killer controls). Ordering contribuisce su dataset piccolo (UCEC), irrilevante su dataset grande (BRCA).
- **Phase 2b** ✅: 2D PE vs 1D ordering — rerun post-fix completato (2026-05-30). UCEC: 2D PE+Transformer≈random, 2D PE+Mamba borderline < Hilbert+Mamba. CCRCC: inversione, 2D PE >> ordering.
- **Phase 2c** ✅: fix codebase completato (4 bug critici); CCRCC BAP1 eseguito; rerun 2D PE UCEC eseguito; mask BiMamba e uniformità encoder 1D/2D fixati (2026-06-02).
- **Phase 3**: multi-scala / tissue-aware Hilbert (multiscale.py)
- **Phase 4**: xLSTM, GLA come backbone alternativi per contributo 2
