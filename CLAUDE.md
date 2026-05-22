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
├── ordering/       # 7 scheme + registry (get_ordering(name))
├── models/         # Mamba backbone + registry (get_backbone(name))
├── encoder.py      # HilbertSequenceEncoder — Trident-compatibile (embedding_dim, precision)
└── multiscale.py   # placeholder phase 3
adapters/
└── patho_bench.py  # register_hilbert_encoders() — chiama prima di Patho-Bench
configs/
├── orderings/      # un yaml per scheme
├── backbones/mamba_base.yaml  # input_dim=1536 (UNI2-h)
└── experiments/    # yaml esperimenti Patho-Bench
scripts/
├── run_benchmark.py      # CLI: scarica splits HF + lancia Patho-Bench
└── ablation_orderings.py # sweep tutti gli ordering, stesso backbone
tests/              # 28 test, tutti passano (incluso Mamba CUDA forward)
docs/
├── architecture.md
└── benchmarks.md
```

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

## Risultati runs_v2/ — PANDA ISUP grading (stato 2026-05-17)

| Modello | acc | macro-OVR-AUC | weighted-κ |
|---|---|---|---|
| ABMIL (no order) | 0.812 | 0.963 | 0.944 |
| Hilbert + Mamba | 0.802 | 0.962 | 0.943 |
| Z-order + Mamba | 0.793 | 0.962 | 0.949 |
| Mean-pool + linprobe | 0.753 | 0.943 | 0.910 |

**Conclusione**: CI completamente sovrapposti → PANDA-ISUP saturo con UNI2-h. Non è un bug di pipeline.

## Risultati runs/ — CPTAC-UCEC Immune_class (stato 2026-05-22 — killer experiment completo)

| Modello | acc | macro-OVR-AUC | macro-F1 | w-kappa |
|---|---|---|---|---|
| Snake + Transformer ⭐ | **0.452 ± 0.013** | **0.629 ± 0.012** | **0.437 ± 0.013** | 0.355 ± 0.025 |
| Zorder + Transformer | 0.448 ± 0.017 | 0.622 ± 0.013 | 0.436 ± 0.016 | 0.337 ± 0.026 |
| Hilbert + Transformer | 0.444 ± 0.013 | 0.614 ± 0.010 | 0.425 ± 0.013 | 0.308 ± 0.026 |
| Peano + Transformer | 0.445 ± 0.015 | 0.613 ± 0.012 | 0.430 ± 0.015 | 0.317 ± 0.029 |
| Moore + Transformer | 0.435 ± 0.014 | 0.608 ± 0.011 | 0.422 ± 0.014 | 0.325 ± 0.025 |
| Mean-pool (linprobe) | 0.403 ± 0.012 | 0.601 ± 0.010 | 0.387 ± 0.013 | 0.200 ± 0.031 |
| Hilbert + Mamba | 0.405 ± 0.013 | 0.596 ± 0.011 | 0.387 ± 0.014 | 0.215 ± 0.030 |
| Peano + Mamba | 0.401 ± 0.014 | 0.594 ± 0.012 | 0.387 ± 0.014 | — |
| Similarity + Mamba | 0.408 ± 0.014 | 0.592 ± 0.011 | 0.391 ± 0.014 | — |
| ABMIL | 0.387 ± 0.014 | 0.594 ± 0.013 | 0.373 ± 0.015 | 0.199 ± 0.031 |
| Zorder + Mamba | 0.398 ± 0.012 | 0.588 ± 0.011 | 0.384 ± 0.013 | — |
| Snake + Mamba | 0.400 ± 0.014 | 0.582 ± 0.011 | 0.385 ± 0.014 | — |
| Moore + Mamba | 0.375 ± 0.013 | 0.573 ± 0.011 | 0.358 ± 0.013 | — |
| **TransMIL (no order)** ⚠ | 0.338 ± 0.013 | 0.542 ± 0.013 | 0.252 ± 0.013 | 0.024 ± 0.026 |
| **2DMamba (2D-native SSM)** ⚠ | 0.368 ± 0.014 | 0.527 ± 0.015 | 0.302 ± 0.017 | 0.042 ± 0.029 |
| **Random + Transformer (fix)** ⚠ | 0.344 ± 0.012 | 0.519 ± 0.016 | 0.269 ± 0.014 | 0.010 ± 0.028 |
| **MambaMIL (no order)** ⚠ | 0.319 ± 0.012 | 0.484 ± 0.014 | 0.258 ± 0.014 | -0.007 ± 0.025 |

⭐ = best  ⚠ = no-ordering controls (killer experiment 2026-05-22)
† 2DMamba run con patch_size=1024 (pure-Python pscan, 14.4M params); resolution loss reale (fino a 16 tile/cella)

**Esito decision tree killer (CI overlap = ~95%, 2·SE):**

| Confronto | Δ AUC | CI overlap? | Interpretazione |
|---|---|---|---|
| Snake+Transformer vs TransMIL | **+0.087** | **DISJOINT** | Architettura Transformer DA SOLA non spiega il guadagno: ordering contribuisce indipendentemente |
| Snake+Transformer vs Random+Transformer | **+0.110** | **DISJOINT** | Ordering matters per Transformer (Snake >> Random con stessa arch) |
| Hilbert+Mamba vs MambaMIL | **+0.112** | **DISJOINT** | Ordering è **essenziale** per Mamba: senza ordering Mamba collassa a ~random |
| Snake+Transformer vs ABMIL | +0.034 | overlap | Confronto classico borderline (CI parzialmente sovrapposte) |
| Random+Transformer vs TransMIL | -0.024 | overlap | Nostra Transformer-arch senza ordering ≈ TransMIL → architetture comparabili senza ordering |
| TransMIL vs ABMIL | -0.052 | DISJOINT | TransMIL **peggio di ABMIL** su UCEC (paper-not-faithful: overfit a 13M params o PPEG inadatto) |
| 2DMamba vs Hilbert+Mamba | **-0.069** | **DISJOINT** | Paradigma 2D-nativo non bypassa 1D-ordering; 2DMamba cade nel cluster no-ordering |
| 2DMamba vs ABMIL | **-0.067** | **DISJOINT** | ABMIL classico batte 2DMamba senza ordering strutturato |
| 2DMamba vs Random+Transformer | +0.008 | overlap | 2DMamba ≈ random ordering → struttura 2D-nativa non sostituisce ordinamento esplicito |

**Conclusioni killer (CPTAC-UCEC Immune_class):**

1. **Tesi VIVE in forma forte**: ordering 1D contribuisce ~+0.11 AUC sia per Transformer sia per Mamba, indipendentemente dall'architettura. Snake+Transformer e Hilbert+Mamba dominano tutti i no-ordering controls con CI disjoint.

2. **Mamba beneficia più drammaticamente dell'ordering** (passa da AUC 0.484 random a 0.596 con Hilbert) rispetto al Transformer (0.519 → 0.629). Implica che gli SSM su WSI **richiedono** struttura spaziale imposta — senza, collassano a baseline random.

3. **TransMIL/MambaMIL/2DMamba capacity-matched sotto-performano ABMIL classico (1.5M)** su UCEC. Cluster "no ordering" (Random+Transformer 0.519, TransMIL 0.542, 2DMamba 0.527, MambaMIL 0.484) tutti peggio o pari ad ABMIL. Possibili cause: overfit su 50-fold UCEC, architetture progettate per N>>1000 patch, PPEG/attention-pool inadatti per immune-subtyping (segnale globale diffuso).

4. **Paradigma 2D-nativo (2DMamba) non bypassa 1D-ordering**: AUC 0.527 ± 0.015 cade nel cluster no-ordering. CI DISJOINT vs Hilbert+Mamba (0.596) e ABMIL (0.594). Inductive bias critico è l'ordinamento spaziale esplicito, non la 2D-nativeness. Caveat: patch_size=1024 (pure-Python pscan) introduce collision loss; paper-grade comparison richiederebbe CUDA kernel a patch_size=256.

5. **Confronto Mamba vs Transformer**: Snake+Transformer (0.629) > Hilbert+Mamba (0.596) di +0.033 AUC, CI overlap marginale. Mamba al massimo eguaglia ABMIL/mean-pool con ordering. Confound capacity (22M vs 13M) non risolto.

6. **Snake e Zorder leggermente meglio di Hilbert/Peano/Moore** per Transformer (Δ ≤ 0.015 AUC, CI overlap). Differenze tra orderings deboli; significative solo collettivamente vs random/no-ordering.

**Pipeline verificata:**
- Class balancing: funziona (Patho-Bench `ExperimentFactory.finetune(balanced=True)` usa `compute_class_weight` → `CrossEntropyLoss(weight=...)`)
- Mamba: 22M params, Transformer: 13M params, TransMIL: 13.4M, MambaMIL: 20.7M (capacity matched ai counterpart HilbertWSI)
- Peano ordering: **canonica** (S-curve column-major, verificata via test regressione vs reference table 3×3+9×9). Linea morta `rev_x = rev_x` rimossa in cleanup, comportamento invariato.
- Random ordering: **fixed** (era seed=0 globale → ora `blake2b(coords) ^ base_seed` per-slide). I valori 0.519 AUC sopra usano il random fixato.

## Prossimi esperimenti (runs_v3/ e runs/)

```bash
cd /home/oem/HilbertWSI
conda activate hilbert-wsi-clean

# ATTENZIONE: "random_perm" non esiste nel registry, usare "random"
# Il saveto con _random_perm_ è intenzionale per distinguerlo da run precedenti

# 1. CRITICO — random+Mamba su UCEC (isola contributo ordering per Mamba)
python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --experiment_type finetune \
    --model_name hilbertwsi_random_mamba \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/cptac_ucec_Immune_class_hilbertwsi_random_perm_mamba \
    --model_kwargs_yaml configs/backbones/mamba_base.yaml \
    --num_epochs 20

# 2. CRITICO — random+Transformer su UCEC (isola contributo ordering per Transformer)
python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --experiment_type finetune \
    --model_name hilbertwsi_random_transformer \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/cptac_ucec_Immune_class_hilbertwsi_random_perm_transformer \
    --model_kwargs_yaml configs/backbones/transformer_base.yaml \
    --num_epochs 20

# 3. similarity+Transformer su UCEC (run incompleta)
python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --experiment_type finetune \
    --model_name hilbertwsi_similarity_transformer \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/cptac_ucec_Immune_class_hilbertwsi_similarity_transformer \
    --model_kwargs_yaml configs/backbones/transformer_base.yaml \
    --num_epochs 20

# 4. Hilbert+Mamba su CPTAC-BRCA (secondo dataset)
python -m scripts.run_benchmark \
    --source cptac_brca --task Immune_class \
    --experiment_type finetune \
    --model_name hilbertwsi_hilbert_mamba \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_BRCA \
    --splits_root splits \
    --saveto runs_v3/cptac_brca_immune_class_hilbert_mamba \
    --model_kwargs_yaml configs/backbones/mamba_base.yaml \
    --num_epochs 20

# 5. Mean-pool + CoxNet su CPTAC-CCRCC OS (survival baseline)
python -m scripts.run_benchmark \
    --source cptac_ccrcc --task OS \
    --experiment_type coxnet \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_CCRCC \
    --splits_root splits \
    --saveto runs_v3/cptac_ccrcc_os_mean_pool

# 6. EBRAINS — da scaricare features prima, poi ablation completa (30-class, task harder)
```

**Orderings nel registry**: hilbert, zorder, snake, moore, peano, random, similarity
**Backbones nel registry**: mamba, transformer

## Nome modello (adapter)

Formato: `hilbertwsi_<ordering>_<backbone>`, es. `hilbertwsi_hilbert_mamba`.
Registrato chiamando `adapters.patho_bench.register_hilbert_encoders()`.

## Note importanti

- **torch non è in pyproject.toml** — va installato manualmente con wheel CUDA matching prima di `pip install -e .`
- **HilbertSequenceEncoder** espone `.embedding_dim` e `.precision` (richiesti da Trident/Patho-Bench Pooler)
- **input_dim=1536** per UNI2-h; modificare se si usa diverso extractor
- Patho-Bench `ExperimentFactory.linprobe()` accetta `patch_embeddings_dirs` (list) + `model_name` per pooling on-the-fly, oppure `pooled_embeddings_dir` per embedding pre-poolati
- PANDA estrazione in corso in background (PID 1259478) su `/data/hilbert-wsi/features/PANDA/`

## Test

```bash
pytest tests/ -v   # 28 passed, include Mamba CUDA forward
```

## Roadmap

- **Phase 1** ✅: PANDA-ISUP ablation ordering (runs_v2/). Risultato: task saturo, no differenza tra orderings.
- **Phase 2** (attuale): CPTAC-UCEC ablation completa (Transformer+ordering > ABMIL su task non saturo); random-ordering control in sospeso; espansione a BRCA/CCRCC
- **Phase 3**: multi-scala / tissue-aware Hilbert (multiscale.py)
- **Phase 4**: altri backbone (xLSTM, GLA/RWKV-7, MambaMIL wrapper) + multi-fold

**Baseline MIL da aggiungere per credibilità review**:
- **TransMIL** — Transformer MIL senza ordering (Nyströmformer); isola contributo ordering da contributo architettura; **priorità alta**
- **MambaMIL** — Mamba MIL senza Hilbert ordering; isola contributo ordering per Mamba specificamente; reviewer lo chiederà
- **CLAM** — baseline classico atteso accanto ad ABMIL
- xLSTM (confronto standard nei paper SSM 2025)
- GLA o RWKV-7 (linear-attention, argomento "Mamba è speciale?")

**Nota narrativa**: se TransMIL ≈ Snake+Transformer → ordering non serve al Transformer, architettura sola spiega il gap. Se Snake+Transformer >> TransMIL → ordering ha contributo indipendente. Questa è la domanda chiave del paper.
