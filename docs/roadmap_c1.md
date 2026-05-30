# Roadmap: Conferma Contributo 1

**Claim**: L'ordinamento SFC causa un incremento causale di AUC rispetto a no-ordering/random, indipendentemente dall'architettura e dalla scala del dataset.

---

## Stato attuale

| Evidenza | Status | Forza |
|---|---|---|
| UCEC 95 slide — Δ=+0.110 Transformer, CI DISJOINT | ✅ | Forte |
| UCEC 95 slide — Δ=+0.112 Mamba, CI DISJOINT | ✅ | Forte |
| 2D PE nel cluster no-ordering su UCEC | ✅ | Forte |
| BRCA 653 slide — saturato, nessun effetto (atteso) | ✅ | Supporto |
| CCRCC ~300 slide — scala intermedia | ❌ MANCANTE | **Critico** |
| xLSTM/GLA — arch-agnostic check | ❌ MANCANTE | Importante |

**Vulnerabilità**: una sola scala non-satura (UCEC, 95 slide). Claim non generalizzabile senza CCRCC.

---

## Esperimenti da eseguire

### Step 1 — CCRCC BAP1_mutation: scala intermedia [BLOCCANTE]

Trio confronto (ordering vs 2D PE vs random) per ciascun backbone.

```bash
# GPU 1 — Transformer trio
for MODEL in hilbertwsi_hilbert_transformer hilbertwsi_2dpe_transformer hilbertwsi_random_transformer; do
  conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ccrcc --task BAP1_mutation \
    --model_name $MODEL \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_CCRCC \
    --splits_root splits \
    --saveto runs/cptac_ccrcc_BAP1_${MODEL} \
    --model_kwargs_yaml configs/backbones/transformer_base.yaml \
    --num_epochs 20
done

# GPU 2 — Mamba trio (in parallelo)
for MODEL in hilbertwsi_hilbert_mamba hilbertwsi_2dpe_mamba hilbertwsi_random_mamba; do
  conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ccrcc --task BAP1_mutation \
    --model_name $MODEL \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_CCRCC \
    --splits_root splits \
    --saveto runs/cptac_ccrcc_BAP1_${MODEL} \
    --model_kwargs_yaml configs/backbones/mamba_base.yaml \
    --num_epochs 20
done
```

**Tempo**: ~3h per backbone (6h totale, 3h con 2 GPU in parallelo).

---

### Step 2 — xLSTM su UCEC: arch-agnostic check [IMPORTANTE]

```bash
for MODEL in hilbertwsi_snake_xlstm hilbertwsi_random_xlstm; do
  conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --model_name $MODEL \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/cptac_ucec_Immune_class_${MODEL} \
    --model_kwargs_yaml configs/backbones/xlstm_base.yaml \
    --num_epochs 20
done
```

**Tempo**: ~2h. Eseguire in parallelo con Step 1 se GPU disponibile.

---

### Step 3 — GLA su UCEC [OPZIONALE]

```bash
for MODEL in hilbertwsi_snake_gla hilbertwsi_random_gla; do
  conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --model_name $MODEL \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/cptac_ucec_Immune_class_${MODEL} \
    --model_kwargs_yaml configs/backbones/gla_base.yaml \
    --num_epochs 20
done
```

**Tempo**: ~2h.

---

### Step 4 — Similarity+Transformer su UCEC [TABELLA]

Colma gap nella tabella ablation (Transformer manca similarity, Mamba ha 0.592).

```bash
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
  --source cptac_ucec --task Immune_class \
  --model_name hilbertwsi_similarity_transformer \
  --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
  --splits_root splits \
  --saveto runs/cptac_ucec_Immune_class_hilbertwsi_similarity_transformer \
  --model_kwargs_yaml configs/backbones/transformer_base.yaml \
  --num_epochs 20
```

**Tempo**: ~45min.

---

## Scheduling ottimale (2 GPU)

```
t=0h    GPU1: CCRCC Transformer trio    GPU2: xLSTM UCEC + similarity
t=3h    GPU1: CCRCC Mamba trio          GPU2: GLA UCEC (opzionale)
t=6h    Analisi risultati
```

**Wall-clock totale**: ~6h.

---

## Criteri di conferma

| Risultato CCRCC | Risultato xLSTM | Claim C1 |
|---|---|---|
| Hilbert >> random, CI DISJOINT | Snake >> random, CI DISJOINT | **Confermato forte**: 2 scale, 3 arch |
| Hilbert >> random, CI DISJOINT | CI overlap | **Confermato**: 2 scale, arch-agnostic debole |
| CI overlap (CCRCC saturo) | CI DISJOINT | **Parziale**: solo dataset piccoli, 3 arch |
| Entrambi CI overlap | — | C1 limitato a UCEC — rivedere claim |

CI check: `Δ AUC > 2*(SE_a + SE_b)` → DISJOINT → effetto causale.

---

## Verifica post-run

```bash
for dir in runs/cptac_ccrcc_BAP1_* runs/cptac_ucec_Immune_class_hilbertwsi_{snake,random}_{xlstm,gla}; do
  [ -f "$dir/test_metrics_summary.json" ] && \
  echo "$dir: $(python -c "import json; d=json.load(open('$dir/test_metrics_summary.json')); auc=d['macro-ovr-auc']; print(f'{auc[\"mean\"]:.3f} ± {auc[\"std_err\"]:.3f}')")"
done
```
