# Roadmap: Conferma Contributo 2

**Claim**: backbone 1D-native (Mamba, Transformer NLP-style, xLSTM, GLA) con ordering SFC competono o superano le architetture MIL standard della letteratura WSI (ABMIL, TransMIL, MambaMIL, CLAM-SB/MB, DSMIL, 2DMamba) **a parità di feature extractor** (UNI2-h, 1536-dim).

**Caveat capacity**: HilbertWSI 13–22M params vs baseline 1–2M. Documentare in tabella paper.

---

## Stato copertura baseline

| Dataset / Task | SFC HilbertWSI | ABMIL | TransMIL | MambaMIL | CLAM-SB | CLAM-MB | DSMIL | 2DMamba |
|---|---|---|---|---|---|---|---|---|
| CPTAC-UCEC / Immune_class | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CPTAC-CCRCC / BAP1_mutation | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| CPTAC-BRCA / Immune_class | ✅ | ✅ | ⚠️ unfinished | ✅ | ✅ | ✅ | ✅ | ❌ |
| PANDA / isup_grade | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Gap critici**:
- **CCRCC**: nessuna baseline → impossibile confronto C2 su scala intermedia (~783 slide)
- **BRCA**: TransMIL non terminato, 2DMamba mancante
- **PANDA**: solo ABMIL e Mamba → non confrontabile

---

## Step 1 — CCRCC BAP1_mutation: baseline [BLOCCANTE C2]

```bash
for MODEL in abmil_baseline transmil_baseline mambamil_baseline \
             clam_sb_baseline clam_mb_baseline dsmil_baseline twodmamba_baseline; do
  YAML_NAME="${MODEL%_baseline}"
  conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ccrcc --task BAP1_mutation \
    --experiment_type finetune \
    --model_name "$MODEL" \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_CCRCC \
    --splits_root splits \
    --saveto runs/cptac_ccrcc_BAP1_${MODEL} \
    --model_kwargs_yaml configs/baselines/${YAML_NAME}_base.yaml \
    --max_seq_len 2048 \
    --num_epochs 20
done
```

Tempo stimato: ~6h sequenziale su 1 GPU.

---

## Step 2 — BRCA Immune_class: gap baseline [IMPORTANTE C2]

```bash
# TransMIL (rerun completo)
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
  --source cptac_brca --task Immune_class \
  --experiment_type finetune \
  --model_name transmil_baseline \
  --patch_features_dir /data/hilbert-wsi/features/CPTAC_BRCA \
  --splits_root splits \
  --saveto runs/cptac_brca_Immune_class_transmil_baseline \
  --model_kwargs_yaml configs/baselines/transmil_base.yaml \
  --num_epochs 20

# 2DMamba BRCA
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
  --source cptac_brca --task Immune_class \
  --experiment_type finetune \
  --model_name twodmamba_baseline \
  --patch_features_dir /data/hilbert-wsi/features/CPTAC_BRCA \
  --splits_root splits \
  --saveto runs/cptac_brca_Immune_class_twodmamba_baseline \
  --model_kwargs_yaml configs/baselines/twodmamba_base.yaml \
  --max_seq_len 4096 \
  --num_epochs 20
```

Tempo: ~4h.

---

## Criterio di conferma C2

| Risultato | Interpretazione |
|---|---|
| HilbertWSI best su ≥2 dataset, CI DISJOINT vs media baseline | **C2 confermato forte** |
| HilbertWSI competitivo (OVERLAP) su ≥2 dataset | **C2 confermato debole** (confound capacity) |
| HilbertWSI peggiore di ABMIL su ≥2 dataset | C2 non regge — rivedere claim |

CI check: Δ AUC > 2·(SE_a + SE_b) → DISJOINT → significativo.

**Nota capacity**: ABMIL (1.5M) competitive con HilbertWSI (13–22M) su UCEC borderline. Interpretare sempre in funzione del N del dataset.

---

## Verifica post-run

```bash
for dir in runs/cptac_ccrcc_BAP1_* runs/cptac_brca_Immune_class_*; do
  jf="$dir/test_metrics_summary.json"
  [ -f "$jf" ] && \
  echo "$dir: $(python -c "
import json; d=json.load(open('$jf'))
auc=d.get('macro-ovr-auc', d.get('auc', {}))
print(f'{auc.get(\"mean\",\"?\"): .3f} ± {auc.get(\"std_err\",\"?\"): .3f}')
")"
done
```
