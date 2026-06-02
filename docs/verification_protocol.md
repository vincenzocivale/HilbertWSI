# Verification Protocol — SFC ordering vs. MIL classico

_Last updated: 2026-05-31. Companion to [`docs/architecture.md`](architecture.md), [`docs/results_summary.md`](results_summary.md)._

Questo documento risponde alla domanda:

> **Riordinare gli embedding di tile WSI lungo una space-filling curve (SFC) 1D,
> e processarli con Transformer/Mamba, porta a un vantaggio misurabile rispetto
> ad applicare Transformer/Mamba MIL classici (ABMIL, TransMIL, MambaMIL,
> CLAM, DSMIL, 2DMamba) sulle stesse feature?**

Definisce: claim falsificabile, confound noti, esperimenti controllati per
isolare ciascun confound, comandi riproducibili, decision tree per
interpretare i risultati.

---

## 1. Claim in forma falsificabile

**C1 (ordering puro)**: a parità di backbone e di pooling head, riordinare le
tile lungo una SFC produce un'AUC superiore rispetto a un ordering casuale
con CI al 95% disgiunti, su task non saturi (≤ ~300 slide).

**C2 (1D-native vs MIL classico)**: a parità di pooling head e di feature
extractor, un backbone 1D-native con ordering SFC è competitivo o superiore
ai modelli MIL classici (ABMIL/TransMIL/MambaMIL/CLAM/DSMIL/2DMamba) su task
non saturi.

I due claim sono distinti: **C1 può essere vero ma C2 falso** se la pool head
delle baseline compensa la mancanza di struttura spaziale; **C1 può essere
falso ma C2 vero** se HilbertWSI vince per pure ragioni di capacity (13–22M
vs. baseline 1–2M).

---

## 2. Confound identificati

L'audit del codice (commit `91bc137` e analisi successiva, 2026-05-31) ha
isolato i seguenti confound che possono falsificare l'inferenza causale se
non controllati:

| Confound | Dettaglio | Mitigazione |
|---|---|---|
| **Pooling head** | HilbertWSI configs usano `pooling: mean` (`*_base.yaml`); le baseline MIL usano tutte attention/CLS pool | Esperimento A3: rerun HilbertWSI con `pooling: attn` |
| **Capacity** | HilbertWSI 13–22M vs ABMIL/CLAM 1.1–1.8M | Documentare; eventualmente Mamba/Transformer depth-matched a 1–2M |
| **PE strategy** | 1D ordering → RoPE su seq index; 2D PE → sin/cos da `(x,y)`. Diverso inductive bias | Esperimento A2: confronto diretto a parità di backbone+pool |
| **Truncation bias** | `max_seq_len` tronca i primi N tile in H5 raster order; bias spaziale verso un quadrante | Default `max_seq_len=None` per non triggerare. Se attivo, applicato simmetricamente a 1D e 2D encoder (fix `91bc137` §8.4) |
| **Dataset saturation** | Task come PANDA-ISUP o BRCA-Immune saturi con UNI2-h → tutti i metodi convergono | Replicare comparisons su scale diverse: 95 (UCEC), ~300 (CCRCC), 653 (BRCA) |
| **Bi vs uni-direzionale Mamba** | HilbertWSI Mamba usa BiMamba2; MambaMIL usa uni-Mamba2 | Documentato. Non eliminabile senza ablation aggiuntivo |
| **Similarity ordering content-dependent** | `SimilarityOrdering` parte da `argmax(feature.norm)` → start dipende dal contenuto | Escludere `similarity` dalla riga "SFC pura" della tabella; tenere come ablation separato |

---

## 3. Esperimenti controllati

Quattro famiglie di esperimenti. Ciascuna isola un fattore.

### A1 — SFC vs. random, stessa architettura, stessa pool

**Isola**: effetto dell'ordering spaziale.
**Esito atteso (pre-registered)**: DISJOINT CI → ordering è causale.

```bash
# UCEC, Transformer (ripetere per ogni backbone in {mamba, xlstm, gla})
for ORD in snake hilbert zorder peano moore random; do
  conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --experiment_type finetune \
    --model_name hilbertwsi_${ORD}_transformer \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/c1_a1_ucec_${ORD}_transformer \
    --model_kwargs_yaml configs/backbones/transformer_base.yaml \
    --num_epochs 20 --balanced --gpu 0
done
```

Confronto chiave (per ciascun backbone):

```
Δ AUC = AUC(snake) − AUC(random)     # o hilbert vs random, scegliere il best
CI:    [Δ − 2·(SE_s + SE_r), Δ + 2·(SE_s + SE_r)]
DISJOINT se Δ > 2·(SE_s + SE_r) → C1 confermato per quel backbone.
```

### A2 — SFC vs. 2D PE, stessa architettura, stessa pool

**Isola**: 1D ordering vs. coordinate spaziali continue come segnale
posizionale.
**Esito atteso**: a priori nessun vincitore. Risultato informa il _meccanismo_
del segnale SFC (è importante l'ordine sequenziale o basta la posizione
codificata?).

```bash
for ENC in hilbertwsi_snake hilbertwsi_2dpe; do
  for BB in mamba transformer; do
    CFG=configs/backbones/${BB}_base.yaml
    [ "${ENC}" = "hilbertwsi_2dpe" ] && CFG=configs/backbones/${BB}_2dpe_base.yaml
    conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
      --source cptac_ucec --task Immune_class \
      --experiment_type finetune \
      --model_name ${ENC}_${BB} \
      --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
      --splits_root splits \
      --saveto runs/c1_a2_ucec_${ENC}_${BB} \
      --model_kwargs_yaml ${CFG} \
      --num_epochs 20 --balanced --gpu 0
  done
done
```

> ⚠ Nota: i risultati 2D PE in `results_summary.md` antecedenti al commit
> `91bc137` (2026-05-30) sono **invalidi** per via di un cast `pe.to(int64)`
> che annullava il PE. Vanno rilanciati. CLAUDE.md li marca con 🔁.

### A3 — HilbertWSI SFC (pool=attn) vs. MIL baselines (pool=attn) — **risposta vera alla domanda dell'utente**

**Isola**: SFC ordering vs. classico MIL, **a parità di pooling head**.
Rimuove il confound principale identificato dall'audit.

**Prerequisito**: serve `configs/backbones/<bb>_attnpool_base.yaml` per
ciascun backbone. Solo `mamba_attnpool_base.yaml` esiste; vanno aggiunti
gli omologhi per `transformer`, `xlstm`, `gla`. Setup minimale (copiare
`*_base.yaml` e cambiare `pooling: mean` → `pooling: attn`).

```bash
# Per Mamba (config già esistente)
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
  --source cptac_ucec --task Immune_class \
  --experiment_type finetune \
  --model_name hilbertwsi_hilbert_mamba \
  --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
  --splits_root splits \
  --saveto runs/c2_a3_ucec_hilbert_mamba_attnpool \
  --model_kwargs_yaml configs/backbones/mamba_attnpool_base.yaml \
  --num_epochs 20 --balanced --gpu 0

# Baseline da confrontare (ABMIL e MambaMIL hanno entrambe pool=attn)
for BL in abmil_baseline mambamil_baseline clam_sb_baseline dsmil_baseline; do
  CFG=configs/baselines/${BL%_baseline}_base.yaml
  conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ucec --task Immune_class \
    --experiment_type finetune \
    --model_name ${BL} \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto runs/c2_a3_ucec_${BL} \
    --model_kwargs_yaml ${CFG} \
    --num_epochs 20 --balanced --gpu 0
done
```

Confronto:

| Pair | Δ AUC interpretato come |
|---|---|
| `hilbertwsi_hilbert_mamba` (attn) vs `mambamil_baseline` | SFC vs no-ordering, **a parità di pool** e arch ≈ uguale (entrambi Mamba2 + attn). Differenze residue: bi vs uni, depth=4 vs 12. |
| `hilbertwsi_hilbert_mamba` (attn) vs `abmil_baseline` | SFC + Mamba sequence model vs pure attention pool sopra UNI2-h. |
| `hilbertwsi_hilbert_mamba` (attn) vs `clam_sb_baseline` | SFC + Mamba vs gated attention. Capacity gap esplicito (22M vs 1.1M). |
| `hilbertwsi_snake_transformer` (attn — config da creare) vs `transmil_baseline` | Stesso confronto per Transformer. |

### A4 — Scaling con dataset size

**Isola**: dataset saturation moderator.
**Esito atteso**: l'effetto SFC dovrebbe ridursi all'aumentare di N.

Ripetere A1 e A3 su:

| Dataset | N slides | Task | Saturazione attesa |
|---|---|---|---|
| CPTAC-UCEC | 95 | Immune_class | non-saturo (regime utile) |
| CPTAC-CCRCC | ~300 | BAP1_mutation o altre mutazioni | intermedio (chiave per scalability) |
| CPTAC-BRCA | 653 | Immune_class | saturo (controllo negativo) |

Comandi: identici a A1/A3 sostituendo `--source` e `--task`.
Per CCRCC con slide grandi (~10k tile mediana) usare `--max_seq_len 2048`.

---

## 4. Test statistico

**2·SE CI overlap test** (Patho-Bench restituisce mean ± SE su 50 fold).

Per due modelli con AUC `μ_a ± SE_a` e `μ_b ± SE_b`:

```
Δ = μ_a − μ_b
half-width = 2 · (SE_a + SE_b)         # conservativo, ~95% CI
DISJOINT se |Δ| > half-width            # effetto causale
OVERLAP  se |Δ| ≤ half-width            # non distinguibile
```

Per K confronti, applicare correzione Bonferroni: usare `2.5·(SE_a + SE_b)`
con K=10. Tutti i confronti riportati in `results_summary.md` usano questo
test.

---

## 5. Decision tree finale

Dato l'output completo degli esperimenti A1–A4:

```
   ┌─ A1 DISJOINT? ───── no ──► C1 falsificato.
   │                            SFC non distinguibile da random.
   │                            → Pubblicare come risultato negativo.
   │                            → Verificare con altri dataset prima di abbandonare.
   yes
   │
   ▼
   ┌─ A3 DISJOINT in favore HilbertWSI? ─ yes ─► C2 confermato.
   │                                              SFC dà vantaggio reale,
   │                                              indipendente da pool head.
   no
   │
   ▼
   ┌─ A3 OVERLAP? ──── yes ───► C1 vero ma C2 debole.
   │                            "SFC ≈ attention pool":
   │                            l'ordering compensa la mancanza di pool
   │                            sofisticato ma non aggiunge oltre.
   │                            Riformulare contributo come:
   │                            "SFC è un'alternativa valida al pool
   │                             di attenzione, non un'aggiunta".
   no
   │
   ▼
   A3 DISJOINT in favore baseline → C2 falsificato.
   Le baseline MIL battono HilbertWSI anche a parità di pool.
   Possibili spiegazioni: capacity gap inverso, bias architettura
   (PPEG in TransMIL, SRMamba in MambaMIL paper), task troppo piccolo.
   → Rivedere claim C2.
```

A4 modera ciascun ramo: l'evidenza è forte se vale **su ≥ 2 dataset di
scala diversa**, debole se vale solo su UCEC.

---

## 6. Stato attuale rispetto al protocollo

Dato lo snapshot dei risultati in `docs/results_summary.md` (2026-05-22):

| Esperimento | Stato | Note |
|---|---|---|
| A1 — UCEC, Transformer (snake vs random) | ✅ Δ=+0.110, DISJOINT | C1 confermato su Transformer |
| A1 — UCEC, Mamba (hilbert vs random) | ⚠ confronto è vs **MambaMIL** (Δ=+0.112), non vs `hilbertwsi_random_mamba` | Manca run `hilbertwsi_random_mamba` (UCEC) — A1 puro non eseguito per Mamba |
| A1 — UCEC, xLSTM, GLA | ❌ Non eseguito | Vedi `docs/roadmap_c1.md` step 2/3 |
| A1 — CCRCC | ❌ Non eseguito (dataset chiave) | Vedi `docs/roadmap_c1.md` step 1 |
| A2 — UCEC | ⚠ Pre-bug `91bc137` (valori invalidi) | 🔁 Rerun richiesto |
| A2 — CCRCC, BRCA | ❌ Non eseguito | Vedi `docs/roadmap_c1.md` |
| **A3 — UCEC, Mamba+attn vs baselines** | ❌ Non eseguito | Centrale per la domanda dell'utente |
| **A3 — UCEC, Transformer+attn vs baselines** | ❌ Non eseguito | Manca `transformer_attnpool_base.yaml` |
| A4 — CCRCC | ❌ Non eseguito | Bloccante per scaling claim |

**Conclusione provvisoria** basata solo sui risultati attualmente
disponibili: C1 ha evidenza solida su UCEC per Transformer; C2 è
**indeterminato** per via del confound pool (HilbertWSI mean vs baseline
attn). Per chiudere la domanda, eseguire **A3 (Mamba)** come prima azione
— config già pronta in `configs/backbones/mamba_attnpool_base.yaml`.

---

## 7. Limitazioni note

1. **Capacity gap**: HilbertWSI 13–22M vs CLAM/DSMIL 1.1M. Anche se C2 fosse
   confermato a parità di pool, resta il confound capacity. Per chiusura
   completa servirebbe HilbertWSI depth-matched a ~2M. Out of scope qui.
2. **BiMamba vs uni-Mamba**: HilbertWSI Mamba scansiona avanti+indietro,
   MambaMIL solo avanti. Questo dà a HilbertWSI un vantaggio strutturale
   nel confronto Mamba diretto. Documentare; eventualmente ablation
   "uni-Mamba + Hilbert".
3. **Truncation in H5 order**: con `--max_seq_len 2048` HilbertWSI e
   `hilbertwsi_2dpe` tronchino i primi 2048 tile (raster H5 — bias verso un
   quadrante). Le baseline MIL **non** tronchino — se `max_seq_len` è
   passato a una baseline, scatena `TypeError` al costruttore. Documentare
   questa asimmetria nei comandi A3 (non passare `--max_seq_len` agli
   esperimenti A3 che includono baselines).
4. **Coverage cross-dataset**: tutti i numeri esistenti vengono da UCEC.
   Cross-dataset evidence è il prossimo step (vedi `docs/roadmap_c1.md`,
   `docs/roadmap_c2.md`).
5. **Test statistico**: 2·SE è approssimazione di 95% CI valida sotto
   normalità di `μ`. Con 50 fold il CLT è soddisfatto. Per task con
   varianza alta tra fold (es. CCRCC con folding non bilanciato) preferire
   bootstrap.
6. **Similarity ordering** è content-dependent: la riga "similarity" nelle
   ablation va interpretata come "ordering condizionato dalle feature" e
   non confusa con un puro segnale spaziale.

---

## 8. Quick reference — comandi per eseguire l'esperimento A3 minimo

Il singolo esperimento più importante per rispondere alla domanda
dell'utente: **HilbertWSI Mamba con attn pool vs MambaMIL su UCEC**
(stesso backbone, stesso pool, l'unica differenza è l'ordinamento).

```bash
# Step 1: HilbertWSI Hilbert + Mamba + attn pool
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
  --source cptac_ucec --task Immune_class \
  --experiment_type finetune \
  --model_name hilbertwsi_hilbert_mamba \
  --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
  --splits_root splits \
  --saveto runs/a3_ucec_hilbert_mamba_attn \
  --model_kwargs_yaml configs/backbones/mamba_attnpool_base.yaml \
  --num_epochs 20 --balanced --gpu 0

# Step 2: MambaMIL (già usa attn pool nativamente)
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
  --source cptac_ucec --task Immune_class \
  --experiment_type finetune \
  --model_name mambamil_baseline \
  --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
  --splits_root splits \
  --saveto runs/a3_ucec_mambamil \
  --model_kwargs_yaml configs/baselines/mambamil_base.yaml \
  --num_epochs 20 --balanced --gpu 0

# Step 3: Leggere AUC ± SE da entrambi i runs
for d in runs/a3_ucec_hilbert_mamba_attn runs/a3_ucec_mambamil; do
  python -c "
import json
m = json.load(open('$d/test_metrics_summary.json'))['macro-ovr-auc']
print(f'$d: AUC = {m[\"mean\"]:.3f} ± {m[\"std_err\"]:.3f}')
"
done

# Step 4: Test 2·SE CI overlap
python -c "
auc_h, se_h = 0.000, 0.000   # sostituire con valori da Step 3
auc_m, se_m = 0.000, 0.000
delta = auc_h - auc_m
hw = 2 * (se_h + se_m)
print(f'Δ = {delta:+.3f}, ±{hw:.3f} → {\"DISJOINT\" if abs(delta) > hw else \"OVERLAP\"}')
"
```

**Decisione**:
- Δ > +2·(SE_h + SE_m) → C2 confermato per Mamba: SFC ordering aggiunge
  segnale anche a parità di pool head.
- Δ < −2·(SE_h + SE_m) → MambaMIL vince anche a parità di ordering→pool:
  C2 falsificato per Mamba.
- |Δ| ≤ 2·(SE_h + SE_m) → SFC ≈ MambaMIL: ordering non aggiunge oltre il
  pool head.

Tempo stimato: ~2h (40min/run × 2 + cleanup).
