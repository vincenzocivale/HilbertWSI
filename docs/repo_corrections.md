# Repo Corrections — Allineamento alla tesi a 2 contributi

_Generato: 2026-05-30. Da rivedere dopo ogni nuovo round di esperimenti._

## Contesto

La repo deve sostenere **due claim sperimentali**:

- **C1 — Ordering**: riordinare gli embedding di tile WSI lungo una space-filling curve (SFC, 1D) migliora rispetto a trattare le tile come patch di un'immagine 2D, **a parità di architettura**.
- **C2 — Architettura**: backbone progettati per dati 1D (Mamba, Transformer NLP-style, xLSTM, GLA) competono o superano le architetture MIL standard della letteratura WSI (ABMIL, TransMIL, MambaMIL, CLAM-SB/MB, DSMIL, 2DMamba) **a parità di feature extractor**.

Le correzioni qui sotto allineano il codice, le config, gli script e la documentazione a questi due claim. Sono raggruppate per **priorità** e per **contributo coperto**.

---

## 1. Stato evidenze (snapshot)

### C1 — copertura ordering vs 2D PE (stessa arch)

| Dataset / Task | SFC×Mamba | SFC×Transformer | 2DPE×Mamba | 2DPE×Transformer | SFC×xLSTM | SFC×GLA | 2DPE×xLSTM | 2DPE×GLA |
|---|---|---|---|---|---|---|---|---|
| CPTAC-UCEC / Immune_class | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| CPTAC-CCRCC / BAP1_mutation | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| CPTAC-BRCA / Immune_class | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| PANDA / isup_grade | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

### C2 — copertura SFC vs baseline MIL

| Dataset / Task | SFC HilbertWSI | ABMIL | TransMIL | MambaMIL | CLAM-SB | CLAM-MB | DSMIL | 2DMamba |
|---|---|---|---|---|---|---|---|---|
| CPTAC-UCEC / Immune_class | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CPTAC-CCRCC / BAP1_mutation | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| CPTAC-BRCA / Immune_class | ✅ | ✅ | ⚠️ unfinished | ✅ | ✅ | ✅ | ✅ | ❌ |
| PANDA / isup_grade | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Gap critici:**
- **C1**: BRCA non ha alcun run 2D PE → impossibile testare C1 su scala alta (653 slide).
- **C1**: xLSTM e GLA non hanno nessun run (né SFC né 2D PE) → C1 non è arch-agnostic oltre Mamba/Transformer.
- **C2**: CCRCC/BAP1 non ha **nessuna** baseline → impossibile validare C2 a scala intermedia (~783 slide).
- **C2**: BRCA TransMIL non terminato; 2DMamba mancante.

---

## 2. Correzioni codice

### 2.1 Documentare `max_seq_len` (NUOVO parametro non documentato) 🟥 ALTA

I file modificati introducono `max_seq_len` su `HilbertSequenceEncoder.__init__()` (hilbert_wsi/encoder.py:39), `TileCoordEncoder.__init__()` (hilbert_wsi/encoder_2d.py:61) e relativi `_build()`/`_build_2dpe()` dell'adapter (adapters/patho_bench.py:314,361). Tronca la sequenza dopo `max_seq_len` tile.

**Azioni**:
- [ ] Aggiungere docstring breve a entrambi gli `__init__` che dichiari: semantica (truncation post-ordering, FIFO), default `None` = no truncation, raccomandazione (CCRCC = 2048 per evitare OOM su slide >10k tile).
- [ ] Aggiornare `docs/architecture.md` § "Module contracts" con la firma estesa.
- [ ] Aggiornare `CLAUDE.md` § "Encoder pair" segnalando che il truncation è **post-ordering**: per gli SFC mantiene le prime 2048 tile lungo la curva (cluster locale spaziale), per 2D PE le prime 2048 tile **nell'ordine arbitrario H5** (potenziale bias). Nota a rischio fairness.
- [ ] Aggiungere test in `tests/test_encoder.py`: input con N=3000, max_seq_len=2048, verificare `idx.shape[1] == 2048` e che il bias dell'ordine truncation sia almeno identico tra encoder 1D e 2D (es. truncare dopo lo stesso shuffle per il path 2D, oppure scegliere strategia diversa — vedi punto sotto).

**Rischio fairness C1**: in TileCoordEncoder il truncation viene applicato **prima** della PE (riga 117-119), quindi sulle prime 2048 tile nell'ordine in cui arrivano dall'H5. Per HilbertSequenceEncoder il truncation è **dopo** il permutation index, quindi sulle prime 2048 tile lungo la SFC. **Questi non sono equivalenti**: 2D PE perde una porzione casuale dello slide, SFC perde la coda della curva (regione spaziale contigua). Decidere:
- **Opzione A**: troncare entrambi sull'ordine H5 prima dell'ordering/PE (più semplice, fair per confronto).
- **Opzione B**: troncare entrambi dopo un random crop deterministico (`blake2b(coords) seed`).
- **Opzione C**: documentare la non-equivalenza e citarla come confound nei risultati CCRCC.

→ **Decisione raccomandata**: opzione A. Modifica `HilbertSequenceEncoder.forward()` per troncare features/coords **prima** di `_compute_perms`, così sia 1D che 2D vedono lo stesso sottoinsieme di tile.

### 2.2 Rimuovere config CCRCC duplicate 🟧 MEDIA

`configs/backbones/{mamba,mamba_2dpe,transformer,transformer_2dpe}_ccrcc.yaml` differiscono dal `_base.yaml` corrispondente **solo** per l'aggiunta di `max_seq_len: 2048`. Generano proliferation (4 file in più ad ogni nuovo dataset).

**Azioni**:
- [ ] Aggiungere `--max_seq_len` come CLI flag a `scripts/run_benchmark.py` che fa override del valore yaml (oppure accetta un secondo `--model_kwargs_yaml_override`).
- [ ] Eliminare i 4 file `*_ccrcc.yaml`.
- [ ] Aggiornare `scripts/run_ccrcc_bap1_c1.sh` per usare i `_base.yaml` + `--max_seq_len 2048`.
- [ ] In alternativa (se si preferisce documentare invece): convertire i 4 file in stub che fanno `extends: mamba_base.yaml` e dichiarare la convenzione in `docs/architecture.md` § "Adding a new backbone".

### 2.3 Aggiungere config 2DPE per xLSTM e GLA 🟥 ALTA (C1)

Senza questi config, C1 non può essere verificato fuori dall'asse Mamba/Transformer.

**File da creare** (copiare il pattern di `mamba_2dpe_base.yaml`):
- `configs/backbones/xlstm_2dpe_base.yaml`
- `configs/backbones/gla_2dpe_base.yaml`

**Verifica**: lanciare un dry-run di `hilbertwsi_2dpe_xlstm` e `hilbertwsi_2dpe_gla` su UCEC per controllare che `_build_2dpe()` (adapters/patho_bench.py:302) accetti i backbone name xlstm/gla — se la dispatch è ristretta solo a mamba/transformer, estenderla.

### 2.4 Chiarire `mamba_attnpool_base.yaml` 🟨 BASSA

Esiste ma non ha pair 2dpe. È un'ablazione di pooling (mean vs attn) o una variante separata? Non documentato.

**Azioni**:
- [ ] Decidere: (i) rimuovere se è un esperimento concluso e già riportato in `docs/benchmarks.md`; (ii) tenere e aggiungere `mamba_attnpool_2dpe_base.yaml` se serve per fairness; (iii) integrare come `pooling: attn` dentro `mamba_base.yaml` con commento.

### 2.5 Hilbert/Random ordering naming consistency 🟨 BASSA

`hilbert_wsi/ordering/random_perm.py` — il nome file include `_perm` ma il registry usa `"random"`. Confermare che `get_ordering("random")` punti a questa classe e che non esistano riferimenti morti a `random_perm`.

**Azioni**:
- [ ] `grep -rn "random_perm" .` → rinominare in `random.py` se nessuno importa direttamente la classe.

---

## 3. Esperimenti mancanti (prioritizzati per claim)

### 3.1 BLOCCANTE C2 — CCRCC BAP1 baselines 🟥

CCRCC è il dataset intermedio. Senza baseline su questa scala, C2 vale solo su UCEC (95 slide).

```bash
for MODEL in abmil_baseline transmil_baseline mambamil_baseline clam_sb_baseline clam_mb_baseline dsmil_baseline twodmamba_baseline; do
  conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ccrcc --task BAP1_mutation \
    --experiment_type finetune \
    --model_name $MODEL \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_CCRCC \
    --splits_root splits \
    --saveto runs/cptac_ccrcc_BAP1_${MODEL} \
    --model_kwargs_yaml configs/baselines/${MODEL%_baseline}_base.yaml \
    --num_epochs 20
done
```

Tempo stimato: ~5h sequenziale su 1 GPU.

### 3.2 BLOCCANTE C1 — BRCA 2D PE 🟥

Senza 2D PE su BRCA, C1 non ha confronto fair su scala alta — solo su UCEC (95) e CCRCC (783).

```bash
for MODEL in hilbertwsi_2dpe_mamba hilbertwsi_2dpe_transformer; do
  conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_brca --task Immune_class \
    --experiment_type finetune \
    --model_name $MODEL \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_BRCA \
    --splits_root splits \
    --saveto runs/cptac_brca_Immune_class_${MODEL} \
    --model_kwargs_yaml configs/backbones/${MODEL#hilbertwsi_2dpe_}_2dpe_base.yaml \
    --num_epochs 20
done
```

Tempo stimato: ~4h.

### 3.3 IMPORTANTE C1+C2 — xLSTM e GLA su UCEC 🟧

Conferma arch-agnostic di C1 e amplia C2.

```bash
for BB in xlstm gla; do
  for ORD in hilbert random; do
    conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
      --source cptac_ucec --task Immune_class \
      --experiment_type finetune \
      --model_name hilbertwsi_${ORD}_${BB} \
      --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
      --splits_root splits \
      --saveto runs/cptac_ucec_Immune_class_hilbertwsi_${ORD}_${BB} \
      --model_kwargs_yaml configs/backbones/${BB}_base.yaml \
      --num_epochs 20
  done
done
```

Tempo: ~4h. **Prerequisito**: 2.3 completato se si vuole anche il braccio 2DPE per chiusura C1.

### 3.4 OPZIONALE — Similarity+Transformer su UCEC

Colma cella mancante nella tabella ablation (CLAUDE.md § risultati).

```bash
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
  --source cptac_ucec --task Immune_class \
  --experiment_type finetune \
  --model_name hilbertwsi_similarity_transformer \
  --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
  --splits_root splits \
  --saveto runs/cptac_ucec_Immune_class_hilbertwsi_similarity_transformer \
  --model_kwargs_yaml configs/backbones/transformer_base.yaml \
  --num_epochs 20
```

---

## 4. Test (coverage gaps)

Coverage attuale: orderings, encoder forward, Mamba CUDA. Mancano:

### 4.1 Test adapter dispatch 🟧

`adapters/patho_bench.py` non ha test su `_parse_name` né su `register_hilbert_encoders`.

**Azioni**:
- [ ] Aggiungere `tests/test_adapter.py` con:
  - `_parse_name("hilbertwsi_snake_transformer")` → `("snake", "transformer")`
  - `_parse_name("hilbertwsi_2dpe_mamba")` → `("2dpe", "mamba")`
  - `_parse_name("hilbertwsi_random_xlstm")` → `("random", "xlstm")` (verificare che xlstm sia accettato)
  - dispatch baseline: `_build_abmil_baseline()` → restituisce `ABMILWrapper`, non solleva
  - smoke test: `register_hilbert_encoders()` non rompe il `trident.slide_encoder_models.load.encoder_factory` originale per modelli non-hilbertwsi.

### 4.2 Test baseline wrappers 🟧

Le 7 baseline (ABMIL/TransMIL/MambaMIL/2DMamba/CLAM-SB/CLAM-MB/DSMIL) non hanno test unitari di forward.

**Azioni**:
- [ ] In `tests/test_baselines.py` smoke test forward per ciascuna: input (1, 1, 256, 1536), verificare shape output `(1, embedding_dim)` e device coerente.

### 4.3 Test max_seq_len truncation 🟥

Vedi 2.1. Bloccante per fairness C1.

---

## 5. Documentazione (consolidamento)

### 5.1 Eliminare duplicazione benchmarks.md ↔ results_summary.md 🟧

La tabella UCEC è duplicata in entrambi (~5 KiB). Versione in `results_summary.md` è più completa (include CLAM/DSMIL aggiornati al 2026-05-22).

**Azioni**:
- [ ] Designare `docs/results_summary.md` come **single source of truth** per i risultati numerici.
- [ ] In `docs/benchmarks.md`, sostituire la tabella UCEC con un link `vedi docs/results_summary.md`. Mantenere solo: comandi di run, struttura `runs/`, lista model name, CI test methodology.

### 5.2 Promuovere `docs/roadmap_c1.md` da untracked a tracked 🟧

File già pronto. Va committato (vedi § 7).

**Azioni**:
- [ ] Aggiungere riferimento in `CLAUDE.md` § Roadmap: link a `docs/roadmap_c1.md` per dettagli step-by-step.
- [ ] Generare il pendant `docs/roadmap_c2.md` con la roadmap per il claim 2 (1D-native vs MIL). Outline:
  - Stato baseline coverage per dataset (vedi § 1).
  - Esperimenti necessari (CCRCC baselines, BRCA 2DMamba+TransMIL, xLSTM/GLA su almeno 2 dataset).
  - Criterio di conferma: HilbertWSI 1D-native ≥ media baseline con CI disjoint su ≥2 dataset.
  - Caveat: HilbertWSI 13-22M vs baseline 1-2M params — documentare confound capacity.

### 5.3 CLAUDE.md — aggiornamenti necessari 🟧

- [ ] Sezione "Encoder pair" → menzionare `max_seq_len` come kwarg dell'encoder.
- [ ] Sezione "Adding a new backbone" → spiegare se basta `_base.yaml` o se serve anche `_2dpe_base.yaml`.
- [ ] Sezione "Risultati" → aggiungere righe CCRCC BAP1 (4 modelli già completati: hilbert/random × mamba/transformer + 2dpe × mamba/transformer).
- [ ] Sezione "Roadmap" → marcare 2c come "in corso" e linkare entrambi i roadmap doc.
- [ ] Verificare il valore `+0.263M overhead` di TileCoordEncoder vs commento in `encoder_2d.py:76` ("0.26M difference"). Allineare.

### 5.4 Architecture.md — aggiornamenti 🟨

- [ ] Aggiungere `max_seq_len` al contract `SequenceBackbone`/encoders.
- [ ] Aggiungere sezione "Adding a 2DPE-pair config" se la decisione 2.3 è di rendere il pair obbligatorio.

---

## 6. Pulizia git

Stato attuale: 4 file modificati staged-out, 6 untracked.

**Modifiche già pronte da committare** (in un singolo commit):

```
M CLAUDE.md                       # aggiornamento risultati 2D PE UCEC
M adapters/patho_bench.py         # max_seq_len passthrough
M hilbert_wsi/encoder.py          # max_seq_len truncation
M hilbert_wsi/encoder_2d.py       # max_seq_len truncation
```

Messaggio suggerito:

```
feat(encoder): add max_seq_len truncation to 1D and 2D PE encoders

Caps sequence length post-ordering to avoid OOM on slides with >10k tiles
(CCRCC: median ~5k, max ~15k vs UCEC ~1.5k). Threading the kwarg through
Patho-Bench adapter to allow per-experiment override via model_kwargs_yaml.

Also updates CLAUDE.md with 2D PE UCEC results (2dpe_transformer=0.545,
2dpe_mamba=0.524) — both fall in no-ordering cluster, confirming C1.
```

**File untracked da decidere caso per caso:**

| File | Decisione |
|---|---|
| `docs/roadmap_c1.md` | Commit (è il roadmap C1 — vedi 5.2) |
| `scripts/run_ccrcc_bap1_c1.sh` | Commit dopo aver applicato 2.2 (usare `_base.yaml` + CLI flag) |
| `configs/backbones/mamba_ccrcc.yaml` | Eliminare (vedi 2.2) |
| `configs/backbones/mamba_2dpe_ccrcc.yaml` | Eliminare (vedi 2.2) |
| `configs/backbones/transformer_ccrcc.yaml` | Eliminare (vedi 2.2) |
| `configs/backbones/transformer_2dpe_ccrcc.yaml` | Eliminare (vedi 2.2) |

⚠️ **NON eliminare i 4 `*_ccrcc.yaml` prima** di aver implementato 2.2 e aver verificato che lo script aggiornato funzioni — altrimenti i prossimi run CCRCC si rompono.

---

## 7. Ordine di esecuzione consigliato

| # | Task | § | Priorità | Tempo |
|---|---|---|---|---|
| 1 | Decidere truncation strategy (Opzione A/B/C) | 2.1 | 🟥 | 30min disc. |
| 2 | Implementare truncation fair + test | 2.1, 4.3 | 🟥 | 1h |
| 3 | Aggiungere `--max_seq_len` CLI a `run_benchmark.py` | 2.2 | 🟧 | 30min |
| 4 | Creare `xlstm_2dpe_base.yaml`, `gla_2dpe_base.yaml` | 2.3 | 🟥 | 15min |
| 5 | Test dispatch adapter | 4.1 | 🟧 | 1h |
| 6 | Test baseline wrappers smoke | 4.2 | 🟧 | 1h |
| 7 | Commit modifiche encoder + max_seq_len | 6 | 🟥 | 5min |
| 8 | Lanciare 3.1 (CCRCC baselines) | 3.1 | 🟥 | 5h GPU |
| 9 | Lanciare 3.2 (BRCA 2DPE) in parallelo | 3.2 | 🟥 | 4h GPU |
| 10 | Lanciare 3.3 (xLSTM/GLA UCEC) | 3.3 | 🟧 | 4h GPU |
| 11 | Consolidare docs (5.1 + 5.2 + 5.3) | 5 | 🟧 | 1h |
| 12 | Aggiornare `results_summary.md` con nuovi numeri | 5.1 | 🟧 | 30min |
| 13 | Eliminare 4 file `*_ccrcc.yaml` | 6 | 🟧 | 5min |
| 14 | Commit roadmap_c1 + roadmap_c2 + script aggiornato | 6 | 🟧 | 5min |

**Wall-clock totale**: ~6h coding + ~13h GPU (in parallelo 2 GPU → 7h).

---

## 8. Bug critici trovati in audit codice (2026-05-30)

Audit eseguito su: `encoder.py`, `encoder_2d.py`, `pos_encoding.py`, `adapters/patho_bench.py`, `models/{mamba,transformer,xlstm,gla}.py`. Findings ordinati per severity.

### 8.1 🔴 CRITICO — `pos_encoding.py:62` dtype cast distrugge il PE

```python
return pe.to(coords.dtype)
```

`coords` in input arriva da H5 come int64 (level-0 pixel). `pe` è float32 in [-1, 1]. Il cast a int64 azzera tutti i valori frazionari → il PE 2D diventa una matrice di 0 e ±1 → **il path 2D è effettivamente senza informazione spaziale**.

**Implicazione per C1**: i risultati `2dpe_transformer=0.545` e `2dpe_mamba=0.524` riportati in `CLAUDE.md` potrebbero essere stati misurati con un PE quasi-nullo. Il confronto "ordering 1D vs 2D PE" non è valido finché il bug non è risolto e i due run non sono rilanciati.

**Fix**:
```python
return pe.to(features.dtype)  # ma features non è in scope qui
```
Soluzione più robusta — castare nel chiamante:
```python
# in TileCoordEncoder._encode (encoder_2d.py:99):
pe = self.pos_enc(coords.to(features.device)).to(features.dtype)
```
E rimuovere il cast in pos_encoding.py:62 (`return pe` invariato come float32).

**Test**: aggiungere `tests/test_pos_encoding.py` con coords int64 → verificare `pe.dtype == float32` e `pe.abs().max() > 0.01`.

**Azione bloccante**: rilanciare 2D PE su UCEC e CCRCC dopo il fix per ottenere risultati validi (sostituisce CLAUDE.md risultati attuali).

### 8.2 🔴 CRITICO — `encoder_2d.py:74` skip_proj forzato rompe xLSTM/GLA

```python
bb_kw["skip_proj"] = True
```

`hilbert_wsi/models/xlstm.py:39-49` e `hilbert_wsi/models/gla.py:138-147` **non accettano** `skip_proj` come kwarg. Tutti i tentativi di costruire `hilbertwsi_2dpe_xlstm` o `hilbertwsi_2dpe_gla` falliscono con `TypeError: __init__() got an unexpected keyword argument 'skip_proj'`.

**Implicazione per C1**: l'arm 2D PE è disponibile solo per Mamba e Transformer. C1 non può essere arch-agnostic finché xLSTM/GLA non supportano skip_proj.

**Fix opzioni**:
- **A (raccomandata)**: aggiungere `skip_proj: bool = False` ai costruttori di xLSTM e GLA, con la stessa semantica di Mamba/Transformer (skip `proj_in` se True). 4 righe per backbone.
- **B**: in `encoder_2d.py` rilevare se il backbone supporta skip_proj via `inspect.signature` e applicarlo solo se sì. Sporcizia ma meno invasivo.

→ **Decisione**: A. Mantiene il contratto SequenceBackbone simmetrico.

### 8.3 🔴 CRITICO — `encoder_2d.py:96-120` mask ignorata

`_encode(self, features, coords)` non ha parametro mask. `forward()` non legge `sample.get("mask")`. Le baseline `mamba.py` / `transformer.py` accettano mask e ne tengono conto nel pooling (mean masked, attn zero-out). In `encoder.py:84-88` (path 1D), la mask viene **gatherata** insieme alle features tramite il permutation index e passata al backbone.

**Implicazione**: se Patho-Bench passa una mask (es. per dataset con N variabile e padding), il path 2D la perde silenziosamente → pooling sbagliato → fairness rotta rispetto al path 1D.

**Fix**:
```python
def _encode(self, features, coords, mask=None):
    h = self.dropout_layer(self.proj_in(features))
    pe = self.pos_enc(coords.to(features.device)).to(features.dtype)
    h = h + pe
    return self.backbone(h, mask=mask)

def forward(self, sample, device="cpu"):
    ...
    mask = sample.get("mask")
    if mask is not None:
        mask = mask.to(device)
        if mask.dim() == 3 and mask.shape[1] == 1:
            mask = mask.reshape(mask.shape[0], -1)
        if self.max_seq_len is not None and mask.shape[1] > self.max_seq_len:
            mask = mask[:, :self.max_seq_len]
    return self._encode(features, coords, mask=mask)
```

**Test**: smoke test con sample che include mask con qualche `False` → confronto pooling con `encoder.py` deve essere coerente (stesso supporto effettivo).

### 8.4 🔴 CRITICO — `encoder.py:81 vs encoder_2d.py:117` truncation asimmetrica

Già coperto in § 2.1. **Stesso fix lì proposto**: troncare features+coords prima di ordering nel path 1D.

### 8.5 🟧 RISCHIO — `_build()` / `_build_2dpe()` droppano silenziosamente kwarg sconosciuti

`adapters/patho_bench.py:302-325` (2dpe) e `:346-376` (1D) fanno `kwargs.pop(...)` per ogni kwarg noto. Se l'utente passa un kwarg non gestito (es. `n_heads=8` invece di `backbone_kwargs: {n_heads: 8}`), viene ignorato. Sintomi: il parametro non ha effetto, run produce risultato inconsistente, no errore.

**Fix**:
```python
# In coda a _build() e _build_2dpe():
if kwargs:
    raise ValueError(f"Unexpected kwargs for {name}: {list(kwargs)}")
```

### 8.6 🟧 RISCHIO — `transformer.py:139` non supporta `pooling: "last"`

Mamba (`mamba.py:72`), xLSTM (`xlstm.py:51`), GLA (`gla.py:149`) accettano `pooling in {"mean", "cls", "last"}`. Transformer accetta solo `{"mean", "cls"}`. Se un'ablazione di pooling è prevista (es. confronto last-token RNN-style), Transformer non può partecipare.

**Decisione**: documentare la limitazione in `CLAUDE.md` § "Sequence backbones" oppure aggiungere supporto a `"last"` in transformer.py (`x[:, -1]`).

### 8.7 🟧 RISCHIO — `mamba.py:72` supporta `pooling: "attn"`, altri no

Asimmetria opposta: solo Mamba ha pooling "attn" (gated attention pool). Una `mamba_attnpool_base.yaml` esiste; se è un'ablazione attiva, va replicata su altri backbone per fairness. Vedi § 2.4.

### 8.8 🟨 NIT — `patho_bench.py:215` docstring errata

```python
# Patho-Bench collate: (B, S, N, D) and (B, S, N, 2). Flatten to (B*S, N, D/2).
```

"D/2" è errato — il flatten è a `(B*S, N, D)`. Aggiornare comment.

### 8.9 🟨 NIT — Verificare ABMILWrapper input_dim

`patho_bench.py:125-126`:
```python
# Trident ABMILSlideEncoder doesn't accept input_dim — remove it
kwargs.pop("input_dim", None)
```

Comment dichiara che ABMILSlideEncoder non accetta `input_dim`. Verificare leggendo Trident source: se in realtà accetta `input_feature_dim` (un alias), il pop scarta info che andrebbe rinominata. Test sul fold di UCEC ABMIL (già funziona, run completo) suggerisce che è OK così — ma verificare prima di future modifiche.

---

## 9. Test di regressione minimi post-fix

Dopo aver applicato i fix § 8.1, 8.2, 8.3, 8.4, eseguire:

```bash
# 1. Test unitari nuovi
pytest tests/test_pos_encoding.py tests/test_encoder.py -v

# 2. Smoke test cross-arch su UCEC (1 epoca, 1 fold) per ciascun backbone × {1D, 2DPE}:
for BB in mamba transformer xlstm gla; do
  for MODE in hilbert 2dpe; do
    conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
      --source cptac_ucec --task Immune_class \
      --experiment_type finetune \
      --model_name hilbertwsi_${MODE}_${BB} \
      --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
      --splits_root splits \
      --saveto /tmp/smoke_ucec_${MODE}_${BB} \
      --model_kwargs_yaml configs/backbones/${BB}_${MODE/hilbert/}base.yaml \
      --num_epochs 1
  done
done
```

Tutti 8 run devono completare senza eccezioni. Solo dopo questo, lanciare i sweep full di § 3.

---

## 10. Criteri di accettazione post-correzioni

La repo è allineata ai 2 contributi quando:

**C1 — verificabile**:
- [ ] Per ogni backbone in {mamba, transformer, xlstm, gla} esiste sia `_base.yaml` che `_2dpe_base.yaml`.
- [ ] Almeno 3 dataset hanno il pair completo (SFC vs 2D PE) per Mamba **e** Transformer.
- [ ] `max_seq_len` tronca in modo fair tra encoder 1D e 2D (stesso sottoinsieme di tile).
- [ ] `docs/roadmap_c1.md` ha tutte le righe ✅ tranne quelle marcate "opzionale".

**C2 — verificabile**:
- [ ] CCRCC BAP1 ha tutte le 7 baseline + HilbertWSI SFC.
- [ ] BRCA Immune_class ha tutte le 7 baseline + HilbertWSI SFC (no gap TransMIL/2DMamba).
- [ ] `docs/roadmap_c2.md` esiste e ha tabella copertura completata per ≥3 dataset.
- [ ] `docs/results_summary.md` riporta il confronto 1D-native vs media baseline con CI test per ciascun dataset.

**Infrastruttura**:
- [ ] `pytest tests/ -v` → tutti i nuovi test passano (dispatch, baselines, max_seq_len).
- [ ] `git status` → working tree clean (no untracked legacy).
- [ ] CLAUDE.md aggiornato a riflettere lo stato corrente (no riferimenti a feature non implementate).
