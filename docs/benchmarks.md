# Reproducing benchmarks

## Dataset status (May 2026)

| Dataset | Features | Status | Path |
|---|---|---|---|
| CPTAC-UCEC | UNI2-h 1536-dim | ✅ Ready (95 slides) | `/data/hilbert-wsi/features/CPTAC_UCEC/*.h5` |
| PANDA | UNI2-h 1536-dim | ⏳ Extracting tar | `/data/hilbert-wsi/features/PANDA/` |
| CPTAC-BRCA | UNI2-h 1536-dim | 🔲 Not downloaded | — |

Features source: `MahmoodLab/UNI2-h-features` on HuggingFace (gated, user: Yuto2007).

Task splits cached: `splits/cptac_ucec/Immune_class/`.

## Prereqs

- Python ≥ 3.10, CUDA-capable GPU.
- `pip install -e ".[mamba,bench,dev]"`.

## Step 1 — Patch features

Use TRIDENT to extract patch features. Mahmood Lab also publishes pre-extracted
UNI2-h features on HuggingFace for the major bench corpora; downloading these
is the fastest path.

```bash
# Example: TRIDENT CLI for one cohort
python -m trident.run_batch_of_slides \
    --slide_dir /data/TCGA-BRCA/svs \
    --feat_extractor uni_v2 \
    --output_dir /data/features/tcga_brca_uni2h
```

## Step 2 — Run a single task

```bash
python -m scripts.run_benchmark \
    --task tcga_brca/BRCA_Subtype \
    --experiment_type linprobe \
    --model_name hilbertwsi_hilbert_mamba \
    --patch_features_dir /data/features/tcga_brca_uni2h \
    --model_kwargs_yaml configs/backbones/mamba_base.yaml \
    --saveto runs/brca_hilbert_mamba
```

`runs/brca_hilbert_mamba/` will contain pooled embeddings, the eval CSV, and a
text dump of the encoder architecture.

## Step 3 — Ordering ablation

```bash
python -m scripts.ablation_orderings \
    --task tcga_brca/BRCA_Subtype \
    --backbone mamba \
    --patch_features_dir /data/features/tcga_brca_uni2h \
    --model_kwargs_yaml configs/backbones/mamba_base.yaml \
    --saveto runs/ablation_brca
```

Runs `hilbertwsi_<scheme>_mamba` for every ordering in
`hilbert_wsi.ordering.available_orderings()` sequentially. Each run writes to
`runs/ablation_brca/<scheme>/`.

## Step 4 — Full Patho-Bench sweep

For multi-task, multi-seed sweeps use the Patho-Bench tmux runner directly
after registering HilbertWSI encoders:

```python
from adapters.patho_bench import register_hilbert_encoders
register_hilbert_encoders()
# then dispatch to patho_bench.Runner with tasks_yaml / hyperparams_yaml
```

A scripted sweep over all 42 Patho-Bench tasks lives outside the v0.1 scope —
contributions welcome.

## Expected ballparks

To be filled in once the first set of runs finishes. Compare against the ABMIL
baseline that ships with Patho-Bench (`abmil` model name) to confirm the
sequence approach is at parity or better before scaling.
