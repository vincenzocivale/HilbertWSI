#!/usr/bin/env bash
# Post-fix re-runs after the SFC-ordering pipeline audit (2026-06-02).
#
# Fixes applied (see hilbert_wsi/pos_encoding.py, encoder.py, encoder_2d.py):
#   BUG A — 2D PE now uses integer grid indices (was near-constant under [0,1] norm).
#   BUG B — length cap is a uniform-random per-slide subsample, NOT a first-N
#           H5-order slab. Here memory is bounded with Patho-Bench --bag_size
#           (uniform sample, re-drawn each epoch) instead of max_seq_len.
#   BUG C — 2D-PE encoder randomises scan order (position only via PE, not raster).
#
# Re-run set (8 sweeps):
#   UCEC 2dpe x {transformer,mamba}   — FULL sequence (matches existing UCEC
#                                       ordering runs -> clean within-UCEC compare)
#   CCRCC {hilbert,random,2dpe} x {transformer,mamba} — bag_size=2048 uniform
#                                       (all 6 identical budget -> clean within-CCRCC)
#
# Cross-dataset caveat: UCEC (full) vs CCRCC (bag 2048) still differ in budget;
# within-dataset comparisons are clean. GPU: 12 GB -> full-seq CCRCC OOMs, hence bag.

cd "$(dirname "$0")/.."

CONDA_ENV="hilbert-wsi-clean"
SPLITS="splits"
EPOCHS=20
CCRCC_BAG=2048
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
SWEEP_LOG="$LOG_DIR/postfix_reruns.log"

UCEC_FEAT="/data/hilbert-wsi/features/CPTAC_UCEC"
CCRCC_FEAT="/data/hilbert-wsi/features/CPTAC_CCRCC"

# run_exp <source> <task> <features> <model_name> <yaml> <saveto> <logfile> [extra args...]
run_exp() {
    local source="$1"; local task="$2"; local feat="$3"; local model_name="$4"
    local yaml="$5"; local saveto="$6"; local logfile="$7"; shift 7
    echo "[$(date '+%F %T')] START $model_name ($source/$task)" | tee -a "$SWEEP_LOG"
    if conda run -n "$CONDA_ENV" python -m scripts.run_benchmark \
        --source "$source" --task "$task" \
        --experiment_type finetune \
        --model_name "$model_name" \
        --patch_features_dir "$feat" \
        --splits_root "$SPLITS" \
        --saveto "$saveto" \
        --model_kwargs_yaml "$yaml" \
        --num_epochs "$EPOCHS" \
        --gpu 0 "$@" \
        2>&1 | tee "$logfile"; then
        echo "[$(date '+%F %T')] OK    $model_name" | tee -a "$SWEEP_LOG"
    else
        echo "[$(date '+%F %T')] FAIL  $model_name (continuing)" | tee -a "$SWEEP_LOG"
    fi
}

echo "[$(date '+%F %T')] START postfix re-runs (8 sweeps)" | tee "$SWEEP_LOG"

# ---- UCEC 2D PE (full sequence, no bag) ----
run_exp cptac_ucec Immune_class "$UCEC_FEAT" hilbertwsi_2dpe_transformer \
    configs/backbones/transformer_2dpe_base.yaml \
    runs/postfix_ucec_2dpe_transformer "$LOG_DIR/postfix_ucec_2dpe_transformer.log"

run_exp cptac_ucec Immune_class "$UCEC_FEAT" hilbertwsi_2dpe_mamba \
    configs/backbones/mamba_2dpe_base.yaml \
    runs/postfix_ucec_2dpe_mamba "$LOG_DIR/postfix_ucec_2dpe_mamba.log"

# ---- CCRCC BAP1 (bag_size=2048 uniform, all 6 arms) ----
for arm in \
    "hilbertwsi_hilbert_transformer configs/backbones/transformer_base.yaml postfix_ccrcc_hilbert_transformer" \
    "hilbertwsi_random_transformer  configs/backbones/transformer_base.yaml postfix_ccrcc_random_transformer" \
    "hilbertwsi_2dpe_transformer    configs/backbones/transformer_2dpe_base.yaml postfix_ccrcc_2dpe_transformer" \
    "hilbertwsi_hilbert_mamba       configs/backbones/mamba_base.yaml postfix_ccrcc_hilbert_mamba" \
    "hilbertwsi_random_mamba        configs/backbones/mamba_base.yaml postfix_ccrcc_random_mamba" \
    "hilbertwsi_2dpe_mamba          configs/backbones/mamba_2dpe_base.yaml postfix_ccrcc_2dpe_mamba" \
; do
    set -- $arm
    run_exp cptac_ccrcc BAP1_mutation "$CCRCC_FEAT" "$1" "$2" "runs/$3" \
        "$LOG_DIR/$3.log" --bag_size "$CCRCC_BAG"
done

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$SWEEP_LOG"
