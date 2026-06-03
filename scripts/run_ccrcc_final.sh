#!/usr/bin/env bash
# Run 6 CCRCC BAP1 experiments with corrected pipeline.
# Waits for postfix_ucec_2dpe_transformer to finish, then starts immediately
# so CCRCC runs in parallel with UCEC 2dpe_mamba from postfix_reruns.sh.
#
# Fixes applied (same as run_postfix_reruns.sh):
#   BUG A — pos_encoding.py grid indices (not [0,1] norm)
#   BUG B — bag_size=2048 uniform sample (not slab truncation)
#   BUG C — encoder_2d.py randomises Mamba scan order
#
# Saveto: runs/ccrcc_final_* (distinct from postfix_ccrcc_* to avoid conflict)

set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="hilbert-wsi-clean"
SOURCE="cptac_ccrcc"
TASK="BAP1_mutation"
FEATURES="/data/hilbert-wsi/features/CPTAC_CCRCC"
SPLITS="splits"
EPOCHS=20
BAG=2048
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
SWEEP_LOG="$LOG_DIR/ccrcc_final.log"

run_exp() {
    local model_name="$1"; local yaml="$2"; local saveto="$3"; local logfile="$4"
    echo "[$(date '+%F %T')] START $model_name" | tee -a "$SWEEP_LOG"
    if conda run -n "$CONDA_ENV" python -m scripts.run_benchmark \
        --source "$SOURCE" --task "$TASK" \
        --experiment_type finetune \
        --model_name "$model_name" \
        --patch_features_dir "$FEATURES" \
        --splits_root "$SPLITS" \
        --saveto "$saveto" \
        --model_kwargs_yaml "$yaml" \
        --num_epochs "$EPOCHS" \
        --bag_size "$BAG" \
        --gpu 0 \
        2>&1 | tee "$logfile"; then
        echo "[$(date '+%F %T')] OK  $model_name" | tee -a "$SWEEP_LOG"
    else
        echo "[$(date '+%F %T')] FAIL $model_name" | tee -a "$SWEEP_LOG"
    fi
}

# Wait for UCEC 2dpe_transformer to finish before starting (UCEC Mamba will
# then be the only other GPU process, leaving ~7-9 GB free for us).
UCEC_DONE="runs/postfix_ucec_2dpe_transformer/test_metrics_summary.json"
echo "[$(date '+%F %T')] Waiting for UCEC 2dpe_transformer to finish..." | tee "$SWEEP_LOG"
while [ ! -f "$UCEC_DONE" ]; do sleep 30; done
echo "[$(date '+%F %T')] UCEC 2dpe_transformer done — starting 6 CCRCC experiments" | tee -a "$SWEEP_LOG"

run_exp hilbertwsi_hilbert_transformer \
    configs/backbones/transformer_base.yaml \
    runs/ccrcc_final_hilbert_transformer \
    "$LOG_DIR/ccrcc_final_hilbert_transformer.log"

run_exp hilbertwsi_random_transformer \
    configs/backbones/transformer_base.yaml \
    runs/ccrcc_final_random_transformer \
    "$LOG_DIR/ccrcc_final_random_transformer.log"

run_exp hilbertwsi_2dpe_transformer \
    configs/backbones/transformer_2dpe_base.yaml \
    runs/ccrcc_final_2dpe_transformer \
    "$LOG_DIR/ccrcc_final_2dpe_transformer.log"

run_exp hilbertwsi_hilbert_mamba \
    configs/backbones/mamba_base.yaml \
    runs/ccrcc_final_hilbert_mamba \
    "$LOG_DIR/ccrcc_final_hilbert_mamba.log"

run_exp hilbertwsi_random_mamba \
    configs/backbones/mamba_base.yaml \
    runs/ccrcc_final_random_mamba \
    "$LOG_DIR/ccrcc_final_random_mamba.log"

run_exp hilbertwsi_2dpe_mamba \
    configs/backbones/mamba_2dpe_base.yaml \
    runs/ccrcc_final_2dpe_mamba \
    "$LOG_DIR/ccrcc_final_2dpe_mamba.log"

echo "[$(date '+%F %T')] ALL 6 CCRCC EXPERIMENTS DONE" | tee -a "$SWEEP_LOG"
