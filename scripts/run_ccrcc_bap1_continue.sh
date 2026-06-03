#!/usr/bin/env bash
# Continue CCRCC BAP1 sweep — experiments 3-6.
# Waits for the postfix_ucec_2dpe_transformer job to finish first.
set -e
cd "$(dirname "$0")/.."

CONDA_ENV="hilbert-wsi-clean"
SOURCE="cptac_ccrcc"
TASK="BAP1_mutation"
FEATURES="/data/hilbert-wsi/features/CPTAC_CCRCC"
SPLITS="splits"
EPOCHS=20
MAX_SEQ_LEN=2048
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

SWEEP_LOG="$LOG_DIR/ccrcc_bap1_continue.log"

run_exp() {
    local model_name="$1"
    local yaml="$2"
    local saveto="$3"
    local logfile="$4"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $model_name" | tee -a "$SWEEP_LOG"
    conda run -n "$CONDA_ENV" python -m scripts.run_benchmark \
        --source "$SOURCE" \
        --task "$TASK" \
        --experiment_type finetune \
        --model_name "$model_name" \
        --patch_features_dir "$FEATURES" \
        --splits_root "$SPLITS" \
        --saveto "$saveto" \
        --model_kwargs_yaml "$yaml" \
        --num_epochs "$EPOCHS" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --gpu 0 \
        2>&1 | tee "$logfile"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $model_name" | tee -a "$SWEEP_LOG"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for UCEC job to finish (postfix_ucec_2dpe_transformer)..." | tee "$SWEEP_LOG"
while pgrep -f "postfix_ucec_2dpe_transformer" > /dev/null 2>&1; do
    sleep 120
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] UCEC job done, starting CCRCC experiments 3-6" | tee -a "$SWEEP_LOG"

run_exp "hilbertwsi_2dpe_transformer" \
    "configs/backbones/transformer_2dpe_base.yaml" \
    "runs/cptac_ccrcc_BAP1_hilbertwsi_2dpe_transformer" \
    "$LOG_DIR/ccrcc_bap1_continue_2dpe_transformer.log"

run_exp "hilbertwsi_hilbert_mamba" \
    "configs/backbones/mamba_base.yaml" \
    "runs/cptac_ccrcc_BAP1_hilbertwsi_hilbert_mamba" \
    "$LOG_DIR/ccrcc_bap1_continue_hilbert_mamba.log"

run_exp "hilbertwsi_random_mamba" \
    "configs/backbones/mamba_base.yaml" \
    "runs/cptac_ccrcc_BAP1_hilbertwsi_random_mamba" \
    "$LOG_DIR/ccrcc_bap1_continue_random_mamba.log"

run_exp "hilbertwsi_2dpe_mamba" \
    "configs/backbones/mamba_2dpe_base.yaml" \
    "runs/cptac_ccrcc_BAP1_hilbertwsi_2dpe_mamba" \
    "$LOG_DIR/ccrcc_bap1_continue_2dpe_mamba.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL 4 REMAINING EXPERIMENTS DONE" | tee -a "$SWEEP_LOG"
