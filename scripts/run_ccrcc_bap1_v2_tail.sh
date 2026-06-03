#!/usr/bin/env bash
# Runs experiments 2-6 of the CCRCC BAP1 v2 sweep.
# Wait for experiment 1 (hilbert_transformer) to finish first.
set -e
cd "$(dirname "$0")/.."

CONDA_ENV="hilbert-wsi-clean"
SOURCE="cptac_ccrcc"
TASK="BAP1_mutation"
FEATURES="/data/hilbert-wsi/features/CPTAC_CCRCC"
SPLITS="splits"
EPOCHS=20
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

SWEEP_LOG="$LOG_DIR/ccrcc_bap1_v2_tail.log"

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
        --gpu 0 \
        2>&1 | tee "$logfile"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $model_name" | tee -a "$SWEEP_LOG"
}

# Wait for experiment 1 to finish
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for hilbert_transformer to finish..." | tee "$SWEEP_LOG"
while [ ! -f "runs/cptac_ccrcc_BAP1_hilbertwsi_hilbert_transformer/test_metrics_summary.json" ]; do
    sleep 60
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] hilbert_transformer done, starting remaining 5..." | tee -a "$SWEEP_LOG"

run_exp "hilbertwsi_random_transformer" \
    "configs/backbones/transformer_base.yaml" \
    "runs/cptac_ccrcc_BAP1_hilbertwsi_random_transformer" \
    "$LOG_DIR/ccrcc_bap1_v2_random_transformer.log"

run_exp "hilbertwsi_2dpe_transformer" \
    "configs/backbones/transformer_2dpe_base.yaml" \
    "runs/cptac_ccrcc_BAP1_hilbertwsi_2dpe_transformer" \
    "$LOG_DIR/ccrcc_bap1_v2_2dpe_transformer.log"

run_exp "hilbertwsi_hilbert_mamba" \
    "configs/backbones/mamba_base.yaml" \
    "runs/cptac_ccrcc_BAP1_hilbertwsi_hilbert_mamba" \
    "$LOG_DIR/ccrcc_bap1_v2_hilbert_mamba.log"

run_exp "hilbertwsi_random_mamba" \
    "configs/backbones/mamba_base.yaml" \
    "runs/cptac_ccrcc_BAP1_hilbertwsi_random_mamba" \
    "$LOG_DIR/ccrcc_bap1_v2_random_mamba.log"

run_exp "hilbertwsi_2dpe_mamba" \
    "configs/backbones/mamba_2dpe_base.yaml" \
    "runs/cptac_ccrcc_BAP1_hilbertwsi_2dpe_mamba" \
    "$LOG_DIR/ccrcc_bap1_v2_2dpe_mamba.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL 6 EXPERIMENTS DONE" | tee -a "$SWEEP_LOG"
