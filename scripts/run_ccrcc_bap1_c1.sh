#!/usr/bin/env bash
# Step 1 roadmap C1: CCRCC BAP1_mutation — ordering vs 2D PE vs random
# Transformer trio then Mamba trio (single GPU, sequential)
# max_seq_len=2048 passed via CLI to handle large CCRCC slides (up to ~15k tiles)

set -euo pipefail

FEATURES=/data/hilbert-wsi/features/CPTAC_CCRCC
SPLITS=splits
EPOCHS=20
LOG=/data/hilbert-wsi/runs/ccrcc_bap1_c1.log

mkdir -p /data/hilbert-wsi/runs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ccrcc_bap1_c1 sweep" | tee -a "$LOG"

# Transformer trio
for MODEL in hilbertwsi_hilbert_transformer hilbertwsi_random_transformer; do
    SAVETO=runs/cptac_ccrcc_BAP1_${MODEL}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $MODEL" | tee -a "$LOG"
    conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
        --source cptac_ccrcc --task BAP1_mutation \
        --experiment_type finetune \
        --model_name "$MODEL" \
        --patch_features_dir "$FEATURES" \
        --splits_root "$SPLITS" \
        --saveto "$SAVETO" \
        --model_kwargs_yaml configs/backbones/transformer_base.yaml \
        --max_seq_len 2048 \
        --num_epochs "$EPOCHS" 2>&1 | tee -a "$LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $MODEL" | tee -a "$LOG"
done

# 2D PE Transformer
MODEL=hilbertwsi_2dpe_transformer
SAVETO=runs/cptac_ccrcc_BAP1_${MODEL}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $MODEL" | tee -a "$LOG"
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ccrcc --task BAP1_mutation \
    --experiment_type finetune \
    --model_name "$MODEL" \
    --patch_features_dir "$FEATURES" \
    --splits_root "$SPLITS" \
    --saveto "$SAVETO" \
    --model_kwargs_yaml configs/backbones/transformer_2dpe_base.yaml \
    --max_seq_len 2048 \
    --num_epochs "$EPOCHS" 2>&1 | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $MODEL" | tee -a "$LOG"

# Mamba trio
for MODEL in hilbertwsi_hilbert_mamba hilbertwsi_random_mamba; do
    SAVETO=runs/cptac_ccrcc_BAP1_${MODEL}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $MODEL" | tee -a "$LOG"
    conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
        --source cptac_ccrcc --task BAP1_mutation \
        --experiment_type finetune \
        --model_name "$MODEL" \
        --patch_features_dir "$FEATURES" \
        --splits_root "$SPLITS" \
        --saveto "$SAVETO" \
        --model_kwargs_yaml configs/backbones/mamba_base.yaml \
        --max_seq_len 2048 \
        --num_epochs "$EPOCHS" 2>&1 | tee -a "$LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $MODEL" | tee -a "$LOG"
done

# 2D PE Mamba
MODEL=hilbertwsi_2dpe_mamba
SAVETO=runs/cptac_ccrcc_BAP1_${MODEL}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $MODEL" | tee -a "$LOG"
conda run -n hilbert-wsi-clean python -m scripts.run_benchmark \
    --source cptac_ccrcc --task BAP1_mutation \
    --experiment_type finetune \
    --model_name "$MODEL" \
    --patch_features_dir "$FEATURES" \
    --splits_root "$SPLITS" \
    --saveto "$SAVETO" \
    --model_kwargs_yaml configs/backbones/mamba_2dpe_base.yaml \
    --max_seq_len 2048 \
    --num_epochs "$EPOCHS" 2>&1 | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $MODEL" | tee -a "$LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL DONE" | tee -a "$LOG"
