#!/usr/bin/env bash
# Launch the 2DMambaMIL run on CPTAC-UCEC Immune_class (Phase 3 experiment 1).
# Standalone so it can run in parallel with other long jobs or sequentially.
#
# Outputs:
#     runs/<saveto>/run.log               stdout+stderr
#     runs/twodmamba_killer_summary.log   master log with timestamps + exit code

set -u
cd "$(dirname "$0")/.."

source /home/oem/miniconda3/etc/profile.d/conda.sh
conda activate hilbert-wsi-clean

# 2D pscan activations are large; expand the allocator to reduce fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_LOG=runs/twodmamba_killer_summary.log
SAVETO=runs/cptac_ucec_Immune_class_twodmamba_baseline

echo "=== 2DMAMBA KILLER START $(date -Iseconds) ===" | tee -a "$MASTER_LOG"
echo "--- saveto=$SAVETO ---" | tee -a "$MASTER_LOG"

mkdir -p "$SAVETO"
python -m scripts.run_benchmark \
    --source cptac_ucec \
    --task Immune_class \
    --experiment_type finetune \
    --model_name twodmamba_baseline \
    --model_kwargs_yaml configs/backbones/twodmamba_base.yaml \
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
    --splits_root splits \
    --saveto "$SAVETO" \
    --num_epochs 20 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --scheduler cosine \
    --optimizer AdamW \
    --balanced \
    --save_which_checkpoints best-val-loss \
    --gpu 0 > "$SAVETO/run.log" 2>&1
rc=$?
echo "=== 2DMAMBA KILLER END $(date -Iseconds) exit=$rc ===" | tee -a "$MASTER_LOG"
