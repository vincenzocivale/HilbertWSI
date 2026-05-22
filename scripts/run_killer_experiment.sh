#!/usr/bin/env bash
# Launch the 3 killer-experiment runs on CPTAC-UCEC Immune_class sequentially.
# Use `;` (not `&&`) so a single run failure does not block the rest.
#
# Usage:
#     bash scripts/run_killer_experiment.sh
#
# Outputs:
#     runs/<saveto>/run.log               per-run stdout+stderr
#     runs/killer_experiment_summary.log  master log with timestamps + exit codes

set -u
cd "$(dirname "$0")/.."

source /home/oem/miniconda3/etc/profile.d/conda.sh
conda activate hilbert-wsi-clean

MASTER_LOG=runs/killer_experiment_summary.log
echo "=== KILLER EXPERIMENT START $(date -Iseconds) ===" | tee -a "$MASTER_LOG"

COMMON_FLAGS=(
    --source cptac_ucec
    --task Immune_class
    --experiment_type finetune
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC
    --splits_root splits
    --num_epochs 20
    --lr 1e-4
    --weight_decay 1e-5
    --scheduler cosine
    --optimizer AdamW
    --balanced
    --save_which_checkpoints best-val-loss
    --gpu 0
)

run_one() {
    local name="$1"; shift
    local saveto="$1"; shift
    local cfg="$1"; shift
    local model="$1"; shift

    echo "--- [$name] START $(date -Iseconds) saveto=$saveto ---" | tee -a "$MASTER_LOG"
    mkdir -p "$saveto"
    python -m scripts.run_benchmark \
        --model_name "$model" \
        --model_kwargs_yaml "$cfg" \
        --saveto "$saveto" \
        "${COMMON_FLAGS[@]}" > "$saveto/run.log" 2>&1
    local rc=$?
    echo "--- [$name] END   $(date -Iseconds) exit=$rc ---" | tee -a "$MASTER_LOG"
}

# 1) Transformer-without-ordering control (TransMIL, ~13M params)
run_one "transmil" \
    runs/cptac_ucec_Immune_class_transmil_baseline \
    configs/backbones/transmil_base.yaml \
    transmil_baseline

# 2) Mamba-without-ordering control (MambaMIL, ~20M params)
run_one "mambamil" \
    runs/cptac_ucec_Immune_class_mambamil_baseline \
    configs/backbones/mambamil_base.yaml \
    mambamil_baseline

# 3) Random-ordering control (Random+HilbertWSI Transformer, ~13M params, fixed per-slide seed)
run_one "random_transformer" \
    runs/cptac_ucec_Immune_class_hilbertwsi_random_transformer \
    configs/backbones/transformer_base.yaml \
    hilbertwsi_random_transformer

echo "=== KILLER EXPERIMENT DONE $(date -Iseconds) ===" | tee -a "$MASTER_LOG"
