#!/usr/bin/env bash
# Run full CPTAC-BRCA Immune_class benchmark: all orderings x backbones + all baselines.
#
# Usage:
#     bash scripts/run_brca_immune_class.sh [--dry-run]
#
# Outputs:
#     runs/cptac_brca_Immune_class_<model>/run.log   per-run stdout+stderr
#     runs/brca_immune_class_summary.log              master log with timestamps + exit codes
#
# On dry-run: prints all commands without executing.

set -u
cd "$(dirname "$0")/.."

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

source /home/oem/miniconda3/etc/profile.d/conda.sh
conda activate hilbert-wsi-clean

MASTER_LOG=runs/brca_immune_class_summary.log
mkdir -p runs
echo "=== BRCA IMMUNE_CLASS START $(date -Iseconds) ===" | tee -a "$MASTER_LOG"

COMMON_FLAGS=(
    --source cptac_brca
    --task Immune_class
    --experiment_type finetune
    --patch_features_dir /data/hilbert-wsi/features/CPTAC_BRCA
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

PASS=0
FAIL=0
SKIP=0

run_one() {
    local label="$1"
    local saveto="$2"
    local cfg="$3"
    local model="$4"

    # Skip already-completed runs (results.json present).
    if [[ -f "$saveto/results.json" ]]; then
        echo "--- [$label] SKIP  $(date -Iseconds) already done ---" | tee -a "$MASTER_LOG"
        SKIP=$((SKIP + 1))
        return 0
    fi

    echo "--- [$label] START $(date -Iseconds) saveto=$saveto ---" | tee -a "$MASTER_LOG"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    [DRY-RUN] python -m scripts.run_benchmark --model_name $model --model_kwargs_yaml $cfg --saveto $saveto ${COMMON_FLAGS[*]}"
        return 0
    fi

    mkdir -p "$saveto"
    python -m scripts.run_benchmark \
        --model_name "$model" \
        --model_kwargs_yaml "$cfg" \
        --saveto "$saveto" \
        "${COMMON_FLAGS[@]}" \
        > "$saveto/run.log" 2>&1
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "--- [$label] OK    $(date -Iseconds) exit=0 ---" | tee -a "$MASTER_LOG"
        PASS=$((PASS + 1))
    else
        echo "--- [$label] FAIL  $(date -Iseconds) exit=$rc  (tail: $saveto/run.log) ---" | tee -a "$MASTER_LOG"
        tail -5 "$saveto/run.log" | sed 's/^/    /' | tee -a "$MASTER_LOG"
        FAIL=$((FAIL + 1))
    fi
}

run_linprobe() {
    local label="$1"
    local saveto="$2"

    if [[ -f "$saveto/results.json" ]]; then
        echo "--- [$label] SKIP  $(date -Iseconds) already done ---" | tee -a "$MASTER_LOG"
        SKIP=$((SKIP + 1))
        return 0
    fi

    echo "--- [$label] START $(date -Iseconds) saveto=$saveto ---" | tee -a "$MASTER_LOG"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    [DRY-RUN] python -m scripts.run_benchmark --experiment_type linprobe --model_name mean-uni_v2 --patch_features_dir /data/hilbert-wsi/features/CPTAC_BRCA --source cptac_brca --task Immune_class --splits_root splits --saveto $saveto"
        return 0
    fi

    mkdir -p "$saveto"
    python -m scripts.run_benchmark \
        --source cptac_brca \
        --task Immune_class \
        --experiment_type linprobe \
        --model_name mean-uni_v2 \
        --patch_features_dir /data/hilbert-wsi/features/CPTAC_BRCA \
        --splits_root splits \
        --saveto "$saveto" \
        --gpu 0 \
        > "$saveto/run.log" 2>&1
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "--- [$label] OK    $(date -Iseconds) exit=0 ---" | tee -a "$MASTER_LOG"
        PASS=$((PASS + 1))
    else
        echo "--- [$label] FAIL  $(date -Iseconds) exit=$rc  (tail: $saveto/run.log) ---" | tee -a "$MASTER_LOG"
        tail -5 "$saveto/run.log" | sed 's/^/    /' | tee -a "$MASTER_LOG"
        FAIL=$((FAIL + 1))
    fi
}

# ── Ordering × Transformer ──────────────────────────────────────────────────
run_one "snake_transformer"   runs/cptac_brca_Immune_class_snake_transformer   configs/backbones/transformer_base.yaml hilbertwsi_snake_transformer
run_one "hilbert_transformer" runs/cptac_brca_Immune_class_hilbert_transformer configs/backbones/transformer_base.yaml hilbertwsi_hilbert_transformer
run_one "zorder_transformer"  runs/cptac_brca_Immune_class_zorder_transformer  configs/backbones/transformer_base.yaml hilbertwsi_zorder_transformer
run_one "peano_transformer"   runs/cptac_brca_Immune_class_peano_transformer   configs/backbones/transformer_base.yaml hilbertwsi_peano_transformer
run_one "moore_transformer"   runs/cptac_brca_Immune_class_moore_transformer   configs/backbones/transformer_base.yaml hilbertwsi_moore_transformer

# ── Ordering × Mamba ────────────────────────────────────────────────────────
run_one "hilbert_mamba"   runs/cptac_brca_Immune_class_hilbert_mamba   configs/backbones/mamba_base.yaml hilbertwsi_hilbert_mamba
run_one "snake_mamba"     runs/cptac_brca_Immune_class_snake_mamba     configs/backbones/mamba_base.yaml hilbertwsi_snake_mamba
run_one "zorder_mamba"    runs/cptac_brca_Immune_class_zorder_mamba    configs/backbones/mamba_base.yaml hilbertwsi_zorder_mamba
run_one "peano_mamba"     runs/cptac_brca_Immune_class_peano_mamba     configs/backbones/mamba_base.yaml hilbertwsi_peano_mamba
run_one "moore_mamba"     runs/cptac_brca_Immune_class_moore_mamba     configs/backbones/mamba_base.yaml hilbertwsi_moore_mamba
run_one "similarity_mamba" runs/cptac_brca_Immune_class_similarity_mamba configs/backbones/mamba_base.yaml hilbertwsi_similarity_mamba

# ── Random controls ─────────────────────────────────────────────────────────
run_one "random_transformer" runs/cptac_brca_Immune_class_random_transformer configs/backbones/transformer_base.yaml hilbertwsi_random_transformer
run_one "random_mamba"       runs/cptac_brca_Immune_class_random_mamba       configs/backbones/mamba_base.yaml       hilbertwsi_random_mamba

# ── No-ordering baselines ───────────────────────────────────────────────────
run_one "transmil"    runs/cptac_brca_Immune_class_transmil_baseline  configs/backbones/transmil_base.yaml  transmil_baseline
run_one "mambamil"    runs/cptac_brca_Immune_class_mambamil_baseline  configs/backbones/mambamil_base.yaml  mambamil_baseline
run_one "abmil"       runs/cptac_brca_Immune_class_abmil_baseline     configs/backbones/abmil_base.yaml     abmil_baseline
run_one "clam_sb"     runs/cptac_brca_Immune_class_clam_sb_baseline   configs/backbones/clam_sb_base.yaml   clam_sb_baseline
run_one "clam_mb"     runs/cptac_brca_Immune_class_clam_mb_baseline   configs/backbones/clam_mb_base.yaml   clam_mb_baseline
run_one "dsmil"       runs/cptac_brca_Immune_class_dsmil_baseline     configs/backbones/dsmil_base.yaml     dsmil_baseline
run_one "twodmamba"   runs/cptac_brca_Immune_class_twodmamba_baseline configs/backbones/twodmamba_base.yaml twodmamba_baseline

# ── Mean-pool linprobe (lower bound) ────────────────────────────────────────
run_linprobe "mean_linprobe" runs/cptac_brca_Immune_class_mean_linprobe

echo "=== BRCA IMMUNE_CLASS DONE $(date -Iseconds)  pass=$PASS fail=$FAIL skip=$SKIP ===" | tee -a "$MASTER_LOG"
