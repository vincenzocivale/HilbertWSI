"""Automated sweep: runs all architectures on a given dataset/task.

Architecture grid
-----------------
Baselines (fixed pooling, no ordering):
  - mean-uni_v2          linprobe   (no trainable params, upper bound of lazy baseline)
  - abmil_baseline       finetune   (classic attention MIL)

HilbertWSI — all orderings × all backbones:
  orderings : hilbert, zorder, snake, moore, peano, random_perm, similarity
  backbones : mamba, transformer
  → 14 finetune runs

Total: 16 runs per dataset/task.

Usage
-----
    python -m scripts.run_sweep \\
        --source cptac_ucec \\
        --task Immune_class \\
        --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \\
        --splits_root splits \\
        --runs_root runs \\
        --gpu 0 \\
        --wandb

    # Dry-run (print commands, do not execute):
    python -m scripts.run_sweep ... --dry_run

    # Skip already-done runs automatically (default on).
    # Force re-run: --no_skip
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

ORDERINGS = ["hilbert", "zorder", "snake", "moore", "peano", "random", "similarity"]
BACKBONES = ["mamba", "transformer"]

BACKBONE_CONFIGS = {
    "mamba": "configs/backbones/mamba_base.yaml",
    "transformer": "configs/backbones/transformer_base.yaml",
}

# Standalone slide encoders (no ordering applied — used as controls in the
# killer experiment to isolate the contribution of space-filling-curve
# ordering from the contribution of the architecture itself).
SLIDE_ENCODER_BASELINES = [
    ("transmil_baseline", "configs/backbones/transmil_base.yaml"),
    ("mambamil_baseline", "configs/backbones/mambamil_base.yaml"),
]

_FINETUNE_DEFAULTS = dict(
    num_epochs=20,
    lr=1e-4,
    weight_decay=1e-5,
    scheduler="cosine",
    optimizer="AdamW",
    # Val split is injected automatically by run_benchmark.py so best-val-loss
    # has something to track.
    save_which_checkpoints="best-val-loss",
    balanced=True,
)


def _all_experiments(args: argparse.Namespace) -> list[dict]:
    """Return ordered list of experiment configs."""
    runs_root = Path(args.runs_root)
    base = dict(
        source=args.source,
        task=args.task,
        patch_features_dir=args.patch_features_dir,
        splits_root=args.splits_root,
        gpu=args.gpu,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        **_FINETUNE_DEFAULTS,
    )

    exps = []

    # --- baselines ---
    exps.append({
        **base,
        "experiment_type": "linprobe",
        "model_name": "mean-uni_v2",
        "saveto": str(runs_root / f"{args.source}_{args.task}_mean-uni_v2"),
    })
    exps.append({
        **base,
        "experiment_type": "finetune",
        "model_name": "abmil_baseline",
        "model_kwargs_yaml": "configs/backbones/abmil_base.yaml",
        "saveto": str(runs_root / f"{args.source}_{args.task}_abmil"),
    })
    # --- killer-experiment controls: standalone slide encoders, no ordering ---
    for model_name, cfg_path in SLIDE_ENCODER_BASELINES:
        exps.append({
            **base,
            "experiment_type": "finetune",
            "model_name": model_name,
            "model_kwargs_yaml": cfg_path,
            "saveto": str(runs_root / f"{args.source}_{args.task}_{model_name}"),
        })

    # --- HilbertWSI grid ---
    for backbone in BACKBONES:
        for ordering in ORDERINGS:
            model_name = f"hilbertwsi_{ordering}_{backbone}"
            exps.append({
                **base,
                "experiment_type": "finetune",
                "model_name": model_name,
                "model_kwargs_yaml": BACKBONE_CONFIGS[backbone],
                "saveto": str(runs_root / f"{args.source}_{args.task}_{model_name}"),
            })

    return exps


# ---------------------------------------------------------------------------
# CLI builder
# ---------------------------------------------------------------------------

def _build_cmd(cfg: dict) -> list[str]:
    """Convert experiment config dict → run_benchmark.py argv."""
    cmd = [
        sys.executable, "-m", "scripts.run_benchmark",
        "--source",          cfg["source"],
        "--task",            cfg["task"],
        "--experiment_type", cfg["experiment_type"],
        "--model_name",      cfg["model_name"],
        "--patch_features_dir", cfg["patch_features_dir"],
        "--splits_root",     cfg["splits_root"],
        "--saveto",          cfg["saveto"],
        "--gpu",             str(cfg["gpu"]),
    ]
    if cfg.get("model_kwargs_yaml"):
        cmd += ["--model_kwargs_yaml", cfg["model_kwargs_yaml"]]
    if cfg.get("wandb"):
        cmd += ["--wandb", "--wandb_project", cfg["wandb_project"]]

    if cfg["experiment_type"] == "finetune":
        cmd += [
            "--num_epochs",             str(cfg["num_epochs"]),
            "--lr",                     str(cfg["lr"]),
            "--weight_decay",           str(cfg["weight_decay"]),
            "--scheduler",              cfg["scheduler"],
            "--optimizer",              cfg["optimizer"],
            "--save_which_checkpoints", cfg["save_which_checkpoints"],
        ]
        if cfg.get("balanced", False):
            cmd.append("--balanced")
        else:
            cmd.append("--no_balanced")
    elif cfg["experiment_type"] == "linprobe":
        pass  # default cost=1.0

    return cmd


def _is_done(saveto: str) -> bool:
    return (Path(saveto) / "test_metrics_summary.json").exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Full architecture sweep on one dataset/task.")
    p.add_argument("--source", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--patch_features_dir", required=True)
    p.add_argument("--splits_root", default="splits")
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", default="hilbertwsi-sequencing")
    p.add_argument("--dry_run", action="store_true", default=False,
                   help="Print commands without executing.")
    p.add_argument("--no_skip", action="store_true", default=False,
                   help="Re-run experiments even if already completed.")
    # Subset flags for quick testing
    p.add_argument("--only_baselines", action="store_true", default=False,
                   help="Run only mean + ABMIL baselines.")
    p.add_argument("--only_backbone", default=None, choices=BACKBONES + [None],
                   help="Run only experiments with this backbone.")
    p.add_argument("--only_ordering", default=None, choices=ORDERINGS + [None],
                   help="Run only experiments with this ordering.")
    args = p.parse_args()

    experiments = _all_experiments(args)

    # Apply subset filters
    if args.only_baselines:
        experiments = [e for e in experiments if "hilbertwsi" not in e["model_name"]]
    if args.only_backbone:
        experiments = [e for e in experiments
                       if "hilbertwsi" not in e["model_name"]
                       or e["model_name"].endswith(args.only_backbone)]
    if args.only_ordering:
        experiments = [e for e in experiments
                       if "hilbertwsi" not in e["model_name"]
                       or f"_{args.only_ordering}_" in e["model_name"]]

    print(f"\n{'=' * 70}")
    print(f"Sweep: {args.source}/{args.task}  |  {len(experiments)} experiments")
    print(f"{'=' * 70}\n")

    done, skipped, failed = 0, 0, 0

    for i, cfg in enumerate(experiments, 1):
        label = f"[{i}/{len(experiments)}] {cfg['model_name']} ({cfg['experiment_type']})"

        if not args.no_skip and _is_done(cfg["saveto"]):
            print(f"SKIP  {label}  (results exist)")
            skipped += 1
            continue

        cmd = _build_cmd(cfg)
        print(f"\nRUN   {label}")
        print(f"      saveto: {cfg['saveto']}")

        if args.dry_run:
            print("      CMD:", " ".join(cmd))
            continue

        result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        if result.returncode == 0:
            print(f"DONE  {label}")
            done += 1
        else:
            print(f"FAIL  {label}  (exit {result.returncode}) — continuing sweep")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"Sweep complete: {done} done, {skipped} skipped, {failed} failed")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
