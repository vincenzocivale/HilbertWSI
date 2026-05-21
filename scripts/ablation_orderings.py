"""Sweep all ordering schemes on a single Patho-Bench task, same backbone.

Usage:
    python -m scripts.ablation_orderings \
        --source panda \
        --task isup_grade \
        --backbone mamba \
        --patch_features_dir /data/hilbert-wsi/features/PANDA/h5_files \
        --splits_root /home/oem/HilbertWSI/splits \
        --model_kwargs_yaml configs/backbones/mamba_base.yaml \
        --saveto runs/ablation_panda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hilbert_wsi.ordering import available_orderings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--experiment_type", default="linprobe",
                   choices=["linprobe", "finetune", "coxnet", "retrieval"])
    p.add_argument("--backbone", default="mamba")
    p.add_argument("--patch_features_dir", required=True)
    p.add_argument("--splits_root", default="splits")
    p.add_argument("--saveto", required=True)
    p.add_argument("--model_kwargs_yaml", default=None)
    p.add_argument("--orderings", nargs="+", default=available_orderings(),
                   help="Subset of orderings to run (default: all).")
    p.add_argument("--balanced", action="store_true", default=True,
                   help="Forward --balanced to run_benchmark (default True).")
    p.add_argument("--no_balanced", action="store_false", dest="balanced")
    p.add_argument("--save_which_checkpoints", default="best-val-loss")
    p.add_argument("--num_epochs", type=int, default=10)
    args = p.parse_args()

    saveto = Path(args.saveto)
    saveto.mkdir(parents=True, exist_ok=True)

    for ordering in args.orderings:
        model_name = f"hilbertwsi_{ordering}_{args.backbone}"
        run_dir = saveto / ordering
        cmd = [
            sys.executable, "-m", "scripts.run_benchmark",
            "--source", args.source,
            "--task", args.task,
            "--experiment_type", args.experiment_type,
            "--model_name", model_name,
            "--patch_features_dir", args.patch_features_dir,
            "--splits_root", args.splits_root,
            "--saveto", str(run_dir),
        ]
        if args.model_kwargs_yaml:
            cmd += ["--model_kwargs_yaml", args.model_kwargs_yaml]
        if args.experiment_type == "finetune":
            cmd += ["--save_which_checkpoints", args.save_which_checkpoints]
            cmd += ["--num_epochs", str(args.num_epochs)]
            cmd.append("--balanced" if args.balanced else "--no_balanced")
        print(f"\n=== Ordering: {ordering} | model: {model_name} ===")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
