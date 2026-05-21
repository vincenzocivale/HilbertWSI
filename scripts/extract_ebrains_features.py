"""Extract UNI2-h features for EBRAINS slides using Trident.

Run this AFTER obtaining EBRAINS slide access from https://ebrains.eu/data

Usage:
    conda activate hilbert-wsi-clean
    python -m scripts.extract_ebrains_features \
        --slides_dir /path/to/ebrains/slides \
        --saveto /data/hilbert-wsi/features/EBRAINS \
        --gpu 0

After extraction, run experiments:
    python -m scripts.ablation_orderings \
        --source ebrains --task diagnosis \
        --experiment_type finetune \
        --backbone mamba \
        --patch_features_dir /data/hilbert-wsi/features/EBRAINS \
        --splits_root splits \
        --model_kwargs_yaml configs/backbones/mamba_base.yaml \
        --saveto runs_v3/ebrains_diagnosis_ablation
"""

from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slides_dir", required=True, help="Dir with EBRAINS .svs/.ndpi slides")
    p.add_argument("--saveto", default="/data/hilbert-wsi/features/EBRAINS")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--magnification", type=float, default=20.0)
    p.add_argument("--batch_limit", type=int, default=256)
    args = p.parse_args()

    from trident import Processor
    from trident.patch_encoder_models.load import encoder_factory

    job_dir = str(Path(args.saveto) / "_trident_job")
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(args.saveto, exist_ok=True)

    device = f"cuda:{args.gpu}"

    proc = Processor(
        job_dir=job_dir,
        wsi_source=args.slides_dir,
        wsi_ext=[".svs", ".ndpi", ".tif", ".scn"],
        search_nested=True,
    )

    print("Step 1: Tissue segmentation...")
    proc.run_segmentation_job()

    print("Step 2: Patch coordinate extraction...")
    coords_dir = proc.run_patching_job(
        patch_size=args.patch_size,
        magnification=args.magnification,
    )

    print("Step 3: UNI2-h feature extraction...")
    encoder = encoder_factory("uni_v2")

    features_dir = proc.run_patch_feature_extraction_job(
        coords_dir=coords_dir,
        patch_encoder=encoder,
        device=device,
        saveas="h5",
        batch_limit=args.batch_limit,
        saveto=args.saveto,
    )
    print(f"Features saved to: {features_dir}")

    n_files = len(glob.glob(os.path.join(features_dir, "*.h5")))
    print(f"Extracted features for {n_files} slides")


if __name__ == "__main__":
    main()
