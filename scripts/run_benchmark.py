"""Run a single Patho-Bench experiment with a HilbertWSI encoder.

Usage — linprobe (mean pooling baseline):
    python -m scripts.run_benchmark \
        --source cptac_ucec --task Immune_class \
        --experiment_type linprobe \
        --model_name mean-uni_v2 \
        --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
        --splits_root splits --saveto runs/cptac_ucec_mean --gpu 0 --wandb

Usage — finetune (end-to-end MIL training, standard setup):
    python -m scripts.run_benchmark \
        --source cptac_ucec --task Immune_class \
        --experiment_type finetune \
        --model_name abmil_baseline \
        --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
        --splits_root splits --saveto runs/cptac_ucec_abmil \
        --model_kwargs_yaml configs/backbones/abmil_base.yaml --gpu 0 --wandb

    python -m scripts.run_benchmark \
        --source cptac_ucec --task Immune_class \
        --experiment_type finetune \
        --model_name hilbertwsi_hilbert_mamba \
        --patch_features_dir /data/hilbert-wsi/features/CPTAC_UCEC \
        --splits_root splits --saveto runs/cptac_ucec_hilbert_mamba \
        --model_kwargs_yaml configs/backbones/mamba_base.yaml --gpu 0 --wandb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from adapters.patho_bench import inject_val_split, register_hilbert_encoders


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _add_linprobe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pooled_embeddings_dir", default=None,
                   help="Dir of pre-pooled slide-level embeddings (skip pooling step).")
    p.add_argument("--cost", type=float, default=1.0,
                   help="Regularisation C for the logistic regression classifier.")


def _add_coxnet_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--alpha", type=float, default=0.1,
                   help="CoxNet regularization strength.")
    p.add_argument("--l1_ratio", type=float, default=0.5,
                   help="CoxNet L1 ratio (0=Ridge, 1=Lasso).")
    p.add_argument("--num_bootstraps", type=int, default=100,
                   help="Bootstrap iterations for CoxNet C-index CI.")


def _add_finetune_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--bag_size", type=int, default=None,
                   help="Patches per bag. None = all (default).")
    p.add_argument("--gradient_accumulation", type=int, default=1)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "gigapath"])
    p.add_argument("--optimizer", default="AdamW", choices=["AdamW", "gigapath"])
    p.add_argument("--balanced", action="store_true", default=True,
                   help="Use class-balanced CrossEntropy weights. Default True.")
    p.add_argument("--no_balanced", action="store_false", dest="balanced",
                   help="Disable class balancing.")
    p.add_argument("--save_which_checkpoints", default="best-val-loss",
                   help="Checkpoint policy: 'all', 'last-N', 'every-N', 'best-val-loss'. "
                        "Default 'best-val-loss' requires a val split (injected automatically).")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction of each fold's train slides re-marked as val "
                        "(stratified). Set to 0 to disable val injection.")
    p.add_argument("--val_seed", type=int, default=7)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a Patho-Bench task with a HilbertWSI model.")
    p.add_argument("--source", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--experiment_type",
                   choices=["linprobe", "finetune", "coxnet", "retrieval"],
                   default="linprobe")
    p.add_argument("--model_name", default=None)
    p.add_argument("--patch_features_dir", default=None)
    p.add_argument("--splits_root", default="splits")
    p.add_argument("--saveto", required=True)
    p.add_argument("--model_kwargs_yaml", default=None)
    p.add_argument("--combine_slides_per_patient", action="store_true", default=True)
    p.add_argument("--no_combine_slides_per_patient", action="store_false",
                   dest="combine_slides_per_patient",
                   help="Treat each slide independently (required for multi-slide patients).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_seq_len", type=int, default=None,
                   help="Cap sequence length (tiles per slide). Overrides model_kwargs_yaml. "
                        "Recommended: 2048 for CCRCC (slides up to ~15k tiles).")
    p.add_argument("--wandb", action="store_true", default=False,
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", default="hilbertwsi-sequencing",
                   help="WandB project name.")

    _add_linprobe_args(p)
    _add_finetune_args(p)
    _add_coxnet_args(p)

    args = p.parse_args()

    if args.model_name:
        register_hilbert_encoders()

    from patho_bench.SplitFactory import SplitFactory      # type: ignore
    from patho_bench.ExperimentFactory import ExperimentFactory  # type: ignore

    Path(args.saveto).mkdir(parents=True, exist_ok=True)
    Path(args.splits_root).mkdir(parents=True, exist_ok=True)

    model_kwargs = _load_yaml(args.model_kwargs_yaml)
    if args.max_seq_len is not None:
        model_kwargs["max_seq_len"] = args.max_seq_len

    split_path, config_path = SplitFactory.from_hf(
        saveto=args.splits_root,
        source=args.source,
        task=args.task,
    )

    # Inject a stratified val subset so 'best-val-loss' has something to track.
    if args.experiment_type == "finetune" and args.val_frac > 0:
        task_cfg = _load_yaml(config_path)
        sample_col = task_cfg.get("sample_col")
        injected = inject_val_split(
            split_path, label_col=args.task,
            val_frac=args.val_frac, seed=args.val_seed,
            sample_col=sample_col,
        )
        if injected:
            print(f"[run_benchmark] Injected val split ({args.val_frac:.0%}) into {split_path}")

    patch_dirs = [args.patch_features_dir] if args.patch_features_dir else None

    if args.experiment_type == "linprobe":
        pooled_dir: str | None = args.pooled_embeddings_dir
        if pooled_dir is None and args.model_name and patch_dirs:
            pooled_dir = str(Path(args.saveto) / "pooled")

        exp = ExperimentFactory.linprobe(
            split=split_path,
            task_config=config_path,
            pooled_embeddings_dir=pooled_dir,
            patch_embeddings_dirs=patch_dirs,
            model_name=args.model_name,
            model_kwargs=model_kwargs,
            saveto=args.saveto,
            combine_slides_per_patient=args.combine_slides_per_patient,
            gpu=args.gpu,
            cost=args.cost,
        )

    elif args.experiment_type == "finetune":
        if not args.model_name:
            p.error("--model_name is required for finetune.")
        if not patch_dirs:
            p.error("--patch_features_dir is required for finetune.")

        exp = ExperimentFactory.finetune(
            split=split_path,
            task_config=config_path,
            patch_embeddings_dirs=patch_dirs,
            model_name=args.model_name,
            model_kwargs=model_kwargs,
            saveto=args.saveto,
            combine_slides_per_patient=args.combine_slides_per_patient,
            gpu=args.gpu,
            bag_size=args.bag_size,
            base_learning_rate=args.lr,
            gradient_accumulation=args.gradient_accumulation,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            scheduler_type=args.scheduler,
            optimizer_type=args.optimizer,
            balanced=args.balanced,
            save_which_checkpoints=args.save_which_checkpoints,
        )

    elif args.experiment_type == "coxnet":
        if not patch_dirs and not getattr(args, "pooled_embeddings_dir", None):
            p.error("--patch_features_dir or --pooled_embeddings_dir required for coxnet.")
        coxnet_pooled_dir: str | None = args.pooled_embeddings_dir
        if coxnet_pooled_dir is None and args.model_name and patch_dirs:
            coxnet_pooled_dir = str(Path(args.saveto) / "pooled")
        exp = ExperimentFactory.coxnet(
            split=split_path,
            task_config=config_path,
            pooled_embeddings_dir=coxnet_pooled_dir,
            patch_embeddings_dirs=patch_dirs,
            model_name=args.model_name,
            model_kwargs=model_kwargs,
            saveto=args.saveto,
            combine_slides_per_patient=args.combine_slides_per_patient,
            gpu=args.gpu,
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
            num_bootstraps=args.num_bootstraps,
        )

    else:
        raise NotImplementedError(f"experiment_type '{args.experiment_type}' not yet wired up.")

    # Build WandB config from all experiment parameters for reproducibility
    wandb_cfg = dict(
        source=args.source,
        task=args.task,
        experiment_type=args.experiment_type,
        model_name=args.model_name,
        model_kwargs=model_kwargs,
        gpu=args.gpu,
        combine_slides_per_patient=args.combine_slides_per_patient,
    )
    if args.experiment_type == "linprobe":
        wandb_cfg.update(cost=args.cost)
    elif args.experiment_type == "coxnet":
        wandb_cfg.update(alpha=args.alpha, l1_ratio=args.l1_ratio)
    elif args.experiment_type == "finetune":
        wandb_cfg.update(
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            bag_size=args.bag_size,
            gradient_accumulation=args.gradient_accumulation,
            scheduler=args.scheduler,
            optimizer=args.optimizer,
            balanced=args.balanced,
        )

    if args.wandb:
        from adapters.wandb_logging import WandBExperiment
        exp = WandBExperiment(exp, wandb_cfg, project=args.wandb_project)

    exp.train()
    exp.test()


if __name__ == "__main__":
    main()
