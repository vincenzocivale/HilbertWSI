"""WandB logging wrapper for Patho-Bench experiments.

Wraps a FinetuningExperiment or LinprobeExperiment. Monkey-patches the
logging hooks (``log_loss``, ``log_lr``, ``init_loggers``) and the
``_accumulate_preds`` method to capture predictions for confusion-matrix logging.

Usage in run_benchmark.py::

    from adapters.wandb_logging import WandBExperiment
    exp = WandBExperiment(exp, cfg)
    exp.train()
    exp.test()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# Test metrics surfaced to WandB summary (read from test_metrics_summary.json).
_TEST_METRICS = [
    "macro-ovr-auc",
    "acc",
    "bacc",
    "macro-f1",
    "weighted_kappa",
]


def _arch_tag(model_name: str) -> str:
    """Derive a short architecture tag like ``mamba+hilbert`` from a model name."""
    if model_name.startswith("hilbertwsi_"):
        parts = model_name[len("hilbertwsi_"):].split("_")
        backbone = parts[-1]
        ordering = "_".join(parts[:-1])
        return f"{backbone}+{ordering}"
    if model_name.startswith("abmil"):
        return "abmil"
    if model_name.startswith("mean-"):
        return "mean_pool"
    return model_name


class WandBExperiment:
    """Thin wrapper that adds WandB logging to any Patho-Bench experiment.

    Logs per-epoch ``{train,val}/loss`` and ``train/lr``; at test time logs
    the confusion matrix and a curated set of test metrics. Drops smooth_rank
    and other MIL-specific signals.
    """

    def __init__(
        self,
        exp: Any,
        cfg: dict[str, Any],
        project: str = "hilbertwsi-sequencing",
    ) -> None:
        self.exp = exp
        self.cfg = cfg
        self.project = project
        self._run = None
        # Latest captured preds/labels (set by patched _accumulate_preds).
        self._captured: dict[str, tuple[Any, Any]] = {}

    def train(self) -> None:
        import wandb

        source = self.cfg.get("source", "unknown")
        task = self.cfg.get("task", "unknown")
        model_name = self.cfg.get("model_name", "unknown")
        exp_type = self.cfg.get("experiment_type", "finetune")
        arch = _arch_tag(model_name)

        self._run = wandb.init(
            project=self.project,
            name=f"{exp_type}/{arch}/{source}/{task}",
            group=f"{source}/{task}",
            job_type=exp_type,
            tags=[source, task, arch, exp_type],
            config=self.cfg,
            reinit=True,
        )

        if hasattr(self.exp, "log_loss"):
            self._patch_logging_hooks()
        if hasattr(self.exp, "_accumulate_preds"):
            self._patch_accumulate_preds()

        self.exp.train()

    def test(self) -> None:
        self.exp.test()
        if self._run is None:
            return

        import wandb

        # 1. Surface curated test metrics from the summary JSON.
        summary_path = Path(self.exp.results_dir) / "test_metrics_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            log_dict: dict[str, float] = {}
            for key in _TEST_METRICS:
                entry = summary.get(key)
                if isinstance(entry, dict) and "mean" in entry:
                    log_dict[f"test/{key}"] = entry["mean"]
                    if "se" in entry:
                        log_dict[f"test/{key}_se"] = entry["se"]
            if log_dict:
                wandb.log(log_dict)
                for k, v in log_dict.items():
                    self._run.summary[k] = v

        # 2. Aggregated confusion matrix from captured preds/labels.
        cap = self._captured.get("test")
        if cap is not None:
            self._log_confusion_matrix("test", *cap)

        # 3. Attach per-fold confusion matrix PNGs already saved by Patho-Bench.
        results_dir = Path(self.exp.results_dir)
        for png in sorted(results_dir.glob("test_metrics/fold_*/confusion_matrices.png")):
            fold = png.parent.name
            wandb.log({f"test/cm_{fold}": wandb.Image(str(png))})

        self._run.finish()
        self._run = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _patch_logging_hooks(self) -> None:
        """Replace Patho-Bench per-epoch logging methods. Drops smooth_rank."""
        exp = self.exp
        original_init_loggers = exp.init_loggers
        original_log_loss = exp.log_loss
        original_log_lr = exp.log_lr
        wrapper = self

        def _init_loggers(save_dir: str):
            return original_init_loggers(save_dir)

        # Use WandB's auto-incrementing step. `epoch` + `fold` are logged as
        # metric fields so panels can use them as x-axis. Avoid explicit step:
        # otherwise train/val of different epochs collide at the same step
        # because Patho-Bench calls log_loss(train) then log_loss(val) per epoch
        # and a step-after-train scheme leaves val one epoch behind train in
        # the plots.

        def _log_loss(step: int) -> None:
            original_log_loss(step)
            import wandb
            mode = exp.mode
            loss = exp.current_epoch_metrics.get("per_sample_loss")
            if loss is None:
                return
            wandb.log(
                {
                    f"{mode}/loss": loss,
                    f"{mode}/epoch": exp.current_epoch,
                    "fold": exp.current_iter,
                }
            )

        def _log_lr(step: int) -> None:
            original_log_lr(step)
            import wandb
            try:
                lr = exp.scheduler.get_last_lr()[0]
            except Exception:
                lr = None
            if lr is not None:
                wandb.log({"train/lr": lr, "train/epoch": exp.current_epoch,
                           "fold": exp.current_iter})

        exp.init_loggers = _init_loggers
        exp.log_loss = _log_loss
        exp.log_lr = _log_lr
        # Disable smooth_rank logging: replace with a no-op that still calls the original
        # (so internal state stays consistent) but skips WandB write.
        if hasattr(exp, "log_smooth_rank"):
            original_log_smooth_rank = exp.log_smooth_rank

            def _log_smooth_rank(step: int) -> None:
                original_log_smooth_rank(step)

            exp.log_smooth_rank = _log_smooth_rank

    def _patch_accumulate_preds(self) -> None:
        """Capture labels/preds emitted by Patho-Bench's per-fold eval pass."""
        exp = self.exp
        original = exp._accumulate_preds
        wrapper = self

        def _accumulate(dataloader, model):
            labels, preds = original(dataloader, model)
            mode = getattr(exp, "mode", None) or "test"
            buf = wrapper._captured.setdefault(mode, ([], []))
            buf[0].append(labels)
            buf[1].append(preds)
            return labels, preds

        # Reset buffers at start of each split's eval (test/val).
        original_eval = getattr(exp, "_eval", None)
        if original_eval is not None:
            def _eval(split: str):
                wrapper._captured[split] = ([], [])
                return original_eval(split)
            exp._eval = _eval

        exp._accumulate_preds = _accumulate

    def _log_confusion_matrix(self, split: str, labels_list, preds_list) -> None:
        """Aggregate per-fold (labels, preds) and log a CM panel to WandB."""
        import wandb

        if not labels_list or not preds_list:
            return
        try:
            y_true = np.concatenate([np.asarray(x).reshape(-1) for x in labels_list])
            preds_arr = [np.asarray(p) for p in preds_list]
            # preds: (N, C) probability arrays -> argmax to class label.
            if preds_arr[0].ndim == 2:
                y_pred = np.concatenate([p.argmax(axis=1) for p in preds_arr])
            else:
                y_pred = np.concatenate(preds_arr)
        except Exception:
            return

        wandb.log(
            {
                f"{split}/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true.tolist(),
                    preds=y_pred.tolist(),
                    class_names=None,
                )
            }
        )
