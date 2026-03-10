"""
End-to-end replication pipeline.

Orchestrates:
  1. Data loading and CV splitting
  2. Per-fold DeepSMOTE and BAGAN training + evaluation
  3. Results aggregation and reporting

The Results object holds all per-fold metrics and provides
display/export helpers for the Jupyter notebook.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"figure.dpi": 120})

from config import Config
from data import MNISTDataManager
from deepsmote import DeepSMOTE
from bagan import BAGAN
from classifier import ClassifierTrainer
from metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────
#  Results container
# ─────────────────────────────────────────────────────────────────

class Results:
    """Stores per-fold metrics and provides comparison-table helpers."""

    # Paper rank-order claim: DeepSMOTE > BAGAN on ACSA and GM (Table II, MNIST)
    PAPER_CLAIM = "DeepSMOTE > BAGAN  on ACSA and GM  (paper Table II, imbalanced test)"

    def __init__(self):
        self.folds: dict[str, list[dict]] = {
            "DeepSMOTE": [],
            "BAGAN":     [],
        }
        self.runtimes: dict[str, float] = {}

    def add_fold(self, method: str, metrics: dict) -> None:
        self.folds[method].append(metrics)

    def summary(self, method: str) -> dict:
        """Mean ± std across folds for a given method."""
        folds = self.folds[method]
        if not folds:
            return {}
        keys = folds[0].keys()
        return {
            k: {
                "mean": float(np.mean([f[k] for f in folds])),
                "std":  float(np.std( [f[k] for f in folds])),
            }
            for k in keys
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame with all fold results."""
        rows = []
        for method, folds in self.folds.items():
            for fold_idx, metrics in enumerate(folds):
                rows.append({
                    "method": method,
                    "fold":   fold_idx + 1,
                    **{k: round(v * 100, 2) for k, v in metrics.items()},
                })
        return pd.DataFrame(rows)

    def comparison_table(self) -> pd.DataFrame:
        """
        Build the paper-style comparison table (mean ± std, %).

        Columns: Method | ACSA | GM | FM
        """
        rows = []
        for method in ["DeepSMOTE", "BAGAN"]:
            s = self.summary(method)
            if not s:
                continue
            row = {"Method": method}
            for metric in ["acsa", "gm", "fm"]:
                m   = s[metric]["mean"] * 100
                std = s[metric]["std"]  * 100
                row[metric.upper()] = f"{m:.2f} ± {std:.2f}"
            rows.append(row)
        df = pd.DataFrame(rows).set_index("Method")
        return df

    def display_table(self) -> None:
        """Print the comparison table to stdout."""
        df = self.comparison_table()
        print("\n" + "=" * 60)
        print("  Replication Results  (MNIST, imbalanced test, 5-fold CV)")
        print("  Metrics in %  (mean ± std across folds)")
        print("=" * 60)
        print(df.to_string())
        print("-" * 60)
        print(f"  Paper claim: {self.PAPER_CLAIM}")
        ds = self.summary("DeepSMOTE")
        bg = self.summary("BAGAN")
        if ds and bg:
            for m in ["acsa", "gm"]:
                direction = "✓" if ds[m]["mean"] > bg[m]["mean"] else "✗"
                print(f"  Replicated {m.upper()}: DeepSMOTE {ds[m]['mean']*100:.2f}% "
                      f"vs BAGAN {bg[m]['mean']*100:.2f}%  [{direction}]")
        print("=" * 60 + "\n")

    def plot_per_fold(self, figsize=(10, 4)) -> plt.Figure:
        """Bar chart of per-fold ACSA, GM, FM for both methods."""
        df = self.to_dataframe()
        metrics = ["ACSA", "GM", "FM"]
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
        colors = {"DeepSMOTE": "#2196F3", "BAGAN": "#FF9800"}

        n_folds = max(len(self.folds["DeepSMOTE"]),
                      len(self.folds["BAGAN"]))
        x = np.arange(n_folds)
        w = 0.35

        for ax, metric in zip(axes, metrics):
            for i, method in enumerate(["DeepSMOTE", "BAGAN"]):
                vals = df[df["method"] == method][metric].values
                ax.bar(x + (i - 0.5) * w, vals, w,
                       label=method, color=colors[method], alpha=0.85)
            ax.set_title(metric)
            ax.set_xlabel("Fold")
            ax.set_xticks(x); ax.set_xticklabels([f"F{i+1}" for i in range(n_folds)])
            ax.set_ylabel("Score (%)")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Per-fold Results — MNIST Replication (Experiment 1, Table II)",
                     fontsize=11)
        plt.tight_layout()
        return fig

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"folds": self.folds, "runtimes": self.runtimes}, f, indent=2)
        print(f"Results saved → {path}")

    @classmethod
    def load_json(cls, path: str) -> "Results":
        with open(path) as f:
            data = json.load(f)
        r = cls()
        r.folds    = data["folds"]
        r.runtimes = data.get("runtimes", {})
        return r


# ─────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────

class ReplicationPipeline:
    """
    Orchestrates the full replication experiment.

    Usage:
        pipeline = ReplicationPipeline(config)
        results  = pipeline.run()           # runs both methods, all folds
        results.display_table()
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self._log(f"Device: {self.device}")

        self.data_mgr = MNISTDataManager(cfg.data)
        self.clf_trainer = ClassifierTrainer(
            cfg.classifier, cfg.data.n_classes, self.device
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, save_path: str | None = None) -> Results:
        """
        Run the full experiment (DeepSMOTE + BAGAN, 5-fold CV).

        Args:
            save_path: optional path to save Results as JSON

        Returns:
            Results object with all fold metrics
        """
        self._log("Loading MNIST and creating imbalanced dataset …")
        X, y = self.data_mgr.get_imbalanced_dataset()
        splits = self.data_mgr.get_cv_splits(X, y)
        self._log(
            f"Dataset: {len(X)} samples, "
            f"class counts: {np.bincount(y).tolist()}"
        )

        results = Results()

        # Run DeepSMOTE
        t0 = time.time()
        self._log("\n── DeepSMOTE (" + str(self.cfg.data.n_folds) + "-fold CV) ──")
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self._log(f"  Fold {fold_idx + 1}/{self.cfg.data.n_folds} …",
                      end=" ")
            metrics = self._run_deepsmote_fold(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )
            results.add_fold("DeepSMOTE", metrics)
            self._log(
                f"ACSA={metrics['acsa']*100:.1f}%  "
                f"GM={metrics['gm']*100:.1f}%  "
                f"FM={metrics['fm']*100:.1f}%"
            )
        results.runtimes["DeepSMOTE"] = time.time() - t0

        # Run BAGAN
        t0 = time.time()
        self._log("\n── BAGAN (" + str(self.cfg.data.n_folds) + "-fold CV) ──")
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self._log(f"  Fold {fold_idx + 1}/{self.cfg.data.n_folds} …",
                      end=" ")
            metrics = self._run_bagan_fold(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )
            results.add_fold("BAGAN", metrics)
            self._log(
                f"ACSA={metrics['acsa']*100:.1f}%  "
                f"GM={metrics['gm']*100:.1f}%  "
                f"FM={metrics['fm']*100:.1f}%"
            )
        results.runtimes["BAGAN"] = time.time() - t0

        self._log("\n── Done ──")
        results.display_table()

        if save_path:
            results.save_json(save_path)

        return results

    def run_deepsmote_only(self, save_path: str | None = None) -> Results:
        """Run only the DeepSMOTE arm of the experiment."""
        self._log("Loading MNIST …")
        X, y = self.data_mgr.get_imbalanced_dataset()
        splits = self.data_mgr.get_cv_splits(X, y)

        results = Results()
        t0 = time.time()
        self._log(f"\n── DeepSMOTE ({self.cfg.data.n_folds}-fold CV) ──")
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self._log(f"  Fold {fold_idx + 1}/{self.cfg.data.n_folds} …", end=" ")
            metrics = self._run_deepsmote_fold(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )
            results.add_fold("DeepSMOTE", metrics)
            self._log(f"ACSA={metrics['acsa']*100:.1f}%  GM={metrics['gm']*100:.1f}%  FM={metrics['fm']*100:.1f}%")
        results.runtimes["DeepSMOTE"] = time.time() - t0
        if save_path:
            results.save_json(save_path)
        return results

    def run_bagan_only(self, save_path: str | None = None) -> Results:
        """Run only the BAGAN arm of the experiment."""
        self._log("Loading MNIST …")
        X, y = self.data_mgr.get_imbalanced_dataset()
        splits = self.data_mgr.get_cv_splits(X, y)

        results = Results()
        t0 = time.time()
        self._log(f"\n── BAGAN ({self.cfg.data.n_folds}-fold CV) ──")
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self._log(f"  Fold {fold_idx + 1}/{self.cfg.data.n_folds} …", end=" ")
            metrics = self._run_bagan_fold(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )
            results.add_fold("BAGAN", metrics)
            self._log(f"ACSA={metrics['acsa']*100:.1f}%  GM={metrics['gm']*100:.1f}%  FM={metrics['fm']*100:.1f}%")
        results.runtimes["BAGAN"] = time.time() - t0
        if save_path:
            results.save_json(save_path)
        return results

    # ------------------------------------------------------------------
    # Per-fold runners
    # ------------------------------------------------------------------

    def _run_deepsmote_fold(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
    ) -> dict:
        torch.manual_seed(self.cfg.data.random_seed)

        ds = DeepSMOTE(self.cfg.deepsmote, self.device)
        ds.train(X_train, y_train, verbose=self.cfg.verbose)

        X_bal, y_bal = ds.generate_balanced(X_train, y_train)
        model = self.clf_trainer.train(X_bal, y_bal, verbose=self.cfg.verbose)
        preds = self.clf_trainer.predict(model, X_test)

        return compute_metrics(y_test, preds)

    def _run_bagan_fold(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
    ) -> dict:
        torch.manual_seed(self.cfg.data.random_seed)

        bagan = BAGAN(self.cfg.bagan, self.cfg.data.n_classes, self.device)
        bagan.train(X_train, y_train, verbose=self.cfg.verbose)

        X_bal, y_bal = bagan.generate_balanced(X_train, y_train)
        model = self.clf_trainer.train(X_bal, y_bal, verbose=self.cfg.verbose)
        preds = self.clf_trainer.predict(model, X_test)

        return compute_metrics(y_test, preds)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _log(self, msg: str, **kwargs) -> None:
        if self.cfg.verbose:
            print(msg, **kwargs)
