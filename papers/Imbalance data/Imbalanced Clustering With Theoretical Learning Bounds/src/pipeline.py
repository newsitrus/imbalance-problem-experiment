"""
pipeline.py — End-to-end experiment pipeline for ICTLB replication.

Usage
-----
    from src import ExperimentPipeline
    pipeline = ExperimentPipeline("config.yaml")
    results  = pipeline.run()          # returns raw per-run DataFrame
    summary  = pipeline.get_summary(results)
    paper_cmp = pipeline.compare_with_paper(summary)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .algorithms import METHOD_DISPLAY_NAMES, get_clusterer
from .data import DATASETS, load_dataset
from .metrics import compute_all_metrics


class ExperimentPipeline:
    """Orchestrates the full experiment: data → clustering → metrics → output.

    Parameters
    ----------
    config_path : str | Path
        Path to config.yaml (absolute or relative to this file's directory).
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            # Resolve relative to the caller's working directory
            config_path = Path.cwd() / config_path
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Absolute paths relative to config file's parent directory
        base = config_path.parent
        self.data_dir    = base / self.cfg["paths"]["data_dir"]
        self.results_dir = base / self.cfg["paths"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.datasets   = self.cfg["experiment"]["datasets"]
        self.methods    = self.cfg["experiment"]["methods"]
        self.n_runs     = self.cfg["experiment"]["n_runs"]
        self.base_seed  = self.cfg["experiment"]["random_seed"]
        self.algo_cfg   = self.cfg["algorithm"]
        self.paper_cfg  = self.cfg.get("paper_results", {})

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """Run the full experiment.

        For each dataset × method × run:
          1. Load and L2-normalise dataset (cached after first download)
          2. Initialise clusterer with paper-matched settings
          3. fit_predict → cluster labels
          4. Compute ACC, F-score, Recall, DCV
          5. Record timing

        Returns
        -------
        pd.DataFrame
            Raw per-run results.  Columns:
            dataset, method, run, ACC, F-score, Recall, DCV, time_s
        Also saves:
            results/raw_results.csv
            results/summary.csv
        """
        if verbose:
            total = len(self.datasets) * len(self.methods) * self.n_runs
            print(f"Starting experiment: {len(self.datasets)} datasets × "
                  f"{len(self.methods)} methods × {self.n_runs} runs "
                  f"= {total} total runs\n")

        rows = []

        for ds_name in self.datasets:
            if verbose:
                print(f"── {ds_name} ", end="", flush=True)

            X, y_true, info = load_dataset(ds_name, data_dir=self.data_dir)
            k = info["k"]

            for method_name in self.methods:
                clusterer = get_clusterer(method_name, self.algo_cfg)

                for run_idx in range(self.n_runs):
                    # Reproducible seed per (dataset, method, run)
                    seed = self.base_seed + hash(ds_name + method_name) % 10_000 + run_idx
                    rng = np.random.default_rng(seed)

                    t0 = time.perf_counter()
                    labels = clusterer.fit_predict(X, k, y_true=y_true, rng=rng)
                    elapsed = time.perf_counter() - t0

                    metrics = compute_all_metrics(y_true, labels)
                    rows.append({
                        "dataset": ds_name,
                        "method":  method_name,
                        "run":     run_idx,
                        **metrics,
                        "time_s":  round(elapsed, 4),
                    })

                if verbose:
                    print(".", end="", flush=True)

            if verbose:
                print()

        if verbose:
            print("\nDone.")

        raw_df = pd.DataFrame(rows)
        raw_df.to_csv(self.results_dir / "raw_results.csv", index=False)

        summary_df = self.get_summary(raw_df)
        summary_df.to_csv(self.results_dir / "summary.csv")

        if verbose:
            print(f"\nResults saved to: {self.results_dir}")

        return raw_df

    def get_summary(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw results: mean over runs per (dataset, method).

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame.  Index: (dataset, method).
            Columns: mean of ACC, F-score, Recall, DCV, time_s.
        """
        metrics = ["ACC", "F-score", "Recall", "DCV", "time_s"]
        summary = (
            raw_df.groupby(["dataset", "method"])[metrics]
            .mean()
            .round(4)
        )
        return summary

    def compare_with_paper(self, summary_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build a comparison table between replication and paper claims.

        Since Tables III–VI in the paper are vector-image graphics (numeric
        values not extractable), the comparison is qualitative:
          • 'Paper claim'  — what the paper text says ICTLB should achieve
          • 'Replicated?'  — whether the replication satisfies that claim

        Parameters
        ----------
        summary_df : pd.DataFrame, optional
            Output of get_summary().  If None, reads results/summary.csv.

        Returns
        -------
        pd.DataFrame
            Comparison table with columns:
            Dataset, Metric, Paper claim (ICTLB), Replicated rank, Match
        """
        if summary_df is None:
            summary_df = pd.read_csv(self.results_dir / "summary.csv",
                                     index_col=[0, 1])

        metrics = ["ACC", "F-score", "Recall"]
        rows = []

        for ds_name in self.datasets:
            ds_cfg = self.paper_cfg.get(ds_name, {})
            ictlb_best = ds_cfg.get("ICTLB_best_metrics", [])
            notes = ds_cfg.get("notes", "—")

            for metric in metrics:
                # Rank all methods for this dataset × metric (higher = better)
                try:
                    ds_slice = summary_df.loc[ds_name, metric]
                    ranked = ds_slice.sort_values(ascending=False)
                    best_method = METHOD_DISPLAY_NAMES.get(ranked.index[0], ranked.index[0])
                    ictlb_val = summary_df.loc[(ds_name, "ictlb"), metric]
                    ictlb_rank = list(ranked.index).index("ictlb") + 1
                except (KeyError, TypeError):
                    best_method = "N/A"
                    ictlb_val   = float("nan")
                    ictlb_rank  = -1

                paper_claim = "Best" if metric in ictlb_best else "Not best"
                replicated_rank = f"Rank {ictlb_rank}" if ictlb_rank > 0 else "N/A"
                match = "✓" if (metric in ictlb_best and ictlb_rank == 1) or \
                               (metric not in ictlb_best and ictlb_rank > 1) else "✗"

                rows.append({
                    "Dataset":             ds_name,
                    "Metric":              metric,
                    "Paper claim (ICTLB)": paper_claim,
                    "ICTLB value":         round(float(ictlb_val), 4) if not np.isnan(ictlb_val) else "N/A",
                    "Best method":         best_method,
                    "Replicated rank":     replicated_rank,
                    "Match":               match,
                    "Paper notes":         notes,
                })

        return pd.DataFrame(rows)

    def get_dataset_info(self) -> pd.DataFrame:
        """Return a metadata table for all configured datasets."""
        rows = []
        for name in self.datasets:
            m = DATASETS[name]
            rows.append({
                "Dataset":  name,
                "Samples":  m["n"],
                "Features": m["n_features"],
                "k":        m["k"],
                "IR":       m["IR"],
                "Minority": m["minority_class_size"],
                "Majority": m["majority_class_size"],
            })
        return pd.DataFrame(rows)
