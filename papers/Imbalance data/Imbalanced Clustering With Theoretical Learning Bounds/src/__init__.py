"""
ICTLB replication package.

Public API
----------
ExperimentPipeline   — run full experiment from a config.yaml path
load_dataset         — load and L2-normalise a single KEEL dataset
dataset_summary      — print dataset metadata table
DATASETS             — registry of all available datasets
METHOD_DISPLAY_NAMES — human-readable method labels
"""

from .data import DATASETS, dataset_summary, load_dataset
from .algorithms import METHOD_DISPLAY_NAMES, get_clusterer
from .metrics import compute_all_metrics
from .pipeline import ExperimentPipeline

__all__ = [
    "ExperimentPipeline",
    "load_dataset",
    "dataset_summary",
    "DATASETS",
    "METHOD_DISPLAY_NAMES",
    "get_clusterer",
    "compute_all_metrics",
]
