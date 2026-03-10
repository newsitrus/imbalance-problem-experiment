"""
smotewb_pkg — SMOTEWB replication package.

Salam & Cengiz, "A novel SMOTE-based resampling technique through noise
detection and the boosting procedure", Expert Systems with Applications, 2022.

Public API
----------
load_dataset(name)   → X, y, info
list_datasets()      → DataFrame
list_classifiers()   → dict
run_experiment(...)  → fold_df, rank_table

RESAMPLER_NAMES      list of valid resampler keys
RESAMPLER_DISPLAY    dict mapping keys to display labels
CLASSIFIER_REGISTRY  dict of classifier configurations
"""

from ._datasets import load_dataset, list_datasets, DATASET_REGISTRY
from ._classifiers import list_classifiers, CLASSIFIER_REGISTRY
from ._experiment import run_experiment, RESAMPLER_NAMES, RESAMPLER_DISPLAY
from ._core import smotewb_resample

__all__ = [
    'load_dataset',
    'list_datasets',
    'list_classifiers',
    'run_experiment',
    'smotewb_resample',
    'RESAMPLER_NAMES',
    'RESAMPLER_DISPLAY',
    'CLASSIFIER_REGISTRY',
    'DATASET_REGISTRY',
]
