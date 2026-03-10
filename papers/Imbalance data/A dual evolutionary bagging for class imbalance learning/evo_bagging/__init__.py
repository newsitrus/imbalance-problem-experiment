"""
evo_bagging — Dual Evolutionary Bagging replication package.

Public API
----------
EvoBagging       : end-to-end pipeline (fit / predict_proba / run_cv)
run_all_baselines: run OverBagging, EE-Bagging, Random Forest with k-fold CV
load_ecoli3      : load the ecoli3 KEEL benchmark dataset
g_mean_score     : geometric mean of sensitivity and specificity
auc_score        : ROC-AUC for binary classification
"""

from .pipeline import EvoBagging
from .baselines import run_all_baselines
from .dataset import load_ecoli3
from .metrics import g_mean_score, auc_score

__all__ = [
    "EvoBagging",
    "run_all_baselines",
    "load_ecoli3",
    "g_mean_score",
    "auc_score",
]
