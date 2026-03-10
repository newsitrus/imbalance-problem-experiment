"""
algorithms.py — Clustering methods from the ICTLB paper.

Methods implemented (Section IV-B of the paper):
  1. KMeans         — standard k-means, density-k-means++ init
  2. KMeansPP       — k-means++ init (sklearn)
  3. MACW           — k-Means with Adaptive Cluster Weights (Eq. 13)
  4. ICTLB          — Imbalanced Clustering with Theoretical Learning Bounds (Eq. 36)

All methods use density-k-means++ initialisation (paper Section IV-B:
"To obtain a better initialization … we generate an initial center C⁰ by the
imbalanced clustering method, i.e., density-k-means++").

Paper settings faithfully replicated:
  α  = −0.5             (paper fixes α = −1/2, Section IV-A)
  W  fixed = 2.0        (paper Section IV-E: "fix one constant W = 2")
  W  tuned ∈ {0.1,0.2,0.3,1,2,3}  (paper Section IV-E)
  W  selected by maximising ACC with true labels (oracle tuning, same as paper)
  max_iter = 300, tol = 1e-4   (convergence not stated in paper; standard choice)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans as _SKLearnKMeans
from sklearn.neighbors import NearestNeighbors

from .metrics import clustering_accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Shared initialisation: density-k-means++
# ──────────────────────────────────────────────────────────────────────────────

def _density_kmeanspp_init(
    X: np.ndarray,
    k: int,
    n_neighbors: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Density-weighted D² seeding (Fan, Chai, Li 2019, ref [31]).

    Extends k-means++ by weighting candidate selection probability by local
    density, giving minority-cluster regions a higher chance of receiving a
    centre.

    Probability of selecting point x as next centre:
        p(x) ∝ density(x) × d²(x, nearest existing centre)
    where density(x) = 1 / (mean_kNN_distance(x) + ε).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = X.shape[0]
    eps = 1e-10

    # Local density via kNN
    k_nn = min(n_neighbors, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_nn, algorithm="auto").fit(X)
    knn_dists, _ = nbrs.kneighbors(X)
    density = 1.0 / (knn_dists.mean(axis=1) + eps)

    # First centre: sample ∝ density
    p = density / density.sum()
    centres = [X[rng.choice(n, p=p)].copy()]

    # Subsequent centres: sample ∝ density × D²
    for _ in range(1, k):
        # Vectorised minimum squared distance to existing centres
        dists_sq = cdist(X, np.array(centres), metric="sqeuclidean").min(axis=1)
        combined = density * dists_sq
        total = combined.sum()
        p = combined / total if total > 0 else np.ones(n) / n
        centres.append(X[rng.choice(n, p=p)].copy())

    return np.array(centres)


# ──────────────────────────────────────────────────────────────────────────────
# Shared update loop: standard Lloyd k-means from given initial centres
# ──────────────────────────────────────────────────────────────────────────────

def _lloyd_updates(
    X: np.ndarray,
    centres: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard Lloyd k-means from given initial centres.

    Returns (labels, final_centres).
    Empty clusters are re-seeded from a random data point (standard practice).
    """
    if rng is None:
        rng = np.random.default_rng()

    centres = centres.copy()
    k = centres.shape[0]

    for _ in range(max_iter):
        old = centres.copy()
        dists = cdist(X, centres, metric="sqeuclidean")
        labels = dists.argmin(axis=1)
        for r in range(k):
            mask = labels == r
            centres[r] = X[mask].mean(axis=0) if mask.any() else X[rng.integers(n := X.shape[0])]
        if np.linalg.norm(centres - old) < tol:
            break

    labels = cdist(X, centres, metric="sqeuclidean").argmin(axis=1)
    return labels, centres


# ──────────────────────────────────────────────────────────────────────────────
# Base interface
# ──────────────────────────────────────────────────────────────────────────────

class BaseClusterer:
    """Shared interface for all clustering methods."""

    def fit_predict(
        self,
        X: np.ndarray,
        k: int,
        y_true: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Fit and return integer cluster labels in [0, k-1]."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# 1. KMeans
# ──────────────────────────────────────────────────────────────────────────────

class KMeans(BaseClusterer):
    """Standard k-means with density-k-means++ initialisation.

    Uses Lloyd updates (same as the paper's baseline setup).
    """

    def __init__(self, max_iter: int = 300, tol: float = 1e-4,
                 n_neighbors_density: int = 5):
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors_density = n_neighbors_density

    def fit_predict(self, X, k, y_true=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        centres = _density_kmeanspp_init(X, k, self.n_neighbors_density, rng)
        labels, _ = _lloyd_updates(X, centres, self.max_iter, self.tol, rng)
        return labels


# ──────────────────────────────────────────────────────────────────────────────
# 2. KMeansPP
# ──────────────────────────────────────────────────────────────────────────────

class KMeansPP(BaseClusterer):
    """k-means++ initialisation (sklearn built-in).

    Uses sklearn's D²-seeded k-means++ followed by Lloyd updates.
    """

    def __init__(self, max_iter: int = 300, tol: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol

    def fit_predict(self, X, k, y_true=None, rng=None):
        seed = int(rng.integers(0, 2 ** 31)) if rng is not None else None
        model = _SKLearnKMeans(
            n_clusters=k, init="k-means++", n_init=1,
            max_iter=self.max_iter, tol=self.tol, random_state=seed,
        )
        return model.fit_predict(X).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# 3. MACW  —  k-Means with Adaptive Cluster Weights  (Eq. 13)
# ──────────────────────────────────────────────────────────────────────────────

class MACW(BaseClusterer):
    """k-Means with Adaptive Cluster Weights.

    Objective (Eq. 12):
        Φ = (1/n) Σ_i  min_r  ||x_i − c_r||² · λ_r^α

    Weight update (Eq. 13):
        numer_r = ( Σ_{x∈θ_r} ||x − c_r||² )^{ 1/(1−α) }
        λ_r     = numer_r / Σ_s numer_s

    All settings match the paper (α = −0.5, density-k-means++ init,
    max_iter=300, tol=1e-4).
    """

    def __init__(self, alpha: float = -0.5, max_iter: int = 300,
                 tol: float = 1e-4, n_neighbors_density: int = 5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors_density = n_neighbors_density

    def fit_predict(self, X, k, y_true=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        alpha = self.alpha
        n = X.shape[0]

        centres = _density_kmeanspp_init(X, k, self.n_neighbors_density, rng)
        lambdas = np.full(k, 1.0 / k)

        for _ in range(self.max_iter):
            old_centres = centres.copy()

            # Assignment: argmin_r  ||x_i − c_r||² · λ_r^α
            w = lambdas ** alpha                                     # (k,)
            dists_sq = cdist(X, centres, metric="sqeuclidean")       # (n, k)
            labels = (dists_sq * w[np.newaxis, :]).argmin(axis=1)

            # Weight update  (Eq. 13)
            numer = np.zeros(k)
            for r in range(k):
                mask = labels == r
                if mask.any():
                    wcss_r = float(np.sum((X[mask] - centres[r]) ** 2))
                    numer[r] = wcss_r ** (1.0 / (1.0 - alpha))
            total = numer.sum()
            lambdas = numer / total if total > 1e-12 else np.full(k, 1.0 / k)

            # Centre update
            for r in range(k):
                mask = labels == r
                centres[r] = X[mask].mean(axis=0) if mask.any() else X[rng.integers(n)]

            if np.linalg.norm(centres - old_centres) < self.tol:
                break

        # Final assignment
        w = lambdas ** alpha
        labels = (cdist(X, centres, metric="sqeuclidean") * w[np.newaxis, :]).argmin(axis=1)
        return labels


# ──────────────────────────────────────────────────────────────────────────────
# 4. ICTLB  —  Imbalanced Clustering with Theoretical Learning Bounds (Eq. 36)
# ──────────────────────────────────────────────────────────────────────────────

class ICTLB(BaseClusterer):
    """Imbalanced Clustering with Theoretical Learning Bounds.

    Weight update (Eq. 36):
        λ_r^α = W_r · |θ_r|^{ α / (2·(1−α)) }

    W selection (Section IV-E):
        Paper: "fix one constant W = 2 and tune another constant W."
        For k=2: W_0 = W_fixed (default 2.0), W_1 tuned over W_grid.
        Best W_1 chosen by maximising ACC against true labels y_true
        (oracle hyperparameter tuning — same procedure as paper).
    """

    def __init__(
        self,
        alpha: float = -0.5,
        W_fixed: float = 2.0,
        W_grid: Optional[list] = None,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_neighbors_density: int = 5,
    ):
        self.alpha = alpha
        self.W_fixed = W_fixed
        self.W_grid = W_grid if W_grid is not None else [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors_density = n_neighbors_density

    def _run_single(
        self,
        X: np.ndarray,
        k: int,
        W: np.ndarray,
        centres_init: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Run one ICTLB instance with a fixed W vector (one W per cluster)."""
        alpha = self.alpha
        exp = alpha / (2.0 * (1.0 - alpha))   # exponent in Eq. 36
        n = X.shape[0]
        centres = centres_init.copy()

        for _ in range(self.max_iter):
            old_centres = centres.copy()
            dists_sq = cdist(X, centres, metric="sqeuclidean")   # (n, k)

            # Iteratively stabilise weights and assignment (inner fixed-point)
            labels = dists_sq.argmin(axis=1)   # warm-start with plain distances
            for _ in range(10):
                sizes = np.array([(labels == r).sum() for r in range(k)], dtype=float)
                sizes = np.maximum(sizes, 1.0)
                lambda_alpha = W * (sizes ** exp)                  # Eq. 36
                new_labels = (dists_sq * lambda_alpha[np.newaxis, :]).argmin(axis=1)
                if (new_labels == labels).all():
                    break
                labels = new_labels

            # Centre update
            for r in range(k):
                mask = labels == r
                centres[r] = X[mask].mean(axis=0) if mask.any() else X[rng.integers(n)]

            if np.linalg.norm(centres - old_centres) < self.tol:
                break

        # Final assignment with converged centres
        dists_sq = cdist(X, centres, metric="sqeuclidean")
        sizes = np.array([(labels == r).sum() for r in range(k)], dtype=float)
        sizes = np.maximum(sizes, 1.0)
        lambda_alpha = W * (sizes ** exp)
        labels = (dists_sq * lambda_alpha[np.newaxis, :]).argmin(axis=1)
        return labels

    def fit_predict(self, X, k, y_true=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Shared initialisation across all W candidates (same starting point)
        centres_init = _density_kmeanspp_init(X, k, self.n_neighbors_density, rng)

        if y_true is None:
            # No labels: use default W_fixed for all clusters
            W = np.full(k, self.W_fixed)
            return self._run_single(X, k, W, centres_init, rng)

        # ── W grid search (paper's oracle procedure) ──────────────────────────
        # Fix W for cluster 0 = W_fixed; tune W for cluster 1 over W_grid.
        # For k > 2: fix all but the last cluster.
        best_acc = -1.0
        best_labels = None

        for w_tune in self.W_grid:
            W = np.full(k, self.W_fixed)
            W[-1] = w_tune                               # tune last cluster's W
            labels = self._run_single(X, k, W, centres_init, rng)
            acc = clustering_accuracy(y_true, labels)
            if acc > best_acc:
                best_acc = acc
                best_labels = labels.copy()

        return best_labels


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

#: Display names used in tables and plots
METHOD_DISPLAY_NAMES = {
    "kmeans":   "k-means",
    "kmeanspp": "k-means++",
    "macw":     "MACW",
    "ictlb":    "ICTLB",
}


def get_clusterer(method: str, cfg: dict) -> BaseClusterer:
    """Instantiate a clusterer by name from the algorithm config section.

    Parameters
    ----------
    method : str
        One of: 'kmeans', 'kmeanspp', 'macw', 'ictlb'
    cfg : dict
        Contents of the ``algorithm`` section from config.yaml.
    """
    alpha   = cfg.get("alpha", -0.5)
    max_iter = cfg.get("max_iter", 300)
    tol     = cfg.get("tol", 1e-4)
    n_nbrs  = cfg.get("n_neighbors_density", 5)
    W_fixed = cfg.get("ictlb_W_fixed", 2.0)
    W_grid  = cfg.get("ictlb_W_grid", [0.1, 0.2, 0.3, 1.0, 2.0, 3.0])

    dispatch = {
        "kmeans":   KMeans(max_iter=max_iter, tol=tol, n_neighbors_density=n_nbrs),
        "kmeanspp": KMeansPP(max_iter=max_iter, tol=tol),
        "macw":     MACW(alpha=alpha, max_iter=max_iter, tol=tol,
                         n_neighbors_density=n_nbrs),
        "ictlb":    ICTLB(alpha=alpha, W_fixed=W_fixed, W_grid=W_grid,
                          max_iter=max_iter, tol=tol,
                          n_neighbors_density=n_nbrs),
    }
    if method not in dispatch:
        raise KeyError(f"Unknown method '{method}'. Available: {list(dispatch)}")
    return dispatch[method]
