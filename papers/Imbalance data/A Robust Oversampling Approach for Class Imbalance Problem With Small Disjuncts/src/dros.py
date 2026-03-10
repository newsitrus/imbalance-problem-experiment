"""
DROS (Disjuncts-Robust Oversampling) — Algorithms 1-4 from:
"A Robust Oversampling Approach for Class Imbalance Problem With Small Disjuncts"
Yi Sun et al., IEEE TKDE, 2023.

Parameters (paper defaults):
    rho   = 0.5      (cone aperture, Eq. 1)
    delta = -0.7660  (direct-interlinked threshold, Eq. 14)
    k     = 7        (nearest majority neighbors, Eq. 8)
    g     = 1.0      (boundary proximity control, Eq. 25)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Algorithm 2 (p.6): Relationships
# Computes direct-interlinked relationship matrix between minority pairs.
# Reference: Equations 12-14
# ---------------------------------------------------------------------------
def _relationships(S_maj, S_min, delta):
    n_min = len(S_min)
    n_maj = len(S_maj)
    I = np.zeros((n_min, n_min), dtype=int)

    for i in range(n_min - 1):
        for j in range(i + 1, n_min):
            # Vectorized over majority points (Eq. 12)
            diff_i = S_min[i] - S_maj          # (n_maj, H)
            diff_j = S_min[j] - S_maj          # (n_maj, H)
            norm_i = np.linalg.norm(diff_i, axis=1)  # (n_maj,)
            norm_j = np.linalg.norm(diff_j, axis=1)  # (n_maj,)

            # Mask out zero norms
            valid = (norm_i > 0) & (norm_j > 0)
            if not np.any(valid):
                continue

            # Inner product of unit vectors (Eq. 12)
            unit_i = diff_i[valid] / norm_i[valid, np.newaxis]
            unit_j = diff_j[valid] / norm_j[valid, np.newaxis]
            D_k = np.sum(unit_i * unit_j, axis=1)

            # Set D_k = 0 for invalid entries (paper convention)
            all_D = np.zeros(n_maj)
            all_D[valid] = D_k

            # Minimum inner product (Eq. 13)
            M_ij = np.min(all_D)

            # Direct-interlinked relationship (Eq. 14)
            if M_ij >= delta:
                I[i][j] = 1
                I[j][i] = 1

    return I


# ---------------------------------------------------------------------------
# Algorithm 3 (p.6): Structures
# Computes light-cone structure for each minority sample.
# Reference: Equations 8-11, 15-23
# ---------------------------------------------------------------------------
def _structures(S_maj, S_min, rho, k, I):
    n_min = len(S_min)
    n_maj = len(S_maj)
    S2 = []

    # Precompute k nearest majority neighbors (Eq. 8)
    k_actual = min(k, n_maj)
    knn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
    knn.fit(S_maj)

    for i in range(n_min):
        x_i = S_min[i]

        # --- Base unit vector (Eqs. 8-11) ---
        _, indices = knn.kneighbors(x_i.reshape(1, -1))
        S_knn = S_maj[indices[0]]               # k nearest majority (Eq. 8)
        z_bar = np.mean(S_knn, axis=0)          # mean center (Eq. 9)

        diff = x_i - z_bar
        norm_diff = np.linalg.norm(diff)
        if norm_diff == 0:
            continue                            # improper: a == 0

        c = diff / norm_diff                    # Eq. 10
        a = -c                                  # Eq. 11

        # --- Vertex (Eqs. 15-21) ---
        sum_positive_p = 0.0
        count_positive = 0

        for j in range(n_min):
            if I[i][j] != 1:
                continue
            d_j = S_min[j] - x_i                # Eq. 16
            p_j = np.dot(d_j, c)                # Eq. 18
            if p_j > 0:                         # Eq. 20: J(p_j) = 1
                sum_positive_p += p_j
                count_positive += 1

        if count_positive == 0:
            continue                            # improper: no interlinked

        p_bar = sum_positive_p / count_positive  # Eq. 19
        v = p_bar * c + x_i                      # Eq. 21

        # --- Radius (Eqs. 22-23) ---
        diff_zv = S_maj - v                                 # (n_maj, H)
        norms_zv = np.linalg.norm(diff_zv, axis=1)         # (n_maj,)
        valid_mask = norms_zv > 0
        if not np.any(valid_mask):
            continue

        unit_zv = diff_zv[valid_mask] / norms_zv[valid_mask, np.newaxis]
        inner_products = unit_zv @ c                        # Eq. 22
        illuminated = inner_products >= rho

        if not np.any(illuminated):
            continue                            # improper: no illuminated majority

        illuminated_dists = norms_zv[valid_mask][illuminated]
        L_v_g = np.min(illuminated_dists)       # nearest illuminated majority
        L_v_x = np.linalg.norm(v - x_i)        # distance vertex to seed

        # Eq. 23
        r = min(L_v_x, L_v_g) + 0.5 * (L_v_g - L_v_x)

        if r <= 0:
            continue                            # improper: non-positive radius

        S2.append({'a': a, 'v': v, 'r': r, 'rho': rho})

    return S2


# ---------------------------------------------------------------------------
# Helper: generate random unit vector satisfying <d, a> >= rho
# Reference: Section 3.4 (high-dimensional sampling strategy)
# ---------------------------------------------------------------------------
def _generate_valid_direction(a, rho, H, max_attempts=1000):
    for _ in range(max_attempts):
        rand_vec = np.random.randn(H)
        candidate = rand_vec + a
        norm = np.linalg.norm(candidate)
        if norm == 0:
            continue
        d_vec = candidate / norm
        if np.dot(d_vec, a) >= rho:
            return d_vec
    # Fallback: a itself always satisfies <a, a> = 1 >= rho
    return a.copy()


# ---------------------------------------------------------------------------
# Algorithm 4 (p.7): DataGeneration
# Generates synthetic minority samples inside valid light-cones.
# Reference: Equations 24-25
# ---------------------------------------------------------------------------
def _data_generation(n_maj, n_min, S2, rho, g):
    n_generate = n_maj - n_min
    if len(S2) == 0 or n_generate <= 0:
        H = len(S2[0]['a']) if S2 else 0
        return np.empty((0, H))

    H = len(S2[0]['a'])
    S_new = np.empty((n_generate, H))

    for i in range(n_generate):
        s = S2[np.random.randint(len(S2))]
        a, v, r = s['a'], s['v'], s['r']

        # Eq. 25
        xi = g + np.random.rand() * (1 - g)

        # Random unit vector satisfying <d, a> >= rho
        d_vec = _generate_valid_direction(a, rho, H)

        # Eq. 24
        S_new[i] = v + (xi * r) * d_vec

    return S_new


# ---------------------------------------------------------------------------
# Algorithm 1 (p.6): DROS — main entry point
# ---------------------------------------------------------------------------
def dros(S_maj, S_min, rho=0.5, k=7, delta=-0.7660, g=1.0):
    """
    DROS oversampling: generate |S_maj| - |S_min| synthetic minority samples.

    Parameters
    ----------
    S_maj : np.ndarray, shape (n_maj, H)
        Majority class training samples.
    S_min : np.ndarray, shape (n_min, H)
        Minority class training samples.
    rho : float
        Cone aperture parameter (Eq. 1). Default 0.5.
    k : int
        Number of nearest majority neighbors (Eq. 8). Default 7.
    delta : float
        Direct-interlinked threshold (Eq. 14). Default -0.7660.
    g : float
        Boundary proximity control (Eq. 25). Default 1.0.

    Returns
    -------
    S_new : np.ndarray, shape (n_maj - n_min, H)
        Synthetic minority samples.
    """
    # Step 1: Compute relationships (Algorithm 2)
    I = _relationships(S_maj, S_min, delta)

    # Step 2: Compute light-cone structures (Algorithm 3)
    S2 = _structures(S_maj, S_min, rho, k, I)

    # Step 3: Generate synthetic samples (Algorithm 4)
    S_new = _data_generation(len(S_maj), len(S_min), S2, rho, g)

    return S_new
