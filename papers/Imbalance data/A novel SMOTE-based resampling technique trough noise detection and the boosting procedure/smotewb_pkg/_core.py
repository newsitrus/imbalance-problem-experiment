"""
SMOTEWB core algorithm — internal module.

Salam & Cengiz, Expert Systems with Applications, 2022.
Do not import this directly; use the public API in smotewb_pkg.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def _adaboost_weights(X, y, M=100, random_state=42):
    """Run AdaBoost for M rounds and return final per-instance weights."""
    rng = np.random.RandomState(random_state)
    n = len(y)
    w = np.ones(n) / n

    for _ in range(M):
        stump = DecisionTreeClassifier(max_depth=1, random_state=rng.randint(2**31 - 1))
        stump.fit(X, y, sample_weight=w)
        pred = stump.predict(X)

        incorrect = (pred != y).astype(float)
        err = np.dot(w, incorrect) / np.sum(w)
        err = np.clip(err, 1e-10, 1 - 1e-10)

        if err >= 0.5:
            break

        alpha = 0.5 * np.log((1 - err) / err)
        w *= np.exp(alpha * np.where(pred == y, -1, 1))
        w /= np.sum(w)

    return w


def _detect_noise(y, weights):
    """
    Label instances as noise using AdaBoost weights and class thresholds.

    T_pos = 2 * n_neg / n^2   (threshold for minority / positive class)
    T_neg = 2 * n_pos / n^2   (threshold for majority / negative class)
    Instance i is noise if weight[i] > T_class(i).
    """
    n = len(y)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    t_pos = 2 * n_neg / (n * n)
    t_neg = 2 * n_pos / (n * n)

    is_noise = np.zeros(n, dtype=bool)
    is_noise[(y == 1) & (weights > t_pos)] = True
    is_noise[(y == 0) & (weights > t_neg)] = True
    return is_noise


def _determine_ki(X_scaled, y, is_noise):
    """
    Compute per-instance k_i and classify each minority instance.

    k_i = number of non-noise positive neighbors before first non-noise
    negative neighbor, capped at k_max = floor(n_neg / n_pos).

    Returns list of dicts: {index, type ('good'|'lonely'|'bad'), k_i, neighbors}
    """
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    k_max = max(1, int(np.floor(n_neg / n_pos)))

    pos_indices = np.where(y == 1)[0]
    non_noise_pos = set(np.where((y == 1) & (~is_noise))[0])
    non_noise_neg = set(np.where((y == 0) & (~is_noise))[0])
    non_noise_all = np.array(sorted(non_noise_pos | non_noise_neg))

    instance_info = []
    for idx in pos_indices:
        x_i = X_scaled[idx]
        candidates = non_noise_all[non_noise_all != idx]

        if len(candidates) == 0:
            itype = 'bad' if is_noise[idx] else 'lonely'
            instance_info.append({'index': idx, 'type': itype, 'k_i': 0, 'neighbors': []})
            continue

        dists = np.linalg.norm(X_scaled[candidates] - x_i, axis=1)
        order = np.argsort(dists)
        sorted_candidates = candidates[order]

        k_i, neighbors = 0, []
        for j in sorted_candidates:
            if k_i >= k_max:
                break
            if j in non_noise_neg:
                break
            k_i += 1
            neighbors.append(j)

        if k_i > 0:
            itype = 'good'
        elif is_noise[idx]:
            itype = 'bad'
        else:
            itype = 'lonely'
        instance_info.append({'index': idx, 'type': itype, 'k_i': k_i, 'neighbors': neighbors})

    return instance_info


def _generate_synthetics(X_scaled, instance_info, n_to_generate, rng):
    """
    Generate synthetic samples (SMOTE interpolation for good, copy for lonely).
    """
    usable = [info for info in instance_info if info['type'] in ('good', 'lonely')]
    if not usable or n_to_generate <= 0:
        return np.empty((0, X_scaled.shape[1]))

    synthetics = []
    for i in range(n_to_generate):
        info = usable[i % len(usable)]
        x_i = X_scaled[info['index']]
        if info['type'] == 'good':
            j = rng.choice(info['neighbors'])
            lam = rng.rand()
            synthetics.append(x_i + lam * (X_scaled[j] - x_i))
        else:
            synthetics.append(x_i.copy())

    return np.array(synthetics)


def smotewb_resample(X, y, M=100, random_state=42):
    """
    SMOTEWB: SMOTE with Boosting — noise-aware adaptive oversampling.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)  — original-scale features
    y : ndarray (n_samples,)              — binary labels: 0=majority, 1=minority
    M : int                               — AdaBoost iterations (paper default: 100)
    random_state : int

    Returns
    -------
    X_resampled : ndarray — balanced dataset (original scale)
    y_resampled : ndarray — balanced labels
    """
    rng = np.random.RandomState(random_state)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    weights = _adaboost_weights(X_scaled, y, M=M, random_state=random_state)
    is_noise = _detect_noise(y, weights)
    instance_info = _determine_ki(X_scaled, y, is_noise)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_to_generate = n_neg - n_pos

    X_syn_scaled = _generate_synthetics(X_scaled, instance_info, n_to_generate, rng)

    if len(X_syn_scaled) > 0:
        X_all_scaled = np.vstack([X_scaled, X_syn_scaled])
        y_all = np.concatenate([y, np.ones(len(X_syn_scaled), dtype=int)])
    else:
        X_all_scaled = X_scaled
        y_all = y.copy()

    return scaler.inverse_transform(X_all_scaled), y_all
