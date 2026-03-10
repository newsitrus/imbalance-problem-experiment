"""
SMOTEWB: SMOTE with Boosting — noise detection and adaptive neighbor count.

Salam & Cengiz, Expert Systems with Applications, 2022.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def _adaboost_weights(X, y, M=100, random_state=42):
    """
    Run AdaBoost for M rounds and return final per-instance weights.

    Uses decision stumps (max_depth=1) as weak learners.
    """
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
    Label instances as noise based on AdaBoost weights and class thresholds.

    T_pos = 2 * n_neg / n^2
    T_neg = 2 * n_pos / n^2
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
    For each positive instance, compute adaptive k_i and classify as good/lonely/bad.

    k_i = number of non-noise positive neighbors before first non-noise negative
    neighbor, capped at k_max = floor(n_neg / n_pos).
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

        # Only consider non-noise instances (excluding self)
        candidates = non_noise_all[non_noise_all != idx]

        if len(candidates) == 0:
            itype = 'bad' if is_noise[idx] else 'lonely'
            instance_info.append({
                'index': idx, 'type': itype, 'k_i': 0, 'neighbors': []
            })
            continue

        # Compute distances to all non-noise candidates
        dists = np.linalg.norm(X_scaled[candidates] - x_i, axis=1)
        order = np.argsort(dists)
        sorted_candidates = candidates[order]

        # Count positive non-noise neighbors before first negative non-noise
        k_i = 0
        neighbors = []
        for j in sorted_candidates:
            if k_i >= k_max:
                break
            if j in non_noise_neg:
                break  # hit a negative neighbor
            # j is a non-noise positive neighbor
            k_i += 1
            neighbors.append(j)

        if k_i > 0:
            instance_info.append({
                'index': idx, 'type': 'good', 'k_i': k_i, 'neighbors': neighbors
            })
        elif is_noise[idx]:
            instance_info.append({
                'index': idx, 'type': 'bad', 'k_i': 0, 'neighbors': []
            })
        else:
            instance_info.append({
                'index': idx, 'type': 'lonely', 'k_i': 0, 'neighbors': []
            })

    return instance_info


def _generate_synthetics(X_scaled, instance_info, n_to_generate, rng):
    """
    Generate synthetic samples from good and lonely instances.

    Good: SMOTE interpolation with one of k_i neighbors.
    Lonely: copy the instance.
    Bad: skip.
    """
    usable = [info for info in instance_info if info['type'] in ('good', 'lonely')]

    if len(usable) == 0 or n_to_generate <= 0:
        return np.empty((0, X_scaled.shape[1]))

    synthetics = []
    idx = 0
    while len(synthetics) < n_to_generate:
        info = usable[idx % len(usable)]
        x_i = X_scaled[info['index']]

        if info['type'] == 'good':
            j = rng.choice(info['neighbors'])
            lam = rng.rand()
            x_syn = x_i + lam * (X_scaled[j] - x_i)
        else:  # lonely
            x_syn = x_i.copy()

        synthetics.append(x_syn)
        idx += 1

    return np.array(synthetics[:n_to_generate])


def smotewb(X, y, M=100, random_state=42):
    """
    SMOTEWB: SMOTE with Boosting.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features) — original-scale features
    y : np.ndarray (n_samples,) — binary labels (0=majority, 1=minority)
    M : int — number of AdaBoost iterations
    random_state : int

    Returns
    -------
    X_resampled : np.ndarray — balanced features (original scale)
    y_resampled : np.ndarray — balanced labels
    """
    rng = np.random.RandomState(random_state)

    # Step 1: scale to [0,1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: AdaBoost weights
    weights = _adaboost_weights(X_scaled, y, M=M, random_state=random_state)

    # Step 3: noise detection
    is_noise = _detect_noise(y, weights)

    # Step 5: per-instance k_i
    instance_info = _determine_ki(X_scaled, y, is_noise)

    # Step 6: generate synthetics
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_to_generate = n_neg - n_pos

    X_syn_scaled = _generate_synthetics(X_scaled, instance_info, n_to_generate, rng)

    # Step 7: combine and descale
    if len(X_syn_scaled) > 0:
        X_all_scaled = np.vstack([X_scaled, X_syn_scaled])
        y_all = np.concatenate([y, np.ones(len(X_syn_scaled), dtype=int)])
    else:
        X_all_scaled = X_scaled
        y_all = y.copy()

    X_resampled = scaler.inverse_transform(X_all_scaled)
    return X_resampled, y_all
