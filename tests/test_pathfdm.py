"""Basic checks on synthetic data: route separation, water-permutation invariance, standardization."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathfdm import embed  # noqa: E402
from pathfdm.evaluation import auc_per_coordinate  # noqa: E402
from pathfdm.loaders import _pack  # noqa: E402


def synthetic(n=20, k=3, shuffle_waters=False, seed=0):
    """Two routes (+y or -y) of a 1-atom solute with k waters that follow the route."""
    rng = np.random.default_rng(seed)
    perm_rng = np.random.default_rng(seed + 1000)      # separate stream: shuffling does not change the noise
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    coords = []
    for lab in labels:
        T = int(rng.integers(30, 60))
        t = np.linspace(0, 1, T)[:, None]
        sign = 1 if lab else -1
        sol = np.hstack([5 * t, sign * 3 * t, 0 * t]) + rng.normal(0, 0.1, (T, 3))
        wat = [np.hstack([0 * t + a, sign * 2 * t, 0 * t]) for a in range(k)]
        if shuffle_waters:
            wat = [wat[i] for i in perm_rng.permutation(k)]
        coords.append(np.hstack([sol] + wat).astype(np.float32))
    ts = _pack(coords, [np.linspace(0, 1, len(c)) for c in coords], [f"t{i}" for i in range(n)], 1, k, [])
    return ts, labels


def test_routes_separate():
    ts, y = synthetic()
    Z = embed(ts.solute_only().fd, metric="l2", n_jobs=1)
    assert auc_per_coordinate(Z, y)[0] > 0.99


def test_water_matching_is_permutation_invariant():
    ts, y = synthetic(seed=1)
    ts_shuffled, _ = synthetic(seed=1, shuffle_waters=True)
    Z = embed(ts.fd, metric="l2", water=True, n_jobs=1)
    Zs = embed(ts_shuffled.fd, metric="l2", water=True, n_jobs=1)
    assert np.allclose(np.abs(np.corrcoef(Z[:, 0], Zs[:, 0])[0, 1]), 1.0, atol=1e-6)


def test_standardized_coordinates():
    ts, _ = synthetic()
    Z = embed(ts.solute_only().fd, metric="l2", n_jobs=1)
    assert np.allclose(Z.mean(0), 0, atol=1e-8) and np.allclose(Z.std(0), 1, atol=1e-8)


if __name__ == "__main__":
    for f in (test_routes_separate, test_water_matching_is_permutation_invariant, test_standardized_coordinates):
        f()
        print("ok", f.__name__)
