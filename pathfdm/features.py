"""Trajectory-level descriptors: four statistics of each per-frame descriptor along a path.

For a path of per-frame descriptors x (n_frames, n_desc), every descriptor gives
    sigma   standard deviation along the path
    mean    mean along the path
    Delta   deviation from linear progression: RMS distance from the straight line joining the first and
            last frame (of the processed path)
    gamma1  skewness
in that order, so the feature matrix has 4 * n_desc columns ordered [sigma, mean, Delta, gamma1] per descriptor.

Paths are smoothed with the predictive noise reducer before the statistics are taken. ``stats_on`` chooses
which statistics use the smoothed path:
    "denoised"  all four statistics on the smoothed path
    "mixed"     sigma, mean, gamma1 on the raw path; Delta on the smoothed path
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew

from .denoise import PredictiveNoiseReducer

STAT_NAMES = ("sigma", "mean", "Delta", "gamma1")
STAT_TEX = {"sigma": r"\sigma({})", "mean": r"\langle {} \rangle", "Delta": r"\Delta({})", "gamma1": r"\gamma_1({})"}


def denoise_path(x: np.ndarray) -> np.ndarray:
    """Predictive denoising with the settings used throughout (m 4, tau 1, k_nn = n/4 within [2, 10], horizon 2)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 2:
        return x
    return PredictiveNoiseReducer(m=4, tau=1, k_nn=min(10, max(2, len(x) // 4)), prediction_horizon=2,
                                  bidirectional=True).denoise(x, n_iterations=2)


def linear_deviation(x: np.ndarray) -> np.ndarray:
    n, c = x.shape
    if n <= 2:
        return np.zeros(c)
    t = np.linspace(0, 1, n)[:, None]
    return np.sqrt(np.mean((x - (x[0] + t * (x[-1] - x[0]))) ** 2, axis=0))


def _skew(x, zero_degenerate=True):
    g = skew(x, axis=0, nan_policy="omit")
    g = np.asarray(g, dtype=float)
    if zero_degenerate:
        g = np.where(~np.isfinite(g) | (np.std(x, axis=0) < 1e-8), 0.0, g)
    return g


def path_statistics(x: np.ndarray, stats_on: str = "denoised", zero_degenerate_skew: bool = True) -> np.ndarray:
    """[sigma, mean, Delta, gamma1] of every column of x, interleaved per column (length 4 * n_desc)."""
    raw = np.nan_to_num(np.asarray(x, dtype=np.float64))
    den = denoise_path(raw)
    if stats_on == "denoised":
        base = den
    elif stats_on == "mixed":
        base = raw
    else:
        raise ValueError("stats_on must be 'denoised' or 'mixed'")
    out = np.empty((raw.shape[1], 4))
    out[:, 0] = np.std(base, axis=0)
    out[:, 1] = np.mean(base, axis=0)
    out[:, 2] = linear_deviation(den)
    out[:, 3] = _skew(base, zero_degenerate_skew)
    return out.ravel()


def feature_matrix(desc_trajs, desc_names, stats_on: str = "denoised", zero_degenerate_skew: bool = True,
                   joint: bool = True):
    """Feature matrix (n_traj, 4 * n_desc) and LaTeX feature names, e.g. r"$\\sigma(d_6)$".

    joint=True smooths all descriptors of a path together (neighbours found in the joint descriptor space);
    joint=False smooths each descriptor on its own.
    """
    if joint:
        X = np.array([path_statistics(d, stats_on, zero_degenerate_skew) for d in desc_trajs])
    else:
        X = np.array([np.concatenate([path_statistics(d[:, [j]], stats_on, zero_degenerate_skew)
                                      for j in range(d.shape[1])]) for d in desc_trajs])
    names = [f"${STAT_TEX[s].format(n)}$" for n in desc_names for s in STAT_NAMES]
    return X, names
