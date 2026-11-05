"""
Functional Diffusion Maps (FDM) with multiple pathway metrics and optional
water permutation invariance.

Supported metrics (``metric=``)
-------------------------------
  "l2"             Functional L2: sqrt( ∫ ||x(t) - y(t)||² dt ), progress-resampled
  "l1"             Functional L1: ∫ ||x(t) - y(t)||_1 dt, progress-resampled
  "dtw"            Multidimensional DTW; ``window`` sets the warping band:
                     None            full DTW, every alignment allowed
                     "sakoe_chiba"   Sakoe-Chiba band of 8% of the longer path (at least 4 frames)
                     "sakoe_chiba_wide"  the same band, widened to at least the length difference |n - m|
                     int             Sakoe-Chiba band of that many frames (around the diagonal,
                                     widened on one side by the length difference |n - m|)
                     float in (0, 1) Sakoe-Chiba band of that fraction of the longer path
  "frechet"        Discrete Fréchet distance (order-aware, max-based)
  "hausdorff"      Classical (symmetric) Hausdorff between frame sets
  "avg_hausdorff"  PRISM weighted average Hausdorff:
                   0.5 * mean_a min_b ||a-b|| + 0.5 * mean_b min_a ||a-b||
                   optionally on k per-trajectory medoids (``n_medoids``), as in
                   PRISM Option 2.
  "msd"            SHINE n-ary MSD dissimilarities, chosen by ``msd_scheme``:
                   "intra", "semi_sum", "min", "max", "inter"

Water handling (``water=``)
---------------------------
  water=False  Plain metric on the full feature vector.
  water=True   For every pair (i, j) the water oxygens of trajectory j are first
               re-labelled to best match those of trajectory i (Hungarian
               assignment on a K x K cost matrix), then the chosen metric is
               evaluated on [solute, permuted waters]. Requires ``water_info``
               (or ``X.water_info``) with keys solute_dim, k_waters,
               atoms_per_water.

               Cost used for the matching (``water_match=``), between the
               path of water a in i and water b in j, each resampled to
               ``water_grid`` points:
                 "auto"      -> "dtw" when metric == "dtw", "l2" for the other
                                order-aware metrics, otherwise "mean"
                 "l2"        -> integrated squared distance on the progress grid
                 "l1"        -> integrated L1 distance on the progress grid
                 "dtw"       -> per-water DTW (Sakoe-Chiba band 10% of the grid)
                 "frechet"   -> discrete Fréchet distance between the two paths
                 "hausdorff" -> symmetric Hausdorff distance between the two
                                sets of positions (order-free)
                 "mean"      -> squared distance between mean water positions
                                over the full trajectory (order-free; a natural
                                choice for set metrics: hausdorff,
                                avg_hausdorff, msd)

Speed (this module vs fdm_metrics.py, same results)
---------------------------------------------------
  * DTW runs in dtaidistance's C code (``use_c=True``); fdm_metrics.py called
    dtw_ndim.distance with its default pure-Python path. Same values.
  * Every trajectory (and its water block) is resampled once, not once per
    pair. Column permutation commutes with resampling along time, so
    permuting the resampled array equals resampling the permuted one.
  * The K x K water DTW costs come from one block call in C.
  * Pairs go to the workers in chunks; trajectories travel as stacked arrays
    (memory-mapped by joblib), not as two pickled arrays per pair.

Controls for confounds (see discussion)
---------------------------------------
  standardize=True        z-score every feature column over all frames of all
                          trajectories before any distance (use for CV spaces;
                          leave False for Cartesian xyz, where it distorts
                          geometry).
  normalize_length=True   divide DTW by sqrt(n + m), where n, m are the lengths
                          actually passed to DTW (after resampling). DTW returns
                          sqrt(sum of per-step costs), which grows like
                          sqrt(path length); this gives the RMS cost per step.
  resample_all=True       resample EVERY trajectory (also short ones) to
                          subsample_t for dtw/frechet, removing length entirely
                          (stricter control; also discards rate information).

  Do not combine standardize=True with Cartesian xyz or with water=True:
  per-column z-scoring distorts geometry and gives each water column its own
  scale, which breaks the permutation invariance of the water matching.

Input ``X``
-----------
  skfda.FDataIrregular, skfda.FDataGrid, a list of (T_i, d) arrays, or a list
  of (times, array) tuples.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, directed_hausdorff
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import svd_flip

try:
    import skfda
except ImportError:  # the module also works on plain lists of arrays
    skfda = None

try:
    from dtaidistance import dtw_ndim, dtw
except ImportError:
    dtw_ndim = None

try:
    from numba import njit
except ImportError:  # pure-python fallback, fine for subsample_t <= ~150
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


VALID_METRICS = {"l2", "l1", "dtw", "frechet", "hausdorff", "avg_hausdorff", "msd"}
VALID_MSD_SCHEMES = {"intra", "semi_sum", "min", "max", "inter"}
ORDER_AWARE = {"l2", "l1", "dtw", "frechet"}      # resampled for cost control
SET_BASED = {"hausdorff", "avg_hausdorff", "msd"}  # order-independent


# =========================================================================== #
#  BASIC HELPERS                                                              #
# =========================================================================== #

def _c(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.double)
    return a[:, None] if a.ndim == 1 else a


def _resample(traj: np.ndarray, times: np.ndarray, n_points: int) -> np.ndarray:
    """Linear resampling onto a uniform progress grid in [0, 1]."""
    if len(traj) == n_points:
        return traj
    if len(traj) == 1:
        return np.repeat(traj, n_points, axis=0)
    t = np.asarray(times, dtype=np.float64).ravel()
    if len(t) != len(traj) or t[0] != 0.0 or t[-1] != 1.0:
        t = np.linspace(0.0, 1.0, len(traj))
    return interp1d(t, traj, axis=0, assume_sorted=True)(np.linspace(0.0, 1.0, n_points))


def _trapz(y: np.ndarray, dt: float) -> float:
    return float(((y[:-1] + y[1:]) * 0.5 * dt).sum())


# =========================================================================== #
#  DP KERNELS (numba-accelerated when available)                              #
# =========================================================================== #

@njit(cache=True)
def _dtw_dp(C, window):
    """DTW on a precomputed squared-distance matrix. Returns sqrt(total cost).
    Band as in dtaidistance: -(max(0, n - m) + window) <= j - i <= max(0, m - n) + window."""
    n, m = C.shape
    INF = 1e300
    acc = np.full((n + 1, m + 1), INF)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - max(0, n - m) - window)
        j_hi = min(m, i + max(0, m - n) + window)
        for j in range(j_lo, j_hi + 1):
            best = acc[i - 1, j - 1]
            if acc[i - 1, j] < best:
                best = acc[i - 1, j]
            if acc[i, j - 1] < best:
                best = acc[i, j - 1]
            acc[i, j] = C[i - 1, j - 1] + best
    return np.sqrt(acc[n, m])


@njit(cache=True)
def _frechet_dp(C):
    """Discrete Fréchet distance on a precomputed Euclidean distance matrix."""
    n, m = C.shape
    ca = np.empty((n, m))
    ca[0, 0] = C[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], C[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], C[0, j])
    for i in range(1, n):
        for j in range(1, m):
            best = ca[i - 1, j - 1]
            if ca[i - 1, j] < best:
                best = ca[i - 1, j]
            if ca[i, j - 1] < best:
                best = ca[i, j - 1]
            ca[i, j] = max(best, C[i, j])
    return ca[n - 1, m - 1]


def _dtw_band(window, n: int, m: int):
    """Band half-width in frames for paths of length n and m, or None for full DTW (see ``window`` in the docstring)."""
    if window is None:
        return None
    if isinstance(window, str):
        key = window.lower().replace("-", "_")
        if key == "sakoe_chiba":
            return max(4, int(0.08 * max(n, m)))
        if key == "sakoe_chiba_wide":
            # 8% band (>= 4 frames), widened to cover the length difference |n - m| (as used for the
            # T4 lysozyme paths in the paper); with dtaidistance this is window = max(w, |n - m| + 1)
            return max(max(4, int(0.08 * max(n, m))) - 1, abs(n - m))
        raise ValueError("window must be None, 'sakoe_chiba', 'sakoe_chiba_wide', an int or a float in (0, 1); "
                         f"got {window!r}")
    if isinstance(window, float) and 0 < window < 1:
        return max(1, int(window * max(n, m)))
    return int(window)


def _dtw(a: np.ndarray, b: np.ndarray, window: int | None) -> float:
    a, b = _c(a), _c(b)
    if not a.flags.writeable:      # memory-mapped worker inputs are read-only; dtaidistance's C code needs a writable buffer
        a = a.copy()
    if not b.flags.writeable:
        b = b.copy()
    if dtw_ndim is not None:
        # dtaidistance returns sqrt(sum of squared Euclidean costs), same as _dtw_dp
        # dtaidistance's window w is a band of w - 1 frames beyond the length difference (see _dtw_dp), hence + 1
        win = None if window is None else window + 1
        return float(dtw_ndim.distance(a, b, window=win, use_c=True))
    return float(_dtw_dp(cdist(a, b, "sqeuclidean"), max(len(a), len(b)) if window is None else window))


# =========================================================================== #
#  BASE METRICS (no water logic)                                              #
# =========================================================================== #

def _msd_set(X: np.ndarray, n_atoms: int) -> float:
    """n-ary MSD of a frame set in O(N): mean pairwise squared distance / N_atoms."""
    N = len(X)
    s = X.sum(axis=0)
    r = (X * X).sum(axis=0)
    return float(2.0 * (N * r.sum() - (s * s).sum()) / (N * N * n_atoms))


def _msd_dissimilarity(A: np.ndarray, B: np.ndarray, scheme: str, n_atoms: int) -> float:
    """SHINE dissimilarities, eqs. 4-8 of Chen et al., JCIM 2025."""
    U = np.vstack([A, B])
    m_u = _msd_set(U, n_atoms)
    if scheme == "intra":
        return m_u
    m_a, m_b = _msd_set(A, n_atoms), _msd_set(B, n_atoms)
    if scheme == "semi_sum":
        return m_u - 0.5 * (m_a + m_b)
    if scheme == "min":
        return m_u - min(m_a, m_b)
    if scheme == "max":
        return m_u - max(m_a, m_b)
    if scheme == "inter":
        na, nb = len(A), len(B)
        return ((na + nb) ** 2 * m_u - (na ** 2 * m_a + nb ** 2 * m_b)) / (na * nb)
    raise ValueError(scheme)


def _base_distance(
    xi, xj, ti, tj, *, metric, n_interp, window, normalize_length,
    msd_scheme, n_atoms, med_i=None, med_j=None, resample_all=False,
) -> float:
    if metric in ("l2", "l1"):
        yi, yj = _resample(xi, ti, n_interp), _resample(xj, tj, n_interp)
        diff = yi - yj
        dt = 1.0 / (n_interp - 1)
        if metric == "l2":
            return float(np.sqrt(max(_trapz((diff * diff).sum(axis=1), dt), 0.0)))
        return _trapz(np.abs(diff).sum(axis=1), dt)

    if metric == "dtw":
        if n_interp is not None and (resample_all or len(xi) > n_interp or len(xj) > n_interp):
            xi, xj = _resample(xi, ti, n_interp), _resample(xj, tj, n_interp)
        d = _dtw(xi, xj, _dtw_band(window, len(xi), len(xj)))
        # d = sqrt(sum of ~L step costs) ~ sqrt(L): divide by sqrt of the
        # lengths DTW actually saw -> RMS cost per step, length-independent
        return d / np.sqrt(len(xi) + len(xj)) if normalize_length else d

    if metric == "frechet":
        if n_interp is not None and (resample_all or len(xi) > n_interp or len(xj) > n_interp):
            xi, xj = _resample(xi, ti, n_interp), _resample(xj, tj, n_interp)
        return float(_frechet_dp(cdist(_c(xi), _c(xj))))

    if metric == "hausdorff":
            # Force arrays to be contiguous and writable in memory
            a = np.ascontiguousarray(_c(xi)).copy()
            b = np.ascontiguousarray(_c(xj)).copy()
            return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))

    if metric == "avg_hausdorff":
        a = xi if med_i is None else xi[med_i]
        b = xj if med_j is None else xj[med_j]
        C = cdist(_c(a), _c(b))
        return float(0.5 * C.min(axis=1).mean() + 0.5 * C.min(axis=0).mean())

    if metric == "msd":
        return _msd_dissimilarity(_c(xi), _c(xj), msd_scheme, n_atoms)

    raise ValueError(f"Unknown metric {metric!r}")


# =========================================================================== #
#  WATER PERMUTATION MATCHING                                                 #
# =========================================================================== #

VALID_WATER_MATCH = {"auto", "l2", "l1", "dtw", "frechet", "hausdorff", "mean"}
GRID_WATER_MATCH = {"l2", "l1", "dtw", "frechet", "hausdorff"}     # need the per-water paths on the grid


@njit(cache=True)
def _frechet_block(Wi, Wj):
    """Discrete Fréchet distance between every water path of i and of j. Wi, Wj: (K, T, c)."""
    K, T = Wi.shape[0], Wi.shape[1]
    out = np.empty((K, K))
    C = np.empty((T, T))
    for a in range(K):
        for b in range(K):
            for p in range(T):
                for q in range(T):
                    acc = 0.0
                    for c in range(Wi.shape[2]):
                        d = Wi[a, p, c] - Wj[b, q, c]
                        acc += d * d
                    C[p, q] = np.sqrt(acc)
            out[a, b] = _frechet_dp(C)
    return out


@njit(cache=True)
def _hausdorff_block(Wi, Wj):
    """Symmetric Hausdorff distance between the position sets of every water of i and of j. Wi, Wj: (K, T, c)."""
    K, T, nc = Wi.shape[0], Wi.shape[1], Wi.shape[2]
    out = np.empty((K, K))
    for a in range(K):
        for b in range(K):
            h_ab = 0.0
            for p in range(T):                       # directed a -> b
                best = 1e300
                for q in range(T):
                    acc = 0.0
                    for c in range(nc):
                        d = Wi[a, p, c] - Wj[b, q, c]
                        acc += d * d
                    if acc < best:
                        best = acc
                if best > h_ab:
                    h_ab = best
            h_ba = 0.0
            for q in range(T):                       # directed b -> a
                best = 1e300
                for p in range(T):
                    acc = 0.0
                    for c in range(nc):
                        d = Wj[b, q, c] - Wi[a, p, c]
                        acc += d * d
                    if acc < best:
                        best = acc
                if best > h_ba:
                    h_ba = best
            out[a, b] = np.sqrt(max(h_ab, h_ba))
    return out


def _water_cost(wi, wj, method, n_grid, mi=None, mj=None):
    """K x K matching cost. wi, wj: (n_grid, K, c) water paths on the progress grid; mi, mj: (K, c) means."""
    if method == "mean":
        return cdist(mi, mj, "sqeuclidean")
    k = wi.shape[1]
    if method in ("l2", "l1"):
        diff = wi[:, :, None, :] - wj[:, None, :, :]                  # (T, K, K, c)
        f = np.einsum("tabc,tabc->tab", diff, diff) if method == "l2" else np.abs(diff).sum(axis=3)
        return ((f[:-1] + f[1:]) * 0.5).sum(axis=0)                    # (K, K), trapezoid on the grid
    if method == "dtw":
        win = max(2, int(0.10 * n_grid))
        series = [np.ascontiguousarray(wi[:, a, :]) for a in range(k)] + [np.ascontiguousarray(wj[:, b, :]) for b in range(k)]
        if dtw_ndim is not None:
            M = dtw_ndim.distance_matrix_fast(series, ndim=wi.shape[2], window=win,
                                              block=((0, k), (k, 2 * k)), compact=False, parallel=False)
            return np.asarray(M[:k, k:], dtype=np.float64)
        return np.array([[_dtw(series[a], series[k + b], win) for b in range(k)] for a in range(k)])
    if method == "frechet":
        return _frechet_block(np.ascontiguousarray(wi.transpose(1, 0, 2)), np.ascontiguousarray(wj.transpose(1, 0, 2)))
    if method == "hausdorff":
        return _hausdorff_block(np.ascontiguousarray(wi.transpose(1, 0, 2)), np.ascontiguousarray(wj.transpose(1, 0, 2)))
    raise ValueError(f"Unknown water_match {method!r}; choose from {sorted(VALID_WATER_MATCH - {'auto'})}")


def _water_assignment(xi, xj, ti, tj, *, ws, we, k, cpw, method, n_grid) -> np.ndarray:
    """Return col_ind such that water b = col_ind[a] of traj j is matched to water a of traj i.
    (Per-pair version, kept for direct use; pairwise_distances uses pre-resampled grids.)"""
    if method == "mean":
        cost = _water_cost(None, None, method, n_grid,
                           xi[:, ws:we].reshape(-1, k, cpw).mean(axis=0), xj[:, ws:we].reshape(-1, k, cpw).mean(axis=0))
    else:
        wi = _resample(xi[:, ws:we], ti, n_grid).reshape(n_grid, k, cpw)
        wj = _resample(xj[:, ws:we], tj, n_grid).reshape(n_grid, k, cpw)
        cost = _water_cost(wi, wj, method, n_grid)
    _, col_ind = linear_sum_assignment(cost)
    return col_ind


def _permute_waters(xj, col_ind, ws, we, k, cpw) -> np.ndarray:
    T = len(xj)
    w = xj[:, ws:we].reshape(T, k, cpw)[:, col_ind, :].reshape(T, -1)
    return np.hstack([xj[:, :ws], w, xj[:, we:]])


# =========================================================================== #
#  PAIR WORKER — module level, explicit arguments (safe for loky)             #
# =========================================================================== #

def _pair_worker(i, j, ti, xi, tj, xj, med_i, med_j, cfg):
    """Per-pair version (as in fdm_metrics.py), kept for direct use."""
    if cfg["water"]:
        ws, we, k, cpw = cfg["ws"], cfg["we"], cfg["k"], cfg["cpw"]
        col = _water_assignment(
            xi, xj, ti, tj, ws=ws, we=we, k=k, cpw=cpw,
            method=cfg["water_match"], n_grid=cfg["water_grid"],
        )
        xj = _permute_waters(xj, col, ws, we, k, cpw)
    d = _base_distance(
        xi, xj, ti, tj,
        metric=cfg["metric"], n_interp=cfg["n_interp"], window=cfg["window"],
        normalize_length=cfg["normalize_length"], msd_scheme=cfg["msd_scheme"],
        n_atoms=cfg["n_atoms"], med_i=med_i, med_j=med_j,
        resample_all=cfg["resample_all"],
    )
    return i, j, d


def _chunk_worker(pairs, store, cfg):
    """Distances for a chunk of pairs from per-trajectory data prepared once in pairwise_distances.

    store: raw (concatenated frames) + off (offsets), times, R (stacked trajectories resampled to n_interp, or
    None), W (water paths on the grid, or None), Wm (mean water positions, or None), meds.
    Reproduces _pair_worker exactly: the same arrays reach the same metric; resampling was done up front."""
    raw, off, times, R, W, Wm, meds = (store[k] for k in ("raw", "off", "times", "R", "W", "Wm", "meds"))
    metric, n_interp = cfg["metric"], cfg["n_interp"]
    out = []
    for i, j in pairs:
        xi, xj = raw[off[i]:off[i + 1]], raw[off[j]:off[j + 1]]
        ti, tj = times[i], times[j]
        # the arrays _base_distance would end up comparing (resampled or not, as in _base_distance)
        if metric in ("l2", "l1"):
            use_r = True
        elif metric in ("dtw", "frechet"):
            use_r = n_interp is not None and (cfg["resample_all"] or len(xi) > n_interp or len(xj) > n_interp)
        else:
            use_r = False
        ai, aj = (R[i], R[j]) if use_r else (xi, xj)
        if cfg["water"]:
            ws, we, k, cpw = cfg["ws"], cfg["we"], cfg["k"], cfg["cpw"]
            method = cfg["water_match"]
            cost = _water_cost(W[i] if W is not None else None, W[j] if W is not None else None, method,
                               cfg["water_grid"], Wm[i] if Wm is not None else None, Wm[j] if Wm is not None else None)
            _, col = linear_sum_assignment(cost)
            aj = _permute_waters(np.asarray(aj), col, ws, we, k, cpw)
        d = _base_distance_pre(ai, aj, metric=metric, window=cfg["window"], n_interp=n_interp,
                               normalize_length=cfg["normalize_length"], msd_scheme=cfg["msd_scheme"],
                               n_atoms=cfg["n_atoms"], med_i=meds[i], med_j=meds[j])
        out.append((i, j, d))
    return out


def _base_distance_pre(ai, aj, *, metric, window, n_interp, normalize_length, msd_scheme, n_atoms, med_i, med_j):
    """_base_distance on arrays that are already resampled where _base_distance would resample them."""
    if metric in ("l2", "l1"):
        diff = ai - aj
        dt = 1.0 / (n_interp - 1)
        if metric == "l2":
            return float(np.sqrt(max(_trapz((diff * diff).sum(axis=1), dt), 0.0)))
        return _trapz(np.abs(diff).sum(axis=1), dt)
    if metric == "dtw":
        d = _dtw(ai, aj, _dtw_band(window, len(ai), len(aj)))
        return d / np.sqrt(len(ai) + len(aj)) if normalize_length else d
    if metric == "frechet":
        return float(_frechet_dp(cdist(_c(ai), _c(aj))))
    if metric == "hausdorff":
        a = np.ascontiguousarray(_c(ai)).copy()
        b = np.ascontiguousarray(_c(aj)).copy()
        return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))
    if metric == "avg_hausdorff":
        a = ai if med_i is None else ai[med_i]
        b = aj if med_j is None else aj[med_j]
        C = cdist(_c(a), _c(b))
        return float(0.5 * C.min(axis=1).mean() + 0.5 * C.min(axis=0).mean())
    if metric == "msd":
        return _msd_dissimilarity(_c(ai), _c(aj), msd_scheme, n_atoms)
    raise ValueError(f"Unknown metric {metric!r}")


# =========================================================================== #
#  INPUT HANDLING, MEDOIDS, DISTANCE MATRIX                                   #
# =========================================================================== #

def _extract_all(X):
    """Return (list_of_times, list_of_arrays, water_info_or_None)."""
    winfo = getattr(X, "water_info", None)
    if skfda is not None and isinstance(X, skfda.FDataIrregular):
        times, trajs = [], []
        for i in range(X.n_samples):
            s = int(X.start_indices[i])
            e = int(X.start_indices[i + 1]) if i + 1 < X.n_samples else len(X.values)
            times.append(np.asarray(X.points[s:e]).ravel())
            trajs.append(np.asarray(X.values[s:e], dtype=np.float64))
        return times, trajs, winfo
    if skfda is not None and isinstance(X, skfda.FDataGrid):
        g = np.asarray(X.grid_points[0]).ravel()
        return [g] * X.n_samples, [np.asarray(X.data_matrix[i], np.float64) for i in range(X.n_samples)], winfo
    if isinstance(X, (list, tuple)):
        times, trajs = [], []
        for item in X:
            if isinstance(item, tuple) and len(item) == 2:
                t, a = item
            else:
                a = item
                t = np.linspace(0.0, 1.0, len(a))
            a = np.asarray(a, dtype=np.float64)
            times.append(np.asarray(t, np.float64).ravel())
            trajs.append(a[:, None] if a.ndim == 1 else a)
        return times, trajs, winfo
    raise TypeError("X must be FDataIrregular, FDataGrid, or a list of (T, d) arrays.")


def _medoid_indices(traj: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k representative frames: k-means, then the frame nearest each centroid.
    Trajectories with <= k frames keep all frames (as in PRISM)."""
    from sklearn.cluster import KMeans
    T = len(traj)
    if T <= k:
        return np.arange(T)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(traj)
    idx = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        d = ((traj[members] - km.cluster_centers_[c]) ** 2).sum(axis=1)
        idx.append(members[np.argmin(d)])
    return np.unique(np.asarray(idx))


def pairwise_distances(
    X,
    metric: str = "dtw",
    *,
    water: bool = False,
    water_info: dict | None = None,
    water_match: str = "auto",
    water_grid: int = 35,
    window: int | float | str | None = None,
    subsample_t: int | None = 100,
    normalize_length: bool = False,
    resample_all: bool = False,
    standardize: bool = False,
    msd_scheme: str = "intra",
    n_atoms: int = 1,
    n_medoids: int | None = None,
    n_jobs: int = 4,
    verbose: int = 0,
) -> np.ndarray:
    """Symmetric (n, n) pathway dissimilarity matrix with zero diagonal."""
    metric = metric.lower()
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}")
    if metric == "msd" and msd_scheme not in VALID_MSD_SCHEMES:
        raise ValueError(f"msd_scheme must be one of {sorted(VALID_MSD_SCHEMES)}")

    times, trajs, x_winfo = _extract_all(X)
    n = len(trajs)

    # ── Standardise feature columns globally (all frames, all trajectories) ──
    if standardize and water:
        warnings.warn("standardize=True with water=True gives each water column its own "
                      "scale and breaks permutation invariance; use standardize=False.")
    if standardize:
        allf = np.vstack(trajs)
        mu, sd = allf.mean(axis=0), allf.std(axis=0)
        sd[sd == 0] = 1.0
        trajs = [(t - mu) / sd for t in trajs]

    # ── Water layout ──
    cfg = dict(
        metric=metric, n_interp=subsample_t if metric in ORDER_AWARE else None,
        window=window, normalize_length=normalize_length, msd_scheme=msd_scheme,
        n_atoms=n_atoms, water=False, resample_all=resample_all,
    )
    if metric in ("l2", "l1") and cfg["n_interp"] is None:
        cfg["n_interp"] = 80
    if water:
        winfo = water_info or x_winfo
        if not winfo or not winfo.get("has_water", True) or not winfo.get("k_waters", 0):
            warnings.warn("water=True but no usable water_info — computing without water matching.")
        else:
            ws = int(winfo["solute_dim"])
            k = int(winfo["k_waters"])
            cpw = int(winfo.get("atoms_per_water", 1)) * 3
            if water_match not in VALID_WATER_MATCH:
                raise ValueError(f"water_match must be one of {sorted(VALID_WATER_MATCH)}")
            if water_match == "auto":
                water_match = "dtw" if metric == "dtw" else ("l2" if metric in ORDER_AWARE else "mean")
            cfg.update(water=True, ws=ws, we=ws + k * cpw, k=k, cpw=cpw,
                       water_match=water_match, water_grid=water_grid)

    # ── PRISM medoids: computed ONCE per trajectory, not per pair.
    #    k-means is invariant to column permutation, so medoid indices stay
    #    valid after the water columns of trajectory j are re-labelled.
    meds = [None] * n
    if metric == "avg_hausdorff" and n_medoids:
        meds = [_medoid_indices(t, n_medoids) for t in trajs]

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if verbose:
        print(f"{len(pairs)} pairs | metric={metric}"
              f"{'/' + msd_scheme if metric == 'msd' else ''} | water={cfg['water']}"
              f"{'/' + cfg['water_match'] if cfg['water'] else ''}"
              f" | standardize={standardize} | normalize_length={normalize_length}")

    # ── Per-trajectory work done once (it used to be repeated for every pair) ──
    trajs = [np.ascontiguousarray(t, dtype=np.float64) for t in trajs]
    off = np.concatenate([[0], np.cumsum([len(t) for t in trajs])]).astype(np.intp)
    R = None
    if cfg["n_interp"] is not None and metric in ORDER_AWARE:
        R = np.stack([_resample(t, tm, cfg["n_interp"]) for t, tm in zip(trajs, times)])
    W = Wm = None
    if cfg["water"]:
        ws, we, k, cpw = cfg["ws"], cfg["we"], cfg["k"], cfg["cpw"]
        if cfg["water_match"] in GRID_WATER_MATCH:
            g = cfg["water_grid"]
            W = np.stack([_resample(t[:, ws:we], tm, g).reshape(g, k, cpw) for t, tm in zip(trajs, times)])
        else:
            Wm = np.stack([t[:, ws:we].reshape(-1, k, cpw).mean(axis=0) for t in trajs])
    store = dict(raw=np.concatenate(trajs), off=off, times=times, R=R, W=W, Wm=Wm, meds=meds)

    n_workers = n_jobs if n_jobs > 0 else max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    if n_workers == 1 or len(pairs) < 2:
        chunks_out = [_chunk_worker(pairs, store, cfg)]
    else:
        n_chunks = min(len(pairs), 8 * n_workers)
        chunks = [pairs[c::n_chunks] for c in range(n_chunks)]       # interleaved: similar cost per chunk
        chunks_out = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
            delayed(_chunk_worker)(c, store, cfg) for c in chunks)
    results = [r for chunk in chunks_out for r in chunk]

    D = np.zeros((n, n))
    for i, j, d in results:
        D[i, j] = D[j, i] = d

    # SHINE dissimilarities are only defined up to ranking and can be negative
    # (min/max/inter). Shift so the kernel sees non-negative distances.
    off = D[~np.eye(n, dtype=bool)]
    if off.min() < 0:
        warnings.warn(f"Negative dissimilarities (min={off.min():.4g}); shifting to >= 0.")
        D = D - off.min()
    np.fill_diagonal(D, 0.0)
    return D


# =========================================================================== #
#  FDM ESTIMATOR                                                              #
# =========================================================================== #

class FDM(TransformerMixin, BaseEstimator):
    """
    Functional Diffusion Maps over any of the metrics above.

    Eigendecomposition uses the symmetric conjugate
        S = Q^{-1/2} K_alpha Q^{-1/2},   P = Q^{-1} K_alpha
    which has exactly the eigenvalues of the Markov matrix P; right
    eigenvectors of P are recovered as psi = Q^{-1/2} v. This is exact
    (unlike symmetrising P itself) and faster/more stable than eig(P).
    """

    def __init__(
        self,
        n_components: int | float = 2,
        kernel: str = "rbf",
        metric: str = "dtw",
        sigma: str | float = "percentil",
        step: int = 1,
        alpha: float = 0.0,
        percentil: float = 50,
        water: bool = False,
        water_info: dict | None = None,
        water_match: str = "auto",
        water_grid: int = 35,
        window: int | float | str | None = None,
        subsample_t: int | None = 100,
        normalize_length: bool = False,
        resample_all: bool = False,
        standardize: bool = False,
        msd_scheme: str = "intra",
        n_atoms: int = 1,
        n_medoids: int | None = None,
        n_jobs: int = 4,
        verbose: int = 0,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.metric = metric
        self.sigma = sigma
        self.step = step
        self.alpha = alpha
        self.percentil = percentil
        self.water = water
        self.water_info = water_info
        self.water_match = water_match
        self.water_grid = water_grid
        self.window = window
        self.subsample_t = subsample_t
        self.normalize_length = normalize_length
        self.resample_all = resample_all
        self.standardize = standardize
        self.msd_scheme = msd_scheme
        self.n_atoms = n_atoms
        self.n_medoids = n_medoids
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ── helpers ──────────────────────────────────────────────────────────────
    def _sigma_from(self, D):
        if self.sigma in ("median", "maximum", "percentil"):
            p = {"median": 50, "maximum": 100}.get(self.sigma, self.percentil)
            off = D[~np.eye(len(D), dtype=bool)]          # exclude zero diagonal
            return float(np.percentile(off, p))
        if isinstance(self.sigma, (int, float)) and self.sigma > 0:
            return float(self.sigma)
        raise ValueError(f"Invalid sigma: {self.sigma!r}")

    def _n_keep(self, ev):
        if isinstance(self.n_components, (int, np.integer)):
            return int(self.n_components)
        thr = self.n_components * abs(ev[0]) ** self.step
        return max(1, int(np.count_nonzero(np.abs(ev) ** self.step > thr)) - 1)

    # ── public API ───────────────────────────────────────────────────────────
    def compute_distances(self, X) -> np.ndarray:
        return pairwise_distances(
            X, self.metric,
            water=self.water, water_info=self.water_info,
            water_match=self.water_match, water_grid=self.water_grid,
            window=self.window, subsample_t=self.subsample_t,
            normalize_length=self.normalize_length, resample_all=self.resample_all,
            standardize=self.standardize,
            msd_scheme=self.msd_scheme, n_atoms=self.n_atoms,
            n_medoids=self.n_medoids, n_jobs=self.n_jobs, verbose=self.verbose,
        )

    def fit(self, X=None, y=None, *, precomputed_D: np.ndarray | None = None):
        D = self.compute_distances(X) if precomputed_D is None else np.asarray(precomputed_D, float)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        self.D_matrix = D

        self.sigma_ = self._sigma_from(D)
        if self.kernel == "rbf":
            K = np.exp(-(D ** 2) / (2.0 * self.sigma_ ** 2))
        elif self.kernel == "laplacian":
            K = np.exp(-D / self.sigma_)
        else:
            raise ValueError(f"Invalid kernel: {self.kernel!r}")
        self.K_matrix = K

        d = K.sum(axis=1) ** self.alpha
        K_alpha = K / np.outer(d, d)
        q = K_alpha.sum(axis=1)
        self.P_matrix = K_alpha / q[:, None]

        S = K_alpha / np.sqrt(np.outer(q, q))
        ev, V = eigh(0.5 * (S + S.T))
        ev, V = ev[::-1], V[:, ::-1]
        psi = V / V[:, [0]]                 # right eigenvectors of P; psi_0 == 1

        self.eigenvals_full = ev
        self.eigenvecs_full = psi

        self.n_components_ = self._n_keep(ev[1:])
        self.eigenvals = ev[1 : self.n_components_ + 1]
        vecs = psi[:, 1 : self.n_components_ + 1]
        self.eigenvecs, _ = svd_flip(vecs, np.zeros_like(vecs).T)

        if self.verbose:
            print(f"σ={self.sigma_:.4g} | λ1..{self.n_components_} = "
                  f"{np.array2string(self.eigenvals, precision=4)}")
        return self

    def fit_transform(self, X=None, y=None, *, precomputed_D: np.ndarray | None = None):
        self.fit(X, precomputed_D=precomputed_D)
        self.embedding = (self.eigenvals ** self.step) * self.eigenvecs
        return self.embedding

    def transform(self, X=None):
        if not hasattr(self, "embedding"):
            raise RuntimeError("Call fit_transform() first.")
        return self.embedding