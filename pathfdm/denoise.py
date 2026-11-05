"""Predictive (nearest-neighbour) noise reduction of trajectories in a delay embedding.

Used to smooth per-frame descriptor time series before the path statistics are computed.
"""
import numpy as np


class PredictiveNoiseReducer:
    """Predictive (nearest-neighbour) denoiser in a delay embedding. Frame-to-frame distances and their ranking are
    computed once per iteration and reused by every forward / backward prediction."""

    def __init__(self, m=5, tau=1, k_nn=15, prediction_horizon=3, bidirectional=True):
        self.m = m
        self.tau = tau
        self.k_nn = k_nn
        self.prediction_horizon = prediction_horizon
        self.bidirectional = bidirectional

    def _embed(self, traj):
        T, d = traj.shape
        n_embed = T - (self.m - 1) * self.tau
        return np.hstack([traj[i * self.tau:i * self.tau + n_embed] for i in range(self.m)])

    def _neighbour_order(self, E):
        """order[q] = frames sorted by distance to frame q. Squared distances from one matrix product
        (|a|^2 + |b|^2 - 2 a.b), which ranks neighbours the same as the explicit norm but ~15x faster."""
        sq = np.einsum("ij,ij->i", E, E)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (E @ E.T)
        np.fill_diagonal(d2, -1.0)              # the query itself always ranks first, as with the exact zero distance
        return np.argsort(d2, axis=1)

    def denoise(self, traj, n_iterations=2):
        traj = np.asarray(traj, dtype=float)
        is_1d = traj.ndim == 1
        if is_1d:
            traj = traj[:, np.newaxis]
        T, d = traj.shape
        if T <= (self.m - 1) * self.tau + 2:
            return traj[:, 0] if is_1d else traj

        current = traj.copy()
        k = self.k_nn
        for _ in range(n_iterations):
            E = self._embed(current)
            n = len(E)
            order = self._neighbour_order(E)
            nbr = order[:, 1:k + 1]                       # k nearest, excluding the first (the query itself)
            preds = np.empty_like(E)
            for p in range(n):
                acc, cnt = None, 0
                for h in range(1, min(self.prediction_horizon + 1, n - p)):      # forward, query = p - 1
                    if p < 1 or p + h >= n:
                        continue
                    succ = nbr[p - 1] + h
                    succ = succ[succ < n]
                    if len(succ):
                        v = E[succ].mean(axis=0); acc = v if acc is None else acc + v; cnt += 1
                if self.bidirectional:
                    for h in range(1, min(self.prediction_horizon + 1, p + 1)):  # backward, query = p + h
                        if p + h >= n or p - h < 0:
                            continue
                        pred = nbr[p + h] - h
                        pred = pred[pred >= 0]
                        if len(pred):
                            v = E[pred].mean(axis=0); acc = v if acc is None else acc + v; cnt += 1
                preds[p] = acc / cnt if cnt else E[p]
            # fold the delay-embedded predictions back onto frames (average over the m delays covering each frame)
            new = np.zeros((T, d)); counts = np.zeros(T)
            for j in range(self.m):
                t0 = j * self.tau
                new[t0:t0 + n] += preds[:, j * d:(j + 1) * d]; counts[t0:t0 + n] += 1
            valid = counts > 0
            new[valid] /= counts[valid, None]
            new[~valid] = current[~valid]
            current = new
        return current[:, 0] if is_1d else current