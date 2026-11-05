"""Scores comparing an embedding with known pathway labels."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score


def auc_per_coordinate(Z, labels, n=4):
    """AUC of each of the first n coordinates for two labels (orientation-free: max(AUC, 1 - AUC))."""
    labels = np.asarray(labels)
    ok = labels >= 0
    out = []
    for k in range(min(n, Z.shape[1])):
        a = roc_auc_score(labels[ok], Z[ok, k])
        out.append(max(a, 1 - a))
    return np.array(out)


def kmeans_ari(Z, labels, k=None, n_use=2, n_init=20, mask=None):
    """ARI between k-means clusters of the first n_use coordinates and the labels (k = number of labels)."""
    labels = np.asarray(labels)
    m = np.ones(len(labels), bool) if mask is None else np.asarray(mask)
    k = k or len(np.unique(labels[m]))
    pred = KMeans(n_clusters=k, n_init=n_init, random_state=0).fit_predict(Z[m, :n_use])
    return adjusted_rand_score(labels[m], pred)


def align_signs(Z, Z_ref):
    """Flip coordinates of Z so that each correlates positively with the same coordinate of Z_ref."""
    Z = Z.copy()
    for k in range(min(Z.shape[1], Z_ref.shape[1])):
        if np.corrcoef(Z[:, k], Z_ref[:, k])[0, 1] < 0:
            Z[:, k] *= -1
    return Z


def compare_embeddings(Z, Z_ref, n=3):
    """Spearman rho between coordinate k of two embeddings of the same trajectories."""
    return np.array([spearmanr(Z[:, k], Z_ref[:, k]).statistic for k in range(n)])
