"""Plots: trajectory space (FDM coordinates), descriptors along the paths, surrogate parity plots, SHAP beeswarm."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def apply_style():
    """Publication style used for the paper figures (serif fonts, inward ticks, small labels)."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee"])
    except Exception:
        pass
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "C059", "Times New Roman"], "font.size": 10,
        "figure.dpi": 150, "text.usetex": False, "mathtext.fontset": "stix",
        "axes.labelsize": 8, "axes.titlesize": 8, "axes.titlelocation": "left", "axes.linewidth": 0.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.top": False, "ytick.right": False,
        "legend.frameon": False, "legend.fontsize": 6, "scatter.edgecolors": "black",
    })


def plot_space(Z, labels=None, label_config=None, dims=(0, 1), ax=None, title=None, s=14, annotate=None,
               legend=True, c=None, cmap="viridis", clabel=None):
    """Scatter of two FDM coordinates, one point per trajectory, coloured by label (or by values ``c``)."""
    ax = ax or plt.subplots(figsize=(3.2, 2.6))[1]
    i, j = dims
    if c is not None:
        sc = ax.scatter(Z[:, i], Z[:, j], c=c, cmap=cmap, s=s, edgecolors="k", linewidths=0.3)
        cb = plt.colorbar(sc, ax=ax, pad=0.02)
        if clabel:
            cb.set_label(clabel, fontsize=7)
    elif labels is None:
        ax.scatter(Z[:, i], Z[:, j], s=s, color="#999999", edgecolors="k", linewidths=0.3)
    else:
        labels = np.asarray(labels)
        for lab, cfg in label_config.items():
            m = labels == lab
            if m.any():
                ax.scatter(Z[m, i], Z[m, j], s=s, color=cfg["color"], edgecolors="k", linewidths=0.3,
                           label=f"{cfg['name']} ({m.sum()})")
        if legend:
            ax.legend(fontsize=6, loc="best")
    if annotate is not None:
        for k, txt in enumerate(annotate):
            ax.annotate(str(txt), (Z[k, i], Z[k, j]), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"FDM {i}")
    ax.set_ylabel(f"FDM {j}")
    if title:
        ax.set_title(title, fontsize=8)
    return ax


def plot_parity(res, labels=None, label_config=None, axs=None, title=None):
    """Predicted vs true coordinate: out-of-bag (left) and mean held-out cross-validation prediction (right)."""
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(6, 2.4))
    preds = [("OOB prediction", res.y_oob, f"R$^2$ = {res.oob_r2:.3f}"),
             ("CV prediction", res.y_cv, f"R$^2$ = {res.cv_r2.mean():.3f} $\\pm$ {res.cv_r2.std():.3f}")]
    y = res.y
    for ax, (kind, yh, r2) in zip(axs, preds):
        if labels is None:
            ax.scatter(y, yh, s=10, color="#999999", edgecolors="k", linewidths=0.3)
        else:
            for lab, cfg in label_config.items():
                m = np.asarray(labels) == lab
                if m.any():
                    ax.scatter(y[m], yh[m], s=10, color=cfg["color"], edgecolors="k", linewidths=0.3,
                               label=cfg["name"])
        lo, hi = min(y.min(), yh.min()), max(y.max(), yh.max())
        pad = 0.05 * (hi - lo)
        xx = np.array([lo - pad, hi + pad])
        ax.plot(xx, xx, "k--", lw=0.8, label="y = x")
        slope, icpt = np.polyfit(y, yh, 1)
        ax.plot(xx, slope * xx + icpt, color="#2a9d8f", lw=1.2, label=f"{r2}, slope {slope:.2f}")
        ax.set_xlim(*xx); ax.set_ylim(*xx)
        ax.set_xlabel("FDM 0 (true)"); ax.set_ylabel(f"FDM 0 ({kind})")
        ax.legend(fontsize=5, loc="lower right")
        if title:
            ax.set_title(title, fontsize=7)
    return axs


def plot_beeswarm(res, max_display=10, title=None, figsize=(3.4, 2.6)):
    """SHAP beeswarm: one point per trajectory, colour = descriptor value (blue high, red low)."""
    import shap
    plt.figure(figsize=figsize)
    shap.summary_plot(res.shap_values, res.X, feature_names=res.names, max_display=max_display, show=False,
                      plot_size=None, cmap=plt.get_cmap("coolwarm_r"))
    ax = plt.gca()
    for coll in ax.collections:
        coll.set_sizes([6])
    ax.tick_params(labelsize=6)
    ax.set_xlabel("SHAP value", fontsize=7)
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=6)
    cbar.set_ylabel("Feature value", fontsize=6)
    if title:
        ax.set_title(title, fontsize=8)
    plt.tight_layout()
    return ax


def _route_legend(ax, labels, label_config, fontsize=6):
    from matplotlib.lines import Line2D
    labels = np.asarray(labels)
    ax.legend(handles=[Line2D([], [], color=cfg["color"], ls="-", lw=1.5, label=f"{cfg['name']} ({(labels == lab).sum()})")
                       for lab, cfg in label_config.items() if (labels == lab).any()], fontsize=fontsize)


def plot_paths(xs, ys, labels=None, label_config=None, ax=None, xlabel=None, ylabel=None, lw=0.6, alpha=0.6,
               legend=True, title=None):
    """Every trajectory as a line in the plane of two per-frame quantities, coloured by label.

    xs, ys : lists (one entry per trajectory) of 1-D arrays of equal length per trajectory, e.g. Phi and Psi,
             or a descriptor and the progress coordinate.
    """
    ax = ax or plt.subplots(figsize=(3.2, 2.6))[1]
    labels = np.full(len(xs), -999) if labels is None else np.asarray(labels)
    cfg = label_config or {}
    for k in np.random.default_rng(0).permutation(len(xs)):     # random order: no route drawn entirely on top
        ax.plot(xs[k], ys[k], ls="-", lw=lw, alpha=alpha, color=cfg.get(labels[k], {"color": "#999999"})["color"])
    if legend and label_config:
        _route_legend(ax, labels, label_config)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    if title:
        ax.set_title(title, fontsize=8)
    return ax


def plot_binned(zs, vs, labels, label_config, edges, ax=None, min_paths=5, xlabel=None, ylabel=None, legend=True,
                title=None):
    """A per-frame quantity as a function of a progress coordinate, per label.

    Every trajectory is first averaged in each bin of ``edges`` (so it counts once per bin, however long it stays
    there); the line is the median over the trajectories of a label and the band the interquartile range. Bins
    reached by fewer than ``min_paths`` trajectories of a label are left out. Returns (ax, table).
    """
    import pandas as pd
    ax = ax or plt.subplots(figsize=(3.2, 2.2))[1]
    edges = np.asarray(edges, float)
    centres = 0.5 * (edges[1:] + edges[:-1])
    labels = np.asarray(labels)
    rows = []
    for lab, cfg in label_config.items():
        idx = np.flatnonzero(labels == lab)
        if not len(idx):
            continue
        A = np.full((len(idx), len(centres)), np.nan)
        for r, k in enumerate(idx):
            b = np.digitize(zs[k], edges) - 1
            v = np.asarray(vs[k], float)
            for j in np.unique(b[(b >= 0) & (b < len(centres))]):
                A[r, j] = v[b == j].mean()
        n = np.isfinite(A).sum(0)
        ok = n >= min_paths
        q1, med, q3 = np.nanpercentile(A[:, ok], [25, 50, 75], axis=0)
        ax.fill_between(centres[ok], q1, q3, color=cfg["color"], alpha=0.2, lw=0)
        ax.plot(centres[ok], med, color=cfg["color"], ls="-", lw=1.2, marker="o", ms=2.5, label=f"{cfg['name']} ({len(idx)})")
        rows += [dict(label=cfg["name"], bin_centre=c, median=m, q25=a, q75=b_, n_paths=int(k))
                 for c, m, a, b_, k in zip(centres[ok], med, q1, q3, n[ok])]
    if legend:
        ax.legend(fontsize=6)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    if title:
        ax.set_title(title, fontsize=8)
    return ax, pd.DataFrame(rows)


def top_descriptor(res, desc_names, rank=0):
    """Per-frame descriptor behind the ``rank``-th feature by SHAP share (features are 4 statistics per descriptor)."""
    j = np.argsort(res.shap_share)[::-1][rank]
    return j // (len(res.names) // len(desc_names))
