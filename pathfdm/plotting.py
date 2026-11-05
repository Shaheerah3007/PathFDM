"""Plots: trajectory space (FDM coordinates), surrogate parity plots, SHAP beeswarm."""
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
