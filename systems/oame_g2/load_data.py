"""OAMe host - G2 guest (4-cyanobenzoate) unbinding: 95 reactive trajectories, dry vs wet routes.

FDM input : guest + host coordinates (fit on host atoms 15-54), optionally followed by the K water oxygens
            that come closest to the host-guest complex at any frame (paper: K = 32), matched between
            trajectories by the Hungarian assignment
Labels    : trajectory_labels.txt (1 = dry, 0 = wet)
Descriptors
    map with water  : water coordinations d1-d12 (PLUMED-style switching functions, normalized):
                      d1-d4 around four guest atoms (L1-L4), d5-d12 around eight virtual sites V1-V8 on the
                      host axis
    solute-only map : guest geometry x, y, r (lateral position of the guest centre relative to the host axis),
                      cos theta (long axis vs host axis), ring normal n_x, n_y, |n_z|, and tau_COO
                      (|cos| between carboxylate-plane and ring normals)

Edit DATA to point at your copy of the data.
"""
from __future__ import annotations

import numpy as np

from pathfdm import find_trajectories, load_trajectories

DATA = {
    "traj_dir": "/home/shaheerah/functional-diffusion-maps/Unbinding_Paths/OAMe_G2",
    "topology": "/home/shaheerah/functional-diffusion-maps/Unbinding_Paths/OAMe_G2/box.gro",
    "reference": "/home/shaheerah/functional-diffusion-maps/Unbinding_Paths/OAMe_G2/conf_template.gro",
    "labels": "/home/shaheerah/functional-diffusion-maps/Unbinding_Paths/OAMe_G2/trajectory_labels.txt",
}

OPTIONS = {
    "solute_sel": "resname MOL or resname OCB",   # guest (15 atoms, first) + host (196 atoms)
    "align_sel": "index 15-54",                    # host atoms used for the fit
    "align_mode": "fit",                           # unwrap + rigid fit (needed for waters)
    "align_weights": "mass",
    "time": "progress",
    "start_frame": 0,
    "max_frames": 5000,
}
K_WATERS = 32                                     # paper setting
GUEST_SEL = "resname MOL"

LABEL_CONFIG = {1: {"color": "#EF553B", "name": "Dry path"}, 0: {"color": "#636EFA", "name": "Wet path"}}

V_SITES = np.array([[20.136, 20.136, 20.0 + 2.5 * i] for i in range(8)])   # virtual sites V1-V8 (Angstrom)
N_GUEST = 15
HYDRATION_NAMES = [f"d_{{{i}}}" for i in range(1, 13)]
GUEST_NAMES = ["x", "y", "r", r"\cos\theta", "n_x", "n_y", "|n_z|", r"\tau_{COO}"]


def load(k_waters: int | None = K_WATERS, solute_sel: str | None = None, n_jobs: int = 8):
    """TrajectorySet (solute + k_waters closest waters; k_waters=None or 0: solute only) and labels."""
    opts = dict(OPTIONS)
    if solute_sel:
        opts["solute_sel"] = solute_sel
    water = {"pick": "closest", "k": k_waters} if k_waters else None
    ts = load_trajectories(find_trajectories(DATA["traj_dir"], "*.dcd"), DATA["topology"], DATA["reference"],
                           water=water, n_jobs=n_jobs, **opts)
    return ts, labels_for(ts)


def labels_for(ts):
    lab = {}
    for line in open(DATA["labels"]):
        p = line.split()
        if len(p) >= 2:
            lab[p[0].rsplit(".", 1)[0]] = int(p[1])
    return np.array([lab.get(n, -1) for n in ts.names])


# ── per-frame descriptors ─────────────────────────────────────────────────────────────────────────────────
def _coordination(a, b, r0, nn, mm, d_max):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).ravel()
    d = np.maximum(d[d <= d_max], 1e-8) / r0
    return float(np.sum((1.0 - d ** nn) / (1.0 - d ** mm)))


def hydration(coords, n_solute_atoms):
    """d1-d12 of every frame. coords: (n_frames, 3 * (n_solute + K)) with the guest atoms first."""
    out = np.empty((len(coords), 12))
    for f, row in enumerate(np.asarray(coords, float)):
        x = row.reshape(-1, 3)
        sol, wat = x[:n_solute_atoms], x[n_solute_atoms:]
        L = [_coordination(sol[i:i + 1], wat, 2.5, 6, 10, 13.0) / 2.5 - 1.0 for i in (0, 3, 7, 8)]
        V = [_coordination(V_SITES[i:i + 1], wat, 2.5, 2, 6, 13.0) / 2.8 - 1.0 for i in range(8)]
        out[f] = L + V
    return out


def guest_geometry(coords):
    """x, y, r, cos theta, n_x, n_y, |n_z|, tau_COO of every frame (guest atoms first in coords)."""
    g = np.asarray(coords, dtype=np.float64).reshape(len(coords), -1, 3)[:, :N_GUEST]
    centre = g[:, [0, 1, 2, 3, 4, 5, 6, 8]].mean(axis=1)
    x, y = centre[:, 0] - V_SITES[0, 0], centre[:, 1] - V_SITES[0, 1]
    axis = (V_SITES[2] - V_SITES[4]) / np.linalg.norm(V_SITES[2] - V_SITES[4])
    long_axis = g[:, 7] - g[:, 5]
    cos_t = long_axis @ axis / np.linalg.norm(long_axis, axis=1)
    ring = g[:, 0:6] - g[:, 0:6].mean(axis=1, keepdims=True)
    n = np.linalg.svd(ring)[2][:, 2]
    n *= np.where(n[:, 2:3] < 0, -1.0, 1.0)
    coo = np.cross(g[:, 9] - g[:, 8], g[:, 10] - g[:, 8])
    coo /= np.linalg.norm(coo, axis=1, keepdims=True)
    tau = np.abs(np.sum(coo * n, axis=1))
    return np.c_[x, y, np.hypot(x, y), cos_t, n[:, 0], n[:, 1], np.abs(n[:, 2]), tau]


def hydration_descriptors(ts):
    return [hydration(c, ts.water_info["n_solute_atoms"]) for c in ts.coords]


def guest_descriptors(ts):
    return [guest_geometry(c) for c in ts.coords]
