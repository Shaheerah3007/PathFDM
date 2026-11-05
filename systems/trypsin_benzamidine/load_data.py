"""Benzamidine unbinding from trypsin: 50 PathGennie escape trajectories from state B, routes B vs B'.

FDM input : benzamidine coordinates (fit on the CA atoms of trypsin onto the funnel template), optionally
            followed by pocket water oxygens (Hungarian-matched between trajectories). Water selections:
              "sites"     the waters occupying the hydration sites V1-V16 in the first frame (paper)
              "near_atom" the waters within 4 A of Asp189 CG in the first frame
              "closest"   the K waters that come closest to benzamidine at any frame
Labels    : two-component Gaussian mixture on the mean s_t (PLUMED Deep-TICA CV) in the window
            z = 0.45-1.0 nm of the funnel coordinate: 0 = B' route, 1 = B route
Descriptors (per frame)
    ligand + water : radius, cos theta, |n_z| of benzamidine + occupancies of V1-V16, FG, FH (PLUMED)
    ligand only    : x, y, r, cos theta, a_x, a_y, n_x, n_y, |n_z| of benzamidine (z left out: it is the
                     progress coordinate of every path)

The PLUMED CVs (cyl.z, radius, st, sw, vwo1-16, FG, FH) are read from COLVAR files computed once with
plumed/plumed_descriptors.dat (PLUMED 2.9 with the pytorch module); see run_plumed() to recompute them.
Data: <repository>/data/<system>/ or $PATHFDM_DATA/<system>/ (see data/README.md).
"""
from __future__ import annotations

import os
from pathlib import Path

# data root: $PATHFDM_DATA if set, else <repository>/data (unpack the Zenodo archive there)
DATA_ROOT = Path(os.environ.get("PATHFDM_DATA", Path(__file__).resolve().parents[2] / "data")) / "trypsin_benzamidine"

import re

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from pathfdm import find_trajectories, load_trajectories

DATA = {
    "traj_dir": str(DATA_ROOT / "trajectories"),
    "topology": str(DATA_ROOT / "start_B.gro"),
    "reference": str(DATA_ROOT / "fit_tem.pdb"),                  # funnel template (223 CA atoms)
    "plumed_script": str(DATA_ROOT / "plumed_descriptors.dat"),
    "colvar_dir": str(DATA_ROOT / "colvars"),
    "colvar_prefix": "",
    "plumed_bin": "plumed",                                        # PLUMED with the pytorch module (only for run_plumed)
}

OPTIONS = {
    "solute_sel": "resname BEN",
    "align_sel": "index 0:3231 and name CA",
    "align_mode": "fit",
    "align_weights": None,          # equal weights, as PLUMED FIT_TO_TEMPLATE TYPE=OPTIMAL with unit occupancies
    "time": "progress",
    "start_frame": 0,
}

LABEL_CONFIG = {0: {"color": "#636EFA", "name": "B' path"}, 1: {"color": "#EF553B", "name": "B path"}}
Z_WINDOW = (0.45, 1.0)              # nm, window of the funnel coordinate used for the labels

ASP_CG_INDEX = 2479
V14_A = np.array([4.070, 3.450, 3.834]) * 10      # funnel axis point (Angstrom)
BEN_C_AMIDINE, BEN_RING, BEN_CN = 3, [7, 8, 10, 12, 14, 16], [0, 3, 4, 7, 8, 10, 12, 14, 16]
N_BEN = 18

HYDRATION_COLS = [f"vwo{i}" for i in range(1, 17)] + ["FG", "FH"]
HYDRATION_TEX = [f"V_{{{i}}}" for i in range(1, 17)] + ["F_G", "F_H"]
LIGAND_TEX = {"radius": "r", "costheta": r"\cos\theta", "ringz": "|n_z|"}
LIG_ONLY = ["x", "y", "r", "costheta", "ax_x", "ax_y", "n_x", "n_y", "ringz"]
LIG_ONLY_TEX = ["x", "y", "r", r"\cos\theta", "a_x", "a_y", "n_x", "n_y", "|n_z|"]


def hydration_sites():
    """V1-V16 positions (Angstrom) from the FIXEDATOM lines of the PLUMED script."""
    txt = open(DATA["plumed_script"]).read()
    m = sorted(re.finditer(r"^v(\d+):\s+FIXEDATOM AT=(\S+)", txt, re.M), key=lambda m: int(m.group(1)))
    return np.array([[float(x) * 10 for x in g.group(2).split(",")] for g in m])


WATER_PICKS = {
    "sites": lambda: {"pick": "sites", "sites": hydration_sites(), "cutoff": 2.0},
    "near_atom": lambda: {"pick": "near_atom", "atom_index": ASP_CG_INDEX, "cutoff": 4.0},
    "closest": lambda k=6: {"pick": "closest", "k": k},
}


def load(water_pick: str | None = "sites", k: int = 6, n_jobs: int = 8):
    """TrajectorySet (benzamidine + waters chosen by ``water_pick``; None = ligand only), COLVARs, labels."""
    water = None if water_pick is None else (WATER_PICKS[water_pick](k) if water_pick == "closest"
                                             else WATER_PICKS[water_pick]())
    files = find_trajectories(DATA["traj_dir"], "traj_*.dcd")
    ts = load_trajectories(files, DATA["topology"], DATA["reference"], water=water, n_jobs=n_jobs, **OPTIONS)
    cvs = [read_colvar(f) for f in ts.files]
    for c, n in zip(cvs, ts.lengths):
        assert len(c) == n, "COLVAR and trajectory lengths differ"
    return ts, cvs, labels_from_cvs(cvs)


def colvar_path(traj):
    name = os.path.splitext(os.path.basename(traj))[0]
    return os.path.join(DATA["colvar_dir"], f"{DATA['colvar_prefix']}{name}.colvar")


def read_colvar(traj):
    path = colvar_path(traj)
    with open(path) as f:
        header = f.readline().split()
    return pd.read_csv(path, sep=r"\s+", comment="#", names=header[2:])


def run_plumed(traj, out_dir):
    """Recompute the COLVAR of one trajectory with ``plumed driver`` (not needed when the COLVARs exist)."""
    import shutil
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    box = os.path.join(out_dir, "work_" + os.path.basename(traj))
    os.makedirs(box, exist_ok=True)
    txt = re.sub(r"^(PRINT\b.*?)FILE=\S+", r"\1FILE=COLVAR", open(DATA["plumed_script"]).read(), flags=re.M)
    open(os.path.join(box, "plumed.dat"), "w").write(txt)
    subprocess.run([DATA["plumed_bin"], "driver", "--mf_dcd", traj, "--plumed", "plumed.dat"], cwd=box, check=True)
    shutil.move(os.path.join(box, "COLVAR"), os.path.join(out_dir, os.path.basename(colvar_path(traj))))
    shutil.rmtree(box, ignore_errors=True)


def labels_from_cvs(cvs):
    """GMM (2 components) on the mean s_t within Z_WINDOW; 1 = B route (higher s_t), 0 = B' route."""
    feat = np.array([c.loc[c["cyl.z"].between(*Z_WINDOW), "st"].mean() for c in cvs])
    ok = ~np.isnan(feat)
    labels = np.full(len(feat), -1)
    gmm = GaussianMixture(2, random_state=0).fit(feat[ok, None])
    labels[ok] = gmm.predict(feat[ok, None])
    if gmm.means_[0, 0] > gmm.means_[1, 0]:
        labels[ok] = 1 - labels[ok]
    return labels


# ── per-frame descriptors ─────────────────────────────────────────────────────────────────────────────────
def _ring_frame(coords):
    sol = np.asarray(coords, float)[:, :N_BEN * 3].reshape(len(coords), N_BEN, 3)
    ring, c_am = sol[:, BEN_RING], sol[:, BEN_C_AMIDINE]
    para = ring[np.arange(len(sol)), np.linalg.norm(ring - c_am[:, None], axis=2).argmax(1)]
    return sol, ring, c_am - para


def orientation(coords):
    """cos theta (amidine axis vs z) and |n_z| (ring normal) of every frame."""
    _, ring, axis = _ring_frame(coords)
    cos_t = axis[:, 2] / np.linalg.norm(axis, axis=1)
    normal = np.linalg.svd(ring - ring.mean(1, keepdims=True))[2][:, 2]
    return np.c_[cos_t, np.abs(normal[:, 2])]


def ligand_geometry(coords):
    sol, ring, axis = _ring_frame(coords)
    cen = sol[:, BEN_CN].mean(1) - V14_A
    ax = axis / np.linalg.norm(axis, axis=1)[:, None]
    n = np.linalg.svd(ring - ring.mean(1, keepdims=True))[2][:, 2]
    n *= np.where(n[:, 2:3] < 0, -1, 1)
    d = {"z": cen[:, 2], "x": cen[:, 0], "y": cen[:, 1], "r": np.hypot(cen[:, 0], cen[:, 1]),
         "costheta": ax[:, 2], "ax_x": ax[:, 0], "ax_y": ax[:, 1], "n_x": n[:, 0], "n_y": n[:, 1],
         "ringz": np.abs(n[:, 2])}
    return np.c_[tuple(d[k] for k in LIG_ONLY)]


def water_descriptors(ts, cvs):
    """radius, cos theta, |n_z|, V1-V16, FG, FH per frame (names: water_descriptor_names())."""
    return [np.c_[c[["radius"]].values, orientation(x), c[HYDRATION_COLS].values] for x, c in zip(ts.coords, cvs)]


def water_descriptor_names():
    return [LIGAND_TEX["radius"], LIGAND_TEX["costheta"], LIGAND_TEX["ringz"]] + HYDRATION_TEX


def ligand_descriptors(ts):
    return [ligand_geometry(x) for x in ts.coords]
