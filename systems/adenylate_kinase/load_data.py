"""Adenylate kinase transition paths between the closed (1AKE) and open (4AKE) structures, from four protocols
(MDAnalysis PSAnalysisTutorial): DIMS, FRODA, and restrained targeted MD at a fast (rTMD-F) and a slow (rTMD-S)
pulling rate, three paths each.

FDM input : backbone coordinates (855 atoms), every frame fitted (unweighted) onto the 1AKE backbone; atoms are
            matched across the different topologies by residue number and atom name
Labels    : protocol
Collective variables (visualization only, not FDM input), computed on the fitted frames:
    q1, q2       fraction of native C-alpha contacts (< 8 A, |i - j| >= 4) of 1AKE and of 4AKE
    theta_NMP    NMP opening angle, theta_LID LID opening angle (degrees, between domain centres of mass)
    zeta, rho    progress (distance remaining to 4AKE, A) along, and RMS deviation (A) from, the straight
                 C-alpha path between the two end structures

Data: <repository>/data/<system>/ or $PATHFDM_DATA/<system>/ (see data/README.md).
"""
from __future__ import annotations

import os
from pathlib import Path

# data root: $PATHFDM_DATA if set, else <repository>/data (unpack the Zenodo archive there)
DATA_ROOT = Path(os.environ.get("PATHFDM_DATA", Path(__file__).resolve().parents[2] / "data")) / "adenylate_kinase"


import MDAnalysis as mda
import numpy as np
from joblib import Parallel, delayed
from MDAnalysis.analysis import align
from MDAnalysis.analysis.align import rotation_matrix

from pathfdm.loaders import _pack

PSA = DATA_ROOT / "PSAnalysisTutorial"     # https://github.com/Becksteinlab/PSAnalysisTutorial
DATA = {
    "methods_dir": str(PSA / "methods"),
    "closed": str(PSA / "structs" / "adk1AKE.pdb"),
    "open": str(PSA / "structs" / "adk4AKE.pdb"),
}
METHODS = ["rTMD-F", "rTMD-S", "DIMS", "FRODA"]
RUNS = 3
METHOD_COLORS = {"rTMD-F": "#4daf4a", "rTMD-S": "#984ea3", "DIMS": "#e4821a", "FRODA": "#377eb8"}
LABEL_CONFIG = {i: {"color": METHOD_COLORS[m], "name": m} for i, m in enumerate(METHODS)}
CV_NAMES = ["q_1", "q_2", r"\theta_\mathrm{NMP}", r"\theta_\mathrm{LID}", r"\zeta", r"\rho"]


def trajectory_list():
    """[(topology, trajectory, method)] for the 12 paths."""
    out = []
    for m in METHODS:
        d = os.path.join(DATA["methods_dir"], m)
        top = os.path.join(d, "top.psf") if os.path.exists(os.path.join(d, "top.psf")) else os.path.join(d, "top.pdb")
        for r in range(1, RUNS + 1):
            out.append((top, os.path.join(d, f"{r:03d}", "fitted_psa.dcd"), m))
    return out


def _references():
    closed, opened = mda.Universe(DATA["closed"]), mda.Universe(DATA["open"])
    align.alignto(opened, closed, select="backbone", match_atoms=False)
    return closed, opened


def _by_name(ref, sel):
    """(resid, name) pairs of a selection of the reference, in reference order."""
    g = ref.select_atoms(sel)
    return list(zip(g.resids.tolist(), g.names.tolist()))


def _match(u, pairs):
    """Atoms of u with the given (resid, name) pairs, in the same order (atoms matched across topologies)."""
    index = {(r, n): a for a, r, n in zip(u.atoms.indices, u.atoms.resids.tolist(), u.atoms.names.tolist())}
    return u.atoms[[index[p] for p in pairs]]


def _native_pairs(ca, cutoff=8.0):
    i, j = np.triu_indices(len(ca), k=4)
    m = np.linalg.norm(ca[i] - ca[j], axis=1) < cutoff
    return i[m], j[m]


def _angle(v1, v2):
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _load_one(top, traj):
    closed, opened = _references()
    ref_bb = closed.select_atoms("backbone").positions.copy()
    ic, fc = closed.select_atoms("name CA").positions.copy(), opened.select_atoms("name CA").positions.copy()
    ip, fp = _native_pairs(ic), _native_pairs(fc)
    v_ab = fc - ic
    total = np.sqrt(np.mean(np.sum(v_ab ** 2, axis=1)))
    sel = {k: _by_name(closed, s) for k, s in [("bb", "backbone"), ("ca", "name CA"),
                                                ("nmp", "resid 35-55 and name CA"),
                                                ("core_nmp", "resid 90-100 and name CA"),
                                                ("core_lid1", "resid 115-125 and name CA"),
                                                ("lid", "resid 125-153 and name CA"),
                                                ("core_lid2", "resid 179-185 and name CA")]}
    u = mda.Universe(top, traj)
    g = {k: _match(u, s) for k, s in sel.items()}
    if g["bb"].n_atoms != len(ref_bb):
        raise ValueError(f"{traj}: {g['bb'].n_atoms} backbone atoms, reference has {len(ref_bb)}")
    ref_com = ref_bb.mean(axis=0)
    coords, cvs = [], []
    for _ in u.trajectory:
        mob = g["bb"].positions
        com = mob.mean(axis=0)
        R, _ = rotation_matrix(mob - com, ref_bb - ref_com)
        u.atoms.positions = (R @ (u.atoms.positions - com).T).T + ref_com
        ca = g["ca"].positions
        q1 = np.mean(np.linalg.norm(ca[ip[0]] - ca[ip[1]], axis=1) < 8.0)
        q2 = np.mean(np.linalg.norm(ca[fp[0]] - ca[fp[1]], axis=1) < 8.0)
        v_nmp, v_lid = g["core_nmp"].center_of_mass(), g["core_lid1"].center_of_mass()
        t_nmp = _angle(g["nmp"].center_of_mass() - v_nmp, g["core_lid1"].center_of_mass() - v_nmp)
        t_lid = _angle(g["core_lid2"].center_of_mass() - v_lid, g["lid"].center_of_mass() - v_lid)
        s = float(np.clip(np.sum((ca - ic) * v_ab) / np.sum(v_ab * v_ab), 0.0, 1.0))
        rho = float(np.sqrt(np.mean(np.sum((ca - (ic + s * v_ab)) ** 2, axis=1))))
        coords.append(g["bb"].positions.ravel().copy())
        cvs.append((q1, q2, t_nmp, t_lid, (1.0 - s) * total, rho))
    return np.array(coords, dtype=np.float32), np.array(cvs)


def load(n_jobs: int = 12):
    """TrajectorySet of the 12 backbone paths, labels (index into METHODS), and the CV paths."""
    tl = trajectory_list()
    res = Parallel(n_jobs=n_jobs)(delayed(_load_one)(t, p) for t, p, _ in tl)
    coords = [r[0] for r in res]
    ts = _pack(coords, [np.linspace(0, 1, len(c)) for c in coords], [p for _, p, _ in tl],
               coords[0].shape[1] // 3, 0, [])
    ts.files = [f"{m}_{i % RUNS + 1}" for i, (_, _, m) in enumerate(tl)]      # readable names
    labels = np.array([METHODS.index(m) for _, _, m in tl])
    print(f"{len(ts)} paths | {coords[0].shape[1] // 3} backbone atoms | {ts.lengths.min()}-{ts.lengths.max()} frames")
    return ts, labels, [r[1] for r in res]
