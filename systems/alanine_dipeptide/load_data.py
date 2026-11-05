"""Alanine dipeptide: C7eq -> C7ax transition paths (PathGennie, 100 reactive paths, alpha_R vs alpha_L routes).

FDM input : aligned Cartesian coordinates of all peptide atoms (backbone fit to the equilibrated structure)
Labels    : trajectory_labels.txt (0 = alpha_R route, 1 = alpha_L route)
Descriptors for the surrogate: Phi, Psi, omega_1, omega_2 (degrees) and the C7 hydrogen-bond distance
(ACE O - NME H, Angstrom), from the unaligned frames.

Data: <repository>/data/<system>/ or $PATHFDM_DATA/<system>/ (see data/README.md).
"""
from __future__ import annotations

import os
from pathlib import Path

# data root: $PATHFDM_DATA if set, else <repository>/data (unpack the Zenodo archive there)
DATA_ROOT = Path(os.environ.get("PATHFDM_DATA", Path(__file__).resolve().parents[2] / "data")) / "alanine_dipeptide"

import glob

import MDAnalysis as mda
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathfdm.data import require
from pathfdm import load_trajectories

DATA = {
    "traj_dir": str(DATA_ROOT / "trajectories"),
    "topology": str(DATA_ROOT / "ala_dipeptide_equilibrated.pdb"),
    "labels": str(DATA_ROOT / "trajectory_labels.txt"),
}

OPTIONS = {
    "solute_sel": "protein",          # atoms that make up the path
    "align_sel": "backbone",          # fit selection
    "align_mode": "aligntraj",        # rigid fit only (no water, no unwrapping needed)
    "align_weights": None,            # unweighted fit (as align.alignto)
    "time": "progress",
    "max_frames": 1000,               # skip longer paths (none are)
}

LABEL_CONFIG = {0: {"color": "#EF553B", "name": r"$\alpha_R$ path"},
                1: {"color": "#636EFA", "name": r"$\alpha_L$ path"}}

DESC_NAMES = [r"\phi", r"\psi", r"\omega_1", r"\omega_2", r"c7\text{-}h"]

PHI_ATOMS = (4, 6, 8, 14)    # ACE C, ALA N, ALA CA, ALA C   (0-based)
PSI_ATOMS = (6, 8, 14, 16)   # ALA N, ALA CA, ALA C, NME N


def trajectory_files():
    return sorted(glob.glob(os.path.join(DATA["traj_dir"], "*.dcd")))


def load(n_jobs: int = 8, **overrides):
    """TrajectorySet of the 100 paths and their route labels (same order)."""
    require(DATA_ROOT)                     # downloads the system from Zenodo if it is missing
    ts = load_trajectories(trajectory_files(), DATA["topology"], DATA["topology"], n_jobs=n_jobs,
                           **{**OPTIONS, **overrides})
    return ts, labels_for(ts)


def labels_for(ts):
    lab = pd.read_csv(DATA["labels"], sep=r"\s+")
    lab_of = dict(zip(lab.iloc[:, 0], lab.iloc[:, 1].astype(int)))
    return np.array([lab_of[os.path.basename(f)] for f in ts.files])


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = -(p1 - p0), p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _descriptors_one(path):
    ref = mda.Universe(DATA["topology"], guess_bonds=True)
    idx = lambda s: ref.select_atoms(s)[0].index
    o_ace, h_nme = idx("resid 1 and name O"), idx("resid 3 and name H")
    om1 = [idx("resid 1 and name CH3"), idx("resid 1 and name C"), idx("resid 2 and name N"), idx("resid 2 and name CA")]
    om2 = [idx("resid 2 and name CA"), idx("resid 2 and name C"), idx("resid 3 and name N"), idx("resid 3 and name C")]
    u = mda.Universe(DATA["topology"], path, guess_bonds=True)
    ag1, ag2 = u.atoms[om1].dihedral, u.atoms[om2].dihedral
    rows = []
    for _ in u.trajectory:
        x = u.atoms.positions.astype(float)
        rows.append([_dihedral(*x[list(PHI_ATOMS)]), _dihedral(*x[list(PSI_ATOMS)]), ag1.value(), ag2.value(),
                     np.linalg.norm(x[o_ace] - x[h_nme])])
    return np.array(rows)


def descriptors(ts, n_jobs: int = 8):
    """Per-frame descriptors of every trajectory: list of (n_frames, 5) arrays, columns as DESC_NAMES."""
    return Parallel(n_jobs=n_jobs)(delayed(_descriptors_one)(f) for f in ts.files)
