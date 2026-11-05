"""Imatinib unbinding from Abl kinase (PDB 2HYY): 200 PathGennie trajectories.

FDM input : imatinib coordinates (68 atoms) after a mass-weighted fit of the protein onto the topology frame
Labels    : trajectory_routes_literature.txt: the three routes of the PathGennie study (Path I, P-loop/hinge;
            Path II, alphaC/catalytic loop; Path III, C-lobe) with Path III split into IIIa and IIIb, plus two
            outlier trajectories

Data: <repository>/data/<system>/ or $PATHFDM_DATA/<system>/ (see data/README.md).
"""
from __future__ import annotations

import os
from pathlib import Path

# data root: $PATHFDM_DATA if set, else <repository>/data (unpack the Zenodo archive there)
DATA_ROOT = Path(os.environ.get("PATHFDM_DATA", Path(__file__).resolve().parents[2] / "data")) / "imatinib_abl"

import pandas as pd

from pathfdm import find_trajectories, load_trajectories

DATA = {
    "traj_dir": str(DATA_ROOT / "trajectories"),          # solvent stripped: protein + imatinib
    "topology": str(DATA_ROOT / "protein_imatinib.gro"),
    "labels": str(DATA_ROOT / "trajectory_routes_literature.txt"),
}

OPTIONS = {
    "solute_sel": "resname LIG",
    "align_sel": "protein",
    "align_mode": "aligntraj",
    "align_weights": "mass",
    "time": "frame",
}

LABEL_CONFIG = {2: {"color": "#009E73", "name": "Path I"}, 1: {"color": "#56B4E9", "name": "Path II"},
                3: {"color": "#CC79A7", "name": "Path IIIa"}, 0: {"color": "#E69F00", "name": "Path IIIb"},
                -1: {"color": "#B7F659", "name": "Outlier 1"}, -2: {"color": "#4245F0", "name": "Outlier 2"}}


N_TRAJ = 200                         # the analysed set: trajectory_0 ... trajectory_199


def load(n_jobs: int = 8):
    files = find_trajectories(DATA["traj_dir"], "*.xtc")[:N_TRAJ]
    ts = load_trajectories(files, DATA["topology"], DATA["topology"],
                           n_jobs=n_jobs, **OPTIONS)
    lab = pd.read_csv(DATA["labels"], sep="\t").set_index("trajectory_name")
    return ts, lab.loc[ts.names, "cluster"].to_numpy().astype(int)
