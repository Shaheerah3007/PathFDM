"""Benzene unbinding from T4 lysozyme L99A: 46 reactive trajectories, egress routes from the literature.

FDM input : benzene coordinates (12 atoms) after a mass-weighted fit of the protein onto the topology frame
Metric    : DTW (paths of very different length; 104-813 frames)
Labels    : trajectory_routes_literature.txt, routes named after the helices that line them
            (Nunes-Alves et al. 2018, 2021): DG, FGH, HJ, EJ, CD, FI

Data: <repository>/data/<system>/ or $PATHFDM_DATA/<system>/ (see data/README.md).
"""
from __future__ import annotations

import os
from pathlib import Path

# data root: $PATHFDM_DATA if set, else <repository>/data (unpack the Zenodo archive there)
DATA_ROOT = Path(os.environ.get("PATHFDM_DATA", Path(__file__).resolve().parents[2] / "data")) / "t4l_benzene"

import numpy as np
import pandas as pd

from pathfdm.data import require
from pathfdm import find_trajectories, load_trajectories

DATA = {
    "traj_dir": str(DATA_ROOT / "trajectories"),          # solvent stripped: protein + benzene
    "topology": str(DATA_ROOT / "protein_benzene.gro"),
    "labels": str(DATA_ROOT / "trajectory_routes_literature.txt"),
}

OPTIONS = {
    "solute_sel": "resname BNZ",
    "align_sel": "protein",
    "align_mode": "aligntraj",
    "align_weights": "mass",
    "time": "frame",
}

LABEL_CONFIG = {0: {"color": "#008241", "name": "DG"}, 1: {"color": "#FDD700", "name": "FGH"},
                2: {"color": "#75C9F9", "name": "HJ"}, 3: {"color": "#CC79A7", "name": "EJ"},
                4: {"color": "#FF5900", "name": "CD"}, 5: {"color": "#2E31F1", "name": "FI"}}


def load(n_jobs: int = 8):
    require(DATA_ROOT)                     # downloads the system from Zenodo if it is missing
    ts = load_trajectories(find_trajectories(DATA["traj_dir"], "*.dcd"), DATA["topology"], DATA["topology"],
                           n_jobs=n_jobs, **OPTIONS)
    lab = pd.read_csv(DATA["labels"], sep="\t").set_index("trajectory_name")
    return ts, lab.loc[ts.names, "cluster"].to_numpy().astype(int)
