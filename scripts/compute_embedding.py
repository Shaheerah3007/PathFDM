"""Compute the FDM embedding of a set of trajectories from the command line and save it as CSV.

Example (ligand unbinding, ligand only):
    python scripts/compute_embedding.py "traj/*.xtc" --topology top.gro --solute "resname LIG" \
        --align protein --align-mode aligntraj --metric l2 -o embedding.csv

Example (ligand + the 32 closest waters, matched between trajectories):
    python scripts/compute_embedding.py "traj/*.dcd" --topology box.gro --reference ref.gro \
        --solute "resname MOL" --align "index 15-54" --waters 32 -o embedding.csv
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathfdm import embed, load_trajectories  # noqa: E402
from pathfdm.loaders import traj_number  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trajectories", help="glob pattern of trajectory files (quote it)")
    p.add_argument("--topology", required=True)
    p.add_argument("--reference", help="fit reference (default: the topology)")
    p.add_argument("--solute", required=True, help="MDAnalysis selection of the path atoms")
    p.add_argument("--align", required=True, help="MDAnalysis selection used for the fit")
    p.add_argument("--align-mode", default="fit", choices=["fit", "aligntraj"])
    p.add_argument("--mass-weighted", action="store_true")
    p.add_argument("--waters", type=int, default=0, help="number of closest water oxygens to include (0 = none)")
    p.add_argument("--metric", default="l2", choices=["l2", "l1", "dtw", "frechet", "hausdorff"])
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("-o", "--output", default="embedding.csv")
    a = p.parse_args()

    files = sorted(glob.glob(a.trajectories), key=traj_number)
    water = {"pick": "closest", "k": a.waters} if a.waters else None
    ts = load_trajectories(files, a.topology, a.reference, solute_sel=a.solute, align_sel=a.align,
                           align_mode=a.align_mode, align_weights="mass" if a.mass_weighted else None,
                           water=water, n_jobs=a.n_jobs)
    Z = embed(ts.fd, n_components=a.n_components, metric=a.metric, alpha=a.alpha, water=bool(water),
              n_jobs=a.n_jobs, normalize_length=a.metric == "dtw")
    df = pd.DataFrame(Z, columns=[f"FDM{i}" for i in range(Z.shape[1])])
    df.insert(0, "trajectory", ts.names)
    df.insert(1, "n_frames", ts.lengths)
    df.to_csv(a.output, index=False)
    print(f"wrote {a.output} ({len(df)} trajectories x {Z.shape[1]} coordinates)")


if __name__ == "__main__":
    main()
