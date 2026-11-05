# Data

The trajectories are archived on Zenodo (DOI: [10.5281/zenodo.23975078](https://doi.org/10.5281/zenodo.23975078)). Download the archives and unpack them in this
folder, so that every system has its own subfolder:

```
data/
  alanine_dipeptide/      trajectories/*.dcd, ala_dipeptide_equilibrated.pdb, trajectory_labels.txt
  oame_g2/                trajectories/trajectory_*.dcd, box.gro, conf_template.gro, trajectory_labels.txt, plumed.dat
  trypsin_benzamidine/    trajectories/traj_*.dcd, colvars/traj_*.colvar, start_B.gro, fit_tem.pdb,
                          plumed_descriptors.dat
  t4l_benzene/            trajectories/*.dcd, protein_benzene.gro, trajectory_routes_literature.txt
  imatinib_abl/           trajectories/trajectory_*.xtc, protein_imatinib.gro, trajectory_routes_literature.txt
  adenylate_kinase/       PSAnalysisTutorial/   (not in the archive, see below)
```

From the repository root (checks the SHA256 sums and unpacks into `data/`):

```bash
python scripts/download_data.py                        # all five systems, ~16 GB
python scripts/download_data.py alanine_dipeptide      # or only some
```

The notebooks do the same for a missing system on first use. By hand:

```bash
cd data
for f in /path/to/download/*.tar; do tar -xf "$f"; done
```

The loaders (`systems/<system>/load_data.py`) read from `data/<system>/`. To keep the data elsewhere, set
`PATHFDM_DATA` to the folder that contains the system subfolders:

```bash
export PATHFDM_DATA=/scratch/pathfdm_data
```

## Contents

| System | Trajectories | Notes |
|---|---|---|
| Alanine dipeptide | 100 PathGennie reactive paths, C7eq → C7ax | as written by PathGennie; labels: 0 = α<sub>R</sub>, 1 = α<sub>L</sub> route |
| OAMe–G2 | 95 reactive unbinding trajectories | full system (waters needed for the water selection); labels: 1 = dry, 0 = wet |
| Trypsin–benzamidine | 50 PathGennie escape trajectories from state B | full system (waters needed); `colvars/` holds the PLUMED CVs (funnel z, radius, Deep-TICA s<sub>t</sub>, hydration-site occupancies V1–V16, FG, FH) computed with `plumed_descriptors.dat` |
| T4 lysozyme L99A–benzene | 46 reactive unbinding trajectories | solvent and ions removed (protein + benzene); labels: literature egress routes |
| Imatinib–Abl (2HYY) | 200 PathGennie unbinding trajectories | solvent and ions removed (protein + imatinib); labels: Path I, II, IIIa, IIIb, two outliers |
| Adenylate kinase | 12 paths (DIMS, FRODA, rTMD-F, rTMD-S) | public: clone https://github.com/Becksteinlab/PSAnalysisTutorial into `data/adenylate_kinase/PSAnalysisTutorial` |

The trypsin COLVARs can be recomputed with `run_plumed()` in `systems/trypsin_benzamidine/load_data.py`; this
needs PLUMED with the pytorch module and the Deep-LDA/Deep-TICA models of the original funnel-metadynamics
study, which are not redistributed here.
