# Data

The trajectories are not stored in this repository. Each system's `systems/<name>/load_data.py` has a `DATA`
dictionary at the top with the paths it reads; edit those paths to point at your copy.

| System | Trajectories | Files read |
|---|---|---|
| Alanine dipeptide | 100 PathGennie reactive paths (`*.dcd`) | equilibrated structure (`ala_dipeptide_equilibrated.pdb`), `trajectory_labels.txt` |
| Adenylate kinase | 12 paths from the MDAnalysis PSAnalysisTutorial (`methods/{DIMS,FRODA,rTMD-F,rTMD-S}/00{1,2,3}/fitted_psa.dcd`) | `structs/adk1AKE.pdb`, `structs/adk4AKE.pdb`, per-method topologies |
| T4 lysozyme L99A – benzene | 46 reactive unbinding trajectories (`*.dcd`) | `pbcmol.gro`, `trajectory_routes_literature.txt` |
| Imatinib – Abl (2HYY) | 200 PathGennie unbinding trajectories (`trajectory_0..199.xtc`) | `pbcmol.gro`, `trajectory_routes_literature.txt` |
| OAMe – G2 | 95 reactive unbinding trajectories (`trajectory_*.dcd`) | `box.gro` (topology), `conf_template.gro` (fit reference), `trajectory_labels.txt` |
| Trypsin – benzamidine | 50 PathGennie escape trajectories (`traj_000..049.dcd`) | `start_B.gro`, funnel template `fit_tem.pdb`, `plumed_descriptors.dat`, one COLVAR per trajectory |

Label files are plain text: one trajectory name and one integer label per line (tab- or space-separated, with
a header line where noted in `load_data.py`).

The trypsin descriptors use PLUMED CVs (funnel coordinate, Deep-TICA s<sub>t</sub>, hydration-site
occupancies). They are read from precomputed COLVAR files; `run_plumed()` in
`systems/trypsin_benzamidine/load_data.py` recomputes them with `plumed driver` (PLUMED 2.9 with the
pytorch module and the Deep-LDA/Deep-TICA models of the original study).
