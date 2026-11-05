# PathFDM

**Functional Diffusion Maps of molecular transition-path ensembles.**

PathFDM treats every transition trajectory (unbinding, conformational change, ...) as one multivariate function
of progress and embeds the whole ensemble in a continuous low-dimensional *trajectory space*, built only from
pairwise distances between trajectories — no collective variables or pathway labels are needed. Trajectories
that follow the same route lie close together; the diffusion coordinates order the routes and the variation
within them. A surrogate model with SHAP attribution then identifies which physical descriptors explain each
coordinate.

Features

* distances between whole trajectories: L² / L¹ after resampling along progress, DTW, discrete Fréchet,
  Hausdorff, PRISM average Hausdorff, SHINE MSD
* **water-permutation-invariant distances**: explicit water molecules are matched between every pair of
  trajectories (Hungarian assignment on per-water path costs), so the embedding does not depend on how waters
  are labelled
* loaders for MD trajectories (MDAnalysis): rigid fit, unwrapping, and water selection (closest waters,
  hydration sites, waters near an atom)
* trajectory-level descriptors (σ, mean, deviation from linear progression Δ, skewness γ₁ of each per-frame
  descriptor) and an ExtraTrees surrogate with out-of-bag and repeated cross-validated R² and SHAP attribution

## Installation

```bash
git clone https://github.com/<user>/PathFDM.git
cd PathFDM
pip install -e ".[fast,notebooks]"      # fast = numba + dtaidistance (recommended for DTW)
```

## Quick start

```python
from pathfdm import find_trajectories, load_trajectories, embed

files = find_trajectories("unbinding_runs/", "*.xtc")
ts = load_trajectories(files, "system.gro", solute_sel="resname LIG", align_sel="protein",
                       align_mode="aligntraj", align_weights="mass")
Z = embed(ts.fd, metric="l2")             # FDM 0, FDM 1, ... (one row per trajectory)
```

With explicit waters (the K = 32 water oxygens that come closest to the solute, matched between trajectories):

```python
ts = load_trajectories(files, "box.gro", "ref.gro", solute_sel="resname MOL", align_sel="index 15-54",
                       align_mode="fit", water={"pick": "closest", "k": 32})
Z = embed(ts.fd, metric="l2", water=True)
```

Interpreting a coordinate:

```python
from pathfdm.features import feature_matrix
from pathfdm.surrogate import fit_surrogate
X, names = feature_matrix(per_frame_descriptors, descriptor_names)   # list of (n_frames, n_desc) arrays
res = fit_surrogate(X, Z[:, 0], names)
print(res.summary())          # out-of-bag R², 5-fold CV x 5 R²
res.table(10)                 # SHAP share of every descriptor
```

`scripts/compute_embedding.py` does load + embed from the command line.

## Method in brief

For trajectories *i*, *j* with distance *d(i, j)*, the Gaussian kernel *K = exp(−d²/2σ²)* (σ = median distance)
is normalized to a Markov matrix *P* (α = 0, standard normalization). Its nontrivial right eigenvectors ψ₁, ψ₂, …
are the diffusion coordinates. The diffusion time *t* only rescales ψₖ by λₖᵗ; `embed` standardizes every
eigenvector across trajectories (zero mean, unit variance), so the result does not depend on *t*, and reports
them as FDM 0, FDM 1, … (signs are arbitrary).

## Repository layout

```
pathfdm/            the package
  fdm.py            trajectory distances (incl. water matching) and the FDM estimator
  loaders.py        MD trajectory loading, alignment, water selection -> TrajectorySet
  features.py       path statistics (σ, mean, Δ, γ₁) after predictive denoising
  denoise.py        predictive nearest-neighbour noise reduction
  surrogate.py      ExtraTrees surrogate, OOB / cross-validated R², SHAP
  evaluation.py     AUC, k-means ARI, embedding comparison
  plotting.py       trajectory-space, parity and beeswarm plots
systems/<name>/     one folder per system studied in the paper
  load_data.py      data paths (edit DATA), loader options, labels, per-frame descriptors
  <name>.ipynb      load -> FDM -> trajectory space -> surrogate accuracy -> SHAP, plus settings comparisons
scripts/            make_notebooks.py (regenerates the notebooks), compute_embedding.py (CLI)
tests/              synthetic checks (route separation, water-permutation invariance)
data/README.md      where the trajectories come from
```

## Systems

| Folder | System | Trajectories | Distance | Surrogate / SHAP |
|---|---|---|---|---|
| `alanine_dipeptide` | Alanine dipeptide C7eq → C7ax | 100 | L² | Φ, Ψ, ω, C7 H-bond |
| `adenylate_kinase` | Adenylate kinase, 4 protocols | 12 | L² | – (CV projections, PCA) |
| `t4l_benzene` | Benzene–T4 lysozyme L99A unbinding | 46 | DTW (α = 0.5) | – |
| `imatinib_abl` | Imatinib–Abl unbinding | 200 | L² | – |
| `oame_g2` | OAMe–G2 host–guest unbinding | 95 | L² + water matching | hydration (map with water), guest geometry (solute-only map) |
| `trypsin_benzamidine` | Benzamidine–trypsin unbinding | 50 | L² + water matching | hydration sites + ligand (map with water), ligand geometry (ligand-only map) |

The trajectories are not included; see `data/README.md`.

## Citation

S. Shahid and S. Chakrabarty, *Disentangling Heterogeneous Trajectory Ensembles through a Data-Driven
Continuous Embedding Interpreted by Feature Attribution* (in preparation).
