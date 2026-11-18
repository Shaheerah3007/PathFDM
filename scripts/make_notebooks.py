"""Write the analysis notebook of every system (systems/<name>/<name>.ipynb).

    python scripts/make_notebooks.py            # write all notebooks
    python scripts/make_notebooks.py oame_g2    # write one

Each notebook loads the data with the system's load_data.py, computes the FDM embedding(s), plots the
trajectory space, and (where descriptors are defined) fits the surrogate model, reports its out-of-bag and
cross-validated accuracy, and shows the SHAP beeswarm.
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

HEADER = '''import sys, warnings
from pathlib import Path
ROOT = Path.cwd().resolve().parents[1]          # repository root (notebook lives in systems/<name>/)
sys.path[:0] = [str(ROOT), str(Path.cwd())]
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathfdm import embed
from pathfdm.features import feature_matrix
from pathfdm.surrogate import fit_surrogate
from pathfdm.evaluation import auc_per_coordinate, kmeans_ari, align_signs, compare_embeddings
from pathfdm.plotting import apply_style, plot_space, plot_parity, plot_beeswarm, plot_paths, plot_binned, top_descriptor
import load_data as D
apply_style()
N_JOBS = 8'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


SURROGATE_HELP = """The surrogate is an ExtraTrees regressor (500 trees) predicting FDM 0 from the standardized
descriptors. **Out-of-bag R²**: each trajectory predicted by the trees that did not see it. **Cross-validation**:
5-fold, repeated 5 times; R² is the mean ± sd over the 25 folds, and the parity plot shows the mean of the five
held-out predictions of every trajectory. The beeswarm shows the SHAP value of each descriptor for every
trajectory (colour = descriptor value)."""

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
NOTEBOOKS = {}

NOTEBOOKS["alanine_dipeptide"] = [
    md("""
# Alanine dipeptide: C5 → C7ax transition paths

100 PathGennie reactive paths, 38 through the α<sub>R</sub> region and 62 through the α<sub>L</sub> region.
FDM is computed from the aligned Cartesian coordinates of all peptide atoms (L² distance, σ = median, α = 0);
no dihedral angles are given to it. The surrogate then asks which backbone torsions explain FDM 0."""),
    code(HEADER),
    md("## Load the paths"),
    code("ts, labels = D.load(n_jobs=N_JOBS)\nprint(dict(zip(*np.unique(labels, return_counts=True))))"),
    md("## FDM embedding"),
    code("""Z, model = embed(ts.fd, metric="l2", sigma="median", alpha=0.0, return_model=True, n_jobs=N_JOBS)
print(f"sigma = {model.sigma_:.3f} | eigenvalues {np.round(model.eigenvals[:4], 4)}")
print("AUC (alpha_R vs alpha_L) of FDM 0-3:", np.round(auc_per_coordinate(Z, labels), 3))
print(f"k-means (k = 2) on FDM 0-1: ARI = {kmeans_ari(Z, labels):.2f}")
plot_space(Z, labels, D.LABEL_CONFIG, title="FDM of 100 alanine dipeptide paths"); plt.show()"""),
    md("## Descriptors and surrogate model\n\n" + SURROGATE_HELP + """

Descriptors: Φ, Ψ, ω<sub>1</sub>, ω<sub>2</sub> and the C7 hydrogen-bond distance, each summarized by
σ, ⟨⟩, Δ and γ<sub>1</sub> (20 descriptors). σ, ⟨⟩ and γ<sub>1</sub> are computed on the raw path and Δ on the
denoised path (`stats_on="mixed"`)."""),
    code("""desc = D.descriptors(ts, n_jobs=N_JOBS)
X, names = feature_matrix(desc, D.DESC_NAMES, stats_on="mixed", zero_degenerate_skew=False)
res = fit_surrogate(X, Z[:, 0], names)
print(res.summary())
res.table(8).round(3)"""),
    code("plot_parity(res, labels, D.LABEL_CONFIG); plt.tight_layout(); plt.show()\nplot_beeswarm(res, max_display=10); plt.show()"),
    md("""## Descriptors along the paths

Left: the paths in the (Φ, Ψ) plane, coloured by route (paper, Fig. b). Right: one descriptor against path
progress; by default the descriptor behind the highest-ranked SHAP feature. Set `X_DESC`, `Y_DESC` and
`DESC` to any of `D.DESC_NAMES` to plot other pairs."""),
    code("""X_DESC, Y_DESC = r"\\phi", r"\\psi"                     # paper: Ramachandran plane
DESC = D.DESC_NAMES[top_descriptor(res, D.DESC_NAMES)]       # or e.g. r"\\omega_1"
col = {n: i for i, n in enumerate(D.DESC_NAMES)}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
plot_paths([d[:, col[X_DESC]] for d in desc], [d[:, col[Y_DESC]] for d in desc], labels, D.LABEL_CONFIG, ax=axs[0],
           xlabel=f"${X_DESC}$ (deg)", ylabel=f"${Y_DESC}$ (deg)", lw=0.5, alpha=0.5)
axs[0].set_xlim(-180, 180); axs[0].set_ylim(-180, 180)
plot_paths([np.linspace(0, 1, len(d)) for d in desc], [d[:, col[DESC]] for d in desc], labels, D.LABEL_CONFIG,
           ax=axs[1], xlabel="path progress", ylabel=f"${DESC}$ (deg)" if DESC != r"c7\\text{-}h" else "C7 H-bond (Å)", lw=0.5, alpha=0.5, legend=False)
plt.tight_layout(); plt.show()"""),
    md("## Setting: statistics on the denoised path\n\nAll four statistics on the denoised path instead of the mixed scheme above."),
    code("""X2, _ = feature_matrix(desc, D.DESC_NAMES, stats_on="denoised")
res2 = fit_surrogate(X2, Z[:, 0], names)
print("all statistics on the denoised path:", res2.summary())
res2.table(5).round(3)"""),
]

NOTEBOOKS["oame_g2"] = [
    md("""
# OAMe–G2: guest unbinding, dry and wet routes

95 reactive unbinding trajectories of 4-cyanobenzoate (G2) from octa-acid methyl (OAMe); on the wet route the
cavity entrance hydrates as the guest leaves, on the dry route it does not (49 dry, 46 wet).

* **Solute-only map**: guest + host coordinates.
* **Map with water**: guest + host + the K = 32 water oxygens that come closest to the complex at any frame,
  matched between every pair of trajectories by the Hungarian assignment.

Both use the L² distance, σ = median, α = 0. The solute-only map is explained with guest-geometry descriptors,
the map with water with the water coordinations d1–d12 and the guest orientation cos θ."""),
    code(HEADER),
    md("## Load (solute + 32 waters)"),
    code("ts, labels = D.load(k_waters=32, n_jobs=N_JOBS)\nsol = ts.solute_only()\nprint(f\"{(labels == 1).sum()} dry, {(labels == 0).sum()} wet\")"),
    md("## FDM embeddings: solute only, with water (matched), with water (not matched)"),
    code("""Z_sol = embed(sol.fd, metric="l2", n_jobs=N_JOBS)
Z_w = embed(ts.fd, metric="l2", water=True, water_match="dtw", n_jobs=N_JOBS)
Z_nm = align_signs(embed(ts.fd, metric="l2", water=False, n_jobs=N_JOBS), Z_w)
maps = {"solute only": Z_sol, "with water, matched": Z_w, "with water, not matched": Z_nm}
fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
for ax, (name, Z) in zip(axs, maps.items()):
    plot_space(Z, labels, D.LABEL_CONFIG, ax=ax, title=f"{name}\\nAUC(FDM 0) = {auc_per_coordinate(Z, labels)[0]:.3f}",
               legend=ax is axs[0])
plt.tight_layout(); plt.show()
print("Spearman rho of FDM 0, solute only vs with water:", np.round(compare_embeddings(Z_sol, Z_w, 1), 3))
print("with vs without matching, FDM 0-2:", np.round(compare_embeddings(Z_nm, Z_w, 3), 3))
pd.DataFrame({n: {"AUC FDM 0": auc_per_coordinate(Z, labels)[0], "k-means ARI (FDM 0-1)": kmeans_ari(Z, labels)}
              for n, Z in maps.items()}).T.round(3)"""),
    md("## Surrogate models\n\n" + SURROGATE_HELP),
    code("""X_w, n_w = feature_matrix(D.water_map_descriptors(ts), D.WATER_MAP_NAMES)
res_w = fit_surrogate(X_w, Z_w[:, 0], n_w)
X_g, n_g = feature_matrix(D.guest_descriptors(ts), D.GUEST_NAMES)
res_s = fit_surrogate(X_g, Z_sol[:, 0], n_g)
print("map with water   (52: d1-d12, cos theta):  ", res_w.summary())
print("solute-only map  (32 guest descriptors):    ", res_s.summary())
pd.concat({"with water": res_w.table(6), "solute only": res_s.table(6)}, axis=1).round(3)"""),
    code("""fig, axs = plt.subplots(2, 2, figsize=(6, 4.8))
plot_parity(res_s, labels, D.LABEL_CONFIG, axs=axs[0], title="solute only")
plot_parity(res_w, labels, D.LABEL_CONFIG, axs=axs[1], title="with water")
plt.tight_layout(); plt.show()
plot_beeswarm(res_w, max_display=10, title="map with water: FDM 0"); plt.show()
plot_beeswarm(res_s, max_display=10, title="solute-only map: FDM 0"); plt.show()"""),
    md("""## Descriptors along the paths

As in the paper figure: (c) water coordination d<sub>6</sub> at V2 against the guest height cyl.z, one line
per trajectory; (f) guest orientation cos θ and (i) the number of waters in the host cavity as functions of
cyl.z, each trajectory averaged in 1 Å bins of cyl.z (median over trajectories and interquartile band). The
dashed line is cyl.z = 6 Å, the upper limit of the bound state. `DESC` (bottom right) plots any hydration or
guest descriptor the same way; by default the one behind the highest-ranked SHAP feature of the map with water.

The paper plots cos θ with the opposite sign (PLUMED convention of the earlier figure): set `COS_SIGN = -1`."""),
    code("""COS_SIGN = 1.0
EDGES = np.arange(-1, 19, 1.0)                     # 1 A bins of cyl.z
hyd, gst = D.hydration_descriptors(ts), D.guest_descriptors(ts)
cz = [D.cyl_z(c) for c in ts.coords]
cav = [D.cavity_waters(c, ts.water_info["n_solute_atoms"]) for c in ts.coords]
DESC = D.WATER_MAP_NAMES[top_descriptor(res_w, D.WATER_MAP_NAMES)]        # or e.g. "d_{6}", r"\\cos\\theta"
pool = {**{n: [h[:, i] for h in hyd] for i, n in enumerate(D.HYDRATION_NAMES)},
        **{n: [g[:, i] for g in gst] for i, n in enumerate(D.GUEST_NAMES)}}
fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.6))
plot_paths([h[:, 5] for h in hyd], cz, labels, D.LABEL_CONFIG, ax=axs[0, 0], xlabel="$d_{6}$ (V2)",
           ylabel="cyl.z (Å)", lw=0.5, alpha=0.5, title="(c)")
plot_binned(cz, [COS_SIGN * g[:, 3] for g in gst], labels, D.LABEL_CONFIG, EDGES, ax=axs[0, 1], xlabel="cyl.z (Å)",
            ylabel=r"$\\cos\\theta$", legend=False, title="(f)")
_, cav_table = plot_binned(cz, cav, labels, D.LABEL_CONFIG, EDGES, ax=axs[1, 0], xlabel="cyl.z (Å)",
                           ylabel="cavity waters", legend=False, title="(i)")
plot_binned(cz, pool[DESC], labels, D.LABEL_CONFIG, EDGES, ax=axs[1, 1], xlabel="cyl.z (Å)", ylabel=f"${DESC}$",
            legend=False, title="top SHAP descriptor" if DESC == D.WATER_MAP_NAMES[top_descriptor(res_w, D.WATER_MAP_NAMES)] else None)
for ax in (axs[0, 1], axs[1, 0], axs[1, 1]):
    ax.axvline(D.BOUND_Z, color="#444444", lw=0.5, ls="--", alpha=0.55)
axs[0, 0].axhline(D.BOUND_Z, color="#444444", lw=0.5, ls="--", alpha=0.55)
plt.tight_layout(); plt.show()"""),
    md("""## Setting: number of waters K, with and without matching

The waters are ranked by closest approach to the complex, so the K closest are the first K of the 64 loaded.
Without matching, waters are compared in this order."""),
    code("""ts64, _ = D.load(k_waters=64, n_jobs=N_JOBS)
rows, fig, axs = [], *plt.subplots(2, 4, figsize=(12, 5))
for col, K in enumerate([16, 32, 48, 64]):
    tk = ts64.first_k_waters(K)
    Zm = embed(tk.fd, metric="l2", water=True, water_match="dtw", n_jobs=N_JOBS)
    Zn = align_signs(embed(tk.fd, metric="l2", water=False, n_jobs=N_JOBS), Zm)
    for row, (kind, Z) in enumerate([("matched", Zm), ("not matched", Zn)]):
        plot_space(Z, labels, D.LABEL_CONFIG, ax=axs[row, col], title=f"K = {K}, {kind}", legend=False, s=10)
        rows.append({"K": K, "waters": kind, "AUC FDM 0": auc_per_coordinate(Z, labels)[0],
                     "k-means ARI": kmeans_ari(Z, labels), "rho FDM 0 vs matched": compare_embeddings(Z, Zm, 1)[0]})
plt.tight_layout(); plt.show()
pd.DataFrame(rows).round(3)"""),
    md("""## Setting: maps without the host

The guest alone, and the guest with the 32 waters that come closest to the guest (matched)."""),
    code("""tg, _ = D.load(k_waters=32, solute_sel=D.GUEST_SEL, n_jobs=N_JOBS)
Z_g = embed(tg.solute_only().fd, metric="l2", n_jobs=N_JOBS)
Z_gw = embed(tg.fd, metric="l2", water=True, water_match="dtw", n_jobs=N_JOBS)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
for ax, (name, Z) in zip(axs, {"guest only": Z_g, "guest + 32 waters": Z_gw}.items()):
    plot_space(Z, labels, D.LABEL_CONFIG, ax=ax, title=f"{name}: AUC(FDM 0) = {auc_per_coordinate(Z, labels)[0]:.3f}",
               legend=ax is axs[0])
plt.tight_layout(); plt.show()"""),
]

NOTEBOOKS["trypsin_benzamidine"] = [
    md("""
# Benzamidine unbinding from trypsin: water-mediated routes B and B′

50 PathGennie escape trajectories from the bound state B. Labels: Gaussian mixture on the mean s<sub>t</sub>
(Deep-TICA CV) in the funnel window z = 0.45–1.0 nm (31 B′, 19 B).

* **Ligand-only map**: benzamidine coordinates.
* **Map with water** (paper): benzamidine + the waters occupying the hydration sites V1–V16 in the first frame
  (six waters), matched between trajectories by the Hungarian assignment.

L² distance, σ = median, α = 0. The PLUMED CVs (s<sub>t</sub>, site occupancies) are read from precomputed
COLVAR files."""),
    code(HEADER),
    md("## Load (benzamidine + hydration-site waters)"),
    code("""ts, cvs, labels = D.load("sites", n_jobs=N_JOBS)
lig = ts.solute_only()
print(f"{(labels == 0).sum()} B', {(labels == 1).sum()} B | waters (resids): {ts.water_ids[0]}")"""),
    md("## FDM embeddings"),
    code("""Z_lig = embed(lig.fd, metric="l2", normalize_length=True, n_jobs=N_JOBS)
Z_w = embed(ts.fd, metric="l2", water=True, water_match="dtw", normalize_length=True, n_jobs=N_JOBS)
Z_nm = align_signs(embed(ts.fd, metric="l2", water=False, normalize_length=True, n_jobs=N_JOBS), Z_w)
maps = {"ligand only": Z_lig, "with water, matched": Z_w, "with water, not matched": Z_nm}
fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
for ax, (name, Z) in zip(axs, maps.items()):
    plot_space(Z, labels, D.LABEL_CONFIG, ax=ax, title=f"{name}\\nAUC(FDM 0) = {auc_per_coordinate(Z, labels)[0]:.3f}",
               legend=ax is axs[0])
plt.tight_layout(); plt.show()
print("with vs without matching, FDM 0-2:", np.round(compare_embeddings(Z_nm, Z_w, 3), 3))
print("All 50 runs start from the same structure, so the six site waters are the same molecules in the same order;")
print("matching therefore changes almost nothing here.")"""),
    md("## Surrogate models\n\n" + SURROGATE_HELP + """

Ligand-only map: 9 ligand descriptors (position x, y, r and orientation of benzamidine; z is left out because it
is the progress coordinate), each smoothed on its own. Map with water: radius, cos θ, |n<sub>z</sub>| and the 18
hydration occupancies (V1–V16, F<sub>G</sub>, F<sub>H</sub>)."""),
    code("""X_l, n_l = feature_matrix(D.ligand_descriptors(ts), D.LIG_ONLY_TEX, joint=False)
res_l = fit_surrogate(X_l, Z_lig[:, 0], n_l)
X_w, n_w = feature_matrix(D.water_descriptors(ts, cvs), D.water_descriptor_names())
res_w = fit_surrogate(X_w, Z_w[:, 0], n_w)
print("ligand only:    ", res_l.summary())
print("ligand + water: ", res_w.summary())
pd.concat({"ligand only": res_l.table(6), "ligand + water": res_w.table(6)}, axis=1).round(3)"""),
    code("""fig, axs = plt.subplots(2, 2, figsize=(6, 4.8))
plot_parity(res_l, labels, D.LABEL_CONFIG, axs=axs[0], title="ligand only")
plot_parity(res_w, labels, D.LABEL_CONFIG, axs=axs[1], title="ligand + water")
plt.tight_layout(); plt.show()
plot_beeswarm(res_w, max_display=10, title="ligand + water: FDM 0"); plt.show()
plot_beeswarm(res_l, max_display=10, title="ligand only: FDM 0"); plt.show()"""),
    md("""## Descriptors along the paths

Left: funnel coordinate z against the water occupancy of hydration site V9, one line per trajectory; the band is
the window z = 0.45–1.0 nm used to assign the routes (paper, Fig. b). Right: any COLVAR column (`DESC`, e.g.
`"st"`, `"vwo9"`, `"radius"`, `"FG"`) against z, each trajectory averaged in 0.05 nm bins of z (median and
interquartile band); by default the hydration site behind the highest-ranked SHAP feature of the map with water."""),
    code("""names_w = D.water_descriptor_names()
cols_w = ["radius", "costheta", "ringz"] + D.HYDRATION_COLS
top = cols_w[top_descriptor(res_w, names_w)]
DESC = top if top in cvs[0].columns else "vwo9"           # or e.g. "st"
z = [c["cyl.z"].to_numpy() for c in cvs]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
plot_paths([D.v9_occupancy(c) for c in cvs], z, labels, D.LABEL_CONFIG, ax=axs[0], xlabel="V9 occupancy",
           ylabel="z (nm)", lw=0.8, alpha=0.75)
axs[0].axhspan(*D.Z_WINDOW, color="k", alpha=0.06)
plot_binned(z, [c[DESC].to_numpy() for c in cvs], labels, D.LABEL_CONFIG, np.arange(0.2, 2.0, 0.05), ax=axs[1],
            xlabel="z (nm)", ylabel=DESC, legend=False)
axs[1].axvspan(*D.Z_WINDOW, color="k", alpha=0.06)
plt.tight_layout(); plt.show()"""),
    md("""## Setting: which waters

* `sites` (above): waters in the hydration sites V1–V16 in the first frame
* `near_atom`: waters within 4 Å of Asp189 CG in the first frame
* `closest`: the six waters that come closest to benzamidine at any frame (differ between trajectories)"""),
    code("""rows = []
fig, axs = plt.subplots(2, 3, figsize=(10, 5.2))
for col, pick in enumerate(["sites", "near_atom", "closest"]):
    tw, _, _ = D.load(pick, k=6, n_jobs=N_JOBS)
    Zm = embed(tw.fd, metric="l2", water=True, water_match="dtw", normalize_length=True, n_jobs=N_JOBS)
    Zn = align_signs(embed(tw.fd, metric="l2", water=False, normalize_length=True, n_jobs=N_JOBS), Zm)
    for row, (kind, Z) in enumerate([("matched", Zm), ("not matched", Zn)]):
        plot_space(Z, labels, D.LABEL_CONFIG, ax=axs[row, col], title=f"{pick} (k = {tw.water_info['k_waters']}), {kind}",
                   legend=False, s=10)
        rows.append({"waters": pick, "k": tw.water_info["k_waters"], "matching": kind,
                     "AUC FDM 0": auc_per_coordinate(Z, labels)[0], "rho FDM 0 vs paper map": compare_embeddings(Z, Z_w, 1)[0]})
plt.tight_layout(); plt.show()
pd.DataFrame(rows).round(3)"""),
]

NOTEBOOKS["t4l_benzene"] = [
    md("""
# Benzene unbinding from T4 lysozyme L99A

46 reactive trajectories labelled by the egress routes reported in the literature (Nunes-Alves et al.), named after
the helices that line them. The paths differ strongly in length (34–814 frames), so they are compared with DTW
(Sakoe–Chiba band of 8 % of the longer path, widened to the length difference of each pair; distances divided
by √(n + m)); σ = median, α = 0.5."""),
    code(HEADER + "\nfrom scipy.stats import spearmanr\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import adjusted_rand_score"),
    code("ts, labels = D.load(n_jobs=N_JOBS)\n{D.LABEL_CONFIG[k]['name']: int((labels == k).sum()) for k in D.LABEL_CONFIG}"),
    md("## FDM embedding (DTW)"),
    code("""Z, model = embed(ts.fd, metric="dtw", window="sakoe_chiba_wide", alpha=0.5, normalize_length=True,
                 return_model=True, n_jobs=N_JOBS)
ari = adjusted_rand_score(labels, KMeans(6, n_init=20, random_state=0).fit_predict(Z[:, :3]))
print(f"sigma = {model.sigma_:.3f} | k-means (k = 6) on FDM 0-2: ARI = {ari:.2f}")
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))
plot_space(Z, labels, D.LABEL_CONFIG, ax=axs[0]); plot_space(Z, labels, D.LABEL_CONFIG, dims=(0, 2), ax=axs[1], legend=False)
plt.tight_layout(); plt.show()"""),
    md("""## Does FDM sort the paths by length?

DTW aligns paths of different length, so FDM could in principle order them by duration alone."""),
    code("""L = ts.lengths
print(f"Spearman rho(FDM 0, length) = {spearmanr(Z[:, 0], L).statistic:.2f}")
print("median length per route:", {D.LABEL_CONFIG[k]['name']: int(np.median(L[labels == k])) for k in D.LABEL_CONFIG})
ari_len = adjusted_rand_score(labels, KMeans(6, n_init=20, random_state=0).fit_predict(np.log(L)[:, None]))
print(f"k-means on log(length) alone: ARI = {ari_len:.2f}")
plot_space(Z, c=np.log10(L), clabel="log10(frames)", title="FDM coloured by path length"); plt.show()"""),
    md("## Setting: kernel bandwidth σ and α"),
    code("""from pathfdm import FDM
from sklearn.preprocessing import StandardScaler
rows = []
for f in (0.5, 0.75, 1.0, 1.5, 2.0):
    for a in (0.0, 0.5, 1.0):          # the DTW distances are computed once and reused
        m = FDM(n_components=10, sigma=f * model.sigma_, alpha=a)
        Zs = StandardScaler().fit_transform(m.fit_transform(None, precomputed_D=model.D_matrix))
        rows.append({"sigma / median": f, "alpha": a,
                     "ARI": adjusted_rand_score(labels, KMeans(6, n_init=20, random_state=0).fit_predict(Zs[:, :3]))})
pd.DataFrame(rows).pivot(index="sigma / median", columns="alpha", values="ARI").round(2)"""),
]

NOTEBOOKS["imatinib_abl"] = [
    md("""
# Imatinib unbinding from Abl kinase (2HYY)

200 PathGennie unbinding trajectories. Labels: the three routes of the PathGennie study — Path I (P-loop/hinge),
Path II (αC/catalytic loop), Path III (C-lobe) — with Path III divided into two adjacent branches IIIa and IIIb,
and two outlier trajectories. FDM on the imatinib coordinates: L² distance, σ = median, α = 0."""),
    code(HEADER + "\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import adjusted_rand_score"),
    code("ts, labels = D.load(n_jobs=N_JOBS)\n{D.LABEL_CONFIG[k]['name']: int((labels == k).sum()) for k in D.LABEL_CONFIG}"),
    md("## FDM embedding"),
    code("""Z, model = embed(ts.fd, metric="l2", return_model=True, n_jobs=N_JOBS)
print(f"sigma = {model.sigma_:.2f}")
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))
plot_space(Z, labels, D.LABEL_CONFIG, ax=axs[0]); plot_space(Z, labels, D.LABEL_CONFIG, dims=(0, 2), ax=axs[1], legend=False)
plt.tight_layout(); plt.show()"""),
    md("## Agreement with the routes"),
    code("""ok = labels >= 0
p4 = KMeans(4, n_init=20, random_state=0).fit_predict(Z[:, :3])
print(f"k-means (k = 4) on FDM 0-2: ARI = {adjusted_rand_score(labels, p4):.3f} (all), "
      f"{adjusted_rand_score(labels[ok], p4[ok]):.3f} (outliers left out)")
coarse = np.where(np.isin(labels, [0, 3]), 3, labels)
p3 = KMeans(3, n_init=20, random_state=0).fit_predict(Z[ok, :3])
print(f"k-means (k = 3), Paths I-III: ARI = {adjusted_rand_score(coarse[ok], p3):.3f} (outliers left out)")
from scipy.spatial.distance import cdist
d = cdist(Z[:, :2], Z[:, :2]); np.fill_diagonal(d, np.inf); nn = d.min(1)
for k in (-1, -2):
    print(f"{D.LABEL_CONFIG[k]['name']}: nearest-neighbour distance {nn[labels == k][0]:.2f} (median {np.median(nn):.3f})")"""),
]

NOTEBOOKS["adenylate_kinase"] = [
    md("""
# Adenylate kinase: transition paths from four protocols

Three paths each from DIMS, FRODA, and restrained targeted MD at a fast (rTMD-F) and a slow (rTMD-S) pulling
rate (MDAnalysis PSAnalysisTutorial). FDM on the backbone coordinates: L² distance, σ = median, α = 0. The
collective variables are computed for comparison only."""),
    code(HEADER + "\nfrom sklearn.decomposition import PCA"),
    code("ts, labels, cvs = D.load()\nprint(ts.names)"),
    md("## Collective-variable projections"),
    code("""fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
for c, m in zip(cvs, labels):
    col = D.LABEL_CONFIG[m]["color"]
    axs[0].plot(c[:, 0], c[:, 1], color=col, lw=0.8); axs[1].plot(c[:, 2], c[:, 3], color=col, lw=0.8)
    axs[2].plot(c[:, 4], c[:, 5], color=col, lw=0.8)
for ax, (i, j) in zip(axs, [(0, 1), (2, 3), (4, 5)]):
    ax.set_xlabel(f"${D.CV_NAMES[i]}$"); ax.set_ylabel(f"${D.CV_NAMES[j]}$")
axs[0].set_xlabel("fraction of 1AKE native contacts"); axs[0].set_ylabel("fraction of 4AKE native contacts")
for k, cfg in D.LABEL_CONFIG.items():
    axs[2].plot([], [], color=cfg["color"], label=cfg["name"])
axs[2].legend(fontsize=6); plt.tight_layout(); plt.show()"""),
    md("## PCA of the backbone coordinates vs FDM"),
    code("""pca = PCA(2).fit(np.vstack(ts.coords))
print("PCA explained variance:", np.round(pca.explained_variance_ratio_, 3))
Z, model = embed(ts.fd, metric="l2", return_model=True)
print(f"sigma = {model.sigma_:.2f}")
fig, axs = plt.subplots(1, 2, figsize=(7, 2.9))
for c, m in zip(ts.coords, labels):
    p = pca.transform(c); axs[0].scatter(p[:, 0], p[:, 1], s=3, color=D.LABEL_CONFIG[m]["color"], alpha=0.5, edgecolors="none")
axs[0].set_xlabel("PC 1"); axs[0].set_ylabel("PC 2"); axs[0].set_title("PCA (frames)")
plot_space(Z, labels, D.LABEL_CONFIG, ax=axs[1], title="FDM (one point per path)")
plt.tight_layout(); plt.show()
pd.DataFrame([{"method": D.METHODS[k], "FDM 0": Z[labels == k, 0].mean(), "FDM 1": Z[labels == k, 1].mean(),
               "sd FDM 1": Z[labels == k, 1].std()} for k in range(4)]).round(2)"""),
]


def main(names):
    for name in names:
        nb = nbf.v4.new_notebook()
        nb.cells = NOTEBOOKS[name]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        out = ROOT / "systems" / name / f"{name}.ipynb"
        nbf.write(nb, out)
        print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:] or list(NOTEBOOKS))
