"""Trajectory loaders: one aligned coordinate path per trajectory, packed into an skfda.FDataIrregular.

Each trajectory becomes a path x(t) of flattened Cartesian coordinates (solute atoms, optionally followed by
selected water oxygens). The returned ``TrajectorySet`` carries the FDataIrregular object used by
``pathfdm.FDM`` together with the per-trajectory arrays and the ``water_info`` that the water matching needs.

Alignment modes
---------------
``align_mode="fit"``        unwrap (no jumps across the box) and rigid fit of ``align_sel`` onto the atoms of
                            ``reference`` (MDAnalysis fit_rot_trans, vectorised); required when waters are kept.
``align_mode="aligntraj"``  MDAnalysis AlignTraj of ``align_sel`` onto the same selection of ``reference``
                            (no unwrapping); ``align_sel`` may be a dict {"mobile": ..., "reference": ...}.

Water selection (``water=``)
----------------------------
``None``                                    solute only
``{"pick": "closest", "k": K}``             the K water oxygens that come closest to the solute at any frame,
                                            ordered by closest approach
``{"pick": "sites", "sites": xyz, "cutoff": c}``
                                            in the first (fitted) frame, the water nearest to each site in
                                            ``xyz`` (Angstrom) if closer than ``c``
``{"pick": "near_atom", "atom_index": i, "cutoff": c}``
                                            in the first frame, the waters within ``c`` of atom ``i``
The same water molecules are followed through the whole trajectory.

Time axis (``time=``): ``"progress"`` = linspace(0, 1, n_frames), ``"frame"`` = frame index.
"""
from __future__ import annotations

import glob
import os
import re
import traceback
from dataclasses import dataclass, field

import MDAnalysis as mda
import numpy as np
import skfda
from joblib import Parallel, delayed
from MDAnalysis.analysis import align
from MDAnalysis.transformations import fit_rot_trans
from scipy.spatial import cKDTree

WATER_O_SEL = "resname SOL HOH TIP3 WAT and (name OW or name O or name OH2)"


def traj_number(path: str) -> int:
    """Last integer in the file name (trajectory_12.dcd -> 12); 0 if there is none."""
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 0


def find_trajectories(folder: str, pattern: str = "*.dcd", sort: str = "number") -> list[str]:
    """Trajectory files in ``folder`` (recursive). sort = "number" (last integer in the name) or "name"."""
    files = glob.glob(os.path.join(folder, "**", pattern), recursive=True) or glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} files in {folder}")
    return sorted(files, key=traj_number) if sort == "number" else sorted(files)


@dataclass
class TrajectorySet:
    """Loaded trajectories. ``fd`` is the input of FDM; ``coords[i]`` is trajectory i as (n_frames, n_features)."""
    fd: skfda.FDataIrregular
    files: list
    coords: list
    times: list
    water_info: dict
    water_ids: list = field(default_factory=list)      # per trajectory: resids of the selected waters

    @property
    def names(self) -> list[str]:
        return [os.path.splitext(os.path.basename(f))[0] for f in self.files]

    @property
    def lengths(self) -> np.ndarray:
        return np.array([len(c) for c in self.coords])

    def __len__(self):
        return len(self.coords)

    def solute_only(self) -> "TrajectorySet":
        """The same trajectories without the water columns."""
        sdim = self.water_info["solute_dim"]
        return _pack([c[:, :sdim] for c in self.coords], self.times, self.files,
                     self.water_info["n_solute_atoms"], 0, [])

    def subset(self, idx) -> "TrajectorySet":
        idx = list(idx)
        return _pack([self.coords[i] for i in idx], [self.times[i] for i in idx], [self.files[i] for i in idx],
                     self.water_info["n_solute_atoms"], self.water_info["k_waters"],
                     [self.water_ids[i] for i in idx] if self.water_ids else [])

    def first_k_waters(self, k: int) -> "TrajectorySet":
        """Keep only the first k selected waters (with pick="closest": the k closest)."""
        sdim, kk = self.water_info["solute_dim"], self.water_info["k_waters"]
        if k > kk:
            raise ValueError(f"only {kk} waters loaded")
        return _pack([c[:, :sdim + 3 * k] for c in self.coords], self.times, self.files,
                     self.water_info["n_solute_atoms"], k, [w[:k] for w in self.water_ids])


def _pack(coords, times, files, n_solute_atoms, k, water_ids) -> TrajectorySet:
    dim = coords[0].shape[1]
    fd = skfda.FDataIrregular(
        start_indices=np.concatenate([[0], np.cumsum([len(t) for t in times[:-1]])]).astype(np.intp),
        points=np.concatenate(times).reshape(-1, 1),
        values=np.concatenate(coords, axis=0),
        argument_names=("progress",),
        coordinate_names=[f"coord_{i}" for i in range(dim)],
    )
    info = {"has_water": k > 0, "n_solute_atoms": n_solute_atoms, "solute_dim": 3 * n_solute_atoms,
            "k_waters": k, "atoms_per_water": 1, "water_dim": 3 * k}
    fd.water_info = info
    return TrajectorySet(fd=fd, files=list(files), coords=list(coords), times=list(times), water_info=info,
                         water_ids=list(water_ids))


def _unwrap_and_fit(pos, boxes, mobile_idx, fit):
    """NoJump then fit_rot_trans (as MDAnalysis applies them per frame), vectorised over frames."""
    frac = np.einsum("fni,fij->fnj", pos, np.linalg.inv(boxes))
    shift = np.zeros_like(frac)
    shift[1:] = np.cumsum(np.round(np.diff(frac, axis=0)), axis=0)
    out = np.empty_like(pos)
    out[0] = pos[0]
    out[1:] = np.einsum("fni,fij->fnj", frac[1:] - shift[1:], boxes[1:])
    w = fit.weights
    for f in range(len(out)):
        mob = out[f, mobile_idx]
        com = (mob * w[:, None]).sum(axis=0) / w.sum()
        rot, _ = align.rotation_matrix(mob - com, fit.ref_coordinates, weights=w)
        out[f] = (np.dot((out[f] - com).astype(np.float32), rot.T) + fit.ref_com).astype(np.float32)
    return out


def _select_waters(water, pos, sol, wat, u, where):
    pick = water["pick"]
    if pick == "closest":
        k = int(water["k"])
        global_min = np.full(wat.shape[1], np.inf, dtype=np.float32)
        for f in range(len(pos)):
            d, _ = cKDTree(sol[f]).query(wat[f], k=1, workers=1)
            np.minimum(global_min, d, out=global_min)
        chosen = np.argpartition(global_min, k)[:k]
        return chosen[np.argsort(global_min[chosen])]
    if pick == "sites":
        d, idx = cKDTree(wat[0]).query(np.asarray(water["sites"], float), k=1)
        return np.array(list(dict.fromkeys(idx[d < water["cutoff"]])), dtype=int)
    if pick == "near_atom":
        ref = pos[0, where(u.atoms[[int(water["atom_index"])]])][0]
        return np.where(np.linalg.norm(wat[0] - ref, axis=1) < water["cutoff"])[0]
    raise ValueError(f"unknown water pick {pick!r}")


def _load_one(path, topology, reference, solute_sel, align_sel, align_mode, align_weights, water, start_frame,
              stride, time, dtype):
    try:
        u = mda.Universe(topology, path)
        ref_u = mda.Universe(reference if reference is not None else topology)
        solute = u.select_atoms(solute_sel)
        if solute.n_atoms == 0:
            raise ValueError(f"solute selection {solute_sel!r} is empty")
        water_ids = []
        if align_mode == "aligntraj":
            if water is not None:
                raise ValueError("water selection needs align_mode='fit' (unwrapped coordinates)")
            sel = align_sel if isinstance(align_sel, dict) else {"mobile": align_sel, "reference": align_sel}
            align.AlignTraj(u, ref_u, select=sel, weights=align_weights, in_memory=True).run()
            frames = range(0, len(u.trajectory), stride)
            coords = np.empty((len(frames), solute.n_atoms * 3), dtype=dtype)
            for i, f in enumerate(frames):
                u.trajectory[f]
                coords[i] = solute.positions.ravel()
        elif align_mode == "fit":
            mobile = u.select_atoms(align_sel)
            fit = fit_rot_trans(mobile, ref_u.atoms, weights=align_weights)
            if align_weights is None:
                fit.weights = np.ones(mobile.n_atoms)
            water_o = u.select_atoms(water.get("selection", WATER_O_SEL)) if water else None
            groups = [solute] + ([water_o] if water else [])
            if water and water["pick"] == "near_atom":
                groups.append(u.atoms[[int(water["atom_index"])]])
            need = sum(groups[1:], groups[0]) | mobile
            where = lambda ag: np.searchsorted(need.indices, ag.indices)
            traj = u.trajectory[::stride]
            pos = np.empty((len(traj), need.n_atoms, 3), dtype=np.float32)
            boxes = np.empty((len(traj), 3, 3))
            for i, ts in enumerate(traj):
                pos[i] = need.positions
                boxes[i] = ts.triclinic_dimensions
            pos = _unwrap_and_fit(pos, boxes, where(mobile), fit)
            sol = pos[:, where(solute)]
            if water:
                wat = pos[:, where(water_o)]
                chosen = _select_waters(water, pos, sol, wat, u, where)   # waters chosen from all frames
                water_ids = water_o[chosen].resids.tolist()
                sol = np.concatenate([sol, wat[:, chosen]], axis=1)
            coords = sol.reshape(len(pos), -1).astype(dtype)
        else:
            raise ValueError(f"align_mode must be 'fit' or 'aligntraj', got {align_mode!r}")
        coords = coords[start_frame:]
        n = len(coords)
        times = np.linspace(0.0, 1.0, n) if time == "progress" else np.arange(start_frame, start_frame + n,
                                                                                 dtype=np.float64) * stride
        return dict(ok=True, path=path, coords=coords, times=times, n_solute=solute.n_atoms, water_ids=water_ids)
    except Exception as e:  # report and skip
        return dict(ok=False, path=path, error=f"{e}\n{traceback.format_exc()}")


def load_trajectories(files, topology, reference=None, *, solute_sel: str, align_sel, align_mode: str = "fit",
                      align_weights=None, water: dict | None = None, start_frame: int = 0, stride: int = 1,
                      time: str = "progress", max_frames: int | None = None, n_jobs: int = 4,
                      dtype=np.float32, verbose: bool = True) -> TrajectorySet:
    """Load, align and pack trajectories (see the module docstring for the options).

    ``topology`` is one file for all trajectories or a list with one topology per trajectory.
    ``reference`` defaults to the topology (its first frame).
    """
    files = list(files)
    tops = topology if isinstance(topology, (list, tuple)) else [topology] * len(files)
    results = Parallel(n_jobs=min(n_jobs, len(files)), backend="loky")(
        delayed(_load_one)(f, t, reference, solute_sel, align_sel, align_mode, align_weights, water, start_frame,
                           stride, time, dtype) for f, t in zip(files, tops))
    good = []
    for r in results:
        if not r["ok"]:
            print(f"[SKIP] {os.path.basename(r['path'])}: {r['error'].splitlines()[0]}")
        elif max_frames is not None and len(r["coords"]) >= max_frames:
            print(f"[SKIP] {os.path.basename(r['path'])}: {len(r['coords'])} frames >= max_frames")
        else:
            good.append(r)
    if not good:
        raise RuntimeError("no trajectory could be loaded")
    ks = {len(r["water_ids"]) for r in good}
    if len(ks) != 1:
        raise ValueError(f"different numbers of selected waters across trajectories: {sorted(ks)}")
    k = ks.pop()
    ts = _pack([r["coords"] for r in good], [r["times"] for r in good], [r["path"] for r in good],
               good[0]["n_solute"], k, [r["water_ids"] for r in good])
    if verbose:
        lens = ts.lengths
        print(f"{len(ts)} trajectories | {good[0]['n_solute']} solute atoms + {k} water oxygens "
              f"| {lens.min()}-{lens.max()} frames | {ts.fd.values.shape[1]} coordinates per frame")
    return ts
