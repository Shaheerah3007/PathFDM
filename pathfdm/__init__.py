"""PathFDM: Functional Diffusion Maps of molecular transition-path ensembles.

    from pathfdm import FDM, load_trajectories, embed
    ts = load_trajectories(files, topology, solute_sel="resname LIG", align_sel="protein", align_mode="aligntraj")
    Z = embed(ts.fd, metric="l2")                      # standardized diffusion coordinates FDM 0, 1, ...
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from .fdm import FDM, pairwise_distances
from .loaders import TrajectorySet, find_trajectories, load_trajectories

__version__ = "0.1.0"


def embed(fd, n_components: int = 10, metric: str = "l2", sigma="median", alpha: float = 0.0, water: bool = False,
          water_match: str = "dtw", return_model: bool = False, **kw):
    """Fit FDM and return the diffusion coordinates standardized per column (FDM 0, FDM 1, ...).

    Each eigenvector is scaled to zero mean and unit variance across trajectories, which removes the
    lambda_k^t factor, so the result does not depend on the diffusion time.
    """
    kw.setdefault("subsample_t", None)
    model = FDM(n_components=n_components, metric=metric, kernel="rbf", sigma=sigma, alpha=alpha, step=1,
                water=water, water_match=water_match, **kw)
    Z = StandardScaler().fit_transform(model.fit_transform(fd))
    return (Z, model) if return_model else Z


__all__ = ["FDM", "pairwise_distances", "TrajectorySet", "find_trajectories", "load_trajectories", "embed"]
