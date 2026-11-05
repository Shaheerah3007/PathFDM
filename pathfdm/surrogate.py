"""Surrogate model of a diffusion coordinate and its SHAP attribution.

An ExtraTrees regressor predicts one FDM coordinate from standardized trajectory-level descriptors.
Accuracy is reported as
    * out-of-bag R^2: each trajectory predicted by the trees whose bootstrap sample did not contain it;
    * 5-fold cross-validation repeated 5 times: held-out R^2 per fold (mean +/- sd over the 25 folds), and the
      mean of the 5 held-out predictions of every trajectory (for parity plots).
Attribution is the SHAP value of every descriptor (TreeExplainer on the model fitted to all trajectories).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

ET_PARAMS = dict(n_estimators=500, bootstrap=True, random_state=42, n_jobs=-1)


@dataclass
class SurrogateResult:
    names: list
    X: np.ndarray                 # standardized descriptors
    y: np.ndarray                 # target coordinate
    model: ExtraTreesRegressor
    shap_values: np.ndarray
    oob_r2: float
    y_oob: np.ndarray
    cv_r2: np.ndarray             # one value per fold
    y_cv: np.ndarray              # mean held-out prediction of every trajectory
    frac_rank1: np.ndarray        # fraction of folds in which a descriptor has the largest mean |SHAP|

    @property
    def shap_share(self) -> np.ndarray:
        imp = np.abs(self.shap_values).mean(axis=0)
        return imp / imp.sum()

    def table(self, top: int | None = None) -> pd.DataFrame:
        t = pd.DataFrame({"feature": self.names, "shap_share": self.shap_share, "frac_folds_rank1": self.frac_rank1})
        t = t.sort_values("shap_share", ascending=False).reset_index(drop=True)
        return t.head(top) if top else t

    def summary(self) -> str:
        return (f"{len(self.names)} descriptors | out-of-bag R2 = {self.oob_r2:.3f} | 5-fold CV x 5: "
                f"R2 = {self.cv_r2.mean():.3f} +/- {self.cv_r2.std():.3f}")


def fit_surrogate(X, y, names, cv: bool = True, n_splits: int = 5, n_repeats: int = 5,
                  standardize: bool = True) -> SurrogateResult:
    X = StandardScaler().fit_transform(X) if standardize else np.asarray(X, float)
    y = np.asarray(y, float)
    model = ExtraTreesRegressor(oob_score=True, **ET_PARAMS).fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    fold_r2, rank1 = [], np.zeros(X.shape[1])
    pred_sum, n_test = np.zeros(len(y)), np.zeros(len(y))
    if cv:
        for tr, te in RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42).split(X):
            m = ExtraTreesRegressor(**ET_PARAMS).fit(X[tr], y[tr])
            yh = m.predict(X[te])
            fold_r2.append(r2_score(y[te], yh))
            pred_sum[te] += yh
            n_test[te] += 1
            rank1[np.argmax(np.abs(shap.TreeExplainer(m).shap_values(X[te])).mean(0))] += 1
        n_folds = len(fold_r2)
    else:
        n_folds, n_test = 1, np.ones(len(y))
    return SurrogateResult(names=list(names), X=X, y=y, model=model, shap_values=sv, oob_r2=model.oob_score_,
                           y_oob=model.oob_prediction_, cv_r2=np.array(fold_r2), y_cv=pred_sum / n_test,
                           frac_rank1=rank1 / n_folds)
