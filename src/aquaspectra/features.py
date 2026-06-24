"""Recursive Feature Elimination (RFE) band ranking.

Paper: wavelength bands are ranked by significance using RFE; classifiers are
then evaluated on the top-k subsets.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


def _make_estimator(name: str, random_state: int):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        )
    if name == "svm":
        # Linear kernel exposes coef_ which RFE needs.
        return SVC(kernel="linear")
    raise ValueError(f"Unknown RFE estimator: {name}")


def rank_bands(
    X: np.ndarray,
    y: np.ndarray,
    band_cols: list[str],
    estimator: str = "random_forest",
    random_state: int = 42,
    step: int | float = 1,
) -> pd.DataFrame:
    """Return a DataFrame ranking every band (rank 1 = most important).

    `step` controls how many features RFE drops per iteration (int = absolute,
    float in (0,1) = fraction). Larger steps are much faster on many bands.
    """
    est = _make_estimator(estimator, random_state)
    rfe = RFE(estimator=est, n_features_to_select=1, step=step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfe.fit(X, y)

    df = pd.DataFrame(
        {
            "band_index": np.arange(len(band_cols)),
            "band": band_cols,
            "rfe_rank": rfe.ranking_,
        }
    ).sort_values("rfe_rank").reset_index(drop=True)
    return df


def top_k_indices(ranking: pd.DataFrame, k: int) -> list[int]:
    return ranking.sort_values("rfe_rank").head(k)["band_index"].tolist()
