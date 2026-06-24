"""End-to-end pipeline orchestrating the paper's methodology."""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .bands import BandStack
from .config import Config
from .features import rank_bands, top_k_indices
from .models import fit_and_evaluate_both
from .ndvi import canopy_mask
from .sampling import sample_points


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def run_sampling(cfg: Config) -> pd.DataFrame:
    stack = BandStack.from_config(cfg)
    print(f"[bands] {stack.n_bands} bands, grid {stack.height}x{stack.width}, "
          f"wl {stack.wavelengths.min():.0f}-{stack.wavelengths.max():.0f} nm")

    canopy = None
    if cfg.get("ndvi", "enabled", default=True):
        canopy = canopy_mask(
            stack,
            cfg.get("ndvi", "red_nm"),
            cfg.get("ndvi", "nir_nm"),
            cfg.get("ndvi", "threshold"),
        )
        print(f"[ndvi] canopy pixels: {int(canopy.sum())} "
              f"({100*canopy.mean():.1f}% of scene)")

    df = sample_points(
        stack,
        cfg.path("labels", "vector_file"),
        cfg.get("labels", "class_field"),
        cfg.get("labels", "subplot_field"),
        canopy,
        cfg.get("sampling", "points_per_subplot", default=200),
        cfg.get("sampling", "random_state", default=42),
    )
    print(f"[sampling] collected {len(df)} samples "
          f"across {df['subplot_id'].nunique()} sub-plots")

    out_dir = _ensure_dir(cfg.path("output", "dir"))
    print("[sampling] writing samples.csv ...")
    df.to_csv(os.path.join(out_dir, cfg.get("output", "samples_csv")),
              index=False, float_format="%.5f")
    return df


def run_modelling(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    band_cols = [c for c in df.columns if c.startswith("b") and c[1:5].isdigit()]
    X = df[band_cols].to_numpy(dtype="float32")
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    print(f"[labels] classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

    rs = cfg.get("model", "random_state", default=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.get("model", "test_size", default=0.2),
        random_state=rs, stratify=y,
    )

    # RFE ranking on training data only (avoid leakage).
    ranking = rank_bands(
        X_tr, y_tr, band_cols,
        estimator=cfg.get("feature_selection", "estimator", default="random_forest"),
        random_state=rs,
        step=cfg.get("feature_selection", "rfe_step", default=1),
    )
    out_dir = _ensure_dir(cfg.path("output", "dir"))
    ranking.to_csv(
        os.path.join(out_dir, cfg.get("output", "band_ranking_csv")), index=False
    )

    rows = []
    best = None  # (kappa, model_name, k, estimator, band_indices)
    for k in cfg.get("feature_selection", "top_k_list", default=[10]):
        idx = top_k_indices(ranking, k)
        wl = ", ".join(
            f"{int(float(band_cols[i].split('_')[1][:-2]))}" for i in idx
        )
        res = fit_and_evaluate_both(
            X_tr[:, idx], y_tr, X_te[:, idx], y_te, cfg.get("model")
        )
        for r in res:
            rows.append({
                "top_k": k, "model": r.name, "accuracy": round(r.accuracy, 4),
                "kappa": round(r.kappa, 4), "wavelengths_nm": wl,
                "best_params": r.best_params,
            })
            if best is None or r.kappa > best[0]:
                best = (r.kappa, r.name, k, r.estimator, idx)
        print(f"[top{k:>2}] " + " | ".join(
            f"{r.name} acc={r.accuracy:.4f} kappa={r.kappa:.4f}" for r in res))

    results = pd.DataFrame(rows)
    results.to_csv(
        os.path.join(out_dir, cfg.get("output", "results_csv")), index=False
    )

    # Persist the best model bundle.
    bundle = {
        "model": best[3],
        "model_name": best[1],
        "top_k": best[2],
        "band_indices": best[4],
        "band_cols": [band_cols[i] for i in best[4]],
        "label_encoder": le,
    }
    joblib.dump(bundle, os.path.join(out_dir, cfg.get("output", "model_file")))
    print(f"[best] {best[1]} with top-{best[2]} bands -> kappa={best[0]:.4f}")
    return results


def run_all(cfg: Config) -> pd.DataFrame:
    df = run_sampling(cfg)
    return run_modelling(cfg, df)
