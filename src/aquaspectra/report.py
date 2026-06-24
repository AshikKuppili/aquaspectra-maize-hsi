"""Generate a multi-page PDF report summarising an AquaSpectra analysis.

Pulls together everything produced by the pipeline (sample spectra, RFE band
ranking, model accuracy/Kappa, and the stress map) into a single, shareable PDF
aimed at agriculture researchers. Uses matplotlib's PdfPages so no extra
dependencies are required.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .config import Config


def _band_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("b") and c[1:5].isdigit()]


def _wl_of(col: str) -> float:
    return float(col.split("_")[1][:-2])


def _title_page(pdf, cfg, meta: dict):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    ax.text(0.5, 0.93, "AquaSpectra", ha="center", fontsize=30,
            weight="bold", color="#2e7d32")
    ax.text(0.5, 0.89, "Maize Crop Water-Stress Detection Report",
            ha="center", fontsize=15)
    ax.text(0.5, 0.86, f"Generated {datetime.now():%Y-%m-%d %H:%M}",
            ha="center", fontsize=10, color="grey")
    ax.hlines(0.835, 0.08, 0.92, color="#2e7d32", lw=1.5)

    lines = [
        ("Scene", ""),
        ("  Spectral bands", f"{meta.get('n_bands', '-')}"),
        ("  Wavelength range", meta.get("wl_range", "-")),
        ("  Grid (W x H)", meta.get("grid", "-")),
        ("  CRS", str(meta.get("crs", "-"))),
        ("", ""),
        ("Canopy / sampling", ""),
        ("  NDVI canopy coverage", meta.get("canopy", "-")),
        ("  Samples collected", meta.get("n_samples", "-")),
        ("  Sub-plots", meta.get("n_subplots", "-")),
        ("  Class distribution", meta.get("class_dist", "-")),
        ("", ""),
        ("Best model", ""),
        ("  Classifier", meta.get("best_model", "-")),
        ("  Band subset", meta.get("best_k", "-")),
        ("  Accuracy", meta.get("best_acc", "-")),
        ("  Cohen's Kappa", meta.get("best_kappa", "-")),
    ]
    y = 0.79
    for k, v in lines:
        if v == "" and k and not k.startswith("  "):
            ax.text(0.10, y, k, fontsize=12, weight="bold", color="#1b5e20")
        else:
            ax.text(0.12, y, k, fontsize=11)
            ax.text(0.62, y, str(v), fontsize=11, weight="bold")
        y -= 0.032

    ax.text(0.08, 0.10,
            "Method: NDVI canopy masking -> RFE band selection -> RF/SVM "
            "(80/20 split, 3-fold CV, Cohen's Kappa).",
            fontsize=9, color="#333333", wrap=True)
    ax.text(0.08, 0.065,
            "Reference: Mohite et al., \"Detection of Crop Water Stress in "
            "Maize using Drone Based Hyperspectral Imaging\", IGARSS 2022, "
            "IEEE. doi:10.1109/IGARSS46834.2022.9884686",
            fontsize=8, color="#555555", wrap=True)
    ax.text(0.08, 0.035,
            "AquaSpectra is an independent reimplementation. Results require "
            "multi-season validation before operational use.",
            fontsize=8, style="italic", color="#777777", wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _spectra_page(pdf, samples: pd.DataFrame):
    cols = _band_cols(samples)
    wl = [_wl_of(c) for c in cols]
    fig, ax = plt.subplots(figsize=(8.27, 5.0))
    for label, g in samples.groupby("label"):
        m = g[cols].mean().values
        sd = g[cols].std().values
        ax.plot(wl, m, label=f"{label} (n={len(g)})", lw=1.8)
        ax.fill_between(wl, m - sd, m + sd, alpha=0.12)
    ax.axvspan(670, 780, color="orange", alpha=0.15,
               label="670-780 nm (key region)")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance")
    ax.set_title("Mean spectral signature by class (\u00b11 SD)")
    ax.legend(fontsize=9)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _ranking_page(pdf, ranking: pd.DataFrame, top: int = 20):
    rk = ranking.sort_values("rfe_rank").head(top)
    fig, ax = plt.subplots(figsize=(8.27, 6.0))
    ax.barh(rk["band"][::-1], (1.0 / rk["rfe_rank"])[::-1], color="seagreen")
    ax.set_xlabel("Importance (1 / RFE rank)")
    ax.set_title(f"Top-{top} RFE-ranked spectral bands")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _results_page(pdf, results: pd.DataFrame):
    num_cols = [c for c in ["top_k", "model", "accuracy", "kappa"]
                if c in results.columns]
    tbl = results[num_cols].copy()

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Model performance by band subset", fontsize=14, y=0.96)

    # --- numeric table (top half) ---
    ax = fig.add_axes([0.08, 0.50, 0.84, 0.40]); ax.axis("off")
    t = ax.table(cellText=tbl.values, colLabels=tbl.columns,
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    if "kappa" in tbl.columns:
        best_pos = list(tbl.index).index(tbl["kappa"].astype(float).idxmax())
        for j in range(len(tbl.columns)):
            t[(best_pos + 1, j)].set_facecolor("#c8e6c9")

    # --- selected wavelengths per subset (bottom half, wrapped text) ---
    if "wavelengths_nm" in results.columns and "top_k" in results.columns:
        ax2 = fig.add_axes([0.08, 0.06, 0.84, 0.38]); ax2.axis("off")
        ax2.text(0, 1.0, "Selected wavelengths (nm) per subset",
                 fontsize=11, weight="bold", color="#1b5e20",
                 transform=ax2.transAxes, va="top")
        seen, y = set(), 0.92
        for _, row in results.iterrows():
            k = row["top_k"]
            if k in seen:
                continue
            seen.add(k)
            ax2.text(0, y, f"top-{k}:", fontsize=9, weight="bold",
                     transform=ax2.transAxes, va="top")
            ax2.text(0.12, y, str(row["wavelengths_nm"]), fontsize=9,
                     transform=ax2.transAxes, va="top", wrap=True)
            y -= 0.085
    pdf.savefig(fig); plt.close(fig)


def _map_page(pdf, map_path: str, classes: list[str]):
    import rasterio
    from matplotlib.colors import ListedColormap, BoundaryNorm
    with rasterio.open(map_path) as ds:
        arr = ds.read(1)
    # Semantic colours: stress=red, no-stress=green, soil=grey.
    class_colors = {
        "NS": "#1a9850",      # No Stress -> green
        "GFWS": "#d73027",    # Grain-Fill Water Stress -> red
    }
    colors = ["#e6e6e6"] + [class_colors.get(c, "#7570b3") for c in classes]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(range(len(colors) + 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8.27, 7.0))
    im = ax.imshow(arr, cmap=cmap, norm=norm)
    ax.set_title("Predicted water-stress map"); ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[i + 0.5 for i in range(len(colors))])
    labels = ["soil/nodata"] + [
        f"{c} (no stress)" if c == "NS"
        else f"{c} (stress)" if c == "GFWS" else c
        for c in classes
    ]
    cbar.ax.set_yticklabels(labels)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _moisture_page(pdf, moisture_path: str):
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(moisture_path) as ds:
        w = 1100; h = int(ds.height * (w / ds.width))
        arr = ds.read(1, out_shape=(1, h, w), resampling=Resampling.average)
    v = arr[np.isfinite(arr)]

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Canopy moisture \u2014 SWIR NDWI", fontsize=14, y=0.96)
    ax = fig.add_axes([0.08, 0.50, 0.84, 0.40])
    lo, hi = (np.nanpercentile(v, [2, 98]) if v.size else (-0.5, 0.8))
    im = ax.imshow(arr, cmap="Blues", vmin=lo, vmax=hi)
    ax.set_title("NDWI = (NIR - SWIR)/(NIR + SWIR)   high = wetter canopy")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = fig.add_axes([0.12, 0.10, 0.76, 0.28])
    if v.size:
        ax2.hist(v, bins=50, color="#1f6fb2", alpha=0.85)
    ax2.set_xlabel("NDWI"); ax2.set_ylabel("pixels")
    ax2.set_title("Moisture distribution over canopy"); ax2.grid(alpha=0.3)

    if v.size:
        txt = (f"canopy pixels: {v.size:,}    "
               f"NDWI min/mean/max: {v.min():.3f} / {v.mean():.3f} / {v.max():.3f}")
        fig.text(0.5, 0.05, txt, ha="center", fontsize=10, color="#333333")
    pdf.savefig(fig); plt.close(fig)


def build_report(cfg: Config, out_pdf: str | None = None,
                 stack_meta: dict | None = None) -> str:
    """Assemble the PDF report from files in the output directory."""
    out_dir = cfg.path("output", "dir")
    out_pdf = out_pdf or os.path.join(out_dir, "AquaSpectra_report.pdf")

    meta = dict(stack_meta or {})

    samples_path = os.path.join(out_dir, cfg.get("output", "samples_csv"))
    ranking_path = os.path.join(out_dir, cfg.get("output", "band_ranking_csv"))
    results_path = os.path.join(out_dir, cfg.get("output", "results_csv"))
    model_path = os.path.join(out_dir, cfg.get("output", "model_file"))
    map_path = os.path.join(out_dir, cfg.get("output", "stress_map"))
    moisture_path = os.path.join(
        out_dir, cfg.get("output", "moisture_map", default="ndwi_real.tif"))

    samples = pd.read_csv(samples_path) if os.path.exists(samples_path) else None
    ranking = pd.read_csv(ranking_path) if os.path.exists(ranking_path) else None
    results = pd.read_csv(results_path) if os.path.exists(results_path) else None

    if samples is not None:
        meta.setdefault("n_samples", f"{len(samples):,}")
        meta.setdefault("n_subplots", samples["subplot_id"].nunique())
        dist = samples["label"].value_counts().to_dict()
        meta.setdefault("class_dist",
                        ", ".join(f"{k}: {v}" for k, v in dist.items()))
    if results is not None and len(results):
        best = results.loc[results["kappa"].astype(float).idxmax()]
        meta.setdefault("best_model", best["model"])
        meta.setdefault("best_k", f"top-{best['top_k']}")
        meta.setdefault("best_acc", f"{float(best['accuracy']):.4f}")
        meta.setdefault("best_kappa", f"{float(best['kappa']):.4f}")

    classes = None
    if os.path.exists(model_path):
        import joblib
        try:
            classes = list(joblib.load(model_path)["label_encoder"].classes_)
        except Exception:  # noqa
            classes = None

    with PdfPages(out_pdf) as pdf:
        _title_page(pdf, cfg, meta)
        if samples is not None:
            _spectra_page(pdf, samples)
        if ranking is not None:
            _ranking_page(pdf, ranking)
        if results is not None:
            _results_page(pdf, results)
        if os.path.exists(map_path) and classes:
            _map_page(pdf, map_path, classes)
        if os.path.exists(moisture_path):
            _moisture_page(pdf, moisture_path)

    return out_pdf
