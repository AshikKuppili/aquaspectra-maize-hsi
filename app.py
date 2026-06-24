"""AquaSpectra - Streamlit UI for maize water-stress detection.

IMPORTANT — large hyperspectral inputs are NOT uploaded through the browser.
The band-separated TIFs are large, so this UI reads them directly from a
**server-side folder path** that you type in. Only small artefacts (an optional
plots.geojson) may be uploaded.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from aquaspectra.config import Config, load_config
from aquaspectra.bands import BandStack
from aquaspectra.ndvi import canopy_mask, compute_ndvi
from aquaspectra.pipeline import run_sampling, run_modelling

st.set_page_config(page_title="AquaSpectra - Maize Water Stress", layout="wide")
st.title("🌽 AquaSpectra - Maize Crop Water-Stress Detection")
st.caption("Drone hyperspectral imaging · IGARSS 2022 (Mohite et al.) replication")


# --------------------------------------------------------------------------- #
# Sidebar: configuration (path-based — no large uploads)
# --------------------------------------------------------------------------- #
st.sidebar.header("Configuration")

mode = st.sidebar.radio(
    "Config source", ["Load config.yaml", "Edit fields"], index=0)

DEFAULT_CFG = os.path.join(os.path.dirname(__file__), "config.yaml")


def build_config() -> Config | None:
    if mode == "Load config.yaml":
        cfg_path = st.sidebar.text_input("Config file path", DEFAULT_CFG)
        if not os.path.exists(cfg_path):
            st.sidebar.error("Config file not found.")
            return None
        cfg = load_config(cfg_path)
        # allow overriding the (large) band folder without touching the file
        bd = st.sidebar.text_input(
            "Override band_dir (server path)",
            cfg.path("input", "band_dir") or "")
        if bd:
            cfg.raw.setdefault("input", {})["band_dir"] = bd
            cfg.root = os.path.dirname(os.path.abspath(cfg_path)) \
                if not os.path.isabs(bd) else cfg.root
        return cfg

    # ---- manual field editing -------------------------------------------- #
    st.sidebar.subheader("Input (server-side paths)")
    band_dir = st.sidebar.text_input("Band folder (one TIF per band)", "data/bands")
    band_glob = st.sidebar.text_input("Band glob", "*.tif")
    wl_csv = st.sidebar.text_input("Wavelengths CSV", "data/wavelengths.csv")
    vector = st.sidebar.text_input("Plots vector (GeoJSON/SHP)", "data/plots.geojson")
    class_field = st.sidebar.text_input("Class field", "stress")
    subplot_field = st.sidebar.text_input("Sub-plot id field", "subplot_id")

    st.sidebar.subheader("NDVI canopy mask")
    ndvi_on = st.sidebar.checkbox("Enable NDVI masking", True)
    red_nm = st.sidebar.number_input("RED nm", value=670.0)
    nir_nm = st.sidebar.number_input("NIR nm", value=800.0)
    thr = st.sidebar.slider("NDVI threshold", 0.0, 1.0, 0.4, 0.05)

    st.sidebar.subheader("Sampling & model")
    ppsp = st.sidebar.number_input("Points / sub-plot", 10, 1000, 200, 10)
    top_k = st.sidebar.text_input("Top-k list", "4,7,10,12,15")
    rfe_step = st.sidebar.number_input("RFE step", 1, 50, 1)

    raw = {
        "input": {"band_dir": band_dir, "band_glob": band_glob,
                  "wavelengths_csv": wl_csv, "bands": [], "block_size": 1024},
        "labels": {"vector_file": vector, "class_field": class_field,
                   "subplot_field": subplot_field},
        "ndvi": {"enabled": ndvi_on, "red_nm": red_nm, "nir_nm": nir_nm,
                 "threshold": thr},
        "sampling": {"points_per_subplot": int(ppsp), "random_state": 42},
        "feature_selection": {
            "method": "rfe",
            "top_k_list": [int(x) for x in top_k.split(",") if x.strip()],
            "estimator": "random_forest", "rfe_step": int(rfe_step)},
        "model": {"test_size": 0.2, "random_state": 42, "cv_folds": 3,
                  "random_forest": {"n_estimators_grid": [100, 200, 300, 500]},
                  "svm": {"kernel": "rbf", "C_grid": [0.1, 1, 10, 100],
                          "gamma_grid": [1, 0.1, 0.01, 0.001]}},
        "output": {"dir": "outputs", "samples_csv": "samples.csv",
                   "results_csv": "band_selection_results.csv",
                   "band_ranking_csv": "band_ranking.csv",
                   "model_file": "best_model.joblib",
                   "stress_map": "stress_map.tif"},
    }
    return Config(raw=raw, root=os.path.dirname(os.path.abspath(__file__)))


cfg = build_config()
if cfg is None:
    st.stop()


@st.cache_resource(show_spinner="Opening band stack…")
def _open_stack(band_dir, band_glob, wl_csv, explicit_key):
    return BandStack.from_config(cfg)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_data, tab_ndvi, tab_train, tab_predict = st.tabs(
    ["📂 Data", "🌱 NDVI / Canopy", "🤖 Train", "🗺️ Stress Map"])

# ---- Data ----------------------------------------------------------------- #
with tab_data:
    st.subheader("Band-separated hyperspectral stack")
    try:
        stack = _open_stack(
            cfg.path("input", "band_dir"),
            cfg.get("input", "band_glob"),
            cfg.path("input", "wavelengths_csv"),
            str(cfg.get("input", "bands")),
        )
    except Exception as e:  # noqa
        st.error(f"Could not open band stack: {e}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bands", stack.n_bands)
    c2.metric("Width × Height", f"{stack.width} × {stack.height}")
    c3.metric("Min λ (nm)", f"{stack.wavelengths.min():.0f}")
    c4.metric("Max λ (nm)", f"{stack.wavelengths.max():.0f}")
    st.write(f"**CRS:** {stack.crs}")

    st.markdown("**False-colour preview** (NIR-Red-Green composite)")
    try:
        ni = stack.band_index_for(800)
        ri = stack.band_index_for(670)
        gi = stack.band_index_for(550)
        comp = np.dstack([stack.read_band(ni), stack.read_band(ri),
                          stack.read_band(gi)])
        p2, p98 = np.nanpercentile(comp, [2, 98])
        comp = np.clip((comp - p2) / (p98 - p2 + 1e-9), 0, 1)
        st.image(comp, clamp=True, use_container_width=True)
    except Exception as e:  # noqa
        st.info(f"Preview unavailable: {e}")

# ---- NDVI ----------------------------------------------------------------- #
with tab_ndvi:
    st.subheader("NDVI soil-background removal")
    if not cfg.get("ndvi", "enabled", default=True):
        st.info("NDVI masking is disabled in the configuration.")
    else:
        if st.button("Compute NDVI canopy mask"):
            with st.spinner("Computing NDVI…"):
                ri = stack.band_index_for(cfg.get("ndvi", "red_nm"))
                ni = stack.band_index_for(cfg.get("ndvi", "nir_nm"))
                ndvi = compute_ndvi(stack.read_band(ri), stack.read_band(ni))
                thr = cfg.get("ndvi", "threshold")
                mask = ndvi >= thr
                colA, colB = st.columns(2)
                with colA:
                    fig, ax = plt.subplots()
                    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=1)
                    ax.set_title("NDVI"); ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.046)
                    st.pyplot(fig)
                with colB:
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(mask, cmap="Greens")
                    ax2.set_title(f"Canopy mask (NDVI ≥ {thr})"); ax2.axis("off")
                    st.pyplot(fig2)
                st.metric("Canopy coverage",
                          f"{100*np.nanmean(mask):.1f}%")

# ---- Train ---------------------------------------------------------------- #
with tab_train:
    st.subheader("Sample → RFE → RF/SVM tuning")
    st.write("Runs the full paper pipeline on the configured scene.")
    if st.button("▶ Run pipeline", type="primary"):
        prog = st.empty()
        with st.spinner("Sampling canopy points…"):
            df = run_sampling(cfg)
        prog.success(f"Collected {len(df)} samples "
                     f"across {df['subplot_id'].nunique()} sub-plots.")
        st.session_state["samples"] = df

        # Mean NS vs GFWS spectra
        band_cols = [c for c in df.columns
                     if c.startswith("b") and c[1:5].isdigit()]
        wl = [float(c.split("_")[1][:-2]) for c in band_cols]
        fig, ax = plt.subplots(figsize=(7, 3))
        for label, g in df.groupby("label"):
            ax.plot(wl, g[band_cols].mean().values, label=str(label))
        ax.axvspan(670, 780, color="orange", alpha=0.15,
                   label="670–780 nm (paper)")
        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Mean reflectance")
        ax.legend(); ax.set_title("Mean spectral signature by class")
        st.pyplot(fig)

        with st.spinner("RFE ranking + RF/SVM tuning (this can take a while)…"):
            results = run_modelling(cfg, df)
        st.session_state["results"] = results

        st.success("Done. Best model saved to outputs/.")
        st.dataframe(results, use_container_width=True)

        best = results.loc[results["kappa"].idxmax()]
        st.metric(f"Best: {best['model']} (top-{best['top_k']})",
                  f"Kappa {best['kappa']:.3f}", f"Acc {best['accuracy']:.3f}")
        st.caption(f"Selected wavelengths (nm): {best['wavelengths_nm']}")

        # Show RFE ranking of top bands
        rk_path = os.path.join(cfg.path("output", "dir"),
                               cfg.get("output", "band_ranking_csv"))
        if os.path.exists(rk_path):
            rk = pd.read_csv(rk_path).head(20)
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            ax3.barh(rk["band"][::-1], (1 / rk["rfe_rank"])[::-1], color="seagreen")
            ax3.set_xlabel("Importance (1 / RFE rank)")
            ax3.set_title("Top-20 RFE-ranked bands")
            st.pyplot(fig3)

# ---- Predict -------------------------------------------------------------- #
with tab_predict:
    st.subheader("Wall-to-wall stress map")
    model_path = os.path.join(cfg.path("output", "dir"),
                              cfg.get("output", "model_file"))
    if not os.path.exists(model_path):
        st.info("Train a model first (Train tab).")
    elif st.button("Generate stress map"):
        import joblib
        from aquaspectra.predict import predict_map

        bundle = joblib.load(model_path)
        out_path = os.path.join(cfg.path("output", "dir"),
                                cfg.get("output", "stress_map"))
        ndvi_on = cfg.get("ndvi", "enabled", default=True)
        with st.spinner("Classifying every canopy pixel…"):
            predict_map(
                stack, bundle["model"], bundle["band_indices"],
                bundle["label_encoder"], out_path,
                canopy_red_nm=cfg.get("ndvi", "red_nm") if ndvi_on else None,
                canopy_nir_nm=cfg.get("ndvi", "nir_nm") if ndvi_on else None,
                canopy_threshold=cfg.get("ndvi", "threshold") if ndvi_on else None,
                block_size=cfg.get("input", "block_size", default=1024),
            )
        import rasterio
        with rasterio.open(out_path) as ds:
            arr = ds.read(1)
        classes = list(bundle["label_encoder"].classes_)
        fig, ax = plt.subplots()
        im = ax.imshow(arr, cmap="RdYlGn_r", vmin=0, vmax=len(classes))
        ax.set_title("Stress map (0=soil)"); ax.axis("off")
        st.pyplot(fig)
        st.caption("Pixel values: " +
                   ", ".join(f"{c}={i+1}" for i, c in enumerate(classes)) +
                   ", soil/nodata=0")
        st.success(f"Saved {out_path}")
