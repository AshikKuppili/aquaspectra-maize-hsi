"""Sample canopy pixels per sub-plot and extract their spectra.

Paper: 200 random canopy points are extracted from each of the 66 sub-plots in
both the NS and GFWS sections (66 x 2 x 200 = 26,400 samples).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import geometry_mask

from .bands import BandStack


def _plot_pixel_mask(stack: BandStack, geom) -> np.ndarray:
    """Boolean mask (True inside the polygon) over the full raster grid."""
    return ~geometry_mask(
        [geom],
        out_shape=(stack.height, stack.width),
        transform=stack.transform,
        invert=False,
    )


def sample_points(
    stack: BandStack,
    plots_path: str,
    class_field: str,
    subplot_field: str | None,
    canopy: np.ndarray | None,
    points_per_subplot: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of sampled spectra with labels.

    Columns: subplot_id, label, b0000_<wl>nm ... plus row/col for reference.
    """
    gdf = gpd.read_file(plots_path)
    if stack.crs is not None and gdf.crs is not None and gdf.crs != stack.crs:
        gdf = gdf.to_crs(stack.crs)

    rng = np.random.default_rng(random_state)
    wl = stack.wavelengths
    band_cols = [f"b{i:04d}_{wl[i]:.0f}nm" for i in range(stack.n_bands)]

    rows_acc, cols_acc, labels_acc, ids_acc = [], [], [], []

    for idx, feat in gdf.iterrows():
        label = feat[class_field]
        sub_id = feat[subplot_field] if subplot_field and subplot_field in gdf.columns else idx

        in_plot = _plot_pixel_mask(stack, feat.geometry)
        if canopy is not None:
            in_plot &= canopy

        rr, cc = np.where(in_plot)
        if rr.size == 0:
            continue

        n = min(points_per_subplot, rr.size)
        pick = rng.choice(rr.size, size=n, replace=False)
        rows_acc.append(rr[pick])
        cols_acc.append(cc[pick])
        labels_acc.extend([label] * n)
        ids_acc.extend([sub_id] * n)

    if not rows_acc:
        raise RuntimeError(
            "No samples collected. Check that plot polygons overlap the raster "
            "and that the NDVI canopy mask is not filtering out everything."
        )

    rows = np.concatenate(rows_acc)
    cols = np.concatenate(cols_acc)
    spectra = stack.read_at(rows, cols)

    df = pd.DataFrame(spectra, columns=band_cols)
    df.insert(0, "row", rows)
    df.insert(1, "col", cols)
    df.insert(0, "subplot_id", ids_acc)
    df.insert(1, "label", labels_acc)

    # Drop rows with any NaN spectra (e.g., nodata pixels).
    df = df.dropna(subset=band_cols).reset_index(drop=True)
    return df
