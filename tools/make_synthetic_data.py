"""Generate a small synthetic hyperspectral dataset for testing the pipeline.

Creates:
  * data/bands/band_XXX.tif   -- one GeoTIFF per band (band-separated, like the
                                 real drone deliverable)
  * data/wavelengths.csv      -- wavelength (nm) for each band, in file order
  * data/plots.geojson        -- sub-plot polygons labelled NS / GFWS

The synthetic reflectance encodes a realistic NS-vs-GFWS difference in the
670-780 nm region so that the RFE/SVM pipeline can recover meaningful bands.
Run:  python tools/make_synthetic_data.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
BANDS = os.path.join(DATA, "bands")

# ---- scene geometry --------------------------------------------------------
N_BANDS = 126
WL = np.linspace(450, 950, N_BANDS)        # 450-950 nm, like the paper
HEIGHT, WIDTH = 140, 220                    # small test grid
RES = 0.05                                  # 5 cm pixels
ORIGIN_X, ORIGIN_Y = 500000.0, 2000000.0    # arbitrary projected origin (UTM-ish)
CRS = "EPSG:32643"                          # UTM 43N (Hyderabad region)
TRANSFORM = from_origin(ORIGIN_X, ORIGIN_Y, RES, RES)

# Field = 2 sections (NS left, GFWS right), each 7 rows x 11 cols sub-plots.
ROWS, COLS = 7, 11
rng = np.random.default_rng(7)


def base_spectrum(stress: bool) -> np.ndarray:
    """A canopy-like reflectance curve; stress lowers RE/NIR reflectance."""
    green = 0.10 * np.exp(-((WL - 550) ** 2) / (2 * 30 ** 2))
    red_dip = -0.05 * np.exp(-((WL - 680) ** 2) / (2 * 15 ** 2))
    nir_plateau = 0.45 / (1 + np.exp(-(WL - 730) / 12))
    water_abs = -0.08 * np.exp(-((WL - 935) ** 2) / (2 * 8 ** 2))
    spec = 0.05 + green + red_dip + nir_plateau + water_abs
    if stress:
        # Stress: reduced RE-NIR reflectance (670-780) + slightly deeper red.
        spec -= 0.12 / (1 + np.exp(-(WL - 725) / 10))
        spec += -0.02 * np.exp(-((WL - 680) ** 2) / (2 * 15 ** 2))
    return spec


def main():
    os.makedirs(BANDS, exist_ok=True)

    sec_w = WIDTH // 2
    cube = np.zeros((N_BANDS, HEIGHT, WIDTH), dtype="float32")
    polys = []

    pad = 4
    ph = (HEIGHT - 2 * pad) / ROWS
    pw = (sec_w - 2 * pad) / COLS

    for section, stress in ((0, False), (1, True)):
        x0 = section * sec_w
        for r in range(ROWS):
            for c in range(COLS):
                r0 = int(pad + r * ph)
                r1 = int(pad + (r + 1) * ph) - 1
                c0 = int(x0 + pad + c * pw)
                c1 = int(x0 + pad + (c + 1) * pw) - 1
                spec = base_spectrum(stress)[:, None, None]
                noise = rng.normal(0, 0.01, (N_BANDS, r1 - r0, c1 - c0)).astype("float32")
                cube[:, r0:r1, c0:c1] = spec + noise

                wx0 = ORIGIN_X + c0 * RES
                wy0 = ORIGIN_Y - r0 * RES
                wx1 = ORIGIN_X + c1 * RES
                wy1 = ORIGIN_Y - r1 * RES
                polys.append({
                    "geometry": box(min(wx0, wx1), min(wy0, wy1),
                                    max(wx0, wx1), max(wy0, wy1)),
                    "stress": "GFWS" if stress else "NS",
                    "subplot_id": f"{'S' if stress else 'C'}_{r}_{c}",
                })

    # Add soil background (low NDVI) in the gaps so NDVI masking has work to do.
    soil = np.linspace(0.18, 0.20, N_BANDS).astype("float32")[:, None]
    empty = cube.sum(axis=0) == 0
    cube[:, empty] = soil + rng.normal(
        0, 0.005, (N_BANDS, int(empty.sum()))).astype("float32")
    cube = np.clip(cube, 0, 1)

    profile = dict(driver="GTiff", height=HEIGHT, width=WIDTH, count=1,
                   dtype="float32", crs=CRS, transform=TRANSFORM, compress="lzw")
    for i in range(N_BANDS):
        path = os.path.join(BANDS, f"band_{i+1:03d}.tif")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(cube[i], 1)

    pd.DataFrame({"wavelength_nm": WL}).to_csv(
        os.path.join(DATA, "wavelengths.csv"), index=False)

    gdf = gpd.GeoDataFrame(polys, crs=CRS)
    gdf.to_file(os.path.join(DATA, "plots.geojson"), driver="GeoJSON")

    print(f"Wrote {N_BANDS} band TIFs to {BANDS}")
    print(f"Wrote wavelengths.csv and plots.geojson ({len(gdf)} sub-plots)")


if __name__ == "__main__":
    main()
