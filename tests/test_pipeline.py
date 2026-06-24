"""Smoke tests for the aquaspectra pipeline using a tiny synthetic scene."""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box

from aquaspectra.config import load_config
from aquaspectra.bands import BandStack
from aquaspectra.ndvi import compute_ndvi, canopy_mask
from aquaspectra.features import rank_bands


def _build_scene(tmpdir):
    n_bands, h, w = 8, 20, 30
    wl = np.linspace(450, 950, n_bands)
    band_dir = os.path.join(tmpdir, "bands")
    os.makedirs(band_dir, exist_ok=True)
    transform = from_origin(0, 0, 1, 1)
    profile = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32",
                   crs="EPSG:32643", transform=transform)
    rng = np.random.default_rng(0)
    cube = rng.uniform(0.1, 0.5, (n_bands, h, w)).astype("float32")
    # make right half "stressed": lower NIR
    cube[5:, :, w // 2:] -= 0.2
    for i in range(n_bands):
        with rasterio.open(os.path.join(band_dir, f"band_{i+1:02d}.tif"),
                           "w", **profile) as ds:
            ds.write(cube[i], 1)
    pd.DataFrame({"wavelength_nm": wl}).to_csv(
        os.path.join(tmpdir, "wavelengths.csv"), index=False)
    return band_dir, cube, wl


def test_bandstack_loads_and_reads(tmp_path):
    band_dir, cube, wl = _build_scene(str(tmp_path))
    refs_glob = os.path.join(band_dir, "*.tif")
    import glob
    from aquaspectra.bands import BandRef
    files = sorted(glob.glob(refs_glob))
    stack = BandStack([BandRef(f, float(w)) for f, w in zip(files, wl)])
    assert stack.n_bands == 8
    assert stack.read_band(0).shape == (20, 30)
    # closest-band lookup
    assert stack.band_index_for(950) == 7


def test_ndvi_and_mask(tmp_path):
    red = np.array([[0.1, 0.2]], dtype="float32")
    nir = np.array([[0.5, 0.2]], dtype="float32")
    ndvi = compute_ndvi(red, nir)
    assert np.isclose(ndvi[0, 0], (0.5 - 0.1) / (0.5 + 0.1))
    assert np.isclose(ndvi[0, 1], 0.0)


def test_rfe_ranks_all_bands(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 8))
    # band 3 carries the signal
    y = (X[:, 3] > 0).astype(int)
    cols = [f"b{i:02d}" for i in range(8)]
    ranking = rank_bands(X, y, cols, step=1)
    assert set(ranking["rfe_rank"]) == set(range(1, 9))
    # the informative band should rank best
    assert ranking.iloc[0]["band"] == "b03"
