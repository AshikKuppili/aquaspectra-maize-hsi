"""NDVI-based canopy masking (soil background removal).

Paper: soil background is removed using an NDVI threshold; analysis is then
performed only over maize canopy pixels.

    NDVI = (NIR - RED) / (NIR + RED)
"""
from __future__ import annotations

import numpy as np

from .bands import BandStack


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red = red.astype("float32")
    nir = nir.astype("float32")
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / denom
    ndvi[denom == 0] = np.nan
    return ndvi


def canopy_mask(stack: BandStack, red_nm: float, nir_nm: float,
                threshold: float) -> np.ndarray:
    """Boolean mask (True = canopy) for the full scene."""
    ri = stack.band_index_for(red_nm)
    ni = stack.band_index_for(nir_nm)
    red = stack.read_band(ri)
    nir = stack.read_band(ni)
    ndvi = compute_ndvi(red, nir)
    mask = ndvi >= threshold
    mask[np.isnan(ndvi)] = False
    return mask


def write_ndvi_map(stack: BandStack, red_nm: float, nir_nm: float,
                   threshold: float, out_tif: str,
                   block_size: int = 2048) -> dict:
    """Compute NDVI tile-by-tile and write a float32 GeoTIFF.

    Returns summary stats {ndvi_min, ndvi_mean, ndvi_max, canopy_fraction}.
    Memory-safe for very large rasters (reads/writes in `block_size` tiles).
    """
    import rasterio

    ri = stack.band_index_for(red_nm)
    ni = stack.band_index_for(nir_nm)

    profile = stack.profile.copy()
    profile.update(count=1, dtype="float32", nodata=np.nan,
                   compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256)

    csum = cmin = cmax = 0.0
    npx = ncanopy = 0
    fmin, fmax = np.inf, -np.inf

    with rasterio.open(out_tif, "w", **profile) as dst:
        for win in stack.iter_blocks(block_size):
            red = stack.read_band(ri, window=win)
            nir = stack.read_band(ni, window=win)
            # treat 0.0 mosaic padding as nodata
            bg = (red == 0) & (nir == 0)
            ndvi = compute_ndvi(red, nir)
            ndvi[bg] = np.nan
            dst.write(ndvi.astype("float32"), 1, window=win)

            v = ndvi[np.isfinite(ndvi)]
            if v.size:
                csum += float(v.sum()); npx += v.size
                fmin = min(fmin, float(v.min()))
                fmax = max(fmax, float(v.max()))
                ncanopy += int((v >= threshold).sum())

    return {
        "ndvi_min": fmin if npx else float("nan"),
        "ndvi_mean": (csum / npx) if npx else float("nan"),
        "ndvi_max": fmax if npx else float("nan"),
        "canopy_fraction": (ncanopy / npx) if npx else 0.0,
        "valid_pixels": npx,
    }
