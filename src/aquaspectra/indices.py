"""Label-free water-stress indices (no training required).

When ground-truth plot labels are NOT available, the paper's supervised RF/SVM
cannot be trained. These spectral indices are physics-based proxies that work on
a single plot or the whole scene:

  NDRE  = (NIR - RedEdge) / (NIR + RedEdge)
          chlorophyll / vigour; declines under stress (higher = healthier).
  WBR   = NIR / WaterAbsorption  (e.g. 780nm / 936nm)
          leaf water content. The ~930-970nm band ABSORBS more (reflects less)
          when the canopy is well-watered, so its reflectance drops and the ratio
          RISES. Hence a HIGHER WBR indicates MORE water (LESS stress).

A combined relative stress score is built by z-normalising both indices over the
valid canopy and averaging. Both indices are "higher = healthier", so:
    stress = -(z(NDRE) + z(WBR)) / 2     (low NDRE + low WBR  ->  higher stress).
"""
from __future__ import annotations

import os

import numpy as np

from .bands import BandStack
from .ndvi import compute_ndvi


def _z(a: np.ndarray) -> np.ndarray:
    m = np.nanmean(a)
    s = np.nanstd(a)
    return (a - m) / s if s > 0 else np.zeros_like(a)


def ndre(rededge: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + rededge
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (nir - rededge) / denom
    out[denom == 0] = np.nan
    return out


def water_band_ratio(nir: np.ndarray, water: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = nir / water
    out[(water == 0) | ~np.isfinite(out)] = np.nan
    return out


def compute_stress_index(stack: BandStack, cfg, window=None):
    """Return dict of arrays: ndvi, ndre, wbr, stress (0-1), canopy mask."""
    red_nm = cfg.get("ndvi", "red_nm")
    nir_nm = cfg.get("ndvi", "nir_nm")
    thr = cfg.get("ndvi", "threshold", default=0.3)
    re_nm = cfg.get("indices", "rededge_nm", default=706.0)
    water_nm = cfg.get("indices", "water_nm", default=936.0)

    red = stack.read_band(stack.band_index_for(red_nm), window=window)
    nir = stack.read_band(stack.band_index_for(nir_nm), window=window)
    rededge = stack.read_band(stack.band_index_for(re_nm), window=window)
    water = stack.read_band(stack.band_index_for(water_nm), window=window)

    ndvi = compute_ndvi(red, nir)
    canopy = np.isfinite(ndvi) & (ndvi >= thr)

    nd = ndre(rededge, nir)
    wb = water_band_ratio(nir, water)
    nd_c = np.where(canopy, nd, np.nan)
    wb_c = np.where(canopy, wb, np.nan)

    # Both NDRE and WBR are "higher = healthier" -> stress is the negated mean.
    # higher stress = low vigour (low NDRE) + low water (low WBR)
    score = -(_z(nd_c) + _z(wb_c)) / 2.0
    # squash to 0-1 across canopy via percentile clip
    valid = score[np.isfinite(score)]
    if valid.size:
        lo, hi = np.nanpercentile(valid, [2, 98])
        stress = np.clip((score - lo) / (hi - lo + 1e-9), 0, 1)
    else:
        stress = score
    stress[~canopy] = np.nan

    return {"ndvi": ndvi, "ndre": nd_c, "wbr": wb_c,
            "stress": stress, "canopy": canopy}


def ndwi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Gao-1996 style canopy water index: (NIR - SWIR)/(NIR + SWIR).

    Mathematically bounded to [-1, 1]; tiny denominators (shadow/edge pixels)
    are masked to avoid numerical blow-up.
    """
    denom = nir + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (nir - swir) / denom
    out[(np.abs(denom) < 0.02) | ~np.isfinite(out)] = np.nan
    out[(out < -1) | (out > 1)] = np.nan
    return out


def write_moisture_map(stack: BandStack, cfg, out_tif: str,
                       block_size: int = 2048) -> dict:
    """Resample a SWIR band onto the reference grid and write an NDWI GeoTIFF.

    SWIR (~2300-2400 nm) is the strongest optical region for liquid water, but
    the SWIR bands sit on a coarser grid, so they are reprojected (WarpedVRT)
    onto the reference (VNIR) grid before the index is computed tile-by-tile.
    Returns summary stats over canopy pixels.
    """
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling

    swir_file = cfg.path("moisture", "swir_file")
    if not swir_file or not os.path.exists(swir_file):
        raise FileNotFoundError(
            f"moisture.swir_file not found: {swir_file!r}. "
            "Point it at a SWIR band GeoTIFF.")

    nir_nm = cfg.get("moisture", "nir_nm", default=cfg.get("ndvi", "nir_nm"))
    red_nm = cfg.get("ndvi", "red_nm")
    thr = cfg.get("ndvi", "threshold", default=0.3)
    ni = stack.band_index_for(nir_nm)
    ri = stack.band_index_for(red_nm)

    profile = stack.profile.copy()
    profile.update(count=1, dtype="float32", nodata=np.nan,
                   compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256)

    csum = 0.0; npx = 0; fmin, fmax = np.inf, -np.inf

    src = rasterio.open(swir_file)
    vrt = WarpedVRT(src, crs=stack.crs, transform=stack.transform,
                    width=stack.width, height=stack.height,
                    resampling=Resampling.bilinear)
    try:
        with rasterio.open(out_tif, "w", **profile) as dst:
            for win in stack.iter_blocks(block_size):
                red = stack.read_band(ri, window=win)
                nir = stack.read_band(ni, window=win)
                swir = vrt.read(1, window=win).astype("float32")
                swir[swir == 0] = np.nan
                ndvi_b = compute_ndvi(red, nir)
                canopy = np.isfinite(ndvi_b) & (ndvi_b >= thr)
                idx = ndwi(nir, swir)
                idx[~canopy] = np.nan
                dst.write(idx.astype("float32"), 1, window=win)

                v = idx[np.isfinite(idx)]
                if v.size:
                    csum += float(v.sum()); npx += v.size
                    fmin = min(fmin, float(v.min()))
                    fmax = max(fmax, float(v.max()))
    finally:
        vrt.close(); src.close()

    return {
        "ndwi_min": fmin if npx else float("nan"),
        "ndwi_mean": (csum / npx) if npx else float("nan"),
        "ndwi_max": fmax if npx else float("nan"),
        "valid_pixels": npx,
    }
