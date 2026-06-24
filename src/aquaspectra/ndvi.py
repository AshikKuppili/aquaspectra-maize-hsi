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
