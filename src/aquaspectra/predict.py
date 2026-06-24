"""Generate a wall-to-wall stress classification map from a trained model."""
from __future__ import annotations

import numpy as np
import rasterio

from .bands import BandStack


def predict_map(
    stack: BandStack,
    model,
    band_indices: list[int],
    label_encoder,
    out_path: str,
    canopy_red_nm: float | None = None,
    canopy_nir_nm: float | None = None,
    canopy_threshold: float | None = None,
    block_size: int = 1024,
) -> str:
    """Classify every canopy pixel and write a single-band GeoTIFF.

    Output values: encoded class id + 1 (so 0 can be reserved for nodata/soil).
    """
    from .ndvi import compute_ndvi

    profile = stack.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")

    do_canopy = (
        canopy_red_nm is not None
        and canopy_nir_nm is not None
        and canopy_threshold is not None
    )
    ri = stack.band_index_for(canopy_red_nm) if do_canopy else None
    ni = stack.band_index_for(canopy_nir_nm) if do_canopy else None

    with rasterio.open(out_path, "w", **profile) as dst:
        for win in stack.iter_blocks(block_size):
            cube = stack.read_window(win)  # (n_bands, h, w)
            nb, h, w = cube.shape
            flat = cube.reshape(nb, -1).T  # (h*w, n_bands)

            valid = ~np.isnan(flat).any(axis=1)
            if do_canopy:
                ndvi = compute_ndvi(cube[ri], cube[ni]).reshape(-1)
                valid &= ndvi >= canopy_threshold

            out = np.zeros(flat.shape[0], dtype="uint8")
            if valid.any():
                X = flat[valid][:, band_indices]
                enc = np.asarray(model.predict(X))  # encoded class ids
                out[valid] = (enc + 1).astype("uint8")

            dst.write(out.reshape(h, w), 1, window=win)

    return out_path
