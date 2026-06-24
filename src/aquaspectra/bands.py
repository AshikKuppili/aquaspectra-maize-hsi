"""Loading band-separated hyperspectral TIFs into an analysis-ready stack.

The drone deliverable is **one (large) GeoTIFF per spectral band**. This module
discovers those files, maps each to a wavelength, validates that they share the
same grid (CRS / transform / shape), and exposes convenient readers:

  * `read_band(index)`           -> full 2D array for one band
  * `read_window(window)`        -> (n_bands, h, w) cube for a sub-window
  * `read_at(rows, cols)`        -> (n_points, n_bands) spectra at pixel coords

Reading is windowed so we never load every full-resolution band into memory at
once (the files are large).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


def _natural_key(s: str):
    """Sort 'band_2.tif' before 'band_10.tif'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# Matches a wavelength embedded in a filename, e.g.:
#   Band_071_521.422nm.tif  -> 521.422
#   B12_730nm.tif           -> 730
#   band_055_700.0_nm.tif   -> 700.0
_WL_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)\s*_?nm", re.IGNORECASE)


def wavelength_from_filename(path: str) -> float | None:
    """Parse a wavelength (nm) embedded in a band filename, or None."""
    m = _WL_IN_NAME.search(os.path.basename(path))
    return float(m.group(1)) if m else None


@dataclass
class BandRef:
    path: str
    wavelength_nm: float


class BandStack:
    """A stack of co-registered single-band GeoTIFFs."""

    def __init__(self, bands: list[BandRef]):
        if not bands:
            raise ValueError("No bands provided to BandStack.")
        self.bands = bands
        self.wavelengths = np.array([b.wavelength_nm for b in bands], dtype=float)

        # Validate common grid using the first band as reference.
        with rasterio.open(bands[0].path) as ref:
            self.crs = ref.crs
            self.transform = ref.transform
            self.height = ref.height
            self.width = ref.width
            self.dtype = ref.dtypes[0]
            self.nodata = ref.nodata
            self.profile = ref.profile.copy()

        for b in bands[1:]:
            with rasterio.open(b.path) as ds:
                if (ds.height, ds.width) != (self.height, self.width):
                    raise ValueError(
                        f"Band '{b.path}' shape {(ds.height, ds.width)} != "
                        f"reference {(self.height, self.width)}. "
                        "All band TIFs must share the same grid."
                    )

    # --------------------------------------------------------------- factories
    @classmethod
    def from_config(cls, cfg) -> "BandStack":
        explicit = cfg.get("input", "bands", default=[]) or []
        if explicit:
            band_dir = cfg.path("input", "band_dir") or cfg.root
            refs = [
                BandRef(
                    path=os.path.join(band_dir, item["file"])
                    if not os.path.isabs(item["file"])
                    else item["file"],
                    wavelength_nm=float(item["wavelength_nm"]),
                )
                for item in explicit
            ]
            return cls(refs)

        # Discover files via glob, map to wavelengths from CSV (or band index).
        band_dir = cfg.path("input", "band_dir")
        pattern = cfg.get("input", "band_glob", default="*.tif")
        files = sorted(glob.glob(os.path.join(band_dir, pattern)), key=_natural_key)
        if not files:
            raise FileNotFoundError(
                f"No band files matching '{pattern}' in '{band_dir}'."
            )

        wl_csv = cfg.path("input", "wavelengths_csv")
        if wl_csv and os.path.exists(wl_csv):
            wl = pd.read_csv(wl_csv)["wavelength_nm"].astype(float).tolist()
            if len(wl) != len(files):
                raise ValueError(
                    f"wavelengths_csv has {len(wl)} rows but {len(files)} "
                    "band files were found; counts must match."
                )
        else:
            # Try parsing wavelengths embedded in filenames (e.g.
            # 'Band_071_521.422nm.tif'); fall back to 1-based band indices.
            parsed = [wavelength_from_filename(f) for f in files]
            if all(p is not None for p in parsed):
                wl = parsed
            else:
                wl = list(range(1, len(files) + 1))

        refs = [BandRef(path=f, wavelength_nm=float(w)) for f, w in zip(files, wl)]
        return cls(refs)

    # ----------------------------------------------------------------- readers
    @property
    def n_bands(self) -> int:
        return len(self.bands)

    def band_index_for(self, wavelength_nm: float) -> int:
        """Index of the band whose wavelength is closest to the target."""
        return int(np.argmin(np.abs(self.wavelengths - wavelength_nm)))

    def read_band(self, index: int, window: Window | None = None) -> np.ndarray:
        with rasterio.open(self.bands[index].path) as ds:
            arr = ds.read(1, window=window, masked=True)
        return arr.astype("float32").filled(np.nan)

    def read_window(self, window: Window | None = None) -> np.ndarray:
        """Return (n_bands, h, w) for the given window (or full scene)."""
        arrays = [self.read_band(i, window=window) for i in range(self.n_bands)]
        return np.stack(arrays, axis=0)

    def read_at(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Return (n_points, n_bands) reflectance spectra at pixel coordinates."""
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        out = np.empty((rows.size, self.n_bands), dtype="float32")
        for i in range(self.n_bands):
            with rasterio.open(self.bands[i].path) as ds:
                full = ds.read(1, masked=True).astype("float32").filled(np.nan)
            out[:, i] = full[rows, cols]
        return out

    def iter_blocks(self, block_size: int = 1024):
        """Yield Windows covering the whole scene in `block_size` tiles."""
        for r in range(0, self.height, block_size):
            for c in range(0, self.width, block_size):
                h = min(block_size, self.height - r)
                w = min(block_size, self.width - c)
                yield Window(col_off=c, row_off=r, width=w, height=h)
