"""Command-line interface for the maize water-stress detection pipeline.

Examples
--------
  # 1. Inspect band stack (counts, wavelengths, grid)
  python -m aquaspectra.cli info --config config.yaml

  # 2. Sample canopy points only -> outputs/samples.csv
  python -m aquaspectra.cli sample --config config.yaml

  # 3. Full run: sample + RFE + RF/SVM tuning + save best model
  python -m aquaspectra.cli run --config config.yaml

  # 4. Predict a wall-to-wall stress map with the saved best model
  python -m aquaspectra.cli predict --config config.yaml

  # 5. NDVI map / soil-canopy mask (no labels required)
  python -m aquaspectra.cli ndvi --config config.yaml

  # 6. SWIR canopy-moisture (NDWI) map (no labels required)
  python -m aquaspectra.cli moisture --config config.yaml

  # 7. Assemble a multi-page PDF report from the outputs
  python -m aquaspectra.cli report --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib

from .bands import BandStack
from .config import load_config
from .pipeline import run_all, run_modelling, run_sampling


def _cmd_info(cfg):
    stack = BandStack.from_config(cfg)
    print(f"Bands      : {stack.n_bands}")
    print(f"Grid       : {stack.height} x {stack.width}")
    print(f"CRS        : {stack.crs}")
    print(f"Wavelengths: {stack.wavelengths.min():.1f} - "
          f"{stack.wavelengths.max():.1f} nm")
    print("First 5    : " + ", ".join(
        f"{b.wavelength_nm:.0f}nm={os.path.basename(b.path)}"
        for b in stack.bands[:5]))


def _cmd_sample(cfg):
    run_sampling(cfg)


def _cmd_run(cfg):
    run_all(cfg)


def _cmd_predict(cfg):
    import pandas as pd  # noqa
    from .predict import predict_map

    out_dir = cfg.path("output", "dir")
    bundle = joblib.load(os.path.join(out_dir, cfg.get("output", "model_file")))
    stack = BandStack.from_config(cfg)
    out_path = os.path.join(out_dir, cfg.get("output", "stress_map"))

    ndvi_on = cfg.get("ndvi", "enabled", default=True)
    predict_map(
        stack,
        bundle["model"],
        bundle["band_indices"],
        bundle["label_encoder"],
        out_path,
        canopy_red_nm=cfg.get("ndvi", "red_nm") if ndvi_on else None,
        canopy_nir_nm=cfg.get("ndvi", "nir_nm") if ndvi_on else None,
        canopy_threshold=cfg.get("ndvi", "threshold") if ndvi_on else None,
        block_size=cfg.get("input", "block_size", default=1024),
    )
    print(f"[predict] wrote {out_path}")
    print(f"          class -> pixel value: "
          + ", ".join(f"{c}={i+1}" for i, c in
                      enumerate(bundle['label_encoder'].classes_)))


def _cmd_report(cfg):
    from .report import build_report
    stack = BandStack.from_config(cfg)
    meta = {
        "n_bands": stack.n_bands,
        "wl_range": f"{stack.wavelengths.min():.1f} - {stack.wavelengths.max():.1f} nm",
        "grid": f"{stack.width} x {stack.height}",
        "crs": stack.crs,
    }
    out = build_report(cfg, stack_meta=meta)
    print(f"[report] wrote {out}")


def _cmd_moisture(cfg):
    from .indices import write_moisture_map
    stack = BandStack.from_config(cfg)
    out_dir = cfg.path("output", "dir")
    os.makedirs(out_dir, exist_ok=True)
    out_tif = os.path.join(out_dir, cfg.get("output", "moisture_map",
                                             default="ndwi.tif"))
    stats = write_moisture_map(
        stack, cfg, out_tif,
        block_size=cfg.get("input", "block_size", default=2048),
    )
    print(f"[moisture] wrote {out_tif}")
    print(f"           NDWI min/mean/max : {stats['ndwi_min']:.3f} / "
          f"{stats['ndwi_mean']:.3f} / {stats['ndwi_max']:.3f}")
    print(f"           valid canopy px   : {stats['valid_pixels']:,}")
    print("           (higher NDWI = wetter canopy; lower = drier/moisture stress)")


def _cmd_ndvi(cfg):
    from .ndvi import write_ndvi_map
    stack = BandStack.from_config(cfg)
    out_dir = cfg.path("output", "dir")
    os.makedirs(out_dir, exist_ok=True)
    out_tif = os.path.join(out_dir, cfg.get("output", "ndvi_map",
                                             default="ndvi.tif"))
    stats = write_ndvi_map(
        stack,
        red_nm=cfg.get("ndvi", "red_nm"),
        nir_nm=cfg.get("ndvi", "nir_nm"),
        threshold=cfg.get("ndvi", "threshold", default=0.3),
        out_tif=out_tif,
        block_size=cfg.get("input", "block_size", default=2048),
    )
    print(f"[ndvi] wrote {out_tif}")
    print(f"       NDVI min/mean/max : {stats['ndvi_min']:.3f} / "
          f"{stats['ndvi_mean']:.3f} / {stats['ndvi_max']:.3f}")
    print(f"       canopy fraction   : {100*stats['canopy_fraction']:.1f}% "
          f"(NDVI >= {cfg.get('ndvi', 'threshold', default=0.3)})")
    print(f"       valid pixels      : {stats['valid_pixels']:,}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aquaspectra", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("info", "ndvi", "moisture", "sample", "run", "predict", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", "-c", default="config.yaml")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    dispatch = {
        "info": _cmd_info,
        "ndvi": _cmd_ndvi,
        "moisture": _cmd_moisture,
        "sample": _cmd_sample,
        "run": _cmd_run,
        "predict": _cmd_predict,
        "report": _cmd_report,
    }
    dispatch[args.command](cfg)


if __name__ == "__main__":
    sys.exit(main())
