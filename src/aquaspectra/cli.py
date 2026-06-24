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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aquaspectra", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("info", "sample", "run", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--config", "-c", default="config.yaml")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    dispatch = {
        "info": _cmd_info,
        "sample": _cmd_sample,
        "run": _cmd_run,
        "predict": _cmd_predict,
    }
    dispatch[args.command](cfg)


if __name__ == "__main__":
    sys.exit(main())
