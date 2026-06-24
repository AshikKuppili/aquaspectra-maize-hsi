# 🌽 AquaSpectra

**Drone hyperspectral detection of grain-fill water stress in maize.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Paper DOI](https://img.shields.io/badge/Paper-10.1109%2FIGARSS46834.2022.9884686-b31b1b.svg)](https://doi.org/10.1109/IGARSS46834.2022.9884686)
[![Built with scikit-learn](https://img.shields.io/badge/built%20with-scikit--learn-f89939.svg)](https://scikit-learn.org/)

AquaSpectra is an open-source, reproducible implementation of the **IGARSS 2022**
methodology for detecting **grain-fill water stress (GFWS)** vs **no-stress (NS)**
in maize from a drone hyperspectral cube (450–950 nm, 126 bands). It performs
NDVI canopy masking, **Recursive Feature Elimination (RFE)** band selection, and
**Random Forest / SVM** classification — and identifies the minimal set of
wavelengths needed for reliable water-stress detection.

> **Reference paper.** This project independently reimplements the method from:
> Mohite *et al.*, *"Detection of Crop Water Stress in Maize using Drone Based
> Hyperspectral Imaging"*, IGARSS 2022, IEEE.
> [doi:10.1109/IGARSS46834.2022.9884686](https://doi.org/10.1109/IGARSS46834.2022.9884686).
> There is **no official code release** for that paper; AquaSpectra is an
> independent, from-scratch implementation built on `rasterio`, `geopandas`, and
> `scikit-learn`. See [Citation](#-citation).

---

## 🧭 What problem does this solve? (plain English)

Maize (corn) is very sensitive to a lack of water, especially during the
**grain-fill** stage when kernels are forming. If water stress goes unnoticed,
yield drops sharply. Traditional ways to detect it — soil moisture sensors, or
manually measuring leaf water — are slow, cover only a few spots, and don't scale
across a whole field.

When a plant is water-stressed, the **way its leaves reflect light changes**
(often before any visible wilting). A special **hyperspectral camera** flown on a
**drone** measures reflected light in 126 narrow colour bands from 450 nm
(blue/visible) to 950 nm (near-infrared) — far more detail than a normal RGB
camera's 3 bands. Those subtle reflectance changes are the fingerprint of stress.

**AquaSpectra reads that drone imagery and answers, for every patch of crop:
"is this plant water-stressed or not?"** — and, importantly, it figures out the
*few* light bands that matter most, so future surveys can be cheaper and faster.

## 🔍 How AquaSpectra works (step by step, in plain English)

1. **Load the images.** The drone delivers one large image file *per colour band*
   (126 files). AquaSpectra stacks them into a single "data cube" where every
   pixel has a full 126-value light spectrum.
2. **Ignore the soil.** Not every pixel is a leaf — some are bare ground. Using
   **NDVI** (a simple "greenness" formula), AquaSpectra keeps only the green
   canopy pixels and throws away soil.
3. **Collect labelled examples.** Across the field's sub-plots — some known to be
   healthy (**NS**), some deliberately water-stressed (**GFWS**) — it samples
   ~200 random canopy pixels each, giving thousands of labelled spectra.
4. **Find the bands that matter.** **RFE** ranks all 126 bands from most to least
   useful for telling stressed vs healthy apart.
5. **Train a classifier.** Two machine-learning models (**Random Forest** and
   **SVM**) learn the difference, and we measure how good they are with a score
   called **Kappa** (higher = better; 1.0 is perfect).
6. **Map the whole field.** The best model is applied to every canopy pixel to
   produce a colour-coded **stress map**.

**Key takeaway from the research:** you don't need all 126 bands — about
**10 bands in the 670–780 nm range** (red / red-edge / near-infrared) are enough
to detect maize water stress reliably.

---

## 🌾 How this helps agriculture researchers

AquaSpectra turns raw drone hyperspectral flights into actionable crop-stress
insight — without writing code. It is useful for:

- **💧 Precision irrigation.** Spot water stress *early* (before visible wilting)
  and exactly *where* it occurs, so water is applied only where needed — saving
  water and protecting yield during the sensitive grain-fill stage.
- **🧬 Variety screening & phenotyping.** Compare how dozens of maize varieties
  (sub-plots) respond to water stress, helping breeders identify
  drought-tolerant lines objectively and at scale.
- **💸 Cheaper future surveys.** By pinpointing the *handful* of wavelengths
  (≈670–780 nm) that actually matter, researchers can justify using simpler,
  lower-cost multispectral cameras instead of full hyperspectral rigs.
- **🛰️ Whole-field coverage.** Replaces a few point measurements (soil sensors,
  manual leaf sampling) with a complete, spatially-explicit **stress map** of
  the entire field — capturing variability sensors miss.
- **⏱️ Fast, non-destructive, repeatable.** No cutting leaves or digging soil;
  the same analysis can be re-run each flight to track stress over time.
- **🔬 Reproducible science.** A documented, open pipeline (NDVI → RFE → RF/SVM →
  Kappa) others can cite, audit, and extend to new crops, stages, or seasons.
- **📊 Decision-ready outputs.** CSV tables (sample spectra, band rankings,
  model accuracy/Kappa) and a GeoTIFF map drop straight into GIS tools,
  reports, or further statistical analysis.

**Typical workflow for a researcher:** fly the field → drop the per-band TIFs in
a folder → mark NS/GFWS sub-plot boundaries → run `setup.ps1` → read the stress
map and the ranked bands. From imagery to insight in minutes.

> ⚠️ **Scope & validation.** Results depend on your sensor, crop stage, and field
> conditions. As the original authors note, identified bands should be validated
> across multiple seasons before operational use. AquaSpectra is a research tool,
> not a substitute for agronomic judgement.

---

## ✨ Highlights

- 📦 **Band-separated TIF input** — reads one (large) GeoTIFF per spectral band
  directly from disk, with windowed/chunked reads.
- 🌱 **NDVI canopy masking** — removes soil background before analysis.
- 🎯 **RFE band selection** — ranks all 126 bands; highlights the discriminative
  670–780 nm Red/Red-Edge/NIR region (the paper's key finding).
- 🤖 **RF & SVM** tuning (80/20 split, 3-fold CV, **Cohen's Kappa**).
- 🗺️ **Wall-to-wall stress map** GeoTIFF from the best model.
- 🖥️ **Streamlit UI** + a **CLI**, plus a synthetic-data generator to try it
  with zero real imagery.

## 🧪 The method (mirrors the paper)

| Step | Module | Description |
|---|---|---|
| 1. Load | `bands.py` | stack band-separated TIFs (validate common grid, windowed reads) |
| 2. Mask | `ndvi.py` | NDVI ≥ threshold → keep maize canopy |
| 3. Sample | `sampling.py` | N random canopy points / sub-plot (paper: 200) → labelled spectra |
| 4. Select | `features.py` | RFE band ranking |
| 5. Model | `models.py` | RF & SVM tuning + Kappa over top-k subsets |
| 6. Map | `predict.py` | classify every canopy pixel → GeoTIFF |

**Paper headline result:** SVM + top-10 bands (mostly 670–780 nm, plus two near
930–940 nm water-absorption) → Kappa ≈ **0.886**.

## 📥 Input data format

The drone deliverable is **one GeoTIFF per spectral band** (band-separated, and
large):

```
data/bands/band_001.tif   # 450 nm
data/bands/band_002.tif   # 454 nm
...
data/bands/band_126.tif   # 950 nm
```

- All band TIFs **must share the same grid** (CRS, transform, width/height).
- Wavelengths come from `data/wavelengths.csv` (one `wavelength_nm` per row, in
  file order) **or** an explicit `input.bands` mapping in `config.yaml`.
- **Labels** are a vector file (`data/plots.geojson` / shapefile) of sub-plot
  polygons, each with a class attribute (`stress` = `NS` / `GFWS`).

## 🚀 Quickstart

### Easiest: one command (for researchers, Windows)

```powershell
# Full setup + build a demo dataset + run analysis + open the web app
.\setup.ps1 -Demo
```

`setup.ps1` checks Python, creates the environment, installs everything, and
launches the web app — no developer experience required. Later, just run
`.\run.ps1` to reopen the app. (Use `.\setup.ps1 -NoLaunch` to install only.)

### Manual setup

```powershell
# 1. environment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ui,dev]"

# 2. generate a synthetic scene (126 band TIFs + plots) — no real data needed
python tools\make_synthetic_data.py

# 3. full pipeline (fast config) + stress map
python -m aquaspectra.cli run     -c config.test.yaml
python -m aquaspectra.cli predict -c config.test.yaml
```

For real data, edit `config.yaml` (`input.band_dir`, `labels.vector_file`,
NDVI red/NIR wavelengths, sampling count, hyper-parameter grids) and run with
`-c config.yaml`.

## 🖥️ CLI

```powershell
python -m aquaspectra.cli info    -c config.yaml   # inspect band stack
python -m aquaspectra.cli sample  -c config.yaml   # sample canopy points
python -m aquaspectra.cli run     -c config.yaml   # sample + RFE + RF/SVM
python -m aquaspectra.cli predict -c config.yaml   # stress map GeoTIFF
```

## 🌐 Web UI (Streamlit)

```powershell
streamlit run app.py
```

> **Large files:** band-separated hyperspectral TIFs are **not uploaded through
> the browser** (they exceed Streamlit's 200 MB upload limit). Instead you type
> the **server-side folder path** where the TIFs already reside and the app
> reads them directly. `.streamlit/config.toml` raises `maxUploadSize` to 2 GB
> only for small optional uploads (e.g. a plots.geojson).

Tabs: **Data** (band info + false-colour preview) · **NDVI / Canopy** (mask +
coverage %) · **Train** (mean NS-vs-GFWS spectra, RFE ranking, RF/SVM Kappa) ·
**Stress Map**.

## 📂 Outputs (`outputs/`)

| File | Description |
|---|---|
| `samples.csv` | sampled spectra (subplot_id, label, per-band reflectance) |
| `band_ranking.csv` | every band ranked by RFE (rank 1 = most important) |
| `band_selection_results.csv` | accuracy & Kappa per model × top-k subset |
| `best_model.joblib` | best estimator + selected band indices + label encoder |
| `stress_map.tif` | per-pixel classification (1 = GFWS, 2 = NS, 0 = soil/nodata) |

## 🗂️ Project layout

```
config.yaml / config.test.yaml   paper-faithful / fast configs
pyproject.toml  requirements.txt
src/aquaspectra/
  bands.py   ndvi.py   sampling.py   features.py
  models.py  predict.py  pipeline.py  cli.py  config.py
app.py                           Streamlit UI
tools/make_synthetic_data.py     synthetic dataset generator
tests/                           pytest smoke tests
```

## 📚 Glossary

| Term | Plain-English meaning |
|---|---|
| **Hyperspectral imaging (HSI)** | A camera that captures many (here 126) very narrow colour bands instead of just Red/Green/Blue, revealing subtle differences invisible to the eye. |
| **Band** | One narrow slice of the light spectrum (e.g. the 730 nm band). Each band is delivered as its own GeoTIFF file. |
| **Wavelength (nm)** | The "colour" of light, measured in nanometres. 450 nm ≈ blue, 550 nm ≈ green, 680 nm ≈ red, 700–1000 nm ≈ near-infrared (invisible). |
| **Reflectance** | The fraction of light a surface bounces back at each wavelength — a leaf's "spectral fingerprint". |
| **Spectrum / spectral signature** | The full curve of reflectance across all bands for a single pixel. |
| **Data cube** | The stacked 3-D array (rows × columns × bands) formed from all band images. |
| **GeoTIFF (.tif)** | An image file that also stores geographic location (CRS + transform) so pixels map to real-world coordinates. |
| **CRS** | Coordinate Reference System — how pixel positions relate to locations on Earth (e.g. UTM zone 43N). |
| **NDVI** | Normalized Difference Vegetation Index = (NIR − Red) / (NIR + Red); a 0–1 "greenness/vegetation" score used to separate plants from soil. |
| **NIR (Near-Infrared)** | Light just beyond visible red (~700–1000 nm); healthy leaves reflect a lot of it. |
| **Red-Edge (RE)** | The sharp rise in reflectance between red and NIR (~680–750 nm); very sensitive to plant stress. |
| **Canopy** | The leafy top layer of the crop (as opposed to soil/background). |
| **Sub-plot** | One small managed cell of the field trial; here each holds a different maize variety. |
| **NS (No-Stress)** | The control group — plants given normal water. |
| **GFWS (Grain-Fill Water Stress)** | Plants deliberately water-stressed during the grain-filling growth stage. |
| **Feature selection** | Picking the most informative inputs (here, which light bands) and discarding the rest. |
| **RFE (Recursive Feature Elimination)** | A method that repeatedly removes the least useful band and re-ranks, producing an importance order. |
| **Random Forest (RF)** | A machine-learning classifier built from many decision trees voting together. |
| **SVM (Support Vector Machine)** | A machine-learning classifier that finds the best boundary separating two classes; "RBF" is a flexible curved-boundary variant. |
| **Hyper-parameter tuning** | Trying different model settings (e.g. number of trees) to find the best-performing one. |
| **Cross-validation (CV)** | Splitting training data into folds to estimate performance reliably; "3-fold" uses 3 splits. |
| **Accuracy** | Percentage of pixels classified correctly. |
| **Kappa (Cohen's Kappa)** | An accuracy measure that corrects for lucky guessing; 1.0 = perfect, 0 = no better than chance. |
| **Ortho-rectification / mosaicing** | Drone pre-processing that stitches and geometrically corrects images so they line up with the map. |
| **Stress map** | The final output raster colour-coding each canopy pixel as stressed or healthy. |

## 📖 Citation

If you use AquaSpectra, please cite **both** this repository (see
[`CITATION.cff`](CITATION.cff)) **and** the original paper:

```bibtex
@inproceedings{mohite2022maize,
  author    = {Mohite, Jayantrao and Sawant, Suryakant and Agarwal, Rishabh
               and Pandit, Ankur and Pappula, Srinivasu},
  title     = {Detection of Crop Water Stress in Maize Using Drone Based
               Hyperspectral Imaging},
  booktitle = {IGARSS 2022 -- 2022 IEEE International Geoscience and Remote
               Sensing Symposium},
  year      = {2022},
  pages     = {5957--5960},
  doi       = {10.1109/IGARSS46834.2022.9884686},
  publisher = {IEEE},
}
```

## 📝 License

[MIT](LICENSE) © 2026 Ashik Kuppili. The referenced paper is © IEEE and is not
redistributed here.
