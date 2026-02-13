# aerosoltools

**Tools for loading and analyzing aerosol instrument data**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](./tests)  
![Docs](https://github.com/NFA-NRCWE/aerosoltools/actions/workflows/deploy-docs.yml/badge.svg)  
[![PyPI version](https://badge.fury.io/py/aerosoltools.svg)](https://pypi.org/project/aerosoltools/)

---

## Overview

`aerosoltools` is a Python library developed at NFA for loading, processing, analyzing, and plotting data from a variety of aerosol instruments. It provides consistent data structures for:

- **1D time-series** (e.g. total number or mass) via `Aerosol1D`
- **2D size-resolved time-series** via `Aerosol2D`
- **Alternative / legacy formats** via `AerosolAlt`

The package includes loaders for common instrument exports, tools for activity segmentation, and convenience methods for **task-based statistics** and **exposure assessment** (e.g. 8 h TWA, short-term limits, peaks).

For full documentation and usage examples, see:

👉 [View the documentation](https://nfa-nrcwe.github.io/aerosoltools/)

---

## 🧰 Provided Loaders

| Instrument    | Function                   | Company                   |
| ------------- | -------------------------- | ------------------------- |
| Aethalometer  | `Load_Aethalometer_file()` | **Aethlabs**      |
| CPC           | `Load_CPC_file()`          | **TSI Inc.**              |
| DiSCmini      | `Load_DiSCmini_file()`     | **Testo**                 |
| DustTrak      | `Load_DustTrak_file()`     | **TSI Inc.**              |
| ELPI          | `Load_ELPI_file()`         | **Dekati Ltd.**           |
| FMPS          | `Load_FMPS_file()`         | **TSI Inc.**              |
| Fourtec       | `Load_Fourtec()`           | **Fourtec Technologies**  |
| Grimm         | `Load_Grimm_file()`        | **GRIMM Aerosol Technik** |
| NS (NanoScan) | `Load_NS_file()`           | **TSI Inc.**              |
| OPC-N3        | `Load_OPCN3_file()`        | **Alphasense Ltd.**       |
| OPS           | `Load_OPS_file()`          | **TSI Inc.**              |
| Partector     | `Load_Partector_file()`    | **naneos GmbH**           |
| SMPS          | `Load_SMPS_file()`         | **TSI Inc.**              |

---

## ✨ Features

- **Unified interface** for loaded aerosol data:
  - Datetime parsing and indexing  
  - Particle data formatting and bin edges/midpoints  
  - Dtype tracking (`dN`, `dM`, `dS`, `dV`, and `/dlogDp` normalization)  
  - Metadata extraction (instrument, units, serial number, etc.)

- **Activity handling**
  - Mark tasks/segments via `mark_activities()`
  - Built-in `"All data"` activity
  - Helper methods to extract activity-specific data

- **Summaries & exposure metrics**
  - `summarize_activities()` – task-based descriptive statistics (duration, PNC, PMx, size metrics, etc.)
  - `summarize_exposure()` – detailed exposure assessment:
    - 1D (`Aerosol1D`): PNC time series
    - 2D (`Aerosol2D`): PNC, MASS, and Pₓ metrics (e.g. PM₂.₅, PM₄.₂, PN₁₀)
    - 8 h (or custom) TWA with background level (value or activity)
    - Short-term limit exceedances (e.g. 15 min window)
    - Peak counts, high percentiles (C95/C99), IQR, durations above limits

- **Pₓ / fraction utilities (2D)**
  - Cumulative and band-limited Pₓ (PM, PN, PS, PV)
  - Reuses previously computed series via internal caching

- **Time operations**
  - Time shifting, cropping, rebinning, and smoothing
  - Handles irregular sampling safely for integration and TWA

- **Plotting**
  - Timeseries plots (with activity shading)
  - Particle size distributions (PSD)
  - Simple correlation/comparison plots

- **Batch loading**
  - `Load_data_from_folder()` to apply a loader across a folder of files

---

## 📦 Installation

Install from PyPI:

```bash
pip install aerosoltools
```

---

## Quickstart

### Load a single instrument file

```python
import aerosoltools as at

elpi = at.Load_ELPI_file("data/elpi_sample.txt")
elpi.plot_timeseries()
```

### Access metadata

```python
elpi.metadata
```

### Mark activities and summarize

```python
activity_periods = {
    "Background": [("2023-09-07 09:06:50", "2023-09-07 09:07:50")],
    "Emission":   [("2023-09-07 09:07:55", "2023-09-07 09:08:30")],
}

elpi.mark_activities(activity_periods)

# Task-based summary across all activities
summary = elpi.summarize_activities()

# Detailed exposure summary for respirable dust (PM4.2) during "Emission"
exp = elpi.summarize_exposure(
    metric="PM4.2",
    activity="Emission",
    background="Background",  # or a float, or None
    short_limit=1.0,
    long_limit=1.0,
)
```

### Batch-load a folder of files

```python
folder_path = "data/cpc_campaign/"
data_list = at.Load_data_from_folder(folder_path, loader=at.Load_CPC_file)
```

---

## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## 🙌 Acknowledgments

Developed by the NRCWE / NFA community to standardize and accelerate aerosol data workflows.

Contributions, issues, and feature requests are very welcome!
