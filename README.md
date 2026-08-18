# aerosoltools

**Tools for loading and analyzing aerosol instrument data**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/NFA-NRCWE/aerosoltools/blob/main/LICENSE)  
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/NFA-NRCWE/aerosoltools/tree/main/tests)  
![Docs](https://github.com/NFA-NRCWE/aerosoltools/actions/workflows/docs-publish.yml/badge.svg)  
[![PyPI version](https://badge.fury.io/py/aerosoltools.svg)](https://pypi.org/project/aerosoltools/)

---

## Overview

`aerosoltools` is a Python library developed at NFA for loading, processing, analyzing, and plotting data from a variety of aerosol instruments. It provides consistent data structures for:

- **1D time-series** (e.g. total number or mass) via `Aerosol1D`
- **2D size-resolved time-series** via `Aerosol2D`
- **Dual-distribution data** (APS aerodynamic + optical) via `Aerosol3d`
- **Instrument-specific classes** with their own physics and accessors —
  `DiSCmini`, `DustTrak`, `ELPI`
- **Non-particle instruments** — `Gas1D`, `Aethalometer`, `Environmental1D`,
  `Partector` (these expose domain accessors instead of `total_concentration`)

The package includes loaders for common instrument exports, tools for activity segmentation, and convenience methods for **task-based statistics** and **exposure assessment** (e.g. 8 h TWA, short-term limits, peaks).

Prefer a point-and-click workflow? An interactive **desktop GUI** is included —
see the **Desktop GUI** section below.

For full documentation and usage examples, see:

👉 [View the documentation](https://nfa-nrcwe.github.io/aerosoltools/)

---

## 🧰 Provided Loaders

| Instrument             | Function                    | Company                   |
| ---------------------- | --------------------------- | ------------------------- |
| Aethalometer           | `load_aethalometer_file()`  | **AethLabs**              |
| APS                    | `load_aps_file()`           | **TSI Inc.**              |
| CPC                    | `load_cpc_file()`           | **TSI Inc.**              |
| DiSCmini               | `load_discmini_file()`      | **Testo**                 |
| DiSCmini (raw export)  | `load_discmini_raw_file()`  | **Testo**                 |
| DustTrak               | `load_dusttrak_file()`      | **TSI Inc.**              |
| ELPI                   | `load_elpi_file()`          | **Dekati Ltd.**           |
| FMPS                   | `load_fmps_file()`          | **TSI Inc.**              |
| Fourtec                | `load_fourtec_file()`       | **Fourtec Technologies**  |
| Grimm                  | `load_grimm_file()`         | **GRIMM Aerosol Technik** |
| NS (NanoScan)          | `load_ns_file()`            | **TSI Inc.**              |
| OPC-N3                 | `load_opcn3_file()`         | **Alphasense Ltd.**       |
| OPS                    | `load_ops_file()`           | **TSI Inc.**              |
| Partector              | `load_partector_file()`     | **naneos GmbH**           |
| Ranger                 | `load_ranger_file()`        | **Aeroqual**              |
| SMPS                   | `load_smps_file()`          | **TSI Inc.**              |
| Weather station        | `load_devlabs_file()`       | **DevLabs**               |

Don't know which loader you need? `load_file()` auto-detects the instrument
from the file itself, and `detect_instrument()` reports what it found.

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
  - `load_data_from_folder()` to apply a loader across a folder of files

---

## 📦 Installation

Install from PyPI:

```bash
pip install aerosoltools
```

To also get the interactive desktop **GUI** (adds the PyQt5 dependency), install
the `gui` extra:

```bash
pip install aerosoltools[gui]
```

---

## Quickstart

### Load a single instrument file

```python
import aerosoltools as at

elpi = at.load_elpi_file("data/elpi_sample.txt")
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
    activities=["Emission"],
    background="Background",  # or a float, or None
    short_limit=1.0,
    long_limit=1.0,
)
```

### Batch-load a folder of files

```python
folder_path = "data/cpc_campaign/"
data = at.load_data_from_folder(folder_path, at.load_cpc_file)
```

---

## 🖥️ Desktop GUI

`aerosoltools` ships with an interactive desktop app for loading multiple
datasets, marking shared tasks, and exploring time series, size distributions,
summaries, overlays and instrument correlations — no coding required.

First install the GUI extra (adds PyQt5):

```bash
pip install aerosoltools[gui]
```

### Create a desktop shortcut (Windows — recommended)

After installing, run this **once** in a terminal:

```bash
aerosoltools-gui-shortcut
```

This places an **AerosolTools** shortcut (with the app icon) on your **Desktop**
and in the **Start Menu**. From then on, just double-click the desktop icon — or
press the Windows key and type *AerosolTools* — to launch the app; no terminal or
Python knowledge is needed afterwards. Re-running the command simply refreshes
the shortcuts.

### Launch from the command line or Python

These work on any platform with the `gui` extra installed:

```bash
aerosoltools-gui                       # open the empty viewer
aerosoltools-gui data/sample_ELPI.txt  # open a file on startup
python -m aerosoltools.gui             # equivalent, via the module
```

```python
from aerosoltools.gui import launch

launch()                  # empty viewer
launch("data/sample.txt") # or pre-load a file
```

---

## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## 🙌 Acknowledgments

Developed by the NRCWE / NFA community to standardize and accelerate aerosol data workflows.

Contributions, issues, and feature requests are very welcome!
