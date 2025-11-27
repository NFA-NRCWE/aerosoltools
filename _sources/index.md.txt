# aerosoltools

Tools for loading and analyzing aerosol instrument data.

---

## Installation

The package is installed by running the code below in a dedicated terminal.

```bash
pip install aerosoltools
```

---

## Overview

`aerosoltools` is a Python library developed at NFA for loading, processing, analyzing, and plotting data from a variety of aerosol instruments. It provides consistent data structures for:

- **1D time-series** (e.g. total number or mass) via `Aerosol1D`
- **2D size-resolved time series** via `Aerosol2D`
- **Alternative / legacy formats** via `AerosolAlt`

The package includes:

- Instrument-specific loaders for common export formats
- Activity segmentation and task-based statistics
- Exposure metrics such as **8 h TWA**, **short-term limits**, and **peak counts**
- Convenience plotting for time series and particle size distributions (PSDs)

---

## Provided loaders

| Instrument    | Function                   | Company                   |
| ------------- | -------------------------- | ------------------------- |
| Aethalometer  | `Load_Aethalometer_file()` | **Magee Scientific**      |
| CPC           | `Load_CPC_file()`          | **TSI Inc.**              |
| DiSCmini      | `Load_DiSCmini_file()`     | **Testo**                 |
| DustTrak      | `Load_DustTrak_file()`     | **TSI Inc.**              |
| ELPI          | `Load_ELPI_file()`         | **Dekati Ltd.**           |
| FMPS          | `Load_FMPS_file()`         | **TSI Inc.**              |
| Fourtec       | `Load_Fourtec_file()`      | **Fourtec Technologies**  |
| Grimm         | `Load_Grimm_file()`        | **GRIMM Aerosol Technik** |
| NS (NanoScan) | `Load_NS_file()`           | **TSI Inc.**              |
| OPC-N3        | `Load_OPCN3_file()`        | **Alphasense Ltd.**       |
| OPS           | `Load_OPS_file()`          | **TSI Inc.**              |
| Partector     | `Load_Partector_file()`    | **naneos GmbH**           |
| SMPS          | `Load_SMPS_file()`         | **TSI Inc.**              |

---

## Key features

### Unified data model

- Datetime parsing and indexing  
- Particle data formatting and bin edges / midpoints  
- Dtype tracking (`dN`, `dM`, `dS`, `dV`, and `/dlogDp` normalization)  
- Metadata for instrument, units, serial number, etc.

### Activities & segmentation

- Mark tasks/segments with `mark_activities()`
- Built-in `"All data"` activity
- Helper methods to extract activity-specific data:
  - `get_activity_data()`
  - `get_activity_extra_data()`

### Summaries & exposure metrics

- `summarize_activities()`  
  - Task-based descriptive statistics  
  - Duration (min + HH:MM)
  - PNC, PMₓ, and size metrics (e.g. mode Dp, median Dp, GMD) for 2D data

- `summarize_exposure()`  
  - 1D (`Aerosol1D`): PNC time series  
  - 2D (`Aerosol2D`): PNC, MASS, and Pₓ metrics (e.g. PM₂.₅, PM₄.₂, PN₁₀)  
  - 8 h (or custom) TWA with:
    - assumed exposure duration
    - background as a constant value or another activity
  - Short-term limit exceedances (e.g. 15 min rolling window)
  - Peak counts, high percentiles (C95/C99), IQR, time above limits

### Pₓ / fraction utilities (2D)

- Cumulative and band-limited Pₓ (PM, PN, PS, PV)
- Internal caching so repeated calls reuse existing series in `extra_data`

### Time operations

- `timeshift()` – shift time stamps  
- `timecrop()` – focus on or exclude intervals  
- `timerebin()` – resample to regular grids  
- `timesmooth()` – rolling smoothing on numeric columns  

Irregular sampling is handled carefully for integration and TWA.

### Plotting

- `plot_timeseries()` – total concentration vs time (with optional activity shading)
- `plot_psd()` – particle size distributions (linear or log scales)
- Correlation/comparison utilities (e.g. `Combine_NS_OPS`, `Plot_correlation`)

### Batch loading

- `Load_data_from_folder()` – apply any loader to a folder of files and collect results.

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

## Acknowledgments

Developed by the NRCWE / NFA community to standardize and accelerate aerosol data workflows.

Contributions, issues, and feature requests are very welcome!

---

```{toctree}
:maxdepth: 0
:caption: Examples

examples/index
```

```{toctree}
:caption: Aerosoltools documentation
:maxdepth: 1

Aerosol1D    <api/aerosoltools/aerosol1d/index>
Aerosol2D    <api/aerosoltools/aerosol2d/index>
AerosolAlt    <api/aerosoltools/aerosolalt/index>
Loaders    <api/aerosoltools/loaders/index>
Utility    <api/aerosoltools/utility/index>
```