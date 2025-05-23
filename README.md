# aerosoltools

**Tools for loading and analyzing aerosol instrument data**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](./tests)
![Docs](https://github.com/NFA-NRCWE/aerosoltools/actions/workflows/deploy-docs.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/aerosoltools.svg)](https://pypi.org/project/aerosoltools/)
---

## Overview

`aerosoltools` is a Python library developed at NFA for loading, processing, analyzing, and plotting data from a variety of aerosol instruments. It provides a consistent data structure for time-resolved and size-resolved measurements using `Aerosol1D`, `Aerosol2D`, and `AerosolAlt` classes.

The package includes loaders for common instrument exports and a batch-loading utility for processing entire folders.

For the full documenation, and examples of use, visit:

[View the documentation](https://nfa-nrcwe.github.io/aerosoltools/)

---

## 🧰 Provided Loaders

| Instrument    | Function                   | Company                   |
| ------------- | -------------------------- | ------------------------- |
| Aethalometer  | `Load_Aethalometer_file()` | **Magee Scientific**      |
| CPC           | `Load_CPC_file()`          | **TSI Inc.**              |
| DiSCmini      | `Load_DiSCmini_file()`     | **Testo**                 |
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

- Unified interface for loaded aerosoldata, automatically handling:
  - Datetime conversion
  - Particle data formatting
  - Bin edge/midpoint parsing
  - Dtype tracking e.g. dN, dV, dM, dS as well as normalization via dlogDp
  - Metadata extraction
- Batch loading via `Load_data_from_folder()`
- Functions for time shifting, cropping, rebinning, and smoothing
- Enables segmentation to group datapoints within specifc timeframes
- Returns structured objects for plotting, statistics, or export
- Functions to plot timeseries, PSD, and correlation plots 

---

## 📦 Installation

The package is available via PyPI and can therefore be installed as:

<pre><code>``` pip install aerosoltools ```</code></pre>

---

## Quickstart
### Load a single instrument file

<pre><code>```
import aerosoltools as at
elpi = at.Load_ELPI_file("data/elpi_sample.txt")
elpi.plot_total()
<pre><code>```
  
### Access metadata
<pre><code>```
elpi.metadata
<pre><code>```
  
### Batch-load a folder of files
<pre><code>```
folder_path = "data/cpc_campaign/"
data = at.Load_data_from_folder(folder_path, loader=at.Load_CPC_file)
<pre><code>```
  
---

📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

🙌 Acknowledgments

Developed by the NRCWE / NFA community to standardize and accelerate aerosol data workflows.

Feel free to contribute, submit issues, or request support!
