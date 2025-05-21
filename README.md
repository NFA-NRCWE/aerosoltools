# aerosoltools

**Tools for loading and analyzing aerosol instrument data**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](./tests)

---

## Overview

`aerosoltools` is a Python library developed at NFA for loading, processing, and analyzing data from a variety of aerosol instruments. It provides a consistent data structure for time-resolved and size-resolved measurements using `Aerosol1D`, `Aerosol2D`, and `AerosolAlt` classes.

The package includes loaders for common instrument exports and a batch-loading utility for processing entire folders.

---

## ✨ Features

- Unified interface for:
  - SMPS, FMPS, NS (TSI)
  - ELPI (Dekati)
  - OPS, CPC, Partector, DiscMini
  - Aethalometer, OPC-N3, Grimm, Fourtec
- Automatically handles:
  - Bin edge/midpoint parsing
  - Unit conversions (e.g., `#/cm³`, `dN/dlogDp`)
  - Metadata extraction
- Batch loading via `Load_data_from_folder()`
- Returns structured objects for plotting, stats, or export

---

## 📦 Installation
Soon to be published on PyPI

## Quickstart
### Load a single instrument file

import aerosoltools as at

elpi = at.Load_ELPI_file("data/elpi_sample.txt")

elpi.plot_total()

### Access metadata

print(elpi._meta["instrument"])

### Batch-load a folder of files
 
from aerosoltools import Load_data_from_folder, Load_CPC_file

folder_path = "data/cpc_campaign/"

data = Load_data_from_folder(folder_path, loader=Load_CPC_file)

## 🧰 Provided Loaders

| Instrument      | Function                  |
|-----------------|---------------------------|
| Aethalometer    | `Load_Aethalometer_file()` |
| CPC             | `Load_CPC_file()`          |
| DiSCmini        | `Load_DiSCmini_file()`     |
| ELPI            | `Load_ELPI_file()`         |
| FMPS            | `Load_FMPS_file()`         |
| Fourtec         | `Load_Fourtec()`           |
| Grimm           | `Load_Grimm_file()`        |
| NS (NanoScan)   | `Load_NS_file()`           |
| OPC-N3          | `Load_OPCN3_file()`        |
| OPS             | `Load_OPS_file()`          |
| Partector       | `Load_Partector_file()`    |
| SMPS            | `Load_SMPS_file()`         |


📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

🙌 Acknowledgments

Developed by the NRCWE / NFA community to standardize and accelerate aerosol data workflows.

Feel free to contribute, submit issues, or request support!
