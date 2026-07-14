"""
Tools for loading and analyzing aerosol instrument data.

This package provides core data structures and loader functions for
post-processing data from a variety of aerosol instruments. It is
intended for workflows involving time-resolved, size-resolved, and
instrument-specific particle measurements.

Naming convention
-----------------
Public functions use PEP 8 ``snake_case`` names (e.g. ``load_elpi_file``,
``plot_correlation``). Classes use ``PascalCase`` (e.g. ``Aerosol2D``,
``DiSCmini``). The legacy ``PascalCase`` function spellings (``Load_ELPI_file``,
``Plot_correlation``, …) have been removed.

Classes
-------
Particle instruments (expose ``total_concentration``):

Aerosol1D
    Time-resolved scalar aerosol data (e.g., total particle concentration).
Aerosol2D
    Time- and size-resolved aerosol data (e.g., particle size distributions).
Aerosol3d
    Dual-distribution size-resolved data (e.g., APS aerodynamic + optical).
DiSCmini
    DiSCmini number/size/LDSA dosimeter; see ``.size`` / ``.ldsa``.
DustTrak
    DustTrak PM mass fractions; see ``.pm1`` / ``.pm2_5`` / … / ``.total``.
ELPI
    Electrical low-pressure impactor (density-dependent size axis).

Non-particle instruments (no ``total_concentration``; use their domain
accessors and ``.dtype`` / ``.unit`` metadata instead):

Gas1D
    Single-gas concentration (e.g., Ranger Cl₂/NO₂); see ``.concentration``.
Aethalometer
    Wavelength-resolved black-carbon mass; see ``.ir_bcc``/``.uv_bcc``/….
Environmental1D
    Environmental / weather channels; see ``.temperature``/``.rh``/….
Partector
    Lung-deposited surface area (LDSA) dosimeter; see ``.ldsa``.

The legacy ``AerosolAlt`` catch-all has been removed; instruments that used it
now have dedicated classes (above).

Loader functions
----------------
load_cpc_file
    Load data from condensation particle counters (CPC, TSI).
load_devlabs_file
    Load weather station data from DevLabs instrument.
load_discmini_file
    Load data from DiSCmini personal dosimeters (Testo).
load_elpi_file
    Load data from electrical low-pressure impactors (ELPI, Dekati).
load_fmps_file
    Load data from fast mobility particle sizers (FMPS, TSI).
load_fourtec_file
    Load environmental logger data (e.g., temperature / relative humidity).
load_grimm_file
    Load data from Grimm optical particle counters.
load_ns_file
    Load data from NanoScan SMPS (NS, TSI).
load_opcn3_file
    Load data from Alphasense OPC-N3 optical particle counters.
load_ops_file
    Load data from optical particle sizers (OPS, TSI).
load_partector_file
    Load data from Naneos Partector particle dosimeters.
load_smps_file
    Load data from scanning mobility particle sizers (SMPS, TSI).
load_aethalometer_file
    Load black carbon mass data from MicroAeth / aethalometers.
load_dusttrak_file
    Load PM mass concentration data from DustTrak instruments.
load_data_from_folder
    Dispatch the appropriate loader over all files in a folder and
    return the combined dataset(s).

Intercomparison (multi-dataset) workflows
-----------------------------------------
combine_size_ranges
    Stitch two range-extending instruments (e.g. NanoScan + OPS) into a
    single size-resolved dataset with a chosen crossover diameter.
combine_measurements
    Concatenate several runs of the same instrument into one time series.
plot_correlation
    Plot and fit correlations between two aerosol time series (e.g.,
    instrument inter-comparisons).
bland_altman_analysis
    Bland-Altman (difference) agreement plot between two datasets.
calibrate_against_reference
    Fit and apply a calibration of one dataset against a reference, returning
    ``(calibrated, model)``.

Advanced multi-dataset workflows (``fit_calibration``, ``apply_calibration``,
``CalibrationModel``, …) live under :mod:`aerosoltools.intercomparison`.

Typical usage example
---------------------
    >>> import aerosoltools as at
    >>> ns = at.load_ns_file("nanoscan_example.txt")
    >>> ops = at.load_ops_file("ops_example.txt")
    >>> combined = at.combine_size_ranges(ns, ops, crossover=400)
"""

from ._core.decay import DecayResult
from ._core.fitting import PSDFitResult, lognormal_modes
from .aerosol1d import Aerosol1D
from .aerosol2d import Aerosol2D
from .aerosol3d import Aerosol3d, CorrelationCube
from .aethalometer import Aethalometer
from .discmini import DiSCmini
from .dusttrak import DustTrak
from .elpi import ELPI
from .environmental import Environmental1D
from .gas1d import Gas1D
from .intercomparison import (
    bland_altman_analysis,
    calibrate_against_reference,
    combine_measurements,
    combine_size_ranges,
    fit_data,
    plot_correlation,
)
from .loaders import (
    INSTRUMENT_LOADERS,
    LoaderError,
    detect_instrument,
    load_aethalometer_file,
    load_aps_file,
    load_cpc_file,
    load_data_from_folder,
    load_devlabs_file,
    load_discmini_file,
    load_discmini_raw_file,
    load_dusttrak_file,
    load_elpi_file,
    load_file,
    load_fmps_file,
    load_fourtec_file,
    load_grimm_file,
    load_ns_file,
    load_opcn3_file,
    load_ops_file,
    load_partector_file,
    load_ranger_file,
    load_smps_file,
)
from .partector import Partector

__all__ = [
    # Particle classes
    "Aerosol1D",
    "Aerosol2D",
    "Aerosol3d",
    "CorrelationCube",
    "DiSCmini",
    "DustTrak",
    "ELPI",
    # Non-particle classes
    "Aethalometer",
    "Environmental1D",
    "Gas1D",
    "Partector",
    # Intercomparison (multi-dataset) workflows
    "bland_altman_analysis",
    "combine_size_ranges",
    "fit_data",
    "plot_correlation",
    "combine_measurements",
    "calibrate_against_reference",
    # Size-distribution fitting
    "lognormal_modes",
    "PSDFitResult",
    # Decay fitting
    "DecayResult",
    # Loading: auto-detect entry point + registry + error base
    "load_file",
    "detect_instrument",
    "INSTRUMENT_LOADERS",
    "LoaderError",
    # Per-instrument loaders
    "load_aps_file",
    "load_aethalometer_file",
    "load_cpc_file",
    "load_devlabs_file",
    "load_discmini_file",
    "load_discmini_raw_file",
    "load_dusttrak_file",
    "load_elpi_file",
    "load_fmps_file",
    "load_fourtec_file",
    "load_grimm_file",
    "load_ns_file",
    "load_opcn3_file",
    "load_ops_file",
    "load_partector_file",
    "load_ranger_file",
    "load_smps_file",
    "load_data_from_folder",
]
