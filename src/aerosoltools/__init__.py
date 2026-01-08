"""
Tools for loading and analyzing aerosol instrument data.

This package provides core data structures and loader functions for
post-processing data from a variety of aerosol instruments. It is
intended for workflows involving time-resolved, size-resolved, and
instrument-specific particle measurements.

Classes
-------
Aerosol1D
    Time-resolved scalar aerosol data (e.g., total particle concentration).
Aerosol2D
    Time- and size-resolved aerosol data (e.g., particle size distributions).
AerosolAlt
    Time-resolved data for instruments reporting alternative metrics
    (e.g., black carbon mass, optical depth, or custom channels).

Loader functions
----------------
Load_CPC_file
    Load data from condensation particle counters (CPC, TSI).
Load_DiSCmini_file
    Load data from DiSCmini personal dosimeters (Testo).
Load_ELPI_file
    Load data from electrical low-pressure impactors (ELPI, Dekati).
Load_FMPS_file
    Load data from fast mobility particle sizers (FMPS, TSI).
Load_Fourtec_file
    Load environmental logger data (e.g., temperature / relative humidity).
Load_Grimm_file
    Load data from Grimm optical particle counters.
Load_NS_file
    Load data from NanoScan SMPS (NS, TSI).
Load_OPCN3_file
    Load data from Alphasense OPC-N3 optical particle counters.
Load_OPS_file
    Load data from optical particle sizers (OPS, TSI).
Load_Partector_file
    Load data from Naneos Partector particle dosimeters.
Load_SMPS_file
    Load data from scanning mobility particle sizers (SMPS, TSI).
Load_Aethalometer_file
    Load black carbon mass data from MicroAeth / aethalometers.
Load_DustTrak_file
    Load PM mass concentration data from DustTrak instruments.
Load_data_from_folder
    Dispatch the appropriate loader over all files in a folder and
    return the combined dataset(s).

Utilities
---------
Combine_NS_OPS
    Combine NanoScan and OPS measurements into a single size-resolved
    dataset with harmonized bin structure.
Plot_correlation
    Plot and fit correlations between two aerosol time series (e.g.,
    instrument inter-comparisons).

Typical usage example
---------------------
    >>> import aerosoltools as at
    >>> ns = at.Load_NS_file("nanoscan_example.txt")
    >>> ops = at.Load_OPS_file("ops_example.txt")
    >>> combined = at.Combine_NS_OPS(ns, ops)
"""

from .aerosol1d import Aerosol1D
from .aerosol2d import Aerosol2D
from .aerosolalt import AerosolAlt
from .loaders import (
    Load_Aethalometer_file,
    Load_CPC_file,
    Load_data_from_folder,
    Load_DiSCmini_file,
    Load_DustTrak_file,
    Load_ELPI_file,
    Load_FMPS_file,
    Load_Fourtec_file,
    Load_Grimm_file,
    Load_NS_file,
    Load_OPCN3_file,
    Load_OPS_file,
    Load_Partector_file,
    Load_SMPS_file,
)
from .utility import  bland_altman_analysis, Combine_NS_OPS, Plot_correlation

__all__ = [
    "Aerosol1D",
    "Aerosol2D",
    "AerosolAlt",
    "bland_altman_analysis"
    "Combine_NS_OPS",
    "Plot_correlation",
    "Load_Aethalometer_file",
    "Load_CPC_file",
    "Load_DiSCmini_file",
    "Load_DustTrak_file",
    "Load_ELPI_file",
    "Load_FMPS_file",
    "Load_Fourtec_file",
    "Load_Grimm_file",
    "Load_NS_file",
    "Load_OPCN3_file",
    "Load_OPS_file",
    "Load_Partector_file",
    "Load_SMPS_file",
    "Load_data_from_folder",
]
