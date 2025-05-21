"""
aerosoltools: Tools for loading and analyzing aerosol instrument data.

This package provides core data structures and loader functions to support
postprocessing of data from a variety of particle instruments used in aerosol science.

Included classes:
    - Aerosol1D      : For time-resolved scalar data (e.g., total concentrations)
    - Aerosol2D      : For size-resolved time-series data (e.g., size distributions)
    - AerosolAlt     : For instruments reporting alternative metrics (e.g., BC mass)

Supported instruments (via loaders):
    - CPC           : Condensation Particle Counter (TSI)
    - DiSCmini      : Personal dosimeter (Testo)
    - ELPI          : Electric Low Pressure Impactor (Dekati)
    - FMPS          : Fast Mobility Particle Sizer (TSI)
    - Fourtec       : Temperature/RH loggers
    - Grimm         : Optical particle counters
    - NS            : NanoScan SMPS (TSI)
    - OPC-N3        : Optical Particle Counter (Alphasense)
    - OPS           : Optical Particle Sizer (TSI)
    - Partector     : Particle dosimeter (Naneos)
    - SMPS          : Scanning Mobility Particle Sizer (TSI)
    - Aethalometer  : BC mass measurements (Magee Scientific)

Utilities:
    - Load_data_from_folder() : Automatically dispatches loaders over a folder of files

Typical usage:
    >>> import aerosoltools as at
    >>> data = at.Load_ELPI_file("example.txt")
    >>> obj = at.Aerosol2D(data)

Author: NRCWE community / NFA
"""

from .aerosol1d import Aerosol1D
from .aerosol2d import Aerosol2D
from .aerosolalt import AerosolAlt
from .loaders import (
    Load_Aethalometer_file,
    Load_CPC_file,
    Load_data_from_folder,
    Load_DiSCmini_file,
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

__all__ = [
    "Aerosol1D",
    "Aerosol2D",
    "AerosolAlt",
    "Load_Aethalometer_file",
    "Load_CPC_file",
    "Load_DiSCmini_file",
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
