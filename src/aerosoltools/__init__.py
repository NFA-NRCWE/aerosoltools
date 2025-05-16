"""
aerosoltools: Tools for loading and analyzing aerosol instrument data.

This package provides:
- The Aerosol1D, Aerosol2D, and AerosolAlt classes for data representation.
- Loader functions for CPC, ELPI, OPS, Partector, DiSCmini, and more.
- Utilities for batch loading from folders and preprocessing.

Usage example:
    import aerosoltools as at

    data = at.Load_ELPI_file("elpi_data.txt")
    aerosol = at.Aerosol2D(data)

Author: NRCWE community / NFA
"""
from .aerosol1d import Aerosol1D
from .aerosol2d import Aerosol2D
from .aerosolalt import AerosolAlt

from .loaders import (
    Load_CPC_file,
    Load_DiSCmini_file,
    Load_ELPI_file,
    Load_NS_file,
    Load_OPS_file,
    Load_Partector_file,
    Load_data_from_folder,
)

__all__ = [
    "Aerosol1D",
    "Aerosol2D",
    "AerosolAlt",
    "Load_CPC_file",
    "Load_DiSCmini_file",
    "Load_ELPI_file",
    "Load_NS_file",
    "Load_OPS_file",
    "Load_Partector_file",
    "Load_data_from_folder",
]
