"""
aerosoltools.loaders: File loaders for various aerosol instruments.

This submodule provides functions to load and parse measurement data from:

- CPC (Condensation Particle Counter)
- DiSCmini
- ELPI (Electrical Low Pressure Impactor)
- NS Sampler
- OPS (Optical Particle Sizer)
- Partector
- And a generic `Load_data_from_folder()` utility for batch loading

Each function is instrument-specific and returns data compatible with
aerosoltools classes like `Aerosol1D` and `Aerosol2D`.
"""
from .CPC import Load_CPC_file
from .Discmini import Load_DiSCmini_file
from .ELPI import Load_ELPI_file
from .NS import Load_NS_file
from .OPS import Load_OPS_file
from .Partector import Load_Partector_file
from .Common import Load_data_from_folder

__all__ = [
    "Load_CPC_file",
    "Load_DiSCmini_file",
    "Load_ELPI_file",
    "Load_NS_file",
    "Load_OPS_file",
    "Load_Partector_file",
    "Load_data_from_folder",
]
