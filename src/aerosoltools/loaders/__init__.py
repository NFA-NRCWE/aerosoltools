"""
aerosoltools.loaders: File loaders for various aerosol instruments.

This submodule contains loader functions for reading data files from a wide range of aerosol instruments,
including:

- Aethalometer (Magee Scientific)
- CPC – Condensation Particle Counter (TSI)
- DiSCmini – Electrostatic classifier (Testo)
- ELPI – Electric Low Pressure Impactor (Dekati)
- FMPS – Fast Mobility Particle Sizer (TSI)
- Fourtec – Bluefish temperature/RH loggers
- Grimm – Optical particle counters (Grimm Aerosol)
- NS – NanoScan (TSI)
- OPC-N3 – Optical Particle Counter (Alphasense)
- OPS – Optical Particle Sizer (TSI)
- Partector – PartectorTEM (Naneos)
- SMPS – Scanning Mobility Particle Sizer (TSI)

Each function is tailored to a specific instrument format and returns data structured for use
with `aerosoltools` classes such as `Aerosol1D` or `Aerosol2D`.

Additionally, the utility function `Load_data_from_folder()` provides a convenient interface
for batch-loading multiple compatible files from a directory.
"""

from .Aethalometer import Load_Aethalometer_file
from .Common import Load_data_from_folder
from .CPC import Load_CPC_file
from .Discmini import Load_DiSCmini_file
from .ELPI import Load_ELPI_file
from .FMPS import Load_FMPS_file
from .Fourtec import Load_Fourtec_file
from .Grimm import Load_Grimm_file
from .NS import Load_NS_file
from .OPCN3 import Load_OPCN3_file
from .OPS import Load_OPS_file
from .Partector import Load_Partector_file
from .SMPS import Load_SMPS_file

__all__ = [
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
