"""
Loaders for instrument-specific aerosol data files.

This subpackage provides a collection of functions for reading and parsing
raw exports from common aerosol instruments. Each loader normalizes the
instrument-specific file format into one of the core classes in
:mod:`aerosoltools` (:class:`Aerosol1D`, :class:`Aerosol2D`,
or :class:`AerosolAlt`).

Supported instruments include:

* Aethalometer (Magee Scientific)
* CPC – Condensation Particle Counter (TSI)
* DiSCmini – Electrostatic dosimeter (Testo)
* DustTrak – Optical particle counter (TSI)
* ELPI – Electric Low Pressure Impactor (Dekati)
* FMPS – Fast Mobility Particle Sizer (TSI)
* Fourtec – Bluefish temperature / RH loggers
* Grimm – Optical particle counters (Grimm Aerosol)
* NS – NanoScan SMPS (TSI)
* OPC-N3 – Optical particle counter (Alphasense)
* OPS – Optical Particle Sizer (TSI)
* Partector – Partector / PartectorTEM (Naneos)
* SMPS – Scanning Mobility Particle Sizer (TSI)

In addition, :func:`Load_data_from_folder` provides a convenience wrapper
for batch-loading and concatenating multiple compatible files from a
directory into a single aerosol object.
"""

from .Aethalometer import Load_Aethalometer_file
from .Common import Load_data_from_folder
from .CPC import Load_CPC_file
from .Dev_weather import Load_Devlabs_file
from .Discmini import Load_DiSCmini_file, Load_DiSCmini_raw_file
from .DustTrak import Load_DustTrak_file
from .ELPI import Load_ELPI_file
from .FMPS import Load_FMPS_file
from .Fourtec import Load_Fourtec_file
from .Grimm import Load_Grimm_file
from .NS import Load_NS_file
from .OPCN3 import Load_OPCN3_file
from .OPS import Load_OPS_file
from .Partector import Load_Partector_file
from .Ranger import Load_Ranger_file
from .SMPS import Load_SMPS_file

__all__ = [
    "Load_Aethalometer_file",
    "Load_CPC_file",
    "Load_DiSCmini_file",
    "Load_DiSCmini_raw_file",
    "Load_DustTrak_file",
    "Load_ELPI_file",
    "Load_FMPS_file",
    "Load_Fourtec_file",
    "Load_Grimm_file",
    "Load_NS_file",
    "Load_OPCN3_file",
    "Load_OPS_file",
    "Load_Partector_file",
    "Load_Ranger_file",
    "Load_SMPS_file",
    "Load_data_from_folder",
    "Load_Devlabs_file",
]
