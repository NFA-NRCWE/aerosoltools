"""
Loaders for instrument-specific aerosol data files.

This subpackage provides a collection of functions for reading and parsing
raw exports from common aerosol instruments. Each loader normalizes the
instrument-specific file format into one of the core classes in
:mod:`aerosoltools` (:class:`Aerosol1D`, :class:`Aerosol2D`,
or an instrument subclass).

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
* Tiger - VOC gas detector (ION)

In addition, :func:`load_data_from_folder` provides a convenience wrapper
for batch-loading and concatenating multiple compatible files from a
directory into a single aerosol object.
"""

from .acsm import load_simple_acsm_file
from .aethalometer import load_aethalometer_file
from .aps import load_aps_file
from .cpc import load_cpc_file
from .dev_weather import load_devlabs_file
from .discmini import load_discmini_file, load_discmini_raw_file
from .dusttrak import load_dusttrak_file
from .elpi import load_elpi_file
from .fmps import load_fmps_file
from .fourtec import load_fourtec_file
from .grimm import load_grimm_file
from .ns import load_ns_file
from .opcn3 import load_opcn3_file
from .ops import load_ops_file
from .partector import load_partector_file
from .ranger import load_ranger_file
from .registry import (
    INSTRUMENT_LOADERS,
    detect_instrument,
    load_file,
    supported_instruments,
)
from .smps import load_smps_file
from .support.exceptions import (
    DelimiterDetectionError,
    EmptyFileError,
    EncodingError,
    FileAccessError,
    FileFormatError,
    InstrumentDetectionError,
    LoaderError,
    MetadataParseError,
    TimestampError,
)
from .support.parsing import load_data_from_folder
from .tiger import load_tiger_file

__all__ = [
    "load_simple_acsm_file",
    "load_aethalometer_file",
    "load_aps_file",
    "load_cpc_file",
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
    "load_tiger_file",
    "load_data_from_folder",
    "load_devlabs_file",
    # Registry / auto-detection
    "load_file",
    "detect_instrument",
    "supported_instruments",
    "INSTRUMENT_LOADERS",
    # Exception hierarchy
    "LoaderError",
    "FileAccessError",
    "EmptyFileError",
    "EncodingError",
    "DelimiterDetectionError",
    "FileFormatError",
    "MetadataParseError",
    "TimestampError",
    "InstrumentDetectionError",
]
