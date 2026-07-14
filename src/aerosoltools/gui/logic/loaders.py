"""GUI shim over the public loader registry.

The instrument registry and automatic file-format detection now live in the
public API (:mod:`aerosoltools.loaders.registry`); the GUI reuses that logic
rather than owning a parallel copy. This module only re-exports the pieces the
GUI's file-open flow references, so a script and the GUI detect and load files
through exactly the same code.
"""

from __future__ import annotations

from ...loaders.registry import (
    INSTRUMENT_LOADERS,
    LOADERS,
    SNIFFERS,
    detect_instrument,
    guess_instrument,
    identify_instrument,
    is_Aethalometer_file,
    is_APS_file,
    is_CPC_file,
    is_DiSCmini_file,
    is_DiSCmini_raw_file,
    is_DustTrak_file,
    is_ELPI_file,
    is_FMPS_file,
    is_Fourtec_file,
    is_Grimm_file,
    is_NS_file,
    is_OPCN3_file,
    is_OPS_file,
    is_Partector_file,
    is_Ranger_file,
    is_SMPS_file,
    load_file,
    require_identified_instrument,
    sniff_instrument,
    supported_instrument_message,
    supported_instrument_names,
    supported_instruments,
)
from ...loaders.support.exceptions import InstrumentDetectionError

#: Historical GUI name for the detection failure (now the public API exception).
UnrecognizedInstrumentError = InstrumentDetectionError

__all__ = [
    "INSTRUMENT_LOADERS",
    "LOADERS",
    "SNIFFERS",
    "UnrecognizedInstrumentError",
    "detect_instrument",
    "guess_instrument",
    "identify_instrument",
    "load_file",
    "require_identified_instrument",
    "sniff_instrument",
    "supported_instrument_message",
    "supported_instrument_names",
    "supported_instruments",
    "is_APS_file",
    "is_Aethalometer_file",
    "is_CPC_file",
    "is_DiSCmini_file",
    "is_DiSCmini_raw_file",
    "is_DustTrak_file",
    "is_ELPI_file",
    "is_FMPS_file",
    "is_Fourtec_file",
    "is_Grimm_file",
    "is_NS_file",
    "is_OPCN3_file",
    "is_OPS_file",
    "is_Partector_file",
    "is_Ranger_file",
    "is_SMPS_file",
]
