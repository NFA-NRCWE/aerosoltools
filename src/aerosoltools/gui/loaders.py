"""Registry mapping instrument names to loader functions for the GUI.

The GUI does not auto-detect instruments from file content. Instead it offers
the user a named list of loaders (this registry) and tries to guess a sensible
default from the file name via :func:`guess_instrument`.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..loaders import (
    Load_Aethalometer_file,
    Load_CPC_file,
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

# Display name -> loader function. Order is preserved in the combo box.
LOADERS: dict[str, Callable] = {
    "CPC": Load_CPC_file,
    "DiSCmini": Load_DiSCmini_file,
    "ELPI": Load_ELPI_file,
    "FMPS": Load_FMPS_file,
    "Fourtec": Load_Fourtec_file,
    "Grimm": Load_Grimm_file,
    "NanoScan (NS)": Load_NS_file,
    "OPC-N3": Load_OPCN3_file,
    "OPS": Load_OPS_file,
    "Partector": Load_Partector_file,
    "SMPS": Load_SMPS_file,
    "Aethalometer": Load_Aethalometer_file,
    "DustTrak": Load_DustTrak_file,
}

# Lower-case filename substrings -> display name (checked in order).
_FILENAME_HINTS: list[tuple[str, str]] = [
    ("aeth", "Aethalometer"),
    ("dusttrak", "DustTrak"),
    ("discmini", "DiSCmini"),
    ("disc", "DiSCmini"),
    ("elpi", "ELPI"),
    ("fmps", "FMPS"),
    ("fourtec", "Fourtec"),
    ("grimm", "Grimm"),
    ("opcn3", "OPC-N3"),
    ("opc-n3", "OPC-N3"),
    ("partector", "Partector"),
    ("smps", "SMPS"),
    ("nanoscan", "NanoScan (NS)"),
    ("ops", "OPS"),
    ("cpc", "CPC"),
    ("_ns", "NanoScan (NS)"),
]


def guess_instrument(filename: str) -> Optional[str]:
    """Guess the instrument display name from a file name.

    Args:
        filename: Path or base name of the data file.

    Returns:
        The matching key in :data:`LOADERS`, or ``None`` if no hint matches.
    """
    name = filename.lower()
    for needle, display in _FILENAME_HINTS:
        if needle in name:
            return display
    return None
