"""Registry and automatic instrument identification for the GUI.

Automatic identification order:

1. Try to recognize the instrument from file content.
2. Fall back to filename naming convention.
3. If both fail, raise a user-facing error.

The GUI should not silently fall back to the first combo-box item, because that
can send unknown files to the wrong loader, e.g. CPC.
"""

from __future__ import annotations

from pathlib import Path
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
from ..loaders.Common import _detect_delimiter


class UnrecognizedInstrumentError(ValueError):
    """Raised when aerosoltools cannot identify an instrument file."""


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

# Lower-case filename substrings -> display name.
#
# Checked after content sniffing. Keep longer/more specific names before shorter
# names where overlap is possible.
_FILENAME_HINTS: list[tuple[str, str]] = [
    ("aethalometer", "Aethalometer"),
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
    ("nanoscan", "NanoScan (NS)"),
    ("_ns", "NanoScan (NS)"),
    ("smps", "SMPS"),
    ("ops", "OPS"),
    ("cpc", "CPC"),
]


def guess_instrument(filename: str) -> Optional[str]:
    """Guess the instrument display name from a file name.

    Args:
        filename:
            Path or base name of the data file.

    Returns:
        The matching key in :data:`LOADERS`, or ``None`` if no hint matches.
    """
    name = filename.lower()
    for needle, display in _FILENAME_HINTS:
        if needle in name:
            return display
    return None


def supported_instrument_names() -> list[str]:
    """Return supported instrument names for user-facing error messages."""
    return list(LOADERS.keys())


def supported_instrument_message(path: str | Path) -> str:
    """Return a helpful error message for unknown instrument files."""
    names = ", ".join(supported_instrument_names())
    return (
        "Could not recognize the instrument type from the file content or "
        "filename.\n\n"
        f"File:\n{path}\n\n"
        "Please rename the file so the filename contains one of the supported "
        f"instrument names:\n\n{names}"
    )


def _read_text_head(path: str | Path, max_lines: int = 120) -> list[str]:
    """Read the first lines of a text-like file using tolerant encodings."""
    path = Path(path)

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"):
        try:
            lines: list[str] = []
            with open(path, "r", encoding=encoding, errors="replace") as fh:
                for _ in range(max_lines):
                    line = fh.readline()
                    if line == "":
                        break
                    lines.append(line)
            return lines
        except Exception:
            continue

    return []


def _head_text(path: str | Path, max_lines: int = 120) -> str:
    """Return lower-case text from the beginning of a file."""
    return "\n".join(_read_text_head(path, max_lines=max_lines)).lower()


def _split_first_line(path: str | Path) -> list[str]:
    """Split the first line using the detected delimiter when possible."""
    try:
        encoding, delimiter = _detect_delimiter(path)
        lines = _read_text_head(path, max_lines=1)
        if not lines:
            return []
        return [x.strip() for x in lines[0].strip().split(delimiter)]
    except Exception:
        lines = _read_text_head(path, max_lines=1)
        if not lines:
            return []

        line = lines[0].strip()
        for delimiter in ("\t", ",", ";", "|"):
            if delimiter in line:
                return [x.strip() for x in line.split(delimiter)]

        return [line]


def is_ELPI_file(path: str | Path) -> bool:
    """Detect ELPI/ELPI+ raw .dat files and Dekati software exports."""
    text = _head_text(path, max_lines=160)

    required_markers = (
        "d50values(um)=",
        "calculateddi(um)=",
        "calculatedmoment=",
        "calculatedtype=",
    )

    return all(marker in text for marker in required_markers) and "[data]" in text


def is_Aethalometer_file(path: str | Path) -> bool:
    """Detect Aethalometer / microAeth exports."""
    text = _head_text(path, max_lines=40)

    return (
        "date / time local" in text
        and "serial number" in text
        and "optical config" in text
    )


def is_SMPS_file(path: str | Path) -> bool:
    """Detect TSI SMPS software exports."""
    text = _head_text(path, max_lines=60)

    return (
        "classifier model" in text
        and "detector model" in text
        and "sample #" in text
        and "start time" in text
    )


def is_NS_file(path: str | Path) -> bool:
    """Detect TSI NanoScan SMPS exports."""
    text = _head_text(path, max_lines=40)

    return (
        "date time" in text
        and "particle density (g/cc)" in text
        and "total conc" in text
        and "file index" in text
    )


def is_OPS_file(path: str | Path) -> bool:
    """Detect TSI OPS exports."""
    first = _split_first_line(path)
    if not first:
        return False

    first_cell = first[0].strip().lower()

    # These are the two branches expected by Load_OPS_file.
    return first_cell in {"sample file", "instrument name"}


def is_Grimm_file(path: str | Path) -> bool:
    """Detect Grimm software or direct instrument exports."""
    first = _split_first_line(path)
    if not first:
        return False

    first_cell = first[0].strip().lower()

    return first_cell == "<header>" or "file name" in first_cell


def is_DustTrak_file(path: str | Path) -> bool:
    """Detect DustTrak DRX exports."""
    text = _head_text(path, max_lines=80)

    return (
        "elapsed time [s]" in text
        and "pm1 [mg/m3]" in text
        and "pm2.5 [mg/m3]" in text
        and "total [mg/m3]" in text
    )


def is_Partector_file(path: str | Path) -> bool:
    """Detect Partector LDSA exports."""
    text = _head_text(path, max_lines=50)

    return (
        "start:" in text
        and "ldsa" in text
        and "tem" in text
        and "flow" in text
    )


def is_DiSCmini_file(path: str | Path) -> bool:
    """Detect DiSCmini converted exports."""
    text = _head_text(path, max_lines=50)

    has_time = "timestamp" in text or "\ntime" in text or "\ttime" in text
    has_core = "number" in text and ("ldsa" in text or "diameter" in text)

    return has_time and has_core


def is_FMPS_file(path: str | Path) -> bool:
    """Detect FMPS exports.

    FMPS files are less self-describing than ELPI/SMPS, so this sniffer is
    intentionally conservative.
    """
    lines = _read_text_head(path, max_lines=20)
    if len(lines) < 16:
        return False

    blob = "\n".join(lines).lower()
    if "fmps" in blob:
        return True

    possible_type_line = lines[12].lower() if len(lines) > 12 else ""
    possible_bin_line = lines[13].lower() if len(lines) > 13 else ""
    possible_time_line = lines[14].lower() if len(lines) > 14 else ""

    has_dtype = any(
        token in possible_type_line
        for token in ("dn", "co", "su", "vo", "ma", "raw")
    )
    has_bins = any(
        size in possible_bin_line
        for size in ("5.6", "10", "15", "20", "30")
    )
    has_time = "elapsed" in possible_time_line or "time" in possible_time_line

    return has_dtype and has_bins and has_time


def is_Fourtec_file(path: str | Path) -> bool:
    """Detect Fourtec text/CSV exports.

    XLS/XLSX files are not sniffed by content here. They can still be detected
    by filename fallback, e.g. a filename containing "fourtec".
    """
    text = _head_text(path, max_lines=40)

    return (
        "internal digital temperature" in text
        and "internal rh" in text
        and "date" in text
        and "time" in text
    )


def is_OPCN3_file(path: str | Path) -> bool:
    """Detect Alphasense OPC-N3 exports."""
    text = _head_text(path, max_lines=10)
    first = _split_first_line(path)
    lower_first = [x.lower() for x in first]

    # Newer format used by the loader: first line ends with "OPC".
    if lower_first and lower_first[-1] == "opc":
        return True

    has_bins = any(col.startswith("bin") for col in lower_first)
    has_pm = any(col in lower_first for col in ("pm1", "pm2.5", "pm10"))
    has_env = any(
        col in lower_first
        for col in ("temp (c)", "rh (%)", "period (s)", "flowrate")
    )

    return (has_bins and has_pm and has_env) or "opc-n3" in text or "opcn3" in text


def is_CPC_file(path: str | Path) -> bool:
    """Conservative CPC detector.

    CPC is not as uniquely marked as ELPI/SMPS/etc., so this sniffer is checked
    late and only returns True when recognizable CPC markers are present.
    """
    text = _head_text(path, max_lines=40)

    full_format = (
        "sample #" in text
        and "start date" in text
        and "start time" in text
        and "[1] conc" in text
    )

    focused_format = (
        "model" in text
        and "serial" in text
        and "sample interval" in text
        and ("concentration" in text or "conc" in text)
    )

    return full_format or focused_format


# Display name -> content sniffer.
#
# Order matters:
# - Strong/distinctive formats first.
# - Weak/generic signatures such as CPC last.
SNIFFERS: dict[str, Callable[[str | Path], bool]] = {
    "ELPI": is_ELPI_file,
    "Aethalometer": is_Aethalometer_file,
    "SMPS": is_SMPS_file,
    "NanoScan (NS)": is_NS_file,
    "OPS": is_OPS_file,
    "Grimm": is_Grimm_file,
    "DustTrak": is_DustTrak_file,
    "Partector": is_Partector_file,
    "DiSCmini": is_DiSCmini_file,
    "FMPS": is_FMPS_file,
    "Fourtec": is_Fourtec_file,
    "OPC-N3": is_OPCN3_file,
    "CPC": is_CPC_file,
}


def sniff_instrument(path: str | Path) -> Optional[str]:
    """Guess instrument from file content only.

    Returns:
        Matching key in LOADERS, or None if content is not recognized.
    """
    matches: list[str] = []

    for display_name, sniffer in SNIFFERS.items():
        try:
            if sniffer(path):
                matches.append(display_name)
        except Exception:
            # A sniffer must never crash GUI loading.
            continue

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    # Tie-break using filename if possible.
    filename_guess = guess_instrument(Path(path).name)
    if filename_guess in matches:
        return filename_guess

    # Otherwise return the strongest match according to SNIFFERS order.
    return matches[0]


def identify_instrument(path: str | Path) -> Optional[str]:
    """Guess instrument using the desired automatic order.

    Order:
        1. File-content sniffer
        2. Filename naming convention
        3. None

    The GUI should show an error when this returns None. It should not silently
    fall back to the combo-box selection.
    """
    return sniff_instrument(path) or guess_instrument(Path(path).name)


def require_identified_instrument(path: str | Path) -> str:
    """Return identified instrument or raise a user-facing error."""
    instrument = identify_instrument(path)

    if instrument is None:
        raise UnrecognizedInstrumentError(supported_instrument_message(path))

    return instrument