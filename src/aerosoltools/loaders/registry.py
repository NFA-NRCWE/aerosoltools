"""Public instrument registry and automatic file-format detection.

Maps canonical instrument names to their loader functions, and identifies an
unknown file from its content (falling back to its filename). This is the API a
script uses to load a file without knowing its instrument up front::

    import aerosoltools as at
    data = at.load_file("mystery_export.csv")      # auto-detects the instrument
    which = at.detect_instrument("mystery_export.csv")   # -> "OPS" or None

Automatic identification order:

1. Recognize the instrument from file **content** (the ``_SNIFFERS``).
2. Fall back to the **filename** naming convention (``_FILENAME_HINTS``).
3. If both fail, :func:`load_file` raises :class:`InstrumentDetectionError`
   (detection never silently guesses, so an unknown file is never sent to the
   wrong loader, e.g. CPC).

The GUI builds its file-open flow on top of this registry rather than owning a
parallel copy of the detection logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .aethalometer import load_aethalometer_file
from .aps import load_aps_file
from .cpc import load_cpc_file
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
from .smps import load_smps_file
from .support.exceptions import InstrumentDetectionError
from .support.parsing import _detect_delimiter

#: Canonical instrument name -> loader function. Order is preserved for the GUI
#: combo box; the keys are the values :func:`detect_instrument` returns.
INSTRUMENT_LOADERS: dict[str, Callable] = {
    "CPC": load_cpc_file,
    "DiSCmini": load_discmini_file,
    "DiSCmini (raw)": load_discmini_raw_file,
    "ELPI": load_elpi_file,
    "FMPS": load_fmps_file,
    "Fourtec": load_fourtec_file,
    "Grimm": load_grimm_file,
    "NanoScan (NS)": load_ns_file,
    "OPC-N3": load_opcn3_file,
    "OPS": load_ops_file,
    "Partector": load_partector_file,
    "SMPS": load_smps_file,
    "Aethalometer": load_aethalometer_file,
    "DustTrak": load_dusttrak_file,
    "Ranger": load_ranger_file,
    "APS": load_aps_file,
}

#: Back-compat alias for the GUI (which historically called this ``LOADERS``).
LOADERS = INSTRUMENT_LOADERS

# Lower-case filename substrings -> display name.
#
# Checked after content sniffing. Keep longer/more specific names before shorter
# names where overlap is possible.
_FILENAME_HINTS: list[tuple[str, str]] = [
    ("aps", "APS"),
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
    ("ranger", "Ranger"),
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
    """Detect TSI OPS exports.

    The first cell is "Sample File" or "Instrument Name" (the two branches
    :func:`load_ops_file` expects). TSI **CPC** exports *also* start with
    "Sample File", so an OPS-specific marker (the "Optical Particle Sizer" name
    or an instrument-model line) is additionally required to avoid grabbing CPC
    files (which are model 3007, with no such marker).
    """
    first = _split_first_line(path)
    if not first:
        return False

    first_cell = first[0].strip().lower()
    if first_cell not in {"sample file", "instrument name"}:
        return False

    text = _head_text(path, max_lines=20)
    return (
        "optical particle sizer" in text
        or "instrument model" in text
        or "model number" in text
    )


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

    return "start:" in text and "ldsa" in text and "tem" in text and "flow" in text


def is_DiSCmini_file(path: str | Path) -> bool:
    """Detect DiSCmini converted (vendor-processed) exports."""
    text = _head_text(path, max_lines=50)

    has_time = "timestamp" in text or "\ntime" in text or "\ttime" in text
    has_core = "number" in text and ("ldsa" in text or "diameter" in text)

    return has_time and has_core


def is_DiSCmini_raw_file(path: str | Path) -> bool:
    """Detect **raw** DiSCmini instrument files (pre vendor conversion).

    Raw files carry the calibration/offset header block and a data table of
    electrometer currents (``Diffusion``/``Filter``) rather than the processed
    ``Number``/``Size``/``LDSA`` columns, so the processed-export sniffer
    (:func:`is_DiSCmini_file`) does not match them.
    """
    text = _head_text(path, max_lines=15)

    return "caldata" in text and "diffusion" in text and "filter" in text


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
        token in possible_type_line for token in ("dn", "co", "su", "vo", "ma", "raw")
    )
    has_bins = any(
        size in possible_bin_line for size in ("5.6", "10", "15", "20", "30")
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
    # PM columns carry units, e.g. "PM1(ug/m3)" / "PM2.5(ug/m3)", so match by
    # prefix rather than exact name.
    has_pm = any(col.startswith(("pm1", "pm2.5", "pm10")) for col in lower_first)
    has_env = any(
        col.startswith(("temp", "rh", "period", "flowrate")) for col in lower_first
    )

    return (has_bins and has_pm and has_env) or "opc-n3" in text or "opcn3" in text


def is_APS_file(path: str | Path) -> bool:
    """Detect TSI APS AIM exports (aerodynamic-only or correlated).

    Distinctive: the AIM header carries "Lower Channel Bound" / "Upper Channel
    Bound" lines, and the data header row names an "Aerodynamic Diameter" or
    "Correlated" column.
    """
    text = _head_text(path, max_lines=8)
    return "lower channel bound" in text and (
        "aerodynamic diameter" in text or "correlated" in text
    )


def is_Ranger_file(path: str | Path) -> bool:
    """Detect Ranger gas / PM sensor CSV exports.

    The Ranger uses interchangeable heads (Cl₂, NO₂, PM, …), so a chlorine
    column is not always present. Two markers are distinctive and appear near
    the top of every export regardless of head: the opening
    "Ranger Serial Number" line, and the sensor status legend
    ("z:zerocal") carried by every header row.
    """
    text = _head_text(path, max_lines=8)
    return "ranger serial number" in text or "z:zerocal" in text


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

    # "Direct" CPC export: starts like a TSI "Sample File" report but has the
    # single count-concentration channel column ("Concentration (#/cm3)").
    direct_format = (
        "concentration (#/cm" in text and "sample #" in text and "start time" in text
    )

    return full_format or focused_format or direct_format


# Display name -> content sniffer.
#
# Order matters:
# - Strong/distinctive formats first.
# - Weak/generic signatures such as CPC last.
SNIFFERS: dict[str, Callable[[str | Path], bool]] = {
    "APS": is_APS_file,
    "ELPI": is_ELPI_file,
    "Aethalometer": is_Aethalometer_file,
    "SMPS": is_SMPS_file,
    "NanoScan (NS)": is_NS_file,
    "OPS": is_OPS_file,
    "Grimm": is_Grimm_file,
    "DustTrak": is_DustTrak_file,
    "Partector": is_Partector_file,
    "DiSCmini (raw)": is_DiSCmini_raw_file,
    "DiSCmini": is_DiSCmini_file,
    "FMPS": is_FMPS_file,
    "Fourtec": is_Fourtec_file,
    "OPC-N3": is_OPCN3_file,
    "Ranger": is_Ranger_file,
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
        raise InstrumentDetectionError(supported_instrument_message(path))

    return instrument


# -- Public entry points ------------------------------------------------------
#: Detect the instrument from a file's content, then its filename. Returns the
#: canonical instrument name (a key of :data:`INSTRUMENT_LOADERS`) or ``None``.
detect_instrument = identify_instrument
#: Canonical names of every instrument this registry can load.
supported_instruments = supported_instrument_names


def load_file(path: str | Path, instrument: str | None = None, **kwargs):
    """Load an instrument file, auto-detecting the instrument when not given.

    The single public entry point for "load this file" without wiring up the
    right loader by hand — the same detection the GUI's file-open uses.

    Args:
        path: Path to the instrument export file.
        instrument: Canonical instrument name (a key of
            :data:`INSTRUMENT_LOADERS`). When ``None`` (default), the instrument
            is detected from the file content, then its filename.
        **kwargs: Forwarded to the selected loader (e.g. ``extra_data=True``).

    Returns:
        The loaded aerosol object (or a list of them for multi-head files).

    Raises:
        InstrumentDetectionError: If ``instrument`` is ``None`` and detection
            fails, or if a given ``instrument`` name is not supported.
    """
    if instrument is None:
        instrument = detect_instrument(path)
        if instrument is None:
            raise InstrumentDetectionError(supported_instrument_message(path))
    try:
        loader = INSTRUMENT_LOADERS[instrument]
    except KeyError:
        raise InstrumentDetectionError(
            f"Unknown instrument {instrument!r}. Supported instruments: "
            f"{', '.join(supported_instruments())}."
        ) from None
    return loader(path, **kwargs)
