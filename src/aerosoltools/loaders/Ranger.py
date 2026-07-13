"""Loader for Ranger gas / PM sensor exports (Cl₂, NO₂, PM, …)."""

from __future__ import annotations

import re
from typing import List, Union

import pandas as pd

from ..aerosol1d import Aerosol1D
from ..gas1d import Gas1D
from .Common import _detect_delimiter

###############################################################################

# Columns that describe the row rather than a measured quantity.
_META_COLS = {"utc time", "local time", "locationid", "location id"}

# Pretty display names for the gases the Ranger interchangeable heads report.
# Anything not listed falls back to the raw column name (e.g. "CO").
_GAS_NAMES = {
    "CL2": "Cl₂",
    "NO2": "NO₂",
    "SO2": "SO₂",
    "O3": "O₃",
    "CO": "CO",
    "CO2": "CO₂",
    "H2S": "H₂S",
    "NH3": "NH₃",
    "HCL": "HCl",
    "HF": "HF",
}

RangerObject = Union[Gas1D, Aerosol1D]


def _is_meta_col(name: str) -> bool:
    """True for the timestamp / location / status columns (non-measurements)."""
    n = name.strip().lower()
    return n in _META_COLS or n.startswith("status")


def _normalize_unit(unit_raw: str) -> str:
    """Map a raw header unit token to the library's display convention."""
    u = unit_raw.strip().lower()
    return {
        "ppm": "ppm",
        "ppb": "ppb",
        "ug/m3": "µg/m³",
        "µg/m3": "µg/m³",
        "mg/m3": "mg/m³",
    }.get(u, unit_raw.strip())


def _parse_measure_column(col: str) -> tuple[str, str]:
    """Split a measurement header like ``"CL2 (PPM)"`` into ``(name, unit)``.

    Args:
        col: Raw header cell, e.g. ``"PM2.5 (ug/m3)"`` or ``"CL2 (PPM)"``.

    Returns:
        A ``(name, unit)`` pair, e.g. ``("PM2.5", "µg/m³")`` or
        ``("CL2", "ppm")``. If no unit is present the unit is an empty string.
    """
    m = re.match(r"^\s*(.*?)\s*\(([^)]*)\)\s*$", col)
    if m:
        return m.group(1).strip(), _normalize_unit(m.group(2))
    return col.strip(), ""


def _gas_display_name(name: str) -> str:
    """Return the subscripted display name for a gas column (fallback: raw)."""
    return _GAS_NAMES.get(name.strip().upper(), name.strip())


def _read_segments(file: str) -> tuple[str, str, list[dict]]:
    """Split a Ranger export into header-delimited segments.

    A Ranger file starts with a serial-number line and then contains one or
    more *segments*, each introduced by its own ``"UTC time, …"`` header row
    (the interchangeable measurement heads write a fresh header whenever the
    measured component changes). This helper walks the raw lines once and
    returns every segment as ``{"header": [...], "rows": [[...], ...]}``.

    Args:
        file: Path to the Ranger ``.csv`` export.

    Returns:
        ``(encoding, serial, segments)`` where ``segments`` preserves file
        order.

    Raises:
        ValueError: If the file contains no recognizable header row.
    """
    encoding, delimiter = _detect_delimiter(file)

    with open(file, "r", encoding=encoding, errors="replace") as fh:
        raw_lines = fh.read().splitlines()

    serial = ""
    segments: list[dict] = []
    current: dict | None = None

    for line in raw_lines:
        if not line.strip():
            continue
        fields = [c.strip() for c in line.split(delimiter)]
        first = fields[0].strip().lower()

        if first.startswith("ranger serial number"):
            if len(fields) > 1 and fields[1].strip():
                serial = fields[1].strip()
            current = None
            continue

        if first == "utc time":
            current = {"header": fields, "rows": []}
            segments.append(current)
            continue

        if current is not None:
            current["rows"].append(fields)

    if not segments:
        raise ValueError(
            "No header row (e.g. 'UTC time, Local time, …') found in the "
            "Ranger file; is this a valid Ranger export?"
        )

    return encoding, serial, segments


def _segment_frame(segment: dict) -> pd.DataFrame:
    """Build a padded DataFrame for one segment using its own header."""
    header = segment["header"]
    width = len(header)
    # Normalise every data row to the header width (Ranger's trailing status
    # column is often empty, and stray rows may be short or long).
    rows = [
        (row + [""] * width)[:width] if len(row) != width else row
        for row in segment["rows"]
    ]
    return pd.DataFrame(rows, columns=header)


def _build_object(
    frame: pd.DataFrame,
    measure_cols: list[str],
    serial: str,
    extra_data: bool,
) -> RangerObject:
    """Turn a per-component frame into a Gas1D (gas) or AerosolAlt (PM)."""
    time_col = "Local time" if "Local time" in frame.columns else frame.columns[0]
    datetimes = pd.to_datetime(frame[time_col], errors="coerce")

    if len(measure_cols) == 1:
        # Single-channel head (a gas such as Cl₂ / NO₂) -> Gas1D.
        name, unit = _parse_measure_column(measure_cols[0])
        display = _gas_display_name(name)

        out = pd.DataFrame(
            {
                "Datetime": datetimes,
                "Concentration": pd.to_numeric(frame[measure_cols[0]], errors="coerce"),
            }
        )
        obj: RangerObject = Gas1D(out.dropna(subset=["Datetime"]).copy())
        obj._meta["unit"] = unit or "ppm"
        obj._meta["dtype"] = display
    else:
        # Multi-channel head (PM fractions) -> plain multi-channel Aerosol1D,
        # one column per fraction (mirrors the DustTrak loader).
        names, units = zip(*(_parse_measure_column(c) for c in measure_cols))
        out = pd.DataFrame({"Datetime": datetimes})
        for src, name in zip(measure_cols, names):
            out[name] = pd.to_numeric(frame[src], errors="coerce")

        obj = Aerosol1D(out.dropna(subset=["Datetime"]).copy())
        obj._meta["unit"] = {n: u for n, u in zip(names, units)}
        obj._meta["dtype"] = {n: "dM" for n in names}
        display = "PM"

    obj._meta["instrument"] = "Ranger"
    obj._meta["serial_number"] = serial
    obj._meta["measurement"] = display

    if extra_data:
        keep = [c for c in ("UTC time", "LocationID") if c in frame.columns]
        status_col = next(
            (c for c in frame.columns if c.strip().lower().startswith("status")), None
        )
        if status_col:
            keep.append(status_col)
        extra = frame[keep].copy()
        extra.insert(0, "Datetime", datetimes.values)
        extra = extra.dropna(subset=["Datetime"]).set_index("Datetime")
        obj._extra_data = extra
        obj._raw_extra_data = extra.copy()

    return obj


###############################################################################


def Load_Ranger_file(
    file: str, extra_data: bool = False
) -> Union[RangerObject, List[RangerObject]]:
    """Description:
        Load a Ranger sensor CSV export. The Ranger uses interchangeable
        measurement heads (e.g. Cl₂, NO₂ or PM), and a single export may
        contain several of them: each head writes its own ``"UTC time, …"``
        header row followed by its readings. This loader splits the file on
        those headers, groups the segments by measured component, and returns
        one aerosol object per component.

    Args:
        file (str):
            Path to the Ranger ``.csv`` export.
        extra_data (bool, optional):
            If ``True``, the non-core columns (UTC time, location id and the
            instrument status flag) are kept in each returned object's
            :attr:`~aerosoltools.aerosol1d.Aerosol1D.extra_data`. Defaults to
            ``False``.

    Returns:
        Aerosol1D | AerosolAlt | list:
            - A **gas** head (a single ``"<GAS> (PPM)"`` column such as
              ``CL2`` or ``NO2``) becomes an :class:`Aerosol1D` with a
              ``"Total_conc"`` column, ``unit`` (``"ppm"``/``"ppb"``) and
              ``dtype`` set to the subscripted gas name (e.g. ``"Cl₂"``).
            - A **PM** head (several ``"PMx (ug/m3)"`` columns) becomes an
              :class:`AerosolAlt` with one column per fraction, per-channel
              ``unit`` (``"µg/m³"``) and ``dtype`` (``"dM"``).
            - If the file holds **more than one** component, a list of such
              objects is returned (in first-appearance order). A file with a
              single component returns that object directly.

            Every object also carries ``instrument`` (``"Ranger"``),
            ``serial_number`` and ``measurement`` (the component label) in its
            metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        ValueError:
            If the file contains no recognizable header row.

    Notes:
        Detailed description:
            The Ranger exports a small self-describing CSV: a serial-number
            line, then one or more header/data blocks. A header looks like::

                UTC time, Local time, LocationID, CL2 (PPM), Status(...)

            The loader uses the **local time** column as the time base (so the
            series aligns on wall-clock time with co-located instruments) and
            treats every column that is not UTC time / Local time / LocationID /
            Status as a measurement. Repeated blocks of the same component
            (e.g. a Cl₂ head that was paused and resumed) are concatenated into
            one continuous, de-duplicated, time-sorted series.

            Because the results are plain :class:`Aerosol1D` / :class:`AerosolAlt`
            objects, all the usual time-series operations — cropping,
            resampling, smoothing, activity marking and exposure summaries —
            work unchanged.

    Examples:
        .. code-block:: python

            import aerosoltools as at

            result = at.load_ranger_file("Ranger_2605-1000088-A.csv")

            # Single component -> one object; multiple -> list of objects.
            objs = result if isinstance(result, list) else [result]
            for obj in objs:
                print(obj.metadata["measurement"], obj.data.shape)
    """
    _encoding, serial, segments = _read_segments(file)

    # Group segments by their measurement columns so repeated blocks of the
    # same head form one continuous series. Preserve first-appearance order.
    groups: "dict[tuple[str, ...], dict]" = {}
    for segment in segments:
        measure_cols = [c for c in segment["header"] if not _is_meta_col(c)]
        if not measure_cols:
            continue
        key = tuple(measure_cols)
        frame = _segment_frame(segment)
        if key in groups:
            groups[key]["frames"].append(frame)
        else:
            groups[key] = {"cols": measure_cols, "frames": [frame]}

    if not groups:
        raise ValueError(
            "No measurement columns found in the Ranger file (only timestamp / "
            "status columns were present)."
        )

    objects: list[RangerObject] = []
    for group in groups.values():
        frame = pd.concat(group["frames"], ignore_index=True)
        obj = _build_object(frame, group["cols"], serial, extra_data)
        # Drop duplicate timestamps and keep the series chronological.
        obj._data = obj._data[~obj._data.index.duplicated(keep="first")].sort_index()
        obj._raw_data = obj._data.copy()
        objects.append(obj)

    return objects[0] if len(objects) == 1 else objects
