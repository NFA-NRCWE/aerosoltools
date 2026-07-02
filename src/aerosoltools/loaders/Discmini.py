from __future__ import annotations

import datetime as dt
import re

import pandas as pd

from ..aerosolalt import AerosolAlt
from .Common import _detect_delimiter

###############################################################################


def _normalize_serial(raw: str | None) -> str | None:
    """Normalize a DiSCmini serial number so it matches across export formats.

    DiSCmini files write the serial inconsistently across firmware / software
    versions — sometimes with an ``"SN"`` prefix (``"SN101923"``) and sometimes
    as bare digits (``"101670"``). Combining datasets keys on the serial, so a
    prefix mismatch makes two files from the *same* instrument look like
    different instruments. This strips a leading ``SN`` (case-insensitive) and
    any surrounding punctuation/whitespace so the stored serial is just the
    identifier itself.

    Args:
        raw: The serial token as read from the file, or ``None``.

    Returns:
        The normalized serial (e.g. ``"101923"``), or ``None`` when ``raw`` is
        empty/``None``.
    """
    if raw is None:
        return None
    s = str(raw).strip().strip("[](){}").strip()
    # Drop a leading "SN" (optionally followed by a separator), case-insensitive.
    s = re.sub(r"^[sS][nN][\s:_-]*", "", s).strip()
    return s or None


def _extract_serial_and_firmware(
    header_lines: list[str],
) -> tuple[str | None, str | None]:
    """Pull the (normalized) serial number and firmware/software version.

    Handles the header variants seen across DiSCmini exports:

    * Vendor-processed files:
      ``[Data recorded with testo DiSCmini SN101923 running firmware 3,42]``
    * Raw instrument files: a ``CalData:`` line whose first token is the serial,
      e.g. ``CalData: SN101923   0.28 ...`` or ``CalData: 101670   6.02 ...``.
    * Older exports: any line containing ``"serial"``.

    Args:
        header_lines: The first handful of lines of the file.

    Returns:
        A ``(serial, firmware)`` tuple; either element may be ``None`` when it
        cannot be found. ``serial`` is normalized via :func:`_normalize_serial`.
    """
    serial = None
    firmware = None
    for ln in header_lines:
        low = ln.lower()
        # Processed header: "...DiSCmini <serial> running firmware <ver>..."
        m = re.search(
            r"disc[m]?ini\s+(\S+)\s+running\s+firmware\s+([\d.,]+)", ln, re.IGNORECASE
        )
        if m:
            serial = serial or m.group(1)
            firmware = firmware or m.group(2).replace(",", ".")
            continue
        # Raw calibration line: "CalData: <serial>  <cal values...>"
        m = re.search(r"caldata:\s*(\S+)", ln, re.IGNORECASE)
        if m and serial is None:
            serial = m.group(1)
            continue
        # Software/firmware version on the first raw line ("... SW-Ver 3.42").
        m = re.search(r"sw-?ver\s*([\d.,]+)", low)
        if m and firmware is None:
            firmware = m.group(1).replace(",", ".")
        # Legacy fallback: an explicit "serial" line.
        if serial is None and "serial" in low:
            toks = ln.strip().replace(",", " ").split()
            if toks:
                serial = toks[-1]
    return _normalize_serial(serial), firmware


###############################################################################


def Load_DiSCmini_file(file: str, extra_data: bool = False) -> AerosolAlt:
    """Description:
        Load a converted DiSCmini export file and return total number
        concentration, mean size, and LDSA as an :class:`AerosolAlt` time
        series.

    Args:
        file (str):
            Path to the DiSCmini ``.txt`` file exported and converted by the
            vendor software.
        extra_data (bool, optional):
            If ``True``, columns that are not part of the core time series
            (datetime, total concentration, mean size, LDSA) are stored in
            the returned object's ``.extra_data`` (and ``._raw_extra_data``)
            DataFrames, indexed by ``Datetime``. Defaults to ``False``.

    Returns:
        AerosolAlt:
            An :class:`~aerosoltools.aerosolalt.AerosolAlt` instance with
            the loaded data

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened. Check the path
            and file permissions.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            :func:`_detect_delimiter`. This usually indicates a corrupted
            or non-text file.
        ValueError:
            If timestamps or header-derived start date/time strings are in
            an unsupported format and cannot be parsed. Check that the file
            was exported using a supported DiSCmini software version and
            that the regional date/time settings are compatible.
        KeyError:
            If expected columns such as ``"TimeStamp"``/``"Time"``,
            ``"Number"``, or size/LDSA columns are missing or renamed in an
            unexpected way. Verify that the file is an unmodified DiSCmini
            export.
        Exception:
            If neither direct parsing nor reconstruction of the datetime
            column succeeds, or if a valid ``"Datetime"`` column cannot be
            produced after all attempts. The raised message will typically
            indicate that the file has not been converted correctly or that
            the datetime format is unsupported.

    Notes:
        Detailed description:
            ``Load_DiSCmini_file`` is designed for DiSCmini data that have
            already been converted by the vendor software to a tab-delimited
            text format.

            Internally, the function:

            - Reads a subset of columns (up to 7) as strings to robustly
              handle locale-specific decimal separators and whitespace.
            - Normalizes column names.
            - Attempts to parse the ``"Datetime"`` column using two common
              formats:

              - ``"%d-%b-%Y %H:%M:%S"`` (e.g. ``01-Oct-2023 12:00:00``)
              - ``"%d-%m-%Y %H:%M:%S"`` (e.g. ``01-10-2023 12:00:00``)

            - If both direct parses fail, it falls back to reconstructing
              absolute time from header information:

              - Searches the header for lines containing ``"start date:"``
                and ``"start time:"``.
              - Parses those as a start datetime.
              - Interprets the ``"Datetime"`` column as elapsed seconds
                since that start time, then creates a real timestamp series.

            - Normalizes numeric fields by:

              - Replacing commas with dots as decimal separators.
              - Removing extraneous whitespace.
              - Coercing invalid entries to ``NaN``.

            - Extracts and **normalizes** the serial number (and firmware
              version) from the header via
              :func:`_extract_serial_and_firmware`, stripping any ``"SN"``
              prefix so files from the same instrument match regardless of the
              export's serial-format convention.

            - Builds the core :class:`AerosolAlt` object from the subset of
              columns that includes

              - ``"Datetime"`` — measurement timestamps.
              - ``"Total_conc"`` — particle number concentration (cm⁻³).
              - ``"Size"`` — mean particle diameter (nm).
              - ``"LDSA"`` — lung-deposited surface area (nm²/cm³).

            - Populates the ``.meta`` dictionary with, e.g.:

              - ``"instrument"`` — set to ``"DiSCmini"``.
              - ``"serial_number"`` — serial number parsed from the header.
              - ``"unit"`` — mapping of variable names to units.
              - ``"dtype"`` — mapping of variable names to data types
                (e.g. ``"dN"`` for number concentration, ``"dS"`` for LDSA).

            - Optionally collects all remaining non-core columns into
              ``.extra_data`` (and ``._raw_extra_data``) when
              ``extra_data=True``.

    Examples:
        A typical workflow is to convert a DiSCmini file using the vendor
        software and then load it for analysis or plotting:

        .. code-block:: python

            import aerosoltools as at

            # Load DiSCmini data with core metrics only
            dm = at.Load_DiSCmini_file("data/discmini_converted.txt")

            # Quick look at total concentration and LDSA
            print(dm.data[["Total_conc", "LDSA"]])

            # List available extra channels
            print(dm_full.extra_data.columns)

            # Plot LDSA over time
            fig, ax = dm.plot_timeseries()
    """
    # Detect encoding + delimiter
    try:
        enc, delim = _detect_delimiter(file, sample_lines=25)  # -> (str, str)
    except Exception as e:
        raise Exception(
            "DiSCmini data has not been converted or delimiter could not be detected."
        ) from e

    # Read first 7 columns as strings; coerce later
    df = pd.read_csv(
        file,
        header=4,
        encoding=enc,
        delimiter="\t",
        usecols=range(0, 7),
        dtype="string",
        na_values=["", "NA", "N/A", "-", "--"],
    )

    # Normalize expected column names
    # Some files use "TimeStamp" (with a separate "Time" column); others use "Time"
    if "TimeStamp" in df.columns:
        # keep "TimeStamp" as datetime-like text; "Time" is redundant in these exports
        if "Time" in df.columns:
            df.drop(columns=["Time"], inplace=True)
        df.rename(
            columns={"TimeStamp": "Datetime", "Number": "Total_conc"}, inplace=True
        )
    else:
        df.rename(columns={"Time": "Datetime", "Number": "Total_conc"}, inplace=True)

    # Parse datetime with two known formats; if both fail, attempt reconstruction from header
    dt_parsed = pd.to_datetime(
        df["Datetime"], format="%d-%b-%Y %H:%M:%S", errors="coerce"
    )
    if dt_parsed.isna().all():
        dt_parsed = pd.to_datetime(
            df["Datetime"], format="%d-%m-%Y %H:%M:%S", errors="coerce"
        )

    if dt_parsed.isna().any():
        # Fallback: reconstruct absolute time from a start date/time in the file header
        try:
            # read minimal header with Python I/O to avoid numpy-encoding stub issues
            with open(file, "r", encoding=enc) as fh:
                header_lines = [next(fh) for _ in range(8)]
            # common patterns:
            #   "[...] start date: YYYY.MM.DD]"
            #   "[...] start time: HH:MM:SS]"
            start_date_line = next(
                (ln for ln in header_lines if "start date:" in ln), None
            )
            start_time_line = next(
                (ln for ln in header_lines if "start time:" in ln), None
            )
            if start_date_line is None or start_time_line is None:
                raise ValueError("Start date/time not found in header.")

            start_date = dt.datetime.strptime(
                start_date_line.split("start date: ")[1].split("]")[0], "%Y.%m.%d"
            )
            start_time = dt.datetime.strptime(
                start_time_line.split("start time: ")[1].split("]")[0], "%H:%M:%S"
            )
            start_dt = start_date + dt.timedelta(
                seconds=start_time.hour * 3600
                + start_time.minute * 60
                + start_time.second
            )

            # When this path is used, the "Datetime" column typically holds elapsed seconds
            # Convert strings like "  12,3" -> "12.3" then to float seconds
            sec = (
                df["Datetime"]
                .fillna("")
                .str.replace(",", ".", regex=False)
                .str.replace(r"\s+", "", regex=True)
            )
            sec_f = pd.to_numeric(sec, errors="coerce")
            if sec_f.isna().all():
                raise ValueError("Elapsed seconds column could not be parsed.")
            dt_parsed = pd.to_datetime(sec_f, unit="s", origin=start_dt)
        except Exception as e:
            raise Exception(
                "Datetime does not match expected format. Ensure file is converted correctly."
            ) from e

    df["Datetime"] = dt_parsed

    # Coerce key numeric columns; accept both commas and dots as decimal separators
    def _to_num(s: pd.Series) -> pd.Series:
        s = (
            s.fillna("")
            .str.replace(",", ".", regex=False)
            .str.replace(r"\s+", "", regex=True)
        )
        return pd.to_numeric(s, errors="coerce")

    # Some exports use "Size" / "LDSA" names consistently
    if "Size" not in df.columns or "LDSA" not in df.columns:
        # Try common alternates if present
        for guess, canonical in [("AvgSize", "Size"), ("LungDepSurfArea", "LDSA")]:
            if guess in df.columns and canonical not in df.columns:
                df.rename(columns={guess: canonical}, inplace=True)

    for col in ["Total_conc", "Size", "LDSA"]:
        if col in df.columns:
            df[col] = _to_num(df[col])

    # Extract the serial number + firmware from the header. The serial is
    # normalized (leading "SN" stripped) so files from the same instrument match
    # regardless of the export's prefix convention (see _extract_serial_and_firmware).
    with open(file, "r", encoding=enc) as fh:
        first_lines = [ln for _, ln in zip(range(10), fh)]
    serial_number, firmware = _extract_serial_and_firmware(first_lines)

    # Build AerosolAlt on the core four columns (order: Datetime, Total_conc, Size, LDSA)
    needed = ["Datetime", "Total_conc", "Size", "LDSA"]
    present = [c for c in needed if c in df.columns]
    if present[:1] != ["Datetime"]:
        raise Exception("Datetime column missing after parsing.")
    DM = AerosolAlt(df[present])

    # Metadata
    DM._meta["instrument"] = "DiSCmini"
    DM._meta["serial_number"] = serial_number
    if firmware is not None:
        DM._meta["firmware"] = firmware
    DM._meta["unit"] = {
        "Total_conc": "cm⁻³",
        "Size": "nm",
        "LDSA": "nm²/cm³",
    }
    DM._meta["dtype"] = {"Total_conc": "dN", "Size": "l", "LDSA": "dS"}

    # Optional extra data (everything except the main 3 numeric cols) indexed by time
    if extra_data:
        keep = set(["Datetime", "Total_conc", "Size", "LDSA"])
        extra_cols = [c for c in df.columns if c not in keep]
        extra_df = df[["Datetime", *extra_cols]].set_index("Datetime")
        DM._extra_data = extra_df
        DM._raw_extra_data = extra_df.copy()

    return DM
