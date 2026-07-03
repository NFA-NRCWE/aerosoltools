from __future__ import annotations

import datetime as dt
import re

import numpy as np
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


###############################################################################
# Raw-file loading (reproduce the vendor "java tool" processing from raw data)
###############################################################################

#: Column layout of the raw DiSCmini data table (after the header block).
_RAW_COLUMNS = [
    "Time",
    "Diffusion",
    "Filter",
    "Temp",
    "Idiff",
    "Ucor",
    "Flow",
    "Batt",
    "Status",
]


def _parse_raw_header(header_lines: list[str]) -> dict:
    """Parse the metadata block at the top of a raw DiSCmini ``.TXT`` file.

    The raw header (before the ``Time  Diffusion  Filter ...`` column row)
    carries the software version, start date/time, the per-instrument
    calibration constants and electrometer offsets, e.g.::

        nw PERSONAL AEROSOL MONITOR Data written with SW-Ver 3.42
        Filename: 6605G55D.TXT
        Averaging Period: 1 sec
        Date and Time: 2026.06.05 06:55:26
        CalData: SN101923    0.28   30.73   -6.45    1.28    1.1319808.76    0.68
         NaCl 2017_02_03
            0.28    30.73    -6.45    1.28    1.13    19808.76    0.68
        Offsets:    -0.75    -0.69
        Sampled:   149393 pC   C:     395   W:      41

    The ``CalData:`` line itself is unreliable for the constants (adjacent
    values can run together, e.g. ``1.1319808.76``); the clean, tab-separated
    copy on the line just above ``Offsets:`` is used instead.

    Args:
        header_lines: Lines of the file up to (and including) the column-header
            row.

    Returns:
        dict: With keys ``serial``, ``firmware``, ``start`` (a
        :class:`datetime.datetime` or ``None``), ``cal`` (list of 7 floats or
        ``None``), ``offsets`` (list of 2 floats or ``None``) and
        ``header_rows`` (number of lines to skip before the data table).

    Raises:
        ValueError: If the seven calibration constants cannot be located.
    """
    serial, firmware = _extract_serial_and_firmware(header_lines)

    start = None
    offsets = None
    cal = None
    data_header_idx = None
    for i, ln in enumerate(header_lines):
        low = ln.lower()
        if "date and time:" in low and start is None:
            token = ln.split(":", 1)[1].strip()
            for fmt in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    start = dt.datetime.strptime(token, fmt)
                    break
                except ValueError:
                    continue
        elif low.startswith("offsets:"):
            offsets = [float(x) for x in re.findall(r"-?\d+\.?\d*", ln)]
            # The clean calibration-constant row is the line directly above.
            if i >= 1:
                nums = re.findall(r"-?\d+\.\d+", header_lines[i - 1])
                if len(nums) >= 7:
                    cal = [float(x) for x in nums[:7]]
        elif low.startswith("time") and "diffusion" in low:
            data_header_idx = i
            break

    if cal is None:
        raise ValueError(
            "Could not read the 7 DiSCmini calibration constants from the raw "
            "header. Verify the file is an unmodified raw instrument export."
        )
    return {
        "serial": serial,
        "firmware": firmware,
        "start": start,
        "cal": cal,
        "offsets": offsets,
        "header_rows": (data_header_idx + 1) if data_header_idx is not None else 10,
    }


def _disc_size_from_ratio(ratio: np.ndarray, cal: list[float]) -> np.ndarray:
    """Mean particle diameter (nm) from the filter/diffusion current ratio.

    .. warning::

        **Provisional.** The exact testo DiSCmini diameter inversion (the
        diffusion-charging model relating the two stage currents to particle
        size) is proprietary and could not be reproduced exactly from the small
        set of calibration files available. This is a best-effort linear
        approximation using the two calibration constants that clearly act as
        the size scale and offset (``cal[1]`` and ``cal[2]``); it is accurate to
        only a few nm and is expected to be replaced once the true inversion is
        confirmed. Because :func:`Load_DiSCmini_raw_file` derives ``Number``
        from ``Size``, the number concentration inherits this approximation.

    Args:
        ratio: Filter/diffusion current ratio (``I_filter / I_diffusion``).
        cal: The 7 calibration constants from the file header.

    Returns:
        numpy.ndarray: Estimated mean diameter in nm, clipped to ``[1, 300]``.
    """
    size = cal[1] * ratio + cal[2]
    return np.clip(size, 1.0, 300.0)


def Load_DiSCmini_raw_file(
    file: str,
    extra_data: bool = False,
    period: int = 10,
) -> AerosolAlt:
    """Description:
        Load a **raw** DiSCmini ``.TXT`` file and reproduce the vendor
        software's processed output (total number concentration, mean size
        and LDSA) directly, without the intermediate commercial conversion.

    Args:
        file (str):
            Path to a raw DiSCmini ``.TXT`` file (the file written by the
            instrument, with the ``CalData:``/``Offsets:`` header and the
            ``Time  Diffusion  Filter ...`` data table).
        extra_data (bool, optional):
            If ``True``, the averaged diagnostic channels (diffusion/filter
            currents, temperature, flow, battery, …) are stored in
            ``.extra_data``. Defaults to ``False``.
        period (int, optional):
            Averaging period in seconds. The vendor tool averages the raw
            1 Hz data into 10 s windows; this reproduces that with
            ``period=10`` (the default).

    Returns:
        AerosolAlt:
            An :class:`~aerosoltools.aerosolalt.AerosolAlt` with
            ``Total_conc`` (cm⁻³), ``Size`` (nm) and ``LDSA`` (nm²/cm³) at the
            chosen averaging period, indexed by **real** timestamps (unlike the
            vendor output, which renumbers time contiguously across gaps).

    Raises:
        ValueError:
            If the calibration constants cannot be parsed from the header (see
            :func:`_parse_raw_header`).

    Notes:
        Detailed description:
            The reproduction was reverse-engineered from paired raw/processed
            DiSCmini files:

            - **Row selection.** Only rows whose ``Status`` byte marks an active
              measurement (low nibble ``B``, e.g. ``"8B"``) are kept; idle /
              pump-off rows (``"88"``/``"89"``, flow ≈ 0.37) are dropped.
            - **Averaging.** Every ``period`` consecutive kept rows are averaged
              into one output row (a trailing partial block is dropped), which
              matches the vendor tool's block count (``floor(valid / period)``).
            - **LDSA** ``= cal[6] · (I_diffusion + I_filter)`` — reproduces the
              vendor value essentially exactly. This follows from the diffusion
              charger physics (miniDiSC application note #8): the charger signal,
              i.e. the total current, is directly proportional to LDSA.
            - **Number** ``= cal[5] · (I_diffusion + I_filter) / Size**cal[4]``.
              The average charge per particle scales as ``q ∝ d**1.1`` (note #8),
              so the total current is ``N · q``; inverting gives this expression
              with ``cal[4] ≈ 1.1``. Exact given ``Size``.
            - **Size** — from the filter/diffusion current ratio via
              :func:`_disc_size_from_ratio` (**provisional**, see its warning).

            ``cal[4]`` (the charge/size exponent, ≈ 1.1), ``cal[5]`` (the
            number-calibration factor) and ``cal[6]`` (the LDSA-per-current
            factor) were confirmed against the vendor output and are consistent
            with the miniDiSC application notes. The diameter inversion (the
            DiSC diffusion-stage deposition calibration) is *not* described by
            those notes, so the metadata records ``size_inversion =
            "provisional"`` as a reminder that ``Size`` (and hence ``Number``)
            is approximate pending the exact algorithm.

    Examples:
        .. code-block:: python

            import aerosoltools as at

            dm = at.Load_DiSCmini_raw_file("data/6605G55D.TXT")
            print(dm.data[["Total_conc", "Size", "LDSA"]])
    """
    enc, _ = _detect_delimiter(file, sample_lines=25)

    # Read the header block (enough lines to reach the data-table header row).
    with open(file, "r", encoding=enc) as fh:
        header_lines = [ln for _, ln in zip(range(15), fh)]
    meta = _parse_raw_header(header_lines)
    cal = meta["cal"]

    # Read the data table.
    raw = pd.read_csv(
        file,
        sep="\t",
        skiprows=meta["header_rows"],
        header=None,
        engine="python",
        names=_RAW_COLUMNS + ["_extra"],
        usecols=range(len(_RAW_COLUMNS)),
    )
    for col in ["Time", "Diffusion", "Filter", "Temp", "Idiff", "Ucor", "Flow", "Batt"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["Status"] = raw["Status"].astype("string").str.strip()

    # Keep only active-measurement rows (Status low nibble "B", e.g. "8B").
    valid = raw[raw["Status"].str.lower().str.endswith("b", na=False)].copy()
    valid = valid.dropna(subset=["Time", "Diffusion", "Filter"]).reset_index(drop=True)
    if valid.empty:
        raise ValueError("No valid measurement rows found in the raw DiSCmini file.")

    # Average every `period` consecutive valid rows into one output row — this
    # reproduces the vendor tool's block structure exactly (its row count is
    # ``floor(valid_rows / period)``). Any trailing partial block is dropped.
    n_blocks = len(valid) // period
    valid = valid.iloc[: n_blocks * period]
    block = np.arange(len(valid)) // period
    avg = (
        valid[["Time", "Diffusion", "Filter", "Temp", "Idiff", "Ucor", "Flow", "Batt"]]
        .groupby(block)
        .mean()
        .reset_index(drop=True)
    )

    diff = avg["Diffusion"].to_numpy()
    filt = avg["Filter"].to_numpy()
    itot = diff + filt

    # LDSA (exact) and Size (provisional) -> Number (from cal + Size).
    ldsa = cal[6] * itot
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(diff != 0, filt / diff, np.nan)
    size = _disc_size_from_ratio(ratio, cal)
    number = cal[5] * itot / np.power(size, cal[4])

    # Real timestamps: start datetime + each window's mean elapsed second.
    start = meta["start"] or dt.datetime(1970, 1, 1)
    times = pd.to_datetime(start) + pd.to_timedelta(avg["Time"].to_numpy(), unit="s")

    out = pd.DataFrame(
        {
            "Datetime": times,
            "Total_conc": np.round(number).astype(float),
            "Size": np.round(size, 1),
            "LDSA": np.round(ldsa, 2),
        }
    )

    DM = AerosolAlt(out)
    DM._meta["instrument"] = "DiSCmini"
    DM._meta["serial_number"] = meta["serial"]
    if meta["firmware"] is not None:
        DM._meta["firmware"] = meta["firmware"]
    DM._meta["calibration"] = cal
    DM._meta["offsets"] = meta["offsets"]
    DM._meta["averaging_period_s"] = period
    # Flag that Size (and the Number derived from it) use the provisional
    # inversion, so downstream code / users know these are approximate.
    DM._meta["size_inversion"] = "provisional"
    DM._meta["unit"] = {
        "Total_conc": "cm⁻³",
        "Size": "nm",
        "LDSA": "nm²/cm³",
    }
    DM._meta["dtype"] = {"Total_conc": "dN", "Size": "l", "LDSA": "dS"}

    if extra_data:
        extra = avg.drop(columns=["Time"]).copy()
        extra.insert(0, "Datetime", times)
        extra = extra.set_index("Datetime")
        DM._extra_data = extra
        DM._raw_extra_data = extra.copy()

    return DM
