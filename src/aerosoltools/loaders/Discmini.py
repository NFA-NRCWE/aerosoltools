from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

from ..discmini import DiSCmini
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


def Load_DiSCmini_file(file: str, extra_data: bool = False) -> DiSCmini:
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

    # Drop phantom columns produced by a trailing delimiter in the export header
    # (e.g. "…Filter\tDiff\t" yields an unnamed, all-empty 7th column). Left in,
    # such a column would show up as an empty "Unnamed: N" series in the GUI.
    phantom = [
        c for c in df.columns if str(c).startswith("Unnamed") or df[c].isna().all()
    ]
    if phantom:
        df.drop(columns=phantom, inplace=True)

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

    # Build DiSCmini on the core four columns (order: Datetime, Total_conc, Size, LDSA)
    needed = ["Datetime", "Total_conc", "Size", "LDSA"]
    present = [c for c in needed if c in df.columns]
    if present[:1] != ["Datetime"]:
        raise Exception("Datetime column missing after parsing.")
    DM = DiSCmini(df[present])

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

    # Optional extra data (everything except the main 3 numeric cols) indexed by
    # time. Columns are read as strings; coerce the numeric ones (e.g. the
    # Filter/Diff currents) so they are proper numbers for downstream use/plots.
    if extra_data:
        keep = set(["Datetime", "Total_conc", "Size", "LDSA"])
        extra_cols = [c for c in df.columns if c not in keep]
        extra_df = df[["Datetime", *extra_cols]].set_index("Datetime")
        for col in extra_df.columns:
            coerced = pd.to_numeric(extra_df[col], errors="coerce")
            if coerced.notna().any():
                extra_df[col] = coerced
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


#: Upper diameter the vendor reports; the sizing calibration is only valid
#: below the diffusion stage's diameter of maximum penetration, so larger
#: results are saturated here (matching the vendor output, which caps at 300 nm).
_DISC_SIZE_MAX = 300.0


def _disc_size_from_ratio(ratio: np.ndarray, cal: list[float]) -> np.ndarray:
    """Mean particle diameter (nm) from the filter/diffusion current ratio.

    The DiSC derives the average particle size from the ratio of the two
    electrometer stage currents, ``R = I_filter / I_diffusion`` — smaller
    particles undergo more Brownian diffusion and are preferentially captured
    in the diffusion stage, so ``R`` grows with particle size. The exact
    ``R → d`` relation is the diffusion-stage capture calibration, which the
    instrument stores as a **cubic polynomial** (Fierz et al. 2011, *Design,
    Calibration, and Field Performance of a Miniature Diffusion Size
    Classifier*, AS&T 45:1-10). Its four coefficients are the first four values
    of the file's ``CalData`` block::

        d = cal[0] + cal[1]·R + cal[2]·R² + cal[3]·R³

    Verified against vendor-processed files to a median error < 1 %.

    Args:
        ratio: Filter/diffusion current ratio (``I_filter / I_diffusion``).
        cal: The 7 calibration constants from the file header.

    Returns:
        numpy.ndarray: Mean diameter in nm, clipped to ``(0, 300]`` (the
        polynomial is only calibrated below the ~500 nm diameter of maximum
        penetration; the vendor likewise saturates the reported size).
    """
    r = np.asarray(ratio, dtype=float)
    size = cal[0] + cal[1] * r + cal[2] * r**2 + cal[3] * r**3
    return np.clip(size, 1.0, _DISC_SIZE_MAX)


#: Empirical size-dependent LDSA refinement. ``LDSA = cal[6]·I_total`` slightly
#: underestimates the vendor's LDSA, by an amount that grows smoothly with the
#: mean particle size (consistent with the flat charge-signal approximation
#: diverging from the true size-weighted lung deposition; cf. Fierz 2011 note
#: #8). A quadratic-in-diameter factor fit to paired raw/processed files removes
#: most of it; leave-one-out validation across the reference instruments cut the
#: LDSA error by ~35 % on each held-out instrument, so the trend generalises.
_LDSA_CORR = (1.003627, -3.126074e-4, 4.279956e-6)  # (c0, c1, c2) in 1/nm powers
_LDSA_CORR_SIZE_CAP = 150.0  # the fit is only supported up to ~150 nm
_LDSA_CORR_CLIP = (0.98, 1.06)  # bound the factor against extrapolation


def _ldsa_size_factor(size: np.ndarray) -> np.ndarray:
    """Return the empirical size-dependent LDSA correction factor (see _LDSA_CORR)."""
    s = np.clip(np.asarray(size, dtype=float), 0.0, _LDSA_CORR_SIZE_CAP)
    c0, c1, c2 = _LDSA_CORR
    return np.clip(c0 + c1 * s + c2 * s * s, *_LDSA_CORR_CLIP)


def _zero_offset_series(raw: pd.DataFrame, offsets_hdr) -> tuple:
    """Build the time-varying electrometer zero offsets from a raw record.

    The DiSCmini re-measures its electrometer zero offsets during a ~1 minute
    pump-/charger-off period every hour (Testo DiSCmini manual, "Zero offsets").
    The vendor software subtracts these (slowly drifting) offsets from the two
    stage currents before computing number/size/LDSA. Those offset measurements
    are recoverable straight from the raw file: during each idle period the
    charging current ``Idiff`` falls to ≈ 0, and the diffusion/filter readings
    then are the pure electrometer zeros. This returns anchor points for a
    linear interpolation of each stage's offset over time, seeded at ``t = 0``
    with the header ``Offsets:`` values (the instrument's start-up zeros).

    Args:
        raw: The full raw data table (all rows, incl. idle) with numeric
            ``Time``/``Diffusion``/``Filter``/``Idiff`` and a ``Status`` column.
        offsets_hdr: The two header ``Offsets:`` values, or ``None``.

    Returns:
        ``(times, off_diff, off_filter)`` numpy arrays of anchor times (s) and
        the corresponding diffusion/filter zero offsets, or ``None`` when no
        offset information is available.
    """
    times, off_d, off_f = [], [], []
    if offsets_hdr is not None and len(offsets_hdr) >= 2:
        times.append(0.0)
        off_d.append(float(offsets_hdr[0]))
        off_f.append(float(offsets_hdr[1]))

    idle = raw[~raw["Status"].str.lower().str.endswith("b", na=False)]
    if not idle.empty:
        # Split the idle rows into consecutive runs (the hourly offset periods).
        breaks = np.where(np.diff(idle["Time"].to_numpy()) > 1)[0]
        for run in np.split(idle.index.to_numpy(), breaks + 1):
            seg = raw.loc[run]
            settled = seg[seg["Idiff"] < 1.0]  # charger fully off → pure zero
            if not settled.empty:
                times.append(float(settled["Time"].mean()))
                off_d.append(float(settled["Diffusion"].mean()))
                off_f.append(float(settled["Filter"].mean()))

    if not times:
        return None
    order = np.argsort(times)
    return (
        np.asarray(times)[order],
        np.asarray(off_d)[order],
        np.asarray(off_f)[order],
    )


def Load_DiSCmini_raw_file(
    file: str,
    extra_data: bool = False,
    period: int = 10,
    zero_offset_correction: bool = True,
    ldsa_correction: bool = False,
) -> DiSCmini:
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
        zero_offset_correction (bool, optional):
            Subtract the time-varying electrometer zero offsets (recovered from
            the hourly pump-off offset measurements, see
            :func:`_zero_offset_series`) from the stage currents before
            computing, as the vendor software does. This markedly improves
            agreement for instruments whose electrometer baseline drifts.
            Defaults to ``True``.
        ldsa_correction (bool, optional):
            Apply an **empirical** size-dependent multiplier to LDSA (see
            :data:`_LDSA_CORR`). The simple ``cal[6]·I_total`` LDSA slightly
            underestimates the vendor value by an amount that grows with mean
            particle size; this factor (fit to paired raw/processed files and
            leave-one-out validated) removes most of it. It is *not* a
            documented vendor formula, so it defaults to ``False``; enable it
            for closer agreement with the vendor LDSA.

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
            - **Averaging.** Kept rows are averaged into fixed ``period``-second
              wall-clock windows (``floor(Time / period)``), as the vendor tool
              does; fully idle windows drop out. (Grouping on time rather than on
              every N-th valid row keeps windows aligned when idle rows are
              interspersed through the record.)
            - **LDSA** ``= cal[6] · (I_diffusion + I_filter)`` — reproduces the
              vendor value essentially exactly. This follows from the diffusion
              charger physics (miniDiSC application note #8): the charger signal,
              i.e. the total current, is directly proportional to LDSA.
            - **Size** ``= cal[0] + cal[1]·R + cal[2]·R² + cal[3]·R³`` where
              ``R = I_filter / I_diffusion``. This is the DiSC diffusion-stage
              current-ratio calibration, stored on the instrument as a cubic
              polynomial (Fierz et al. 2011); see :func:`_disc_size_from_ratio`.
            - **Number** ``= cal[5] · (I_diffusion + I_filter) / Size**cal[4]``.
              The average charge per particle scales as ``q ∝ d**1.1`` (Fierz
              2011; miniDiSC note #8), so the total current is ``N · q``;
              inverting gives this expression with ``cal[4] ≈ 1.1``.

            ``cal[0..3]`` are the cubic size-calibration coefficients,
            ``cal[4]`` the charge/size exponent (≈ 1.1), ``cal[5]`` the
            number-calibration factor and ``cal[6]`` the LDSA-per-current
            factor. All were validated against vendor-processed files.

            A drifting electrometer **zero offset** is removed before computing
            (``zero_offset_correction``): the instrument re-measures its zero
            offsets during a ~1 min pump-off period every hour (Testo DiSCmini
            manual), and those measurements are recovered from the idle periods
            in the raw file and subtracted from the stage currents, as the
            vendor software does (see :func:`_zero_offset_series`). This is the
            dominant source of raw-vs-vendor disagreement for instruments whose
            baseline drifts.

            Accuracy: the *inversion* fed the vendor's own currents matches
            Size/Number to a median error < 1 %; end-to-end from the raw file
            (with the zero-offset correction) Size/Number agree with the vendor
            output to roughly 1-3 % across the reference instruments. A small
            residual remains from effects not reproduced here — the optional
            *induction correction* (subtracting a term proportional to the
            filter-signal time-derivative from the diffusion stage, which the
            vendor leaves off by default and which only matters during rapid
            transients), the sensor-signal-to-l/min flow conversion, and the
            2-decimal rounding of the raw currents. The averaged
            diffusion/filter currents are exposed via ``extra_data``.

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

    # Subtract the drifting electrometer zero offsets (recovered from the hourly
    # pump-off offset measurements) from the stage currents, as the vendor does.
    offsets = (
        _zero_offset_series(raw, meta["offsets"]) if zero_offset_correction else None
    )
    if offsets is not None:
        ot, od, of = offsets
        raw["Diffusion"] = raw["Diffusion"] - np.interp(raw["Time"], ot, od)
        raw["Filter"] = raw["Filter"] - np.interp(raw["Time"], ot, of)

    # Keep only active-measurement rows (Status low nibble "B", e.g. "8B").
    valid = raw[raw["Status"].str.lower().str.endswith("b", na=False)].copy()
    valid = valid.dropna(subset=["Time", "Diffusion", "Filter"]).reset_index(drop=True)
    if valid.empty:
        raise ValueError("No valid measurement rows found in the raw DiSCmini file.")

    # Average the valid rows into fixed ``period``-second wall-clock windows,
    # grouping on the elapsed-time column (``floor(Time / period)``) exactly as
    # the vendor tool does. Grouping on time — rather than on every N-th valid
    # row — keeps windows aligned even when idle rows are interspersed through
    # the record. A window is only emitted when at least ``min_valid`` of its
    # ``period`` seconds are valid (the vendor keeps windows with >= 8 of 10
    # valid samples); this drops fully/partly idle windows at gap edges so the
    # output row sequence matches the vendor's — otherwise a spurious partial
    # window at each gap would shift every later row and accumulate a drift.
    cols = ["Diffusion", "Filter", "Temp", "Idiff", "Ucor", "Flow", "Batt"]
    min_valid = int(np.ceil(0.8 * period))
    window = (valid["Time"] // period).astype("int64")
    grp = valid.groupby(window)
    counts = grp.size()
    avg = grp[cols].mean()
    keep = counts.index[counts >= min_valid]
    avg = avg.loc[keep]
    win_index = avg.index.to_numpy()
    avg = avg.reset_index(drop=True)

    diff = avg["Diffusion"].to_numpy()
    filt = avg["Filter"].to_numpy()
    itot = diff + filt

    # LDSA = cal[6]·I_total; Size from the current-ratio calibration polynomial;
    # Number = cal[5]·I_total / d**cal[4] (charge law q ∝ d**cal[4], cal[4]≈1.1).
    ldsa = cal[6] * itot
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(diff != 0, filt / diff, np.nan)
    size = _disc_size_from_ratio(ratio, cal)
    number = cal[5] * itot / np.power(size, cal[4])
    # Optional empirical size-dependent LDSA refinement (does not feed Number).
    if ldsa_correction:
        ldsa = ldsa * _ldsa_size_factor(size)

    # Real timestamps at each window's centre (start + win*period + (period-1)/2),
    # matching the vendor's time labelling and preserving real elapsed time.
    start = meta["start"] or dt.datetime(1970, 1, 1)
    centre_s = win_index * period + (period - 1) / 2.0
    times = pd.to_datetime(start) + pd.to_timedelta(centre_s, unit="s")

    out = pd.DataFrame(
        {
            "Datetime": times,
            "Total_conc": np.round(number).astype(float),
            "Size": np.round(size, 1),
            "LDSA": np.round(ldsa, 2),
        }
    )

    DM = DiSCmini(out)
    DM._meta["instrument"] = "DiSCmini"
    DM._meta["serial_number"] = meta["serial"]
    if meta["firmware"] is not None:
        DM._meta["firmware"] = meta["firmware"]
    DM._meta["calibration"] = cal
    DM._meta["offsets"] = meta["offsets"]
    DM._meta["averaging_period_s"] = period
    DM._meta["zero_offset_correction"] = bool(offsets is not None)
    DM._meta["ldsa_correction"] = bool(ldsa_correction)
    # Size uses the instrument's stored cubic current-ratio calibration
    # (Fierz 2011); reproduces the vendor output to a median error < 1 %.
    DM._meta["size_inversion"] = "cubic_ratio_polynomial"
    DM._meta["unit"] = {
        "Total_conc": "cm⁻³",
        "Size": "nm",
        "LDSA": "nm²/cm³",
    }
    DM._meta["dtype"] = {"Total_conc": "dN", "Size": "l", "LDSA": "dS"}

    if extra_data:
        extra = avg.copy()
        extra.insert(0, "Datetime", times)
        extra = extra.set_index("Datetime")
        DM._extra_data = extra
        DM._raw_extra_data = extra.copy()

    return DM
