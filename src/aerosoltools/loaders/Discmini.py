from __future__ import annotations

import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..aerosolalt import AerosolAlt
from .Common import detect_delimiter

def Load_DiSCmini_file(file: str, extra_data: bool = False) -> AerosolAlt:
    """
    Load and parse data from a DiSCmini .txt file (after conversion), returning an AerosolAlt object.

    This function extracts the datetime, total particle number concentration, average size,
    and LDSA from the DiSCmini export file. It also stores serial number and units
    as metadata, and optionally attaches extra columns as `.extra_data`.

    Parameters
    ----------
    file : str
        Path to the .txt file exported from DiSCmini software (after conversion).
    extra_data : bool, optional
        If True, attaches unused data columns as `._extra_data` in the AerosolAlt class.

    Returns
    -------
    DM : AerosolAlt
        Object containing parsed time series data and instrument metadata.

    Raises
    ------
    Exception
        If the file has not been converted correctly, or the datetime format is unrecognized.

    Notes
    -----
    - The returned data contains: 'Datetime', 'Total_conc' (cm⁻³), 'Size' (nm), and 'LDSA' (nm²/cm³).
    - Automatically detects encoding and delimiter using `detect_delimiter()`.
    - Two known datetime formats are supported: `%d-%b-%Y %H:%M:%S` and `%d-%m-%Y %H:%M:%S`,
      with a fallback that reconstructs absolute time from a start timestamp when present.
    """
    # Detect encoding + delimiter
    try:
        enc, delim = detect_delimiter(file, sample_lines=25)  # -> (str, str)
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
        df.rename(columns={"TimeStamp": "Datetime", "Number": "Total_conc"}, inplace=True)
    else:
        df.rename(columns={"Time": "Datetime", "Number": "Total_conc"}, inplace=True)

    # Parse datetime with two known formats; if both fail, attempt reconstruction from header
    dt_parsed = pd.to_datetime(df["Datetime"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    if dt_parsed.isna().all():
        dt_parsed = pd.to_datetime(df["Datetime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")

    if dt_parsed.isna().any():
        # Fallback: reconstruct absolute time from a start date/time in the file header
        try:
            # read minimal header with Python I/O to avoid numpy-encoding stub issues
            with open(file, "r", encoding=enc) as fh:
                header_lines = [next(fh) for _ in range(8)]
            # common patterns:
            #   "[...] start date: YYYY.MM.DD]"
            #   "[...] start time: HH:MM:SS]"
            start_date_line = next((ln for ln in header_lines if "start date:" in ln), None)
            start_time_line = next((ln for ln in header_lines if "start time:" in ln), None)
            if start_date_line is None or start_time_line is None:
                raise ValueError("Start date/time not found in header.")

            start_date = dt.datetime.strptime(
                start_date_line.split("start date: ")[1].split("]")[0], "%Y.%m.%d"
            )
            start_time = dt.datetime.strptime(
                start_time_line.split("start time: ")[1].split("]")[0], "%H:%M:%S"
            )
            start_dt = start_date + dt.timedelta(
                seconds=start_time.hour * 3600 + start_time.minute * 60 + start_time.second
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
        s = s.fillna("").str.replace(",", ".", regex=False).str.replace(r"\s+", "", regex=True)
        return pd.to_numeric(s, errors="coerce")

    # Some exports use "Size" / "LDSA" names consistently
    if "Size" not in df.columns or "LDSA" not in df.columns:
        # Try common alternates if present (adjust if you’ve seen other labels)
        for guess, canonical in [("AvgSize", "Size"), ("LungDepSurfArea", "LDSA")]:
            if guess in df.columns and canonical not in df.columns:
                df.rename(columns={guess: canonical}, inplace=True)

    for col in ["Total_conc", "Size", "LDSA"]:
        if col in df.columns:
            df[col] = _to_num(df[col])

    # Extract serial number (robustly read a small header block)
    with open(file, "r", encoding=enc) as fh:
        meta_line = next(fh)  # line 0
        # Often serial appears on early lines; scan a few
        first_lines = [meta_line] + [next(fh) for _ in range(6)]
    serial_number = None
    for ln in first_lines:
        if "serial" in ln.lower():
            # naive grab: last space-separated token or adjust to your actual pattern
            toks = ln.strip().replace(",", " ").split()
            serial_number = toks[-1]
            break
    if serial_number is None:
        # fallback to numpy reader if needed (use file handle to avoid encoding kw warnings)
        with open(file, "r", encoding=enc) as fh2:
            arr = np.genfromtxt(fh2, delimiter=delim, skip_header=1, max_rows=1, dtype=str)
        serial_number = str(arr).split(" ")[-1]

    # Build AerosolAlt on the core four columns (order: Datetime, Total_conc, Size, LDSA)
    needed = ["Datetime", "Total_conc", "Size", "LDSA"]
    present = [c for c in needed if c in df.columns]
    if present[:1] != ["Datetime"]:
        raise Exception("Datetime column missing after parsing.")
    DM = AerosolAlt(df[present])

    # Metadata
    DM._meta["instrument"] = "DiSCmini"
    DM._meta["serial_number"] = serial_number
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
