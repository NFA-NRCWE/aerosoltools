# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as datetime
import Common as Com
from aerosol1d import Aerosol1D

###############################################################################

def Load_CPC_file(file: str):
    """
    Load particle number concentration data from a CPC `.txt` export file.

    This function parses timestamped CPC measurements and returns them in an
    `Aerosol1D` object. It also extracts relevant metadata including instrument
    serial number and units.

    Parameters
    ----------
    file : str
        Path to a `.txt` file exported from the CPC software.

    Returns
    -------
    CPC : Aerosol1D
        Object containing the total concentration time series and metadata.

    Notes
    -----
    - The time column is not timestamped in the file and is reconstructed from
      the provided start time and assumes 1-second resolution.
    - Units are assumed to be in cm⁻³.
    - Requires `Com.detect_delimiter()` for automatic encoding/delimiter detection.
    """
    # Detect encoding and delimiter
    encoding, delimiter = Com.detect_delimiter(file)

    # Read CPC data block (time, concentration), ignoring last lines
    df = pd.read_csv(
        file,
        header=14,
        skipfooter=3,
        usecols=[0, 1],
        encoding=encoding,
        delimiter=delimiter,
        engine="python"
    )
    df.columns = ["Time", "Total_conc"]

    # Read metadata (serial, start date/time)
    meta = np.genfromtxt(
        file,
        delimiter=delimiter,
        encoding=encoding,
        skip_header=4,
        max_rows=6,
        dtype=str
    )
    try:
        start_dt = datetime.datetime.strptime(f"{meta[0, 1]} {meta[1, 1]}", "%m/%d/%y %H:%M:%S")
    except Exception as e:
        raise ValueError(f"Unable to parse start datetime: {e}")

    # Construct datetime column assuming 1-second interval
    df["Datetime"] = pd.date_range(start=start_dt, periods=len(df), freq="1s")

    # Clean and convert concentration values
    df["Total_conc"] = pd.to_numeric(df["Total_conc"], errors="coerce")
    df.dropna(subset=["Total_conc"], inplace=True)

    # Create Aerosol1D object
    CPC = Aerosol1D(df[["Datetime", "Total_conc"]])
    CPC._meta["instrument"] = "CPC"
    CPC._meta["serial_number"] = meta[5, 1][5:-3]  # Trims "CPC: XYZ123.xxx" to "XYZ123"
    CPC._meta["unit"] = "cm$^{-3}$"
    CPC._meta["dtype"] = "dN"

    return CPC

