# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd

from ..aerosol1d import Aerosol1D
from .Common import detect_delimiter

###############################################################################


def Load_CPC_file(file: str, extra_data: bool = False) -> Aerosol1D:
    """
    Load CPC data file and determine the appropriate parsing routine.

    This function detects whether the input file is exported directly from a CPC
    or whether it has been through the AIM software, based on its column
    structure, and routes it accordingly.
    The loaded data is then formatted and returned as an Aerosol1D class instance.

    Parameters
    ----------
    file : str or Path
        Path to the CPC data file (.txt format).
    extra_data : bool, optional
        If True, includes all non-core metadata in the `.extra_data` attribute (only for full format).

    Returns
    -------
    CPC : Aerosol1D
        Object containing datetime and total concentration data parsed from the CPC file.

    Raises
    ------
    Exception
        If the file format cannot be identified.
    """
    encoding, delimiter = detect_delimiter(file)
    col_count = len(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=4,
            max_rows=1,
            dtype=str,
        )
    )

    if col_count == 4:
        return Load_CPC_focused(file, encoding, delimiter)
    elif col_count == 14:
        return Load_CPC_full(file, extra_data, encoding, delimiter)
    else:
        raise Exception("Error in determining CPC data structure")


def Load_CPC_focused(file: str, encoding: str, delimiter: str) -> Aerosol1D:
    """
    Load and parse CPC data in 'focused' format.

    Focused files contain timestamp and concentration columns starting at line 14.
    The function constructs a time-indexed DataFrame and extracts relevant metadata.

    Parameters
    ----------
    file : str or Path
        Path to the focused CPC data file.
    encoding : str
        Encoding used to read the file.
    delimiter : str
        Delimiter used in the file.

    Returns
    -------
    CPC : Aerosol1D
        Object containing datetime and concentration data from the CPC export.
    """
    df = pd.read_csv(
        file,
        header=14,
        skipfooter=3,
        usecols=[0, 1],
        encoding=encoding,
        delimiter=delimiter,
        engine="python",
    )
    df.columns = ["Time", "Total_conc"]

    meta = np.genfromtxt(
        file,
        delimiter=delimiter,
        encoding=encoding,
        skip_header=4,
        max_rows=6,
        dtype="str",
    )

    start_datetime = datetime.datetime.strptime(
        f"{meta[0,1]} {meta[1,1]}", "%m/%d/%y %H:%M:%S"
    )
    df["Datetime"] = [
        start_datetime + datetime.timedelta(seconds=i + 1) for i in range(len(df))
    ]

    df = pd.concat([df["Datetime"], df["Total_conc"]], axis=1)
    df["Total_conc"] = pd.to_numeric(df["Total_conc"], errors="coerce")
    df = df.dropna()

    CPC = Aerosol1D(df)
    CPC._meta["instrument"] = "CPC"
    CPC._meta["serial_number"] = meta[5, 1][5:-3]
    CPC._meta["unit"] = "cm$^{-3}$"

    return CPC


def Load_CPC_full(
    file: str, extra_data: bool, encoding: str, delimiter: str
) -> Aerosol1D:
    """
    Load and parse CPC data in 'full' format.

    Full-format CPC files contain metadata and additional operational parameters,
    with datetimes and concentrations extracted from named columns.

    Parameters
    ----------
    file : str or Path
        Path to the full CPC data file.
    extra_data : bool
        If True, stores all additional columns in the `.extra_data` attribute.
    encoding : str
        Encoding used to read the file.
    delimiter : str
        Delimiter used in the file.

    Returns
    -------
    CPC : Aerosol1D
        Object containing datetime and concentration data from the CPC export.
    """
    df = pd.read_csv(
        file, header=2, encoding=encoding, delimiter=delimiter, engine="python"
    )

    df = df.rename(columns={"Sample #": "Datetime", "[1] Conc": "Total_conc"})
    df["Datetime"] = pd.to_datetime(
        df["Start Date"] + df["Start Time"], format="%m/%d/%y%H:%M:%S"
    )

    data_df = pd.concat([df["Datetime"], df["Total_conc"]], axis=1)

    CPC = Aerosol1D(data_df.copy())
    CPC._meta["instrument"] = "CPC"
    CPC._meta["serial_number"] = df["Instrument ID"].iloc[0][5:-3]
    CPC._meta["unit"] = "cm$^{-3}$"

    if extra_data:
        extra_df = df.drop(columns=["Start Date", "Start Time", "Total_conc"])
        extra_df.set_index("Datetime", inplace=True)
        CPC._extra_data = extra_df

    return CPC
