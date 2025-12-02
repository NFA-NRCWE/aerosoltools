import datetime as datetime

import numpy as np
import pandas as pd

from ..aerosolalt import AerosolAlt
from .Common import detect_delimiter

###############################################################################


def Load_DustTrak_file(file: str, extra_data: bool = False):
    """
    Load DustTrak data from a .csv file.

    This function reads the PM and Total mass values from DustTrak DRX, reconstructs
    the datetime index from the file's metadata, and returns an `AerosolAlt` object
    containing the structured data.

    Parameters
    ----------
    file : str
        Path to the Partector `.txt` file.
    extra_data : bool, optional
        If True, attaches "Alarms" and "Errors" columns
        as `extra_data` in the returned class. Default is False.

    Returns
    -------
    Dust : AerosolAlt
        A class instance containing datetime-indexed PM1,2.5,4,10 and Total mass.
        Metadata includes instrument info.

    Notes
    -----
    - Mass values are loaded as mg/m³ and returned as in units of ug/m³`.
    - Requires `Com.detect_delimiter()` for automatic delimiter/encoding detection.
    """

    try:
        encoding, delimiter = detect_delimiter(file, sample_lines=30)
    except Exception:
        delimiter = ","

    # Read main data
    df = pd.read_csv(file, delimiter=delimiter, header=35)
    df.rename(
        columns={
            "Elapsed Time [s]": "Datetime",
            "PM1 [mg/m3]": "PM1",
            "PM2.5 [mg/m3]": "PM2.5",
            "PM4 [mg/m3]": "PM4",
            "PM10 [mg/m3]": "PM10",
            "TOTAL [mg/m3]": "Total",
        },
        inplace=True,
    )

    # Read header metadata
    meta_lines = np.genfromtxt(file, delimiter=delimiter, max_rows=8, dtype="str")
    try:
        start_str = f"{meta_lines[7,1]} {meta_lines[6,1]}"
        start_time = datetime.datetime.strptime(start_str, "%d/%m/%Y %H:%M:%S")
    except Exception as e:
        raise ValueError(f"Unable to parse start datetime from metadata: {e}")

    # Convert time column to absolute datetime
    df["Datetime"] = pd.to_timedelta(df["Datetime"], unit="s") + start_time
    data_columns = df.columns[1:6]

    df[data_columns] = df[data_columns] * 1000
    # Create AerosolAlt object
    Dust = AerosolAlt(df[df.columns[:6]])
    Dust._meta["instrument"] = meta_lines[0, 1]
    Dust._meta["model_number"] = meta_lines[1, 1]
    Dust._meta["serial_number"] = meta_lines[2, 1]
    Dust._meta["unit"] = {'PM1':"ug/m³",'PM2.5':"ug/m³",'PM4':"ug/m³",'PM10':"ug/m³",'Total':"ug/m³"}
    Dust._meta["dtype"] = {'PM1':"dM",'PM2.5':'dM','PM4':'dM','PM10':'dM','Total':'dM'}

    # Attach extra data
    if extra_data:
        extra_df = df["Datetime", "Alarms", "Errors"].set_index("Datetime")
        Dust._extra_data = extra_df
        Dust._raw_extra_data = extra_df.copy()
    return Dust
