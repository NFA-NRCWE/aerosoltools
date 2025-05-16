# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as datetime
import Common as Com
from ..aerosolalt import AerosolAlt
from matplotlib.dates import date2num

###############################################################################

def Load_Partector_file(file: str, extra_data: bool = False):
    """
    Load Partector LDSA data from a .txt file.

    This function reads the flow, LDSA, and TEM (filter trigger) columns, reconstructs
    the datetime index from the file's metadata, and returns an `AerosolAlt` object
    containing the structured data.

    Parameters
    ----------
    file : str
        Path to the Partector `.txt` file.
    extra_data : bool, optional
        If True, attaches all additional columns (except LDSA, TEM, Flow)
        as `extra_data` in the returned class. Default is False.

    Returns
    -------
    Par : AerosolAlt
        A class instance containing datetime-indexed LDSA, TEM flag, and flow.
        Metadata includes sample volume estimates and instrument info.

    Notes
    -----
    - LDSA is returned in units of `nm²/cm³`.
    - Flow is averaged over TEM==1 samples and reported in `l/min`.
    - Sample volume is calculated only for the period where TEM==1.
    - Requires `Com.detect_delimiter()` for automatic delimiter/encoding detection.
    """

    try:
        encoding, delimiter = Com.detect_delimiter(file, sample_lines=30)
    except Exception:
        delimiter = "\t"

    # Read main data
    df = pd.read_csv(file, delimiter=delimiter, header=10)
    df.rename(columns={"time": "Datetime", "flow": "Flow"}, inplace=True)

    # Read header metadata
    meta_lines = np.genfromtxt(file, delimiter=delimiter, max_rows=10, dtype="str")
    try:
        start_str = meta_lines[4].split("Start: ")[1].split("\n")[0]
        start_time = datetime.datetime.strptime(start_str, "%d.%m.%Y %H:%M:%S")
    except Exception as e:
        raise ValueError(f"Unable to parse start datetime from metadata: {e}")

    # Convert time column to absolute datetime
    df["Datetime"] = pd.to_timedelta(df["Datetime"], unit="s") + start_time

    # Sample duration estimate from TEM flag
    is_templog = df["TEM"] == 1
    if is_templog.sum() < 2:
        raise ValueError("Not enough TEM sampling points to estimate volume.")

    if is_templog.sum() > 0 and is_templog.sum() != len(is_templog):
        if (df["TEM"].diff() == 1).sum() > 1:
            print(f"⚠️ Warning: More than one TEM sampling period detected in {file}")

    tem_times = df.loc[is_templog, "Datetime"]
    tem_start, tem_end = tem_times.iloc[0], tem_times.iloc[-1]
    avg_flow_ml_min = df.loc[is_templog, "Flow"].mean() * 1000  # l/min → ml/min

    duration_min = (date2num(tem_end) - date2num(tem_start)) * 24 * 60
    sample_volume_ml = avg_flow_ml_min * duration_min

    # Package sample info
    sample_meta = {
        "Start": [tem_start],
        "End": [tem_end],
        "Sample_vol [ml]": [sample_volume_ml]
    }

    # Create AerosolAlt object
    Par = AerosolAlt(df[["Datetime", "LDSA", "TEM", "Flow"]])
    Par._meta["instrument"] = "Partector"
    Par._meta["serial_number"] = meta_lines[0][-3:]
    Par._meta["unit"] = {"LDSA": "nm$^{2}$/cm$^{3}$", "TEM": "bool", "Flow": "l/min"}
    Par._meta["TEM_samples"] = pd.DataFrame(sample_meta)

    # Attach extra data
    if extra_data:
        extra_df = df.drop(columns=["LDSA", "TEM", "Flow"]).set_index("Datetime")
        Par._extra_data = extra_df

    return Par
