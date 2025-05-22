# -*- coding: utf-8 -*-

import datetime as datetime

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .Common import detect_delimiter

###############################################################################


def Load_OPS_file(file: str, extra_data: bool = False):
    """
    Load data from an OPS (Optical Particle Sizer) file and route to the appropriate parser.

    This function inspects the file header and determines whether the file was exported
    via the AIM software or directly from the instrument. Based on this, it dispatches
    to the correct loader: `Load_OPS_AIM` or `Load_OPS_Direct`.

    Parameters
    ----------
    file : str
        Path to the OPS data file.
    extra_data : bool, optional
        If True, attaches unused columns to the returned object as `._extra_data`.
        Passed directly to the underlying loader. Default is False.

    Returns
    -------
    OPS : Aerosol2D
        A class containing size-resolved particle data and instrument metadata.

    Raises
    ------
    Exception
        If the file type cannot be identified from the header.

    Notes
    -----
    - This function depends on `Com.detect_delimiter()` and assumes either AIM-exported
      or direct instrument export file formats.
    - If new formats are introduced, this function should be updated accordingly.
    """
    encoding, delimiter = detect_delimiter(file)

    # Peek at the first line to determine file type
    first_line = np.genfromtxt(
        file,
        delimiter=delimiter,
        encoding=encoding,
        skip_header=0,
        max_rows=1,
        dtype=str,
    )[0]

    if first_line == "Sample File":
        return Load_OPS_AIM(
            file, extra_data=extra_data, encoding=encoding, delimiter=delimiter
        )
    elif first_line == "Instrument Name":
        return Load_OPS_Direct(
            file, extra_data=extra_data, encoding=encoding, delimiter=delimiter
        )
    else:
        raise Exception("Unrecognized OPS file format. Unable to parse.")


###############################################################################


def Load_OPS_AIM(
    file: str, extra_data: bool = False, encoding: str = None, delimiter: str = None
) -> Aerosol2D:
    """
    Load data from OPS instrument as exported by AIM software.

    Parameters
    ----------
    file : str
        Path to the OPS AIM-exported data file.
    extra_data : bool, optional
        If True, includes all non-distribution columns in `.extra_data`.
    encoding : str, optional
        Encoding format. If None, detected automatically.
    delimiter : str, optional
        Delimiter format. If None, detected automatically.

    Returns
    -------
    OPS : Aerosol2D
        Object containing time-resolved particle number distributions and metadata.

    Raises
    ------
    Exception
        If only one of encoding or delimiter is provided.
    """
    if encoding is None and delimiter is None:
        encoding, delimiter = detect_delimiter(file)
    elif encoding is None or delimiter is None:
        raise Exception("Either provide both encoding and delimiter, or neither.")

    df = pd.read_csv(file, header=13, encoding=encoding, delimiter=delimiter)

    bin_mids = np.round(np.array(df.columns[17:33], dtype=float) * 1000, 1)

    bin_lb = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, skip_header=10, max_rows=1
    )[17:-1]
    bin_ub = np.genfromtxt(
        file,
        delimiter=delimiter,
        encoding=encoding,
        skip_header=11,
        max_rows=1,
        usecols=-2,
    )
    bin_edges = np.append(bin_lb, [bin_ub]) * 1000

    df.rename(columns={"Sample #": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Start Time"], format="%m/%d/%Y %H:%M:%S"
    )
    df.drop(columns=["Date", "Start Time"], inplace=True)

    dist_data = df.iloc[:, 15:31].to_numpy()

    if extra_data:
        ops_extra = df.drop(columns=df.columns[13:])
        ops_extra.set_index("Datetime", inplace=True)

    meta = np.genfromtxt(
        file,
        delimiter=delimiter,
        encoding=encoding,
        skip_header=1,
        max_rows=7,
        dtype=str,
    )
    weight = meta[6, 1]
    dtype_desc = meta[5, 1]
    density = 1.0

    unit_dict = {"Nu": "cm⁻³", "Su": "nm²/cm³", "Vo": "nm³/cm³", "Ma": "ug/m³"}
    dtype_dict = {"Nu": "dN", "Su": "dS", "Vo": "dV", "Ma": "dM"}

    try:
        unit = unit_dict[weight[:2]]
        if "dlogDp" in dtype_desc:
            dtype = dtype_dict[weight[:2]] + "/dlogDp"
        elif "dDp" in dtype_desc:
            dtype = dtype_dict[weight[:2]] + "/dDp"
        else:
            dtype = dtype_dict[weight[:2]]
    except KeyError:
        raise Exception("Unit and/or data type does not match the expected format.")

    total_conc = pd.DataFrame(np.nansum(dist_data, axis=1), columns=["Total_conc"])
    dist_data = pd.DataFrame(dist_data, columns=bin_mids.astype(str))
    final_df = pd.concat([df["Datetime"], total_conc, dist_data], axis=1)

    OPS = Aerosol2D(final_df)
    OPS._meta["bin_edges"] = bin_edges
    OPS._meta["bin_mids"] = bin_mids
    OPS._meta["density"] = density
    OPS._meta["instrument"] = "OPS"
    OPS._meta["serial_number"] = meta[1, 1]
    OPS._meta["unit"] = unit
    OPS._meta["dtype"] = dtype

    OPS.convert_to_number_concentration()
    OPS.unnormalize_logdp()

    if extra_data:
        OPS._extra_data = ops_extra

    return OPS


###############################################################################


def Load_OPS_Direct(
    file: str, extra_data: bool = False, encoding: str = None, delimiter: str = None
) -> Aerosol2D:
    """
    Load data from OPS exported directly from the instrument (not via AIM).

    Parameters
    ----------
    file : str
        Path to the raw OPS export file.
    extra_data : bool, optional
        If True, includes all non-distribution columns in `.extra_data`.
    encoding : str, optional
        Encoding format. If None, detected automatically.
    delimiter : str, optional
        Delimiter format. If None, detected automatically.

    Returns
    -------
    OPS : Aerosol2D
        Object containing particle concentrations corrected from counts and metadata.
    """
    if encoding is None and delimiter is None:
        encoding, delimiter = detect_delimiter(file)

    df = pd.read_csv(file, header=37, encoding=encoding, delimiter=delimiter)

    meta_raw = pd.read_csv(
        file,
        encoding=encoding,
        delimiter=delimiter,
        header=None,
        nrows=35,
        dtype={0: str},
    )
    meta = meta_raw.set_index(0).squeeze().to_dict()

    start_time = datetime.datetime.strptime(
        meta["Test Start Date"] + " " + meta["Test Start Time"], "%Y/%m/%d %H:%M:%S"
    )
    df["Datetime"] = [
        start_time + datetime.timedelta(seconds=s) for s in df["Elapsed Time [s]"]
    ]
    df.drop(columns=["Elapsed Time [s]"], inplace=True)

    sample_interval = datetime.datetime.strptime(
        meta["Sample Interval [H:M:S]"], "%H:%M:%S"
    ).second
    deadtime = np.array(df["Deadtime (s)"])

    dist_data = df.iloc[:, 0:16].to_numpy()
    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # ignore warnings from division by zero (they are set to NaN)
        dist_data = np.true_divide(
            dist_data, (16.67 * (sample_interval - deadtime[:, np.newaxis]))
        )

    if extra_data:
        ops_extra = df.drop(columns=df.columns[0:16])
        ops_extra["Bin 17"] = ops_extra["Bin 17"] / (
            16.67 * (sample_interval - deadtime)
        )
        ops_extra.set_index("Datetime", inplace=True)
    else:
        ops_extra = pd.DataFrame([])

    bin_edges = (
        np.array([meta[f"Bin {i} Cut Point (um)"] for i in range(1, 18)], dtype=float)
        * 1000
    )
    bin_mids = np.array((bin_edges[1:] + bin_edges[:-1]) / 2, dtype=float).round(1)

    total_conc = pd.DataFrame(np.nansum(dist_data, axis=1), columns=["Total_conc"])
    dist_data = pd.DataFrame(dist_data, columns=bin_mids.astype(str))
    final_df = pd.concat([df["Datetime"], total_conc, dist_data], axis=1)

    OPS = Aerosol2D(final_df)
    OPS._meta["bin_edges"] = bin_edges
    OPS._meta["bin_mids"] = bin_mids
    OPS._meta["density"] = float(meta.get("Density", 1.0))
    OPS._meta["instrument"] = "OPS"
    OPS._meta["serial_number"] = meta.get("Serial Number", "Unknown")
    OPS._meta["unit"] = "cm⁻³"
    OPS._meta["dtype"] = "dN"

    if extra_data:
        OPS._extra_data = ops_extra

    return OPS
