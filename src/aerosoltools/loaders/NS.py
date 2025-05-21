# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .Common import detect_delimiter

###############################################################################


def Load_NS_file(file: str, extra_data: bool = False) -> Aerosol2D:
    """
    Load and process NanoScan SMPS data exported in CSV format.

    This function reads NanoScan data, extracts time-resolved size distributions,
    computes bin edges and midpoints, and parses metadata including density and unit type.

    Parameters
    ----------
    file : str
        Path to the NanoScan CSV export file.
    extra_data : bool, optional
        If True, retains all non-distribution columns in the `.extra_data` attribute. Default is False.

    Returns
    -------
    NS : Aerosol2D
        Aerosol2D object containing:
        - Time-indexed size distribution and total concentration data
        - Instrument metadata (e.g., bin edges, serial number, density, unit)

    Raises
    ------
    Exception
        If the unit format or data type is unrecognized.

    Notes
    -----
    - Bin midpoints are extracted from column headers.
    - Bin edges are estimated using geometric means between bin midpoints.
    - Size distribution columns are assumed to span from column 1 to 13 (inclusive).
    """
    # Auto-detect file encoding and delimiter
    encoding, delimiter = detect_delimiter(file)

    # Load full dataset
    ns_df = pd.read_csv(
        file, delimiter=delimiter, decimal=".", header=5, encoding=encoding
    )
    ns_df.drop(columns=["File Index", "Sample #", "Total Conc"], inplace=True)

    # Extract bin midpoints and calculate bin edges
    bin_mids = np.array(ns_df.columns[1:14], dtype=float)
    bin_edges = np.append(10, np.append(np.sqrt(bin_mids[1:] * bin_mids[:-1]), 420))

    # Parse datetime
    ns_df.rename(columns={"Date Time": "Datetime"}, inplace=True)
    ns_df["Datetime"] = pd.to_datetime(ns_df["Datetime"], format="%Y/%m/%d %H:%M:%S")

    # Isolate size distribution data
    size_data = ns_df[bin_mids.astype(str)].copy()

    # Extract optional extra data
    if extra_data:
        ns_extra = ns_df.drop(columns=bin_mids.astype(str))
        ns_extra.set_index("Datetime", inplace=True)
    else:
        ns_extra = pd.DataFrame([])

    # Extract metadata
    serial_number = str(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            skip_header=2,
            max_rows=1,
            usecols=1,
            dtype=str,
            encoding=encoding,
        )
    )

    dtype_line = str(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            skip_header=5,
            max_rows=1,
            dtype=str,
            encoding=encoding,
        )
    )
    dtype = dtype_line.split(" ")[0]
    density = ns_df["Particle Density (g/cc)"].iloc[0]

    unit_dict = {
        "dN": "cm⁻³",
        "dS": "nm²/cm³",
        "dV": "nm³/cm³",
        "dM": "ug/m³",
    }

    try:
        unit = unit_dict[dtype[:2]]
    except KeyError:
        raise Exception("Unit and/or data type does not match the expected")

    # Compute total concentration and structure final output
    total_conc = size_data.sum(axis=1)
    total_col = pd.DataFrame({"Total_conc": total_conc})
    data_out = pd.concat([ns_df["Datetime"], total_col, size_data], axis=1)

    # Create Aerosol2D object
    NS = Aerosol2D(data_out)
    NS._meta["instrument"] = "NS"
    NS._meta["bin_edges"] = bin_edges.round(1)
    NS._meta["bin_mids"] = bin_mids.round(1)
    NS._meta["density"] = density
    NS._meta["serial_number"] = serial_number
    NS._meta["unit"] = unit
    NS._meta["dtype"] = dtype

    NS.convert_to_number_concentration()
    NS.unnormalize_logdp()

    if extra_data:
        NS._extra_data = ns_extra

    return NS
