# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .Common import detect_delimiter

###############################################################################


def Load_FMPS_file(file: str) -> Aerosol2D:
    """
    Dispatcher for FMPS file loading. Detects raw format and raises exception,
    otherwise routes to the FMPS software-export parser in order to load the
    data correctly.

    Parameters
    ----------
    file : str
        Path to the FMPS-exported file.

    Returns
    -------
    FMPS : Aerosol2D
        Structured aerosol size distribution object.

    Raises
    ------
    Exception
        If the FMPS file is raw and unsupported.
    """
    encoding, delimiter = detect_delimiter(file)
    header_check = str(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=12,
            max_rows=1,
            dtype=str,
        )
    )

    if "Raw" in header_check:
        raise Exception(
            f"{file} is exported as raw, and needs to be treated by the software."
        )
    return _load_fmps_software(file, encoding, delimiter)


###############################################################################


def _load_fmps_software(file: str, encoding: str, delimiter: str) -> Aerosol2D:
    """
    Load data from FMPS exported file and convert into Aerosol2D object.

    Parameters
    ----------
    file : str
        Path to the FMPS file.
    encoding : str
        File encoding.
    delimiter : str
        Delimiter used in the file.

    Returns
    -------
    FMPS : Aerosol2D
        Processed particle distribution with metadata.
    """
    # Load bin values and concentration data
    bin_mids = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, skip_header=13, max_rows=1
    )[1:-11]
    bin_edges = np.append(5.6, (bin_mids[1:] + bin_mids[:-1]) / 2)
    bin_edges = np.append(bin_edges, 560)

    data_array = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, skip_header=15
    )
    dist_data = data_array[:, 1:-11]
    total_conc = pd.DataFrame(np.nansum(dist_data, axis=1), columns=["Total_conc"])

    # Parse timestamp
    time_format = str(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=14,
            max_rows=1,
            dtype=str,
        )[0]
    )
    try:
        datetime_df = _parse_danish_datetime(file, delimiter, encoding, time_format)
    except (IndexError, ValueError, KeyError):
        datetime_df = _parse_standard_datetime(file, delimiter, encoding)

    # Extract metadata
    datatype = str(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=12,
            max_rows=1,
            dtype=str,
            usecols=0,
        )
    ).split(" ")[0]
    serial_number = str(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=4,
            max_rows=1,
            dtype=str,
        )[2]
    )[-8:]

    dtype_dict = {"Co": "dN", "dN": "dN", "Su": "dS", "Vo": "dV", "Ma": "dM"}
    unit_dict = {"dN": "cm⁻³", "dS": "nm²/cm³", "dV": "nm³/cm³", "dM": "ug/m³"}

    try:
        if "dlogDp" in datatype:
            dtype = dtype_dict[datatype[:2]] + "/dlogDp"
        else:
            dtype = dtype_dict[datatype[:2]]
        unit = unit_dict[dtype[:2]]
    except KeyError:
        raise Exception("Unit and/or data type does not match the expected format.")

    # Format data
    dist_df = pd.DataFrame(dist_data, columns=bin_mids.astype(str))
    output_df = pd.concat([datetime_df, total_conc, dist_df], axis=1)

    # Create Aerosol2D object
    FMPS = Aerosol2D(output_df)
    FMPS._meta = {
        "instrument": "FMPS",
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "density": 1.0,
        "serial_number": serial_number,
        "unit": unit,
        "dtype": dtype,
    }
    FMPS.convert_to_number_concentration()
    FMPS.unnormalize_logdp()
    return FMPS


###############################################################################


def _parse_danish_datetime(file, delimiter, encoding, time_format):
    """Parses FMPS datetime strings in Danish date format."""
    date_str = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, max_rows=1, dtype=str
    )[1]
    date_str = date_str.split('"')[1]

    day, month_name, year = date_str.split(" ")[:3]
    hour, minute, second = map(int, date_str.split(" ")[-1].split(":"))

    months = {
        "januar": 1,
        "februar": 2,
        "marts": 3,
        "april": 4,
        "maj": 5,
        "juni": 6,
        "juli": 7,
        "august": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "december": 12,
    }
    start_dt = datetime.datetime(
        int(year), months[month_name], int(day.replace(".", "")), hour, minute, second
    )

    if "Elapsed" in time_format:
        times = np.genfromtxt(
            file, delimiter=delimiter, encoding=encoding, skip_header=15, dtype=float
        )[:, 0]
        return pd.DataFrame(
            [start_dt + datetime.timedelta(seconds=int(t)) for t in times],
            columns=["Datetime"],
        )
    else:
        time_list = np.genfromtxt(
            file, delimiter=delimiter, encoding=encoding, skip_header=15, dtype=str
        )[:, 0]
        step = datetime.datetime.strptime(
            time_list[1], "%H:%M:%S"
        ) - datetime.datetime.strptime(time_list[0], "%H:%M:%S")
        return pd.DataFrame(
            [start_dt + i * step for i in range(len(time_list))], columns=["Datetime"]
        )


###############################################################################


def _parse_standard_datetime(file, delimiter, encoding):
    """Parses FMPS datetime strings in standard English format."""
    fmps_date = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, max_rows=1, dtype=str
    )[2:]
    month_map = {
        k: v
        for v, k in enumerate(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            1,
        )
    }

    month_str = fmps_date[0].split(" ")[1][:3]
    day = int(fmps_date[0].split(" ")[2])
    year = int(fmps_date[1].split(" ")[1])
    time = fmps_date[1].split(" ")[3] + " " + fmps_date[1].split(" ")[4][:2]
    base_time = datetime.datetime.strptime(time, "%I:%M:%S %p")

    start_dt = datetime.datetime(
        year,
        month_map[month_str],
        day,
        base_time.hour,
        base_time.minute,
        base_time.second,
    )

    time_strs = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, skip_header=15, dtype=str
    )[:, 0]
    times = [datetime.datetime.strptime(t, "%I:%M:%S %p") for t in time_strs]
    step = times[1] - times[0]

    return pd.DataFrame(
        [start_dt + i * step for i in range(len(times))], columns=["Datetime"]
    )
