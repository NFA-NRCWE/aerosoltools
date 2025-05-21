# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .Common import detect_delimiter

###############################################################################


def Load_Grimm_file(file: str) -> Aerosol2D:
    """
    Load data from a Grimm spectrometer file, either software-exported or instrument-direct.
    The file type is detected automatically and passed to the correct loader.

    Parameters
    ----------
    file : str
        Path to the Grimm data file.

    Returns
    -------
    grimm : Aerosol2D
        Aerosol2D object with datetime-resolved particle concentration data.

    Raises
    ------
    Exception
        If the file format is not recognized.
    """
    encoding, delimiter = detect_delimiter(file)
    header_line = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, max_rows=1, dtype=str
    )

    try:
        if "File name" in header_line[0]:
            return Load_Grimm_inst(file, encoding, delimiter)
    except:
        if header_line == "<Header>":
            return Load_Grimm_soft(file, encoding, delimiter)
        else:
            raise Exception("Unrecognized Grimm file format.")


###############################################################################


def Load_Grimm_soft(file: str, encoding: str, delimiter: str) -> Aerosol2D:
    """
    Load Grimm data exported via software.

    Parameters
    ----------
    file : str
        Path to the software-exported Grimm data file.
    encoding : str
        File encoding.
    delimiter : str
        Field delimiter.

    Returns
    -------
    grimm : Aerosol2D
        Object with parsed size-distribution and metadata.
    """
    df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, header=13)
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df = df.dropna().reset_index(drop=True)

    df["Datetime"] = [
        datetime.datetime.strptime(s.split(".")[0][-19:], "%d/%m/%Y %H:%M:%S")
        for s in df["Datetime"]
    ]

    bin_edges_str = df.columns[1:]
    bin_edges = (
        np.array([float(label[:-3]) for label in bin_edges_str] + [34]) * 1000
    )  # nm
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    meta = np.genfromtxt(
        file,
        delimiter=delimiter,
        encoding=encoding,
        skip_header=1,
        max_rows=10,
        dtype=str,
    )
    dtype_raw = meta[6].split(" ")[1]
    location = meta[1].split(":")[1]
    serial_number = meta[6].split(":")[1]

    # Normalize based on data type
    if "g/m" in dtype_raw:
        norm_vector = (np.pi / 6) * (bin_mids / 1e7) ** 3 * 1 * 1e12
    elif "/l" in dtype_raw:
        norm_vector = 1000
    else:
        raise Exception("Unsupported data type format in software export.")

    raw_data = df.iloc[:, 1:].astype(float).values / norm_vector
    total_conc = pd.DataFrame(np.nansum(raw_data, axis=1), columns=["Total_conc"])
    dist_df = pd.DataFrame(raw_data, columns=bin_mids.astype(str))
    final_df = pd.concat([df["Datetime"], total_conc, dist_df], axis=1)

    grimm = Aerosol2D(final_df)
    grimm._meta = {
        "instrument": "Grimm",
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "density": 1.0,
        "serial_number": serial_number,
        "location": location,
        "unit": "cm⁻³",
        "dtype": "dN",
    }

    return grimm


###############################################################################


def Load_Grimm_inst(file: str, encoding: str, delimiter: str) -> Aerosol2D:
    """
    Load Grimm data exported directly from the instrument.

    Parameters
    ----------
    file : str
        Path to the direct-export Grimm file.
    encoding : str
        File encoding.
    delimiter : str
        Field delimiter.

    Returns
    -------
    grimm : Aerosol2D
        Parsed object with datetime and size-resolved particle data.
    """
    df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, header=1)
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df = df.dropna().reset_index(drop=True)

    # Normalize incomplete timestamps by appending midnight
    times = []
    for t in df["Datetime"]:
        try:
            times.append(datetime.datetime.strptime(t, "%m/%d/%Y %I:%M:%S %p"))
        except ValueError:
            times.append(
                datetime.datetime.strptime(t + " 12:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
            )
    df["Datetime"] = times

    bin_labels = df.columns[1:-1]
    bin_edges = (
        np.array([float(b.split("-")[0]) for b in bin_labels] + [32, 34]) * 1000
    )  # nm
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    meta = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, max_rows=1, dtype=str
    )
    dtype_raw = meta[2].split(" ")[1]

    if "Mass" in dtype_raw:
        norm_vector = (np.pi / 6) * (bin_mids / 1e7) ** 3 * 1 * 1e12
    elif "Count" in dtype_raw:
        norm_vector = 1000
    else:
        raise Exception("Unsupported data type format in instrument export.")

    raw_data = df.iloc[:, 1:].astype(float).values / norm_vector
    total_conc = pd.DataFrame(np.nansum(raw_data, axis=1), columns=["Total_conc"])
    dist_df = pd.DataFrame(raw_data, columns=bin_mids.astype(str))
    final_df = pd.concat([df["Datetime"], total_conc, dist_df], axis=1)

    grimm = Aerosol2D(final_df)
    grimm._meta = {
        "instrument": "Grimm",
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "density": 1.0,
        "serial_number": "Unknown",
        "unit": "cm⁻³",
        "dtype": "dN",
    }

    return grimm
