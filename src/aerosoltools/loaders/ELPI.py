# -*- coding: utf-8 -*-

from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
from ..aerosol2d import Aerosol2D
from .Common import detect_delimiter

def load_ELPI_metadata(file_path: Union[str, Path], 
                       delimiter: str = "\t", 
                       encoding: str = "utf-8") -> dict:
    """
    Extract metadata from an ELPI-formatted data file.

    This function reads the first ~36 lines of an ELPI export file to parse metadata 
    defined as key=value pairs. Values separated by the specified delimiter are 
    interpreted as lists, and numeric values are automatically converted to float.

    Parameters
    ----------
    file_path : str or Path
        Path to the ELPI data file.
    delimiter : str, optional
        Delimiter used for separating list values in metadata (default is tab).
    encoding : str, optional
        Encoding of the input file (default is 'utf-8').

    Returns
    -------
    dict
        Dictionary containing parsed metadata. Scalar values are converted to float 
        if possible. Tabular values are returned as lists (of floats or strings).
    """
    metadata = {}
    
    with open(file_path, 'r', encoding=encoding) as f:
        for row, line in enumerate(f):
            if row >= 36:
                break
            line = line.strip()
    
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
    
                # Split tab-separated values
                if delimiter in value:
                    items = value.split(delimiter)
                    try:
                        # Convert to list of floats if possible
                        items = [float(v) for v in items]
                        value = items
                    except ValueError:
                        value = items  # leave as strings
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # leave as string if not a float
    
                metadata[key] = value
    
    return metadata

def Load_ELPI_file(file: str, extra_data: bool = False):
    """
    Load data from an ELPI (.txt) file and convert it into an `Aerosol2D` object.

    This function reads ELPI exports (usually .txt files), extracts datetime and 
    particle size distribution information, applies unit conversions (e.g., from dW/dlogDp),
    and calculates total concentration and size-resolved particle data in cm⁻³.

    Parameters
    ----------
    file : str
        Path to the ELPI-exported .dat file.
    extra_data : bool, optional
        If True, retains and returns all non-distribution data in `.extra_data`. Default is False.

    Returns
    -------
    ELPI : Aerosol2D
        Object containing parsed size distribution data, total concentration, 
        and instrument metadata.

    Raises
    ------
    Exception
        If unit or weight format cannot be determined, or parsing fails.

    Notes
    -----
    - Bin mids and edges are stored in nanometers.
    - Normalization is done to convert to number concentration `dN`.
    - Supports dynamic density-aware edge recomputation when density ≠ 1.
    """
    encoding, delimiter = detect_delimiter(file, sample_lines=100)

    # Load metadata and bin descriptors
    meta = load_ELPI_metadata(file, delimiter, encoding)
    bin_edges = np.array(meta["D50values(um)"], dtype=float) * 1000
    bin_mids = np.array(meta["CalculatedDi(um)"], dtype=float) * 1000

    # Recalculate bin edges if non-unit density (mass data, not geometric cutpoints)
    if meta["Density(g/cm^3)"] != 1.0:
        bin_edges[1:-1] = np.sqrt(bin_mids[1:] * bin_mids[:-1])
        bin_edges[0] = bin_edges[1] ** 2 / bin_edges[2]
        bin_edges[-1] = bin_edges[-2] ** 2 / bin_edges[-3]
        print("################# Warning! #################")
        print("          Density ≠ 1.0 assumed           ")
        print("  Bin edges estimated via geometric means ")
        print("###########################################")

    # Load main data table, handling possible variations
    try:
        df = pd.read_csv(file, sep=delimiter, header=36, encoding=encoding)
        df = df.iloc[1:].reset_index(drop=True)
    except pd.errors.ParserError:
        df = pd.read_csv(file, sep=delimiter, header=None, skiprows=42, encoding=encoding)
        with open(file, encoding=encoding) as f:
            header_line = f.readlines()[39].strip().split(delimiter)
        while len(header_line) < df.shape[1]:
            header_line.append(f"Unnamed_{len(header_line)}")
        df.columns = header_line

    # Parse datetime
    df = df.rename(columns={"Date Time (yyyy/mm/dd hh:mm)": "Datetime"})
    try:
        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y/%m/%d %H:%M:%S")

    # Extract size distribution data and extra metadata
    dist_data = df.iloc[:, 34:48].copy()
    extra_df = df.drop(df.columns[33:47], axis=1)

    # Normalize bin data by reported type
    if "dW/dlogDp" in meta["CalculatedType"]:
        dlogDp = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
        dist_data *= dlogDp
    elif "dW/dDp" in meta["CalculatedType"]:
        dDp = bin_edges[1:] - bin_edges[:-1]
        dist_data *= dDp / 1000  # convert nm to µm
    elif "dW" in meta["CalculatedType"]:
        pass
    else:
        raise Exception("Unrecognized CalculatedType: expected dW, dW/dDp, or dW/dlogDp")

    # Apply moment-type normalization
    if "Mass" in meta["CalculatedMoment"]:
        density = meta["Density(g/cm^3)"]
        norm_vector = (np.pi / 6) * (bin_mids / 1e7) ** 3 * density * 1e12
    elif "Volume" in meta["CalculatedMoment"]:
        norm_vector = (np.pi / 6) * (bin_mids / 1e7) ** 3
    elif "Surface" in meta["CalculatedMoment"]:
        norm_vector = np.pi * bin_mids ** 2 * 1e-6
    elif "Number" in meta["CalculatedMoment"]:
        norm_vector = 1
    else:
        raise Exception("Unrecognized CalculatedMoment: expected Number, Surface, Mass, or Volume")

    dist_data = dist_data / norm_vector

    # Total concentration and column formatting
    total_conc = pd.DataFrame(np.nansum(dist_data, axis=1), columns=["Total_conc"])
    bin_mids = bin_mids.round(1)
    dist_data.columns = [str(mid) for mid in bin_mids]

    final_df = pd.concat([df["Datetime"], total_conc, dist_data], axis=1)

    # Construct Aerosol2D object
    ELPI = Aerosol2D(final_df)

    # Finalize metadata
    meta["density"] = meta.pop("Density(g/cm^3)")
    meta["bin_edges"] = bin_edges.round(1)
    meta["bin_mids"] = bin_mids
    meta["instrument"] = "ELPI"
    meta["serial_number"] = str(np.genfromtxt(file, delimiter=delimiter, skip_header=0, max_rows=1,
                                              dtype=str, encoding=encoding)).split(",")[1][1:-1]
    meta["dtype"] = "dN"
    meta["unit"] = "cm$^{-3}$"

    # Clean metadata
    for key in ["CalculatedDi(um)", "CalculatedType", "CalculatedMoment"]:
        meta.pop(key, None)

    ELPI._meta = meta
    if extra_data:
        extra_df.set_index('Datetime',inplace=True)
        ELPI._extra_data = extra_df

    return ELPI
