# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .Common import detect_delimiter

###############################################################################


def load_ELPI_metadata(
    file_path: Union[str, Path], delimiter: str = "\t", encoding: str = "utf-8"
) -> dict:
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

    with open(file_path, "r", encoding=encoding) as f:
        for row, line in enumerate(f):
            if row >= 36:
                break
            line = line.strip()

            if "=" in line:
                key, value = line.split("=", 1)
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


###############################################################################


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

    encoding, delimiter = detect_delimiter(file)

    # Load metadata and bin descriptors
    meta = load_ELPI_metadata(file, delimiter, encoding)
    try:
        bin_edges = np.array(meta["D50values(um)"], dtype=float) * 1000
    except ValueError:
        bin_edges = np.array(meta["D50values(um)"].split("\t"), dtype=float) * 1000
    try:
        bin_mids = np.array(meta["CalculatedDi(um)"], dtype=float) * 1000
    except ValueError:
        bin_mids = np.array(meta["CalculatedDi(um)"].split("\t"), dtype=float) * 1000

    # Recalculate bin edges if non-unit density (mass data, not geometric cutpoints)
    if meta["Density(g/cm^3)"] != 1.0:
        bin_edges[1:-1] = np.sqrt(bin_mids[1:] * bin_mids[:-1])
        bin_edges[0] = bin_edges[1] ** 2 / bin_edges[2]
        bin_edges[-1] = bin_edges[-2] ** 2 / bin_edges[-3]
        print("################# Warning! #################")
        print("          Density ≠ 1.0 assumed           ")
        print("  Bin edges estimated via geometric means ")
        print("###########################################")

    # --- Load main data table robustly + parse datetime (no column drops) ---
    with open(file, encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    # Find the [Data] marker
    try:
        data_idx = next(i for i, line in enumerate(lines) if line.strip() == "[Data]")
    except StopIteration:
        raise ValueError("Couldn't find the [Data] marker in the file.")

    # Find the header row right before [Data]
    # Prefer the last non-section line that contains the delimiter
    j = data_idx - 1
    while j >= 0 and (not lines[j].strip() or lines[j].lstrip().startswith("[")):
        j -= 1
    if j < 0 or delimiter not in lines[j]:
        # search a bit further up as a fallback
        found = False
        for k in range(data_idx - 1, max(-1, data_idx - 25), -1):
            if delimiter in lines[k] and not lines[k].lstrip().startswith("["):
                j = k
                found = True
                break
        if not found:
            raise ValueError("Couldn't find the header line before [Data].")

    header = [
        h.strip() for h in lines[j].lstrip("\ufeff").rstrip("\r\n").split(delimiter)
    ]

    # Read the data rows AFTER the [Data] line
    df = pd.read_csv(
        file,
        sep=delimiter,
        header=None,
        skiprows=data_idx + 1,
        encoding=encoding,
        engine="python",
        on_bad_lines="skip",
    )

    # Align header length with actual number of columns (no reordering)
    if len(header) < df.shape[1]:
        header += [f"Unnamed_{i}" for i in range(len(header), df.shape[1])]
    elif len(header) > df.shape[1]:
        header = header[: df.shape[1]]
    df.columns = header

    # Force the first column to be called "Datetime" (keep order intact)
    cols = list(df.columns)
    cols[0] = "Datetime"
    df.columns = cols

    # Parse datetimes: strict formats first, then permissive fallback
    formats = ["%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"]
    parsed = None
    for fmt in formats:
        s = pd.to_datetime(df["Datetime"], format=fmt, errors="coerce")
        if s.notna().mean() > 0.98:  # accept if ~all rows parse
            parsed = s
            break
    if parsed is None:
        s = pd.to_datetime(df["Datetime"], errors="coerce")
        parsed = s
    if parsed.isna().mean() > 0.2:
        raise ValueError(
            "Datetime parsing failed for many rows — check the first column."
        )
    df["Datetime"] = parsed
    # --- end robust load + parse ---

    # Extract size distribution data and extra metadata (indices preserved)
    dist_data = df.iloc[:, 34:48].copy()
    extra_df = df.drop(df.columns[33:47], axis=1)

    # Checks unit format (dW, dW/dP, dW/dlogDp)
    Unit_dict = {"Nu": "cm⁻³", "Su": "nm²/cm³", "Vo": "nm³/cm³", "Ma": "ug/m³"}
    dtype_dict = {"Nu": "dN", "Su": "dS", "Vo": "dV", "Ma": "dM"}

    try:
        Unit = Unit_dict[meta["CalculatedMoment"][:2]]
        dtype = dtype_dict[meta["CalculatedMoment"][:2]] + meta["CalculatedType"][2:]
    except (KeyError, TypeError) as e:
        raise Exception(
            "Unit and/or data type does not match the expected - ensure that the data has been converted from Current (fA) via the ELPI software"
        ) from e

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

    # Serial number (kept from your original approach)
    if delimiter == ",":
        serial_n = str(
            np.genfromtxt(
                file,
                delimiter=delimiter,
                skip_header=0,
                max_rows=1,
                dtype=str,
                encoding=encoding,
            )
        )[1][1:-1]
    else:
        serial_n = str(
            np.genfromtxt(
                file,
                delimiter=delimiter,
                skip_header=0,
                max_rows=1,
                dtype=str,
                encoding=encoding,
            )
        ).split(",")[1][1:-1]

    meta["serial_number"] = serial_n
    meta["dtype"] = dtype
    meta["unit"] = Unit

    # Clean metadata
    for key in ["CalculatedDi(um)", "CalculatedType", "CalculatedMoment"]:
        meta.pop(key, None)

    ELPI._meta = meta
    ELPI.convert_to_number_concentration()
    ELPI.unnormalize_logdp()

    if extra_data:
        extra_df.set_index("Datetime", inplace=True)
        ELPI._extra_data = extra_df

    return ELPI
