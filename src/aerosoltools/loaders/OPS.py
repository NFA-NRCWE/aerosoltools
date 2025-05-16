# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as datetime
import Common as Com
from ..aerosol2d import Aerosol2D

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
    encoding, delimiter = Com.detect_delimiter(file)

    # Peek at the first line to determine file type
    first_line = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding,
        skip_header=0, max_rows=1, dtype=str
    )[0]

    if first_line == "Sample File":
        return Load_OPS_AIM(file, extra_data=extra_data, encoding=encoding, delimiter=delimiter)
    elif first_line == "Instrument Name":
        return Load_OPS_Direct(file, extra_data=extra_data, encoding=encoding, delimiter=delimiter)
    else:
        raise Exception("Unrecognized OPS file format. Unable to parse.")

    
###############################################################################

def Load_OPS_AIM(file: str, extra_data: bool = False, encoding: str = None, delimiter: str = None):
    """
    Load OPS (Optical Particle Sizer) data exported by AIM software.

    This function processes AIM-exported OPS files by extracting timestamped
    size distribution data, converting units to number concentration, and constructing
    an `Aerosol2D` object. It handles bin normalization and applies mass/surface/number
    corrections based on file metadata.

    Parameters
    ----------
    file : str
        Path to the AIM-exported OPS data file.
    extra_data : bool, optional
        If True, attach all columns before sizebin data as `.extra_data`. Default is False.
    encoding : str, optional
        File encoding. If None, detected automatically (along with delimiter).
    delimiter : str, optional
        Field delimiter (e.g., ',' or '\t'). If None, detected automatically.

    Returns
    -------
    OPS : Aerosol2D
        Time-indexed object containing total concentration and size-resolved
        data, with metadata and optional extra data.

    Raises
    ------
    Exception
        If the file cannot be parsed, or expected metadata values are missing.

    Notes
    -----
    - Output data is normalized to `cm⁻³`.
    - Unit type (`dN`, `dS`, `dW`) is inferred and standardized to `dN`.
    - Bin midpoints and edges are stored in the `.meta` dictionary.
    - This loader is specific to the AIM format; see `Load_OPS_Direct` for instrument exports.
    """
    # Auto-detect encoding and delimiter if not provided
    if encoding is None and delimiter is None:
        encoding, delimiter = Com.detect_delimiter(file)
    elif encoding is None or delimiter is None:
        raise ValueError("Provide both encoding and delimiter, or let them be auto-detected.")

    # Read main table
    df = pd.read_csv(file, header=13, encoding=encoding, delimiter=delimiter)

    # Extract bin mids from column headers and convert to nm
    bin_mids = np.round(df.columns[17:33].astype(float) * 1000, 1)

    # Read bin edges from metadata section
    lb_edges = np.genfromtxt(file, delimiter=delimiter, encoding=encoding,
                              skip_header=10, max_rows=1)[17:-1]
    ub_edge = np.genfromtxt(file, delimiter=delimiter, encoding=encoding,
                             skip_header=11, max_rows=1, usecols=-2)
    bin_edges = np.append(lb_edges, ub_edge) * 1000  # convert to nm

    # Build datetime column from date and time
    df = df.rename(columns={"Sample #": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Start Time"], format="%m/%d/%Y %H:%M:%S")
    df.drop(columns=["Date", "Start Time"], inplace=True)

    # Extract raw sizebin data
    raw_data = df.iloc[:, 15:31].to_numpy()

    # Extract metadata from header lines
    meta = np.genfromtxt(file, delimiter=delimiter, encoding=encoding,
                         skip_header=1, max_rows=7, dtype=str)
    dtype = meta[5, 1]
    weight_type = meta[6, 1]

    # Normalize to raw dW if needed
    if "dW/dlogDp" in dtype:
        dlogDp = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
        raw_data *= dlogDp
    elif "dW/dDp" in dtype:
        dDp = bin_edges[1:] - bin_edges[:-1]
        raw_data *= dDp / 1000  # µm → cm
    elif "dW" in dtype:
        pass  # already raw mass
    else:
        raise ValueError("Unknown dtype format. Expected dW, dW/dDp, or dW/dlogDp.")

    # Apply weight-type normalization to get dN
    if "Mass" in weight_type:
        density = float(df.iloc[0, 40])
        volume = (np.pi / 6) * (bin_mids / 1e7)**3  # cm³
        norm_vector = volume * density * 1e12  # µg/cm³
    elif "Surface" in weight_type:
        norm_vector = np.pi * (bin_mids)**2 * 1e-6  # nm² to µm²/cm³
        density = 1
    elif "Number" in weight_type:
        norm_vector = 1
        density = 1
    else:
        raise ValueError("Unknown weight format. Expected Mass, Surface, or Number.")

    # Recalculate bin mids (ensures consistency with edges)
    bin_mids = ((bin_edges[1:] + bin_edges[:-1]) / 2).round(1)

    # Convert to dN
    normalized_data = raw_data / norm_vector
    total_conc = np.nansum(normalized_data, axis=1)

    # Build final DataFrame
    df_total = pd.DataFrame(total_conc, columns=["Total_conc"])
    df_bins = pd.DataFrame(normalized_data, columns=bin_mids.astype(str))
    df_final = pd.concat([df["Datetime"], df_total, df_bins], axis=1)

    # Instantiate Aerosol2D object
    OPS = Aerosol2D(df_final)
    OPS._meta["bin_edges"] = bin_edges
    OPS._meta["bin_mids"] = bin_mids
    OPS._meta["density"] = density
    OPS._meta["instrument"] = "OPS"
    OPS._meta["serial_number"] = meta[1, 1]
    OPS._meta["unit"] = "cm$^{-3}$"
    OPS._meta["dtype"] = "dN"

    # Handle optional extra data
    if extra_data:
        extra_df = df.drop(columns=df.columns[13:]).set_index("Datetime")
        OPS._extra_data = extra_df

    return OPS

###############################################################################

def Load_OPS_Direct(file: str, extra_data: bool = False, encoding: str = None, delimiter: str = None):
    """
    Load OPS (Optical Particle Sizer) data exported directly from the instrument.

    This function processes raw OPS data files exported directly from the device,
    converts particle counts to concentrations, and constructs an `Aerosol2D` object
    with appropriate metadata. The function supports optional inclusion of extra
    metadata and raw columns.

    Parameters
    ----------
    file : str
        Path to the CSV file exported directly from the OPS instrument.
    extra_data : bool, optional
        If True, attaches all non-sizebin columns and bin 17 data to `.extra_data`.
        Default is False.
    encoding : str, optional
        Character encoding for the file (e.g., 'utf-8'). If None, will be auto-detected.
    delimiter : str, optional
        Field delimiter (e.g., ',' or '\t'). If None, will be auto-detected.

    Returns
    -------
    OPS : Aerosol2D
        Time-indexed data object containing total concentration and size-resolved
        particle data, along with instrument metadata.

    Raises
    ------
    Exception
        If the file cannot be parsed, or metadata lines are malformed.

    Notes
    -----
    - Converts count data to concentration in cm⁻³ using flow rate and sample duration.
    - Bin 17 (particles >10 µm) is excluded from the main dataset but included in `.extra_data`.
    - Requires `Com.detect_delimiter()` for auto-formatting detection.
    """
    if encoding is None and delimiter is None:
        encoding, delimiter = Com.detect_delimiter(file)

    # Load measurement data, excluding last header-only bin
    df = pd.read_csv(file, header=37, encoding=encoding, delimiter=delimiter)

    # Extract metadata as key-value dict
    meta = pd.read_csv(
        file, header=None, nrows=35, encoding=encoding, delimiter=delimiter,
        dtype={0: str}
    ).set_index(0).squeeze().to_dict()

    # Parse starting datetime from metadata
    start_datetime = datetime.datetime.strptime(
        f"{meta['Test Start Date']} {meta['Test Start Time']}",
        "%Y/%m/%d %H:%M:%S"
    )

    # Convert elapsed time to full timestamps
    df["Datetime"] = pd.to_timedelta(df["Elapsed Time [s]"], unit="s") + start_datetime
    df.drop(columns=["Elapsed Time [s]"], inplace=True)

    # Determine sample length from metadata
    sample_interval = datetime.datetime.strptime(
        meta["Sample Interval [H:M:S]"], "%H:%M:%S"
    )
    sample_length = datetime.timedelta(
        hours=sample_interval.hour,
        minutes=sample_interval.minute,
        seconds=sample_interval.second
    ).total_seconds()

    # Apply correction: counts to concentration (excluding Bin 17)
    deadtime = df["Deadtime (s)"].to_numpy()
    counts = df.iloc[:, 1:17].to_numpy()  # Bin 1–16

    # Convert counts to concentration using flow rate (16.67 cm³/s)
    conc = np.true_divide(counts, 16.67 * (sample_length - deadtime[:, np.newaxis]))

    # If requested, store Bin 17 and other columns as extra data
    if extra_data:
        extra = df.drop(columns=df.columns[1:17]).copy()
        extra["Bin 17"] = extra["Bin 17"] / (16.67 * (sample_length - deadtime))
        extra.set_index("Datetime", inplace=True)
    else:
        extra = pd.DataFrame([])

    # Define bin edges and midpoints
    bin_edges = np.array([float(meta[f"Bin {i} Cut Point (um)"]) for i in range(1, 18)]) * 1000  # nm
    bin_mids = ((bin_edges[1:] + bin_edges[:-1]) / 2).round(1)  # nm

    # Compute total concentration
    total_conc = pd.DataFrame(np.nansum(conc, axis=1), columns=["Total_conc"])
    conc_df = pd.DataFrame(conc, columns=bin_mids.astype(str))
    df_final = pd.concat([df["Datetime"], total_conc, conc_df], axis=1)

    # Package into class
    OPS = Aerosol2D(df_final)
    OPS._meta["bin_edges"] = bin_edges
    OPS._meta["bin_mids"] = bin_mids
    OPS._meta["density"] = meta["Density"]
    OPS._meta["instrument"] = "OPS"
    OPS._meta["serial_number"] = meta["Serial Number"]
    OPS._meta["unit"] = "cm$^{-3}$"
    OPS._meta["dtype"] = "dN"
    if extra_data:
        OPS._extra_data = extra

    return OPS
