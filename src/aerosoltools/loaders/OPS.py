import datetime as datetime
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
    file: str,
    extra_data: bool = False,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
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
    ValueError
        If only one of encoding or delimiter is provided.
    """

    # Detect when both are omitted
    if encoding is None and delimiter is None:
        encoding, delimiter = detect_delimiter(file)  # -> Tuple[str, str]

    # If only one was provided, that’s ambiguous
    if (encoding is None) != (delimiter is None):
        raise ValueError("Either provide both encoding and delimiter, or neither.")

    # Normalize to concrete strings for type checker
    enc: str = encoding  # type: ignore[assignment]
    delim: str = delimiter  # type: ignore[assignment]

    # --- main table via pandas ------------------------------------------------
    df = pd.read_csv(file, header=13, encoding=enc, delimiter=delim)

    # OPS: mid diameters are in columns 17..32 (inclusive) in µm -> convert to nm
    bin_mids: NDArray[np.float64] = np.round(
        np.array(df.columns[17:33], dtype=float) * 1000.0, 1
    )

    # --- edges via genfromtxt, using file handles to avoid 'encoding' kw warning ----
    # lower edges row
    with open(file, "r", encoding=enc) as fh1:
        bin_lb = np.genfromtxt(
            fh1,
            delimiter=delim,
            skip_header=10,
            max_rows=1,
        )[
            17:-1
        ]  # all but the last column on that row

    # upper edge is the penultimate column on the next row
    with open(file, "r", encoding=enc) as fh2:
        # read a single value from the (-2) column on that row
        bin_ub_arr = np.genfromtxt(
            fh2,
            delimiter=delim,
            skip_header=11,
            max_rows=1,
            usecols=[-2],  # list to satisfy numpy's type stubs
        )
    # Ensure scalar float
    bin_ub = (
        float(bin_ub_arr)
        if isinstance(bin_ub_arr, (np.floating, float))
        else float(np.asarray(bin_ub_arr).item())
    )

    # Build edges in nm
    bin_edges: NDArray[np.float64] = np.append(bin_lb, [bin_ub]) * 1000.0

    # --- timestamp and columns ----------------------------------------------
    # The AIM export has "Date", "Start Time" -> build a single datetime column
    df.rename(columns={"Sample #": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Start Time"], format="%m/%d/%Y %H:%M:%S"
    )
    df.drop(columns=["Date", "Start Time"], inplace=True)

    # Distribution block (positions may vary in different exports; adjust if needed)
    dist_block = df.iloc[:, 15:31]
    dist_data = dist_block.to_numpy()
    total_conc = pd.DataFrame(np.nansum(dist_data, axis=1), columns=["Total_conc"])
    dist_df = pd.DataFrame(dist_data, columns=bin_mids.astype(str))
    final_df = pd.concat([df["Datetime"], total_conc, dist_df], axis=1)

    # Optional extra-data (everything beyond column 13 kept, indexed by time)
    if extra_data:
        ops_extra = df.drop(columns=df.columns[13:])
        ops_extra.set_index("Datetime", inplace=True)

    # --- metadata block ------------------------------------------------------
    with open(file, "r", encoding=enc) as fh3:
        meta = np.genfromtxt(
            fh3,
            delimiter=delim,
            skip_header=1,
            max_rows=7,
            dtype=str,
        )
    weight = str(meta[6, 1])  # e.g., "Nu" prefix present
    dtype_desc = str(meta[5, 1])
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
    except KeyError as e:
        raise ValueError(
            "Unit and/or data type does not match the expected format."
        ) from e

    # --- assemble object -----------------------------------------------------
    OPS = Aerosol2D(final_df)
    OPS._meta["bin_edges"] = bin_edges
    OPS._meta["bin_mids"] = bin_mids
    OPS._meta["density"] = density
    OPS._meta["instrument"] = "OPS"
    OPS._meta["serial_number"] = str(meta[1, 1])
    OPS._meta["unit"] = unit
    OPS._meta["dtype"] = dtype

    OPS.convert_to_number_concentration()
    OPS.unnormalize_logdp()

    if extra_data:
        OPS._extra_data = ops_extra  # type: ignore[assignment]

    return OPS


###############################################################################


def Load_OPS_Direct(
    file: str,
    extra_data: bool = False,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> Aerosol2D:
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
    # Detect when both are omitted
    both_missing = encoding is None and delimiter is None
    if both_missing:
        encoding, delimiter = detect_delimiter(file)  # -> Tuple[str, str]

    # If only one was provided, that’s ambiguous
    if (encoding is None) != (delimiter is None):
        raise ValueError("Either provide both encoding and delimiter, or neither.")

    enc: str = encoding  # type: ignore[assignment]
    delim: str = delimiter  # type: ignore[assignment]

    # Load measurement data, excluding last header-only bin
    df = pd.read_csv(file, header=37, encoding=enc, delimiter=delim)

    # Extract metadata as key-value dict
    meta = (
        pd.read_csv(
            file,
            header=None,
            nrows=35,
            encoding=enc,
            delimiter=delim,
            dtype={0: str},
        )
        .set_index(0)
        .squeeze()
        .to_dict()  # type: ignore
    )

    # Parse starting datetime from metadata
    start_datetime = datetime.datetime.strptime(
        f"{meta['Test Start Date']} {meta['Test Start Time']}", "%Y/%m/%d %H:%M:%S"
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
        seconds=sample_interval.second,
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
    bin_edges = (
        np.array([float(meta[f"Bin {i} Cut Point (um)"]) for i in range(1, 18)]) * 1000
    )  # nm
    bin_mids = ((bin_edges[1:] + bin_edges[:-1]) / 2).round(1)  # nm

    # Compute total concentration
    total_conc = pd.DataFrame(np.nansum(conc, axis=1), columns=["Total_conc"])
    conc_df = pd.DataFrame(conc, columns=bin_mids.astype(str))
    df_final = pd.concat([df["Datetime"], total_conc, conc_df], axis=1)

    # Package into class
    OPS = Aerosol2D(df_final)
    OPS._meta["bin_edges"] = bin_edges
    OPS._meta["bin_mids"] = bin_mids
    OPS._meta["density"] = float(meta["Density"])
    OPS._meta["instrument"] = "OPS"
    OPS._meta["serial_number"] = meta["Serial Number"]
    OPS._meta["unit"] = "cm$^{-3}$"
    OPS._meta["dtype"] = "dN"
    if extra_data:
        OPS._extra_data = extra

    return OPS
