# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import Common as Com
from ..aerosol2d import Aerosol2D

###############################################################################

def Load_NS_file(file: str, extra_data: bool = False) -> Aerosol2D:
    """
    Load and process NanoScan SMPS data exported as .csv files.

    Parameters
    ----------
    file : str
        Path to the NanoScan data file (CSV format).
    extra_data : bool, optional
        If True, returns additional columns in the `_extra_data` attribute of the class.
        Default is False.

    Returns
    -------
    NS : Aerosol2D
        Aerosol2D object containing:
        - Datetime-indexed total and binned concentration data
        - Metadata (instrument, density, bin info, dtype/unit)
        - Optionally extra data not part of core size distribution

    Notes
    -----
    - Bin midpoints are extracted from headers.
    - Bin edges are computed assuming log-normal spacing.
    - Units are converted based on header metadata.
    - Total concentration is recomputed by summing over bins.
    """
    # Auto-detect encoding and delimiter
    encoding, delimiter = Com.detect_delimiter(file)

    # Load full dataset
    ns_df = pd.read_csv(file, delimiter=delimiter, decimal='.', header=5, encoding=encoding)
    ns_df.drop(columns=['File Index', 'Sample #', 'Total Conc'], inplace=True)

    # Extract size bins and compute bin edges
    bin_mids = np.array(ns_df.columns[1:14], dtype=float)
    bin_edges = np.append(
        10, np.append(np.sqrt(bin_mids[1:] * bin_mids[:-1]), 420)
    )  # Assume logarithmic spacing

    # Parse datetime
    ns_df.rename(columns={'Date Time': 'Datetime'}, inplace=True)
    ns_df['Datetime'] = pd.to_datetime(ns_df['Datetime'], format="%Y/%m/%d %H:%M:%S")

    # Isolate size distribution data
    size_data = ns_df[bin_mids.astype(str)].copy()

    # Optional extra data
    if extra_data:
        ns_extra = ns_df.drop(columns=bin_mids.astype(str))
        ns_extra.set_index('Datetime', inplace=True)
    else:
        ns_extra = pd.DataFrame([])

    # Read dtype from header
    dtype_line = str(np.genfromtxt(file, delimiter=delimiter, skip_header=5, max_rows=1, dtype=str, encoding=encoding))
    dtype = dtype_line.strip()

    serial_number = str(np.genfromtxt(file, delimiter=delimiter, skip_header=2, max_rows=1, usecols=1, dtype=str, encoding=encoding))
    density = ns_df['Particle Density (g/cc)'].iloc[0]

    # Normalize if dlogDp or dDp present
    if 'dlogDp' in dtype:
        dlog_dp = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
        size_data = size_data.multiply(dlog_dp, axis=1)

    # Convert to number concentration
    if 'dM' in dtype:
        norm_vector = (np.pi / 6) * (bin_mids / 1e7) ** 3 * density * 1e12  # μg/m³
    elif 'dV' in dtype:
        norm_vector = (np.pi / 6) * (bin_mids / 1e7) ** 3  # nm³/cm³
    elif 'dS' in dtype:
        norm_vector = np.pi * bin_mids ** 2  # nm²/cm³
    elif 'dN' in dtype:
        norm_vector = 1
    else:
        raise ValueError(f"Unknown dtype format in header: {dtype}")

    # Final normalization
    size_data = size_data / norm_vector

    # Compute total concentration and reformat
    total_conc = size_data.sum(axis=1)
    total_col = pd.DataFrame({'Total_conc': total_conc})

    data_out = pd.concat([ns_df['Datetime'], total_col, size_data], axis=1)

    # Assemble output
    NS = Aerosol2D(data_out)
    NS._meta['instrument'] = 'NS'
    NS._meta['bin_edges'] = bin_edges.round(1)
    NS._meta['bin_mids'] = bin_mids.round(1)
    NS._meta['density'] = density
    NS._meta['serial_number'] = serial_number
    NS._meta['unit'] = 'cm$^{-3}$'
    NS._meta['dtype'] = 'dN'
    if extra_data:
        NS._extra_data = ns_extra

    return NS

