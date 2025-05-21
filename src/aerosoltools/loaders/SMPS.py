# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .Common import detect_delimiter

###############################################################################


def load_SMPS_metadata(
    file_path: Union[str, Path], delimiter: str = ",", encoding: str = "iso-8859-1"
) -> dict:
    """
    Extract metadata from an SMPS-formatted data file.

    Parameters
    ----------
    file_path : str or Path
        Path to the SMPS export file.
    delimiter : str, optional
        Delimiter used in the file (default is comma).
    encoding : str, optional
        File encoding (default is 'iso-8859-1').

    Returns
    -------
    dict
        Parsed key-value metadata, with list or float conversion when possible.
    """
    metadata = {}
    with open(file_path, "r", encoding=encoding) as f:
        for i, line in enumerate(f):
            if i >= 25:
                break
            line = line.strip()
            if "," in line:
                key, value = line.split(",", 1)
                key = key.strip()
                value = value.strip()

                if delimiter in value:
                    try:
                        value = [float(v) for v in value.split(delimiter)]
                    except ValueError:
                        value = value.split(delimiter)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

                metadata[key] = value
    return metadata


###############################################################################


def Load_SMPS_file(file: str, extra_data: bool = False) -> Aerosol2D:
    """
    Load SMPS data exported as a text file and structure it into an Aerosol2D object.

    Parameters
    ----------
    file : str
        Path to the SMPS .txt export file.
    extra_data : bool, optional
        If True, stores additional columns in `.extra_data`. Default is False.

    Returns
    -------
    SMPS : Aerosol2D
        Object containing time-resolved particle size distribution and metadata.
    """
    encoding, delimiter = detect_delimiter(file)
    df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, header=25)
    meta = load_SMPS_metadata(file, delimiter, encoding)

    # Parse datetime
    df.rename(columns={"Sample #": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Start Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df.drop(columns=["Date", "Start Time"], inplace=True)

    # Bin columns and conversion
    bin_cols = df.columns[7:103]  # columns 9–105 (Python 0-based)
    bin_mids = np.array(bin_cols, dtype=float)

    # Define full SMPS bin midpoints and identify matching edges
    bin_mid_log = np.round(np.array([10 ** ((i - 0.5) / 64.0) for i in range(192)]), 1)
    size_idx = np.nonzero(np.isin(bin_mid_log, bin_mids))[0]
    bin_idx = np.append(size_idx, size_idx[-1] + 1)
    bin_edges = np.round([10 ** ((i - 1) / 64.0) for i in range(192)], 2)[bin_idx]

    # Extract and format main data
    dist_data = df[bin_cols].to_numpy()
    total_conc = np.nansum(dist_data, axis=1)
    df_total = pd.DataFrame(total_conc, columns=["Total_conc"])
    df_dist = pd.DataFrame(dist_data, columns=bin_mids.astype(str))
    final_df = pd.concat([df["Datetime"], df_total, df_dist], axis=1)

    # Unit / dtype interpretation
    density = 1.0
    unit_dict = {"Nu": "cm⁻³", "Su": "nm²/cm³", "Vo": "nm³/cm³", "Ma": "ug/m³"}
    dtype_dict = {"Nu": "dN", "Su": "dS", "Vo": "dV", "Ma": "dM"}

    try:
        weight_prefix = meta["Weight"][:2]
        unit = unit_dict[weight_prefix]
        if "dlogDp" in meta["Units"]:
            dtype = dtype_dict[weight_prefix] + "/dlogDp"
        elif "dDp" in meta["Units"]:
            dtype = dtype_dict[weight_prefix] + "/dDp"
        else:
            dtype = dtype_dict[weight_prefix]
    except Exception:
        raise Exception("Unit and/or data type does not match the expected format.")

    # Construct object
    smps = Aerosol2D(final_df)
    smps._meta = {
        **{k: v for k, v in meta.items() if k not in ("Weight", "Units")},
        "instrument": "SMPS",
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "density": density,
        "serial_number": f"Classifier: {meta['Classifier Model'][2]}, Detector: {meta['Detector Model'][2]}",
        "unit": unit,
        "dtype": dtype,
    }

    smps.convert_to_number_concentration()
    smps.unnormalize_logdp()

    if extra_data:
        extra_df = df.drop(columns=bin_cols)
        extra_df.set_index("Datetime", inplace=True)
        smps._extra_data = extra_df

    return smps
