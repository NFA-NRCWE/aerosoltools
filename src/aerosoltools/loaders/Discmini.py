# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from .Common import detect_delimiter
from .aerosolalt import AerosolAlt

###############################################################################

def Load_DiSCmini_file(file: str, extra_data: bool = False):
    """
    Load and parse data from a DiSCmini .txt file (after conversion), returning an AerosolAlt object.

    This function extracts the datetime, total particle number concentration, average size,
    and LDSA from the DiSCmini export file. It also stores serial number and units
    as metadata, and optionally attaches extra columns as `.extra_data`.

    Parameters
    ----------
    file : str
        Path to the .txt file exported from DiSCmini software (after conversion).
    extra_data : bool, optional
        If True, attaches unused data columns as `._extra_data` in the AerosolAlt class.

    Returns
    -------
    DM : AerosolAlt
        Object containing parsed time series data and instrument metadata.

    Raises
    ------
    Exception
        If the file has not been converted correctly, or the datetime format is unrecognized.

    Notes
    -----
    - The returned data contains: 'Datetime', 'Total_conc' (cm⁻³), 'Size' (nm), and 'LDSA' (nm²/cm³).
    - Automatically detects encoding and delimiter using `Com.detect_delimiter()`.
    - Two known datetime formats are supported: `%d-%b-%Y %H:%M:%S` and `%d-%m-%Y %H:%M:%S`.
    """
    try:
        encoding, delimiter = Com.detect_delimiter(file, sample_lines=12)
    except Exception:
        raise Exception("DiSCmini data has not been converted or delimiter could not be detected.")

    # Load selected columns: DateTime, Number, Size, LDSA, etc.
    df = pd.read_csv(file, header=4, encoding=encoding, delimiter="\t", usecols=range(0, 7))
    df.drop(columns=['Time'], inplace=True)
    df.rename(columns={'TimeStamp': 'Datetime', 'Number': 'Total_conc'}, inplace=True)

    # Attempt datetime parsing using known formats
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d-%b-%Y %H:%M:%S")
    except ValueError:
        try:
            df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d-%m-%Y %H:%M:%S")
        except Exception:
            raise Exception("Datetime does not match expected format. Ensure file is converted correctly.")

    # Extract serial number from metadata (row 2, position 6)
    meta_line = np.genfromtxt(file, delimiter=delimiter, encoding=encoding, skip_header=1, max_rows=1, dtype=str)
    serial_number = str(meta_line).split(" ")[5]

    # Create AerosolAlt instance
    DM = AerosolAlt(df.iloc[:, 0:4])  # Datetime, Total_conc, Size, LDSA

    # Set metadata
    DM._meta['instrument'] = "DiSCmini"
    DM._meta['serial_number'] = serial_number
    DM._meta['unit'] = {
        'Total_conc': 'cm$^{-3}$',
        'Size': 'nm',
        'LDSA': 'nm$^{2}$/cm$^{3}$'
    }
    DM._meta['dtype'] = {
        'Total_conc': 'dN',
        'Size': 'l',
        'LDSA': 'dS'
    }

    # Attach extra data if requested
    if extra_data:
        extra_df = df.drop(columns=list(df)[1:4]).set_index('Datetime')
        DM._extra_data = extra_df

    return DM
