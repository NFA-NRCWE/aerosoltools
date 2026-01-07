# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd

from ..aerosolalt import AerosolAlt
from .Common import detect_delimiter

###############################################################################


def Load_Fourtec_file(file: str) -> AerosolAlt:
    """
    Load data from a Fourtec Bluefish CSV or Excel export file.

    This function extracts datetime, temperature, and humidity from logged
    sensor data and stores it in an AerosolAlt object with metadata.

    Parameters
    ----------
    file : str
        Path to the Fourtec file (.csv or .xlsx format).

    Returns
    -------
    Fourtec : AerosolAlt
        Object containing:
        - Datetime, Temperature (°C), and RH (%)
        - Instrument metadata including serial number
    """
    if file.lower().endswith(".csv"):
        encoding, delimiter = detect_delimiter(file)
        df = pd.read_csv(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skiprows=8,
            usecols=[0, 1, 2, 4],
        )
        df.rename(
            columns={
                "Date": "Date",
                "Time": "Time",
                "Internal Digital Temperature": "Temperature",
                "Internal RH": "RH",
            },
            inplace=True,
        )

        # Combine date and time into a single datetime column
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], format="%d-%m-%Y %H:%M:%S"
        )
        df.drop(columns=["Date", "Time"], inplace=True)

        # Extract serial number
        SN = str(
            np.genfromtxt(
                file,
                delimiter=delimiter,
                encoding=encoding,
                skip_header=1,
                max_rows=1,
                dtype=str,
            )[2]
        )

    else:
        df = pd.read_excel(file, skiprows=8, usecols=[0, 1, 2, 4])
        df.rename(
            columns={
                "Date": "Date",
                "Time": "Time",
                "Internal Digital Temperature": "Temperature",
                "Internal RH": "RH",
            },
            inplace=True,
        )

        # Combine date and time into a single datetime column
        df["Datetime"] = [
            datetime.datetime.combine(date.date(), time)
            for date, time in zip(df["Date"], df["Time"])
        ]
        df.drop(columns=["Date", "Time"], inplace=True)

        # Extract serial number from second row, column index 2
        SN = str(pd.read_excel(file, skiprows=1, nrows=1, usecols=[2]).iloc[0, 0])

    # Package into AerosolAlt object
    fourtec = AerosolAlt(df[["Datetime", "Temperature", "RH"]])
    fourtec._meta["instrument"] = "Fourtec"
    fourtec._meta["serial_number"] = SN
    fourtec._meta["unit"] = {"Temperature": "°C", "RH": "%"}

    return fourtec
