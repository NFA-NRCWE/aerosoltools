import datetime

import numpy as np
import pandas as pd

from ..environmental import Environmental1D
from .support.parsing import _detect_delimiter

###############################################################################


def load_fourtec_file(file: str) -> Environmental1D:
    """Description:
        Load a Fourtec Bluefish CSV or Excel export and return temperature and
        relative humidity as an :class:`Environmental1D` time series.

    Args:
        file (str):
            Path to the Fourtec export file (``.csv`` or ``.xlsx``).

    Returns:
        Environmental1D:
            Fourtec measurements with a datetime index and columns for
            temperature (°C) and relative humidity (%), plus basic metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If a ``.csv`` file cannot be decoded using the encodings tried by
            :func:`_detect_delimiter`.
        ValueError:
            If the date/time fields in the export cannot be parsed into valid
            timestamps. Check that the file uses the expected Fourtec formats.

    Notes:
        Detailed description:
            This loader is tailored to Fourtec Bluefish logger exports produced
            by the vendor software in either text (CSV) or Excel (XLSX) format.
            The files typically contain a short header block followed by a
            tabular data section with separate ``Date`` and ``Time`` columns
            and channels for internal temperature and relative humidity.

            Internally, the function:

            - Branches on the file extension:

              - For ``.csv`` files:

                - Uses :func:`_detect_delimiter` to infer text encoding and
                  field delimiter.
                - Reads the main data block with :func:`pandas.read_csv`,
                  skipping the header rows and selecting the expected columns
                  (date, time, temperature, RH).
                - Renames ``"Internal Digital Temperature"`` to
                  ``"Temperature"`` and ``"Internal RH"`` to ``"RH"``.
                - Combines the ``"Date"`` and ``"Time"`` columns into a single
                  ``"Datetime"`` column using the format
                  ``"%d-%m-%Y %H:%M:%S"`` and drops the original date/time
                  columns.
                - Extracts the instrument serial number from the header using
                  :func:`numpy.genfromtxt`.

              - For ``.xlsx`` files:

                - Reads the main data block with :func:`pandas.read_excel`,
                  again skipping header rows and selecting the same logical
                  columns.
                - Renames temperature and RH columns in the same way as for
                  CSV.
                - Builds ``"Datetime"`` by combining the separate date and time
                  columns via :func:`datetime.datetime.combine`, then drops
                  the original date/time columns.
                - Extracts the serial number from the second row in the header
                  area with a small :func:`pandas.read_excel` call.

            - Constructs an :class:`Environmental1D` object using the core columns
              ``"Datetime"``, ``"Temperature"``, and ``"RH"``.
            - Populates metadata with:

              - ``instrument`` set to ``"Fourtec"``,
              - ``serial_number`` taken from the header,
              - ``unit`` mapping: ``{"Temperature": "°C", "RH": "%"}``.

            No additional channels are currently stored in ``extra_data``; only
            the core environmental measurements are kept.

      Examples:
          Typical usage is to load a Fourtec logger file and use the
          resulting temperature and RH time series alongside aerosol
          measurements:

          .. code-block:: python

              import aerosoltools as at

              # Load Fourtec data from CSV or Excel
              fourtec = at.load_fourtec_file("data/Fourtec_export.csv")

              # Inspect the data
              print(fourtec.data.head())

              # Check metadata (instrument, serial number, units)
              print(fourtec.metadata)

              # Plot temperature and RH over time
              fig, ax = fourtec.plot_timeseries()
    """
    # Branch on CSV vs Excel format
    if file.lower().endswith(".csv"):
        # Detect encoding and delimiter, then read main data block
        encoding, delimiter = _detect_delimiter(file)
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
            df["Date"] + " " + df["Time"], format="%d-%m-%Y %H:%M:%S", errors="raise"  # type: ignore
        )
        df.drop(columns=["Date", "Time"], inplace=True)

        # Extract serial number from header
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
        # Read main data block from Excel
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

        # Extract serial number from the second row (header area)
        SN = str(pd.read_excel(file, skiprows=0, nrows=1, usecols=[2]).iloc[0, 0])

    # Build Environmental1D with core variables only
    fourtec = Environmental1D(df[["Datetime", "Temperature", "RH"]])
    fourtec._meta["instrument"] = "Fourtec"
    fourtec._meta["serial_number"] = SN
    fourtec._meta["unit"] = {"Temperature": "°C", "RH": "%"}

    return fourtec
