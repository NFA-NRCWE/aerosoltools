
import numpy as np
import pandas as pd

from ..environmental import Environmental1D
from .Common import Load_data_from_folder, _detect_delimiter

###############################################################################


def Load_Devlabs_file(folder_path: str, freq='1min', start=None, end=None) -> Environmental1D:
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
              fourtec = at.Load_Fourtec_file("data/Fourtec_export.csv")

              # Inspect the data
              print(fourtec.data.head())

              # Check metadata (instrument, serial number, units)
              print(fourtec.metadata)

              # Plot temperature and RH over time
              fig, ax = fourtec.plot_timeseries()
    """
    Parameters=["Temp","RH","W_speed","W_direction","PM2.5","CO2","CO","NO2"]

    Search_words={"Temp"    : "temperature",
                  "RH"      : "relative_humidity",
                  "W_speed" : "wind_speed",
                  "W_direction" : "wind_direction",
                  "PM2.5"   : "pm2.5",
                  "CO2"     : "carbon_dioxide",
                  "CO"      : "carbon_monoxide",
                  "NO2"     : "nitrogen_dioxide"}
    Data={}
    for param in Parameters:
        Data[param]=Load_data_from_folder(folder_path, load_parameter, search_word = Search_words[param], parameter=param)
        Data[param]._data[param]=pd.to_numeric(Data[param].data[param],errors='coerce')

    # Extract serial number from header
    SN = Data["Temp"]._meta["serial_number"]
    for param in Parameters:
        if Data[param]._meta["serial_number"]!= SN:
            raise ValueError("Trying to combine data from multiple sensors")

    if start is None :
        start   = min([Data[i ].time[0] for i in Parameters])
    if end is None :
        end     = max([Data[i ].time[-1] for i in Parameters])

    df={}
    data=Data.copy()
    for param in Parameters:
        data[param]._data[param]=pd.to_numeric(data[param].data[param],errors='coerce')

    for param in Parameters:
        df[param]=data[param].timerebin(freq,start,end).data[param]

    Combined_df=pd.concat(df.values(),axis=1)

    # Build Environmental1D with core variables only
    devlabs = Environmental1D(Combined_df)
    devlabs._meta["instrument"] = "DevLabs Weatherstation"
    devlabs._meta["serial_number"] = SN
    devlabs._meta["unit"] = {
        "Temperature"  : "°C",
        "RH"           : "%",
        "W_speed"      : "m/s",
        "W_direc"      : "°",
        "PM2.5"        : "µg/m3",
        "CO2"          : "ppm",
        "CO"           : "ppm",
        "NO2"          : "ppb"}
    devlabs._meta["dtype"] = {
        "Temperature"  : "Temp",
        "RH"           : "RH",
        "W_speed"      : "W-speed",
        "W_direc"      : "W-direc",
        "PM2.5"        : "PM2.5",
        "CO2"          : "CO$_2$",
        "CO"           : "CO",
        "NO2"          : "NO$_2$"}

    return devlabs

###############################################################################

def load_parameter (file: str,parameter) -> Environmental1D:
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
              fourtec = at.Load_Fourtec_file("data/Fourtec_export.csv")

              # Inspect the data
              print(fourtec.data.head())

              # Check metadata (instrument, serial number, units)
              print(fourtec.metadata)

              # Plot temperature and RH over time
              fig, ax = fourtec.plot_timeseries()
    """
    # Branch on CSV vs Excel format

    # Detect encoding and delimiter, then read main data block
    encoding, delimiter = _detect_delimiter(file)
    df = pd.read_csv(
        file,
        delimiter=delimiter,
        encoding=encoding,
        usecols=[0, 2],
    )
    df.rename(
        columns={
            "Timestamp": "Datetime",
            df.columns[1]: parameter,
        },
        inplace=True,
    )

    # Combine date and time into a single datetime column
    df["Datetime"] = pd.to_datetime(
        df["Datetime"], format="%Y-%m-%d %H:%M:%S", errors="raise"  # type: ignore
    )

    # Extract serial number from header
    SN = str(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=1,
            max_rows=1,
            dtype=str,
        )[1]
    )

    # Build Environmental1D with core variables only
    devlabs = Environmental1D(df)
    devlabs._meta["serial_number"] = SN

    return devlabs
