import datetime

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .Common import _detect_delimiter

###############################################################################


def Load_FMPS_file(file: str) -> Aerosol2D:
    """Description:
        Load an FMPS size-distribution export and return it as an
        :class:`Aerosol2D` number-size distribution with metadata.

    Args:
        file (str):
            Path to the FMPS data file exported from the FMPS software.

    Returns:
        Aerosol2D:
            FMPS size distributions with a datetime index, total concentration,
            size-resolved bins, and associated metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            :func:`_detect_delimiter`.
        Exception:
            If the file appears to be a raw (unconverted) FMPS export, or if
            the detected data type cannot be interpreted as a supported FMPS
            format.

    Notes:
        Detailed description:
            This loader is intended for FMPS files that have already been
            converted using the FMPS software (i.e. not raw current files).
            The file should contain a header with instrument and data settings
            followed by a tabular size-distribution block.

            Internally, the function:

            - Uses :func:`_detect_delimiter` to infer file encoding and field
              delimiter.
            - Reads a header line (after a fixed number of header rows) and
              checks whether it contains the word ``"Raw"``. If so, an
              exception is raised, as raw exports must first be converted
              with the FMPS software.
            - The data is loaded:

              - reads FMPS bin mid diameters and reconstructs bin edges,
              - loads the size-resolved concentration matrix and computes
                ``"Total_conc"`` per time step,
              - parses timestamps using localized Danish or standard English
                date/time formats,
              - extracts metadata from the header (data type marker and serial
                number),
              - infers the base data type (Co, dN, Su, Vo, Ma) and whether the
                values are normalized by ``/dlogDp``,
              - builds an :class:`Aerosol2D` instance and attaches bin edges,
                bin mids, instrument name, serial number, unit, and dtype,
              - calls ``_convert_to_number_concentration()`` followed by
                :meth:`Aerosol2D.unnormalize_logdp` to express the final
                distribution as number concentration per bin (dN, cm⁻³).

            The returned :class:`Aerosol2D` object is therefore ready for
            further analysis or plotting using the standard aerosoltools
            interface.

        Theory:
            FMPS exports can represent different moments or metrics of the
            particle size distribution optionally normalized by ``/dlogDp``.
            The moment depends on the chosen data type:

            - Co / dN: number-based concentration,
            - Su: surface-based moment (dS),
            - Vo: volume-based moment (dV),
            - Ma: mass-based moment (dM),

            The FMPS header encodes this in a type string (e.g. ``"dN/dlogDp"``).
            The loader interprets this string to assign the appropriate unit
            and dtype and then uses internal helpers to convert the distribution
            to number concentration (dN, cm⁻³) and remove any ``/dlogDp``
            normalization so that the output is directly comparable to other
            number-size distributions in aerosoltools.

    Examples:
        Typical usage is to load a converted FMPS file and work directly
        with the resulting number-size distribution:

        .. code-block:: python

            import aerosoltools as at

            # Load FMPS data as a 2D number-size distribution
            fmps = at.Load_FMPS_file("data/FMPS_export.txt")

            # Inspect the data
            print(fmps.data)

            # Inspect bin_edges and metadata
            print(fmps.bin_edges)
            print(fmps.metadata)

            # Plot the time-integrated size distribution
            fig, ax = fmps.plot_psd()
    """
    # Detect encoding and delimiter once for the file
    encoding, delimiter = _detect_delimiter(file)

    # Peek at a header line to decide whether this is a raw or processed export
    header_check = str(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=12,
            max_rows=1,
            dtype=str,
        )
    )

    if "Raw" in header_check:
        raise Exception(
            f"{file} is exported as raw and must first be converted using the FMPS software."
        )

    # Delegate to the software-export parser
    return _load_fmps_software(file, encoding, delimiter)


###############################################################################


def _load_fmps_software(file: str, encoding: str, delimiter: str) -> Aerosol2D:
    """Parse an FMPS software-exported file into an :class:`Aerosol2D`.

    This internal helper assumes the file has already been converted using
    the FMPS software (i.e. not “raw”). It performs the following steps:

    - Reads bin mid diameters and reconstructs bin edges.
    - Loads the size-resolved concentration matrix and computes total
      concentration per time step.
    - Parses timestamps via localized date/time helpers
      (:func:`_parse_danish_datetime` or :func:`_parse_standard_datetime`).
    - Infers the base data type (``Co``, ``dN``, ``Su``, ``Vo``, ``Ma``) and
      whether it is normalized by ``dlogDp``.
    - Constructs an :class:`Aerosol2D` instance, then converts to number
      concentration and removes any ``/dlogDp`` normalization.

    Args:
        file:
            Path to the FMPS file.
        encoding:
            Text encoding used to read the file.
        delimiter:
            Field delimiter used in the file.

    Returns:
        Aerosol2D:
            Particle size distribution with metadata and total number
            concentration.
    """
    # Load bin mids and derive bin edges
    bin_mids = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, skip_header=13, max_rows=1
    )[1:-11]
    bin_edges = np.append(5.6, (bin_mids[1:] + bin_mids[:-1]) / 2)
    bin_edges = np.append(bin_edges, 560)

    # Load main data matrix and compute total concentration
    data_array = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, skip_header=15
    )
    dist_data = data_array[:, 1:-11]
    total_conc = pd.DataFrame(np.nansum(dist_data, axis=1), columns=["Total_conc"])

    # Parse time axis using localized or standard datetime formats
    time_format = str(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=14,
            max_rows=1,
            dtype=str,
        )[0]
    )
    try:
        datetime_df = _parse_danish_datetime(file, delimiter, encoding, time_format)
    except (IndexError, ValueError, KeyError):
        datetime_df = _parse_standard_datetime(file, delimiter, encoding, time_format)

    # Extract metadata from header (data type string and serial number)
    with open(file, "r", encoding=encoding, newline="") as fh:
        for _ in range(12):
            fh.readline()
        line = fh.readline()

    first_field = line.split(delimiter)[0].strip()
    datatype = first_field.split(" ")[0]
    try:
        serial_number = str(
            np.genfromtxt(
                file,
                delimiter=delimiter,
                encoding=encoding,
                skip_header=4,
                max_rows=1,
                dtype=str,
            )[2]
        )[-8:]
    except:
        serial_number = str(
            np.genfromtxt(
                file,
                delimiter=delimiter,
                encoding=encoding,
                skip_header=4,
                max_rows=1,
                dtype=str,
            )[1]
        ).split(",")[1][-8:]

    # Infer dtype and unit from the datatype marker
    dtype_dict = {"Co": "dN", "dN": "dN", "Su": "dS", "Vo": "dV", "Ma": "dM"}
    unit_dict = {"dN": "cm⁻³", "dS": "nm²/cm³", "dV": "nm³/cm³", "dM": "ug/m³"}

    try:
        if "dlogDp" in datatype:
            dtype = dtype_dict[datatype[:2]] + "/dlogDp"
        else:
            dtype = dtype_dict[datatype[:2]]
        unit = unit_dict[dtype[:2]]
    except KeyError as e:
        raise Exception(
            "Unit and/or data type does not match the expected FMPS format. "
            "Check that the file is an FMPS software export."
        ) from e

    # Assemble the distribution DataFrame (Datetime, Total_conc, size bins)
    dist_df = pd.DataFrame(dist_data, columns=bin_mids.astype(str))
    output_df = pd.concat([datetime_df, total_conc, dist_df], axis=1)

    # Create Aerosol2D object and populate metadata
    FMPS = Aerosol2D(output_df)
    FMPS._meta = {
        "instrument": "FMPS",
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "density": 1.0,
        "serial_number": serial_number,
        "unit": unit,
        "dtype": dtype,
    }

    # Convert to number concentration and undo any dlogDp normalization
    FMPS._convert_to_number_concentration()
    FMPS.unnormalize_logdp()

    return FMPS


###############################################################################


def _parse_danish_datetime(
    file: str, delimiter: str, encoding: str, time_format: str
) -> pd.DataFrame:
    """Parse FMPS datetimes written in Danish date format.

    This helper reads the start date/time from the header in Danish prose
    (e.g. ``"29. januar 2025 14:32:10"``), constructs a start timestamp, and
    then converts either:

    - elapsed seconds since start (when ``"Elapsed"`` appears in ``time_format``), or
    - clock times for each row (if not elapsed),

    into an absolute ``Datetime`` column.

    Args:
        file:
            Path to the FMPS file.
        delimiter:
            Field delimiter used in the file.
        encoding:
            Text encoding used to read the file.
        time_format:
            Time format marker read from the header, used to distinguish
            elapsed time vs. wall-clock times.

    Returns:
        pandas.DataFrame:
            Single-column DataFrame with a ``"Datetime"`` column.
    """
    # Read the header line containing the Danish date string
    date_str = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, max_rows=1, dtype=str
    )[1]
    date_str = date_str.split('"')[1]

    # Parse Danish month name and clock time
    day, month_name, year = date_str.split(" ")[:3]
    hour, minute, second = map(int, date_str.split(" ")[-1].split(":"))

    months = {
        "januar": 1,
        "februar": 2,
        "marts": 3,
        "april": 4,
        "maj": 5,
        "juni": 6,
        "juli": 7,
        "august": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "december": 12,
    }
    start_dt = datetime.datetime(
        int(year), months[month_name], int(day.replace(".", "")), hour, minute, second
    )

    # Convert either elapsed seconds or clock times into absolute datetimes
    if "Elapsed" in time_format:
        times = np.genfromtxt(
            file, delimiter=delimiter, encoding=encoding, skip_header=15, dtype=float
        )[:, 0]
        dt_index = [start_dt + datetime.timedelta(seconds=int(t)) for t in times]
    else:
        time_list = np.genfromtxt(
            file, delimiter=delimiter, encoding=encoding, skip_header=15, dtype=str
        )[:, 0]
        step = datetime.datetime.strptime(
            time_list[1], "%H:%M:%S"
        ) - datetime.datetime.strptime(time_list[0], "%H:%M:%S")
        dt_index = [start_dt + i * step for i in range(len(time_list))]

    return pd.DataFrame(dt_index, columns=["Datetime"])


###############################################################################


def _parse_standard_datetime(
    file: str, delimiter: str, encoding: str, time_format: str
) -> pd.DataFrame:
    """Parse FMPS datetimes written in standard English format.

    This helper reads month/day/year and time-of-day information from the
    FMPS header in English (e.g. ``"Jan 02 2025 02:12:34 PM"``),
    constructs a start timestamp, and then converts either:

    - elapsed seconds since start (when ``"Elapsed"`` appears in ``time_format``), or
    - clock times for each row (if not elapsed),

    into an absolute ``Datetime`` column.

    Args:
        file:
            Path to the FMPS file.
        delimiter:
            Field delimiter used in the file.
        encoding:
            Text encoding used to read the file.
        time_format:
            Time format marker read from the header, used to distinguish
            elapsed time vs. wall-clock times.

    Returns:
        pandas.DataFrame:
            Single-column DataFrame with a ``"Datetime"`` column.
    """
    # Read the main date/time tokens from the header
    fmps_date = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, max_rows=1, dtype=str
    )[1].split(",")[1:]#[2:]
    month_map = {
        k: v
        for v, k in enumerate(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            1,
        )
    }

    month_str = fmps_date[0].split(" ")[1][:3]
    day = int(fmps_date[0].split(" ")[2])
    year = int(fmps_date[1].split(" ")[1])
    time = fmps_date[1].split(" ")[3] + " " + fmps_date[1].split(" ")[4][:2]
    base_time = datetime.datetime.strptime(time, "%I:%M:%S %p")

    start_dt = datetime.datetime(
        year,
        month_map[month_str],
        day,
        base_time.hour,
        base_time.minute,
        base_time.second,
    )

    # Load the time strings for each row and build a regular time grid
    if "Elapsed" in time_format:
        times = np.genfromtxt(
            file, delimiter=delimiter, encoding=encoding, skip_header=15, dtype=float
        )[:, 0]
        return pd.DataFrame(
            [start_dt + datetime.timedelta(seconds=int(t)) for t in times],
            columns=["Datetime"],
        )
    else:
        time_strs = np.genfromtxt(
            file, delimiter=delimiter, encoding=encoding, skip_header=15, dtype=str
        )[:, 0]
        times = [datetime.datetime.strptime(t, "%I:%M:%S %p") for t in time_strs]
        step = times[1] - times[0]

        return pd.DataFrame(
            [start_dt + i * step for i in range(len(times))], columns=["Datetime"]
        )


