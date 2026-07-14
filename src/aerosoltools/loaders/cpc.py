import datetime

import numpy as np
import pandas as pd

from ..aerosol1d import Aerosol1D
from .support.exceptions import FileFormatError
from .support.parsing import _detect_delimiter

###############################################################################


def load_cpc_file(file: str, extra_data: bool = False) -> Aerosol1D:
    """Load a TSI CPC (Condensation Particle Counter) export file, automatically
    detect its format, and return a time series of total particle number
    concentration as an :class:`Aerosol1D` object.

    Args:
        file (str):
            Path to the CPC data file (typically ``.txt`` or ``.csv``) exported
            from the instrument software.
        extra_data (bool, optional):
            If ``True`` and the file is in *full* format, non-core diagnostic
            and operational parameters are stored in the returned object's
            ``.extra_data`` attribute. For *focused* format files, this flag
            has no effect because only minimal channels are available.
            Defaults to ``False``.

    Returns:
        Aerosol1D:
            An :class:`~aerosoltools.aerosol1d.Aerosol1D` instance containing:

            - ``.data`` with a datetime index and a ``"Total_conc"`` column
              holding particle number concentration in ``cm⁻³``.
            - ``.meta`` (metadata) populated with, e.g.:

              - ``"instrument"`` — set to ``"CPC"``.
              - ``"serial_number"`` — instrument serial number parsed from
                the header.
              - ``"unit"`` — set to ``"cm⁻³"``.
              - ``"dtype"`` — set to ``"dN"`` (number concentration).

            If ``extra_data=True`` and the file is a *full* CPC export,
            additional diagnostic channels are provided in ``.extra_data``.

    Raises:
        FileNotFoundError:
            If the specified ``file`` path does not exist or cannot be opened.
            Verify the path and that you have read permissions.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            ``_detect_delimiter``. Check that the file is a valid CPC
            export and not a binary or corrupted file.
        ValueError:
            If ``_detect_delimiter`` cannot reliably determine a delimiter
            from the sampled lines. This can happen for very short or malformed
            files.
        FileFormatError:
            If the detected column structure does not match any supported CPC
            format (currently focused with 4 columns or full with 14 columns).
            In this case, check that the file is an unmodified CPC export and
            that the header layout has not changed.

    Notes:
        Detailed description:
            ``load_cpc_file`` is the main entry point for reading CPC data.
            It is designed to handle at least two common export layouts:

            - Focused format directly from the instrument (compact output):

              - Contains only a time column and total number concentration.
              - Identified by having 4 columns in the numeric data section.
              - Internally parsed by a specialized focused-format loader.

            - Full format via TSI AIM software (diagnostic-rich output):

              - Contains additional operational and diagnostic parameters
                (e.g. flow, temperature, status flags) in separate columns.
              - Identified by having 14 columns in the numeric data section.
              - Internally parsed by a specialized full-format loader, which
                also exposes optional diagnostics via ``.extra_data`` when
                requested.

    Examples:
        Typical usage is to load a single CPC file (regardless of its
        export format) and immediately access a clean time series:

        .. code-block:: python

            import aerosoltools as at

            # Load a CPC export (focused or full format)
            cpc = at.load_cpc_file("data/CPC_2023-10-01.txt", extra_data=True)

            # Inspect the main total concentration time series
            print(cpc.data)

            # Plot the time series of total concentration
            fig, ax = cpc.plot_timeseries()
    """
    encoding, delimiter = _detect_delimiter(file)
    col_count = len(
        np.genfromtxt(
            file,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=4,
            max_rows=1,
            dtype=str,
        )
    )

    if col_count == 4:
        return _load_cpc_focused(file, encoding, delimiter)
    elif col_count == 14:
        return _load_cpc_full(file, extra_data, encoding, delimiter)
    else:
        raise FileFormatError("Error in determining CPC data structure")


###############################################################################


def _load_cpc_focused(file: str, encoding: str, delimiter: str) -> Aerosol1D:
    """Load and parse CPC data exported in *focused* format.

    Focused CPC exports contain a compact layout with only time and total
    concentration columns. This function reconstructs a continuous time
    index based on the start date and time stored in the file header and
    returns the result as an :class:`Aerosol1D` object.

    Args:
        file: Path to the focused CPC data file.
        encoding: Text encoding to use when reading the file (e.g. ``"utf-8"``).
        delimiter: Field delimiter used in the file (e.g. ``","``, ``";"``).

    Returns:
        Aerosol1D: An :class:`Aerosol1D` instance with:

        - a datetime index reconstructed from the start date/time and sampling
          interval,
        - a single ``"Total_conc"`` column containing particle number
          concentration (cm⁻³),
        - instrument metadata including ``instrument``, ``serial_number``,
          and ``unit``.
    """
    df = pd.read_csv(
        file,
        header=14,
        skipfooter=3,
        usecols=[0, 1],
        encoding=encoding,
        delimiter=delimiter,
        engine="python",
    )
    df.columns = ["Time", "Total_conc"]

    meta = np.genfromtxt(
        file,
        delimiter=delimiter,
        encoding=encoding,
        skip_header=4,
        max_rows=6,
        dtype="str",
    )

    start_datetime = datetime.datetime.strptime(
        f"{meta[0,1]} {meta[1,1]}", "%m/%d/%y %H:%M:%S"
    )
    df["Datetime"] = [
        start_datetime + datetime.timedelta(seconds=i + 1) for i in range(len(df))
    ]

    df = pd.concat([df["Datetime"], df["Total_conc"]], axis=1)
    df["Total_conc"] = pd.to_numeric(df["Total_conc"], errors="coerce")
    df = df.dropna()

    CPC = Aerosol1D(df)
    CPC._meta["instrument"] = "CPC"
    CPC._meta["serial_number"] = meta[5, 1][5:-3]
    CPC._meta["unit"] = "cm⁻³"
    CPC._meta["dtype"] = "dN"
    return CPC


###############################################################################


def _load_cpc_full(
    file: str, extra_data: bool, encoding: str, delimiter: str
) -> Aerosol1D:
    """Load and parse CPC data exported in *full* format.

    Full-format CPC files include a richer set of operational parameters and
    diagnostics in addition to the main concentration time series. This
    function extracts the datetime and total concentration, attaches
    instrument metadata, and optionally stores all remaining columns in
    ``.extra_data``.

    Args:
        file: Path to the full CPC data file.
        extra_data: If ``True``, all non-core columns (beyond datetime and
            ``"Total_conc"``) are stored in the returned object's
            ``.extra_data`` attribute.
        encoding: Text encoding to use when reading the file.
        delimiter: Field delimiter used in the file.

    Returns:
        Aerosol1D: An :class:`Aerosol1D` instance with:

        - a datetime index from the CPC ``"Start Date"`` and ``"Start Time"``,
        - a ``"Total_conc"`` column in cm⁻³,
        - instrument metadata (``instrument``, ``serial_number``, ``unit``),
        - optionally, extra diagnostic columns in ``.extra_data`` when
          ``extra_data=True``.
    """
    df = pd.read_csv(
        file, header=2, encoding=encoding, delimiter=delimiter, engine="python"
    )

    df = df.rename(columns={"Sample #": "Datetime", "[1] Conc": "Total_conc"})
    df["Datetime"] = pd.to_datetime(
        df["Start Date"] + df["Start Time"], format="%m/%d/%y%H:%M:%S"
    )

    data_df = pd.concat([df["Datetime"], df["Total_conc"]], axis=1)

    CPC = Aerosol1D(data_df.copy())
    CPC._meta["instrument"] = "CPC"
    CPC._meta["serial_number"] = df["Instrument ID"].iloc[0][5:-3]
    CPC._meta["unit"] = "cm⁻³"
    CPC._meta["dtype"] = "dN"

    if extra_data:
        extra_df = df.drop(columns=["Start Date", "Start Time", "Total_conc"])
        extra_df.set_index("Datetime", inplace=True)
        CPC._extra_data = extra_df

    return CPC
