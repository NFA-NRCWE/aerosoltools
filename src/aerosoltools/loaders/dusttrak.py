import datetime

import pandas as pd

from ..dusttrak import DustTrak
from .support.construct import build_1d
from .support.exceptions import TimestampError
from .support.reading import read_cells, read_table, sniff
from .support.units import resolve_extra_columns

###############################################################################


def load_dusttrak_file(file: str, extra_data: bool = True) -> DustTrak:
    """Load a DustTrak DRX export file and return PM mass concentrations and
    total mass as a :class:`DustTrak` time series.

    Args:
        file (str):
            Path to the DustTrak DRX data file (typically a ``.csv`` export).
        extra_data (bool, optional):
            If ``True``, non-core columns (e.g. alarms, errors, diagnostics)
            are stored in ``extra_data``. Defaults to ``True``, matching every
            other loader.

    Returns:
        DustTrak:
            DustTrak measurements with a datetime index and PM channels
            (PM1, PM2.5, PM4, PM10, Total) in µg/m³.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded with the inferred or fallback
            encoding.
        ValueError:
            If the start datetime cannot be parsed from the header, or if
            elapsed time cannot be converted to timestamps.
        KeyError:
            If expected columns (elapsed time or PM channels) are missing or
            renamed unexpectedly.

    Notes:
        Detailed description:
            This loader assumes a standard DustTrak DRX ASCII/CSV export with
            a metadata header followed by a tabular data block. Internally it:

            - Tries to infer file encoding and delimiter using
              ``_detect_delimiter``; if this fails, it falls back to
              ``encoding="latin-1"`` and ``delimiter=","``.
            - Reads the data block starting at row 35 and renames key columns:

              - ``"Elapsed Time [s]"`` → ``"Datetime"`` (elapsed seconds),
              - ``"PM1 [mg/m3]"`` → ``"PM1"``,
              - ``"PM2.5 [mg/m3]"`` → ``"PM2.5"``,
              - ``"PM4 [mg/m3]"`` → ``"PM4"``,
              - ``"PM10 [mg/m3]"`` → ``"PM10"``,
              - ``"TOTAL [mg/m3]"`` → ``"Total"``.

            - Reads the first 8 header lines to extract instrument name,
              model number, serial number, and the start date/time. The
              start datetime is parsed using ``"%d/%m/%Y %H:%M:%S"``.
            - Converts the elapsed seconds in ``"Datetime"`` to absolute
              timestamps by adding a time delta to the parsed start datetime.
            - Converts all PM channels from mg/m³ to µg/m³.
            - Creates a :class:`DustTrak` object using ``Datetime`` and the
              PM channels as the main data frame, and attaches metadata:

              - ``instrument``, ``model_number``, ``serial_number``,
              - per-channel units (µg/m³),
              - per-channel dtype (``"dM"``).

            - If ``extra_data=True``, any remaining columns are moved to
              ``extra_data`` (and ``_raw_extra_data``) indexed by ``Datetime``,
              so alarms or diagnostic signals are preserved.

    Examples:
        Typical usage is to load one or more DustTrak DRX files for QA/QC
        or exposure assessment:

        .. code-block:: python

            import aerosoltools as at

            # Load core PM fractions
            dust = at.load_dusttrak_file("data/DustTrak_DRX_2023-10-01.csv")

            # Inspect PM2.5 and PM10 time series
            print(dust.data[["PM2.5", "PM10"]].head())

            # Plot total mass concentration
            fig, ax = dust.plot_timeseries()
    """
    # Try to detect encoding and delimiter from the file
    try:
        encoding, delimiter = sniff(file, sample_lines=30)
    except Exception:
        # Fallback: assume a reasonable default encoding and comma delimiter
        encoding = "latin-1"
        delimiter = ","

    # Read main data block
    df = read_table(file, encoding=encoding, delimiter=delimiter, header=35)
    df.rename(
        columns={
            "Elapsed Time [s]": "Datetime",
            "PM1 [mg/m3]": "PM1",
            "PM2.5 [mg/m3]": "PM2.5",
            "PM4 [mg/m3]": "PM4",
            "PM10 [mg/m3]": "PM10",
            "TOTAL [mg/m3]": "Total",
        },
        inplace=True,
    )

    # Read header metadata (instrument info and start date/time)
    meta_lines = read_cells(
        file, skip_header=0, max_rows=8, encoding=encoding, delimiter=delimiter
    )
    try:
        # Typical pattern: line 7 = date, line 6 = time
        start_str = f"{meta_lines[7, 1]} {meta_lines[6, 1]}"
        start_time = datetime.datetime.strptime(start_str, "%d/%m/%Y %H:%M:%S")
    except Exception as e:
        raise TimestampError(
            f"Unable to parse start datetime from metadata: {e}"
        ) from e

    # Convert elapsed seconds to absolute datetime
    df["Datetime"] = pd.to_timedelta(df["Datetime"], unit="s") + start_time

    # Convert mg/m³ -> µg/m³ for PM channels
    data_columns = ["PM1", "PM2.5", "PM4", "PM10", "Total"]
    df[data_columns] = df[data_columns] * 1000.0

    # Build DustTrak object on the core columns (Datetime + 5 PM channels)
    Dust = build_1d(
        df[["Datetime", *data_columns]],
        cls=DustTrak,
        instrument=str(meta_lines[0, 1]),
        serial_number=str(meta_lines[2, 1]),
        unit={col: "µg/m³" for col in data_columns},
        dtype={col: "dM" for col in data_columns},
        extra_meta={"model_number": str(meta_lines[1, 1])},
    )

    # Optional extra data (e.g. alarms / errors) indexed by time. The DRX writes
    # these headers without units, so nothing is stripped -- resolve_extra_columns
    # simply leaves unrecognised headers alone and records no unit for them.
    if extra_data:
        extra_cols = [c for c in df.columns if c not in ["Datetime", *data_columns]]
        if extra_cols:
            extra_df = df[["Datetime", *extra_cols]].set_index("Datetime")
            extra_df, column_units = resolve_extra_columns(extra_df)
            Dust._extra_data = extra_df
            Dust._raw_extra_data = extra_df.copy()
            if column_units:
                Dust._meta["column_units"] = column_units

    return Dust
