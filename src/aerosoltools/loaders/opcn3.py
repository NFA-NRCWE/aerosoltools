from typing import Optional

import numpy as np
import pandas as pd

from ..aerosol1d import Aerosol1D
from ..aerosol2d import Aerosol2D
from .support.construct import build_1d, build_2d
from .support.exceptions import FileFormatError
from .support.reading import read_cells, read_table, sniff
from .support.units import resolve_extra_columns

###############################################################################


def load_opcn3_file(file: str, extra_data: bool = True, PM_focus: bool = False):
    """Load an OPC-N3 export and return either a size-resolved distribution
    (:class:`Aerosol2D`) or PM-focused time series (:class:`Aerosol1D`)
    with metadata.

    Args:
        file (str):
            Path to the OPC-N3 data file.
        extra_data (bool, optional):
            If ``True``, auxiliary channels (e.g. temperature, RH, flow, PM
            channels) are stored in ``extra_data`` when supported by the
            underlying loader. Defaults to ``True`` — everything in the file is
            kept.
        PM_focus (bool, optional):
            If ``True``, return a compact PM-focused :class:`Aerosol1D` with
            total number concentration and PM1/PM2.5/PM10 mass as reported
            by the instrument. If ``False`` (default), return a fully
            size-resolved :class:`Aerosol2D` with binned number concentrations.

    Returns:
        Aerosol2D | Aerosol1D:
            Parsed OPC-N3 dataset as either a size-resolved number-size
            distribution or a PM-only time series, depending on ``PM_focus``
            and the detected file format.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            ``sniff``.
        ValueError:
            If the delimiter detection or CSV parsing fails in a way that
            prevents reading the header or main data table.
        FileFormatError:
            If the file format cannot be recognised from the first line and
            thus cannot be dispatched to a supported OPC-N3 loader.

    Notes:
        Detailed description:
            This loader is designed for loading data from an alphasense OPC-N3
            low-cost particle counter. It reads CSV formats produced by
            different firmware/software revisions. The formats differ mainly in
            how timestamps, bin labels, and PM channels are laid out.

            Internally, the function:

            - Uses ``sniff`` to infer file encoding and field
              delimiter.
            - Reads the first line to classify the file:

              - Legacy/old format (few columns, typically no trailing
                ``"OPC"`` marker):

                - Detected when ``len(first_line) <= 34``.
                - If ``PM_focus=False`` (default), the file is passed to
                  :func:`_load_opcn3_file_old`, which:

                  - parses ISO datetime strings into a ``Datetime`` column,
                  - interprets legacy bin labels (``Bin1``, ``Bin2``, …),
                  - converts raw counts to number concentration in ``cm⁻³``
                    using the reported sampling period and flow,
                  - infers bin edges/mids and returns an :class:`Aerosol2D`
                    with ``Datetime``, ``Total_conc``, and one column per size
                    bin.

                - If ``PM_focus=True``, the file is passed to
                  :func:`_load_opcn3_file_pm_old`, which:

                  - computes ``Total_conc`` from the size bins,
                  - extracts PM1, PM2.5 and PM10 mass concentrations,
                  - returns an :class:`Aerosol1D` with ``Datetime``,
                    ``Total_conc``, ``PM1``, ``PM2.5`` and ``PM10``.

              - Newer format (AIM-like structure, trailing ``"OPC"`` in
                the first line):

                - Detected when ``first_line[-1] == "OPC"``.
                - If ``PM_focus=False``, the file is passed to
                  :func:`_load_opcn3_file_new`, which:

                  - parses ISO datetime strings from a ``Date`` column,
                  - uses a fixed set of bin edges from the OPC-N3 specification,
                  - converts raw counts to number concentration in ``cm⁻³``
                    using sampling period and flow,
                  - returns an :class:`Aerosol2D` with ``Datetime``,
                    ``Total_conc`` and size-bin columns.

                - If ``PM_focus=True``, the file is passed to
                  :func:`_load_opcn3_file_pm_new`, which:

                  - computes ``Total_conc`` from the size bins,
                  - returns an :class:`Aerosol1D` with ``Datetime``,
                    ``Total_conc``, ``PM1``, ``PM2.5`` and ``PM10``.

            - If neither legacy nor newer patterns match, an ``Exception`` is
              raised indicating that the OPC-N3 format is not supported.

            Across all loaders:

            - ``extra_data=True`` causes selected auxiliary variables
              (temperature, RH, period, flow, PM channels) to be stored in
              ``extra_data`` and ``_raw_extra_data`` indexed by ``Datetime``.
            - Size-resolved loaders return number concentration per bin in
              ``cm⁻³`` with metadata for bin edges, bin mids, density,
              instrument name, and dtype.
            - PM-focused loaders return compact time series of total number
              concentration and PM metrics with corresponding units and dtypes.

    Examples:
        Typical usage is to load OPC-N3 data either as a full size
        distribution or as a compact PM time series:

        .. code-block:: python

            import aerosoltools as at

            # 1) Load full size-resolved distribution (Aerosol2D)
            opcn_dist = at.load_opcn3_file("data/OPCN3_legacy.csv")

            print(opcn_dist.data.head())
            print(opcn_dist.bin_edges)
            print(opcn_dist.metadata)

            # 2) Load only total and PM metrics (multi-channel Aerosol1D)
            opcn_pm = at.load_opcn3_file(
                "data/OPCN3_legacy.csv",
                extra_data=True,
                PM_focus=True,
            )

            print(opcn_pm.data.head())
            print(opcn_pm.metadata)

            # 3) Plot the time-integrated size distribution
            fig, ax = opcn_dist.plot_psd()
    """
    encoding, delimiter = sniff(file)

    # Peek at the first line to determine file type
    first_line = read_cells(file, skip_header=0, encoding=encoding, delimiter=delimiter)

    if len(first_line) <= 34:
        if PM_focus:
            return _load_opcn3_file_pm_old(
                file, extra_data=extra_data, encoding=encoding, delimiter=delimiter
            )
        else:
            return _load_opcn3_file_old(
                file, extra_data=extra_data, encoding=encoding, delimiter=delimiter
            )
    elif first_line[-1] == "OPC":
        if PM_focus:
            return _load_opcn3_file_pm_new(
                file, extra_data=extra_data, encoding=encoding, delimiter=delimiter
            )
        else:
            return _load_opcn3_file_new(
                file, extra_data=extra_data, encoding=encoding, delimiter=delimiter
            )
    else:
        raise FileFormatError("Unrecognized OPS file format. Unable to parse.")


###############################################################################


def _load_opcn3_file_old(
    file: str,
    extra_data: bool = False,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> Aerosol2D:
    """Load legacy OPC-N3 CSV exports as size-resolved number concentrations.

    This loader handles older OPC-N3 CSV formats where bin labels are
    encoded as ``Bin1``, ``Bin2``, … in the header. It converts raw bin
    counts to number concentrations in cm⁻³ using the reported sampling
    period and flow, infers bin edges/midpoints, and returns an
    :class:`Aerosol2D` object.

    Args:
        file: Path to the OPC-N3 CSV export file.
        extra_data: If ``True``, non-bin columns (e.g. period, flow,
            temperature, RH) are stored in ``.extra_data`` indexed by
            ``Datetime``. If ``False`` (default), only the time series of
            binned concentrations and ``Total_conc`` are kept.
        encoding: Optional text encoding to use. If ``None`` (default),
            both encoding and delimiter are auto-detected together.
        delimiter: Optional CSV delimiter. If ``None`` (default), both
            encoding and delimiter are auto-detected together.

    Returns:
        Aerosol2D: Object with columns:

        * ``Datetime``
        * ``Total_conc`` (cm⁻³)
        * one column per size bin (cm⁻³), named by bin mid diameter in nm,

        plus metadata such as ``"bin_edges"``, ``"bin_mids"``, ``"density"``,
        ``"instrument"`` and ``"dtype"``.

    Raises:
        ValueError: If exactly one of ``encoding`` or ``delimiter`` is
            provided (both must be given or neither).
    """
    enc, delim = sniff(file, encoding, delimiter)

    df = read_table(file, delimiter=delim, encoding=enc)
    df.rename(columns={"date": "Datetime"}, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Parse ISO-formatted timestamp strings (e.g., '2024-01-21T15:30:01.000Z')
    df["Datetime"] = pd.to_datetime(
        df["Datetime"].str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S"
    )

    # Determine bin edges/mids
    bin_cols = df.columns[1:25]
    bin_edges = (
        np.array([float(col.split("Bin")[1]) for col in bin_cols] + [40.0]) * 1000
    )  # in nm
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Correct Period and FlowRate values
    df["Period"] = df["Period"].astype(float) / 100  # seconds
    df["FlowRate"] = df["FlowRate"].astype(float) / 100  # ml/s

    df.rename(
        columns={
            "Period": "Period (s)",
            "FlowRate": "FlowRate (ml/s)",
            "Temperature.C": "Temp (C)",
            "RH.p": "RH (%)",
        },
        inplace=True,
    )

    # Optional: store extra data (units resolved from headers)
    column_units: dict[str, str] = {}
    if extra_data:
        extra_df = df.drop(columns=bin_cols)
        extra_df.set_index("Datetime", inplace=True)
        extra_df, column_units = resolve_extra_columns(extra_df)
    else:
        extra_df = pd.DataFrame([])

    # Compute sample volume per row
    sample_volume = df["Period (s)"] * df["FlowRate (ml/s)"]  # in cm³

    # Convert counts to concentrations
    raw_counts = df[bin_cols].astype(float).values
    concentrations = raw_counts / sample_volume.values[:, None]  # type: ignore # #/cm³

    total_conc = np.nansum(concentrations, axis=1)
    total_df = pd.DataFrame(total_conc, columns=["Total_conc"])
    bin_df = pd.DataFrame(concentrations, columns=bin_mids.astype(str))

    final_df = pd.concat([df["Datetime"], total_df, bin_df], axis=1)

    # Already plain number (dN, cm⁻³); no post-construction conversion.
    OPCN = build_2d(
        final_df,
        bin_edges=bin_edges,
        bin_mids=bin_mids,
        instrument="OPCN",
        serial_number="unknown",
        unit="cm⁻³",
        dtype="dN",
        density=1.65,
        to_number=False,
        unnormalize=False,
    )

    if extra_data:
        OPCN._extra_data = extra_df
        OPCN._raw_extra_data = extra_df.copy()
        OPCN._meta["column_units"] = column_units
    return OPCN


###############################################################################


def _load_opcn3_file_new(
    file: str,
    extra_data: bool = False,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> Aerosol2D:
    """Load newer OPC-N3 CSV exports as size-resolved number concentrations.

    This loader handles updated OPC-N3 CSV formats where the date column
    is named ``Date`` and certain PM channels (``PM1``, ``PM2.5``,
    ``PM10``) are already present. It converts raw bin counts to number
    concentrations in cm⁻³ using the reported sampling period and flow,
    uses a fixed set of bin edges from the OPC-N3 specification, and
    returns an :class:`Aerosol2D` object.

    Args:
        file: Path to the OPC-N3 CSV export file.
        extra_data: If ``True``, non-bin columns (e.g. period, flow,
            temperature, RH, PM channels) are stored in ``.extra_data``
            indexed by ``Datetime``. If ``False`` (default), only
            ``Datetime``, ``Total_conc`` and bin concentrations are kept.
        encoding: Optional text encoding to use. If ``None`` (default),
            both encoding and delimiter are auto-detected together.
        delimiter: Optional CSV delimiter. If ``None`` (default), both
            encoding and delimiter are auto-detected together.

    Returns:
        Aerosol2D: Object with columns:

        * ``Datetime``
        * ``Total_conc`` (cm⁻³)
        * one column per size bin (cm⁻³), named by bin mid diameter in nm,

        plus metadata such as ``"bin_edges"``, ``"bin_mids"``, ``"density"``,
        ``"instrument"`` and ``"dtype"``.

    Raises:
        ValueError: If exactly one of ``encoding`` or ``delimiter`` is
            provided (both must be given or neither).
    """
    enc, delim = sniff(file, encoding, delimiter)

    df = read_table(file, delimiter=delim, encoding=enc)
    df.rename(columns={"Date": "Datetime"}, inplace=True)
    df.drop(columns="OPC", inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Parse ISO-formatted timestamp strings (e.g., '2024-01-21T15:30:01.000Z')
    df["Datetime"] = pd.to_datetime(
        df["Datetime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S"
    )

    # Determine bin edges/mids
    bin_cols = df.columns[1:25]
    bin_edges = (
        np.array(
            [
                0.35,
                0.46,
                0.66,
                1.0,
                1.3,
                1.7,
                2.3,
                3.0,
                4.0,
                5.2,
                6.5,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
                18.0,
                20.0,
                22.0,
                25.0,
                28.0,
                31.0,
                34.0,
                37.0,
                40.0,
            ]
        )
        * 1000
    )  # in nm
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Correct Period and FlowRate values
    df["Period"] = df["Period"].astype(float) / 100  # seconds
    df["FlowRate"] = df["FlowRate"].astype(float) / 100  # ml/s

    df.rename(
        columns={
            "Period": "Period (s)",
            "FlowRate": "FlowRate (ml/s)",
            "Temp": "Temp (C)",
            "RH": "RH (%)",
            "PM1": "PM1(ug/m3)",
            "PM2.5": "PM2.5(ug/m3)",
            "PM10": "PM10(ug/m3)",
        },
        inplace=True,
    )

    # Optional: store extra data (units resolved from headers)
    column_units: dict[str, str] = {}
    if extra_data:
        extra_df = df.drop(columns=bin_cols)
        extra_df.set_index("Datetime", inplace=True)
        extra_df, column_units = resolve_extra_columns(extra_df)
    else:
        extra_df = pd.DataFrame([])

    # Compute sample volume per row
    sample_volume = df["Period (s)"] * df["FlowRate (ml/s)"]  # in cm³

    # Convert counts to concentrations
    raw_counts = df[bin_cols].astype(float).values
    concentrations = raw_counts / sample_volume.values[:, None]  # type: ignore # #/cm³

    total_conc = np.nansum(concentrations, axis=1)
    total_df = pd.DataFrame(total_conc, columns=["Total_conc"])
    bin_df = pd.DataFrame(concentrations, columns=bin_mids.astype(str))

    final_df = pd.concat([df["Datetime"], total_df, bin_df], axis=1)

    # Already plain number (dN, cm⁻³); no post-construction conversion.
    OPCN = build_2d(
        final_df,
        bin_edges=bin_edges,
        bin_mids=bin_mids,
        instrument="OPCN",
        serial_number="unknown",
        unit="cm⁻³",
        dtype="dN",
        density=1.65,
        to_number=False,
        unnormalize=False,
    )

    if extra_data:
        OPCN._extra_data = extra_df
        OPCN._raw_extra_data = extra_df.copy()
        OPCN._meta["column_units"] = column_units
    return OPCN


###############################################################################


def _load_opcn3_file_pm_old(
    file: str,
    extra_data: bool = False,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> Aerosol1D:
    """Load legacy OPC-N3 CSV exports focusing on total and PM metrics only.

    This loader handles older OPC-N3 formats but returns a compact
    :class:`Aerosol1D` object containing only:

    * ``Total_conc`` (number concentration)
    * ``PM1`` (µg/m³)
    * ``PM2.5`` (µg/m³)
    * ``PM10`` (µg/m³)

    instead of the full size-resolved distribution.

    Args:
        file: Path to the OPC-N3 CSV export file.
        extra_data: If ``True``, a subset of auxiliary variables
            (e.g. temperature, RH, period) is stored in ``.extra_data``
            indexed by ``Datetime``. If ``False`` (default), only the
            main PM metrics and total concentration are kept.
        encoding: Optional text encoding to use. If ``None`` (default),
            both encoding and delimiter are auto-detected together.
        delimiter: Optional CSV delimiter. If ``None`` (default), both
            encoding and delimiter are auto-detected together.

    Returns:
        Aerosol1D: Object with columns:

        * ``Datetime``
        * ``Total_conc`` (cm⁻³)
        * ``PM1`` (µg/m³)
        * ``PM2.5`` (µg/m³)
        * ``PM10`` (µg/m³),

        plus metadata describing instrument type, density, units and dtypes.

    Raises:
        ValueError: If exactly one of ``encoding`` or ``delimiter`` is
            provided (both must be given or neither).
    """
    enc, delim = sniff(file, encoding, delimiter)

    df = read_table(file, delimiter=delim, encoding=enc)
    df.rename(columns={"date": "Datetime"}, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Parse ISO-formatted timestamp strings (e.g., '2024-01-21T15:30:01.000Z')
    df["Datetime"] = pd.to_datetime(
        df["Datetime"].str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S"
    )

    # Determine bin edges/mids
    bin_cols = df.columns[1:25]

    # Correct Period and FlowRate values
    df["Period"] = df["Period"].astype(float) / 100  # seconds
    df["FlowRate"] = df["FlowRate"].astype(float) / 100  # ml/s

    df.rename(
        columns={
            "Period": "Period (s)",
            "FlowRate": "FlowRate (ml/s)",
            "Temperature.C": "Temp (C)",
            "RH.p": "RH (%)",
            "PM1(ug/m3)": "PM1",
            "PM2.5(ug/m3)": "PM2.5",
            "PM10(ug/m3)": "PM10",
        },
        inplace=True,
    )

    # Optional: store extra data (units resolved from headers)
    column_units: dict[str, str] = {}
    if extra_data:
        extra_df = df[["Datetime", "Temp (C)", "RH (%)", "Period (s)"]]
        extra_df.set_index("Datetime", inplace=True)
        extra_df, column_units = resolve_extra_columns(extra_df)
    else:
        extra_df = pd.DataFrame([])

    # Compute sample volume per row
    sample_volume = df["Period (s)"] * df["FlowRate (ml/s)"]  # in cm³

    # Convert counts to concentrations
    raw_counts = df[bin_cols].astype(float).values
    concentrations = raw_counts / sample_volume.values[:, None]  # type: ignore # #/cm³

    total_conc = np.nansum(concentrations, axis=1)
    total_df = pd.DataFrame(total_conc, columns=["Total_conc"])
    PM_df = df[["PM1", "PM2.5", "PM10"]]

    final_df = pd.concat([df["Datetime"], total_df, PM_df], axis=1)

    OPCN = build_1d(
        final_df,
        instrument="OPCN",
        serial_number="unknown",
        unit={
            "Total_conc": "cm⁻³",
            "PM1": "µg/m³",
            "PM2.5": "µg/m³",
            "PM10": "µg/m³",
        },
        dtype={"Total_conc": "dN", "PM1": "dM", "PM2.5": "dM", "PM10": "dM"},
        extra_meta={"density": 1.65},
    )

    if extra_data:
        OPCN._extra_data = extra_df
        OPCN._raw_extra_data = extra_df.copy()
        OPCN._meta["column_units"] = column_units
    return OPCN


###############################################################################


def _load_opcn3_file_pm_new(
    file: str,
    extra_data: bool = False,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> Aerosol1D:
    """Load newer OPC-N3 CSV exports focusing on total and PM metrics only.

    This loader handles updated OPC-N3 formats and returns a compact
    :class:`Aerosol1D` object containing only:

    * ``Total_conc`` (number concentration)
    * ``PM1`` (µg/m³)
    * ``PM2.5`` (µg/m³)
    * ``PM10`` (µg/m³)

    rather than the full binned size distribution.

    Args:
        file: Path to the OPC-N3 CSV export file.
        extra_data: If ``True``, a subset of auxiliary variables
            (e.g. temperature, RH, period) is stored in ``.extra_data``
            indexed by ``Datetime``. If ``False`` (default), only the
            main PM metrics and total concentration are kept.
        encoding: Optional text encoding to use. If ``None`` (default),
            both encoding and delimiter are auto-detected together.
        delimiter: Optional CSV delimiter. If ``None`` (default), both
            encoding and delimiter are auto-detected together.

    Returns:
        Aerosol1D: Object with columns:

        * ``Datetime``
        * ``Total_conc`` (cm⁻³)
        * ``PM1`` (µg/m³)
        * ``PM2.5`` (µg/m³)
        * ``PM10`` (µg/m³),

        plus metadata describing instrument type, density, units and dtypes.

    Raises:
        ValueError: If exactly one of ``encoding`` or ``delimiter`` is
            provided (both must be given or neither).
    """
    enc, delim = sniff(file, encoding, delimiter)

    df = read_table(file, delimiter=delim, encoding=enc)
    df.rename(columns={"Date": "Datetime"}, inplace=True)
    df.drop(columns="OPC", inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Parse ISO-formatted timestamp strings (e.g., '2024-01-21T15:30:01.000Z')
    df["Datetime"] = pd.to_datetime(
        df["Datetime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S"
    )

    # Determine bin edges/mids
    bin_cols = df.columns[1:25]

    # Correct Period and FlowRate values
    df["Period"] = df["Period"].astype(float) / 100  # seconds
    df["FlowRate"] = df["FlowRate"].astype(float) / 100  # ml/s

    df.rename(
        columns={
            "Period": "Period (s)",
            "FlowRate": "FlowRate (ml/s)",
            "Temp": "Temp (C)",
            "RH": "RH (%)",
        },
        inplace=True,
    )

    # Optional: store extra data (units resolved from headers)
    column_units: dict[str, str] = {}
    if extra_data:
        extra_df = df[["Datetime", "Temp (C)", "RH (%)", "Period (s)"]]
        extra_df.set_index("Datetime", inplace=True)
        extra_df, column_units = resolve_extra_columns(extra_df)
    else:
        extra_df = pd.DataFrame([])

    # Compute sample volume per row
    sample_volume = df["Period (s)"] * df["FlowRate (ml/s)"]  # in cm³

    # Convert counts to concentrations
    raw_counts = df[bin_cols].astype(float).values
    concentrations = raw_counts / sample_volume.values[:, None]  # type: ignore # #/cm³

    total_conc = np.nansum(concentrations, axis=1)
    total_df = pd.DataFrame(total_conc, columns=["Total_conc"])
    PM_df = df[["PM1", "PM2.5", "PM10"]]

    final_df = pd.concat([df["Datetime"], total_df, PM_df], axis=1)

    OPCN = build_1d(
        final_df,
        instrument="OPCN",
        serial_number="unknown",
        unit={
            "Total_conc": "cm⁻³",
            "PM1": "µg/m³",
            "PM2.5": "µg/m³",
            "PM10": "µg/m³",
        },
        dtype={"Total_conc": "dN", "PM1": "dM", "PM2.5": "dM", "PM10": "dM"},
        extra_meta={"density": 1.65},
    )

    if extra_data:
        OPCN._extra_data = extra_df
        OPCN._raw_extra_data = extra_df.copy()
        OPCN._meta["column_units"] = column_units
    return OPCN
