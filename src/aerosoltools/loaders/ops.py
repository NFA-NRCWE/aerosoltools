import datetime
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..aerosol2d import Aerosol2D
from .support.construct import build_2d
from .support.exceptions import FileFormatError
from .support.reading import read_cells, read_table, sniff
from .support.units import resolve_extra_columns

###############################################################################


def load_ops_file(file: str, extra_data: bool = True):
    """Load a TSI OPS spectrometer export and return it as an
    :class:`Aerosol2D` number-size distribution with metadata.

    Args:
        file (str):
            Path to the OPS data file exported either via the AIM software or
            directly from the OPS instrument.
        extra_data (bool, optional):
            If ``True``, auxiliary channels (e.g. status, environmental
            variables, Bin 17 for direct exports) are stored in ``extra_data``
            when supported by the underlying loader. Defaults to ``True`` —
            everything in the file is kept.

    Returns:
        Aerosol2D:
            OPS size distributions with a datetime index, total concentration,
            size-resolved bins, and associated metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            ``sniff``.
        FileFormatError:
            If the first header line cannot be recognised as either an AIM
            export (``"Sample File"``) or a direct OPS export
            (``"Instrument Name"``), and the file therefore cannot be routed
            to a supported loader.

    Notes:
        Detailed description:
            This loader supports two main OPS export flavours:

            - AIM software exports (AIM-generated CSV),
            - direct instrument exports written by the OPS itself.

            Internally, the function:

            - Uses ``sniff`` to infer file encoding and field
              delimiter.
            - Reads the first line of the file using :func:`numpy.genfromtxt`
              and inspects the first token:

              - If the line starts with ``"Sample File"``, the file is treated
                as an AIM export and passed to :func:`_load_ops_aim`, which:

                - reads the main data table starting at the AIM header,
                - reconstructs a single ``Datetime`` column from ``"Date"`` and
                  ``"Start Time"``,
                - extracts OPS mid diameters (in µm) from specific columns,
                  converts them to nm, and builds bin edges from lower/upper
                  cutpoints in the header,
                - sums over the size bins to compute ``"Total_conc"`` for each
                  time step,
                - interprets the metadata block to infer the underlying moment
                  (Nu, Su, Vo, Ma) and normalisation (e.g. ``/dlogDp``),
                - constructs an :class:`Aerosol2D` with ``Datetime``,
                  ``Total_conc`` and one column per size bin (named by bin
                  midpoint in nm), and attaches metadata such as bin edges,
                  bin mids, density, serial number, unit and dtype,
                - converts the distribution to number concentration and removes
                  any ``/dlogDp`` normalisation.

              - If the line starts with ``"Instrument Name"``, the file is
                treated as a direct OPS export and passed to
                :func:`_load_ops_direct`, which:

                - reads a metadata block from the top of the file (including
                  test start date/time, sample interval, bin cutpoints, flow
                  and density),
                - reconstructs absolute timestamps from the test start time and
                  the ``"Elapsed Time [s]"`` column,
                - converts counts in each size bin (Bin 1–16) to number
                  concentration in ``cm⁻³`` using the nominal flow rate and
                  sample interval adjusted for dead time,
                - computes ``"Total_conc"`` as the sum over bins,
                - defines bin edges from the reported bin cut points and
                  derives bin mid diameters in nm,
                - builds an :class:`Aerosol2D` with ``Datetime``, ``Total_conc``
                  and size-bin columns, and attaches metadata including bin
                  edges/mids, density, serial number, unit and dtype.

            - If ``extra_data=True``:

              - AIM exports preserve non-distribution columns (e.g. flags,
                additional channels) in ``extra_data``.
              - Direct exports store Bin 17 (converted to concentration when
                available) and other non-size-bin columns in ``extra_data``
                indexed by ``Datetime``.

            The returned :class:`Aerosol2D` is therefore a number-size
            distribution (dN, cm⁻³) with OPS-specific binning and metadata,
            ready for further analysis or plotting.

        Theory:
            OPS exports provide binned particle counts and, depending on the
            export type, may encode different moments or normalisations:

            - AIM exports may report number, surface, volume or mass based
              moments (Nu, Su, Vo, Ma) and can be normalised by ``dlogDp`` or
              ``dDp``. The internal AIM loader interprets this from the
              metadata and uses internal helpers to convert the distribution to
              number concentration (dN, cm⁻³) and remove any ``/dlogDp`` or
              ``/dDp`` normalisation.
            - Direct OPS exports report counts per bin over a known sample
              interval and flow. These are converted to number concentrations
              by

              .. math::

                  C = \\frac{N}{Q \\cdot (\\Delta t - t_\\mathrm{dead})},

              where :math:`N` is the count in the bin, :math:`Q` is the
              volumetric flow (cm³/s), :math:`\\Delta t` is the sample
              interval, and :math:`t_\\mathrm{dead}` is the recorded dead time.

            Bin edges (in µm) are taken from the OPS metadata and converted to
            nm; mid diameters are defined as geometric means of neighbouring
            edges and used to label the size-bin columns.

    Examples:
        Typical usage is to load OPS data from either AIM or direct
        instrument exports and work directly with the resulting
        number-size distribution:

        .. code-block:: python

            import aerosoltools as at

            # Load OPS data (AIM or direct export)
            ops = at.load_ops_file("data/OPS_export.csv", extra_data=True)

            # Inspect the first few rows
            print(ops.data.head())

            # Inspect bin edges and metadata
            print(ops.bin_edges)
            print(ops.metadata)

            # Plot a time-integrated or mean size distribution
            fig, ax = ops.plot_psd()
    """
    encoding, delimiter = sniff(file)

    # Peek at the first line to determine file type
    first_line = read_cells(
        file, skip_header=0, encoding=encoding, delimiter=delimiter
    )[0]

    if first_line == "Sample File":
        return _load_ops_aim(
            file, extra_data=extra_data, encoding=encoding, delimiter=delimiter
        )
    elif first_line == "Instrument Name":
        return _load_ops_direct(
            file, extra_data=extra_data, encoding=encoding, delimiter=delimiter
        )
    else:
        raise FileFormatError("Unrecognized OPS file format. Unable to parse.")


###############################################################################


def _load_ops_aim(
    file: str,
    extra_data: bool = False,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> Aerosol2D:
    """Load OPS data exported via AIM software into an :class:`Aerosol2D`.

    This loader handles OPS files exported through the AIM software. It
    reads the size-bin mid diameters and distribution block, reconstructs
    a single time axis from separate date and time columns, and interprets
    the metadata block to infer units and data type (e.g. ``dN/dlogDp``).

    Args:
        file: Path to the OPS AIM-exported data file.
        extra_data: If ``True``, non-distribution columns (e.g. status
            flags, environmental data) are stored in ``.extra_data``
            indexed by ``Datetime``. If ``False`` (default), only the
            time series of binned concentrations and ``Total_conc`` are
            kept.
        encoding: Optional text encoding to use. If ``None`` (default),
            both encoding and delimiter are auto-detected together.
        delimiter: Optional field delimiter to use (e.g. ``","`` or
            ``"\\t"``). If ``None`` (default), both encoding and delimiter
            are auto-detected together.

    Returns:
        Aerosol2D: Object with columns:

        * ``Datetime``
        * ``Total_conc`` (number concentration)
        * one column per size bin, named by bin mid diameter in nm,

        along with metadata entries such as ``"bin_edges"``, ``"bin_mids"``,
        ``"density"``, ``"instrument"``, ``"serial_number"``, ``"unit"`` and
        ``"dtype"``.

    Raises:
        ValueError: If exactly one of ``encoding`` or ``delimiter`` is
            provided (both must be given or neither).
        ValueError: If the unit / data-type combination in the metadata
            cannot be mapped to an expected format.
    """

    enc, delim = sniff(file, encoding, delimiter)

    # --- main table via pandas ------------------------------------------------
    df = read_table(file, header=13, encoding=enc, delimiter=delim)

    # OPS: mid diameters are in columns 17..32 (inclusive) in µm -> convert to nm
    bin_mids: NDArray[np.float64] = np.round(
        np.array(df.columns[17:33], dtype=float) * 1000.0, 1
    )

    # --- edges via genfromtxt, using file handles to avoid 'encoding' kw warning ----
    # lower edges row
    with open(file, "r", encoding=enc) as fh1:
        bin_lb = np.genfromtxt(
            fh1,
            delimiter=delim,
            skip_header=10,
            max_rows=1,
        )[
            17:-1
        ]  # all but the last column on that row

    # upper edge is the penultimate column on the next row
    with open(file, "r", encoding=enc) as fh2:
        # read a single value from the (-2) column on that row
        bin_ub_arr = np.genfromtxt(
            fh2,
            delimiter=delim,
            skip_header=11,
            max_rows=1,
            usecols=[-2],  # list to satisfy numpy's type stubs
        )
    # Ensure scalar float
    bin_ub = (
        float(bin_ub_arr)
        if isinstance(bin_ub_arr, (np.floating, float))
        else float(np.asarray(bin_ub_arr).item())
    )

    # Build edges in nm
    bin_edges: NDArray[np.float64] = np.append(bin_lb, [bin_ub]) * 1000.0
    # Updates the Bin_mids to correctly be the average between upper and lower bin edge
    bin_mids = ((bin_edges[1:] + bin_edges[:-1]) / 2).round(1)  # nm
    # --- timestamp and columns ----------------------------------------------
    # The AIM export has "Date", "Start Time" -> build a single datetime column
    df.rename(columns={"Sample #": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Start Time"], format="%m/%d/%Y %H:%M:%S"  # type: ignore
    )
    df.drop(columns=["Date", "Start Time"], inplace=True)

    # Distribution block (positions may vary in different exports; adjust if needed)
    dist_block = df.iloc[:, 15:31]
    dist_data = dist_block.to_numpy()
    total_conc = pd.DataFrame(np.nansum(dist_data, axis=1), columns=["Total_conc"])
    dist_df = pd.DataFrame(dist_data, columns=bin_mids.astype(str))
    final_df = pd.concat([df["Datetime"], total_conc, dist_df], axis=1)

    # Optional extra-data (the env/metadata columns kept, indexed by time)
    column_units: dict[str, str] = {}
    if extra_data:
        ops_extra = df.drop(columns=df.columns[13:])
        ops_extra.set_index("Datetime", inplace=True)
        ops_extra, column_units = resolve_extra_columns(ops_extra)

    # --- metadata block ------------------------------------------------------
    with open(file, "r", encoding=enc) as fh3:
        meta = np.genfromtxt(
            fh3,
            delimiter=delim,
            skip_header=1,
            max_rows=7,
            dtype=str,
        )
    weight = str(meta[6, 1])  # e.g., "Nu" prefix present
    dtype_desc = str(meta[5, 1])
    density = 1.0

    unit_dict = {"Nu": "cm⁻³", "Su": "nm²/cm³", "Vo": "nm³/cm³", "Ma": "ug/m³"}
    dtype_dict = {"Nu": "dN", "Su": "dS", "Vo": "dV", "Ma": "dM"}

    try:
        unit = unit_dict[weight[:2]]
        if "dlogDp" in dtype_desc:
            dtype = dtype_dict[weight[:2]] + "/dlogDp"
        elif "dDp" in dtype_desc:
            dtype = dtype_dict[weight[:2]] + "/dDp"
        else:
            dtype = dtype_dict[weight[:2]]
    except KeyError as e:
        raise FileFormatError(
            "Unit and/or data type does not match the expected format."
        ) from e

    # --- assemble object -----------------------------------------------------
    OPS = build_2d(
        final_df,
        bin_edges=bin_edges,
        bin_mids=bin_mids,
        instrument="OPS",
        serial_number=str(meta[1, 1]),
        unit=unit,
        dtype=dtype,
        density=density,
    )

    if extra_data:
        OPS._extra_data = ops_extra  # type: ignore[assignment]
        OPS._meta["column_units"] = column_units

    return OPS


###############################################################################


def _load_ops_direct(
    file: str,
    extra_data: bool = False,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> Aerosol2D:
    """Load OPS data exported directly from the instrument into :class:`Aerosol2D`.

    This loader handles raw OPS CSV exports written directly by the
    instrument (not via AIM). It parses the metadata header, reconstructs
    absolute timestamps from a test start time and elapsed seconds,
    converts counts to concentrations in cm⁻³ using the sample interval
    and flow, and builds an :class:`Aerosol2D` object.

    Bin 17 (particles above the last cut point) is excluded from the main
    distribution but can be preserved in ``.extra_data`` when
    ``extra_data=True``.

    Args:
        file: Path to the CSV file exported directly from the OPS
            instrument.
        extra_data: If ``True``, non-size-bin columns plus Bin 17 (as a
            concentration) are stored in ``.extra_data`` indexed by
            ``Datetime``. If ``False`` (default), only the main bins and
            ``Total_conc`` are kept.
        encoding: Optional text encoding to use. If ``None`` (default),
            both encoding and delimiter are auto-detected together.
        delimiter: Optional field delimiter to use (e.g. ``","`` or
            ``"\\t"``). If ``None`` (default), both encoding and delimiter
            are auto-detected together.

    Returns:
        Aerosol2D: Object with columns:

        * ``Datetime``
        * ``Total_conc`` (cm⁻³)
        * one column per size bin (cm⁻³), named by bin mid diameter in nm,

        along with metadata such as ``"bin_edges"``, ``"bin_mids"``,
        ``"density"``, ``"instrument"``, ``"serial_number"``, ``"unit"``,
        and ``"dtype"``.

    Raises:
        ValueError: If exactly one of ``encoding`` or ``delimiter`` is
            provided (both must be given or neither).
        Exception: If the header metadata is malformed or cannot be parsed.
    """
    enc, delim = sniff(file, encoding, delimiter)

    # Load measurement data, excluding last header-only bin
    df = read_table(file, header=37, encoding=enc, delimiter=delim)

    # Extract metadata as key-value dict
    meta = (
        read_table(
            file,
            header=None,
            nrows=35,
            encoding=enc,
            delimiter=delim,
            dtype={0: str},
        )
        .set_index(0)
        .squeeze()
        .to_dict()  # type: ignore
    )

    # Parse starting datetime from metadata
    start_datetime = datetime.datetime.strptime(
        f"{meta['Test Start Date']} {meta['Test Start Time']}", "%Y/%m/%d %H:%M:%S"
    )

    # Convert elapsed time to full timestamps
    df["Datetime"] = pd.to_timedelta(df["Elapsed Time [s]"], unit="s") + start_datetime
    df.drop(columns=["Elapsed Time [s]"], inplace=True)

    # Determine sample length from metadata
    sample_interval = datetime.datetime.strptime(
        meta["Sample Interval [H:M:S]"], "%H:%M:%S"
    )
    sample_length = datetime.timedelta(
        hours=sample_interval.hour,
        minutes=sample_interval.minute,
        seconds=sample_interval.second,
    ).total_seconds()

    # Apply correction: counts to concentration (excluding Bin 17)
    deadtime = df["Deadtime (s)"].to_numpy()
    counts = df.iloc[:, 0:16].to_numpy()  # Bin 1–16

    # Convert counts to concentration using flow rate (16.67 cm³/s)
    conc = np.true_divide(counts, 16.67 * (sample_length - deadtime[:, np.newaxis]))

    # If requested, store Bin 17 and other columns as extra data
    column_units: dict[str, str] = {}
    if extra_data:
        extra = df.drop(columns=df.columns[0:16]).copy()
        try:
            extra["Bin 17"] = extra["Bin 17"] / (16.67 * (sample_length - deadtime))
        except KeyError:
            pass
        extra.set_index("Datetime", inplace=True)
        # "Rel. Humidity" has no in-header unit; relative humidity is always %.
        extra, column_units = resolve_extra_columns(
            extra, curated={"Rel. Humidity": "%"}
        )
    else:
        extra = pd.DataFrame([])

    # Define bin edges and midpoints
    bin_edges = (
        np.array([float(meta[f"Bin {i} Cut Point (um)"]) for i in range(1, 18)]) * 1000
    )  # nm
    bin_mids = ((bin_edges[1:] + bin_edges[:-1]) / 2).round(1)  # nm

    # Compute total concentration
    total_conc = pd.DataFrame(np.nansum(conc, axis=1), columns=["Total_conc"])
    conc_df = pd.DataFrame(conc, columns=bin_mids.astype(str))
    df_final = pd.concat([df["Datetime"], total_conc, conc_df], axis=1)

    # Package into class. The counts→concentration step above already produced
    # plain number (dN, cm⁻³), so no post-construction conversion is applied.
    OPS = build_2d(
        df_final,
        bin_edges=bin_edges,
        bin_mids=bin_mids,
        instrument="OPS",
        serial_number=meta["Serial Number"],
        unit="cm⁻³",
        dtype="dN",
        density=float(meta["Density"]),
        to_number=False,
        unnormalize=False,
    )
    if extra_data:
        OPS._extra_data = extra
        OPS._meta["column_units"] = column_units

    return OPS
