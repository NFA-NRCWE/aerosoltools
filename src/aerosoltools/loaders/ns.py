import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .support.construct import build_2d
from .support.exceptions import FileFormatError
from .support.reading import read_cells, read_table, sniff

###############################################################################


def load_ns_file(file: str, extra_data: bool = False) -> Aerosol2D:
    """Load a NanoScan SMPS (NS) CSV export and return it as an
    :class:`Aerosol2D` number-size distribution with metadata.

    Args:
        file (str):
            Path to the NanoScan CSV export file.
        extra_data (bool, optional):
            If ``True``, non-size-bin columns (e.g. status, density) are stored
            in ``extra_data`` indexed by ``Datetime``. Defaults to ``False``.

    Returns:
        Aerosol2D:
            NanoScan size distributions with a datetime index, total
            concentration, size-resolved bins, and associated metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the CSV file cannot be decoded using the encodings tried by
            ``_detect_delimiter``.
        ValueError:
            If the reported data type/unit string cannot be mapped to a known
            NanoScan format (e.g. dN, dS, dV, dM), or if the datetime column
            cannot be parsed.

    Notes:
        Detailed description:
            This loader is tailored to NanoScan SMPS CSV exports produced by
            the TSI AIM instrument software. The file typically contains several
            header rows, followed by a table with:

            - a ``"Date Time"`` column,
            - size-bin columns labelled by mid diameters (nm),
            - additional metadata columns (e.g. particle density).

            Internally, the function:

            - Uses ``_detect_delimiter`` to infer file encoding and field
              delimiter.
            - Reads the main data table, starting at the NanoScan data header
              (``header=5``).
            - Drops columns that are not part of the size distribution or time:
              ``"File Index"``, ``"Sample #"``, and ``"Total Conc"``.
            - Extracts bin mid diameters from the first 13 size-bin column
              headers and converts them to floats. From these, it computes bin
              edges in nm using geometric means between adjacent mids, with
              fixed outer edges at 10 nm and 420 nm.
            - Renames ``"Date Time"`` to ``"Datetime"`` and parses timestamps
              using the format ``"%Y/%m/%d %H:%M:%S"``.
            - Selects the size-distribution columns (those matching the
              bin-mid labels) as numeric data.
            - Optionally collects all non-size-bin columns into ``extra_data``
              when ``extra_data=True``, with ``Datetime`` as the index.
            - Reads a header line to extract the instrument serial number.
            - Reads the first data row to infer the data-type string
              (e.g. ``"dN/dlogDp"``) and extracts particle density.
            - Maps the data-type prefix (``"dN"``, ``"dS"``, ``"dV"``, ``"dM"``)
              to a physical unit via a small lookup, raising a ``ValueError``
              if the type is unknown.
            - Computes ``"Total_conc"`` as the sum over all size bins for each
              time step and concatenates ``Datetime``, ``Total_conc`` and the
              size-bin columns into a single DataFrame.
            - Creates an :class:`Aerosol2D` object from this DataFrame and
              populates metadata:

              - ``instrument`` set to ``"NS"``,
              - ``bin_edges`` and ``bin_mids`` (rounded to one decimal place),
              - ``density`` (float),
              - ``serial_number``,
              - ``unit`` (string) and ``dtype`` (raw type string from header).

            - Calls ``_convert_to_number_concentration()`` followed by
              :meth:`Aerosol2D.unnormalize_logdp` so that the final
              distribution is expressed as number concentration per bin
              (dN, cm⁻³) without ``/dlogDp`` normalisation.
            - If ``extra_data=True``, attaches the non-size-bin columns as
              ``extra_data`` indexed by ``Datetime``.

        Theory:
            NanoScan exports can represent different moments of the particle
            size distribution (number, surface, volume, mass) and may be
            normalised by ``/dlogDp``. The raw type string typically encodes
            this, for example:

            - ``dN/dlogDp`` — number concentration per logarithmic diameter
              interval,
            - ``dS/dlogDp`` — surface-based moment,
            - ``dV/dlogDp`` — volume-based moment,
            - ``dM/dlogDp`` — mass-based moment.

            From the two-letter prefix (``dN``, ``dS``, ``dV``, ``dM``), the
            loader assigns the corresponding unit (cm⁻³, nm²/cm³, nm³/cm³,
            µg/m³). The internal helper then converts any supported moment to
            number concentration and removes the ``/dlogDp`` normalisation so
            that the resulting distribution is directly comparable to other
            number-size distributions in aerosoltools.

      Examples:
          Typical usage is to load a NanoScan CSV file and work directly
          with the resulting number-size distribution:

          .. code-block:: python

              import aerosoltools as at

              # Load NanoScan data (keep extra metadata columns)
              ns = at.load_ns_file("data/NanoScan_export.csv", extra_data=True)

              # Inspect the first few rows
              print(ns.data.head())

              # Check bin edges and metadata
              print(ns.bin_edges)
              print(ns.metadata)

              # Plot a time-integrated or mean size distribution
              fig, ax = ns.plot_psd()
    """
    # Detect encoding and delimiter for the NanoScan export
    encoding, delimiter = sniff(file)

    # Load the full table from the main data header line
    ns_df = read_table(
        file, delimiter=delimiter, decimal=".", header=5, encoding=encoding
    )

    # Drop columns that are not part of the size distribution or time
    ns_df.drop(columns=["File Index", "Sample #", "Total Conc"], inplace=True)

    # Extract bin mid diameters from header and compute bin edges (nm)
    bin_mids = np.array(ns_df.columns[1:14], dtype=float)
    bin_edges = np.append(10.0, np.append(np.sqrt(bin_mids[1:] * bin_mids[:-1]), 420.0))

    # Parse datetime column and rename to "Datetime"
    ns_df.rename(columns={"Date Time": "Datetime"}, inplace=True)
    ns_df["Datetime"] = pd.to_datetime(ns_df["Datetime"], format="%Y/%m/%d %H:%M:%S")

    # Select size-distribution columns as numeric data
    size_data = ns_df[bin_mids.astype(str)].copy()

    # Optionally keep all non-size-bin columns as extra_data
    if extra_data:
        ns_extra = ns_df.drop(columns=bin_mids.astype(str))
        ns_extra.set_index("Datetime", inplace=True)
    else:
        ns_extra = pd.DataFrame([])

    # Read a header line for instrument serial number
    with open(file, "r", encoding=encoding, newline="") as fh:
        for _ in range(2):
            fh.readline()
        line = fh.readline()
    fields = line.rstrip("\n").split(delimiter)
    serial_number = fields[1].strip()

    # Inspect first data row to infer dtype string and density
    dtype_line = str(
        read_cells(file, skip_header=5, encoding=encoding, delimiter=delimiter)
    )
    dtype = dtype_line.split(" ")[0]
    density = ns_df["Particle Density (g/cc)"].iloc[0]

    # Map data type prefix to physical unit
    unit_dict = {
        "dN": "cm⁻³",
        "dS": "nm²/cm³",
        "dV": "nm³/cm³",
        "dM": "ug/m³",
    }
    try:
        unit = unit_dict[dtype[:2]]
    except KeyError as e:
        raise FileFormatError(
            "Unit and/or data type does not match the expected NanoScan format."
        ) from e

    # Compute total concentration per time step from size bins
    total_conc = size_data.sum(axis=1)
    total_col = pd.DataFrame({"Total_conc": total_conc})
    data_out = pd.concat([ns_df["Datetime"], total_col, size_data], axis=1)

    # Instantiate, attach metadata, standardise to number and undo /dlogDp.
    NS = build_2d(
        data_out,
        bin_edges=bin_edges.round(1),
        bin_mids=bin_mids.round(1),
        instrument="NS",
        serial_number=serial_number,
        unit=unit,
        dtype=dtype,
        density=float(density),
    )

    # Attach optional extra channels
    if extra_data:
        NS._extra_data = ns_extra

    return NS
