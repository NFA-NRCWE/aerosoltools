import datetime

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .support.construct import build_2d
from .support.exceptions import FileFormatError
from .support.reading import read_cells, read_table, sniff

###############################################################################


def load_grimm_file(file: str) -> Aerosol2D:
    """Load a Grimm spectrometer export and return it as an
    :class:`Aerosol2D` number-size distribution with metadata.

    Args:
        file (str):
            Path to the Grimm data file exported either directly from the
            instrument or via the Grimm software.

    Returns:
        Aerosol2D:
            Grimm size distributions with a datetime index, total concentration,
            size-resolved bins, and associated metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            ``sniff``.
        ValueError:
            If the file header cannot be recognised as either an
            instrument-direct export or a software export, or if subsequent
            parsing fails due to an unsupported data type.

    Notes:
        Detailed description:
            This loader is designed for Grimm optical particle counter /
            spectrometer exports in the text formats used by the instrument
            and the Grimm software.

            Internally, the function:

            - Uses ``sniff`` to infer file encoding and field
              delimiter.
            - Reads the first header line with :func:`numpy.genfromtxt` and
              inspects its first cell to decide which parser to use:

              - Instrument-direct export:

                - Identified when the first header cell contains
                  ``"File name"``.
                - The file is passed to :func:`_load_grimm_inst`, which:

                  - reads the main data table (header near the top of the file),
                  - parses timestamps from US-style date/time strings,
                  - converts bin labels like ``"0.25-0.3"`` µm into bin edges
                    and mid-diameters in nm,
                  - reads top-of-file metadata to determine whether the data
                    are given as mass or counts,
                  - normalises the size-resolved data to number concentration
                    in ``cm⁻³`` based on this type,
                  - builds an :class:`Aerosol2D` object with datetime, total
                    concentration, size-bin columns, and basic Grimm metadata
                    (bin_edges, bin_mids, unit, dtype).

              - Software export:

                - Identified when the first header cell is ``"<Header>"``.
                - The file is passed to :func:`_load_grimm_soft`, which:

                  - reads the software-exported table with date/time in the
                    first column and one column per size bin,
                  - parses timestamps from combined date/time strings,
                  - derives bin edges and mid-diameters from column labels
                    like ``"0.25µm"`` (in µm) and converts them to nm,
                  - reads metadata to determine the underlying data type
                    (mass per volume vs counts per litre),
                  - normalises the distribution to number concentration in
                    ``cm⁻³`` using the appropriate conversion,
                  - builds an :class:`Aerosol2D` object with datetime, total
                    concentration, size-bin columns, and Grimm metadata
                    (bin_edges, bin_mids, unit, dtype, location, serial_number).

            - If neither pattern is detected in the first header cell, a
              ``ValueError`` is raised indicating an unrecognised Grimm
              format.

            The resulting :class:`Aerosol2D` instance always contains a
            ``"Datetime"`` column, a ``"Total_conc"`` column with total number
            concentration, and one column per size bin named by its midpoint
            (nm), together with basic instrument metadata.

        Theory:
            Grimm exports may present data either as:

            - mass concentration per size bin, or
            - counts per litre per size bin.

            The internal loaders convert these to number concentration per
            bin in ``cm⁻³`` as follows:

            - For mass-based exports, a unit density and spherical
              particles are assumed. The particle volume for a bin with
              midpoint diameter :math:`D` is approximated as:

              .. math::

                  V = \\frac{\\pi}{6} \\cdot D^3

              Using this volume and density, mass is converted to number
              concentration.

            - For count-per-litre exports, number concentration in
              ``cm⁻³`` is obtained by dividing by 1000 (1 L = 1000 cm³).

            These assumptions yield a consistent number-size distribution
            (dN, cm⁻³) that is comparable to other instruments handled by
            aerosoltools.

    Examples:
        Typical usage is to load a Grimm file (instrument or software
        export) and work directly with the resulting number-size
        distribution:

        .. code-block:: python

            import aerosoltools as at

            # Load Grimm data as a 2D number-size distribution
            grimm = at.load_grimm_file("data/Grimm_export.txt")

            # Inspect the data
            print(grimm.data.head())

            # Inspect bin edges and metadata
            print(grimm.bin_edges)
            print(grimm.metadata)

            # Plot the time series
            fig, ax = grimm.plot_timeseries()
    """
    encoding, delimiter = sniff(file)
    header_line = read_cells(
        file, skip_header=0, encoding=encoding, delimiter=delimiter
    )

    # Normalise first header cell to a plain string
    first = header_line[0] if isinstance(header_line, np.ndarray) else header_line

    # Instrument-direct export: header contains "File name"
    if isinstance(first, str) and "File name" in first:
        return _load_grimm_inst(file, encoding, delimiter)

    # Software export: header often starts with "<Header>"
    if isinstance(first, str) and first.strip() == "<Header>":
        return _load_grimm_soft(file, encoding, delimiter)

    raise FileFormatError("Unrecognised Grimm file format.")


###############################################################################


def _load_grimm_soft(file: str, encoding: str, delimiter: str) -> Aerosol2D:
    """Load Grimm data exported via the Grimm software.

    This loader parses software-exported Grimm files, extracts the datetime,
    converts bin labels into numeric diameters, normalises mass/count data to
    number concentration, and returns an :class:`Aerosol2D` instance.

    Args:
        file:
            Path to the software-exported Grimm data file.
        encoding:
            Text encoding used for the file.
        delimiter:
            Field delimiter used in the file.

    Returns:
        Aerosol2D:
            Object with size-resolved number concentration, total concentration,
            and metadata (bin edges/mids, serial number, unit, dtype).

    Raises:
        ValueError:
            If the underlying data type is not recognised as a supported
            mass or count format.
    """
    # Read main data table
    df = read_table(file, delimiter=delimiter, encoding=encoding, header=13)
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df = df.dropna().reset_index(drop=True)

    # Parse timestamps from combined date/time string
    df["Datetime"] = [
        datetime.datetime.strptime(s.split(".")[0][-19:], "%d/%m/%Y %H:%M:%S")
        for s in df["Datetime"]
    ]

    # Derive bin edges and mids from column labels (e.g. "0.25µm")
    bin_edges_str = df.columns[1:]
    bin_edges = np.array([float(label[:-3]) for label in bin_edges_str] + [34]) * 1000.0
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Read metadata block
    meta = np.genfromtxt(
        file,
        delimiter=delimiter,
        encoding=encoding,
        skip_header=1,
        max_rows=10,
        dtype=str,
    )
    dtype_raw = meta[6].split(" ")[1]
    location = meta[1].split(":")[1]
    serial_number = meta[6].split(":")[1]

    # Determine normalisation factor based on reported data type
    if "g/m" in dtype_raw:
        # Mass -> number: assume unit density and spherical particles
        norm_vector = (np.pi / 6.0) * (bin_mids / 1e7) ** 3 * 1.0 * 1e12
    elif "/l" in dtype_raw:
        # Counts per litre -> counts per cm³
        norm_vector = 1000.0
    else:
        raise FileFormatError("Unsupported data type format in Grimm software export.")

    # Normalise size-resolved data to number concentration
    raw_data = df.iloc[:, 1:].astype(float).values / norm_vector
    total_conc = pd.DataFrame(np.nansum(raw_data, axis=1), columns=["Total_conc"])
    dist_df = pd.DataFrame(raw_data, columns=bin_mids.astype(str))

    # Assemble final DataFrame
    final_df = pd.concat([df["Datetime"], total_conc, dist_df], axis=1)

    # Build Aerosol2D object and attach metadata (already plain number).
    grimm = build_2d(
        final_df,
        bin_edges=bin_edges,
        bin_mids=bin_mids,
        instrument="Grimm",
        serial_number=serial_number,
        unit="cm⁻³",
        dtype="dN",
        density=1.0,
        extra_meta={"location": location},
        to_number=False,
        unnormalize=False,
    )

    return grimm


###############################################################################


def _load_grimm_inst(file: str, encoding: str, delimiter: str) -> Aerosol2D:
    """Load Grimm data exported directly from the instrument.

    This loader handles instrument-direct Grimm exports, where the header
    structure and bin labels differ slightly from the software exports. It
    parses the timestamps, converts bin labels to numeric diameters, and
    normalises to number concentration.

    Args:
        file:
            Path to the instrument-direct Grimm data file.
        encoding:
            Text encoding used for the file.
        delimiter:
            Field delimiter used in the file.

    Returns:
        Aerosol2D:
            Object with size-resolved number concentration, total concentration,
            and Grimm metadata (bin edges/mids, unit, dtype).

    Raises:
        ValueError:
            If the data type (mass vs count) is not recognised from the header.
    """
    # Read main table with header near top of file
    df = read_table(file, delimiter=delimiter, encoding=encoding, header=1)
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df = df.dropna().reset_index(drop=True)

    # Parse timestamps; if time is missing, append a default midnight time
    times: list[datetime.datetime] = []
    for t in df["Datetime"]:
        try:
            times.append(datetime.datetime.strptime(t, "%m/%d/%Y %I:%M:%S %p"))
        except ValueError:
            times.append(
                datetime.datetime.strptime(t + " 12:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
            )
    df["Datetime"] = times

    # Convert bin labels like "0.25-0.3" into edges and mids (nm)
    bin_labels = df.columns[1:-1]
    first_edges = [float(b.split("-")[0]) for b in bin_labels]
    bin_edges = np.array(first_edges + [32.0, 34.0]) * 1000.0
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Read top-of-file metadata to infer type (mass vs count)
    meta = np.genfromtxt(
        file, delimiter=delimiter, encoding=encoding, max_rows=1, dtype=str
    )
    dtype_raw = meta[2].split(" ")[1]

    # Determine normalisation factor
    if "Mass" in dtype_raw:
        # Mass -> number (assuming unit density, spherical particles)
        norm_vector = (np.pi / 6.0) * (bin_mids / 1e7) ** 3 * 1.0 * 1e12
    elif "Count" in dtype_raw:
        # Counts per litre -> cm³
        norm_vector = 1000.0
    else:
        raise FileFormatError(
            "Unsupported data type format in Grimm instrument export."
        )

    # Normalise raw size-resolved data to number concentration
    raw_data = df.iloc[:, 1:].astype(float).values / norm_vector
    total_conc = pd.DataFrame(np.nansum(raw_data, axis=1), columns=["Total_conc"])
    dist_df = pd.DataFrame(raw_data, columns=bin_mids.astype(str))
    final_df = pd.concat([df["Datetime"], total_conc, dist_df], axis=1)

    # Build Aerosol2D object and metadata (serial number often not embedded
    # cleanly); the data are already plain number.
    grimm = build_2d(
        final_df,
        bin_edges=bin_edges,
        bin_mids=bin_mids,
        instrument="Grimm",
        serial_number="Unknown",
        unit="cm⁻³",
        dtype="dN",
        density=1.0,
        to_number=False,
        unnormalize=False,
    )

    return grimm
