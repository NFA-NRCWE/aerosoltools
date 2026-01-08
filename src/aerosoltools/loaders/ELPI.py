from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .Common import _detect_delimiter

###############################################################################


def _load_ELPI_metadata(
    file_path: Union[str, Path],
    delimiter: str = "\t",
    encoding: str = "utf-8",
) -> dict:
    """Parse ELPI header metadata into a structured dictionary.

    This helper reads the first ~36 lines of an ELPI export file and extracts
    key–value pairs written in the form ``key=value``. Values containing the
    specified delimiter are interpreted as lists; individual items are
    converted to ``float`` where possible, otherwise kept as strings. Scalar
    values are likewise converted to ``float`` when feasible.

    Args:
        file_path:
            Path to the ELPI data file (typically a ``.txt`` or ``.dat`` export).
        delimiter:
            Field separator used for list-like metadata values.
            Defaults to tab (``"\\t"``), which is the typical ELPI format.
        encoding:
            Text encoding used to read the file. Defaults to ``"utf-8"``.

    Returns:
        dict:
            A dictionary mapping metadata keys to parsed values. Entries may be:

            * ``float`` for scalar numeric values,
            * ``str`` for scalar non-numeric values,
            * ``list[float]`` / ``list[str]`` for delimited sequences.

    Notes:
        Only the early header region (first ~36 lines) is scanned. Lines that
        do not contain ``"="`` are ignored.
    """
    metadata: dict = {}

    # Scan the header region and collect key=value pairs
    with open(file_path, "r", encoding=encoding) as f:
        for row, line in enumerate(f):
            if row >= 36:
                break
            line = line.strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Split potential list values
            if delimiter in value:
                items = value.split(delimiter)
                try:
                    items = [float(v) for v in items]
                    value = items
                except ValueError:
                    # Fall back to list of strings if not purely numeric
                    value = items
            else:
                # Try to interpret as a single float; otherwise keep as string
                try:
                    value = float(value)
                except ValueError:
                    pass

            metadata[key] = value

    return metadata


###############################################################################


def Load_ELPI_file(file: str, extra_data: bool = False) -> Aerosol2D:
    """Description:
        Load an ELPI size-distribution export and return it as an
        :class:`Aerosol2D` number-size distribution with metadata.

    Args:
        file (str):
            Path to the ELPI data file (typically a ``.txt`` or ``.dat`` export).
        extra_data (bool, optional):
            If ``True``, non-distribution columns (operational parameters,
            status, etc.) are stored in ``extra_data``. Defaults to ``False``.

    Returns:
        Aerosol2D:
            ELPI size distributions with a datetime index, total concentration,
            size-resolved bins, and associated metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            :func:`_detect_delimiter`.
        ValueError:
            If the ``[Data]`` marker or header line cannot be found, or if
            datetime parsing fails for a substantial fraction of rows.
        KeyError:
            If required metadata entries (e.g. cutpoints, moments) are missing
            from the header.
        Exception:
            If the calculated moment/type in the ELPI metadata cannot be mapped
            to a known combination (Nu, Su, Vo, Ma), indicating an unsupported
            or unconverted export.

    Notes:
        Detailed description:
            This loader is tailored to ELPI (Electrical Low Pressure Impactor)
            exports generated via the ELPI Dekati software. The file should
            contain a header region with key–value metadata and a subsequent
            data block marked by ``[Data]``.

            Internally, the function:

            - Uses :func:`_detect_delimiter` to infer file encoding and field
              delimiter.
            - Calls :func:`_load_ELPI_metadata` to parse the header region into
              a metadata dictionary. In particular, it reads:

              - ``"D50values(um)"`` — used to construct ``bin_edges`` (cutpoints),
              - ``"CalculatedDi(um)"`` — used as geometric bin mid-diameters,
              - ``"Density(g/cm^3)"`` — particle density used for cutpoints,
              - ``"CalculatedMoment"`` / ``"CalculatedType"`` — to determine
                which moment the data represent (Nu, Su, Vo, Ma).

            - Converts bin edges and mid-diameters from µm to nm and, if the
              density differs from 1 g/cm³, recalculates internal bin edges
              using geometric means of adjacent midpoints before continuing.
            - Reads the data block after ``[Data]`` and assigns the aligned
              header as column names.
            - Renames the first column to ``"Datetime"`` and parses timestamps
              by trying several explicit formats (with and without fractional
              seconds), then a permissive fallback. If more than ~20% of rows
              fail, a ``ValueError`` is raised.
            - Extracts the size-distribution block (columns 34–47) as the core
              distribution data and removes these columns from a copy used as
              potential extra data.
            - Determines the physical unit and data type from
              ``"CalculatedMoment"`` / ``"CalculatedType"`` using a small
              lookup (Nu → dN, Su → dS, Vo → dV, Ma → dM). If this fails, an
              exception is raised: A likely cause is that the data is given in
              current (fA) and has not yet been converted with the Dekati software.
            - Computes ``"Total_conc"`` as the sum over all size bins, rounds
              bin mid-diameters to one decimal place, and renames the
              distribution columns to the stringified midpoints.
            - Assembles the final data frame as:

              - ``Datetime``,
              - ``Total_conc``,
              - size-bin columns named by bin midpoint (nm).

            - Constructs an :class:`Aerosol2D` object from this table and
              populates metadata:

              - ``bin_edges`` and ``bin_mids`` in nm,
              - ``density`` (g/cm³),
              - ``instrument`` set to ``"ELPI"``,
              - ``serial_number`` extracted from the first header line,
              - ``unit`` (string) and ``dtype`` (string) for the distribution.

            - Calls ``_convert_to_number_concentration()`` followed by
              :meth:`Aerosol2D.unnormalize_logdp` to ensure the distribution
              is expressed as number concentration per bin (dN, cm⁻³) rather
              than a moment normalized by ``dlogDp``.
            - If ``extra_data=True``, all non-distribution columns are stored
              in ``extra_data`` (with ``Datetime`` as index) and preserved in
              ``_raw_extra_data`` for later use.

        Theory:
            ELPI data can represent different geometric moments of the particle
            size distribution depending on the export settings:

            - ``Nu`` — number-based moment (dN),
            - ``Su`` — surface-based moment (dS),
            - ``Vo`` — volume-based moment (dV),
            - ``Ma`` — mass-based moment (dM).

            The metadata fields ``"CalculatedMoment"`` and ``"CalculatedType"``
            encode which of these is present. The loader uses this to assign
            the correct unit and dtype strings and then converts the
            distribution to number concentration (dN, cm⁻³) using internal
            helpers.

            When the density differs from 1 g/cm³, the nominal cutpoints are
            mass-based; to approximate number-based cutpoints, inner bin edges
            are recomputed using geometric means of adjacent mid-diameters.
            This preserves a consistent bin structure when changing from mass
            to number metrics.

    Examples:
        Typical usage is to load an ELPI export and directly access a
        number-size distribution for further analysis:

        .. code-block:: python

            import aerosoltools as at

            # Load ELPI data as a 2D number-size distribution
            elpi = at.Load_ELPI_file("data/ELPI_export.txt", extra_data=True)

            # Inspect the data
            print(elpi.data)

            # Inspect bin_edges and metadata
            print(elpi.bin_edges)
            print(elpi.metadata)

            # Plot the time-integrated size distribution
            fig, ax = elpi.plot_psd()
    """
    # Detect encoding and delimiter for this file
    encoding, delimiter = _detect_delimiter(file)

    # Load metadata and derive bin descriptors (convert µm -> nm)
    meta = _load_ELPI_metadata(file, delimiter, encoding)
    try:
        bin_edges = np.array(meta["D50values(um)"], dtype=float) * 1000
    except ValueError:
        bin_edges = np.array(meta["D50values(um)"].split("\t"), dtype=float) * 1000
    try:
        bin_mids = np.array(meta["CalculatedDi(um)"], dtype=float) * 1000
    except ValueError:
        bin_mids = np.array(meta["CalculatedDi(um)"].split("\t"), dtype=float) * 1000

    # Recalculate bin edges for non-unit density (mass-based cutpoints)
    if meta["Density(g/cm^3)"] != 1.0:
        bin_edges[1:-1] = np.sqrt(bin_mids[1:] * bin_mids[:-1])
        bin_edges[0] = bin_edges[1] ** 2 / bin_edges[2]
        bin_edges[-1] = bin_edges[-2] ** 2 / bin_edges[-3]
        print("################# Warning! #################")
        print("          Density ≠ 1.0 assumed           ")
        print("  Bin edges estimated via geometric means ")
        print("###########################################")

    # --- Load main data table and parse timestamps --------------------------
    with open(file, encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    # Locate the [Data] section marker
    try:
        data_idx = next(i for i, line in enumerate(lines) if line.strip() == "[Data]")
    except StopIteration:
        raise ValueError("Couldn't find the [Data] marker in the file.")

    # Find the header line immediately before [Data] (skipping blank/section lines)
    j = data_idx - 1
    while j >= 0 and (not lines[j].strip() or lines[j].lstrip().startswith("[")):
        j -= 1
    if j < 0 or delimiter not in lines[j]:
        # As a fallback, scan further up for a plausible header
        found = False
        for k in range(data_idx - 1, max(-1, data_idx - 25), -1):
            if delimiter in lines[k] and not lines[k].lstrip().startswith("["):
                j = k
                found = True
                break
        if not found:
            raise ValueError("Couldn't find the header line before [Data].")

    header = [
        h.strip() for h in lines[j].lstrip("\ufeff").rstrip("\r\n").split(delimiter)
    ]

    # Read only the data rows after [Data]
    df = pd.read_csv(
        file,
        sep=delimiter,
        header=None,
        skiprows=data_idx + 1,
        encoding=encoding,
        engine="python",
        on_bad_lines="skip",
    )

    # Align header length with actual number of columns (without reordering)
    if len(header) < df.shape[1]:
        header += [f"Unnamed_{i}" for i in range(len(header), df.shape[1])]
    elif len(header) > df.shape[1]:
        header = header[: df.shape[1]]
    df.columns = header

    # Force the first column to be named "Datetime"
    cols = list(df.columns)
    cols[0] = "Datetime"
    df.columns = cols

    # Parse datetime column: try known formats, then a permissive fallback
    formats = ["%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"]
    parsed = None
    for fmt in formats:
        s = pd.to_datetime(df["Datetime"], format=fmt, errors="coerce")
        if s.notna().mean() > 0.98:
            parsed = s
            break
    if parsed is None:
        parsed = pd.to_datetime(df["Datetime"], errors="coerce")
    if parsed.isna().mean() > 0.2:
        raise ValueError(
            "Datetime parsing failed for many rows — check the first column."
        )
    df["Datetime"] = parsed

    # --- Extract size distribution and non-distribution data ----------------
    dist_data = df.iloc[:, 34:48].copy()
    extra_df = df.drop(df.columns[33:47], axis=1)

    # Determine unit and dtype from ELPI "CalculatedMoment" / "CalculatedType"
    unit_map = {"Nu": "cm⁻³", "Su": "nm²/cm³", "Vo": "nm³/cm³", "Ma": "ug/m³"}
    dtype_map = {"Nu": "dN", "Su": "dS", "Vo": "dV", "Ma": "dM"}

    try:
        prefix = meta["CalculatedMoment"][:2]
        Unit = unit_map[prefix]
        dtype = dtype_map[prefix] + meta["CalculatedType"][2:]
    except (KeyError, TypeError) as e:
        raise Exception(
            "Unit and/or data type does not match the expected values. "
            "Ensure the data has been converted from current (fA) using the ELPI software."
        ) from e

    # Compute total concentration and relabel size-bin columns by bin mids
    total_conc = pd.DataFrame(np.nansum(dist_data, axis=1), columns=["Total_conc"])
    bin_mids = bin_mids.round(1)
    dist_data.columns = [str(mid) for mid in bin_mids]

    final_df = pd.concat([df["Datetime"], total_conc, dist_data], axis=1)

    # Build Aerosol2D object
    ELPI = Aerosol2D(final_df)

    # Populate and clean metadata
    meta["density"] = meta.pop("Density(g/cm^3)")
    meta["bin_edges"] = bin_edges.round(1)
    meta["bin_mids"] = bin_mids
    meta["instrument"] = "ELPI"

    # Extract serial number from the very first header line
    if delimiter == ",":
        serial_n = str(
            np.genfromtxt(
                file,
                delimiter=delimiter,
                skip_header=0,
                max_rows=1,
                dtype=str,
                encoding=encoding,
            )
        )[1][1:-1]
    else:
        serial_n = str(
            np.genfromtxt(
                file,
                delimiter=delimiter,
                skip_header=0,
                max_rows=1,
                dtype=str,
                encoding=encoding,
            )
        ).split(",")[1][1:-1]

    meta["serial_number"] = serial_n
    meta["dtype"] = dtype
    meta["unit"] = Unit

    for key in ["CalculatedDi(um)", "CalculatedType", "CalculatedMoment"]:
        meta.pop(key, None)

    ELPI._meta = meta

    # Convert to number concentration and remove dlogDp normalization
    ELPI._convert_to_number_concentration()
    ELPI.unnormalize_logdp()

    # Attach non-distribution columns as extra_data if requested
    if extra_data:
        extra_df.set_index("Datetime", inplace=True)
        ELPI._extra_data = extra_df

    return ELPI
