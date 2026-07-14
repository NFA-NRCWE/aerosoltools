from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .support.exceptions import FileFormatError
from .support.parsing import _detect_delimiter

###############################################################################


def _load_SMPS_metadata(
    file_path: Union[str, Path],
    delimiter: str = ",",
    encoding: str = "iso-8859-1",
) -> dict:
    """Extract metadata from an SMPS-formatted export file.

    The function scans the first ~25 lines of an SMPS text export, parses
    key–value pairs separated by a comma, and attempts to coerce values into
    floats or lists of floats when possible. Remaining values are left as
    strings.

    Args:
        file_path: Path to the SMPS export file.
        delimiter: Delimiter used inside the value field for lists
            (defaults to ``","``).
        encoding: File encoding used to open the metadata header
            (defaults to ``"iso-8859-1"``).

    Returns:
        dict: A mapping from metadata keys to parsed values. Values are:

        - ``float`` where a single numeric value is detected.
        - ``list[float]`` or ``list[str]`` where a delimited list is present.
        - ``str`` otherwise.
    """
    metadata: dict = {}
    with open(file_path, "r", encoding=encoding) as f:
        for i, line in enumerate(f):
            if i >= 25:
                break
            line = line.strip()
            if "," in line:
                key, value = line.split(",", 1)
                key = key.strip()
                value = value.strip()

                if delimiter in value:
                    try:
                        value = [float(v) for v in value.split(delimiter)]
                    except ValueError:
                        value = value.split(delimiter)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

                metadata[key] = value
    return metadata


###############################################################################


def load_smps_file(file: str, extra_data: bool = False) -> Aerosol2D:
    """Load an SMPS size-distribution export and return it as an
    :class:`Aerosol2D` number-size distribution with metadata.

    Args:
        file (str):
            Path to the SMPS ``.txt`` or CSV export file.
        extra_data (bool, optional):
            If ``True``, non-distribution columns (e.g. status, flow, voltages)
            are stored in ``extra_data`` indexed by ``Datetime``. Defaults to
            ``False``.

    Returns:
        Aerosol2D:
            SMPS size distributions with a datetime index, total concentration,
            size-resolved bins, and associated metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            ``_detect_delimiter``.
        ValueError:
            If CSV parsing fails in a way that prevents reading the header or
            main data table.
        FileFormatError:
            If the weight/unit combination in the SMPS header does not match
            the expected format and the data type cannot be interpreted.

    Notes:
        Detailed description:
            This loader targets SMPS (Scanning Mobility Particle Sizer) exports
            where the first ~25 lines contain metadata and the main data table
            starts at a fixed header row.

            Internally, the function:

            - Uses ``_detect_delimiter`` to infer file encoding and field
              delimiter, then reads the main data block with
              :func:`pandas.read_csv` using ``header=25``.
            - Calls :func:`_load_SMPS_metadata` to parse the first ~25 lines
              into a metadata dictionary, attempting to coerce values to floats
              or lists where possible.
            - Renames ``"Sample #"`` to ``"Datetime"`` and constructs absolute
              timestamps by combining:

              - ``"Date"`` (e.g. ``"%d/%m/%Y"``),
              - ``"Start Time"`` (e.g. ``"%H:%M:%S"``),

              into a single ``Datetime`` column, after which the original
              ``"Date"`` and ``"Start Time"`` columns are dropped.

            - Identifies the SMPS size-bin columns as ``df.columns[7:103]``
              (Python 0-based, corresponding to the instrument’s diameter
              range) and converts the headers to floating mid diameters in nm:

              - ``bin_cols`` → text mid diameters,
              - ``bin_mids`` → ``np.array(bin_cols, dtype=float)``.

            - Reconstructs the full SMPS bin structure using the theoretical
              logarithmic grid:

              - builds an array of 192 logarithmically spaced midpoints

                .. code-block:: python

                    bin_mid_log = np.round(
                        np.array([10 ** ((i - 0.5) / 64.0) for i in range(192)]), 1
                    )

              - finds indices where these theoretical midpoints match the
                midpoints present in the file,
              - uses those indices to select the corresponding bin edges from

                .. code-block:: python

                    all_edges = np.round(
                        [10 ** ((i - 1) / 64.0) for i in range(192)], 2
                    )

              - and slices them to yield ``bin_edges`` for the exported range.

            - Extracts the size-distribution matrix from ``bin_cols``, computes
              the total number concentration for each timestamp via
              ``np.nansum(...)``, and assembles a final DataFrame:

              - ``Datetime``,
              - ``Total_conc``,
              - one column per size bin (column names are the bin midpoints in
                nm, converted to strings).

            - Interprets the weight and unit information from the metadata:

              - reads ``meta["Weight"]`` to determine the base moment prefix
                (``"Nu"``, ``"Su"``, ``"Vo"``, ``"Ma"``),
              - reads ``meta["Units"]`` to detect whether the data are
                normalised by ``dlogDp`` or ``dDp``,
              - maps these to:

                + a physical unit (e.g. ``"cm⁻³"`` for number),
                + a dtype string (e.g. ``"dN/dlogDp"``).

              If this mapping fails (unknown prefix or units string), an
              ``Exception`` is raised.

            - Constructs an :class:`Aerosol2D` object from the assembled
              DataFrame and attaches metadata:

              - all parsed metadata except ``"Weight"`` and ``"Units"``,
              - ``instrument`` set to ``"SMPS"``,
              - ``bin_edges`` and ``bin_mids`` in nm,
              - ``density`` (fixed at 1.0 here),
              - ``serial_number`` built from the classifier and detector model
                entries in the metadata,
              - ``unit`` and ``dtype`` derived from the weight/unit mapping.

            - Calls ``_convert_to_number_concentration()`` followed by
              :meth:`Aerosol2D.unnormalize_logdp` to express the distribution
              as number concentration per bin (dN, cm⁻³) rather than the
              original reported moment (which may be normalised by ``dlogDp``
              or ``dDp``).

            - If ``extra_data=True``, all non-size-bin columns are kept in a
              separate DataFrame with ``Datetime`` as index and stored in
              ``extra_data``.

        Theory:
            SMPS instruments typically use a logarithmically spaced diameter
            grid with a fixed number of channels per decade. In this export:

            - Theoretical bin midpoints are generated via:

              .. math::

                  D_{p,\\mathrm{mid}} = 10^{(i - 0.5)/64},\\quad i = 0,\\dots,191,

              which produces 64 channels per decade on a log10 scale.
            - Bin edges are then defined as:

              .. math::

                  D_{p,\\mathrm{edge}} = 10^{(i - 1)/64},\\quad i = 0,\\dots,191,

              and sliced to match the subset of midpoints present in the file.

            The weight prefix (``Nu``, ``Su``, ``Vo``, ``Ma``) encodes whether
            the reported moment is based on number, surface, volume or mass.
            The loader uses this to determine the correct unit/dtype string and
            eventually converts the distribution to number concentration
            (dN, cm⁻³) using internal helpers.

    Examples:
        Typical usage is to load SMPS size distributions for further
        processing or plotting:

        .. code-block:: python

            import aerosoltools as at

            # Load SMPS data as a 2D number-size distribution
            smps = at.load_smps_file("data/SMPS_export.txt", extra_data=True)

            # Inspect the data
            print(smps.data.head())

            # Inspect bin edges and metadata
            print(smps.bin_edges)
            print(smps.metadata)

            # Plot a time-integrated size distribution
            fig, ax = smps.plot_psd()
    """
    encoding, delimiter = _detect_delimiter(file)
    df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, header=25)
    meta = _load_SMPS_metadata(file, delimiter, encoding)

    # Parse datetime
    df.rename(columns={"Sample #": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Start Time"], format="%d/%m/%Y %H:%M:%S"  # type: ignore
    )
    df.drop(columns=["Date", "Start Time"], inplace=True)

    # Bin columns and conversion
    bin_cols = df.columns[7:103]  # columns 9–105 (Python 0-based)
    bin_mids = np.array(bin_cols, dtype=float)

    # Define full SMPS bin midpoints and identify matching edges
    bin_mid_log = np.round(np.array([10 ** ((i - 0.5) / 64.0) for i in range(192)]), 1)
    size_idx = np.nonzero(np.isin(bin_mid_log, bin_mids))[0]
    bin_idx = np.append(size_idx, size_idx[-1] + 1)
    bin_edges = np.round([10 ** ((i - 1) / 64.0) for i in range(192)], 2)[bin_idx]

    # Extract and format main data
    dist_data = df[bin_cols].to_numpy()
    total_conc = np.nansum(dist_data, axis=1)
    df_total = pd.DataFrame(total_conc, columns=["Total_conc"])
    df_dist = pd.DataFrame(dist_data, columns=bin_mids.astype(str))
    final_df = pd.concat([df["Datetime"], df_total, df_dist], axis=1)

    # Unit / dtype interpretation
    density = 1.0
    unit_dict = {"Nu": "cm⁻³", "Su": "nm²/cm³", "Vo": "nm³/cm³", "Ma": "ug/m³"}
    dtype_dict = {"Nu": "dN", "Su": "dS", "Vo": "dV", "Ma": "dM"}

    try:
        weight_prefix = meta["Weight"][:2]
        unit = unit_dict[weight_prefix]
        if "dlogDp" in meta["Units"]:
            dtype = dtype_dict[weight_prefix] + "/dlogDp"
        elif "dDp" in meta["Units"]:
            dtype = dtype_dict[weight_prefix] + "/dDp"
        else:
            dtype = dtype_dict[weight_prefix]
    except Exception:
        raise FileFormatError(
            "Unit and/or data type does not match the expected format."
        )

    # Construct object
    smps = Aerosol2D(final_df)
    smps._meta = {
        **{k: v for k, v in meta.items() if k not in ("Weight", "Units")},
        "instrument": "SMPS",
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "density": density,
        "serial_number": (
            f"Classifier: {meta['Classifier Model'][2]}, "
            f"Detector: {meta['Detector Model'][2]}"
        ),
        "unit": unit,
        "dtype": dtype,
    }

    smps._convert_to_number_concentration()
    smps.unnormalize_logdp()

    if extra_data:
        extra_df = df.drop(columns=bin_cols)
        extra_df.set_index("Datetime", inplace=True)
        smps._extra_data = extra_df

    return smps
