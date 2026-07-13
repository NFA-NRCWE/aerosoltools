import pandas as pd

from ..aethalometer import Aethalometer
from .Common import _detect_delimiter

###############################################################################


def Load_Aethalometer_file(file: str, extra_data: bool = False) -> Aethalometer:
    """Description:
        Load time-resolved black carbon data from an Aethalometer export file
        into an :class:`aerosoltools.AerosolAlt` object.

    Args:
        file (str):
            Path to the Aethalometer export file (typically ``.csv`` or
            ``.txt``) containing time-resolved concentration data.
        extra_data (bool, optional):
            If ``True``, non-core variables (e.g. diagnostics, sensor status
            or quality flags) are stored in ``AerosolAlt.extra_data`` with
            a ``Datetime`` index. If ``False`` (default), only the main
            measurement channels are kept in ``.data``.

    Returns:
        AerosolAlt:
            An :class:`~aerosoltools.aerosolalt.AerosolAlt` instance with:

            - ``.data`` containing the primary BC concentration channels
              (at least IR/UV/blue/green/red) indexed by ``Datetime``.
            - ``.metadata`` (``.meta``) populated with, e.g.:

              - ``"serial_number"`` — Aethalometer serial number.
              - ``"optical_config"`` — optical configuration string.
              - ``"instrument"`` — set to ``"Aethalometer"``.
              - ``"unit"`` — per-channel dict: ``"ng/m³"`` for the BC
                channels and ``""`` (dimensionless) for ``"AAE"``.
              - ``"dtype"`` — per-channel dict: ``"dM"`` (mass concentration)
                for the BC channels and ``"AAE"`` for the Ångström exponent.

            If ``extra_data=True``, additional columns that are not part of
            the core BC channels are stored in ``.extra_data``.

    Raises:
        FileNotFoundError:
            If the specified ``file`` path does not exist.
        UnicodeDecodeError:
            If the file encoding cannot be correctly decoded using the
            detected encoding. Check that the file is not corrupted and
            matches the expected export format.
        KeyError:
            If expected columns (e.g. ``"Date / time local"``,
            ``"Serial number"``, ``"Optical config"``, or the BC channels)
            are missing or renamed in an unexpected way. Verify that the
            file is an unmodified Aethalometer export.
        ValueError:
            If the datetime strings in the ``"Date / time local"`` column
            cannot be parsed using the expected format
            ``\"%Y-%m-%dT%H:%M:%S\"``. Check that the locale and export
            format match the loader assumptions.
        Exception:
            If the file is read successfully but, after dropping empty rows,
            no data remain (i.e. the dataset is effectively empty). This
            typically indicates an export problem or a file containing only
            headers.

    Notes:
        Detailed description:
            This loader is tailored to Aethalometer (e.g. MicroAeth) ASCII or
            CSV exports produced by the vendor software. Internally, it:

            - Auto-detects file delimiter and encoding.
            - Reads the file and drops the ``"Readable status"`` column
              if present, and removes fully empty rows.
            - Extracts metadata from the first row (serial number, optical
              configuration) and stores it in the internal ``.meta``
              dictionary.
            - Detects the file structure: for wide exports (many columns,
              typically ``>= 70``), it standardizes verbose column names
              such as ``"Biomass BCc  (ng/m^3)"`` and
              ``"Fossil fuel BCc  (ng/m^3)"`` to shorter labels
              (``"Biomass BCc"``, ``"Fossil fuel BCc"``) and includes
              them along with ``"AAE"`` as core channels. For narrower
              exports, only the main spectral BC channels are treated as
              core.
            - Builds a new :class:`AerosolAlt` object from these core
              columns, with ``Datetime`` as the time index.
            - Optionally collects all remaining non-core columns into
              ``.extra_data`` if ``extra_data=True``.

        Theory:
            The loader does not perform any transformations on the physical
            values themselves beyond basic type conversion; it assumes the
            Aethalometer has already converted raw optical attenuation to BC
            mass concentration (BCc) in units of ``ng/m³`` using the
            manufacturer’s algorithms and calibration constants.

            The different color channels (IR, UV, blue, green, red) provide
            wavelength-resolved BC mass concentrations. Optional channels
            such as ``"Biomass BCc"`` and ``"Fossil fuel BCc"`` are derived
            quantities based on multi-wavelength absorption (e.g. via Ångström
            exponent or source apportioning models) as implemented in the
            vendor software.

    Examples:
        Load a single Aethalometer file

        .. code-block:: python

            import aerosoltools as at

            # Load Aethalometer data with core BC channels only
            aeth = at.Load_Aethalometer_file("data/aeth_export.csv")
    """
    # Detect file encoding and delimiter (comma, semicolon, etc.)
    enc, delim = _detect_delimiter(file)

    # Read raw file, drop known non-data column and rows that are completely empty
    df = (
        pd.read_csv(file, delimiter=delim, encoding=enc, header=0, decimal=".")
        .drop(columns=["Readable status"])
        .dropna()
    )
    if df.empty:
        raise Exception("Empty data set")

    # Normalize row index and the datetime column name
    df = df.reset_index(drop=True)
    df.rename(
        columns={
            "Date / time local": "Datetime",
        },
        inplace=True,
    )

    # Parse timestamps to pandas datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%dT%H:%M:%S")

    # Extract instrument metadata from the first row. Per-channel unit/dtype
    # dicts are attached below, once the core channels are known.
    meta = {
        "serial_number": df["Serial number"].iloc[0],
        "optical_config": df["Optical config"].iloc[0],
        "instrument": "Aethalometer",
    }

    # Remove non-measurement / header-like columns no longer needed
    df.drop(columns=list(df.columns[:6]) + ["Optical config"], inplace=True)

    # Choose core measurement channels depending on file structure
    if len(df.columns) >= 70:
        # Normalize verbose column names to concise labels
        df.rename(
            columns={
                "Biomass BCc  (ng/m^3)": "Biomass BCc",
                "Fossil fuel BCc  (ng/m^3)": "Fossil fuel BCc",
            },
            inplace=True,
        )

        core_cols = [
            "Datetime",
            "IR BCc",
            "UV BCc",
            "Blue BCc",
            "Green BCc",
            "Red BCc",
            "Biomass BCc",
            "Fossil fuel BCc",
            "AAE",
        ]
    else:
        core_cols = [
            "Datetime",
            "IR BCc",
            "UV BCc",
            "Blue BCc",
            "Green BCc",
            "Red BCc",
        ]

    # Attach per-channel unit/dtype metadata (the AerosolAlt convention: a
    # name→value dict keyed by channel). The BC channels are mass
    # concentrations in ng/m³; the Ångström exponent (AAE) is dimensionless.
    channels = core_cols[1:]  # everything except "Datetime"
    meta["unit"] = {c: ("" if "AAE" in c else "ng/m³") for c in channels}
    meta["dtype"] = {c: ("AAE" if "AAE" in c else "dM") for c in channels}

    # Build Aethalometer object from core channels and attach metadata
    data = df[core_cols]
    aeth = Aethalometer(data.copy())
    aeth._meta = meta

    # Optionally store all remaining variables as extra_data
    if extra_data:
        extra_df = df.drop(columns=core_cols[1:])
        extra_df.set_index("Datetime", inplace=True)
        aeth._extra_data = extra_df

    return aeth
