"""Loader for Ranger chlorine (Cl₂) gas-sensor exports."""

import pandas as pd

from ..aerosol1d import Aerosol1D
from .Common import _detect_delimiter

###############################################################################


def Load_Ranger_file(file: str, extra_data: bool = False) -> Aerosol1D:
    """Description:
        Load a Ranger chlorine-sensor CSV export and return the chlorine (Cl₂)
        concentration as an :class:`Aerosol1D` time series.

    Args:
        file (str):
            Path to the Ranger ``.csv`` export.
        extra_data (bool, optional):
            If ``True``, the non-core columns (UTC time, location id and the
            instrument status flag) are kept in the returned object's
            :attr:`~aerosoltools.aerosol1d.Aerosol1D.extra_data`. Defaults to
            ``False``.

    Returns:
        Aerosol1D:
            An :class:`~aerosoltools.aerosol1d.Aerosol1D` instance with:

            - ``.data`` carrying a datetime index (local time) and a
              ``"Total_conc"`` column holding the Cl₂ concentration,
            - metadata: ``instrument`` (``"Ranger"``), ``serial_number``,
              ``unit`` (``"ppm"``) and ``dtype`` (``"Cl₂"``).

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        ValueError:
            If the expected chlorine-concentration column is missing or the
            timestamps cannot be parsed.

    Notes:
        Detailed description:
            The Ranger exports a small self-describing CSV: a serial-number
            line, a header row and then one reading per minute. The relevant
            columns are::

                UTC time, Local time, LocationID, CL2 (PPM), Status(...)

            The loader uses the **local time** column as the time base (so the
            series aligns on wall-clock time with co-located aerosol
            instruments), the ``CL2 (PPM)`` column as the concentration, and
            reads the serial number from the first line. Because the result is
            a plain :class:`Aerosol1D`, all of the usual time-series
            operations — cropping, resampling/averaging, smoothing, activity
            marking and exposure summaries — work unchanged (with ``"PNC"``
            resolving to the Cl₂ series).

    Examples:
        .. code-block:: python

            import aerosoltools as at

            cl2 = at.load_ranger_file("Ranger_2605-1000088-A.csv")
            print(cl2.data.head())
            print(cl2.metadata)         # instrument / serial / unit
            fig, ax = cl2.plot_total_conc()
    """
    encoding, delimiter = _detect_delimiter(file)

    # First line holds the serial number, e.g. "Ranger Serial Number:,2605-..."
    with open(file, "r", encoding=encoding, errors="replace") as f:
        first_line = f.readline().strip()
    serial = ""
    if delimiter in first_line:
        parts = first_line.split(delimiter)
        if len(parts) > 1:
            serial = parts[1].strip()

    # The header row follows the serial line.
    df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]

    conc_col = next((c for c in df.columns if c.upper().startswith("CL2")), None)
    if conc_col is None:
        raise ValueError(
            "No chlorine-concentration column (e.g. 'CL2 (PPM)') found in the "
            "Ranger file."
        )
    time_col = "Local time" if "Local time" in df.columns else df.columns[0]

    df["Datetime"] = pd.to_datetime(df[time_col], errors="raise")
    df["Total_conc"] = pd.to_numeric(df[conc_col], errors="coerce")

    ranger = Aerosol1D(df[["Datetime", "Total_conc"]].copy())
    ranger._meta["instrument"] = "Ranger"
    ranger._meta["serial_number"] = serial
    ranger._meta["unit"] = "ppm"
    ranger._meta["dtype"] = "Cl₂"

    if extra_data:
        keep = [c for c in ("UTC time", "LocationID") if c in df.columns]
        status_col = next(
            (c for c in df.columns if c.lower().startswith("status")), None
        )
        if status_col:
            keep.append(status_col)
        extra = df[["Datetime", *keep]].copy()
        extra.set_index("Datetime", inplace=True)
        ranger._extra_data = extra
        ranger._raw_extra_data = extra.copy()

    return ranger
