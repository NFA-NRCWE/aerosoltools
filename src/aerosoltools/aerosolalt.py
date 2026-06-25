"""Public class for alternative-metric aerosol instruments (:class:`AerosolAlt`).

Holds the data model (``__init__``); the alternative-metric behaviour lives
in :class:`aerosoltools._core.alt.AltMixin`, which this class composes. The
public API is unchanged.
"""

from __future__ import annotations

import pandas as pd

from ._core.alt import AltMixin
from .aerosol1d import Aerosol1D


class AerosolAlt(AltMixin, Aerosol1D):
    """Description:
        One-dimensional aerosol time series with multiple scalar channels.

    Args:
        dataframe (pandas.DataFrame):
            Input data with a time column or time index and one or more scalar
            data columns. The first time-like column is converted to a
            DatetimeIndex if needed.

    Notes:
        Detailed description:
            :class:`AerosolAlt` extends :class:`Aerosol1D` for instruments that
            log several scalar channels in the same time series, such as:

            * multiple size-integrated aerosol metrics (e.g. PM1, PM2.5, PM10),
            * LDSA, flow, or other auxiliary variables,
            * environmental or diagnostic signals.

            The underlying data model is the same as for :class:`Aerosol1D`
            (time-indexed 1D series), but plotting and summary methods accept a
            ``parameter`` argument that lets you choose which column in
            :attr:`data` is treated as the “primary” signal.

            Internally, the constructor delegates to :class:`Aerosol1D`:

            * Ensures a DatetimeIndex is present (or creates one from a time
              column).
            * Stores the main time series in ``data``.
            * Keeps non-core channels in ``extra_data`` (if attached later).
            * Carries metadata such as ``instrument``, ``unit`` and ``dtype``
              either as scalars or per-column mappings.

            All activity-handling functionality (e.g. marking activity periods,
            cropping by activity) is inherited from :class:`Aerosol1D` and works
            transparently for any scalar channel.

    Examples:
        A typical workflow is to use :class:`AerosolAlt` for instruments
        with several scalar outputs (e.g. LDSA + flow + flags) and then
        select which one to visualise or summarise:

        .. code-block:: python

            import aerosoltools as at
            import pandas as pd

            # Example: construct from a DataFrame with multiple channels
            df = pd.DataFrame(
                {
                    "Datetime": pd.date_range("2024-01-01", periods=100, freq="1min"),
                    "LDSA": 50.0,
                    "Flow": 2.0,
                    "Flag": 0,
                }
            )

            alt = at.AerosolAlt(df)

            # Plot LDSA as the primary channel
            fig, ax = alt.plot_total_conc(parameter="LDSA")

            # Summarise Flow instead of LDSA
            summary = alt.summarize(parameter="Flow")
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__(dataframe)
