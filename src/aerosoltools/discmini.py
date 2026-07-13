"""Public class for DiSCmini personal dosimeters (:class:`DiSCmini`).

The DiSCmini reports particle **number** concentration (kept as the primary
:attr:`total_concentration`), a mean particle :attr:`size`, and lung-deposited
surface area :attr:`ldsa`. It is a particle instrument, so it composes plain
:class:`Aerosol1D` and only adds typed accessors for its extra channels.
"""

from __future__ import annotations

import pandas as pd

from .aerosol1d import Aerosol1D


class DiSCmini(Aerosol1D):
    """DiSCmini number / size / LDSA time series.

    Args:
        dataframe (pandas.DataFrame): Input data with a time column/index and
            ``Total_conc`` (number concentration), ``Size`` and ``LDSA`` columns.

    Attributes:
        size (pandas.Series): Mean particle size (nm).
        ldsa (pandas.Series): Lung-deposited surface area (nm²/cm³).
    """

    @property
    def size(self) -> pd.Series:
        """Mean particle size (nm)."""
        if "Size" not in self.data.columns:
            raise AttributeError("This DiSCmini dataset has no 'Size' channel.")
        return self.data["Size"]

    @property
    def ldsa(self) -> pd.Series:
        """Lung-deposited surface area (nm²/cm³)."""
        if "LDSA" not in self.data.columns:
            raise AttributeError("This DiSCmini dataset has no 'LDSA' channel.")
        return self.data["LDSA"]
