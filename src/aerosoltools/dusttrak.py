"""Public class for TSI DustTrak PM monitors (:class:`DustTrak`).

The DustTrak reports particle **mass** as several PM fractions
(PM1/PM2.5/PM4/PM10) plus a total. It is a particle instrument; its primary
series (:attr:`total_concentration`) is the total PM mass channel.
"""

from __future__ import annotations

import pandas as pd

from .aerosol1d import Aerosol1D


class DustTrak(Aerosol1D):
    """DustTrak PM mass-fraction time series.

    Args:
        dataframe (pandas.DataFrame): Input data with a time column/index and
            ``PM1``, ``PM2.5``, ``PM4``, ``PM10`` and ``Total`` mass channels
            (µg/m³).

    Attributes:
        pm1, pm2_5, pm4, pm10, total (pandas.Series): PM mass fractions and the
            total PM mass (µg/m³).
    """

    @property
    def total_concentration(self) -> pd.Series:
        """Total PM mass concentration — the DustTrak ``"Total"`` channel."""
        if "Total" in self.data.columns:
            return self.data["Total"]
        return super().total_concentration

    def _channel(self, name: str) -> pd.Series:
        if name not in self.data.columns:
            available = ", ".join(map(str, self.data.columns))
            raise AttributeError(
                f"DustTrak has no '{name}' channel; available: {available}."
            )
        return self.data[name]

    @property
    def pm1(self) -> pd.Series:
        """PM1 mass concentration (µg/m³)."""
        return self._channel("PM1")

    @property
    def pm2_5(self) -> pd.Series:
        """PM2.5 mass concentration (µg/m³)."""
        return self._channel("PM2.5")

    @property
    def pm4(self) -> pd.Series:
        """PM4 mass concentration (µg/m³)."""
        return self._channel("PM4")

    @property
    def pm10(self) -> pd.Series:
        """PM10 mass concentration (µg/m³)."""
        return self._channel("PM10")

    @property
    def total(self) -> pd.Series:
        """Total PM mass concentration (µg/m³)."""
        return self._channel("Total")
