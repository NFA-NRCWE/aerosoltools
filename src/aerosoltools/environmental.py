"""Public class for environmental / weather loggers (:class:`Environmental1D`).

Multi-channel environmental time series — temperature, relative humidity,
pressure, wind, and sometimes co-logged gas/PM channels (e.g. Fourtec Bluefish,
DevLabs weather station). These are not particle number concentrations, so
``total_concentration`` does not apply; each channel is accessed by name.
"""

from __future__ import annotations

import pandas as pd

from ._core.nonparticle import _NonParticleMixin
from .aerosol1d import Aerosol1D


class Environmental1D(_NonParticleMixin, Aerosol1D):
    """Multi-channel environmental time series.

    Reuses the :class:`Aerosol1D` machinery but does **not** expose
    ``total_concentration``. The primary channel defaults to temperature when
    present, else the first data column.

    Args:
        dataframe (pandas.DataFrame): Input data with a time column/index and
            one or more environmental channels (e.g. ``"Temperature"``,
            ``"RH"``, ``"Pressure"``).

    Attributes:
        temperature (pandas.Series): Air temperature (°C), when present.
        relative_humidity / rh (pandas.Series): Relative humidity (%), when
            present.
        pressure (pandas.Series): Air pressure, when present.
    """

    #: Primary summary metrics (see ``available_metrics``): T and RH.
    _primary_metric_keys = ("Temperature", "RH")

    def _primary_name(self) -> str:
        for candidate in ("Temperature", "Temp"):
            if candidate in self._data.columns:
                return candidate
        return super()._primary_name()

    @property
    def temperature(self) -> pd.Series:
        """Air temperature (°C)."""
        return self._require("Temperature")

    @property
    def relative_humidity(self) -> pd.Series:
        """Relative humidity (%)."""
        return self._require("RH")

    #: Short alias for :attr:`relative_humidity`.
    rh = relative_humidity

    @property
    def pressure(self) -> pd.Series:
        """Air pressure."""
        return self._require("Pressure")
