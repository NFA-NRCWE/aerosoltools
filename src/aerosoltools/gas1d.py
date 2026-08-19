"""Public class for single-gas monitors (:class:`Gas1D`).

Time-resolved gas concentration from an instrument that measures **one gas at a
time** (e.g. a Ranger head reading Cl₂ or NO₂). The measured species is carried
in :attr:`dtype` and the unit (ppm/ppb) in :attr:`unit`; the reading itself is
exposed as :attr:`concentration`.
"""

from __future__ import annotations

import pandas as pd

from ._core.metrics import MetricSpec, classify_unit
from ._core.nonparticle import _NonParticleMixin
from .aerosol1d import Aerosol1D


class Gas1D(_NonParticleMixin, Aerosol1D):
    """Time-resolved single-gas concentration.

    Reuses the time-series / activity / plotting / summary machinery of
    :class:`Aerosol1D`, but does **not** expose ``total_concentration`` (gases
    are not a particle number concentration).

    Args:
        dataframe (pandas.DataFrame): Input data with a time column/index and a
            single gas-concentration column named ``"Concentration"``.

    Attributes:
        concentration (pandas.Series): The gas concentration time series.

    Notes:
        The gas species is stored in :attr:`dtype` (e.g. ``"Cl₂"``/``"NO₂"``)
        and the unit in :attr:`unit` (``"ppm"``/``"ppb"``).
    """

    _primary_column = "Concentration"

    @property
    def concentration(self) -> pd.Series:
        """Gas concentration time series (species in :attr:`dtype`)."""
        return self._require("Concentration")

    def _species(self) -> str:
        """The measured gas species used as the metric key (e.g. ``"Cl₂"``)."""
        sp = str(self.dtype)
        return sp if sp and sp != "Unknown dtype" else "Concentration"

    def available_metrics(self) -> list[MetricSpec]:
        """A single metric keyed by the measured gas species.

        Keying by species (not the generic ``"Concentration"`` column) means two
        Cl₂ sensors merge into one summary column while Cl₂ and NO₂ stay separate.
        """
        unit = self.unit
        return [
            MetricSpec(
                self._species(), self._species(), unit, classify_unit(unit)[0], True
            )
        ]

    def _get_metric_series(self, metric_name: str):
        # Accept the species key (or a plain "Concentration") -> the gas reading.
        if metric_name == self._species() or metric_name.upper() == "CONCENTRATION":
            return self.concentration.astype(float), self.unit
        return super()._get_metric_series(metric_name)
