"""Public class for single-gas monitors (:class:`Gas1D`).

Time-resolved gas concentration from an instrument that measures **one gas at a
time** (e.g. a Ranger head reading Cl₂ or NO₂). The measured species is carried
in :attr:`dtype` and the unit (ppm/ppb) in :attr:`unit`; the reading itself is
exposed as :attr:`concentration`.
"""

from __future__ import annotations

import pandas as pd

from ._core.nonparticle import _NonParticleMixin
from .aerosolalt import AerosolAlt


class Gas1D(_NonParticleMixin, AerosolAlt):
    """Time-resolved single-gas concentration.

    Reuses the time-series / activity / plotting / summary machinery of
    :class:`AerosolAlt`, but does **not** expose ``total_concentration`` (gases
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
