"""Public class for Naneos Partector LDSA dosimeters (:class:`Partector`).

Lung-deposited surface area (LDSA, nm²/cm³) plus a TEM sampling flag and flow.
The Partector reports no particle number, so ``total_concentration`` does not
apply; LDSA is the primary channel.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ._core.nonparticle import _NonParticleMixin
from .aerosol1d import Aerosol1D


class Partector(_NonParticleMixin, Aerosol1D):
    """Partector LDSA time series with TEM sampling metadata.

    Reuses the :class:`Aerosol1D` machinery but does **not** expose
    ``total_concentration`` (the Partector reports no particle number). The
    primary channel is :attr:`ldsa`.

    Args:
        dataframe (pandas.DataFrame): Input data with a time column/index and
            ``"LDSA"`` (and typically ``"TEM"`` / ``"Flow"``) columns.

    Attributes:
        ldsa (pandas.Series): Lung-deposited surface area (nm²/cm³).
        flow (pandas.Series): Sample flow (l/min), when present.
        tem_samples (pandas.DataFrame | None): TEM sampling summary
            (``Start`` / ``End`` / ``Sample_vol [ml]``), from metadata.
    """

    _primary_column = "LDSA"

    @property
    def ldsa(self) -> pd.Series:
        """Lung-deposited surface area (nm²/cm³)."""
        return self._require("LDSA")

    @property
    def flow(self) -> pd.Series:
        """Sample flow (l/min)."""
        return self._require("Flow")

    @property
    def tem_samples(self) -> Optional[pd.DataFrame]:
        """TEM sampling summary table, if the loader recorded one."""
        return self._meta.get("TEM_samples")
