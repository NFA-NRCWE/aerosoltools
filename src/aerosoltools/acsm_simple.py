"""Public class for aethalometers / MicroAeth black-carbon monitors.

Wavelength-resolved black-carbon mass concentration (BCc, ng/m³). The primary
BC channels — IR/UV/Blue/Green/Red and, for wide exports, Biomass/Fossil-fuel
BCc plus the Ångström exponent (AAE) — are kept first-class in :attr:`data` and
exposed as direct properties (``.ir_bcc``, ``.uv_bcc``, …). The many raw vendor
columns live in :attr:`extra_data`.
"""

from __future__ import annotations

import pandas as pd

from ._core.nonparticle import _NonParticleMixin
from .aerosol1d import Aerosol1D


class ACSM_simple(_NonParticleMixin, Aerosol1D):
    """Time of flight mass spec. resolved mass concentration.
    Values are simplified attribution of particle type according to;
    Organic, sulfates, nitrate, ammonia and chlorine.

    Reuses the :class:`Aerosol1D` machinery but does **not** expose
    ``total_concentration`` (returned values are mass concentration).
    The conventional primary channel is organic masss concentration
    (:attr:`org`).

    Args:
        dataframe (pandas.DataFrame): Input data with a time column/index and
            the BC channels (``"Org"``, ``"SO4"``, …).

    Attributes:
        Org, SO4, NO3, NH4, Chlorine (pandas.Series):
            Per-wavelength black-carbon mass concentration (µg/m³).
    """

    _primary_column = "Org"
    #: Primary summary metric (see ``available_metrics``): IR-equivalent BC.
    _primary_metric_keys = ("Org",)

    @property
    def org(self) -> pd.Series:
        """Organic mass concentration (µg/m³)."""
        return self._require("Org")

    @property
    def sulfate(self) -> pd.Series:
        """Sulfate mass concentration (µg/m³)."""
        return self._require("SO4")

    @property
    def nitrate(self) -> pd.Series:
        """Nitrate mass concentration (µg/m³)."""
        return self._require("NO3")

    @property
    def ammonia(self) -> pd.Series:
        """Ammonia mass concentration (µg/m³)."""
        return self._require("NH4")

    @property
    def chlorine(self) -> pd.Series:
        """Chlorine mass concentration (µg/m³)."""
        return self._require("Chlorine")

