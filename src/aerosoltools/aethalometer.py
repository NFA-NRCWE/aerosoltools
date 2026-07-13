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
from .aerosolalt import AerosolAlt


class Aethalometer(_NonParticleMixin, AerosolAlt):
    """Wavelength-resolved black-carbon mass concentration.

    Reuses the :class:`AerosolAlt` machinery but does **not** expose
    ``total_concentration`` (black carbon is not a particle number
    concentration). The conventional primary channel is IR-equivalent BC
    (:attr:`ir_bcc`).

    Args:
        dataframe (pandas.DataFrame): Input data with a time column/index and
            the BC channels (``"IR BCc"``, ``"UV BCc"``, …).

    Attributes:
        ir_bcc, uv_bcc, blue_bcc, green_bcc, red_bcc (pandas.Series):
            Per-wavelength black-carbon mass concentration (ng/m³).
        biomass_bcc, fossil_bcc (pandas.Series): Source-apportioned BC, present
            only for wide (``>= 70``-column) exports.
        aae (pandas.Series): Ångström absorption exponent (dimensionless),
            present only for wide exports.
    """

    _primary_column = "IR BCc"

    @property
    def ir_bcc(self) -> pd.Series:
        """IR-equivalent black-carbon mass concentration (ng/m³)."""
        return self._require("IR BCc")

    @property
    def uv_bcc(self) -> pd.Series:
        """UV black-carbon mass concentration (ng/m³)."""
        return self._require("UV BCc")

    @property
    def blue_bcc(self) -> pd.Series:
        """Blue-wavelength black-carbon mass concentration (ng/m³)."""
        return self._require("Blue BCc")

    @property
    def green_bcc(self) -> pd.Series:
        """Green-wavelength black-carbon mass concentration (ng/m³)."""
        return self._require("Green BCc")

    @property
    def red_bcc(self) -> pd.Series:
        """Red-wavelength black-carbon mass concentration (ng/m³)."""
        return self._require("Red BCc")

    @property
    def biomass_bcc(self) -> pd.Series:
        """Biomass-apportioned black carbon (ng/m³); wide exports only."""
        return self._require("Biomass BCc")

    @property
    def fossil_bcc(self) -> pd.Series:
        """Fossil-fuel-apportioned black carbon (ng/m³); wide exports only."""
        return self._require("Fossil fuel BCc")

    @property
    def aae(self) -> pd.Series:
        """Ångström absorption exponent (dimensionless); wide exports only."""
        return self._require("AAE")
