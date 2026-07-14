"""Structural typing for the facade state the ``_core`` mixins rely on.

The topic mixins (statistics, activities, time_ops, corrections, fitting, decay,
size_distribution, …) are composed onto the ``Aerosol1D``/``Aerosol2D`` facades
and freely use facade state that *no mixin declares* — ``self._data``,
``self.time``, ``self.copy_self()``, ``self._ensure_data_robustness()`` and, for
size-resolved data, ``self._sizebin_headers`` / ``self.bin_mids``. That contract
was entirely implicit.

These ``Protocol`` classes make it explicit, so a type checker (and a reader)
knows exactly what a mixin expects of its host. They are **typing-only**: at
runtime each mixin still subclasses ``object`` (via a ``TYPE_CHECKING`` guard),
so the composition and MRO of the facades are unchanged.

A mixin opts in with::

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from ._protocols import AerosolData
        _Host = AerosolData
    else:
        _Host = object

    class SummaryMixin(_Host):
        ...
"""

from __future__ import annotations

from typing import Protocol, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class AerosolData(Protocol):
    """The 1-D facade state and helpers any topic mixin may use on ``self``."""

    # -- backing frames / metadata (mutated in place by processing mixins) --
    _data: pd.DataFrame
    _extra_data: pd.DataFrame
    _raw_data: pd.DataFrame
    _raw_extra_data: pd.DataFrame
    _meta: dict
    _activity_periods: dict

    # -- public read-only views --------------------------------------------
    @property
    def data(self) -> pd.DataFrame: ...
    @property
    def time(self) -> pd.DatetimeIndex: ...
    @property
    def metadata(self) -> dict: ...
    @property
    def dtype(self) -> str: ...
    @property
    def unit(self) -> Union[str, dict]: ...
    @property
    def instrument(self) -> str: ...
    @property
    def serial_number(self) -> str: ...
    @property
    def _primary(self) -> pd.Series: ...
    @property
    def activities(self) -> dict: ...
    @property
    def activity_periods(self) -> dict: ...

    # -- helpers the mixins call -------------------------------------------
    def copy_self(self) -> "AerosolData":
        """Return an independent deep copy of the object."""
        ...

    def _ensure_data_robustness(self, vals) -> pd.Series:
        """Coerce a column/array to the object's clean, numeric contract."""
        ...


class SizeResolvedData(AerosolData, Protocol):
    """Adds the size-resolved (2-D) state the size/PSD/correction mixins use."""

    _sizebin_headers: list

    @property
    def bin_mids(self) -> NDArray[np.float64]: ...
    @property
    def bin_edges(self) -> NDArray[np.float64]: ...
    @property
    def density(self) -> float: ...
