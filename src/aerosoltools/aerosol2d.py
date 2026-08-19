"""Public 2D (size-resolved) aerosol class (:class:`Aerosol2D`).

Holds the data model (``__init__`` and the size-axis properties) plus the
fundamental data-robustness helper. The heavy behaviour — basis
conversions, Pₓ fractions, lognormal fitting, corrections, plotting and
summaries — lives in topic mixins under :mod:`aerosoltools._core`, which
this class composes. The public API is unchanged.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._core.corrections import CorrectionMixin
from ._core.fitting import FitMixin
from ._core.fractions import FractionMixin
from ._core.plotting2d import Plot2DMixin
from ._core.size_distribution import SizeConversionMixin
from ._core.statistics2d import Summary2DMixin
from .aerosol1d import Aerosol1D

try:
    from typing import override  # Python 3.12+
except ImportError:  # pragma: no cover - typing_extensions fallback
    from typing_extensions import override  # noqa: F401


class Aerosol2D(
    SizeConversionMixin,
    FractionMixin,
    FitMixin,
    CorrectionMixin,
    Plot2DMixin,
    Summary2DMixin,
    Aerosol1D,
):
    """
    A class for managing time-resolved, size-distributed aerosol data.

    This class extends `Aerosol1D` to handle datasets that contain particle
    size distributions (e.g., number, mass, or surface area concentration
    across particle size bins). It supports transformation between physical
    representations (dN, dS, dV, dW), visualization, activity segmentation,
    and summary statistics including PM values and particle size metrics.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A DataFrame containing the data to load. The first column should
        contain time stamps or be the DataFrame index. The second column should
        be the total concentration. All remaining columns must represent
        concentration values in size bins with bin midpoints as column headers.

    Notes
    -----
    All data handling is done with `pandas`. Input DataFrames are expected to
    have particle size bin midpoints as column headers, and the class assumes
    these are numeric and represent diameters in nanometers.
    """

    def __init__(self, dataframe):
        super().__init__(dataframe)

    def _frozen_bins(self, key: str) -> NDArray[np.float64]:
        """Cached read-only float64 view of ``_meta[key]`` (bin_mids/bin_edges).

        The size axis only ever changes by *reassigning* a new array in
        ``_meta`` (load, rebin, density recompute, combine) — never by mutating
        one in place — so a cheap identity + length + endpoint signature detects
        any change and rebuilds the cache. Public :attr:`bin_mids`/
        :attr:`bin_edges` copy this before returning (callers may mutate); the
        frozen array also gives :attr:`_sizebin_headers` a stable identity to
        memoise against.
        """
        src = self._meta[key]
        sig = (id(src), len(src), float(src[0]), float(src[-1]))
        cache = self.__dict__.setdefault("_bins_cache", {})
        entry = cache.get(key)
        if entry is None or entry[0] != sig:
            arr = np.asarray(src, dtype=np.float64)
            if arr is src:  # never freeze the caller's stored array in place
                arr = arr.copy()
            arr.setflags(write=False)
            entry = (sig, arr)
            cache[key] = entry
        return entry[1]

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        """Particle size bin edges in nanometers.

        Returns:
            numpy.ndarray: One-dimensional array of bin edge diameters in
            nanometers (dtype ``float64``). Length is ``n + 1`` when there are
            ``n`` size bins. A copy is returned so callers cannot mutate the
            internal metadata.
        """
        # .copy() so callers can't mutate the cached internal array
        return self._frozen_bins("bin_edges").copy()

    @property
    def bin_mids(self) -> NDArray[np.float64]:
        """Particle size bin midpoints in nanometers.

        Returns:
            numpy.ndarray: One-dimensional array of bin midpoint diameters in
            nanometers (dtype ``float64``). Length is ``n`` for ``n`` size
            bins. A copy is returned so callers cannot mutate the internal
            metadata.
        """
        return self._frozen_bins("bin_mids").copy()

    @property
    def density(self) -> float:
        """Assumed particle density in g/cm³.

        Returns:
            float: Particle density used for conversions between number, volume,
            surface area, and mass distributions. Falls back to the value
            stored in the metadata (typically set at load time or via
            :meth:`set_density`). Defaults to 1.0 g/cm³ if not explicitly set in
            metadata.
        """
        return float(self._meta.get("density", 1.0))

    @property
    def metadata(self) -> dict:
        """Metadata associated with the size-resolved dataset.

        Returns:
            dict: Dictionary of metadata extracted or defined for this object,
            including bin edges/mids, units, data type (dN/dS/dV/dM),
            instrument information, density, and any additional fields stored
            in ``self._meta``.
        """
        return self._meta

    @property
    def size_data(self) -> pd.DataFrame:
        """Size-bin concentration data.

        Returns:
            pandas.DataFrame: Subset of :attr:`data` containing only the
            columns that represent size-resolved concentration values, ordered
            according to :attr:`bin_mids` (via :attr:`_sizebin_headers`). Each
            column corresponds to a size bin, and each row to a time stamp.
        """
        return self.data.loc[:, self._sizebin_headers]

    @property
    def _sizebin_headers(self) -> list[str]:
        """Column labels for size-bin concentration data.

        Returns:
            list[str]: Column names used in :attr:`data` for the size-bin
            distribution, derived from :attr:`bin_mids` (converted to strings).
            These headers define which columns are treated as the size
            distribution in methods such as :meth:`convert_to_number_concentration`,
            :meth:`convert_to_mass_concentration`, and plotting utilities.
        """
        # Memoise the (repeatedly requested) header list against the frozen
        # bin_mids identity, so ``size_data`` doesn't rebuild it on every access.
        mids = self._frozen_bins("bin_mids")
        cache = self.__dict__.get("_headers_cache")
        if cache is None or cache[0] is not mids:
            cache = (mids, [str(x) for x in mids])
            self.__dict__["_headers_cache"] = cache
        return cache[1]
