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

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        """Particle size bin edges in nanometers.

        Returns:
            numpy.ndarray: One-dimensional array of bin edge diameters in
            nanometers (dtype ``float64``). Length is ``n + 1`` when there are
            ``n`` size bins. A copy is returned so callers cannot mutate the
            internal metadata.
        """
        # ensure an array of floats; .copy() so callers can't mutate your internal state
        return np.asarray(self._meta["bin_edges"], dtype=np.float64).copy()

    @property
    def bin_mids(self) -> NDArray[np.float64]:
        """Particle size bin midpoints in nanometers.

        Returns:
            numpy.ndarray: One-dimensional array of bin midpoint diameters in
            nanometers (dtype ``float64``). Length is ``n`` for ``n`` size
            bins. A copy is returned so callers cannot mutate the internal
            metadata.
        """
        return np.asarray(self._meta["bin_mids"], dtype=np.float64).copy()

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
        return [str(x) for x in self.bin_mids]

    def _ensure_data_robustness(self, vals) -> pd.Series:
        """Validity mask from the original object (keeps alignment with self.time)

        This returns a cleaned series, so that no new data is generated,
        where before the total_conc was NaN.
        Args:
            vals (np.array):
                array of data structured as a column of data from either data
                extra data.
        Returns:
            pd.Series: Time series of the requested Pₓ metric, indexed by
            :attr:`time`. Empty or invalid time steps (where
            :attr:`total_concentration` is NaN) are returned as NaN.
        """

        valid_mask = self.total_concentration.notna()
        series = pd.Series(vals, index=self.time)

        return series.where(valid_mask, np.nan)
