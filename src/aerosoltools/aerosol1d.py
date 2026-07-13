"""Public 1D aerosol time-series class (:class:`Aerosol1D`).

This module holds the data model — ``__init__`` and the read-only properties —
plus a couple of fundamental helpers. The heavier behaviour is grouped by topic
into mixins under :mod:`aerosoltools._core` (time operations, activities, summary
statistics and plotting) which this class composes, so the file stays readable
while the public API is unchanged: every method is still available on
``Aerosol1D``.
"""

from __future__ import annotations

import copy
from typing import Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._core.activities import ActivityMixin
from ._core.decay import DecayFitMixin
from ._core.plotting import Plot1DMixin
from ._core.statistics import SummaryMixin
from ._core.time_ops import TimeOpsMixin

# Matplotlib styling applied on import, kept here so that importing the package
# preserves the same default plot appearance it has always had.
params = {
    "legend.fontsize": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.figsize": (19, 10),
}
plt.rcParams.update(params)


class Aerosol1D(TimeOpsMixin, ActivityMixin, SummaryMixin, Plot1DMixin, DecayFitMixin):
    """Handle 1D aerosol time-series measurements.

    This class manages time-indexed aerosol concentration data (for example
    total particle number or mass concentration) and provides utilities for
    resampling, smoothing, marking activity segments, cropping, shifting,
    summarizing, and plotting.

    It is intended for pre- and post-processing of aerosol datasets collected
    with portable or stationary particle counters that report a single
    concentration value per time step.

    Args:
        dataframe (pandas.DataFrame): Input data. If the index is not a
            :class:`pandas.DatetimeIndex`, the first column is interpreted as
            timestamps and converted using :func:`pandas.to_datetime`, then
            set as the index. The first data column is assumed to represent
            total particle concentration.

    Attributes:
        data (pandas.DataFrame): Main data frame containing the time index,
            the total concentration column, and any activity masks (e.g.
            ``"All data"``, ``"Peak"``, user-defined activities).
        extra_data (pandas.DataFrame): Additional columns extracted from the
            raw file that are not considered core aerosol measurements
            (e.g. environmental or meta signals).
        activities (list[str]): List of defined activity labels for which
            boolean mask columns exist in :attr:`data`.
        activity_periods (dict[str, list[tuple[pandas.Timestamp, pandas.Timestamp]]]):
            Mapping from activity name to a list of (start, end) time intervals
            where that activity is considered active.
        metadata (dict): Dictionary of metadata such as unit, data type,
            instrument type and serial number. Internally stored in
            ``self._meta``.

    Notes:
        Users are encouraged to interact with the class via its public
        properties and methods (e.g. :attr:`data`, :meth:`mark_activities`,
        :meth:`timerebin`, :meth:`plot_total_conc`) rather than modifying
        private attributes directly.
    """

    def __init__(self, dataframe):
        self._meta = {}
        self._extra_data = pd.DataFrame([])
        self._activities = []
        self._activity_periods = {}

        # Automatically handle timestamp column
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            timestamp_col = dataframe.columns[0]
            dataframe.loc[:, timestamp_col] = pd.to_datetime(dataframe[timestamp_col])
            dataframe.set_index(timestamp_col, inplace=True)

        # Ensure there is a meaningful column name
        if dataframe.columns[0] is None or dataframe.columns[0] == 0:
            dataframe.columns = ["Total_conc"]

        self._data = dataframe.copy()
        self._raw_data = dataframe.copy()
        self._raw_extra_data = pd.DataFrame([])
        self._data.loc[:, "All data"] = True
        self._activities.append("All data")
        self._activity_periods["All data"] = [(self.time.min(), self.time.max())]

    ###########################################################################
    """############################ Properties #############################"""
    ###########################################################################

    @property
    def activities(self) -> List[str]:
        """Names of all defined activity labels.

        Returns:
            list[str]: Activity names for which a boolean mask column exists
            in :attr:`data` (for example ``"All data"``, ``"Peak"``, or
            user-defined labels created via :meth:`mark_activities`).
        """
        return self._activities

    @property
    def activity_periods(self) -> Dict:
        """Time periods associated with each activity label.

        Returns:
            dict: Mapping from activity name (str) to a list of
            ``(start, end)`` timestamp tuples, where each tuple defines a
            contiguous period during which the activity is active.
        """
        return self._activity_periods

    @property
    def data(self) -> pd.DataFrame:
        """Main data frame with time index and measurement columns.

        Returns:
            pandas.DataFrame: The full data table, including the time index,
            the primary measurement columns (for example total concentration),
            and any boolean activity mask columns.
        """
        return self._data

    @property
    def dtype(self) -> str:
        """Data type descriptor for the primary measurements.

        Returns:
            str: A short description of the data type, for example
            ``"dN"`` (number concentration), ``"dM"`` (mass concentration),
            or ``"dN/dlogDp"`` when normalized. Falls back to
            ``"Unknown dtype"`` if not set in the metadata.
        """
        return self._meta.get("dtype", "Unknown dtype")

    @property
    def extra_data(self) -> pd.DataFrame:
        """Additional non-core data columns.

        Returns:
            pandas.DataFrame: Data frame containing extra columns that were
            extracted from the raw data file but are not considered part of the
            core aerosol time series (for example environmental or metadata
            channels). The index is aligned with :attr:`data`.
        """
        return self._extra_data

    @property
    def instrument(self) -> str:
        """Instrument used to acquire the measurements.

        Returns:
            str: Instrument name or description (for example model/type).
            Falls back to ``"Unknown instrument"`` if not set in the metadata.
        """
        return self._meta.get("instrument", "Unknown instrument")

    @property
    def measurement(self):
        """Human-readable name of the measured quantity, if known.

        Returns:
            The label describing *what* the primary series represents — for
            example ``"Cl₂"`` or ``"NO₂"`` for a gas monitor — as set by the
            loader in ``metadata["measurement"]``. Returns ``None`` when no
            explicit label was stored (in which case callers fall back to a
            generic name such as ``"Total concentration"`` or the channel name).
            This is distinct from :attr:`dtype` (``dN``/``dM``/…) and
            :attr:`unit`; it names the quantity rather than its basis or units.
        """
        return self._meta.get("measurement", None)

    @property
    def metadata(self) -> Dict:
        """Metadata associated with the dataset.

        Returns:
            dict: Dictionary containing metadata such as unit, data type,
            instrument type, and serial number. Internally stored in
            ``self._meta``.
        """
        return self._meta

    @property
    def original_data(self) -> pd.DataFrame:
        """Unmodified original main data frame.

        Returns:
            pandas.DataFrame: A copy of the raw data as it was immediately after
            loading and initial normalization of the time index and column
            names, before any further processing steps (such as cropping,
            smoothing, or rebinning).
        """
        return self._raw_data

    @property
    def original_extra_data(self) -> pd.DataFrame:
        """Unmodified original extra-data frame.

        Returns:
            pandas.DataFrame: A copy of the raw extra-data table (if any) before
            any processing steps. This will typically be empty if no extra data
            were extracted at load time.
        """
        return self._raw_extra_data

    @property
    def serial_number(self) -> str:
        """Instrument serial number.

        Returns:
            str: Serial number of the instrument, if available in the metadata.
            Falls back to ``"Unknown serial number"`` if not set.
        """
        return self._meta.get("serial_number", "Unknown serial number")

    @property
    def time(self) -> pd.DatetimeIndex:
        """Time index of the measurements.

        Returns:
            pandas.DatetimeIndex: The time stamps corresponding to each row in
            :attr:`data`. This is the index of the main data frame.
        """
        return cast(pd.DatetimeIndex, self._data.index)

    @property
    def total_concentration(self) -> pd.Series:
        """Total aerosol concentration time series.

        Returns:
            pandas.Series: The total concentration over time. If a column named
            ``"Total_conc"`` exists in :attr:`data`, that column is
            returned; otherwise the first data column is used as a fallback.
        """
        if "Total_conc" in self._data.columns:
            return self._data["Total_conc"]
        else:
            return self._data.iloc[:, 0]

    @property
    def _primary(self) -> pd.Series:
        """The dataset's primary channel (package-internal, not public API).

        This is the generic hook that reusable operations (e.g. default peak
        detection, default summaries) and the GUI use when they just need "the
        main channel of this dataset", independent of what that channel *is*.

        For particle classes it is the :attr:`total_concentration`. Non-particle
        classes (gases, black carbon, environmental, LDSA-only) override this to
        return their own primary channel, since ``total_concentration`` does not
        apply to them.

        Returns:
            pandas.Series: The primary measurement series.
        """
        return self.total_concentration

    @property
    def unit(self) -> str:
        """Measurement unit for the primary data.

        Returns:
            str: Unit string, for example ``"#/cm³"`` for number concentration
            or ``"µg/m³"`` for mass concentration. Falls back to
            ``"Unknown unit"`` if not set in the metadata.
        """
        return self._meta.get("unit", "Unknown unit")

    ###########################################################################
    """########################### Core helpers ###########################"""
    ###########################################################################

    def _ensure_data_robustness(self, vals) -> pd.Series:
        """Validity mask from the original object (keeps alignment with self.time)

        This returns a cleaned serires, so that no new data is generated,
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

    ###########################################################################

    ###########################################################################
    """############################# Functions #############################"""
    ###########################################################################

    def calibrate(self, m: float = 1, b: float = 0):
        """
        Apply a correction to the total conc and mark the data as calibrated
        by a linear function

        Parameters
        ----------
        m : float
            The calibration value to be multiplied to the data for correction.
        b : float
            A constant offset to be removed. By default is zero and should be
            used cautionsly.

        Returns
        -------
        None

        """
        self._data["Total_conc"] = self._ensure_data_robustness(
            self._data["Total_conc"] * m + b
        )

        self._meta["calibrated"] = {"m": m, "b": b}

    ###########################################################################

    def copy_self(self):
        """Description:
            Create and return a deep copy of the aerosol time-series object.

        Returns:
            Aerosol1D: A new object with independent copies of all data,
                extra_data, metadata, and activity definitions.

        Notes:
            Detailed description:
                The returned object is completely detached from the original:
                changing masks, cropping, rebinning, or smoothing on the copy
                does not affect the source instance, and vice versa. Both
                the original and the copy retain their own raw_data and
                raw_extra_data snapshots.

        Examples:
            Use this when you want to experiment with processing steps
            without changing your original dataset:

            .. code-block:: python

                cpc_raw = cpc_data
                cpc_proc = cpc_data.copy_self()
                cpc_proc.timerebin("1min").timesmooth(window=11)
        """
        return copy.deepcopy(self)
