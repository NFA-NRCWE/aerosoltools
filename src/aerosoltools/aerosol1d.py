from __future__ import annotations

import copy
import datetime as dt
import os as os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tabulate import tabulate

params = {
    "legend.fontsize": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.figsize": (19, 10),
}
plt.rcParams.update(params)
DateLike = Union[dt.datetime, pd.Timestamp]


# aerosol1d class definition
class Aerosol1D:
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
        if "Total Concentration" in self._data.columns:
            return self._data["Total_conc"]
        else:
            return self._data.iloc[:, 0]

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
    """### Shared helpers for summarize_activities / summarize_exposure  ###"""
    ###########################################################################

    @staticmethod
    def _format_hhmm(total_minutes: float) -> str:
        """Format a total duration in minutes as ``"HH:MM"``.

        Args:
            total_minutes (float): Total duration in minutes. May be any
                finite non-negative value. Non-finite values (NaN, inf) are
                treated as zero and returned as ``"00:00"``.

        Returns:
            str: A zero-padded string of the form ``"HH:MM"``.
        """
        if not np.isfinite(total_minutes):
            return "00:00"
        td = pd.to_timedelta(total_minutes, unit="m")
        total_sec = int(td.total_seconds())
        h = total_sec // 3600
        m = (total_sec % 3600) // 60
        return f"{h:02d}:{m:02d}"

    ###########################################################################

    def _dt_minutes(self) -> pd.Series:
        """Compute per-step durations in minutes for the main time axis.

        Returns:
            pandas.Series: A 1D series indexed by :attr:`time` where each
            element represents the duration (in minutes) associated with
            that time step.
        """
        t = self.time
        if len(t) == 1:
            return pd.Series([1.0], index=t)

        dt = (t.to_series().shift(-1) - t.to_series()).dt.total_seconds().abs()
        med = float(np.nanmedian(dt)) if np.isfinite(np.nanmedian(dt)) else 60.0
        dt.iloc[-1] = med
        return dt / 60.0

    ###########################################################################

    def _get_metric_series(self, metric_name: str) -> tuple[pd.Series, str]:
        """Return a time series and unit for a named 1D metric.

        This base implementation supports:

            * "PNC": maps to :attr:`total_concentration` with :attr:`unit`.
            * Any other name: looked up as a numeric column in :attr:`data`
              or :attr:`extra_data`.

        Subclasses (e.g. Aerosol2D) override this to support additional
        derived metrics such as MASS or Pₓ.

        Args:
            metric_name (str): Name of the metric, case-sensitive for
                data/extra_data lookup, case-insensitive for "PNC".

        Returns:
            tuple[pandas.Series, str]: (series, unit) where series is
            indexed by :attr:`time`.

        Raises:
            ValueError: If the metric (other than "PNC") cannot be found
                in data or extra_data.
        """
        mu = metric_name.upper()

        if mu == "PNC":
            series = self.total_concentration.astype(float)
            return series, self.unit

        # Look up in main data first, then extra_data
        if metric_name in self._data.select_dtypes(exclude="bool").columns:
            series = self._data[metric_name].astype(float)
        elif metric_name in self._extra_data.select_dtypes(exclude="bool").columns:
            series = self._extra_data[metric_name].astype(float)
        else:
            raise ValueError(
                f"Metric '{metric_name}' not found in main data or extra_data."
            )

        # For 1D, we do not have per-metric units, so we reuse self.unit.
        return series, self.unit

    ###########################################################################
    """############################# Functions #############################"""
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

    ###########################################################################

    def get_activity_data(self, activity_name):
        """Description:
            Return main data restricted to a given activity period.

        Args:
            activity_name (str): Name of the activity/boolean mask column
                to use (for example "All data", "Peak", or a user-defined
                label created via mark_activities).

        Returns:
            pandas.DataFrame: Copy of the main data for time steps where the
                selected activity is True, with all activity mask columns
                removed. The index is the filtered DatetimeIndex.

        Raises:
            ValueError: If activity_name is not present in self.activities.
                Ensure you call mark_activities or Peak_finder first, or
                check available labels via the activities property.

        Notes:
            Detailed description:
                This method filters self.data using the chosen activity mask
                and then drops all boolean activity columns so that the result
                only contains “measurement-like” columns (for example
                Total_conc and any other numeric channels). The returned
                DataFrame is safe to modify without affecting the internal
                storage of the Aerosol1D object.

            Theory:
                The method performs a simple time-based selection of rows
                flagged by an activity mask.

        Examples:
            Extract a clean dataset for a specific task before further
            analysis:

            .. code-block:: python

                task_df = data.get_activity_data("Task")
                task_df["Total_conc"].plot()
        """

        if activity_name not in self.activities:
            raise ValueError(
                f"Activity '{activity_name}' not found in available activities: {self.activities}"
            )

        return self._data[self._data[activity_name]].drop(columns=self.activities)

    ###########################################################################

    def get_activity_extra_data(self, activity_name):
        """Description:
            Return extra_data restricted to a given activity period.

        Args:
            activity_name (str): Name of the activity/boolean mask column
                in self.data to use as a selector.

        Returns:
            pandas.DataFrame: Subset of extra_data aligned to the time
                periods where the chosen activity is True. If extra_data is
                empty, an empty DataFrame is returned.

        Raises:
            ValueError: If activity_name is not present in self.activities.

        Notes:
            Detailed description:
                The method uses the activity mask stored in self.data but
                returns rows from self.extra_data. This is useful to inspect
                auxiliary channels (for example temperature, pressure, or
                instrument status signals) corresponding to specific tasks
                or events, without mixing them with the primary aerosol
                time series.

        Examples:
            Correlate environmental parameters with a task:

            .. code-block:: python

                env_task = data.get_activity_extra_data("Task")
                env_task[["Temperature", "RelativeHumidity"]].plot()
        """

        if activity_name not in self.activities:
            raise ValueError(
                f"Activity '{activity_name}' not found in available activities: {self.activities}"
            )

        return self.extra_data[self._data[activity_name]]

    ###########################################################################

    def mark_activities(self, activity_periods, mode: str = "union"):
        """Description:
            Define or update boolean activity masks on the time axis.

        Args:
            activity_periods (dict): Mapping from activity name (str) to
                one of:

                    - a (start, end) tuple,
                    - a list of (start, end) tuples, or
                    - None (no active periods).

                Each start/end can be a pandas.Timestamp, datetime, or any
                string understood by pandas.to_datetime.
            mode (str): How to combine new periods with existing masks of
                the same name. One of:

                    - "union": existing OR new periods (default),
                    - "replace": overwrite any existing mask,
                    - "intersection": keep only overlapping periods.

        Returns:
            None: The object is modified in place by adding/updating
                boolean columns in self.data and entries in
                self.activity_periods.

        Raises:
            ValueError: If mode is not "union", "replace", or "intersection".
                Check the spelling of mode and choose among the supported
                options.

        Notes:
            Detailed description:
                For each activity key, a boolean Series is built over the
                current time index. All timestamps falling between any of
                the supplied (start, end) pairs (inclusive) are set to True,
                and others to False. Depending on mode, this new mask is
                merged with any existing column of the same name. The
                normalized list of (start, end) periods is stored in
                self.activity_periods[activity].

            Theory:
                The method implements generic time segmentation. Activities
                can represent tasks, locations, process states, or anything
                else defined by the user.

        Examples:
            Mark a measurement into “Task” and “Background” segments:

            .. code-block:: python

                periods = {
                    "Task": [("2025-01-24 09:00", "2025-01-24 11:00"),
                                ("2025-01-24 12:30", "2025-01-24 14:00")
                                ],
                    "Background": [("2025-01-24 08:30", "2025-01-24 09:00")]
                }
                data.mark_activities(periods, mode="replace")
                data.plot_total_conc(mark_activities=True)
        """

        if mode not in {"union", "replace", "intersection"}:
            raise ValueError(
                f"Invalid mode {mode!r}. "
                "Expected one of {'union', 'replace', 'intersection'}."
            )

        for activity, periods in activity_periods.items():
            # Initialize column with False on the timeline
            col = pd.Series(False, index=self.time, dtype=bool)

            # Normalize periods to a list of (start, end)
            if isinstance(periods, tuple) and len(periods) == 2:
                periods = [periods]
            elif periods is None:
                periods = []

            # Mark True within each period
            for start, end in periods:
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                mask = (self.time >= start_ts) & (self.time <= end_ts)
                col[mask] = True

            # Update or create column
            if activity in self._data.columns:
                existing = (
                    self._data[activity]
                    .reindex(self.time)  # align to current index
                    .fillna(False)
                    .astype(bool)
                )
                if mode == "replace":
                    updated = col
                elif mode == "intersection":
                    updated = existing & col
                else:  # "union"
                    updated = existing | col
                # assign back (preserves position if column already exists)
                self._data.loc[:, activity] = updated.values
            else:
                # new column
                self._data[activity] = col

            # Track metadata
            if activity not in self._activities:
                self._activities.append(activity)
            self._activity_periods[activity] = periods

    ###########################################################################

    def Peak_finder(
        self,
        window: int = 15,
        ratio: float = 2.5,
        method: str = "median",
        specific_data: str = "",
    ):
        """Description:
            Detect peaks and mark them as an activity labeled 'Peak'.

        Args:
            window (int): Rolling window size in number of samples used to
                compute the local baseline and spread. Defaults to 15.
            ratio (float): Threshold factor on the rolling standard
                deviation. A time step is flagged as peak when
                value - baseline > ratio * rolling_std. Defaults to 2.5.
            method (str): Aggregation used as the rolling baseline, one of
                "mean", "median", "sum", "min", "max". Defaults to "median".
            specific_data (str): Optional column name to use for peak
                detection. If empty, the total_concentration time series is
                used. Otherwise, the name is looked up first in self.data
                (numeric columns) and then in extra_data.

        Returns:
            None: The method adds/updates a boolean "Peak" column in
                self.data and updates self.activity_periods["Peak"].

        Raises:
            ValueError: If method is not one of the supported names. Choose
                one of "mean", "median", "sum", "min", "max".
            ValueError: If specific_data is not found as a numeric column
                in either self.data or extra_data. Check column names via
                data.columns or extra_data.columns.

        Notes:
            Detailed description:
                The method computes a rolling baseline and rolling standard
                deviation over the chosen data series. Points where the
                deviation from the baseline exceeds ratio times the rolling
                standard deviation are marked as peaks. Contiguous True
                segments in this mask are converted to (start, end) time
                intervals and stored as the "Peak" activity.

            Theory:
                This is a simple statistical peak detector based on local
                outlier detection relative to a moving baseline and spread.
                It highlights periods where the signal is unusually high
                compared to its recent history.

        Examples:
            Automatically tag high-exposure episodes:

            .. code-block:: python

                data.Peak_finder(window=31, ratio=3.0)
                data.plot_total_conc(mark_activities=["Peak"])
                peak_df = data.get_activity_data("Peak")
        """

        if method not in ["mean", "median", "sum", "min", "max"]:
            raise ValueError(
                "Invalid method. Choose from 'mean', 'median', 'sum', 'min', 'max'."
            )

        window = int(window)
        Data_return = self.copy_self()

        if specific_data == "":
            Data_return = Data_return.total_concentration
        else:
            if specific_data in self._data.select_dtypes(exclude="bool").columns:
                Data_return = Data_return._data[specific_data]

            elif (
                specific_data in self._extra_data.select_dtypes(exclude="bool").columns
            ):
                Data_return = Data_return._extra_data[specific_data]
            else:
                raise ValueError(
                    f"Invalid data title. No column named {specific_data} can be found."
                )

        Test = getattr(
            Data_return.rolling(window=window, center=True, min_periods=2), method
        )()
        Std = getattr(
            Data_return.rolling(window=window, center=True, min_periods=2), "std"
        )()

        # Change Med array from median value to the difference between median and actual value.
        mask = Data_return - Test > Std * ratio

        peak_col = {"Peak": mask}
        # Track metadata
        if "Peak" not in self._activities:
            self._activities.append("Peak")
            self._data = pd.concat([self.data, pd.DataFrame(peak_col)], axis=1)
        else:
            self._data["Peak"] = mask

        periods = list(
            zip(
                mask.index[
                    mask & ~mask.shift(fill_value=False)
                ],  # starts: True preceded by False/NaN
                mask.index[
                    mask & ~mask.shift(-1, fill_value=False)
                ],  # ends: True followed by False/NaN
            )
        )

        self._activity_periods["Peak"] = periods

    ###########################################################################

    def plot_total_conc(
        self,
        ax: Axes | None = None,
        mark_activities: bool | Sequence[str] = False,
    ) -> tuple[Figure, Axes]:
        """Description:
            Plot the time series of total aerosol concentration.

        Args:
            ax (matplotlib.axes.Axes | None): Optional axis to draw the plot on.
                If None, a new figure and axes are created.
            mark_activities (bool | Sequence[str]): Controls highlighting
                of activity periods:

                    - False: no highlighting.
                    - True: shade all activities except "All data".
                    - sequence of str: shade only the named activities.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
                figure and axes containing the plot.

        Raises:
            None: Any errors usually stem from invalid Matplotlib axes or
                malformed time indices.

        Notes:
            Detailed description:
                The method draws total_concentration versus time with a
                grid and labels derived from dtype and unit. The x-axis is
                formatted using Matplotlib's concise date formatter. If
                mark_activities is enabled, each selected activity is drawn
                as semi-transparent vertical spans covering its active
                periods, with a legend entry per activity.

        Examples:
            Inspect an instrument time series with highlighted tasks:

            .. code-block:: python

                fig, ax = data.plot_total_conc(mark_activities=True)
                fig.savefig("total_concentration.png", dpi=150)
        """

        new_fig_created = False

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            new_fig_created = True
        else:
            fig = ax.figure

        # Plot main data
        ax.plot(self.time, self.total_concentration, linestyle="-")

        # Format x-axis
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel("Time")
        if "/" in self.dtype:
            total_conc_dtype = self.dtype.split("/")[0]
            ax.set_ylabel(f"{total_conc_dtype}, {self.unit}")
        else:
            ax.set_ylabel(f"{self.dtype}, {self.unit}")
        ax.grid(True)

        # Highlight activities
        if mark_activities and hasattr(self, "_activity_periods"):
            # Exclude "All data" unless explicitly requested
            all_activities = sorted(self._activity_periods.keys())
            color_map = plt.colormaps.get_cmap("gist_ncar")
            activity_colors = {
                activity: color_map(i / max(1, len(all_activities)))
                for i, activity in enumerate(all_activities)
            }

            if mark_activities is True:
                selected_activities = [a for a in all_activities if a != "All data"]
            elif isinstance(mark_activities, list):
                selected_activities = [
                    a for a in mark_activities if a in self._activity_periods
                ]
            else:
                selected_activities = []

            for activity in selected_activities:
                color = activity_colors[activity]
                first = True
                for start, end in self._activity_periods[activity]:
                    ax.axvspan(
                        cast(float, mdates.date2num(pd.Timestamp(start))),
                        cast(float, mdates.date2num(pd.Timestamp(end))),
                        color=color,
                        alpha=0.3,
                        label=activity if first else None,
                        zorder=3,
                    )
                    first = False
            # Clip x-axis to actual data range
            left = float(mdates.date2num(self.time.min()))
            right = float(mdates.date2num(self.time.max()))
            ax.set_xlim(left, right)
            ax.legend()

        if new_fig_created:
            fig.tight_layout()  # type: ignore

        return fig, ax  # type: ignore

    ###########################################################################

    def summarize_activities(
        self,
        filename: Optional[str] = None,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Description:
            Summarize 1D aerosol metrics per activity.

        Args:
            filename (str | None): Optional path to a CSV or Excel file.
                If provided, the summary table is appended to the file
                (creating it if it does not exist). If None, nothing is
                written to disk.
            metrics (list[str] | None): List of metric names to summarize.
                If None, a default set is used: ["PNC"].

                * "PNC" refers to total_concentration.
                * Any other name is looked up in data or extra_data
                  (numeric columns only).

        Returns:
            pandas.DataFrame: Summary table with one row per activity and
            columns:

                * "Segment"
                * "Duration (HH:MM)"
                * For each metric M: "M [unit] mean" and "M [unit] std".

        Raises:
            ValueError: If a requested metric (other than "PNC") cannot
                be found in data or extra_data.

        Notes:
            The per-activity means and standard deviations are simple
            sample statistics over the selected time steps. Durations are
            based on the actual sampling intervals (via _dt_minutes).

        Examples:

            .. code-block:: python

                data.summarize_activities()
        """

        # Defaults: only PNC if nothing else is specified
        if metrics is None:
            metrics = ["PNC"]

        # Precompute dt in minutes for the whole series
        dt_mins = self._dt_minutes()

        # Prepare series and units for all requested metrics
        metric_series: dict[str, pd.Series] = {}
        metric_units: dict[str, str] = {}
        for name in metrics:
            series, unit = self._get_metric_series(name)
            metric_series[name] = series
            metric_units[name] = unit

        rows: list[list[Any]] = []

        for activity in self.activities:
            mask = self.data[activity].astype(bool)
            if mask.sum() == 0:
                continue

            dt_seg = dt_mins.loc[mask]
            duration_min = float(dt_seg.sum())
            duration_hhmm = self._format_hhmm(duration_min)

            row: list[Any] = [activity, duration_hhmm]

            for name in metrics:
                s = metric_series[name].loc[mask]
                if s.empty or not np.isfinite(s).any():
                    mean_val = float("nan")
                    std_val = float("nan")
                else:
                    mean_val = float(s.mean())
                    std_val = float(s.std())
                row += [round(mean_val, 3), round(std_val, 3)]

            rows.append(row)

        # Build column labels
        columns: list[str] = ["Segment", "Duration (HH:MM)"]
        for name in metrics:
            unit = metric_units[name]
            label = f"{name} [{unit}]"
            columns += [f"{label} mean", f"{label} std"]

        summary = pd.DataFrame(rows, columns=columns)

        # Console output (transposed for readability)
        if not summary.empty:
            summary_t = summary.set_index("Segment").T
            print("\nSummary of aerosol metrics per activity:\n")
            print(tabulate(summary_t, headers="keys", tablefmt="pretty", floatfmt=".3f"))  # type: ignore

        # Optional file output (append if exists)
        if filename:
            fname = str(filename)
            lower = fname.lower()
            if lower.endswith(".csv"):
                try:
                    existing = pd.read_csv(fname)
                    combined = pd.concat([existing, summary], ignore_index=True)
                except FileNotFoundError:
                    combined = summary
                combined.to_csv(fname, index=False)
            elif lower.endswith((".xls", ".xlsx")):
                try:
                    existing = pd.read_excel(fname)
                    combined = pd.concat([existing, summary], ignore_index=True)
                except FileNotFoundError:
                    combined = summary
                combined.to_excel(fname, index=False)
            else:
                # Fallback: treat as CSV
                try:
                    existing = pd.read_csv(fname)
                    combined = pd.concat([existing, summary], ignore_index=True)
                except FileNotFoundError:
                    combined = summary
                combined.to_csv(fname, index=False)

            print(f"\nActivity summary appended to: {filename}")

        return summary

    ###########################################################################

    def summarize_exposure(
        self,
        metric: str = "PNC",
        background: Union[float, str, None] = None,
        exposure_hours: Optional[float] = None,
        short_limit: float = 1.0,
        long_limit: float = 1.0,
        short_window: str = "15min",
        twa_window: str = "8h",
        peak_ratio: float = 2.5,
        filename: Optional[str] = None,
        activities: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Description:
            Summarize exposure metrics for one 1D metric across activities.

        Args:
            metric (str): Exposure metric name based on the underlying
                1D time series. Default is "PNC", corresponding to the
                total particle number concentration. Supported forms:

                    * "PNC": total number concentration, mapped to
                      :attr:`total_concentration`.
                    * Any other string: interpreted as the name of a
                      numeric column in :attr:`data` or :attr:`extra_data`
                      (case-sensitive). For example, "MASS" or "TEMP"
                      if such columns exist.

            background (float | str | None): Background level used when
                computing the time-weighted average over ``twa_window``.
                The same background level is used for all activities in
                the output. Possible entries are:

                    * None: assume zero background.
                    * float: constant background level in metric units.
                    * str: name of an activity; the TWA of ``metric`` over
                      that activity is used as background.

            exposure_hours (float | None): Assumed duration of exposure for
                each activity, in hours, when embedding it into the TWA
                window. If None, the measured activity duration is used for
                that activity. If a positive value is given, the same
                exposure duration is applied for all activities.

            short_limit (float): Short-term concentration limit in metric
                units (for example a 15-min STEL). This value is reported
                in the output as ``"STEL [unit]"``.

            long_limit (float): Long-term concentration limit in metric
                units (for example an 8-h OEL). This value is reported in
                the output as ``"Exposure limit [unit]"``.

            short_window (str): Rolling window used for short-term (STEL)
                evaluation, given as a pandas offset string (for example
                "15min"). This is reported as ``"STEL window [offset]"`` and
                used to construct a time-based rolling mean.

            twa_window (str): Total duration of the TWA window as a pandas
                offset string (for example "8h"). This is reported as
                ``"TWA window [offset]"`` and used to embed the segment
                exposure into a reference window (for example an 8-h shift).

            peak_ratio (float): Factor used in peak detection; peaks are
                flagged when the metric exceeds::

                    baseline + peak_ratio * rolling_std

                where ``baseline`` is a rolling median and ``rolling_std`` is
                a rolling standard deviation over a short window.

            filename (str | None): Optional path to a CSV/Excel file to
                which the non-transposed result rows are appended. If the
                file exists, rows are appended; otherwise the file is
                created with a header. Supported extensions are ".csv",
                ".xls", ".xlsx".

            activities (Sequence[str] | None): Activities to summarize.
                If None (default), all defined activities in
                :attr:`activities` are summarized (for example "All data",
                "Background", "Task"). Activities with no marked time steps
                are skipped.

        Returns:
            pandas.DataFrame: One row per activity segment with summary
            statistics for the chosen metric. Column names embed their units
            in square brackets where applicable (for example "Max [µg/m³]").
            The set of columns matches that of
            :meth:`Aerosol2D.summarize_exposure`, restricted to a single
            1D metric.

        Raises:
            ValueError: If ``metric`` cannot be found (except "PNC").
            ValueError: If a background activity name is given but does not
                exist or has no samples.
            ValueError: If ``short_window`` or ``twa_window`` cannot be
                parsed as pandas-style durations.
            ValueError: If ``exposure_hours`` is negative.
            TypeError: If ``background`` is not None, float, or str.

        Notes:
            Detailed description:
                The method first obtains the requested metric time series
                from the 1D data:

                    - For "PNC", the series is taken from
                      :attr:`total_concentration`.
                    - For any other metric name, the series is taken from a
                      numeric column in :attr:`data` or :attr:`extra_data`
                      (if present).

                For each selected activity, the method:

                    1. Extracts the metric time series within the activity
                       using the corresponding boolean mask.
                    2. Computes per-step durations from the actual sampling
                       times via :meth:`_dt_minutes`.
                    3. Forms a time-weighted segment mean (using the per-step
                       durations), which is used internally when embedding
                       the activity into the TWA window.
                    4. Computes instantaneous percentiles, peaks, STEL-style
                       exceedances, and long-term time-above-limit measures.
                    5. Combines the segment mean with the specified background
                       level to obtain an overall TWA for the chosen window.

                The returned DataFrame contains the following columns
                (per activity):

                    - "Segment"
                        Name of the activity segment.

                    - "Metric"
                        Name of the metric that was summarized, e.g.
                        "PNC", "MASS", or any other 1D column name.

                    - "Duration [HH:MM]"
                        Duration of the activity derived from the actual
                        sampling intervals, formatted as "HH:MM" (hours and
                        minutes).

                    - "Max [unit]"
                        Maximum of the instantaneous metric values during the
                        activity. The unit is the metric unit (for example
                        "cm⁻³" or "µg/m³").

                    - "95th percentile [unit]"
                    - "75th percentile [unit]"
                    - "50th percentile [unit]"
                    - "25th percentile [unit]"
                    - "5th percentile [unit]"
                        Percentiles of the instantaneous metric values during
                        the activity, all in the same unit as the metric.

                    - "Peaks [count]"
                        Number of distinct peak events detected in the
                        activity, where a peak is a contiguous period in which
                        the metric exceeds::

                            baseline + peak_ratio * rolling_std

                        with ``baseline`` the rolling median and
                        ``rolling_std`` the rolling standard deviation over a
                        short sliding window, covering at least 3 datapoints
                        and at most 15 datapoints, depending on how many
                        samples are available in the segment.

                    - "STEL [unit]"
                        Short-term exposure limit used in the STEL
                        exceedance calculations (echoing ``short_limit``),
                        with the same concentration unit as the metric.

                    - "STEL window [offset]"
                        Rolling window length used for STEL evaluation,
                        echoing ``short_window`` (for example "15min").

                    - "STEL exceedance [min]"
                        Total time in minutes during the activity where the
                        **full-window** rolling mean is greater than or equal
                        to the STEL. Only complete windows are counted in this
                        measure.

                    - "STEL episodes [count]"
                        Number of distinct STEL exceedance episodes based on
                        the full-window rolling mean. Each contiguous block of
                        time where the full-window mean is at or above the
                        STEL is counted as one episode.

                    - "Exposure limit [unit]"
                        Long-term exposure limit value used in the long-term
                        evaluation (echoing ``long_limit``), with the same
                        unit as the metric.

                    - "Exposure limit exceedance [min]"
                        Total time in minutes during the activity where the
                        instantaneous metric values are greater than or equal
                        to the exposure limit.

                    - "TWA concentration [unit]"
                        Time-weighted average concentration of the metric over
                        the full TWA window (``"TWA window [offset]"``),
                        combining the internal segment mean (optionally
                        stretched to ``exposure_hours``) with the specified
                        background level for the remainder of the window::

                            TWA = (segment_mean * exposure_minutes
                                   + background * (twa_minutes - exposure_minutes))
                                  / twa_minutes

                        where ``twa_minutes`` is the duration of
                        ``twa_window`` in minutes and ``exposure_minutes`` is
                        either the measured activity duration or
                        ``exposure_hours * 60``, whichever is used.

                    - "TWA window [offset]"
                        Duration of the TWA window over which
                        "TWA concentration [unit]" is evaluated, echoing
                        ``twa_window`` (for example "8h").

            Theory:
                Conceptually, the method mirrors typical occupational hygiene
                practice for a single 1D metric:

                    - Exposure within an activity is represented by
                      time-weighted averages and distributional descriptors
                      (max and percentiles).
                    - Short-term limits (STEL) are evaluated using rolling
                      means over a specified window, counting both total time
                      above the limit and distinct exceedance episodes.
                    - Long-term limits are represented by cumulative
                      time-above-limit measures.
                    - An overall TWA for an 8-h (or user-specified) reference
                      period is constructed by mixing the task exposure with a
                      background level for the rest of the window.

                This ensures a transparent link between raw 1D time-series
                behaviour and regulatory-style metrics such as STEL and
                8-hour OELs.

        Examples:
            Summarize total particle number concentration (PNC) exposure
            for all activities in a CPC dataset, using the "Background"
            activity as background level and a 15-min STEL with an 8-h
            TWA window:

            .. code-block:: python

                cpc.summarize_exposure(
                        metric="PNC",
                        background="Background",
                        short_limit=5e4,
                        long_limit=2e4,
                        short_window="15min",
                        twa_window="8h",
                )

            Other 1D metrics stored as columns (for example "MASS") can
            be handled in the same way simply by changing the ``metric``
            argument.
        """

        # --- obtain the time series for the requested metric -------------------
        series_source, unit = self._get_metric_series(metric)

        # --- time deltas in minutes for integration (irregular sampling safe) ---
        dt_mins = self._dt_minutes()

        # --- parse windows ------------------------------------------------------
        try:
            short_minutes = pd.to_timedelta(short_window).total_seconds() / 60.0
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not parse short_window {short_window!r}.") from exc

        try:
            twa_minutes = pd.to_timedelta(twa_window).total_seconds() / 60.0
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not parse twa_window {twa_window!r}.") from exc

        # --- background level (one value, reused for all activities) -----------
        if background is None:
            bg = 0.0
        elif isinstance(background, (int, float)):
            bg = float(background)
        elif isinstance(background, str):
            if background not in self.activities:
                raise ValueError(
                    f"Background activity '{background}' not found in activities."
                )
            mask_bg = self.data[background].astype(bool)
            if mask_bg.sum() == 0:
                raise ValueError(
                    f"Background activity '{background}' has no marked time steps."
                )
            s_bg = series_source.loc[mask_bg]
            dt_bg = dt_mins.loc[mask_bg]
            dur_bg = float(dt_bg.sum())
            bg = float((s_bg * dt_bg).sum() / max(dur_bg, 1e-9))
        else:
            raise TypeError(
                "background must be None, a float, or an activity name (str)."
            )

        # --- assumed exposure duration within the TWA window --------------------
        if exposure_hours is not None and exposure_hours < 0:
            raise ValueError("exposure_hours must be non-negative.")

        # --- which activities to summarize -------------------------------------
        if activities is None:
            activity_list = list(self.activities)
        else:
            activity_list = list(activities)

        # Column labels with units embedded (same for all rows)
        duration_label = "Duration [HH:MM]"
        max_label = f"Max [{unit}]"
        p95_label = f"95th percentile [{unit}]"
        p75_label = f"75th percentile [{unit}]"
        p50_label = f"50th percentile [{unit}]"
        p25_label = f"25th percentile [{unit}]"
        p5_label = f"5th percentile [{unit}]"
        peaks_label = "Peaks [count]"
        stel_label = f"STEL [{unit}]"
        stel_window_label = "STEL window [offset]"
        stel_exceed_label = "STEL exceedance [min]"
        stel_episodes_label = "STEL episodes [count]"
        exp_limit_label = f"Exposure limit [{unit}]"
        exp_exceed_label = "Exposure limit exceedance [min]"
        twa_conc_label = f"TWA concentration [{unit}]"
        twa_window_label = "TWA window [offset]"

        rows: list[dict[str, Any]] = []

        for activity in activity_list:
            if activity not in self.activities:
                raise ValueError(f"Activity '{activity}' not found in activities.")

            mask = self.data[activity].astype(bool)
            if mask.sum() == 0:
                # Skip empty activities
                continue

            s = series_source.loc[mask]
            dtm = dt_mins.loc[mask]

            # --- basic duration & segment mean ----------------------------------
            duration_min = float(dtm.sum())
            duration_hhmm = self._format_hhmm(duration_min)
            if duration_min <= 0:
                # Skip pathological segments rather than crashing everything
                continue

            seg_mean = float((s * dtm).sum() / duration_min)
            seg_max = float(s.max())

            # percentiles in metric units
            p95, p75, p50, p25, p5 = map(
                float, np.nanpercentile(s, [95, 75, 50, 25, 5])
            )

            # --- short-term limit: full-window rolling mean (episodes) ----------
            if len(dtm) > 0 and np.isfinite(dtm.median()) and dtm.median() > 0:
                dt_med = float(dtm.median())
            else:
                dt_med = short_minutes  # fall back to something non-zero

            if dt_med <= 0:
                dt_med = short_minutes

            n_win = max(1, int(round(short_minutes / dt_med)))
            if n_win > s.size:
                n_win = s.size

            if s.size == 1:
                s_roll_full = s.copy()
            else:
                s_roll_full = s.rolling(window=n_win, min_periods=n_win).mean()

            full_exceed_mask = s_roll_full >= float(short_limit)
            short_full_exceed_min = float(dtm.where(full_exceed_mask, 0.0).sum())
            short_full_episodes = int(
                ((full_exceed_mask) & (~full_exceed_mask.shift(fill_value=False))).sum()
            )

            # --- peaks using rolling baseline -----------------------------------
            window_size = max(3, min(15, s.size))
            baseline = s.rolling(
                window=window_size, center=True, min_periods=1
            ).median()
            spread = s.rolling(window=window_size, center=True, min_periods=1).std()
            peak_mask = (s - baseline) > (spread * float(peak_ratio))
            peak_count = int(((peak_mask) & (~peak_mask.shift(fill_value=False))).sum())

            # --- window TWA with background -------------------------------------
            if exposure_hours is None:
                exposure_minutes = duration_min
            else:
                exposure_minutes = float(exposure_hours) * 60.0

            if exposure_minutes >= twa_minutes:
                window_twa = seg_mean
            else:
                window_twa = float(
                    (
                        seg_mean * exposure_minutes
                        + bg * (twa_minutes - exposure_minutes)
                    )
                    / twa_minutes
                )

            # instantaneous time above long-term limit within the activity
            long_exceed_min = float(dtm.where(s >= float(long_limit), 0.0).sum())

            rows.append(
                {
                    "Segment": activity,
                    "Metric": metric,
                    duration_label: duration_hhmm,
                    max_label: round(seg_max, 3),
                    p95_label: round(p95, 3),
                    p75_label: round(p75, 3),
                    p50_label: round(p50, 3),
                    p25_label: round(p25, 3),
                    p5_label: round(p5, 3),
                    peaks_label: peak_count,
                    stel_label: float(short_limit),
                    stel_window_label: short_window,
                    stel_exceed_label: round(short_full_exceed_min, 2),
                    stel_episodes_label: short_full_episodes,
                    exp_limit_label: float(long_limit),
                    exp_exceed_label: round(long_exceed_min, 2),
                    twa_conc_label: round(window_twa, 3),
                    twa_window_label: twa_window,
                }
            )

        result = pd.DataFrame(rows)

        # --- append to file if requested ---------------------------------------
        if filename and not result.empty:
            fname = str(filename)
            lower = fname.lower()
            if lower.endswith(".csv"):
                if os.path.exists(fname):
                    result.to_csv(fname, mode="a", header=False, index=False)
                else:
                    result.to_csv(fname, mode="w", header=True, index=False)
            elif lower.endswith((".xlsx", ".xls")):
                if os.path.exists(fname):
                    existing = pd.read_excel(fname)
                    combined = pd.concat([existing, result], ignore_index=True)
                else:
                    combined = result
                combined.to_excel(fname, index=False)
            else:
                raise ValueError(
                    f"Unsupported file extension for '{filename}'. Use .csv or .xlsx."
                )

        # --- pretty terminal print (transposed) --------------------------------
        if not result.empty:
            result_t = result.set_index("Segment").T
            display_df = result_t.reset_index().rename(columns={"index": "Metric"})
            table_data = cast(
                Dict[str, Sequence[Any]],
                display_df.to_dict(orient="list"),
            )

            print(
                f"\nExposure summary by segment for metric '{metric}' ({unit}) [1D]:\n"
            )
            print(
                tabulate(
                    table_data,
                    headers="keys",
                    tablefmt="pretty",
                    floatfmt=".3f",
                )
            )

        return result

    ###########################################################################

    def timecrop(
        self,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        inplace: bool = True,
        focus: bool = True,
        crop_extra: bool = True,
    ) -> "Aerosol1D":
        """Description:
            Crop the time axis to include or exclude a time interval.

        Args:
            start (str | pandas.Timestamp | None): Start of the interval.
                If None, cropping starts at the first timestamp.
            end (str | pandas.Timestamp | None): End of the interval.
                If None, cropping ends at the last timestamp.
            inplace (bool): If True, modify the current object and return
                it. If False, return a cropped deep copy.
            focus (bool): If True, keep only the interval [start, end].
                If False, remove that interval and keep data before start
                and after end.
            crop_extra (bool): If True, also crop extra_data to the same
                time mask (when inplace=True or on the copy).

        Returns:
            Aerosol1D: The cropped object (self when inplace=True, or a
                new instance when inplace=False).

        Raises:
            ValueError: May be raised by pandas.to_datetime if start or
                end strings cannot be parsed. Check the date format
                (for example "YYYY-MM-DD HH:MM:SS").

        Notes:
            Detailed description:
                A boolean mask is built over the current time index based
                on start and end. Depending on focus, this mask either
                selects or removes the specified interval. All columns in
                self.data are subset accordingly. When crop_extra is True,
                extra_data is cropped using the same mask, keeping time
                alignment between main and extra channels.

        Examples:
            Drop a calibration period at the start of a measurement:

            .. code-block:: python

                data.timecrop(end="2025-01-24 09:00", focus=False)
                data.plot_total_conc()
        """

        mask = pd.Series(True, index=self.time)
        if start is not None:
            mask &= self.time >= pd.to_datetime(start)
        if end is not None:
            mask &= self.time <= pd.to_datetime(end)
        if not focus:
            mask = ~mask

        if inplace:
            self._data = self._data.loc[mask]
            if crop_extra and len(self._extra_data) > 0:
                self._extra_data = self._extra_data.loc[mask]
            return self
        else:
            obj = self.copy_self()
            obj._data = obj._data.loc[mask]
            if crop_extra and len(obj._extra_data) > 0:
                obj._extra_data = obj._extra_data.loc[mask]
            return obj

    ##########################################################################

    def timerebin(
        self,
        freq: str = "s",
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        method: Union[str, Callable] = "mean",
        inplace: bool = True,
    ) -> "Aerosol1D":
        """Description:
            Resample the time series to a new regular frequency.

        Args:
            freq (str): Target sampling interval as a pandas offset alias
                (for example "30s", "1min", "5min", "1H"). Defaults to "s".
            start (datetime | pandas.Timestamp | None): Optional start time
                for the resampling window. If None, the first timestamp is
                used.
            end (datetime | pandas.Timestamp | None): Optional end time for
                the resampling window. If None, the last timestamp is used.
            method (str | Callable): Aggregation function for numeric
                columns. Any aggregation accepted by pandas.DataFrame.resample
                (for example "mean", "sum", "median") or a custom callable.
            inplace (bool): If True, update this object and return it. If
                False, leave this instance unchanged and return a rebinned
                deep copy.

        Returns:
            Aerosol1D: The rebinned object (self when inplace=True, or a
                new instance otherwise) with a regular DatetimeIndex and
                resampled main and extra data.

        Raises:
            ValueError: May be raised by pandas if freq is not a valid
                offset alias or if start/end cannot be interpreted as
                timestamps.

        Notes:
            Detailed description:
                Numeric columns in self.data are resampled using the chosen
                method, while boolean columns are aggregated with a logical
                maximum so that any True within a bin keeps the bin True.
                The resulting DataFrame is reindexed to a fully regular
                time grid between the floored start and end times. Extra
                data, if present, are resampled in the same way and aligned
                to the new index. Activity masks such as "All data" are
                maintained and missing boolean values are filled with False.

        Examples:
            Convert a high‑frequency logger to 1‑minute averages:

            .. code-block:: python

                data.timerebin(freq="1min", method="mean")
                data.plot_total_conc()
        """

        # Select numeric and boolean columns
        numeric_cols = list(self._data.select_dtypes(exclude="bool").columns)
        bool_cols = list(self._data.select_dtypes(include="bool").columns)

        # Resample main data
        rebinned_numeric = (
            self._data[numeric_cols].resample(freq).agg(method)
            if numeric_cols
            else pd.DataFrame()
        )
        rebinned_bool = (
            self._data[bool_cols].resample(freq).max().astype(bool)
            if bool_cols
            else pd.DataFrame()
        )
        rebinned = pd.concat([rebinned_numeric, rebinned_bool], axis=1)

        # Determine window and align to grid
        start_ts = (
            pd.Timestamp(start) if start is not None else pd.Timestamp(self.time[0])
        )
        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(self.time[-1])
        start_time = start_ts.floor(freq)
        end_time = end_ts.floor(freq)

        # Reindex to aligned range
        time_index = pd.date_range(start=start_time, end=end_time, freq=freq)
        rebinned = rebinned.reindex(time_index)

        # Resample extra data (if any)
        rebinned_extra = pd.DataFrame(index=time_index)
        if len(self._extra_data) > 0:
            num_extra = list(self._extra_data.select_dtypes(exclude="bool").columns)
            bool_extra = list(self._extra_data.select_dtypes(include="bool").columns)

            r_num = (
                self._extra_data[num_extra].resample(freq).agg(method)
                if num_extra
                else pd.DataFrame(index=time_index)
            )
            r_bool = (
                self._extra_data[bool_extra].resample(freq).max().astype(bool)
                if bool_extra
                else pd.DataFrame(index=time_index)
            )

            rebinned_extra = pd.concat([r_num, r_bool], axis=1).reindex(time_index)

            # Fill missing booleans
            if bool_extra:
                existing = [c for c in bool_extra if c in rebinned_extra.columns]
                if existing:
                    rebinned_extra[existing] = rebinned_extra[existing].fillna(False)

        # Assign results
        if inplace:
            self._data = rebinned

            # Fill missing booleans
            if "All data" in self._data:
                self._data["All data"] = self._data["All data"].fillna(False)
            if bool_cols:
                existing = [c for c in bool_cols if c in self._data.columns]
                if existing:
                    self._data[existing] = self._data[existing].fillna(False)

            if len(self._extra_data) > 0:
                self._extra_data = rebinned_extra
            return self
        else:
            obj = self.copy_self()
            obj._data = rebinned
            if len(self._extra_data) > 0:
                obj._extra_data = rebinned_extra
            return obj

    ###########################################################################

    def timeshift(
        self,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        inplace: bool = True,
        shift_extra: bool = True,
    ):
        """Description:
            Shift the time index by a constant offset.

        Args:
            seconds (float): Offset in seconds; positive shifts the series
                forward in time, negative backward. Defaults to 0.
            minutes (float): Offset in minutes, converted internally to
                seconds and added to seconds. Defaults to 0.
            hours (float): Offset in hours, converted internally to
                seconds and added to seconds. Defaults to 0.
            inplace (bool): If True, shift this object in place and return
                it. If False, return a shifted deep copy. Default is True.
            shift_extra (bool): If True, also shift the index of
                extra_data so that main and extra channels remain aligned.
                Default is True.

        Returns:
            Aerosol1D: Object with shifted time index (self when
                inplace=True, otherwise a new instance).

        Raises:
            None: Unless invalid types are passed for the offset arguments.

        Notes:
            Detailed description:
                A pandas.Timedelta is constructed from the combined offset:
                total_shift = seconds + 60*minutes + 3600*hours.
                This timedelta is added to the index of self.data, and
                optionally to self.extra_data. Activity masks and other
                columns are left unchanged; only timestamps move.

        Examples:
            Align measurements from two instruments with a known clock
            offset:

            .. code-block:: python

                cpc.timeshift(minutes=+2)
                elpi.timeshift(minutes=-1)
                cpc.plot_total_conc()
                elpi.plot_total_conc()
        """

        total_seconds = seconds + 60 * minutes + 3600 * hours
        delta = pd.to_timedelta(total_seconds, unit="s")

        target = self if inplace else self.copy_self()
        target._data.index = target._data.index + delta

        if shift_extra and len(target._extra_data) > 0:
            # Shift extra_data if it shares the same index
            target._extra_data.index = target._extra_data.index + delta

        return target

    ###########################################################################

    def timesmooth(self, window: int = 5, method: str = "mean", inplace: bool = True):
        """Description:
            Apply a centered rolling smoother to numeric columns.

        Args:
            window (int): Rolling window size in number of samples.
                Defaults to 5.
            method (str): Aggregation used for smoothing, one of
                "mean", "median", "sum", "min", "max". Defaults to "mean".
            inplace (bool): If True, replace the numeric columns of this
                object with their smoothed versions. If False, return a
                smoothed deep copy.

        Returns:
            Aerosol1D: Object with smoothed numeric columns and unchanged
                boolean/activity columns (self when inplace=True, otherwise
                a new instance).

        Raises:
            ValueError: If method is not among the supported names.
                Choose one of "mean", "median", "sum", "min", "max".

        Notes:
            Detailed description:
                The method separates numeric and boolean columns in
                self.data. A centered rolling window with at least one
                valid sample (min_periods=1) is applied to numeric columns
                using the requested aggregation, producing a smoothed
                version of each numeric time series. Boolean columns
                untouched.

            Theory:
                The smoothing is a simple moving-window operation. It
                can help reduce noise in total concentration or other
                instrument channels prior to further analysis.

        Examples:
            Smooth a noisy CPC time series while preserving activities:

            .. code-block:: python

                smooth = data.timesmooth(window=11, method="median", inplace=False)
                smooth.plot_total_conc(mark_activities=True)
        """

        if method not in ["mean", "median", "sum", "min", "max"]:
            raise ValueError(
                "Invalid method. Choose from 'mean', 'median', 'sum', 'min', 'max'."
            )

        numeric_cols = self._data.select_dtypes(exclude="bool").columns
        bool_cols = self._data.select_dtypes(include="bool").columns

        smoothed_numeric = getattr(
            self._data[numeric_cols].rolling(window=window, center=True, min_periods=1),
            method,
        )()
        preserved_bool = self._data[bool_cols]

        smoothed = pd.concat([smoothed_numeric, preserved_bool], axis=1)

        if inplace:
            self._data = smoothed
            return self
        else:
            new_obj = self.copy_self()
            new_obj._data = smoothed
            return new_obj
