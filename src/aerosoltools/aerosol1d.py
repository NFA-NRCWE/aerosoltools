from __future__ import annotations

import copy
import datetime as dt
from typing import Callable, Dict, List, Optional, Sequence, Union, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
    """
    Class for handling 1D aerosol time series measurements.

    This class manages time-indexed aerosol concentration data (e.g., total particle concentration).
    It provides utilities for resampling, smoothing, marking activity segments, cropping,
    shifting, summarizing, and plotting data. It is particularly suited for pre- and post-processing
    of aerosol datasets collected via portable or stationary particle counters.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A DataFrame containing time-indexed aerosol data. If the index is not a DatetimeIndex,
        the first column will be interpreted as timestamps and set as the index automatically.
        The first data column is assumed to represent total particle concentration.

    Notes
    -----
    This class is intended for workflows that involve segmenting and analyzing time-resolved
    total aerosol concentration data. Users should interact primarily through public methods
    and properties, rather than modifying internal attributes directly.
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
            dataframe.columns = ["Total Concentration"]

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
    def activities(self) -> List:
        """
        List of defined activity labels.

        Returns
        -------
        list of str
            List of activity names that have been marked in the dataset.
        """
        return self._activities

    @property
    def activity_periods(self) -> Dict:
        """
        Dictionary of activity names and their associated time periods.

        Returns
        -------
        dict
            Dictionary where keys are activity names and values are lists of (start, end) tuples.
        """
        return self._activity_periods

    @property
    def data(self) -> pd.DataFrame:
        """
        Dataframe with all data, times, and activity columns if marked.

        Returns
        -------
        pd.DataFrame
            The full DataFrame.
        """
        return self._data

    @property
    def dtype(self):
        """
        Data type description of the measurements.

        Returns
        -------
        str
            The type of data (e.g., particle number concentration, mass concentration).
        """
        return self._meta.get("dtype", "Uknown dtype")

    @property
    def extra_data(self) -> pd.DataFrame:
        """
        Extra data columns which were extracted from the raw data file

        Returns
        -------
        pd.DataFrame
            Dataframe of extra data columns which are not native to the Aerosol class object.
        """
        return self._extra_data

    @property
    def instrument(self) -> str:
        """
        Instrument used for the measurements.

        Returns
        -------
        str
            Name or description of the instrument.
        """
        return self._meta.get("instrument", "Uknown instrument")

    @property
    def metadata(self) -> Dict:
        """
        Return overiew of meta data

        Returns
        -------
        dict
            Contains; unit, data type, instrument type and serial_number.
        """
        return self._meta

    @property
    def original_data(self) -> pd.DataFrame:
        """
        Unmodified original dataset.

        Returns
        -------
        pandas.DataFrame
            Copy of the raw, original data before any processing.
        """
        return self._raw_data

    @property
    def original_extra_data(self) -> pd.DataFrame:
        """
        Unmodified original dataset.

        Returns
        -------
        pandas.DataFrame
            Copy of the raw, original data before any processing.
        """
        return self._raw_extra_data

    @property
    def serial_number(self) -> str:
        """
        Serial number of instrument

        Returns
        -------
        str
            Serial number of instrument
        """
        return self._meta.get("serial_number", "Uknown serial number")

    @property
    def time(self) -> pd.DatetimeIndex:
        """
        Timestamps of the dataset.

        Returns
        -------
        pandas.DatetimeIndex
            Time index of the measurements.
        """
        return cast(pd.DatetimeIndex, self._data.index)

    @property
    def total_concentration(self) -> pd.Series:
        """
        Total concentration measurements.

        Returns
        -------
        pandas.Series
            Total concentration data over time.
        """
        if "Total Concentration" in self._data.columns:
            return self._data["Total Concentration"]
        else:
            return self._data.iloc[:, 0]

    @property
    def unit(self) -> str:
        """
        Unit of the measurements.

        Returns
        -------
        str
            Unit string (e.g., "#/cm³", "µg/m³").
        """
        return self._meta.get("unit", "Uknown unit")

    ###########################################################################
    """############################# Functions #############################"""
    ###########################################################################

    def copy_self(self):
        """
        Create a deep copy of the current Aerosol1D  object.

        Returns
        -------
        Aerosol1D
            A deep copy of the current instance.
        """
        return copy.deepcopy(self)

    ###########################################################################

    def get_activity_data(self, activity_name):
        """
        Extract data corresponding to a specified activity.

        Parameters
        ----------
        activity_name : str
            Name of the activity to extract.

        Returns
        -------
        pandas.DataFrame
            Subset of the data where the specified activity is active (True).
        """
        if activity_name not in self.activities:
            raise ValueError(
                f"Activity '{activity_name}' not found in available activities: {self.activities}"
            )

        return self._data[self._data[activity_name]].drop(columns=self.activities)

    ###########################################################################

    def get_activity_extra_data(self, activity_name):
        """
        Extract data corresponding to a specified activity.

        Parameters
        ----------
        activity_name : str
            Name of the activity to extract.

        Returns
        -------
        pandas.DataFrame
            Subset of the data where the specified activity is active (True).
        """
        if activity_name not in self.activities:
            raise ValueError(
                f"Activity '{activity_name}' not found in available activities: {self.activities}"
            )

        return self.extra_data[self._data[activity_name]]

    ###########################################################################
    def mark_activities(self, activity_periods, mode="union"):
        """
        Mark activities in the data by adding one boolean column per activity.

           Parameters
           ----------
           activity_periods : dict
               Dictionary where keys are activity names (str) and values are
               (start, end) tuples or list of (start, end) tuples.
            mode:
              - "union": existing OR new (accumulate periods) [default]
              - "replace": overwrite existing with new
              - "intersection": existing AND new

           Returns
           -------
           None

        """
        
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

    # def mark_activities(self, activity_periods):
    #     """
    #     Mark activities in the data by adding one boolean column per activity.

    #     Parameters
    #     ----------
    #     activity_periods : dict
    #         Dictionary where keys are activity names (str) and values are
    #         (start, end) tuples or list of (start, end) tuples.

    #     Returns
    #     -------
    #     None
    #     """
    #     new_cols = {}

    #     for activity, periods in activity_periods.items():
    #         # Initialize column with False
    #         col = pd.Series(False, index=self.time)

    #         # Normalize periods
    #         if isinstance(periods, tuple) and len(periods) == 2:
    #             periods = [periods]

    #         for start, end in periods:
    #             mask = (self.time >= pd.Timestamp(start)) & (
    #                 self.time <= pd.Timestamp(end)
    #             )
    #             col[mask] = True

    #         new_cols[activity] = col

    #         # Track metadata
    #         if activity not in self._activities:
    #             self._activities.append(activity)
    #         self._activity_periods[activity] = periods

    #     # Add all new activity columns at once
    #     self._data = pd.concat([self._data, pd.DataFrame(new_cols)], axis=1)

    ###########################################################################

    def Peak_finder(
        self,
        window: int = 15,
        ratio: float = 2.5,
        method: str = "median",
        specific_data: str = "",
    ):
        """
        Function for locating peaks based on diviance from standard deviation
        over a chosen period.

        Parameters
        ----------
        window : int
            Width of the window around which the peak should be found, with each
            integer being a time step away from the investigated point.
        ratio : float
            Value over which the the difference from median should be larger than
            the standard deviation to be counted as a peak

        Returns
        -------
        mask : np.array
            array of boolean values indicating found peaks.

        """

        if method not in ["mean", "median", "sum", "min", "max"]:
            raise ValueError(
                "Invalid method. Choose from 'mean', 'median', 'sum', 'min', 'max'."
            )

        window = int(window)
        Data_return = self.copy_self()

        if specific_data == "":
            Data_return = Data_return._data["Total_conc"]
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
            self._data = pd.concat([self._data, pd.DataFrame(peak_col)], axis=1)
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
        """
        Plot the total concentration over time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An existing Matplotlib Axes object. If None, a new figure and axes are created.
        mark_activities : bool or list of str, optional
            If True, highlights all activity periods **except "All data"**.
            If a list of activity names is provided, only those will be highlighted.
            If False (default), no activities are marked.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Matplotlib figure object.
        ax : matplotlib.axes.Axes
            The Matplotlib axes object with the plot.
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

    def summarize(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Summarize total concentration statistics for each defined activity,
        including 'All data'.

        Parameters
        ----------
        filename : str, optional
            Path to an Excel file where the summary will be saved. If None, no file is saved.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing summary statistics.
        """
        rows = []

        # Loop through all activities (including "All data")
        for activity in self.activities:
            try:
                subset = self.data[self.data[activity]][self.total_concentration.name]
            except KeyError:
                subset = self.data[self.data[activity]].iloc[:, 0]

            if not subset.empty:
                rows.append(
                    [
                        activity,
                        subset.min(),
                        subset.max(),
                        subset.mean(),
                        subset.std(),
                        len(subset),
                    ]
                )

        # Create DataFrame
        summary = pd.DataFrame(
            rows, columns=["Segment", "Min", "Max", "Mean", "Std", "N datapoints"]
        )
        summary_rounded = summary.round(3)

        # Console output
        print("\nSummary of total concentration:\n")
        print(
            tabulate(
                summary_rounded.values.tolist(),
                headers=summary_rounded.columns.tolist(),
                tablefmt="pretty",
                floatfmt=".3f",
            )
        )

        # Optionally save
        if filename:
            summary_rounded.to_excel(filename, index=False)
            print(f"\nSummary saved to: {filename}")

        return summary_rounded

    ###########################################################################

    def timecrop(
        self,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        inplace: bool = True,
        focus: bool = True,
    ) -> "Aerosol1D":
        """
        Crop the data to a specified time window.

        Parameters
        ----------
        start : str or pd.Timestamp, optional
            Start time. If None, cropping starts from the earliest available time.
            If a string, it should have the format “YYYY-MM-DD HH:MM:SS”, e.g., “2025-01-24 20:00:00”.
        end : str or pd.Timestamp, optional
            End time. If None, cropping ends at the latest available time.
            If a string, it should have the format “YYYY-MM-DD HH:MM:SS”, e.g., “2025-01-25 20:00:00”.
        inplace : bool, optional
            If True (default), modifies the current object. If False, returns a new cropped instance.
        focus : bool, optional
            If True (default), keep the selected time window.
            If False, remove the selected time window and keep everything else.

        Returns
        -------
        Aerosol1D
            Instance of Aerosol1D. Either the modified current object (if inplace=True) or a new cropped object (if inplace=False).
        """
        mask = pd.Series(True, index=self.time)

        if start is not None:
            start = pd.to_datetime(start)
            mask &= self.time >= start
        if end is not None:
            end = pd.to_datetime(end)
            mask &= self.time <= end

        # Invert if we're *removing* the selected window
        if not focus:
            mask = ~mask

        if inplace:
            self._data = self._data.loc[mask]
            self._extra_data = self._extra_data.loc[mask]
            return self
        else:
            cropped = self.copy_self()
            cropped._data = self._data.loc[mask]
            return cropped

    ##########################################################################

    def timerebin(
        self,
        freq: str = "s",
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        method: Union[str, Callable] = "mean",
        inplace: bool = True,
    ) -> "Aerosol1D":
        """
        Resample the data to a new time frequency using an aggregation function.

        This method resamples numeric columns with the chosen aggregation and boolean
        columns with a logical max, aligns the time index to the frequency grid, and
        optionally applies the same operation to any extra-data frame.

        Parameters
        ----------
        freq : str, optional
            Resampling frequency. Naming convention is 's', 'min', 'h', 'D' for
            seconds, minutes, hours, and days, optionally with an integer (e.g.,
            '30s', '5min', '1H', '1D'). Default is 's'.
        start : datetime | pandas.Timestamp | None, optional
            Start of the resampling window. If None, the first timestamp in the data
            is used. The start is snapped down to the nearest frequency grid using
            ``Timestamp.floor(freq)``.
        end : datetime | pandas.Timestamp | None, optional
            End of the resampling window. If None, the last timestamp in the data
            is used. The end is snapped down to the nearest frequency grid using
            ``Timestamp.floor(freq)``.
        method : str or callable, optional
            Aggregation method applied to numeric columns (e.g., 'mean', 'median',
            'sum', 'min', 'max', or a custom function). Default is 'mean'.
        inplace : bool, optional
            If True, modifies the object in place. If False, returns a new rebinned
            object. Default is True.

        Returns
        -------
        Aerosol1D
            Instance of ``Aerosol1D`` (or subclass) with rebinned time index. Returns
            ``self`` if ``inplace=True``, otherwise a new instance.

        Notes
        -----
        - Boolean columns are aggregated with logical max and missing booleans after
        reindexing are filled with ``False``.
        - Extra data (``_extra_data``) is resampled and aligned if present.
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
    ):
        """
        Shift the time index by a given number of seconds and/or minutes.

        Parameters
        ----------
        seconds : float, optional
            Number of seconds to shift. Defaults to 0.
        minutes : float, optional
            Number of minutes to shift. Defaults to 0.
        inplace : bool, optional
            Whether to modify the object in place. Defaults to True.

        Returns
        -------
        Aerosol1D
            Instance of Aerosol1D with shifted time index. Either the modified current object (if inplace=True) or a new shifted object (if inplace=False).
        """
        total_seconds = seconds + 60 * minutes + 3600 * hours
        total_shift = pd.to_timedelta(total_seconds, unit="s")

        if inplace:
            self._data.index = self._data.index + total_shift
            if len(self._extra_data)>0:
                self._extra_data.index = self._extra_data.index + total_shift
            return self
        else:
            shifted = self.copy_self()
            shifted._data.index = shifted._data.index + total_shift
            if len(shifted._extra_data)>0:
                shifted._extra_data.index = shifted._extra_data.index + total_shift
            return shifted

    ###########################################################################

    def timesmooth(self, window: int = 5, method: str = "mean", inplace: bool = True):
        """
        Apply rolling window smoothing to the data.

        Parameters
        ----------
        window : int, optional
            Size of the moving window (in number of samples). Default is 5.
        method : str, optional
            Aggregation method to use: 'mean', 'median', 'sum', 'min', or 'max'. Default is 'mean'.
        inplace : bool, optional
            If True, modifies the current object. If False, returns a new smoothed instance. Default is True.

        Returns
        -------
        Aerosol1D
            Instance of Aerosol1D with smoothed data.
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
