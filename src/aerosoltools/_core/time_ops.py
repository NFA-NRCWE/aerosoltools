"""Time-axis operations for 1D/2D aerosol data: crop, rebin, shift, smooth."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Callable, Optional, Union

import pandas as pd

if TYPE_CHECKING:  # avoid a circular import at runtime; only for type hints
    from ..aerosol1d import Aerosol1D

DateLike = Union[dt.datetime, pd.Timestamp]


class TimeOpsMixin:
    """Crop, rebin, shift and smooth the time axis."""

    def timecrop(
        self,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        inplace: bool = True,
        focus: bool = True,
        crop_extra: bool = True,
    ) -> "Aerosol1D":
        """Crop the time axis to include or exclude a time interval.

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

    def timerebin(
        self,
        freq: str = "s",
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        method: Union[str, Callable] = "mean",
        inplace: bool = True,
    ) -> "Aerosol1D":
        """Resample the time series to a new regular frequency.

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

        # Select numeric and boolean columns. Use ``include="number"`` (not
        # ``exclude="bool"``) so that text/object columns — e.g. status or
        # comment channels in some instruments' extra data — are not fed to a
        # numeric aggregation, which would raise. Such columns are dropped on
        # resampling, as they cannot be meaningfully aggregated over time.
        numeric_cols = list(self._data.select_dtypes(include="number").columns)
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
            num_extra = list(self._extra_data.select_dtypes(include="number").columns)
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

    def timeshift(
        self,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        inplace: bool = True,
        shift_extra: bool = True,
    ):
        """Shift the time index by a constant offset.

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

    def timesmooth(self, window: int = 5, method: str = "mean", inplace: bool = True):
        """Apply a centered rolling smoother to numeric columns.

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
