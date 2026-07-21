"""Activity/segment handling: mark periods, slice by activity, detect peaks."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd


class ActivityMixin:
    """Define, retrieve and auto-detect activity periods."""

    def mark_activities(self, activity_periods, mode: str = "union"):
        """Define or update boolean activity masks on the time axis.

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

    def rename_activity(self, old_name: str, new_name: str) -> None:
        """Rename an existing activity, keeping its periods and mask intact.

        Args:
            old_name (str): Current activity name.
            new_name (str): New name for the activity.

        Returns:
            None: The object is modified in place: the boolean mask column
                in self.data and the entries in self.activities and
                self.activity_periods are renamed.

        Raises:
            ValueError: If old_name is not a known activity, or new_name is
                already used by a different activity.
        """
        if old_name not in self._activities:
            raise ValueError(f"No activity named {old_name!r} to rename.")
        if new_name == old_name:
            return
        if new_name in self._activities:
            raise ValueError(f"An activity named {new_name!r} already exists.")

        self._data.rename(columns={old_name: new_name}, inplace=True)
        self._activities[self._activities.index(old_name)] = new_name
        self._activity_periods[new_name] = self._activity_periods.pop(old_name)

    def get_activity_data(self, activity_name):
        """Return main data restricted to one or more activity periods.

        Args:
            activity_name (str | Sequence[str]): Name of the activity/boolean mask
                column to use, or multiple names. If multiple are provided, rows
                are returned only where *all* selected activities are True.

        Returns:
            pandas.DataFrame: Copy of the main data for time steps where the
                selected activity (or combined activities) is True, with all
                activity mask columns removed.

        Raises:
            ValueError: If any activity name is not present in self.activities.
            TypeError: If activity_name is neither a string nor an iterable of strings.
        """
        # Normalize to a list of activity names
        if isinstance(activity_name, str):
            activity_names = [activity_name]
        else:
            try:
                activity_names = list(activity_name)
            except TypeError as e:
                raise TypeError(
                    "activity_name must be a string or an iterable of strings."
                ) from e

        if not activity_names:
            raise ValueError("At least one activity name must be provided.")

        # Validate
        missing = [a for a in activity_names if a not in self.activities]
        if missing:
            raise ValueError(
                f"Activity(ies) {missing} not found in available activities: {self.activities}"
            )

        # Combined mask: True only where ALL selected activities are True
        mask = self._data[activity_names].all(axis=1)

        return self._data.loc[mask].drop(columns=self.activities).copy()

    def get_activity_extra_data(self, activity_name):
        """Return extra_data restricted to a given activity period.

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

    def Peak_finder(
        self,
        window: int = 15,
        ratio: float = 2.5,
        method: str = "median",
        specific_data: str = "",
    ):
        """Detect peaks and mark them as an activity labeled 'Peak'.

        Args:
            window (int): Rolling window size in number of samples used to
                compute the local baseline and spread. Defaults to 15.
            ratio (float): Threshold factor on the rolling standard
                deviation. A time step is flagged as peak when
                value - baseline > ratio * rolling_std. Defaults to 2.5.
            method (str): Aggregation used as the rolling baseline, one of
                "mean", "median", "sum", "min", "max". Defaults to "median".
            specific_data (str): Optional column name to use for peak
                detection. If empty, the dataset's primary channel is used
                (total concentration for particle instruments). Otherwise, the
                name is looked up first in self.data (numeric columns) and then
                in extra_data.

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
            Data_return = Data_return._primary
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

    def Mark_threshold(
        self,
        activity: str,
        threshold: Union[int, float],
        metric: Optional[str] = None,
        threshold_direction: str = "above",
    ):
        """Mark every time step whose value lies on the requested side of a
        fixed threshold as an activity, and store the contiguous runs as
        its periods.

            This is a threshold-based segmenter: unlike :meth:`Peak_finder`
            (which compares each point to a moving baseline), it tags samples
            purely by whether they exceed — or fall below — a single, constant
            limit. It is the natural way to flag exposures above an
            occupational exposure limit (OEL) or any other fixed concentration
            cut-off.

        Args:
            activity (str): Name to give the marked activity (e.g. ``"OEL"``).
            threshold (int, float): The fixed limit each sample is compared to.
            metric (str, optional): Column to compare against the threshold. If
                ``None`` (default), the first column of ``self.data`` (the total
                concentration) is used. Otherwise the name is looked up first
                among the numeric columns of ``self.data`` and then of
                ``extra_data``.
            threshold_direction (str): Which side of the threshold counts as the
                activity. ``"above"`` or ``">"`` flags samples strictly greater
                than ``threshold``; ``"below"`` or ``"<"`` flags samples strictly
                less than it. Defaults to ``"above"``.

        Returns:
            None: The method adds/updates a boolean column named ``activity`` in
                ``self.data`` and stores its contiguous runs in
                ``self.activity_periods[activity]``.

        Raises:
            ValueError: If ``metric`` is not found as a numeric column in either
                ``self.data`` or ``extra_data``. Check column names via
                ``data.columns`` or ``extra_data.columns``.
            ValueError: If ``threshold_direction`` is not one of the supported
                names. Choose one of ``"above"``, ``"below"``, ``">"`` or ``"<"``.

        Notes:
            Detailed description:
                The chosen metric is compared element-wise to ``threshold`` to
                build a boolean mask (NaN samples compare False and are never
                flagged). The mask is stored as the ``activity`` column, and its
                contiguous True segments are converted to ``(start, end)`` time
                intervals and stored as the activity's periods.

        Examples:
            Flag the periods where the total concentration exceeds an OEL and
            shade them on the time series:

            .. code-block:: python

                data.Mark_threshold("OEL", threshold=1e4)
                data.plot_total_conc(mark_activities=["OEL"])
                oel_df = data.get_activity_data("OEL")
        """

        Data_return = self.copy_self()

        if metric is None:
            # Default to the dataset's primary channel (total concentration for
            # particle instruments, the main channel otherwise) — matching
            # Peak_finder and removing the "first column = primary" assumption.
            metric = str(getattr(self._primary, "name", None) or self.data.columns[0])

        if metric in self._data.select_dtypes(exclude="bool").columns:
            Data_return = Data_return._data[metric]

        elif metric in self._extra_data.select_dtypes(exclude="bool").columns:
            Data_return = Data_return._extra_data[metric]
        else:
            raise ValueError(
                f"Invalid data title. No column named {metric} can be found."
            )

        # Build the boolean mask for the requested side of the threshold.
        if threshold_direction == "above" or threshold_direction == ">":
            mask = Data_return > threshold
        elif threshold_direction == "below" or threshold_direction == "<":
            mask = Data_return < threshold
        else:
            raise ValueError(
                "Invalid threshold_direction. Either chose 'above' or 'below'"
            )

        peak_col = {activity: mask}

        # Track metadata
        if activity not in self._activities:
            self._activities.append(activity)
            self._data = pd.concat([self.data, pd.DataFrame(peak_col)], axis=1)
        else:
            self._data[activity] = mask

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

        self._activity_periods[activity] = periods

    # snake_case aliases for PEP 8 consistency
    peak_finder = Peak_finder
    mark_threshold = Mark_threshold
