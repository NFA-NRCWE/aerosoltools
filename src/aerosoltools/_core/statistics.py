"""Activity and exposure summary statistics for 1D aerosol data."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
from tabulate import tabulate

from . import _stats


class SummaryMixin:
    """Per-activity and exposure (STEL/TWA) summaries."""

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
            series = self._primary.astype(float)
            return series, self.unit_of(getattr(self._primary, "name", None))

        # Look up in main data first, then extra_data
        if metric_name in self._data.select_dtypes(exclude="bool").columns:
            series = self._data[metric_name].astype(float)
        elif metric_name in self._extra_data.select_dtypes(exclude="bool").columns:
            series = self._extra_data[metric_name].astype(float)
        else:
            raise ValueError(
                f"Metric '{metric_name}' not found in main data or extra_data."
            )

        # Per-column unit (multi-channel instruments carry a per-column dict).
        return series, self.unit_of(metric_name)

    def summarize_activities(
        self,
        filename: Optional[str] = None,
        sheet_name: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        stats: Optional[Sequence[str]] = None,
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
            stats (Sequence[str] | None): Which per-activity statistics to
                report for each metric. If None, defaults to
                ``["mean", "std"]`` (the historical behaviour). Each entry
                must be one of "mean", "std", "min", "max", "median".

        Returns:
            pandas.DataFrame: Summary table with one row per activity and
            columns:

                * "Segment"
                * "Duration (HH:MM)"
                * For each metric M and requested stat S: "M [unit] S".

        Raises:
            ValueError: If a requested metric (other than "PNC") cannot
                be found in data or extra_data, or if ``stats`` contains an
                unrecognised name.

        Notes:
            The per-activity statistics are simple sample statistics over
            the selected time steps. Durations are based on the actual
            sampling intervals (via _dt_minutes).

        Examples:

            .. code-block:: python

                data.summarize_activities()
                data.summarize_activities(stats=["mean", "min", "max"])
        """

        # Defaults: only PNC if nothing else is specified
        if metrics is None:
            metrics = ["PNC"]
        if stats is None:
            stats = ["mean", "std"]

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
                values = _stats.compute_stats(s, stats)
                row += [round(values[stat], 3) for stat in stats]

            rows.append(row)

        # Build column labels
        columns: list[str] = ["Segment", "Duration (HH:MM)"]
        for name in metrics:
            unit = metric_units[name]
            label = f"{name} [{unit}]"
            columns += [f"{label} {stat}" for stat in stats]

        summary = pd.DataFrame(rows, columns=columns)

        # Console output (transposed for readability)
        if not summary.empty:
            summary_t = summary.set_index("Segment").T
            print("\nSummary of aerosol metrics per activity:\n")
            print(tabulate(summary_t, headers="keys", tablefmt="pretty", floatfmt=".3f"))  # type: ignore

        # Optional file output (append if exists)
        if filename and not summary.empty:
            fname = str(filename)
            lower = fname.lower()
            if sheet_name:
                shname = str(sheet_name)
            else:
                shname = f"{self.instrument} summary"

            if lower.endswith(".csv"):
                if os.path.exists(fname):
                    summary.to_csv(fname, mode="a", header=False, index=False)
                else:
                    summary.to_csv(fname, mode="w", header=True, index=False)
            elif lower.endswith((".xlsx", ".xls")):
                if os.path.exists(fname):
                    with pd.ExcelWriter(
                        filename,
                        engine="openpyxl",
                        mode="a",
                        if_sheet_exists="replace",  # requires pandas ≥ 1.4
                    ) as writer:
                        summary.to_excel(writer, sheet_name=shname, index=False)
                else:
                    summary.to_excel(fname, sheet_name=shname, index=False)
            else:
                raise ValueError(
                    f"Unsupported file extension for '{filename}'. Use .csv or .xlsx."
                )

            print(f"\nActivity summary appended to: {filename}")

        return summary

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
