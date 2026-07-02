"""Activity and exposure summaries for size-resolved (2D) data."""

import os
from typing import Any, Iterable, Mapping, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
from tabulate import tabulate

from . import _stats

try:
    from typing import override  # Python 3.12+
except ImportError:  # pragma: no cover - typing_extensions fallback
    from typing_extensions import override  # noqa: F401


class Summary2DMixin:
    """Activity and exposure summaries for size-resolved (2D) data."""

    @override
    def summarize_activities(
        self,
        filename: Optional[str] = None,
        sheet_name: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        stats: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Description:
            Summarize size-resolved aerosol metrics per activity.

        Args:
            filename (str | None): Optional Excel file path. If provided,
                the summary table is written to this file (one sheet,
                activities as rows). If None, no file is written.
            sheet_name (str | None): Optional sheet name. If provided
                the data is written in a sheet named as sheet_name.
                If one already exsits it overwrites the exsisitng sheet.
            metrics (list[str] | None): List of metric names to compute.
                If None, a default set is used: ["PNC", "PM1", "PM2.5",
                "PM4", "PM10", "MASS", "MODE", "MEDIAN", "GMD"].
            stats (Sequence[str] | None): Which per-activity statistics to
                report for each metric. If None, defaults to
                ``["mean", "std"]`` (the historical behaviour). Each entry
                must be one of "mean", "std", "min", "max", "median". The
                "mean" column keeps its unsuffixed name for backward
                compatibility; any other stat is suffixed, e.g. "... min".

        Returns:
            pandas.DataFrame: Summary table with:
                * "Segment"
                * "Duration (HH:MM)"
                * For each metric M and requested stat S: "M [unit]" (for
                  S="mean") or "M [unit] S" (otherwise).

        Raises:
            ValueError: If a metric name cannot be interpreted (for
                example malformed PMx string) or is unsupported.
            ValueError: If internal preparation for a Pₓ metric fails
                (for example missing PSD columns or inconsistent bin
                metadata), or if ``stats`` contains an unrecognised name.

        Notes:
            Detailed description:
                For each activity, the method computes the total duration
                and the requested set of metrics. PNC and MASS are total
                number and mass concentrations. MODE, MEDIAN and GMD are
                size metrics derived from the number distribution. Metrics
                of the form PMx, PNx, PSx, PVx are Pₓ values at cut
                diameter x (in µm) for mass, number, surface area and
                volume, respectively. Each metric is reported as mean and
                standard deviation over the activity. A transposed version
                of the table is printed to the terminal for quick inspection.

            Theory:
                The method combines time-weighted activity durations with
                standard PSD-derived metrics: bulk concentrations, Pₓ, and
                central size metrics (mode, median, geometric mean).
                Pₓ metrics are determined using penetration curves given in
                EN 481 / ISO 7708, mimicing cyclone cut-offs rather than
                sharp size cut-offs for PMₓ calculations.

        Examples:
            Generate a task-level summary of exposure metrics:

            .. code-block:: python

                elpi.summarize_activities(
                    filename="activity_summary_elpi.xlsx",
                    metrics=["PNC", "PM2.5", "PM10", "MODE", "GMD"],
                )
        """

        # --- defaults --------------------------------------------------------
        if metrics is None:
            metrics = [
                "PNC",
                "PM1",
                "PM2.5",
                "PM4",
                "PM10",
                "MASS",
                "MODE",
                "MEDIAN",
                "GMD",
            ]
        if stats is None:
            stats = ["mean", "std"]

        # --- helper: duration in minutes per time step (shared helper) -------
        dt_mins = self._dt_minutes()

        def _px_label(dchar: str, cutoff: float, lower_lim: float = 0) -> str:
            if lower_lim == 0:
                return f"P{dchar}{cutoff:g}"
            else:
                return f"P{dchar}{lower_lim:g}-{cutoff:g}"

        metrics_upper = [m.upper() for m in metrics]

        want_pnc = "PNC" in metrics_upper
        want_mass = "MASS" in metrics_upper
        want_mode = "MODE" in metrics_upper
        want_median = "MEDIAN" in metrics_upper
        want_gmd = "GMD" in metrics_upper
        want_any_size_metric = want_mode or want_median or want_gmd

        # Gather requested Pₓ cutoffs by dtype char (M/N/S/V) using shared parser
        pm_requests: dict[str, set[float]] = {
            "M": set(),
            "N": set(),
            "S": set(),
            "V": set(),
        }
        for name_upper in metrics_upper:
            parsed = self._parse_px_metric_scalar(name_upper)
            if parsed:
                dchar, cutoff, lower_lim = parsed
                pm_requests[dchar].add((cutoff, lower_lim))

        # --- prepare prerequisite data only if needed ------------------------
        # Extract the base array once and convert to target dtypes via numpy,
        # avoiding expensive full-object deep copies.
        dtype_map = {"M": "dM", "N": "dN", "S": "dS", "V": "dV"}
        current_base = str(self.dtype).replace("/dlogDp", "")

        number_df = None
        mass_df = None
        if want_pnc or want_any_size_metric or want_mass:
            base_arr, _, _ = self._as_base_array()

        if want_pnc or want_any_size_metric:
            if current_base != "dN":
                number_arr = self._convert_array(
                    base_arr, self.bin_mids, current_base, "dN", self.density
                )
            else:
                number_arr = base_arr
            number_df = pd.DataFrame(
                number_arr, index=self.time, columns=self._sizebin_headers
            )

        if want_mass:
            if current_base != "dM":
                mass_arr = self._convert_array(
                    base_arr, self.bin_mids, current_base, "dM", self.density
                )
            else:
                mass_arr = base_arr
            mass_df = pd.DataFrame(
                mass_arr, index=self.time, columns=self._sizebin_headers
            )

        # Precompute Pₓ series directly via _px_fraction_series (no copy_self needed)
        px_extra: dict[str, dict[str, pd.Series]] = {}
        for dchar, cutoffs in pm_requests.items():
            if not cutoffs:
                continue
            px_extra[dchar] = {}
            for pm, lower_lim in sorted(cutoffs):
                label = _px_label(dchar, pm, lower_lim)
                px_extra[dchar][label] = self._px_fraction_series(
                    dtype=dtype_map[dchar], upper=pm, lower=lower_lim
                )

        # --- compute per-activity --------------------------------------------
        rows: list[list[float | str]] = []
        bin_mids = np.asarray(self.bin_mids, dtype=float)

        for activity in self.activities:
            mask = self.data[activity]
            if mask.sum() == 0:
                continue

            # Duration of this activity (min and HH:MM)
            duration_minutes = float(dt_mins.loc[mask].sum())
            duration_hhmm = self._format_hhmm(duration_minutes)

            # Initialize with NaNs
            nan_stats = {stat: float("nan") for stat in stats}
            pnc_stats = dict(nan_stats)
            mass_stats = dict(nan_stats)
            mode_stats = dict(nan_stats)
            median_stats = dict(nan_stats)
            gmd_stats = dict(nan_stats)

            # PNC (from number_df)
            if want_pnc and number_df is not None:
                num_act = number_df.loc[mask]
                s = self._ensure_data_robustness(num_act.sum(axis=1))
                pnc_stats = _stats.compute_stats(s, stats)

            # Total mass (from mass_df)
            if want_mass and mass_df is not None:
                mass_act = mass_df.loc[mask]
                s = self._ensure_data_robustness(mass_act.sum(axis=1))
                mass_stats = _stats.compute_stats(s, stats)

            # Size metrics (mode/median/GMD) on number distribution
            if want_any_size_metric and number_df is not None:
                num_act = number_df.loc[mask]

                mode_list: list[float] = []
                med_list: list[float] = []
                gmd_list: list[float] = []

                for _, row in num_act.iterrows():  # type: ignore
                    dist = row.to_numpy(dtype=float)  # type: ignore
                    tot = float(dist.sum())
                    if tot <= 0:
                        continue

                    if want_mode:
                        mode_idx = int(np.argmax(dist))
                        mode_list.append(float(bin_mids[mode_idx]))

                    if want_median:
                        cum = np.cumsum(dist)
                        cum /= cum[-1]
                        med_idx = int(np.searchsorted(cum, 0.5))
                        med_list.append(float(bin_mids[med_idx]))

                    if want_gmd:
                        positive = dist > 0
                        if not np.any(positive):
                            continue
                        dpos = dist[positive]
                        mpos = bin_mids[positive]
                        gval = float(np.exp(np.sum(np.log(mpos) * dpos) / dpos.sum()))
                        gmd_list.append(gval)

                if want_mode and mode_list:
                    mode_stats = _stats.compute_stats(pd.Series(mode_list), stats)
                if want_median and med_list:
                    median_stats = _stats.compute_stats(pd.Series(med_list), stats)
                if want_gmd and gmd_list:
                    gmd_stats = _stats.compute_stats(pd.Series(gmd_list), stats)

            # Pₓ metrics collected in requested order
            px_values: dict[str, dict[str, float]] = {}
            for name, name_upper in zip(metrics, metrics_upper):
                parsed = self._parse_px_metric_scalar(name_upper)
                if not parsed:
                    continue
                dchar, cutoff, lower_lim = parsed
                dchar_dict = px_extra.get(dchar)
                if dchar_dict is None:
                    raise ValueError(
                        f"Internal error: dtype '{dchar}' was not prepared."
                    )
                label = _px_label(dchar, cutoff, lower_lim)
                if label not in dchar_dict:
                    raise ValueError(f"Internal error: PX column '{label}' not found.")
                ser = dchar_dict[label].loc[mask]
                px_values[label] = _stats.compute_stats(ser, stats)

            # Assemble row
            row: list[float | str] = [activity, duration_hhmm]
            for name, name_upper in zip(metrics, metrics_upper):
                if name_upper == "PNC":
                    row += [round(pnc_stats[stat], 2) for stat in stats]
                elif name_upper == "MASS":
                    row += [round(mass_stats[stat], 2) for stat in stats]
                elif name_upper == "MODE":
                    row += [round(mode_stats[stat], 1) for stat in stats]
                elif name_upper == "MEDIAN":
                    row += [round(median_stats[stat], 1) for stat in stats]
                elif name_upper == "GMD":
                    row += [round(gmd_stats[stat], 1) for stat in stats]
                else:
                    parsed = self._parse_px_metric_scalar(name_upper)
                    if not parsed:
                        raise ValueError(f"Unsupported metric '{name}'.")
                    dchar, cutoff, lower_lim = parsed
                    label = _px_label(dchar, cutoff, lower_lim)
                    vals = px_values[label]
                    row += [round(vals[stat], 2) for stat in stats]
            rows.append(row)

        # --- column headers with explicit units ---------------------------------
        columns: list[str] = ["Segment", "Duration (HH:MM)"]

        unit_map_px = {
            "M": "µg/m³",
            "N": "cm⁻³",
            "S": "nm²/cm³",
            "V": "nm³/cm³",
        }

        for name, name_upper in zip(metrics, metrics_upper):
            parsed = self._parse_px_metric_scalar(name_upper)
            if parsed:
                dchar, _, _ = parsed
                unit = unit_map_px[dchar]
                label = f"{name} [{unit}]"
            elif name_upper == "PNC":
                label = "PNC [cm⁻³]"
            elif name_upper == "MASS":
                label = "Total Mass [µg/m³]"
            elif name_upper == "MODE":
                label = "Mode Dp [nm]"
            elif name_upper == "MEDIAN":
                label = "Median Dp [nm]"
            elif name_upper == "GMD":
                label = "GMD [nm]"
            else:
                label = name
            # "mean" keeps the bare label (historical default columns are
            # unchanged); any other stat gets an explicit suffix.
            columns += [
                label if stat == "mean" else f"{label} {stat}" for stat in stats
            ]

        summary = pd.DataFrame(rows, columns=columns)
        # --- append to file if requested ---------------------------------------
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
            print(f"\nActivity summary saved to: {filename}")

        summary_t = summary.set_index("Segment").T
        print("\nSummary of aerosol properties (transposed):\n")
        print(tabulate(summary_t, headers="keys", tablefmt="pretty", floatfmt=".3f"))  # type: ignore

        return summary

    @override
    def summarize_exposure(
        self,
        metric: str = "PM4.2",
        background: Union[float, str, None] = None,
        exposure_hours: Optional[float] = None,
        short_limit: float = 1.0,
        long_limit: float = 1.0,
        short_window: str = "15min",
        twa_window: str = "8h",
        peak_ratio: float = 2.5,
        filename: Optional[str] = None,
        sheet_name: Optional[str] = None,
        activities: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Description:
            Summarize exposure metrics for one PSD-derived metric across activities.

        Args:
            metric (str): Exposure metric name derived from the underlying
                particle size distribution (PSD). Default is "PM4.2",
                corresponding to respirable dust.Supported forms:

                    - "PNC": total number concentration.
                    - "MASS": total mass concentration.
                    - "PM<x>", "PN<x>", "PS<x>", "PV<x>": cumulative Pₓ at cut
                      diameter <x> in µm for mass, number, surface, volume
                      (for example "PM4.2").
                    - "PM<a>-<b>", "PN<a>-<b>", "PS<a>-<b>", "PV<a>-<b>":
                      band-limited Pₓ between diameters <a> and <b> in µm
                      (for example "PM1-4", "PM1-4.2"). These are computed
                      using the EN 481 / ISO 7708 penetration curves.

            background (float | str | None): Background level used when
                computing the time-weighted average over ``twa_window``.
                The same background level is used for all activities in
                the output. Default is None. Possible entries are:

                    * None: assume zero background.
                    * float: constant background level in metric units.
                    * str: name of an activity; the TWA of ``metric`` over
                      that activity is used as background.

            exposure_hours (float | None): Assumed duration of exposure for
                each activity, in hours, when embedding it into the TWA
                window. If None, the measured activity duration is used for
                that activity. If a positive value is given, the same
                exposure duration is applied for all activities.
                Default is None.

            short_limit (float): Short-term concentration limit in metric
                units (for example a 15-min STEL). This value is reported in
                the output as ``"STEL [unit]"``. Default is 1.0 (in metric
                units).

            long_limit (float): Long-term concentration limit in metric
                units (for example an 8-h OEL). This value is reported in the
                output as ``"Exposure limit [unit]"``. Default is 1.0 (in
                metric units).

            short_window (str): Rolling window used for short-term (STEL)
                evaluation, given as a pandas offset string (for example
                "15min"). This is reported as ``"STEL window"``.
                Default is "15min" (15 minutes).

            twa_window (str): Total duration of the TWA window as a pandas
                offset string (for example "8h"). This is reported as
                ``"TWA window"``. Default is "8h" (8 hours).

            peak_ratio (float): Factor used in peak detection; peaks are
                flagged when the metric exceeds::

                    baseline + peak_ratio * rolling_std,

                where ``baseline`` is a rolling median and ``rolling_std`` is
                a rolling standard deviation over a short window.
                Default is 2.5.

            filename (str | None): Optional path to a CSV/Excel file to which
                the non-transposed result rows are appended. If the file
                exists, rows are appended; otherwise the file is created with
                a header. Supported extensions are ".csv", ".xls", ".xlsx".

            activities (Sequence[str] | None): Activities to summarize.
                If None (default), all defined activities in
                :attr:`activities` are summarized (for example "All data",
                "Background", "Emission", "Decay"). Activities with no
                marked time steps are skipped.

        Returns:
            pandas.DataFrame: One row per activity segment with summary
            statistics for the chosen metric. Column names embed their units
            in square brackets. See Notes below (Detailed description) for a
            complete list of columns.

        Raises:
            ValueError: If ``metric`` cannot be parsed (unsupported string).
            ValueError: If a background activity name is given but does not
                exist or has no samples.
            ValueError: If ``short_window`` or ``twa_window`` cannot be
                parsed as pandas-style durations.
            ValueError: If ``exposure_hours`` is negative.
            TypeError: If ``background`` is not None, float or str.

        Notes:
            Detailed description:
                The method first derives the requested metric time series
                from the underlying 2D PSD (for example PNC from a number
                distribution, PM4.2 from a mass-based Pₓ, or a band metric
                such as PM1–4). Where possible, existing Pₓ series already
                stored in :attr:`extra_data` are reused; otherwise they are
                computed on a working copy and then used to build the metric
                time series.

                For each selected activity, the method:

                    1. Extracts the metric time series within the activity
                       using the activity mask.
                    2. Computes per-step durations from the actual sampling
                       times.
                    3. Forms a time-weighted segment mean, used internally
                       when embedding the activity into the TWA window.
                    4. Computes instantaneous percentiles, peaks, STEL-style
                       exceedances and long-term time-above-limit measures.
                    5. Combines the segment mean with the specified background
                       level to obtain an overall TWA for the chosen window.

                The returned DataFrame contains the following columns
                (per activity):

                    - "Segment"
                        Name of the activity segment.

                    - "Metric"
                        Name of the metric that was summarized, e.g.
                        "PM4.2", "PM1-4", "PNC", "MASS".

                    - "Duration [HH:MM]"
                        Duration of the activity derived from the actual
                        sampling intervals, formatted as "HH:MM" (hours and
                        minutes).

                    - "Max [unit]"
                        Maximum of the instantaneous metric values during the
                        activity. The unit is the metric unit (e.g. "µg/m³",
                        "cm⁻³", "nm²/cm³", "nm³/cm³").

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

                            baseline + peak_ratio * rolling_std,

                        with ``baseline`` the rolling median and
                        ``rolling_std`` the rolling standard deviation over a
                        short sliding window, covering at least 3 datapoints
                        and maximum 15 datapoints, depending on the number
                        of available datapoints in the given segment.

                    - "STEL [unit]"
                        Short-term exposure limit used in the STEL
                        exceedance calculations (echoing ``short_limit``),
                        with the same concentration unit as the metric.

                    - "STEL window"
                        Rolling window length used for STEL evaluation,
                        echoing ``short_window`` (for example "15min").

                    - "STEL exceedance [min]"
                        Total time in minutes during the activity where the
                        **full-window** rolling mean is greater than or equal
                        to the STEL. Only complete windows are counted in
                        this measure.

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
                        the full TWA window (``"TWA window"``), combining the
                        internal segment mean (optionally stretched to
                        ``exposure_hours``) with the specified background
                        level for the remainder of the window::

                            TWA = (segment_mean * exposure_minutes
                                   + background * (twa_minutes - exposure_minutes))
                                  / twa_minutes

                        where ``twa_minutes`` is the duration of
                        ``twa_window`` in minutes and ``exposure_minutes`` is
                        either the measured activity duration or
                        ``exposure_hours * 60``, whichever is used.

                    - "TWA window"
                        Duration of the TWA window over which
                        "TWA concentration [unit]" is evaluated, echoing
                        ``twa_window`` (for example "8h").

            Theory:
                Conceptually, the method mirrors typical occupational hygiene
                practice:

                    * Exposure within an activity is represented by
                      time-weighted averages and distributional
                      descriptors (max and percentiles).
                    * Short-term limits (STEL) are evaluated using rolling
                      means over a specified window, counting both total time
                      above the limit and distinct exceedance episodes.
                    * Long-term limits are represented by cumulative
                      time-above-limit measures.
                    * An overall TWA for an 8-h (or user-specified) reference
                      period is constructed by mixing the task exposure with a
                      background level for the rest of the window.

                This ensures a transparent link between raw time series
                behaviour and regulatory style metrics such as STEL and
                8-hour OELs.

        Examples:
            Summarize respirable PM4.2 exposure for all activities in an
            ELPI dataset, using the "Background" activity as background
            level and a 15-min STEL with an 8-h TWA window:

            .. code-block:: python

                elpi.summarize_exposure(
                    metric="PM4.2",
                    background="Background",
                    short_limit=5.0,
                    long_limit=10.0,
                    short_window="15min",
                    twa_window="8h",
                )

            Band-limited metrics are handled in the same way, e.g. for a
            PM1–4.2 band::

            .. code-block:: python

                elpi.summarize_exposure(metric="PM1-4.2")
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
            if sheet_name:
                shname = str(sheet_name)
            else:
                shname = f"{metric} summary"

            if lower.endswith(".csv"):
                if os.path.exists(fname):
                    result.to_csv(fname, mode="a", header=False, index=False)
                else:
                    result.to_csv(fname, mode="w", header=True, index=False)
            elif lower.endswith((".xlsx", ".xls")):
                if os.path.exists(fname):
                    with pd.ExcelWriter(
                        filename,
                        engine="openpyxl",
                        mode="a",
                        if_sheet_exists="replace",  # requires pandas ≥ 1.4
                    ) as writer:
                        result.to_excel(writer, sheet_name=shname, index=False)
                else:
                    result.to_excel(fname, sheet_name=shname, index=False)
            else:
                raise ValueError(
                    f"Unsupported file extension for '{filename}'. Use .csv or .xlsx."
                )
            print(f"\nExposure summary saved to: {filename}")

        # --- pretty terminal print (transposed) --------------------------------
        if not result.empty:
            result_t = result.set_index("Segment").T
            display_df = result_t.reset_index().rename(columns={"index": "Metric"})
            table_data = cast(
                Mapping[str, Iterable[Any]],
                display_df.to_dict(orient="list"),
            )

            print(f"\nExposure summary by segment for metric '{metric}' ({unit}):\n")
            print(
                tabulate(
                    table_data,
                    headers="keys",
                    tablefmt="pretty",
                    floatfmt=".3f",
                )
            )

        return result
