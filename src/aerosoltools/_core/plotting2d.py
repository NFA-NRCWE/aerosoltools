"""Plotting for 2D aerosol data: PSD, PM time series and time-size heatmaps."""

from typing import Any, Optional, Sequence, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray

from . import _shading


class Plot2DMixin:
    """Plot size distributions, PM time series and time-size heatmaps."""

    def _activity_psd_stats(self, activity: str, normalize: bool = True):
        """Mean and std of the (already dtype-resolved) PSD for one activity.

        Internal helper factored out of :meth:`plot_psd` so other code (the
        GUI's PSD tab, which needs the per-bin σ to draw error bars on top of
        bars) can reuse the exact same mean/std computation instead of
        duplicating the normalization logic.

        Args:
            activity: Activity name; must be a boolean column in ``self.data``.
            normalize: Whether to return the log-diameter-normalized
                (dx/dlogDp) form, matching the ``normalize`` flag of
                :meth:`plot_psd`.

        Returns:
            tuple[NDArray, NDArray, NDArray]: ``(bin_mids, mean, std)``, each
                a float array over the size bins. ``mean``/``std`` are all-NaN
                when the activity has no samples.
        """
        is_already_normalized = "/dlogDp" in self.dtype
        bin_columns = self._sizebin_headers
        bin_mids = np.asarray(self.bin_mids, dtype=float)
        log_bin_edges = np.log10(self.bin_edges)
        dlog_dp = np.diff(log_bin_edges)
        factor_series = pd.Series(dlog_dp, index=bin_columns)

        subset = self.data[self.data[activity]]
        if subset.empty:
            nan = np.full(bin_mids.shape, np.nan)
            return bin_mids, nan, nan

        if normalize:
            if not is_already_normalized:
                act_data = subset[bin_columns].copy().div(factor_series, axis=1)
            else:
                act_data = subset[bin_columns].copy()
        else:
            if is_already_normalized:
                act_data = subset[bin_columns].copy().mul(factor_series, axis=1)
            else:
                act_data = subset[bin_columns].copy()

        avg_act = act_data.mean().to_numpy(dtype=float)
        std_act = act_data.std().to_numpy(dtype=float)
        return bin_mids, avg_act, std_act

    def plot_psd(
        self,
        activities: Optional[list[str]] = None,
        normalize: bool = True,
        ax=None,
        dtype: str | None = None,
    ):
        """Description:
            Plot mean particle size distributions for one or more activities.

        Args:
            activities (list[str] | None): Names of activities to include.
                If None, all activities in self.activities are considered.
                Activities that do not exist are skipped.
            normalize (bool): If True, plot PSDs in log-diameter–
                normalized form (dx/dlogDp). If the underlying data are not
                normalized, a temporary division by Δlog₁₀(Dp) is applied.
                If False, PSDs are shown in base units, undoing any stored
                normalisation if needed.
            ax (matplotlib.axes.Axes | None): Axis to plot into. If None,
                a new figure and axes are created.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
                figure and axes with the PSD plot.

        Raises:
            None: Aside from Matplotlib or data consistency errors (for
                example invalid bin_edges or empty activities).

        Notes:
            Detailed description:
                For each selected activity, the method filters rows where
                the activity mask is True, optionally converts the data to
                normalized or base form for plotting, and computes mean
                and standard deviation across time in each size bin. It
                then plots the mean PSD as a line on a logarithmic
                diameter axis with a shaded ±1σ envelope. Colors are
                assigned per activity and a legend is added.

            Theory:
                Log-diameter–normalized PSDs (for example dN/dlogDp) are
                often preferred for visual comparison because equal
                horizontal distances represent equal decades in size. The
                method supports both normalized and base distributions so
                that you can inspect either representation.

        Examples:
            Compare PSDs during two tasks:

            .. code-block:: python

                elpi.mark_activities({
                    "Task A": [("2025-01-24 09:00", "2025-01-24 10:00")],
                    "Task B": [("2025-01-24 10:00", "2025-01-24 11:00")],
                })
                elpi.plot_psd(activities=["Task A", "Task B"], normalize=True)
        """

        clas = self if dtype is None else self.dtype_converter(dtype, False)

        new_fig_created = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            new_fig_created = True
        else:
            fig = ax.figure

        # Set up axes: log diameter on x, grid on both scales
        ax.set_xscale("log")
        ax.set_xlabel("Particle diameter (nm)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Determine current normalization state
        is_already_normalized = "/dlogDp" in clas.dtype
        bin_mids = clas.bin_mids

        # Choose y-label based on requested vs. current normalization
        if normalize and not is_already_normalized:
            y_label_dtype = f"{clas.dtype}/dlogDp"
        elif not normalize and is_already_normalized:
            y_label_dtype = clas.dtype.replace("/dlogDp", "")
        else:
            y_label_dtype = clas.dtype
        ax.set_ylabel(f"{y_label_dtype}, {clas.unit}")

        # Assign colors per activity
        all_activities = sorted(clas._activity_periods.keys())
        color_map = plt.colormaps.get_cmap("gist_ncar")
        activity_colors = {
            activity: color_map(i / max(1, len(all_activities)))
            for i, activity in enumerate(all_activities)
        }

        # Determine which activities to plot
        selected_activities = activities if activities is not None else clas.activities

        for activity in selected_activities:
            if activity not in clas.activities:
                print(f"Activity '{activity}' not found. Skipping.")
                continue

            _, avg_act, std_act = clas._activity_psd_stats(
                activity, normalize=normalize
            )
            if np.all(np.isnan(avg_act)):
                continue
            color = activity_colors.get(activity, None)

            ax.plot(bin_mids, avg_act, label=activity, color=color or "black")
            ax.fill_between(
                bin_mids,
                avg_act - std_act,
                avg_act + std_act,
                color=color or "black",
                alpha=0.3,
            )

        ax.legend()
        if new_fig_created:
            fig.tight_layout()

        return fig, ax

    def plot_PM_timeseries(
        self,
        PM_values: list[float] | None = None,
        dtype: str = "dM",
        activity: str = "All data",
        fraction: bool = False,
        cumulative: bool = False,
        mark_activities: bool | Sequence[str] = False,
    ):
        """Description:
            Plot time series of one or more size-selective Pₓ metrics.

        Args:
            PM_values (list[float]): Cut diameters in µm defining the
                Pₓ series to compute (for example [0.5, 2.5, 10]).
            dtype (str): Base distribution type for Pₓ evaluation, one of
                "dN", "dS", "dV", "dM". Defaults to "dM" (mass-based).
            activity (str): Name of the activity mask selecting which time
                steps to plot. Must be a boolean column in data.
            fraction (bool): If False (default), plot Pₓ in absolute
                units and stack bands between successive PM_values. If
                True, plot the largest Pₓ on the primary axis and the
                fractional contributions of each Pₓ on a secondary axis.
            cumulative (bool): Controls how bands/legend values are
                interpreted:

                    * False: legend reports band-wise contributions between
                      successive PM_values (for example PM10 − PM2.5).
                    * True: legend reports cumulative Pₓ at each cut
                      (for example PM2.5, PM10).

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
                figure and primary axes. When fraction=True, a secondary
                y-axis for fractions is also created.

        Raises:
            Exception: If the number of PM_values exceeds the internal
                color palette. Reduce PM_values or extend the color list.
            KeyError: If activity is not a defined mask column in data.
                Check data.columns and mark_activities/Peak_finder calls.
            ValueError: If dtype is not one of "dN", "dS", "dV", "dM".

        Notes:
            Detailed description:
                The method works on a converted copy of the data (using
                dtype_converter) to compute Pₓ series for each requested
                cut diameter via PM_calc. It then restricts to the chosen
                activity and draws either stacked absolute bands or
                fractional contributions relative to the largest Pₓ. Mean
                ± standard deviation for each series (or band) are shown
                in the legend for quick comparison.

            Theory:
                Pₓ metrics reflect the contribution of different size
                ranges to overall exposure, following EN 481 / ISO 7708
                penetration curves for the chosen base distribution.
                Visualising absolute vs fractional Pₓ helps understand
                whether coarse or fine particles dominate during a task.

        Examples:
            Examine how PM0.5, PM2.5 and PM10 evolve during a shift:

            .. code-block:: python

                fig, ax = elpi.plot_PM_timeseries(
                    PM_values=[0.5, 2.5, 10],
                    dtype="dM",
                    activity="All data",
                    fraction=False,
                )
        """

        if PM_values is None:
            PM_values = [0.5, 2.5, 10]

        # Color palette for stacking PM bands
        colors = [
            "brown",
            "chocolate",
            "darkorange",
            "gold",
            "olive",
            "darkgreen",
            "teal",
            "deepskyblue",
            "darkblue",
        ]
        if len(PM_values) > len(colors):
            raise Exception(
                "Number of PM values are above the limit. Reduce the number of PM-values"
            )

        # Work on a copy to avoid modifying base distributions
        data_copy = self.copy_self()
        data_copy.dtype_converter(dtype=dtype)

        # Compute P-series for each requested PM limit
        for i in PM_values:
            data_copy.PM_calc(dtype=dtype, PM=i)

        # Restrict to selected activity
        mask = self.data[activity]
        PM_data = data_copy.extra_data.loc[mask]

        # Create figure/axes. Tick and legend font sizes come from the shared
        # rcParams set on import (see aerosoltools.aerosol1d), so this figure
        # matches the rest of the library instead of hardcoding sizes.
        figure, ax = plt.subplots()

        # Highlight activities (shared helper; "All data" excluded unless asked).
        if mark_activities and hasattr(self, "_activity_periods"):
            selected = _shading.resolve_activities(
                self._activity_periods, mark_activities
            )
            _shading.shade_activities(ax, self._activity_periods, selected, zorder=3)
            # Clip x-axis to actual data range
            left = float(mdates.date2num(self.time.min()))
            right = float(mdates.date2num(self.time.max()))
            ax.set_xlim(left, right)

        # Fractional mode: total on primary axis, fractions on secondary axis
        if fraction:
            total_name = f"P{dtype[-1]}{PM_values[-1]}"
            total = PM_data[total_name]
            ax.plot(total, color="k", label="Total", lw=3)

            ax2 = ax.twinx()
            for i, pmv in enumerate(PM_values):
                pm = f"P{dtype[-1]}{pmv}"
                # Safe ratio
                ratio = PM_data[pm].where(total != 0, np.nan) / total.where(
                    total != 0, np.nan
                )

                if i == 0:
                    avg, sd = float(PM_data[pm].mean()), float(PM_data[pm].std())
                    ax2.fill_between(
                        self.time[mask],
                        ratio,
                        alpha=0.75,
                        color=colors[i],
                        label=f"{pm}: {avg:.2f}±{sd:.2f}",
                    )
                else:
                    pm_1 = f"P{dtype[-1]}{PM_values[i-1]}"
                    if cumulative:
                        avg, sd = float(PM_data[pm].mean()), float(PM_data[pm].std())
                        ax2.fill_between(
                            self.time[mask],
                            PM_data[pm_1].where(total != 0, np.nan)
                            / total.where(total != 0, np.nan),
                            ratio,
                            alpha=0.75,
                            color=colors[i],
                            label=f"{pm}: {avg:.2f}±{sd:.2f}",
                        )
                    else:
                        band = PM_data[pm] - PM_data[pm_1]
                        avg, sd = float(band.mean()), float(band.std())
                        ax2.fill_between(
                            self.time[mask],
                            PM_data[pm_1].where(total != 0, np.nan)
                            / total.where(total != 0, np.nan),
                            ratio,
                            alpha=0.75,
                            color=colors[i],
                            label=f"{pm}: {avg:.2f}±{sd:.2f}",
                        )

            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.set_ylim(0, 1)
            ax2.set_ylabel(f"P{dtype[-1]} fraction")
            ax2.legend(
                loc="best",
                title=f"Average values ({data_copy.unit})",
            )
        else:
            for i, pmv in enumerate(PM_values):
                pm = f"P{dtype[-1]}{pmv}"
                if i == 0:
                    avg, sd = float(PM_data[pm].mean()), float(PM_data[pm].std())
                    ax.fill_between(
                        self.time[mask],
                        PM_data[pm],
                        alpha=1,
                        color=colors[i],
                        label=f"{pm}: {avg:.2f}±{sd:.2f}",
                    )
                else:
                    pm_1 = f"P{dtype[-1]}{PM_values[i-1]}"
                    if cumulative:
                        avg, sd = float(PM_data[pm].mean()), float(PM_data[pm].std())
                        ax.fill_between(
                            self.time[mask],
                            PM_data[pm_1],
                            PM_data[pm],
                            color=colors[i],
                            label=f"{pm}: {avg:.2f}±{sd:.2f}",
                        )
                    else:
                        band = PM_data[pm] - PM_data[pm_1]
                        avg, sd = float(band.mean()), float(band.std())
                        ax.fill_between(
                            self.time[mask],
                            PM_data[pm_1],
                            PM_data[pm],
                            color=colors[i],
                            label=f"{pm}: {avg:.2f}±{sd:.2f}",
                        )
            ax.legend(
                loc="best",
                title=f"Average values ({data_copy.unit})",
            )

        ax.set_ylim(0)
        ax.set_ylabel(f"P{dtype[-1]}, {data_copy.unit}")
        loc = mdates.AutoDateLocator()
        fmt = mdates.ConciseDateFormatter(loc)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)

        return ax.figure, ax

    def plot_timeseries(
        self,
        y_tot: tuple[float, float] = (0, 0),
        y_3d: tuple[float, float] = (0, 0),
        log: bool = True,
        ax1: Axes | None = None,
        ax2: Axes | None = None,
        mark_activities: bool | Sequence[str] = False,
        dtype: str | None = None,
    ) -> tuple[Figure, NDArray[Any]]:
        """Description:
            Plot total concentration and a time–size heatmap in one figure.

        Args:
            y_tot (tuple[float, float]): Y-limits for the total
                concentration panel (ymin, ymax). Use (0, 0) for automatic
                limits; if a non-zero entry is given with zero partner,
                the non-zero value is used directly, while the zero is
                replaced by the max/min in the data.
            y_3d (tuple[float, float]): Color-scale limits for the 2D PSD
                panel (zmin, zmax). Use (0, 0) for automatic limits. To
                enforce a strictly positive lower limit for log scaling,
                set zmin > 0 and zmax = 0 for automatic upper limit, e.g.
                y_3d = (1, 0).
            log (bool): If True, the function attempts to use a logarithmic
                color scale for the 2D panel. If the PSD values used for
                the mesh (after any clipping from y_3d) are not strictly
                positive or the lower limit is ≤ 0, the function
                automatically falls back to a linear color scale and prints
                a warning to the terminal. If False, a linear color scale
                is used directly.
            ax1 (matplotlib.axes.Axes | None): Axis for the top (total
                concentration) plot. If provided, ax2 must also be
                provided.
            ax2 (matplotlib.axes.Axes | None): Axis for the bottom
                (time–size) plot. If provided, ax1 must also be provided.
            mark_activities (bool | Sequence[str]): Passed to
                plot_total_conc to control activity highlighting. True
                shades all activities except "All data"; a sequence
                restricts shading to specific activities.
            dtype (str | None): Designator for the desired datatype to be
                plotted, independent from current datatype.
                Chose between; 'dN', 'dS', 'dV', 'dM'
        Returns:
            tuple[matplotlib.figure.Figure, numpy.ndarray]: A tuple with
                the figure and a NumPy array [ax1, ax2, colorbar].

        Raises:
            ValueError: If only one of ax1 or ax2 is supplied. Provide
                both or neither.

        Notes:
            Detailed description:
                The method first draws the total concentration time series
                (via plot_total_conc) in the top panel, optionally with
                activity shading and custom y-limits. The bottom panel
                shows a pcolormesh of particle size distribution (PSD)
                with values as a function of time and particle diameter.
                A shared colorbar is added and labeled with the
                current dtype and unit, and the y-axis for the PSD is
                log-scaled in diameter.

                By default, the color scale for the PSD uses a logarithmic
                normalization (log=True). If the PSD values (after any
                clipping via y_3d) include zeros or negatives, or if the
                color-scale lower limit is ≤ 0, the method automatically
                falls back to a linear color scale and prints a clearly
                visible warning to the terminal. To avoid this fallback and
                enforce log scaling, you can specify a strictly positive
                lower limit via y_3d, for example y_3d = (1, 0), which
                clips all values below 1 and lets the method safely use a
                log color scale.

            Theory:
                The heatmap represents the evolution of the size
                distribution over time. Combining this with total
                concentration in one figure makes it easier to relate bulk
                peaks to specific size modes or shifts in the
                distribution.

        Examples:
            Create an overview plot of an ELPI or SMPS data set:

            .. code-block:: python

                fig, (ax1, ax2, cbar) = elpi.plot_timeseries()
                fig.savefig("elpi_timeseries.png", dpi=150)
        """

        clas = self if dtype is None else self.dtype_converter(dtype, False)

        # Require both axes or neither when passing external axes
        if (ax1 is None) != (ax2 is None):
            raise ValueError("You must provide both ax1 and ax2, or neither.")

        # Create figure/axes if not provided
        if ax1 is None and ax2 is None:
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
        else:
            assert ax1 is not None and ax2 is not None
            fig: Figure = ax1.get_figure()  # type: ignore
        ax1 = cast(Axes, ax1)
        ax2 = cast(Axes, ax2)

        time = clas.time
        total = clas.total_concentration
        data = clas.size_data
        bin_edges = clas.bin_edges

        # --- Top panel: total concentration (with optional activity shading) ---
        _, ax_new = clas.plot_total_conc(ax=ax1, mark_activities=mark_activities)
        ax1 = ax_new
        ax1 = cast(Axes, ax1)

        # Optionally fix y-limits of the total concentration plot
        if y_tot != (0, 0):
            ymin = y_tot[0] if y_tot[0] != 0 else total.min() * 0.98
            ymax = y_tot[1] if y_tot[1] != 0 else total.max() * 1.02
            ax1.set_ylim(ymin, ymax)

        # --- Construct time edges for pcolormesh (center ± ½Δt) ---------------
        dt = (time[1] - time[0]) / 2
        time_edges = pd.DatetimeIndex(np.append(time - dt, [time[-1] + dt]))  # type: ignore

        x_grid, y_grid = np.meshgrid(time_edges, bin_edges, indexing="ij")

        # --- Handle color scale limits ----------------------------------------
        z_data = data.copy()
        if y_3d != (0, 0):
            zmin, zmax = y_3d
            if zmin != 0:
                z_data = z_data.clip(lower=zmin)
            if zmax == 0:
                zmax = z_data.max().max()
        else:
            zmin = z_data.min().min()
            zmax = z_data.max().max()

        # --- Choose color normalization (log or linear, with fallback) --------
        use_log = log
        if log:
            # Check for non-positive values or non-positive lower limit
            has_non_positive = (z_data <= 0).any().any()
            if has_non_positive or zmin <= 0:
                # Fallback to linear with a big, clear warning
                print(
                    "\n" + "=" * 72 + "\nWARNING (plot_timeseries):"
                    "\n  PSD data contain zero or negative values, or the lower"
                    "\n  color-scale limit is ≤ 0. Falling back to *linear*"
                    "\n  color scale for the concentration heatmap."
                    "\n"
                    "\n  To enforce log-scaled concentration, set y_3d to use a"
                    "\n  strictly positive lower limit, for example:"
                    "\n      y_3d = (1, 0)"
                    "\n  which clips values below 1 and allows log scaling."
                    "\n" + "=" * 72 + "\n"
                )
                use_log = False

        if use_log:
            # Safety: ensure vmin > 0 for LogNorm
            vmin = max(zmin, np.nextafter(0, 1))
            norm = LogNorm(vmin=vmin, vmax=zmax)
        else:
            norm = Normalize(vmin=zmin, vmax=zmax)

        # --- Bottom panel: size–time pcolormesh -------------------------------
        mesh = ax2.pcolormesh(
            x_grid, y_grid, z_data, cmap="jet", norm=norm, shading="flat"
        )

        # Axis labels and scales
        ax2.set_yscale("log")
        ax2.set_ylabel("Dp, nm")
        ax2.set_xlabel("Time")
        # The panels share the time x-axis, so the top one's "Time" label would
        # just mirror the bottom one — clear it regardless of whether the axes
        # were created here or passed in by the caller.
        ax1.set_xlabel("")
        ax2.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )

        # Shared colorbar for both panels
        col = fig.colorbar(mesh, ax=[ax1, ax2])
        col.set_label(f"{clas.dtype}, {clas.unit}")

        # Basic styling
        ax1.tick_params(axis="y", which="both", direction="out", length=6, width=2)
        ax2.tick_params(axis="y", which="both", direction="out", length=6, width=2)

        return fig, np.array([ax1, ax2, col], dtype=object)
