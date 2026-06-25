"""Plotting for 1D aerosol data (total-concentration time series)."""

from __future__ import annotations

from typing import Sequence, Union, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Plot1DMixin:
    """Plot the total-concentration time series."""

    def plot_total_conc(
        self,
        ax: Axes | None = None,
        mark_activities: bool | Sequence[str] = False,
        parameter: Union[int, str] = 0,
    ) -> tuple[Figure, Axes]:
        """Description:
            Plot a selected scalar channel versus time for an :class:`AerosolAlt`
            object.

        Args:
            ax (matplotlib.axes.Axes | None, optional):
                Existing Matplotlib axes to draw on. If ``None``, a new figure
                and axes are created. Defaults to ``None``.
            mark_activities (bool | Sequence[str], optional):
                Control highlighting of activity periods defined on the object:

                * ``False`` – no shading (default).
                * ``True`` – shade all activities except ``"All data"``.
                * sequence of str – shade only the named activities.

            parameter (int | str, optional):
                Index or column name of the signal to plot. If ``int``, it is
                interpreted as a positional index into :attr:`data.columns`. If
                ``str``, it is treated as a column label. Defaults to ``0``.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                The figure and axes containing the time-series plot.

        Raises:
            LookupError:
                If ``parameter`` does not correspond to a valid column index or
                column label.

        Notes:
            Detailed description:
                This method overrides :meth:`Aerosol1D.plot_total_conc` to allow
                plotting of *any* scalar column stored in :attr:`data`, not only
                ``"Total_conc"``. It is particularly useful when an instrument
                logs multiple scalar metrics alongside time.

                Internally, the method:

                * Resolves ``parameter``:

                  - if an integer, it is used as a positional index into
                    :attr:`data.columns`,
                  - if a string, it must match a column label.

                * Creates or reuses a Matplotlib axes (depending on ``ax``).
                * Plots the selected column against the object’s time index.
                * Configures the x-axis with a concise datetime formatter via
                  :mod:`matplotlib.dates`.
                * Determines the appropriate ``dtype`` and ``unit``:

                  - if global (scalar), they are used as-is;
                  - if per-column mappings, the entry for the chosen
                    ``parameter`` is used.

                  The y-axis label is constructed from the base dtype
                  (e.g. ``"dN"`` from ``"dN/dlogDp"`` when applicable)
                  and the corresponding unit.

                * Optionally shades activity periods defined by
                  :meth:`Aerosol1D.mark_activities` when ``mark_activities`` is
                  ``True`` or a list of activity names. Each activity receives
                  a distinct colour, and overlapping shaded regions are clipped
                  to the data’s time extent.

                * Calls ``fig.tight_layout()`` when it creates the figure to
                  ensure margins and labels do not overlap.

        Examples:
            A typical use is to visualise one of several channels stored in
            an :class:`AerosolAlt` object, optionally with activity periods
            highlighted:

            .. code-block:: python

                import aerosoltools as at

                # Suppose 'alt' is an AerosolAlt with columns: ["LDSA", "Flow", "Flag"]
                alt = at.Load_Partector_file("data/Partector_log.txt")

                # Plot LDSA with activity shading
                fig, ax = alt.plot_total_conc(
                    parameter="LDSA",
                    mark_activities=True,
                )

                # Plot Flow on existing axes without activity shading
                fig2, ax2 = plt.subplots()
                alt.plot_total_conc(ax=ax2, parameter="Flow", mark_activities=False)
        """
        # Resolve which column to use based on the requested parameter.
        if isinstance(parameter, int):
            if parameter >= len(self.data.columns):
                raise LookupError("Chosen parameter is invalid")
            parameter = self.data.columns[parameter]
        elif isinstance(parameter, str):
            if parameter not in self._data and parameter not in self._extra_data:
                raise LookupError(f"Chosen parameter '{parameter}' is invalid")
        else:
            raise LookupError("Chosen parameter is invalid")

        # if isinstance(parameter, int):
        #     if parameter >= len(self._raw_data.columns):
        #         raise LookupError("Chosen parameter is invalid")
        #     parameter = self.data.columns[parameter]
        # elif isinstance(parameter, str):
        #     pass
        # else:
        #     raise LookupError("Chosen parameter is invalid")

        new_fig_created = False

        # Create or reuse axes.
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            new_fig_created = True
        else:
            fig = ax.figure

        # Plot the selected data column against time.
        if parameter in self._data:
            ax.plot(self.time, self.data[parameter], linestyle="-")
        elif parameter in self._extra_data:
            ax.plot(self.time, self.extra_data[parameter], linestyle="-")
        else:
            raise KeyError(f"Parameter '{parameter}' not found in data or extra_data")

        # Format x-axis as dates.
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel("Time")

        # Resolve dtype/unit for the chosen parameter (can be scalar or per-column).
        if isinstance(self.dtype, str):
            Dtype = self.dtype
        else:
            Dtype = self.dtype[parameter]

        if isinstance(self.unit, str):
            Unit = self.unit
        else:
            Unit = self.unit[parameter]  # type: ignore[index]

        if "/" in Dtype:
            total_conc_dtype = Dtype.split("/")[0]
            ax.set_ylabel(f"{total_conc_dtype}, {Unit}")
        else:
            ax.set_ylabel(f"{Dtype}, {Unit}")
        ax.grid(True)

        # Optionally highlight activity periods as shaded regions.
        if mark_activities and hasattr(self, "_activity_periods"):
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

            # Clamp x-limits to the actual data range and show legend.
            left = float(mdates.date2num(self.time.min()))
            right = float(mdates.date2num(self.time.max()))
            ax.set_xlim(left, right)
            ax.legend()

        if new_fig_created:
            fig.tight_layout()  # type: ignore[call-arg]

        return fig, ax  # type: ignore[return-value]
