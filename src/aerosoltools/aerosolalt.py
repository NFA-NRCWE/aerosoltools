from __future__ import annotations

try:
    from typing import override  # Python 3.12+
except ImportError:  # Python ≤ 3.11
    from typing_extensions import override

from typing import Optional, Sequence, Union, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tabulate import tabulate

from .aerosol1d import Aerosol1D

params = {
    "legend.fontsize": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.figsize": (19, 10),
}
plt.rcParams.update(params)


class AerosolAlt(Aerosol1D):
    """Description:
        One-dimensional aerosol time series with multiple scalar channels.

    Args:
        dataframe (pandas.DataFrame):
            Input data with a time column or time index and one or more scalar
            data columns. The first time-like column is converted to a
            DatetimeIndex if needed.

    Notes:
        Detailed description:
            :class:`AerosolAlt` extends :class:`Aerosol1D` for instruments that
            log several scalar channels in the same time series, such as:

            * multiple size-integrated aerosol metrics (e.g. PM1, PM2.5, PM10),
            * LDSA, flow, or other auxiliary variables,
            * environmental or diagnostic signals.

            The underlying data model is the same as for :class:`Aerosol1D`
            (time-indexed 1D series), but plotting and summary methods accept a
            ``parameter`` argument that lets you choose which column in
            :attr:`data` is treated as the “primary” signal.

            Internally, the constructor delegates to :class:`Aerosol1D`:

            * Ensures a DatetimeIndex is present (or creates one from a time
              column).
            * Stores the main time series in ``data``.
            * Keeps non-core channels in ``extra_data`` (if attached later).
            * Carries metadata such as ``instrument``, ``unit`` and ``dtype``
              either as scalars or per-column mappings.

            All activity-handling functionality (e.g. marking activity periods,
            cropping by activity) is inherited from :class:`Aerosol1D` and works
            transparently for any scalar channel.

    Examples:
        A typical workflow is to use :class:`AerosolAlt` for instruments
        with several scalar outputs (e.g. LDSA + flow + flags) and then
        select which one to visualise or summarise:

        .. code-block:: python

            import aerosoltools as at
            import pandas as pd

            # Example: construct from a DataFrame with multiple channels
            df = pd.DataFrame(
                {
                    "Datetime": pd.date_range("2024-01-01", periods=100, freq="1min"),
                    "LDSA": 50.0,
                    "Flow": 2.0,
                    "Flag": 0,
                }
            )

            alt = at.AerosolAlt(df)

            # Plot LDSA as the primary channel
            fig, ax = alt.plot_total_conc(parameter="LDSA")

            # Summarise Flow instead of LDSA
            summary = alt.summarize(parameter="Flow")
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__(dataframe)

    ###########################################################################
    """############################# Functions #############################"""
    ###########################################################################

    @override
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
            if parameter >= len(self._raw_data.columns):
                raise LookupError("Chosen parameter is invalid")
            parameter = self.data.columns[parameter]
        elif isinstance(parameter, str):
            pass
        else:
            raise LookupError("Chosen parameter is invalid")

        new_fig_created = False

        # Create or reuse axes.
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            new_fig_created = True
        else:
            fig = ax.figure

        # Plot the selected data column against time.
        ax.plot(self.time, self.data[parameter], linestyle="-")

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

    ###########################################################################

    def summarize(
        self,
        filename: Optional[str] = None,
        *,
        parameter: Union[int, str] = 0,
    ) -> pd.DataFrame:
        """Description:
            Summarise basic statistics of a selected scalar channel by activity.

        Args:
            filename (str | None, optional):
                Path to an Excel file to write the summary table to. If
                ``None`` (default), the summary is not saved to disk and is
                only printed and returned.
            parameter (int | str, optional):
                Index or column name of the signal to summarise. If an
                ``int`` is given, it is interpreted as a positional index into
                :attr:`data.columns`. If a ``str`` is given, it is interpreted
                as a column label. Defaults to ``0``.

        Returns:
            pandas.DataFrame:
                A summary table with one row per activity (including
                ``"All data"``), containing at least:

                ``["Segment", "Min", "Max", "Mean", "Std", "N datapoints"]``.

        Raises:
            LookupError:
                If ``parameter`` does not correspond to a valid column index or
                column label.

        Notes:
            Detailed description:
                This method extends :meth:`Aerosol1D.summarize` by allowing the
                user to choose which column in :attr:`data` is summarised. It
                uses activity flags previously defined on the object to compute
                statistics for each segment.

                Internally, the method:

                - Resolves ``parameter``:

                  - if integer, used as a positional index into
                    :attr:`data.columns`,
                  - if string, used as a column label,

                  raising :class:`LookupError` if the resolution fails or if
                  the index is out of range.

                - Iterates over all activities in :attr:`activities` (including
                  the default ``"All data"`` segment), and for each:

                  - selects rows belonging to that activity, using either:

                    - an activity mask in :attr:`data[activity]`, and
                    - the selected parameter column,

                  - computes:

                    - minimum,
                    - maximum,
                    - mean,
                    - standard deviation,
                    - number of data points.

                - Collects these into a :class:`pandas.DataFrame` and rounds
                  the numeric values to three decimal places for readability.
                - Prints a nicely formatted version of the table to the
                  console using :func:`tabulate`, labelling columns as
                  ``"Segment"``, ``"Min"``, ``"Max"``, ``"Mean"``, ``"Std"``,
                  and ``"N datapoints"``.
                - If ``filename`` is provided, writes the rounded summary to an
                  Excel file (without index) and prints the output path.

                The method then returns the rounded summary DataFrame so it can
                be used programmatically (e.g. in reports or further analysis).

        Examples:
            A typical use is to summarise different scalar channels for
            predefined activities (e.g. measurement phases, locations,
            scenarios):

            .. code-block:: python

                import aerosoltools as at

                # Load an LDSA-like time series
                alt = at.Load_Partector_file("data/Partector_log.txt")

                # Suppose activities have been marked already on 'alt'
                # Summarise LDSA by activity
                summary_ldsa = alt.summarize(parameter="LDSA")

                # Summarise Flow channel and save to Excel
                summary_flow = alt.summarize(
                    filename="partector_flow_summary.xlsx",
                    parameter="Flow",
                )
        """
        # Resolve which column to summarise based on the requested parameter.
        if isinstance(parameter, int):
            if parameter >= len(self._raw_data.columns):
                raise LookupError("Chosen parameter is invalid")
            parameter = self.data.columns[parameter]
        elif isinstance(parameter, str):
            pass
        else:
            raise LookupError("Chosen parameter is invalid")

        rows: list[list[object]] = []

        # Loop through all activities (including "All data") and collect stats.
        for activity in self.activities:
            try:
                subset = self.data[self.data[activity]][
                    self.total_concentration[parameter].name  # type: ignore[index]
                ]
            except KeyError:
                subset = self.data[self.data[activity]][parameter]

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

        # Build the summary table and round for readability.
        summary = pd.DataFrame(
            rows, columns=["Segment", "Min", "Max", "Mean", "Std", "N datapoints"]
        )
        summary_rounded = summary.round(3)

        # Print a nicely formatted version to the console.
        print("\nSummary of total concentration:\n")
        print(
            tabulate(
                summary_rounded,  # type: ignore
                headers="keys",
                tablefmt="pretty",
                floatfmt=".3f",
            )  # type: ignore[arg-type]
        )

        # Optionally save to Excel.
        if filename:
            summary_rounded.to_excel(filename, index=False)
            print(f"\nSummary saved to: {filename}")

        return summary_rounded
