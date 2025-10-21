from __future__ import annotations

import datetime as dt
from typing import Tuple

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit
from scipy.stats import theilslopes

from .aerosol2d import Aerosol2D

###############################################################################


def Combine_NS_OPS(
    NS_data: Aerosol2D,
    OPS_data: Aerosol2D,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> Aerosol2D:
    """
    Function to combine Nanoscan and OPS data. If no start or end time are specified
    then the function will use the first and last datapoints that exist for both
    instruments.

    Parameters
    ----------
    NS_data : Aerosol2D
        An Aerosol2D data set of a Nanoscan.

    OPS_data : Aerosol2D
        An Aerosol2D data set of an OPS.

    start : datetime, optional
        The starting time for combining the two datasets. If not speicifed, a
        starting point is found as earliest data point of the two datasets. The default is None.

    end : datetime, optional
        The end time for combining the two datasets. If not speicifed, an end
        point is found as earliest data point of the two datasets. The default is None.

    Returns
    -------
    Combined_NS_OPS : Aerosol2D
        A class containing size-resolved particle data and instrument metadata from the combination
        NS and OPS datasets, with newly calculated total concentrations.
        The last bin of the NS has been ignored and the second
        to last has been shortened and its number reduced accordingly.
    """

    # Initialize combination by confirming the data type of the two data sets.
    NS = NS_data.copy_self()
    OPS = OPS_data.copy_self()
    NS.convert_to_number_concentration()
    NS.unnormalize_logdp()
    OPS.convert_to_number_concentration()
    OPS.unnormalize_logdp()

    # Round all the datetime values, so they do not have seconds, in order to ease the alignment process

    NS_time_delta = NS.time[1] - NS.time[0]
    OPS_time_delta = OPS.time[1] - OPS.time[0]
    # Determine the slowest frequency for the purpose of aligning the data.
    freq = max(NS_time_delta, OPS_time_delta)
    freq = f"{freq.seconds}s"

    # Determine the start time for combining the data either from the specified time
    # or from the first datapoint which exist for both instruments
    if start:
        start = start.replace(second=0)
    else:
        start = min(OPS.time[0], NS.time[0])

    # Determine the end time for combining the data either from the specified time
    # or from the last datapoint which exist for both instruments
    if end:
        end = end.replace(second=0)
    else:
        end = max(OPS.time[-1], NS.time[-1])

    assert isinstance(start, dt.datetime)
    assert isinstance(end, dt.datetime)
    # Grab the NS data within the specifed start and end times
    NS_time_matched = NS.timerebin(freq, start, end, inplace=False)
    OPS_time_matched = OPS.timerebin(freq, start, end, inplace=False)

    NS_bins = NS.bin_edges.astype(float)
    OPS_bins = OPS.bin_edges.astype(float)
    OPS0 = float(OPS_bins[0])

    # index j such that NS_bins[j] < OPS0 <= NS_bins[j+1]
    j = int(np.searchsorted(NS_bins, OPS0, side="right") - 1)
    if j < 0 or j >= len(NS_bins) - 1:
        raise ValueError("OPS lower edge is outside the NS bin range.")

    # fraction of the NS bin to keep
    factor_reduction = (OPS0 - NS_bins[j]) / (NS_bins[j + 1] - NS_bins[j])

    # combined edges and mids
    ns_ops_edges = np.concatenate([NS_bins[: j + 1], OPS_bins])
    bin_mids = np.round(0.5 * (ns_ops_edges[:-1] + ns_ops_edges[1:]), 1)
    bin_mids[:j] = NS.bin_mids[:j]

    # Select the relevant columns
    ns_df = NS_time_matched.data.drop(columns=NS_time_matched.data.columns[j + 2 :])
    ops_df = OPS_time_matched.data.drop(columns=["Total_conc"])

    # scale the last kept NS bin by the reduction factor
    last_ns_col = ns_df.columns[-1]
    ns_df[last_ns_col] = ns_df[last_ns_col].astype(float) * factor_reduction

    # --- merge by index (datetimes) ---
    combined = pd.concat([ns_df, ops_df], axis=1)

    # --- rename size-bin columns to string mids in one shot ---
    old_size_cols = list(combined.columns[1 : 1 + len(bin_mids)])
    rename_map = dict(zip(old_size_cols, [str(x) for x in bin_mids], strict=False))
    combined.rename(columns=rename_map, inplace=True)

    # --- total concentration where both instruments have any data at a timestamp ---
    mask = ns_df[ns_df.columns[1:]].notna().any(axis=1) & ops_df[
        OPS._sizebin_headers
    ].notna().any(axis=1)
    # sum only over the size-bin columns (first column is datetime/index)
    sizebin_span = combined.columns[1 : 1 + (len(ns_ops_edges) - 1)]
    combined["Total_conc"] = combined[sizebin_span].sum(axis=1).where(mask, np.nan)

    # --- build the result object and propagate metadata/activities ---
    res = Aerosol2D(combined)
    res._activities = NS.activities
    res._activity_periods = NS.activity_periods
    res._meta["bin_edges"] = ns_ops_edges
    res._meta["bin_mids"] = bin_mids
    res._meta["density"] = NS._meta["density"]
    res._meta["instrument"] = "NS_OPS"
    res._meta["serial_number"] = f"NS: {NS.serial_number}, OPS: {OPS.serial_number}"
    res._meta["unit"] = "cm⁻³"
    res._meta["dtype"] = "dN"
    res._raw_extra_data = pd.concat([NS._raw_extra_data, OPS._raw_extra_data], axis=1)

    return res


###############################################################################


def Plot_correlation(
    X: ArrayLike,
    Y: ArrayLike,
    ax_in: Axes | None = None,
    *,
    intercept: bool = True,
    uniform_scaling: bool = True,
    outlier_influence: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Function to plot the correlation between two sets of values, which have been
    aligned so as to have sensible comparison points.
    X and Y must have the same length. This can be accomplished by using the
    averaging function to generate time associated data of same dimensions.

    Parameters
    ----------
    X: Numpy.array
        First set of values.
    Y: Numpy.array
        Second set of values.
    ax_in : matplotlib.axes.Axes, optional
        Axes to draw on; if None, a new Figure/Axes is created.
    intercept : bool, optional
        If True, fit y = A*x + B; if False, fit y = A*x with B fixed at 0.
    uniform_scaling : bool, optional
        If True, both axes are scaled by a common factor to share limits.
    outlier_influence : bool, optional
        If True, use least-squares (curve_fit). If False, use Theil–Sen slope.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handles for the axis of the plot.
    """

    # --- helpers -------------------------------------------------------------
    def linear_func(
        x: NDArray[np.float64], A: float, B: float = 0.0
    ) -> NDArray[np.float64]:
        return A * x + B

    def r2_score(y_true: NDArray[np.float64], y_fit: NDArray[np.float64]) -> float:
        ss_res = float(np.sum((y_true - y_fit) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        return round(1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0), 3)

    # --- coerce inputs, drop NaN/inf rows -----------------------------------
    x_arr = np.asarray(X)
    y_arr = np.asarray(Y)

    # Handle datetime-like X → float date numbers (Matplotlib expects float)
    if np.issubdtype(x_arr.dtype, np.datetime64):
        x_num = mdates.date2num(pd.to_datetime(x_arr).to_pydatetime())
        x = np.asarray(x_num, dtype=np.float64)
    else:
        # If X is a sequence of Python datetimes
        if x_arr.size > 0 and isinstance(
            x_arr.flat[0], (dt.datetime, np.datetime64, pd.Timestamp)
        ):
            x_num = mdates.date2num(pd.to_datetime(x_arr).to_pydatetime())
            x = np.asarray(x_num, dtype=np.float64)
        else:
            x = np.asarray(x_arr, dtype=np.float64)

    y = np.asarray(y_arr, dtype=np.float64)

    # Stack, filter invalid rows
    z = np.column_stack((x, y)).astype(np.float64, copy=False)
    mask = ~(np.isnan(z).any(axis=1) | np.isinf(z).any(axis=1))
    z = z[mask]
    if z.size == 0:
        # create an empty plot to keep signature behavior predictable
        fig, ax = plt.subplots()
        return fig, ax
    x = z[:, 0]
    y = z[:, 1]

    # --- fit -----------------------------------------------------------------
    # Initialize SEs to avoid "possibly unbound" warnings
    SE_A: float = 0.0
    SE_B: float = 0.0
    A: float
    B: float

    if outlier_influence:
        if intercept:
            params, cov = curve_fit(linear_func, x, y, p0=[1.0, 0.0])
            A, B = float(params[0]), float(params[1])
            # cov can be None if fit fails; guard
            if cov is not None and cov.size >= 2:
                diag = np.sqrt(np.diag(cov))
                SE_A = float(diag[0])
                SE_B = float(diag[1])
        else:
            # fit y = A*x  (B = 0)
            def linear_through_origin(
                xv: NDArray[np.float64], a: float
            ) -> NDArray[np.float64]:
                return a * xv

            params, cov = curve_fit(linear_through_origin, x, y, p0=[1.0])
            A = float(params[0])
            B = 0.0
            if cov is not None and cov.size >= 1:
                SE_A = float(np.sqrt(cov[0, 0]))
    else:
        # Theil–Sen robust fit (returns slope, intercept, lo_slope, hi_slope)
        slope, intercept_ts, _, _ = theilslopes(y, x)
        A = float(slope)  # type: ignore
        B = float(intercept_ts)  # type: ignore
        # keep SE_A/SE_B at 0.0; we don't draw bands for this branch

    y_fit = linear_func(x, A, B)
    r2 = r2_score(y, y_fit)

    # --- scaling and limits --------------------------------------------------
    if uniform_scaling:
        factor = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0))
    else:
        factor = 1.0

    x_min = float(np.min(x) / factor) if np.min(x) <= 0 else 0.0
    x_max = 0.0 if np.max(x) <= 0 else float(np.max(z) / factor)
    fit_x = np.linspace(x_min, x_max, 200, dtype=np.float64)
    fit_y = A * fit_x + (B / factor)

    # Label string
    if B == 0.0:
        label = f"y={round(A, 2)}$\\cdot$x"
    elif B > 0.0:
        label = f"y={round(A, 2)}$\\cdot$x \n + {round(B, 2)}"
    else:
        label = f"y={round(A, 2)}$\\cdot$x \n {round(B, 2)}"

    # --- axes creation -------------------------------------------------------
    if ax_in is None:
        fig, ax = plt.subplots()
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.legend(fontsize=25)
        ax.grid(True)
    else:
        ax = ax_in
        fig = ax.figure

    # --- drawing -------------------------------------------------------------
    ax.plot([x_min, x_max], [x_min, x_max], ls="--", c="k", lw=3)  # 1:1 line
    ax.plot(x / factor, y / factor, "bo")  # points
    ax.plot(fit_x, fit_y, "r-", lw=3)  # fit
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=25, va="top")
    r2_y = 0.80 if B == 0.0 else 0.65
    ax.text(
        0.05,
        r2_y,
        f"r$^2$: {round(r2, 2)}",
        transform=ax.transAxes,
        fontsize=25,
        va="top",
    )

    # Uncertainty band only for the least-squares branch
    if outlier_influence:
        band = np.sqrt((SE_A * fit_x) ** 2 + (SE_B / factor) ** 2)
        ax.fill_between(fit_x, fit_y - band, fit_y + band, alpha=0.33)

    return fig, ax  # type: ignore


###############################################################################
def Plot_correlation_df(df: pd.DataFrame, fig_text: str = "", *Plotsettings):
    """
    Function to plot multiple correlation plots together in a n*n grid, where n is the number of instruments minus 1.
    X and Y must have the same length. This can be accomplished by using the
    averaging function to generate time associated data of same dimensions.

    Parameters
    ----------
    df: dataframe
        A dataframe of the instruments structured with an instrument per column,
        with instrument handle equal to name or location.

    fig_text: str, optional
        If chosen a string is added as a figure text in the unused region of the plot.

    *Plotsettings: list of input to
        Input to the Plot_calibration function:
            intercept=True, uniform_scaling=True, outlier_influence=True
        Call on this
    unifomr_scaling: boolean, optional
        Boolean that can be turned off so as to not scale axis the max value.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    gs : matplotlib.gridspace._subplots.AxesSubplot
        Handles for the gridspace of the plot.
    or
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handles for the axis of the plot.
    """

    instruments = list(df.columns)
    n = 0

    if len(instruments) == 2:
        fig, ax = Plot_correlation(
            df[instruments[0]], df[instruments[1]], *Plotsettings
        )
        fig.figsize = (18, 18)  # type: ignore
        ax.set_ylabel(instruments[1])
        ax.set_xlabel(instruments[0])
    else:
        n = len(instruments) - 1

        fig = plt.figure(figsize=(18, 18), constrained_layout=True)
        gs = fig.add_gridspec(n, n)

        axes = {}
        # Create only the subplots you need (upper triangle incl. diagonal)
        for i in range(n):
            for j in range(i, n):
                ax = fig.add_subplot(gs[j, i])
                axes[(j, i)] = ax

                try:
                    _, _ = Plot_correlation(
                        df[instruments[i]], df[instruments[j + 1]], ax, *Plotsettings
                    )
                except Exception:
                    pass

                if i == 0:
                    ax.set_ylabel(instruments[j + 1])
                if j == n - 1:
                    ax.set_xlabel(instruments[i])
    if len(fig_text) > 0:
        # One description box spanning top row, columns 2..n (i.e., gs[0, 1:])
        if n > 1:
            desc_ax = fig.add_subplot(gs[0, 1:])  # type: ignore # uses only empty cells
            desc_ax.axis("off")
            desc_ax.text(
                0.0,
                0.5,
                fig_text,
                ha="left",
                va="center",
                fontsize=40,
                wrap=True,
                bbox=dict(boxstyle="round,pad=0.6", fc="whitesmoke", ec="0.5", lw=1),
                transform=desc_ax.transAxes,
            )
        else:
            # Fallback if there's only one column in the grid
            ax.text(  # type: ignore
                0.5,
                0.90,
                fig_text,
                ha="center",
                va="top",
                transform=ax.transAxes, # type: ignore
                fontsize=40,  # type: ignore
            )
    plt.tight_layout()
    if n == 0:
        return fig, ax  # type: ignore
    else:
        return fig, gs  # type: ignore
