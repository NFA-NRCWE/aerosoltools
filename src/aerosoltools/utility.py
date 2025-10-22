from __future__ import annotations

import datetime as dt
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from numpy.typing import NDArray
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

# =========================
# Small utilities
# =========================


def _ts(x) -> pd.Timestamp:
    """Coerce strings/datetime/np.datetime64 to pandas Timestamp."""
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, (dt.datetime, dt.date, np.datetime64, str)):
        return pd.to_datetime(x)
    return pd.to_datetime(x)


def _infer_freq(idx: pd.DatetimeIndex) -> Optional[str]:
    """Infer 'S', '1T', '1H' from a DatetimeIndex; fallback via median delta."""
    if len(idx) < 3:
        return None
    f = pd.infer_freq(idx)
    if f:
        return f
    d = np.diff(idx.view("i8"))  # ns
    if d.size == 0:
        return None
    sec = int(round(np.median(d) / 1e9))
    if sec < 60:
        return f"{max(1, sec)}S"
    if sec < 3600:
        return f"{max(1, sec // 60)}T"
    return f"{max(1, sec // 3600)}H"


def _coarser(rule_a: str, rule_b: str) -> str:
    """Return the coarser (larger) cadence among two rules (S/T/H/D-ish)."""

    def to_s(rule: str) -> float:
        r = rule.upper()
        num = "".join(ch for ch in r if ch.isdigit())
        n = int(num) if num else 1
        unit = "".join(ch for ch in r if ch.isalpha()) or "S"
        return n * {"S": 1, "T": 60, "MIN": 60, "H": 3600, "D": 86400}.get(unit, 1)

    return rule_a if to_s(rule_a) >= to_s(rule_b) else rule_b


def _select_column_from_obj(obj, column: str) -> pd.Series:
    """
    Return the Series for `column` from obj.data or obj.extra_data (in that order).
    Assumes indices are DatetimeIndex; caller handles time-cropping / coercion.
    Raises KeyError if not found anywhere.
    """
    if column in obj.data.columns:
        return obj.data[column]
    # extra_data may be empty or missing
    extra = getattr(obj, "extra_data", None)
    if extra is not None and column in extra.columns:
        return extra[column]
    raise KeyError(f"Column '{column}' not found in obj.data or obj.extra_data.")


def _extract_series(
    obj,
    column: str,
    start_time: Optional[pd.Timestamp | str] = None,
    end_time: Optional[pd.Timestamp | str] = None,
) -> pd.Series:
    """
    Get the requested column as a float Series indexed by time.
    Uses obj.timecrop(start,end,inplace=False) when a window is provided.
    """
    if start_time is not None and end_time is not None and hasattr(obj, "timecrop"):
        obj = obj.timecrop(_ts(start_time), _ts(end_time), inplace=False)

    s = _select_column_from_obj(obj, column).copy()
    # Index is a DatetimeIndex by contract
    s = s.sort_index()
    s = pd.to_numeric(s, errors="coerce")
    return s


def _align_series(
    X,
    Y,
    column: str,
    start_time: Optional[pd.Timestamp | str],
    end_time: Optional[pd.Timestamp | str],
    *,
    match: str = "exact",  # "exact" | "nearest" | "rebin"
    tolerance: Union[str, pd.Timedelta] = "30s",
    rebin_freq: Optional[str] = None,  # when match="rebin"
    rebin_method: Union[str, Callable] = "mean",  # when match="rebin"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract + align `column` from two aerosol objects per chosen strategy.
    Drops rows with NaN/±inf.
    """
    sx = _extract_series(X, column, start_time, end_time).rename("x")
    sy = _extract_series(Y, column, start_time, end_time).rename("y")

    if match == "exact":
        xy = pd.concat([sx, sy], axis=1, join="inner")

    elif match == "nearest":
        # mutual nearest within tolerance to avoid asymmetric pairings
        dx = sx.reset_index().rename(columns={sx.index.name or "index": "time"})
        dy = sy.reset_index().rename(columns={sy.index.name or "index": "time"})
        dx = dx.sort_values("time")
        dy = dy.sort_values("time")
        tol = pd.to_timedelta(tolerance)

        left = pd.merge_asof(
            dx, dy, on="time", direction="nearest", tolerance=tol
        ).dropna(subset=["y"])
        right = pd.merge_asof(
            dy, dx, on="time", direction="nearest", tolerance=tol
        ).dropna(subset=["x"])

        left["key"] = list(zip(left["time"], left["y"]))
        right["key"] = list(zip(right["time"], right["x"]))
        keep = set(left["key"]).intersection(set(right["key"]))
        left = left[left["key"].isin(keep)]

        xy = left[["x", "y"]].set_index(pd.DatetimeIndex(left["time"]))

    elif match == "rebin":
        # target cadence: explicit or coarser of the two inferred
        if rebin_freq is None:
            fx = _infer_freq(sx.index) or "S"  # type: ignore
            fy = _infer_freq(sy.index) or "S"  # type: ignore
            target = _coarser(fx, fy)
        else:
            target = rebin_freq

        st = _ts(start_time) if start_time is not None else None
        et = _ts(end_time) if end_time is not None else None

        def _rb(obj):
            # .timerebin also resamples extra_data in your implementation
            tmp = obj.timerebin(
                freq=target, start=st, end=et, method=rebin_method, inplace=False
            )
            s = _select_column_from_obj(tmp, column)  # check data then extra_data
            return pd.to_numeric(s, errors="coerce").sort_index()

        sxr = _rb(X).rename("x")
        syr = _rb(Y).rename("y")
        xy = pd.concat([sxr, syr], axis=1, join="inner")

    else:
        raise ValueError("match must be 'exact', 'nearest', or 'rebin'")

    vals = xy.to_numpy(dtype=float, copy=False)
    m = np.isfinite(vals).all(axis=1)
    if not m.any():
        return np.array([], float), np.array([], float)

    xy = xy.loc[m]
    return xy["x"].to_numpy(float), xy["y"].to_numpy(float)


def _linear(x: NDArray[np.float64], A: float, B: float = 0.0) -> NDArray[np.float64]:
    return A * x + B


def _r2(y_true: NDArray[np.float64], y_fit: NDArray[np.float64]) -> float:
    ss_res = float(np.sum((y_true - y_fit) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return round(1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0), 3)


# =========================
# Main plotting function
# =========================


def Plot_correlation(
    X,
    Y,
    ax_in: Axes | None = None,
    *,
    start_time: pd.Timestamp | str | None = None,
    end_time: pd.Timestamp | str | None = None,
    column: str = "Total_conc",
    match: str = "exact",  # "exact" | "nearest" | "rebin"
    tolerance: Union[str, pd.Timedelta] = "30s",
    rebin_freq: Optional[str] = None,
    rebin_method: Union[str, Callable] = "mean",
    intercept: bool = True,
    uniform_scaling: bool = True,
    outlier_influence: bool = True,
) -> Tuple[Figure | SubFigure, Axes]:
    """
    Plot the correlation between two aerosol datasets over a time window.

    The function selects the requested column (default "Total_conc") from each object,
    looking first in ``obj.data`` and then in ``obj.extra_data``. It crops the data to
    the specified time window (``start_time`` / ``end_time`` may be strings or
    ``pandas.Timestamp`` and are parsed with ``pandas.to_datetime``), aligns the two
    series according to the chosen strategy, removes invalid values, and fits a linear
    model for visualization.

    Parameters
    ----------
    X, Y : Aerosol1D-like
        Aerosol datasets exposing ``.data`` (DatetimeIndex). The column to correlate is
        taken from ``obj.data[column]`` if present; otherwise from
        ``obj.extra_data[column]``. If ``match='rebin'`` is used, objects should
        implement ``.timerebin``.
    ax_in : matplotlib.axes.Axes, optional
        Axes to draw on; if None, a new Figure/Axes is created.
    start_time, end_time : pandas.Timestamp or str, optional
        Inclusive analysis window. If both are provided, each object is cropped via
        ``obj.timecrop(start_time, end_time, inplace=False)``. Strings are parsed with
        ``pandas.to_datetime``.
    column : str, optional
        Name of the variable to correlate (default "Total_conc"). Looked up in
        ``obj.data`` first, then ``obj.extra_data``.
    match : {"exact", "nearest", "rebin"}, optional
        - "exact"  : keep only timestamps present in both series (default).
        - "nearest": pair to nearest neighbor within ``tolerance`` (mutual nearest).
        - "rebin"  : rebin both inputs to a common cadence with ``obj.timerebin``.
    tolerance : str or pandas.Timedelta, optional
        Max separation for ``match="nearest"`` (e.g., "30s"). Default "30s".
    rebin_freq : str, optional
        Target cadence for ``match="rebin"``. If None, uses the coarser cadence inferred
        from the inputs.
    rebin_method : str or callable, optional
        Aggregation passed to ``obj.timerebin`` when ``match="rebin"`` (default "mean").
    intercept : bool, optional
        If True, fit ``y = A*x + B``; else fit ``y = A*x`` with ``B = 0``.
    uniform_scaling : bool, optional
        If True, scale both axes by a common factor so limits are shared.
    outlier_influence : bool, optional
        If True, use least-squares (``curve_fit``) and draw a 1σ band;
        if False, use Theil–Sen (robust), without band.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the created/used figure.
    ax : matplotlib.axes.Axes
        Handle for the axes containing the correlation plot.

    Notes
    -----
    - Rows with NaN or ±inf in either series are removed before fitting.
    - With ``match="rebin"``, rebinning is performed via
      ``obj.timerebin(freq=rebin_freq or inferred, start=start_time, end=end_time,
      method=rebin_method, inplace=False)`` for each object, then aligned by inner join.
    - The plot shows a 1:1 line, fitted line, equation, and R².
    """
    # --- align + clean -------------------------------------------------------
    x_vals, y_vals = _align_series(
        X,
        Y,
        column,
        start_time,
        end_time,
        match=match,
        tolerance=tolerance,
        rebin_freq=rebin_freq,
        rebin_method=rebin_method,
    )

    # Always return a top-level Figure to keep type hints simple
    if ax_in is None:
        fig, ax = plt.subplots()
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.legend(fontsize=15)
        ax.grid(True)
    else:
        ax = ax_in
        fig = ax.figure

    x = x_vals.astype(np.float64, copy=False)
    y = y_vals.astype(np.float64, copy=False)

    # --- fit -----------------------------------------------------------------
    SE_A = 0.0
    SE_B = 0.0
    if outlier_influence:
        if intercept:
            params, cov = curve_fit(_linear, x, y, p0=[1.0, 0.0])
            A, B = float(params[0]), float(params[1])
            if cov is not None and cov.size >= 2:
                d = np.sqrt(np.diag(cov))
                SE_A, SE_B = float(d[0]), float(d[1])
        else:

            def _lin0(xv: NDArray[np.float64], a: float) -> NDArray[np.float64]:
                return a * xv

            params, cov = curve_fit(_lin0, x, y, p0=[1.0])
            A, B = float(params[0]), 0.0
            if cov is not None and cov.size >= 1:
                SE_A = float(np.sqrt(cov[0, 0]))
    else:
        slope, intercept_ts, _, _ = theilslopes(y, x)
        A, B = float(slope), float(intercept_ts)  # type: ignore

    y_fit = _linear(x, A, B)
    r2 = _r2(y, y_fit)

    # --- scaling and limits --------------------------------------------------
    factor = (
        float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0))
        if uniform_scaling
        else 1.0
    )
    x_min = float(np.min(x) / factor) if np.min(x) <= 0 else 0.0
    x_max = 0.0 if np.max(x) <= 0 else float(max(np.max(x), np.max(y)) / factor)
    fit_x = np.linspace(x_min, x_max, 200, dtype=np.float64)
    fit_y = A * fit_x + (B / factor)

    # --- draw ----------------------------------------------------------------
    ax.plot([x_min, x_max], [x_min, x_max], ls="--", c="k", lw=3)  # 1:1 line
    ax.plot(x / factor, y / factor, "bo")  # scatter
    ax.plot(fit_x, fit_y, "r-", lw=3)  # fit
    label = (
        f"y={round(A,2)}·x"
        if B == 0.0
        else (
            f"y={round(A,2)}·x + {round(B,2)}"
            if B > 0
            else f"y={round(A,2)}·x {round(B,2)}"
        )
    )
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=15, va="top")
    ax.text(
        0.05,
        0.80 if B == 0.0 else 0.65,
        f"r$^2$: {round(r2,2)}",
        transform=ax.transAxes,
        fontsize=15,
        va="top",
    )

    if outlier_influence:
        band = np.sqrt((SE_A * fit_x) ** 2 + (SE_B / factor) ** 2)
        ax.fill_between(fit_x, fit_y - band, fit_y + band, alpha=0.33)
    ax.set_xlabel("X: {0}".format(X.instrument), fontsize=15)
    ax.set_ylabel("Y: {0}".format(Y.instrument), fontsize=15)
    return fig, ax
