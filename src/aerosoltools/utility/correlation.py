"""Correlation, regression and Bland-Altman agreement analysis between
two aerosol time series, with the time-alignment helpers they share."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.stats import norm, theilslopes

from ._common import _coarser, _infer_freq, _ts


def _select_column_from_obj(obj, parameter: str) -> pd.Series:
    """Select a named column from an aerosol-like object.

    The function looks for the requested parameter first in ``obj.data`` and, if
    not found, in ``obj.extra_data``. It assumes both attributes (if present)
    are pandas objects indexed by time.

    Args:
        obj: Object exposing at least a ``data`` attribute (pandas DataFrame)
            and optionally an ``extra_data`` attribute (pandas DataFrame).
        parameter: Column name to retrieve.

    Returns:
        pandas.Series: The selected column as a Series with time index.

    Raises:
        KeyError: If the column is not found in either ``data`` or
            ``extra_data``.
    """
    if parameter in obj.data.columns:
        return obj.data[parameter]
    # extra_data may be empty or missing
    extra = getattr(obj, "extra_data", None)
    if extra is not None and parameter in extra.columns:
        return extra[parameter]
    raise KeyError(f"Column '{parameter}' not found in obj.data or obj.extra_data.")


def _resolve_unit(obj, parameter: str | None = None) -> str:
    """Return the display unit for ``parameter`` as a plain string.

    ``obj.unit`` is a single string for :class:`Aerosol1D`/:class:`Aerosol2D`
    but a ``{column: unit}`` dict for :class:`AerosolAlt` (e.g. DiSCmini). This
    resolves the per-parameter unit so axis labels never render a whole dict.

    Args:
        obj: Aerosol object exposing a ``unit`` attribute.
        parameter: Column name to look up when ``unit`` is a dict; when it is
            missing/``None`` the first entry is used.

    Returns:
        The unit string (empty when unavailable).
    """
    unit = getattr(obj, "unit", "")
    if isinstance(unit, dict):
        if parameter is not None and parameter in unit:
            return str(unit[parameter])
        return str(next(iter(unit.values()), "")) if unit else ""
    return str(unit)


def _extract_series(
    obj,
    parameter: str,
    start_time: pd.Timestamp | str | None = None,
    end_time: pd.Timestamp | str | None = None,
) -> pd.Series:
    """Return a numeric time series from an object, optionally time-cropped.

    If both ``start_time`` and ``end_time`` are provided and the object
    implements ``timecrop``, a non-destructive crop is applied before the
    column is selected. The resulting column is then converted to numeric,
    coercing non-numeric entries to NaN and sorted by time.

    Args:
        obj: Object exposing ``data`` and/or ``extra_data`` and optionally
            ``timecrop``. Typically an :class:`Aerosol1D` or :class:`Aerosol2D`.
        parameter: Name of the column to extract from ``data`` or ``extra_data``.
        start_time: Optional start of the time window. May be a
            :class:`pandas.Timestamp` or a string parseable by
            :func:`pandas.to_datetime`.
        end_time: Optional end of the time window. Same rules as ``start_time``.

    Returns:
        pandas.Series: A float Series indexed by time, cropped and coerced
        to numeric.
    """
    if start_time is not None and end_time is not None and hasattr(obj, "timecrop"):
        obj = obj.timecrop(_ts(start_time), _ts(end_time), inplace=False)

    s = _select_column_from_obj(obj, parameter).copy()
    # Index is a DatetimeIndex by contract
    s = s.sort_index()
    s = pd.to_numeric(s, errors="coerce")
    return s


def _activity_period_mask(index, X, Y, activity: str) -> NDArray[np.bool_]:
    """Boolean mask over ``index`` selecting timestamps inside an activity.

    An activity's occurrences are absolute-time ``(start, end)`` intervals
    shared across datasets, so a timestamp belongs to the activity when it falls
    within any of those intervals. The periods are read from whichever of the two
    objects defines the activity (they are projected from the same registry).

    Args:
        index: Timestamps to test (a :class:`pandas.DatetimeIndex` or similar).
        X: First aerosol-like object.
        Y: Second aerosol-like object.
        activity: Name of the activity whose periods define the selection.

    Returns:
        numpy.ndarray: Boolean array, ``True`` where the timestamp is inside the
        activity.

    Raises:
        ValueError: If neither object defines ``activity``.
    """
    periods = None
    for obj in (X, Y):
        registry = getattr(obj, "_activity_periods", None)
        if registry and activity in registry:
            periods = registry[activity]
            break
    if periods is None:
        raise ValueError(f"Activity '{activity}' is not defined on either dataset.")

    idx = pd.DatetimeIndex(index)
    mask = np.zeros(len(idx), dtype=bool)
    for start, end in periods:
        mask |= (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return mask


def _align_series(
    X,
    Y,
    parameter: str | tuple,
    start_time: pd.Timestamp | str | None,
    end_time: pd.Timestamp | str | None,
    *,
    match: str = "exact",  # "exact" | "nearest" | "rebin"
    tolerance: str | pd.Timedelta = "30s",
    rebin_freq: str | None = None,  # when match="rebin"
    rebin_method: str | Callable = "mean",  # when match="rebin"
    activity: str | None = None,  # restrict to one activity's periods
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and time-align a variable from two aerosol-like objects.

    This helper is used by :func:`Plot_correlation` to obtain two aligned
    numeric arrays for one or two given parameters. It delegates column selection to
    :func:`_extract_series`, then applies one of three alignment strategies:

    * ``"exact"`` – inner join on identical timestamps.
    * ``"nearest"`` – mutual nearest-neighbor pairing within ``tolerance``.
    * ``"rebin"`` – rebin both objects to a common cadence via ``timerebin``
      and join on timestamps.

    Rows containing NaN or ±inf in either series are dropped.

    Args:
        X: First aerosol-like object, typically :class:`Aerosol1D` or
            :class:`Aerosol2D`.
        Y: Second aerosol-like object.
        parameter: Name of the variable or variables to extract from each object.
            If tuple the parameters are read as (parameter_X, parameter_Y)
        start_time: Optional start of the analysis window (string or
            :class:`pandas.Timestamp`).
        end_time: Optional end of the analysis window (string or
            :class:`pandas.Timestamp`).
        match: Alignment strategy (``"exact"``, ``"nearest"``, or ``"rebin"``).
        tolerance: Maximum time separation allowed for ``match="nearest"``.
        rebin_freq: Target frequency string for ``match="rebin"``. If ``None``,
            the coarser cadence inferred from the two series is used.
        rebin_method: Aggregation method passed to ``timerebin`` when
            ``match="rebin"`` (e.g. ``"mean"`` or a callable).

    Returns:
        tuple[np.ndarray, np.ndarray]: Two 1D NumPy arrays (x, y) containing
        aligned, finite values suitable for regression or plotting.
    """
    if type(parameter) is str:
        sx = _extract_series(X, parameter, start_time, end_time).rename("x")
        sy = _extract_series(Y, parameter, start_time, end_time).rename("y")
    elif type(parameter) is tuple:
        sx = _extract_series(X, parameter[0], start_time, end_time).rename("x")
        sy = _extract_series(Y, parameter[1], start_time, end_time).rename("y")
    else:
        raise ValueError("Parameter not str or tuple.")
    # If one side has no data at all in this window, fail early
    if sx.empty or sy.empty:
        raise ValueError(
            "No data for the requested parameter/time window in one or both objects. "
            "Check 'start_time'/'end_time' and 'parameter'."
        )

    match = match.lower()
    if match == "exact":
        # Inner join on identical timestamps
        xy = pd.concat([sx, sy], axis=1, join="inner")

    elif match == "nearest":
        # One-way nearest-neighbor match: align Y to X's timeline
        dx = sx.reset_index().rename(columns={sx.index.name or "index": "time"})
        dy = sy.reset_index().rename(columns={sy.index.name or "index": "time"})
        dx = dx.sort_values("time")
        dy = dy.sort_values("time")
        tol = pd.to_timedelta(tolerance)

        merged = pd.merge_asof(
            dx,
            dy,
            on="time",
            direction="nearest",
            tolerance=tol,
        )
        # Drop rows where Y has no acceptable neighbor
        merged = merged.dropna(subset=["y"])

        if merged.empty:
            raise ValueError(
                "No matching timestamps between the two series within the given "
                "tolerance. Try increasing 'tolerance' or using match='rebin'."
            )

        xy = merged.set_index("time")[["x", "y"]]

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
            tmp = obj.timerebin(
                freq=target, start=st, end=et, method=rebin_method, inplace=False
            )
            s = _select_column_from_obj(tmp, parameter)
            return pd.to_numeric(s, errors="coerce").sort_index()

        sxr = _rb(X).rename("x")
        syr = _rb(Y).rename("y")
        xy = pd.concat([sxr, syr], axis=1, join="inner")

    else:
        raise ValueError("match must be 'exact', 'nearest', or 'rebin'")

    # If alignment produced nothing at all, the two instruments never lined up
    # under the current settings — most often an 'exact' match between
    # instruments that log on different clocks/cadences. Point at nearest/rebin.
    if xy.empty:
        if match == "exact":
            raise ValueError(
                "No exact timestamp matches between the two instruments. They "
                "likely log on different clocks or sampling intervals, so no two "
                "samples share the same timestamp. In 'Time alignment', switch "
                "'Match' to 'nearest' (with a tolerance, e.g. 30s) or 'rebin' to "
                "a common interval, then Compute again."
            )
        raise ValueError(
            "The two instruments produced no overlapping data under the current "
            "time-alignment settings. Try a larger tolerance, 'rebin' to a common "
            "interval, or widen the time window."
        )

    # Optionally keep only the aligned points that fall inside one activity.
    # The activity periods are absolute-time, so this works uniformly for all
    # three match modes (and for activities with several occurrences).
    if activity is not None and activity != "All data":
        xy = xy.loc[_activity_period_mask(xy.index, X, Y, activity)]
        if xy.empty:
            raise ValueError(
                f"No aligned data points fall inside activity '{activity}'. The "
                "two instruments do align elsewhere, but not within this "
                "activity's time window — check that the activity covers the "
                "side-by-side period, or (if their timestamps differ) switch "
                "'Match' to 'nearest' or 'rebin'."
            )

    # Remove rows with non-finite values in either series
    vals = xy.to_numpy(dtype=float, copy=False)
    m = np.isfinite(vals).all(axis=1)
    if not m.any():
        raise ValueError(
            "No overlapping valid data points between the two series for the "
            "requested time window and alignment settings."
        )

    xy = xy.loc[m]
    return xy["x"].to_numpy(float), xy["y"].to_numpy(float)


def _linear(x: NDArray[np.float64], A: float, B: float = 0.0) -> NDArray[np.float64]:
    """Simple linear model ``y = A·x + B``.

    Args:
        x: 1D array of predictor values.
        A: Slope of the linear relationship.
        B: Intercept term (default 0).

    Returns:
        numpy.ndarray: Predicted values ``y`` for each ``x``.
    """
    return A * x + B


def _r2(y_true: NDArray[np.float64], y_fit: NDArray[np.float64]) -> float:
    """Compute the coefficient of determination (R²).

    R² is defined as ``1 - SS_res / SS_tot``, where ``SS_res`` is the residual
    sum of squares and ``SS_tot`` is the total sum of squares around the mean
    of ``y_true``. The value is rounded to three decimal places.

    Args:
        y_true: Observed data values.
        y_fit: Fitted or predicted data values with the same shape as
            ``y_true``.

    Returns:
        float: R² between 0 and 1 (or 0 if ``SS_tot`` is zero).
    """
    ss_res = float(np.sum((y_true - y_fit) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return round(1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0), 3)


def fit_data(
    target_data,
    ref_data,
    parameter: str | int = 0,
    fit_function=None,
    variable_guess: dict = {"m": 1},
    start_time: pd.Timestamp | str | None = None,
    end_time: pd.Timestamp | str | None = None,
    # inplace: bool = True
):
    # m: Union[int, float, list] = 1, b: Union[int, float, list] = 0, inplace: bool = True):
    """
    Apply a correction to the total conc and mark the data as calibrated
    by a linear function. The calibration value is applied to the size data.

    Args:
        X: First aerosol-like object, typically :class:`Aerosol1D` or
            :class:`Aerosol2D`.
        Y: Second aerosol-like object.
        parameter: Name of the variable or variables to extract from each object.
            If tuple the parameters are read as (parameter_X, parameter_Y)
        parameter (int | str, optional):
            Index or column name of the signal to plot. If ``int``, it is
            interpreted as a positional index into :attr:`data.columns`. If
            ``str``, it is treated as a column label. Defaults to ``0``.
            If 'bin' is chosen, the calibration will go through each size bin,
            and apply the calibration function.
        fit_function (function):
            A defined function to apply to the calibration.
            If none is chosen, an assumed linear calibration is used using y = m*x +b
        Variables (dict):
            The calibration value to be multiplied to the data for correction.
            If m is provided as a list, it should be of equal length to the number
            of bins. The total concentration is then recalculated as the sum.
        start_time: Optional start of the analysis window (string or
            :class:`pandas.Timestamp`).
        end_time: Optional end of the analysis window (string or
            :class:`pandas.Timestamp`).
        inplace (bool): If ``True`` (default), modify the current instance and
            return it. If ``False``, perform the conversion on a deep copy
            and return the new instance.

    Returns:
        out (Aerosold2D):
            If inplace ``True`` the calibration function applies the calibration
            to the acted upon dataset, if inplace ``False`` a copy of the calibrated
            dataset is returned.
            In addition to the data with the applied calibration, a
    None

    """

    if fit_function is None:
        fit_function = _linear

    # Cleaning up the data and removing rows where either value is nan
    x, y = _align_series(target_data, ref_data, parameter, start_time, end_time)

    # Apply the fit using curve_fit for a function with or without an intercept.
    parameters, covariance = curve_fit(
        fit_function, x, y, p0=[variable_guess[i] for i in variable_guess]
    )
    SE = np.sqrt(np.diag(covariance))

    fit = fit_function(x, *parameters)
    r2 = _r2(y, fit)

    Fit = {}
    Fit_error = {}
    variables = list(variable_guess.keys())
    for i in range(0, len(variables)):
        Fit[variables[i]] = parameters[i]
        Fit_error[variables[i]] = SE[i]

    return Fit, Fit_error, r2


def Plot_correlation(
    X,
    Y,
    ax_in: Axes | None = None,
    *,
    start_time: pd.Timestamp | str | None = None,
    end_time: pd.Timestamp | str | None = None,
    parameter: str | tuple | None = None,
    match: str = "exact",  # "exact" | "nearest" | "rebin"
    tolerance: str | pd.Timedelta = "30s",
    rebin_freq: str = None,
    rebin_method: str | Callable = "mean",
    intercept: bool = True,
    uniform_scaling: bool = True,
    outlier_influence: bool = True,
    activity: str | None = None,
) -> tuple[Figure, Axes]:
    """Description:
        Create a correlation plot between the same variable from two aerosol
        datasets, including regression line, 1:1 line, and R².

    Args:
        X:
            First aerosol dataset. Typically an :class:`Aerosol1D` or
            :class:`Aerosol2D` instance exposing ``data`` (and optionally
            ``extra_data`` and ``timerebin``).
        Y:
            Second aerosol dataset with the same interface requirements as
            ``X``.
        ax_in (matplotlib.axes.Axes | None, optional):
            Existing Matplotlib axes to draw on. If ``None``, a new figure
            and axes are created. Default is None.
        start_time (pandas.Timestamp | str | None, optional):
            Inclusive start of the analysis window. If provided together with
            ``end_time`` and the objects implement ``timecrop``, the data are
            cropped to this period before correlation is computed. Strings are
            parsed with :func:`pandas.to_datetime`. Default is None, meaning
            start from first common timestamp.
        end_time (pandas.Timestamp | str | None, optional):
            Inclusive end of the analysis window. Same parsing rules as
            ``start_time``.
        parameter (str | tuple, optional):
            Name of the variable to correlate. The function first looks for
            this parameter in ``obj.data`` and then in ``obj.extra_data``.
            If tuple the parameters are read as (parameter_X, parameter_Y).
            If None the first data column is chosen for each dataset.
            The default is ``\"Total_conc\"``

        match (str, optional):
            Strategy for aligning the two time series in time. One of:

            - ``\"exact\"`` (default): Keep only timestamps that are present
              in both series.
            - ``\"nearest\"``: Match values from ``Y`` to the timeline of
              ``X`` using nearest timestamps within ``tolerance``.
            - ``\"rebin\"``: Rebin both datasets to a common time step using
              ``timerebin`` and then join on timestamps.

        tolerance (str | pandas.Timedelta, optional):
            Maximum allowed separation between timestamps when
            ``match=\"nearest\"`` is used. Can be a pandas offset string
            (e.g. ``\"30s\"``) or a :class:`pandas.Timedelta`. Ignored for
            other ``match`` modes.
        rebin_freq (str | None, optional):
            Target resampling rule for ``match=\"rebin\"`` (e.g. ``\"1min\"``).
            If ``None``, the coarser cadence inferred from the two series is
            chosen automatically. Default is None.
        rebin_method (str | Callable, optional):
            Aggregation method passed to ``timerebin`` when ``match=\"rebin\"``
            is used (e.g. ``\"mean\"``, ``\"median\"``, or a custom function).
            Default is ``\"mean\"``.
        intercept (bool, optional):
            If ``True`` (default), fit a full linear model
            ``y = A·x + B``. If ``False``, constrain the fit to pass through
            the origin (``y = A·x``).
        uniform_scaling (bool, optional):
            If ``True`` (default), both axes are scaled by a common factor so
            that the same numerical range is shown on x and y. If ``False``,
            each axis is scaled independently.
        outlier_influence (bool, optional):
            If ``True`` (default), use standard least-squares regression
            (:func:`scipy.optimize.curve_fit`) and draw a 1σ confidence band
            around the fitted line. If ``False``, use the robust
            Theil–Sen estimator (:func:`scipy.stats.theilslopes`) without a
            confidence band.
        activity (str | None, optional):
            If given, restrict the correlation to the timestamps inside this
            activity's marked periods (absolute-time, so multiple occurrences are
            supported). Useful for correlating only a window where the two
            instruments measured side by side. ``None`` (default) or
            ``\"All data\"`` uses the full overlapping record.

    Returns:
        tuple[Figure, Axes]:
            The figure and axes containing the correlation scatter plot, the
            1:1 line, and the regression line with its equation and R² in
            the legend.

    Raises:
        ValueError:
            If one or both objects contain no data for the requested
            ``parameter`` and time window, if the chosen alignment strategy
            yields no matching timestamps, or if all overlapping points are
            non-finite (NaN/inf) after cleaning.
        KeyError:
            If ``parameter`` is not found in either ``data`` or ``extra_data`` of
            one or both objects.
        RuntimeError:
            If the regression fit fails to converge (e.g. due to degenerate
            or extremely ill-conditioned data) and :func:`curve_fit` or the
            Theil–Sen estimator raises a fitting-related error.

    Notes:
        Detailed description:
            ``Plot_correlation`` is a convenience function for quickly
            comparing two aerosol datasets measuring the same physical
            quantity, such as total particle number concentration from two
            instruments. The function:

            * Extracts the requested ``parameter`` from each object.
            * Aligns the series in time using the selected ``match`` mode
              (exact timestamps, nearest neighbors, or common rebinned
              cadence).
            * Removes rows where either series is NaN or infinite.
            * Fits a linear model relating ``Y`` to ``X``, optionally
              including an intercept and using either standard or robust
              regression.
            * Computes and reports the coefficient of determination (R²).
            * Plots the scatter of aligned data points, the 1:1 line, the
              fitted regression line, and (optionally) a confidence band
              around the fit.

            Axis labels are automatically derived from ``X.instrument`` and
            ``Y.instrument`` (if available), giving a quick visual summary of
            how well two instruments agree.

        Theory:
            The regression models used are simple linear relationships:

            * With intercept: ``y = A·x + B``
            * Without intercept: ``y = A·x``

            When ``outlier_influence=True``, the parameters ``A`` and ``B``
            are obtained by minimizing the least-squares error using
            :func:`scipy.optimize.curve_fit`. Standard errors of the fit
            parameters are derived from the covariance matrix and propagated
            to form an approximate 1σ confidence band.

            When ``outlier_influence=False``, the Theil–Sen estimator is used
            (:func:`scipy.stats.theilslopes`). This approach is more robust to
            outliers, but no confidence band is drawn.

    Examples:
        A typical use case is to compare the agreement between two
        instruments over the same time period:

        .. code-block:: python

            import aerosoltools as at

            # Load two datasets measuring total number concentration
            smps = at.Load_SMPS_file("smps_data.txt")
            ops = at.Load_OPS_file("ops_data.txt")

            # Plot correlation of total concentration over a work shift
            fig, ax = at.Plot_correlation(
                smps,
                ops,
                start_time="2023-10-01 08:00",
                end_time="2023-10-01 16:00",
                parameter="Total_conc",
                match="nearest",
                tolerance="60s",
                intercept=True,
                uniform_scaling=True,
                outlier_influence=False,
            )
    """
    if parameter is None:
        p_X = X.data.columns[0]
        p_Y = Y.data.columns[0]
    elif type(parameter) is str:
        p_X = parameter
        p_Y = parameter
    elif type(parameter) is tuple:
        p_X = parameter[0]
        p_Y = parameter[1]

    # --- align + clean -------------------------------------------------------
    x_vals, y_vals = _align_series(
        X,
        Y,
        (p_X, p_Y),
        start_time,
        end_time,
        match=match,
        tolerance=tolerance,
        rebin_freq=rebin_freq,
        rebin_method=rebin_method,
        activity=activity,
    )

    # Always return a top-level Figure to keep type hints simple
    if ax_in is None:
        fig, ax = plt.subplots()
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
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
    ax.plot([x_min, x_max], [x_min, x_max], ls="--", c="k", lw=3, label="1:1 line")
    ax.plot(x / factor, y / factor, "bo", label="Datapoints")
    label = (
        f"y={round(A,2)}·x"
        if B == 0.0
        else (
            f"y={round(A,2)}·x + {round(B,2)}"
            if B > 0
            else f"y={round(A,2)}·x {round(B,2)}"
        )
    )
    ax.plot(fit_x, fit_y, "r-", lw=3, label=label + f", r$^2$: {round(r2,2)}")

    if outlier_influence:
        band = np.sqrt((SE_A * fit_x) ** 2 + (SE_B / factor) ** 2)
        ax.fill_between(fit_x, fit_y - band, fit_y + band, alpha=0.33)

    ax.set_xlabel(f"{X.instrument} : {p_X}", fontsize=15)
    ax.set_ylabel(f"{Y.instrument} : {p_Y}", fontsize=15)
    ax.legend(fontsize=15)
    ax.grid(True)
    return fig, ax  # type: ignore


def bland_altman_analysis(
    X,
    Y,
    ax_in: Axes | None = None,
    method: str = "BA",
    C: float = 0.95,
    *,
    start_time: pd.Timestamp | str | None = None,
    end_time: pd.Timestamp | str | None = None,
    parameter: str | tuple | None = None,
    match: str = "exact",  # "exact" | "nearest" | "rebin",C=0.95):
    tolerance: str | pd.Timedelta = "30s",
    rebin_freq: str | None = None,
    rebin_method: str | Callable = "mean",
    activity: str | None = None,
):
    """
    Plot Bland-Altman (difference plot) to highlight difference between two groups
    of data.

    Args:
       X:
           First aerosol dataset. Typically an :class:`Aerosol1D` or
           :class:`Aerosol2D` instance exposing ``data`` (and optionally
           ``extra_data`` and ``timerebin``).
       Y:
           Second aerosol dataset with the same interface requirements as
           ``X``.
       ax_in (matplotlib.axes.Axes | None, optional):
           Existing Matplotlib axes to draw on. If ``None``, a new figure
           and axes are created. Default is None.
       method (str | optional):
            Choose which analysis method is desired between 'BA', 'Gi', 'Eu';
            'BA' - Bland-Altman : straight-forward comparison between two samples
            'Gi' - Giavarina : percentage difference in relation to mean
            'Eu' - Euser : logarithmic of means an difference
            Deault is 'BA' resulting in the standard Bland-Altman analysis.
       C (float | optional):
            Confidence interval. Default is 0.95
       start_time (pandas.Timestamp | str | None, optional):
           Inclusive start of the analysis window. If provided together with
           ``end_time`` and the objects implement ``timecrop``, the data are
           cropped to this period before correlation is computed. Strings are
           parsed with :func:`pandas.to_datetime`. Default is None, meaning
           start from first common timestamp.
       end_time (pandas.Timestamp | str | None, optional):
           Inclusive end of the analysis window. Same parsing rules as
           ``start_time``.
       parameter (str | tuple, optional):
           Name of the variable to correlate. The function first looks for
           this column in ``obj.data`` and then in ``obj.extra_data``.
           If tuple the parameters are read as (parameter_X, parameter_Y).
           If None the first data column is chosen for each dataset.
           The default is ``\"Total_conc\"``
       match (str, optional):
           Strategy for aligning the two time series in time. One of:

           - ``\"exact\"`` (default): Keep only timestamps that are present
             in both series.
           - ``\"nearest\"``: Match values from ``Y`` to the timeline of
             ``X`` using nearest timestamps within ``tolerance``.
           - ``\"rebin\"``: Rebin both datasets to a common time step using
             ``timerebin`` and then join on timestamps.

       tolerance (str | pandas.Timedelta, optional):
           Maximum allowed separation between timestamps when
           ``match=\"nearest\"`` is used. Can be a pandas offset string
           (e.g. ``\"30s\"``) or a :class:`pandas.Timedelta`. Ignored for
           other ``match`` modes.
       rebin_freq (str | None, optional):
           Target resampling rule for ``match=\"rebin\"`` (e.g. ``\"1min\"``).
           If ``None``, the coarser cadence inferred from the two series is
           chosen automatically. Default is None.
       rebin_method (str | Callable, optional):
           Aggregation method passed to ``timerebin`` when ``match=\"rebin\"``
           is used (e.g. ``\"mean\"``, ``\"median\"``, or a custom function).
           Default is ``\"mean\"``.
       activity (str | None, optional):
           If given, restrict the comparison to the timestamps inside this
           activity's marked periods (absolute-time, multiple occurrences
           supported). ``None`` (default) or ``\"All data\"`` uses the full
           overlapping record.

    Returns:
        tuple[Figure, Axes]:
            The figure and axes containing the correlation scatter plot, the
            1:1 line, and the regression line with its equation and R² in
            the legend.


    Notes:
        Detailed description:
            ``bland_altman_analysis`` is a function for quickly
            comparing two aerosol datasets measuring the same physical
            quantity, such as total particle number concentration from two
            instruments. The function:

            * Extracts the requested ``parameter`` from each object.
            * Aligns the series in time using the selected ``match`` mode
              (exact timestamps, nearest neighbors, or common rebinned
              cadence).
            * Removes rows where either series is NaN or infinite.
            * Fits a linear model relating ``Y`` to ``X``, optionally
              including an intercept and using either standard or robust
              regression.
            * Computes and reports the coefficient of determination (R²).
            * Plots the scatter of aligned data points, the 1:1 line, the
              fitted regression line, and (optionally) a confidence band
              around the fit.

            Axis labels are automatically derived from ``X.instrument`` and
            ``Y.instrument`` (if available), giving a quick visual summary of
            how well two instruments agree.

        Theory:
            The regression models used are simple linear relationships:

            * With intercept: ``y = A·x + B``
            * Without intercept: ``y = A·x``

            When ``outlier_influence=True``, the parameters ``A`` and ``B``
            are obtained by minimizing the least-squares error using
            :func:`scipy.optimize.curve_fit`. Standard errors of the fit
            parameters are derived from the covariance matrix and propagated
            to form an approximate 1σ confidence band.

            When ``outlier_influence=False``, the Theil–Sen estimator is used
            (:func:`scipy.stats.theilslopes`). This approach is more robust to
            outliers, but no confidence band is drawn.
    """

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

    if parameter is None:
        p_X = X.data.columns[0]
        p_Y = Y.data.columns[0]
    elif type(parameter) is str:
        p_X = parameter
        p_Y = parameter
    elif type(parameter) is tuple:
        p_X = parameter[0]
        p_Y = parameter[1]

    # Allign time
    x_vals, y_vals = _align_series(
        X,
        Y,
        (p_X, p_Y),
        start_time,
        end_time,
        match=match,
        tolerance=tolerance,
        rebin_freq=rebin_freq,
        rebin_method=rebin_method,
        activity=activity,
    )

    x = x_vals.astype(np.float64, copy=False)
    y = y_vals.astype(np.float64, copy=False)

    means = (x + y) / 2
    diffs = x - y

    # Average difference (aka the bias)
    bias = np.median(diffs)  #!!!
    if method == "Gi":
        diffs = diffs / means * 100
        bias = np.median(diffs)  #!!!
    elif method == "Eu":
        x_log = np.log10(x)
        y_log = np.log10(y)
        # means = (x + y) / 2
        diffs = x_log - y_log

    # Sample standard deviation
    s = np.std(diffs, ddof=1)  # Use ddof=1 to get the sample standard deviation
    loas = norm.interval(C, bias, s)
    if method == "Eu":
        diffs = x - y

    # Per-parameter unit strings (X.unit may be a dict on AerosolAlt).
    unit_x = _resolve_unit(X, p_X)
    diff_label = f"Difference ({unit_x})" if unit_x else "Difference"

    # Dict
    Diff = {
        "BA": diff_label,
        "Gi": "Difference (%)",
        "Eu": diff_label,
    }

    # --- Plot ----------------------------------------------------------------
    ax.scatter(means, diffs, c="k", s=20, alpha=0.6, marker="o")

    # Labels
    title_param = p_X if p_X == p_Y else f"{p_X} vs {p_Y}"
    ax.set_title(f"Bland-Altman Plot for {title_param}")
    ax.set_xlabel(f"Mean ({unit_x})" if unit_x else "Mean")
    ax.set_ylabel(Diff[method])
    # Get axis limits
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    # Set y-axis limits
    max_y = max(abs(bottom), abs(top))
    ax.set_ylim(-max_y * 1.1, max_y * 1.1)
    # Set x-axis limits
    domain = right - left
    ax.set_xlim(left, left + domain * 1.1)
    # Plot the zero line
    ax.axhline(y=0, c="k", lw=0.5)
    # Plot the bias and the limits of agreement. The bias / LOA values are shown
    # in the legend (not annotated on the plot) so labels never overlap when the
    # lines lie close together.
    unit_suffix = f" {unit_x}" if (unit_x and method != "Gi") else ""
    conf_pct = f"{C * 100:g}%"
    ax.axhline(y=bias, c="grey", ls="--", label=f"Bias = {bias:+.2f}{unit_suffix}")

    # --- Plot the limits of the agreement--------------------------------------
    if method == "Eu":
        # Convert the LOAs from horizontal lines in the log space to gradients of
        # diagonal lines in the native space
        lower_loa_m = 2 * (10 ** (loas[0] - bias) - 1) / (10 ** (loas[0] - bias) + 1)
        upper_loa_m = 2 * (10 ** (loas[1] - bias) - 1) / (10 ** (loas[1] - bias) + 1)
        # Plot the limits of agreement (values shown in the legend).
        x = np.array([left, right])
        y = upper_loa_m * x + bias
        ax.plot(
            x,
            y,
            c="grey",
            ls="--",
            label=f"Upper LOA ({conf_pct}) = {upper_loa_m:+.2f} × Mean + Bias",
        )
        y = lower_loa_m * x + bias
        ax.plot(
            x,
            y,
            c="grey",
            ls="--",
            label=f"Lower LOA ({conf_pct}) = {lower_loa_m:+.2f} × Mean + Bias",
        )
        ax.legend(loc="upper right", fontsize=9)
        print(f"For the differences, μ = {bias:.2f} {unit_x} and s = {s:.2f} {unit_x}")
        return fig, ax, loas
    else:
        ax.axhline(
            y=loas[1],
            c="grey",
            ls="--",
            label=f"Upper LOA ({conf_pct}) = {loas[1]:+.2f}{unit_suffix}",
        )
        ax.axhline(
            y=loas[0],
            c="grey",
            ls="--",
            label=f"Lower LOA ({conf_pct}) = {loas[0]:+.2f}{unit_suffix}",
        )

    ax.legend(loc="upper right", fontsize=9)
    print(f"For the differences, μ = {bias:.2f} {unit_x} and s = {s:.2f} {unit_x}")

    return fig, ax
