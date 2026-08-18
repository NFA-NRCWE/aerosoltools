"""Correlation, regression and Bland-Altman agreement analysis between two
aerosol time series.

The two datasets are time-aligned by the shared helpers in
:mod:`~aerosoltools.intercomparison._alignment`; this module only fits the
relationship and draws the comparison.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.stats import norm, theilslopes

from ._alignment import _align_series, _resolve_unit


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
):
    """Fit a function relating one dataset's signal to another's.

    Aligns the chosen signal from ``target_data`` and ``ref_data`` on their
    common time base (dropping samples that are NaN in either), fits
    ``fit_function`` to the pair with :func:`scipy.optimize.curve_fit`, and
    reports the fitted coefficients, their 1σ standard errors, and the R² of the
    fit. Nothing is modified — this only computes the relationship (e.g. to build
    a calibration from it later).

    Args:
        target_data: The dataset whose signal is the fit's ``x`` (e.g. the field
            instrument being characterised).
        ref_data: The dataset whose signal is the fit's ``y`` (e.g. the
            reference instrument).
        parameter (int | str): Which signal to extract from each dataset. An
            ``int`` is a positional index into ``data.columns``; a ``str`` is a
            column label. Defaults to ``0``.
        fit_function (callable | None): Model ``f(x, *coeffs)`` to fit. Defaults
            to a linear model ``y = m*x + b`` (:func:`_linear`) when ``None``.
        variable_guess (dict): Initial guesses for the fit coefficients, keyed by
            name (e.g. ``{"m": 1}`` or ``{"A": 1, "B": 0}``); the keys also name
            the returned coefficients. Defaults to ``{"m": 1}``.
        start_time: Optional start of the analysis window (string or
            :class:`pandas.Timestamp`).
        end_time: Optional end of the analysis window (string or
            :class:`pandas.Timestamp`).

    Returns:
        tuple[dict, dict, float]: ``(coeffs, errors, r_squared)`` where
        ``coeffs`` maps each ``variable_guess`` key to its fitted value,
        ``errors`` maps each key to its 1σ standard error, and ``r_squared`` is
        the coefficient of determination of the fit.
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


def plot_correlation(
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
    """Create a correlation plot between the same variable from two aerosol
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
            ``plot_correlation`` is a convenience function for quickly
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
            smps = at.load_smps_file("smps_data.txt")
            ops = at.load_ops_file("ops_data.txt")

            # Plot correlation of total concentration over a work shift
            fig, ax = at.plot_correlation(
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

    # Per-parameter unit strings (X.unit may be a per-column dict).
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


def wind_rose(
    X,
    Y,
    parameter: str = "Total_conc",
    speed_log=False,
    wind_resolution: tuple = (8, 15),
    ax_in=None,
    start_time: pd.Timestamp | str | None = None,
    end_time: pd.Timestamp | str | None = None,
    rebin_freq: str | None = "1min",
    rebin_method: str = "mean",
    activity: str | None = None,
    min_observations: int = 3,
):
    """
    Function to generate a heat-map wind-rose depiction of data.
    The functions combines the simultatious data of wind speed and wind direction
    from an environmental class and combines it with a

    Args:
       X:
           First dataset. A :class:`Environmental1D` with data for wind speed
           and wind direction.
       Y:
           Second aerosol-like object. This provides the data to be plotted
           in the heatmap. This can also be the first data set X.
       parameter (str, optional):
           Name of the variable to correlate. The function first looks for
           this column in ``Y.data`` and then in ``Y.extra_data``.
           The default is ``\"Total_conc\"``.
       speed_log (bool, optional):
           Determines whether the radial axis should be displayed as with
           windspeeds spreadout in logspace. Default is False.
        wind_resolution (tuple, optional):
           Provides the number of bins along each wind dimension. The default
           (8,15) result in 8 sections around the compass with 15 sections
           along the radial axis marking the wind speed.
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
           overlapping record. The viable activities must be marked in dataset X.
       min_observations (int, optional):
           The minimum number of datapoints going into the calculation of a
           bin average. Depending on the rebin freq this migth remain low,
           if freq is high. Default is 3.

    Returns:
        tuple[Figure, Axes]:
            The figure and axes containing the wind-rose polar heatmap, with
            colorbar to the right and details of parameter and instrument to
            the top left.


    Notes:
        Detailed description:
            ``wind_rose`` is creating a depiction of the average of a chosen
            parameter data using a radial heat-map to associate the desired
            parameter of interest with wind speed and direction.

            * Extracts the requested ``parameter`` from dataset Y.
            * Aligns the series in time using the selected timerebin
            * Removes rows where either series is NaN or infinite.
            * Create bins in the polar space according to the chosen resolution.
            * Plots polar heatmap showing the average concentration/strength
              of the chosen parameter in color along the compass directions.

            Axis labels are automatically derived from ``X.instrument`` and
            ``Y.instrument``.

        Theory:
            The regression models used are simple linear relationships:


    """
    # ------------------------------------------------------------------
    # Construct dataframe and remove invalid observations
    # ------------------------------------------------------------------

    # Always return a top-level Figure
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.grid(True)
    else:
        fig = ax_in.figure
        if ax_in.name == "polar":
            ax = ax_in
        else:
            subplotspec = ax_in.get_subplotspec()
            ax_in.remove()
            ax = fig.add_subplot(
                subplotspec,
                projection="polar",
            )

    wind = X.timerebin(
        freq=rebin_freq,
        start=start_time,
        end=end_time,
        method=rebin_method,
        inplace=False,
    )

    data = Y.timerebin(
        freq=rebin_freq,
        start=start_time,
        end=end_time,
        method=rebin_method,
        inplace=False,
    )

    # Mark activities
    if activity is None:
        activity = "All data"

    data.mark_activities(wind.activity_periods)

    if "W_direction" in wind.data.columns:
        df = {"w_dir": wind.get_activity_data(activity)["W_direction"]}

        if "W_speed" in wind.data.columns:
            df["w_speed"] = wind.get_activity_data(activity)["W_speed"]
        else:
            raise ValueError("The chosen data does not contain data on wind-speed")
    else:
        raise ValueError("The chosen data does not contain data on wind-direction")

    if parameter in data.data.columns:
        df["data"] = data.get_activity_data(activity)[parameter]
    elif parameter in data.extra_data.columns:
        df["data"] = data.get_activity_extra_data(activity)[parameter]
    else:
        raise ValueError(f"{parameter} is not present in the chosen dataset")

    df = pd.DataFrame(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["w_dir", "w_speed", "data"])

    if df.empty:
        raise ValueError("No valid wind/concentration observations remain.")

    n_dir_bins, n_speed_bins = wind_resolution

    # Generate wind direction bins
    dir_width = 360.0 / n_dir_bins
    dir_shift = dir_width / 2

    wd = df["w_dir"] % 360
    wd_shifted = (wd + dir_shift) % 360
    dir_edges = np.linspace(0, 360, n_dir_bins + 1)

    df["dir_bin"] = pd.cut(
        wd_shifted, bins=dir_edges, right=False, include_lowest=True, labels=False
    )
    # Generate wind speed bins
    if speed_log:
        positive_speed = df.loc[df["w_speed"] > 0, "w_speed"]
        speed_min = positive_speed.min()
        speed_max = positive_speed.max()
        speed_edges = np.geomspace(speed_min, speed_max * (1 + 1e-10), n_speed_bins + 1)
    else:
        speed_min = max(0, df["w_speed"].min())
        speed_max = df["w_speed"].max()
        speed_edges = np.linspace(speed_min, speed_max * (1 + 1e-10), n_speed_bins + 1)
    df["speed_bin"] = pd.cut(
        df["w_speed"], bins=speed_edges, right=False, include_lowest=True, labels=False
    )
    # Remove data outside the defined bins
    df_grid = df.dropna(subset=["dir_bin", "speed_bin", "data"]).copy()
    df_grid["dir_bin"] = df_grid["dir_bin"].astype(int)
    df_grid["speed_bin"] = df_grid["speed_bin"].astype(int)
    # Calculate mean and counts
    grid = df_grid.groupby(["speed_bin", "dir_bin"])["data"].mean().unstack()
    counts = df_grid.groupby(["speed_bin", "dir_bin"])["data"].count().unstack()
    # Force the requested resolution
    grid = grid.reindex(
        index=range(n_speed_bins),
        columns=range(n_dir_bins),
    )
    counts = counts.reindex(
        index=range(n_speed_bins),
        columns=range(n_dir_bins),
        fill_value=0,
    )
    # Mask cells without sufficient observations
    grid = grid.where(counts >= min_observations)
    assert grid.shape == (n_speed_bins, n_dir_bins)
    # Coordinate grid
    plot_dir_edges = np.deg2rad(
        np.linspace(
            -dir_shift,
            360 - dir_shift,
            n_dir_bins + 1,
        )
    )
    Theta, R = np.meshgrid(
        plot_dir_edges,
        speed_edges,
    )
    C = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    pcm = ax.pcolormesh(
        Theta,
        R,
        C,
        shading="flat",
        cmap="viridis",
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(
        [
            "N",
            "NE",
            "E",
            "SE",
            "S",
            "SW",
            "W",
            "NW",
        ]
    )
    # Radial scale
    if speed_log:
        ax.set_rscale("log")
        ax.set_rlim(
            speed_edges[0],
            speed_edges[-1],
        )
    else:
        ax.set_rlim(
            speed_edges[0],
            speed_edges[-1],
        )
    # Addition of unit and dtype to the color scale
    if isinstance(data._meta["unit"], dict):
        fig.colorbar(
            pcm,
            ax=ax,
            label=f"{data._meta['dtype'][parameter]} ({data._meta['unit'][parameter]})",
        )
    else:
        fig.colorbar(pcm, ax=ax, label=f"{data._meta['dtype']} ({data._meta['unit']})")
    # Finishing touches for the
    ax.set_title(f"{data._meta['instrument']} \n{parameter}", loc="left")
    ax.text(
        1.00,
        1.05,
        "Wind speed (m/s)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
    )
    return fig, ax
