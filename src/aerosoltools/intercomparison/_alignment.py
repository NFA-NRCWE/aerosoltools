"""Shared time-alignment helpers used across the intercomparison workflows.

Two datasets rarely share an identical clock, so every cross-dataset workflow
(correlation, combination, calibration) first has to line the two time series
up. This module owns that shared machinery: frequency/timestamp coercion, the
column selection that tolerates ``data`` vs ``extra_data`` and per-channel unit
dicts, and :func:`_align_series` (exact / nearest / rebin matching, optional
activity restriction). It is private — users reach it only through the public
functions that build on it.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _ts(x) -> pd.Timestamp:
    """Coerce an input into a :class:`pandas.Timestamp`.

    This helper accepts strings, Python datetime objects, NumPy datetime64, and
    already-constructed :class:`pandas.Timestamp` objects and returns a
    normalized ``Timestamp`` instance.

    Args:
        x: A value representing a point in time (e.g. ``str``,
            :class:`datetime.datetime`, :class:`datetime.date`,
            :class:`numpy.datetime64`, or :class:`pandas.Timestamp`).

    Returns:
        pandas.Timestamp: The input converted to a ``Timestamp``.
    """
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, (dt.datetime, dt.date, np.datetime64, str)):
        return pd.to_datetime(x)
    return pd.to_datetime(x)


def _infer_freq(idx: pd.DatetimeIndex) -> Optional[str]:
    """Infer a reasonable resampling rule from a time index.

    The function first tries :func:`pandas.infer_freq`. If that fails, it falls
    back to estimating the cadence from the median inter-sample spacing and
    returns a rule like ``"1S"``, ``"5T"`` or ``"1H"``.

    Args:
        idx: Datetime index from which to infer a sampling frequency.

    Returns:
        str | None: A pandas offset alias representing the inferred cadence
        (e.g. ``"1S"``, ``"30T"``, ``"1H"``), or ``None`` if it cannot be
        determined.
    """
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
    """Return the coarser (slower) cadence between two resampling rules.

    The rules are interpreted as simple second-, minute-, hour- or day-based
    frequencies (e.g. ``"S"``, ``"10S"``, ``"5T"``, ``"1H"``, ``"1D"``), and
    compared by their corresponding period length in seconds.

    Args:
        rule_a: First pandas-style frequency string.
        rule_b: Second pandas-style frequency string.

    Returns:
        str: The rule corresponding to the larger time step (coarser cadence).
    """

    def to_s(rule: str) -> float:
        r = rule.upper()
        num = "".join(ch for ch in r if ch.isdigit())
        n = int(num) if num else 1
        unit = "".join(ch for ch in r if ch.isalpha()) or "S"
        return n * {"S": 1, "T": 60, "MIN": 60, "H": 3600, "D": 86400}.get(unit, 1)

    return rule_a if to_s(rule_a) >= to_s(rule_b) else rule_b


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
    but a ``{column: unit}`` dict for multi-channel instruments (e.g. DiSCmini). This
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

    This helper is used by :func:`plot_correlation` to obtain two aligned
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
