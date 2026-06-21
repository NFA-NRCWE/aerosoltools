from __future__ import annotations

__all__ = ["Combine_NS_OPS", "Plot_correlation", "combine_measurements"]

import datetime as dt
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.stats import norm, theilslopes

from .aerosol2d import Aerosol2D

###############################################################################


def Combine_NS_OPS(
    NS_data: Aerosol2D,
    OPS_data: Aerosol2D,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    *,
    match: str = "rebin",  # "exact" | "nearest" | "rebin"
    tolerance: str | pd.Timedelta = "30s",
    rebin_freq: str | None = None,
    rebin_method: str | Callable = "mean",
) -> Aerosol2D:
    """Description:
        Combine data from two instruments of overlapping number size distributions,
        commonly NanoScan (NS) and OPS, into one time-aligned Aerosol2D spectrum.
        The function can also combine FMPS + OPS or NS + APS with the smallest
        bin of the instrument with the larger size range being the cutting point
        of the instrument with smaller size range.

    Args:
        NS_data (Aerosol2D):
            Measurements from the instrument with the smaller size range as an
            :class:`~aerosoltools.aerosol2d.Aerosol2D` instance, containing
            a time-resolved size distribution.
        OPS_data (Aerosol2D):
            Measurements from the instrument with the larger size range as an
            :class:`~aerosoltools.aerosol2d.Aerosol2D` instance, containing
            a time-resolved size distribution.
        start (pandas.Timestamp | str | None, optional):
            Start time of the period used for combining the two instruments.
            If ``None``, the later of the two available start times
            (NS vs OPS) is used. Strings are parsed with
            :func:`pandas.to_datetime`. Default is None.
        end (pandas.Timestamp | str | None, optional):
            End time of the period used for combining the two instruments.
            If ``None``, the earlier of the two available end times
            (NS vs OPS) is used. Default is None.
        match (str, optional):
            Strategy for aligning the two time series in time. Default is
            ``\"rebin\"``. Options are:

            * ``\"rebin\"`` (default): Rebin both instruments to a common
              time step using :meth:`Aerosol2D.timerebin`, then intersect
              timestamps.
            * ``\"exact\"``: Use only timestamps that are present in both
              datasets without resampling.
            * ``\"nearest\"``: Match OPS values to NS timestamps using the
              nearest available OPS point within ``tolerance``.

        tolerance (str | pandas.Timedelta, optional):
            Maximum allowed separation between NS and OPS timestamps when
            ``match=\"nearest\"`` is used. Can be a pandas offset string
            (e.g. ``\"30s\"``) or a :class:`pandas.Timedelta`. Ignored
            for other ``match`` modes. Default is 30s.
        rebin_freq (str | None, optional):
            Target resampling rule for ``match=\"rebin\"`` (e.g. ``\"1min\"``).
            If ``None``, the coarser of the inferred NS and OPS cadences is
            chosen automatically. Default is None.
        rebin_method (str | Callable, optional):
            Aggregation method passed to :meth:`Aerosol2D.timerebin` when
            ``match=\"rebin\"`` is used (e.g. ``\"mean\"``, ``\"median\"``,
            or a custom function). Default is ``\"mean\"``.

    Returns:
        Aerosol2D:
            A new :class:`~aerosoltools.aerosol2d.Aerosol2D` object containing
            the merged NS+OPS number size distribution.

    Raises:
        ValueError:
            If the requested time interval has no overlap between NS and OPS,
            or if the chosen ``match`` strategy produces no common timestamps.
            Also raised if the lowest OPS bin edge falls outside the NS bin
            range so that no consistent splice point can be defined.
        TypeError:
            If ``NS_data`` or ``OPS_data`` does not behave like an
            :class:`Aerosol2D` instance (e.g. missing required attributes such
            as ``time``, ``bin_edges``, or ``timerebin``).

    Notes:
        Detailed description:
            This function is intended for combining data from a NanoScan SMPS
            and an OPS into a single, continuous number size distribution in
            diameter space. Internally, the following steps are carried out:

            * Both inputs are copied to avoid modifying the originals.
            * Each dataset is converted to number concentration (dN) in
              ``cm⁻³`` and de-normalized from ``/dlogDp`` if needed.
            * The time series from NS and OPS are aligned using the specified
              ``match`` mode and the chosen time window (``start``/``end``).
            * The overlapping size range is determined from the NS and OPS
              bin edges. The OPS lower bin edge defines the splice point.
            * The NanoScan bin that overlaps the OPS lower edge is truncated,
              and its concentration is scaled by the fraction of bin width
              that remains below the splice point.
            * All NS bins below the splice point and all OPS bins are
              concatenated to form a new set of bin edges and midpoints.
            * The size-resolved concentrations are merged, and the total
              number concentration is recomputed from the combined size bins
              for each aligned timestamp.
            * NanoScan activities and metadata are propagated to the result,
              and the instrument metadata is set to ``\"NS_OPS\"``.

            The resulting class object includes:

            * Combined size-bin edges and midpoints covering the NS+OPS range.
            * Recomputed total number concentration in ``cm⁻³`` for each
              timestamp.
            * Propagated NanoScan activities and metadata, with the instrument
              set to ``\"NS_OPS\"``.

        Theory:
            The function assumes both instruments measure size-resolved
            particle number concentration and uses a simple geometric
            bin-splicing approach:

            * Number concentration per bin is first expressed as dN (cm⁻³)
              without ``/dlogDp`` normalization.
            * The shared diameter range is determined from the reported bin
              edges; a single splice bin in the NanoScan is truncated so that
              the OPS lower edge becomes the exact boundary between NS and OPS.
            * Conservation of particle number within the truncated bin is
              approximated by scaling with the retained width fraction.

            This produces a piecewise distribution that is directly usable for
            further number-based metrics (e.g. total PNC, modal fits, etc.).

    Examples:
        A typical use case is combining co-located NanoScan and OPS
        measurements into a single spectrum before analysis or reporting:

        .. code-block:: python

            import aerosoltools as at

            NS = at.Load_NS_file("nanoscan.csv")
            OPS = at.Load_OPS_file("ops.csv")

            # Combine on a 1-minute grid using time rebinning
            combined = at.Combine_NS_OPS(
                NS, OPS,
                start="2023-10-01 08:00",
                end="2023-10-01 16:00",
            )

            # Plot the resulting size distribution
            fig, ax = combined.plot_timeseries()
    """

    # --- copy + force dN, unnormalized ---------------------------------------
    NS = NS_data.copy_self()
    OPS = OPS_data.copy_self()
    NS._convert_to_number_concentration()
    NS.unnormalize_logdp()
    OPS._convert_to_number_concentration()
    OPS.unnormalize_logdp()

    # --- determine time window -----------------------------------------------
    ns_idx = NS.time
    ops_idx = OPS.time

    # default overlap window
    default_start = max(ns_idx.min(), ops_idx.min())
    default_end = min(ns_idx.max(), ops_idx.max())

    if start is None:
        st = default_start
    else:
        st = _ts(start)

    if end is None:
        et = default_end
    else:
        et = _ts(end)

    if st > et:
        raise ValueError(
            "No overlapping time range between NS and OPS for the "
            "requested start/end times."
        )

    # --- align in time according to 'match' ----------------------------------
    match = match.lower()
    if match not in {"exact", "nearest", "rebin"}:
        raise ValueError("match must be one of 'exact', 'nearest', or 'rebin'.")

    if match == "exact":
        # Crop both and intersect their time indices – only exact coincidences
        NS_tm = NS.timecrop(st, et, inplace=False)
        OPS_tm = OPS.timecrop(st, et, inplace=False)

        common_idx = NS_tm.data.index.intersection(OPS_tm.data.index)
        if common_idx.empty:
            raise ValueError(
                "No common timestamps between NS and OPS with match='exact'. "
                "Consider using match='nearest' or match='rebin'."
            )
        NS_tm._data = NS_tm._data.loc[common_idx]
        OPS_tm._data = OPS_tm._data.loc[common_idx]

    elif match == "nearest":
        # Crop both to the window
        NS_tm = NS.timecrop(st, et, inplace=False)
        OPS_tm = OPS.timecrop(st, et, inplace=False)

        tol = pd.to_timedelta(tolerance)

        # Union of both time axes within the window
        target_index = NS_tm.data.index.union(OPS_tm.data.index).sort_values()

        # Reindex each to the union grid by nearest neighbor (within tolerance)
        ns_aligned = NS_tm.data.reindex(target_index, method="nearest", tolerance=tol)
        ops_aligned = OPS_tm.data.reindex(target_index, method="nearest", tolerance=tol)

        NS_tm._data = ns_aligned
        OPS_tm._data = ops_aligned

    else:  # match == "rebin"
        # Infer a common cadence if none is given
        if rebin_freq is None:
            f_ns = _infer_freq(ns_idx) or "S"
            f_ops = _infer_freq(ops_idx) or "S"
            freq = _coarser(f_ns, f_ops)
        else:
            freq = rebin_freq

        NS_tm = NS.timerebin(
            freq=freq, start=st, end=et, method=rebin_method, inplace=False
        )
        OPS_tm = OPS.timerebin(
            freq=freq, start=st, end=et, method=rebin_method, inplace=False
        )

        common_idx = NS_tm.data.index.intersection(OPS_tm.data.index)
        if common_idx.empty:
            raise ValueError(
                "No overlapping timestamps after rebinning NS and OPS. "
                "Check 'rebin_freq' or the requested time window."
            )
        NS_tm._data = NS_tm._data.loc[common_idx]
        OPS_tm._data = OPS_tm._data.loc[common_idx]

    # --- size-domain combination (same logic as before) ----------------------
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
    # keep original NS midpoints where appropriate
    bin_mids[:j] = NS.bin_mids[:j]

    # Select the relevant NS columns (truncate above splice bin)
    ns_df = NS_tm.data.drop(columns=NS_tm.data.columns[j + 2 :])
    # OPS: drop its Total_conc; we recompute it later
    ops_df = OPS_tm.data.drop(columns=["Total_conc"])

    # scale the last kept NS bin by the reduction factor
    last_ns_col = ns_df.columns[-1]
    ns_df[last_ns_col] = ns_df[last_ns_col].astype(float) * factor_reduction

    # merge NS and OPS by index (already aligned in time)
    combined = pd.concat([ns_df, ops_df], axis=1)

    # rename size-bin columns to string mids in one shot
    old_size_cols = list(combined.columns[1 : 1 + len(bin_mids)])
    rename_map = dict(zip(old_size_cols, [str(x) for x in bin_mids], strict=False))
    combined.rename(columns=rename_map, inplace=True)

    # total concentration where both instruments have any data at a timestamp
    mask = ns_df[ns_df.columns[1:]].notna().any(axis=1) & ops_df[
        OPS._sizebin_headers
    ].notna().any(axis=1)

    # sum only over the size-bin columns (first column is the first bin)
    sizebin_span = combined.columns[1 : 1 + (len(ns_ops_edges) - 1)]
    combined["Total_conc"] = combined[sizebin_span].sum(axis=1).where(mask, np.nan)

    # --- build result object and propagate metadata/activities ----------------
    res = Aerosol2D(combined)
    res._activities = NS.activities
    res._activity_periods = NS.activity_periods
    res._meta["bin_edges"] = ns_ops_edges
    res._meta["bin_mids"] = bin_mids
    res._meta["density"] = NS._meta["density"]
    res._meta["instrument"] = f"{NS_data.instrument} + {OPS_data.instrument}"
    res._meta["serial_number"] = f"NS: {NS.serial_number}, OPS: {OPS.serial_number}"
    res._meta["unit"] = "cm⁻³"
    res._meta["dtype"] = "dN"
    res._raw_extra_data = pd.concat([NS._raw_extra_data, OPS._raw_extra_data], axis=1)

    return res


###############################################################################


def combine_measurements(datasets, *, require_same_serial: bool = True):
    """Description:
        Concatenate several measurements from the *same* instrument into one
        continuous time series — for example the same monitor run on three
        separate days, with gaps in between. The inputs are joined along the
        time axis, sorted, de-duplicated, and returned as a single new object
        of the same class. Gaps between recordings are preserved (no
        interpolation is performed).

    Args:
        datasets (Sequence):
            Two or more aerosol objects of the *same* class (all
            :class:`~aerosoltools.aerosol1d.Aerosol1D`, all
            :class:`~aerosoltools.aerosol2d.Aerosol2D`, ...). For size-resolved
            data the size-bin structure (``bin_edges``) must be identical.
        require_same_serial (bool, optional):
            If ``True`` (default), raise when the datasets do not all share the
            same ``serial_number``. Set to ``False`` to combine regardless
            (e.g. when serial numbers are missing). Default is True.

    Returns:
        A new object of the same class as the inputs, spanning the union of all
        input time ranges, with instrument metadata taken from the first input
        and the union of all user-defined activity periods re-marked on the
        combined time axis.

    Raises:
        ValueError:
            If no datasets are given, the classes differ, the size-bin
            structure differs (for 2D data), or the serial numbers differ while
            ``require_same_serial`` is True.

    Examples:
        .. code-block:: python

            import aerosoltools as at

            d1 = at.Load_OPS_file("ops_day1.txt")
            d2 = at.Load_OPS_file("ops_day2.txt")
            d3 = at.Load_OPS_file("ops_day3.txt")
            full = at.combine_measurements([d1, d2, d3])
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("Provide at least one dataset to combine.")
    if len(datasets) == 1:
        return datasets[0].copy_self()

    base = datasets[0]
    cls = type(base)
    for d in datasets[1:]:
        if type(d) is not cls:
            raise ValueError(
                "All datasets must be the same aerosol type to combine "
                f"(got {cls.__name__} and {type(d).__name__})."
            )

    if require_same_serial:
        serials = {str(d.serial_number) for d in datasets}
        if len(serials) > 1:
            raise ValueError(
                "Datasets have different serial numbers "
                f"({', '.join(sorted(serials))}); refusing to combine. "
                "Pass require_same_serial=False to override."
            )

    # Size-resolved data must share an identical bin structure.
    if "bin_edges" in base._meta:
        base_edges = np.asarray(base._meta["bin_edges"], dtype=float)
        for d in datasets[1:]:
            edges = np.asarray(d._meta.get("bin_edges", []), dtype=float)
            if edges.shape != base_edges.shape or not np.allclose(edges, base_edges):
                raise ValueError(
                    "Size-bin structure differs between datasets; cannot "
                    "concatenate. (Use Combine_NS_OPS for different instruments.)"
                )

    def _concat_sorted(frames):
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=0)
        # Keep the first occurrence of any duplicated timestamp, then sort.
        out = out[~out.index.duplicated(keep="first")].sort_index()
        return out

    # Concatenate the numeric main data (boolean activity masks are dropped and
    # re-derived below) and the extra data.
    combined = _concat_sorted(
        [d._data.select_dtypes(exclude="bool") for d in datasets]
    )
    combined_extra = _concat_sorted([d._extra_data for d in datasets])

    result = cls(combined.copy())
    # Restore instrument metadata (instrument, serial, unit, dtype, bins, ...).
    result._meta = dict(base._meta)
    result._raw_data = combined.copy()
    result._extra_data = combined_extra
    result._raw_extra_data = combined_extra.copy()

    # Union the user-defined activity periods across all inputs and re-mark them
    # on the combined time axis.
    merged: dict = {}
    for d in datasets:
        for name, periods in getattr(d, "_activity_periods", {}).items():
            if name == "All data":
                continue
            merged.setdefault(name, [])
            merged[name].extend(list(periods))
    for name, periods in merged.items():
        result.mark_activities({name: periods}, mode="replace")

    return result


###############################################################################

# =========================
# Small utilities
# =========================


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


###############################################################################


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


###############################################################################


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


###############################################################################


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


###############################################################################


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


###############################################################################


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


###############################################################################


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


###############################################################################


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


###############################################################################

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
    parameter: str | tuple | None = None,
    match: str = "exact",  # "exact" | "nearest" | "rebin"
    tolerance: str | pd.Timedelta = "30s",
    rebin_freq: str = None,
    rebin_method: str | Callable = "mean",
    intercept: bool = True,
    uniform_scaling: bool = True,
    outlier_influence: bool = True,
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


###############################################################################


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

    # Dict
    Diff = {
        "BA": f"Difference ({X.unit})",
        "Gi": "Difference (%)",
        "Eu": f"Difference ({X.unit})",
    }  #'Log10 of difference'}

    # --- Plot ----------------------------------------------------------------
    ax.scatter(means, diffs, c="k", s=20, alpha=0.6, marker="o")

    # Labels
    ax.set_title(f"Bland-Altman Plot for {p_X} vz {p_Y}")
    ax.set_xlabel(f"Mean ({X.unit})")
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
    # Plot the bias and the limits of agreement
    fs = 15
    ax.axhline(y=bias, c="grey", ls="--")
    ax.annotate("Bias", (right, bias), (0, 7), textcoords="offset pixels", fontsize=fs)
    ax.annotate(
        f"{bias:+4.2f}",
        (right, bias),
        (0, -12),
        textcoords="offset pixels",
        fontsize=fs,
    )

    # --- Plot the limits of the agreement--------------------------------------
    if method == "Eu":
        # Convert the LOAs from horizontal lines in the log space to gradients of
        # diagonal lines in the native space
        lower_loa_m = 2 * (10 ** (loas[0] - bias) - 1) / (10 ** (loas[0] - bias) + 1)
        upper_loa_m = 2 * (10 ** (loas[1] - bias) - 1) / (10 ** (loas[1] - bias) + 1)
        # Plot the limits of agreement
        x = np.array([left, right])
        y = upper_loa_m * x + bias
        ax.plot(x, y, c="grey", ls="--")
        ax.annotate(
            "Upper LOA", (right, y[1]), (0, 6), textcoords="offset pixels", fontsize=fs
        )
        ax.annotate(
            f"{upper_loa_m:+4.2f} × Mean + Bias",
            (right, y[1]),
            (0, -15),
            textcoords="offset pixels",
            fontsize=fs,
        )
        y = lower_loa_m * x + bias
        ax.plot(x, y, c="grey", ls="--")
        ax.annotate(
            "Lower LOA", (right, y[1]), (0, 6), textcoords="offset pixels", fontsize=fs
        )
        ax.annotate(
            f"{lower_loa_m:+4.2f} × Mean + Bias",
            (right, y[1]),
            (0, -15),
            textcoords="offset pixels",
            fontsize=fs,
        )
        return fig, ax, loas
    else:
        ax.axhline(y=loas[1], c="grey", ls="--")
        ax.axhline(y=loas[0], c="grey", ls="--")
        # Annotations
        ax.annotate(
            "Upper LOA",
            (right, loas[1]),
            (0, 6),
            textcoords="offset pixels",
            fontsize=fs,
        )
        ax.annotate(
            f"{loas[1]:+4.2f}",
            (right, loas[1]),
            (0, -15),
            textcoords="offset pixels",
            fontsize=fs,
        )
        ax.annotate(
            "Lower LOA",
            (right, loas[0]),
            (0, 6),
            textcoords="offset pixels",
            fontsize=fs,
        )
        ax.annotate(
            f"{loas[0]:+4.2f}",
            (right, loas[0]),
            (0, -15),
            textcoords="offset pixels",
            fontsize=fs,
        )

    print(f"For the differences, μ = {bias:.2f} {X.unit} and s = {s:.2f} {X.unit}")

    return fig, ax


###############################################################################
