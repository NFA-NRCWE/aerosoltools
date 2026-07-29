"""Combine measurements from one or more instruments into one dataset."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from ._alignment import _coarser, _infer_freq, _ts


def _align_two_in_time(
    lower, upper, start, end, match, tolerance, rebin_freq, rebin_method
):
    """Time-align two Aerosol2D objects; return the aligned (lower, upper) copies.

    Shared by :func:`combine_size_ranges`. The inputs must already be plain,
    unnormalized number (dN). Alignment honours the ``match`` strategy
    ("exact" | "nearest" | "rebin") over the chosen (or overlapping) window.
    """
    lo_idx = lower.time
    up_idx = upper.time
    st = _ts(start) if start is not None else max(lo_idx.min(), up_idx.min())
    et = _ts(end) if end is not None else min(lo_idx.max(), up_idx.max())
    if st > et:
        raise ValueError(
            "No overlapping time range between the two instruments for the "
            "requested start/end times."
        )

    match = match.lower()
    if match not in {"exact", "nearest", "rebin"}:
        raise ValueError("match must be one of 'exact', 'nearest', or 'rebin'.")

    if match == "exact":
        lo_tm = lower.timecrop(st, et, inplace=False)
        up_tm = upper.timecrop(st, et, inplace=False)
        common = lo_tm.data.index.intersection(up_tm.data.index)
        if common.empty:
            raise ValueError(
                "No common timestamps between the instruments with "
                "match='exact'. Consider match='nearest' or match='rebin'."
            )
        lo_tm._data = lo_tm._data.loc[common]
        up_tm._data = up_tm._data.loc[common]

    elif match == "nearest":
        lo_tm = lower.timecrop(st, et, inplace=False)
        up_tm = upper.timecrop(st, et, inplace=False)
        tol = pd.to_timedelta(tolerance)
        target = lo_tm.data.index.union(up_tm.data.index).sort_values()
        lo_tm._data = lo_tm.data.reindex(target, method="nearest", tolerance=tol)
        up_tm._data = up_tm.data.reindex(target, method="nearest", tolerance=tol)

    else:  # rebin
        if rebin_freq is None:
            freq = _coarser(_infer_freq(lo_idx) or "S", _infer_freq(up_idx) or "S")
        else:
            freq = rebin_freq
        lo_tm = lower.timerebin(
            freq=freq, start=st, end=et, method=rebin_method, inplace=False
        )
        up_tm = upper.timerebin(
            freq=freq, start=st, end=et, method=rebin_method, inplace=False
        )
        common = lo_tm.data.index.intersection(up_tm.data.index)
        if common.empty:
            raise ValueError(
                "No overlapping timestamps after rebinning. Check 'rebin_freq' "
                "or the requested time window."
            )
        lo_tm._data = lo_tm._data.loc[common]
        up_tm._data = up_tm._data.loc[common]

    return lo_tm, up_tm


def combine_size_ranges(
    lower_data: Aerosol2D,
    upper_data: Aerosol2D,
    crossover: float | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    *,
    match: str = "rebin",  # "exact" | "nearest" | "rebin"
    tolerance: str | pd.Timedelta = "30s",
    rebin_freq: str | None = None,
    rebin_method: str | Callable = "mean",
) -> Aerosol2D:
    """Stitch two size-resolved instruments that together **extend** the size
    range (e.g. NanoScan + OPS, FMPS + OPS, NS/FMPS + APS) into one
    time-aligned :class:`~aerosoltools.aerosol2d.Aerosol2D` number spectrum.

        One instrument supplies the small-size end, the other the large-size
        end, and a single ``crossover`` diameter marks where one takes over from
        the other. The cut is a **hard cap at whole size bins**: every bin in the
        result comes from exactly one instrument (no bin mixes counts from both
        and no counts are scaled). The two instruments may be passed in either
        order — the one reaching the larger sizes is used as the upper end.

    Args:
        lower_data (Aerosol2D):
            One of the two instruments (order does not matter).
        upper_data (Aerosol2D):
            The other instrument.
        crossover (float | None, optional):
            Diameter (nm) where the upper instrument takes over from the lower
            one. Bins of the lower instrument with an upper edge at/below the
            crossover are kept; bins of the upper instrument with a lower edge
            at/above it are kept; the two boundary bins have their shared edge
            set to ``crossover`` (counts unchanged). If ``None``, the upper
            instrument's lowest bin edge is used (legacy behaviour).
        start, end (pandas.Timestamp | str | None, optional):
            Time window to combine over. Defaults to the instruments' overlap.
        match (str, optional):
            Time-alignment strategy: ``"rebin"`` (default), ``"exact"`` or
            ``"nearest"`` (see below).
        tolerance (str | pandas.Timedelta, optional):
            Max timestamp separation for ``match="nearest"``. Default ``"30s"``.
        rebin_freq (str | None, optional):
            Target cadence for ``match="rebin"``; defaults to the coarser of the
            two inferred cadences.
        rebin_method (str | Callable, optional):
            Aggregation for ``match="rebin"``. Default ``"mean"``.

    Returns:
        Aerosol2D:
            The merged number size distribution (dN, cm⁻³), with the lower
            instrument's activities/metadata propagated and ``instrument`` set
            to ``"<lower> + <upper>"``.

    Raises:
        ValueError:
            If the two instruments do not form a size-range extension (one must
            reach smaller and the other larger — combining two instruments that
            cover the *same* range, e.g. ELPI + SMPS, is rejected); if the
            crossover lies outside the shared range; or if the time alignment
            yields no common timestamps.

    Notes:
        Number concentration per bin is expressed as plain dN (cm⁻³, no
        ``/dlogDp``) before stitching. Because the cut snaps to whole bins, a
        boundary bin may be marginally widened to meet the crossover, but its
        counts are never split or scaled — so total number is conserved to the
        bin the crossover falls in.

    Examples:
        .. code-block:: python

            import aerosoltools as at

            ns = at.load_ns_file("nanoscan.csv")
            ops = at.load_ops_file("ops.csv")
            combined = at.combine_size_ranges(ns, ops, crossover=400)
            fig, ax = combined.plot_timeseries()
    """
    # --- copy + force dN, unnormalized ---------------------------------------
    a = lower_data.copy_self()
    b = upper_data.copy_self()
    for obj in (a, b):
        obj._convert_to_number_concentration()
        obj.unnormalize_logdp()

    # Order by size range: the instrument reaching larger sizes is the upper one.
    if float(a.bin_edges[-1]) > float(b.bin_edges[-1]):
        lower, upper = b, a
    else:
        lower, upper = a, b

    lo_edges = np.asarray(lower.bin_edges, dtype=float)
    up_edges = np.asarray(upper.bin_edges, dtype=float)

    # --- range-extension guard ----------------------------------------------
    # A meaningful stitch needs the upper instrument to reach above the lower
    # one, and a shared overlap wide enough to place a crossover. The lower
    # instrument supplies the fine (small-diameter) end below the crossover and
    # the upper supplies the coarse end above it. It is fine for one
    # instrument's range to *contain* the other's (e.g. an ELPI spanning
    # 6-9890 nm combined with a NanoScan spanning 10-420 nm): the portion of the
    # wider instrument below the crossover is simply discarded in favour of the
    # finer instrument. Only two instruments reaching the *same* top size (no
    # coarse extension at all) are rejected — there is no sensible crossover.
    extends = up_edges[-1] > lo_edges[-1]
    overlaps = lo_edges[-1] > up_edges[0]
    if not (extends and overlaps):
        raise ValueError(
            "These two instruments do not form a size-range extension with a "
            f"shared overlap (lower {lo_edges[0]:.1f}-{lo_edges[-1]:.1f} nm, "
            f"upper {up_edges[0]:.1f}-{up_edges[-1]:.1f} nm). One instrument "
            "must reach larger sizes than the other, with an overlap to cross "
            "over in."
        )

    if crossover is None:
        crossover = float(up_edges[0])
    Dc = float(crossover)
    lo_min = max(lo_edges[0], up_edges[0])
    hi_max = min(lo_edges[-1], up_edges[-1])
    if not (lo_min <= Dc <= hi_max):
        raise ValueError(
            f"crossover {Dc:.1f} nm is outside the instruments' shared overlap "
            f"({lo_min:.1f}-{hi_max:.1f} nm)."
        )

    # --- time alignment ------------------------------------------------------
    lo_tm, up_tm = _align_two_in_time(
        lower, upper, start, end, match, tolerance, rebin_freq, rebin_method
    )

    # --- size-domain stitch (hard cap at whole bins, no scaling) -------------
    # Lower keeps bins whose upper edge <= Dc; upper keeps bins whose lower edge
    # >= Dc; the shared boundary edge is set to Dc (counts unchanged).
    n_lo = int(np.searchsorted(lo_edges, Dc, side="right") )  # lower bins kept
    first_up = int(np.searchsorted(up_edges, Dc, side="left"))  # first upper edge >= Dc
    n_up = (len(up_edges) - 1) - first_up  # upper bins kept
    if n_lo < 1 or n_up < 1:
        raise ValueError(
            "The chosen crossover leaves one instrument with no bins; pick a "
            "crossover further inside the shared overlap."
        )

    edges = np.concatenate([lo_edges[:n_lo], [Dc], up_edges[first_up + 1 :]])
    mids = np.round(np.sqrt(edges[:-1] * edges[1:]), 1)
    # Keep original midpoints for the untouched interior bins; only the two
    # boundary bins (last lower, first upper) get a recomputed geometric mid.
    lo_mids = np.asarray(lower.bin_mids, dtype=float)
    up_mids = np.asarray(upper.bin_mids, dtype=float)
    if n_lo >= 2:
        mids[: n_lo - 1] = lo_mids[: n_lo - 1]
    if n_up >= 2:
        mids[n_lo + 1 :] = up_mids[first_up + 1 :]

    lo_headers = list(lower._sizebin_headers)
    up_headers = list(upper._sizebin_headers)
    lo_keep = lo_headers[:n_lo]
    up_keep = up_headers[first_up:]

    lower_df = lo_tm.data.loc[:, lo_keep]
    upper_df = up_tm.data[up_keep]
    combined = pd.concat([lower_df, upper_df], axis=1)

    new_mids = [str(m) for m in mids]
    combined = combined.rename(columns=dict(zip(lo_keep + up_keep, new_mids)))
    # Calculate total concentration and place it first
    combined.insert( 0,"Total_conc",  combined.sum(axis=1) )

    res = Aerosol2D(combined)
    res.mark_activities(lower.activity_periods)
    # res._activities = lower.activities
    # res._activity_periods = lower.activity_periods
    res._meta["bin_edges"] = edges
    res._meta["bin_mids"] = mids
    res._meta["density"] = lower._meta.get("density", 1.0)
    res._meta["instrument"] = f"{lower.instrument} + {upper.instrument}"
    res._meta["serial_number"] = (
        f"{lower.instrument}: {lower.serial_number}, "
        f"{upper.instrument}: {upper.serial_number}"
    )
    res._meta["unit"] = "cm⁻³"
    res._meta["dtype"] = "dN"
    res._raw_extra_data = pd.concat(
        [lower._raw_extra_data, upper._raw_extra_data], axis=1
    )
    return res


def combine_measurements(datasets, *, require_same_serial: bool = True):
    """Concatenate several measurements from the *same* instrument into one
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

            d1 = at.load_ops_file("ops_day1.txt")
            d2 = at.load_ops_file("ops_day2.txt")
            d3 = at.load_ops_file("ops_day3.txt")
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
                    "concatenate. (Use combine_size_ranges for different instruments.)"
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
    combined = _concat_sorted([d._data.select_dtypes(exclude="bool") for d in datasets])
    combined_extra = _concat_sorted([d._extra_data for d in datasets])

    result = cls(combined.copy())
    # Restore instrument metadata (instrument, serial, unit, dtype, bins, ...).
    result._meta = dict(base._meta)
    result._raw_data = combined.copy()
    result._extra_data = combined_extra
    result._raw_extra_data = combined_extra.copy()

    # Correlated APS (Aerosol3d): the concat above covers the aerodynamic axis,
    # but the optical companion and the optical×aerodynamic correlation matrix
    # must be concatenated too — otherwise the combined object silently drops to
    # an aerodynamic-only record and loses its Aero↔Optical capabilities.
    from ..aerosol3d import Aerosol3d  # local import avoids a circular import

    if isinstance(result, Aerosol3d):
        opticals = [getattr(d, "_optical", None) for d in datasets]
        if all(o is not None for o in opticals):
            result._optical = combine_measurements(opticals, require_same_serial=False)
        corrs = [getattr(d, "_correlation", None) for d in datasets]
        if all(c is not None for c in corrs):
            result._correlation = _concat_sorted(corrs)

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
