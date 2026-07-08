"""Combine measurements from one or more instruments into one dataset."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from ._common import _coarser, _infer_freq, _ts


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
    """Description:
        Stitch two size-resolved instruments that together **extend** the size
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

            ns = at.Load_NS_file("nanoscan.csv")
            ops = at.Load_OPS_file("ops.csv")
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
    # A meaningful stitch needs the lower instrument to reach below the upper
    # one and the upper to reach above the lower one, with a shared overlap. Two
    # instruments covering essentially the same range (e.g. ELPI + SMPS) are
    # rejected — there is no sensible single crossover.
    extends = up_edges[-1] > lo_edges[-1] and lo_edges[0] < up_edges[0]
    overlaps = lo_edges[-1] > up_edges[0]
    if not (extends and overlaps):
        raise ValueError(
            "These two instruments do not form a size-range extension with a "
            f"shared overlap (lower {lo_edges[0]:.1f}-{lo_edges[-1]:.1f} nm, "
            f"upper {up_edges[0]:.1f}-{up_edges[-1]:.1f} nm). Combining "
            "instruments that cover the same range is not supported."
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
    n_lo = int(np.searchsorted(lo_edges, Dc, side="right") - 1)  # lower bins kept
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

    lower_df = lo_tm.data.drop(
        columns=[h for h in lo_headers if h not in lo_keep] + ["Total_conc"],
        errors="ignore",
    )
    upper_df = up_tm.data[up_keep]
    combined = pd.concat([lower_df, upper_df], axis=1)

    new_mids = [str(m) for m in mids]
    combined = combined.rename(columns=dict(zip(lo_keep + up_keep, new_mids)))
    combined["Total_conc"] = combined[new_mids].sum(axis=1)

    res = Aerosol2D(combined)
    res._activities = lower.activities
    res._activity_periods = lower.activity_periods
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

    warnings.warn(
        "Combine_NS_OPS is deprecated; use combine_size_ranges (it works for "
        "any pair of range-extending instruments and lets you set the "
        "crossover diameter).",
        DeprecationWarning,
        stacklevel=2,
    )
    return combine_size_ranges(
        NS_data,
        OPS_data,
        crossover=None,
        start=start,
        end=end,
        match=match,
        tolerance=tolerance,
        rebin_freq=rebin_freq,
        rebin_method=rebin_method,
    )


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
