"""Combine measurements from one or more instruments into one dataset."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from ._common import _coarser, _infer_freq, _ts


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
