"""Shared helpers for the aerosoltools GUI.

These functions adapt the aerosoltools data classes to the needs of the GUI:
resolving per-column dtype/unit metadata (which may be scalars *or* dicts on
:class:`AerosolAlt`), enumerating plottable columns, shading activity spans on
a Matplotlib axis, and adding/removing activities interactively.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib as mpl
import matplotlib.dates as mdates
import pandas as pd

from ..aerosol1d import Aerosol1D
from ..aerosol2d import Aerosol2D

# Sentinel used by plottable_columns to mark the canonical "total" series.
TOTAL = "<total>"


def is_2d(obj) -> bool:
    """Return True if ``obj`` is a size-resolved :class:`Aerosol2D`."""
    return isinstance(obj, Aerosol2D)


def describe(obj: Aerosol1D, column: str | None = None) -> Tuple[str, str]:
    """Resolve the (dtype, unit) pair for a column.

    Both ``obj.dtype`` and ``obj.unit`` may be plain strings (Aerosol1D /
    Aerosol2D) or per-column dicts (AerosolAlt). This helper returns sensible
    strings in either case.

    Args:
        obj: The aerosol object.
        column: Column name to look up in the per-column case. When ``None``
            and the metadata is a dict, the first entry is used.

    Returns:
        A ``(dtype, unit)`` tuple of display strings.
    """
    dtype = obj.dtype
    unit = obj.unit

    def _resolve(meta):
        if isinstance(meta, dict):
            if column is not None and column in meta:
                return str(meta[column])
            if meta:
                return str(next(iter(meta.values())))
            return ""
        return str(meta)

    return _resolve(dtype), _resolve(unit)


def plottable_columns(obj: Aerosol1D) -> List[Tuple[str, str, str]]:
    """Enumerate columns that can be plotted as a 1D time series.

    Returns:
        A list of ``(label, kind, name)`` tuples where ``kind`` is one of
        ``"total"``, ``"data"`` or ``"extra"``. The first entry is always the
        canonical total concentration series.
    """
    cols: List[Tuple[str, str, str]] = [("Total concentration", "total", TOTAL)]

    activities = set(getattr(obj, "activities", []))
    size_headers = set(getattr(obj, "_sizebin_headers", [])) if is_2d(obj) else set()

    numeric = obj.data.select_dtypes(exclude="bool")
    for name in numeric.columns:
        if name in activities or name in size_headers:
            continue
        if name == "Total_conc":
            continue  # already represented by the canonical "total" entry
        cols.append((f"{name} (data)", "data", name))

    if obj.extra_data is not None and not obj.extra_data.empty:
        extra_numeric = obj.extra_data.select_dtypes(exclude="bool")
        for name in extra_numeric.columns:
            cols.append((f"{name} (extra)", "extra", name))

    return cols


def series_for(obj: Aerosol1D, kind: str, name: str) -> pd.Series:
    """Return the pandas Series for a ``(kind, name)`` selection."""
    if kind == "total":
        return obj.total_concentration
    if kind == "extra":
        return obj.extra_data[name]
    return obj.data[name]


def user_activities(obj: Aerosol1D) -> List[str]:
    """Return user-defined activity names (excluding the built-in 'All data')."""
    return [a for a in obj.activities if a != "All data"]


def shade_activities(ax, obj: Aerosol1D, include_all_data: bool = False) -> None:
    """Shade each activity period on ``ax`` as a translucent vertical span.

    Mirrors the styling used by the core plotting methods (the ``gist_ncar``
    colormap, alpha 0.3) so the interactive view matches library output.
    """
    periods = getattr(obj, "_activity_periods", {})
    all_names = sorted(periods.keys())
    if not all_names:
        return

    cmap = mpl.colormaps["gist_ncar"]
    colors = {
        name: cmap(i / max(1, len(all_names))) for i, name in enumerate(all_names)
    }

    selected = (
        all_names if include_all_data else [a for a in all_names if a != "All data"]
    )
    drew_label = False
    for name in selected:
        first = True
        for start, end in periods[name]:
            ax.axvspan(
                mdates.date2num(pd.Timestamp(start)),
                mdates.date2num(pd.Timestamp(end)),
                color=colors[name],
                alpha=0.3,
                label=name if first else None,
                zorder=1,
            )
            first = False
            drew_label = True
    if drew_label:
        ax.legend(loc="upper right", fontsize=8)


def add_activity(obj: Aerosol1D, name: str, start, end) -> None:
    """Append a period to an activity, keeping any periods already defined.

    A task can occur several times, so this accumulates the new ``(start, end)``
    onto the existing list before calling ``mark_activities``. (``mark_activities``
    unions the boolean mask but *replaces* ``activity_periods`` with whatever list
    it is given, so the full list must be passed for the shading to show every
    occurrence.)
    """
    periods = list(obj._activity_periods.get(name, []))
    periods.append((pd.Timestamp(start), pd.Timestamp(end)))
    obj.mark_activities({name: periods}, mode="union")


def delete_activity(obj: Aerosol1D, name: str) -> None:
    """Remove an activity and its mask column.

    The aerosoltools classes have no public delete method, so this manipulates
    the (package-internal) activity bookkeeping directly. It is a no-op for the
    built-in ``"All data"`` activity.
    """
    if name == "All data":
        return
    if name in obj.data.columns:
        obj._data = obj._data.drop(columns=[name])
    if name in obj._activities:
        obj._activities.remove(name)
    obj._activity_periods.pop(name, None)
