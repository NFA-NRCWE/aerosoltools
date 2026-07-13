"""Shared helpers for the aerosoltools GUI.

These functions adapt the aerosoltools data classes to the needs of the GUI:
resolving per-column dtype/unit metadata (which may be scalars *or* dicts on
multi-channel instruments), enumerating plottable columns, shading activity spans on
a Matplotlib axis, and adding/removing activities interactively.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .._core import _shading
from .._core._labels import base_dtype  # re-exported for GUI use
from .._core.nonparticle import _NonParticleMixin
from ..aerosol1d import Aerosol1D
from ..aerosol2d import Aerosol2D

__all__ = [
    "TOTAL",
    "THRESHOLD_COLOR",
    "base_dtype",
    "is_2d",
    "is_particle",
    "describe",
    "measurement_label",
    "plottable_columns",
    "grouped_columns",
    "series_for",
    "user_activities",
    "shade_activities",
    "draw_threshold",
    "delete_activity",
    "set_activity_periods",
]

# Sentinel used by plottable_columns to mark the canonical "total" series.
TOTAL = "<total>"

# Colour used for the concentration-threshold (e.g. OEL) marker line. A strong
# red reads as a "limit" and stays distinct from the data and activity shading.
THRESHOLD_COLOR = "#d62728"


def draw_threshold(ax, value, label: str | None = None) -> None:
    """Draw a horizontal threshold line (e.g. an OEL) across ``ax``.

    Used by the time-series-style plots so it is visually obvious at which times
    the concentration rose above (or fell below) a user-defined limit such as an
    occupational exposure limit. The line is added with a legend entry and the
    legend is (re)drawn so the label shows alongside any activity/series labels
    already present.

    Args:
        ax: Axis to draw on.
        value: Threshold value in the axis' current y-units. Ignored when
            ``None`` or not finite.
        label: Legend text; defaults to ``"Threshold (<value>)"``.
    """
    if value is None:
        return
    try:
        y = float(value)
    except (TypeError, ValueError):
        return
    if not pd.notna(y):
        return
    text = label.strip() if (label and label.strip()) else f"Threshold ({y:g})"
    ax.axhline(
        y, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.6, zorder=5, label=text
    )
    # Rebuild the legend so the threshold appears next to any existing labels.
    ax.legend(loc="upper right", fontsize=8)


def is_2d(obj) -> bool:
    """Return True if ``obj`` is a size-resolved :class:`Aerosol2D`."""
    return isinstance(obj, Aerosol2D)


def is_particle(obj) -> bool:
    """Return True if ``obj`` measures a particle quantity.

    Particle instruments have a meaningful ``total_concentration``; the
    non-particle classes (gases, black carbon, environmental, LDSA-only) do not,
    so callers should offer/serve the generic "Total concentration" metric only
    when this is True.
    """
    return not isinstance(obj, _NonParticleMixin)


def describe(obj: Aerosol1D, column: str | None = None) -> Tuple[str, str]:
    """Resolve the (dtype, unit) pair for a column.

    Both ``obj.dtype`` and ``obj.unit`` may be plain strings (Aerosol1D /
    Aerosol2D) or per-column dicts (multi-channel instruments). This helper returns sensible
    strings in either case.

    Args:
        obj: The aerosol object.
        column: Column name to look up in the per-column case. When ``None``
            and the metadata is a dict, the first entry is used.

    Returns:
        A ``(dtype, unit)`` tuple of display strings.
    """
    # Delegate to the core per-column-aware accessors (single source of truth).
    return obj.dtype_of(column), obj.unit_of(column)


def measurement_label(obj: Aerosol1D, column: str | None = None) -> Tuple[str, str]:
    """Resolve a ``(name, unit)`` display pair for a dataset's primary series.

    ``name`` is *what* the series measures — the loader-supplied
    :attr:`~aerosoltools.Aerosol1D.measurement` (e.g. ``"Cl₂"``) when set, else
    the selected/primary channel name for a multi-channel object, else the
    generic ``"Total concentration"``. ``unit`` comes from :func:`describe`, so
    per-channel dict units on multi-channel instruments are handled. This lets panes
    label a gas or black-carbon series correctly instead of always writing
    "Total concentration".

    Args:
        obj: The aerosol object.
        column: A specific column to label (its own name/unit); ``None`` for the
            object's primary series.

    Returns:
        A ``(name, unit)`` tuple of display strings.
    """
    _dtype, unit = describe(obj, column)
    name = getattr(obj, "measurement", None)
    if not name:
        if column is not None:
            name = column
        elif "Total_conc" in obj.data.columns:
            name = "Total concentration"
        else:
            # Multi-channel object: the primary is the first non-activity column.
            activities = set(getattr(obj, "activities", []))
            data_cols = [c for c in obj.data.columns if c not in activities]
            name = data_cols[0] if data_cols else "Total concentration"
    return str(name), unit


def plottable_columns(obj: Aerosol1D) -> List[Tuple[str, str, str]]:
    """Enumerate columns that can be plotted as a 1D time series.

    Returns:
        A list of ``(label, kind, name)`` tuples where ``kind`` is one of
        ``"total"``, ``"data"`` or ``"extra"``. For particle instruments the
        first entry is the canonical total concentration series; non-particle
        instruments (gas, black carbon, LDSA, T/RH) have no total concentration,
        so their named channels are listed directly instead.
    """
    cols: List[Tuple[str, str, str]] = []
    if is_particle(obj):
        cols.append(("Total concentration", "total", TOTAL))

    activities = set(getattr(obj, "activities", []))
    size_headers = set(getattr(obj, "_sizebin_headers", [])) if is_2d(obj) else set()

    def _plottable(series: pd.Series) -> bool:
        """A column is plottable only if it holds some finite numeric value.

        Restricting to numeric, non-empty columns keeps text/all-blank columns
        (e.g. a stray empty column from a trailing delimiter in an export) out of
        the series picker, so selecting one can never crash the plot.
        """
        numeric = pd.to_numeric(series, errors="coerce")
        return bool(numeric.notna().any())

    for name in obj.data.select_dtypes(exclude="bool").columns:
        if name in activities or name in size_headers:
            continue
        if name == "Total_conc":
            continue  # already represented by the canonical "total" entry
        if _plottable(obj.data[name]):
            cols.append((f"{name} (data)", "data", name))

    if obj.extra_data is not None and not obj.extra_data.empty:
        for name in obj.extra_data.select_dtypes(exclude="bool").columns:
            if _plottable(obj.extra_data[name]):
                cols.append((f"{name} (extra)", "extra", name))

    return cols


def grouped_columns(
    obj: Aerosol1D,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Split :func:`plottable_columns` into ``(standard, extra)`` groups.

    ``standard`` holds the canonical total plus the core data channels;
    ``extra`` holds the (often numerous) housekeeping/diagnostic columns. The
    pickers show the standard group up front and tuck the extra group behind a
    labelled separator so instruments with 20–30 diagnostic channels don't
    overwhelm the selector. Non-plottable columns are already excluded upstream.
    """
    standard: List[Tuple[str, str, str]] = []
    extra: List[Tuple[str, str, str]] = []
    for label, kind, name in plottable_columns(obj):
        (extra if kind == "extra" else standard).append((label, kind, name))
    return standard, extra


def series_for(obj: Aerosol1D, kind: str, name: str) -> pd.Series:
    """Return the pandas Series for a ``(kind, name)`` selection.

    Non-total series are coerced to numeric (invalid entries become ``NaN``) so
    a column stored as text still plots and can never raise on conversion.
    """
    if kind == "total":
        # ``_primary`` is the total concentration for particle instruments and
        # the main channel for non-particle ones, so this never raises.
        return obj._primary
    series = obj.extra_data[name] if kind == "extra" else obj.data[name]
    return pd.to_numeric(series, errors="coerce")


def user_activities(obj: Aerosol1D) -> List[str]:
    """Return user-defined activity names (excluding the built-in 'All data')."""
    return [a for a in obj.activities if a != "All data"]


def shade_activities(ax, obj: Aerosol1D, include_all_data: bool = False) -> None:
    """Shade each activity period on ``ax`` as a translucent vertical span.

    Thin wrapper over :func:`aerosoltools._core._shading.shade_activities` so the
    interactive view matches the library's own plots exactly (same colormap,
    alpha and per-activity colours). Uses the GUI's draw order (behind the data)
    and compact legend.

    Each span is **clamped to the object's own data range** before drawing: an
    activity may be scoped to several datasets and so extend past *this* one's
    data (e.g. a task shared with a longer recording). Shading only where the
    data actually exists keeps the plot's x-limits on the data instead of
    stretching the view out to reach a span with no samples under it.
    """
    periods = getattr(obj, "_activity_periods", {})
    periods = _clamp_periods_to_data(periods, obj)
    selected = _shading.resolve_activities(
        periods, True, include_all_data=include_all_data
    )
    _shading.shade_activities(
        ax, periods, selected, zorder=1, legend_kw={"loc": "upper right", "fontsize": 8}
    )


def _clamp_periods_to_data(periods: dict, obj: Aerosol1D) -> dict:
    """Clamp each activity's spans to ``obj``'s data range (keys preserved).

    Every activity name is kept (so the shared per-activity colour assignment,
    which is keyed on the full name set, does not shift); only the drawn spans
    are trimmed to ``[t_min, t_max]`` and any span entirely outside the data is
    dropped. Returns the original mapping unchanged when the object has no time.
    """
    t = getattr(obj, "time", None)
    if t is None or len(t) == 0:
        return periods
    tmin, tmax = pd.Timestamp(t.min()), pd.Timestamp(t.max())
    clamped: dict = {}
    for name, spans in periods.items():
        kept = []
        for start, end in spans:
            s = max(pd.Timestamp(start), tmin)
            e = min(pd.Timestamp(end), tmax)
            if e > s:
                kept.append((s, e))
        clamped[name] = kept
    return clamped


# -- per-object activity mutation -----------------------------------------
# These are the single place where the GUI mutates an aerosol object's activity
# bookkeeping. The project layer (gui/project.py) owns the shared registry and
# loops over datasets, delegating each per-object change to one of these — so the
# "how to change one object" policy lives here, not spread across both layers.


def set_activity_periods(obj: Aerosol1D, name: str, periods) -> None:
    """Replace one activity's periods on a single object (``replace`` mode)."""
    norm = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in periods]
    obj.mark_activities({name: norm}, mode="replace")


def delete_activity(obj: Aerosol1D, name: str) -> None:
    """Remove an activity and its mask column from a single object.

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


def rename_activity(obj: Aerosol1D, old_name: str, new_name: str) -> None:
    """Rename an activity on a single object; no-op if it isn't present.

    An activity may be scoped to only some datasets, so the object handed in
    here need not carry it (or ``"All data"``); in those cases there is nothing
    to rename.
    """
    if old_name in ("All data", new_name) or old_name not in getattr(
        obj, "_activities", []
    ):
        return
    obj.rename_activity(old_name, new_name)
