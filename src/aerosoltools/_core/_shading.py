"""Shared activity-period shading for the plotting mixins and the GUI.

The 1D and 2D plotting methods and the GUI both shade activity periods as
translucent vertical spans with one consistent colour per activity (assigned
from the ``gist_ncar`` colormap over *all* activities, so a given activity keeps
the same colour across every plot). This module holds that logic once so the
three call sites cannot drift apart.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

#: Colormap activities are coloured from, and the span transparency. Kept here
#: so every shaded plot (library or GUI) matches.
ACTIVITY_CMAP = "gist_ncar"
ACTIVITY_ALPHA = 0.3


def resolve_activities(activity_periods: Dict, mark, include_all_data: bool = False) -> List[str]:
    """Resolve a ``mark`` spec to the list of activity names to shade.

    Args:
        activity_periods: Mapping of activity name -> list of ``(start, end)``.
        mark: ``True`` shades every activity (optionally including ``"All data"``);
            a sequence of names shades only those; anything falsy shades none.
        include_all_data: Whether ``"All data"`` is included when ``mark`` is ``True``.

    Returns:
        Activity names to shade, in the stable (sorted) colour order.
    """
    all_names = sorted(activity_periods.keys())
    if mark is True:
        return [a for a in all_names if include_all_data or a != "All data"]
    if isinstance(mark, (list, tuple, set)):
        return [a for a in all_names if a in mark]
    return []


def activity_colors(activity_periods: Dict) -> Dict:
    """Map each activity to a stable colour from :data:`ACTIVITY_CMAP`."""
    all_names = sorted(activity_periods.keys())
    cmap = plt.colormaps.get_cmap(ACTIVITY_CMAP)
    return {
        name: cmap(i / max(1, len(all_names))) for i, name in enumerate(all_names)
    }


def shade_activities(
    ax,
    activity_periods: Dict,
    selected: Sequence[str],
    *,
    alpha: float = ACTIVITY_ALPHA,
    zorder: int = 2,
    legend: bool = True,
    legend_kw: dict | None = None,
) -> bool:
    """Shade ``selected`` activities on ``ax`` as translucent vertical spans.

    Each activity's occurrences share one colour and a single legend entry.

    Args:
        ax: Axis to draw on.
        activity_periods: Mapping of activity name -> list of ``(start, end)``.
        selected: Activities to shade (see :func:`resolve_activities`).
        alpha: Span transparency.
        zorder: Draw order for the spans.
        legend: Whether to (re)draw the legend when anything was shaded.
        legend_kw: Extra keyword arguments forwarded to ``ax.legend``.

    Returns:
        ``True`` if at least one span was drawn.
    """
    colors = activity_colors(activity_periods)
    drew = False
    for name in selected:
        first = True
        for start, end in activity_periods.get(name, []):
            ax.axvspan(
                mdates.date2num(pd.Timestamp(start)),
                mdates.date2num(pd.Timestamp(end)),
                color=colors.get(name, "black"),
                alpha=alpha,
                label=name if first else None,
                zorder=zorder,
            )
            first = False
            drew = True
    if drew and legend:
        ax.legend(**(legend_kw or {}))
    return drew
