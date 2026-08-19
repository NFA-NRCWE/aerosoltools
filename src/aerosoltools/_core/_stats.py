"""Shared helper for computing named summary statistics on a time-step series.

Used by :mod:`statistics` and :mod:`statistics2d` so ``summarize_activities``
computes the same set of stats (mean/std/min/max/median) the same way for
both 1D and 2D aerosol objects.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

#: Stat name -> function computing it from a pandas Series.
_STAT_FUNCS = {
    "mean": lambda s: float(s.mean()),
    "std": lambda s: float(s.std()),
    "min": lambda s: float(s.min()),
    "max": lambda s: float(s.max()),
    "median": lambda s: float(s.median()),
}

#: Recognised stat names, in a stable (insertion) order.
VALID_STATS = tuple(_STAT_FUNCS)


def compute_stats(series: pd.Series, stats: Sequence[str]) -> dict[str, float]:
    """Return ``{stat: value}`` for each name in ``stats``.

    Args:
        series: Values to summarize; NaNs are ignored by the underlying
            pandas reductions. May be empty.
        stats: Stat names, each one of :data:`VALID_STATS`.

    Returns:
        dict[str, float]: One entry per requested stat, each NaN when
        ``series`` is empty or has no finite values.

    Raises:
        ValueError: If a name in ``stats`` is not one of :data:`VALID_STATS`.
    """
    unknown = [name for name in stats if name not in _STAT_FUNCS]
    if unknown:
        raise ValueError(
            f"Unknown stat(s) {unknown}. Expected one of {sorted(VALID_STATS)}."
        )
    if series.empty or not np.isfinite(series).any():
        return {name: float("nan") for name in stats}
    return {name: _STAT_FUNCS[name](series) for name in stats}
