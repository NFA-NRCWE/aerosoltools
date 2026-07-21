"""Shared axis-limit policy for the plot tabs.

One place that answers "what should the axis limits be, given what is currently
drawn?" — so every pane derives its limits the same way instead of each rolling
its own. It reads the *data artists* actually on the axes (lines, bars/patches
and filled ±σ bands) and returns tight limits with a small margin. Panes call
this whenever a display setting or the dataset selection changes; a pane that
just needs Matplotlib's own autoscale can keep using ``relim``/``autoscale_view``
(:meth:`_PlotTab._sync_toolbar_home`), which follows the same "limits from the
drawn data" principle.

Kept Qt-free and dependency-light so it can be reasoned about on its own.
"""

from __future__ import annotations

import numpy as np

# Log-scale headroom factors (multiplicative), matching the PSD panes' feel.
_LOG_LO, _LOG_HI = 0.8, 1.25


def _artist_values(ax):
    """Return ``(xs, ys)`` lists of value arrays from a single axis' data artists."""
    xs, ys = [], []
    for line in ax.lines:
        xs.append(np.asarray(line.get_xdata(), dtype=float))
        ys.append(np.asarray(line.get_ydata(), dtype=float))
    for patch in ax.patches:  # bars span x..x+width and y..y+height
        try:
            x0 = float(patch.get_x())
            y0 = float(patch.get_y())
            xs.append(np.array([x0, x0 + float(patch.get_width())]))
            ys.append(np.array([y0, y0 + float(patch.get_height())]))
        except (TypeError, ValueError):
            continue
    for coll in ax.collections:  # ±σ fill_between envelopes, scatter, …
        for path in coll.get_paths():
            v = path.vertices
            if v.size:
                xs.append(np.asarray(v[:, 0], dtype=float))
                ys.append(np.asarray(v[:, 1], dtype=float))
    return xs, ys


def data_bounds(axes):
    """Return ``(x_values, y_values)`` finite arrays across ``axes`` (or None each)."""
    xs, ys = [], []
    for ax in axes:
        ax_xs, ax_ys = _artist_values(ax)
        xs += ax_xs
        ys += ax_ys

    def _finite(arrs):
        if not arrs:
            return None
        a = np.concatenate(arrs)
        a = a[np.isfinite(a)]
        return a if a.size else None

    return _finite(xs), _finite(ys)


def _linear_range(vals, margin, anchor_zero):
    """Tight ``(lo, hi)`` for linear data with a fractional ``margin``."""
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - min(0.0, lo) if anchor_zero else hi - lo
    m = margin * span if span > 0 else (margin * abs(hi) or 1.0)
    bottom = (0.0 if lo >= 0 else lo - m) if anchor_zero else lo - m
    return bottom, hi + m


def _log_range(vals):
    """Tight ``(lo, hi)`` for log data (positive values only), or None."""
    pos = vals[vals > 0]
    if pos.size == 0:
        return None
    return float(pos.min()) * _LOG_LO, float(pos.max()) * _LOG_HI


def limits(axes, *, log_x=False, log_y=False, margin=0.05, y_anchor_zero=False):
    """Compute ``(xlim, ylim)`` from the data drawn on ``axes``.

    Args:
        axes: One or more axes to read artists from (e.g. an axis and its twin).
        log_x / log_y: Whether that axis is log-scaled (positive-only range).
        margin: Fractional padding added to a linear range.
        y_anchor_zero: Anchor a non-negative linear y-range at 0 (for PSDs etc.).

    Returns:
        ``(xlim, ylim)`` where each is a ``(lo, hi)`` tuple or ``None`` when
        there is nothing to measure on that axis.
    """
    xv, yv = data_bounds(axes)
    xlim = None
    if xv is not None:
        xlim = _log_range(xv) if log_x else _linear_range(xv, margin, False)
    ylim = None
    if yv is not None:
        ylim = _log_range(yv) if log_y else _linear_range(yv, margin, y_anchor_zero)
    return xlim, ylim


def y_limits(axes, *, log_y=False, margin=0.05, anchor_zero=True):
    """Convenience wrapper returning just the y-limits (see :func:`limits`)."""
    return limits(axes, log_y=log_y, margin=margin, y_anchor_zero=anchor_zero)[1]
