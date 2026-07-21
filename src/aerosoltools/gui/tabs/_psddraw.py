"""Shared PSD drawing helpers for the single-PSD and PSD-comparison tabs.

Both panes render mean particle size distributions via the library's own
:meth:`Aerosol2D.plot_psd` (so the curve matches the rest of aerosoltools) and
then recolour / optionally re-draw them as bars with a ±σ spread. Keeping the
"draw one PSD curve" logic here means the single-dataset pane (which adds
lognormal fitting) and the comparison pane (display only) stay consistent.
"""

from __future__ import annotations

import numpy as np


def draw_psd_curve(ax, obj, act, *, color, label, normalize, bars, show_band):
    """Draw one mean-PSD curve for ``(obj, act)`` onto ``ax``; return ``(x, y)``.

    Returns the mean curve's ``(bin_mids, values)`` (for fitting / scoring), or
    ``None`` when the activity has no samples in this object. When ``bars`` is
    set the line is replaced by edge-aligned bars (with error bars for the ±σ
    spread); otherwise the line is recoloured and the ±σ band kept only when
    ``show_band`` is on.
    """
    n_lines = len(ax.lines)
    n_coll = len(ax.collections)
    try:
        obj.plot_psd(activities=[act], normalize=normalize, ax=ax)
    except Exception:
        return None
    new_lines = list(ax.lines)[n_lines:]
    new_coll = list(ax.collections)[n_coll:]
    if not new_lines:
        return None
    main = new_lines[0]
    xd = np.asarray(main.get_xdata(), dtype=float)
    yd = np.asarray(main.get_ydata(), dtype=float)

    if bars:
        for ln in new_lines:  # replace plot_psd's line with bars
            ln.remove()
        for coll in new_coll:  # a fill_between reads poorly behind bars
            coll.remove()
        sigma = None
        if show_band:
            try:
                _, _, sigma = obj._activity_psd_stats(act, normalize=normalize)
            except Exception:
                sigma = None
        _draw_bars(ax, obj, xd, yd, color, label, sigma)
        return xd, yd

    for j, line in enumerate(new_lines):
        line.set_color(color)
        line.set_label(label if j == 0 else "_nolegend_")
    for coll in new_coll:  # fill_between ±1σ envelope
        if show_band:
            coll.set_color(color)
            coll.set_alpha(0.18)
        else:
            coll.remove()
    return xd, yd


def _draw_bars(ax, obj, bin_mids, heights, color, label, sigma=None) -> None:
    """Draw one PSD as edge-aligned bars so each size bin's width is visible."""
    edges = np.asarray(obj.bin_edges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if heights.size != edges.size - 1:
        # bin_edges/levels mismatch — fall back to a plain line.
        ax.plot(bin_mids, heights, color=color, lw=2.0, label=label)
        return
    ax.bar(
        edges[:-1],
        heights,
        width=np.diff(edges),
        align="edge",
        facecolor=color,
        edgecolor=color,
        linewidth=0.7,
        alpha=0.55,
        label=label,
        zorder=3,
    )
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        centers = edges[:-1] + np.diff(edges) / 2
        finite = np.isfinite(heights) & np.isfinite(sigma)
        if finite.any():
            ax.errorbar(
                centers[finite],
                heights[finite],
                yerr=sigma[finite],
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                alpha=0.8,
                zorder=4,
            )


def data_ylim(ax, log_y: bool):
    """Compute y-limits from the *data* artists on ``ax`` (lines, bars, bands).

    Thin wrapper over the shared axis-limit policy (:mod:`_autoscale`): the range
    is measured from the plotted data (mean curves, bars and the ±σ band), so a
    lognormal fit's near-zero tails or a huge σ spread never drive the limits.
    A non-negative linear PSD is anchored at 0. Returns ``(bottom, top)`` or
    ``None`` when there is nothing to measure.
    """
    from . import _autoscale

    return _autoscale.y_limits([ax], log_y=log_y, anchor_zero=True)
