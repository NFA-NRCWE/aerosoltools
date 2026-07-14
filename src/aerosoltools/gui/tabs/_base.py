"""Shared base class and helpers for the GUI's data tabs.

:class:`_PlotTab` provides the embedded Matplotlib figure + navigation toolbar
and the "Save plot…" export pipeline; :func:`_export_table` / :func:`_tune_table`
are table helpers; :func:`_active_color_cycle` reads the live colour cycle so
tab colours stay theme-correct on screen and in exports.
"""

from __future__ import annotations

import io
import math
import os
import pickle
import traceback

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from ..qt import Figure, FigureCanvas, NavigationToolbar, QtCore, QtWidgets
from ..view import theme


def _export_table(
    parent, df: pd.DataFrame, default_stem: str, with_index: bool
) -> None:
    """Save a DataFrame to .xlsx (or .csv) via a file dialog.

    Args:
        parent: Widget used as the dialog parent.
        df: The table to export.
        default_stem: Suggested file-name stem (no extension).
        with_index: Whether to write the DataFrame index (e.g. timestamps).
    """
    if df is None or df.empty:
        QtWidgets.QMessageBox.information(
            parent, "Nothing to export", "The table is empty."
        )
        return
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        "Export table",
        f"{default_stem}.xlsx",
        "Excel workbook (*.xlsx);;CSV file (*.csv)",
    )
    if not path:
        return
    try:
        if path.lower().endswith(".csv"):
            df.to_csv(path, index=with_index)
        else:
            df.to_excel(path, index=with_index)
    except Exception:
        QtWidgets.QMessageBox.warning(
            parent, "Export failed", traceback.format_exc(limit=2)
        )
        return
    QtWidgets.QMessageBox.information(parent, "Exported", f"Table saved to:\n{path}")


def _tune_table(view: "QtWidgets.QTableView") -> None:
    """Size table columns to fit header + content so labels never clip.

    ``ResizeToContents`` derives each column width from the (style-aware) size
    hint, which includes header padding — unlike fixed/interactive widths,
    which can clip styled headers. ``setResizeContentsPrecision`` caps how many
    rows are sampled so this stays fast on very large frames.
    """
    hh = view.horizontalHeader()
    hh.setResizeContentsPrecision(60)
    hh.setMinimumSectionSize(48)
    hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    view.verticalHeader().setResizeContentsPrecision(60)


class _PlotTab(QtWidgets.QWidget):
    """Base class providing an embedded Matplotlib figure + navigation toolbar."""

    #: Short tag used in the suggested export file name.
    export_tag = "plot"

    #: Whether the shared cursor-zoom / right-drag-pan navigation is attached.
    #: Panes with their own drag/rotate navigation (e.g. the 3D APS view) opt
    #: out by setting this False.
    interactive_nav = True

    def __init__(self, main, nrows: int = 1):
        """Build the embedded figure, canvas, toolbar and a Save-plot button.

        Args:
            main: The owning :class:`MainWindow`.
            nrows: Number of stacked axes the export figure should expect.
        """
        super().__init__()
        self.main = main
        self.figure = Figure(figsize=(8, 5), layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._nrows = nrows

        self._layout = QtWidgets.QVBoxLayout(self)
        self.controls = QtWidgets.QHBoxLayout()
        self._layout.addLayout(self.controls)
        self._layout.addWidget(self.toolbar)
        self._layout.addWidget(self.canvas, stretch=1)

        # A "Save plot…" button; subclasses add it to their controls row.
        self.save_btn = QtWidgets.QPushButton("Save plot…")
        self.save_btn.setToolTip(
            "Save the current plot as a high-resolution image (PNG / PDF / SVG) "
            "using the light export style."
        )
        self.save_btn.clicked.connect(self.save_figure)

        # Direct-manipulation navigation: scroll to zoom toward the cursor and
        # right-drag to pan, so the toolbar tools are rarely needed. Panes gate
        # these while a modal interaction (fit editing, area marking) owns the
        # mouse — see :meth:`_scroll_reserved` / :meth:`_pan_locked`.
        self._pan_state = None
        if self.interactive_nav:
            self.canvas.mpl_connect("scroll_event", self._nav_scroll)
            self.canvas.mpl_connect("button_press_event", self._nav_press)
            self.canvas.mpl_connect("motion_notify_event", self._nav_motion)
            self.canvas.mpl_connect("button_release_event", self._nav_release)

    @property
    def obj(self):
        """Active aerosol object (proxied from the main window)."""
        return self.main.obj

    def _split_with_side(self, side_widget, sizes=(820, 300)):
        """Lay the plot out left of ``side_widget`` with a draggable divider.

        Moves the controls row, toolbar and canvas into a left pane and places a
        horizontal :class:`QSplitter` between it and ``side_widget`` so the user
        can resize the side panel (like the datasets dock) instead of it being a
        fixed width. ``self._left_col`` is left pointing at the left pane's layout
        so subclasses can still insert into it.

        Args:
            side_widget: The right-hand side panel (already populated).
            sizes: Initial ``(left, side)`` widths in pixels.

        Returns:
            QtWidgets.QSplitter: The created splitter.
        """
        self._left_col = QtWidgets.QVBoxLayout()
        self._layout.removeItem(self.controls)
        self._layout.removeWidget(self.toolbar)
        self._layout.removeWidget(self.canvas)
        self._left_col.addLayout(self.controls)
        self._left_col.addWidget(self.toolbar)
        self._left_col.addWidget(self.canvas, stretch=1)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(self._left_col)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(side_widget)
        # The plot pane takes the extra space; the side panel keeps its width.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes(list(sizes))
        self._layout.addWidget(splitter, stretch=1)
        return splitter

    def _sync_toolbar_home(self) -> None:
        """Reset the nav toolbar's "home" view to the axes' current limits.

        The Matplotlib toolbar caches the view it should return to on
        "Reset original view" the first time the user zooms/pans, and
        otherwise keeps whatever was pushed earlier. If the data changes
        (e.g. cropping the dataset) and the axes autoscale to a new range,
        that cached home view is now stale — clicking "Reset original view"
        would jump back to the pre-crop limits instead of the fresh ones.
        Clearing the stack and pushing the just-drawn view fixes that.

        Autoscale is applied lazily (on draw), so ``push_current`` can otherwise
        capture the *pre-autoscale* limits and leave "home" too wide after data
        (a dataset or activity) is removed. Finalise the view first so the pushed
        home matches the current data.
        """
        for ax in self.figure.axes:
            if ax.get_autoscalex_on() or ax.get_autoscaley_on():
                ax.relim()
                ax.autoscale_view()
        self.toolbar.update()
        self.toolbar.push_current()

    def autoscale_to_data(
        self,
        ax=None,
        *,
        log_x: bool | None = None,
        log_y: bool | None = None,
        y_anchor_zero: bool = False,
        extra_axes=(),
        set_x: bool = True,
        set_y: bool = True,
    ) -> None:
        """Set ``ax``'s limits from the data currently drawn on it (the shared policy).

        Reads the drawn artists via :mod:`_autoscale` (lines, bars and ±σ bands)
        and applies tight limits with a small margin, so panes get consistent,
        never-stale limits by calling this after any display/dataset change. Pass
        ``extra_axes`` (e.g. a twin axis) to include their y-data in the y-range;
        ``set_x`` / ``set_y`` restrict which axis is updated. ``log_x`` / ``log_y``
        default to the axis' current scale (so a log axis gets a positive range).
        """
        from . import _autoscale

        ax = ax if ax is not None else getattr(self, "ax", None)
        if ax is None:
            return
        if log_x is None:
            log_x = ax.get_xscale() == "log"
        if log_y is None:
            log_y = ax.get_yscale() == "log"
        xlim, _ = _autoscale.limits([ax], log_x=log_x)
        _, ylim = _autoscale.limits(
            [ax, *extra_axes], log_y=log_y, y_anchor_zero=y_anchor_zero
        )
        if set_x and xlim is not None:
            ax.set_xlim(*xlim)
        if set_y and ylim is not None:
            ax.set_ylim(*ylim)

    def current_time_xlim(self):
        """Return the (xmin, xmax) of the time axis as Matplotlib date numbers.

        Returns ``None`` for tabs whose x-axis is not time (e.g. PSD), so the
        "crop to current view" action knows this tab can't define a time window.
        """
        return None

    def _show_message(self, msg: str) -> None:
        """Print a centered message (used for errors), leaving the tab redrawable.

        Tabs that keep a persistent ``self.ax`` (created once and reused every
        redraw) must not have it detached here: if the whole figure were cleared
        and a *new* axis added, ``self.ax`` would be orphaned and every later
        redraw would paint onto an axis no longer in the figure — the canvas
        would appear frozen on this message. So when a live ``self.ax`` exists,
        the message is drawn on it (``clear`` alone keeps it attached); the next
        redraw's ``ax.clear()`` then restores a normal plotting axis. Tabs that
        rebuild the figure each redraw (no persistent ``self.ax``) fall back to
        clearing the figure.
        """
        ax = getattr(self, "ax", None)
        if ax is not None and ax in self.figure.axes:
            ax.clear()
        else:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            wrap=True,
            fontsize=10,
            transform=ax.transAxes,
        )
        self.canvas.draw_idle()

    def _export_stem(self) -> str:
        """Suggested export file-name stem derived from the source file."""
        base = os.path.splitext(os.path.basename(self.main.source_path or "plot"))[0]
        return f"{base}_{self.export_tag}"

    def save_figure(self) -> None:
        """Save the plot for publication: high-DPI, light, with enlarged text.

        Saves the *live* on-screen figure — so any interactive edits (relabelled
        axes, recoloured curves) and the current zoom are kept. The save runs on
        a detached copy that is restyled to a light, print-friendly look (white
        background, dark frame/labels) **and** has its text and line widths
        enlarged to publication sizes, so the figure stays legible once it is
        placed (and usually shrunk) on a page. Data-artist colours are left as
        shown and the on-screen figure is never modified.
        """
        if self.obj is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save plot",
            f"{self._export_stem()}.png",
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)",
        )
        if not path:
            return
        try:
            rc = theme.export_rc()
            fig = _detached_copy(self.figure)
            _lighten_for_export(fig, rc)
            _enlarge_for_export(fig, rc)
            fig.savefig(path, dpi=theme.EXPORT_DPI, facecolor=fig.get_facecolor())
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "Save failed", traceback.format_exc(limit=2)
            )
            return
        QtWidgets.QMessageBox.information(self, "Saved", f"Plot saved to:\n{path}")

    def refresh(self) -> None:  # pragma: no cover - overridden
        """Redraw the tab from the current object (overridden by subclasses)."""
        raise NotImplementedError

    # -- shared direct-manipulation navigation ----------------------------
    # Scroll zooms toward the cursor; right-drag pans. Both defer to a pane's
    # modal interaction (fit editing, area marking) via the two predicates
    # below, and to an active toolbar pan/zoom tool, so limits stay locked while
    # a mode owns the mouse (and the wheel/clicks do that mode's job instead).

    def _scroll_reserved(self, event) -> bool:  # pragma: no cover - overridden
        """True when a pane mode claims the wheel (so cursor-zoom stands down)."""
        return False

    def _pan_locked(self) -> bool:  # pragma: no cover - overridden
        """True when a pane mode locks the view (so right-drag pan is disabled)."""
        return False

    @staticmethod
    def _rescale_range(lo, hi, center, scale, is_log):
        """Zoom a ``(lo, hi)`` range about ``center`` by ``scale`` (log-aware)."""
        if center is None:
            return None
        if is_log:
            if lo <= 0 or hi <= 0 or center <= 0:
                return None
            lo, hi, center = math.log10(lo), math.log10(hi), math.log10(center)
        new_lo = center - (center - lo) * scale
        new_hi = center + (hi - center) * scale
        if is_log:
            return 10**new_lo, 10**new_hi
        return new_lo, new_hi

    def _nav_scroll(self, event) -> None:
        """Zoom the axes under the cursor toward the cursor point on scroll."""
        ax = event.inaxes
        if ax is None or self.toolbar.mode:
            return
        if self._pan_locked() or self._scroll_reserved(event):
            return
        scale = 1.0 / 1.2 if event.step > 0 else 1.2  # scroll up = zoom in
        xr = self._rescale_range(
            *ax.get_xlim(), event.xdata, scale, ax.get_xscale() == "log"
        )
        yr = self._rescale_range(
            *ax.get_ylim(), event.ydata, scale, ax.get_yscale() == "log"
        )
        if xr is not None:
            ax.set_xlim(*xr)
        if yr is not None:
            ax.set_ylim(*yr)
        self.canvas.draw_idle()

    def _nav_press(self, event) -> None:
        """Begin a right-drag pan on the axes under the cursor."""
        if event.button != 3 or event.inaxes is None:
            return
        if self.toolbar.mode or self._pan_locked():
            return
        ax = event.inaxes
        self._pan_state = {
            "ax": ax,
            "x": event.x,
            "y": event.y,
            "xlim": ax.get_xlim(),
            "ylim": ax.get_ylim(),
        }

    def _nav_motion(self, event) -> None:
        """Pan the grabbed axes so the point under the cursor tracks the drag."""
        pan = self._pan_state
        if pan is None or event.x is None or event.y is None:
            return
        ax = pan["ax"]
        trans = ax.transData
        # Shift the axes' corner points by the cursor's pixel delta, then map
        # back to data coordinates — correct for linear and log axes alike.
        corners = np.array(
            [[pan["xlim"][0], pan["ylim"][0]], [pan["xlim"][1], pan["ylim"][1]]]
        )
        disp = trans.transform(corners)
        disp[:, 0] -= event.x - pan["x"]
        disp[:, 1] -= event.y - pan["y"]
        new = trans.inverted().transform(disp)
        ax.set_xlim(new[0, 0], new[1, 0])
        ax.set_ylim(new[0, 1], new[1, 1])
        self.canvas.draw_idle()

    def _nav_release(self, event) -> None:
        """End a right-drag pan."""
        if event.button == 3:
            self._pan_state = None


def _active_color_cycle() -> list:
    """Return the colour cycle of the *currently active* rcParams profile.

    Reading the live ``axes.prop_cycle`` keeps colours theme-correct on the
    embedded canvas (dark or light). Exports save the live figure as-is (see
    :meth:`_PlotTab.save_figure`), so these on-screen colours carry straight
    through to the saved image.
    """
    cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color")
    return list(cycle) if cycle else theme.mpl_cycle()


def _detached_copy(fig):
    """Return an independent copy of ``fig`` (via pickle) with an Agg canvas.

    Restyling a copy means the export pipeline can never alter the live,
    on-screen figure — a failure mid-restyle leaves the canvas untouched.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    buf = io.BytesIO()
    pickle.dump(fig, buf)
    buf.seek(0)
    copy = pickle.load(buf)
    FigureCanvasAgg(copy)  # attach a canvas so savefig works
    return copy


def _is_light(color) -> bool:
    """True for near-white / pale colours (which vanish on a white export)."""
    try:
        r, g, b = mcolors.to_rgb(color)
    except (ValueError, TypeError):
        return False
    return (0.299 * r + 0.587 * g + 0.114 * b) > 0.7


def _darken_texts(texts, dark: str) -> None:
    """Darken pale text (and pale text boxes) so they read on a white export."""
    for t in texts:
        if _is_light(t.get_color()):
            t.set_color(dark)
        box = t.get_bbox_patch()
        if box is not None and not _is_light(box.get_facecolor()):
            box.set_facecolor("white")


def _lighten_for_export(fig, rc: dict) -> None:
    """Recolour a figure's *structural* elements to the light export palette.

    Only the frame, ticks, labels, grid, legend and pale annotation text are
    touched — data artists (lines, bars, scatter, meshes) keep their on-screen
    colours, so interactive recolouring and the displayed look survive the
    export. ``rc`` is the :func:`theme.export_rc` profile (passed in so the
    caller can share it with :func:`_enlarge_for_export`).
    """
    txt = rc["text.color"]
    fig.set_facecolor(rc["figure.facecolor"])
    fig.set_edgecolor(rc["figure.facecolor"])
    for ax in fig.axes:
        ax.set_facecolor(rc["axes.facecolor"])
        for spine in ax.spines.values():
            spine.set_edgecolor(rc["axes.edgecolor"])
        ax.tick_params(axis="both", which="both", colors=rc["xtick.color"])
        ax.xaxis.label.set_color(txt)
        ax.yaxis.label.set_color(txt)
        if ax.get_title():
            ax.title.set_color(txt)
        for gl in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            gl.set_color(rc["grid.color"])
        _darken_texts(ax.texts, txt)
        leg = ax.get_legend()
        if leg is not None:
            frame = leg.get_frame()
            frame.set_facecolor(rc["legend.facecolor"])
            frame.set_edgecolor(rc["legend.edgecolor"])
            for t in leg.get_texts():
                t.set_color(rc["legend.labelcolor"])
    _darken_texts(fig.texts, txt)


def _enlarge_for_export(fig, rc: dict) -> None:
    """Resize a figure's text and lines to the export profile's print sizes.

    The embedded figure carries the compact *on-screen* sizes (small axis
    labels, ticks, legends and thin lines); saved as-is they become unreadable
    once the figure is shrunk onto a page. This sets the structural text (axis
    labels, ticks, title, offset/exponent text, legend and annotations) to the
    larger :func:`theme.export_rc` sizes and thickens every data line to at
    least the profile's line width, so a saved figure reads well at print scale.
    Colours are handled separately by :func:`_lighten_for_export`.

    Notes:
        Tick-label sizes are set via :meth:`Axes.tick_params` rather than by
        resizing the existing tick-label artists: a ``savefig`` triggers a
        redraw in which the formatter regenerates the tick labels, which would
        otherwise reset any directly-set size back to the rcParams default.
    """
    label = rc["axes.labelsize"]
    title = rc["axes.titlesize"]
    tick = rc["xtick.labelsize"]
    legend = rc["legend.fontsize"]
    min_lw = rc["lines.linewidth"]
    min_ms = min_lw * theme.EXPORT_MARKER_SCALE

    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(label)
        ax.yaxis.label.set_fontsize(label)
        if ax.get_title():
            ax.title.set_fontsize(title)
        ax.tick_params(axis="both", which="major", labelsize=tick)
        ax.tick_params(axis="both", which="minor", labelsize=tick)
        # The scientific-notation exponent ("1e3") sits apart from the ticks.
        ax.xaxis.get_offset_text().set_fontsize(tick)
        ax.yaxis.get_offset_text().set_fontsize(tick)
        # Free-standing annotations (placeholders, the PSD "Fitting:" note): one
        # notch below the axis labels so they stay readable but not dominant.
        for t in ax.texts:
            t.set_fontsize(tick)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(legend)
            leg_title = leg.get_title()
            if leg_title is not None and leg_title.get_text():
                leg_title.set_fontsize(legend)
        for line in ax.get_lines():
            if line.get_linewidth() < min_lw:
                line.set_linewidth(min_lw)
            marker = line.get_marker()
            if marker not in (None, "None", "", " ") and line.get_markersize() < min_ms:
                line.set_markersize(min_ms)

    for t in fig.texts:
        t.set_fontsize(label)
