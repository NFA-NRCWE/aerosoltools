"""Shared base class and helpers for the GUI's data tabs.

:class:`_PlotTab` provides the embedded Matplotlib figure + navigation toolbar
and the "Save plot…" export pipeline; :func:`_export_table` / :func:`_tune_table`
are table helpers; :func:`_active_color_cycle` reads the live colour cycle so
tab colours stay theme-correct on screen and in exports.
"""

from __future__ import annotations

import os
import traceback

import matplotlib as mpl
import pandas as pd

from .. import theme
from ..qt import Figure, FigureCanvas, NavigationToolbar, QtCore, QtWidgets


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

    #: Figure size (inches) used when exporting; subclasses may override.
    export_figsize = theme.EXPORT_FIGSIZE
    #: Short tag used in the suggested export file name.
    export_tag = "plot"

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

    def current_time_xlim(self):
        """Return the (xmin, xmax) of the time axis as Matplotlib date numbers.

        Returns ``None`` for tabs whose x-axis is not time (e.g. PSD), so the
        "crop to current view" action knows this tab can't define a time window.
        """
        return None

    def _show_message(self, msg: str) -> None:
        """Clear the figure and print a centered message (used for errors)."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True, fontsize=10)
        self.canvas.draw_idle()

    def _export_stem(self) -> str:
        """Suggested export file-name stem derived from the source file."""
        base = os.path.splitext(os.path.basename(self.main.source_path or "plot"))[0]
        return f"{base}_{self.export_tag}"

    def save_figure(self) -> None:
        """Render the current view into a fresh, high-quality figure and save it.

        The export figure is built under a fixed rcParams profile (see
        :func:`theme.export_rc`) with a deliberate size + font sizes, so the
        output is well proportioned regardless of the on-screen layout.
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
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        try:
            with mpl.rc_context(theme.export_rc()):
                fig = Figure(figsize=self.export_figsize, layout="constrained")
                FigureCanvasAgg(fig)  # attach a canvas so savefig works
                self._render_export(fig)
                fig.savefig(path, dpi=theme.EXPORT_DPI)
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "Save failed", traceback.format_exc(limit=2)
            )
            return
        QtWidgets.QMessageBox.information(self, "Saved", f"Plot saved to:\n{path}")

    def refresh(self) -> None:  # pragma: no cover - overridden
        """Redraw the tab from the current object (overridden by subclasses)."""
        raise NotImplementedError

    def _render_export(self, fig) -> None:  # pragma: no cover - overridden
        """Draw the current view onto ``fig`` for export. Overridden per tab."""
        raise NotImplementedError


def _active_color_cycle() -> list:
    """Return the colour cycle of the *currently active* rcParams profile.

    Reading the live ``axes.prop_cycle`` keeps colours theme-correct on both
    paths automatically: the dark/light screen cycle on the embedded canvas, and
    the light export cycle while ``_render_export`` runs under
    :func:`theme.export_rc`. This is why this tab does not need the screen-only
    brighten hack that :class:`PSDTab` uses.
    """
    cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color")
    return list(cycle) if cycle else theme.mpl_cycle()
