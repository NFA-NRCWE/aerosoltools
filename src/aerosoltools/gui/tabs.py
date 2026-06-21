"""Tab widgets for the aerosoltools GUI.

Each tab is a self-contained QWidget that reads the *current* aerosol object
from the parent :class:`MainWindow` (passed in as ``main``) and renders itself
on :meth:`refresh`. Plot tabs embed a Matplotlib canvas plus the standard
navigation toolbar (zoom/pan), and reuse the library's own plotting methods
wherever possible so the GUI stays a thin presentation layer.
"""

from __future__ import annotations

import contextlib
import io
import os
import traceback

import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.widgets import SpanSelector

from ..utility import Plot_correlation, bland_altman_analysis
from . import helpers, theme
from .models import PandasTableModel
from .qt import Figure, FigureCanvas, NavigationToolbar, QtCore, QtWidgets


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
        self.save_btn.clicked.connect(self.save_figure)

    @property
    def obj(self):
        return self.main.obj

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
        raise NotImplementedError

    def _render_export(self, fig) -> None:  # pragma: no cover - overridden
        """Draw the current view onto ``fig`` for export. Overridden per tab."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Raw data tab
# ---------------------------------------------------------------------------
class RawDataTab(QtWidgets.QWidget):
    """Tabular view of the object's main data (and optionally extra data)."""

    def __init__(self, main):
        super().__init__()
        self.main = main

        layout = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["Main data", "Extra data"])
        self.source.currentIndexChanged.connect(self.refresh)
        bar.addWidget(QtWidgets.QLabel("Show:"))
        bar.addWidget(self.source)
        self.info = QtWidgets.QLabel("")
        bar.addWidget(self.info)
        bar.addStretch(1)
        self.export_btn = QtWidgets.QPushButton("Export to Excel…")
        self.export_btn.clicked.connect(self._export)
        bar.addWidget(self.export_btn)
        layout.addLayout(bar)

        self.model = PandasTableModel()
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.model)
        self.view.setAlternatingRowColors(True)
        _tune_table(self.view)
        layout.addWidget(self.view)

    @property
    def obj(self):
        return self.main.obj

    def refresh(self) -> None:
        if self.obj is None:
            return
        if self.source.currentText() == "Extra data":
            df = self.obj.extra_data
        else:
            df = self.obj.data
        self.model.set_dataframe(df)
        if df.shape[1] == 0:
            self.info.setText("   (no extra data in this file)")
        else:
            self.info.setText(f"   {df.shape[0]} rows × {df.shape[1]} columns")

    def _export(self) -> None:
        if self.obj is None:
            return
        which = "extra" if self.source.currentText() == "Extra data" else "data"
        df = self.obj.extra_data if which == "extra" else self.obj.data
        base = os.path.splitext(os.path.basename(self.main.source_path or "data"))[0]
        # Keep the timestamp index for raw data exports.
        _export_table(self, df, f"{base}_{which}", with_index=True)


# ---------------------------------------------------------------------------
# Summary tab
# ---------------------------------------------------------------------------
class SummaryTab(QtWidgets.QWidget):
    """Per-activity tabular summaries (activity stats and exposure metrics)."""

    def __init__(self, main):
        super().__init__()
        self.main = main

        layout = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Activity summary", "Exposure summary"])
        self.kind.currentIndexChanged.connect(self._on_kind_change)
        bar.addWidget(QtWidgets.QLabel("Type:"))
        bar.addWidget(self.kind)

        # Metric selector: a kind drop-down, plus a cut-off drop-down that is
        # only shown for size-selective fractions (PM/PN/PS/PV). This replaces
        # the old free-text field, which let users type invalid metric strings.
        self.metric_label = QtWidgets.QLabel("Metric:")
        bar.addWidget(self.metric_label)
        self.metric_kind = QtWidgets.QComboBox()
        self.metric_kind.currentTextChanged.connect(self._on_metric_kind_change)
        bar.addWidget(self.metric_kind)
        self.metric_cut = QtWidgets.QComboBox()
        self.metric_cut.setEditable(True)  # allow custom cut-offs too
        self.metric_cut.setFixedWidth(80)
        bar.addWidget(self.metric_cut)
        self.metric_cut_label = QtWidgets.QLabel("µm")
        bar.addWidget(self.metric_cut_label)
        self._synced_obj = None

        self.compute = QtWidgets.QPushButton("Compute")
        self.compute.setObjectName("primary")
        self.compute.clicked.connect(self.refresh)
        bar.addWidget(self.compute)
        bar.addStretch(1)
        self.export_btn = QtWidgets.QPushButton("Export to Excel…")
        self.export_btn.clicked.connect(self._export)
        bar.addWidget(self.export_btn)
        layout.addLayout(bar)

        # Second row: exposure-limit parameters (only shown for exposure).
        self.exp_bar = QtWidgets.QHBoxLayout()
        self.short_limit = self._add_field("STEL (short-term limit):", "1.0", width=80)
        self.short_window = self._add_field("over", "15min", width=70)
        self.long_limit = self._add_field("OEL (8h limit):", "1.0", width=80)
        self.twa_window = self._add_field("TWA window", "8h", width=70)
        self.exp_bar.addStretch(1)
        layout.addLayout(self.exp_bar)

        self.model = PandasTableModel()
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.model)
        self.view.setAlternatingRowColors(True)
        _tune_table(self.view)
        layout.addWidget(self.view)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._on_kind_change()

    def _add_field(self, label: str, default: str, width: int) -> QtWidgets.QLineEdit:
        """Add a labelled line-edit to the exposure-parameter row."""
        lbl = QtWidgets.QLabel(label)
        edit = QtWidgets.QLineEdit(default)
        edit.setFixedWidth(width)
        self.exp_bar.addWidget(lbl)
        self.exp_bar.addWidget(edit)
        # Stash the label so visibility can be toggled together with the field.
        edit._label = lbl  # type: ignore[attr-defined]
        return edit

    @property
    def obj(self):
        return self.main.obj

    _FRACTION_KINDS = ("PM", "PN", "PS", "PV")

    def _limit_fields(self):
        return (self.short_limit, self.short_window, self.long_limit, self.twa_window)

    def _exposure_widgets(self):
        """All widgets (and their labels) shown only in Exposure mode."""
        widgets = [
            self.metric_label,
            self.metric_kind,
            self.metric_cut,
            self.metric_cut_label,
        ]
        for field in self._limit_fields():
            widgets.append(field)
            label = getattr(field, "_label", None)
            if label is not None:
                widgets.append(label)
        return widgets

    def _ensure_metric_options(self) -> None:
        """Populate the metric drop-downs for the loaded object (once per file)."""
        if self.obj is None or self._synced_obj is self.obj:
            return
        self._synced_obj = self.obj
        self.metric_kind.blockSignals(True)
        self.metric_cut.blockSignals(True)
        self.metric_kind.clear()
        self.metric_cut.clear()
        if helpers.is_2d(self.obj):
            self.metric_kind.addItems(["PNC", "MASS", "PM", "PN", "PS", "PV"])
            self.metric_cut.addItems(
                ["0.1", "0.25", "0.5", "1", "2.5", "4", "4.2", "10"]
            )
            self.metric_kind.setCurrentText("PM")
            self.metric_cut.setCurrentText("4.2")
        else:
            # 1D / AerosolAlt: offer PNC plus the available numeric channels.
            names = ["PNC"]
            for _label, kind, name in helpers.plottable_columns(self.obj):
                if kind in ("data", "extra") and name not in names:
                    names.append(name)
            self.metric_kind.addItems(names)
        self.metric_kind.blockSignals(False)
        self.metric_cut.blockSignals(False)
        self._on_metric_kind_change()

    def _on_metric_kind_change(self, *_args) -> None:
        exposure = self.kind.currentText() == "Exposure summary"
        is_fraction = self.metric_kind.currentText().strip() in self._FRACTION_KINDS
        show_cut = exposure and is_fraction
        self.metric_cut.setVisible(show_cut)
        self.metric_cut_label.setVisible(show_cut)

    def _build_metric(self) -> str:
        kind = self.metric_kind.currentText().strip()
        if kind in self._FRACTION_KINDS:
            return f"{kind}{self.metric_cut.currentText().strip()}"
        return kind

    def _on_kind_change(self) -> None:
        exposure = self.kind.currentText() == "Exposure summary"
        if exposure:
            self._ensure_metric_options()
        for widget in self._exposure_widgets():
            widget.setVisible(exposure)
        if exposure:
            self._on_metric_kind_change()  # refine cut-off visibility

    @staticmethod
    def _to_float(text: str, default: float) -> float:
        try:
            return float(text.strip())
        except (ValueError, AttributeError):
            return default

    def refresh(self) -> None:
        if self.obj is None:
            return
        # Keep the metric drop-downs in sync with the loaded object.
        if self._synced_obj is not self.obj:
            self._synced_obj = None
            self._ensure_metric_options()
        self.status.setText("")
        try:
            # The core summarize_* methods print their table to stdout; swallow
            # it so the GUI is the single source of truth and a non-UTF-8 console
            # cannot crash the method on the µg/m³ / cm⁻³ glyphs mid-print.
            with contextlib.redirect_stdout(io.StringIO()):
                if self.kind.currentText() == "Exposure summary":
                    df = self.obj.summarize_exposure(
                        metric=self._build_metric(),
                        short_limit=self._to_float(self.short_limit.text(), 1.0),
                        long_limit=self._to_float(self.long_limit.text(), 1.0),
                        short_window=self.short_window.text().strip() or "15min",
                        twa_window=self.twa_window.text().strip() or "8h",
                    )
                else:
                    df = self.obj.summarize_activities()
            if df is None or df.empty:
                self.model.set_dataframe(pd.DataFrame())
                self.status.setText("No data to summarize for the current activities.")
            else:
                self.model.set_dataframe(df)
        except Exception as exc:  # surface errors instead of crashing the GUI
            self.model.set_dataframe(pd.DataFrame())
            self.status.setText(f"Could not compute summary: {exc}")

    def _export(self) -> None:
        df = self.model.dataframe
        kind = (
            "exposure" if self.kind.currentText() == "Exposure summary" else "activity"
        )
        base = os.path.splitext(os.path.basename(self.main.source_path or "summary"))[0]
        _export_table(self, df, f"{base}_{kind}_summary", with_index=False)


# ---------------------------------------------------------------------------
# Activity period editor
# ---------------------------------------------------------------------------
class ActivityEditorDialog(QtWidgets.QDialog):
    """Add, remove, or adjust the time periods of a single activity.

    Lets the user fine-tune a task's occurrences (each a start/end pair)
    without deleting and re-marking the whole activity.
    """

    def __init__(self, parent, name, periods, default_start, default_end):
        super().__init__(parent)
        self.setWindowTitle(f"Edit periods — {name}")
        self.resize(480, 320)
        self._default = (default_start, default_end)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                f"Periods for '{name}'. Adjust the start/end times, or add/remove rows."
            )
        )

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Start", "End"])
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        layout.addWidget(self.table, stretch=1)

        for start, end in periods:
            self._add_row(start, end)
        if not periods:
            self._add_row()

        row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add period")
        add_btn.clicked.connect(lambda: self._add_row())
        rem_btn = QtWidgets.QPushButton("Remove selected")
        rem_btn.clicked.connect(self._remove_row)
        row.addWidget(add_btn)
        row.addWidget(rem_btn)
        row.addStretch(1)
        layout.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_editor(self, value) -> QtWidgets.QDateTimeEdit:
        editor = QtWidgets.QDateTimeEdit()
        editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        editor.setCalendarPopup(True)
        editor.setDateTime(QtCore.QDateTime(pd.Timestamp(value).to_pydatetime()))
        return editor

    def _add_row(self, start=None, end=None) -> None:
        start = self._default[0] if start is None else start
        end = self._default[1] if end is None else end
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setCellWidget(r, 0, self._make_editor(start))
        self.table.setCellWidget(r, 1, self._make_editor(end))

    def _remove_row(self) -> None:
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def periods(self) -> list:
        """Return the edited list of ``(start, end)`` pairs (end > start only)."""
        out = []
        for r in range(self.table.rowCount()):
            start = pd.Timestamp(self.table.cellWidget(r, 0).dateTime().toPyDateTime())
            end = pd.Timestamp(self.table.cellWidget(r, 1).dateTime().toPyDateTime())
            if end > start:
                out.append((start, end))
        return out


# ---------------------------------------------------------------------------
# Time series tab (with interactive activity marking)
# ---------------------------------------------------------------------------
class TimeSeriesTab(_PlotTab):
    """Plot a selectable column over time and mark activities by dragging."""

    export_tag = "timeseries"

    def __init__(self, main):
        super().__init__(main, nrows=1)

        self.column = QtWidgets.QComboBox()
        # Changing the series should rescale; toggles/caps preserve the view.
        self.column.currentIndexChanged.connect(lambda: self.refresh(reset_view=True))
        self.controls.addWidget(QtWidgets.QLabel("Series:"))
        self.controls.addWidget(self.column)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.stateChanged.connect(lambda: self.refresh(reset_view=False))
        self.controls.addWidget(self.log_y)

        self.controls.addWidget(QtWidgets.QLabel("Y min:"))
        self.ymin = QtWidgets.QLineEdit()
        self.ymin.setPlaceholderText("auto")
        self.ymin.setFixedWidth(70)
        self.ymin.editingFinished.connect(lambda: self.refresh(reset_view=False))
        self.controls.addWidget(self.ymin)

        self.controls.addWidget(QtWidgets.QLabel("Y max:"))
        self.ymax = QtWidgets.QLineEdit()
        self.ymax.setPlaceholderText("auto")
        self.ymax.setFixedWidth(70)
        self.ymax.editingFinished.connect(lambda: self.refresh(reset_view=False))
        self.controls.addWidget(self.ymax)

        self.show_acts = QtWidgets.QCheckBox("Show activities")
        self.show_acts.setChecked(True)
        self.show_acts.stateChanged.connect(lambda: self.refresh(reset_view=False))
        self.controls.addWidget(self.show_acts)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # View bookkeeping: preserve zoom/pan across non-rescaling refreshes.
        self._key = None
        self._has_drawn = False
        # Identity of the object whose columns currently populate the selector,
        # so the series list re-syncs when the active dataset changes.
        self._cols_obj = None

        # Mark-task toggle: a button that stays visually pressed while marking.
        self.mark_mode = QtWidgets.QPushButton("Mark activities")
        self.mark_mode.setObjectName("toggle")
        self.mark_mode.setCheckable(True)
        self.mark_mode.setToolTip(
            "Toggle marking on, then drag across the plot to add a task period."
        )
        self.mark_mode.toggled.connect(self._toggle_mark_mode)

        # Activities side panel (mark toggle + list + edit + delete).
        self.act_list = QtWidgets.QListWidget()
        self.act_list.setMaximumWidth(260)
        self.act_list.itemDoubleClicked.connect(lambda _item: self._edit_selected())
        self.edit_btn = QtWidgets.QPushButton("Edit selected activity")
        self.edit_btn.clicked.connect(self._edit_selected)
        self.del_btn = QtWidgets.QPushButton("Delete selected activity")
        self.del_btn.clicked.connect(self._delete_selected)
        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Activities:"))
        side.addWidget(self.act_list, stretch=1)
        side.addWidget(self.mark_mode)
        side.addWidget(self.edit_btn)
        side.addWidget(self.del_btn)
        hint = QtWidgets.QLabel(
            "Tip: click 'Mark activities', then drag across the plot to add a "
            "task period (pick an existing task name to add another occurrence). "
            "Double-click a task to edit its periods. Click 'Marking' again to "
            "stop and zoom/pan."
        )
        hint.setWordWrap(True)
        side.addWidget(hint)

        # Two columns: a left column holding the (compact) data-adjustments box
        # — attached later — plus the view controls, toolbar and plot; and a
        # full-height activities panel on the right.
        self._left_col = QtWidgets.QVBoxLayout()
        self._layout.removeItem(self.controls)
        self._layout.removeWidget(self.toolbar)
        self._layout.removeWidget(self.canvas)
        self._left_col.addLayout(self.controls)
        self._left_col.addWidget(self.toolbar)
        self._left_col.addWidget(self.canvas, stretch=1)

        body = QtWidgets.QHBoxLayout()
        body.addLayout(self._left_col, stretch=1)
        body.addLayout(side)
        self._layout.addLayout(body, stretch=1)

        self.ax = self.figure.add_subplot(111)
        self._span = SpanSelector(
            self.ax,
            self._on_span,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="tab:red"),
            button=1,
        )
        self._span.set_active(False)

    def attach_adjust_controls(self, adjust_box) -> None:
        """Embed the shared "Data adjustments" box atop the left column.

        The box is built and owned by :class:`MainWindow` (so its handlers can
        operate on the loaded object), but lives inside this tab so that data
        adjustments happen where the data is shown.
        """
        self._left_col.insertWidget(0, adjust_box)

    # -- behaviour ---------------------------------------------------------
    def _toggle_mark_mode(self) -> None:
        active = self.mark_mode.isChecked()
        # Reflect state in the button label ("Mark activities" -> "Marking").
        self.mark_mode.setText("Marking" if active else "Mark activities")
        self._span.set_active(active)
        # Deactivate any active toolbar pan/zoom so it doesn't grab the drag.
        if active and self.toolbar.mode:
            mode = str(self.toolbar.mode)
            if "pan" in mode:
                self.toolbar.pan()  # toggles pan off
            elif "zoom" in mode:
                self.toolbar.zoom()  # toggles zoom off

    def _on_span(self, xmin: float, xmax: float) -> None:
        if not self.mark_mode.isChecked() or self.obj is None:
            return
        if xmax <= xmin:
            return
        # SpanSelector gives Matplotlib date floats (tz-aware UTC); strip tz so
        # the comparison against the naive time index in mark_activities works.
        start = pd.Timestamp(mdates.num2date(xmin)).tz_localize(None)
        end = pd.Timestamp(mdates.num2date(xmax)).tz_localize(None)

        # Offer existing tasks (pick one to add another occurrence) plus a new
        # default name. The combo is editable so a brand-new name can be typed.
        existing = self.main.project.user_activities()
        default = f"Task {len(existing) + 1}"
        items = existing + [default]
        name, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Mark task",
            "Pick an existing task to add another occurrence, or type a new name:",
            items,
            len(items) - 1,  # preselect the new default
            True,  # editable
        )
        if not ok or not name.strip():
            return
        # Tasks are shared across all datasets in the project.
        self.main.project.add_activity(name.strip(), start, end)
        # Keep the current zoom/pan so the view does not snap back after marking.
        self.main.refresh_all(reset_view=False)

    def _delete_selected(self) -> None:
        item = self.act_list.currentItem()
        if item is None or self.obj is None:
            return
        # Deleting a task removes it from every dataset in the project.
        self.main.project.delete_activity(item.text())
        self.main.refresh_all(reset_view=False)

    def _edit_selected(self) -> None:
        item = self.act_list.currentItem()
        if item is None or self.obj is None:
            return
        name = item.text()
        periods = list(self.obj._activity_periods.get(name, []))
        tmin = pd.Timestamp(self.obj.time.min())
        tmax = pd.Timestamp(self.obj.time.max())
        dlg = ActivityEditorDialog(self, name, periods, tmin, tmax)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        # Replace the task's periods across every dataset in the project.
        self.main.project.set_activity_periods(name, dlg.periods())
        self.main.refresh_all(reset_view=False)

    # -- rendering ---------------------------------------------------------
    def _sync_columns(self) -> None:
        self.column.blockSignals(True)
        current = self.column.currentData()
        self.column.clear()
        for label, kind, name in helpers.plottable_columns(self.obj):
            self.column.addItem(label, userData=(kind, name))
        # Restore previous selection if still present.
        if current is not None:
            idx = self.column.findData(current)
            if idx >= 0:
                self.column.setCurrentIndex(idx)
        self.column.blockSignals(False)

    def _sync_activities(self) -> None:
        self.act_list.clear()
        self.act_list.addItems(helpers.user_activities(self.obj))

    def _cap(self, edit) -> float | None:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _plot_on(self, ax) -> None:
        """Draw the currently selected series onto ``ax`` (no view/cap logic)."""
        kind, name = self.column.currentData() or ("total", helpers.TOTAL)
        series = helpers.series_for(self.obj, kind, name)

        ax.clear()
        ax.plot(series.index, series.to_numpy(), lw=1.5)

        col_for_units = None if kind == "total" else name
        dtype, unit = helpers.describe(self.obj, col_for_units)
        base_dtype = dtype.split("/")[0] if "/" in dtype else dtype
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{base_dtype}, {unit}".strip(", "))
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        if self.log_y.isChecked():
            ax.set_yscale("log")
        if self.show_acts.isChecked():
            helpers.shade_activities(ax, self.obj)

    def refresh(self, reset_view: bool | None = None) -> None:
        if self.obj is None:
            return
        if reset_view is None:
            reset_view = getattr(self.main, "_reset_view", True)
        # Re-sync the series list when first shown or when the active dataset
        # (and hence the available columns) has changed.
        if self.column.count() == 0 or self._cols_obj is not self.obj:
            self._sync_columns()
            self._cols_obj = self.obj
        self._sync_activities()

        kind, name = self.column.currentData() or ("total", helpers.TOTAL)
        # Decide whether to keep the existing view: same series, already drawn,
        # and the caller did not request a rescale.
        key = (kind, name)
        preserve = (not reset_view) and self._has_drawn and key == self._key
        prev_xlim = self.ax.get_xlim() if preserve else None
        prev_ylim = self.ax.get_ylim() if preserve else None

        try:
            self._plot_on(self.ax)
        except Exception as exc:
            self._show_message(f"Could not read series: {exc}")
            return

        # Restore the previous view, then apply any explicit Y caps on top.
        if preserve and prev_xlim is not None:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)
        ymin, ymax = self._cap(self.ymin), self._cap(self.ymax)
        if ymin is not None or ymax is not None:
            self.ax.set_ylim(bottom=ymin, top=ymax)

        self._key = key
        self._has_drawn = True
        self.canvas.draw_idle()

    def _render_export(self, fig) -> None:
        ax = fig.add_subplot(111)
        self._plot_on(ax)
        # Match the on-screen view (zoom/crop and any Y caps already applied).
        ax.set_xlim(self.ax.get_xlim())
        ax.set_ylim(self.ax.get_ylim())

    def current_time_xlim(self):
        if self._has_drawn and self.ax.has_data():
            return self.ax.get_xlim()
        return None


# ---------------------------------------------------------------------------
# PSD tab (Aerosol2D only)
# ---------------------------------------------------------------------------
class PSDTab(_PlotTab):
    """Mean particle size distribution per activity (uses ``plot_psd``)."""

    export_tag = "psd"

    def __init__(self, main):
        super().__init__(main, nrows=1)

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.setChecked(True)
        self.normalize.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.normalize)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.log_y)

        self.controls.addWidget(QtWidgets.QLabel("Activities:"))
        self.act_list = QtWidgets.QListWidget()
        self.act_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.act_list.setMaximumHeight(70)
        self.act_list.itemSelectionChanged.connect(self.refresh)
        self.controls.addWidget(self.act_list)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        self.ax = self.figure.add_subplot(111)

    def _sync_activities(self) -> None:
        self.act_list.blockSignals(True)
        selected = {i.text() for i in self.act_list.selectedItems()}
        self.act_list.clear()
        for name in self.obj.activities:
            self.act_list.addItem(name)
            if name in selected:
                self.act_list.item(self.act_list.count() - 1).setSelected(True)
        self.act_list.blockSignals(False)

    def _plot_on(self, fig) -> None:
        """Draw the PSD onto a fresh axis on ``fig``."""
        selected = [i.text() for i in self.act_list.selectedItems()] or None
        fig.clear()
        ax = fig.add_subplot(111)
        self.obj.plot_psd(
            activities=selected,
            normalize=self.normalize.isChecked(),
            ax=ax,
        )
        if self.log_y.isChecked():
            ax.set_yscale("log")

    def refresh(self) -> None:
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        self._sync_activities()
        try:
            self._plot_on(self.figure)
            self.ax = self.figure.axes[0] if self.figure.axes else None
            # The core plot_psd picks colours tuned for a white background; on
            # the dark theme they are too dark, so brighten them for the screen
            # only (exports keep the core's report colours via _render_export).
            if self.ax is not None and theme.is_dark():
                self._brighten_for_dark(self.ax)
            self.canvas.draw_idle()
        except Exception:
            self._show_message("Could not draw PSD:\n" + traceback.format_exc(limit=1))

    @staticmethod
    def _brighten_for_dark(ax) -> None:
        """Recolour PSD lines (and their ±1σ fills) with the bright cycle."""
        cycle = theme.mpl_cycle()
        for i, line in enumerate(ax.get_lines()):
            line.set_color(cycle[i % len(cycle)])
        for i, coll in enumerate(ax.collections):  # fill_between envelopes
            coll.set_color(cycle[i % len(cycle)])
            coll.set_alpha(0.20)
        if ax.get_legend() is not None:
            ax.legend()  # rebuild so legend swatches match the new colours

    def _render_export(self, fig) -> None:
        self._plot_on(fig)


# ---------------------------------------------------------------------------
# Heatmap tab (Aerosol2D only)
# ---------------------------------------------------------------------------
class HeatmapTab(_PlotTab):
    """Total concentration + time/size heatmap (uses ``plot_timeseries``)."""

    export_figsize = theme.EXPORT_FIGSIZE_TALL
    export_tag = "heatmap"

    def __init__(self, main):
        super().__init__(main, nrows=2)

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.normalize)

        self.log = QtWidgets.QCheckBox("Log color scale")
        self.log.setChecked(True)
        self.log.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.log)

        self.controls.addWidget(QtWidgets.QLabel("Color min:"))
        self.cmin = QtWidgets.QLineEdit()
        self.cmin.setPlaceholderText("auto")
        self.cmin.setFixedWidth(70)
        self.cmin.editingFinished.connect(self.refresh)
        self.controls.addWidget(self.cmin)

        self.controls.addWidget(QtWidgets.QLabel("Color max:"))
        self.cmax = QtWidgets.QLineEdit()
        self.cmax.setPlaceholderText("auto")
        self.cmax.setFixedWidth(70)
        self.cmax.editingFinished.connect(self.refresh)
        self.controls.addWidget(self.cmax)

        self.show_acts = QtWidgets.QCheckBox("Show activities")
        self.show_acts.setChecked(True)
        self.show_acts.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.show_acts)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

    def _cap(self, edit) -> float:
        """Return the field value, or 0.0 (which plot_timeseries reads as auto)."""
        text = edit.text().strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0

    def _plot_on(self, fig) -> None:
        """Draw the total-conc + heatmap panels onto ``fig``."""
        # plot_timeseries adds its own colorbar axes, so build panels fresh.
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        # y_3d caps the colour scale: (min, max), where 0 means "automatic".
        # Note: for a log colour scale the lower cap must be > 0.
        y_3d = (self._cap(self.cmin), self._cap(self.cmax))

        # Normalize on a working copy so the loaded object is left untouched.
        target = self.obj
        if self.normalize.isChecked():
            target = self.obj.copy_self()
            target.normalize_logdp()

        target.plot_timeseries(
            log=self.log.isChecked(),
            y_3d=y_3d,
            ax1=ax1,
            ax2=ax2,
            mark_activities=self.show_acts.isChecked(),
        )

    def refresh(self) -> None:
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        try:
            self._plot_on(self.figure)
            self.canvas.draw_idle()
        except Exception:
            self._show_message(
                "Could not draw heatmap:\n" + traceback.format_exc(limit=1)
            )

    def _render_export(self, fig) -> None:
        self._plot_on(fig)

    def current_time_xlim(self):
        # The top (total-conc) panel shares the time x-axis with the heatmap.
        axes = self.figure.axes
        if axes and axes[0].has_data():
            return axes[0].get_xlim()
        return None


# ---------------------------------------------------------------------------
# PM bands tab (Aerosol2D only)
# ---------------------------------------------------------------------------
class PMBandsTab(_PlotTab):
    """Stacked size-selective Pₓ bands over time.

    Reuses the library's ``dtype_converter`` + ``PM_calc`` on a working copy for
    the numerics, then renders the stacked bands on the embedded axis.
    """

    export_tag = "pm_bands"

    def __init__(self, main):
        super().__init__(main, nrows=1)

        self.dtype = QtWidgets.QComboBox()
        self.dtype.addItems(["dM", "dN", "dS", "dV"])
        self.dtype.currentIndexChanged.connect(self.refresh)
        self.controls.addWidget(QtWidgets.QLabel("Basis:"))
        self.controls.addWidget(self.dtype)

        self.values = QtWidgets.QLineEdit("0.5, 2.5, 10")
        self.values.setFixedWidth(140)
        self.values.editingFinished.connect(self.refresh)
        self.controls.addWidget(QtWidgets.QLabel("Cut-offs (µm):"))
        self.controls.addWidget(self.values)

        self.cumulative = QtWidgets.QCheckBox("Cumulative")
        self.cumulative.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.cumulative)

        self.activity = QtWidgets.QComboBox()
        self.activity.currentIndexChanged.connect(self.refresh)
        self.controls.addWidget(QtWidgets.QLabel("Activity:"))
        self.controls.addWidget(self.activity)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        self.ax = self.figure.add_subplot(111)

    def _parse_values(self) -> list[float]:
        out: list[float] = []
        for part in self.values.text().split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(float(part))
            except ValueError:
                continue
        return sorted(out)

    def _sync_activities(self) -> None:
        self.activity.blockSignals(True)
        current = self.activity.currentText()
        self.activity.clear()
        self.activity.addItems(self.obj.activities)
        idx = self.activity.findText(current)
        self.activity.setCurrentIndex(idx if idx >= 0 else 0)
        self.activity.blockSignals(False)

    def _plot_on(self, ax) -> None:
        """Compute and draw the stacked Pₓ bands onto ``ax``."""
        values = self._parse_values()
        if not values:
            raise ValueError("Enter one or more numeric cut-off diameters (µm).")

        dtype = self.dtype.currentText()
        dchar = dtype[-1]
        work = self.obj.copy_self()
        work.dtype_converter(dtype=dtype)
        for pm in values:
            work.pm_calc(dtype=dtype, PM=pm)
        activity = self.activity.currentText()
        mask = work.data[activity].astype(bool)
        pm_data = work.extra_data.loc[mask]
        _, unit = helpers.describe(work)

        ax.clear()
        x = pm_data.index
        labels = [f"P{dchar}{v:g}" for v in values]
        cmap = ["#5e3c99", "#998ec3", "#d8daeb", "#fee0b6", "#f1a340", "#b35806"]
        present = [(i, lab) for i, lab in enumerate(labels) if lab in pm_data.columns]

        if self.cumulative.isChecked():
            # Stacked cumulative areas: 0→Pₓ₀, Pₓ₀→Pₓ₁, …  (top of stack = total).
            prev = None
            for i, label in present:
                series = pm_data[label]
                lower = 0 if prev is None else prev
                ax.fill_between(
                    x, lower, series, label=label, color=cmap[i % len(cmap)], alpha=0.9
                )
                prev = series
        else:
            # Differential bands, each drawn from zero, so the concentration in
            # each size range is shown independently (largest first so the
            # smaller ranges stay visible on top).
            bands = []
            prev_label = None
            for i, label in present:
                if prev_label is None:
                    band = pm_data[label]
                    rng = f"0–{label}"
                else:
                    band = pm_data[label] - pm_data[prev_label]
                    rng = f"{prev_label}–{label}"
                bands.append((i, rng, band))
                prev_label = label
            for i, rng, band in sorted(bands, key=lambda t: -float(np.nanmean(t[2]))):
                avg = float(np.nanmean(band))
                ax.fill_between(
                    x, 0, band, label=f"{rng} (μ={avg:.2g})",
                    color=cmap[i % len(cmap)], alpha=0.6,
                )

        ax.set_ylabel(f"P{dchar}, {unit}")
        ax.set_xlabel("Time")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        ax.legend(loc="upper right")

    def refresh(self) -> None:
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        self._sync_activities()
        try:
            self._plot_on(self.ax)
            self.canvas.draw_idle()
        except ValueError as exc:
            self._show_message(str(exc))
        except Exception:
            self._show_message(
                "Could not compute PM bands:\n" + traceback.format_exc(limit=1)
            )

    def _render_export(self, fig) -> None:
        ax = fig.add_subplot(111)
        self._plot_on(ax)

    def current_time_xlim(self):
        if self.ax.has_data():
            return self.ax.get_xlim()
        return None


# ---------------------------------------------------------------------------
# Overlay tab (multi-dataset comparison)
# ---------------------------------------------------------------------------
class OverlayTab(_PlotTab):
    """Overlay one metric over time for several datasets at once.

    Unlike the single-view tabs, this reads *all* of the project's datasets. A
    per-dataset **view-only** time shift lets the user slide instruments in time
    to line up peaks; "Apply shifts permanently" bakes those shifts into the
    datasets' time axes (and re-projects the shared, absolute-time tasks).
    """

    export_tag = "overlay"

    def __init__(self, main):
        super().__init__(main, nrows=1)

        self.metric = QtWidgets.QComboBox()
        self.metric.currentIndexChanged.connect(self.refresh)
        self.controls.addWidget(QtWidgets.QLabel("Metric:"))
        self.controls.addWidget(self.metric)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.stateChanged.connect(self._draw)
        self.controls.addWidget(self.log_y)

        self.normalize = QtWidgets.QCheckBox("Normalize (0–1)")
        self.normalize.setToolTip(
            "Scale each series to 0–1 to compare shapes across instruments with "
            "different units/magnitudes."
        )
        self.normalize.stateChanged.connect(self._draw)
        self.controls.addWidget(self.normalize)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Side panel: per-dataset include + shift table, plus an apply button.
        self._building = False
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Dataset", "Shift (min)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumWidth(320)
        self.table.itemChanged.connect(self._on_item_changed)

        self.apply_btn = QtWidgets.QPushButton("Apply shifts permanently")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.setToolTip(
            "Bake the current view shifts into the datasets (modifies their time "
            "axes). Shared tasks keep their absolute times."
        )
        self.apply_btn.clicked.connect(self._apply_shifts)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Datasets to overlay:"))
        side.addWidget(self.table, stretch=1)
        side.addWidget(self.apply_btn)
        hint = QtWidgets.QLabel(
            "Tick datasets to overlay. Adjust 'Shift (min)' to slide a dataset in "
            "time and line up peaks (view only until you apply permanently)."
        )
        hint.setWordWrap(True)
        side.addWidget(hint)

        # Left column (controls + toolbar + plot) and a full-height side panel.
        self._left_col = QtWidgets.QVBoxLayout()
        self._layout.removeItem(self.controls)
        self._layout.removeWidget(self.toolbar)
        self._layout.removeWidget(self.canvas)
        self._left_col.addLayout(self.controls)
        self._left_col.addWidget(self.toolbar)
        self._left_col.addWidget(self.canvas, stretch=1)
        body = QtWidgets.QHBoxLayout()
        body.addLayout(self._left_col, stretch=1)
        body.addLayout(side)
        self._layout.addLayout(body, stretch=1)

        self.ax = self.figure.add_subplot(111)

    # -- data access -------------------------------------------------------
    @property
    def _datasets(self):
        return self.main.project.datasets

    def _metric_options(self) -> list:
        names = ["Total concentration"]
        seen = set(names)
        for ds in self._datasets:
            for _label, kind, name in helpers.plottable_columns(ds.obj):
                if kind == "total" or name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return names

    def _series_for(self, ds):
        metric = self.metric.currentText()
        if metric == "Total concentration":
            s = ds.obj.total_concentration
        elif metric in ds.obj.data.columns:
            s = ds.obj.data[metric]
        elif ds.obj.extra_data is not None and metric in ds.obj.extra_data.columns:
            s = ds.obj.extra_data[metric]
        else:
            return None
        return pd.to_numeric(s, errors="coerce")

    # -- table sync --------------------------------------------------------
    def _sync_metric(self) -> None:
        current = self.metric.currentText()
        self.metric.blockSignals(True)
        self.metric.clear()
        self.metric.addItems(self._metric_options())
        idx = self.metric.findText(current)
        self.metric.setCurrentIndex(idx if idx >= 0 else 0)
        self.metric.blockSignals(False)

    def _sync_table(self) -> None:
        self._building = True
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for ds in self._datasets:
            r = self.table.rowCount()
            self.table.insertRow(r)
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(
                QtCore.Qt.Checked if ds.overlay_on else QtCore.Qt.Unchecked
            )
            chk.setData(QtCore.Qt.UserRole, ds.id)
            self.table.setItem(r, 0, chk)
            name = QtWidgets.QTableWidgetItem(ds.label)
            name.setFlags(QtCore.Qt.ItemIsEnabled)
            self.table.setItem(r, 1, name)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-100000.0, 100000.0)
            spin.setDecimals(2)
            spin.setValue(ds.view_shift.total_seconds() / 60.0)
            spin.valueChanged.connect(lambda v, d=ds: self._on_shift(d, v))
            self.table.setCellWidget(r, 2, spin)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self._building = False

    # -- interaction -------------------------------------------------------
    def _on_item_changed(self, item) -> None:
        if self._building or item.column() != 0:
            return
        ds = self.main.project.get(item.data(QtCore.Qt.UserRole))
        if ds is not None:
            ds.overlay_on = item.checkState() == QtCore.Qt.Checked
            self._draw()

    def _on_shift(self, ds, minutes: float) -> None:
        if self._building:
            return
        ds.view_shift = pd.Timedelta(minutes=float(minutes))
        self._draw()

    def _apply_shifts(self) -> None:
        moved = [ds for ds in self._datasets if ds.view_shift != pd.Timedelta(0)]
        if not moved:
            QtWidgets.QMessageBox.information(
                self, "Apply shifts", "There are no view shifts to apply."
            )
            return
        ans = QtWidgets.QMessageBox.question(
            self,
            "Apply shifts permanently",
            f"Permanently shift {len(moved)} dataset(s) by their current view "
            "shift?\nThis changes their time axes; shared tasks keep their "
            "absolute times.",
        )
        if ans != QtWidgets.QMessageBox.Yes:
            return
        for ds in moved:
            ds.obj.timeshift(seconds=ds.view_shift.total_seconds())
            self.main.project._apply_activities(ds)
            ds.view_shift = pd.Timedelta(0)
        self.main._sync_crop_fields()
        self.main.refresh_all(reset_view=True)
        self.main._refresh_sidebar()

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        if not self._datasets:
            return
        self._sync_metric()
        self._sync_table()
        self._draw()

    def _draw_on(self, ax) -> None:
        ax.clear()
        plotted = 0
        for ds in self._datasets:
            if not ds.overlay_on:
                continue
            s = self._series_for(ds)
            if s is None or s.empty:
                continue
            x = s.index + ds.view_shift  # view-only shift
            y = s.to_numpy(dtype=float)
            if self.normalize.isChecked():
                lo, hi = np.nanmin(y), np.nanmax(y)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    y = (y - lo) / (hi - lo)
            label = ds.label
            mins = ds.view_shift.total_seconds() / 60.0
            if mins:
                label += f" ({mins:+.0f} min)"
            ax.plot(x, y, lw=1.4, label=label)
            plotted += 1

        ax.set_xlabel("Time")
        ax.set_ylabel(
            "Normalized (0–1)" if self.normalize.isChecked()
            else self.metric.currentText()
        )
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        if self.log_y.isChecked():
            ax.set_yscale("log")
        if plotted:
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "Tick one or more datasets to overlay.",
                ha="center", va="center", transform=ax.transAxes,
            )

    def _draw(self) -> None:
        try:
            self._draw_on(self.ax)
        except Exception:
            self._show_message(
                "Could not draw overlay:\n" + traceback.format_exc(limit=1)
            )
            return
        self.canvas.draw_idle()

    def _render_export(self, fig) -> None:
        self._draw_on(fig.add_subplot(111))

    def current_time_xlim(self):
        if self.ax.has_data():
            return self.ax.get_xlim()
        return None


# ---------------------------------------------------------------------------
# Combined PSD tab (multi-dataset comparison)
# ---------------------------------------------------------------------------
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


class CombinedPSDTab(_PlotTab):
    """Overlay mean particle size distributions across datasets and tasks.

    A comparison tab (like :class:`OverlayTab`): it reads *all* of the project's
    size-resolved (2D) datasets rather than the active one. The user ticks which
    datasets and which activities to compare; one mean-PSD curve is drawn per
    (dataset × activity) pair via the library's own :meth:`Aerosol2D.plot_psd`
    (``ax=`` shared), then relabelled/recoloured so each combination is distinct.
    """

    export_tag = "combined_psd"

    def __init__(self, main):
        super().__init__(main, nrows=1)

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.setChecked(True)
        self.normalize.stateChanged.connect(self._draw)
        self.controls.addWidget(self.normalize)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.stateChanged.connect(self._draw)
        self.controls.addWidget(self.log_y)

        self.band = QtWidgets.QCheckBox("±σ band")
        self.band.setToolTip(
            "Shade each curve's ±1σ spread. Off by default, since the bands "
            "overlap heavily when several curves are compared."
        )
        self.band.stateChanged.connect(self._draw)
        self.controls.addWidget(self.band)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Side panel: datasets to compare (checkable) + activities (multi-select).
        self._building = False
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.setMaximumWidth(280)
        self.ds_list.itemChanged.connect(self._on_ds_changed)

        self.act_list = QtWidgets.QListWidget()
        self.act_list.setMaximumWidth(280)
        self.act_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.act_list.itemSelectionChanged.connect(self._draw)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Datasets to compare:"))
        side.addWidget(self.ds_list, stretch=1)
        side.addWidget(QtWidgets.QLabel("Activities:"))
        side.addWidget(self.act_list, stretch=1)
        hint = QtWidgets.QLabel(
            "Tick the 2D datasets and the activities to compare. One mean-PSD "
            "curve is drawn per dataset × activity."
        )
        hint.setWordWrap(True)
        side.addWidget(hint)

        # Left column (controls + toolbar + plot) and a full-height side panel.
        self._left_col = QtWidgets.QVBoxLayout()
        self._layout.removeItem(self.controls)
        self._layout.removeWidget(self.toolbar)
        self._layout.removeWidget(self.canvas)
        self._left_col.addLayout(self.controls)
        self._left_col.addWidget(self.toolbar)
        self._left_col.addWidget(self.canvas, stretch=1)
        body = QtWidgets.QHBoxLayout()
        body.addLayout(self._left_col, stretch=1)
        body.addLayout(side)
        self._layout.addLayout(body, stretch=1)

        self.ax = self.figure.add_subplot(111)

    # -- data access -------------------------------------------------------
    @property
    def _datasets_2d(self):
        return [d for d in self.main.project.datasets if helpers.is_2d(d.obj)]

    def _selected_activities(self) -> list:
        return [i.text() for i in self.act_list.selectedItems()]

    # -- list sync ---------------------------------------------------------
    def _sync_datasets(self) -> None:
        self._building = True
        self.ds_list.blockSignals(True)
        self.ds_list.clear()
        for ds in self._datasets_2d:
            item = QtWidgets.QListWidgetItem(ds.label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked if ds.psd_on else QtCore.Qt.Unchecked
            )
            item.setData(QtCore.Qt.UserRole, ds.id)
            self.ds_list.addItem(item)
        self.ds_list.blockSignals(False)
        self._building = False

    def _sync_activities(self) -> None:
        self.act_list.blockSignals(True)
        selected = {i.text() for i in self.act_list.selectedItems()}
        self.act_list.clear()
        names = ["All data"] + self.main.project.user_activities()
        for name in names:
            self.act_list.addItem(name)
            if name in selected:
                self.act_list.item(self.act_list.count() - 1).setSelected(True)
        # First time shown (nothing selected yet): default to "All data".
        if not selected and self.act_list.count():
            self.act_list.item(0).setSelected(True)
        self.act_list.blockSignals(False)

    # -- interaction -------------------------------------------------------
    def _on_ds_changed(self, item) -> None:
        if self._building:
            return
        ds = self.main.project.get(item.data(QtCore.Qt.UserRole))
        if ds is not None:
            ds.psd_on = item.checkState() == QtCore.Qt.Checked
            self._draw()

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        self._sync_datasets()
        self._sync_activities()
        self._draw()

    def _draw_on(self, ax) -> None:
        ax.clear()
        datasets = [d for d in self._datasets_2d if d.psd_on]
        activities = self._selected_activities()
        colors = _active_color_cycle()
        normalize = self.normalize.isChecked()
        show_band = self.band.isChecked()

        plotted = 0
        ci = 0
        single = len(datasets) == 1 or len(activities) == 1
        for ds in datasets:
            for act in activities:
                if act not in ds.obj.activities:
                    continue
                n_lines = len(list(ax.lines))
                n_coll = len(list(ax.collections))
                try:
                    ds.obj.plot_psd(activities=[act], normalize=normalize, ax=ax)
                except Exception:
                    continue
                new_lines = list(ax.lines)[n_lines:]
                new_coll = list(ax.collections)[n_coll:]
                if not new_lines:
                    continue  # activity had no samples in this dataset
                color = colors[ci % len(colors)]
                ci += 1
                # Label by whichever dimension actually varies, to keep the
                # legend uncluttered when comparing along just one axis.
                if single and len(activities) == 1:
                    label = ds.label
                elif single and len(datasets) == 1:
                    label = act
                else:
                    label = f"{ds.label} – {act}"
                for j, line in enumerate(new_lines):
                    line.set_color(color)
                    line.set_label(label if j == 0 else "_nolegend_")
                for coll in new_coll:  # fill_between ±1σ envelope
                    if show_band:
                        coll.set_color(color)
                        coll.set_alpha(0.18)
                    else:
                        coll.remove()
                plotted += 1

        if not plotted:
            ax.clear()
            msg = (
                "Tick at least one dataset and one activity to compare."
                if self._datasets_2d
                else "Load size-resolved (2D) datasets to compare their PSDs."
            )
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        if self.log_y.isChecked():
            ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)

    def _draw(self) -> None:
        try:
            self._draw_on(self.ax)
        except Exception:
            self._show_message(
                "Could not draw combined PSD:\n" + traceback.format_exc(limit=1)
            )
            return
        self.canvas.draw_idle()

    def _render_export(self, fig) -> None:
        self._draw_on(fig.add_subplot(111))


# ---------------------------------------------------------------------------
# Cross-instrument summary tab (multi-dataset comparison)
# ---------------------------------------------------------------------------
class CrossSummaryTab(QtWidgets.QWidget):
    """One combined activity/exposure summary table across several datasets.

    The single-dataset :class:`SummaryTab` runs ``summarize_activities`` /
    ``summarize_exposure`` on the active object. This comparison tab runs the
    same core methods on **each ticked dataset**, prepends ``Dataset`` /
    ``Instrument`` columns, and concatenates the per-dataset tables into one
    (columns align by name, so metrics that only some instruments report simply
    leave blanks for the others). It is compute-on-demand (a ``Compute`` button)
    rather than recomputing on every refresh, since exposure stats over many
    datasets can be costly.
    """

    _FRACTION_KINDS = ("PM", "PN", "PS", "PV")

    def __init__(self, main):
        super().__init__()
        self.main = main

        layout = QtWidgets.QVBoxLayout(self)
        body = QtWidgets.QHBoxLayout()
        layout.addLayout(body, stretch=1)

        # -- left: datasets to include ------------------------------------
        self._building = False
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.setMaximumWidth(240)
        self.ds_list.itemChanged.connect(self._on_ds_changed)
        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Datasets to include:"))
        side.addWidget(self.ds_list, stretch=1)
        body.addLayout(side)

        # -- right: controls + table --------------------------------------
        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, stretch=1)

        bar = QtWidgets.QHBoxLayout()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Activity summary", "Exposure summary"])
        self.kind.currentIndexChanged.connect(self._on_kind_change)
        bar.addWidget(QtWidgets.QLabel("Type:"))
        bar.addWidget(self.kind)

        # Metric selector (exposure only): a kind drop-down plus a cut-off
        # drop-down shown only for the size-selective fractions (PM/PN/PS/PV).
        self.metric_label = QtWidgets.QLabel("Metric:")
        bar.addWidget(self.metric_label)
        self.metric_kind = QtWidgets.QComboBox()
        self.metric_kind.currentTextChanged.connect(self._on_metric_kind_change)
        bar.addWidget(self.metric_kind)
        self.metric_cut = QtWidgets.QComboBox()
        self.metric_cut.setEditable(True)
        self.metric_cut.setFixedWidth(80)
        bar.addWidget(self.metric_cut)
        self.metric_cut_label = QtWidgets.QLabel("µm")
        bar.addWidget(self.metric_cut_label)

        self.compute = QtWidgets.QPushButton("Compute")
        self.compute.setObjectName("primary")
        self.compute.clicked.connect(self._compute)
        bar.addWidget(self.compute)
        bar.addStretch(1)
        self.export_btn = QtWidgets.QPushButton("Export to Excel…")
        self.export_btn.clicked.connect(self._export)
        bar.addWidget(self.export_btn)
        right.addLayout(bar)

        # Second row: exposure-limit parameters (only shown for exposure).
        self.exp_bar = QtWidgets.QHBoxLayout()
        self.short_limit = self._add_field("STEL (short-term limit):", "1.0", width=80)
        self.short_window = self._add_field("over", "15min", width=70)
        self.long_limit = self._add_field("OEL (8h limit):", "1.0", width=80)
        self.twa_window = self._add_field("TWA window", "8h", width=70)
        self.exp_bar.addStretch(1)
        right.addLayout(self.exp_bar)

        self.model = PandasTableModel()
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.model)
        self.view.setAlternatingRowColors(True)
        _tune_table(self.view)
        right.addWidget(self.view, stretch=1)

        self.status = QtWidgets.QLabel(
            "Tick datasets and click Compute to build the combined summary."
        )
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        self._on_kind_change()

    # -- small helpers -----------------------------------------------------
    def _add_field(self, label: str, default: str, width: int) -> QtWidgets.QLineEdit:
        lbl = QtWidgets.QLabel(label)
        edit = QtWidgets.QLineEdit(default)
        edit.setFixedWidth(width)
        self.exp_bar.addWidget(lbl)
        self.exp_bar.addWidget(edit)
        edit._label = lbl  # type: ignore[attr-defined]
        return edit

    def _selected_datasets(self) -> list:
        return [d for d in self.main.project.datasets if d.summary_on]

    def _limit_fields(self):
        return (self.short_limit, self.short_window, self.long_limit, self.twa_window)

    def _exposure_widgets(self):
        widgets = [
            self.metric_label,
            self.metric_kind,
            self.metric_cut,
            self.metric_cut_label,
        ]
        for field in self._limit_fields():
            widgets.append(field)
            label = getattr(field, "_label", None)
            if label is not None:
                widgets.append(label)
        return widgets

    # -- metric options ----------------------------------------------------
    def _ensure_metric_options(self) -> None:
        """Populate the metric drop-downs from the union of selected datasets."""
        selected = self._selected_datasets()
        any_2d = any(helpers.is_2d(d.obj) for d in selected)
        cur_kind = self.metric_kind.currentText()
        cur_cut = self.metric_cut.currentText()

        self.metric_kind.blockSignals(True)
        self.metric_cut.blockSignals(True)
        self.metric_kind.clear()
        self.metric_cut.clear()

        kinds = ["PNC"]
        if any_2d:
            kinds += ["MASS", "PM", "PN", "PS", "PV"]
        for d in selected:  # add any extra numeric channels present
            for _label, kind, name in helpers.plottable_columns(d.obj):
                if kind in ("data", "extra") and name not in kinds:
                    kinds.append(name)
        self.metric_kind.addItems(kinds)
        self.metric_cut.addItems(["0.1", "0.25", "0.5", "1", "2.5", "4", "4.2", "10"])

        # Preserve the user's selection across rebuilds where still valid.
        ki = self.metric_kind.findText(cur_kind)
        self.metric_kind.setCurrentText(
            cur_kind if ki >= 0 else ("PM" if any_2d else "PNC")
        )
        self.metric_cut.setCurrentText(cur_cut or "4.2")
        self.metric_kind.blockSignals(False)
        self.metric_cut.blockSignals(False)
        self._on_metric_kind_change()

    def _on_metric_kind_change(self, *_args) -> None:
        exposure = self.kind.currentText() == "Exposure summary"
        is_fraction = self.metric_kind.currentText().strip() in self._FRACTION_KINDS
        show_cut = exposure and is_fraction
        self.metric_cut.setVisible(show_cut)
        self.metric_cut_label.setVisible(show_cut)

    def _build_metric(self) -> str:
        kind = self.metric_kind.currentText().strip()
        if kind in self._FRACTION_KINDS:
            return f"{kind}{self.metric_cut.currentText().strip()}"
        return kind

    def _on_kind_change(self) -> None:
        exposure = self.kind.currentText() == "Exposure summary"
        if exposure:
            self._ensure_metric_options()
        for widget in self._exposure_widgets():
            widget.setVisible(exposure)
        if exposure:
            self._on_metric_kind_change()

    @staticmethod
    def _to_float(text: str, default: float) -> float:
        try:
            return float(text.strip())
        except (ValueError, AttributeError):
            return default

    # -- dataset list sync -------------------------------------------------
    def _sync_datasets(self) -> None:
        self._building = True
        self.ds_list.blockSignals(True)
        self.ds_list.clear()
        for ds in self.main.project.datasets:
            item = QtWidgets.QListWidgetItem(f"{ds.label}  ({ds.instrument})")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked if ds.summary_on else QtCore.Qt.Unchecked
            )
            item.setData(QtCore.Qt.UserRole, ds.id)
            self.ds_list.addItem(item)
        self.ds_list.blockSignals(False)
        self._building = False

    def _on_ds_changed(self, item) -> None:
        if self._building:
            return
        ds = self.main.project.get(item.data(QtCore.Qt.UserRole))
        if ds is not None:
            ds.summary_on = item.checkState() == QtCore.Qt.Checked
        # The available metrics may change with the selection (e.g. a 2D
        # dataset added/removed), so keep the metric drop-downs in step.
        if self.kind.currentText() == "Exposure summary":
            self._ensure_metric_options()

    def refresh(self) -> None:
        """Re-sync the dataset list; the table itself is built on Compute."""
        self._sync_datasets()
        if self.kind.currentText() == "Exposure summary":
            self._ensure_metric_options()

    # -- compute -----------------------------------------------------------
    def _compute(self) -> None:
        datasets = self._selected_datasets()
        if not datasets:
            self.model.set_dataframe(pd.DataFrame())
            self.status.setText("Tick at least one dataset to include.")
            return
        exposure = self.kind.currentText() == "Exposure summary"
        metric = self._build_metric() if exposure else None

        frames: list[pd.DataFrame] = []
        skipped: list[str] = []
        # The core summarize_* methods print their result table to stdout; with
        # many datasets that is just noise (the user reads the GUI table), and on
        # a non-UTF-8 console the unit glyphs (µg/m³, cm⁻³) can even raise an
        # encoding error mid-method. Swallow that console output during compute.
        with contextlib.redirect_stdout(io.StringIO()):
            for ds in datasets:
                try:
                    if exposure:
                        df = ds.obj.summarize_exposure(
                            metric=metric,
                            short_limit=self._to_float(self.short_limit.text(), 1.0),
                            long_limit=self._to_float(self.long_limit.text(), 1.0),
                            short_window=self.short_window.text().strip() or "15min",
                            twa_window=self.twa_window.text().strip() or "8h",
                        )
                    else:
                        df = ds.obj.summarize_activities()
                except Exception as exc:  # e.g. a PM metric on a 1D instrument
                    skipped.append(f"{ds.label} ({exc})")
                    continue
                if df is None or df.empty:
                    continue
                df = df.copy()
                df.insert(0, "Instrument", ds.instrument)
                df.insert(0, "Dataset", ds.label)
                frames.append(df)

        if not frames:
            self.model.set_dataframe(pd.DataFrame())
            msg = "No data to summarize for the current selection."
            if skipped:
                msg += "  Skipped — " + "; ".join(skipped)
            self.status.setText(msg)
            return

        combined = pd.concat(frames, ignore_index=True, sort=False)
        self.model.set_dataframe(combined)
        note = f"{len(frames)} dataset(s) combined, {len(combined)} rows."
        if skipped:
            note += "  Skipped — " + "; ".join(skipped)
        self.status.setText(note)

    def _export(self) -> None:
        df = self.model.dataframe
        kind = (
            "exposure" if self.kind.currentText() == "Exposure summary" else "activity"
        )
        _export_table(
            self, df, f"cross_instrument_{kind}_summary", with_index=False
        )


# ---------------------------------------------------------------------------
# Correlation / Bland–Altman tab (two-dataset comparison)
# ---------------------------------------------------------------------------
class CorrelationTab(_PlotTab):
    """Correlate (or Bland–Altman compare) one parameter between two datasets.

    A two-dataset comparison: pick an **X** and a **Y** dataset and a parameter
    they share, and the tab draws the library's own
    :func:`~aerosoltools.utility.Plot_correlation` (scatter + 1:1 + regression +
    R²) or :func:`~aerosoltools.utility.bland_altman_analysis` (difference plot)
    on the embedded axis. Time alignment between the two instruments is delegated
    to the core via ``match`` ("exact"/"nearest"/"rebin") + ``tolerance``. The
    plot is built on **Compute** (alignment can be costly), so a plain ``refresh``
    only re-syncs the dataset/parameter selectors and leaves the plot in place.
    """

    export_tag = "correlation"

    def __init__(self, main):
        super().__init__(main, nrows=1)

        self.analysis = QtWidgets.QComboBox()
        self.analysis.addItems(["Correlation", "Bland–Altman"])
        self.analysis.currentIndexChanged.connect(self._on_analysis_change)
        self.controls.addWidget(QtWidgets.QLabel("Analysis:"))
        self.controls.addWidget(self.analysis)

        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.currentIndexChanged.connect(self._on_pair_change)
        self.controls.addWidget(QtWidgets.QLabel("X:"))
        self.controls.addWidget(self.x_combo)

        self.y_combo = QtWidgets.QComboBox()
        self.y_combo.currentIndexChanged.connect(self._on_pair_change)
        self.controls.addWidget(QtWidgets.QLabel("Y:"))
        self.controls.addWidget(self.y_combo)

        self.param = QtWidgets.QComboBox()
        self.param.setMinimumWidth(120)
        self.controls.addWidget(QtWidgets.QLabel("Parameter:"))
        self.controls.addWidget(self.param)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # -- side panel: time alignment + analysis options + Compute -------
        self.match = QtWidgets.QComboBox()
        self.match.addItems(["exact", "nearest", "rebin"])
        self.match.currentTextChanged.connect(self._on_match_change)
        self.tolerance = QtWidgets.QLineEdit("30s")
        self.tolerance.setToolTip("Max timestamp separation for 'nearest' match.")
        self.rebin_freq = QtWidgets.QLineEdit()
        self.rebin_freq.setPlaceholderText("auto")
        self.rebin_freq.setToolTip("Common time step for 'rebin', e.g. 1min (blank = auto).")
        self.rebin_method = QtWidgets.QComboBox()
        self.rebin_method.addItems(["mean", "median", "min", "max", "sum"])

        align = QtWidgets.QGroupBox("Time alignment")
        align_col = QtWidgets.QVBoxLayout(align)
        align_col.addWidget(self._field_row("Match:", self.match))
        self._tol_row = self._field_row("Tolerance:", self.tolerance)
        self._freq_row = self._field_row("Rebin to:", self.rebin_freq)
        self._method_row = self._field_row("Aggregation:", self.rebin_method)
        align_col.addWidget(self._tol_row)
        align_col.addWidget(self._freq_row)
        align_col.addWidget(self._method_row)

        # Correlation-only options.
        self.intercept = QtWidgets.QCheckBox("Fit intercept (y = A·x + B)")
        self.intercept.setChecked(True)
        self.uniform = QtWidgets.QCheckBox("Uniform axis scaling")
        self.uniform.setChecked(True)
        self.robust = QtWidgets.QCheckBox("Robust fit (Theil–Sen)")
        self.robust.setToolTip(
            "Use the outlier-resistant Theil–Sen estimator instead of "
            "least-squares (drops the confidence band)."
        )
        self.corr_box = QtWidgets.QGroupBox("Regression")
        corr_col = QtWidgets.QVBoxLayout(self.corr_box)
        corr_col.addWidget(self.intercept)
        corr_col.addWidget(self.uniform)
        corr_col.addWidget(self.robust)

        # Bland–Altman-only options.
        self.ba_method = QtWidgets.QComboBox()
        self.ba_method.addItem("Bland–Altman (difference)", "BA")
        self.ba_method.addItem("Giavarina (% of mean)", "Gi")
        self.ba_method.addItem("Euser (log)", "Eu")
        self.ba_conf = QtWidgets.QDoubleSpinBox()
        self.ba_conf.setRange(0.50, 0.999)
        self.ba_conf.setSingleStep(0.01)
        self.ba_conf.setDecimals(3)
        self.ba_conf.setValue(0.95)
        self.ba_box = QtWidgets.QGroupBox("Bland–Altman")
        ba_col = QtWidgets.QVBoxLayout(self.ba_box)
        ba_col.addWidget(self._field_row("Method:", self.ba_method))
        ba_col.addWidget(self._field_row("Confidence:", self.ba_conf))

        self.compute_btn = QtWidgets.QPushButton("Compute")
        self.compute_btn.setObjectName("primary")
        self.compute_btn.clicked.connect(self._draw)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(align)
        side.addWidget(self.corr_box)
        side.addWidget(self.ba_box)
        side.addWidget(self.compute_btn)
        hint = QtWidgets.QLabel(
            "Pick two datasets and a shared parameter, then click Compute. Use "
            "'nearest'/'rebin' match when the two instruments log on different "
            "time stamps."
        )
        hint.setWordWrap(True)
        side.addWidget(hint)
        side.addStretch(1)
        side_w = QtWidgets.QWidget()
        side_w.setLayout(side)
        side_w.setMaximumWidth(300)

        # Left column (controls + toolbar + plot) and a side panel.
        self._left_col = QtWidgets.QVBoxLayout()
        self._layout.removeItem(self.controls)
        self._layout.removeWidget(self.toolbar)
        self._layout.removeWidget(self.canvas)
        self._left_col.addLayout(self.controls)
        self._left_col.addWidget(self.toolbar)
        self._left_col.addWidget(self.canvas, stretch=1)
        body = QtWidgets.QHBoxLayout()
        body.addLayout(self._left_col, stretch=1)
        body.addWidget(side_w)
        self._layout.addLayout(body, stretch=1)

        self.ax = self.figure.add_subplot(111)
        self._has_drawn = False
        self._on_analysis_change()
        self._on_match_change()

    # -- small helpers -----------------------------------------------------
    @staticmethod
    def _field_row(label_text: str, widget) -> QtWidgets.QWidget:
        """Wrap ``label + widget`` in a row so both can be hidden together."""
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QtWidgets.QLabel(label_text))
        h.addWidget(widget, stretch=1)
        return row

    def _dataset(self, ds_id):
        return self.main.project.get(ds_id)

    # -- visibility toggles ------------------------------------------------
    def _on_analysis_change(self, *_a) -> None:
        is_corr = self.analysis.currentText() == "Correlation"
        self.corr_box.setVisible(is_corr)
        self.ba_box.setVisible(not is_corr)

    def _on_match_change(self, *_a) -> None:
        mode = self.match.currentText()
        self._tol_row.setVisible(mode == "nearest")
        self._freq_row.setVisible(mode == "rebin")
        self._method_row.setVisible(mode == "rebin")

    # -- selector sync -----------------------------------------------------
    def _sync_pair(self) -> None:
        datasets = self.main.project.datasets
        for combo, default_idx in ((self.x_combo, 0), (self.y_combo, 1)):
            combo.blockSignals(True)
            prev = combo.currentData()
            combo.clear()
            for ds in datasets:
                combo.addItem(ds.label, ds.id)
            idx = combo.findData(prev)
            if idx < 0:
                idx = min(default_idx, combo.count() - 1)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _common_parameters(self) -> list:
        """Numeric column names present in *both* selected datasets."""
        xds = self._dataset(self.x_combo.currentData())
        yds = self._dataset(self.y_combo.currentData())
        if xds is None or yds is None:
            return []

        def numeric_cols(obj) -> set:
            cols = set(obj.data.select_dtypes(exclude="bool").columns)
            extra = getattr(obj, "extra_data", None)
            if extra is not None and not extra.empty:
                cols |= set(extra.select_dtypes(exclude="bool").columns)
            return cols - set(getattr(obj, "activities", []))

        common = numeric_cols(xds.obj) & numeric_cols(yds.obj)
        ordered = ["Total_conc"] if "Total_conc" in common else []
        ordered += sorted(c for c in common if c != "Total_conc")
        return ordered

    def _sync_params(self) -> None:
        self.param.blockSignals(True)
        prev = self.param.currentText()
        self.param.clear()
        self.param.addItems(self._common_parameters())
        idx = self.param.findText(prev)
        if idx >= 0:
            self.param.setCurrentIndex(idx)
        self.param.blockSignals(False)

    def _on_pair_change(self, *_a) -> None:
        self._sync_params()

    def refresh(self) -> None:
        """Re-sync selectors only; the plot is (re)built on Compute."""
        self._sync_pair()
        self._sync_params()

    # -- rendering ---------------------------------------------------------
    def _draw_on(self, ax) -> None:
        xds = self._dataset(self.x_combo.currentData())
        yds = self._dataset(self.y_combo.currentData())
        if xds is None or yds is None:
            raise ValueError("Load at least two datasets to compare.")
        if xds is yds:
            raise ValueError("Pick two different datasets for X and Y.")
        if not self.param.count():
            raise ValueError(
                "The two datasets share no common parameter to correlate."
            )

        parameter = self.param.currentText().strip() or None
        mode = self.match.currentText()
        kwargs = dict(
            parameter=parameter,
            match=mode,
            tolerance=self.tolerance.text().strip() or "30s",
        )
        if mode == "rebin":
            kwargs["rebin_freq"] = self.rebin_freq.text().strip() or None
            kwargs["rebin_method"] = self.rebin_method.currentText()

        if self.analysis.currentText() == "Correlation":
            Plot_correlation(
                xds.obj,
                yds.obj,
                ax_in=ax,
                intercept=self.intercept.isChecked(),
                uniform_scaling=self.uniform.isChecked(),
                outlier_influence=not self.robust.isChecked(),
                **kwargs,
            )
        else:
            bland_altman_analysis(
                xds.obj,
                yds.obj,
                ax_in=ax,
                method=self.ba_method.currentData(),
                C=float(self.ba_conf.value()),
                **kwargs,
            )

    def _draw(self) -> None:
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            self._draw_on(ax)
            self.ax = ax
            self._has_drawn = True
            self.canvas.draw_idle()
        except Exception as exc:
            self._show_message(f"Could not compute: {exc}")

    def _render_export(self, fig) -> None:
        self._draw_on(fig.add_subplot(111))
