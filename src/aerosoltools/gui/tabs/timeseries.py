"""Time series tab with interactive activity marking."""

from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.widgets import SpanSelector

from .. import helpers
from ..qt import QtCore, QtWidgets
from ..widgets import ThresholdControls
from ._base import _PlotTab


class ActivityEditorDialog(QtWidgets.QDialog):
    """Add, remove, or adjust the time periods of a single activity.

    Lets the user fine-tune a task's occurrences (each a start/end pair)
    without deleting and re-marking the whole activity.
    """

    def __init__(self, parent, name, periods, default_start, default_end):
        """Build the period table and add/remove controls.

        Args:
            parent: Parent widget.
            name: Activity being edited.
            periods: Existing ``(start, end)`` pairs.
            default_start: Default start for newly added rows.
            default_end: Default end for newly added rows.
        """
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
        """Return a datetime editor initialised to ``value``."""
        editor = QtWidgets.QDateTimeEdit()
        editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        editor.setCalendarPopup(True)
        editor.setDateTime(QtCore.QDateTime(pd.Timestamp(value).to_pydatetime()))
        return editor

    def _add_row(self, start=None, end=None) -> None:
        """Append a start/end editor row (defaulting where unset)."""
        start = self._default[0] if start is None else start
        end = self._default[1] if end is None else end
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setCellWidget(r, 0, self._make_editor(start))
        self.table.setCellWidget(r, 1, self._make_editor(end))

    def _remove_row(self) -> None:
        """Remove the currently selected period row."""
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


class TimeSeriesTab(_PlotTab):
    """Plot a selectable column over time and mark activities by dragging."""

    export_tag = "timeseries"

    def __init__(self, main):
        """Build the series selector, view controls, plot and activities panel."""
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

        # Concentration-threshold (e.g. OEL) overlay. State lives on the project
        # so it survives tab rebuilds and is saved with the project.
        self.threshold = ThresholdControls()
        self.threshold.set_state(self.main.project.plot_thresholds.get(self.export_tag))
        self.threshold.changed.connect(self._on_threshold_changed)
        self.controls.addWidget(self.threshold)

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

        # Two panes split by a draggable divider: a left pane holding the
        # (compact) data-adjustments box — attached later via
        # :meth:`attach_adjust_controls`, which inserts into ``self._left_col`` —
        # plus the view controls, toolbar and plot; and a resizable activities
        # panel on the right.
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        self._split_with_side(side_widget, sizes=(880, 280))

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
        """Enter or leave task-marking mode (toggles the span selector)."""
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
        """Handle a dragged span: prompt for a task name and add the period."""
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

    def _on_threshold_changed(self) -> None:
        """Persist the threshold overlay state and redraw (keeping the view)."""
        self.main.project.plot_thresholds[self.export_tag] = self.threshold.state()
        self.refresh(reset_view=False)

    def _delete_selected(self) -> None:
        """Delete the selected activity across every dataset."""
        item = self.act_list.currentItem()
        if item is None or self.obj is None:
            return
        # Deleting a task removes it from every dataset in the project.
        self.main.project.delete_activity(item.text())
        self.main.refresh_all(reset_view=False)

    def _edit_selected(self) -> None:
        """Open the period editor for the selected activity."""
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
        """Rebuild the series selector, preserving the selection."""
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
        """Refresh the activities list from the active object."""
        self.act_list.clear()
        self.act_list.addItems(helpers.user_activities(self.obj))

    def _cap(self, edit) -> float | None:
        """Parse a Y-axis cap field, returning None when blank or invalid."""
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
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{helpers.base_dtype(dtype)}, {unit}".strip(", "))
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        if self.log_y.isChecked():
            ax.set_yscale("log")
        if self.show_acts.isChecked():
            helpers.shade_activities(ax, self.obj)
        # Threshold line drawn last so it sits on top of the data/shading.
        helpers.draw_threshold(
            ax, self.threshold.threshold_value(), self.threshold.legend_text()
        )

    def refresh(self, reset_view: bool | None = None) -> None:
        """Redraw the selected series and re-sync the selectors.

        Args:
            reset_view: Autoscale when True; preserve the current zoom/pan when
                False. Defaults to the window's pending reset-view flag.
        """
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

    def current_time_xlim(self):
        """Time-axis limits as date numbers, or None."""
        if self._has_drawn and self.ax.has_data():
            return self.ax.get_xlim()
        return None
