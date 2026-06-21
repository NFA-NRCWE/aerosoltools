"""Multi-dataset time-series overlay comparison tab."""

from __future__ import annotations

import traceback

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .. import helpers
from ..qt import QtCore, QtWidgets
from ._base import _PlotTab


class OverlayTab(_PlotTab):
    """Overlay one metric over time for several datasets at once.

    Unlike the single-view tabs, this reads *all* of the project's datasets. A
    per-dataset **view-only** time shift lets the user slide instruments in time
    to line up peaks; "Apply shifts permanently" bakes those shifts into the
    datasets' time axes (and re-projects the shared, absolute-time tasks).
    """

    export_tag = "overlay"

    def __init__(self, main):
        """Build the overlay controls and the per-dataset include/shift table."""
        super().__init__(main, nrows=1)

        self.metric = QtWidgets.QComboBox()
        self.metric.currentIndexChanged.connect(self.refresh)
        self.controls.addWidget(QtWidgets.QLabel("Metric:"))
        self.controls.addWidget(self.metric)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        # A scale change rescales the axes, so don't preserve the old view.
        self.log_y.stateChanged.connect(lambda: self._draw())
        self.controls.addWidget(self.log_y)

        self.normalize = QtWidgets.QCheckBox("Normalize (0–1)")
        self.normalize.setToolTip(
            "Scale each series to 0–1 to compare shapes across instruments with "
            "different units/magnitudes."
        )
        self.normalize.stateChanged.connect(lambda: self._draw())
        self.controls.addWidget(self.normalize)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Side panel: per-dataset include + shift table, plus an apply button.
        self._building = False
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Dataset", "Shift (min)"])
        self.table.verticalHeader().setVisible(False)
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

        # Plot on the left of a resizable divider, side panel on the right.
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        self._split_with_side(side_widget)

        self.ax = self.figure.add_subplot(111)

    # -- data access -------------------------------------------------------
    @property
    def _datasets(self):
        """All datasets in the project."""
        return self.main.project.datasets

    def _metric_options(self) -> list:
        """Metric names available across the datasets."""
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
        """Return the numeric series for a dataset's chosen metric, or None."""
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
        """Repopulate the metric combo, preserving the selection."""
        current = self.metric.currentText()
        self.metric.blockSignals(True)
        self.metric.clear()
        self.metric.addItems(self._metric_options())
        idx = self.metric.findText(current)
        self.metric.setCurrentIndex(idx if idx >= 0 else 0)
        self.metric.blockSignals(False)

    def _sync_table(self) -> None:
        """Rebuild the include/shift table from the datasets."""
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
        """Persist a dataset's overlay-include flag and redraw."""
        if self._building or item.column() != 0:
            return
        ds = self.main.project.get(item.data(QtCore.Qt.UserRole))
        if ds is not None:
            ds.overlay_on = item.checkState() == QtCore.Qt.Checked
            # Keep the current zoom/pan: ticking a dataset shouldn't reset it.
            self._draw(preserve=True)

    def _on_shift(self, ds, minutes: float) -> None:
        """Persist a dataset's view-only time shift and redraw."""
        if self._building:
            return
        ds.view_shift = pd.Timedelta(minutes=float(minutes))
        # Preserve the current view so the user can zoom in, then slide a
        # dataset and watch it move within the same window.
        self._draw(preserve=True)

    def _apply_shifts(self) -> None:
        """Bake the current view shifts permanently into the datasets' time axes."""
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
        self.main.adjust_box.sync_crop_fields()
        self.main.refresh_all(reset_view=True)
        self.main._refresh_sidebar()

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        """Re-sync the metric combo and table, then redraw."""
        if not self._datasets:
            return
        self._sync_metric()
        self._sync_table()
        self._draw()

    def _draw_on(self, ax) -> None:
        """Draw each included dataset's (optionally normalized, shifted) series."""
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
            "Normalized (0–1)"
            if self.normalize.isChecked()
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
                0.5,
                0.5,
                "Tick one or more datasets to overlay.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _draw(self, preserve: bool = False) -> None:
        """Redraw onto the embedded axis, reporting any error inline.

        Args:
            preserve: When True, keep the current axis limits (zoom/pan) across
                the redraw instead of autoscaling — used for view-only shifts and
                include toggles so the user's zoom survives.
        """
        prev = None
        if preserve and self.ax.has_data():
            prev = (self.ax.get_xlim(), self.ax.get_ylim())
        try:
            self._draw_on(self.ax)
        except Exception:
            self._show_message(
                "Could not draw overlay:\n" + traceback.format_exc(limit=1)
            )
            return
        if prev is not None:
            self.ax.set_xlim(prev[0])
            self.ax.set_ylim(prev[1])
        self.canvas.draw_idle()

    def _render_export(self, fig) -> None:
        """Draw the overlay onto a fresh export figure."""
        self._draw_on(fig.add_subplot(111))

    def current_time_xlim(self):
        """Time-axis limits as date numbers, or None."""
        if self.ax.has_data():
            return self.ax.get_xlim()
        return None
