"""Multi-dataset time-series overlay comparison tab."""

from __future__ import annotations

import traceback

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .. import helpers
from ..qt import QtCore, QtWidgets
from ..widgets import ThresholdControls
from ._base import _active_color_cycle, _PlotTab


def _format_hms(td: pd.Timedelta) -> str:
    """Render a timedelta as a signed ``[-]H:MM:SS`` string (whole seconds)."""
    total = int(round(td.total_seconds()))
    sign = "-" if total < 0 else ""
    total = abs(total)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h}:{m:02d}:{s:02d}"


def _parse_hms(text: str):
    """Parse ``[-]H:MM:SS`` / ``M:SS`` / bare seconds into a timedelta, or None.

    A bare number is read as seconds, so small nudges (a few seconds) are easy
    to type. Returns ``None`` on unparseable input so the caller can revert.
    """
    text = text.strip()
    if not text:
        return pd.Timedelta(0)
    sign = 1
    if text[0] in "+-":
        sign = -1 if text[0] == "-" else 1
        text = text[1:].strip()
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 1:
        h, m, s = 0.0, 0.0, nums[0]
    elif len(parts) == 2:
        h, m, s = 0.0, nums[0], nums[1]
    elif len(parts) == 3:
        h, m, s = nums
    else:
        return None
    return pd.Timedelta(seconds=sign * (h * 3600 + m * 60 + s))


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

        # Concentration-threshold (e.g. OEL) overlay; state persists on the project.
        self.threshold = ThresholdControls()
        self.threshold.set_state(self.main.project.plot_thresholds.get(self.export_tag))
        self.threshold.changed.connect(self._on_threshold_changed)
        self.controls.addWidget(self.threshold)

        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Side panel: per-dataset include + shift table, plus an apply button.
        self._building = False
        self._ax2 = None  # secondary y-axis for a second unit (e.g. ppm)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Dataset", "Shift (h:mm:ss)"])
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
            "Tick datasets to overlay. Set 'Shift (h:mm:ss)' to slide a dataset "
            "in time and line up peaks (view only until you apply permanently)."
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
            edit = QtWidgets.QLineEdit(_format_hms(ds.view_shift))
            edit.setToolTip(
                "View-only time shift as [-]h:mm:ss (e.g. -0:00:30 to move the "
                "dataset 30 s earlier). A bare number is read as seconds."
            )
            edit.editingFinished.connect(lambda e=edit, d=ds: self._on_shift_edit(d, e))
            self.table.setCellWidget(r, 2, edit)
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
            # Rescale to the now-visible datasets so the axis limits (and the
            # "reset view" target) follow the shown data instead of stretching to
            # cover a dataset that was just hidden.
            self._draw()

    def _on_shift_edit(self, ds, edit) -> None:
        """Parse an h:mm:ss shift field, persist the view-only shift and redraw."""
        if self._building:
            return
        td = _parse_hms(edit.text())
        if td is None:  # unparseable — revert to the current shift
            edit.setText(_format_hms(ds.view_shift))
            return
        ds.view_shift = td
        edit.blockSignals(True)
        edit.setText(_format_hms(td))  # normalise the display
        edit.blockSignals(False)
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

    def _on_threshold_changed(self) -> None:
        """Persist the threshold overlay state and redraw (keeping the view)."""
        self.main.project.plot_thresholds[self.export_tag] = self.threshold.state()
        self._draw(preserve=True)

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        """Re-sync the metric combo and table, then redraw."""
        if not self._datasets:
            return
        self._sync_metric()
        self._sync_table()
        self._draw()

    def _draw_on(self, ax) -> None:
        """Draw each included dataset's (optionally normalized, shifted) series.

        When the datasets carry different concentration units (e.g. particle
        counts in cm⁻³ and a gas in ppm), the second unit is drawn on a secondary
        right-hand y-axis so both data types are legible on one plot (each axis
        autoscales to its own series).
        """
        ax.clear()
        # Drop any secondary axis from the previous draw before rebuilding.
        if self._ax2 is not None:
            try:
                self._ax2.remove()
            except Exception:
                pass
            self._ax2 = None

        metric = self.metric.currentText()
        normalize = self.normalize.isChecked()

        # Gather each included, non-empty series with its unit (units are only
        # reliably known for the total-concentration metric).
        entries = []  # (x, y, label, unit)
        for ds in self._datasets:
            if not ds.overlay_on:
                continue
            s = self._series_for(ds)
            if s is None or s.empty:
                continue
            unit = (
                helpers.describe(ds.obj)[1] if metric == "Total concentration" else ""
            )
            x = s.index + ds.view_shift  # view-only shift
            y = s.to_numpy(dtype=float)
            if normalize:
                lo, hi = np.nanmin(y), np.nanmax(y)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    y = (y - lo) / (hi - lo)
            label = ds.label
            if ds.view_shift != pd.Timedelta(0):
                label += f" ({_format_hms(ds.view_shift)})"
            entries.append((x, y, label, unit))

        # Distinct units among the plotted series — a second one triggers the
        # secondary axis (but not when everything is normalised to 0–1).
        units = []
        for _x, _y, _label, unit in entries:
            if unit and unit not in units:
                units.append(unit)
        use_dual = (
            (not normalize) and metric == "Total concentration" and len(units) >= 2
        )
        primary_unit = units[0] if units else ""

        ax2 = ax.twinx() if use_dual else None
        self._ax2 = ax2

        # Assign colours across *all* series (both axes) from the shared cycle so
        # two datasets never collide, even when split over the two axes.
        colors = _active_color_cycle()
        for i, (x, y, label, unit) in enumerate(entries):
            target = ax2 if (use_dual and unit != primary_unit) else ax
            target.plot(x, y, lw=1.4, label=label, color=colors[i % len(colors)])
        plotted = len(entries)

        ax.set_xlabel("Time")
        if normalize:
            ax.set_ylabel("Normalized (0–1)")
        elif metric == "Total concentration" and primary_unit:
            ax.set_ylabel(f"{metric} [{primary_unit}]")
            if use_dual:
                ax2.set_ylabel(f"{metric} [{units[1]}]")
        else:
            ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        if self.log_y.isChecked():
            ax.set_yscale("log")
            if ax2 is not None:
                ax2.set_yscale("log")

        if not plotted:
            ax.text(
                0.5,
                0.5,
                "Tick one or more datasets to overlay.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Threshold line (e.g. OEL) on the primary axis, then a combined legend
        # spanning both axes (the threshold rebuilds its own legend on ax alone,
        # so build the final one afterwards to keep the right-axis series in it).
        helpers.draw_threshold(
            ax, self.threshold.threshold_value(), self.threshold.legend_text()
        )
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels() if ax2 is not None else ([], [])
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

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
        else:
            self._sync_toolbar_home()
        self.canvas.draw_idle()

    def current_time_xlim(self):
        """Time-axis limits as date numbers, or None."""
        if self.ax.has_data():
            return self.ax.get_xlim()
        return None
