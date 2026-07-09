"""Multi-dataset time-series overlay comparison tab."""

from __future__ import annotations

import traceback

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from ..._core import _shading
from .. import helpers
from ..qt import QtCore, QtWidgets
from ..widgets import ThresholdControls, WheelLineEdit
from . import _autoscale
from ._base import _active_color_cycle, _PlotTab

#: Line styles encoding which y-axis a series is read against (axis 0/1/2).
_AXIS_STYLES = ["-", "--", ":"]
_AXIS_WORDS = ["solid, left", "dashed, right", "dotted, right"]


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

    When the datasets carry different concentration units (e.g. particle counts
    in cm⁻³, a gas in ppm and black carbon in ng/m³) each unit gets its own
    y-axis (left, right, then a second offset right spine — up to three), and the
    **line style** encodes which axis a series belongs to (solid/dashed/dotted)
    while the **colour** stays tied to the dataset.
    """

    export_tag = "overlay"

    def __init__(self, main):
        """Build the overlay controls and the per-dataset include/shift table."""
        super().__init__(main, nrows=1)

        self.metric = QtWidgets.QComboBox()
        self.metric.currentIndexChanged.connect(self.refresh)
        self.controls.addWidget(QtWidgets.QLabel("Metric:"))
        self.controls.addWidget(self.metric)

        # Per-unit log toggles are rebuilt whenever the shown units change; one
        # checkbox per distinct unit (each unit has its own y-axis/scale).
        self._log_units: set[str] = set()
        self.log_box = QtWidgets.QWidget()
        self._log_layout = QtWidgets.QHBoxLayout(self.log_box)
        self._log_layout.setContentsMargins(0, 0, 0, 0)
        self._log_checks: dict = {}
        self.controls.addWidget(self.log_box)

        self.normalize = QtWidgets.QCheckBox("Normalize (0–1)")
        self.normalize.setToolTip(
            "Scale each series to 0–1 to compare shapes across instruments with "
            "different units/magnitudes."
        )
        self.normalize.stateChanged.connect(self._on_normalize)
        self.controls.addWidget(self.normalize)

        self.show_acts = QtWidgets.QCheckBox("Show activities")
        self.show_acts.setToolTip(
            "Shade the project's activities across every dataset, so you can see "
            "how the different instruments behave within the same task window."
        )
        self.show_acts.stateChanged.connect(lambda: self._draw(preserve=True))
        self.controls.addWidget(self.show_acts)

        # Concentration-threshold (e.g. OEL) overlay; state persists on the project.
        self.threshold = ThresholdControls()
        self.threshold.set_state(self.main.project.plot_thresholds.get(self.export_tag))
        self.threshold.changed.connect(self._on_threshold_changed)
        self.controls.addWidget(self.threshold)

        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Side panel: per-dataset include + shift table, plus an apply button.
        self._building = False
        self._extra_axes: list = []  # secondary y-axes for further units
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Dataset", "Shift"])
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
            "Tick datasets to overlay. Set 'Shift' (h:mm:ss) to slide a dataset "
            "in time and line up peaks — click the field and scroll the mouse "
            "wheel for a smooth ±1 s nudge (Ctrl = ±1 min). View only until you "
            "apply permanently."
        )
        hint.setWordWrap(True)
        side.addWidget(hint)

        # Plot on the left of a resizable divider, side panel on the right.
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        self._split_with_side(side_widget)

        self.ax = self.figure.add_subplot(111)
        self.canvas.setToolTip(
            "Colour identifies the dataset; line style identifies the y-axis: "
            "solid = left, dashed = right, dotted = 2nd right."
        )

    # -- data access -------------------------------------------------------
    @property
    def _datasets(self):
        """All datasets in the project."""
        return self.main.project.datasets

    def _metric_groups(self):
        """Return ``(standard_names, extra_groups)`` for the metric picker.

        ``standard_names`` is the total plus the core data channels shared across
        datasets; ``extra_groups`` is a list of ``(header, [names])`` — one per
        dataset that exposes extra housekeeping columns — so those (often many)
        columns are tucked behind a labelled, per-instrument separator.
        """
        standard = ["Total concentration"]
        seen = set(standard)
        extra_groups = []
        for ds in self._datasets:
            std, extra = helpers.grouped_columns(ds.obj)
            for _label, kind, name in std:
                if kind == "total" or name in seen:
                    continue
                seen.add(name)
                standard.append(name)
            enames, eseen = [], set()
            for _label, _kind, name in extra:
                if name in eseen:
                    continue
                eseen.add(name)
                enames.append(name)
            if enames:
                extra_groups.append((f"Extra: {ds.label}", enames))
        return standard, extra_groups

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

    def _name_unit_for(self, ds):
        """Return the ``(name, unit)`` label pair for ``ds`` under the metric.

        Units are only resolved for the total-concentration metric (where they
        drive the per-unit axes); a named column keeps a single shared axis.
        """
        if self.metric.currentText() == "Total concentration":
            return helpers.measurement_label(ds.obj)
        return self.metric.currentText(), ""

    def _gather_units(self) -> list:
        """Distinct units among the currently included, non-empty series (≤3)."""
        if self.normalize.isChecked():
            return []
        units: list = []
        for ds in self._datasets:
            if not ds.overlay_on:
                continue
            s = self._series_for(ds)
            if s is None or s.empty:
                continue
            u = self._name_unit_for(ds)[1]
            if u not in units:
                units.append(u)
        return units

    # -- table sync --------------------------------------------------------
    def _sync_metric(self) -> None:
        """Repopulate the metric combo (grouped), preserving the selection."""
        current = self.metric.currentText()
        self.metric.blockSignals(True)
        self.metric.clear()
        standard, extra_groups = self._metric_groups()
        for name in standard:
            self.metric.addItem(name)
        for header, names in extra_groups:
            self._add_metric_header(f"── {header} ──")
            for name in names:
                self.metric.addItem(name)
        idx = self.metric.findText(current)
        self.metric.setCurrentIndex(idx if idx >= 0 else 0)
        self.metric.blockSignals(False)

    def _add_metric_header(self, text: str) -> None:
        """Add a non-selectable section header to the metric combo."""
        self.metric.addItem(text)
        item = self.metric.model().item(self.metric.count() - 1)
        if item is not None:
            item.setEnabled(False)

    def _sync_log_controls(self) -> None:
        """Rebuild the per-unit 'Log' checkboxes for the currently shown units."""
        while self._log_layout.count():
            w = self._log_layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._log_checks = {}
        units = self._gather_units()
        self.log_box.setVisible(bool(units))
        for u in units:
            cb = QtWidgets.QCheckBox(f"Log {u}" if u else "Log Y")
            cb.setChecked(u in self._log_units)
            cb.toggled.connect(lambda on, unit=u: self._on_log_toggle(unit, on))
            self._log_layout.addWidget(cb)
            self._log_checks[u] = cb

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
            edit = WheelLineEdit(_format_hms(ds.view_shift))
            edit.setFixedWidth(74)
            edit.setToolTip(
                "View-only time shift as [-]h:mm:ss (e.g. -0:00:30 to move the "
                "dataset 30 s earlier). A bare number is read as seconds. With "
                "the field focused, scroll to nudge ±1 s (Ctrl = ±1 min)."
            )
            edit.editingFinished.connect(lambda e=edit, d=ds: self._on_shift_edit(d, e))
            edit.stepped.connect(
                lambda direction, coarse, e=edit, d=ds: self._on_shift_step(
                    d, e, direction, coarse
                )
            )
            self.table.setCellWidget(r, 2, edit)
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)
        self.table.blockSignals(False)
        self._building = False

    # -- interaction -------------------------------------------------------
    def _on_normalize(self) -> None:
        """Redraw for a normalize toggle (units become irrelevant)."""
        self._sync_log_controls()
        self._draw()

    def _on_log_toggle(self, unit: str, on: bool) -> None:
        """Persist a unit's log-scale choice and redraw (keeping the view)."""
        if on:
            self._log_units.add(unit)
        else:
            self._log_units.discard(unit)
        self._draw(preserve=True)

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
            self._sync_log_controls()
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

    def _on_shift_step(self, ds, edit, direction: int, coarse: bool) -> None:
        """Nudge a dataset's view shift by a wheel tick (±1 s, or ±1 min coarse)."""
        if self._building:
            return
        step = pd.Timedelta(minutes=1) if coarse else pd.Timedelta(seconds=1)
        ds.view_shift = ds.view_shift + direction * step
        edit.blockSignals(True)
        edit.setText(_format_hms(ds.view_shift))
        edit.blockSignals(False)
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
        """Re-sync the metric combo, table and log controls, then redraw."""
        if not self._datasets:
            return
        self._sync_metric()
        self._sync_table()
        self._sync_log_controls()
        self._draw()

    def _clear_extra_axes(self) -> None:
        """Drop any secondary axes from the previous draw."""
        for extra in self._extra_axes:
            try:
                extra.remove()
            except Exception:
                pass
        self._extra_axes = []

    def _gather_entries(self, normalize: bool) -> list:
        """Collect ``(ds, x, y, name, unit)`` for each included, non-empty series."""
        entries = []
        for ds in self._datasets:
            if not ds.overlay_on:
                continue
            s = self._series_for(ds)
            if s is None or s.empty:
                continue
            name, unit = self._name_unit_for(ds)
            x = mdates.date2num((s.index + ds.view_shift).to_pydatetime())
            y = s.to_numpy(dtype=float)
            if normalize:
                lo, hi = np.nanmin(y), np.nanmax(y)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    y = (y - lo) / (hi - lo)
            label = ds.label
            if ds.view_shift != pd.Timedelta(0):
                label += f" ({_format_hms(ds.view_shift)})"
            entries.append((ds, x, y, label, name, unit))
        return entries

    def _draw_on(self, ax) -> None:
        """Draw each included dataset's series, grouping units onto their axes."""
        ax.clear()
        self._clear_extra_axes()
        normalize = self.normalize.isChecked()
        entries = self._gather_entries(normalize)
        cycle = _active_color_cycle()

        if not entries:
            ax.set_xlabel("Time")
            ax.text(
                0.5,
                0.5,
                "Tick one or more datasets to overlay.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        if normalize:
            self._draw_single(ax, entries, cycle)
        else:
            self._draw_multi_axis(ax, entries, cycle)

        # x-range from all series (the twin axes share this axis), then activity
        # shading and a combined legend spanning every axis.
        xs = np.concatenate([e[1] for e in entries])
        xs = xs[np.isfinite(xs)]
        if xs.size:
            ax.set_xlim(float(xs.min()), float(xs.max()))
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        if self.show_acts.isChecked():
            periods = self.main.project.activities
            selected = _shading.resolve_activities(periods, True)
            _shading.shade_activities(ax, periods, selected, zorder=1, legend=False)

        # Threshold line on the primary axis, then one legend across all axes.
        helpers.draw_threshold(
            ax, self.threshold.threshold_value(), self.threshold.legend_text()
        )
        handles, labels = ax.get_legend_handles_labels()
        for extra in self._extra_axes:
            h, ll = extra.get_legend_handles_labels()
            handles += h
            labels += ll
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=8)

    def _draw_single(self, ax, entries, cycle) -> None:
        """Draw all series on one axis (the normalized 0–1 view)."""
        for i, (ds, x, y, label, _name, _unit) in enumerate(entries):
            ax.plot(x, y, lw=1.4, label=label, color=ds.color or cycle[i % len(cycle)])
        ax.set_ylabel("Normalized (0–1)")
        ylim = _autoscale.y_limits([ax], log_y=False, anchor_zero=True)
        if ylim is not None:
            ax.set_ylim(*ylim)

    def _draw_multi_axis(self, ax, entries, cycle) -> None:
        """Draw series with one y-axis per unit (up to three), style = axis."""
        unit_order: list = []
        for _ds, _x, _y, _label, _name, unit in entries:
            if unit not in unit_order:
                unit_order.append(unit)
        if len(unit_order) > 3:
            shown = ", ".join(u or "(dimensionless)" for u in unit_order[:3])
            extra = ", ".join(u or "(dimensionless)" for u in unit_order[3:])
            QtWidgets.QMessageBox.warning(
                self,
                "Too many units",
                "The overlay can show at most three different units at once "
                f"(one per y-axis). Showing: {shown}.\nNot shown: {extra}.\n\n"
                "Untick some datasets to reduce the number of distinct units.",
            )
            unit_order = unit_order[:3]

        # Build one axis per unit: left, right, then an offset right spine.
        axis_for = {}
        for idx, unit in enumerate(unit_order):
            if idx == 0:
                axis_for[unit] = ax
            else:
                twin = ax.twinx()
                if idx == 2:
                    twin.spines["right"].set_position(("outward", 60))
                self._extra_axes.append(twin)
                axis_for[unit] = twin

        names_by_axis: dict = {u: set() for u in unit_order}
        for i, (ds, x, y, label, name, unit) in enumerate(entries):
            if unit not in axis_for:
                continue  # a 4th+ unit that was dropped above
            aidx = unit_order.index(unit)
            target = axis_for[unit]
            color = ds.color or cycle[i % len(cycle)]
            target.plot(x, y, lw=1.4, ls=_AXIS_STYLES[aidx], label=label, color=color)
            names_by_axis[unit].add(name)

        # Label each axis by its (shared) measurement + unit and its style hint,
        # and apply that unit's own log scale and autoscaled limits.
        for aidx, unit in enumerate(unit_order):
            axis = axis_for[unit]
            names = names_by_axis[unit]
            base = next(iter(names)) if len(names) == 1 else "Concentration"
            unit_part = f" [{unit}]" if unit else ""
            axis.set_ylabel(f"{base}{unit_part} ({_AXIS_WORDS[aidx]})")
            is_log = unit in self._log_units
            axis.set_yscale("log" if is_log else "linear")
            ylim = _autoscale.y_limits([axis], log_y=is_log, anchor_zero=not is_log)
            if ylim is not None:
                axis.set_ylim(*ylim)

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
