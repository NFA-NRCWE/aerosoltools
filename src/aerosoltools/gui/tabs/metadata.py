"""Metadata tab: inspect the active dataset's metadata, size bins and density.

Shows the metadata the program extracted from the raw file and maintains during
handling (instrument, serial, units, loader-specific fields …) as a read-only
table, a full per-bin size table (edges + midpoints), and hosts the density
control and a size-bin crop — the single, instrument-specific place for these.
"""

from __future__ import annotations

import traceback

import numpy as np

from .. import calibration as calib
from .. import helpers
from ..qt import QtWidgets

#: Metadata keys shown via the dedicated size-bin table instead of a raw dump.
_BIN_KEYS = {"bin_edges", "bin_mids"}
#: Keys not shown in the general metadata dump (surfaced elsewhere in the pane):
#: the bin arrays go to the size table, and the raw ``calibrated`` dict is shown
#: as human-readable functions via the calibration controls (items 6/7).
_HIDDEN_KEYS = _BIN_KEYS | {"calibrated"}


def _format_value(value) -> str:
    """Render a metadata value compactly (summarise big arrays).

    Plain floats are rendered with a few significant figures so binary-float
    noise (e.g. a density of ``0.9999999999999998`` after stepping) shows as a
    clean ``1`` instead of sixteen digits.
    """
    if isinstance(value, np.ndarray) or (
        isinstance(value, (list, tuple)) and len(value) > 6
    ):
        arr = np.asarray(value, dtype=float).ravel()
        return f"{arr.size} values" if arr.size else "(empty)"
    if isinstance(value, float) and not isinstance(value, bool):
        return f"{value:.6g}"
    return str(value)


class MetadataTab(QtWidgets.QWidget):
    """Metadata + size-bin inspection, density control and size-bin cropping."""

    def __init__(self, main):
        """Build the density/axis/crop controls and the metadata + bin tables."""
        super().__init__()
        self.main = main

        layout = QtWidgets.QVBoxLayout(self)

        # -- measurement label (all datasets) ---------------------------------
        self.meas_row = QtWidgets.QWidget()
        mrow = QtWidgets.QHBoxLayout(self.meas_row)
        mrow.setContentsMargins(0, 0, 0, 0)
        mrow.addWidget(QtWidgets.QLabel("Measurement:"))
        self.meas_edit = QtWidgets.QLineEdit()
        self.meas_edit.setMaximumWidth(240)
        self.meas_edit.setToolTip(
            "Name of the measured quantity (e.g. 'Cl₂', 'IR BCc'). Shown on the "
            "y-axis and legend across the panes instead of the generic 'Total "
            "concentration'. Clear the field to restore the default label."
        )
        self.meas_edit.editingFinished.connect(self._apply_measurement)
        mrow.addWidget(self.meas_edit)
        mrow.addStretch(1)
        layout.addWidget(self.meas_row)

        # -- density (size-resolved data only) --------------------------------
        self.density_row = QtWidgets.QWidget()
        drow = QtWidgets.QHBoxLayout(self.density_row)
        drow.setContentsMargins(0, 0, 0, 0)
        drow.addWidget(QtWidgets.QLabel("Particle density (g/cm³):"))
        self.density_spin = QtWidgets.QDoubleSpinBox()
        self.density_spin.setRange(0.1, 25.0)
        self.density_spin.setSingleStep(0.1)
        self.density_spin.setValue(1.0)
        self.density_spin.setToolTip(
            "Particle density used for mass-based conversions (and, for the "
            "ELPI, to recompute the density-dependent particle sizes)."
        )
        drow.addWidget(self.density_spin)
        self.apply_all = QtWidgets.QCheckBox("apply to all datasets")
        drow.addWidget(self.apply_all)
        self.apply_btn = QtWidgets.QPushButton("Set density")
        self.apply_btn.clicked.connect(self._apply_density)
        drow.addWidget(self.apply_btn)
        drow.addStretch(1)
        layout.addWidget(self.density_row)

        # -- APS axis selector (correlated Aerosol3d only) --------------------
        self.axis_row = QtWidgets.QWidget()
        arow = QtWidgets.QHBoxLayout(self.axis_row)
        arow.setContentsMargins(0, 0, 0, 0)
        arow.addWidget(QtWidgets.QLabel("Show axis (APS):"))
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItem("Aerodynamic", "aerodynamic")
        self.axis_combo.addItem("Optical", "optical")
        self.axis_combo.setToolTip(
            "For a correlated APS, choose which size axis the other tabs show "
            "and analyse — aerodynamic or optical (both behave as 2D data)."
        )
        self.axis_combo.currentIndexChanged.connect(self._on_axis_change)
        arow.addWidget(self.axis_combo)
        arow.addStretch(1)
        layout.addWidget(self.axis_row)

        # -- size-bin crop (size-resolved data only) --------------------------
        self.crop_row = QtWidgets.QWidget()
        crow = QtWidgets.QHBoxLayout(self.crop_row)
        crow.setContentsMargins(0, 0, 0, 0)
        crow.addWidget(QtWidgets.QLabel("Crop size bins — keep from"))
        self.crop_from = QtWidgets.QComboBox()
        self.crop_to = QtWidgets.QComboBox()
        for c in (self.crop_from, self.crop_to):
            c.setToolTip("Bin-edge diameters (nm) to keep between (inclusive).")
        crow.addWidget(self.crop_from)
        crow.addWidget(QtWidgets.QLabel("to"))
        crow.addWidget(self.crop_to)
        crow.addWidget(QtWidgets.QLabel("nm"))
        self.crop_btn = QtWidgets.QPushButton("Apply crop")
        self.crop_btn.setToolTip(
            "Drop the size bins outside the chosen range (e.g. remove the two "
            "smallest bins). Structural — use the sidebar Reload to undo."
        )
        self.crop_btn.clicked.connect(self._apply_crop)
        crow.addWidget(self.crop_btn)
        crow.addStretch(1)
        layout.addWidget(self.crop_row)

        # -- tables: general metadata (+ calibration) + per-bin sizes ---------
        tables = QtWidgets.QHBoxLayout()

        # Left: the general metadata table with the calibration summary/controls
        # directly beneath it (a total-concentration calibration is shown here).
        left = QtWidgets.QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Field", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        left.addWidget(self.table, stretch=1)
        self._build_calibration_controls(left)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        tables.addWidget(left_widget, stretch=3)

        # Right: the per-bin size table, with a per-bin calibration column.
        self.bins = QtWidgets.QTableWidget(0, 5)
        self.bins.setHorizontalHeaderLabels(
            ["#", "lower (nm)", "mid (nm)", "upper (nm)", "calibration"]
        )
        self.bins.verticalHeader().setVisible(False)
        self.bins.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.bins.setToolTip(
            "All size-bin edges and midpoints, plus each bin's calibration "
            "function when a per-bin calibration is applied."
        )
        tables.addWidget(self.bins, stretch=2)
        layout.addLayout(tables, stretch=1)

    def _build_calibration_controls(self, parent_layout) -> None:
        """Build the calibration summary + on/off/reset controls (item 6/7)."""
        self.cal_group = QtWidgets.QGroupBox("Calibration")
        cal = QtWidgets.QVBoxLayout(self.cal_group)
        cal.setContentsMargins(8, 4, 8, 6)
        row = QtWidgets.QHBoxLayout()
        self.cal_enable = QtWidgets.QCheckBox("Apply calibration")
        self.cal_enable.setToolTip(
            "Toggle the fitted calibration on or off. When off, the original "
            "(uncalibrated) data is shown across every pane."
        )
        self.cal_enable.toggled.connect(self._on_cal_toggle)
        self.cal_reset = QtWidgets.QPushButton("Reset")
        self.cal_reset.setToolTip(
            "Remove the calibration and restore the original data."
        )
        self.cal_reset.clicked.connect(self._on_cal_reset)
        row.addWidget(self.cal_enable)
        row.addWidget(self.cal_reset)
        row.addStretch(1)
        cal.addLayout(row)
        self.cal_summary = QtWidgets.QLabel("")
        self.cal_summary.setWordWrap(True)
        self.cal_summary.setStyleSheet("color: palette(mid);")
        cal.addWidget(self.cal_summary)
        parent_layout.addWidget(self.cal_group)
        self.cal_group.setVisible(False)

    @property
    def obj(self):
        """The active dataset's parent object (aerodynamic axis for a 3D APS)."""
        return getattr(self.main, "active_obj", None) or self.main.obj

    # -- actions -----------------------------------------------------------
    def _apply_density(self) -> None:
        """Push the chosen density onto the active dataset (or all).

        The value is rounded so repeated spin-box stepping cannot store a noisy
        density (e.g. ``0.9999999999999998``) that then shows with many digits.
        """
        self.main.apply_density(
            round(float(self.density_spin.value()), 4),
            all_datasets=self.apply_all.isChecked(),
        )

    def _on_axis_change(self) -> None:
        """Switch the active APS axis (aerodynamic/optical) for all tabs."""
        self.main.set_active_axis(self.axis_combo.currentData() or "aerodynamic")

    def _apply_measurement(self) -> None:
        """Store an edited measurement label on the active object (or clear it)."""
        obj = self.obj
        if obj is None:
            return
        text = self.meas_edit.text().strip()
        current = obj.metadata.get("measurement")
        if text == (current or "") or (not text and current is None):
            return  # no change
        if text:
            obj._meta["measurement"] = text
        else:
            obj._meta.pop("measurement", None)
        self.main.refresh_all(reset_view=False)

    def _on_cal_toggle(self, checked: bool) -> None:
        """Enable/disable the active dataset's calibration and refresh all panes."""
        ds = self.main.project.active
        if ds is None or ds.calibration is None:
            return
        if bool(checked) == ds.calibration_enabled:
            return
        calib.set_calibration_enabled(ds, checked)
        self.main.project._apply_activities(ds)  # obj was rebuilt
        self.main.refresh_all(reset_view=False)

    def _on_cal_reset(self) -> None:
        """Remove the active dataset's calibration and restore the original data."""
        ds = self.main.project.active
        if ds is None or ds.calibration is None:
            return
        calib.reset_calibration(ds)
        self.main.project._apply_activities(ds)
        self.main.refresh_all(reset_view=False)

    def _apply_crop(self) -> None:
        """Keep only the bins between the two chosen edges (conservative rebin)."""
        obj = self.obj
        if obj is None or not helpers.is_2d(obj):
            return
        lo = self.crop_from.currentData()
        hi = self.crop_to.currentData()
        if lo is None or hi is None or lo >= hi:
            QtWidgets.QMessageBox.warning(
                self, "Crop size bins", "Pick a valid 'from' edge below the 'to' edge."
            )
            return
        edges = np.asarray(obj.bin_edges, dtype=float)
        keep = edges[(edges >= lo - 1e-6) & (edges <= hi + 1e-6)]
        if keep.size < 2:
            QtWidgets.QMessageBox.warning(
                self, "Crop size bins", "The chosen range leaves no whole bins."
            )
            return
        try:
            obj.rebin_bin_edges(keep)
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "Crop failed", traceback.format_exc(limit=1)
            )
            return
        self.main.refresh_all(reset_view=True)

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        """Repopulate the metadata/size tables and the density/crop controls."""
        obj = self.obj
        if obj is None:
            self.table.setRowCount(0)
            self.bins.setRowCount(0)
            return

        # Seed the editable measurement label from the resolved default.
        self.meas_edit.blockSignals(True)
        self.meas_edit.setText(helpers.measurement_label(obj)[0])
        self.meas_edit.blockSignals(False)

        is2d = helpers.is_2d(obj)
        self.density_row.setVisible(is2d)
        self.crop_row.setVisible(is2d)
        if is2d:
            self.density_spin.blockSignals(True)
            self.density_spin.setValue(float(getattr(obj, "density", 1.0)))
            self.density_spin.blockSignals(False)

        is_3d = getattr(obj, "is_correlated", False)
        self.axis_row.setVisible(is_3d)
        if is_3d:
            self.axis_combo.blockSignals(True)
            self.axis_combo.setCurrentIndex(
                1
                if getattr(self.main, "_active_axis", "aerodynamic") == "optical"
                else 0
            )
            self.axis_combo.blockSignals(False)

        # General metadata (bin arrays go to the dedicated size table below).
        meta = dict(getattr(obj, "metadata", {}) or {})
        rows = [("class", type(obj).__name__)]
        rows += [
            (str(k), _format_value(v)) for k, v in meta.items() if k not in _HIDDEN_KEYS
        ]
        self.table.setRowCount(len(rows))
        for r, (key, val) in enumerate(rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(key))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(val))
        self.table.resizeColumnToContents(0)

        self._fill_bins(obj if is2d else None)
        self._refresh_calibration()

    def _refresh_calibration(self) -> None:
        """Update the calibration summary + on/off state for the active dataset."""
        ds = self.main.project.active
        has = ds is not None and ds.calibration is not None
        self.cal_group.setVisible(bool(has))
        if not has:
            return
        self.cal_enable.blockSignals(True)
        self.cal_enable.setChecked(ds.calibration_enabled)
        self.cal_enable.blockSignals(False)
        self.cal_summary.setText(calib.calibration_summary(ds) or "")

    def _fill_bins(self, obj) -> None:
        """Fill the per-bin size table, its calibration column and the crop pickers."""
        if obj is None:
            self.bins.setRowCount(0)
            return
        edges = np.asarray(obj.bin_edges, dtype=float)
        mids = np.asarray(obj.bin_mids, dtype=float)
        ds = self.main.project.active
        cal_texts = calib.bin_calibration_texts(ds) if ds is not None else None
        self.bins.setRowCount(len(mids))
        for i, mid in enumerate(mids):
            cal = cal_texts[i] if (cal_texts and i < len(cal_texts)) else ""
            cells = [
                str(i + 1),
                f"{edges[i]:g}",
                f"{mid:g}",
                f"{edges[i + 1]:g}",
                cal,
            ]
            for c, text in enumerate(cells):
                self.bins.setItem(i, c, QtWidgets.QTableWidgetItem(text))
        self.bins.resizeColumnsToContents()

        # Crop selectors: the bin edges (from-edge < to-edge).
        cur_lo = self.crop_from.currentData()
        cur_hi = self.crop_to.currentData()
        for combo, default in ((self.crop_from, edges[0]), (self.crop_to, edges[-1])):
            combo.blockSignals(True)
            combo.clear()
            for e in edges:
                combo.addItem(f"{e:g}", float(e))
            combo.blockSignals(False)
        # Restore the previous selection when still valid, else span the range.
        self._select_edge(self.crop_from, cur_lo, edges[0])
        self._select_edge(self.crop_to, cur_hi, edges[-1])

    @staticmethod
    def _select_edge(combo, value, fallback) -> None:
        """Select ``value`` in an edge combo if present, else ``fallback``."""
        target = value if value is not None else fallback
        idx = combo.findData(target)
        combo.setCurrentIndex(idx if idx >= 0 else combo.findData(fallback))
