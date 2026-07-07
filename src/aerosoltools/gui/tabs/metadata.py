"""Metadata tab: inspect the active dataset's metadata, size bins and density.

Shows the metadata the program extracted from the raw file and maintains during
handling (instrument, serial, units, loader-specific fields …) as a read-only
table, a full per-bin size table (edges + midpoints), and hosts the density
control and a size-bin crop — the single, instrument-specific place for these.
"""

from __future__ import annotations

import traceback

import numpy as np

from .. import helpers
from ..qt import QtWidgets

#: Metadata keys shown via the dedicated size-bin table instead of a raw dump.
_BIN_KEYS = {"bin_edges", "bin_mids"}


def _format_value(value) -> str:
    """Render a metadata value compactly (summarise big arrays)."""
    if isinstance(value, np.ndarray) or (
        isinstance(value, (list, tuple)) and len(value) > 6
    ):
        arr = np.asarray(value, dtype=float).ravel()
        return f"{arr.size} values" if arr.size else "(empty)"
    return str(value)


class MetadataTab(QtWidgets.QWidget):
    """Metadata + size-bin inspection, density control and size-bin cropping."""

    def __init__(self, main):
        """Build the density/axis/crop controls and the metadata + bin tables."""
        super().__init__()
        self.main = main

        layout = QtWidgets.QVBoxLayout(self)

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

        # -- tables: general metadata + per-bin sizes -------------------------
        tables = QtWidgets.QHBoxLayout()
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Field", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        tables.addWidget(self.table, stretch=3)

        self.bins = QtWidgets.QTableWidget(0, 4)
        self.bins.setHorizontalHeaderLabels(
            ["#", "lower (nm)", "mid (nm)", "upper (nm)"]
        )
        self.bins.verticalHeader().setVisible(False)
        self.bins.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.bins.setToolTip("All size-bin edges and midpoints, for inspection.")
        tables.addWidget(self.bins, stretch=2)
        layout.addLayout(tables, stretch=1)

    @property
    def obj(self):
        """The active dataset's parent object (aerodynamic axis for a 3D APS)."""
        return getattr(self.main, "active_obj", None) or self.main.obj

    # -- actions -----------------------------------------------------------
    def _apply_density(self) -> None:
        """Push the chosen density onto the active dataset (or all)."""
        self.main.apply_density(
            float(self.density_spin.value()), all_datasets=self.apply_all.isChecked()
        )

    def _on_axis_change(self) -> None:
        """Switch the active APS axis (aerodynamic/optical) for all tabs."""
        self.main.set_active_axis(self.axis_combo.currentData() or "aerodynamic")

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
            (str(k), _format_value(v)) for k, v in meta.items() if k not in _BIN_KEYS
        ]
        self.table.setRowCount(len(rows))
        for r, (key, val) in enumerate(rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(key))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(val))
        self.table.resizeColumnToContents(0)

        self._fill_bins(obj if is2d else None)

    def _fill_bins(self, obj) -> None:
        """Fill the per-bin size table and the crop-range selectors."""
        if obj is None:
            self.bins.setRowCount(0)
            return
        edges = np.asarray(obj.bin_edges, dtype=float)
        mids = np.asarray(obj.bin_mids, dtype=float)
        self.bins.setRowCount(len(mids))
        for i, mid in enumerate(mids):
            cells = [str(i + 1), f"{edges[i]:g}", f"{mid:g}", f"{edges[i + 1]:g}"]
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
