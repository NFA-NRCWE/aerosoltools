"""Metadata tab: inspect the active dataset's metadata and set its density.

Shows the metadata the program extracted from the raw file and maintains during
handling (instrument, serial, units, size range, loader-specific fields …) as a
read-only table, and hosts the particle-density control (per dataset, or applied
to all datasets) — the single place density is set, to avoid inconsistent
per-tab entry.
"""

from __future__ import annotations

import numpy as np

from .. import helpers
from ..qt import QtWidgets


def _format_value(key: str, value) -> str:
    """Render a metadata value compactly (summarise big arrays)."""
    if isinstance(value, np.ndarray) or (
        isinstance(value, (list, tuple)) and len(value) > 6
    ):
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size:
            return f"{arr.size} values, {arr.min():g} … {arr.max():g}"
        return "(empty)"
    return str(value)


class MetadataTab(QtWidgets.QWidget):
    """Read-only metadata view plus the density control for the active dataset."""

    def __init__(self, main):
        """Build the metadata table and the density control row."""
        super().__init__()
        self.main = main

        layout = QtWidgets.QVBoxLayout(self)

        # Density control (size-resolved data only).
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
        self.apply_all.setToolTip(
            "When ticked, the density is set on every loaded dataset; otherwise "
            "only on this one."
        )
        drow.addWidget(self.apply_all)
        self.apply_btn = QtWidgets.QPushButton("Set density")
        self.apply_btn.clicked.connect(self._apply_density)
        drow.addWidget(self.apply_btn)
        drow.addStretch(1)
        layout.addWidget(self.density_row)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Field", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=1)

    @property
    def obj(self):
        """The active dataset's parent object (aerodynamic axis for a 3D APS)."""
        return getattr(self.main, "active_obj", None) or self.main.obj

    def _apply_density(self) -> None:
        """Push the chosen density onto the active dataset (or all)."""
        self.main.apply_density(
            float(self.density_spin.value()), all_datasets=self.apply_all.isChecked()
        )

    def refresh(self) -> None:
        """Repopulate the metadata table and density control for the active data."""
        obj = self.obj
        if obj is None:
            self.table.setRowCount(0)
            return

        is2d = helpers.is_2d(obj)
        self.density_row.setVisible(is2d)
        if is2d:
            self.density_spin.blockSignals(True)
            self.density_spin.setValue(float(getattr(obj, "density", 1.0)))
            self.density_spin.blockSignals(False)

        meta = dict(getattr(obj, "metadata", {}) or {})
        # A couple of derived, user-friendly rows up top.
        rows = [("class", type(obj).__name__)]
        if is2d:
            edges = np.asarray(obj.bin_edges, dtype=float)
            rows.append(
                (
                    "size range",
                    f"{len(obj.bin_mids)} bins, "
                    f"{edges.min():g} – {edges.max():g} nm",
                )
            )
        rows += [(str(k), _format_value(k, v)) for k, v in meta.items()]

        self.table.setRowCount(len(rows))
        for r, (key, val) in enumerate(rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(key))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(val))
        self.table.resizeColumnToContents(0)
