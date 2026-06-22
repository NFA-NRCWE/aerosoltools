"""Raw data table tab."""

from __future__ import annotations

import os

from ..models import PandasTableModel
from ..qt import QtWidgets
from ._base import _export_table, _tune_table


class RawDataTab(QtWidgets.QWidget):
    """Tabular view of the object's main data (and optionally extra data)."""

    def __init__(self, main):
        """Build the main/extra-data selector, info label and table."""
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
        self.export_btn.setToolTip(
            "Save the displayed table (with timestamps) to an .xlsx or .csv file."
        )
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
        """Active aerosol object (proxied from the main window)."""
        return self.main.obj

    def refresh(self) -> None:
        """Show the active object's main or extra data table."""
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
        """Save the displayed table (keeping timestamps) to a file."""
        if self.obj is None:
            return
        which = "extra" if self.source.currentText() == "Extra data" else "data"
        df = self.obj.extra_data if which == "extra" else self.obj.data
        base = os.path.splitext(os.path.basename(self.main.source_path or "data"))[0]
        # Keep the timestamp index for raw data exports.
        _export_table(self, df, f"{base}_{which}", with_index=True)
