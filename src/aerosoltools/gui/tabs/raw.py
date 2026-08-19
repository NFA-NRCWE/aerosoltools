"""Raw data table tab."""

from __future__ import annotations

import os

from ..logic import helpers
from ..qt import QtWidgets
from ..view.models import PandasTableModel
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
        # Export basis (size-resolved data only): the dataset stays dN, but the
        # user can export number/mass/surface/volume.
        bar.addWidget(QtWidgets.QLabel("Export as:"))
        self.export_dtype = QtWidgets.QComboBox()
        self.export_dtype.addItems(["dN", "dM", "dS", "dV"])
        self.export_dtype.setToolTip(
            "Distribution basis for the exported size data (number/mass/surface/"
            "volume). The dataset itself stays as number (dN)."
        )
        bar.addWidget(self.export_dtype)
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
        """Show the active object's main or extra data table.

        For the main table the dtype/unit is shown and, for size-resolved (2D)
        data, the otherwise-cryptic numeric column headers are annotated as
        size-bin midpoint diameters (in nm) — both in the info line and as
        per-column header tooltips — so it is clear what the numbers mean.
        """
        if self.obj is None:
            return
        extra = self.source.currentText() == "Extra data"
        df = self.obj.extra_data if extra else self.obj.data
        self.model.set_dataframe(df)

        if df.shape[1] == 0:
            self.model.set_header_tooltips({})
            self.info.setText("   (no extra data in this file)")
            return

        parts = [f"{df.shape[0]} rows × {df.shape[1]} columns"]
        tooltips: dict = {}
        if not extra:
            dtype, unit = helpers.describe(self.obj)
            parts.append(f"values: {dtype} [{unit}]")
            if helpers.is_2d(self.obj):
                parts.append("numeric headers = size-bin midpoint Ø (nm)")
                for name in self.obj._sizebin_headers:
                    tooltips[name] = (
                        f"Size-bin midpoint diameter: {name} nm — "
                        f"values in {dtype} [{unit}]"
                    )
        self.model.set_header_tooltips(tooltips)
        self.info.setText("   " + "   |   ".join(parts))

    def _export(self) -> None:
        """Save the displayed table (keeping timestamps) to a file."""
        if self.obj is None:
            return
        which = "extra" if self.source.currentText() == "Extra data" else "data"
        if which == "extra":
            df = self.obj.extra_data
        else:
            # Convert a copy to the chosen basis for export (dataset stays dN).
            disp = self.export_dtype.currentText()
            obj = self.obj
            if disp != "dN" and helpers.is_2d(obj):
                obj = obj.copy_self()
                obj.dtype_converter(disp)
            df = obj.data
            which = disp if helpers.is_2d(self.obj) else "data"
        base = os.path.splitext(os.path.basename(self.main.source_path or "data"))[0]
        # Keep the timestamp index for raw data exports.
        _export_table(self, df, f"{base}_{which}", with_index=True)
