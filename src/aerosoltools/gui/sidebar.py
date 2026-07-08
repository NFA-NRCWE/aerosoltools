"""Left-hand dataset sidebar for the multi-dataset GUI.

A thin presentation widget: it shows the project's datasets, lets the user pick
the active one, and exposes add / remove / rename actions via Qt signals. All
state lives in :class:`~aerosoltools.gui.project.Project`; this widget only
renders it and reports user intent back to the :class:`MainWindow`.
"""

from __future__ import annotations

from typing import List, Optional

from .qt import QtCore, QtWidgets

_ID_ROLE = QtCore.Qt.UserRole


class DatasetSidebar(QtWidgets.QWidget):
    """List of datasets with selection + add/remove/rename controls."""

    dataset_selected = QtCore.pyqtSignal(int)
    add_requested = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal(int)
    rename_requested = QtCore.pyqtSignal(int)
    reload_requested = QtCore.pyqtSignal(int)
    join_requested = QtCore.pyqtSignal(int)
    combine_ns_ops_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        """Build the add / list / rename / remove / combine controls."""
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.add_btn = QtWidgets.QPushButton("Add file…")
        self.add_btn.setObjectName("primary")
        self.add_btn.setToolTip(
            "Import one or more data files as new datasets (Ctrl+O)."
        )
        self.add_btn.clicked.connect(self.add_requested.emit)
        layout.addWidget(self.add_btn)

        self.list = QtWidgets.QListWidget()
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(False)
        self.list.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.list, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self.rename_btn = QtWidgets.QPushButton("Rename")
        self.rename_btn.setToolTip("Rename the selected dataset.")
        self.rename_btn.clicked.connect(self._emit_rename)
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setToolTip("Remove the selected dataset from the project.")
        self.remove_btn.clicked.connect(self._emit_remove)
        self.reload_btn = QtWidgets.QPushButton("Reload")
        self.reload_btn.setToolTip(
            "Reload the selected dataset from its source file (discards "
            "conversions, cropping and activities on it)."
        )
        self.reload_btn.clicked.connect(self._emit_reload)
        row.addWidget(self.rename_btn)
        row.addWidget(self.remove_btn)
        row.addWidget(self.reload_btn)
        layout.addLayout(row)

        # Combine actions.
        self.join_btn = QtWidgets.QPushButton("Join same instrument")
        self.join_btn.setToolTip(
            "Merge all datasets that share the selected dataset's instrument and "
            "serial number into one continuous time series."
        )
        self.join_btn.clicked.connect(self._emit_join)
        layout.addWidget(self.join_btn)

        self.combine_btn = QtWidgets.QPushButton("Combine size ranges…")
        self.combine_btn.setToolTip(
            "Stitch two range-extending size instruments (e.g. NanoScan/FMPS + "
            "OPS/APS) into one distribution at a crossover diameter you choose."
        )
        self.combine_btn.clicked.connect(self.combine_ns_ops_requested.emit)
        layout.addWidget(self.combine_btn)

        self._update_buttons()

    # -- rendering ---------------------------------------------------------
    def set_datasets(self, datasets: List, active_id: Optional[int]) -> None:
        """Rebuild the list to mirror the project (without re-emitting signals)."""
        self.list.blockSignals(True)
        self.list.clear()
        for ds in datasets:
            item = QtWidgets.QListWidgetItem(self._describe(ds))
            item.setData(_ID_ROLE, ds.id)
            self.list.addItem(item)
            if ds.id == active_id:
                item.setSelected(True)
                self.list.setCurrentItem(item)
        self.list.blockSignals(False)
        self._update_buttons()

    @staticmethod
    def _describe(ds) -> str:
        """Return the multi-line list label for a dataset (name, instrument, span)."""
        span = ds.time_span()
        if span is not None:
            start, end = span
            if start.normalize() == end.normalize():
                when = f"{start:%Y-%m-%d %H:%M}–{end:%H:%M}"
            else:
                when = f"{start:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M}"
        else:
            when = "no data"
        return f"{ds.label}\n{ds.instrument}  ·  {ds.n_points()} pts\n{when}"

    def _selected_id(self) -> Optional[int]:
        """Return the id of the selected dataset, or None."""
        item = self.list.currentItem()
        return None if item is None else int(item.data(_ID_ROLE))

    def _update_buttons(self) -> None:
        """Enable or disable the action buttons for the current selection."""
        has_sel = self.list.currentItem() is not None
        self.rename_btn.setEnabled(has_sel)
        self.remove_btn.setEnabled(has_sel)
        self.reload_btn.setEnabled(has_sel)
        self.join_btn.setEnabled(has_sel)
        self.combine_btn.setEnabled(self.list.count() >= 2)

    # -- signals -----------------------------------------------------------
    def _on_select(self) -> None:
        """Emit :attr:`dataset_selected` for the newly selected dataset."""
        self._update_buttons()
        ds_id = self._selected_id()
        if ds_id is not None:
            self.dataset_selected.emit(ds_id)

    def _emit_remove(self) -> None:
        """Emit :attr:`remove_requested` for the selected dataset."""
        ds_id = self._selected_id()
        if ds_id is not None:
            self.remove_requested.emit(ds_id)

    def _emit_rename(self) -> None:
        """Emit :attr:`rename_requested` for the selected dataset."""
        ds_id = self._selected_id()
        if ds_id is not None:
            self.rename_requested.emit(ds_id)

    def _emit_reload(self) -> None:
        """Emit :attr:`reload_requested` for the selected dataset."""
        ds_id = self._selected_id()
        if ds_id is not None:
            self.reload_requested.emit(ds_id)

    def _emit_join(self) -> None:
        """Emit :attr:`join_requested` for the selected dataset."""
        ds_id = self._selected_id()
        if ds_id is not None:
            self.join_requested.emit(ds_id)
