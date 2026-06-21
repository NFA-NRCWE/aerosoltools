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
    join_requested = QtCore.pyqtSignal(int)
    combine_ns_ops_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.add_btn = QtWidgets.QPushButton("Add file…")
        self.add_btn.setObjectName("primary")
        self.add_btn.clicked.connect(self.add_requested.emit)
        layout.addWidget(self.add_btn)

        self.list = QtWidgets.QListWidget()
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(False)
        self.list.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.list, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self.rename_btn = QtWidgets.QPushButton("Rename")
        self.rename_btn.clicked.connect(self._emit_rename)
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.clicked.connect(self._emit_remove)
        row.addWidget(self.rename_btn)
        row.addWidget(self.remove_btn)
        layout.addLayout(row)

        # Combine actions.
        self.join_btn = QtWidgets.QPushButton("Join same instrument")
        self.join_btn.setToolTip(
            "Merge all datasets that share the selected dataset's instrument and "
            "serial number into one continuous time series."
        )
        self.join_btn.clicked.connect(self._emit_join)
        layout.addWidget(self.join_btn)

        self.combine_btn = QtWidgets.QPushButton("Combine NS + OPS…")
        self.combine_btn.setToolTip(
            "Combine a NanoScan and an OPS dataset into one merged size "
            "distribution."
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
        item = self.list.currentItem()
        return None if item is None else int(item.data(_ID_ROLE))

    def _update_buttons(self) -> None:
        has_sel = self.list.currentItem() is not None
        self.rename_btn.setEnabled(has_sel)
        self.remove_btn.setEnabled(has_sel)
        self.join_btn.setEnabled(has_sel)
        self.combine_btn.setEnabled(self.list.count() >= 2)

    # -- signals -----------------------------------------------------------
    def _on_select(self) -> None:
        self._update_buttons()
        ds_id = self._selected_id()
        if ds_id is not None:
            self.dataset_selected.emit(ds_id)

    def _emit_remove(self) -> None:
        ds_id = self._selected_id()
        if ds_id is not None:
            self.remove_requested.emit(ds_id)

    def _emit_rename(self) -> None:
        ds_id = self._selected_id()
        if ds_id is not None:
            self.rename_requested.emit(ds_id)

    def _emit_join(self) -> None:
        ds_id = self._selected_id()
        if ds_id is not None:
            self.join_requested.emit(ds_id)
