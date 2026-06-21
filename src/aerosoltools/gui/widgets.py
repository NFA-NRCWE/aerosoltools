"""Small reusable widgets and dialogs for the aerosoltools GUI.

These are presentation-only helpers with no business logic: a tab bar that sizes
its tabs so labels never clip, the modal dialog used to pick the two datasets for
a NanoScan + OPS combine, and the read-only keyboard-shortcut reference.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from .qt import QtWidgets


class SlackTabBar(QtWidgets.QTabBar):
    """Tab bar that pads each tab's width hint so labels never clip.

    Qt's default size hint for a stylesheet-padded tab can under-allocate
    width, clipping the first/last characters of the label. Adding a fixed
    slack to the hint guarantees the full text is shown.
    """

    def tabSizeHint(self, index):  # noqa: N802
        """Return the default size hint widened so tab labels never clip."""
        size = super().tabSizeHint(index)
        size.setWidth(size.width() + 28)
        return size


class CombineNSOPSDialog(QtWidgets.QDialog):
    """Pick a NanoScan and an OPS dataset (+ time-match) to combine."""

    def __init__(self, parent, datasets):
        """Build the NS/OPS dataset pickers and match selector.

        Args:
            parent: Parent widget.
            datasets: Candidate (2D) datasets to choose from.
        """
        super().__init__(parent)
        self.setWindowTitle("Combine NS + OPS")
        self._datasets = datasets

        form = QtWidgets.QFormLayout(self)
        self.ns_combo = QtWidgets.QComboBox()
        self.ops_combo = QtWidgets.QComboBox()
        for d in datasets:
            label = f"{d.label}  ({d.instrument})"
            self.ns_combo.addItem(label, d.id)
            self.ops_combo.addItem(label, d.id)
        self._preselect(self.ns_combo, ("nano", "ns"))
        self._preselect(self.ops_combo, ("ops",))

        self.match_combo = QtWidgets.QComboBox()
        self.match_combo.addItems(["rebin", "nearest", "exact"])

        form.addRow("NanoScan (smaller sizes):", self.ns_combo)
        form.addRow("OPS (larger sizes):", self.ops_combo)
        form.addRow("Time match:", self.match_combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _preselect(self, combo, keys) -> None:
        """Preselect the first dataset whose instrument matches ``keys``."""
        for i in range(combo.count()):
            ds_id = combo.itemData(i)
            d = next(x for x in self._datasets if x.id == ds_id)
            text = f"{d.instrument} {d.instrument_key}".lower()
            if any(k in text for k in keys):
                combo.setCurrentIndex(i)
                return

    def result(self):
        """Return the chosen ``(ns_dataset, ops_dataset, match)`` tuple."""
        ns = next(d for d in self._datasets if d.id == self.ns_combo.currentData())
        ops = next(d for d in self._datasets if d.id == self.ops_combo.currentData())
        return ns, ops, self.match_combo.currentText()


class KeyboardShortcutsDialog(QtWidgets.QDialog):
    """Read-only reference listing the application's keyboard shortcuts.

    The window passes the live ``(keys, description)`` pairs it collected while
    building the menu bar, so this dialog always mirrors the real bindings rather
    than a hand-maintained copy.
    """

    def __init__(self, parent, shortcuts: Iterable[Tuple[str, str]]):
        """Build the shortcuts table from ``(keys, description)`` pairs."""
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Available keyboard shortcuts:"))

        # A two-column, non-editable table: keys on the left, action on the right.
        rows = list(shortcuts)
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for r, (keys, description) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(keys))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(description))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.resize(420, 320)
