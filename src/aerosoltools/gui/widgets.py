"""Small reusable widgets and dialogs for the aerosoltools GUI.

These are presentation-only helpers with no business logic: a tab bar that sizes
its tabs so labels never clip, the modal dialog used to pick the two datasets for
a NanoScan + OPS combine, and the read-only keyboard-shortcut reference.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from .qt import QtCore, QtWidgets


class ThresholdControls(QtWidgets.QWidget):
    """Inline controls for overlaying a concentration threshold (e.g. an OEL).

    A check-box switches a horizontal limit line on/off; the value field sets
    where it sits (in the plot's *current* y-units) and the label field sets its
    legend text. :attr:`changed` fires whenever any of the three is edited, so
    the owning tab can persist the state and redraw. Kept presentation-only —
    the actual line is drawn by :func:`helpers.draw_threshold`.
    """

    #: Emitted when the enable box, value or label changes.
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        """Build the enable check-box plus value and legend-text fields."""
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.enable = QtWidgets.QCheckBox("Threshold")
        self.enable.setToolTip(
            "Overlay a horizontal limit line — e.g. an occupational exposure "
            "limit (OEL) — so it is clear at which times the concentration rose "
            "above it."
        )
        self.value = QtWidgets.QLineEdit()
        self.value.setPlaceholderText("value")
        self.value.setFixedWidth(70)
        self.value.setToolTip(
            "Threshold level, in the units currently shown on the y-axis."
        )
        self.label = QtWidgets.QLineEdit()
        self.label.setPlaceholderText("legend, e.g. OEL")
        self.label.setFixedWidth(120)
        self.label.setToolTip("Legend text shown next to the threshold line.")

        lay.addWidget(self.enable)
        lay.addWidget(self.value)
        lay.addWidget(self.label)

        self.enable.stateChanged.connect(lambda _state: self.changed.emit())
        self.value.editingFinished.connect(self.changed.emit)
        self.label.editingFinished.connect(self.changed.emit)

    def state(self) -> dict:
        """Return the current ``{"on", "value", "label"}`` state (JSON-safe)."""
        return {
            "on": self.enable.isChecked(),
            "value": self.value.text().strip(),
            "label": self.label.text().strip(),
        }

    def set_state(self, state: Optional[dict]) -> None:
        """Restore a previously saved state without emitting :attr:`changed`."""
        if not state:
            return
        widgets = (self.enable, self.value, self.label)
        for w in widgets:
            w.blockSignals(True)
        self.enable.setChecked(bool(state.get("on")))
        self.value.setText(str(state.get("value", "")))
        self.label.setText(str(state.get("label", "")))
        for w in widgets:
            w.blockSignals(False)

    def threshold_value(self) -> Optional[float]:
        """The float threshold when enabled and parseable, else ``None``."""
        if not self.enable.isChecked():
            return None
        try:
            return float(self.value.text().strip())
        except ValueError:
            return None

    def legend_text(self) -> str:
        """The user's legend text (may be empty)."""
        return self.label.text().strip()


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
