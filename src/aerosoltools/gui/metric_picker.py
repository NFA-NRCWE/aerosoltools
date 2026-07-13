"""Metric-picker dialog for the Summary pane.

Lists the metrics available across the ticked datasets — grouped by instrument —
from each dataset's :meth:`~aerosoltools.Aerosol1D.available_metrics` catalog, so
the user picks metrics intentionally (with each instrument's primary metrics
pre-checked) instead of typing free-text tokens that silently mis-resolve. It
doubles as the discovery window: "what can I summarise, and from which
instrument?"
"""

from __future__ import annotations

from .qt import QtCore, QtWidgets


def metric_catalog(datasets) -> list[tuple[str, list]]:
    """Return ``[(instrument, [MetricSpec, …]), …]`` across ``datasets``.

    Grouped by instrument in first-seen order; metrics de-duplicated by ``key``
    within each instrument group (so N datasets of one instrument list its
    metrics once).
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    for ds in datasets:
        instr = ds.instrument
        try:
            specs = ds.obj.available_metrics()
        except Exception:
            specs = []
        if instr not in groups:
            groups[instr] = {}
            order.append(instr)
        for m in specs:
            groups[instr].setdefault(m.key, m)
    return [(instr, list(groups[instr].values())) for instr in order]


def default_keys(datasets) -> list[str]:
    """The union of each instrument's primary (``default``) metric keys."""
    keys: list[str] = []
    for _instr, specs in metric_catalog(datasets):
        for m in specs:
            if m.default and m.key not in keys:
                keys.append(m.key)
    return keys


class MetricPickerDialog(QtWidgets.QDialog):
    """Grouped-by-instrument checklist of available metrics.

    Args:
        datasets: The ticked GUI datasets (each with ``.obj`` and ``.instrument``).
        selected_keys: Keys to pre-check; when falsy, each instrument's primary
            metrics are pre-checked instead.
        single: If True, only one metric may be selected (exposure summary).
    """

    def __init__(self, datasets, selected_keys, single: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose metrics")
        self._single = single
        self.resize(440, 500)

        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Pick one metric to summarise exposure for."
            if single
            else "Pick the metric(s) to summarise. Grouped by instrument; each "
            "metric is computed for every ticked dataset that provides it. "
            "Comparable quantities at different unit scales (e.g. ng/m³ and "
            "µg/m³) merge into one column."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Metric", "Unit"])
        self.tree.setRootIsDecorated(True)
        layout.addWidget(self.tree, stretch=1)

        preset = set(selected_keys or [])
        self._key_items: dict[str, list] = {}
        for instr, specs in metric_catalog(datasets):
            head = QtWidgets.QTreeWidgetItem(self.tree, [instr, ""])
            head.setFlags(QtCore.Qt.ItemIsEnabled)
            font = head.font(0)
            font.setBold(True)
            head.setFont(0, font)
            for m in specs:
                child = QtWidgets.QTreeWidgetItem(head, [m.label, m.unit])
                child.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                checked = (m.key in preset) if preset else bool(m.default)
                # In single mode never pre-check more than one.
                if (
                    single
                    and checked
                    and any(
                        it.checkState(0) == QtCore.Qt.Checked
                        for items in self._key_items.values()
                        for it in items
                    )
                ):
                    checked = False
                child.setCheckState(
                    0, QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
                )
                child.setData(0, QtCore.Qt.UserRole, m.key)
                self._key_items.setdefault(m.key, []).append(child)
            head.setExpanded(True)

        self.tree.resizeColumnToContents(0)
        self.tree.itemChanged.connect(self._on_item_changed)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_item_changed(self, item, col: int) -> None:
        """Enforce single-select, or sync a key that appears under >1 instrument."""
        if col != 0:
            return
        key = item.data(0, QtCore.Qt.UserRole)
        if key is None:
            return
        state = item.checkState(0)
        self.tree.blockSignals(True)
        if self._single and state == QtCore.Qt.Checked:
            for k, items in self._key_items.items():
                for it in items:
                    if it is not item:
                        it.setCheckState(0, QtCore.Qt.Unchecked)
        else:
            for it in self._key_items.get(key, []):
                if it is not item:
                    it.setCheckState(0, state)
        self.tree.blockSignals(False)

    def selected_keys(self) -> list[str]:
        """The checked metric keys, in catalog order."""
        keys = []
        for key, items in self._key_items.items():
            if any(it.checkState(0) == QtCore.Qt.Checked for it in items):
                keys.append(key)
        return keys
