"""Dropdown for the substance being measured, showing its exposure limits.

The popup is a table — substance, CAS, 8-hour and short-term limits (mg/m³) and
the order's remarks — while the closed combo shows only the name; typing part of
a name filters the list. Also holds the helpers that turn a pick into the
self-contained record a project stores (the limit, its fraction variants and the
order it came from), so a saved comparison survives a later list unchanged.
"""

from __future__ import annotations

from ...exposure_limits import (
    REMARK_CODES,
    ExposureLimit,
    ExposureLimitList,
    basis_note,
)
from ..qt import QtCore, QtGui, QtWidgets

_FRACTION_TEXT = {
    "respirable": "respirable fraction",
    "inhalable": "inhalable fraction",
    "thoracic": "thoracic fraction",
    "total": "total dust (no fraction named)",
}


def format_value(value: float | None) -> str:
    """Compact number for a limit field (``""`` for ``None``)."""
    return "" if value is None else f"{value:.6g}"


def stel_text(limit: ExposureLimit) -> str:
    """The short-term limit as listed in the picker, e.g. ``"0.2 (2×)"``."""
    if limit.stel is None:
        return "–"
    suffix = " (ceiling)" if limit.ceiling else " (2×)" if limit.stel_derived else ""
    return format_value(limit.stel) + suffix


def short_term_name(limit: ExposureLimit) -> str:
    """What the short-term limit is, e.g. ``"short-term (15 min)"``/``"ceiling"``."""
    return "ceiling" if limit.ceiling else f"short-term ({limit.stel_minutes} min)"


def describe_limit(limit: ExposureLimit, source_label: str) -> str:
    """Multi-line tooltip text explaining one limit and where it comes from."""
    lines = [limit.name]
    if limit.cas:
        lines.append("CAS: " + ", ".join(limit.cas))
    if limit.twa is not None:
        lines.append(f"8-hour limit: {format_value(limit.twa)} {limit.unit}")
    else:
        lines.append("No 8-hour limit.")
    if limit.stel is not None:
        short = (
            f"{short_term_name(limit).capitalize()} limit: "
            f"{format_value(limit.stel)} {limit.unit}"
        )
        if limit.stel_derived:
            short += f" — twice the 8-hour limit ({limit.stel_rule})"
        lines.append(short)
    else:
        lines.append("No short-term limit.")
    lines.append("Applies to: " + _FRACTION_TEXT.get(limit.fraction, limit.fraction))
    note = basis_note(limit)
    if note:
        lines.append(note)
    for code in limit.remark_codes:
        lines.append(f"{code}: {REMARK_CODES.get(code, '')}")
    lines.extend(limit.notes)
    lines.append(f"Source: {source_label}, Bilag 2, Afsnit {limit.section}")
    return "\n".join(lines)


def pick_record(limit: ExposureLimit, limits: ExposureLimitList) -> dict:
    """A self-contained, JSON-safe record of a picked substance and its source.

    Holds the limit, its fraction variants (so the one that fits a metric can
    be chosen later without the list) and the order it came from.
    """
    return {
        "substance": limit.name,
        "limit": limit.to_dict(),
        "variants": [
            v.to_dict() for v in limits.siblings(limit) if v.name != limit.name
        ],
        "source": limits.source.label,
        "source_short": limits.source.short_label,
        "eli": limits.source.eli,
    }


def record_limit(record: dict) -> ExposureLimit:
    """Rebuild the :class:`ExposureLimit` a :func:`pick_record` describes."""
    if isinstance(record.get("limit"), dict):
        return ExposureLimit.from_dict(record["limit"])
    # Records written before the limit was stored whole.
    return ExposureLimit(
        name=record.get("substance", ""),
        cas=tuple(record.get("cas") or ()),
        twa=record.get("twa"),
        stel=record.get("stel"),
        unit=record.get("unit") or "mg/m³",
        stel_rule=record.get("stel_rule") or "",
        remarks=record.get("remarks") or "",
    )


def record_candidates(record: dict) -> list[ExposureLimit]:
    """The picked limit, then its fraction variants (for ``applicable_limit``)."""
    variants = [ExposureLimit.from_dict(v) for v in record.get("variants") or ()]
    return [record_limit(record), *variants]


class ExposureLimitCombo(QtWidgets.QComboBox):
    """Pick a substance from an exposure-limit list (or "none").

    The popup lists every substance with its CAS numbers, 8-hour and short-term
    limits and remarks; a short-term value marked "(2×)" is twice the 8-hour
    limit, as the order prescribes where it gives no separate value. Type part
    of a name to filter; hover a row for the full explanation. :attr:`picked`
    fires on a user choice only.
    """

    #: Emitted when the user chooses an entry (not on programmatic changes).
    picked = QtCore.pyqtSignal()

    NONE_TEXT = "— none: type the limits —"
    COLUMNS = ("Substance", "CAS", "8-h (mg/m³)", "Short-term (mg/m³)", "Remarks")

    def __init__(self, parent=None):
        """Build the combo with a multi-column table popup and name search."""
        super().__init__(parent)
        self._limits: ExposureLimitList | None = None
        self._rows: list[ExposureLimit | None] = [None]
        model = QtGui.QStandardItemModel(0, len(self.COLUMNS), self)
        model.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.setModel(model)
        view = QtWidgets.QTreeView(self)
        view.setRootIsDecorated(False)
        view.setUniformRowHeights(True)
        view.setAllColumnsShowFocus(True)
        view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setView(view)
        self.setModelColumn(0)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLength)
        self.setMinimumContentsLength(26)
        # Type part of a name to find it among ~100 substances.
        self.setEditable(True)
        self.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.lineEdit().setPlaceholderText("type to search…")
        completer = self.completer()
        completer.setFilterMode(QtCore.Qt.MatchContains)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        completer.activated[str].connect(self._on_completed)
        self.lineEdit().editingFinished.connect(self._restore_text)
        self.activated.connect(lambda _index: self.picked.emit())

    def limits(self) -> ExposureLimitList | None:
        """The list currently offered."""
        return self._limits

    def set_limits(self, limits: ExposureLimitList) -> None:
        """Offer ``limits``, keeping the current substance selected if still listed."""
        current = self.current_limit()
        self._limits = limits
        model = self.model()
        self.blockSignals(True)
        model.removeRows(0, model.rowCount())
        none_item = QtGui.QStandardItem(self.NONE_TEXT)
        none_item.setToolTip("Leave the limit fields as typed.")
        model.appendRow(
            [none_item] + [QtGui.QStandardItem("") for _ in self.COLUMNS[1:]]
        )
        self._rows = [None]
        for lim in limits:
            cells = (
                lim.name,
                ", ".join(lim.cas),
                format_value(lim.twa) or "–",
                stel_text(lim),
                lim.remarks,
            )
            tip = describe_limit(lim, limits.source.label)
            items = []
            for text in cells:
                item = QtGui.QStandardItem(text)
                item.setToolTip(tip)
                item.setEditable(False)
                items.append(item)
            model.appendRow(items)
            self._rows.append(lim)
        self.blockSignals(False)
        self.set_current_name(current.name if current else None)
        self.setToolTip(
            f"Particulate substances with a limit in {limits.source.label} "
            f"(Bilag 2, Afsnit {', '.join(limits.source.sections)}; fibres "
            "excluded). Type to search. (2×) = short-term limit set to twice the "
            "8-hour limit by the order."
        )

    def current_limit(self) -> ExposureLimit | None:
        """The selected substance's limits, or ``None`` for the "none" entry."""
        index = self.currentIndex()
        return self._rows[index] if 0 <= index < len(self._rows) else None

    def set_current_name(self, name: str | None) -> None:
        """Select a substance by name (or "none") without emitting :attr:`picked`."""
        index = 0
        if name:
            for i, lim in enumerate(self._rows):
                if lim is not None and lim.name == name:
                    index = i
                    break
        self.blockSignals(True)
        self.setCurrentIndex(index)
        self.setEditText(self.itemText(index))
        self.blockSignals(False)

    def _on_completed(self, text: str) -> None:
        """A search match was chosen: select it as a user pick."""
        index = self.findText(text, QtCore.Qt.MatchExactly)
        if index >= 0:
            self.setCurrentIndex(index)
            self.picked.emit()

    def _restore_text(self) -> None:
        """Show the selected entry again if the search text matched nothing."""
        if self.lineEdit().text() != self.itemText(self.currentIndex()):
            self.setEditText(self.itemText(self.currentIndex()))

    def showPopup(self):  # noqa: N802 (Qt override)
        """Size the table popup to its columns before opening it."""
        view = self.view()
        width = 0
        for col in range(len(self.COLUMNS)):
            view.resizeColumnToContents(col)
            width += view.columnWidth(col)
        view.setMinimumWidth(width + view.verticalScrollBar().sizeHint().width() + 8)
        super().showPopup()
