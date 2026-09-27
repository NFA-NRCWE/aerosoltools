"""Browse the exposure-limit list in use and update it from Retsinformation."""

from __future__ import annotations

import contextlib
from datetime import datetime

from ...exposure_limits import (
    RetsinformationError,
    fetch_exposure_limits,
    find_current_order,
)
from ..logic import exposure_limits
from ..qt import QtCore, QtWidgets
from ..view.models import PandasTableModel


@contextlib.contextmanager
def _busy():
    """Show a wait cursor while a (blocking) network request runs."""
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


class ExposureLimitsDialog(QtWidgets.QDialog):
    """Show the occupational exposure limits in use, and fetch a newer order.

    *Check for a newer order* follows Retsinformation's record of which order
    replaced which (a revised list is a new order, so there is no fixed "latest"
    address) and offers the one in force; an order can also be fetched by its
    number. A fetched list is saved for this user and used for new picks —
    picks already made in a project keep the values and source they were made
    with. :attr:`changed` is True once a new list was saved.
    """

    def __init__(self, parent=None):
        """Build the source header, the limits table and the update controls."""
        super().__init__(parent)
        self.setWindowTitle("Occupational exposure limits")
        self.changed = False
        self._newer_eli: str | None = None

        layout = QtWidgets.QVBoxLayout(self)
        self.header = QtWidgets.QLabel("")
        self.header.setWordWrap(True)
        self.header.setTextFormat(QtCore.Qt.RichText)
        self.header.setOpenExternalLinks(True)
        layout.addWidget(self.header)

        self.model = PandasTableModel()
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, stretch=1)

        self.excluded = QtWidgets.QLabel("")
        self.excluded.setWordWrap(True)
        self.excluded.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.excluded)

        box = QtWidgets.QGroupBox("Update from Retsinformation")
        grid = QtWidgets.QGridLayout(box)
        self.check_btn = QtWidgets.QPushButton("Check for a newer order")
        self.check_btn.setToolTip(
            "Ask Retsinformation whether the order in use is still in force and, "
            "if not, which order replaced it."
        )
        self.check_btn.clicked.connect(self._check)
        self.get_newer_btn = QtWidgets.QPushButton("Download it")
        self.get_newer_btn.setEnabled(False)
        self.get_newer_btn.clicked.connect(lambda: self._download(self._newer_eli))
        grid.addWidget(self.check_btn, 0, 0)
        grid.addWidget(self.get_newer_btn, 0, 1)
        self.check_status = QtWidgets.QLabel("")
        self.check_status.setWordWrap(True)
        grid.addWidget(self.check_status, 1, 0, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Or fetch an order:"), 2, 0)
        self.eli_edit = QtWidgets.QLineEdit()
        self.eli_edit.setPlaceholderText(
            "e.g. 2026/613, an ELI link, or BEK nr 613 af 29/06/2026"
        )
        grid.addWidget(self.eli_edit, 2, 1)
        fetch_btn = QtWidgets.QPushButton("Fetch")
        fetch_btn.clicked.connect(lambda: self._download(self.eli_edit.text()))
        grid.addWidget(fetch_btn, 2, 2)
        grid.setColumnStretch(1, 1)
        layout.addWidget(box)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.resize(900, 560)
        self._show_list()

    def _show_list(self) -> None:
        """Fill the header and table from the list currently in use."""
        limits = exposure_limits.active_list()
        src = limits.source
        retrieved = src.retrieved[:10] if src.retrieved else "?"
        in_force = f", in force from {src.in_force_from}" if src.in_force_from else ""
        self.header.setText(
            f"<b>{src.label}</b>{in_force} — {src.title}<br>"
            f"{len(limits)} limits for dust (Bilag 2, Afsnit B) · status "
            f"'{src.status or '?'}' when retrieved on {retrieved} · "
            f"<a href='{src.eli}'>{src.eli}</a>"
        )
        df = limits.to_dataframe().drop(columns=["Unit"])
        df = df.rename(
            columns={
                "8-h limit": "8-h (mg/m³)",
                "Short-term limit": "Short-term (mg/m³)",
            }
        )
        self.model.set_dataframe(df)
        self.table.resizeColumnsToContents()
        self.excluded.setText(
            "Not listed: fibres, whose limits are counts per cm³ ("
            + ", ".join(limits.excluded)
            + "). Short-term rule “Jf. § 3, stk. 2” = twice the 8-hour limit."
        )

    def _check(self) -> None:
        """Follow the order in use to the one in force now."""
        current = exposure_limits.active_list().source
        try:
            with _busy():
                found = find_current_order(current.eli)
        except RetsinformationError as exc:
            self.check_status.setText(f"Could not check: {exc}")
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        if found.superseded:
            path = " → ".join(step.label for step in found.chain)
            text = f"{current.label} has been replaced: {path} (in force)."
            self._newer_eli = found.current.eli
            self.get_newer_btn.setText(f"Download {found.current.label}")
            self.get_newer_btn.setEnabled(True)
        else:
            text = f"{current.label} is still in force (checked {stamp})."
            self._newer_eli = None
            self.get_newer_btn.setEnabled(False)
        if found.amendments:
            text += (
                " Note: it has been amended by "
                + ", ".join(a.label for a in found.amendments)
                + " — those changes are not part of the parsed list."
            )
        self.check_status.setText(text)

    def _download(self, eli: str | None) -> None:
        """Fetch and parse an order, show what changes, and save it on confirmation."""
        if not eli or not eli.strip():
            return
        old = exposure_limits.active_list()
        try:
            with _busy():
                new = fetch_exposure_limits(eli)
        except (RetsinformationError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Exposure limits", str(exc))
            return
        diff = exposure_limits.compare_lists(old, new)
        summary = "\n".join(diff[:25]) + ("\n…" if len(diff) > 25 else "")
        if not diff:
            summary = "The limits are the same as in the list in use."
        status = new.source.status or "?"
        answer = QtWidgets.QMessageBox.question(
            self,
            "Use this order?",
            f"{new.source.label} (status '{status}'): {len(new)} dust limits.\n\n"
            f"Compared with {old.source.label}:\n{summary}\n\n"
            "Save it and use it for new picks? (Picks already made in a project "
            "keep their values and source.)",
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        path = exposure_limits.save_user_list(new)
        self.changed = True
        in_use = exposure_limits.active_list().source
        note = f"Saved to {path}."
        if in_use.label != new.source.label:
            note += f" {in_use.label} is newer, so it stays in use."
        self.check_status.setText(note)
        self.get_newer_btn.setEnabled(False)
        self._show_list()
