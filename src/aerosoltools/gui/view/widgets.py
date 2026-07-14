"""Small reusable widgets and dialogs for the aerosoltools GUI.

These are presentation-only helpers with no business logic: a tab bar that sizes
its tabs so labels never clip, the modal dialog used to pick two size-resolved
datasets and a crossover for combining their ranges, the checklist dialog for
joining same-instrument datasets into one recording, and the read-only
keyboard-shortcut reference.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

from ..qt import Figure, FigureCanvas, QtCore, QtWidgets


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


class WheelLineEdit(QtWidgets.QLineEdit):
    """A line edit that emits a step signal on mouse-wheel while focused.

    Used for the Overlay time-shift field: once the field has keyboard focus,
    scrolling nudges the value up/down for a quick, smooth adjustment (Ctrl for a
    coarse step). The widget stays value-agnostic — it emits
    ``stepped(direction, coarse)`` and lets the owner parse/format the value — so
    it can drive any stepped text field. Requiring focus first avoids hijacking
    wheel scrolling of the surrounding table.
    """

    #: Emitted on a focused wheel tick: ``(+1|-1 direction, coarse?)``.
    stepped = QtCore.pyqtSignal(int, bool)

    def wheelEvent(self, event):  # noqa: N802 (Qt override)
        """Turn a wheel tick into a :attr:`stepped` signal when focused."""
        if not self.hasFocus():
            event.ignore()
            return
        dy = event.angleDelta().y()
        if dy == 0:
            event.ignore()
            return
        coarse = bool(event.modifiers() & QtCore.Qt.ControlModifier)
        self.stepped.emit(1 if dy > 0 else -1, coarse)
        event.accept()


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


class TwoRowTabs(QtWidgets.QWidget):
    """A tab widget with two rows of tabs over one shared content area.

    The two :class:`SlackTabBar` rows share a single :class:`QStackedWidget`, so
    every pane still fills the whole window but all tabs fit without the scroll
    arrows. Dataset-specific panes go on the top row, project/comparison panes on
    the bottom row. Provides the small subset of the ``QTabWidget`` API the main
    window uses (:meth:`add_tab`, :meth:`clear`).
    """

    def __init__(self, parent=None):
        """Build the two tab bars and the shared stacked content area."""
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.bars = (SlackTabBar(), SlackTabBar())
        for row, bar in enumerate(self.bars):
            bar.setExpanding(False)
            bar.setDrawBase(False)
            bar.setUsesScrollButtons(True)
            bar.setProperty("inactiveRow", False)
            # tabBarClicked (not currentChanged) so re-clicking the row's forced
            # selection still switches back to it.
            bar.tabBarClicked.connect(lambda i, r=row: self._activate(r, i))
            v.addWidget(bar)
        self.stack = QtWidgets.QStackedWidget()
        v.addWidget(self.stack, 1)
        self._stack_index: list[list[int]] = [[], []]

    def currentWidget(self):  # noqa: N802 (mirrors QTabWidget)
        """Return the currently shown pane (or None)."""
        return self.stack.currentWidget()

    def current_text(self):
        """Title of the currently shown tab (across both rows), or None."""
        cur = self.stack.currentIndex()
        for row in (0, 1):
            for i, si in enumerate(self._stack_index[row]):
                if si == cur:
                    return self.bars[row].tabText(i)
        return None

    def select_text(self, text: str) -> bool:
        """Activate the first tab whose title matches ``text``; True if found."""
        for row in (0, 1):
            for i in range(self.bars[row].count()):
                if self.bars[row].tabText(i) == text:
                    self._activate(row, i)
                    return True
        return False

    def add_tab(self, widget, text: str, row: int) -> None:
        """Add ``widget`` as a pane, with a tab labelled ``text`` on ``row``."""
        si = self.stack.addWidget(widget)
        bar = self.bars[row]
        bar.blockSignals(True)
        bar.addTab(text)
        bar.blockSignals(False)
        self._stack_index[row].append(si)

    def finalize(self) -> None:
        """Select the first tab (preferring the top row) after (re)building."""
        for row in (0, 1):
            if self.bars[row].count():
                self._activate(row, 0)
                return

    def clear(self) -> None:
        """Remove every tab and pane."""
        for bar in self.bars:
            bar.blockSignals(True)
            while bar.count():
                bar.removeTab(0)
            bar.blockSignals(False)
        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.setParent(None)
        self._stack_index = [[], []]

    def _activate(self, row: int, i: int) -> None:
        """Show the pane for tab ``i`` of ``row`` and mark that row active."""
        if i < 0 or i >= self.bars[row].count():
            return
        bar = self.bars[row]
        bar.blockSignals(True)
        bar.setCurrentIndex(i)
        bar.blockSignals(False)
        self.stack.setCurrentIndex(self._stack_index[row][i])
        # Only the active row draws its tab as selected (QSS: inactiveRow).
        for r, b in enumerate(self.bars):
            b.setProperty("inactiveRow", r != row)
            b.style().unpolish(b)
            b.style().polish(b)


class CombineInstrumentsDialog(QtWidgets.QDialog):
    """Pick two size-resolved datasets and a crossover to stitch their ranges.

    Shows each instrument's covered size range on a shared log axis (nm) and a
    draggable crossover line that snaps to whole bin edges, so the user sets
    exactly where one instrument hands over to the other. Returns the two
    datasets, the crossover diameter (nm) and the time-match mode; the combining
    itself is done by :func:`aerosoltools.combine_size_ranges`.
    """

    def __init__(self, parent, datasets):
        """Build the dataset pickers, range plot and crossover control.

        Args:
            parent: Parent widget.
            datasets: Candidate (2D) datasets to choose from.
        """
        super().__init__(parent)
        self.setWindowTitle("Combine size ranges")
        self._datasets = datasets
        self._crossover = None  # nm
        self._edges = np.array([])  # snap targets (overlap bin edges)
        self._dragging = False

        layout = QtWidgets.QVBoxLayout(self)

        pick = QtWidgets.QFormLayout()
        self.a_combo = QtWidgets.QComboBox()
        self.b_combo = QtWidgets.QComboBox()
        for d in datasets:
            label = f"{d.label}  ({d.instrument})"
            self.a_combo.addItem(label, d.id)
            self.b_combo.addItem(label, d.id)
        self._preselect(self.a_combo, ("nano", "ns", "fmps", "smps"))
        self._preselect(self.b_combo, ("ops", "aps"))
        if (
            self.b_combo.currentIndex() == self.a_combo.currentIndex()
            and len(datasets) > 1
        ):
            self.b_combo.setCurrentIndex(
                1 - self.a_combo.currentIndex()
                if self.a_combo.currentIndex() < 2
                else 0
            )
        self.a_combo.currentIndexChanged.connect(self._redraw)
        self.b_combo.currentIndexChanged.connect(self._redraw)
        pick.addRow("Instrument 1:", self.a_combo)
        pick.addRow("Instrument 2:", self.b_combo)
        self.match_combo = QtWidgets.QComboBox()
        self.match_combo.addItems(["rebin", "nearest", "exact"])
        pick.addRow("Time match:", self.match_combo)
        layout.addLayout(pick)

        self.figure = Figure(figsize=(6.5, 2.4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, stretch=1)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.resize(560, 380)
        self._redraw()

    def _preselect(self, combo, keys) -> None:
        """Preselect the first dataset whose instrument matches ``keys``."""
        for i in range(combo.count()):
            ds_id = combo.itemData(i)
            d = next(x for x in self._datasets if x.id == ds_id)
            text = f"{d.instrument} {d.instrument_key}".lower()
            if any(k in text for k in keys):
                combo.setCurrentIndex(i)
                return

    def _selected(self):
        """Return the two currently chosen datasets."""
        a = next(d for d in self._datasets if d.id == self.a_combo.currentData())
        b = next(d for d in self._datasets if d.id == self.b_combo.currentData())
        return a, b

    def _redraw(self, *_a) -> None:
        """Redraw the range bars + crossover for the current selection."""
        a, b = self._selected()
        ok = a is not b
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(ok)
        ax = self.ax
        ax.clear()
        if not ok:
            self.info.setText("Pick two different datasets.")
            self._edges = np.array([])
            self.canvas.draw_idle()
            return

        ea = np.asarray(a.obj.bin_edges, dtype=float)
        eb = np.asarray(b.obj.bin_edges, dtype=float)
        lo_end, hi_end = min(ea[0], eb[0]), max(ea[-1], eb[-1])
        overlap_lo = max(ea[0], eb[0])
        overlap_hi = min(ea[-1], eb[-1])

        for y, e, ds, color in ((1, ea, a, "#4c72b0"), (0, eb, b, "#dd8452")):
            ax.plot(
                [e[0], e[-1]],
                [y, y],
                "-",
                color=color,
                lw=8,
                alpha=0.35,
                solid_capstyle="butt",
            )
            ax.plot(e, np.full_like(e, y), "|", color=color, ms=10, mew=1.0)
            ax.text(
                e[0],
                y + 0.18,
                f"{ds.instrument}",
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
            )

        # Snap targets: bin edges of both instruments inside the overlap.
        edges = np.unique(np.concatenate([ea, eb]))
        self._edges = edges[(edges >= overlap_lo) & (edges <= overlap_hi)]
        if self._crossover is None or not (overlap_lo <= self._crossover <= overlap_hi):
            self._crossover = float(np.sqrt(overlap_lo * overlap_hi))
            self._snap()

        if overlap_hi <= overlap_lo:
            self.info.setText(
                "These instruments do not overlap in size — they cannot be "
                "stitched at a crossover."
            )
            self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        else:
            ax.axvspan(overlap_lo, overlap_hi, color="0.5", alpha=0.08)
        self._xline = ax.axvline(self._crossover, color="crimson", lw=1.6)

        ax.set_xscale("log")
        ax.set_xlim(max(1.0, lo_end * 0.7), hi_end * 1.3)
        ax.set_ylim(-0.6, 1.7)
        ax.set_yticks([])
        ax.set_xlabel("Diameter (nm)")
        ax.grid(True, which="both", axis="x", alpha=0.25)
        self.figure.tight_layout()
        self._update_info()
        self.canvas.draw_idle()

    def _snap(self) -> None:
        """Snap the crossover to the nearest overlap bin edge."""
        if self._edges.size:
            self._crossover = float(
                self._edges[int(np.argmin(np.abs(self._edges - self._crossover)))]
            )

    def _update_info(self) -> None:
        """Refresh the textual summary under the plot."""
        a, b = self._selected()
        lo = a if float(a.obj.bin_edges[-1]) <= float(b.obj.bin_edges[-1]) else b
        hi = b if lo is a else a
        self.info.setText(
            f"Crossover {self._crossover:g} nm — "
            f"{lo.instrument} below, {hi.instrument} above. "
            "Drag the red line (snaps to bin edges)."
        )

    # -- crossover dragging ------------------------------------------------
    def _on_press(self, event) -> None:
        if event.inaxes is self.ax and event.xdata is not None and self._edges.size:
            self._dragging = True
            self._crossover = float(event.xdata)
            self._snap()
            self._xline.set_xdata([self._crossover, self._crossover])
            self._update_info()
            self.canvas.draw_idle()

    def _on_motion(self, event) -> None:
        if self._dragging and event.inaxes is self.ax and event.xdata is not None:
            self._crossover = float(event.xdata)
            self._snap()
            self._xline.set_xdata([self._crossover, self._crossover])
            self._update_info()
            self.canvas.draw_idle()

    def _on_release(self, _event) -> None:
        self._dragging = False

    def result(self):
        """Return ``(dataset_a, dataset_b, crossover_nm, match)``."""
        a, b = self._selected()
        return a, b, self._crossover, self.match_combo.currentText()


# Back-compat alias for the previous name.
CombineNSOPSDialog = CombineInstrumentsDialog


class JoinDatasetsDialog(QtWidgets.QDialog):
    """Pick which same-instrument datasets to concatenate into one recording.

    Only datasets from the *same instrument type* as the one the user clicked are
    offered — joining across different instruments would fail, so that is a hard
    block enforced by the caller. Within that type, every dataset gets a
    check-box; the ones whose serial number matches the clicked dataset are
    pre-checked and tagged "suggested", but the user is free to tick others
    (e.g. two units of the same OPS model) or untick a suggested one. At least
    two must stay ticked to enable *Join*.
    """

    def __init__(self, parent, datasets, seed):
        """Build the checklist of candidate datasets.

        Args:
            parent: Parent widget.
            datasets: Candidate datasets — all sharing ``seed``'s instrument.
            seed: The dataset the user invoked the join from; its serial number
                drives which candidates are pre-checked.
        """
        super().__init__(parent)
        self.setWindowTitle("Join datasets")
        self._datasets = list(datasets)
        self._checks: list[QtWidgets.QCheckBox] = []

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            f"Concatenate '{seed.instrument}' datasets into one continuous "
            "recording. Datasets that share the same serial number are "
            "suggested (pre-ticked); tick or untick to change the selection. "
            "The chosen datasets are replaced by the combined one."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        seed_serial = str(seed.serial_number)
        box = QtWidgets.QGroupBox("Datasets to join")
        vb = QtWidgets.QVBoxLayout(box)
        for d in self._datasets:
            serial = str(d.serial_number)
            suggested = serial == seed_serial
            tag = "  — suggested (same serial)" if suggested else ""
            chk = QtWidgets.QCheckBox(f"{d.label}   [serial {serial}]{tag}")
            chk.setChecked(suggested)
            chk.setProperty("ds_id", d.id)
            chk.toggled.connect(self._sync_ok)
            self._checks.append(chk)
            vb.addWidget(chk)
        vb.addStretch(1)
        layout.addWidget(box, stretch=1)

        self.warn = QtWidgets.QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #c0562b;")
        layout.addWidget(self.warn)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Join")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.resize(460, 320)
        self._sync_ok()

    def _sync_ok(self, *_a) -> None:
        """Enable *Join* only when ≥2 datasets are ticked; warn on mixed serials."""
        picked = self.selected_ids()
        ok_btn = self.buttons.button(QtWidgets.QDialogButtonBox.Ok)
        ok_btn.setEnabled(len(picked) >= 2)
        serials = {str(d.serial_number) for d in self._datasets if d.id in picked}
        if len(picked) < 2:
            self.warn.setText("Tick at least two datasets to join.")
        elif len(serials) > 1:
            self.warn.setText(
                "Warning: the selected datasets have different serial numbers — "
                "joining them assumes they belong to one continuous recording."
            )
        else:
            self.warn.setText("")

    def selected_ids(self) -> list:
        """Return the ids of the currently ticked datasets."""
        return [chk.property("ds_id") for chk in self._checks if chk.isChecked()]


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
