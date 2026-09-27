"""Small reusable widgets and dialogs for the aerosoltools GUI.

These are presentation-only helpers with no business logic: the threshold and
exposure-limit line controls shared by the plot tabs, a tab bar that sizes its
tabs so labels never clip, the modal dialog used to pick two size-resolved
datasets and a crossover for combining their ranges, the checklist dialog for
joining same-instrument datasets into one recording, and the read-only
keyboard-shortcut reference.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

from ...exposure_limits import ExposureLimit, ExposureLimitList, applicable_limit
from ..qt import Figure, FigureCanvas, QtCore, QtWidgets
from .exposure_limit_picker import (
    ExposureLimitCombo,
    format_value,
    pick_record,
    record_candidates,
    record_limit,
    short_term_name,
)

#: Line styles of a substance's limits: the 8-hour limit dashed, the
#: short-term limit dotted (a typed threshold is dashed too).
TWA_STYLE = "--"
STEL_STYLE = ":"


class ExposureLimitLinesDialog(QtWidgets.QDialog):
    """Choose the substance whose exposure limits a plot shows as lines.

    A searchable substance list plus toggles for the 8-hour and short-term
    lines. Every change applies at once through ``on_change(record)`` — a
    ``pick_record`` with a ``"show"`` dict, or ``None`` to remove the lines — so
    the plot updates behind the dialog.
    """

    def __init__(
        self,
        parent,
        limits: ExposureLimitList,
        record: Optional[dict],
        on_change: Callable[[Optional[dict]], None],
    ):
        """Build the substance picker, line toggles and applicability note.

        Args:
            parent: Parent widget.
            limits: The list to pick from.
            record: The current link (``None`` when the lines are typed/off).
            on_change: Called with the new link after every change.
        """
        super().__init__(parent)
        self.setWindowTitle("Exposure-limit lines")
        self._limits = limits
        self._on_change = on_change
        show = (record or {}).get("show") or {}

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.combo = ExposureLimitCombo()
        self.combo.set_limits(limits)
        self.combo.set_current_name(record.get("substance") if record else None)
        self.combo.picked.connect(self._emit)
        form.addRow("Substance:", self.combo)
        self.show_twa = QtWidgets.QCheckBox("8-hour limit (dashed)")
        self.show_stel = QtWidgets.QCheckBox("Short-term limit (dotted)")
        self.show_twa.setChecked(bool(show.get("twa", True)))
        self.show_stel.setChecked(bool(show.get("stel", True)))
        lines = QtWidgets.QHBoxLayout()
        for box in (self.show_twa, self.show_stel):
            box.toggled.connect(self._emit)
            lines.addWidget(box)
        lines.addStretch(1)
        form.addRow("Lines:", lines)
        layout.addLayout(form)

        self.note = QtWidgets.QLabel("")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        source = QtWidgets.QLabel(
            f"Limits from {limits.source.label}. Both limits apply at the same "
            "time: the 8-hour limit to the shift's time-weighted average, the "
            "short-term limit to 15-minute averages."
        )
        source.setWordWrap(True)
        source.setStyleSheet("color: palette(mid);")
        layout.addWidget(source)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        remove = buttons.addButton("Remove lines", QtWidgets.QDialogButtonBox.ResetRole)
        remove.clicked.connect(self._remove)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)
        self.resize(620, self.sizeHint().height())

    def set_note(self, text: str) -> None:
        """Show whether the lines apply to the plotted series."""
        self.note.setText(text)

    def _emit(self, *_args) -> None:
        """Pass the current choice to the plot."""
        limit = self.combo.current_limit()
        if limit is None:
            self._on_change(None)
            return
        record = pick_record(limit, self._limits)
        record["show"] = {
            "twa": self.show_twa.isChecked(),
            "stel": self.show_stel.isChecked(),
        }
        self._on_change(record)

    def _remove(self) -> None:
        """Unlink the substance and close."""
        self.combo.set_current_name(None)
        self._on_change(None)
        self.accept()


class ThresholdControls(QtWidgets.QWidget):
    """Inline controls for a threshold line or a substance's exposure-limit lines.

    A check-box switches the lines on/off. Typed by hand, the value field sets
    one line (in the plot's *current* y-units) and the label field its legend
    text. The *OEL…* button instead links the controls to a substance from the
    Danish limit-value order: its 8-hour limit (dashed) and short-term limit
    (dotted) are drawn together, since both apply at once. Linked lines are
    converted to the plotted unit on every draw, hidden when the series is not
    a comparable concentration, and flagged in :attr:`warning` when the series'
    size fraction does not fit the limit's (PM1 against a total-dust limit).
    Typing a value unlinks. :attr:`changed` fires whenever any of this changes,
    so the owning tab can persist the state and redraw; the lines themselves
    are drawn by :func:`helpers.draw_thresholds`.
    """

    #: Emitted when the enable box, value, label or linked limit changes.
    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        parent=None,
        limits_provider: Optional[Callable[[], ExposureLimitList]] = None,
    ):
        """Build the enable check-box, value and legend fields and *OEL…* button.

        Args:
            parent: Parent widget.
            limits_provider: Returns the exposure-limit list the *OEL…* button
                offers; without one the button is hidden.
        """
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        # The linked substance (a pick_record plus "show": {"twa", "stel"}), the
        # y-unit at the last draw, and the warning that draw produced.
        self._oel: Optional[dict] = None
        self._unit: Optional[str] = None
        self._warning = ""
        self._limits_provider = limits_provider
        self._dialog: Optional[ExposureLimitLinesDialog] = None

        self.enable = QtWidgets.QCheckBox("Threshold")
        self.enable.setToolTip(
            "Overlay horizontal limit lines — a typed value, or a substance's "
            "occupational exposure limits (OEL…) — so it is clear at which times "
            "the concentration rose above them."
        )
        self.value = QtWidgets.QLineEdit()
        self.value.setPlaceholderText("value")
        self.value.setFixedWidth(90)
        self.label = QtWidgets.QLineEdit()
        self.label.setPlaceholderText("legend, e.g. OEL")
        self.label.setFixedWidth(120)
        self.label.setToolTip("Legend text shown next to the line(s).")
        self.oel_btn = QtWidgets.QPushButton("OEL…")
        self.oel_btn.setToolTip(
            "Show a substance's 8-hour and short-term exposure limits (Danish "
            "limit-value order) as lines, converted to the plotted unit."
        )
        self.oel_btn.clicked.connect(self._open_dialog)
        self.oel_btn.setVisible(limits_provider is not None)

        lay.addWidget(self.enable)
        lay.addWidget(self.value)
        lay.addWidget(self.label)
        lay.addWidget(self.oel_btn)

        self.enable.stateChanged.connect(lambda _state: self.changed.emit())
        self.value.editingFinished.connect(self.changed.emit)
        self.label.editingFinished.connect(self.changed.emit)
        # Typing a value by hand replaces a linked substance.
        self.value.textEdited.connect(self._unlink)
        self._sync_value_tip()

    # -- state ------------------------------------------------------------
    def state(self) -> dict:
        """Return the current ``{"on", "value", "label", "oel"}`` state (JSON-safe)."""
        return {
            "on": self.enable.isChecked(),
            "value": self.value.text().strip(),
            "label": self.label.text().strip(),
            "oel": dict(self._oel) if self._oel else None,
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
        self._oel = self._upgrade(state.get("oel"))
        self._sync_value_tip()

    @staticmethod
    def _upgrade(oel) -> Optional[dict]:
        """A stored link, upgraded from the one-line ``"kind"`` form if needed."""
        if not isinstance(oel, dict):
            return None
        oel = dict(oel)
        if "show" not in oel:
            kind = oel.pop("kind", None)
            oel["show"] = {"twa": kind != "stel", "stel": kind != "twa"}
        return oel

    def legend_text(self) -> str:
        """The user's legend text (may be empty)."""
        return self.label.text().strip()

    @property
    def warning(self) -> str:
        """Why linked lines are hidden or may mislead, from the last draw (or "")."""
        return self._warning

    # -- lines ------------------------------------------------------------
    def threshold_lines(
        self,
        unit: Optional[str],
        metrics: Sequence[Tuple[str, Optional[float]]] = (),
    ) -> list:
        """The lines to draw on an axis in ``unit`` that shows ``metrics``.

        Args:
            unit: The axis' y-unit (``None`` if unknown, e.g. a normalised axis).
            metrics: ``(name, size cut in µm)`` of each series on that axis
                (see ``logic.exposure_limits.series_size_cut``), for the
                size-fraction check of a linked substance.

        Returns:
            ``[{"value", "label", "linestyle"}, …]`` — empty when switched off.
            :attr:`warning` then says why linked lines are hidden, or which
            series' size fraction the limit does not fit.
        """
        self._unit = unit
        self._warning = ""
        if self._oel is not None:
            return self._linked_lines(unit, metrics)
        if not self.enable.isChecked():
            return []
        try:
            value = float(self.value.text().strip())
        except ValueError:
            return []
        label = self.legend_text() or f"Threshold ({value:g})"
        return [{"value": value, "label": label, "linestyle": TWA_STYLE}]

    def _linked_lines(self, unit: Optional[str], metrics) -> list:
        """The linked substance's 8-hour and short-term lines in ``unit``."""
        rec = self._oel
        limit = record_limit(rec)
        if not limit.is_comparable_to(unit):
            self._warning = (
                f"{limit.name}: the limits are in {limit.unit}, the plotted series "
                f"in {unit or 'an unknown unit'} — no limit lines."
            )
            self._show_values(limit, None)
            return []
        self._warning = self._fraction_warning(rec, metrics)
        self._show_values(limit, unit)
        if not self.enable.isChecked():
            return []
        show = rec.get("show") or {}
        base = self.legend_text() or limit.name
        source = f" ({rec['source_short']})" if rec.get("source_short") else ""
        lines = []
        if show.get("twa", True) and limit.twa is not None:
            lines.append(
                {
                    "value": limit.twa_in(unit),
                    "label": f"{base} – 8-h OEL{source}",
                    "linestyle": TWA_STYLE,
                }
            )
        if show.get("stel", True) and limit.stel is not None:
            lines.append(
                {
                    "value": limit.stel_in(unit),
                    "label": f"{base} – {short_term_name(limit)}{source}",
                    "linestyle": STEL_STYLE,
                }
            )
        return lines

    @staticmethod
    def _fraction_warning(rec: dict, metrics) -> str:
        """Warn about series whose size fraction the linked limit does not fit."""
        candidates = record_candidates(rec)
        picked = candidates[0]
        messages = []
        for name, cut in metrics:
            limit, check = applicable_limit(candidates, name, cut)
            if limit.name == picked.name and check.applies:
                continue
            if check.applies:
                messages.append(
                    f"⚠ The lines are for {picked.fraction} dust ('{picked.name}'); "
                    f"for {name} use '{limit.name}'."
                )
            else:
                messages.append(f"⚠ {check.message}")
        return "\n".join(dict.fromkeys(messages))

    def _show_values(self, limit: ExposureLimit, unit: Optional[str]) -> None:
        """Show the linked limits in the value field as ``"8-hour / short-term"``."""
        if unit is None:
            text = "–"
        else:
            text = " / ".join(
                format_value(v) or "–"
                for v in (limit.twa_in(unit), limit.stel_in(unit))
            )
        self.value.blockSignals(True)
        self.value.setText(text)
        self.value.blockSignals(False)
        self._sync_value_tip()

    def _sync_value_tip(self) -> None:
        """Explain in the value field's tooltip what it shows."""
        if self._oel is None:
            self.value.setToolTip(
                "Threshold level, in the units currently shown on the y-axis."
            )
            return
        rec = self._oel
        tip = (
            f"8-hour / short-term limits for {rec.get('substance')} "
            f"({rec.get('source')}), in the plotted unit"
            f"{' (' + self._unit + ')' if self._unit else ''}. Type a value to "
            "replace them with a single threshold."
        )
        if self._warning:
            tip += "\n" + self._warning
        self.value.setToolTip(tip)

    def _unlink(self, *_args) -> None:
        """Drop the linked substance (the value becomes a typed threshold)."""
        if self._oel is not None:
            self._oel = None
            self._warning = ""
            self._sync_value_tip()

    # -- OEL dialog ---------------------------------------------------------
    def _open_dialog(self) -> None:
        """Open the substance picker; changes apply to the plot at once."""
        if self._limits_provider is None:
            return
        dlg = ExposureLimitLinesDialog(
            self, self._limits_provider(), self._oel, self._set_link
        )
        dlg.set_note(self._warning)
        self._dialog = dlg
        try:
            dlg.exec_()
        finally:
            self._dialog = None

    def _set_link(self, record: Optional[dict]) -> None:
        """Link the lines to a substance (``None``: remove them), then redraw."""
        self._oel = record
        widgets = (self.enable, self.value, self.label)
        for w in widgets:
            w.blockSignals(True)
        if record is not None:
            self.enable.setChecked(True)
            self.label.setText(record.get("substance", ""))
        else:
            self.enable.setChecked(False)
            self.value.setText("")
            self.label.setText("")
        for w in widgets:
            w.blockSignals(False)
        self._warning = ""
        self.changed.emit()  # the tab redraws, refreshing the warning
        if self._dialog is not None:
            self._dialog.set_note(self._warning or "The lines apply to the plot.")
        self._sync_value_tip()


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


def repopulate_combo(combo, items, default_index: int = 0) -> None:
    """Repopulate a ``QComboBox`` with ``items``, preserving the selection.

    Signals are blocked while the combo is rebuilt (so no spurious
    ``currentIndexChanged`` fires). The previously-shown text is re-selected if
    it is still present; otherwise the combo falls back to ``default_index``.
    Shared by the tabs whose activity picker is refreshed on every redraw.
    """
    combo.blockSignals(True)
    current = combo.currentText()
    combo.clear()
    combo.addItems(list(items))
    idx = combo.findText(current)
    combo.setCurrentIndex(idx if idx >= 0 else default_index)
    combo.blockSignals(False)


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
