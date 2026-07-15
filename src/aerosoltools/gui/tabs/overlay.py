"""Multi-dataset time-series overlay comparison tab.

Overlays up to **three metrics** over time for several datasets at once. Colour
identifies the dataset; line style identifies the *metric slot* (solid, dash-dot,
dotted); and each metric is assigned to one of up to three y-axes (left, right,
2nd right) so quantities in different units stay readable together. A per-dataset
view-only time shift lets the user slide instruments in time to line peaks up.
"""

from __future__ import annotations

import traceback

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..._core import _shading
from ..logic import helpers
from ..qt import QtCore, QtWidgets
from ..view.widgets import ThresholdControls, WheelLineEdit
from . import _autoscale
from ._base import _active_color_cycle, _PlotTab

#: Line styles encoding which *metric slot* a series belongs to. Slot 1 (the
#: primary, usually total concentration) is solid so a fast-oscillating,
#: high-time-resolution series (e.g. a CPC) still reads clearly; the secondary
#: slots use dash-dot / dotted, which stay distinguishable when zoomed out.
_METRIC_STYLES = ["-", "-.", ":"]

#: Sentinel shown in the metric pickers for an unused slot.
_NONE = "— none —"

#: Words describing each axis by its drawing position (left, right, 2nd right).
_POSITION_WORDS = ["Left axis", "Right axis", "2nd right axis"]

#: Concentration bases offered (as suffixed options) for the particle total.
_TOTAL_BASES = ["dN", "dM", "dS", "dV"]

#: Superscript digits/sign mapped to plain characters for unit comparison.
_SUP = str.maketrans(
    {
        "⁻": "-",
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
)


def _unit_key(unit: str) -> str:
    """Normalise a unit string (drop LaTeX/superscript/case/spaces) for comparison.

    So ``"cm⁻³"``, ``"cm^-3"`` and the CPC's LaTeX ``"cm$^{-3}$"`` all reduce to
    the same key and are recognised as one unit.
    """
    key = (unit or "").strip().lower().translate(_SUP)
    for ch in "$^{}\\ ":
        key = key.replace(ch, "")
    return key


#: Normalised keys of number-concentration units (share one y-axis / label).
_NUMBER_UNIT_KEYS = {
    _unit_key(u)
    for u in (
        "cm⁻³",
        "cm-3",
        "#/cm³",
        "1/cm³",
        "p/cm³",
        "particles/cm³",
        "n/cm³",
        "cm$^{-3}$",
    )
}


def _format_hms(td: pd.Timedelta) -> str:
    """Render a timedelta as a signed ``[-]H:MM:SS`` string (whole seconds)."""
    total = int(round(td.total_seconds()))
    sign = "-" if total < 0 else ""
    total = abs(total)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h}:{m:02d}:{s:02d}"


def _parse_hms(text: str):
    """Parse ``[-]H:MM:SS`` / ``M:SS`` / bare seconds into a timedelta, or None.

    A bare number is read as seconds, so small nudges (a few seconds) are easy
    to type. Returns ``None`` on unparseable input so the caller can revert.
    """
    text = text.strip()
    if not text:
        return pd.Timedelta(0)
    sign = 1
    if text[0] in "+-":
        sign = -1 if text[0] == "-" else 1
        text = text[1:].strip()
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 1:
        h, m, s = 0.0, 0.0, nums[0]
    elif len(parts) == 2:
        h, m, s = 0.0, nums[0], nums[1]
    elif len(parts) == 3:
        h, m, s = nums
    else:
        return None
    return pd.Timedelta(seconds=sign * (h * 3600 + m * 60 + s))


def _canon_unit(unit: str) -> str:
    """Canonicalise a unit so number-concentration spellings compare/display equal."""
    if _unit_key(unit) in _NUMBER_UNIT_KEYS:
        return "cm⁻³"
    return (unit or "").strip()


class OverlayTab(_PlotTab):
    """Overlay up to three metrics over time for several datasets at once.

    Unlike the single-view tabs, this reads *all* of the project's datasets.

    Encoding:
        * **Colour** identifies the dataset.
        * **Line style** identifies the metric slot (solid / dash-dot / dotted).
        * Each metric is drawn against a **user-assigned y-axis** (1/2/3), which
          become the left, right and 2nd-right spines. Per-axis min/max and log
          controls live in a second controls row. Series with **different units
          never share an axis**: if two are assigned to the same axis, the
          conflicting one is moved to a free/compatible axis automatically so a
          real unit label can always be shown (same-unit series may still be
          split across axes on purpose, e.g. very different scales).

    A per-dataset **view-only** time shift lets the user slide instruments in
    time to line up peaks; "Apply shifts permanently" bakes those shifts into the
    datasets' time axes (and re-projects the shared, absolute-time tasks).
    """

    export_tag = "overlay"

    def __init__(self, main):
        """Build the overlay controls and the per-dataset include/shift table."""
        super().__init__(main, nrows=1)

        self._conv_cache: dict = {}  # per-draw basis-conversion memo (see _converted)

        # -- metric row (row 1): up to three narrow metric pickers, each with a
        #    small target-axis selector, plus the concentration-basis selector.
        self.metric_combos: list = []
        self.axis_combos: list = []
        for slot in range(3):
            self.controls.addWidget(QtWidgets.QLabel(f"M{slot + 1}:"))
            combo = QtWidgets.QComboBox()
            # A small minimum keeps the controls row (and hence the middle pane)
            # from claiming an excessive minimum width; it still expands to show
            # the long option text when there is room.
            combo.setMinimumWidth(80)
            combo.setMaximumWidth(160)
            combo.setToolTip(
                "Metric for this slot (colour = dataset, line style = slot). "
                "The 'Total concentration (dN/dM/dS/dV)' options are particle "
                "totals; named measurements (e.g. a black-carbon or gas channel) "
                "appear under their own name. Choose '— none —' to leave the slot "
                "unused."
            )
            combo.currentIndexChanged.connect(self.refresh)
            self.controls.addWidget(combo)
            self.metric_combos.append(combo)
            ax_combo = QtWidgets.QComboBox()
            ax_combo.addItems(["→1", "→2", "→3"])
            ax_combo.setCurrentIndex(slot)  # default M1→axis1, M2→axis2, M3→axis3
            ax_combo.setToolTip(
                "Which y-axis (1=left, 2=right, 3=2nd right). Metrics with "
                "different units are never placed on the same axis — a conflict "
                "is moved to a free/compatible axis automatically."
            )
            ax_combo.setMaximumWidth(48)
            ax_combo.currentIndexChanged.connect(lambda *_: self._draw())
            self.controls.addWidget(ax_combo)
            self.axis_combos.append(ax_combo)

        self.normalize = QtWidgets.QCheckBox("Normalize (0–1)")
        self.normalize.setToolTip(
            "Scale each series to 0–1 to compare shapes across instruments with "
            "different units/magnitudes (all on one axis)."
        )
        self.normalize.stateChanged.connect(self._on_normalize)
        self.controls.addWidget(self.normalize)

        self.show_acts = QtWidgets.QCheckBox("Show activities")
        self.show_acts.setToolTip(
            "Shade the project's activities across every dataset, so you can see "
            "how the different instruments behave within the same task window."
        )
        self.show_acts.stateChanged.connect(lambda: self._draw(preserve=True))
        self.controls.addWidget(self.show_acts)

        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # -- axis-limits row (row 2): per-axis min / max / log controls. Built
        #    here, inserted into the left column after _split_with_side.
        self.controls2 = QtWidgets.QHBoxLayout()
        self.axis_groups: list = []
        self.axis_labels: list = []
        self.axis_min: list = []
        self.axis_max: list = []
        self.axis_log: list = []
        for i in range(3):
            group = QtWidgets.QWidget()
            gl = QtWidgets.QHBoxLayout(group)
            gl.setContentsMargins(0, 0, 0, 0)
            lbl = QtWidgets.QLabel(f"Axis {i + 1}")
            mn = QtWidgets.QLineEdit()
            mn.setPlaceholderText("min")
            mn.setFixedWidth(58)
            mn.editingFinished.connect(lambda: self._draw(preserve=True))
            mx = QtWidgets.QLineEdit()
            mx.setPlaceholderText("max")
            mx.setFixedWidth(58)
            mx.editingFinished.connect(lambda: self._draw(preserve=True))
            lg = QtWidgets.QCheckBox("log")
            lg.toggled.connect(lambda *_: self._draw(preserve=True))
            for w in (lbl, mn, mx, lg):
                gl.addWidget(w)
            self.controls2.addWidget(group)
            self.axis_groups.append(group)
            self.axis_labels.append(lbl)
            self.axis_min.append(mn)
            self.axis_max.append(mx)
            self.axis_log.append(lg)

        # Concentration-threshold (e.g. OEL) overlay lives on the second row to
        # keep the metric row (and thus the middle pane's minimum width) narrow.
        self.threshold = ThresholdControls()
        self.threshold.set_state(self.main.project.plot_thresholds.get(self.export_tag))
        self.threshold.changed.connect(self._on_threshold_changed)
        self.controls2.addWidget(self.threshold)
        self.controls2.addStretch(1)

        # Side panel: per-dataset include + shift table, plus an apply button.
        self._building = False
        self._extra_axes: list = []  # secondary y-axes for further metrics
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Dataset", "Shift"])
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_item_changed)

        self.apply_btn = QtWidgets.QPushButton("Apply shifts permanently")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.setToolTip(
            "Bake the current view shifts into the datasets (modifies their time "
            "axes). Shared tasks keep their absolute times."
        )
        self.apply_btn.clicked.connect(self._apply_shifts)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Datasets to overlay:"))
        side.addWidget(self.table, stretch=1)
        side.addWidget(self.apply_btn)
        hint = QtWidgets.QLabel(
            "Tick datasets to overlay. Set 'Shift' (h:mm:ss) to slide a dataset "
            "in time and line up peaks — click the field and scroll the mouse "
            "wheel for a smooth ±1 s nudge (Ctrl = ±1 min). View only until you "
            "apply permanently."
        )
        hint.setWordWrap(True)
        side.addWidget(hint)

        # Plot on the left of a resizable divider, side panel on the right.
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        self._split_with_side(side_widget)
        # Insert the axis-limits row just under the metric row (index 0 is the
        # metric controls, then the toolbar and canvas follow).
        self._left_col.insertLayout(1, self.controls2)

        self.ax = self.figure.add_subplot(111)
        self.canvas.setToolTip(
            "Colour identifies the dataset; line style identifies the metric "
            "slot: solid = M1, dash-dot = M2, dotted = M3."
        )

    # -- data access -------------------------------------------------------
    @property
    def _datasets(self):
        """All datasets in the project."""
        return self.main.project.datasets

    @staticmethod
    def _measurement_of(obj):
        """A dataset's named measurement (e.g. 'IR BCc'), or None for a plain total.

        Datasets that carry a measurement name are *not* particle number totals,
        so they are offered under that name rather than folded into the generic
        "Total concentration" metric (which would mislabel/mismix them).
        """
        meas = getattr(obj, "measurement", None)
        if not meas:
            meas = (getattr(obj, "metadata", {}) or {}).get("measurement")
        return meas or None

    def _metric_options(self):
        """Return ``(standard_names, extra_groups)`` shared by all metric pickers.

        ``standard_names`` lists the particle-total options ("Total concentration
        (dN/dM/dS/dV)"), then any named measurements (e.g. an aethalometer's
        "IR BCc"), then the shared core data channels. ``extra_groups`` is one
        ``(header, [names])`` per *instrument* — the extra housekeeping columns
        of all that instrument's datasets are unioned into a single group, so N
        datasets of the same instrument never spawn "Extra: ELPI", "Extra: ELPI
        (2)", … duplicates.
        """
        has_number = has_2d = False
        measurements: list = []
        channels: list = []
        seen_channels: set = set()
        extra_by_instr: dict = {}
        for ds in self._datasets:
            obj = ds.obj
            meas = self._measurement_of(obj)
            if meas:
                if meas not in measurements:
                    measurements.append(meas)
            elif helpers.is_particle(obj):
                # Only genuine particle instruments contribute to the generic
                # "Total concentration" totals; non-particle datasets expose
                # their named channels below instead.
                has_number = True
                if helpers.is_2d(obj):
                    has_2d = True
            std, extra = helpers.grouped_columns(obj)
            for _label, kind, name in std:
                if kind == "total" or name in seen_channels:
                    continue
                seen_channels.add(name)
                channels.append(name)
            names = extra_by_instr.setdefault(ds.instrument, [])
            for _label, _kind, name in extra:
                if name not in names:
                    names.append(name)

        totals: list = []
        if has_number:
            totals.append("Total concentration (dN)")
        if has_2d:
            totals += [f"Total concentration ({b})" for b in _TOTAL_BASES[1:]]
        standard = totals + measurements + channels
        groups = [
            (f"Extra: {instr}", names)
            for instr, names in extra_by_instr.items()
            if names
        ]
        return standard, groups

    def _converted(self, obj, basis: str):
        """``(converted_copy, unit)`` for ``obj`` on ``basis``, memoised per draw.

        Several metric slots can request the same dataset on the same basis in
        one redraw; caching the converted copy for the duration of a single
        :meth:`_gather_entries` pass avoids re-copying and re-converting the same
        object N times. The cache is reset each pass, so it never outlives the
        data it was built from.
        """
        key = (id(obj), basis)
        cached = self._conv_cache.get(key)
        if cached is None:
            cached = helpers.converted_copy(obj, basis)
            self._conv_cache[key] = cached
        return cached

    def _series_for(self, ds, metric: str):
        """Return ``(series, name, unit)`` for a dataset's metric, or None.

        "Total concentration (dN/dM/dS/dV)" is the generic particle total, only
        for datasets *without* a named measurement (dM/dS/dV need size-resolved
        data); a named measurement resolves to that dataset's primary series;
        otherwise the metric is looked up as a data / extra column.
        """
        obj = ds.obj
        meas = self._measurement_of(obj)
        if metric.startswith("Total concentration (") and metric.endswith(")"):
            if not helpers.is_particle(obj):
                # Non-particle instruments (gas, black carbon, LDSA, T/RH) are
                # never a generic particle total; they appear under their own
                # named channel/measurement instead.
                return None
            basis = metric[metric.index("(") + 1 : -1]
            if meas:  # a named measurement is not a generic particle total
                return None
            if basis != "dN":
                if not helpers.is_2d(obj):
                    return None
                conv, unit = self._converted(obj, basis)
                s = pd.to_numeric(conv.total_concentration, errors="coerce")
                return s, f"Total concentration ({basis})", unit
            _dtype, unit = helpers.describe(obj)
            s = pd.to_numeric(obj.total_concentration, errors="coerce")
            return s, "Total concentration", unit
        if meas is not None and metric == meas:
            _dtype, unit = helpers.describe(obj)
            # ``_primary`` is total_concentration for particle instruments and
            # the dataset's main channel for non-particle ones (e.g. a gas).
            return pd.to_numeric(obj._primary, errors="coerce"), metric, unit
        if metric in obj.data.columns:
            # A raw data/extra column: only carry a unit the column genuinely
            # owns (a per-column dict entry, e.g. Flow → l/min). Housekeeping
            # columns have none, so leave the unit blank and label by header
            # text rather than stitch on the primary series' unit (e.g. cm⁻³).
            unit = helpers.column_unit(obj, metric) or ""
            return pd.to_numeric(obj.data[metric], errors="coerce"), metric, unit
        if obj.extra_data is not None and metric in obj.extra_data.columns:
            unit = helpers.column_unit(obj, metric) or ""
            return pd.to_numeric(obj.extra_data[metric], errors="coerce"), metric, unit
        return None

    def _dataset_color(self, ds, cycle):
        """Stable colour for a dataset (its own colour, else by project order)."""
        if ds.color:
            return ds.color
        ids = [d.id for d in self._datasets]
        return cycle[ids.index(ds.id) % len(cycle)]

    # -- selectors sync ----------------------------------------------------
    def _sync_metrics(self) -> None:
        """Repopulate the three metric combos (grouped), preserving selections."""
        standard, groups = self._metric_options()
        for slot, combo in enumerate(self.metric_combos):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_NONE)
            for name in standard:
                combo.addItem(name)
            for header, names in groups:
                self._add_header(combo, f"── {header} ──")
                for name in names:
                    combo.addItem(name)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif slot == 0:
                default = combo.findText("Total concentration (dN)")
                combo.setCurrentIndex(default if default >= 0 else 0)
            else:
                combo.setCurrentIndex(0)  # — none —
            combo.blockSignals(False)

    @staticmethod
    def _add_header(combo, text: str) -> None:
        """Add a non-selectable section header to a combo."""
        combo.addItem(text)
        item = combo.model().item(combo.count() - 1)
        if item is not None:
            item.setEnabled(False)

    def _active_metrics(self):
        """Chosen ``(slot, metric, uaxis)`` triples for slots with a real metric."""
        out = []
        for slot, combo in enumerate(self.metric_combos):
            metric = combo.currentText()
            if not metric or metric == _NONE or metric.startswith("──"):
                continue
            uaxis = self.axis_combos[slot].currentIndex() + 1
            out.append((slot, metric, uaxis))
        return out

    def _sync_table(self) -> None:
        """Rebuild the include/shift table from the datasets."""
        self._building = True
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for ds in self._datasets:
            r = self.table.rowCount()
            self.table.insertRow(r)
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(
                QtCore.Qt.Checked if ds.overlay_on else QtCore.Qt.Unchecked
            )
            chk.setData(QtCore.Qt.UserRole, ds.id)
            self.table.setItem(r, 0, chk)
            name = QtWidgets.QTableWidgetItem(ds.label)
            name.setFlags(QtCore.Qt.ItemIsEnabled)
            self.table.setItem(r, 1, name)
            edit = WheelLineEdit(_format_hms(ds.view_shift))
            edit.setFixedWidth(74)
            edit.setToolTip(
                "View-only time shift as [-]h:mm:ss (e.g. -0:00:30 to move the "
                "dataset 30 s earlier). A bare number is read as seconds. With "
                "the field focused, scroll to nudge ±1 s (Ctrl = ±1 min)."
            )
            edit.editingFinished.connect(lambda e=edit, d=ds: self._on_shift_edit(d, e))
            edit.stepped.connect(
                lambda direction, coarse, e=edit, d=ds: self._on_shift_step(
                    d, e, direction, coarse
                )
            )
            self.table.setCellWidget(r, 2, edit)
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)
        self.table.blockSignals(False)
        self._building = False

    # -- interaction -------------------------------------------------------
    def _on_normalize(self) -> None:
        """Redraw for a normalize toggle (axis assignment/limits become moot)."""
        self._draw()

    def _on_item_changed(self, item) -> None:
        """Persist a dataset's overlay-include flag and redraw."""
        if self._building or item.column() != 0:
            return
        ds = self.main.project.get(item.data(QtCore.Qt.UserRole))
        if ds is not None:
            ds.overlay_on = item.checkState() == QtCore.Qt.Checked
            self._draw()

    def _on_shift_edit(self, ds, edit) -> None:
        """Parse an h:mm:ss shift field, persist the view-only shift and redraw."""
        if self._building:
            return
        td = _parse_hms(edit.text())
        if td is None:  # unparseable — revert to the current shift
            edit.setText(_format_hms(ds.view_shift))
            return
        ds.view_shift = td
        edit.blockSignals(True)
        edit.setText(_format_hms(td))  # normalise the display
        edit.blockSignals(False)
        self._draw(preserve=True)

    def _on_shift_step(self, ds, edit, direction: int, coarse: bool) -> None:
        """Nudge a dataset's view shift by a wheel tick (±1 s, or ±1 min coarse)."""
        if self._building:
            return
        step = pd.Timedelta(minutes=1) if coarse else pd.Timedelta(seconds=1)
        ds.view_shift = ds.view_shift + direction * step
        edit.blockSignals(True)
        edit.setText(_format_hms(ds.view_shift))
        edit.blockSignals(False)
        self._draw(preserve=True)

    def _apply_shifts(self) -> None:
        """Bake the current view shifts permanently into the datasets' time axes."""
        moved = [ds for ds in self._datasets if ds.view_shift != pd.Timedelta(0)]
        if not moved:
            QtWidgets.QMessageBox.information(
                self, "Apply shifts", "There are no view shifts to apply."
            )
            return
        ans = QtWidgets.QMessageBox.question(
            self,
            "Apply shifts permanently",
            f"Permanently shift {len(moved)} dataset(s) by their current view "
            "shift?\nThis changes their time axes; shared tasks keep their "
            "absolute times.",
        )
        if ans != QtWidgets.QMessageBox.Yes:
            return
        for ds in moved:
            ds.obj.timeshift(seconds=ds.view_shift.total_seconds())
            self.main.project._apply_activities(ds)
            ds.view_shift = pd.Timedelta(0)
        self.main.adjust_box.sync_crop_fields()
        self.main.refresh_all(reset_view=True)
        self.main._refresh_sidebar()

    def _on_threshold_changed(self) -> None:
        """Persist the threshold overlay state and redraw (keeping the view)."""
        self.main.project.plot_thresholds[self.export_tag] = self.threshold.state()
        self._draw(preserve=True)

    def _cap(self, edit) -> float | None:
        """Parse an axis-limit field, returning None when blank or invalid."""
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        """Re-sync the metric combos and table, then redraw."""
        if not self._datasets:
            return
        self._sync_metrics()
        self._sync_table()
        self._draw()

    def _clear_extra_axes(self) -> None:
        """Drop any secondary axes from the previous draw."""
        for extra in self._extra_axes:
            try:
                extra.remove()
            except Exception:
                pass
        self._extra_axes = []

    def _gather_entries(self, normalize: bool) -> list:
        """Collect one entry dict per (dataset × active metric slot) series."""
        self._conv_cache: dict = {}  # per-pass basis-conversion memo (see _converted)
        entries = []
        for slot, metric, uaxis in self._active_metrics():
            for ds in self._datasets:
                if not ds.overlay_on:
                    continue
                res = self._series_for(ds, metric)
                if res is None:
                    continue
                s, name, unit = res
                if s is None or s.empty:
                    continue
                x = mdates.date2num((s.index + ds.view_shift).to_pydatetime())
                y = s.to_numpy(dtype=float)
                if normalize:
                    lo, hi = np.nanmin(y), np.nanmax(y)
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        y = (y - lo) / (hi - lo)
                entries.append(
                    {
                        "ds": ds,
                        "slot": slot,
                        "uaxis": uaxis,
                        "x": x,
                        "y": y,
                        "name": name,
                        "unit": unit,
                    }
                )
        return entries

    def _draw_on(self, ax) -> None:
        """Draw each included dataset's series, grouping metrics onto their axes."""
        ax.clear()
        self._clear_extra_axes()
        normalize = self.normalize.isChecked()
        entries = self._gather_entries(normalize)
        cycle = _active_color_cycle()

        # Axis-limit controls only apply to the multi-axis view.
        for group in self.axis_groups:
            group.setVisible(not normalize)

        if not entries:
            ax.set_xlabel("Time")
            ax.text(
                0.5,
                0.5,
                "Pick a metric and tick one or more datasets to overlay.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        if normalize:
            self._draw_normalized(ax, entries, cycle)
        else:
            self._draw_multi_axis(ax, entries, cycle)

        # x-range from all series (the twin axes share this axis), then activity
        # shading and a combined, per-axis-grouped legend.
        xs = np.concatenate([e["x"] for e in entries])
        xs = xs[np.isfinite(xs)]
        if xs.size:
            ax.set_xlim(float(xs.min()), float(xs.max()))
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        if self.show_acts.isChecked():
            periods = self.main.project.activities
            selected = _shading.resolve_activities(periods, True)
            _shading.shade_activities(ax, periods, selected, zorder=1, legend=False)

        # Threshold line on the primary axis (drawn without its own legend).
        tv = self.threshold.threshold_value()
        threshold_handle = None
        if tv is not None and np.isfinite(tv):
            ax.axhline(
                tv,
                color=helpers.THRESHOLD_COLOR,
                linestyle="--",
                linewidth=1.6,
                zorder=5,
            )
            text = self.threshold.legend_text() or f"Threshold ({tv:g})"
            threshold_handle = (
                Line2D([], [], color=helpers.THRESHOLD_COLOR, ls="--", lw=1.6),
                text,
            )
        self._build_legend(ax, entries, normalize, threshold_handle)

    def _draw_normalized(self, ax, entries, cycle) -> None:
        """Draw all series on one axis, scaled to 0–1 (colour=dataset, style=slot)."""
        for e in entries:
            color = self._dataset_color(e["ds"], cycle)
            ax.plot(e["x"], e["y"], lw=1.4, ls=_METRIC_STYLES[e["slot"]], color=color)
        ax.set_ylabel("Normalized (0–1)")
        ylim = _autoscale.y_limits([ax], log_y=False, anchor_zero=True)
        if ylim is not None:
            ax.set_ylim(*ylim)

    def _resolve_axis_units(self, entries) -> tuple:
        """Assign each entry a physical axis so no axis mixes canonical units.

        Honours each entry's requested axis (its ``uaxis``) when the unit is
        compatible with what is already there; otherwise moves it to the lowest
        numbered axis that is free or already carries that same unit. This keeps
        the user's freedom to spread *same-unit* series across axes (e.g. two
        cm⁻³ datasets far apart in scale) while never rendering two different
        units on one axis. Entries that cannot be placed (all three axes already
        hold other units) are returned as ``overflow`` and not drawn.

        Sets ``entry["_uaxis"]`` on each drawn entry. Returns
        ``(drawn, overflow)``.
        """
        axis_unit: dict = {}  # physical axis number -> canonical unit on it
        drawn: list = []
        overflow: list = []
        for e in entries:
            want = e["uaxis"]
            cu = _canon_unit(e["unit"])
            placed = None
            if axis_unit.get(want, cu) == cu:
                placed = want
            else:
                for cand in (1, 2, 3):
                    if axis_unit.get(cand, cu) == cu:
                        placed = cand
                        break
            if placed is None:
                overflow.append(e)
                continue
            axis_unit[placed] = cu
            e["_uaxis"] = placed
            drawn.append(e)
        return drawn, overflow

    def _draw_multi_axis(self, ax, entries, cycle) -> None:
        """Draw series with one y-axis per unit (up to three); never mix units."""
        drawn, overflow = self._resolve_axis_units(entries)
        used = sorted({e["_uaxis"] for e in drawn})
        # Map each used axis number to a Matplotlib axis: first used axis to the
        # left spine, then right, then an outward 2nd-right spine (so skipping an
        # axis number never leaves an empty middle spine).
        axis_for: dict = {}
        self._axis_position: dict = {}
        for pos, uaxis in enumerate(used):
            if pos == 0:
                target = ax
            else:
                target = ax.twinx()
                if pos == 2:
                    target.spines["right"].set_position(("outward", 60))
                self._extra_axes.append(target)
            axis_for[uaxis] = target
            self._axis_position[uaxis] = pos

        for e in drawn:
            target = axis_for[e["_uaxis"]]
            color = self._dataset_color(e["ds"], cycle)
            e["_color"] = color
            e["_style"] = _METRIC_STYLES[e["slot"]]
            target.plot(e["x"], e["y"], lw=1.4, ls=e["_style"], color=color)

        # Label each axis by its metric(s) + unit, and apply its log/limits. Each
        # axis now carries exactly one canonical unit, so the unit always shows.
        for uaxis, target in axis_for.items():
            ax_entries = [e for e in drawn if e["_uaxis"] == uaxis]
            names = list(dict.fromkeys(e["name"] for e in ax_entries))
            canon = {_canon_unit(e["unit"]) for e in ax_entries}
            unit = next(iter(canon)) if len(canon) == 1 else ""
            label = ", ".join(names[:2]) + ("…" if len(names) > 2 else "")
            target.set_ylabel(f"{label} [{unit}]" if unit else label)
            is_log = self.axis_log[uaxis - 1].isChecked()
            target.set_yscale("log" if is_log else "linear")
            self._apply_axis_limits(target, uaxis, is_log)

        # Warn (on the plot) about any metric that could not get its own unit
        # axis because all three axes are already taken by other units.
        if overflow:
            dropped = list(
                dict.fromkeys(
                    f"{e['name']} [{_canon_unit(e['unit'])}]" for e in overflow
                )
            )
            ax.text(
                0.5,
                0.99,
                "⚠ Only 3 y-axes: no room for " + ", ".join(dropped),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color=helpers.THRESHOLD_COLOR,
                zorder=6,
            )

    def _apply_axis_limits(self, target, uaxis: int, is_log: bool) -> None:
        """Autoscale an axis to its data, then apply any explicit min/max caps."""
        ylim = _autoscale.y_limits([target], log_y=is_log, anchor_zero=not is_log)
        if ylim is None:
            return
        lo, hi = ylim
        umin = self._cap(self.axis_min[uaxis - 1])
        umax = self._cap(self.axis_max[uaxis - 1])
        if umin is not None:
            lo = umin
        if umax is not None:
            hi = umax
        target.set_ylim(lo, hi)

    def _build_legend(self, ax, entries, normalize: bool, threshold_handle) -> None:
        """Build one legend, grouped under a header for each y-axis.

        Each axis gets a "── Left/Right axis — <metric> ──" separator header,
        then one entry per dataset drawn on it (labelled with the metric too when
        the axis carries more than one). This keeps the axis *labels* clean while
        still making it obvious which lines belong to which axis.
        """
        multi_metric = len({e["slot"] for e in entries}) > 1
        handles: list = []
        labels: list = []
        if threshold_handle is not None:
            handles.append(threshold_handle[0])
            labels.append(threshold_handle[1])

        def _entry_label(e) -> str:
            lbl = e["ds"].label
            if e["ds"].view_shift != pd.Timedelta(0):
                lbl += f" ({_format_hms(e['ds'].view_shift)})"
            if multi_metric:
                lbl += f" — {e['name']}"
            return lbl

        if normalize:
            for e in entries:
                handles.append(
                    Line2D(
                        [],
                        [],
                        color=self._dataset_color(e["ds"], _active_color_cycle()),
                        ls=_METRIC_STYLES[e["slot"]],
                        lw=1.4,
                    )
                )
                labels.append(_entry_label(e))
        else:
            # Group by the *resolved* physical axis (``_uaxis``), set on the
            # entries that were actually drawn; overflow entries are skipped.
            drawn = [e for e in entries if "_uaxis" in e]
            for uaxis in sorted({e["_uaxis"] for e in drawn}):
                pos = self._axis_position.get(uaxis, 0)
                ax_entries = [e for e in drawn if e["_uaxis"] == uaxis]
                names = list(dict.fromkeys(e["name"] for e in ax_entries))
                head = _POSITION_WORDS[pos] if pos < len(_POSITION_WORDS) else "Axis"
                metric_part = names[0] if len(names) == 1 else ", ".join(names[:2])
                handles.append(Line2D([], [], ls="none", marker="none"))
                labels.append(f"── {head}: {metric_part} ──")
                for e in ax_entries:
                    handles.append(
                        Line2D([], [], color=e["_color"], ls=e["_style"], lw=1.4)
                    )
                    labels.append(_entry_label(e))
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=8)

    def _draw(self, preserve: bool = False) -> None:
        """Redraw onto the embedded axis, reporting any error inline.

        Args:
            preserve: When True, keep the current *time* (x) view across the
                redraw so a view-only shift or toggle does not snap the zoom
                back. The y-axes always follow the per-axis autoscale + caps.
        """
        prev_x = None
        if preserve and self.ax.has_data():
            prev_x = self.ax.get_xlim()
        try:
            self._draw_on(self.ax)
        except Exception:
            self._show_message(
                "Could not draw overlay:\n" + traceback.format_exc(limit=1)
            )
            return
        if prev_x is not None:
            self.ax.set_xlim(prev_x)
        else:
            self._sync_toolbar_home()
        self.canvas.draw_idle()

    def current_time_xlim(self):
        """Time-axis limits as date numbers, or None."""
        if self.ax.has_data():
            return self.ax.get_xlim()
        return None
