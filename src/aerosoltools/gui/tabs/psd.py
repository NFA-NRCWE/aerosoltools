"""Particle size distribution (PSD) tab.

Works for one *or* several datasets: it reads the project's size-resolved (2D)
datasets and overlays one mean-PSD curve per ticked (dataset × activity), so a
single instrument behaves like the old single-view PSD tab while multiple
instruments are compared on the same axes.

On top of the comparison it offers **lognormal fitting** (via
:meth:`Aerosol2D.fit_psd`): the user picks one plotted curve as the *fit target*
(emphasised; the rest dim, or hidden via "Only target"), shapes one or more
modes **by eye** — click sets a mode's peak (μ, height), scrolling sets its width
(σ) — and may then press **Fit** to optimise those guesses. The hand-built
("manual") and optimised states are drawn differently so it is always clear
which is showing. Each fit is stored per (dataset × activity) on the dataset and
saved with the project; editing or deleting a task drops its fits.
"""

from __future__ import annotations

import traceback

import numpy as np

from .. import helpers
from ..qt import QtCore, QtWidgets
from . import _psdfit as fit
from ._base import _active_color_cycle, _PlotTab

# Modes table columns.
_COL_MU, _COL_SIGMA, _COL_PEAK, _COL_BIND = range(4)


class PSDTab(_PlotTab):
    """Overlay mean particle size distributions and fit lognormal modes.

    Reads *all* of the project's size-resolved (2D) datasets rather than the
    active one. The user ticks which datasets and which activities to compare;
    one mean-PSD curve is drawn per (dataset × activity) pair via the library's
    own :meth:`Aerosol2D.plot_psd` (``ax=`` shared), then relabelled/recoloured
    so each combination is distinct.

    The "Lognormal fit" side panel selects one plotted curve as the *fit target*
    and shapes modes on it by eye (click = peak, scroll = width); :meth:`_run_fit`
    optionally optimises them. Fits are kept per (dataset × activity) on the
    dataset (:attr:`Dataset.psd_fits`) so they persist in the saved project.
    """

    export_tag = "psd"

    def __init__(self, main):
        """Build the PSD controls, fit panel, dataset/activity lists and plot."""
        super().__init__(main, nrows=1)

        # Fitting state. ``_modes`` mirrors the current target's stored fit (a
        # list of {mu, sigma, peak, bound, *_err?} dicts); ``_fitted`` flags
        # whether they came from an optimisation (vs a hand-built guess). The
        # target object/colour/line are captured during the last full draw so
        # the overlay can be repainted cheaply (without recomputing the PSDs).
        self._modes: list[dict] = []
        self._fitted = False
        self._building_table = False
        self._target_obj = None
        self._target_color = None
        self._target_line = None
        self._target_xy = None  # (bin_mids, mean) of the target, for fit/R²
        self._overlay_artists: list = []

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.setChecked(True)
        self.normalize.setToolTip(
            "Plot dx/dlogDp. Lognormal fits are defined in this space, so it is "
            "enabled automatically while fitting."
        )
        # NB: wrap _draw in a lambda for the signals that carry a value (an int
        # state/index). _draw now takes a ``preserve`` flag, and PyQt5 would
        # otherwise pass that int straight into it — a scale/normalize change
        # must rescale, so swallow the argument and call _draw() with no view
        # preservation.
        self.normalize.stateChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.normalize)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.stateChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.log_y)

        self.band = QtWidgets.QCheckBox("±σ band")
        self.band.setToolTip(
            "Shade each curve's ±1σ spread. Off by default, since the bands "
            "overlap heavily when several curves are compared."
        )
        self.band.stateChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.band)

        self.controls.addWidget(QtWidgets.QLabel("Display:"))
        self.display_mode = QtWidgets.QComboBox()
        self.display_mode.addItems(["Lines", "Bars"])
        self.display_mode.setToolTip(
            "Bars show each size bin's width (best for a single PSD); lines are "
            "clearer when several curves are compared on the same axes."
        )
        self.display_mode.currentIndexChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.display_mode)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Side panel: datasets to compare (checkable) + activities (multi-select).
        self._building = False
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.itemChanged.connect(self._on_ds_changed)

        self.act_list = QtWidgets.QListWidget()
        self.act_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.act_list.itemSelectionChanged.connect(self._on_targets_changed)

        self.ds_list.setToolTip(
            "Tick the 2D datasets and activities to compare. One mean-PSD curve "
            "is drawn per dataset × activity."
        )
        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Datasets to compare:"))
        side.addWidget(self.ds_list, stretch=1)
        side.addWidget(QtWidgets.QLabel("Activities:"))
        side.addWidget(self.act_list, stretch=1)
        # The fitting controls live in the side panel (horizontal space) rather
        # than above the plot, so the canvas keeps its full height — a PSD drawn
        # on a short, wide canvas looks misleadingly flat.
        side.addWidget(self._build_fit_panel(), stretch=0)

        # Plot on the left of a resizable divider, side panel on the right.
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        self._split_with_side(side_widget, sizes=(700, 380))

        self.ax = self.figure.add_subplot(111)
        # Direct manipulation of modes (only while "Edit on plot" is active):
        # click sets the selected mode's peak; scroll sets its width.
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    # -- fit panel construction -------------------------------------------
    def _build_fit_panel(self) -> QtWidgets.QGroupBox:
        """Construct the compact "Lognormal fit" box (target + modes + actions).

        Kept deliberately tight (short labels, a few-row scrolling modes table,
        merged option rows) so it occupies little of the side panel.
        """
        box = QtWidgets.QGroupBox("Lognormal fit")
        # Take only the height it needs, so the dataset/activity lists above keep
        # the side panel's spare vertical space.
        box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        outer = QtWidgets.QVBoxLayout(box)
        outer.setSpacing(4)
        outer.setContentsMargins(6, 4, 6, 6)

        # Row 1: which plotted curve to fit + "show only that curve".
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(QtWidgets.QLabel("Target:"))
        self.fit_target = QtWidgets.QComboBox()
        self.fit_target.setToolTip(
            "Which plotted curve the fit applies to. It is drawn bold and named "
            "on the plot; the other curves dim."
        )
        self.fit_target.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.fit_target.currentIndexChanged.connect(self._on_target_changed)
        row1.addWidget(self.fit_target, stretch=1)
        self.isolate_chk = QtWidgets.QCheckBox("Only")
        self.isolate_chk.setToolTip(
            "Hide the other curves and show only the fit target."
        )
        # Showing/hiding curves must rescale to the now-visible data, even while
        # a fit is being edited (which otherwise pins the view).
        self.isolate_chk.stateChanged.connect(lambda *_: self._draw(force_rescale=True))
        row1.addWidget(self.isolate_chk)
        outer.addLayout(row1)

        # Modes table: μ / σ / peak height (all editable) + a "bind μ" check.
        # Compact rows; extra modes scroll rather than grow the panel.
        self.modes_table = QtWidgets.QTableWidget(0, 4)
        self.modes_table.setHorizontalHeaderLabels(["μ (nm)", "σ", "Peak", "Bind"])
        self.modes_table.setToolTip(
            "Peak diameter μ, geometric SD σ, and peak height (dx/dlogDp). "
            "Tick Bind to hold a mode's μ near its value during the fit."
        )
        self.modes_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.modes_table.verticalHeader().setVisible(False)
        self.modes_table.verticalHeader().setDefaultSectionSize(22)
        self.modes_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.modes_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.modes_table.setMinimumHeight(58)
        self.modes_table.setMaximumHeight(110)
        self.modes_table.itemSelectionChanged.connect(self._redraw_overlay)
        self.modes_table.itemChanged.connect(self._on_mode_edited)
        outer.addWidget(self.modes_table)

        # Row 2: edit-on-plot toggle + add/remove mode (short labels).
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(4)
        self.edit_btn = QtWidgets.QPushButton("Edit on plot")
        self.edit_btn.setObjectName("toggle")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setToolTip(
            "Shape modes by eye: with a mode selected, click the plot to set its "
            "peak (μ and height) and scroll over the plot to widen/narrow it (σ). "
            "With no mode selected, a click adds one."
        )
        self.edit_btn.toggled.connect(self._toggle_edit)
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.setToolTip("Add a mode at the target's mid-range, then shape it.")
        add_btn.clicked.connect(self._add_mode)
        rem_btn = QtWidgets.QPushButton("Del")
        rem_btn.setToolTip("Remove the selected mode.")
        rem_btn.clicked.connect(self._remove_mode)
        row2.addWidget(self.edit_btn, stretch=1)
        row2.addWidget(add_btn)
        row2.addWidget(rem_btn)
        outer.addLayout(row2)

        # Row 3: fit / clear + options, all on one line.
        row3 = QtWidgets.QHBoxLayout()
        row3.setSpacing(4)
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("primary")
        self.fit_btn.setToolTip("Optimise the current modes to the target curve.")
        self.fit_btn.clicked.connect(self._run_fit)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setToolTip("Remove all modes and the fit overlay.")
        clear_btn.clicked.connect(self._clear_fit)
        self.local_chk = QtWidgets.QCheckBox("Local")
        self.local_chk.setChecked(True)
        self.local_chk.setToolTip(
            "Fit each mode only to the bins near it (a window scaled by its "
            "width), so modes follow the peaks instead of one broad lognormal "
            "stretching across the whole size range. Turn off to fit globally. "
            "The shaded band on the plot marks the window the fit uses."
        )
        # Toggling Local changes the shaded fit-window band, so repaint it.
        self.local_chk.stateChanged.connect(lambda *_: self._redraw_overlay())
        self.log_scaling = QtWidgets.QCheckBox("Log")
        self.log_scaling.setChecked(True)
        self.log_scaling.setToolTip(
            "Fit against log10(dx/dlogDp) so weakly populated modes are not "
            "swamped by the dominant mode (recommended)."
        )
        self.tolerance = QtWidgets.QLineEdit("10")
        self.tolerance.setFixedWidth(36)
        self.tolerance.setToolTip(
            "Percent window a bound μ may move within during the fit."
        )
        row3.addWidget(self.fit_btn, stretch=1)
        row3.addWidget(clear_btn)
        row3.addWidget(self.local_chk)
        row3.addWidget(self.log_scaling)
        row3.addWidget(QtWidgets.QLabel("tol%"))
        row3.addWidget(self.tolerance)
        outer.addLayout(row3)

        self.fit_status = QtWidgets.QLabel(
            "Edit on plot: click sets a mode's peak, scroll sets its width. "
            "Fit to optimise (optional)."
        )
        self.fit_status.setWordWrap(True)
        outer.addWidget(self.fit_status)
        return box

    # -- data access -------------------------------------------------------
    @property
    def _datasets_2d(self):
        """The project's size-resolved (2D) datasets."""
        return [d for d in self.main.project.datasets if helpers.is_2d(d.obj)]

    def _selected_activities(self) -> list:
        """Names of the activities ticked in the activity list."""
        return [i.text() for i in self.act_list.selectedItems()]

    def _available_targets(self) -> list:
        """List of ``(ds, activity)`` pairs that actually have a PSD to plot.

        Mirrors the selection logic in :meth:`_draw_on` / ``plot_psd`` so the
        fit-target dropdown offers exactly the curves on screen.
        """
        out = []
        for ds in [d for d in self._datasets_2d if d.psd_on]:
            for act in self._selected_activities():
                if act not in ds.obj.activities:
                    continue
                try:
                    subset = ds.obj.data[ds.obj.data[act]]
                except Exception:
                    continue
                if not subset.empty:
                    out.append((ds, act))
        return out

    def _current_target(self):
        """Return ``(ds, activity)`` for the selected fit target, or ``None``."""
        data = self.fit_target.currentData()
        if data is None:
            return None
        ds = self.main.project.get(data[0])
        if ds is None:
            return None
        return ds, data[1]

    # -- list / target sync ------------------------------------------------
    def _sync_datasets(self) -> None:
        """Rebuild the dataset checklist from the project's 2D datasets."""
        self._building = True
        self.ds_list.blockSignals(True)
        self.ds_list.clear()
        for ds in self._datasets_2d:
            item = QtWidgets.QListWidgetItem(ds.label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if ds.psd_on else QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, ds.id)
            self.ds_list.addItem(item)
        self.ds_list.blockSignals(False)
        self._building = False

    def _sync_activities(self) -> None:
        """Rebuild the activity multi-select, preserving the selection."""
        self.act_list.blockSignals(True)
        selected = {i.text() for i in self.act_list.selectedItems()}
        self.act_list.clear()
        names = ["All data"] + self.main.project.user_activities()
        for name in names:
            self.act_list.addItem(name)
            if name in selected:
                self.act_list.item(self.act_list.count() - 1).setSelected(True)
        # First time shown (nothing selected yet): default to "All data".
        if not selected and self.act_list.count():
            self.act_list.item(0).setSelected(True)
        self.act_list.blockSignals(False)

    def _sync_fit_target(self) -> None:
        """Rebuild the fit-target dropdown from the plottable curves."""
        self.fit_target.blockSignals(True)
        prev = self.fit_target.currentData()
        self.fit_target.clear()
        for ds, act in self._available_targets():
            self.fit_target.addItem(f"{ds.label} – {act}", (ds.id, act))
        if prev is not None:
            idx = self.fit_target.findData(prev)
            if idx >= 0:
                self.fit_target.setCurrentIndex(idx)
        self.fit_target.blockSignals(False)
        has_target = self.fit_target.count() > 0
        for w in (self.fit_target, self.fit_btn, self.edit_btn, self.isolate_chk):
            w.setEnabled(has_target)

    # -- target fit load / store (persisted on the dataset) ----------------
    def _load_target_fit(self) -> None:
        """Load the current target's stored modes into the working state."""
        target = self._current_target()
        if target is None:
            self._modes = []
            self._fitted = False
            self._write_modes_to_table()
            return
        ds, act = target
        rec = ds.psd_fits.get(act)
        if rec and rec.get("modes"):
            self._modes = fit.clean_modes(rec["modes"])
            self._fitted = bool(rec.get("optimized"))
        else:
            self._modes = []
            self._fitted = False
        self._write_modes_to_table()

    def _store_target_fit(self) -> None:
        """Persist the working modes onto the current target's dataset."""
        target = self._current_target()
        if target is None:
            return
        ds, act = target
        if self._modes:
            ds.psd_fits[act] = {
                "modes": fit.clean_modes(self._modes),
                "optimized": bool(self._fitted),
            }
        else:
            ds.psd_fits.pop(act, None)

    # -- interaction -------------------------------------------------------
    def _on_ds_changed(self, item) -> None:
        """Persist a dataset's include flag and redraw."""
        if self._building:
            return
        ds = self.main.project.get(item.data(QtCore.Qt.UserRole))
        if ds is not None:
            ds.psd_on = item.checkState() == QtCore.Qt.Checked
            self._on_targets_changed()

    def _on_targets_changed(self) -> None:
        """Dataset/activity selection changed: re-sync target, reload fit, draw."""
        self._sync_fit_target()
        self._load_target_fit()
        self._draw()

    def _on_target_changed(self, _index: int = 0) -> None:
        """Fit target changed: load that curve's stored modes and redraw."""
        self._load_target_fit()
        self._draw()

    def _toggle_edit(self) -> None:
        """Enter/leave on-plot editing (and drop any active pan/zoom)."""
        active = self.edit_btn.isChecked()
        self.edit_btn.setText("Editing…" if active else "Edit on plot")
        if active and self.toolbar.mode:
            mode = str(self.toolbar.mode)
            if "pan" in mode:
                self.toolbar.pan()
            elif "zoom" in mode:
                self.toolbar.zoom()
        # Emphasis depends on edit mode, so reflect the change immediately — but
        # keep the user's current zoom: they may have zoomed in to place modes,
        # and flipping editing on/off should not throw that away.
        self._draw(preserve=True)

    def _selected_row(self) -> int:
        """Index of the selected mode, or -1."""
        row = self.modes_table.currentRow()
        return row if 0 <= row < len(self._modes) else -1

    def _on_press(self, event) -> None:
        """Click on the plot: set the selected mode's peak, or add a new mode."""
        if not self.edit_btn.isChecked() or event.inaxes is not self.ax:
            return
        if event.button != 1 or self.toolbar.mode:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None or x <= 0 or y <= 0:
            return
        row = self._selected_row()
        if row >= 0:
            self._modes[row]["mu"] = float(x)
            self._modes[row]["peak"] = float(y)
            self._modes[row].pop("mu_err", None)
        else:
            self._modes.append(
                {
                    "mu": float(x),
                    "sigma": fit.DEFAULT_SIGMA,
                    "peak": float(y),
                    "bound": False,
                }
            )
            row = len(self._modes) - 1
        self._fitted = False
        self._ensure_normalized()
        self._store_target_fit()
        self._write_modes_to_table()
        self.modes_table.selectRow(row)
        self._draw()

    def _on_scroll(self, event) -> None:
        """Scroll over the plot: widen/narrow the selected mode (σ)."""
        if not self.edit_btn.isChecked() or event.inaxes is not self.ax:
            return
        if self.toolbar.mode:
            return
        row = self._selected_row()
        if row < 0:
            return
        m = self._modes[row]
        # Scroll up → narrower, scroll down → wider. A small per-notch step
        # (~6%) lets the width be fine-tuned rather than jumping.
        sigma = float(m["sigma"]) * (1.06 ** (-event.step))
        m["sigma"] = float(min(fit.SIGMA_MAX, max(fit.SIGMA_MIN, sigma)))
        m.pop("sigma_err", None)
        self._fitted = False
        self._ensure_normalized()
        self._store_target_fit()
        # Update just the σ cell (keeping the row selected) and repaint only the
        # overlay, so rapid scrolling stays smooth.
        self._building_table = True
        self.modes_table.blockSignals(True)
        cell = self.modes_table.item(row, _COL_SIGMA)
        if cell is not None:
            cell.setText(f"{m['sigma']:.2f}")
        self.modes_table.blockSignals(False)
        self._building_table = False
        self._redraw_overlay()

    def _add_mode(self) -> None:
        """Append a default mode centred on the target's size range."""
        target = self._current_target()
        if target is None:
            return
        bm = np.asarray(target[0].obj.bin_mids, dtype=float)
        mu = float(np.sqrt(bm.min() * bm.max()))
        peak = 1.0
        if self._target_xy is not None:
            ydata = np.asarray(self._target_xy[1], dtype=float)
            if np.isfinite(ydata).any():
                peak = float(np.nanmax(ydata))
        self._modes.append(
            {"mu": mu, "sigma": fit.DEFAULT_SIGMA, "peak": peak, "bound": False}
        )
        self._fitted = False
        self._ensure_normalized()
        self._store_target_fit()
        self._write_modes_to_table()
        self.modes_table.selectRow(len(self._modes) - 1)
        self._draw()

    def _remove_mode(self) -> None:
        """Remove the selected mode."""
        row = self._selected_row()
        if row >= 0:
            del self._modes[row]
            self._fitted = False
            self._store_target_fit()
            self._write_modes_to_table()
            self._draw()

    def _clear_fit(self) -> None:
        """Drop all modes and any fit overlay (for the current target)."""
        self._modes = []
        self._fitted = False
        self._store_target_fit()
        self._write_modes_to_table()
        self.fit_status.setText("Fit cleared.")
        self._draw()

    def _ensure_normalized(self) -> None:
        """Force the dx/dlogDp view, the space lognormal fits live in."""
        if not self.normalize.isChecked():
            self.normalize.blockSignals(True)
            self.normalize.setChecked(True)
            self.normalize.blockSignals(False)

    # -- modes table <-> state --------------------------------------------
    def _write_modes_to_table(self) -> None:
        """Render ``self._modes`` into the table (errors shown as tooltips)."""
        self._building_table = True
        self.modes_table.blockSignals(True)
        self.modes_table.setRowCount(len(self._modes))
        for r, m in enumerate(self._modes):
            self._set_cell(r, _COL_MU, f"{m['mu']:.1f}", m.get("mu_err"))
            self._set_cell(r, _COL_SIGMA, f"{m['sigma']:.2f}", m.get("sigma_err"))
            self._set_cell(r, _COL_PEAK, f"{m['peak']:.3g}", None)
            bind = QtWidgets.QTableWidgetItem()
            bind.setFlags(
                QtCore.Qt.ItemIsUserCheckable
                | QtCore.Qt.ItemIsEnabled
                | QtCore.Qt.ItemIsSelectable
            )
            bind.setCheckState(
                QtCore.Qt.Checked if m.get("bound") else QtCore.Qt.Unchecked
            )
            bind.setToolTip(
                "Hold this mode's peak diameter near its value during the fit."
            )
            self.modes_table.setItem(r, _COL_BIND, bind)
        self.modes_table.blockSignals(False)
        self._building_table = False

    def _set_cell(self, row: int, col: int, text: str, err) -> None:
        """Set an editable numeric cell, with the fit's ±error as a tooltip."""
        item = QtWidgets.QTableWidgetItem(text)
        if err is not None and np.isfinite(err):
            item.setToolTip(f"± {err:.3g} (1σ fit uncertainty)")
        self.modes_table.setItem(row, col, item)

    def _on_mode_edited(self, item) -> None:
        """A typed/checked edit invalidates the optimised state; re-preview."""
        if self._building_table:
            return
        self._read_table_into_modes()
        self._fitted = False
        self._store_target_fit()
        self._draw()

    def _read_table_into_modes(self) -> None:
        """Rebuild ``self._modes`` from the table cells (dropping bad rows)."""
        modes = []
        for r in range(self.modes_table.rowCount()):
            try:
                mu = float(self.modes_table.item(r, _COL_MU).text())
                sigma = float(self.modes_table.item(r, _COL_SIGMA).text())
                peak = float(self.modes_table.item(r, _COL_PEAK).text())
            except (TypeError, ValueError, AttributeError):
                continue
            bind_item = self.modes_table.item(r, _COL_BIND)
            bound = (
                bind_item is not None and bind_item.checkState() == QtCore.Qt.Checked
            )
            modes.append({"mu": mu, "sigma": sigma, "peak": peak, "bound": bound})
        self._modes = modes

    def _valid_modes(self) -> list:
        """Modes with sane values as ``(index, mu, sigma, peak, bound)`` tuples."""
        return fit.valid_modes(self._modes)

    def _modes_as_triples(self) -> list:
        """Valid modes as ``(mu, sigma, factor)`` for plotting/evaluation."""
        return fit.modes_as_triples(self._modes)

    # -- fitting -----------------------------------------------------------
    def _run_fit(self) -> None:
        """Optimise the seeded modes on the active target via ``_psdfit.run_fit``."""
        target = self._current_target()
        if target is None:
            self.fit_status.setText("No fit target selected.")
            return
        ds, act = target
        try:
            tol = float(self.tolerance.text())
        except ValueError:
            tol = 10.0

        new, message = fit.run_fit(
            ds.obj,
            act,
            self._modes,
            log_scaling=self.log_scaling.isChecked(),
            tolerance=tol,
            local=self.local_chk.isChecked(),
        )
        if message is not None:  # nothing to fit, or the optimisation failed
            self.fit_status.setText(message)
            return

        self._modes = new
        self._fitted = True
        self._ensure_normalized()
        self._store_target_fit()
        self._write_modes_to_table()
        self._draw()
        self._report_fit_quality(len(new))

    def _report_fit_quality(self, n_modes: int) -> None:
        """Show the in-window R² and any degenerate-mode warning for the fit."""
        xy = self._target_xy
        if xy is None:
            self.fit_status.setText(f"Optimised {n_modes} mode(s).")
            return
        result = fit.fit_quality(
            xy[0], xy[1], self._modes, local=self.local_chk.isChecked()
        )
        if result is None:
            self.fit_status.setText(f"Optimised {n_modes} mode(s).")
            return
        r2, scope, flagged = result
        msg = f"Optimised {n_modes} mode(s) — R² ({scope}) = {r2:.3f}"
        if flagged:
            msg += f" — mode {', '.join(flagged)} degenerate; add a mode or bind μ."
        self.fit_status.setText(msg)

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        """Re-sync the dataset/activity/target lists, reload the fit and redraw."""
        self._sync_datasets()
        self._sync_activities()
        self._sync_fit_target()
        self._load_target_fit()
        self._draw()

    def _draw_on(self, ax) -> None:
        """Draw one mean-PSD curve per (dataset × activity), plus the fit overlay.

        While fitting, the target curve is drawn bold and named, the others dim
        (or are hidden when "Only" is ticked), and the target's lognormal modes
        and combined total are overlaid.
        """
        ax.clear()
        self._overlay_artists = []
        self._target_obj = self._target_color = self._target_line = None
        self._target_xy = None
        datasets = [d for d in self._datasets_2d if d.psd_on]
        activities = self._selected_activities()
        colors = _active_color_cycle()
        normalize = self.normalize.isChecked()
        show_band = self.band.isChecked()
        bars = self.display_mode.currentText() == "Bars"
        target = self.fit_target.currentData()
        isolate = self.isolate_chk.isChecked() and target is not None
        # Emphasise the target while there are modes or the user is editing, so a
        # plain comparison (nothing being fitted) looks unchanged.
        emphasize = target is not None and (
            bool(self._modes) or self.edit_btn.isChecked()
        )

        plotted = 0
        ci = 0
        single = len(datasets) == 1 or len(activities) == 1
        # When several datasets are compared, colour each by its own stable
        # dataset colour so it reads the same across every plot; when a single
        # dataset's activities are compared, cycle colours so they stay distinct.
        multi_ds = len(datasets) > 1
        for ds in datasets:
            for act in activities:
                if act not in ds.obj.activities:
                    continue
                is_target = (
                    target is not None and ds.id == target[0] and act == target[1]
                )
                if isolate and not is_target:
                    continue
                n_lines = len(list(ax.lines))
                n_coll = len(list(ax.collections))
                try:
                    ds.obj.plot_psd(activities=[act], normalize=normalize, ax=ax)
                except Exception:
                    continue
                new_lines = list(ax.lines)[n_lines:]
                new_coll = list(ax.collections)[n_coll:]
                if not new_lines:
                    continue  # activity had no samples in this dataset
                if multi_ds and ds.color:
                    color = ds.color
                else:
                    color = colors[ci % len(colors)]
                    ci += 1
                # Label by whichever dimension actually varies, to keep the
                # legend uncluttered when comparing along just one axis.
                if single and len(activities) == 1:
                    label = ds.label
                elif single and len(datasets) == 1:
                    label = act
                else:
                    label = f"{ds.label} – {act}"
                # Capture the target's (bin_mids, mean) for the fit overlay/R²,
                # independent of how the curve is drawn (line or bars).
                main = new_lines[0]
                xd = np.asarray(main.get_xdata(), dtype=float)
                yd = np.asarray(main.get_ydata(), dtype=float)
                if is_target:
                    self._target_obj = ds.obj
                    self._target_color = color
                    self._target_xy = (xd, yd)

                if bars:
                    for ln in new_lines:  # replace plot_psd's line with bars
                        ln.remove()
                    for coll in new_coll:  # fill_between reads poorly behind bars
                        coll.remove()
                    sigma = None
                    if show_band:
                        # Same mean/std computation as the ±σ band in line mode,
                        # shown here as error bars on top of each bar instead.
                        try:
                            _, _, sigma = ds.obj._activity_psd_stats(
                                act, normalize=normalize
                            )
                        except Exception:
                            sigma = None
                    self._draw_bars(
                        ax, ds.obj, xd, yd, color, label, is_target, emphasize, sigma
                    )
                    plotted += 1
                    continue

                for j, line in enumerate(new_lines):
                    line.set_color(color)
                    line.set_label(label if j == 0 else "_nolegend_")
                    if is_target and j == 0:
                        self._target_line = line
                    if is_target and emphasize:
                        line.set_linewidth(3.0)
                        line.set_zorder(5)
                    elif emphasize:
                        line.set_alpha(0.28)
                for coll in new_coll:  # fill_between ±1σ envelope
                    if show_band:
                        coll.set_color(color)
                        coll.set_alpha(0.18 if is_target or not emphasize else 0.08)
                    else:
                        coll.remove()
                plotted += 1

        if not plotted:
            ax.clear()
            msg = (
                "Tick at least one dataset and one activity to compare."
                if self._datasets_2d
                else "Load size-resolved (2D) datasets to compare their PSDs."
            )
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        # Name the curve being fitted, so it is unambiguous among several.
        if emphasize and self._target_obj is not None:
            ax.text(
                0.015,
                0.97,
                f"Fitting: {self.fit_target.currentText()}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.5,
                color=self._target_color,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc=ax.get_facecolor(),
                    ec=self._target_color,
                    alpha=0.85,
                ),
            )

        # Measure the y-range from the *data* artists (mean curves, bars and the
        # ±σ band when shown) before the fit overlay is painted, so the lognormal
        # fit's near-zero tails never drive the limits. This keeps the view tight
        # around the data even when the σ spread is huge or a log scale would
        # otherwise reach the fit's tiny values (see _apply_data_ylim).
        data_ylim = self._data_ylim(ax)

        self._paint_overlay(ax, remove_existing=False)

        if self.log_y.isChecked():
            ax.set_yscale("log")
        if data_ylim is not None:
            ax.set_ylim(*data_ylim)
        ax.legend(loc="upper right", fontsize=8)

    def _draw_bars(
        self, ax, obj, bin_mids, heights, color, label, is_target, emphasize, sigma=None
    ) -> None:
        """Draw one PSD as edge-aligned bars, so each size bin's width is visible.

        Bars are best for a single curve; when several are compared they overlap,
        so non-target curves are drawn faintly while fitting.

        Args:
            sigma: Per-bin standard deviation, or None. When given, drawn as an
                error bar anchored at the top of each bar (the ±σ band's
                equivalent in bar mode, where a fill_between would be hidden
                behind the bars).
        """
        edges = np.asarray(obj.bin_edges, dtype=float)
        heights = np.asarray(heights, dtype=float)
        if heights.size != edges.size - 1:
            # bin_edges/levels mismatch — fall back to a plain line.
            ax.plot(bin_mids, heights, color=color, lw=2.0, label=label)
            return
        alpha = 0.16 if (emphasize and not is_target) else 0.6
        ax.bar(
            edges[:-1],
            heights,
            width=np.diff(edges),
            align="edge",
            facecolor=color,
            edgecolor=color,
            linewidth=0.7,
            alpha=alpha,
            label=label,
            zorder=5 if (is_target and emphasize) else 3,
        )
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            centers = edges[:-1] + np.diff(edges) / 2
            finite = np.isfinite(heights) & np.isfinite(sigma)
            if finite.any():
                ax.errorbar(
                    centers[finite],
                    heights[finite],
                    yerr=sigma[finite],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=3,
                    alpha=0.8 if (is_target or not emphasize) else 0.3,
                    zorder=6 if (is_target and emphasize) else 4,
                )

    def _fit_window_spans(self, bin_mids) -> list:
        """X-intervals the lognormal fit uses, for the shaded fit-window band.

        With "Local" on this is each valid mode's local window
        (μ · σ^±:data:`_psdfit.FIT_LOCAL_SIGMAS`), merged where they overlap;
        with it off it is the whole measured size range. Clamped to the data
        range so the band never widens the view.
        """
        valid = self._valid_modes()
        if not valid:
            return []
        lo_data, hi_data = float(np.min(bin_mids)), float(np.max(bin_mids))
        if not self.local_chk.isChecked():
            return [(lo_data, hi_data)]
        raw = []
        for _idx, mu, sigma, _peak, _bound in valid:
            half = sigma**fit.FIT_LOCAL_SIGMAS
            raw.append((max(mu / half, lo_data), min(mu * half, hi_data)))
        raw.sort()
        merged = [list(raw[0])]
        for lo, hi in raw[1:]:
            if lo <= merged[-1][1]:  # overlaps the previous window -> extend it
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return [(lo, hi) for lo, hi in merged if hi > lo]

    def _paint_overlay(self, ax, remove_existing: bool = False) -> None:
        """Draw the modes + total on the target; track the artists for reuse.

        Manual (hand-built) fits are drawn dashed in orange; optimised fits solid
        in crimson — so the two states are never confused. The selected mode's
        peak gets a hollow handle.
        """
        if remove_existing:
            for artist in self._overlay_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
        self._overlay_artists = []
        obj = self._target_obj
        if obj is None or not self._modes:
            return
        if not self.normalize.isChecked():
            txt = ax.text(
                0.015,
                0.88,
                "Enable Normalize to overlay the fit.",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="crimson",
            )
            self._overlay_artists.append(txt)
            return
        valid = self._valid_modes()
        if not valid:
            return
        triples = self._modes_as_triples()
        bm = np.asarray(obj.bin_mids, dtype=float)

        # Light band(s) marking the x-window the fit actually uses, drawn behind
        # everything else. Makes it obvious when a mode is being fit to an
        # unsuitable region (or when "Local" narrows the fit to a poor slice of
        # the spectrum) — a common reason a fit fails or looks wrong.
        for lo, hi in self._fit_window_spans(bm):
            band = ax.axvspan(lo, hi, facecolor="#7f7f7f", alpha=0.12, lw=0, zorder=0)
            self._overlay_artists.append(band)

        dp = np.logspace(np.log10(bm.min()), np.log10(bm.max()), 300)
        total, per_mode = fit.lognormal_modes(dp, triples)
        optimized = self._fitted
        col = "crimson" if optimized else "darkorange"
        sel = self.modes_table.currentRow()

        for comp in per_mode:
            (ln,) = ax.plot(dp, comp, ls=":", lw=1.0, color=col, alpha=0.55, zorder=6)
            self._overlay_artists.append(ln)
        for idx, mu, _sigma, peak, _bound in valid:
            is_sel = idx == sel
            (mk,) = ax.plot(
                [mu],
                [peak],
                marker="o",
                ms=9 if is_sel else 6,
                mfc="white" if is_sel else col,
                mec=col,
                mew=1.6,
                ls="None",
                zorder=9,
            )
            self._overlay_artists.append(mk)
        (total_ln,) = ax.plot(
            dp,
            total,
            ls="-" if optimized else "--",
            lw=2.6,
            color=col,
            zorder=8,
            label="Optimised fit" if optimized else "Manual fit (not optimised)",
        )
        self._overlay_artists.append(total_ln)

    def _redraw_overlay(self) -> None:
        """Repaint only the fit overlay (modes unchanged → PSD curves kept)."""
        if self._target_obj is None or self._building_table:
            return
        self._paint_overlay(self.ax, remove_existing=True)
        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    def _data_ylim(self, ax):
        """Compute y-limits from the data artists currently on ``ax``.

        Called from :meth:`_draw_on` *before* the fit overlay is painted, so it
        sees only the data: the mean-PSD lines, the bars, and the ±σ band when
        it is shown. This is what fixes the "huge σ blows up the y-axis" and
        "log scale dives to 1e-24 to reach the fit's tail" problems — the limits
        track the plotted data plus a small margin, never the lognormal tails.

        Returns ``(bottom, top)`` or ``None`` when there is nothing to measure.
        """
        vals = []
        for line in ax.lines:
            vals.append(np.asarray(line.get_ydata(), dtype=float))
        for patch in ax.patches:  # bars: base .. base+height
            base = float(patch.get_y())
            vals.append(np.array([base, base + float(patch.get_height())]))
        for coll in ax.collections:  # ±σ fill_between envelopes
            for path in coll.get_paths():
                vals.append(np.asarray(path.vertices[:, 1], dtype=float))
        if not vals:
            return None
        allv = np.concatenate(vals)
        allv = allv[np.isfinite(allv)]
        if allv.size == 0:
            return None
        ymax = float(allv.max())
        if self.log_y.isChecked():
            pos = allv[allv > 0]
            if pos.size == 0 or ymax <= 0:
                return None
            # A little headroom each side, in log space.
            return float(pos.min()) * 0.8, ymax * 1.25
        ymin = float(allv.min())
        # Linear: anchor at 0 for the usual non-negative PSD, but follow the data
        # down if the ±σ band dips negative. Margin is ~5% of the visible span.
        span = ymax - min(0.0, ymin)
        margin = 0.05 * span if span > 0 else (0.05 * abs(ymax) or 1.0)
        bottom = 0.0 if ymin >= 0 else ymin - margin
        return bottom, ymax + margin

    def _draw(self, preserve: bool = False, force_rescale: bool = False) -> None:
        """Redraw onto the embedded axis, reporting any error in the figure.

        Args:
            preserve: Keep the current zoom/pan across the redraw instead of
                letting the data-driven limits take over. Always on while editing
                a fit (so placing/typing modes never snaps the view back), and
                requested explicitly when toggling editing on/off.
            force_rescale: Autoscale to the data even while editing — used when
                the set of visible curves changes (e.g. the "Only" toggle) so the
                limits follow the shown data rather than the hidden ones.
        """
        preserve = (preserve or self.edit_btn.isChecked()) and not force_rescale
        prev = None
        if preserve and self.ax.has_data():
            prev = (self.ax.get_xlim(), self.ax.get_ylim())
        try:
            self._draw_on(self.ax)
        except Exception:
            self._show_message(
                "Could not draw combined PSD:\n" + traceback.format_exc(limit=1)
            )
            return
        if prev is not None:
            self.ax.set_xlim(prev[0])
            self.ax.set_ylim(prev[1])
        else:
            self._sync_toolbar_home()
        self.canvas.draw_idle()
