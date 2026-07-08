"""Single-dataset PSD tab with lognormal fitting.

Shows **one** mean particle size distribution at a time — the active dataset's,
for a task chosen from a drop-down — and hosts the lognormal fitting. Because
only one PSD is ever shown, the fitting rules stay simple: the user shapes one
or more modes by eye (click sets a mode's peak μ/height, scrolling sets its
width σ) and may press **Fit** to optimise them. Manual and optimised states are
drawn differently. Each fit is stored per (dataset × task) on the dataset and
saved with the project. Comparing several datasets/activities is done in the
separate PSD comparison pane (which has no fitting).
"""

from __future__ import annotations

import traceback

import numpy as np

from .. import helpers
from ..qt import QtCore, QtWidgets
from . import _psddraw as draw
from . import _psdfit as fit
from ._base import _active_color_cycle, _PlotTab

# Modes table columns.
_COL_MU, _COL_SIGMA, _COL_PEAK, _COL_BIND = range(4)


class SinglePSDTab(_PlotTab):
    """Show one dataset's mean PSD for a chosen task and fit lognormal modes."""

    export_tag = "psd"

    def __init__(self, main):
        """Build the PSD display controls, task selector, fit panel and plot."""
        super().__init__(main, nrows=1)

        self._modes: list[dict] = []
        self._fitted = False
        self._building_table = False
        self._target_obj = None
        self._target_xy = None  # (bin_mids, mean) of the shown PSD, for fit/R²
        self._overlay_artists: list = []

        self.controls.addWidget(QtWidgets.QLabel("Task:"))
        self.task = QtWidgets.QComboBox()
        self.task.setToolTip("Which activity of the active dataset to show and fit.")
        self.task.currentIndexChanged.connect(self._on_task_changed)
        self.controls.addWidget(self.task)

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.setChecked(True)
        self.normalize.setToolTip(
            "Plot dx/dlogDp. Lognormal fits are defined in this space, so it is "
            "enabled automatically while fitting."
        )
        self.normalize.stateChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.normalize)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.stateChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.log_y)

        self.band = QtWidgets.QCheckBox("±σ band")
        self.band.setToolTip("Shade the curve's ±1σ spread.")
        self.band.stateChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.band)

        self.controls.addWidget(QtWidgets.QLabel("Display:"))
        self.display_mode = QtWidgets.QComboBox()
        self.display_mode.addItems(["Bars", "Lines"])
        self.display_mode.setToolTip(
            "Bars show each size bin's width (good for a single PSD); lines are "
            "smoother for reading off a fit."
        )
        self.display_mode.currentIndexChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.display_mode)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self._build_fit_panel(), stretch=0)
        side.addStretch(1)
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        self._split_with_side(side_widget, sizes=(760, 320))

        self.ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    # -- fit panel construction -------------------------------------------
    def _build_fit_panel(self) -> QtWidgets.QGroupBox:
        """Construct the compact "Lognormal fit" box (modes + actions)."""
        box = QtWidgets.QGroupBox("Lognormal fit")
        outer = QtWidgets.QVBoxLayout(box)
        outer.setSpacing(4)
        outer.setContentsMargins(6, 4, 6, 6)

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
        self.modes_table.setMinimumHeight(80)
        self.modes_table.setMaximumHeight(160)
        self.modes_table.itemSelectionChanged.connect(self._redraw_overlay)
        self.modes_table.itemChanged.connect(self._on_mode_edited)
        outer.addWidget(self.modes_table)

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
        add_btn.setToolTip("Add a mode at the mid-range, then shape it.")
        add_btn.clicked.connect(self._add_mode)
        rem_btn = QtWidgets.QPushButton("Del")
        rem_btn.setToolTip("Remove the selected mode.")
        rem_btn.clicked.connect(self._remove_mode)
        row2.addWidget(self.edit_btn, stretch=1)
        row2.addWidget(add_btn)
        row2.addWidget(rem_btn)
        outer.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        row3.setSpacing(4)
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("primary")
        self.fit_btn.setToolTip("Optimise the current modes to the shown curve.")
        self.fit_btn.clicked.connect(self._run_fit)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setToolTip("Remove all modes and the fit overlay.")
        clear_btn.clicked.connect(self._clear_fit)
        self.local_chk = QtWidgets.QCheckBox("Local")
        self.local_chk.setChecked(True)
        self.local_chk.setToolTip(
            "Fit each mode only to the bins near it (a window scaled by its "
            "width), so modes follow the peaks. The shaded band marks the window."
        )
        self.local_chk.stateChanged.connect(lambda *_: self._redraw_overlay())
        self.log_scaling = QtWidgets.QCheckBox("Log")
        self.log_scaling.setChecked(True)
        self.log_scaling.setToolTip(
            "Fit against log10(dx/dlogDp) so weakly populated modes are not "
            "swamped by the dominant mode (recommended)."
        )
        self.tolerance = QtWidgets.QLineEdit("10")
        self.tolerance.setFixedWidth(36)
        self.tolerance.setToolTip("Percent window a bound μ may move within.")
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

    # -- target (active dataset × task) -----------------------------------
    @property
    def _active_ds(self):
        """The active dataset, or None."""
        return self.main.project.active

    def _current_task(self) -> str:
        """The selected task name (falling back to 'All data')."""
        return self.task.currentText() or "All data"

    def _sync_task(self) -> None:
        """Repopulate the task drop-down from the active dataset's activities."""
        obj = self.obj
        self.task.blockSignals(True)
        current = self.task.currentText()
        self.task.clear()
        names = (
            list(getattr(obj, "activities", ["All data"])) if obj is not None else []
        )
        if "All data" not in names:
            names = ["All data"] + names
        for name in names:
            self.task.addItem(name)
        idx = self.task.findText(current)
        self.task.setCurrentIndex(idx if idx >= 0 else 0)
        self.task.blockSignals(False)

    def _load_target_fit(self) -> None:
        """Load the current (dataset × task) stored modes into the working state."""
        ds = self._active_ds
        if ds is None:
            self._modes, self._fitted = [], False
            self._write_modes_to_table()
            return
        rec = ds.psd_fits.get(self._current_task())
        if rec and rec.get("modes"):
            self._modes = fit.clean_modes(rec["modes"])
            self._fitted = bool(rec.get("optimized"))
        else:
            self._modes, self._fitted = [], False
        self._write_modes_to_table()

    def _store_target_fit(self) -> None:
        """Persist the working modes onto the current (dataset × task)."""
        ds = self._active_ds
        if ds is None:
            return
        task = self._current_task()
        if self._modes:
            ds.psd_fits[task] = {
                "modes": fit.clean_modes(self._modes),
                "optimized": bool(self._fitted),
            }
        else:
            ds.psd_fits.pop(task, None)

    def _on_task_changed(self, _index: int = 0) -> None:
        """Task changed: load that task's stored modes and redraw."""
        self._load_target_fit()
        self._draw()

    # -- interaction -------------------------------------------------------
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
        sigma = float(m["sigma"]) * (1.06 ** (-event.step))
        m["sigma"] = float(min(fit.SIGMA_MAX, max(fit.SIGMA_MIN, sigma)))
        m.pop("sigma_err", None)
        self._fitted = False
        self._ensure_normalized()
        self._store_target_fit()
        self._building_table = True
        self.modes_table.blockSignals(True)
        cell = self.modes_table.item(row, _COL_SIGMA)
        if cell is not None:
            cell.setText(f"{m['sigma']:.2f}")
        self.modes_table.blockSignals(False)
        self._building_table = False
        self._redraw_overlay()

    def _add_mode(self) -> None:
        """Append a default mode centred on the dataset's size range."""
        obj = self.obj
        if obj is None or not helpers.is_2d(obj):
            return
        bm = np.asarray(obj.bin_mids, dtype=float)
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
        """Drop all modes and any fit overlay (for the current task)."""
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
            bind.setToolTip("Hold this mode's peak diameter near its value in the fit.")
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
        """Optimise the seeded modes on the shown PSD via ``_psdfit.run_fit``."""
        obj = self.obj
        if obj is None or not helpers.is_2d(obj):
            self.fit_status.setText("No size-resolved dataset to fit.")
            return
        act = self._current_task()
        try:
            tol = float(self.tolerance.text())
        except ValueError:
            tol = 10.0
        new, message = fit.run_fit(
            obj,
            act,
            self._modes,
            log_scaling=self.log_scaling.isChecked(),
            tolerance=tol,
            local=self.local_chk.isChecked(),
        )
        if message is not None:
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
        """Re-sync the task list, reload the fit and redraw."""
        self._sync_task()
        self._load_target_fit()
        self._draw()

    def _draw_on(self, ax) -> None:
        """Draw the active dataset's mean PSD for the task, plus the fit overlay."""
        ax.clear()
        self._overlay_artists = []
        self._target_obj = None
        self._target_xy = None
        obj = self.obj
        if obj is None or not helpers.is_2d(obj):
            ax.text(
                0.5,
                0.5,
                "Select a size-resolved (2D) dataset to view its PSD.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        act = self._current_task()
        if act not in obj.activities:
            ax.text(
                0.5,
                0.5,
                f"No samples for '{act}' in this dataset.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        ds = self._active_ds
        cycle = _active_color_cycle()
        color = (ds.color if ds is not None else None) or cycle[0]
        xy = draw.draw_psd_curve(
            ax,
            obj,
            act,
            color=color,
            label=act,
            normalize=self.normalize.isChecked(),
            bars=self.display_mode.currentText() == "Bars",
            show_band=self.band.isChecked(),
        )
        if xy is None:
            ax.text(
                0.5,
                0.5,
                f"No samples for '{act}' in this dataset.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        self._target_obj = obj
        self._target_xy = xy

        # Measure the y-range from the data before the fit overlay is painted, so
        # the lognormal fit's near-zero tails never drive the limits.
        data_ylim = draw.data_ylim(ax, self.log_y.isChecked())
        self._paint_overlay(ax)
        if self.log_y.isChecked():
            ax.set_yscale("log")
        if data_ylim is not None:
            ax.set_ylim(*data_ylim)
        ax.legend(loc="upper right", fontsize=8)

    def _fit_window_spans(self, bin_mids) -> list:
        """X-intervals the lognormal fit uses, for the shaded fit-window band."""
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
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return [(lo, hi) for lo, hi in merged if hi > lo]

    def _paint_overlay(self, ax, remove_existing: bool = False) -> None:
        """Draw the modes + total on the shown curve; track the artists for reuse."""
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
        """Repaint only the fit overlay (modes unchanged → PSD curve kept)."""
        if self._target_obj is None or self._building_table:
            return
        self._paint_overlay(self.ax, remove_existing=True)
        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    def _draw(self, preserve: bool = False) -> None:
        """Redraw onto the embedded axis, reporting any error in the figure.

        Keeps the current zoom/pan while a fit is being edited (so placing/typing
        modes never snaps the view back).
        """
        preserve = preserve or self.edit_btn.isChecked()
        prev = None
        if preserve and self.ax.has_data():
            prev = (self.ax.get_xlim(), self.ax.get_ylim())
        try:
            self._draw_on(self.ax)
        except Exception:
            self._show_message("Could not draw PSD:\n" + traceback.format_exc(limit=1))
            return
        if prev is not None:
            self.ax.set_xlim(prev[0])
            self.ax.set_ylim(prev[1])
        else:
            self._sync_toolbar_home()
        self.canvas.draw_idle()
