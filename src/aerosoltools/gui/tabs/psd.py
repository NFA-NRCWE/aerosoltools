"""Particle size distribution (PSD) tab.

Works for one *or* several datasets: it reads the project's size-resolved (2D)
datasets and overlays one mean-PSD curve per ticked (dataset × activity), so a
single instrument behaves like the old single-view PSD tab while multiple
instruments are compared on the same axes.

On top of the comparison it offers **lognormal fitting** (via
:meth:`Aerosol2D.fit_psd`): the user picks one of the plotted curves as the *fit
target*, seeds one or more modes (by clicking the plot or typing μ/σ/peak), and
fits them. Each mode plus the combined *total* fit is overlaid on the target.
"""

from __future__ import annotations

import traceback

import numpy as np

from .. import helpers
from ..qt import QtCore, QtWidgets
from ._base import _active_color_cycle, _PlotTab

# A lognormal mode is parameterised here by its peak diameter ``mu`` (nm), its
# geometric standard deviation ``sigma`` (dimensionless), and the *peak height*
# of the dx/dlogDp curve. :meth:`Aerosol2D.fit_psd` instead works with a
# ``factor`` scaling parameter; the two are related by the value of the
# lognormal at its peak, ``peak = factor / (sqrt(2π) · log10(sigma))``. Storing
# the peak height (rather than ``factor``) lets the user read and click a value
# that matches the curve, and lets a single click seed a mode that visually
# peaks where the user clicked.
_SQRT_2PI = np.sqrt(2.0 * np.pi)
#: Default geometric standard deviation for a freshly added/placed mode.
_DEFAULT_SIGMA = 1.8


def _peak_to_factor(peak: float, sigma: float) -> float:
    """Convert a desired dx/dlogDp peak height to ``fit_psd``'s ``factor``."""
    return float(peak) * _SQRT_2PI * np.log10(sigma)


def _factor_to_peak(factor: float, sigma: float) -> float:
    """Convert ``fit_psd``'s ``factor`` back to a dx/dlogDp peak height."""
    return float(factor) / (_SQRT_2PI * np.log10(sigma))


def _lognormal_modes(dp_nm, modes):
    """Evaluate a sum of lognormal modes (dx/dlogDp) at diameters ``dp_nm``.

    Mirrors the lognormal used inside :meth:`Aerosol2D.fit_psd` so the overlay
    matches the fit exactly.

    Args:
        dp_nm: Diameters in nm to evaluate at.
        modes: Iterable of ``(mu, sigma, factor)`` triples.

    Returns:
        ``(total, per_mode)`` where ``total`` is the summed curve and
        ``per_mode`` is the list of each mode's individual curve.
    """
    x = np.log10(np.asarray(dp_nm, dtype=float))
    total = np.zeros_like(x)
    per_mode = []
    for mu, sigma, factor in modes:
        s = np.log10(sigma)
        comp = factor * (1.0 / (_SQRT_2PI * s)) * np.exp(-((x - np.log10(mu)) ** 2) / (2.0 * s**2))
        per_mode.append(comp)
        total = total + comp
    return total, per_mode


# Modes table columns.
_COL_MU, _COL_SIGMA, _COL_PEAK, _COL_BIND = range(4)


class PSDTab(_PlotTab):
    """Overlay mean particle size distributions and fit lognormal modes.

    Reads *all* of the project's size-resolved (2D) datasets rather than the
    active one. The user ticks which datasets and which activities to compare;
    one mean-PSD curve is drawn per (dataset × activity) pair via the library's
    own :meth:`Aerosol2D.plot_psd` (``ax=`` shared), then relabelled/recoloured
    so each combination is distinct. With a single dataset ticked it is just
    that instrument's PSD per activity.

    A "Lognormal fit" panel on top lets the user select one plotted curve as the
    *fit target* (highlighted; the rest dim), seed modes by clicking the plot,
    and fit them with :meth:`Aerosol2D.fit_psd`. Each mode and the combined
    total fit are overlaid on the target.
    """

    export_tag = "psd"

    def __init__(self, main):
        """Build the PSD controls, fit panel, dataset/activity lists and plot."""
        super().__init__(main, nrows=1)

        # Fitting state: the working list of modes (dicts with mu/sigma/peak/
        # bound and optional *_err from the last fit), whether they represent a
        # committed fit (vs an uncommitted guess), and the currently emphasised
        # target line captured during the last draw.
        self._modes: list[dict] = []
        self._fitted = False
        self._building_table = False
        self._target_line = None

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.setChecked(True)
        self.normalize.setToolTip(
            "Plot dx/dlogDp. Lognormal fits are defined in this space, so it is "
            "enabled automatically while fitting."
        )
        self.normalize.stateChanged.connect(self._draw)
        self.controls.addWidget(self.normalize)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.stateChanged.connect(self._draw)
        self.controls.addWidget(self.log_y)

        self.band = QtWidgets.QCheckBox("±σ band")
        self.band.setToolTip(
            "Shade each curve's ±1σ spread. Off by default, since the bands "
            "overlap heavily when several curves are compared."
        )
        self.band.stateChanged.connect(self._draw)
        self.controls.addWidget(self.band)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Side panel: datasets to compare (checkable) + activities (multi-select).
        self._building = False
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.itemChanged.connect(self._on_ds_changed)

        self.act_list = QtWidgets.QListWidget()
        self.act_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.act_list.itemSelectionChanged.connect(self._on_targets_changed)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Datasets to compare:"))
        side.addWidget(self.ds_list, stretch=1)
        side.addWidget(QtWidgets.QLabel("Activities:"))
        side.addWidget(self.act_list, stretch=1)
        hint = QtWidgets.QLabel(
            "Tick the 2D datasets and the activities to compare. One mean-PSD "
            "curve is drawn per dataset × activity."
        )
        hint.setWordWrap(True)
        side.addWidget(hint)

        # Plot on the left of a resizable divider, side panel on the right.
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        self._split_with_side(side_widget)

        # Fitting top panel, above the view controls in the left column.
        self._left_col.insertWidget(0, self._build_fit_panel())

        self.ax = self.figure.add_subplot(111)
        # Click-to-place seeds/moves a mode at the clicked diameter and height.
        self.canvas.mpl_connect("button_press_event", self._on_click)

    # -- fit panel construction -------------------------------------------
    def _build_fit_panel(self) -> QtWidgets.QGroupBox:
        """Construct the "Lognormal fit" control box (target + modes + actions)."""
        box = QtWidgets.QGroupBox("Lognormal fit")
        outer = QtWidgets.QVBoxLayout(box)
        outer.setSpacing(6)

        # Row 1: which plotted curve to fit.
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Fit target:"))
        self.fit_target = QtWidgets.QComboBox()
        self.fit_target.setToolTip(
            "Which plotted curve the fit applies to. The target is drawn bold; "
            "the other curves dim."
        )
        self.fit_target.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.fit_target.currentIndexChanged.connect(self._on_target_changed)
        row1.addWidget(self.fit_target, stretch=1)
        outer.addLayout(row1)

        # Modes table: μ / σ / peak height (all editable) + a "bind μ" check.
        self.modes_table = QtWidgets.QTableWidget(0, 4)
        self.modes_table.setHorizontalHeaderLabels(
            ["μ peak (nm)", "σ (GSD)", "Peak height", "Bind μ"]
        )
        self.modes_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.modes_table.verticalHeader().setVisible(False)
        self.modes_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.modes_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.modes_table.setMinimumHeight(96)
        self.modes_table.setMaximumHeight(160)
        self.modes_table.itemChanged.connect(self._on_mode_edited)
        outer.addWidget(self.modes_table)

        # Row 2: place-by-click toggle + add/remove mode.
        row2 = QtWidgets.QHBoxLayout()
        self.place_btn = QtWidgets.QPushButton("Place mode (click plot)")
        self.place_btn.setObjectName("toggle")
        self.place_btn.setCheckable(True)
        self.place_btn.setToolTip(
            "Toggle on, then click the plot to set the selected mode's peak "
            "diameter and height (or add a new mode if none is selected)."
        )
        self.place_btn.toggled.connect(self._toggle_place)
        add_btn = QtWidgets.QPushButton("Add mode")
        add_btn.clicked.connect(self._add_mode)
        rem_btn = QtWidgets.QPushButton("Remove")
        rem_btn.clicked.connect(self._remove_mode)
        row2.addWidget(self.place_btn, stretch=1)
        row2.addWidget(add_btn)
        row2.addWidget(rem_btn)
        outer.addLayout(row2)

        # Row 3: fitting options.
        row3 = QtWidgets.QHBoxLayout()
        self.log_scaling = QtWidgets.QCheckBox("Log-scaled fit")
        self.log_scaling.setChecked(True)
        self.log_scaling.setToolTip(
            "Fit against log10(dx/dlogDp) so weakly populated modes are not "
            "swamped by the dominant mode (recommended)."
        )
        row3.addWidget(self.log_scaling)
        row3.addWidget(QtWidgets.QLabel("Bind tol. %:"))
        self.tolerance = QtWidgets.QLineEdit("10")
        self.tolerance.setFixedWidth(48)
        self.tolerance.setToolTip(
            "Percent window a bound μ may move within during the fit."
        )
        row3.addWidget(self.tolerance)
        row3.addStretch(1)
        outer.addLayout(row3)

        # Row 4: fit / clear.
        row4 = QtWidgets.QHBoxLayout()
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("primary")
        self.fit_btn.clicked.connect(self._run_fit)
        clear_btn = QtWidgets.QPushButton("Clear fit")
        clear_btn.clicked.connect(self._clear_fit)
        row4.addWidget(self.fit_btn, stretch=1)
        row4.addWidget(clear_btn)
        outer.addLayout(row4)

        self.fit_status = QtWidgets.QLabel(
            "Pick a fit target, then click the plot to seed a mode and press Fit."
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
        self.fit_target.setEnabled(has_target)
        self.fit_btn.setEnabled(has_target)
        self.place_btn.setEnabled(has_target)

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
        """Re-sync the fit-target dropdown then redraw (dataset/activity change)."""
        self._sync_fit_target()
        self._draw()

    def _on_target_changed(self, _index: int = 0) -> None:
        """The fit target changed: the modes are now a guess for the new curve."""
        self._fitted = False
        self._draw()

    def _toggle_place(self) -> None:
        """Enter/leave click-to-place mode (and drop any active pan/zoom)."""
        active = self.place_btn.isChecked()
        self.place_btn.setText("Placing… (click plot)" if active else "Place mode (click plot)")
        if active and self.toolbar.mode:
            mode = str(self.toolbar.mode)
            if "pan" in mode:
                self.toolbar.pan()
            elif "zoom" in mode:
                self.toolbar.zoom()

    def _on_click(self, event) -> None:
        """Seed or move a mode at the clicked (diameter, height)."""
        if not self.place_btn.isChecked() or event.inaxes is not self.ax:
            return
        if event.button != 1 or self.toolbar.mode:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None or x <= 0 or y <= 0:
            return
        row = self.modes_table.currentRow()
        if 0 <= row < len(self._modes):
            self._modes[row]["mu"] = float(x)
            self._modes[row]["peak"] = float(y)
            self._modes[row].pop("mu_err", None)
        else:
            self._modes.append(
                {"mu": float(x), "sigma": _DEFAULT_SIGMA, "peak": float(y), "bound": False}
            )
        self._fitted = False
        self._ensure_normalized()
        self._write_modes_to_table()
        self._draw()

    def _add_mode(self) -> None:
        """Append a default mode centred on the target's size range."""
        target = self._current_target()
        mu, peak = 100.0, 1.0
        if target is not None:
            bm = np.asarray(target[0].obj.bin_mids, dtype=float)
            mu = float(np.sqrt(bm.min() * bm.max()))
        if self._target_line is not None:
            ydata = np.asarray(self._target_line.get_ydata(), dtype=float)
            if np.isfinite(ydata).any():
                peak = float(np.nanmax(ydata))
        self._modes.append({"mu": mu, "sigma": _DEFAULT_SIGMA, "peak": peak, "bound": False})
        self._fitted = False
        self._ensure_normalized()
        self._write_modes_to_table()
        self.modes_table.selectRow(len(self._modes) - 1)
        self._draw()

    def _remove_mode(self) -> None:
        """Remove the selected mode."""
        row = self.modes_table.currentRow()
        if 0 <= row < len(self._modes):
            del self._modes[row]
            self._fitted = False
            self._write_modes_to_table()
            self._draw()

    def _clear_fit(self) -> None:
        """Drop all modes and any fit overlay."""
        self._modes = []
        self._fitted = False
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
            bind.setToolTip("Hold this mode's peak diameter near its value during the fit.")
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
        """A typed/checked edit invalidates the committed fit; re-preview."""
        if self._building_table:
            return
        self._read_table_into_modes()
        self._fitted = False
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
            bound = bind_item is not None and bind_item.checkState() == QtCore.Qt.Checked
            modes.append({"mu": mu, "sigma": sigma, "peak": peak, "bound": bound})
        self._modes = modes

    def _valid_modes(self) -> list:
        """Modes with sane values as ``(mu, sigma, peak, bound)`` tuples."""
        out = []
        for m in self._modes:
            try:
                mu, sigma, peak = float(m["mu"]), float(m["sigma"]), float(m["peak"])
            except (TypeError, ValueError):
                continue
            if mu > 0 and sigma > 1.0 and peak > 0:
                out.append((mu, sigma, peak, bool(m.get("bound"))))
        return out

    def _modes_as_triples(self) -> list:
        """Valid modes as ``(mu, sigma, factor)`` for plotting/evaluation."""
        return [(mu, sigma, _peak_to_factor(peak, sigma)) for mu, sigma, peak, _ in self._valid_modes()]

    # -- fitting -----------------------------------------------------------
    def _run_fit(self) -> None:
        """Fit the seeded modes to the active target via ``Aerosol2D.fit_psd``."""
        target = self._current_target()
        if target is None:
            self.fit_status.setText("No fit target selected.")
            return
        ds, act = target
        valid = self._valid_modes()
        if not valid:
            self.fit_status.setText("Add at least one mode first (click the plot or 'Add mode').")
            return

        mu = [v[0] for v in valid]
        sigma = [v[1] for v in valid]
        factor = [_peak_to_factor(v[2], v[1]) for v in valid]
        binding = []
        for v in valid:
            binding += [v[3], False, False]  # bind μ only
        try:
            tol = float(self.tolerance.text())
        except ValueError:
            tol = 10.0

        try:
            modes, err = ds.obj.fit_psd(
                period=act,
                mu=mu,
                sigma=sigma,
                factor=factor,
                log_scaling=self.log_scaling.isChecked(),
                binding=binding,
                tolerance=tol,
            )
        except Exception as exc:
            self.fit_status.setText(f"Fit failed: {exc}")
            return

        new = []
        for i in range(len(modes["mu"])):
            ss = float(modes["sigma"][i])
            ff = float(modes["factor"][i])
            new.append(
                {
                    "mu": float(modes["mu"][i]),
                    "sigma": ss,
                    "peak": _factor_to_peak(ff, ss),
                    "bound": valid[i][3] if i < len(valid) else False,
                    "mu_err": float(err["mu"][i]),
                    "sigma_err": float(err["sigma"][i]),
                    "factor_err": float(err["factor"][i]),
                }
            )
        self._modes = new
        self._fitted = True
        self._ensure_normalized()
        self._write_modes_to_table()
        self._draw()
        self._report_fit_quality(len(new))

    def _report_fit_quality(self, n_modes: int) -> None:
        """Show an R² (in log space) of the total fit against the target curve."""
        line = self._target_line
        triples = self._modes_as_triples()
        if line is None or not triples:
            self.fit_status.setText(f"Fitted {n_modes} mode(s).")
            return
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if mask.sum() < 3:
            self.fit_status.setText(f"Fitted {n_modes} mode(s).")
            return
        yhat, _ = _lognormal_modes(x[mask], triples)
        yt = np.log10(y[mask])
        yp = np.log10(np.clip(yhat, 1e-30, None))
        ss_res = np.nansum((yt - yp) ** 2)
        ss_tot = np.nansum((yt - np.nanmean(yt)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Flag degenerate modes so a poor fit is obvious: a σ pinned near the
        # solver's ceiling (≈5) or a peak that landed outside the measured size
        # range usually means the data needs another mode (or a bound μ).
        lo, hi = float(np.min(x[mask])), float(np.max(x[mask]))
        warns = []
        for i, m in enumerate(self._modes, start=1):
            if m["sigma"] >= 4.5:
                warns.append(f"mode {i} σ≈{m['sigma']:.1f} (very broad)")
            if m["mu"] < 0.9 * lo or m["mu"] > 1.1 * hi:
                warns.append(f"mode {i} μ={m['mu']:.0f} nm off-range")
        msg = f"Fitted {n_modes} mode(s) — R² (log) = {r2:.3f}"
        if warns:
            msg += " — check: " + "; ".join(warns) + ". Try another mode or bind μ."
        self.fit_status.setText(msg)

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        """Re-sync the dataset/activity/target lists and redraw."""
        self._sync_datasets()
        self._sync_activities()
        self._sync_fit_target()
        self._draw()

    def _draw_on(self, ax) -> None:
        """Draw one mean-PSD curve per (dataset x activity) onto ``ax``.

        The current fit target (if any) is emphasised and its lognormal modes +
        total fit overlaid; the other curves are dimmed.
        """
        ax.clear()
        self._target_line = None
        datasets = [d for d in self._datasets_2d if d.psd_on]
        activities = self._selected_activities()
        colors = _active_color_cycle()
        normalize = self.normalize.isChecked()
        show_band = self.band.isChecked()
        target = self.fit_target.currentData()

        plotted = 0
        ci = 0
        single = len(datasets) == 1 or len(activities) == 1
        target_obj = None
        target_color = None
        # Only emphasise the target / dim the rest while the user is fitting, so
        # a plain comparison (no modes seeded) looks unchanged.
        emphasize = bool(self._modes) and target is not None
        for ds in datasets:
            for act in activities:
                if act not in ds.obj.activities:
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
                is_target = target is not None and ds.id == target[0] and act == target[1]
                for j, line in enumerate(new_lines):
                    line.set_color(color)
                    line.set_label(label if j == 0 else "_nolegend_")
                    if is_target and emphasize:
                        line.set_linewidth(2.6)
                        line.set_zorder(5)
                        if j == 0:
                            self._target_line = line
                            target_obj = ds.obj
                            target_color = color
                    elif emphasize:
                        line.set_alpha(0.35)
                for coll in new_coll:  # fill_between ±1σ envelope
                    if show_band:
                        coll.set_color(color)
                        coll.set_alpha(0.18 if is_target or not emphasize else 0.10)
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

        if target_obj is not None and self._modes:
            self._draw_fit_overlay(ax, target_obj, target_color, normalize)

        if self.log_y.isChecked():
            ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)

    def _draw_fit_overlay(self, ax, obj, color, normalize: bool) -> None:
        """Overlay each lognormal mode and the combined total on the target."""
        triples = self._modes_as_triples()
        if not triples:
            return
        if not normalize:
            ax.text(
                0.02,
                0.98,
                "Enable Normalize (dx/dlogDp) to overlay the fit.",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="crimson",
            )
            return
        bm = np.asarray(obj.bin_mids, dtype=float)
        dp = np.logspace(np.log10(bm.min()), np.log10(bm.max()), 300)
        total, per_mode = _lognormal_modes(dp, triples)
        for comp in per_mode:
            ax.plot(dp, comp, ls="--", lw=1.1, color=color or "0.5", alpha=0.6)
        committed = self._fitted
        ax.plot(
            dp,
            total,
            ls="-" if committed else ":",
            lw=2.4,
            color="crimson",
            zorder=6,
            label="Total fit" if committed else "Total fit (guess)",
        )

    def _draw(self) -> None:
        """Redraw onto the embedded axis, reporting any error in the figure."""
        try:
            self._draw_on(self.ax)
        except Exception:
            self._show_message(
                "Could not draw combined PSD:\n" + traceback.format_exc(limit=1)
            )
            return
        self.canvas.draw_idle()

    def _render_export(self, fig) -> None:
        """Draw the comparison (and any fit overlay) onto a fresh export figure."""
        self._draw_on(fig.add_subplot(111))
