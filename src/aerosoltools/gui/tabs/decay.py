"""Decay / source-strength fitting tab.

Fits a single-zone emission + decay peak model (see
:mod:`aerosoltools._core.decay`) to windows of a chosen metric on the active
dataset. **Several** decays can be fitted on one plot: each marked window is a
persistent fit (saved with the project), shown as a shaded region plus its fit
curve. In **Adjust** mode a fit is selected by clicking inside its region — its
draggable handles appear and its parameters load into the fields above the plot
— and deselected by clicking outside it (or pressing Esc); with Adjust off the
mouse zooms/pans the plot. A results table below the canvas lists every fit's
parameters (and, optionally, a per-size-bin breakdown) and can be exported.
"""

from __future__ import annotations

import contextlib
import io
import traceback

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.widgets import SpanSelector

from ..._core import decay as _decay
from ..logic import helpers
from ..qt import QtCore, QtWidgets
from ._base import _export_table, _PlotTab

#: Fraction metrics that need a cut-off diameter alongside the kind.
_FRACTION_KINDS = ("PM", "PN", "PS", "PV")

#: Model label shown in the selector -> value passed to ``fit_decay``.
_MODEL_CHOICES = [
    ("Auto (best fit)", "auto"),
    ("Zeroth order (constant removal)", "zeroth_order"),
    ("First order (air exchange + wall loss)", "first_order"),
    ("Second order (coagulation)", "second_order"),
    ("Combined (first + second order)", "combined"),
]

#: Map digits and a sign to Unicode superscripts for exponent rendering.
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

#: Colours for the draggable emission-start / source-off markers.
_T0_COLOR = "#2e8b57"
_PEAK_COLOR = "#d1495b"
#: Colour of the draggable background (P0) line.
_BG_COLOR = "#7f7f7f"
#: Curve colour for a manual guess preview (vs the optimised fit).
_GUESS_COLOR = "#e08214"
#: Fill colour of a fit's marked region (selected brighter than the rest).
_REGION_COLOR = "#4c72b0"

#: Factor a single scroll notch multiplies/divides the decay rate by.
_SCROLL_RATE_STEP = 1.15

#: Per-bin fits below this R² are flagged as too poor and excluded.
_MIN_BIN_R2 = 0.5


def _fmt(value, sig: int = 4) -> str:
    """Format a number, rendering scientific notation as ``m×10ⁿ``."""
    if value is None or not np.isfinite(value):
        return "n/a"
    text = f"{value:.{sig}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        power = str(int(exponent))
        return f"{mantissa}×10{power.translate(_SUPERSCRIPT)}"
    return text


def _fmt_pm(value, err, conv: float = 1.0) -> str:
    """Format ``value`` with an optional ``± err`` (both scaled by ``conv``)."""
    if value is None or not np.isfinite(value):
        return "n/a"
    text = _fmt(value * conv)
    if err is not None and np.isfinite(err) and err > 0:
        text += f" ± {_fmt(err * conv, 2)}"
    return text


def _fmt_time(ts) -> str:
    """Render a timestamp as ``HH:MM:SS`` (date dropped for compactness)."""
    if ts is None:
        return ""
    try:
        return pd.Timestamp(ts).strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return ""


class DecayTab(_PlotTab):
    """Fit one or more emission + decay peaks; report source strength/loss rates."""

    export_tag = "decay"

    def __init__(self, main):
        """Build the metric/model/parameter controls, plot and results table."""
        super().__init__(main, nrows=1)

        self.controls.addWidget(QtWidgets.QLabel("Metric:"))
        self.metric_kind = QtWidgets.QComboBox()
        self.metric_kind.currentTextChanged.connect(self._on_metric_kind_change)
        self.controls.addWidget(self.metric_kind)
        self.metric_cut = QtWidgets.QComboBox()
        self.metric_cut.setEditable(True)
        self.metric_cut.setFixedWidth(64)
        self.metric_cut.currentIndexChanged.connect(self._on_metric_cut_change)
        self.metric_cut.lineEdit().editingFinished.connect(self._on_metric_cut_change)
        self.controls.addWidget(self.metric_cut)
        self.metric_cut_label = QtWidgets.QLabel("µm")
        self.controls.addWidget(self.metric_cut_label)

        self.controls.addWidget(QtWidgets.QLabel("Model:"))
        self.model = QtWidgets.QComboBox()
        for label, value in _MODEL_CHOICES:
            self.model.addItem(label, value)
        self.model.setToolTip(
            "Loss model to fit:\n"
            "• Zeroth order — constant removal (losses independent of conc.).\n"
            "• First order — losses ∝ concentration (air exchange + deposition).\n"
            "• Second order — losses ∝ concentration² (coagulation).\n"
            "• Combined — first + second order together.\n"
            "• Auto — fit all and pick the best (favouring the simpler)."
        )
        self.model.currentIndexChanged.connect(self._recompute_selected)
        self.controls.addWidget(self.model)

        self.controls.addWidget(QtWidgets.QLabel("Volume (m³):"))
        self.volume = QtWidgets.QLineEdit()
        self.volume.setPlaceholderText("optional")
        self.volume.setFixedWidth(60)
        self.volume.setToolTip(
            "Chamber/room volume in m³. Converts the fitted emission rate into a "
            "source strength (emission rate × volume) and a total emitted amount."
        )
        self.volume.editingFinished.connect(self._recompute_all_and_draw)
        self.controls.addWidget(self.volume)

        self.controls.addWidget(QtWidgets.QLabel("ACH (1/h):"))
        self.ach = QtWidgets.QLineEdit()
        self.ach.setPlaceholderText("optional")
        self.ach.setFixedWidth(52)
        self.ach.setToolTip(
            "An independently known air exchange rate (1/h). When given, the "
            "wall-loss/deposition rate is estimated as the fitted first-order "
            "loss minus this value."
        )
        self.ach.editingFinished.connect(self._recompute_all_and_draw)
        self.controls.addWidget(self.ach)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.setToolTip("Log-scale the concentration (y) axis.")
        self.log_y.stateChanged.connect(self._on_logy)
        self.controls.addWidget(self.log_y)

        self.per_bin = QtWidgets.QCheckBox("Per size bin")
        self.per_bin.setToolTip(
            "Also fit each size bin separately over every marked region (same "
            "window and timing), adding a row per bin to the table. Bins with "
            "R² < 0.5 are flagged as poorly fit and excluded."
        )
        self.per_bin.stateChanged.connect(self._recompute_all_and_draw)
        self.controls.addWidget(self.per_bin)

        # Mark-new: arm a one-shot drag that creates a new fitting region.
        self.mark_btn = QtWidgets.QPushButton("Mark new fitting area")
        self.mark_btn.setObjectName("toggle")
        self.mark_btn.setCheckable(True)
        self.mark_btn.setToolTip(
            "Toggle on, then drag across the plot to add a new decay fit over "
            "that window. It becomes the selected fit."
        )
        self.mark_btn.toggled.connect(self._toggle_mark)
        self.controls.addWidget(self.mark_btn)

        # Adjust: while on, clicks/handles/wheel edit the selected fit and the
        # view is locked; while off, the mouse zooms/pans the plot.
        self.adjust_btn = QtWidgets.QPushButton("Adjust")
        self.adjust_btn.setObjectName("toggle")
        self.adjust_btn.setCheckable(True)
        self.adjust_btn.setToolTip(
            "Toggle on to edit fits: click inside a region to select it (its "
            "handles appear and its parameters load), drag the handles or scroll "
            "to change the decay rate, click outside (or press Esc) to deselect. "
            "Toggle off to zoom/pan the plot."
        )
        self.adjust_btn.toggled.connect(self._toggle_adjust)
        self.controls.addWidget(self.adjust_btn)

        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Second controls row: selected-fit actions + guess parameters.
        self._layout.insertLayout(1, self._build_guess_row())

        self.ax = self.figure.add_subplot(111)

        self.result_label = QtWidgets.QLabel(
            "Click 'Mark new fitting area', then drag on the plot to select a "
            "window spanning the rise, peak and decay. Turn on 'Adjust' to shape "
            "a fit (click inside its region to select it); turn it off to "
            "zoom/pan. Scroll to zoom, right-drag to pan."
        )
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # Put the plot (toolbar + canvas + status) above the results table in a
        # draggable vertical splitter, so the user can trade figure height for
        # table height (e.g. a short, wide figure with a tall table).
        self._layout.removeWidget(self.toolbar)
        self._layout.removeWidget(self.canvas)
        plot_area = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plot_area)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(self.toolbar)
        pv.addWidget(self.canvas, stretch=1)
        pv.addWidget(self.result_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(plot_area)
        splitter.addWidget(self._build_table_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([460, 180])
        self._layout.addWidget(splitter, stretch=1)

        # -- fit state --------------------------------------------------------
        self._cols_obj = None
        self._sel = None  # index of the selected fit in the dataset's decay_fits
        self._results: list = []  # cached result dict (or None) per fit
        self._bin_results: list = []  # per-fit list of (bin_mid, result, excluded)
        # Drawn-handle artists for the selected fit + drag bookkeeping.
        self._t0_line = self._peak_line = self._bg_line = self._peak_marker = None
        self._region_patches: list = []
        self._dragging = None  # "t0" | "peak" | "bg" | "peakval" while dragging

        # ``interactive=False``: the drag rectangle is transient (it vanishes on
        # release). Each fit's region is drawn by us as an ``axvspan`` instead, so
        # an interactive/persistent selection would otherwise linger over the
        # last-marked window and survive deleting that fit.
        self.span = SpanSelector(
            self.ax,
            self._on_span,
            "horizontal",
            useblit=True,
            interactive=False,
            props=dict(alpha=0.12, facecolor=_REGION_COLOR),
            button=1,
        )
        self.span.set_active(False)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)

    def _build_guess_row(self) -> QtWidgets.QHBoxLayout:
        """Build the selected-fit actions, guess parameters and timing fields."""
        row = QtWidgets.QHBoxLayout()
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("primary")
        self.fit_btn.setToolTip(
            "Optimise the loss kinetics of the selected fit from the current "
            "guess (the background, peak and timing you set are held fixed)."
        )
        self.fit_btn.clicked.connect(self._on_fit)
        row.addWidget(self.fit_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset guess")
        self.reset_btn.setToolTip(
            "Clear the selected fit's manual values and re-fit automatically."
        )
        self.reset_btn.clicked.connect(self._on_reset)
        row.addWidget(self.reset_btn)

        self.del_btn = QtWidgets.QPushButton("Delete fit")
        self.del_btn.setToolTip("Remove the selected fit and its region.")
        self.del_btn.clicked.connect(self._on_delete)
        row.addWidget(self.del_btn)

        row.addSpacing(10)
        row.addWidget(QtWidgets.QLabel("P₀:"))
        self.bg_edit = QtWidgets.QLineEdit()
        self.bg_edit.setFixedWidth(70)
        self.bg_edit.setPlaceholderText("auto")
        self.bg_edit.setToolTip("Background concentration P₀ (held fixed in the fit).")
        self.bg_edit.editingFinished.connect(self._apply_guess_fields)
        row.addWidget(self.bg_edit)

        row.addWidget(QtWidgets.QLabel("Peak:"))
        self.peak_edit = QtWidgets.QLineEdit()
        self.peak_edit.setFixedWidth(80)
        self.peak_edit.setPlaceholderText("auto")
        self.peak_edit.setToolTip("Peak concentration at source-off (held fixed).")
        self.peak_edit.editingFinished.connect(self._apply_guess_fields)
        row.addWidget(self.peak_edit)

        row.addWidget(QtWidgets.QLabel("Rate (1/h):"))
        self.rate_edit = QtWidgets.QLineEdit()
        self.rate_edit.setFixedWidth(68)
        self.rate_edit.setPlaceholderText("auto")
        self.rate_edit.setToolTip(
            "First-order-equivalent loss rate (1/h). Scroll over the plot in "
            "Adjust mode, or type a value."
        )
        self.rate_edit.editingFinished.connect(self._apply_guess_fields)
        row.addWidget(self.rate_edit)

        row.addWidget(QtWidgets.QLabel("Emission start:"))
        self.t0_edit = QtWidgets.QLineEdit()
        self.t0_edit.setFixedWidth(84)
        self.t0_edit.setPlaceholderText("HH:MM:SS")
        self.t0_edit.setToolTip(
            "Emission start time (within the window). Type HH:MM:SS or a full "
            "timestamp, or drag the green line. Snaps the dashed marker there."
        )
        self.t0_edit.editingFinished.connect(self._apply_time_fields)
        row.addWidget(self.t0_edit)

        row.addWidget(QtWidgets.QLabel("Source-off:"))
        self.tp_edit = QtWidgets.QLineEdit()
        self.tp_edit.setFixedWidth(84)
        self.tp_edit.setPlaceholderText("HH:MM:SS")
        self.tp_edit.setToolTip(
            "Source-off (peak) time. Type HH:MM:SS or a full timestamp, or drag "
            "the red line."
        )
        self.tp_edit.editingFinished.connect(self._apply_time_fields)
        row.addWidget(self.tp_edit)

        row.addStretch(1)
        return row

    def _build_table_panel(self) -> QtWidgets.QWidget:
        """Build the results table + export button shown below the plot."""
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("Fitted decays:"))
        head.addStretch(1)
        self.export_btn = QtWidgets.QPushButton("Export table…")
        self.export_btn.setToolTip("Export the fitted-parameter table to Excel / CSV.")
        self.export_btn.clicked.connect(self._export_table)
        head.addWidget(self.export_btn)
        v.addLayout(head)
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # Only a small floor; the vertical splitter governs the actual height.
        self.table.setMinimumHeight(80)
        v.addWidget(self.table)
        return box

    # -- option syncing ------------------------------------------------------
    def _sync_metric_options(self) -> None:
        """Repopulate the metric drop-down for the active dataset.

        Particle number-concentration datasets keep the PNC (+ size-selective
        Pₓ / MASS for 2-D) kinds, which the cut-off field parameterises. Every
        other dataset — gases (Cl₂/NO₂ ppm), black carbon, LDSA, PM-mass, T/RH —
        instead offers its own named metrics from :meth:`available_metrics`, so
        the decay fit works on whatever quantity the instrument actually
        measured rather than an inapplicable PNC.
        """
        obj = self.obj
        is2d = helpers.is_2d(obj)
        number_primary = obj._has_number_primary()
        if number_primary:
            kinds = ["PNC"]
            if is2d:
                kinds += ["MASS", "PM", "PN", "PS", "PV"]
        else:
            # Named metrics (Cl₂, BC channels, LDSA, PM1/2.5/10, T, RH, …); none
            # of these use a size cut-off.
            kinds = obj._metric_keys() or [str(obj.data.columns[0])]

        current = self.metric_kind.currentText()
        self.metric_kind.blockSignals(True)
        self.metric_kind.clear()
        self.metric_kind.addItems(kinds)
        idx = self.metric_kind.findText(current)
        self.metric_kind.setCurrentIndex(idx if idx >= 0 else 0)
        self.metric_kind.blockSignals(False)
        self.metric_cut.blockSignals(True)
        self.metric_cut.clear()
        self.metric_cut.addItems(["0.1", "0.25", "0.5", "1", "2.5", "4", "4.2", "10"])
        self.metric_cut.blockSignals(False)
        # Per-bin decay only applies to size-resolved particle data.
        self.per_bin.setEnabled(is2d and number_primary)
        if not (is2d and number_primary):
            self.per_bin.setChecked(False)
        self._on_metric_kind_change()

    def _on_metric_kind_change(self, *_args) -> None:
        """Show the cut-off field only for size-selective fractions, then redraw."""
        is_fraction = self.metric_kind.currentText().strip() in _FRACTION_KINDS
        self.metric_cut.setVisible(is_fraction)
        self.metric_cut_label.setVisible(is_fraction)
        # A different metric has its own data range/units: refit every region.
        self._recompute_all()
        self._draw(reset=True)

    def _on_metric_cut_change(self, *_args) -> None:
        """Refit when the fraction cut-off diameter changes."""
        if self.metric_kind.currentText().strip() not in _FRACTION_KINDS:
            return
        self._recompute_all()
        self._draw(reset=True)

    def _on_logy(self, *_args) -> None:
        """Toggle the y-axis log scale, keeping the current fits."""
        self._draw()

    def _build_metric(self) -> str:
        """Assemble the metric string (kind plus cut-off where relevant)."""
        kind = self.metric_kind.currentText().strip()
        if kind in _FRACTION_KINDS:
            return f"{kind}{self.metric_cut.currentText().strip()}"
        return kind

    @staticmethod
    def _to_float(text: str, default=None):
        """Parse a field as a float, returning ``default`` when blank/invalid."""
        text = text.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    # -- fit-spec access --------------------------------------------------
    @property
    def _active_ds(self):
        """The active dataset, or None."""
        return self.main.project.active

    def _specs(self) -> list:
        """The active dataset's list of decay-fit specs (empty if none)."""
        ds = self._active_ds
        return ds.decay_fits if ds is not None else []

    def _selected_spec(self):
        """The currently selected fit spec, or None."""
        specs = self._specs()
        if self._sel is not None and 0 <= self._sel < len(specs):
            return specs[self._sel]
        return None

    @staticmethod
    def _new_overrides() -> dict:
        """Empty override set for a fresh fit spec."""
        return {
            "t0": None,
            "peak_time": None,
            "background": None,
            "peakval": None,
            "rate": None,
        }

    # -- rendering -------------------------------------------------------
    def refresh(self) -> None:
        """Re-sync selectors for the active dataset, recompute fits and redraw."""
        if self.obj is None:
            return
        changed = self._cols_obj is not self.obj
        if changed:
            self._sel = None
            self._sync_metric_options()
            self._cols_obj = self.obj
        self._recompute_all()
        # Rescale on a dataset switch or an explicit reset request; otherwise
        # keep the current zoom so unrelated actions don't snap the view back.
        self._draw(reset=changed or getattr(self.main, "_reset_view", True))

    def _series(self):
        """Return ``(times, values, unit, metric)`` for the whole active metric."""
        metric = self._build_metric()
        series, unit = self.obj._get_metric_series(metric)
        times = self.obj.time
        values = np.asarray(series, dtype=float)
        finite = np.isfinite(values)
        return times[finite], values[finite], unit, metric

    # -- fitting -----------------------------------------------------------
    def _fit_spec(self, spec: dict, optimize=None):
        """Run ``fit_decay`` for one spec; return the result dict or None."""
        if self.obj is None:
            return None
        ov = spec.get("overrides", {})
        model = spec.get("model", self.model.currentData())
        if not spec.get("optimized", True) and model == "auto" and spec.get("_result"):
            # Keep the last optimised family while previewing under Auto.
            model = spec["_result"]["model"]
        opt = spec.get("optimized", True) if optimize is None else optimize
        with contextlib.redirect_stdout(io.StringIO()):
            return self.obj.fit_decay(
                spec["window"],
                metric=self._build_metric(),
                model=model,
                volume=self._to_float(self.volume.text(), None),
                air_exchange_rate=self._to_float(self.ach.text(), None),
                emission_start=ov.get("t0"),
                peak_time=ov.get("peak_time"),
                background=ov.get("background"),
                peak_concentration=ov.get("peakval"),
                decay_rate=ov.get("rate"),
                optimize=opt,
                snap_to_samples=False,
            )

    def _fit_bins(self, spec: dict) -> list:
        """Fit each size bin over ``spec``'s window; return (mid, result, excluded)."""
        obj = self.obj
        if obj is None or not helpers.is_2d(obj):
            return []
        ov = spec.get("overrides", {})
        model = spec.get("model", self.model.currentData())
        volume = self._to_float(self.volume.text(), None)
        ach = self._to_float(self.ach.text(), None)
        out = []
        mids = np.asarray(obj.bin_mids, dtype=float)
        for header, mid in zip(obj._sizebin_headers, mids):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = obj.fit_decay(
                        spec["window"],
                        metric=self._build_metric(),
                        model=model,
                        volume=volume,
                        air_exchange_rate=ach,
                        emission_start=ov.get("t0"),
                        peak_time=ov.get("peak_time"),
                        background=ov.get("background"),
                        peak_concentration=ov.get("peakval"),
                        decay_rate=ov.get("rate"),
                        optimize=True,
                        snap_to_samples=False,
                        series=obj.data[header],
                        unit=obj.unit,
                    )
            except Exception:
                out.append((float(mid), None, True))
                continue
            excluded = not np.isfinite(res.get("r_squared", np.nan)) or (
                res["r_squared"] < _MIN_BIN_R2
            )
            out.append((float(mid), res, excluded))
        return out

    def _recompute_all(self) -> None:
        """Recompute every fit's result (and per-bin rows) for the active dataset."""
        specs = self._specs()
        self._results = []
        self._bin_results = []
        for spec in specs:
            try:
                res = self._fit_spec(spec)
            except Exception:
                res = None
            spec["_result"] = res
            self._results.append(res)
            if self.per_bin.isChecked() and res is not None:
                try:
                    self._bin_results.append(self._fit_bins(spec))
                except Exception:
                    self._bin_results.append([])
            else:
                self._bin_results.append([])

    def _recompute_all_and_draw(self) -> None:
        """Recompute all fits and redraw (kept view); used by shared controls."""
        self._recompute_all()
        self._draw()

    def _recompute_selected(self, *_args) -> None:
        """Recompute just the selected fit after a per-fit control change."""
        spec = self._selected_spec()
        if spec is None:
            return
        spec["model"] = self.model.currentData()
        self._recompute_one(self._sel)
        self._draw()

    def _recompute_one(self, index: int) -> None:
        """Recompute the result (and per-bin rows) for a single fit index."""
        specs = self._specs()
        if not (0 <= index < len(specs)):
            return
        spec = specs[index]
        try:
            res = self._fit_spec(spec)
        except Exception:
            res = None
        spec["_result"] = res
        while len(self._results) < len(specs):
            self._results.append(None)
            self._bin_results.append([])
        self._results[index] = res
        if self.per_bin.isChecked() and res is not None:
            self._bin_results[index] = self._fit_bins(spec)
        else:
            self._bin_results[index] = []

    # -- window selection & interaction ----------------------------------
    def _toggle_mark(self) -> None:
        """Arm/disarm a one-shot drag that creates a new fitting region."""
        active = self.mark_btn.isChecked()
        self.mark_btn.setText("Marking…" if active else "Mark new fitting area")
        self.span.set_active(active)
        if active:
            # Marking and adjusting are mutually exclusive; drop any pan/zoom.
            if self.adjust_btn.isChecked():
                self.adjust_btn.setChecked(False)
            self._drop_toolbar_mode()

    def _toggle_adjust(self) -> None:
        """Enter/leave Adjust mode (edit the selected fit vs zoom/pan)."""
        active = self.adjust_btn.isChecked()
        self.adjust_btn.setText("Adjusting…" if active else "Adjust")
        if active:
            if self.mark_btn.isChecked():
                self.mark_btn.setChecked(False)
            self._drop_toolbar_mode()
        else:
            self._select(None)  # leaving adjust hides the handles
        self._draw()

    def _drop_toolbar_mode(self) -> None:
        """Deactivate any active toolbar pan/zoom so drags reach this tab."""
        if self.toolbar.mode:
            mode = str(self.toolbar.mode)
            if "pan" in mode:
                self.toolbar.pan()
            elif "zoom" in mode:
                self.toolbar.zoom()

    def _scroll_reserved(self, event) -> bool:
        """In Adjust mode with a fit selected, the wheel sets the decay rate."""
        return self.adjust_btn.isChecked() and self._selected_spec() is not None

    def _pan_locked(self) -> bool:
        """Lock the view while marking or adjusting."""
        return self.mark_btn.isChecked() or self.adjust_btn.isChecked()

    def _on_span(self, xmin: float, xmax: float) -> None:
        """Create a new fit from the dragged window and select it."""
        if xmax - xmin < 1e-9 or self._active_ds is None:
            return
        if self.mark_btn.isChecked():
            self.mark_btn.setChecked(False)  # one-shot
        window = (self._num_to_ts(xmin), self._num_to_ts(xmax))
        spec = {
            "window": window,
            "metric": self._build_metric(),
            "model": self.model.currentData(),
            "optimized": True,
            "overrides": self._new_overrides(),
        }
        self._specs().append(spec)
        self._sel = len(self._specs()) - 1
        self._recompute_one(self._sel)
        # Adjusting the new fit right away is the common next step.
        self.adjust_btn.setChecked(True)
        self._draw()

    def _on_fit(self) -> None:
        """Optimise the loss kinetics of the selected fit."""
        spec = self._selected_spec()
        if spec is None:
            self.result_label.setText("Select a fit first (click inside its region).")
            return
        spec["optimized"] = True
        self._recompute_one(self._sel)
        self._draw()

    def _on_reset(self) -> None:
        """Clear the selected fit's manual guesses and re-auto-fit."""
        spec = self._selected_spec()
        if spec is None:
            return
        spec["overrides"] = self._new_overrides()
        spec["optimized"] = True
        self._recompute_one(self._sel)
        self._draw()

    def _on_delete(self) -> None:
        """Delete the selected fit and its region."""
        spec = self._selected_spec()
        if spec is None:
            return
        del self._specs()[self._sel]
        self._sel = None
        self._recompute_all()
        self._draw(reset=True)

    def _apply_guess_fields(self) -> None:
        """Read the P₀/peak/rate fields into the selected fit and preview."""
        spec = self._selected_spec()
        if spec is None:
            return
        fields = (self.bg_edit, self.peak_edit, self.rate_edit)
        if not any(f.isModified() for f in fields):
            return
        for f in fields:
            f.setModified(False)
        ov = spec["overrides"]
        ov["background"] = self._to_float(self.bg_edit.text(), None)
        ov["peakval"] = self._to_float(self.peak_edit.text(), None)
        rate_h = self._to_float(self.rate_edit.text(), None)
        ov["rate"] = (rate_h / 3600.0) if rate_h is not None else None
        spec["optimized"] = False
        self._recompute_one(self._sel)
        self._draw()

    def _apply_time_fields(self) -> None:
        """Read the emission-start / source-off time fields into the selected fit."""
        spec = self._selected_spec()
        if spec is None:
            return
        fields = (self.t0_edit, self.tp_edit)
        if not any(f.isModified() for f in fields):
            return
        for f in fields:
            f.setModified(False)
        ov = spec["overrides"]
        ov["t0"] = self._parse_time(self.t0_edit.text(), spec["window"][0])
        ov["peak_time"] = self._parse_time(self.tp_edit.text(), spec["window"][0])
        self._recompute_one(self._sel)
        self._draw()

    @staticmethod
    def _parse_time(text: str, ref):
        """Parse a full timestamp or a HH:MM:SS on ``ref``'s date, or None."""
        text = text.strip()
        if not text:
            return None
        try:
            return pd.Timestamp(text)
        except (ValueError, TypeError):
            pass
        try:
            t = pd.Timestamp(text).time()
        except (ValueError, TypeError):
            try:
                t = pd.to_datetime(text, format="%H:%M:%S").time()
            except (ValueError, TypeError):
                return None
        return pd.Timestamp(pd.Timestamp(ref).date()).replace(
            hour=t.hour, minute=t.minute, second=t.second
        )

    def _select(self, index) -> None:
        """Select a fit by index (or None to deselect) and reload its fields."""
        self._sel = index
        spec = self._selected_spec()
        if spec is not None:
            self.model.blockSignals(True)
            mi = self.model.findData(spec.get("model", "auto"))
            self.model.setCurrentIndex(mi if mi >= 0 else 0)
            self.model.blockSignals(False)

    def _on_scroll(self, event) -> None:
        """Adjust-mode scroll over the plot: change the selected fit's decay rate."""
        if not self.adjust_btn.isChecked() or event.inaxes is not self.ax:
            return
        spec = self._selected_spec()
        if spec is None or self.toolbar.mode:
            return
        ov = spec["overrides"]
        base = ov.get("rate")
        if base is None and spec.get("_result"):
            base = spec["_result"].get("decay_rate")
        if not base or not np.isfinite(base) or base <= 0:
            base = 1.0 / 1800.0
        ov["rate"] = float(max(base * (_SCROLL_RATE_STEP**event.step), 1e-9))
        spec["optimized"] = False
        self._recompute_one(self._sel)
        self._draw()

    def _on_key(self, event) -> None:
        """Esc deselects the current fit (hiding its handles)."""
        if event.key == "escape" and self._sel is not None:
            self._select(None)
            self._draw()

    # -- draggable handles ------------------------------------------------
    def _on_press(self, event) -> None:
        """Handle a left-click: grab a handle, select a region, or deselect."""
        if event.button != 1 or event.inaxes is not self.ax:
            return
        if not self.adjust_btn.isChecked():
            return  # not editing: clicks do nothing (nav owns the mouse)
        # If a fit is selected, a click on one of its handles starts a drag.
        if self._selected_spec() is not None and self._grab_handle(event):
            return
        # Otherwise (re)select whichever region the click fell inside, or none.
        idx = self._region_at(event.xdata)
        self._select(idx)
        self._draw()

    def _region_at(self, xdata):
        """Index of the fit whose window contains ``xdata`` (last wins), or None."""
        if xdata is None:
            return None
        click = self._num_to_ts(xdata)
        found = None
        for i, spec in enumerate(self._specs()):
            start, end = spec["window"]
            if start <= click <= end:
                found = i
        return found

    def _grab_handle(self, event) -> bool:
        """Grab the selected fit's handle nearest the click; True if grabbed."""
        if self._result_selected() is None:
            return False
        if self._peak_marker is not None and event.y is not None:
            mx = mdates.date2num(self._peak_marker.get_xdata()[0])
            my = self._peak_marker.get_ydata()[0]
            px, py = self.ax.transData.transform((mx, my))
            if abs(event.x - px) <= 8 and abs(event.y - py) <= 8:
                self._grab("peakval")
                return True
        for name, line in (("t0", self._t0_line), ("peak", self._peak_line)):
            px = self._line_x_pixels(line)
            if px is not None and abs(event.x - px) <= 6:
                self._grab(name)
                return True
        py = self._line_y_pixels(self._bg_line)
        if py is not None and event.y is not None and abs(event.y - py) <= 6:
            self._grab("bg")
            return True
        return False

    def _result_selected(self):
        """The selected fit's cached result dict, or None."""
        spec = self._selected_spec()
        return spec.get("_result") if spec else None

    def _line_x_pixels(self, line) -> float:
        """Screen-x (pixels) of a vertical marker line, or None."""
        if line is None:
            return None
        xnum = line.get_xdata()[0]
        return float(self.ax.transData.transform((mdates.date2num(xnum), 0))[0])

    def _line_y_pixels(self, line) -> float:
        """Screen-y (pixels) of a horizontal marker line, or None."""
        if line is None:
            return None
        yval = line.get_ydata()[0]
        x0 = self.ax.get_xlim()[0]
        return float(self.ax.transData.transform((x0, yval))[1])

    def _grab(self, name: str) -> None:
        """Start dragging handle ``name`` (suspending the span selector)."""
        self._dragging = name
        self.span.set_active(False)

    def _on_motion(self, event) -> None:
        """Move the grabbed handle to follow the cursor."""
        if self._dragging is None or event.inaxes is not self.ax:
            return
        if self._dragging in ("t0", "peak"):
            if event.xdata is None:
                return
            line = self._t0_line if self._dragging == "t0" else self._peak_line
            num = mdates.num2date(event.xdata)
            line.set_xdata([num, num])
        elif self._dragging == "bg":
            if event.ydata is None:
                return
            self._bg_line.set_ydata([event.ydata, event.ydata])
        elif self._dragging == "peakval":
            if event.ydata is None:
                return
            self._peak_marker.set_ydata([event.ydata])
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        """Commit a dragged handle as an override and preview."""
        if self._dragging is None:
            return
        name, self._dragging = self._dragging, None
        self.span.set_active(self.mark_btn.isChecked())
        spec = self._selected_spec()
        if spec is None:
            return
        ov = spec["overrides"]
        if name in ("t0", "peak"):
            line = self._t0_line if name == "t0" else self._peak_line
            ts = self._num_to_ts(mdates.date2num(line.get_xdata()[0]))
            ov["t0" if name == "t0" else "peak_time"] = ts
        elif name == "bg":
            ov["background"] = float(self._bg_line.get_ydata()[0])
        elif name == "peakval":
            ov["peakval"] = float(self._peak_marker.get_ydata()[0])
        spec["optimized"] = False
        self._recompute_one(self._sel)
        self._draw()

    @staticmethod
    def _num_to_ts(xnum: float) -> pd.Timestamp:
        """Matplotlib date number -> tz-naive pandas Timestamp."""
        return pd.Timestamp(mdates.num2date(xnum)).tz_localize(None)

    # -- drawing ----------------------------------------------------------
    def _draw(self, reset: bool = False) -> None:
        """Redraw the series, every fit region/curve, and the selected handles."""
        if self.obj is None:
            return
        prev = None
        if not reset and self.ax.has_data():
            prev = (self.ax.get_xlim(), self.ax.get_ylim())
        try:
            self._draw_on()
        except Exception:
            self.result_label.setText(
                "Could not draw:\n" + traceback.format_exc(limit=1)
            )
            return
        if prev is not None:
            self.ax.set_xlim(prev[0])
            self.ax.set_ylim(prev[1])
        else:
            self._sync_toolbar_home()
        self.canvas.draw_idle()
        self._rebuild_table()
        self._update_status()

    def _draw_on(self) -> None:
        """Paint the base series, all fit regions/curves and the selected handles."""
        ax = self.ax
        ax.clear()
        self._t0_line = self._peak_line = self._bg_line = self._peak_marker = None
        self._region_patches = []
        times, values, unit, metric = self._series()
        ax.plot(times, values, "-", lw=0.8, color="#4c72b0", alpha=0.5)
        ax.plot(times, values, "o", ms=2.5, color="#4c72b0", label=metric)
        helpers.shade_activities(ax, self.obj)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{metric}, {unit}".strip(", "))
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        ax.set_yscale("log" if self.log_y.isChecked() else "linear")

        for i, spec in enumerate(self._specs()):
            res = spec.get("_result")
            selected = i == self._sel
            self._draw_region(ax, spec, res, selected)
        ax.legend(loc="upper right", fontsize=8)

    def _draw_region(self, ax, spec, res, selected: bool) -> None:
        """Shade a fit's window; draw its curve, and handles when selected."""
        start, end = spec["window"]
        patch = ax.axvspan(
            start,
            end,
            color=_REGION_COLOR,
            alpha=0.16 if selected else 0.06,
            lw=0,
            zorder=0,
        )
        self._region_patches.append(patch)
        if res is None:
            return
        span_s = (end - start).total_seconds()
        t_fit = np.linspace(0.0, float(span_s), 400)
        t_fit = np.unique(np.concatenate([t_fit, [res["peak_time_s"]]]))
        c_fit = _decay.decay_curve(res["model"], t_fit, res["model_popt"])
        t_dates = start + pd.to_timedelta(t_fit, unit="s")
        optimized = spec.get("optimized", True)
        color = _PEAK_COLOR if optimized else _GUESS_COLOR
        ax.plot(
            t_dates,
            c_fit,
            ls="-" if optimized else "--",
            lw=2.0,
            color=color,
            zorder=6,
        )
        if not selected:
            return
        # Selected fit: draggable background line, peak handle and timing lines.
        self._bg_line = ax.axhline(res["background"], color=_BG_COLOR, ls="--", lw=1.2)
        self._peak_marker = ax.plot(
            [res["peak_time"]],
            [res["peak_concentration"]],
            marker="o",
            ms=9,
            mfc="white",
            mec=_PEAK_COLOR,
            mew=1.8,
            ls="None",
            zorder=9,
        )[0]
        t0_ts = res["window_start"] + pd.to_timedelta(res["emission_start_s"], unit="s")
        self._t0_line = ax.axvline(t0_ts, color=_T0_COLOR, ls="--", lw=1.4)
        self._peak_line = ax.axvline(
            res["peak_time"], color=_PEAK_COLOR, ls=":", lw=1.4
        )
        self._sync_fit_fields(spec, res, t0_ts)

    def _sync_fit_fields(self, spec, res, t0_ts) -> None:
        """Reflect the selected fit's parameters into the editable fields."""
        pairs = (
            (self.bg_edit, _fmt(res.get("background")) if res else ""),
            (self.peak_edit, _fmt(res.get("peak_concentration")) if res else ""),
            (self.rate_edit, _fmt(res.get("decay_rate_per_hour")) if res else ""),
            (self.t0_edit, _fmt_time(t0_ts)),
            (self.tp_edit, _fmt_time(res.get("peak_time")) if res else ""),
        )
        for edit, text in pairs:
            edit.blockSignals(True)
            edit.setText("" if text == "n/a" else text)
            edit.setModified(False)
            edit.blockSignals(False)

    def _update_status(self) -> None:
        """Set the one-line status under the plot for the selected fit."""
        spec = self._selected_spec()
        if spec is None:
            n = len(self._specs())
            self.result_label.setText(
                f"{n} fit(s). "
                + (
                    "Turn on 'Adjust' and click inside a region to edit a fit, "
                    "or 'Mark new fitting area' to add one."
                    if n
                    else "Click 'Mark new fitting area', then drag to add a decay fit."
                )
            )
            return
        res = spec.get("_result")
        if res is None:
            self.result_label.setText("This region could not be fitted.")
            return
        model = res["model"].replace("_", " ")
        state = "optimised" if spec.get("optimized", True) else "guess (press Fit)"
        self.result_label.setText(
            f"Selected fit {self._sel + 1}: {model}, {state} — "
            f"R² = {res['r_squared']:.4f}, decay R² = {res['decay_r_squared']:.4f}. "
            "Drag the handles, scroll to change the rate, or edit the fields."
        )

    # -- results table ----------------------------------------------------
    def _table_columns(self, unit: str) -> list:
        """Ordered ``(header, key)`` columns; units filled from the metric unit."""
        u = unit or "conc"
        return [
            ("Fit", "fit"),
            ("Size bin (nm)", "bin"),
            ("Metric", "metric"),
            ("Region start", "rstart"),
            ("Region end", "rend"),
            ("Emission start", "t0"),
            ("Source-off", "tp"),
            ("Model", "model"),
            ("n", "n"),
            ("R²", "r2"),
            ("decay R²", "dr2"),
            (f"Background [{u}]", "bg"),
            (f"Peak [{u}]", "peak"),
            ("Decay rate [1/h]", "rate"),
            (f"Emission rate [{u}/s]", "emis"),
            ("First-order loss [1/h]", "loss"),
            ("Half-life [h]", "half"),
            (f"Zeroth-order a [{u}/s]", "zeroth"),
            ("2nd-order rate", "second"),
            ("Wall loss [1/h]", "wall"),
            ("Source strength", "source"),
            ("Total emitted", "emitted"),
            ("Note", "note"),
        ]

    def _row_cells(self, res, fit_label, bin_label, note) -> dict:
        """Build the display values for one result row."""
        if res is None:
            return {"fit": fit_label, "bin": bin_label, "note": note or "no fit"}
        errs = res.get("errors", {}) or {}
        t0_ts = res["window_start"] + pd.to_timedelta(res["emission_start_s"], unit="s")
        cells = {
            "fit": fit_label,
            "bin": bin_label,
            "metric": res.get("metric", ""),
            "rstart": "",  # filled from the spec window in _table_rows
            "rend": "",
            "t0": _fmt_time(t0_ts),
            "tp": _fmt_time(res.get("peak_time")),
            "model": res["model"].replace("_", " "),
            "n": str(res.get("n_points", "")),
            "r2": f"{res['r_squared']:.4f}",
            "dr2": f"{res['decay_r_squared']:.4f}",
            "bg": _fmt_pm(res.get("background"), errs.get("P0")),
            "peak": _fmt(res.get("peak_concentration")),
            "rate": _fmt(res.get("decay_rate_per_hour")),
            "emis": _fmt_pm(res.get("emission_rate"), errs.get("E")),
            "loss": _fmt(res.get("loss_rate_per_hour")),
            "half": _fmt(res.get("half_life_hours")),
            "zeroth": _fmt(res.get("zeroth_order_rate")),
            "second": _fmt(res.get("second_order_rate")),
            "wall": _fmt(res.get("wall_loss_rate_per_hour")),
            "source": _fmt(res.get("source_strength")),
            "emitted": _fmt(res.get("total_emitted")),
            "note": note or "",
        }
        return cells

    def _table_rows(self):
        """Yield ``(cells, unit)`` for every fit (and per-bin) result row."""
        rows = []
        unit = ""
        for i, spec in enumerate(self._specs()):
            res = self._results[i] if i < len(self._results) else None
            start, end = spec["window"]
            base = self._row_cells(res, str(i + 1), "", None)
            base["rstart"] = _fmt_time(start)
            base["rend"] = _fmt_time(end)
            if res is not None:
                unit = res.get("unit", unit)
            rows.append(base)
            bins = self._bin_results[i] if i < len(self._bin_results) else []
            for mid, bres, excluded in bins:
                note = "excluded (R²<0.5)" if excluded else ""
                cells = self._row_cells(bres, f"{i + 1}·bin", f"{mid:g}", note)
                cells["rstart"] = _fmt_time(start)
                cells["rend"] = _fmt_time(end)
                rows.append(cells)
        return rows, unit

    def _rebuild_table(self) -> None:
        """Repopulate the results table from the cached fit results."""
        rows, unit = self._table_rows()
        columns = self._table_columns(unit)
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels([h for h, _ in columns])
        self.table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for c, (_h, key) in enumerate(columns):
                item = QtWidgets.QTableWidgetItem(str(cells.get(key, "")))
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    def _results_dataframe(self) -> pd.DataFrame:
        """The results table as a DataFrame (for export)."""
        rows, unit = self._table_rows()
        columns = self._table_columns(unit)
        data = [{h: cells.get(key, "") for h, key in columns} for cells in rows]
        return pd.DataFrame(data, columns=[h for h, _ in columns])

    def _export_table(self) -> None:
        """Export the fitted-parameter table to Excel / CSV."""
        df = self._results_dataframe()
        _export_table(self, df, self._export_stem(), with_index=False)

    def current_time_xlim(self):
        """Time-axis limits as date numbers, or None."""
        if self.ax.has_data():
            return self.ax.get_xlim()
        return None
