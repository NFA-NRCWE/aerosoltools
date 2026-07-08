"""Decay / source-strength fitting tab.

Fits a single-zone emission + decay peak model (see
:mod:`aerosoltools._core.decay`) to a window of a chosen metric on the active
dataset. The window is chosen by **dragging on the plot** (no need to pre-mark a
task); the emission start and the source-off (peak) are detected automatically
and can be nudged by dragging their dashed markers. From the fit it reports the
emission (source) strength and the loss kinetics — zeroth order (constant
removal), first order (air exchange + wall deposition), second order
(coagulation) or both combined — plus, when a chamber volume or a known air
exchange rate are supplied, the source strength and wall-loss rate.

Like the PSD tab's lognormal fitting, the fit is **interactive**: selecting a
window seeds an automatic fit, then the user can shape the *initial guess* by
eye — drag the background line, drag the peak height, scroll to change the decay
rate, or type exact values — and see a live dashed **preview** of that guess.
Pressing **Fit** optimises the loss kinetics from the current guess (solid
curve); **Reset guess** clears the manual values and re-fits automatically.
"""

from __future__ import annotations

import traceback

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.widgets import SpanSelector

from ..._core import decay as _decay
from .. import helpers
from ..qt import QtCore, QtWidgets
from ._base import _PlotTab

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

#: Factor a single scroll notch multiplies/divides the decay rate by.
_SCROLL_RATE_STEP = 1.15


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


class DecayTab(_PlotTab):
    """Fit an emission + decay peak to estimate source strength and loss rates."""

    export_tag = "decay"

    def __init__(self, main):
        """Build the metric/model/parameter controls and the interactive plot."""
        super().__init__(main, nrows=1)

        self.controls.addWidget(QtWidgets.QLabel("Metric:"))
        self.metric_kind = QtWidgets.QComboBox()
        self.metric_kind.currentTextChanged.connect(self._on_metric_kind_change)
        self.controls.addWidget(self.metric_kind)
        self.metric_cut = QtWidgets.QComboBox()
        self.metric_cut.setEditable(True)
        self.metric_cut.setFixedWidth(64)
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
        self.model.currentIndexChanged.connect(self._recompute)
        self.controls.addWidget(self.model)

        self.controls.addWidget(QtWidgets.QLabel("Volume (m³):"))
        self.volume = QtWidgets.QLineEdit()
        self.volume.setPlaceholderText("optional")
        self.volume.setFixedWidth(64)
        self.volume.setToolTip(
            "Chamber/room volume in m³. Converts the fitted emission rate into a "
            "source strength (emission rate × volume) and a total emitted amount."
        )
        self.volume.editingFinished.connect(self._recompute)
        self.controls.addWidget(self.volume)

        self.controls.addWidget(QtWidgets.QLabel("Known ACH (1/h):"))
        self.ach = QtWidgets.QLineEdit()
        self.ach.setPlaceholderText("optional")
        self.ach.setFixedWidth(56)
        self.ach.setToolTip(
            "An independently known air exchange rate (1/h). When given, the "
            "wall-loss/deposition rate is estimated as the fitted first-order "
            "loss minus this value."
        )
        self.ach.editingFinished.connect(self._recompute)
        self.controls.addWidget(self.ach)

        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Second controls row: the interactive initial-guess parameters + the
        # Fit / Reset actions (mirrors the PSD tab's manual-then-optimise flow).
        self._layout.insertLayout(1, self._build_guess_row())

        self.ax = self.figure.add_subplot(111)

        self.result_label = QtWidgets.QLabel(
            "Drag on the plot to select a window spanning the rise, peak and "
            "decay. Then shape the guess: drag the emission start (green) / "
            "source-off (red) markers, drag the background line or the peak "
            "handle, scroll to change the decay rate, or type values above — the "
            "dashed preview updates live. Press Fit to optimise."
        )
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._layout.addWidget(self.result_label)

        # Fit state: the selected window, the (optional) manual overrides for the
        # emission start / source-off (times), the background P0, the peak
        # concentration and the decay rate (1/s), whether the shown fit is an
        # optimisation or a manual-guess preview, plus the current fit result.
        self._cols_obj = None
        self._window = None  # (start, end) timestamps
        self._t0_override = None
        self._peak_override = None  # source-off (peak) TIME
        self._bg_override = None  # background P0 (concentration)
        self._peakval_override = None  # peak concentration
        self._rate_override = None  # decay rate (1/s, first-order-equivalent)
        self._optimized = False
        self._result = None
        self._t0_line = None
        self._peak_line = None
        self._bg_line = None  # horizontal background (P0) line
        self._peak_marker = None  # (peak_time, peak_conc) handle
        self._dragging = None  # "t0" | "peak" | "bg" | "peakval" while dragging

        self.span = SpanSelector(
            self.ax,
            self._on_span,
            "horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
            props=dict(alpha=0.12, facecolor="#4c72b0"),
        )
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _build_guess_row(self) -> QtWidgets.QHBoxLayout:
        """Build the initial-guess parameter fields and Fit / Reset buttons."""
        row = QtWidgets.QHBoxLayout()
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("primary")
        self.fit_btn.setToolTip(
            "Optimise the loss kinetics to the decay, starting from the current "
            "guess (the background, peak and timing you set are held fixed)."
        )
        self.fit_btn.clicked.connect(self._on_fit)
        row.addWidget(self.fit_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset guess")
        self.reset_btn.setToolTip(
            "Clear the manual background / peak / decay-rate values and re-fit "
            "the window automatically."
        )
        self.reset_btn.clicked.connect(self._on_reset)
        row.addWidget(self.reset_btn)

        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Guess —  P₀:"))
        self.bg_edit = QtWidgets.QLineEdit()
        self.bg_edit.setFixedWidth(72)
        self.bg_edit.setPlaceholderText("auto")
        self.bg_edit.setToolTip(
            "Background concentration P₀. Drag the grey line on the plot, or type "
            "a value. Held fixed during the fit."
        )
        self.bg_edit.editingFinished.connect(self._apply_guess_fields)
        row.addWidget(self.bg_edit)

        row.addWidget(QtWidgets.QLabel("Peak:"))
        self.peak_edit = QtWidgets.QLineEdit()
        self.peak_edit.setFixedWidth(84)
        self.peak_edit.setPlaceholderText("auto")
        self.peak_edit.setToolTip(
            "Peak concentration at source-off. Drag the hollow peak handle up/"
            "down, or type a value. Held fixed during the fit."
        )
        self.peak_edit.editingFinished.connect(self._apply_guess_fields)
        row.addWidget(self.peak_edit)

        row.addWidget(QtWidgets.QLabel("Decay rate (1/h):"))
        self.rate_edit = QtWidgets.QLineEdit()
        self.rate_edit.setFixedWidth(72)
        self.rate_edit.setPlaceholderText("auto")
        self.rate_edit.setToolTip(
            "First-order-equivalent loss rate (1/h). Scroll over the plot to "
            "change it, or type a value. Seeds the fit; with a manual value the "
            "preview uses it directly."
        )
        self.rate_edit.editingFinished.connect(self._apply_guess_fields)
        row.addWidget(self.rate_edit)

        row.addStretch(1)
        return row

    # -- option syncing ------------------------------------------------------
    def _sync_metric_options(self) -> None:
        """Repopulate the metric drop-down for the active object's dimensionality."""
        is2d = helpers.is_2d(self.obj)
        current = self.metric_kind.currentText()
        self.metric_kind.blockSignals(True)
        self.metric_kind.clear()
        kinds = ["PNC"]
        if is2d:
            kinds += ["MASS", "PM", "PN", "PS", "PV"]
        self.metric_kind.addItems(kinds)
        idx = self.metric_kind.findText(current)
        self.metric_kind.setCurrentIndex(idx if idx >= 0 else 0)
        self.metric_kind.blockSignals(False)
        self.metric_cut.clear()
        self.metric_cut.addItems(["0.1", "0.25", "0.5", "1", "2.5", "4", "4.2", "10"])
        self._on_metric_kind_change()

    def _on_metric_kind_change(self, *_args) -> None:
        """Show the cut-off field only for the size-selective fractions, then refit."""
        is_fraction = self.metric_kind.currentText().strip() in _FRACTION_KINDS
        self.metric_cut.setVisible(is_fraction)
        self.metric_cut_label.setVisible(is_fraction)
        self._draw_base()
        self._recompute()

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

    # -- rendering -------------------------------------------------------
    def refresh(self) -> None:
        """Re-sync the selectors for the active dataset and redraw the series."""
        if self.obj is None:
            return
        if self._cols_obj is not self.obj:
            # Reset the window/overrides *before* syncing metrics, so the refit
            # that a metric change triggers does not run against a stale window
            # from the previous dataset.
            self._window = None
            self._t0_override = self._peak_override = None
            self._bg_override = self._peakval_override = self._rate_override = None
            self._optimized = False
            self._result = None
            self._sync_metric_options()
            self._cols_obj = self.obj
        # Re-run an existing fit on refresh (e.g. after a density change) so the
        # mass-based results and curve track the new density; otherwise just
        # redraw the raw segment. The optimise/preview state is preserved.
        if self._window is not None:
            self._recompute()
        else:
            self._draw_base()

    def _series(self):
        """Return ``(times, values, unit)`` for the whole active metric series."""
        metric = self._build_metric()
        series, unit = self.obj._get_metric_series(metric)
        times = self.obj.time
        values = np.asarray(series, dtype=float)
        finite = np.isfinite(values)
        return times[finite], values[finite], unit, metric

    def _draw_base(self) -> None:
        """Plot the full metric series (with activity shading), no fit curve."""
        if self.obj is None:
            return
        ax = self.ax
        ax.clear()
        self._t0_line = self._peak_line = None
        self._bg_line = self._peak_marker = None
        try:
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
        except Exception:
            pass
        self._sync_toolbar_home()
        self.canvas.draw_idle()

    # -- window selection & fitting --------------------------------------
    def _on_span(self, xmin: float, xmax: float) -> None:
        """Store the dragged window (date numbers) and auto-fit it fresh."""
        if xmax - xmin < 1e-9:
            return
        self._window = (self._num_to_ts(xmin), self._num_to_ts(xmax))
        # A fresh window drops every manual override and auto-detects/optimises.
        self._t0_override = self._peak_override = None
        self._bg_override = self._peakval_override = self._rate_override = None
        self._recompute(optimize=True)

    def _on_fit(self) -> None:
        """Optimise the loss kinetics from the current guess."""
        if self._window is None:
            self.result_label.setText("Select a window on the plot first.")
            return
        self._recompute(optimize=True)

    def _on_reset(self) -> None:
        """Clear the manual background / peak / rate guesses and re-auto-fit."""
        self._bg_override = self._peakval_override = self._rate_override = None
        self._recompute(optimize=True)

    def _apply_guess_fields(self) -> None:
        """Read the guess fields into overrides and preview (if user-edited).

        ``editingFinished`` also fires on focus-out with unchanged text, so act
        only when the user actually modified a field — otherwise clicking away
        from an optimised fit would silently drop it back to a preview.
        """
        fields = (self.bg_edit, self.peak_edit, self.rate_edit)
        if not any(f.isModified() for f in fields):
            return
        for f in fields:
            f.setModified(False)
        self._bg_override = self._to_float(self.bg_edit.text(), None)
        self._peakval_override = self._to_float(self.peak_edit.text(), None)
        rate_h = self._to_float(self.rate_edit.text(), None)
        self._rate_override = (rate_h / 3600.0) if rate_h is not None else None
        self._recompute(optimize=False)

    def _recompute(self, *_args, optimize=None) -> None:
        """Fit/preview for the current window, overrides and mode, then redraw.

        ``optimize`` forces the optimise (True) or preview (False) state; when
        ``None`` the current state is kept (used by control changes that should
        not switch between fitting and previewing). Signal slots pass through
        ``*_args`` (an int index/state) which is ignored.
        """
        if self.obj is None or self._window is None:
            return
        if optimize is not None:
            self._optimized = bool(optimize)
        metric = self._build_metric()
        volume = self._to_float(self.volume.text(), None)
        ach = self._to_float(self.ach.text(), None)
        # While previewing a manual guess under "Auto", keep the model family the
        # last optimisation chose — otherwise scrubbing the rate could silently
        # jump between model families (e.g. a linear zeroth-order seed suddenly
        # scoring best), which is disorienting mid-adjustment.
        model = self.model.currentData()
        if not self._optimized and model == "auto" and self._result is not None:
            model = self._result["model"]
        try:
            result = self.obj.fit_decay(
                self._window,
                metric=metric,
                model=model,
                volume=volume,
                air_exchange_rate=ach,
                emission_start=self._t0_override,
                peak_time=self._peak_override,
                background=self._bg_override,
                peak_concentration=self._peakval_override,
                decay_rate=self._rate_override,
                optimize=self._optimized,
            )
        except Exception as exc:
            self._result = None
            self.result_label.setText(f"Fit failed: {exc}")
            return
        self._result = result
        try:
            self._draw_fit(metric, result)
        except Exception:
            self.result_label.setText(
                "Fit succeeded but the plot failed:\n" + traceback.format_exc(limit=1)
            )
            return
        self._sync_guess_fields(result)
        self.result_label.setText(self._format_result(result))

    def _sync_guess_fields(self, result: dict) -> None:
        """Show the fit's current background / peak / decay-rate in the fields."""
        pairs = (
            (self.bg_edit, result.get("background")),
            (self.peak_edit, result.get("peak_concentration")),
            (self.rate_edit, result.get("decay_rate_per_hour")),
        )
        for edit, value in pairs:
            edit.blockSignals(True)
            edit.setText("" if value is None else f"{value:.4g}")
            edit.setModified(False)
            edit.blockSignals(False)

    def _draw_fit(self, metric: str, result: dict) -> None:
        """Redraw the series with the fit/preview curve and draggable handles."""
        self._draw_base()
        ax = self.ax
        start, end = self._window
        span_s = (end - start).total_seconds()
        # Dense grid, plus the exact peak time so the cusp is not clipped.
        t_fit = np.linspace(0.0, float(span_s), 600)
        peak_s = result["peak_time_s"]
        t_fit = np.unique(np.concatenate([t_fit, [peak_s]]))
        c_fit = _decay.decay_curve(result["model"], t_fit, result["model_popt"])
        t_dates = start + pd.to_timedelta(t_fit, unit="s")
        optimized = self._optimized
        curve_color = _PEAK_COLOR if optimized else _GUESS_COLOR
        if optimized:
            label = f"Fit ({result['model'].replace('_', ' ')})"
        else:
            label = "Guess (preview)"
        ax.plot(
            t_dates,
            c_fit,
            ls="-" if optimized else "--",
            lw=2.0,
            color=curve_color,
            label=label,
        )

        # Draggable background (P0) line and peak handle.
        self._bg_line = ax.axhline(
            result["background"],
            color=_BG_COLOR,
            ls="--",
            lw=1.2,
            label="background P₀",
        )
        self._peak_marker = ax.plot(
            [result["peak_time"]],
            [result["peak_concentration"]],
            marker="o",
            ms=9,
            mfc="white",
            mec=_PEAK_COLOR,
            mew=1.8,
            ls="None",
            zorder=9,
            label="peak",
        )[0]

        t0_ts = result["window_start"] + pd.to_timedelta(
            result["emission_start_s"], unit="s"
        )
        self._t0_line = ax.axvline(
            t0_ts, color=_T0_COLOR, ls="--", lw=1.4, label="emission start"
        )
        self._peak_line = ax.axvline(
            result["peak_time"], color=_PEAK_COLOR, ls=":", lw=1.4, label="source-off"
        )
        ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    # -- draggable handles ------------------------------------------------
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

    def _on_press(self, event) -> None:
        """Grab a handle if the click landed on it (disabling the span picker)."""
        if event.inaxes is not self.ax or event.x is None or self._result is None:
            return
        # Peak handle first (it sits on the source-off line): needs x *and* y.
        if self._peak_marker is not None and event.y is not None:
            mx = mdates.date2num(self._peak_marker.get_xdata()[0])
            my = self._peak_marker.get_ydata()[0]
            px, py = self.ax.transData.transform((mx, my))
            if abs(event.x - px) <= 8 and abs(event.y - py) <= 8:
                self._grab("peakval")
                return
        for name, line in (("t0", self._t0_line), ("peak", self._peak_line)):
            px = self._line_x_pixels(line)
            if px is not None and abs(event.x - px) <= 6:
                self._grab(name)
                return
        py = self._line_y_pixels(self._bg_line)
        if py is not None and event.y is not None and abs(event.y - py) <= 6:
            self._grab("bg")

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
        self.span.set_active(True)
        if name in ("t0", "peak"):
            line = self._t0_line if name == "t0" else self._peak_line
            ts = self._num_to_ts(mdates.date2num(line.get_xdata()[0]))
            if name == "t0":
                self._t0_override = ts
            else:
                self._peak_override = ts
        elif name == "bg":
            self._bg_override = float(self._bg_line.get_ydata()[0])
        elif name == "peakval":
            self._peakval_override = float(self._peak_marker.get_ydata()[0])
        self._recompute(optimize=False)

    def _on_scroll(self, event) -> None:
        """Scroll over the plot: speed up / slow down the decay rate."""
        if self._result is None or event.inaxes is not self.ax:
            return
        if self.toolbar.mode:  # don't fight an active pan/zoom
            return
        base = self._rate_override
        if base is None:
            base = self._result.get("decay_rate")
        if not base or not np.isfinite(base) or base <= 0:
            base = 1.0 / 1800.0
        # Scroll up -> faster decay (larger rate), down -> slower.
        self._rate_override = float(max(base * (_SCROLL_RATE_STEP**event.step), 1e-9))
        self._recompute(optimize=False)

    @staticmethod
    def _num_to_ts(xnum: float) -> pd.Timestamp:
        """Matplotlib date number -> tz-naive pandas Timestamp."""
        return pd.Timestamp(mdates.num2date(xnum)).tz_localize(None)

    # -- results ----------------------------------------------------------
    def _format_result(self, result: dict) -> str:
        """Render the fit dict as a short multi-line results summary."""
        unit = result["unit"]
        model_name = result["model"].replace("_", " ")
        if self._optimized:
            header = (
                f"Model: {model_name}   R² = {result['r_squared']:.4f} "
                f"(decay {result['decay_r_squared']:.4f})   n = {result['n_points']}"
            )
        else:
            header = (
                f"Preview (initial guess) — {model_name}, R² = "
                f"{result['r_squared']:.4f}. Press Fit to optimise."
            )
        lines = [
            header,
            f"Background P0 = {_fmt(result['background'])} {unit}   "
            f"Peak = {_fmt(result['peak_concentration'])} {unit}",
            f"Decay rate = {_fmt(result['decay_rate_per_hour'])} 1/h   "
            f"Emission rate E = {_fmt(result['emission_rate'])} {unit}/s",
        ]
        if "zeroth_order_rate" in result:
            lines.append(
                f"Constant removal a = {_fmt(result['zeroth_order_rate'])} {unit}/s"
            )
        if "loss_rate_per_hour" in result:
            lines.append(
                f"First-order loss = {_fmt(result['loss_rate_per_hour'])} 1/h"
                f"   (half-life {_fmt(result['half_life_hours'], 3)} h)"
            )
        if "second_order_rate" in result:
            lines.append(
                f"Second-order (coagulation) rate C = "
                f"{_fmt(result['second_order_rate'])} ({unit}·s)⁻¹"
            )
        if "wall_loss_rate_per_hour" in result:
            wall = result["wall_loss_rate_per_hour"]
            lines.append(
                "Estimated wall-loss/deposition rate = "
                f"{_fmt(wall)} 1/h (= first-order loss − known ACH)"
            )
            if wall < 0:
                lines.append(
                    "  ⚠ Negative wall loss: the fitted first-order loss is "
                    "below the entered air-exchange rate (ACH). Check the ACH "
                    "value or the fit window."
                )
        if "source_strength" in result:
            lines.append(
                f"Source strength ≈ {_fmt(result['source_strength'])} "
                f"{result['source_strength_unit']}   "
                f"(total emitted ≈ {_fmt(result['total_emitted'])} "
                f"{result['total_emitted_unit']})"
            )
        return "\n".join(lines)

    def current_time_xlim(self):
        """Time-axis limits as date numbers, or None."""
        if self.ax.has_data():
            return self.ax.get_xlim()
        return None
