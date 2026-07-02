"""Decay / source-strength fitting tab.

Fits a single-zone emission + decay peak model (see
:mod:`aerosoltools._core.decay`) to one activity/time window of a chosen
metric on the active dataset. From the fit it reports the emission (source)
strength and the loss kinetics — a first-order loss (air exchange + wall
deposition), a second-order loss (coagulation), or both combined — and, when
the chamber volume or an independently known air exchange rate are supplied,
the derived source strength and wall-loss rate.
"""

from __future__ import annotations

import traceback

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from ..._core import decay as _decay
from .. import helpers
from ..qt import QtCore, QtWidgets
from ._base import _PlotTab

#: Fraction metrics that need a cut-off diameter alongside the kind.
_FRACTION_KINDS = ("PM", "PN", "PS", "PV")

#: Model label shown in the selector -> value passed to ``fit_decay``.
_MODEL_CHOICES = [
    ("Auto (best fit)", "auto"),
    ("First order (air exchange + wall loss)", "first_order"),
    ("Second order (coagulation)", "second_order"),
    ("Combined (first + second order)", "combined"),
]

#: Map digits and a sign to Unicode superscripts for exponent rendering.
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _fmt(value, sig: int = 4) -> str:
    """Format a number, rendering scientific notation as ``m×10ⁿ``.

    ``:.g`` uses ``e+NN`` for large/small magnitudes (e.g. ``1.01e+09``),
    which reads as raw code; convert that tail into a proper superscript power
    of ten. Values that fit in plain decimal are returned unchanged.
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    text = f"{value:.{sig}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        power = str(int(exponent))  # normalise "+09" -> "9", "-05" -> "-5"
        return f"{mantissa}×10{power.translate(_SUPERSCRIPT)}"
    return text


class DecayTab(_PlotTab):
    """Fit an emission + decay peak to estimate source strength and loss rates."""

    export_tag = "decay"

    def __init__(self, main):
        """Build the segment/metric/model/parameter controls and the plot."""
        super().__init__(main, nrows=1)

        self.controls.addWidget(QtWidgets.QLabel("Segment:"))
        self.segment = QtWidgets.QComboBox()
        self.segment.setToolTip(
            "Activity/time window to fit — mark a task covering the peak "
            "(background → rise → peak → decay) in the Time series tab."
        )
        self.segment.currentIndexChanged.connect(lambda: self._draw_segment_only())
        self.controls.addWidget(self.segment)

        self.controls.addWidget(QtWidgets.QLabel("Metric:"))
        self.metric_kind = QtWidgets.QComboBox()
        self.metric_kind.currentTextChanged.connect(self._on_metric_kind_change)
        self.controls.addWidget(self.metric_kind)
        self.metric_cut = QtWidgets.QComboBox()
        self.metric_cut.setEditable(True)
        self.metric_cut.setFixedWidth(70)
        self.controls.addWidget(self.metric_cut)
        self.metric_cut_label = QtWidgets.QLabel("µm")
        self.controls.addWidget(self.metric_cut_label)

        self.controls.addWidget(QtWidgets.QLabel("Model:"))
        self.model = QtWidgets.QComboBox()
        for label, value in _MODEL_CHOICES:
            self.model.addItem(label, value)
        self.model.setToolTip(
            "Loss model to fit:\n"
            "• First order — losses ∝ concentration (air exchange + deposition).\n"
            "• Second order — losses ∝ concentration² (coagulation).\n"
            "• Combined — both terms together.\n"
            "• Auto — fit all three and pick the best (favouring the simpler)."
        )
        self.controls.addWidget(self.model)

        self.controls.addWidget(QtWidgets.QLabel("Volume (m³):"))
        self.volume = QtWidgets.QLineEdit()
        self.volume.setPlaceholderText("optional")
        self.volume.setFixedWidth(70)
        self.volume.setToolTip(
            "Chamber/room volume in m³. Converts the fitted emission rate into a "
            "source strength (emission rate × volume) and a total emitted amount."
        )
        self.controls.addWidget(self.volume)

        self.controls.addWidget(QtWidgets.QLabel("Known ACH (1/h):"))
        self.ach = QtWidgets.QLineEdit()
        self.ach.setPlaceholderText("optional")
        self.ach.setFixedWidth(60)
        self.ach.setToolTip(
            "An independently known air exchange rate (1/h, e.g. from a "
            "tracer-gas test). When given, the wall-loss/deposition rate is "
            "estimated as the fitted first-order loss minus this value."
        )
        self.controls.addWidget(self.ach)

        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("primary")
        self.fit_btn.setToolTip(
            "Fit the emission + decay peak model to the selected segment."
        )
        self.fit_btn.clicked.connect(self._fit)
        self.controls.addWidget(self.fit_btn)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        self.ax = self.figure.add_subplot(111)

        self.result_label = QtWidgets.QLabel(
            "Mark a peak (rise + decay) as a task in Time series, pick it as the "
            "segment here, choose a model, then click Fit."
        )
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._layout.addWidget(self.result_label)

        # Identity of the object the selectors were last synced from, so the
        # metric list re-syncs when the active dataset changes.
        self._cols_obj = None

    # -- option syncing ------------------------------------------------------
    def _sync_segments(self) -> None:
        """Repopulate the segment combo from the active object's activities."""
        current = self.segment.currentText()
        self.segment.blockSignals(True)
        self.segment.clear()
        self.segment.addItems(list(self.obj.activities))
        idx = self.segment.findText(current)
        self.segment.setCurrentIndex(idx if idx >= 0 else 0)
        self.segment.blockSignals(False)

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
        """Show the cut-off field only for the size-selective fractions."""
        is_fraction = self.metric_kind.currentText().strip() in _FRACTION_KINDS
        self.metric_cut.setVisible(is_fraction)
        self.metric_cut_label.setVisible(is_fraction)

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
        """Re-sync the selectors for the active dataset and redraw the segment."""
        if self.obj is None:
            return
        if self._cols_obj is not self.obj:
            self._sync_metric_options()
            self._cols_obj = self.obj
        self._sync_segments()
        self._draw_segment_only()

    def _segment_series(self, segment: str, metric: str):
        """Return ``(times, values, unit)`` for one segment of a metric."""
        series, unit = self.obj._get_metric_series(metric)
        mask = self.obj.data[segment].astype(bool)
        times = self.obj.time[mask]
        values = np.asarray(series.loc[mask], dtype=float)
        finite = np.isfinite(values)
        return times[finite], values[finite], unit

    def _draw_segment_only(self) -> None:
        """Plot the raw segment data (no fit curve) after a selection change."""
        if self.obj is None:
            return
        ax = self.ax
        ax.clear()
        segment = self.segment.currentText()
        try:
            if not segment or segment not in self.obj.activities:
                raise ValueError("No segment selected.")
            metric = self._build_metric()
            times, values, unit = self._segment_series(segment, metric)
            ax.plot(times, values, "o", ms=3, color="tab:blue", label=segment)
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{metric}, {unit}".strip(", "))
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(mdates.AutoDateLocator())
            )
            ax.legend(loc="upper right", fontsize=8)
        except Exception:
            pass  # an empty/placeholder axis is fine before a segment is picked
        self._sync_toolbar_home()
        self.canvas.draw_idle()

    def _fit(self) -> None:
        """Run the emission + decay fit and draw the fitted curve + results."""
        if self.obj is None:
            return
        segment = self.segment.currentText()
        if not segment:
            self.result_label.setText("Pick a segment to fit.")
            return
        metric = self._build_metric()
        volume = self._to_float(self.volume.text(), None)
        ach = self._to_float(self.ach.text(), None)
        try:
            result = self.obj.fit_decay(
                segment,
                metric=metric,
                model=self.model.currentData(),
                volume=volume,
                air_exchange_rate=ach,
            )
        except Exception as exc:
            self.result_label.setText(f"Fit failed: {exc}")
            return
        try:
            self._draw_fit(segment, metric, result)
        except Exception:
            self._show_message(
                "Fit succeeded but the plot failed:\n" + traceback.format_exc(limit=1)
            )
        self.result_label.setText(self._format_result(result))

    def _draw_fit(self, segment: str, metric: str, result: dict) -> None:
        """Draw the segment data with the fitted emission + decay curve overlaid."""
        ax = self.ax
        ax.clear()
        times, values, unit = self._segment_series(segment, metric)

        ax.plot(
            times,
            values,
            "o",
            ms=3,
            color="tab:blue",
            alpha=0.6,
            label=f"{segment} (data)",
        )
        # Dense curve from the fitted model over the window (seconds from start).
        t_start = times[0]
        span_s = (times[-1] - t_start).total_seconds()
        t_fit = np.linspace(0.0, float(span_s), 400)
        c_fit = _decay.decay_curve(result["model"], t_fit, result["model_popt"])
        t_fit_dates = t_start + pd.to_timedelta(t_fit, unit="s")
        label = f"Fit ({result['model'].replace('_', ' ')})"
        ax.plot(t_fit_dates, c_fit, "-", lw=2.0, color="crimson", label=label)

        # Mark the fitted peak time.
        peak_t = result["peak_time"]
        ax.axvline(peak_t, color="gray", ls=":", lw=1.2, alpha=0.8)

        ax.set_xlabel("Time")
        ax.set_ylabel(f"{metric}, {unit}".strip(", "))
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        ax.legend(loc="upper right", fontsize=8)
        self._sync_toolbar_home()
        self.canvas.draw_idle()

    @staticmethod
    def _format_result(result: dict) -> str:
        """Render the fit dict as a short multi-line results summary."""
        unit = result["unit"]
        model_name = result["model"].replace("_", " ")
        lines = [
            f"Model: {model_name}   R² = {result['r_squared']:.4f}   "
            f"n = {result['n_points']}",
            f"Background P0 = {_fmt(result['background'])} {unit}   "
            f"Peak = {_fmt(result['peak_concentration'])} {unit}",
            f"Emission rate E = {_fmt(result['emission_rate'])} {result['unit']}/s",
        ]
        if "loss_rate_per_hour" in result:
            line = (
                f"First-order loss = {_fmt(result['loss_rate_per_hour'])} 1/h"
                f"   (half-life {_fmt(result['half_life_hours'], 3)} h)"
            )
            lines.append(line)
        if "second_order_rate" in result:
            lines.append(
                f"Second-order (coagulation) rate C = "
                f"{_fmt(result['second_order_rate'])} ({unit}·s)⁻¹"
            )
        if "wall_loss_rate_per_hour" in result:
            lines.append(
                "Estimated wall-loss/deposition rate = "
                f"{_fmt(result['wall_loss_rate_per_hour'])} 1/h "
                "(= first-order loss − known ACH)"
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
