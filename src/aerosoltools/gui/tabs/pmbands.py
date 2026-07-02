"""Stacked PM-bands tab (Aerosol2D)."""

from __future__ import annotations

import traceback

import matplotlib.dates as mdates
import numpy as np

from .. import helpers
from ..qt import QtWidgets
from ._base import _PlotTab

#: A categorical, high-contrast palette (Okabe–Ito, colour-blind safe) so the
#: bands stay easy to tell apart — unlike the old pale diverging scheme whose
#: middle colours (and the first band's edge) looked alike.
_BAND_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#666666",  # grey
]


class PMBandsTab(_PlotTab):
    """Stacked size-selective Pₓ bands over time.

    Reuses the library's ``dtype_converter`` + ``PM_calc`` on a working copy for
    the numerics, then renders the stacked bands on the embedded axis.
    """

    export_tag = "pm_bands"

    def __init__(self, main):
        """Build the PM-bands controls and plot."""
        super().__init__(main, nrows=1)

        self.dtype = QtWidgets.QComboBox()
        self.dtype.addItems(["dM", "dN", "dS", "dV"])
        self.dtype.setToolTip(
            "Distribution basis the Pₓ bands are computed on: mass (dM), number "
            "(dN), surface (dS) or volume (dV)."
        )
        self.dtype.currentIndexChanged.connect(self.refresh)
        self.controls.addWidget(QtWidgets.QLabel("Basis:"))
        self.controls.addWidget(self.dtype)

        self.values = QtWidgets.QLineEdit("0.5, 2.5, 10")
        self.values.setFixedWidth(140)
        self.values.setToolTip(
            "Comma-separated cut diameters in µm (e.g. 0.5, 2.5, 10). Each gives a "
            "size-selective fraction such as PM2.5 = mass below 2.5 µm."
        )
        self.values.editingFinished.connect(self.refresh)
        self.controls.addWidget(QtWidgets.QLabel("Cut-offs (µm):"))
        self.controls.addWidget(self.values)

        self.cumulative = QtWidgets.QCheckBox("Cumulative")
        self.cumulative.setToolTip(
            "Stack cumulative fractions (each band is everything below its cut-off) "
            "instead of independent size bands between successive cut-offs."
        )
        self.cumulative.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.cumulative)

        self.activity = QtWidgets.QComboBox()
        self.activity.currentIndexChanged.connect(self.refresh)
        self.controls.addWidget(QtWidgets.QLabel("Activity:"))
        self.controls.addWidget(self.activity)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        self.ax = self.figure.add_subplot(111)

    def _parse_values(self) -> list[float]:
        """Parse the comma-separated cut-off field into sorted floats."""
        out: list[float] = []
        for part in self.values.text().split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(float(part))
            except ValueError:
                continue
        return sorted(out)

    def _sync_activities(self) -> None:
        """Repopulate the activity combo, preserving the selection."""
        self.activity.blockSignals(True)
        current = self.activity.currentText()
        self.activity.clear()
        self.activity.addItems(self.obj.activities)
        idx = self.activity.findText(current)
        self.activity.setCurrentIndex(idx if idx >= 0 else 0)
        self.activity.blockSignals(False)

    def _plot_on(self, ax) -> None:
        """Compute and draw the stacked Pₓ bands onto ``ax``."""
        values = self._parse_values()
        if not values:
            raise ValueError("Enter one or more numeric cut-off diameters (µm).")

        dtype = self.dtype.currentText()
        dchar = dtype[-1]
        work = self.obj.copy_self()
        work.dtype_converter(dtype=dtype)
        for pm in values:
            work.pm_calc(dtype=dtype, PM=pm)
        activity = self.activity.currentText()
        mask = work.data[activity].astype(bool)
        pm_data = work.extra_data.loc[mask]
        _, unit = helpers.describe(work)

        ax.clear()
        x = pm_data.index
        labels = [f"P{dchar}{v:g}" for v in values]
        cmap = _BAND_COLORS
        present = [(i, lab) for i, lab in enumerate(labels) if lab in pm_data.columns]

        if self.cumulative.isChecked():
            # Stacked cumulative areas: 0→Pₓ₀, Pₓ₀→Pₓ₁, …  (top of stack = total).
            # Each layer also gets a crisp top outline in its own colour so the
            # boundaries between stacked bands read clearly.
            prev = None
            for i, label in present:
                series = pm_data[label]
                lower = 0 if prev is None else prev
                color = cmap[i % len(cmap)]
                ax.fill_between(x, lower, series, label=label, color=color, alpha=0.85)
                ax.plot(x, series, color=color, lw=1.4, alpha=0.95)
                prev = series
        else:
            # Differential bands, each drawn from zero so the concentration in
            # each size range is shown independently. Tallest band first (drawn
            # at the back); every band also gets a full-opacity outline so a
            # large-particle curve is never lost behind a dominant smaller one.
            bands = []
            prev_label = None
            for i, label in present:
                if prev_label is None:
                    band = pm_data[label]
                    rng = f"0–{label}"
                else:
                    band = pm_data[label] - pm_data[prev_label]
                    rng = f"{prev_label}–{label}"
                bands.append((i, rng, band))
                prev_label = label
            for i, rng, band in sorted(bands, key=lambda t: -float(np.nanmean(t[2]))):
                avg = float(np.nanmean(band))
                color = cmap[i % len(cmap)]
                ax.fill_between(x, 0, band, color=color, alpha=0.30)
                # Spell out "mean" (rather than "μ") so it isn't mistaken for the
                # µ in µg/m³ already shown on the y-axis.
                ax.plot(
                    x,
                    band,
                    color=color,
                    lw=1.8,
                    label=f"{rng} (mean {avg:.2g} {unit})",
                )

        ax.set_ylabel(f"P{dchar}, {unit}")
        ax.set_xlabel("Time")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        ax.legend(loc="upper right")

    def refresh(self) -> None:
        """Recompute and redraw the stacked PM bands."""
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        self._sync_activities()
        try:
            self._plot_on(self.ax)
            self._sync_toolbar_home()
            self.canvas.draw_idle()
        except ValueError as exc:
            self._show_message(str(exc))
        except Exception:
            self._show_message(
                "Could not compute PM bands:\n" + traceback.format_exc(limit=1)
            )

    def current_time_xlim(self):
        """Time-axis limits as date numbers, or None."""
        if self.ax.has_data():
            return self.ax.get_xlim()
        return None
