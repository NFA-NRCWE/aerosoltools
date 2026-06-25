"""Time/size heatmap tab (Aerosol2D)."""

from __future__ import annotations

import traceback

from .. import helpers
from ..qt import QtCore, QtWidgets
from ._base import _PlotTab

#: Lower color-scale limit forced when a log scale is requested but the data
#: contain zeros (a log scale cannot show ≤ 0), so the heatmap stays log instead
#: of silently falling back to linear.
_LOG_FLOOR = 0.1


class HeatmapTab(_PlotTab):
    """Total concentration + time/size heatmap (uses ``plot_timeseries``)."""

    export_tag = "heatmap"

    def __init__(self, main):
        """Build the heatmap controls and the two stacked panels."""
        super().__init__(main, nrows=2)

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.normalize)

        self.log = QtWidgets.QCheckBox("Log color scale")
        self.log.setChecked(True)
        self.log.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.log)

        self.controls.addWidget(QtWidgets.QLabel("Color min:"))
        self.cmin = QtWidgets.QLineEdit()
        self.cmin.setPlaceholderText("auto")
        self.cmin.setFixedWidth(70)
        self.cmin.editingFinished.connect(self.refresh)
        self.controls.addWidget(self.cmin)

        self.controls.addWidget(QtWidgets.QLabel("Color max:"))
        self.cmax = QtWidgets.QLineEdit()
        self.cmax.setPlaceholderText("auto")
        self.cmax.setFixedWidth(70)
        self.cmax.editingFinished.connect(self.refresh)
        self.controls.addWidget(self.cmax)

        self.show_acts = QtWidgets.QCheckBox("Show activities")
        self.show_acts.setChecked(True)
        self.show_acts.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.show_acts)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Tracks whether the previous draw floored the lower colour limit, so the
        # warning bubble pops only when that state newly turns on (not on every
        # unrelated redraw, e.g. tweaking Color max).
        self._last_floored = False

    def _cap(self, edit) -> float:
        """Return the field value, or 0.0 (which plot_timeseries reads as auto)."""
        text = edit.text().strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0

    def _plot_on(self, fig) -> bool:
        """Draw the total-conc + heatmap panels onto ``fig``.

        Returns ``True`` when the lower colour limit had to be floored to keep a
        log scale (the data contained zeros and the user left ``Color min`` on
        auto), so the caller can warn the user.
        """
        # plot_timeseries adds its own colorbar axes, so build panels fresh.
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        # Normalize on a working copy so the loaded object is left untouched.
        target = self.obj
        if self.normalize.isChecked():
            target = self.obj.copy_self()
            target.normalize_logdp()

        # y_3d caps the colour scale: (min, max), where 0 means "automatic".
        # A log colour scale needs a strictly positive lower cap; when the user
        # leaves Color min on auto but the data contain zeros, plot_timeseries
        # would silently drop back to a linear scale. Floor the lower cap to
        # _LOG_FLOOR in that case so the (more useful) log scale is kept.
        lower = self._cap(self.cmin)
        floored = False
        if self.log.isChecked() and lower <= 0:
            if bool((target.size_data <= 0).to_numpy().any()):
                lower = _LOG_FLOOR
                floored = True
        y_3d = (lower, self._cap(self.cmax))

        target.plot_timeseries(
            log=self.log.isChecked(),
            y_3d=y_3d,
            ax1=ax1,
            ax2=ax2,
            mark_activities=self.show_acts.isChecked(),
        )
        return floored

    def _warn_floor(self) -> None:
        """Pop a non-blocking 'speech bubble' next to the Color min field."""
        pos = self.cmin.mapToGlobal(QtCore.QPoint(0, self.cmin.height()))
        QtWidgets.QToolTip.showText(
            pos,
            f"Lower colour limit set to {_LOG_FLOOR:g} because the data contain "
            "zeros, which a log colour scale can't show.\nType a value in "
            "'Color min' (or untick 'Log color scale') to override.",
            self.cmin,
        )

    def refresh(self) -> None:
        """Redraw the total-concentration + heatmap panels."""
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        try:
            floored = self._plot_on(self.figure)
            self.canvas.draw_idle()
        except Exception:
            self._show_message(
                "Could not draw heatmap:\n" + traceback.format_exc(limit=1)
            )
            return
        # Warn only on the transition into the floored state, so the bubble does
        # not reappear on every redraw while the floor stays in effect.
        if floored and not self._last_floored:
            self._warn_floor()
        self._last_floored = floored

    def current_time_xlim(self):
        # The top (total-conc) panel shares the time x-axis with the heatmap.
        """Time-axis limits of the top panel as date numbers, or None."""
        axes = self.figure.axes
        if axes and axes[0].has_data():
            return axes[0].get_xlim()
        return None
