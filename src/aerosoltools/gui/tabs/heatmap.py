"""Time/size heatmap tab (Aerosol2D)."""

from __future__ import annotations

import traceback

from .. import helpers, theme
from ..qt import QtWidgets
from ._base import _PlotTab


class HeatmapTab(_PlotTab):
    """Total concentration + time/size heatmap (uses ``plot_timeseries``)."""

    export_figsize = theme.EXPORT_FIGSIZE_TALL
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

    def _cap(self, edit) -> float:
        """Return the field value, or 0.0 (which plot_timeseries reads as auto)."""
        text = edit.text().strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0

    def _plot_on(self, fig) -> None:
        """Draw the total-conc + heatmap panels onto ``fig``."""
        # plot_timeseries adds its own colorbar axes, so build panels fresh.
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        # y_3d caps the colour scale: (min, max), where 0 means "automatic".
        # Note: for a log colour scale the lower cap must be > 0.
        y_3d = (self._cap(self.cmin), self._cap(self.cmax))

        # Normalize on a working copy so the loaded object is left untouched.
        target = self.obj
        if self.normalize.isChecked():
            target = self.obj.copy_self()
            target.normalize_logdp()

        target.plot_timeseries(
            log=self.log.isChecked(),
            y_3d=y_3d,
            ax1=ax1,
            ax2=ax2,
            mark_activities=self.show_acts.isChecked(),
        )

    def refresh(self) -> None:
        """Redraw the total-concentration + heatmap panels."""
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        try:
            self._plot_on(self.figure)
            self.canvas.draw_idle()
        except Exception:
            self._show_message(
                "Could not draw heatmap:\n" + traceback.format_exc(limit=1)
            )

    def _render_export(self, fig) -> None:
        """Draw the panels onto a fresh export figure."""
        self._plot_on(fig)

    def current_time_xlim(self):
        # The top (total-conc) panel shares the time x-axis with the heatmap.
        """Time-axis limits of the top panel as date numbers, or None."""
        axes = self.figure.axes
        if axes and axes[0].has_data():
            return axes[0].get_xlim()
        return None
