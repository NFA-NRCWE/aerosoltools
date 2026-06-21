"""Particle size distribution tab (Aerosol2D)."""

from __future__ import annotations

import traceback

from .. import helpers, theme
from ..qt import QtWidgets
from ._base import _PlotTab


class PSDTab(_PlotTab):
    """Mean particle size distribution per activity (uses ``plot_psd``)."""

    export_tag = "psd"

    def __init__(self, main):
        """Build the PSD controls and plot."""
        super().__init__(main, nrows=1)

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.setChecked(True)
        self.normalize.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.normalize)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.log_y)

        self.controls.addWidget(QtWidgets.QLabel("Activities:"))
        self.act_list = QtWidgets.QListWidget()
        self.act_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.act_list.setMaximumHeight(70)
        self.act_list.itemSelectionChanged.connect(self.refresh)
        self.controls.addWidget(self.act_list)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        self.ax = self.figure.add_subplot(111)

    def _sync_activities(self) -> None:
        """Rebuild the activity multi-select, preserving the selection."""
        self.act_list.blockSignals(True)
        selected = {i.text() for i in self.act_list.selectedItems()}
        self.act_list.clear()
        for name in self.obj.activities:
            self.act_list.addItem(name)
            if name in selected:
                self.act_list.item(self.act_list.count() - 1).setSelected(True)
        self.act_list.blockSignals(False)

    def _plot_on(self, fig) -> None:
        """Draw the PSD onto a fresh axis on ``fig``."""
        selected = [i.text() for i in self.act_list.selectedItems()] or None
        fig.clear()
        ax = fig.add_subplot(111)
        self.obj.plot_psd(
            activities=selected,
            normalize=self.normalize.isChecked(),
            ax=ax,
        )
        if self.log_y.isChecked():
            ax.set_yscale("log")

    def refresh(self) -> None:
        """Redraw the mean PSD for the selected activities."""
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        self._sync_activities()
        try:
            self._plot_on(self.figure)
            self.ax = self.figure.axes[0] if self.figure.axes else None
            # The core plot_psd picks colours tuned for a white background; on
            # the dark theme they are too dark, so brighten them for the screen
            # only (exports keep the core's report colours via _render_export).
            if self.ax is not None and theme.is_dark():
                self._brighten_for_dark(self.ax)
            self.canvas.draw_idle()
        except Exception:
            self._show_message("Could not draw PSD:\n" + traceback.format_exc(limit=1))

    @staticmethod
    def _brighten_for_dark(ax) -> None:
        """Recolour PSD lines (and their ±1σ fills) with the bright cycle."""
        cycle = theme.mpl_cycle()
        for i, line in enumerate(ax.get_lines()):
            line.set_color(cycle[i % len(cycle)])
        for i, coll in enumerate(ax.collections):  # fill_between envelopes
            coll.set_color(cycle[i % len(cycle)])
            coll.set_alpha(0.20)
        if ax.get_legend() is not None:
            ax.legend()  # rebuild so legend swatches match the new colours

    def _render_export(self, fig) -> None:
        """Draw the PSD onto a fresh export figure."""
        self._plot_on(fig)
