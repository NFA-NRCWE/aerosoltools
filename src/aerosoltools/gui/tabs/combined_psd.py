"""Multi-dataset combined-PSD comparison tab."""

from __future__ import annotations

import traceback

from .. import helpers
from ..qt import QtCore, QtWidgets
from ._base import _active_color_cycle, _PlotTab


class CombinedPSDTab(_PlotTab):
    """Overlay mean particle size distributions across datasets and tasks.

    A comparison tab (like :class:`OverlayTab`): it reads *all* of the project's
    size-resolved (2D) datasets rather than the active one. The user ticks which
    datasets and which activities to compare; one mean-PSD curve is drawn per
    (dataset × activity) pair via the library's own :meth:`Aerosol2D.plot_psd`
    (``ax=`` shared), then relabelled/recoloured so each combination is distinct.
    """

    export_tag = "combined_psd"

    def __init__(self, main):
        """Build the combined-PSD controls, dataset/activity lists and plot."""
        super().__init__(main, nrows=1)

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.setChecked(True)
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
        self.ds_list.setMaximumWidth(280)
        self.ds_list.itemChanged.connect(self._on_ds_changed)

        self.act_list = QtWidgets.QListWidget()
        self.act_list.setMaximumWidth(280)
        self.act_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.act_list.itemSelectionChanged.connect(self._draw)

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

        # Left column (controls + toolbar + plot) and a full-height side panel.
        self._left_col = QtWidgets.QVBoxLayout()
        self._layout.removeItem(self.controls)
        self._layout.removeWidget(self.toolbar)
        self._layout.removeWidget(self.canvas)
        self._left_col.addLayout(self.controls)
        self._left_col.addWidget(self.toolbar)
        self._left_col.addWidget(self.canvas, stretch=1)
        body = QtWidgets.QHBoxLayout()
        body.addLayout(self._left_col, stretch=1)
        body.addLayout(side)
        self._layout.addLayout(body, stretch=1)

        self.ax = self.figure.add_subplot(111)

    # -- data access -------------------------------------------------------
    @property
    def _datasets_2d(self):
        """The project's size-resolved (2D) datasets."""
        return [d for d in self.main.project.datasets if helpers.is_2d(d.obj)]

    def _selected_activities(self) -> list:
        """Names of the activities ticked in the activity list."""
        return [i.text() for i in self.act_list.selectedItems()]

    # -- list sync ---------------------------------------------------------
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

    # -- interaction -------------------------------------------------------
    def _on_ds_changed(self, item) -> None:
        """Persist a dataset's include flag and redraw."""
        if self._building:
            return
        ds = self.main.project.get(item.data(QtCore.Qt.UserRole))
        if ds is not None:
            ds.psd_on = item.checkState() == QtCore.Qt.Checked
            self._draw()

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        """Re-sync the dataset/activity lists and redraw."""
        self._sync_datasets()
        self._sync_activities()
        self._draw()

    def _draw_on(self, ax) -> None:
        """Draw one mean-PSD curve per (dataset x activity) onto ``ax``."""
        ax.clear()
        datasets = [d for d in self._datasets_2d if d.psd_on]
        activities = self._selected_activities()
        colors = _active_color_cycle()
        normalize = self.normalize.isChecked()
        show_band = self.band.isChecked()

        plotted = 0
        ci = 0
        single = len(datasets) == 1 or len(activities) == 1
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
                for j, line in enumerate(new_lines):
                    line.set_color(color)
                    line.set_label(label if j == 0 else "_nolegend_")
                for coll in new_coll:  # fill_between ±1σ envelope
                    if show_band:
                        coll.set_color(color)
                        coll.set_alpha(0.18)
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

        if self.log_y.isChecked():
            ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)

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
        """Draw the comparison onto a fresh export figure."""
        self._draw_on(fig.add_subplot(111))
