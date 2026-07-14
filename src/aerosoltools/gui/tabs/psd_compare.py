"""PSD comparison tab — overlay mean PSDs from several datasets/activities.

A display-only pane (the lognormal fitting lives in the single-dataset PSD tab):
it reads the project's size-resolved (2D) datasets and overlays one mean-PSD
curve per ticked (dataset × activity), so different instruments and activities
can be compared on one axis. Each dataset keeps its own stable colour. Bar/line,
log-Y and the ±σ band are available, but there is no fitting here — that keeps
this pane simple and predictable.
"""

from __future__ import annotations

import traceback

from ..app.sidebar import _color_icon
from ..logic import helpers
from ..qt import QtCore, QtGui, QtWidgets
from ..state.project import shade_of
from . import _psddraw as draw
from ._base import _active_color_cycle, _PlotTab


class ComparisonPSDTab(_PlotTab):
    """Overlay mean particle size distributions from several datasets/activities."""

    export_tag = "psd_comparison"

    def __init__(self, main):
        """Build the display controls, dataset/activity lists and the plot."""
        super().__init__(main, nrows=1)

        self.normalize = QtWidgets.QCheckBox("Normalize (dx/dlogDp)")
        self.normalize.setChecked(True)
        self.normalize.setToolTip("Plot dx/dlogDp (per decade of diameter).")
        self.normalize.stateChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.normalize)

        self.log_y = QtWidgets.QCheckBox("Log Y")
        self.log_y.stateChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.log_y)

        self.band = QtWidgets.QCheckBox("±σ band")
        self.band.setToolTip(
            "Shade each curve's ±1σ spread. Off by default, since the bands "
            "overlap heavily when several curves are compared."
        )
        self.band.stateChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.band)

        self.controls.addWidget(QtWidgets.QLabel("Display:"))
        self.display_mode = QtWidgets.QComboBox()
        self.display_mode.addItems(["Lines", "Bars"])
        self.display_mode.setToolTip(
            "Bars show each size bin's width (best for a single PSD); lines are "
            "clearer when several curves are compared on the same axes."
        )
        self.display_mode.currentIndexChanged.connect(lambda *_: self._draw())
        self.controls.addWidget(self.display_mode)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Side panel: datasets to compare (checkable) + activities (multi-select).
        self._building = False
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.itemChanged.connect(self._on_ds_changed)
        self.act_list = QtWidgets.QListWidget()
        self.act_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.act_list.itemSelectionChanged.connect(lambda: self._draw())
        self.ds_list.setToolTip(
            "Tick the 2D datasets and activities to compare. One mean-PSD curve "
            "is drawn per dataset × activity."
        )
        # Per-curve colour swatches (click to override the auto shade).
        self.color_list = QtWidgets.QListWidget()
        self.color_list.setToolTip(
            "One row per drawn curve. Each activity defaults to a shade of its "
            "dataset's colour; click a row to pick a custom colour."
        )
        self.color_list.itemClicked.connect(self._on_swatch_clicked)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Datasets to compare:"))
        side.addWidget(self.ds_list, stretch=1)
        side.addWidget(QtWidgets.QLabel("Activities:"))
        side.addWidget(self.act_list, stretch=1)
        side.addWidget(QtWidgets.QLabel("Curve colours (click to change):"))
        side.addWidget(self.color_list, stretch=1)
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        self._split_with_side(side_widget, sizes=(760, 320))

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
        if not selected and self.act_list.count():
            self.act_list.item(0).setSelected(True)  # default to "All data"
        self.act_list.blockSignals(False)

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

    def _activity_color(self, ds, act, n_selected: int) -> str:
        """Resolve a curve's colour: user override, else a shade of the base.

        With a single activity selected the datasets keep their own base colours
        (so a multi-instrument comparison is read by dataset colour); with
        several activities, each activity auto-shades its dataset's base colour
        (light/dark variants) so same-dataset curves stay distinguishable. A
        stored override always wins.
        """
        override = ds.activity_colors.get(act)
        if override:
            return override
        if n_selected <= 1:
            return ds.color or _active_color_cycle()[0]
        names = ["All data"] + self.main.project.user_activities()
        k = names.index(act) if act in names else 0
        return shade_of(ds.color, k, len(names))

    def _draw_on(self, ax) -> None:
        """Overlay one mean-PSD curve per (ticked dataset × selected activity)."""
        ax.clear()
        datasets = [d for d in self._datasets_2d if d.psd_on]
        activities = self._selected_activities()
        normalize = self.normalize.isChecked()
        show_band = self.band.isChecked()
        bars = self.display_mode.currentText() == "Bars"
        single = len(datasets) == 1 or len(activities) == 1

        plotted = 0
        drawn: list = []  # (ds, act, color, label) for the swatch list
        for ds in datasets:
            for act in activities:
                if act not in ds.obj.activities:
                    continue
                color = self._activity_color(ds, act, len(activities))
                # Label by whichever dimension varies, to keep the legend tidy.
                if single and len(activities) == 1:
                    label = ds.label
                elif single and len(datasets) == 1:
                    label = act
                else:
                    label = f"{ds.label} – {act}"
                xy = draw.draw_psd_curve(
                    ax,
                    ds.obj,
                    act,
                    color=color,
                    label=label,
                    normalize=normalize,
                    bars=bars,
                    show_band=show_band,
                )
                if xy is not None:
                    plotted += 1
                    drawn.append((ds, act, color, label))

        self._sync_color_list(drawn)

        if not plotted:
            msg = (
                "Tick at least one dataset and one activity to compare."
                if self._datasets_2d
                else "Load size-resolved (2D) datasets to compare their PSDs."
            )
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        ylim = draw.data_ylim(ax, self.log_y.isChecked())
        if self.log_y.isChecked():
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="upper right", fontsize=8)

    def _sync_color_list(self, drawn: list) -> None:
        """Populate the swatch list with one row per drawn curve."""
        self.color_list.blockSignals(True)
        self.color_list.clear()
        for ds, act, color, label in drawn:
            item = QtWidgets.QListWidgetItem(_color_icon(color), label)
            item.setData(QtCore.Qt.UserRole, (ds.id, act, color))
            self.color_list.addItem(item)
        self.color_list.blockSignals(False)

    def _on_swatch_clicked(self, item) -> None:
        """Open a colour dialog for a curve and store the chosen override."""
        data = item.data(QtCore.Qt.UserRole)
        if not data:
            return
        ds_id, act, color = data
        ds = self.main.project.get(ds_id)
        if ds is None:
            return
        chosen = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(color), self, f"Colour for {ds.label} – {act}"
        )
        if chosen.isValid():
            ds.activity_colors[act] = chosen.name()
            self._draw()

    def _draw(self) -> None:
        """Redraw onto the embedded axis, reporting any error in the figure."""
        try:
            self._draw_on(self.ax)
        except Exception:
            self._show_message(
                "Could not draw combined PSD:\n" + traceback.format_exc(limit=1)
            )
            return
        self._sync_toolbar_home()
        self.canvas.draw_idle()
