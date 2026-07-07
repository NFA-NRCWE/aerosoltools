"""Aerodynamic vs optical comparison tab (correlated APS / :class:`Aerosol3d`).

Shows the correlated optical×aerodynamic concentration matrix as a heat map — the
view that is unique to a correlated APS, mapping how each particle population's
optically reported size relates to its aerodynamically reported size.
"""

from __future__ import annotations

import traceback

from ..qt import QtWidgets
from ._base import _PlotTab


class AeroOpticalTab(_PlotTab):
    """Compare aerodynamic and optical sizing for a correlated APS dataset."""

    export_tag = "aero_optical"

    def __init__(self, main):
        """Build the activity/scale controls and the comparison heat map."""
        super().__init__(main, nrows=1)

        self.controls.addWidget(QtWidgets.QLabel("Activity:"))
        self.activity = QtWidgets.QComboBox()
        self.activity.setToolTip(
            "Restrict the comparison to one marked activity (or the whole record)."
        )
        self.activity.currentIndexChanged.connect(lambda: self.refresh())
        self.controls.addWidget(self.activity)

        self.controls.addWidget(QtWidgets.QLabel("View:"))
        self.view = QtWidgets.QComboBox()
        self.view.addItem("3D (over time)", "3d")
        self.view.addItem("2D (time-aggregated)", "2d")
        self.view.setToolTip(
            "3D: optical Ø (x), aerodynamic Ø (y), time (z), concentration as "
            "colour — drag to rotate. 2D: the time-summed comparison heat map."
        )
        self.view.currentIndexChanged.connect(lambda: self.refresh())
        self.controls.addWidget(self.view)

        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        self._cols_obj = None

    def _obj3d(self):
        """The active dataset's parent object if it's a correlated APS, else None."""
        obj = self.main.active_obj
        if obj is not None and getattr(obj, "is_correlated", False):
            return obj
        return None

    def refresh(self) -> None:
        """Redraw the comparison for the active correlated APS dataset."""
        obj = self._obj3d()
        self.figure.clear()
        if obj is None:
            self.canvas.draw_idle()
            return

        # Keep the activity picker in sync with the dataset.
        if self._cols_obj is not obj:
            self._cols_obj = obj
            self.activity.blockSignals(True)
            current = self.activity.currentText()
            self.activity.clear()
            self.activity.addItems(list(obj.activities))
            idx = self.activity.findText(current)
            self.activity.setCurrentIndex(idx if idx >= 0 else 0)
            self.activity.blockSignals(False)

        act = self.activity.currentText()
        act = None if act in ("", "All data") else act
        is_3d = self.view.currentData() == "3d"
        # Constrained layout collapses a 3D axes + colorbar; use it only in 2D.
        self.figure.set_layout_engine("none" if is_3d else "constrained")
        ax = self.figure.add_subplot(111, projection="3d" if is_3d else None)
        try:
            if is_3d:
                obj.plot_aero_optical_3d(activity=act, ax=ax)
            else:
                obj.plot_aero_vs_optical(activity=act, ax=ax)
        except Exception:
            ax.remove()
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Could not draw comparison:\n" + traceback.format_exc(limit=1),
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
            )
        self._sync_toolbar_home()
        self.canvas.draw_idle()
