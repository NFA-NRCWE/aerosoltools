"""Aerodynamic vs optical comparison tab (correlated APS / :class:`Aerosol3d`).

Two linked panels: the top shows total concentration over time with a movable
time cursor; the bottom shows, for the selected time, a 3-D **bar** plot of the
correlated optical×aerodynamic distribution — one bar per (optical, aerodynamic)
size bin, its height *and* colour both the concentration (so it reads like a
surface with clearly separated, unequally sized bins). Drag the cursor on the
top panel to scan through time; the colour scale is fixed to the dataset's
global range so bars are comparable across times.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..qt import QtWidgets
from ._base import _PlotTab

_CMAP = "jet"  # match the 2-D heatmaps


class AeroOpticalTab(_PlotTab):
    """Time-cursor + 3-D bar view of a correlated APS record."""

    export_tag = "aero_optical"

    def __init__(self, main):
        """Build the (empty) controls and the two-panel figure with a cursor."""
        super().__init__(main, nrows=2)
        self.controls.addWidget(
            QtWidgets.QLabel("Drag the line on the top plot to pick a time.")
        )
        self.normalize = QtWidgets.QCheckBox("Normalize (dlogDp)")
        self.normalize.setToolTip(
            "Divide each cell by BOTH log bin widths — dN/(dlogDp_optical · "
            "dlogDp_aerodynamic) — so bins of unequal width are comparable. The "
            "top total-concentration panel stays on the raw counts."
        )
        self.normalize.stateChanged.connect(self.refresh)
        self.controls.addWidget(self.normalize)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # Constrained layout fights a 3-D axes + colorbar; place axes manually.
        self.figure.set_layout_engine("none")
        self._obj = None  # currently drawn correlated object
        self._times = None  # correlation time index
        self._total = None  # total concentration per time
        self._matrix = None  # raw (n_times, n_optical, n_aero)
        self._disp = None  # displayed matrix (raw or dlogDp-normalized)
        self._opt_edges = None
        self._aero_edges = None
        self._norm = None
        self._sel = 0  # selected time index
        self._cursor = None
        self._dragging = False

        self.ax_top = None
        self.ax_bot = None
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _obj3d(self):
        """Active parent object if it's a correlated APS, else None."""
        obj = self.main.active_obj
        return obj if getattr(obj, "is_correlated", False) else None

    # -- rendering ---------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild both panels for the active correlated APS dataset."""
        obj = self._obj3d()
        self.figure.clear()
        self._cursor = None
        if obj is None:
            self._obj = None
            self.canvas.draw_idle()
            return

        corr = obj.correlation
        if obj is not self._obj:
            self._sel = 0  # start at the first time for a newly shown dataset
        self._obj = obj
        self._times = corr.index
        n_opt = len(obj.optical.bin_mids)
        n_aero = len(obj.bin_mids)
        # (n_times, n_optical, n_aero) — matrix columns are (optical, aero).
        self._matrix = corr.to_numpy(dtype=float).reshape(len(corr), n_opt, n_aero)
        # Total concentration is always the raw (physical) sum over both axes.
        self._total = np.nansum(self._matrix, axis=(1, 2))
        self._opt_edges = np.asarray(obj.optical.bin_edges, dtype=float)
        self._aero_edges = np.asarray(obj.bin_edges, dtype=float)
        # Displayed matrix: optionally dN/(dlogDp_optical · dlogDp_aerodynamic),
        # so unequally sized bins are comparable across both size axes.
        if self.normalize.isChecked():
            dlog_opt = np.diff(np.log10(self._opt_edges))  # length n_opt
            dlog_aero = np.diff(np.log10(self._aero_edges))  # length n_aero
            self._disp = self._matrix / (
                dlog_opt[None, :, None] * dlog_aero[None, None, :]
            )
        else:
            self._disp = self._matrix
        gmax = np.nanmax(self._disp) if self._disp.size else 1.0
        self._norm = Normalize(vmin=0.0, vmax=gmax if gmax > 0 else 1.0)
        self._sel = min(self._sel, len(self._times) - 1)

        self.ax_top = self.figure.add_axes([0.10, 0.72, 0.78, 0.22])
        self.ax_bot = self.figure.add_axes([0.02, 0.02, 0.86, 0.62], projection="3d")
        # One shared colorbar for the bar heights/colours (global scale).
        sm = ScalarMappable(norm=self._norm, cmap=_CMAP)
        sm.set_array([])
        self.figure.colorbar(
            sm,
            ax=self.ax_bot,
            pad=0.08,
            shrink=0.7,
            label=self._conc_label(obj),
        )

        self._draw_top()
        self._draw_bottom()
        self._sync_toolbar_home()
        self.canvas.draw_idle()

    def _conc_label(self, obj) -> str:
        """Colourbar label reflecting whether the cells are dlogDp-normalized."""
        if self.normalize.isChecked():
            return f"dN / (dlogDp_opt · dlogDp_aero)  ({obj.unit})"
        return f"Concentration ({obj.unit})"

    def _draw_top(self) -> None:
        """Total concentration vs time with the draggable time cursor."""
        ax = self.ax_top
        ax.clear()
        ax.plot(self._times, self._total, "-", lw=1.0, color="#4c72b0")
        ax.set_ylabel(f"Total\n({self._obj.unit})", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        ax.tick_params(labelsize=8)
        self._cursor = ax.axvline(
            self._times[self._sel], color="crimson", lw=1.6, picker=5
        )

    def _draw_bottom(self) -> None:
        """3-D bar plot of the optical×aerodynamic distribution at the cursor."""
        ax = self.ax_bot
        ax.clear()
        mat = self._disp[self._sel]  # (n_optical, n_aero)
        xe = np.log10(self._opt_edges)  # optical on x
        ye = np.log10(self._aero_edges)  # aerodynamic on y
        xs, ys, zs, dxs, dys, dzs, colors = [], [], [], [], [], [], []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isfinite(v) or v <= 0:
                    continue
                xs.append(xe[i])
                ys.append(ye[j])
                zs.append(0.0)
                dxs.append(xe[i + 1] - xe[i])
                dys.append(ye[j + 1] - ye[j])
                dzs.append(v)
                colors.append(v)
        if dzs:
            cmap = ScalarMappable(norm=self._norm, cmap=_CMAP).to_rgba(colors)
            ax.bar3d(xs, ys, zs, dxs, dys, dzs, color=cmap, shade=True)
        self._log_ticks(ax.set_xticks, ax.set_xticklabels, self._opt_edges)
        self._log_ticks(ax.set_yticks, ax.set_yticklabels, self._aero_edges)
        ax.set_xlabel("Optical Ø (nm)", fontsize=8)
        ax.set_ylabel("Aerodynamic Ø (nm)", fontsize=8)
        if self.normalize.isChecked():
            zlabel = "Norm. conc."
        else:
            zlabel = f"Conc. ({self._obj.unit})"
        ax.set_zlabel(zlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(
            f"t = {self._times[self._sel].strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9
        )

    @staticmethod
    def _log_ticks(set_ticks, set_labels, values) -> None:
        """Decade ticks (labelled in nm) on a log₁₀-valued 3-D axis."""
        lo = int(np.floor(np.log10(values.min())))
        hi = int(np.ceil(np.log10(values.max())))
        decades = np.arange(lo, hi + 1)
        set_ticks(decades)
        set_labels([f"{10.0**d:g}" for d in decades])

    # -- time-cursor dragging ---------------------------------------------
    def _on_press(self, event) -> None:
        if event.inaxes is self.ax_top and event.xdata is not None:
            self._dragging = True
            self._set_time(event.xdata)

    def _on_motion(self, event) -> None:
        if self._dragging and event.inaxes is self.ax_top and event.xdata is not None:
            self._set_time(event.xdata)

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _set_time(self, xdata) -> None:
        """Snap the cursor to the nearest time and redraw the bottom panel."""
        if self._times is None or self._cursor is None:
            return
        target = mdates.num2date(xdata).replace(tzinfo=None)
        idx = int(np.argmin(np.abs(self._times.to_numpy() - np.datetime64(target))))
        if idx == self._sel:
            return
        self._sel = idx
        self._cursor.set_xdata([self._times[idx], self._times[idx]])
        self._draw_bottom()
        self.canvas.draw_idle()
