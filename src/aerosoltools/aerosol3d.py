"""Public dual-axis aerosol class (:class:`Aerosol3d`).

The Aerodynamic Particle Sizer (APS) can size the same particle population two
ways at once: by **aerodynamic** diameter (time of flight) and by **optical**
diameter (side-scatter intensity). :class:`Aerosol3d` holds both. It extends
:class:`~aerosoltools.aerosol2d.Aerosol2D`, using the **aerodynamic**
distribution as its primary size axis, so every 2D capability (basis
conversions, Pₓ fractions, PSD fitting, heatmaps, summaries …) works on it
unchanged. The **optical** distribution is carried alongside as a companion
:class:`Aerosol2D` at :attr:`optical`, and — when the file was recorded in
*correlated* mode — the full optical×aerodynamic count matrix is kept so the two
techniques can be compared directly (:meth:`plot_aero_vs_optical`).

When only one of the two size measurements is present (the APS was run in
aerodynamic-only, or optical-only, mode), the object simply behaves as a plain
2D distribution on that axis (``is_correlated`` is then ``False``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .aerosol2d import Aerosol2D


class Aerosol3d(Aerosol2D):
    """Aerodynamic + optical size distributions from a correlated APS record.

    Args:
        dataframe (pandas.DataFrame): The **aerodynamic** distribution
            (Datetime, ``Total_conc`` and one column per aerodynamic size bin),
            exactly as :class:`Aerosol2D` expects.
        optical (Aerosol2D | None): The companion **optical** distribution on
            the same time base, or ``None`` for an aerodynamic-only record.
        correlation (pandas.DataFrame | None): The optical×aerodynamic matrix,
            indexed by time with a two-level column index
            ``(optical_mid, aero_mid)`` holding the per-cell concentration.
            ``None`` unless the file was recorded in correlated mode.
    """

    def __init__(self, dataframe, optical=None, correlation=None):
        super().__init__(dataframe)
        self._optical: Optional[Aerosol2D] = optical
        self._correlation: Optional[pd.DataFrame] = correlation

    # -- properties --------------------------------------------------------
    @property
    def optical(self) -> Optional[Aerosol2D]:
        """The optical-diameter distribution as an :class:`Aerosol2D`, or None."""
        return self._optical

    @property
    def aerodynamic(self) -> Aerosol2D:
        """The aerodynamic-diameter distribution as a standalone Aerosol2D."""
        return self.as_2d("aerodynamic")

    @property
    def is_correlated(self) -> bool:
        """True when both size axes (and their correlation matrix) are present."""
        return self._optical is not None and self._correlation is not None

    @property
    def correlation(self) -> Optional[pd.DataFrame]:
        """The time-indexed optical×aerodynamic matrix (``(optical, aero)`` cols)."""
        return self._correlation

    def axis_view(self, axis: str = "aerodynamic") -> Aerosol2D:
        """Return the live aerodynamic or optical axis with activities synced.

        Unlike :meth:`as_2d` (which returns an independent copy), this returns
        the *live* companion object with the parent's current activities copied
        onto it — so switching axes in the GUI keeps marked tasks, fits and
        views consistent across both. The aerodynamic axis is ``self``.

        Args:
            axis: ``"aerodynamic"`` (default) or ``"optical"``.

        Returns:
            Aerosol2D: ``self`` for the aerodynamic axis, or the (activity-
            synced) optical companion; falls back to ``self`` when there is no
            optical axis.
        """
        if axis.lower().startswith("o") and self._optical is not None:
            opt = self._optical
            opt._activities = list(self._activities)
            opt._activity_periods = dict(self._activity_periods)
            for name in self._activities:
                if name in self._data.columns and len(opt._data) == len(self._data):
                    opt._data[name] = self._data[name].to_numpy()
            return opt
        return self

    def as_2d(self, axis: str = "aerodynamic") -> Aerosol2D:
        """Return one size axis as an independent :class:`Aerosol2D`.

        Use this to work with only the aerodynamic or only the optical sizing
        (the returned object is fully detached — analysing or cropping it does
        not touch the parent).

        Args:
            axis: ``"aerodynamic"`` (default) or ``"optical"``.

        Returns:
            Aerosol2D: A standalone copy of the requested distribution.

        Raises:
            ValueError: If ``axis`` is not recognised, or ``"optical"`` is
                requested on an aerodynamic-only record.
        """
        axis = axis.lower()
        if axis in ("aerodynamic", "aero", "a"):
            twin = Aerosol2D(self._data.copy())
            twin._meta = dict(self._meta)
            twin._activities = list(self._activities)
            twin._activity_periods = dict(self._activity_periods)
            twin._extra_data = self._extra_data.copy()
            return twin
        if axis in ("optical", "optic", "o"):
            if self._optical is None:
                raise ValueError("This record has no optical distribution.")
            return self._optical.copy_self()
        raise ValueError("axis must be 'aerodynamic' or 'optical'.")

    # -- time operations propagate to the optical companion ----------------
    # The correlation matrix is deliberately kept at load resolution; the
    # comparison plot selects the relevant time window from it, so it stays a
    # faithful record even after the distributions are cropped/rebinned.
    def _sync(self, name, args, kwargs):
        """Run a time op on the aero axis, then mirror it on the optical one."""
        inplace = kwargs.get("inplace", True)
        target = self if inplace else self.copy_self()
        getattr(Aerosol2D, name)(target, *args, **{**kwargs, "inplace": True})
        if target._optical is not None:
            getattr(target._optical, name)(*args, **{**kwargs, "inplace": True})
        if name == "timecrop" and target._correlation is not None:
            start, end = (list(args) + [None, None])[:2]
            start = kwargs.get("start", start)
            end = kwargs.get("end", end)
            idx = target._correlation.index
            mask = np.ones(len(idx), dtype=bool)
            if start is not None:
                mask &= idx >= pd.Timestamp(start)
            if end is not None:
                mask &= idx <= pd.Timestamp(end)
            target._correlation = target._correlation.loc[mask]
        return target

    def timecrop(self, *args, **kwargs):
        """Crop both size axes to a time window (see :meth:`Aerosol2D.timecrop`)."""
        return self._sync("timecrop", args, kwargs)

    def timerebin(self, *args, **kwargs):
        """Rebin both size axes in time (see :meth:`Aerosol2D.timerebin`)."""
        return self._sync("timerebin", args, kwargs)

    def timesmooth(self, *args, **kwargs):
        """Smooth both size axes in time (see :meth:`Aerosol2D.timesmooth`)."""
        return self._sync("timesmooth", args, kwargs)

    def timeshift(self, *args, **kwargs):
        """Time-shift both size axes (see :meth:`Aerosol2D.timeshift`)."""
        return self._sync("timeshift", args, kwargs)

    # -- the comparison plot ----------------------------------------------
    def plot_aero_vs_optical(
        self,
        activity: Optional[str] = None,
        window: Optional[tuple] = None,
        ax=None,
        log_color: bool = True,
        cmap: str = "viridis",
    ):
        """Compare the aerodynamic and optical sizing of the same particles.

        Draws the correlated optical×aerodynamic concentration matrix (summed
        over the chosen time window) as a heat map — aerodynamic diameter on the
        x-axis, optical diameter on the y-axis — with the 1:1 line overlaid.
        This is where the APS is unique: it shows, particle-population by
        particle-population, how the optically reported size maps onto the
        aerodynamically reported size.

        Args:
            activity: Optional activity name to restrict the sum to. Overrides
                ``window`` when given.
            window: Optional ``(start, end)`` time window (Timestamp-parseable).
                Defaults to the object's current time span.
            ax: Existing Matplotlib axes; a new figure is made when ``None``.
            log_color: Use a logarithmic colour scale (default True).
            cmap: Matplotlib colormap name.

        Returns:
            tuple: ``(figure, axes)``.

        Raises:
            ValueError: If the record is not correlated (no optical axis).
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if not self.is_correlated:
            raise ValueError(
                "plot_aero_vs_optical needs a correlated APS record (both "
                "aerodynamic and optical sizing)."
            )

        corr = self._correlation
        idx = corr.index
        if activity is not None and activity in self.activities:
            mask = self._data[activity].astype(bool)
            mask = mask.reindex(idx, fill_value=False).to_numpy()
        elif window is not None:
            start, end = window
            mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        else:
            mask = (idx >= self.time.min()) & (idx <= self.time.max())

        matrix = corr.loc[mask].sum(axis=0)  # Series indexed by (optical, aero)
        grid = matrix.unstack()  # rows: optical mid, cols: aero mid
        opt_mids = grid.index.to_numpy(dtype=float)
        aero_mids = grid.columns.to_numpy(dtype=float)
        z = grid.to_numpy(dtype=float)
        z = np.where(z > 0, z, np.nan)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.figure

        norm = LogNorm(vmin=np.nanmin(z), vmax=np.nanmax(z)) if log_color else None
        mesh = ax.pcolormesh(
            aero_mids, opt_mids, z, shading="nearest", cmap=cmap, norm=norm
        )
        lo = min(aero_mids.min(), opt_mids.min())
        hi = max(aero_mids.max(), opt_mids.max())
        ax.plot([lo, hi], [lo, hi], "--", color="0.7", lw=1.2, label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Aerodynamic diameter (nm)")
        ax.set_ylabel("Optical diameter (nm, nominal)")
        ax.set_title("APS: optical vs aerodynamic sizing")
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(mesh, ax=ax, label=f"Concentration ({self.unit})")
        fig.tight_layout()
        return fig, ax
