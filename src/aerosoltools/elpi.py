"""Public class for Dekati ELPI impactors (:class:`ELPI`).

The ELPI reports a *geometric* particle diameter that depends on the assumed
particle density (it classifies by aerodynamic behaviour). Changing the density
must therefore recompute the diameters — and re-attribute the per-bin number —
rather than only rescaling mass. :class:`ELPI` is otherwise an ordinary
:class:`Aerosol2D`; it only overrides the density-recompute hook so that this
instrument-specific physics lives on the ELPI class instead of leaking into the
shared size-distribution code.
"""

from __future__ import annotations

from .aerosol2d import Aerosol2D


class ELPI(Aerosol2D):
    """Electrical Low-Pressure Impactor size distribution.

    Behaves exactly like :class:`Aerosol2D` except that :meth:`set_density`
    rebuilds the density-dependent size axis (via the loader's
    ``recalculate_ELPI_density``) instead of only rescaling mass.
    """

    def _recompute_diameters_for_density(self, density: float, old: float) -> bool:
        # Lazy import avoids a circular import (the loader imports this class).
        from .loaders.elpi import recalculate_ELPI_density

        return bool(recalculate_ELPI_density(self, density))
