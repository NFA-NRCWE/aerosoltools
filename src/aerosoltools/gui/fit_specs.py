"""Typed, JSON-round-trippable fit specifications for project persistence.

The GUI stores two kinds of user-created fits on a :class:`~aerosoltools.gui.
project.Dataset` and writes them into the saved project file: lognormal **PSD
fits** (per activity, on the Particle-size-distribution tab) and **decay /
source fits** (per marked window, on the Decay tab). Those used to be free
untyped dicts, cleaned/restored by hand in ``projectio``. The dataclasses here
give each an explicit type that owns its own coercion + serialization
(``to_dict``/``from_dict``), mirroring
:class:`~aerosoltools.gui.summary_cache.SummaryCacheEntry`, so ``projectio``
just round-trips typed objects.

The interactive fitters (``tabs/_psdfit`` and the Decay tab) still work with the
per-mode / override *dicts* — those live inside these specs unchanged — so only
the stored container is typed, not the whole fitting engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def _clean_mode(m) -> dict:
    """Coerce one lognormal-mode mapping to a plain ``{mu, sigma, peak, bound}``."""
    return {
        "mu": float(m["mu"]),
        "sigma": float(m["sigma"]),
        "peak": float(m["peak"]),
        "bound": bool(m.get("bound", False)),
    }


@dataclass
class PsdFitSpec:
    """One stored lognormal PSD fit (the modes for a dataset × activity).

    Attributes:
        modes: Lognormal modes as plain ``{mu, sigma, peak, bound}`` dicts — the
            shape the interactive fitter (``tabs/_psdfit``) reads and writes.
        optimized: Whether the modes came from an optimized fit (vs. hand-placed
            starting guesses), used by the tab to show fit quality.
    """

    modes: List[dict] = field(default_factory=list)
    optimized: bool = False

    def to_dict(self) -> dict:
        """JSON-safe dict for project persistence (modes coerced to floats)."""
        return {
            "modes": [_clean_mode(m) for m in self.modes],
            "optimized": bool(self.optimized),
        }

    @classmethod
    def from_dict(cls, data) -> "PsdFitSpec":
        """Rebuild a spec from :meth:`to_dict` output (tolerant of missing keys)."""
        data = data or {}
        return cls(
            modes=[_clean_mode(m) for m in (data.get("modes") or [])],
            optimized=bool(data.get("optimized", False)),
        )
