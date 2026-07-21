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
from typing import List, Optional, Tuple

import pandas as pd

# The manual-override fields a decay fit can carry (all optional): the two are
# timestamps, the rest floats. Kept as a plain dict on the live spec because the
# Decay tab mutates them in place; DecayFitSpec only owns their (de)serialization.
_DECAY_TS_OVERRIDES = ("t0", "peak_time")
_DECAY_NUM_OVERRIDES = ("background", "peakval", "rate")


def _ts_or_none(value):
    """ISO string for a timestamp-like value, or None when missing/unparseable."""
    if value is None:
        return None
    try:
        return pd.Timestamp(value).isoformat()
    except (ValueError, TypeError):
        return None


def _parse_ts(value):
    """pd.Timestamp for a value, or None when empty/unparseable."""
    if value is None:
        return None
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError):
        return None


def _num_or_none(value):
    """float(value) or None when missing/unparseable."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


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


@dataclass
class DecayFitSpec:
    """One stored decay / source fit (a marked window and its fit settings).

    This is the persistence type for the Decay tab's fits. The tab keeps each
    fit as a *live dict* it mutates in place (and hangs a transient ``_result``
    on), so this spec is a data-transfer object: :meth:`from_live` reads a live
    dict, :meth:`to_dict`/:meth:`from_dict` (de)serialize JSON, and
    :meth:`to_live` rebuilds the live dict. The recomputed fit result itself is
    never stored — it is recomputed on load so it tracks density / calibration.

    Attributes:
        window: ``(start, end)`` timestamps of the marked region.
        metric: Metric the fit runs on (e.g. ``"PNC"``).
        model: Kinetic model name (``"auto"`` or an explicit model).
        optimized: Whether the loss kinetics were optimized (vs. hand-set).
        per_bin: Whether the fit was also run per size bin.
        overrides: Manual guesses ``{t0, peak_time, background, peakval, rate}``
            (timestamps for the first two, floats for the rest; any may be None).
    """

    window: Tuple[pd.Timestamp, pd.Timestamp]
    metric: str = "PNC"
    model: str = "auto"
    optimized: bool = True
    per_bin: bool = False
    overrides: dict = field(default_factory=dict)

    @staticmethod
    def _clean_overrides(ov, parse) -> dict:
        """Coerce an override mapping, applying ``parse`` to the timestamp fields."""
        ov = ov or {}
        cleaned = {key: parse(ov.get(key)) for key in _DECAY_TS_OVERRIDES}
        cleaned.update({key: _num_or_none(ov.get(key)) for key in _DECAY_NUM_OVERRIDES})
        return cleaned

    @classmethod
    def from_live(cls, spec) -> Optional["DecayFitSpec"]:
        """Build a spec from the Decay tab's live dict, or None if it has no window."""
        window = (spec or {}).get("window")
        if not window or len(window) != 2 or None in window:
            return None
        return cls(
            window=(pd.Timestamp(window[0]), pd.Timestamp(window[1])),
            metric=str(spec.get("metric", "PNC")),
            model=str(spec.get("model", "auto")),
            optimized=bool(spec.get("optimized", True)),
            per_bin=bool(spec.get("per_bin", False)),
            overrides=cls._clean_overrides(spec.get("overrides"), _parse_ts),
        )

    def to_dict(self) -> dict:
        """JSON-safe dict for project persistence (timestamps as ISO strings)."""
        return {
            "window": [_ts_or_none(self.window[0]), _ts_or_none(self.window[1])],
            "metric": self.metric,
            "model": self.model,
            "optimized": bool(self.optimized),
            "per_bin": bool(self.per_bin),
            "overrides": self._clean_overrides(self.overrides, _ts_or_none),
        }

    @classmethod
    def from_dict(cls, data) -> Optional["DecayFitSpec"]:
        """Rebuild a spec from :meth:`to_dict` output, or None if the window is gone."""
        data = data or {}
        window = data.get("window")
        if not window or len(window) != 2 or None in window:
            return None
        return cls(
            window=(pd.Timestamp(window[0]), pd.Timestamp(window[1])),
            metric=str(data.get("metric", "PNC")),
            model=str(data.get("model", "auto")),
            optimized=bool(data.get("optimized", True)),
            per_bin=bool(data.get("per_bin", False)),
            overrides=cls._clean_overrides(data.get("overrides"), _parse_ts),
        )

    def to_live(self) -> dict:
        """Rebuild the live in-memory dict the Decay tab mutates."""
        return {
            "window": (self.window[0], self.window[1]),
            "metric": self.metric,
            "model": self.model,
            "optimized": bool(self.optimized),
            "per_bin": bool(self.per_bin),
            "overrides": self._clean_overrides(self.overrides, _parse_ts),
        }
