"""Does an exposure limit apply to a measured metric? The size-fraction check.

Limits are set for a health-related size fraction (EN 481 / ISO 7708): the
*respirable* fraction (50 % cut ≈ 4 µm), the *thoracic* (≈ 10 µm), the
*inhalable* (≈ 100 µm) — or *total* dust where the order names no fraction. A
direct-reading metric covers particles up to its own cut: PM2.5 up to 2.5 µm, a
size spectrometer's total mass up to its largest size bin. A comparison is only
fair when both cover the same particles:

* ``"match"`` — the metric's cut suits the limit's fraction: the limit applies.
* ``"conservative"`` — the metric also counts larger particles than the fraction
  (PM10 against a respirable limit): exposure is overestimated.
* ``"underestimates"`` — the metric misses part of the fraction (PM1 against
  total dust): exposure is underestimated.
* ``"unknown"`` — the metric's size range is not known.

Total and inhalable dust include particles no direct-reading instrument samples
fully, so a metric reaching at least 10 µm is accepted for them, with that
caveat. A limit *measured as elemental carbon* (diesel exhaust) applies to a
black-carbon metric instead of a size cut.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from ._model import ExposureLimit

#: Nominal 50 % cut (µm) of each fraction a limit can apply to.
FRACTION_CUT_UM: dict[str, float] = {
    "respirable": 4.0,
    "thoracic": 10.0,
    "inhalable": 100.0,
    "total": math.inf,
}
#: Relative band around a respirable/thoracic cut that still counts as a match
#: (respirable: PM3–PM5, thoracic: PM7.5–PM12.5).
_MATCH_BAND = (0.75, 1.25)
#: Smallest cut (µm) accepted as a stand-in for total or inhalable dust.
_COARSE_CUT_UM = 10.0

_PM = re.compile(r"\bPM\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
#: An instrument's whole measured range ("Total", "TSP").
_WHOLE_RANGE = re.compile(r"^\s*(?:total|tsp)\b", re.IGNORECASE)
#: A mass derived from a whole size distribution ("MASS", "Mass concentration").
_DERIVED_MASS = re.compile(r"^\s*mass(?:\s+concentration)?\s*$", re.IGNORECASE)
_BLACK_CARBON = re.compile(r"\bBC|black carbon|elemental carbon|\bEC\b", re.IGNORECASE)


@dataclass(frozen=True)
class FractionCheck:
    """Outcome of :func:`check_fraction`.

    Attributes:
        status: ``"match"``, ``"conservative"``, ``"underestimates"`` or
            ``"unknown"`` (see the module docstring).
        message: Plain-language explanation — a warning when the limit does not
            apply, a caveat (or empty) when it does.
    """

    status: str
    message: str = ""

    @property
    def applies(self) -> bool:
        """Whether the limit applies to the metric."""
        return self.status == "match"


def metric_size_cut(metric: str, upper_um: float | None = None) -> float | None:
    """Largest particle size (µm) a metric covers.

    Args:
        metric: The metric's name, e.g. ``"PM2.5"``, ``"Total"``, ``"MASS"``.
        upper_um: Upper edge of the instrument's size range in µm, when known
            (a size spectrometer's largest bin). A size spectrometer reports
            ``PM10`` even when it only sizes up to 0.4 µm, so a PM cut is
            capped at this.

    Returns:
        The cut in µm (``math.inf`` for a "Total" channel of unknown range),
        or ``None`` when it cannot be told from the name.
    """
    name = metric or ""
    m = _PM.search(name)
    if m:
        cut = float(m.group(1).replace(",", "."))
        return min(cut, upper_um) if upper_um else cut
    if _WHOLE_RANGE.match(name):
        return upper_um if upper_um else math.inf
    if _DERIVED_MASS.match(name):
        return upper_um
    return None


def check_fraction(
    limit: ExposureLimit, metric: str, size_cut_um: float | None
) -> FractionCheck:
    """Check whether ``limit`` applies to a metric covering particles up to a cut.

    Args:
        limit: The exposure limit (its :attr:`~ExposureLimit.fraction` and
            :attr:`~ExposureLimit.basis` decide).
        metric: The metric's name, used in messages and to recognise black
            carbon.
        size_cut_um: The metric's cut from :func:`metric_size_cut`.
    """
    name = metric or "the metric"
    if limit.basis == "elemental carbon":
        if _BLACK_CARBON.search(metric or ""):
            return FractionCheck(
                "match",
                f"{name} (black carbon) is used as a measure of elemental carbon.",
            )
        return FractionCheck(
            "conservative",
            f"The limit is for elemental carbon, but {name} counts all particle "
            "mass — exposure is overestimated.",
        )
    fraction = limit.fraction
    if size_cut_um is None:
        return FractionCheck(
            "unknown",
            f"The size range of {name} is not known — check that it matches the "
            f"{fraction} fraction the limit applies to.",
        )
    reach = "all sizes" if math.isinf(size_cut_um) else f"≤ {size_cut_um:g} µm"
    if fraction in ("total", "inhalable"):
        if size_cut_um >= _COARSE_CUT_UM:
            return FractionCheck(
                "match",
                f"Direct-reading instruments under-sample the coarse particles of "
                f"{fraction} dust, so {name} may underestimate it.",
            )
        return FractionCheck(
            "underestimates",
            f"{name} ({reach}) misses the coarse particles of {fraction} dust — "
            "exposure is underestimated.",
        )
    cut = FRACTION_CUT_UM[fraction]
    low, high = cut * _MATCH_BAND[0], cut * _MATCH_BAND[1]
    if size_cut_um < low:
        return FractionCheck(
            "underestimates",
            f"{name} ({reach}) misses part of the {fraction} fraction (≈ PM{cut:g}) "
            "— exposure is underestimated.",
        )
    if size_cut_um > high:
        return FractionCheck(
            "conservative",
            f"{name} ({reach}) also counts particles larger than the {fraction} "
            f"fraction (≈ PM{cut:g}) — exposure is overestimated.",
        )
    return FractionCheck("match")


def basis_note(limit: ExposureLimit) -> str:
    """Caveat for a limit expressed as an element's mass (``""`` if none).

    E.g. titanium dioxide's limit is *beregnet som Ti*: a particle mass
    concentration includes the oxygen too, so comparing it is conservative.
    """
    basis = limit.basis
    if not basis or basis == "elemental carbon":
        return ""
    return (
        f"The limit is for the {basis} content (beregnet som {basis}); a particle "
        "mass concentration includes the rest of the compound, so comparing it "
        "is conservative."
    )


def applicable_limit(
    candidates: Sequence[ExposureLimit], metric: str, size_cut_um: float | None
) -> tuple[ExposureLimit, FractionCheck]:
    """Choose which fraction variant of a substance applies to a metric.

    Args:
        candidates: The picked limit first, then its fraction variants (see
            :meth:`ExposureLimitList.siblings`), e.g. ``"Kvarts, respirabel"``
            then ``"Kvarts, total"``.
        metric: The metric's name.
        size_cut_um: The metric's cut from :func:`metric_size_cut`.

    Returns:
        The first candidate that applies, with its check — or, if none does,
        the first candidate with the reason it does not.
    """
    first = candidates[0]
    first_check = check_fraction(first, metric, size_cut_um)
    if first_check.applies:
        return first, first_check
    for other in candidates[1:]:
        check = check_fraction(other, metric, size_cut_um)
        if check.applies:
            return other, check
    return first, first_check
