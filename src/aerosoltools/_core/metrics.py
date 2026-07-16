"""Metric catalog + unit registry shared by summaries (core, not GUI-specific).

Every data class exposes :meth:`available_metrics` — the list of quantities it
can actually summarise, each described by a :class:`MetricSpec`. This gives both
the GUI metric picker and scripting users of ``summarize_activities`` /
``summarize_exposure`` a clear, instrument-aware answer to "what can I ask for?",
instead of a metric token being blindly applied to every dataset.

The unit registry classifies a unit string into a physical ``dimension`` and a
canonical unit, and converts values between different scales of the *same*
dimension (e.g. ``ng/m³`` ↔ ``µg/m³``). Summaries use this to merge the *same*
metric reported by different instruments at different unit scales into a single
column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Superscript digits/sign → plain characters, so ``cm⁻³`` compares like ``cm-3``.
_SUP = str.maketrans(
    {
        "⁻": "-",
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
)


def unit_key(unit: str) -> str:
    """Normalise a unit string for comparison (case/superscript/LaTeX/space-free).

    So ``"cm⁻³"``, ``"cm^-3"``, ``"cm$^{-3}$"`` all reduce to one key, and the
    two LDSA spellings ``"nm²/cm³"`` and ``"nm$^{2}$/cm$^{3}$"`` collapse to the
    same key. (Mirrors ``gui/tabs/overlay._unit_key``.)
    """
    key = (unit or "").strip().lower().translate(_SUP)
    for ch in "$^{}\\ ":
        key = key.replace(ch, "")
    return key


#: Normalised unit key → (dimension, canonical unit, scale-to-canonical).
#: ``value_in_canonical = value * scale``. Only units of the *same* dimension are
#: ever converted into one another.
_UNIT_TABLE: dict[str, tuple[str, str, float]] = {
    # number concentration
    **{
        unit_key(u): ("number", "cm⁻³", 1.0)
        for u in ("cm⁻³", "cm-3", "#/cm³", "1/cm³", "p/cm³", "n/cm³", "particles/cm³")
    },
    # mass concentration (canonical µg/m³)
    unit_key("µg/m³"): ("mass", "µg/m³", 1.0),
    unit_key("ug/m³"): ("mass", "µg/m³", 1.0),
    unit_key("mg/m³"): ("mass", "µg/m³", 1.0e3),
    unit_key("ng/m³"): ("mass", "µg/m³", 1.0e-3),
    # surface-area / volume concentration
    unit_key("nm²/cm³"): ("surface", "nm²/cm³", 1.0),
    unit_key("nm³/cm³"): ("volume", "nm³/cm³", 1.0),
    # gas mixing ratio (canonical ppm)
    unit_key("ppm"): ("mixing_ratio", "ppm", 1.0),
    unit_key("ppb"): ("mixing_ratio", "ppm", 1.0e-3),
    unit_key("ppt"): ("mixing_ratio", "ppm", 1.0e-6),
    # -- environmental / housekeeping --------------------------------------
    # Each is its own dimension (never cross-converted with a different one),
    # but the many raw header spellings a file may use (``C``/``degC``/``°C``,
    # ``ug/m3``, ``mL/min`` …) collapse to one canonical glyph so the same
    # quantity from two instruments compares equal. See parse_header_unit.
    # temperature (canonical °C)
    **{unit_key(u): ("temperature", "°C", 1.0) for u in ("°C", "C", "degC", "celsius")},
    # relative humidity (canonical %)
    **{unit_key(u): ("humidity", "%", 1.0) for u in ("%", "%RH", "RH%")},
    # pressure (canonical hPa == mbar)
    unit_key("hPa"): ("pressure", "hPa", 1.0),
    unit_key("mbar"): ("pressure", "hPa", 1.0),
    unit_key("Pa"): ("pressure", "hPa", 1.0e-2),
    unit_key("kPa"): ("pressure", "hPa", 10.0),
    unit_key("bar"): ("pressure", "hPa", 1.0e3),
    unit_key("atm"): ("pressure", "hPa", 1013.25),
    # volumetric flow (canonical l/min)
    **{unit_key(u): ("flow", "l/min", 1.0) for u in ("l/min", "L/min", "lpm")},
    unit_key("mL/min"): ("flow", "l/min", 1.0e-3),
    # particle size / diameter (canonical nm)
    unit_key("nm"): ("size", "nm", 1.0),
    **{unit_key(u): ("size", "nm", 1.0e3) for u in ("µm", "um")},
    # elapsed time / duration (canonical s)
    unit_key("s"): ("time", "s", 1.0),
    unit_key("sec"): ("time", "s", 1.0),
    unit_key("ms"): ("time", "s", 1.0e-3),
    unit_key("min"): ("time", "s", 60.0),
    **{unit_key(u): ("time", "s", 3600.0) for u in ("h", "hr")},
    # speed (canonical km/h) — e.g. a GPS ground speed channel
    unit_key("km/h"): ("speed", "km/h", 1.0),
    unit_key("m/s"): ("speed", "km/h", 3.6),
}


def classify_unit(unit: str) -> tuple[str, str, float]:
    """Return ``(dimension, canonical_unit, scale_to_canonical)`` for ``unit``.

    Unknown units fall back to a per-unit ``"other:"`` dimension (so they never
    merge with anything else), their own string as the canonical unit, and a
    scale of ``1.0``.
    """
    key = unit_key(unit)
    if key in _UNIT_TABLE:
        return _UNIT_TABLE[key]
    return (f"other:{key}", (unit or "").strip(), 1.0)


def canonical_unit(unit: str) -> str:
    """The canonical unit string for ``unit``'s dimension (e.g. ng/m³ → µg/m³)."""
    return classify_unit(unit)[1]


#: A trailing unit in round/square brackets: ``"Sample temp (C)"`` → ``"C"``,
#: ``"Pressure [mBar]"`` → ``"mBar"``. The bracket group must sit at the very end
#: so a leading ``"[1] Conc"`` (an index) is not mistaken for a unit.
_BRACKET_UNIT_RE = re.compile(
    r"^(?P<name>.*?)\s*[\(\[]\s*(?P<unit>[^)\]]+?)\s*[\)\]]\s*$"
)
#: A dot-suffixed unit: ``"Temperature.C"`` → ``"C"``. The suffix must be letters
#: (or ``%``), so ``"PM2.5"`` / ``"Bin0.35"`` (numeric) never match.
_DOT_UNIT_RE = re.compile(r"^(?P<name>.+)\.(?P<unit>[A-Za-z%µ°]+)$")


def parse_header_unit(header: str) -> tuple[str, str | None]:
    """Split a column header into ``(name, canonical_unit)`` when it embeds one.

    A unit is extracted **only** when the bracketed/dot-suffixed token is a unit
    the registry actually recognises (:func:`classify_unit`), so non-units are
    left as part of the name rather than fabricated into a unit — e.g. an index
    ``"[1] Conc"``, a coordinate format ``"GPS lat (ddmm.mmmmm)"``, a note
    ``"UB with RI"``, or a distribution weighting ``"Concentration (dW)"`` all
    return ``(header, None)``. A recognised unit is returned canonicalised
    (``"(C)"`` → ``"°C"``, ``"(ug/m3)"`` → ``"µg/m³"``, ``"[mBar]"`` → ``"hPa"``).

    Args:
        header: A raw column header from an instrument export.

    Returns:
        ``(name, unit)`` where ``unit`` is the canonical unit string, or
        ``(header, None)`` when no recognised unit is embedded.
    """
    if not header:
        return header, None
    for rx in (_BRACKET_UNIT_RE, _DOT_UNIT_RE):
        m = rx.match(header)
        if m is None:
            continue
        dimension, canonical, _scale = classify_unit(m.group("unit"))
        if not dimension.startswith("other:"):
            name = m.group("name").strip()
            return (name or header), canonical
    return header, None


#: Distribution basis → canonical user-facing quantity name. Replaces the
#: confusing ``"Total concentration (dX)"`` notation with a plain physical name,
#: so a *measured* mass (e.g. a DustTrak) reads "Mass concentration" just like a
#: *derived* one — comparable, one name. The ``dX`` basis stays internal.
BASIS_QUANTITY: dict[str, str] = {
    "dN": "Number concentration",
    "dM": "Mass concentration",
    "dS": "Surface area concentration",
    "dV": "Volume concentration",
}
#: Reverse lookup: canonical quantity name → basis.
QUANTITY_BASIS: dict[str, str] = {v: k for k, v in BASIS_QUANTITY.items()}


#: Canonical cross-instrument environmental metrics: a display name → the set of
#: instrument column names (lower-cased) that mean the *same* physical quantity.
#: Only genuinely-shared **ambient** channels are listed, so different
#: instruments' temperature/RH/pressure columns collapse into one pickable
#: metric. Internal/sensor channels are deliberately absent (e.g. an
#: aethalometer's ``Internal temp``, an ELPI's ``Temperature1``) so they are
#: never merged with the ambient quantity.
_CANONICAL_METRICS: dict[str, set[str]] = {
    "Ambient temperature": {
        "temperature",
        "temp",
        "sample temp",
        "air temperature",
        "t",
    },
    "Ambient relative humidity": {
        "rh",
        "humidity",
        "rel. humidity",
        "relative humidity",
        "sample rh",
    },
    "Ambient pressure": {
        "pressure",
        "ambient pressure",
        "sample pressure",
        "barometric pressure",
    },
}


def canonical_metric_for(column: str) -> str | None:
    """Return the canonical environmental metric ``column`` belongs to, or None.

    Matches a (cleaned) instrument column name against :data:`_CANONICAL_METRICS`
    so the same ambient quantity from different instruments (``"Sample temp"``,
    ``"Temp"``, ``"T"`` → ``"Ambient temperature"``) can be combined. Returns
    ``None`` for anything not on the curated list — including internal/sensor
    channels — so unrecognised columns stay instrument-local.
    """
    key = (column or "").strip().lower()
    for canonical, aliases in _CANONICAL_METRICS.items():
        if key in aliases:
            return canonical
    return None


def convert_value(x, from_unit: str, to_unit: str):
    """Convert a value/array from ``from_unit`` to ``to_unit`` (same dimension).

    Returns ``x`` unchanged if the two units are not in the same known dimension
    (there is no meaningful linear conversion), so callers never silently
    fabricate cross-dimension numbers.
    """
    fdim, _fcanon, fscale = classify_unit(from_unit)
    tdim, _tcanon, tscale = classify_unit(to_unit)
    if fdim != tdim or tscale == 0:
        return x
    return x * (fscale / tscale)


@dataclass(frozen=True)
class MetricSpec:
    """A quantity a dataset can be summarised on.

    Attributes:
        key: Token accepted by ``summarize_activities`` / ``summarize_exposure``
            (e.g. ``"PNC"``, ``"LDSA"``, ``"PM2.5"``, ``"IR BCc"``, ``"Cl₂"``).
        label: Human-readable name for display/columns.
        unit: The metric's unit (as the dataset reports it).
        dimension: Physical dimension used to group/merge comparable metrics
            (from :func:`classify_unit`).
        default: Whether this metric is part of the instrument's primary set
            (used as the summary default and pre-checked in the GUI picker).
    """

    key: str
    label: str
    unit: str
    dimension: str
    default: bool = False

    @property
    def canonical_unit(self) -> str:
        """The canonical unit for this metric's dimension."""
        return canonical_unit(self.unit)
