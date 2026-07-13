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
    # environmental / misc (each its own dimension; not cross-converted)
    unit_key("°C"): ("temperature", "°C", 1.0),
    unit_key("%"): ("humidity", "%", 1.0),
    unit_key("nm"): ("size", "nm", 1.0),
    unit_key("l/min"): ("flow", "l/min", 1.0),
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
