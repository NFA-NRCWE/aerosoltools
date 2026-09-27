"""Data model for occupational exposure limits (OELs) and their legal source.

An :class:`ExposureLimit` is one row of a limit-value list (a substance with its
8-hour and short-term limits); an :class:`ExposureLimitList` is a whole parsed
list together with its :class:`LimitSource` — the order's number, date and ELI
link — so a comparison can always state exactly which version it used.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd

from .._core.metrics import classify_unit, convert_value

#: Remark (*anmærkning*) codes of Bilag 2, as defined by the order's own legend.
REMARK_CODES: dict[str, str] = {
    "E": "Has an EU limit value (the Danish limit may be stricter).",
    "L": "The short-term value is a ceiling value (loftværdi) that must never "
    "be exceeded.",
    "H": "Can be absorbed through the skin.",
    "K": "Considered potentially carcinogenic (covered by the Danish order on "
    "carcinogenic substances).",
}

#: Name fragment → particle fraction. Per the order, a limit that names no
#: fraction applies to *total* dust.
_FRACTIONS = (
    ("respirab", "respirable"),
    ("inhalerb", "inhalable"),
    ("thorak", "thoracic"),
)

#: A size-fraction qualifier in a name ("…, respirabel fraktion", "…, totalstøv"),
#: removed to find the entries that are fraction variants of one substance.
_FRACTION_QUALIFIER = re.compile(
    r",?\s*\b(?:respirab|inhalerb|thorak|total)\w*(?:\s+(?:fraktion|støv))?\b",
    re.IGNORECASE,
)
#: "beregnet som Ti" — the limit applies to that element's (or group's) mass.
_CALCULATED_AS = re.compile(r"beregnet som\s+([^\s,]+)")
#: A footnote giving a short-term reference period other than 15 minutes.
_REFERENCE_PERIOD = re.compile(r"referenceperiode på (\d+) minut", re.IGNORECASE)

#: Serialisation tag + version of the JSON files written by :meth:`to_json`.
_FORMAT = "aerosoltools.exposure_limits"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ExposureLimit:
    """One substance's occupational exposure limits.

    Attributes:
        name: Substance/material name exactly as the order lists it (Danish),
            e.g. ``"Kvarts, respirabel"``. The limit is tied to this name, not
            to the CAS numbers.
        cas: CAS numbers listed for the substance (may be empty, and the order
            notes the list is not always exhaustive).
        twa: 8-hour time-weighted-average limit, in :attr:`unit`; ``None`` when
            the order gives only a short-term limit.
        stel: Short-term limit (average over :attr:`stel_minutes`, normally 15
            min; a ceiling when :attr:`ceiling`), in :attr:`unit`; ``None``
            when the order gives none.
        unit: Unit of :attr:`twa` / :attr:`stel` (``"mg/m³"`` for dust).
        stel_rule: The order's reference when :attr:`stel` is *derived* rather
            than listed — e.g. ``"Jf. § 3, stk. 2"``, which makes the short-term
            limit twice the 8-hour limit. Empty when the value is listed.
        remarks: The raw remark (*anmærkning*) text, e.g. ``"EK"``; see
            :data:`REMARK_CODES`.
        year: Year the entry was added or last changed (the order's *Årstal*).
        notes: Footnotes and later-dated values attached to the entry.
        section: The Bilag 2 section the entry comes from: ``"B"`` (dust),
            ``"A"`` (the particulate entries of the gases/vapours/particles
            list) or ``"C"`` (process-specific welding-fume limits).
    """

    name: str
    cas: tuple[str, ...] = ()
    twa: float | None = None
    stel: float | None = None
    unit: str = "mg/m³"
    stel_rule: str = ""
    remarks: str = ""
    year: int | None = None
    notes: tuple[str, ...] = ()
    section: str = "B"

    @property
    def stel_derived(self) -> bool:
        """Whether :attr:`stel` was derived from :attr:`twa` (not listed)."""
        return bool(self.stel_rule)

    @property
    def remark_codes(self) -> tuple[str, ...]:
        """The remark letters (``E``/``L``/``H``/``K``) in :attr:`remarks`."""
        m = re.match(r"[ELHK]+", self.remarks.strip())
        return tuple(m.group(0)) if m else ()

    @property
    def ceiling(self) -> bool:
        """Whether the short-term value is a ceiling value (remark ``L``)."""
        return "L" in self.remark_codes

    @property
    def fraction(self) -> str:
        """The particle fraction the limit applies to.

        One of ``"respirable"``, ``"inhalable"``, ``"thoracic"`` or
        ``"total"`` — read from the name; the order specifies total dust
        wherever no fraction is named.
        """
        low = self.name.lower()
        for stem, fraction in _FRACTIONS:
            if stem in low:
                return fraction
        return "total"

    @property
    def base_name(self) -> str:
        """The name without its fraction qualifier, shared by fraction variants.

        ``"Kvarts, total"`` and ``"Kvarts, respirabel"`` both give ``"Kvarts"``.
        """
        base = _FRACTION_QUALIFIER.sub("", self.name)
        return " ".join(base.replace(" ,", ",").split()).strip(" ,")

    @property
    def basis(self) -> str | None:
        """What the limit is expressed as, when not the substance's own mass.

        The element or group of *"beregnet som …"* (e.g. ``"Ti"`` for titanium
        dioxide), or ``"elemental carbon"`` for diesel exhaust. ``None`` when
        the limit applies to the substance itself.
        """
        m = _CALCULATED_AS.search(self.name)
        if m:
            return m.group(1)
        if any("elementært kulstof" in note.lower() for note in self.notes):
            return "elemental carbon"
        return None

    @property
    def stel_minutes(self) -> int:
        """Reference period of the short-term limit, in minutes (15 unless noted)."""
        for note in self.notes:
            m = _REFERENCE_PERIOD.search(note)
            if m:
                return int(m.group(1))
        return 15

    def twa_in(self, unit: str) -> float | None:
        """The 8-hour limit expressed in ``unit`` (``None`` when there is none).

        Args:
            unit: Target unit of the same dimension, e.g. ``"µg/m³"``.

        Raises:
            ValueError: If ``unit`` is not a concentration of the same kind
                (e.g. a number concentration for a mass-based limit).
        """
        return self._convert(self.twa, unit)

    def stel_in(self, unit: str) -> float | None:
        """The short-term limit expressed in ``unit`` (``None`` when none).

        Args:
            unit: Target unit of the same dimension, e.g. ``"µg/m³"``.

        Raises:
            ValueError: If ``unit`` is not a concentration of the same kind.
        """
        return self._convert(self.stel, unit)

    def is_comparable_to(self, unit: str | None) -> bool:
        """Whether a series in ``unit`` can be compared against this limit."""
        if not unit:
            return False
        return classify_unit(unit)[0] == classify_unit(self.unit)[0]

    def _convert(self, value: float | None, unit: str) -> float | None:
        """Convert ``value`` from :attr:`unit` to ``unit`` (same dimension only)."""
        if value is None:
            return None
        if not self.is_comparable_to(unit):
            raise ValueError(
                f"The limit for {self.name!r} is in {self.unit}; it cannot be "
                f"expressed in {unit!r}. Compare it against a "
                f"{classify_unit(self.unit)[0]} concentration instead."
            )
        return float(convert_value(value, self.unit, unit))

    def to_dict(self) -> dict:
        """JSON-safe dict of the fields (tuples become lists)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ExposureLimit:
        """Rebuild a limit from :meth:`to_dict` output."""
        kwargs = dict(data)
        for key in ("cas", "notes"):
            kwargs[key] = tuple(kwargs.get(key) or ())
        return cls(**kwargs)


@dataclass(frozen=True)
class LimitSource:
    """Provenance of an :class:`ExposureLimitList`.

    Attributes:
        title: The order's title.
        number: The order's number (*BEK nr*).
        date: Date the order was issued, ISO ``YYYY-MM-DD``.
        eli: The order's ELI link on Retsinformation.
        status: Retsinformation status when retrieved (``"Valid"`` = in force,
            ``"Historic"`` = replaced).
        in_force_from: Date the order took effect (ISO), when stated.
        accession_number: Retsinformation's accession number.
        retrieved: When the order was downloaded (ISO timestamp, UTC).
        sections: Which Bilag 2 sections were parsed.
    """

    title: str
    number: int
    date: str
    eli: str
    status: str = ""
    in_force_from: str | None = None
    accession_number: str = ""
    retrieved: str = ""
    sections: tuple[str, ...] = ("B",)

    @property
    def label(self) -> str:
        """Citation in Retsinformation's style, e.g. ``BEK nr 613 af 29/06/2026``."""
        try:
            issued = date.fromisoformat(self.date).strftime("%d/%m/%Y")
        except ValueError:
            issued = self.date
        return f"BEK nr {self.number} af {issued}"

    @property
    def short_label(self) -> str:
        """Compact citation for plot legends, e.g. ``BEK 613/2026``."""
        return f"BEK {self.number}/{self.date[:4]}"

    def to_dict(self) -> dict:
        """JSON-safe dict of the fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LimitSource:
        """Rebuild a source from :meth:`to_dict` output."""
        kwargs = dict(data)
        kwargs["sections"] = tuple(kwargs.get("sections") or ("B",))
        return cls(**kwargs)


@dataclass(frozen=True)
class ExposureLimitList:
    """A parsed limit-value list plus the legal source it came from.

    Iterate it for the :class:`ExposureLimit` rows, look one up by name with
    :meth:`get` or ``limits["Kvarts, respirabel"]``, or view it as a table with
    :meth:`to_dataframe`.

    Attributes:
        source: Which order (number, date, ELI) the limits were parsed from.
        limits: The substances, sorted by name.
        excluded: ``(name, reason)`` for entries with a mass limit that were
            deliberately left out — fibres (counted per cm³), mercury (also a
            vapour), volatile metal compounds, and Afsnit A entries not
            classified as particulate. Gases and vapours with a ppm limit are
            not listed.
    """

    source: LimitSource
    limits: tuple[ExposureLimit, ...] = ()
    excluded: tuple[tuple[str, str], ...] = field(default=())

    def __iter__(self) -> Iterator[ExposureLimit]:
        return iter(self.limits)

    def __len__(self) -> int:
        return len(self.limits)

    def names(self) -> list[str]:
        """Substance names, in list order."""
        return [lim.name for lim in self.limits]

    def get(self, name: str) -> ExposureLimit | None:
        """Look a substance up by name (exact, then case-insensitive)."""
        for lim in self.limits:
            if lim.name == name:
                return lim
        folded = name.casefold()
        for lim in self.limits:
            if lim.name.casefold() == folded:
                return lim
        return None

    def __getitem__(self, name: str) -> ExposureLimit:
        lim = self.get(name)
        if lim is None:
            close = difflib.get_close_matches(name, self.names(), n=3, cutoff=0.5)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise KeyError(f"No exposure limit named {name!r}.{hint}")
        return lim

    def siblings(self, limit: ExposureLimit) -> list[ExposureLimit]:
        """The fraction variants of ``limit``'s substance (``limit`` included).

        E.g. ``"Kvarts, total"`` and ``"Kvarts, respirabel"``.
        """
        base = limit.base_name.casefold()
        found = [lim for lim in self.limits if lim.base_name.casefold() == base]
        return found or [limit]

    def to_dataframe(self) -> pd.DataFrame:
        """The list as a table (one row per substance)."""
        rows = [
            {
                "Substance": lim.name,
                "CAS": ", ".join(lim.cas),
                "8-h limit": lim.twa,
                "Short-term limit": lim.stel,
                "Unit": lim.unit,
                "Short-term rule": lim.stel_rule,
                "Remarks": lim.remarks,
                "Fraction": lim.fraction,
                "Basis": lim.basis or "",
                "Section": lim.section,
                "Year": lim.year,
                "Notes": " ".join(lim.notes),
            }
            for lim in self.limits
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        """JSON-safe representation (see :meth:`to_json`)."""
        return {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "source": self.source.to_dict(),
            "excluded": [list(pair) for pair in self.excluded],
            "limits": [lim.to_dict() for lim in self.limits],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExposureLimitList:
        """Rebuild a list from :meth:`to_dict` output.

        Raises:
            ValueError: If ``data`` is not an exposure-limit list, or was
                written by a newer, incompatible version.
        """
        if data.get("format") != _FORMAT:
            raise ValueError("Not an aerosoltools exposure-limit list.")
        if int(data.get("format_version", 0)) > _FORMAT_VERSION:
            raise ValueError(
                "This exposure-limit list was written by a newer aerosoltools; "
                "update aerosoltools to read it."
            )
        # Early lists stored excluded entries as bare names (all fibres).
        excluded = tuple(
            (item, "fibre") if isinstance(item, str) else (str(item[0]), str(item[1]))
            for item in data.get("excluded") or ()
        )
        return cls(
            source=LimitSource.from_dict(data["source"]),
            limits=tuple(ExposureLimit.from_dict(d) for d in data.get("limits", [])),
            excluded=excluded,
        )

    def to_json(self, path: str | Path) -> None:
        """Write the list (with its source) to a UTF-8 JSON file."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=1)
        Path(path).write_text(text + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> ExposureLimitList:
        """Read a list written by :meth:`to_json`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
