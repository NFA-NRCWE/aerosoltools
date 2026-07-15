"""Shared unit resolution for the loaders' *extra* (housekeeping) columns.

Instrument exports carry many non-core columns — temperatures, humidities,
flows, pressures, diagnostics — whose units, when present at all, are embedded
in the header in a jumble of conventions (``Sample temp (C)``, ``Pressure
[mBar]``, ``Temperature.C``). This helper gives every loader one place to turn
that raw frame into a clean ``extra_data`` frame plus a ``{column: unit}`` map:

* a recognised trailing unit is stripped from the header (so the column is named
  ``"Sample temp"`` and its unit ``"°C"`` is recorded separately), and
* the unit is canonicalised via the shared registry (``(C)`` → ``°C``,
  ``(ug/m3)`` → ``µg/m³``), so the same quantity from two instruments compares
  equal.

Columns with no recognisable unit keep their header verbatim and simply get no
entry in the map — callers then show the header text alone rather than borrow
the primary series' unit. Only genuine units are ever attached (see
:func:`aerosoltools._core.metrics.parse_header_unit`), so an index ``[1] Conc``,
a coordinate format ``(ddmm.mmmmm)`` or a note ``(with RI)`` is never mistaken
for one.
"""

from __future__ import annotations

import pandas as pd

from ..._core.metrics import canonical_unit, parse_header_unit


def resolve_extra_columns(
    df: pd.DataFrame, curated: dict[str, str] | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Clean an ``extra_data`` frame's headers and resolve their units.

    Args:
        df: The non-core columns to keep as ``extra_data`` (index already set).
        curated: Optional ``{header: unit}`` overrides for columns whose unit is
            known from the instrument spec but not written in the header (e.g. a
            DiSCmini diffusion current). Curated units are canonicalised too and
            win over header parsing; the column keeps its original name.

    Returns:
        ``(clean_df, column_units)`` where ``clean_df`` is ``df`` with recognised
        trailing units stripped from the column names, and ``column_units`` maps
        each resolved column (by its final name) to a canonical unit string.
    """
    curated = curated or {}
    rename: dict[str, str] = {}
    units: dict[str, str] = {}
    taken = {str(c) for c in df.columns}
    for col in df.columns:
        raw = str(col)
        if raw in curated:
            units[raw] = canonical_unit(curated[raw])
            continue
        name, unit = parse_header_unit(raw)
        if unit is None:
            continue
        # Strip the unit from the header when the cleaned name is usable and does
        # not clash with another column; otherwise keep the raw header as-is.
        if name and name != raw and name not in taken:
            rename[raw] = name
            taken.add(name)
            units[name] = unit
        else:
            units[raw] = unit
    clean = df.rename(columns=rename) if rename else df
    return clean, units
