"""Small shared helpers for plot/axis labels.

Centralises the one-line conventions that were previously inlined at several
call sites, so the whole codebase derives axis labels the same way.
"""

from __future__ import annotations

#: Units of the size-fraction metrics integrated from a size distribution
#: (``PM2.5`` → mass, ``PN10`` → number, …), keyed by the letter after ``P``.
PX_UNITS = {"M": "µg/m³", "N": "cm⁻³", "S": "nm²/cm³", "V": "nm³/cm³"}

_DISTRIBUTION_TYPES = {"dN", "dM", "dS", "dV"}


def base_dtype(dtype: str) -> str:
    """Return the base distribution type, dropping any normalisation suffix.

    For example ``"dN/dlogDp" -> "dN"`` and ``"dM" -> "dM"``. Used wherever a
    y-axis label or a dtype selector needs the underlying quantity rather than
    its log-diameter-normalised form.
    """
    if not dtype:
        return dtype
    return dtype.split("/")[0]


def _primary_name(obj):
    """Column name of the object's primary channel, or ``None``."""
    try:
        return getattr(obj._primary, "name", None)
    except (AttributeError, IndexError, KeyError):
        return None


def channel_unit(obj, column=None) -> str | None:
    """Return ``column``'s own unit, or ``None`` when it has none on record.

    Unlike :meth:`~aerosoltools.Aerosol1D.unit_of`, which falls back to the
    primary series' unit for any column it does not know, a unit is only
    returned when it genuinely belongs to ``column``:

    * the loader-resolved per-column unit (``metadata["column_units"]``);
    * a per-column ``unit`` dict entry (e.g. a Partector's ``Flow`` → l/min);
    * the object's single scalar unit, for its primary channel — and, when its
      dtype is a distribution type (dN/dM/dS/dV, so the unit covers every
      measured column), for the other columns of :attr:`data` too (size bins,
      an ACSM's species), never for the housekeeping :attr:`extra_data`;
    * the unit of a size-fraction metric derived from a size distribution
      (``PM2.5`` → µg/m³, ``PN10`` → cm⁻³, ``MASS`` → µg/m³).

    A housekeeping column without unit metadata returns ``None``; its header
    often carries the unit already (``"Temperature (C)"``).

    Args:
        obj: The aerosol object.
        column: Column/channel name; ``None`` for the primary channel.

    Returns:
        The unit string when known, else ``None``.
    """
    primary = _primary_name(obj)
    if column is None:
        column = primary
    meta = getattr(obj, "_meta", None) or {}
    column_units = meta.get("column_units") or {}
    if column in column_units:
        return str(column_units[column])
    unit = meta.get("unit")
    if isinstance(unit, dict):
        if column in unit:
            return str(unit[column])
    elif unit and column is not None:
        if column == primary:
            return str(unit)
        data = getattr(obj, "data", None)
        masks = set(getattr(obj, "activities", None) or [])
        dtype = meta.get("dtype")
        if (
            isinstance(dtype, str)
            and base_dtype(dtype) in _DISTRIBUTION_TYPES
            and data is not None
            and column in data.columns
            and column not in masks
        ):
            return str(unit)
    # Size-fraction metrics of a size-resolved object (PM_calc & co.).
    parse = getattr(obj, "_parse_px_metric_scalar", None)
    if callable(parse) and isinstance(column, str):
        name = column.upper()
        if name == "MASS":
            return PX_UNITS["M"]
        try:
            parsed = parse(name)
        except ValueError:
            parsed = None
        if parsed:
            return PX_UNITS[parsed[0]]
    return None


def channel_label(obj, column) -> str:
    """Axis label for one channel of ``obj`` (GitHub #39).

    A particle instrument's total concentration keeps its distribution-type
    label (``"dN, cm⁻³"``), and a size bin adds its diameter (``"dN (337.0
    nm), cm⁻³"``). Any other channel is labelled by *its own* name and unit: a
    named parameter (``"PM2.5, µg/m³"``, ``"Flow, l/min"``), or the main series
    of an instrument without a total concentration (``"LDSA, nm²/cm³"``, ``"IR
    BCc, ng/m³"``, ``"Temperature, °C"``) — the dtype describes the primary
    series and mislabelled other channels (a PM2.5 mass series as ``"dN,
    cm⁻³"``). A gas monitor's main series is named by its gas (``"Cl₂,
    ppm"``). The unit is left out when the channel has none on record (see
    :func:`channel_unit`).

    Args:
        obj: The aerosol object.
        column: Column/channel name.

    Returns:
        The label, e.g. ``"PM2.5, µg/m³"``.
    """
    if column == "Total_conc":
        return f"{base_dtype(obj.dtype_of(column))}, {obj.unit_of(column)}"
    if column in (getattr(obj, "_sizebin_headers", None) or []):
        dtype = base_dtype(obj.dtype_of(column))
        return f"{dtype} ({column} nm), {obj.unit_of(column)}"
    name = str(column)
    if column == _primary_name(obj):
        species = getattr(obj, "_species", None)  # Gas1D: the gas is its dtype
        name = str(
            getattr(obj, "measurement", None)
            or (species() if callable(species) else None)
            or column
        )
    unit = channel_unit(obj, column)
    return f"{name}, {unit}" if unit else name
