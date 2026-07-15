"""Shared object-construction helpers for the size-resolved loaders.

The 2-D loaders all finish the same way: build the class, set the same handful
of ``_meta`` fields (instrument / bin axis / serial / unit / dtype / density),
convert to number concentration and undo any ``/dlogDp`` normalization. This
helper centralises that tail so a loader doesn't repeat it (and doesn't reach
into the core private conversion methods directly).
"""

from __future__ import annotations

from typing import Optional

from ...aerosol1d import Aerosol1D
from ...aerosol2d import Aerosol2D


def build_2d(
    data,
    *,
    bin_edges,
    bin_mids,
    instrument: str,
    serial_number,
    unit: str,
    dtype: str,
    density: float = 1.0,
    cls: Optional[type] = None,
    extra_meta: Optional[dict] = None,
    to_number: bool = True,
    unnormalize: bool = True,
) -> Aerosol2D:
    """Construct a size-resolved object and finish loading it.

    Builds ``cls(data)`` (default :class:`Aerosol2D`), writes the canonical
    metadata, then — unless disabled — converts the distribution to number
    concentration and removes any ``/dlogDp`` normalization, matching what the
    loaders did inline.

    Args:
        data: The assembled ``DataFrame`` (Datetime index or column, Total_conc,
            one column per size bin) the class expects.
        bin_edges / bin_mids: Size-bin edges and midpoints (nm).
        instrument: Instrument name for ``_meta["instrument"]``.
        serial_number: Instrument serial number.
        unit: Concentration unit string.
        dtype: Distribution dtype string (e.g. ``"dN"``, ``"dM/dlogDp"``).
        density: Particle density in g/cm³ (default 1.0).
        cls: Concrete class to instantiate (e.g. :class:`ELPI`); defaults to
            :class:`Aerosol2D`.
        extra_meta: Optional extra metadata merged into ``_meta`` (e.g. the ELPI
            charger/cutpoint fields) *before* the conversion runs.
        to_number: Convert to number concentration (``_convert_to_number_
            concentration``) after construction. Default ``True``.
        unnormalize: Undo ``/dlogDp`` normalization after construction. Default
            ``True``.

    Returns:
        The constructed, fully-populated object.
    """
    cls = cls or Aerosol2D
    obj = cls(data)
    # Apply any instrument extra metadata first, then the canonical fields, so
    # the canonical fields always win on a key collision.
    if extra_meta:
        obj._meta.update(extra_meta)
    obj._meta.update(
        {
            "instrument": instrument,
            "bin_edges": bin_edges,
            "bin_mids": bin_mids,
            "density": density,
            "serial_number": serial_number,
            "unit": unit,
            "dtype": dtype,
        }
    )
    if to_number:
        obj._convert_to_number_concentration()
    if unnormalize:
        obj.unnormalize_logdp()
    return obj


def build_1d(
    data,
    *,
    instrument: str,
    serial_number,
    unit=None,
    dtype: Optional[str] = None,
    cls: Optional[type] = None,
    extra_meta: Optional[dict] = None,
) -> Aerosol1D:
    """Construct a 1-D (time-series) object and set its canonical metadata.

    The 1-D / non-particle loaders finish by building the class and writing a
    small, mostly-shared set of ``_meta`` fields; this centralises that tail.
    Unlike :func:`build_2d` there is no size axis and no number/normalization
    conversion. ``unit`` may be a string or a per-channel ``dict`` (multi-channel
    instruments), and ``dtype`` is optional (some non-particle classes carry it
    per channel via ``extra_meta`` instead).

    Args:
        data: The assembled ``DataFrame`` the class expects.
        instrument: Instrument name for ``_meta["instrument"]``.
        serial_number: Instrument serial number.
        unit: Concentration unit string, or a per-channel ``{col: unit}`` dict.
            Omitted from ``_meta`` when ``None``.
        dtype: Distribution/quantity dtype string; omitted when ``None``.
        cls: Concrete class to instantiate (e.g. :class:`Gas1D`,
            :class:`Partector`); defaults to :class:`Aerosol1D`.
        extra_meta: Optional extra metadata merged into ``_meta`` *before* the
            canonical fields (so instrument/serial/unit/dtype win on collision).

    Returns:
        The constructed, populated object.
    """
    cls = cls or Aerosol1D
    obj = cls(data)
    if extra_meta:
        obj._meta.update(extra_meta)
    obj._meta["instrument"] = instrument
    obj._meta["serial_number"] = serial_number
    if unit is not None:
        obj._meta["unit"] = unit
    if dtype is not None:
        obj._meta["dtype"] = dtype
    return obj
