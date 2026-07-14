"""Shared object-construction helpers for the size-resolved loaders.

The 2-D loaders all finish the same way: build the class, set the same handful
of ``_meta`` fields (instrument / bin axis / serial / unit / dtype / density),
convert to number concentration and undo any ``/dlogDp`` normalization. This
helper centralises that tail so a loader doesn't repeat it (and doesn't reach
into the core private conversion methods directly).
"""

from __future__ import annotations

from typing import Optional

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
    if extra_meta:
        obj._meta.update(extra_meta)
    if to_number:
        obj._convert_to_number_concentration()
    if unnormalize:
        obj.unnormalize_logdp()
    return obj
