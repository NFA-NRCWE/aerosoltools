"""Shared behaviour for non-particle time-series instruments.

:class:`_NonParticleMixin` is composed into the data classes for instruments
whose primary quantity is **not** a particle number/size concentration — gases
(:class:`~aerosoltools.Gas1D`), black carbon
(:class:`~aerosoltools.Aethalometer`), environmental loggers
(:class:`~aerosoltools.Environmental1D`) and LDSA-only dosimeters
(:class:`~aerosoltools.Partector`).

These classes reuse the reusable time-series / activity / plotting / summary
machinery of :class:`~aerosoltools.Aerosol1D` (via
:class:`~aerosoltools.AerosolAlt`), but the "total concentration" concept does
not apply to them. This mixin therefore:

* removes :attr:`total_concentration` (it raises with a pointer to the correct
  accessor), and
* redefines the package-internal :attr:`_primary` hook so reusable operations
  (default peak detection, default summaries) and the GUI still have a
  well-defined "main channel" to fall back on.

The measured species/quantity is carried in metadata (``.dtype`` / ``.measurement``),
never in the attribute name; each concrete class exposes stable, discoverable
domain accessors (e.g. ``.concentration``, ``.ldsa``, ``.ir_bcc``).
"""

from __future__ import annotations

import pandas as pd


class _NonParticleMixin:
    """Neutralise the particle-only ``total_concentration`` and supply ``_primary``.

    Concrete classes may set the class attribute :attr:`_primary_column` (the
    name of the main data column) or override :meth:`_primary_name`; otherwise
    the first non-boolean data column is used.
    """

    #: Name of the primary data column, or ``None`` to auto-detect.
    _primary_column: str | None = None

    @property
    def total_concentration(self) -> pd.Series:  # type: ignore[override]
        """Not available: this instrument does not measure particle number.

        Raises:
            AttributeError: Always. Use the class's dedicated accessor (e.g.
                ``.concentration`` / ``.ldsa`` / ``.ir_bcc``) or a named channel
                in :attr:`data` instead.
        """
        channels = [
            c
            for c in self._data.columns  # type: ignore[attr-defined]
            if self._data[c].dtype != bool  # type: ignore[attr-defined]
        ]
        raise AttributeError(
            f"{type(self).__name__} does not measure particle number, so it has "
            f"no 'total_concentration'. Use this class's dedicated accessor or a "
            f"named channel in .data ({', '.join(map(str, channels))})."
        )

    def _primary_name(self) -> str:
        """Return the column name of the primary channel."""
        col = self._primary_column or self._meta.get("primary")  # type: ignore[attr-defined]
        if col is not None and col in self._data.columns:  # type: ignore[attr-defined]
            return col
        # Fall back to the first non-boolean (i.e. non-activity-mask) column.
        return self._data.select_dtypes(exclude="bool").columns[0]  # type: ignore[attr-defined]

    @property
    def _primary(self) -> pd.Series:
        """The dataset's primary channel (package-internal, not public API)."""
        return self._data[self._primary_name()]  # type: ignore[attr-defined]

    def _require(self, column: str) -> pd.Series:
        """Return a data column, raising a clear error if it is absent."""
        if column not in self._data.columns:  # type: ignore[attr-defined]
            available = ", ".join(map(str, self._data.columns))  # type: ignore[attr-defined]
            raise AttributeError(
                f"{type(self).__name__} has no '{column}' channel in this file; "
                f"available channels: {available}."
            )
        return self._data[column]  # type: ignore[attr-defined]
