"""Regression tests for derived-column invalidation (Policy 1).

Memoised derived columns (MASS, Pₓ) are cached in ``extra_data`` and reused if
present. Any data-changing mutation must drop them so they recompute from the
current data; loaded housekeeping columns must be left intact. Before this fix a
cached MASS survived an ELPI density change ~19 % stale.
"""

from __future__ import annotations

import os

import numpy as np

import aerosoltools as at

_DATA = os.path.join(os.path.dirname(__file__), "data")


def _mass_pm(obj):
    m, _ = obj._get_metric_series("MASS")
    p, _ = obj._get_metric_series("PM2.5")
    return float(np.nansum(m)), float(np.nansum(p))


def test_density_change_invalidates_mass_and_px():
    """ELPI set_density must not leave a stale cached MASS/Pₓ."""
    a = at.load_elpi_file(os.path.join(_DATA, "Sample_ELPI.txt"))
    _mass_pm(a)  # caches MASS/PM2.5 at the original density
    a.set_density(2.0)  # changes bin diameters
    m_after, p_after = _mass_pm(a)  # cached path after the change

    fresh = at.load_elpi_file(os.path.join(_DATA, "Sample_ELPI.txt"))
    fresh.set_density(2.0)
    m_fresh, p_fresh = _mass_pm(fresh)

    assert m_after == m_fresh  # was ~19 % off before the fix
    assert p_after == p_fresh


def test_smoothing_invalidates_derived_metrics():
    """Rolling smooth must recompute MASS/Pₓ, not reuse the pre-smooth cache."""
    a = at.load_ops_file(os.path.join(_DATA, "Sample_OPS2.txt"))
    _mass_pm(a)
    a.timesmooth(window=11, method="mean")
    m_after, p_after = _mass_pm(a)

    fresh = at.load_ops_file(os.path.join(_DATA, "Sample_OPS2.txt"))
    fresh.timesmooth(window=11, method="mean")
    m_fresh, p_fresh = _mass_pm(fresh)

    assert m_after == m_fresh
    assert p_after == p_fresh


def test_crop_drops_derived_but_keeps_loaded_columns():
    """A crop invalidates derived columns yet preserves loaded housekeeping ones."""
    obj = at.load_ops_file(os.path.join(_DATA, "Sample_OPS2.txt"))
    _mass_pm(obj)  # MASS/PM2.5 now cached in extra_data
    assert "MASS" in obj.extra_data.columns
    loaded = [c for c in obj.extra_data.columns if c not in ("MASS", "PM2.5")]

    t = obj.time
    obj.timecrop(start=str(t[0]), end=str(t[len(t) // 2]))

    # Derived columns are gone (will recompute on demand)...
    assert "MASS" not in obj.extra_data.columns
    # ...but the loaded housekeeping columns survive (cropped, not dropped).
    assert all(c in obj.extra_data.columns for c in loaded)
    assert len(obj.extra_data) == len(obj.time)


def test_derived_registry_carried_across_copy():
    """``_derived_columns`` (and thus invalidation) survives copy_self."""
    obj = at.load_ops_file(os.path.join(_DATA, "Sample_OPS2.txt"))
    obj._get_metric_series("MASS")
    assert "MASS" in obj._derived_columns
    assert "MASS" in obj.copy_self()._derived_columns
