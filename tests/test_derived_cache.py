"""Tests for the GUI derived-copy cache and generation invalidation.

The cache memoises basis-converted copies per ``(dataset, generation, basis)``;
``Dataset.touch()`` (called at every mutation site) bumps the generation so a
stale converted view is never handed out. The cache itself is Qt-free; the
mutation-wiring test is headless/offscreen.
"""

from __future__ import annotations

import os

import pytest

import aerosoltools as at

_DATA = os.path.join(os.path.dirname(__file__), "data")


def _ops_dataset():
    from aerosoltools.gui.state.project import Dataset

    path = os.path.join(_DATA, "Sample_OPS2.txt")
    return Dataset(at.load_ops_file(path), path, "OPS")


def test_cache_reuses_until_touch_then_refreshes():
    from aerosoltools.gui.state.derived_cache import DerivedCache

    ds = _ops_dataset()
    cache = DerivedCache()

    v1, unit = cache.converted(ds, "dM")
    v2, _ = cache.converted(ds, "dM")
    assert v1 is v2  # reused within a generation
    assert "dM" in str(v1.dtype) and unit == "µg/m³"

    # dN is the identity — the loaded object itself, no copy.
    assert cache.converted(ds, "dN")[0] is ds.obj

    ds.touch()
    v3, _ = cache.converted(ds, "dM")
    assert v3 is not v1  # generation bumped → recomputed


def test_project_owns_one_cache():
    from aerosoltools.gui.state.project import Project

    p = Project()
    assert p.derived is not None
    # clear() is a no-op on an empty cache but must exist as the single reset.
    p.derived.clear()


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from aerosoltools.gui.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gui_mutation_bumps_generation(app):
    """A crop through the adjustments controls invalidates the active dataset."""
    from aerosoltools.gui.app.main_window import MainWindow

    win = MainWindow()
    win.load_file(os.path.join(_DATA, "Sample_OPS2.txt"), instrument="OPS")
    ds = win.project.active
    gen0 = ds.generation

    t = ds.obj.time
    win.adjust_box._do_crop(str(t[0]), str(t[len(t) // 2]))
    assert ds.generation > gen0
