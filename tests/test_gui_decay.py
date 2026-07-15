"""Headless GUI test for the Decay tab's metric selection.

Guarded so it is skipped when the optional PyQt5 GUI extra is not installed; it
runs on the offscreen Qt platform so it needs no display.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def app():
    from aerosoltools.gui.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _decay_tab(window):
    from aerosoltools.gui.tabs.decay import DecayTab

    return window.findChildren(DecayTab)[0]


def test_decay_metric_options_follow_dataset_type(app):
    """The Decay tab offers dataset-appropriate metrics, not just PNC.

    Regression: selecting a non-particle dataset (a Ranger gas monitor measuring
    Cl₂ in ppm) used to leave PNC as the only metric, so drawing raised
    "'PNC' ... is not available for this Gas1D".
    """
    from aerosoltools.gui.app.main_window import MainWindow

    win = MainWindow()

    # Gas monitor: the only metric is its measured species, and drawing works.
    win.load_file(os.path.join(_DATA, "Sample_Ranger.csv"), instrument="Ranger")
    tab = _decay_tab(win)
    tab.refresh()
    kinds = [tab.metric_kind.itemText(i) for i in range(tab.metric_kind.count())]
    assert kinds == ["Cl₂"]
    assert "PNC" not in kinds
    _times, values, unit, metric = tab._series()
    assert metric == "Cl₂" and unit == "ppm" and len(values) > 0
    tab._draw()  # previously raised ValueError for the Gas1D

    # Particle size-resolved dataset still exposes PNC + the derived fractions.
    win.load_file(os.path.join(_DATA, "Sample_OPS2.txt"), instrument="OPS")
    tab.refresh()
    p_kinds = [tab.metric_kind.itemText(i) for i in range(tab.metric_kind.count())]
    assert "PNC" in p_kinds and "PM" in p_kinds
