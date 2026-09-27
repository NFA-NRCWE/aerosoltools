"""GUI test: the 2D heatmap shows dx/dlogDp by default (GitHub #40)."""

from __future__ import annotations

import os

import pytest

_DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from aerosoltools.gui.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _colorbar_label(tab) -> str:
    return tab.figure.axes[-1].get_ylabel()


def test_heatmap_is_normalized_by_default(app):
    from aerosoltools.gui.app.main_window import MainWindow
    from aerosoltools.gui.tabs.heatmap import HeatmapTab

    win = MainWindow()
    win.load_file(os.path.join(_DATA, "Sample_OPS.csv"), instrument="OPS")
    tab = win.findChildren(HeatmapTab)[0]
    tab.refresh()

    assert tab.normalize.isChecked()
    assert _colorbar_label(tab).startswith("dN/dlogDp")
    # The loaded dataset itself is left as it was.
    assert "/dlogDp" not in tab.obj.dtype

    tab.normalize.setChecked(False)
    assert _colorbar_label(tab).startswith("dN,")
