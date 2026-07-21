"""Axis-label regression tests for the single-dataset display panes.

Covers the rule that a series is labelled by the column's *own* name, and a unit
is appended only when the column genuinely owns one — a per-column ``unit`` dict
entry (e.g. a Partector's ``Flow`` → l/min) or the scalar unit of the primary
channel. A housekeeping ``extra_data`` column (e.g. ``"Temperature (C)"``) has no
unit metadata, so it must not inherit the *primary* series' unit (``cm⁻³``).

The PyQt5 tests are guarded/offscreen; the ``column_unit`` helper test is pure.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

_DATA = os.path.join(os.path.dirname(__file__), "data")


def test_column_unit_only_reports_genuine_units():
    """``helpers.column_unit`` returns a unit only when the column owns one."""
    import aerosoltools as at
    from aerosoltools.gui.logic import helpers

    # Multi-channel instrument: units live in a per-column dict.
    part = at.load_partector_file(os.path.join(_DATA, "Sample_Partector.txt"))
    assert helpers.column_unit(part, "Flow") == part._meta["unit"]["Flow"]
    assert helpers.column_unit(part, "LDSA") == part._meta["unit"]["LDSA"]
    assert helpers.column_unit(part, "not a column") is None

    # Scalar-unit instrument: only the primary channel carries the scalar unit;
    # an injected housekeeping column must not inherit it.
    ops = at.load_ops_file(os.path.join(_DATA, "Sample_OPS2.txt"))
    ops._extra_data = pd.DataFrame(
        {"Temperature (C)": [21.0] * len(ops.time)}, index=ops.time
    )
    primary = getattr(ops._primary, "name", None)
    assert helpers.column_unit(ops, primary) == ops.unit_of()
    assert helpers.column_unit(ops, "Temperature (C)") is None
    # The un-strict accessor still (wrongly) falls back to the primary unit.
    assert ops.unit_of("Temperature (C)") == ops.unit_of()


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from aerosoltools.gui.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_timeseries_ylabel_names_column_and_skips_bogus_unit(app):
    """Data channels keep their real unit; extra columns show header text only."""
    from aerosoltools.gui.app.main_window import MainWindow
    from aerosoltools.gui.tabs.timeseries import TimeSeriesTab

    win = MainWindow()
    win.load_file(os.path.join(_DATA, "Sample_Partector.txt"), instrument="Partector")
    obj = win.project.active.obj
    obj._extra_data = pd.DataFrame(
        {"Temperature (C)": [21.0] * len(obj.time)}, index=obj.time
    )

    tab = win.findChildren(TimeSeriesTab)[0]
    tab.refresh(reset_view=True)
    tab._sync_columns()

    labels = {}
    for i in range(tab.column.count()):
        if tab.column.itemData(i) is None:
            continue  # section header
        tab.column.setCurrentIndex(i)
        tab._plot_on(tab.ax)
        labels[tab.column.itemText(i)] = tab.ax.get_ylabel()

    # A real per-column unit is shown, named by the column (not "Unknown dtype").
    assert labels["Flow (data)"] == "Flow, l/min"
    # The housekeeping column shows its header only — no fabricated "cm⁻³"/"dN".
    extra_label = labels["Temperature (C) (extra)"]
    assert extra_label == "Temperature (C)"
    assert "cm" not in extra_label and "dN" not in extra_label


def test_decay_display_unit_drops_fabricated_extra_unit(app):
    """Decay keeps derived/species units but drops a raw column's fake unit."""
    from aerosoltools.gui.app.main_window import MainWindow
    from aerosoltools.gui.tabs.decay import DecayTab

    win = MainWindow()

    # 2-D particle: derived PNC/MASS/Pₓ keep their real (hardcoded) units.
    win.load_file(os.path.join(_DATA, "Sample_OPS2.txt"), instrument="OPS")
    tab = win.findChildren(DecayTab)[0]
    tab.refresh()
    assert tab._display_unit("PNC", "cm⁻³") == "cm⁻³"
    assert tab._display_unit("MASS", "µg/m³") == "µg/m³"

    # Gas: the species metric keeps ppm.
    win.load_file(os.path.join(_DATA, "Sample_Ranger.csv"), instrument="Ranger")
    gtab = win.findChildren(DecayTab)[0]
    gtab.refresh()
    gobj = win.project.active.obj
    _times, _vals, unit, metric = gtab._series()
    assert metric == "Cl₂" and unit == "ppm"

    # A raw housekeeping column on the 1-D gas drops the primary's fabricated unit.
    gobj._data["Temperature (C)"] = 21.0
    fabricated = gobj.unit_of("Temperature (C)")  # == "ppm" via fallback
    assert gtab._display_unit("Temperature (C)", fabricated) == ""
