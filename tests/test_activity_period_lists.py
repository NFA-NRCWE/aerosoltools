"""Pasting and copying lists of activity periods (GitHub #38).

The parser (``gui.logic.periods``) is Qt-free; the dialog and the Time series
tab are exercised headlessly.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from aerosoltools.gui.logic.periods import (
    format_activities,
    format_periods,
    parse_periods,
)

_DATA = os.path.join(os.path.dirname(__file__), "data")
_LO = pd.Timestamp("2026-09-01 08:00")
_HI = pd.Timestamp("2026-09-05 16:00")
T = pd.Timestamp


def _pairs(text, lo=_LO, hi=_HI):
    return [(str(s), str(e)) for s, e in parse_periods(text, lo, hi).periods]


def test_excel_columns_with_a_header():
    text = "Start\tEnd\n2026-09-03 10:00:00\t2026-09-03 11:30:00\n"
    parsed = parse_periods(text, _LO, _HI)
    assert parsed.periods == [(T("2026-09-03 10:00"), T("2026-09-03 11:30"))]
    assert parsed.skipped == 0  # the header is not counted as a bad row
    assert parsed.dayfirst is None  # ISO is never ambiguous


def test_iso_dates_stay_iso_without_a_data_range():
    """dateutil would read 2026-09-03 as 9 March when asked for day-first."""
    parsed = parse_periods("2026-09-03 10:00\t2026-09-03 11:00")
    assert parsed.periods == [(T("2026-09-03 10:00"), T("2026-09-03 11:00"))]


@pytest.mark.parametrize(
    "text",
    [
        "03-09-2026 10:00\t03-09-2026 11:00",
        "03/09/2026 10:00;03/09/2026 11:00",
        "03.09.2026 10.00\t03.09.2026 11.00",
    ],
)
def test_day_first_dates_are_read_to_fit_the_data(text):
    parsed = parse_periods(text, _LO, _HI)
    assert parsed.periods == [(T("2026-09-03 10:00"), T("2026-09-03 11:00"))]
    assert parsed.dayfirst is True


def test_month_first_dates_when_only_they_fit_the_data():
    parsed = parse_periods("09/03/2026 10:00 AM\t09/03/2026 11:00 AM", _LO, _HI)
    assert parsed.periods == [(T("2026-09-03 10:00"), T("2026-09-03 11:00"))]
    assert parsed.dayfirst is False


def test_clock_times_use_the_first_day_and_roll_over_midnight():
    assert _pairs("10:00\t11:30\n23:30\t00:30") == [
        ("2026-09-01 10:00:00", "2026-09-01 11:30:00"),
        ("2026-09-01 23:30:00", "2026-09-02 00:30:00"),
    ]


def test_name_and_number_columns_are_ignored():
    text = (
        "1\tTask 1\t2026-09-03 10:00\t2026-09-03 11:00\n"
        "2\tEmission 2\t2026-09-03 12:00\t2026-09-03 13:00"
    )
    assert _pairs(text) == [
        ("2026-09-03 10:00:00", "2026-09-03 11:00:00"),
        ("2026-09-03 12:00:00", "2026-09-03 13:00:00"),
    ]


def test_other_separators_and_excel_serials():
    assert _pairs("2026-09-03 10:00 - 2026-09-03 11:00") == [
        ("2026-09-03 10:00:00", "2026-09-03 11:00:00")
    ]
    assert _pairs('"2026-09-03 10:00","2026-09-03 11:00"') == [
        ("2026-09-03 10:00:00", "2026-09-03 11:00:00")
    ]
    assert _pairs("46268.4166667\t46268.5") == [
        ("2026-09-03 10:00:00", "2026-09-03 12:00:00")
    ]


def test_bad_rows_are_counted_not_kept():
    text = "2026-09-03 10:00\t2026-09-03 09:00\nonly 1 value\n\nStart\tEnd\n"
    parsed = parse_periods(text, _LO, _HI)
    assert parsed.periods == []
    assert parsed.skipped == 2


def test_copied_text_round_trips():
    periods = [
        (T("2026-09-03 10:00"), T("2026-09-03 11:00")),
        (T("2026-09-04 09:15:30"), T("2026-09-04 10:00")),
    ]
    assert parse_periods(format_periods(periods), _LO, _HI).periods == periods
    all_text = format_activities({"A": periods[:1], "B": periods[1:]})
    assert all_text.splitlines() == [
        "Activity\tStart\tEnd",
        "A\t2026-09-03 10:00:00\t2026-09-03 11:00:00",
        "B\t2026-09-04 09:15:30\t2026-09-04 10:00:00",
    ]
    # Pasting the three-column copy into one activity keeps both periods.
    assert parse_periods(all_text, _LO, _HI).periods == periods


# -- GUI ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from aerosoltools.gui.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dialog(periods):
    from aerosoltools.gui.tabs.timeseries import ActivityEditorDialog

    return ActivityEditorDialog(None, "Task", periods, _LO, _HI)


def test_paste_replaces_the_placeholder_row(app):
    dlg = _dialog([])
    assert dlg.table.rowCount() == 1  # placeholder
    dlg.paste_text("03-09-2026 10:00\t03-09-2026 11:00\n04-09-2026 10:00\tx")
    assert dlg.periods() == [(T("2026-09-03 10:00"), T("2026-09-03 11:00"))]
    assert "day-month-year" in dlg.status.text()
    assert "1 row(s)" in dlg.status.text()


def test_paste_adds_to_existing_periods_and_copy(app):
    from aerosoltools.gui.qt import QtWidgets

    existing = [(T("2026-09-02 10:00"), T("2026-09-02 11:00"))]
    dlg = _dialog(existing)
    dlg.paste_text("2026-09-03 10:00\t2026-09-03 11:00")
    assert dlg.periods() == existing + [(T("2026-09-03 10:00"), T("2026-09-03 11:00"))]
    dlg.paste_text("nothing useful")
    assert dlg.table.rowCount() == 2
    assert "No start/end pairs" in dlg.status.text()

    dlg._copy()
    copied = QtWidgets.QApplication.clipboard().text()
    assert copied.splitlines()[0] == "Start\tEnd"
    assert parse_periods(copied, _LO, _HI).periods == dlg.periods()


def test_new_activity_from_a_list_and_copy_all(app, monkeypatch):
    from aerosoltools.gui.app.main_window import MainWindow
    from aerosoltools.gui.qt import QtWidgets
    from aerosoltools.gui.tabs.timeseries import ActivityEditorDialog, TimeSeriesTab

    win = MainWindow()
    win.load_file(os.path.join(_DATA, "Sample_CPC_AIM.txt"))
    win.load_file(os.path.join(_DATA, "Sample_OPS.csv"), instrument="OPS")
    tab = win.findChildren(TimeSeriesTab)[0]
    t = tab.obj.time
    text = format_periods([(t[5], t[15]), (t[30], t[40])])
    QtWidgets.QApplication.clipboard().setText(text)

    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", staticmethod(lambda *a: ("Pasted", True))
    )
    monkeypatch.setattr(
        ActivityEditorDialog, "exec_", lambda self: QtWidgets.QDialog.Accepted
    )
    tab._new_from_list()

    proj = win.project
    assert proj.activities["Pasted"] == [(t[5], t[15]), (t[30], t[40])]
    # Like a marked task, a new task applies to the active dataset only.
    assert proj.activity_scope("Pasted") == {proj.active_id}
    assert "Pasted" in tab.obj.activities

    tab._copy_all_periods()
    rows = QtWidgets.QApplication.clipboard().text().splitlines()
    assert rows[0] == "Activity\tStart\tEnd"
    assert len(rows) == 3 and all(r.startswith("Pasted\t") for r in rows[1:])

    # Nothing usable on the clipboard: say so, and create nothing.
    told = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", staticmethod(lambda *a: told.append(a))
    )
    QtWidgets.QApplication.clipboard().setText("no times here")
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", staticmethod(lambda *a: ("Empty", True))
    )
    tab._new_from_list()
    assert told and "Empty" not in proj.activities


def test_add_activity_keeps_its_scope(app):
    """set_activity_periods' new ``scope`` must not reset add_activity's scope."""
    from aerosoltools.gui.state.project import Project

    proj = Project()
    proj.add_activity("A", _LO, _HI, scope={7})
    assert proj.activity_scope("A") == {7}
    proj.set_activity_periods("A", [(_LO, _HI)], scope={9})  # existing: kept
    assert proj.activity_scope("A") == {7}
