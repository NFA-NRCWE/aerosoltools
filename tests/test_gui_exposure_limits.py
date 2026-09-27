"""GUI tests for picking a substance's exposure limits (headless / offscreen).

Covers the Summary tab's substance dropdown (fills STEL/OEL in the metric's unit,
refuses a non-mass metric, records the source with the result) and the plot
threshold's *OEL* link (converted per axis unit, hidden on a number axis).
"""

from __future__ import annotations

import dataclasses
import os

import pytest

_DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from aerosoltools.gui.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _isolated_user_lists(app, tmp_path, monkeypatch):
    """Keep tests off the real user's fetched lists (bundled list only)."""
    from aerosoltools.gui.logic import exposure_limits

    monkeypatch.setattr(exposure_limits, "user_dir", lambda: tmp_path / "oel")
    exposure_limits.reload()
    yield
    exposure_limits.reload()


def _summary_tab(instrument_file: str, instrument: str):
    from aerosoltools.gui.app.main_window import MainWindow
    from aerosoltools.gui.tabs.summary import SummaryTab

    win = MainWindow()
    win.load_file(os.path.join(_DATA, instrument_file), instrument=instrument)
    tab = win.findChildren(SummaryTab)[0]
    tab.refresh()
    tab.kind.setCurrentText("Exposure summary")
    return win, tab


def _pick(tab, name: str) -> None:
    combo = tab.oel_combo
    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if combo.itemText(i) == name)
    )
    combo.picked.emit()


def test_summary_pick_fills_limits_and_records_source(app):
    """Picking a substance fills OEL/STEL in µg/m³ and tags the result table."""
    _win, tab = _summary_tab("Sample_DustTrak.csv", "DustTrak")
    assert tab._exposure_unit() == "µg/m³"
    _pick(tab, "Kvarts, respirabel")
    assert tab.long_limit.text() == "100" and tab.short_limit.text() == "200"
    assert tab.long_limit._unit.text() == "µg/m³"
    assert "BEK nr 613 af 29/06/2026" in tab.oel_source.text()

    tab._compute()
    df = tab.model.dataframe
    assert set(df["Substance"]) == {"Kvarts, respirabel"}
    assert set(df["Limit source"]) == {"BEK nr 613 af 29/06/2026"}
    assert set(df["Exposure limit [µg/m³]"]) == {100.0}
    assert set(df["STEL [µg/m³]"]) == {200.0}

    # The pick is stored with the cached result and restored with it.
    tab.kind.setCurrentText("Activity summary")
    tab._oel_pick = None
    tab.kind.setCurrentText("Exposure summary")
    assert tab._oel_pick["substance"] == "Kvarts, respirabel"
    assert tab.oel_combo.currentText() == "Kvarts, respirabel"


def test_summary_pick_refuses_number_metric(app):
    """A mass-based limit is not written into a number-concentration summary."""
    _win, tab = _summary_tab("Sample_OPS2.txt", "OPS")
    tab._metric_keys_by_kind["Exposure summary"] = ["PNC"]
    before = (tab.long_limit.text(), tab.short_limit.text())
    _pick(tab, "Kvarts, respirabel")
    assert (tab.long_limit.text(), tab.short_limit.text()) == before
    assert "Choose a mass metric" in tab.status.text()


def test_summary_typed_limit_clears_pick(app):
    """Typing a limit by hand un-picks the substance."""
    _win, tab = _summary_tab("Sample_DustTrak.csv", "DustTrak")
    _pick(tab, "Træstøv, inhalerbart")
    assert tab._oel_pick is not None
    tab.long_limit.textEdited.emit("500")
    assert tab._oel_pick is None
    assert tab.oel_combo.currentText() == tab.oel_combo.NONE_TEXT


def test_threshold_linked_limit_follows_axis_unit(app):
    """A linked OEL line converts per axis unit, hides on a number axis, persists."""
    from aerosoltools.gui.logic import exposure_limits
    from aerosoltools.gui.view.widgets import ThresholdControls

    limits = exposure_limits.active_list()
    ctl = ThresholdControls(limits_provider=exposure_limits.active_list)
    assert ctl.threshold_value("µg/m³") is None  # nothing set yet

    ctl._apply_limit(limits["Kvarts, respirabel"], limits, "stel")
    assert ctl.threshold_value("µg/m³") == pytest.approx(200.0)
    assert ctl.value.text() == "200"
    assert ctl.threshold_value("mg/m³") == pytest.approx(0.2)
    assert ctl.threshold_value("cm⁻³") is None  # not comparable: no line
    assert "short-term" in ctl.legend_text() and "BEK 613/2026" in ctl.legend_text()

    # The link survives a save/restore, then typing a value replaces it.
    again = ThresholdControls(limits_provider=exposure_limits.active_list)
    again.set_state(ctl.state())
    assert again.threshold_value("µg/m³") == pytest.approx(200.0)
    again.value.setText("42")
    again.value.textEdited.emit("42")
    assert again.state()["oel"] is None
    assert again.threshold_value("cm⁻³") == pytest.approx(42.0)


def test_threshold_menu_disabled_on_number_axis(app):
    """The OEL menu explains, and disables picks, when the axis is not mass."""
    from aerosoltools.gui.logic import exposure_limits
    from aerosoltools.gui.view.widgets import ThresholdControls

    ctl = ThresholdControls(limits_provider=exposure_limits.active_list)
    ctl.threshold_value("cm⁻³")
    ctl._fill_oel_menu()
    subs = [a.menu() for a in ctl._oel_menu.actions() if a.menu()]
    assert subs and not any(m.isEnabled() for m in subs)
    ctl.threshold_value("µg/m³")
    ctl._fill_oel_menu()
    subs = [a.menu() for a in ctl._oel_menu.actions() if a.menu()]
    assert all(m.isEnabled() for m in subs)


def test_newer_user_list_takes_over(app):
    """A saved newer order becomes the list offered, and changes are reported."""
    from aerosoltools.gui.logic import exposure_limits

    bundled = exposure_limits.active_list()
    quartz = bundled["Kvarts, respirabel"]
    newer = dataclasses.replace(
        bundled,
        source=dataclasses.replace(bundled.source, number=999, date="2027-01-15"),
        limits=tuple(
            dataclasses.replace(lim, twa=0.05, stel=0.1) if lim is quartz else lim
            for lim in bundled.limits
        ),
    )
    path = exposure_limits.save_user_list(newer)
    assert path.name == "dk_bek_2027_999.json"
    assert exposure_limits.active_list().source.label == "BEK nr 999 af 15/01/2027"
    diff = exposure_limits.compare_lists(bundled, exposure_limits.active_list())
    assert diff == [
        "~ Kvarts, respirabel: 8-h 0.1 → 0.05 mg/m³; short-term 0.2 → 0.1 mg/m³"
    ]


def test_dialog_check_and_download_newer_order(app, monkeypatch):
    """The dialog follows a replacement, previews the change and saves it."""
    from aerosoltools.exposure_limits import retsinformation as ri
    from aerosoltools.gui.app import exposure_limits_dialog as dlg_mod
    from aerosoltools.gui.logic import exposure_limits
    from aerosoltools.gui.qt import QtWidgets

    bundled = exposure_limits.active_list()
    old = ri.OrderStatus(eli=bundled.source.eli, title=ri.ORDER_TITLE, in_force=False)
    new = ri.OrderStatus(
        eli=ri.eli_url("2027/5"), title=ri.ORDER_TITLE, in_force=True, date="2027-01-02"
    )
    monkeypatch.setattr(
        dlg_mod,
        "find_current_order",
        lambda eli: ri.CurrentOrder(current=new, chain=(old, new)),
    )
    newer = dataclasses.replace(
        bundled,
        source=dataclasses.replace(
            bundled.source, number=5, date="2027-01-02", eli=new.eli
        ),
    )
    monkeypatch.setattr(dlg_mod, "fetch_exposure_limits", lambda eli: newer)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes
    )

    dlg = dlg_mod.ExposureLimitsDialog()
    dlg._check()
    assert "has been replaced" in dlg.check_status.text()
    assert dlg.get_newer_btn.isEnabled()
    dlg._download(dlg._newer_eli)
    assert dlg.changed
    assert exposure_limits.active_list().source.label == "BEK nr 5 af 02/01/2027"
    assert "BEK nr 5 af 02/01/2027" in dlg.header.text()
