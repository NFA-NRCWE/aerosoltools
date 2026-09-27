"""GUI tests for picking a substance's exposure limits (headless / offscreen).

Covers the Summary tab's substance dropdown (fills STEL/OEL with the fraction
variant that fits the metric, lists no limit where none fits, records the source
and applicability with the result) and the plot threshold's *OEL…* link (both
the 8-hour and short-term lines, converted per axis unit, hidden on a number
axis, flagged when the series' size fraction does not fit).
"""

from __future__ import annotations

import dataclasses
import math
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


def _limits():
    from aerosoltools.gui.logic import exposure_limits

    return exposure_limits.active_list()


def _summary_tab(instrument_file: str, instrument: str, metric: str):
    from aerosoltools.gui.app.main_window import MainWindow
    from aerosoltools.gui.tabs.summary import SummaryTab

    win = MainWindow()
    win.load_file(os.path.join(_DATA, instrument_file), instrument=instrument)
    tab = win.findChildren(SummaryTab)[0]
    tab.refresh()
    tab.kind.setCurrentText("Exposure summary")
    tab._metric_keys_by_kind["Exposure summary"] = [metric]
    return win, tab


def _pick(tab, name: str) -> None:
    combo = tab.oel_combo
    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if combo.itemText(i) == name)
    )
    combo.picked.emit()


def _linked(name: str, show=None):
    """A ThresholdControls linked to ``name`` (both lines unless ``show``)."""
    from aerosoltools.gui.logic import exposure_limits
    from aerosoltools.gui.view.exposure_limit_picker import pick_record
    from aerosoltools.gui.view.widgets import ThresholdControls

    ctl = ThresholdControls(limits_provider=exposure_limits.active_list)
    record = pick_record(_limits()[name], _limits())
    record["show"] = show or {"twa": True, "stel": True}
    ctl._set_link(record)
    return ctl


def test_summary_fills_the_limits_that_fit_the_metric(app):
    """PM4 gets the respirable limit, in µg/m³, and the table records it."""
    _win, tab = _summary_tab("Sample_DustTrak.csv", "DustTrak", "PM4")
    assert tab._exposure_unit() == "µg/m³"
    _pick(tab, "Kvarts, respirabel")
    assert tab.long_limit.text() == "100" and tab.short_limit.text() == "200"
    assert tab.long_limit._unit.text() == "µg/m³"
    assert tab.short_window.text() == "15min"
    assert "BEK nr 613 af 29/06/2026" in tab.oel_source.text()

    tab._compute()
    df = tab.model.dataframe
    assert set(df["Substance"]) == {"Kvarts, respirabel"}
    assert set(df["Limit source"]) == {"BEK nr 613 af 29/06/2026"}
    assert set(df["Limit applies"]) == {"yes"}
    assert set(df["Exposure limit [µg/m³]"]) == {100.0}
    assert set(df["STEL [µg/m³]"]) == {200.0}

    # The pick is stored with the cached result and restored with it.
    tab.kind.setCurrentText("Activity summary")
    tab._oel_pick = None
    tab.kind.setCurrentText("Exposure summary")
    assert tab._oel_pick["substance"] == "Kvarts, respirabel"
    assert tab.oel_combo.currentText() == "Kvarts, respirabel"


def test_summary_uses_the_variant_that_fits(app):
    """A Total channel gets the total-dust variant of the picked substance."""
    _win, tab = _summary_tab("Sample_DustTrak.csv", "DustTrak", "Total")
    _pick(tab, "Kvarts, respirabel")
    assert tab.long_limit.text() == "300" and tab.short_limit.text() == "600"
    tab._compute()
    df = tab.model.dataframe
    assert set(df["Substance"]) == {"Kvarts, total"}
    assert all(v.startswith("yes") for v in df["Limit applies"])


def test_summary_lists_no_limit_for_a_mismatched_fraction(app):
    """PM1 against a respirable/total limit: no limit is listed, with the reason."""
    _win, tab = _summary_tab("Sample_DustTrak.csv", "DustTrak", "PM1")
    _pick(tab, "Kvarts, respirabel")
    assert tab.long_limit.text() == "" and tab.short_limit.text() == ""
    assert "No limit for Kvarts applies to PM1" in tab.status.text()
    tab._compute()
    df = tab.model.dataframe
    assert df["Exposure limit [µg/m³]"].isna().all()
    assert df["STEL [µg/m³]"].isna().all()
    assert all(v.startswith("no – ") for v in df["Limit applies"])
    assert "No exposure limit applies" in tab.status.text()


def test_summary_lists_conservative_comparisons_with_a_warning(app):
    """PM10 against a respirable-only limit is listed, flagged as conservative."""
    _win, tab = _summary_tab("Sample_DustTrak.csv", "DustTrak", "PM10")
    _pick(tab, "Krystallinsk siliciumdioxid, respirabelt støv")
    assert tab.long_limit.text() == "100" and tab.short_limit.text() == "200"
    assert tab.status.text().startswith("⚠ Conservative comparison:")
    assert "overestimated" in tab.status.text()
    tab._compute()
    df = tab.model.dataframe
    assert set(df["Exposure limit [µg/m³]"]) == {100.0}
    assert all(v.startswith("yes (conservative) – ") for v in df["Limit applies"])


def test_summary_welding_limit_has_no_short_term_value(app):
    """A welding limit fills the OEL only; the STEL columns stay blank."""
    _win, tab = _summary_tab("Sample_DustTrak.csv", "DustTrak", "Total")
    _pick(tab, "Svejserøg, MIG/MAG, almindeligt konstruktionsstål, sædvanlig primer")
    assert tab.long_limit.text() == "1600" and tab.short_limit.text() == ""
    tab._compute()
    df = tab.model.dataframe
    assert set(df["Exposure limit [µg/m³]"]) == {1600.0}
    assert df["STEL [µg/m³]"].isna().all()


def test_summary_pick_refuses_number_metric(app):
    """A mass-based limit is not written into a number-concentration summary."""
    _win, tab = _summary_tab("Sample_OPS2.txt", "OPS", "PNC")
    _pick(tab, "Kvarts, respirabel")
    assert tab.long_limit.text() == "" and tab.short_limit.text() == ""
    assert "Choose a mass metric" in tab.status.text()


def test_summary_typed_limit_clears_pick(app):
    """Typing a limit by hand un-picks the substance."""
    _win, tab = _summary_tab("Sample_DustTrak.csv", "DustTrak", "Total")
    _pick(tab, "Træstøv, inhalerbart")
    assert tab._oel_pick is not None and tab.long_limit.text() == "1000"
    tab.long_limit.textEdited.emit("500")
    assert tab._oel_pick is None
    assert tab.oel_combo.currentText() == tab.oel_combo.NONE_TEXT


def test_threshold_draws_both_limits_in_the_axis_unit(app):
    """A linked substance gives an 8-hour (dashed) and short-term (dotted) line."""
    ctl = _linked("Kvarts, respirabel")
    lines = ctl.threshold_lines("µg/m³", [("PM4", 4.0)])
    assert [(line["value"], line["linestyle"]) for line in lines] == [
        (100.0, "--"),
        (200.0, ":"),
    ]
    assert "8-h OEL" in lines[0]["label"] and "short-term (15 min)" in lines[1]["label"]
    assert "BEK 613/2026" in lines[0]["label"]
    assert ctl.warning == "" and ctl.value.text() == "100 / 200"
    assert [line["value"] for line in ctl.threshold_lines("mg/m³")] == [0.1, 0.2]
    # Not a mass axis: no lines, and a reason.
    assert ctl.threshold_lines("cm⁻³", [("PNC", None)]) == []
    assert "no limit lines" in ctl.warning


def test_threshold_warns_about_size_fraction(app):
    """Lines stay, but a PM1 or Total series gets a size-fraction warning."""
    ctl = _linked("Kvarts, respirabel")
    assert len(ctl.threshold_lines("µg/m³", [("PM1", 1.0)])) == 2
    assert "underestimated" in ctl.warning
    ctl.threshold_lines("µg/m³", [("Total", math.inf)])
    assert "'Kvarts, total'" in ctl.warning  # the variant that fits


def test_threshold_line_choice_and_persistence(app):
    """Either line can be hidden; the link survives save/restore until typed over."""
    from aerosoltools.gui.logic import exposure_limits
    from aerosoltools.gui.view.widgets import ThresholdControls

    ctl = _linked("Carbon black", show={"twa": True, "stel": False})
    assert [line["value"] for line in ctl.threshold_lines("mg/m³")] == [3.5]

    again = ThresholdControls(limits_provider=exposure_limits.active_list)
    again.set_state(ctl.state())
    assert [line["value"] for line in again.threshold_lines("µg/m³")] == [3500.0]
    again.value.setText("42")
    again.value.textEdited.emit("42")
    assert again.state()["oel"] is None
    assert [line["value"] for line in again.threshold_lines("cm⁻³")] == [42.0]

    # A link saved by the earlier one-line version ("kind") still loads.
    old = {**ctl.state(), "oel": {**ctl.state()["oel"], "kind": "stel"}}
    del old["oel"]["show"]
    again.set_state(old)
    assert [line["value"] for line in again.threshold_lines("mg/m³")] == [7.0]


def test_threshold_dialog_applies_choices(app):
    """The OEL… dialog links a substance and its line toggles at once."""
    from aerosoltools.gui.view.widgets import ExposureLimitLinesDialog

    seen = []
    dlg = ExposureLimitLinesDialog(None, _limits(), None, seen.append)
    dlg.combo.set_current_name("Titandioxid, beregnet som Ti")
    dlg.combo.picked.emit()
    assert seen[-1]["substance"] == "Titandioxid, beregnet som Ti"
    assert seen[-1]["show"] == {"twa": True, "stel": True}
    dlg.show_stel.setChecked(False)
    assert seen[-1]["show"] == {"twa": True, "stel": False}
    dlg._remove()
    assert seen[-1] is None


def test_timeseries_draws_limit_lines_and_warning(app):
    """End to end: a DustTrak PM1 plot shows both lines and a fraction warning."""
    from aerosoltools.gui.app.main_window import MainWindow
    from aerosoltools.gui.tabs.timeseries import TimeSeriesTab
    from aerosoltools.gui.view.exposure_limit_picker import pick_record

    win = MainWindow()
    win.load_file(os.path.join(_DATA, "Sample_DustTrak.csv"), instrument="DustTrak")
    tab = win.findChildren(TimeSeriesTab)[0]
    tab.refresh(reset_view=True)
    tab._sync_columns()
    index = next(
        i
        for i in range(tab.column.count())
        if (tab.column.itemData(i) or ("", ""))[1] == "PM1"
    )
    tab.column.setCurrentIndex(index)
    record = pick_record(_limits()["Kvarts, respirabel"], _limits())
    record["show"] = {"twa": True, "stel": True}
    tab.threshold._set_link(record)
    tab._plot_on(tab.ax)
    from aerosoltools.gui.logic.helpers import THRESHOLD_COLOR

    styles = sorted(
        line.get_linestyle()
        for line in tab.ax.get_lines()
        if line.get_color() == THRESHOLD_COLOR
    )
    assert styles == ["--", ":"]
    assert any("underestimated" in t.get_text() for t in tab.ax.texts)


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
    """The Tools dialog follows a replacement, previews the change and saves it."""
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
