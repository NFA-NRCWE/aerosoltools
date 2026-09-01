"""The Decay tab's three modes of working, and the draggable level handles.

Follow-up to the #30 rework, from maintainer testing:

1. A value typed into P0 came back changed -- it was silently clipped to the
   window's lowest post-peak sample.
2. The start-level line could not be dragged, only typed.
3. It was not visible which numbers came from the algorithm, which from the
   user, and whether they were current.

The three intended modes: mark a region and press Fit (algorithm picks its own
starting values); set values by hand and press Fit (algorithm optimises from
them); or set values by hand and stop there (the values are used as-is, with
parameters and R2 still computed for the drawn curve).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data")
_WINDOW = (pd.Timestamp("2023-09-11 14:45:00"), pd.Timestamp("2023-09-11 15:35:00"))


@pytest.fixture(scope="module")
def app():
    from aerosoltools.gui.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(app):
    """A Decay tab with one fit over the documented NanoScan decay window."""
    from aerosoltools.gui.app.main_window import MainWindow

    win = MainWindow()
    win.load_file(os.path.join(_DATA, "Sample_NS.csv"), "NanoScan (NS)")
    stack = win.tabs.stack
    tab = next(
        stack.widget(i)
        for i in range(stack.count())
        if type(stack.widget(i)).__name__ == "DecayTab"
    )
    tab.refresh()
    tab._specs().append(
        {
            "window": _WINDOW,
            "metric": tab._build_metric(),
            "model": "first_order",
            "optimized": True,
            "overrides": tab._new_overrides(),
        }
    )
    tab._sel = 0
    tab._recompute_all()
    tab._draw()
    return tab


class _Event:
    """Minimal stand-in for a Matplotlib mouse event."""

    def __init__(self, **kw):
        self.x = self.y = self.xdata = self.ydata = None
        self.inaxes = None
        self.__dict__.update(kw)


def _set_field(tab, edit, text):
    edit.setText(text)
    edit.setModified(True)
    tab._apply_guess_fields()


# -- 1. a typed level is not clipped -----------------------------------------


def test_typed_background_is_used_verbatim(tab):
    """It used to come back clipped to the window's lowest post-peak sample."""
    _set_field(tab, tab.bg_edit, "8000")
    assert tab._results[0]["background"] == pytest.approx(8000.0, abs=1e-6)


def test_typed_background_survives_the_redraw_into_the_field(tab):
    """The field is refilled from the result, so a clipped value showed up there."""
    _set_field(tab, tab.bg_edit, "8000")
    tab._draw()
    assert tab._to_float(tab.bg_edit.text(), None) == pytest.approx(8000.0, rel=1e-3)


def test_a_background_above_the_window_minimum_is_allowed(tab):
    """A separately measured background may sit above the noisy window floor."""
    values = tab.obj.total_concentration
    mask = (tab.obj.time >= _WINDOW[0]) & (tab.obj.time <= _WINDOW[1])
    floor = float(np.nanmin(values[mask]))

    above = round(floor * 1.2)
    _set_field(tab, tab.bg_edit, str(above))
    assert above > floor
    assert tab._results[0]["background"] == pytest.approx(above, abs=1e-6)


# -- 2. the start level is draggable -----------------------------------------


def test_start_level_line_is_drawn_and_grabbable(tab):
    line = tab._start_line
    assert line is not None, "no start-level handle drawn for the selected fit"

    xs = line.get_xdata()
    event = _Event(
        x=tab._x_pixels(xs[0]) + 5, y=tab._line_y_pixels(line), inaxes=tab.ax
    )
    assert tab._grab_handle(event) is True
    assert tab._dragging == "start"


def test_dragging_the_start_level_sets_it(tab):
    line = tab._start_line
    xs = line.get_xdata()
    tab._grab_handle(
        _Event(x=tab._x_pixels(xs[0]) + 5, y=tab._line_y_pixels(line), inaxes=tab.ax)
    )

    target = float(line.get_ydata()[0]) * 1.35
    move = _Event(inaxes=tab.ax, ydata=target)
    tab._on_motion(move)
    tab._on_release(move)

    assert tab._results[0]["start_concentration"] == pytest.approx(target, abs=1e-6)
    assert tab._selected_spec()["overrides"]["start"] == pytest.approx(target)


def test_background_line_still_grabbable_beside_the_start_segment(tab):
    """The start segment must not swallow the background line's grab zone."""
    xs = tab._start_line.get_xdata()
    event = _Event(
        x=tab._x_pixels(xs[-1]) + 200,
        y=tab._line_y_pixels(tab._bg_line),
        inaxes=tab.ax,
    )
    assert tab._grab_handle(event) is True
    assert tab._dragging == "bg"


def test_start_segment_stays_long_enough_to_grab(tab):
    """It spans the pre-emission part, floored so it never collapses to a point."""
    xs = tab._start_line.get_xdata()
    region = (_WINDOW[1] - _WINDOW[0]).total_seconds()
    span = (pd.Timestamp(xs[-1]) - pd.Timestamp(xs[0])).total_seconds()
    assert span >= 0.2 * region


# -- 3. the mode is visible ---------------------------------------------------


def test_mode_is_auto_before_anything_is_touched(tab):
    assert tab._fit_mode(tab._selected_spec()) == "auto"


def test_mode_becomes_manual_when_a_value_is_set(tab):
    _set_field(tab, tab.bg_edit, "8000")
    assert tab._fit_mode(tab._selected_spec()) == "manual"


def test_mode_becomes_guided_after_pressing_fit(tab):
    _set_field(tab, tab.bg_edit, "8000")
    tab._on_fit()
    assert tab._fit_mode(tab._selected_spec()) == "guided"
    # The guess seeded the fit rather than surviving it.
    assert tab._results[0]["background"] != pytest.approx(8000.0, rel=1e-3)


def test_reset_returns_the_fit_to_auto(tab):
    _set_field(tab, tab.bg_edit, "8000")
    tab._on_reset()
    assert tab._fit_mode(tab._selected_spec()) == "auto"


def test_mode_is_reported_in_the_results_table(tab):
    rows, unit = tab._table_rows()
    assert ("Mode", "mode") in tab._table_columns(unit)
    assert rows[0]["mode"] == "auto fit"

    _set_field(tab, tab.bg_edit, "8000")
    rows, _ = tab._table_rows()
    assert rows[0]["mode"] == "your values (not fitted)"


def test_manual_values_still_get_parameters_and_r_squared(tab):
    """Mode 3: no Fit pressed, but the table must not go stale or blank."""
    before = tab._results[0]["r_squared"]
    _set_field(tab, tab.bg_edit, "8000")
    res = tab._results[0]

    assert np.isfinite(res["r_squared"])
    assert res["r_squared"] != pytest.approx(before, rel=1e-6)
    assert np.isfinite(res["decay_rate_per_hour"])
    rows, _ = tab._table_rows()
    assert rows[0]["r2"] not in ("", "n/a")
    assert rows[0]["rate"] not in ("", "n/a")


def test_every_edit_path_recomputes_the_table(tab):
    """Dragging, typing and Fit must all refresh the displayed numbers."""
    seen = []
    for action in (
        lambda: _set_field(tab, tab.bg_edit, "9000"),
        lambda: _set_field(tab, tab.start_edit, "60000"),
        lambda: tab._on_fit(),
    ):
        action()
        rows, _ = tab._table_rows()
        seen.append(rows[0]["r2"])
        assert tab._results[0] is tab._selected_spec()["_result"]
    assert len(set(seen)) > 1, "the table never changed across three edits"
